"""按 SASC 官方 evaluation/evaluator.py 的口径算 Sm / Emax / maxF / MAE。

流程: 模型在 test 上出 sigmoid 概率 -> 存 PNG(/255) -> 用官方 Eval_thread 评测。
预测会先做逐图 min-max 归一化(官方代码里有), 与论文表一致。
"""
import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "vendor"))
from evaluation.evaluator import Eval_thread          # noqa: E402
from nn_net import Net                                 # noqa: E402
from train import UNet4, USOD                          # noqa: E402


class Pairs:
    """给 Eval_thread 的最小 loader: 逐条 yield (PIL 灰度预测, PIL 灰度 GT)。"""

    def __init__(self, pairs):
        self.pairs = pairs

    def __iter__(self):
        for p, g in self.pairs:
            yield Image.open(p).convert("L"), Image.open(g).convert("L")

    def __len__(self):
        return len(self.pairs)


@torch.no_grad()
def dump_preds(ckpt, arch, data, split, out_dir, bs=8, no_depth=False,
               use_post=False, depth_blocks=8, use_msf=True, patch=16,
               fuse_simple=False, same_kv=False, attn_up_idx="", attn_up_mode="stack",
               net_py=False, carafe_idx="", carafe_k=5, carafe_cm=64,
               dss_q=False, dss_add=False, dss_mid=None, dss_hp=True,
               strip_k=(7, 15),
               v1_dep_at="", v1_keep_fuse_dep=False, v1_dep_on_v=False,
               v1_mla_rank=0):
    dev = torch.device("cuda")
    ds = USOD(data, split, 352)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    if arch == "unet":
        m = UNet4(32)
    elif arch == "v1":
        from nn_v1 import NetV1
        dep_at = tuple(int(s) for s in str(v1_dep_at).split(",") if s.strip())
        m = NetV1(depth=depth_blocks, use_depth=True, same_kv=same_kv,
                  dep_blocks=dep_at,
                  keep_fuse_dep=v1_keep_fuse_dep or (not dep_at),
                  dep_on_v=v1_dep_on_v, latent_rank=v1_mla_rank)
    elif attn_up_idx or net_py or carafe_idx or dss_q or dss_add:
        from net import Net as NetAttn
        idx = tuple(int(s) for s in str(attn_up_idx).split(",") if s.strip())
        cidx = tuple(int(s) for s in str(carafe_idx).split(",") if s.strip())
        dep_at = tuple(int(s) for s in str(v1_dep_at).split(",") if s.strip())
        m = NetAttn(size=352, num_classes=1, use_depth=not no_depth, patch=patch,
                    depth=depth_blocks, fuse_simple=fuse_simple, same_kv=same_kv,
                    attn_up_idx=idx, attn_up_mode=attn_up_mode,
                    carafe_idx=cidx, carafe_k=carafe_k, carafe_cm=carafe_cm,
                    dss_quarter=dss_q, dss_add=dss_add, dss_mid=dss_mid,
                    dss_hp=dss_hp,
                    dep_blocks=dep_at, dep_on_v=v1_dep_on_v,
                    keep_fuse_dep=v1_keep_fuse_dep or (not dep_at),
                    strip_k=strip_k)
    else:
        m = Net(size=352, num_classes=1, use_depth=not no_depth,
                use_msf=use_msf, patch=patch, depth=depth_blocks,
                fuse_simple=fuse_simple)
    m = m.to(dev)
    ck = torch.load(ckpt, map_location=dev)
    m.load_state_dict(ck["model"])
    m.eval()
    print(f"loaded {ckpt} (ep{ck.get('epoch')}, val IoU {ck.get('iou'):.4f})", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    k = 0
    for rgb, dep, _ in ld:
        prob = m(rgb.to(dev), dep.to(dev)).sigmoid().cpu().numpy()[:, 0]
        for b in range(prob.shape[0]):
            n = ds.names[k]
            cv2.imwrite(os.path.join(out_dir, n),
                        np.clip(prob[b] * 255.0, 0, 255).astype(np.uint8))
            k += 1
    print(f"wrote {k} preds -> {out_dir}", flush=True)
    return ds.names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="net30.pt")
    p.add_argument("--arch", default="net", choices=["net", "unet", "v1", "v2"])
    p.add_argument("--attn-up-idx", default="",
                   help="对应训练的 --attn-up-idx, 如 0 或 0,1 (非空时走 net.py)")
    p.add_argument("--attn-up-mode", default="stack", choices=["stack", "refine"],
                   help="对应训练的 --attn-up-mode")
    p.add_argument("--carafe-idx", default="",
                   help="对应训练的 --carafe-idx, 如 0 或 0,1 (走 net.py)")
    p.add_argument("--carafe-k", type=int, default=5, help="对应训练的 --carafe-k")
    p.add_argument("--carafe-cm", type=int, default=64, help="对应训练的 --carafe-cm")
    p.add_argument("--dss-q", action="store_true",
                   help="对应训练的 --dss-q (DSS 在 1/4, 1/16 只做深度注入)")
    p.add_argument("--dss-add", action="store_true",
                   help="对应训练的 --dss-add (保留 1/16 的 Fusion, 另在 1/4 加 DSS)")
    p.add_argument("--dss-mid", type=int, default=None, help="对应训练的 --dss-mid")
    p.add_argument("--no-dss-hp", action="store_true",
                   help="对应训练的 --no-dss-hp (v7 权重无 LearnableHighPass)")
    p.add_argument("--strip-k", default="7,15",
                   help="对应训练的 --strip-k (Fusion 长条卷积的核长)")
    p.add_argument("--net-py", action="store_true",
                   help="对应训练的 --net-py: 走 net.py 的 Net (128 通道 fuse/decoder)")
    p.add_argument("--same-kv", action="store_true", help="对应 v1 的 --same-kv 消融")
    p.add_argument("--v1-dep-at", default="",
                   help="对应 v1 的 --v1-dep-at, 如 1,3,5 (空 = fuse 相加旧行为)")
    p.add_argument("--v1-keep-fuse-dep", action="store_true",
                   help="对应 v1 的 --v1-keep-fuse-dep")
    p.add_argument("--v1-dep-on-v", action="store_true",
                   help="对应 v1 的 --v1-dep-on-v")
    p.add_argument("--v1-mla-rank", type=int, default=0,
                   help="对应 v1 的 --v1-mla-rank")
    p.add_argument("--data", default="E:/vscode/SASC-USOD-main/data/USOD10K")
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default="net30")
    p.add_argument("--reuse", action="store_true", help="复用已有预测, 只重算指标")
    p.add_argument("--no-depth", action="store_true", help="对应训练的 --no-depth 消融")
    p.add_argument("--no-msf", action="store_true", help="对应训练的 --no-msf")
    p.add_argument("--simple-fuse", action="store_true", help="对应训练的 --simple-fuse")
    p.add_argument("--patch", type=int, default=16, help="对应训练的 --patch")
    p.add_argument("--depth-blocks", type=int, default=8, help="对应训练的 --depth-blocks")
    p.add_argument("--cpu", action="store_true", help="指标在 CPU 上算(避免和训练抢显存)")
    a = p.parse_args()

    pred_dir = f"preds_{a.tag}"
    if not a.reuse:
        names = dump_preds(a.ckpt, a.arch, a.data, a.split, pred_dir, no_depth=a.no_depth,
                           use_post=False, depth_blocks=a.depth_blocks,
                           use_msf=not a.no_msf, patch=a.patch,
                           fuse_simple=a.simple_fuse, same_kv=a.same_kv,
                           attn_up_idx=a.attn_up_idx, attn_up_mode=a.attn_up_mode,
                           net_py=a.net_py, carafe_idx=a.carafe_idx,
                           carafe_k=a.carafe_k, carafe_cm=a.carafe_cm,
                           dss_q=a.dss_q, dss_add=a.dss_add, dss_mid=a.dss_mid,
                           dss_hp=not a.no_dss_hp,
                           strip_k=tuple(int(s) for s in str(a.strip_k).split(",") if s.strip()),
                           v1_dep_at=a.v1_dep_at,
                           v1_keep_fuse_dep=a.v1_keep_fuse_dep,
                           v1_dep_on_v=a.v1_dep_on_v,
                           v1_mla_rank=a.v1_mla_rank)
    else:
        names = sorted(os.path.basename(x) for x in glob.glob(os.path.join(pred_dir, "*")))

    gt_dir = os.path.join(a.data, a.split, "gt352")
    pairs = [(os.path.join(pred_dir, n), os.path.join(gt_dir, n)) for n in names]
    missing = [n for n in names if not os.path.exists(os.path.join(gt_dir, n))]
    assert not missing, f"GT 缺失: {missing[:3]}"

    out = f"metrics_{a.tag}"
    os.makedirs(out, exist_ok=True)
    ev = Eval_thread(Pairs(pairs), method=a.tag, dataset=f"USOD10K-{a.split}",
                     output_dir=out, cuda=not a.cpu)
    print(ev.run())
    with open(os.path.join(out, "result.txt")) as f:
        print(f.read())


if __name__ == "__main__":
    main()
