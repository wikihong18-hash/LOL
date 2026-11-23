# -*- coding: utf-8 -*-
"""LOL (Locate with Once Look) 推理脚本。

用法:
    # 单张
    python predict.py --rgb  a.jpg --depth a.png --out a_pred.png
    # 整个目录 (按文件名配对 RGB/ 与 depth/)
    python predict.py --rgb-dir RGB --depth-dir depth --out-dir preds

网络固定在 352x352 上训练, 内部会 resize; 输出会还原到原图尺寸。
预测为 0~255 单通道 PNG (显著目标概率)。
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from net import Net

SIZE = 352


def build_model(ckpt, device):
    """按 v7 (即 LOL) 的精确结构构造并加载权重。"""
    m = Net(size=SIZE, num_classes=1, use_depth=True, patch=16, depth=8,
            dss_add=True, dss_hp=False)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = sd.get("model", sd)
    m.load_state_dict(sd)
    return m.to(device).eval()


def load_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_depth(path):
    """读深度并做 1~99 百分位归一化 (与训练时一致, 只对原图做, 再 resize)。"""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.ndim == 3:
        d = d.max(axis=2)
    d = d.astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1) * 255.0


@torch.no_grad()
def predict(model, rgb, dep, device):
    h, w = rgb.shape[:2]
    rgb_r = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    dep_r = cv2.resize(dep, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(np.ascontiguousarray(rgb_r.transpose(2, 0, 1))).float().div(255.0)
    d = torch.from_numpy(np.ascontiguousarray(dep_r[None])).float().div(255.0)
    x, d = x[None].to(device), d[None].to(device)
    logits = model(x, d)
    if isinstance(logits, tuple):
        logits = logits[0]
    prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser("LOL inference")
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "LOL.pt"))
    ap.add_argument("--rgb", default="", help="单张 RGB 路径")
    ap.add_argument("--depth", default="", help="单张深度图路径")
    ap.add_argument("--out", default="", help="单张输出路径 (.png)")
    ap.add_argument("--rgb-dir", default="", help="批量: RGB 目录")
    ap.add_argument("--depth-dir", default="", help="批量: 深度目录 (与 RGB 同名)")
    ap.add_argument("--out-dir", default="preds", help="批量: 输出目录")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, device)
    print(f"loaded {args.ckpt}  params={sum(p.numel() for p in model.parameters()):,}  device={device}")

    if args.rgb:
        assert args.depth, "需要 --depth (LOL 是 RGB-D 模型)"
        dep = load_depth(args.depth)
        prob = predict(model, load_rgb(args.rgb), dep, device)
        out = args.out or os.path.splitext(args.rgb)[0] + "_pred.png"
        cv2.imwrite(out, (prob * 255.0).round().astype(np.uint8))
        print(f"saved {out}  mean={prob.mean():.4f}  max={prob.max():.4f}")
        return

    assert args.rgb_dir and args.depth_dir, "要么给 --rgb/--depth, 要么给 --rgb-dir/--depth-dir"
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.rgb_dir, "*")))
    n = 0
    for f in files:
        name = os.path.basename(f)
        dep_path = os.path.join(args.depth_dir, name)
        if not os.path.exists(dep_path):
            print(f"skip {name}: 深度缺失")
            continue
        prob = predict(model, load_rgb(f), load_depth(dep_path), device)
        cv2.imwrite(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png"),
                    (prob * 255.0).round().astype(np.uint8))
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(files)}")
    print(f"done: {n} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
