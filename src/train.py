import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F

from nn_net import Net, DiceBCELoss, dice_score


def _edge_dice(logits, target, eps=1e-6):
    """原论文 edge_loss = BCE + Dice(sigmoid)。"""
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    den = p.sum((2, 3)) + target.sum((2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def _router_target(edge, k_dilate=3, k_smooth=3):
    """router 监督目标: GT 边界先膨胀再平滑 -> 边界附近 w_bs 应该为 1。"""
    edge = edge.float()
    edge = F.max_pool2d(edge, k_dilate, 1, k_dilate // 2)
    edge = F.avg_pool2d(edge, k_smooth, 1, k_smooth // 2)
    return edge.clamp(0, 1)


def get_aux_weights(epoch, args):
    """前 aux_stage1_epochs 轮强监督, 之后退火 (与原实现一致)。"""
    if epoch <= args.aux_stage1_epochs:
        return args.lam_router1, args.lam_edge1
    return args.lam_router2, args.lam_edge2


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------
class USOD(Dataset):
    """USOD10K: 返回 (rgb(3,H,W), depth(1,H,W), mask(1,H,W)), 均为 float32 [0,1]。

    cache=True 时直接读已建好的 352 缓存 (rgb352/depth352/gt352), 比解码 640x480
    原图快约 30%; cache=False 时读 RGB/ 与 depth/ 原图再 resize。
    """

    def __init__(self, root, split, size=352, cache=True, aug=False, limit=0, ram=False,
                 edge=False):
        self.root = os.path.join(root, split)
        self.size = size
        self.cache = cache
        self.aug = aug
        self.edge = edge
        self.names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.root, "GT", "*")))
        if limit:
            self.names = self.names[:limit]
        if not self.names:
            raise FileNotFoundError(f"no GT found under {self.root}")
        self.ram = ram
        if ram:
            if not cache:
                raise ValueError("--ram-cache 需要 352 缓存 (去掉 --no-cache)")
            self._preload()

    def _preload(self):
        """整集读进内存, uint8。352 下每张 5 通道 620 KB, 7178 张约 4.45 GB。
        注意: 内存版必须配 num_workers=0, 否则 spawn 会把整个数组复制进每个 worker。"""
        from concurrent.futures import ThreadPoolExecutor

        def one(n):
            rgb = cv2.cvtColor(self._read(os.path.join(self.root, "rgb352", n), cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB)
            dep = self._read(os.path.join(self.root, "depth352", n), cv2.IMREAD_GRAYSCALE)
            g = self._read(os.path.join(self.root, "gt352", n), cv2.IMREAD_GRAYSCALE)
            return rgb, dep, g

        t = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(one, self.names))
        self.rgb = np.stack([o[0] for o in out])
        self.dep = np.stack([o[1] for o in out])
        self.gt = np.stack([o[2] for o in out])
        gb = (self.rgb.nbytes + self.dep.nbytes + self.gt.nbytes) / 2**30
        print(f"  ram-cache {len(self.names)} imgs, {gb:.2f} GB, {time.time()-t:.0f}s", flush=True)

    def __len__(self):
        return len(self.names)

    def _read(self, path, flag):
        im = cv2.imread(path, flag)
        if im is None:
            raise FileNotFoundError(path)
        return im

    def __getitem__(self, i):
        n = self.names[i]
        s = self.size

        if self.ram:
            rgb, dep, g = self.rgb[i], self.dep[i], self.gt[i]
            if self.aug and np.random.rand() < 0.5:
                rgb, dep, g = rgb[:, ::-1], dep[:, ::-1], g[:, ::-1]
            if self.aug and np.random.rand() < 0.3:
                a = 1.0 + np.random.uniform(-0.2, 0.2)
                b = np.random.uniform(-18, 18)
                rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
            dep = dep[None].astype(np.float32) / 255.0
            mask = (g[None] > 127).astype(np.float32)
            return rgb, dep, mask

        if self.cache:
            rgb = self._read(os.path.join(self.root, "rgb352", n), cv2.IMREAD_COLOR)
            dep = self._read(os.path.join(self.root, "depth352", n), cv2.IMREAD_GRAYSCALE)
            g = self._read(os.path.join(self.root, "gt352", n), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if self.edge:
                e = self._read(os.path.join(self.root, "edge352", n), cv2.IMREAD_GRAYSCALE)
        else:
            rgb = cv2.cvtColor(self._read(os.path.join(self.root, "RGB", n), cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB)
            dep = self._read(os.path.join(self.root, "depth", n), cv2.IMREAD_UNCHANGED)
            if dep.ndim == 3:
                dep = dep.max(axis=2)
            dep = dep.astype(np.float32)
            lo, hi = np.percentile(dep, 1), np.percentile(dep, 99)
            dep = np.clip((dep - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
            g = self._read(os.path.join(self.root, "GT", n), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, (s, s), interpolation=cv2.INTER_LINEAR)
            g = cv2.resize(g, (s, s), interpolation=cv2.INTER_NEAREST)

        if rgb.shape[0] != s or rgb.shape[1] != s:
            rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, (s, s), interpolation=cv2.INTER_LINEAR)
            g = cv2.resize(g, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.aug:
            if np.random.rand() < 0.5:                       # 水平翻转, 多路同步
                rgb, dep, g = rgb[:, ::-1], dep[:, ::-1], g[:, ::-1]
                if self.edge:
                    e = e[:, ::-1]
            if np.random.rand() < 0.3:                       # 亮度/对比度微扰, 只动 RGB
                a = 1.0 + np.random.uniform(-0.2, 0.2)
                b = np.random.uniform(-18, 18)
                rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255)

        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
        dep = np.ascontiguousarray(dep[None]).astype(np.float32) / 255.0
        mask = (np.ascontiguousarray(g[None]).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        if self.edge:
            emask = (np.ascontiguousarray(e[None]).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
            return rgb, dep, mask, emask
        return rgb, dep, mask


class UNet4(nn.Module):
    """对照用: 把 (rgb, depth) 拼成 4 通道喂标准 UNet, 与 Net 同数据同增强。"""

    def __init__(self, w=32):
        super().__init__()
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "underwater_depth"))
        from train_selector import UNet
        self.m = UNet(4, w)

    def forward(self, rgb, dep):
        return self.m(torch.cat([rgb, dep], 1))


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _match(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def apply_freeze(model, prefixes, keep_sub=()):
    """把匹配前缀的参数 requires_grad 置 False, 返回冻结的张量数。

    keep_sub 里的子串命中则不冻结 (例如 v1 的 dep_norm: 它属于 blocks,
    但深度注入的归一化需要能学, 否则缩放/偏移被锁死在 LN 默认值)。
    """
    n_frozen = 0
    for n_, prm in model.named_parameters():
        if _match(n_, prefixes) and not any(s in n_ for s in keep_sub):
            prm.requires_grad = False
            n_frozen += prm.numel()
    return n_frozen


def freeze_bn_eval(model, prefixes):
    """冻结区域内的 BN/Dropout 强制切到 eval, 否则 running stats 会继续漂移。"""
    for n_, m in model.named_modules():
        if _match(n_, prefixes) and not isinstance(m, (type(None),)):
            m.eval()


# ----------------------------------------------------------------------------
# 训练 / 评测
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, args):
    model.train()
    if getattr(args, "_freeze_prefixes", None):
        freeze_bn_eval(model, args._freeze_prefixes)
    running, n_seen = 0.0, 0
    run_lr, run_le = 0.0, 0.0
    lam_r, lam_e = get_aux_weights(epoch, args)
    t0 = time.time()
    for step, batch in enumerate(loader):
        rgb, depth, mask = batch[0], batch[1], batch[2]
        gt_edge = batch[3] if len(batch) > 3 else None
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
        aux = None
        if args.aux_edge and gt_edge is not None:
            gt_edge = gt_edge.to(device)
            logits, aux = model(rgb, depth, return_aux=True)
        else:
            logits = model(rgb, depth)
        loss = criterion(logits, mask)
        l_r = l_e = 0.0
        if aux is not None and aux.get("edge_logits") is not None:
            tgt = aux["edge_logits"].shape[-2:]
            gt_s = F.interpolate(gt_edge, size=tgt, mode="nearest")
            l_r = F.binary_cross_entropy_with_logits(
                aux["router_logits"], _router_target(gt_s, args.router_k, args.router_k))
            l_e = (F.binary_cross_entropy_with_logits(aux["edge_logits"], gt_s)
                   + _edge_dice(aux["edge_logits"], gt_s))
            loss = loss + lam_r * l_r + lam_e * l_e
            run_lr += float(l_r) * rgb.shape[0]
            run_le += float(l_e) * rgb.shape[0]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        bs = rgb.shape[0]
        running += loss.item() * bs
        n_seen += bs
        if step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            el = time.time() - t0
            print(f"  ep{epoch} step{step:>4}/{len(loader)} loss {loss.item():.4f} "
                  f"lr {lr:.2e} {el/(step+1)*1000:.0f} ms/it", flush=True)

    return (running / max(1, n_seen), time.time() - t0,
            run_lr / max(1, n_seen), run_le / max(1, n_seen))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_dice = tot_iou = 0.0
    n_seen = 0
    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
        logits = model(rgb, depth)
        bs = rgb.shape[0]
        tot_loss += criterion(logits, mask).item() * bs
        tot_dice += dice_score(logits, mask) * bs
        pred = (logits.sigmoid() > 0.5).float()
        dims = (1, 2, 3)
        inter = (pred * mask).sum(dims)
        union = pred.sum(dims) + mask.sum(dims) - inter
        tot_iou += ((inter + 1e-6) / (union + 1e-6)).mean().item() * bs
        n_seen += bs
    return tot_loss / n_seen, tot_dice / n_seen, tot_iou / n_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="E:/vscode/SASC-USOD-main/data/USOD10K")
    p.add_argument("--size", type=int, default=352, help="必须是 32 的倍数")
    p.add_argument("--train-n", type=int, default=0, help="0 = 用全部")
    p.add_argument("--val-n", type=int, default=0, help="0 = 用全部")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--no-cache", action="store_true", help="读原始 640x480 而不读 352 缓存")
    p.add_argument("--ram-cache", action="store_true", help="整集读进内存, 自动 num_workers=0")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=str, default="checkpoint_net.pt")
    p.add_argument("--tag", type=str, default="net")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-from", default="", help="从已有 .pt 权重初始化 (strict=False)")
    p.add_argument("--freeze", default="", help="冻结的模块前缀, 逗号分隔, 例如 fuse,decoder")
    p.add_argument("--lr-new", type=float, default=0.0, help="可训练部分的 lr (0=沿用 --lr)")
    p.add_argument("--lr-dep-norm", type=float, default=0.0,
                   help="depth 注入的 dep_norm 单独 lr (0=沿用 --lr); "
                        "冻结主干训练时建议 5e-5")
    p.add_argument("--patience", type=int, default=0,
                   help="早停: 连续多少轮 val IoU 无提升就停 (0=关闭)")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="早停: 视为提升的最小幅度")
    p.add_argument("--dropout", type=float, default=0.0, help="v2 的 MLP dropout")
    p.add_argument("--arch", default="net", choices=["net", "unet"], help="unet = 1.93M 对照")
    p.add_argument("--no-depth", action="store_true",
                   help="消融: 移除 depth_stem 与 rgb+depth 相加, 只用 RGB")
    p.add_argument("--simple-fuse", action="store_true",
                   help="消融: 融合模块去掉 Laplace/长短条, 只留 3x3+1x1")
    p.add_argument("--same-kv", action="store_true",
                   help="消融: KV 不做 2x2 下采样, 与 Q 同分辨率 (注意力算力 x4)")
    p.add_argument("--v1", action="store_true",
                   help="用 v1 (net30) 的原始结构 (无 MSF, 11x11 token)")
    p.add_argument("--v1-dep-at", default="",
                   help="v1: 深度注入到哪几个 Block (0 起算), 如 1,3,5 = 第 2/4/6 个; "
                        "注入方式 dep 先 LayerNorm 再与特征相加, 然后才进 k_conv; "
                        "给了此参数则 fuse 不再吃深度 (除非再给 --v1-keep-fuse-dep)")
    p.add_argument("--v1-keep-fuse-dep", action="store_true",
                   help="v1: 保留 fuse 里的深度相加 (与 --v1-dep-at 同时给 = 两处都注入)")
    p.add_argument("--v1-dep-on-v", action="store_true",
                   help="v1: 深度同时进 V 分支 (默认只进 K, 只调相似度不进内容)")
    p.add_argument("--v1-mla-rank", type=int, default=0,
                   help="v1: MLA 低秩压缩 KV, 0=关闭; 建议 32 (dim/4) 或 16")
    p.add_argument("--dep-at", default="",
                   help="net.py: 深度注入到哪几个 Block (0 起算), 如 1,3,5; "
                        "dep 先 LayerNorm 再与特征相加, 然后才进 k_conv; "
                        "给了此参数则 fuse 默认不再吃深度 (除非再给 --keep-fuse-dep)")
    p.add_argument("--dep-on-v", action="store_true",
                   help="net.py: 深度同时进 V 分支 (默认只进 K)")
    p.add_argument("--keep-fuse-dep", action="store_true",
                   help="net.py: 保留 fuse 里的深度相加 (与 --dep-at 同时给 = 两处都注入)")
    p.add_argument("--dep-zero-init", action="store_true",
                   help="net.py: 深度注入支 LN weight 置 0, 初始化时等价于不注入")
    p.add_argument("--strip-k", default="7,15",
                   help="net.py: Fusion 长条卷积的两级核长, 如 7,15(默认) / 5,9 / 3,5")
    p.add_argument("--init-crop", action="store_true",
                   help="init-from 时对形状不一致的 4D 卷积核做中心裁剪, 保留旧权重")
    p.add_argument("--attn-up-mode", default="stack", choices=["stack", "refine"],
                   help="stack = token 堆叠上采样(等价 nearest); refine = bilinear + token 域注意力残差")
    p.add_argument("--attn-up-plain", action="store_true",
                   help="配合 --attn-up-idx: 去掉注意力, 只保留 nearest 上采样 (定位消融)")
    p.add_argument("--attn-up-idx", default="",
                   help="解码器哪几级用 AttnUpSample 代替 bilinear, 逗号分隔, 如 0 或 0,1"
                        " (空=全用 bilinear; 非空时自动走 net.py 的 Net)")
    p.add_argument("--net-py", action="store_true",
                   help="走 net.py 的 Net (128 通道 fuse/decoder), 即使不用注意力上采样")
    p.add_argument("--carafe-idx", default="",
                   help="解码器哪几级用 CARAFE 代替 bilinear, 逗号分隔, 如 0 或 0,1 (走 net.py)")
    p.add_argument("--carafe-k", type=int, default=5, help="CARAFE 重组核大小")
    p.add_argument("--carafe-cm", type=int, default=64, help="CARAFE 通道压缩宽度")
    p.add_argument("--dss-q", action="store_true",
                   help="把 DSS 融合块从 1/16 移到 1/4 (88×88), 1/16 只做深度注入+1×1")
    p.add_argument("--dss-add", action="store_true",
                   help="保留 1/16 的 Fusion, 另在 1/4 加一个 DSS (对照 --dss-q)")
    p.add_argument("--dss-mid", type=int, default=None,
                   help="DSS 软路由的中间层宽度 (默认 max(C//4, 8))")
    p.add_argument("--no-dss-hp", action="store_true",
                   help="DSS 的 BS 分支只吃固定拉普拉斯, 去掉可学习高通 "
                        "(用于和加了 hp 之前的 v7 权重严格对齐)")
    p.add_argument("--bce-weight", type=float, default=0.5, help="BCE 项权重")
    p.add_argument("--aux-edge", action="store_true",
                   help="对 DSS 的 edge_logits / router_logits 加辅助监督 (SASC 原实现)")
    p.add_argument("--aux-stage1-epochs", type=int, default=15, help="强监督持续到第几轮")
    p.add_argument("--lam-router1", type=float, default=0.5)
    p.add_argument("--lam-edge1", type=float, default=0.1)
    p.add_argument("--lam-router2", type=float, default=0.1)
    p.add_argument("--lam-edge2", type=float, default=0.03)
    p.add_argument("--router-k", type=int, default=3, help="router 目标的膨胀/平滑核")
    p.add_argument("--dice-weight", type=float, default=0.5, help="soft Dice 项权重")
    p.add_argument("--no-msf", action="store_true",
                   help="消融: 关掉输入侧多尺度融合(MSF), 回到 stem 下采样 + patch8")
    p.add_argument("--depth-blocks", type=int, default=8, help="DownKV 块数")
    p.add_argument("--patch", type=int, default=16, choices=[4, 8, 16, 32],
                   help="patch 大小: 决定 token 数与解码器级数")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  size {args.size}  bs {args.batch_size}", flush=True)

    cache = not args.no_cache
    if args.ram_cache and args.num_workers > 0:
        print("[warn] --ram-cache incompatible with workers, forcing num_workers=0", flush=True)
        args.num_workers = 0
    train_ds = USOD(args.data, "train", args.size, cache=cache, aug=not args.no_aug,
                    limit=args.train_n, ram=args.ram_cache, edge=args.aux_edge)
    val_ds = USOD(args.data, "val", args.size, cache=cache, aug=False, limit=args.val_n)
    print(f"train {len(train_ds)} imgs | val {len(val_ds)} imgs | cache={cache}", flush=True)

    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=pin,
                          persistent_workers=args.num_workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=pin, persistent_workers=True)

    if args.v1:
        from nn_v1 import NetV1
        dep_at = tuple(int(s) for s in str(args.v1_dep_at).split(",") if s.strip())
        keep_fuse_dep = args.v1_keep_fuse_dep or (not dep_at)   # 空 -> 退回原行为
        model = NetV1(size=args.size, num_classes=1, use_depth=not args.no_depth,
                      depth=args.depth_blocks, same_kv=args.same_kv,
                      dep_blocks=dep_at, keep_fuse_dep=keep_fuse_dep,
                      dep_on_v=args.v1_dep_on_v,
                      latent_rank=args.v1_mla_rank).to(device)
        print(f"v1: depth -> blocks {dep_at or 'None (fuse 相加)'} "
              f"keep_fuse_dep={keep_fuse_dep} dep_on_{'K+V' if args.v1_dep_on_v else 'K'} "
              f"mla_rank={args.v1_mla_rank or 'off'} "
              f"-> {sum(1 for b in model.blocks if b.attn.dep_norm is not None)} 块注入", flush=True)
    elif args.attn_up_idx or args.net_py or args.carafe_idx or args.dss_q or args.dss_add:
        # net.py 是自包含版本, 与 nn_net 逐张量同名同形, 但多了 attn_up_idx
        from net import Net as NetAttn
        idx = tuple(int(s) for s in str(args.attn_up_idx).split(",") if s.strip())
        cidx = tuple(int(s) for s in str(args.carafe_idx).split(",") if s.strip())
        dep_at = tuple(int(s) for s in str(args.dep_at).split(",") if s.strip())
        keep_fuse = args.keep_fuse_dep or (not dep_at)
        strip_k = tuple(int(s) for s in str(args.strip_k).split(",") if s.strip())
        model = NetAttn(size=args.size, num_classes=1, use_depth=not args.no_depth,
                        patch=args.patch, fuse_simple=args.simple_fuse,
                        same_kv=args.same_kv, depth=args.depth_blocks,
                        attn_up_idx=idx, attn_up_plain=args.attn_up_plain,
                        attn_up_mode=args.attn_up_mode,
                        carafe_idx=cidx, carafe_k=args.carafe_k,
                        carafe_cm=args.carafe_cm,
                        dss_quarter=args.dss_q, dss_add=args.dss_add,
                        dss_mid=args.dss_mid,
                        dss_hp=not args.no_dss_hp,
                        dep_blocks=dep_at, dep_on_v=args.dep_on_v,
                        keep_fuse_dep=keep_fuse,
                        dep_zero_init=args.dep_zero_init,
                        strip_k=strip_k).to(device)
        print(f"attn_up_idx {idx} plain={args.attn_up_plain} carafe_idx {cidx} "
              f"(k={args.carafe_k} cm={args.carafe_cm}) dss_q={args.dss_q} "
              f"-> {sum(1 for u in model.decoder.ups if type(u).__name__ == 'AttnUpBlock')}"
              f"/{len(model.decoder.ups)} 级注意力上采样, "
              f"{sum(1 for u in model.decoder.ups if type(u).__name__ == 'CarafeUpBlock')}"
              f" 级 CARAFE, "
              f"fuse={type(model.fuse).__name__}, "
              f"refine={'DSS@' + str(model.decoder.refine.channels) if model.decoder.refine is not None else 'None'}"
              f", depth->blocks {dep_at or 'None'} keep_fuse_dep={keep_fuse} "
              f"dep_on_{'K+V' if args.dep_on_v else 'K'} "
              f"dep_zero_init={getattr(model, 'dep_zero_init', False)} "
              f"strip_k={getattr(model.fuse, 'strip_k', None)}",
              flush=True)
    else:
        model = (UNet4(32) if args.arch == "unet"
                 else Net(size=args.size, num_classes=1, use_depth=not args.no_depth,
                          use_msf=not args.no_msf, patch=args.patch,
                          fuse_simple=args.simple_fuse,
                          same_kv=args.same_kv,
                          depth=args.depth_blocks)).to(device)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        sd = ck.get("model", ck)
        own = model.state_dict()
        if args.init_crop:
            # 只对形状不同的 4D 卷积核做中心裁剪(换小核时保住旧权重的中段)
            n_crop = 0
            for k_, v in list(sd.items()):
                if k_ not in own or v.dim() != 4 or own[k_].dim() != 4:
                    continue
                t = own[k_]
                if v.shape == t.shape or v.shape[:2] != t.shape[:2]:
                    continue
                dh, dw = v.shape[2] - t.shape[2], v.shape[3] - t.shape[3]
                if dh < 0 or dw < 0 or dh % 2 or dw % 2:
                    continue
                h0, w0 = dh // 2, dw // 2
                sd[k_] = v[:, :, h0:h0 + t.shape[2], w0:w0 + t.shape[3]].contiguous()
                n_crop += 1
            if n_crop:
                print(f"  init-crop: {n_crop} 个卷积核按中心裁剪载入", flush=True)
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        skipped = [k for k in sd if k not in keep]
        own.update(keep)
        model.load_state_dict(own)
        print(f"init from {args.init_from} (ep{ck.get('epoch')}): "
              f"loaded {len(keep)}/{len(sd)} tensors", flush=True)
        if skipped:
            print(f"  skipped {len(skipped)}: {skipped[:6]}", flush=True)

    prefixes = [s.strip() for s in args.freeze.split(",") if s.strip()]
    args._freeze_prefixes = prefixes
    n_all = sum(p.numel() for p in model.parameters())
    if prefixes:
        n_fro = apply_freeze(model, prefixes, keep_sub=("dep_norm",))
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze [{','.join(prefixes)}]: frozen {n_fro:,} / {n_all:,}, "
              f"trainable {n_tr:,} ({n_tr / n_all * 100:.1f}%)", flush=True)
        if n_tr == 0:
            raise SystemExit("freeze 把全部参数都冻住了, 检查前缀是否写错")
    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    decay, no_decay, dep_ln = [], [], []
    for n_, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        # dep_norm 是深度注入的通路, 单独给一档 lr: 冻结主干时它是唯一能学的环节,
        # 用 decoder 的 2e-4 容易几轮就把注入幅度推过冲
        if "dep_norm" in n_:
            dep_ln.append(prm)
        elif prm.ndim <= 1:
            no_decay.append(prm)
        else:
            decay.append(prm)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": args.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    base_lr = args.lr_new if args.lr_new > 0 else args.lr
    if dep_ln:
        dn_lr = args.lr_dep_norm if args.lr_dep_norm > 0 else base_lr
        groups.append({"params": dep_ln, "weight_decay": 0.0, "lr": dn_lr})
        print(f"dep_norm: {len(dep_ln)} tensors @ lr {dn_lr:.2e} "
              f"(base {base_lr:.2e})", flush=True)
    optimizer = torch.optim.AdamW(groups, lr=base_lr)
    total_steps = max(1, len(train_ld)) * args.epochs
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    npar = sum(p.numel() for p in model.parameters())
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {npar:,}  trainable: {ntr:,}")
    print(f"steps/epoch: {len(train_ld)}  total: {total_steps}", flush=True)

    hist = {"tag": args.tag, "params": npar, "trainable": ntr,
            "freeze": prefixes, "init_from": args.init_from,
            "size": args.size, "bs": args.batch_size,
            "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
            "use_depth": not args.no_depth, "use_msf": not args.no_msf,
            "fuse_simple": args.simple_fuse,
            "bce_weight": args.bce_weight, "dice_weight": args.dice_weight,
            "attn_up_idx": args.attn_up_idx,
            "depth_blocks": args.depth_blocks, "rows": []}
    best = -1.0
    bad = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, secs, l_r, l_e = train_one_epoch(
            model, train_ld, criterion, optimizer, scheduler, device, epoch, args
        )
        va_loss, va_dice, va_iou = evaluate(model, val_ld, criterion, device)
        el = (time.time() - t_start) / 3600
        if args.aux_edge:
            print(f"[ep{epoch}] train {tr_loss:.4f} (router {l_r:.4f} edge {l_e:.4f}) "
                  f"| val_loss {va_loss:.4f} | dice {va_dice:.4f} | IoU {va_iou:.4f} "
                  f"| {secs/60:.2f} min | 累计 {el:.2f} h", flush=True)
        else:
            print(f"[ep{epoch}] train {tr_loss:.4f} | val_loss {va_loss:.4f} "
                  f"| dice {va_dice:.4f} | IoU {va_iou:.4f} | {secs/60:.2f} min | 累计 {el:.2f} h",
                  flush=True)
        hist["rows"].append(dict(ep=epoch, train=tr_loss, val_loss=va_loss,
                                 dice=va_dice, iou=va_iou, sec=secs))
        json.dump(hist, open(args.out.replace(".pt", "") + ".json", "w"), indent=1)

        if va_iou > best + args.min_delta:
            best = va_iou
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "iou": va_iou}, args.out)
            print(f"  saved -> {args.out} (IoU {va_iou:.4f})", flush=True)
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"early stop: no gain over {best:.4f}+{args.min_delta} "
                      f"for {bad} epochs, stop at ep{epoch}", flush=True)
                break

    print(f"done. best val IoU: {best:.4f}  total {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
