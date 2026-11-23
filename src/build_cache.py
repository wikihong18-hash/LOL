# -*- coding: utf-8 -*-
"""生成 352 尺寸缓存 (rgb352 / depth352 / gt352), 供 USOD(cache=True) 使用。

    python build_cache.py --data /path/to/USOD10K --splits train,val,test

规则与 USOD(cache=False) 完全一致:
  rgb   : BGR 读入 -> resize(LINEAR) -> 原样写盘 (通道顺序不动)
  depth : 1~99 百分位归一化到 0~255 -> resize(LINEAR) -> uint8 写盘
  gt    : 灰度 -> resize(NEAREST) -> 原样写盘
已存在的文件默认跳过, --force 覆盖。
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser("build 352 cache")
    ap.add_argument("--data", required=True, help="USOD10K 根目录")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    s = args.size
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        root = os.path.join(args.data, split)
        gts = sorted(glob.glob(os.path.join(root, "GT", "*")))
        if not gts:
            print(f"[{split}] 无 GT, 跳过")
            continue
        for sub in ("rgb352", "depth352", "gt352"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)

        n = 0
        for gp in gts:
            name = os.path.basename(gp)
            out_r = os.path.join(root, "rgb352", name)
            out_d = os.path.join(root, "depth352", name)
            out_g = os.path.join(root, "gt352", name)
            if not args.force and all(os.path.exists(p) for p in (out_r, out_d, out_g)):
                continue

            rgb = cv2.imread(os.path.join(root, "RGB", name), cv2.IMREAD_COLOR)
            if rgb is None:
                print(f"  缺 RGB: {name}")
                continue
            dep = cv2.imread(os.path.join(root, "depth", name), cv2.IMREAD_UNCHANGED)
            if dep is None:
                print(f"  缺 depth: {name}")
                continue
            if dep.ndim == 3:
                dep = dep.max(axis=2)
            dep = dep.astype(np.float32)
            lo, hi = np.percentile(dep, 1), np.percentile(dep, 99)
            dep = np.clip((dep - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)

            rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, (s, s), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (s, s), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(out_r, rgb)
            cv2.imwrite(out_d, dep.astype(np.uint8))
            cv2.imwrite(out_g, gt)
            n += 1
            if n % 500 == 0:
                print(f"  [{split}] {n}/{len(gts)}", flush=True)
        print(f"[{split}] 新增 {n} 张, 共 {len(gts)} 张")


if __name__ == "__main__":
    main()
