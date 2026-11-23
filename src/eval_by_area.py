# -*- coding: utf-8 -*-
import os, sys, time, numpy as np, torch
from torch.utils.data import DataLoader
from train import USOD
from net import Net

DEV = "cuda"
DATA = os.environ.get("LOL_DATA", "E:/vscode/SASC-USOD-main/data/USOD10K")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# 命令行: python eval_by_area.py [limit] "tag|ckpt|strip_k|dss" ...
#   例: "v8s59|net30_v8s59.pt|5,9|1"
if len(sys.argv) > 2:
    MODELS = []
    for spec in sys.argv[2:]:
        parts = spec.split("|")
        tag, ckpt = parts[0], parts[1]
        sk = tuple(int(x) for x in parts[2].split(",")) if len(parts) > 2 and parts[2] else (7, 15)
        dss = (len(parts) <= 3) or parts[3] != "0"
        MODELS.append((tag, ckpt, dss, sk))
else:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS = [("LOL", os.path.join(_ROOT, "weights", "LOL.pt"), True, (7, 15))]
BINS = [(0.0, 0.01, "XS <1%"), (0.01, 0.05, "S 1-5%"),
        (0.05, 0.20, "M 5-20%"), (0.20, 1.01, "L >20%")]

t0 = time.time()
ds = USOD(DATA, "test", 352, cache=True, aug=False, limit=LIMIT)
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
print("test n=%d  build %.1fs" % (len(ds), time.time() - t0), flush=True)

res = {}
for tag, ckpt, dss, sk in MODELS:
    m = Net(size=352, patch=16, use_depth=True, dss_add=dss, strip_k=sk).to(DEV).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    r = m.load_state_dict(sd, strict=False)
    if dss and len(r.missing_keys) > 0:
        b = m.decoder.refine.hp.block
        b[4].weight.data.zero_(); b[4].bias.data.zero_()
        print("  -> hp 末级 BN 置零, 退化为纯拉普拉斯 (missing %d)" % len(r.missing_keys), flush=True)
    st = {k: [0, 0.0, 0.0, 0.0] for _, _, k in BINS}
    t1 = time.time(); n = 0
    with torch.no_grad():
        for rgb, dep, mask in ld:
            p = (torch.sigmoid(m(rgb.to(DEV), dep.to(DEV))) > 0.5).float().cpu()
            area = mask.flatten(1).mean(1)
            for i in range(mask.shape[0]):
                a = float(area[i])
                for lo, hi, nm in BINS:
                    if lo <= a < hi:
                        pi, gi = p[i].flatten(), mask[i].flatten()
                        inter = float((pi * gi).sum()); uni = float(((pi + gi) > 0).sum())
                        st[nm][0] += 1; st[nm][1] += inter; st[nm][2] += uni
                        st[nm][3] += inter / uni if uni > 0 else 1.0
                        break
            n += mask.shape[0]
            if n % 400 < 8:
                print("  %s %d/%d  %.1fs" % (tag, n, len(ds), time.time() - t1), flush=True)
    res[tag] = st
    print("[%s] missing %d unexpected %d | %.1fs" % (tag, len(r.missing_keys), len(r.unexpected_keys), time.time() - t1), flush=True)
    del m; torch.cuda.empty_cache()

print("\npooled IoU (交集总和/并集总和)")
print("%-9s" % "area" + "".join("%18s" % t for t, *_ in MODELS), flush=True)
for _, _, nm in BINS:
    line = "%-9s" % nm
    for tag, *_ in MODELS:
        s = res[tag][nm]
        line += "%18s" % ("%.4f (n=%d)" % (s[1] / max(s[2], 1), s[0]) if s[0] else "-")
    print(line, flush=True)
line = "%-9s" % "ALL"
for tag, *_ in MODELS:
    s = res[tag]
    i = sum(v[1] for v in s.values()); u = sum(v[2] for v in s.values()); c = sum(v[0] for v in s.values())
    line += "%18s" % ("%.4f (n=%d)" % (i / max(u, 1), c))
print(line, flush=True)

print("\nper-image mean IoU")
print("%-9s" % "area" + "".join("%18s" % t for t, *_ in MODELS), flush=True)
for _, _, nm in BINS:
    line = "%-9s" % nm
    for tag, *_ in MODELS:
        s = res[tag][nm]
        line += "%18s" % ("%.4f" % (s[3] / s[0]) if s[0] else "-")
    print(line, flush=True)
