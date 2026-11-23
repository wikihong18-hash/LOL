# LOL · Locate with Once Look

RGB-D 显著目标检测（USOD-10K）轻量模型。单次前视即可定位显著目标：RGB 与深度各过一遍主干，在 1/16 分辨率融合，解码回全分辨率。

- 参数量 **3,211,947**（推理全参；训练时只更新 812,259）
- 输入 352×352，输出同尺寸概率图
- 权重：`weights/LOL.pt`（epoch 13，val IoU 0.7859）

## 结果（USOD10K test，1026 张）

| MAE ↓ | maxFm ↑ | meanFm ↑ | maxEm ↑ | meanEm ↑ | Sm ↑ | AP ↑ | AUC ↑ |
|---|---|---|---|---|---|---|---|
| **0.0353** | **0.8737** | **0.8608** | **0.9351** | **0.9290** | **0.8833** | **0.8894** | **0.9637** |

按 GT 面积分档（per-image mean IoU）：

| 面积档 | n | IoU |
|---|---|---|
| XS <1% | 40 | 0.5378 |
| S 1–5% | 279 | 0.7544 |
| M 5–20% | 443 | 0.8204 |
| L >20% | 264 | 0.8148 |
| 全部（pooled） | 1026 | 0.7821 |

小目标（<1%）是明确短板，pooled 值会被少数大预测拉高，评估请用 per-image mean。

## 架构

```
RGB ──► MSF(1/2,1/4) ──► patch_embed 16×16 ──► 8 × Block ──► LayerNorm ──┐
                                      (22×22, dim128)                     ├─► Fusion ──► Decoder(4 级) ──► mask
Depth ──► DepthStem ────────────────────────────────────────────────────┘
```

- **Block**：PVT 风格 SRA 的简化版。484 个 query token（22×22），K/V 用 2×2 stride2 卷积下采样到 121 个（11×11），4 头，head_dim 32。注意力之后接 GELU MLP（ratio 4）。
- **Fusion**（1/16 分辨率）：`f1 + depth → 3×3 → 1×1 升通道`，然后双分支并联：
  - 分支 A：固定 3×3 拉普拉斯算子（depthwise，无参数）→ 3×3 卷积
  - 分支 B：可学习长条卷积 `1×7 → 7×1 → 1×15 → 15×1`，每级 dw + 1×1 + BN + ReLU
  - concat(256) → 3×3 depthwise → 1×1 降回 128
  - 15-tap 在 22×22 上覆盖 68% 宽度，用极低参数代价换全局线状上下文，对小目标尤其关键（缩短核长的消融在所有面积档均为负收益）。
- **DSS**（1/4 分辨率）：软路由把特征分成细节支（拉普拉斯）与区域支，逐像素权重 `w` 由轻量 router 输出，末层零初始化，训练从"不拆分"平滑开始。
- **Decoder**：4 级上采样，其中两级为注意力上采样块，末端 DSS refine（64 通道）。

## 用法

### 推理

```bash
cd src

# 单张（LOL 是 RGB-D 模型，深度必填）
python predict.py --rgb a.jpg --depth a.png --out a_pred.png

# 批量：RGB 与深度按同名配对
python predict.py --rgb-dir RGB --depth-dir depth --out-dir preds
```

深度会先做 1–99 百分位归一化再 resize（与训练一致），输出 PNG 为 0–255 概率，尺寸还原到原图。

### 评测

```bash
cd src
python eval_sod_metrics.py --ckpt ../weights/LOL.pt \
    --data /path/to/USOD10K --split test --tag LOL \
    --patch 16 --depth-blocks 8 --dss-add --no-dss-hp
```

评测走 352 缓存（`rgb352/`、`depth352/`、`gt352/`），没有缓存先建一次：

```bash
python build_cache.py --data /path/to/USOD10K --splits train,val,test
```

`--no-dss-hp` 必须加：v7 权重不含 `LearnableHighPass`，否则结构不匹配。

面积分档：

```bash
LOL_DATA=/path/to/USOD10K python eval_by_area.py 0 "LOL|../weights/LOL.pt|7,15|1"
```

### 训练复现

```bash
cd src
python train.py --net-py --patch 16 --depth-blocks 8 --dss-add --no-dss-hp \
    --init-from <v6.1.pt> --freeze msf,patch_embed,depth_stem,blocks,norm \
    --epochs 20 --batch-size 8 --lr 2e-4 \
    --bce-weight 0.25 --dice-weight 0.75 --seed 1 \
    --tag LOL --out LOL.pt
```

原始训练是两阶段：v6.1 全量训练后被冻结，只训 decoder 20 轮（lr 2e-4，AdamW，cosine + warmup，BCE 0.25 + Dice 0.75）。包内不含 v6.1 起点；没有它就只能从零训练，50 轮约到 val 0.7783 / Sm 0.8758，比 warm start 低约 0.008。

## 目录

```
LOL/
├── src/
│   ├── net.py                 模型定义（Net / Fusion / Block / DSSBlock / MaskDecoder）
│   ├── nn_net.py              早期版本与 DiceBCELoss（被 train/eval 引用）
│   ├── train.py               训练脚本 + USOD 数据集
│   ├── eval_sod_metrics.py    六项指标（官方 Eval_thread）
│   ├── eval_by_area.py        按 GT 面积分档
│   ├── build_cache.py         生成 352 缓存 (rgb352/depth352/gt352)
│   └── predict.py             推理（单图 / 批量）
├── weights/
│   ├── LOL.pt                 seed 1，最终权重
│   └── LOL_seed0.pt           seed 0，用于复核种子波动
├── configs/LOL_config.json    完整结构与训练配置
├── vendor/evaluation/         指标评测器
└── results/metrics.txt
```

## 依赖

Python 3.10+，`torch>=2.0`、`torchvision`、`opencv-python`、`numpy`、`pillow`。推理 3.2M 参数，单张 352 输入在笔记本 5060 上约 16 ms（不含前后处理）。
