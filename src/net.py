"""RGB-D 分割网络 — 自包含单文件版本, 不依赖 blocks.py。

这是与 blocks.py 那套(GroupedAttention / 4 向量 / 交叉注意力)完全不同的另一个网络:
主干仍是 8 个 pre-norm DownKV Block 串行堆叠, 没有向量池化, 也没有交叉注意力。

流程 (默认 size=352, patch=16, grid=22x22):

    rgb ── MultiScaleFusion ── 1/1·1/2·1/4 三层 → 由小到大上采样融合 → 32ch @1/1
        ── PatchEmbed ──────── depthwise patch(k=patch,s=patch) + 1x1 → 128ch + sincos 位置嵌入
        ── 8 × Block ───────── pre-norm 残差, 块内 DownKVAttention (KV 用 2x2 步长卷积下采样)
        ── LayerNorm ───────── f1 (B, 128, gh, gw)

    depth ─ DepthStem ──────── 1/4 下采样 + 切块平均池化 → 128ch @grid

    f1 + depth ── fuse ────── 3x3 → 1x1 (128 通道) → 拉普拉斯分支 ∥ 长条分支 → concat → 3x3 → 1x1
              ── MaskDecoder ─ 逐级 ×2 上采样, 在 1/4 与 1/2 处接 MSF 的 skip → mask

与 nn_net.py 的关系: 模块属性名和内部结构逐一同名同形, nn_net 训练出来的 checkpoint
可以 100% 载入本文件, 一个张量都不丢。

冻结: 用 model.freeze("msf", "patch_embed", "depth_stem", "blocks", "norm", "fuse")
      只训练 decoder; BN 的 running stats 需要在每个 epoch 开头额外切到 eval, 见 train.py。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 基础组件
# ----------------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    """Conv2d(no bias) -> BatchNorm2d -> ReLU"""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


def _sincos_1d(pos, dim, temperature=10000.0):
    """1D sine-cosine embedding: (L,) -> (L, dim)."""
    omega = 1.0 / (temperature ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2)))
    out = pos[:, None].float() * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1)


POS_EMBED_SCALE = 0.05   # sincos 原始 std≈0.56; 乘 0.05 后 ≈0.028, 与 patch 投影输出同量级


def build_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """固定(不可训练) 2D sincos 位置嵌入, 返回 (1, C, gh, gw)。"""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half = embed_dim // 2
    emb_h = _sincos_1d(torch.arange(grid_h), half)   # (gh, C/2)
    emb_w = _sincos_1d(torch.arange(grid_w), half)   # (gw, C/2)
    pos = torch.cat(
        [emb_h[:, None, :].expand(grid_h, grid_w, half),
         emb_w[None, :, :].expand(grid_h, grid_w, half)],
        dim=-1,
    )                                                # (gh, gw, C)
    pos = pos * POS_EMBED_SCALE
    return pos.permute(2, 0, 1).unsqueeze(0)         # (1, C, gh, gw)


class MultiScaleFusion(nn.Module):
    """RGB -> 1/1、1/2、1/4 三层特征, 再由小到大上采样融合回 1/1。

        c1 : in_ch -> w1   k3 s1   (H)
        c2 : w1    -> w2   k3 s2   (H/2)
        c3 : w2    -> w3   k3 s2   (H/4)
        g2 = fuse2(cat[up(c3), c2])   -> (H/2, w2)
        g1 = fuse1(cat[up(g2), c1])   -> (H,   w1)
        out = 3x3(g1)                 -> (H,   out_ch)

    1/1 分辨率上做 3x3 很贵, 所以两级融合都用 1x1, 只在最后做一次 3x3。
    return_scales=True 时额外返回 (f1, g2, f3), 直接当 decoder 的 1/1、1/2、1/4 skip。
    """

    def __init__(self, in_ch=3, widths=(8, 16, 32), out_ch=16):
        super().__init__()
        w1, w2, w3 = widths
        self.c1 = ConvBNReLU(in_ch, w1, 3, 1, 1)      # 1/1
        self.c2 = ConvBNReLU(w1, w2, 3, 2, 1)         # 1/2
        self.c3 = ConvBNReLU(w2, w3, 3, 2, 1)         # 1/4
        self.fuse2 = nn.Sequential(
            nn.Conv2d(w3 + w2, w2, 1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(w2 + w1, w1, 1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )
        self.out = ConvBNReLU(w1, out_ch, 3, 1, 1)
        self.out_channels = out_ch

    def forward(self, x, return_scales=False):
        f1 = self.c1(x)                                            # 1/1
        f2 = self.c2(f1)                                           # 1/2
        f3 = self.c3(f2)                                           # 1/4

        u3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        g2 = self.fuse2(torch.cat([u3, f2], dim=1))                # 1/2
        u2 = F.interpolate(g2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        g1 = self.fuse1(torch.cat([u2, f1], dim=1))                # 1/1
        out = self.out(g1)                                         # (B, out_ch, H, W)

        return (out, (f1, g2, f3)) if return_scales else out


class PatchEmbed(nn.Module):
    """1/1 分辨率特征直接切 patch -> (B, dim, gh, gw), 加固定 2D sincos 位置嵌入。

    投影用 depthwise patch + 1x1: 每个输入通道独立在 patch 内做线性组合, 再跨通道混合。
    参数量 in_ch*patch*patch + in_ch*dim, 比 dense 版本小两个数量级。
    """

    def __init__(self, in_ch=16, dim=128, grid=(11, 11), patch=32, dense=False):
        super().__init__()
        self.patch_size = patch
        if dense:
            self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch, padding=0)
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=patch, stride=patch,
                          padding=0, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, dim, kernel_size=1, bias=True),
            )
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(dim, *grid),
            persistent=False,
        )

    def forward(self, x):
        """-> (B, dim, gh, gw), 不展平(与 nn_net 保持一致, 展平在主流程里做)"""
        out = self.proj(x)
        pe = self.pos_embed
        if pe.shape[-2:] != out.shape[-2:]:
            pe = F.interpolate(pe, size=out.shape[-2:], mode="bilinear", align_corners=False)
        return out + pe


class DownKVAttention(nn.Module):
    """多头注意力, 但 K/V 不做线性投影。

    Q: 正常 Linear 投影。
    K/V: token 先还原成 2D 特征图 (B, C, gh, gw), 再用 kernel=stride=2 的卷积下采样 1/2,
         展平后得到 N/4 个 token。softmax 只在 N/4 个 key 上做。

    kv_kernel=2 -> KV 数量 1/4 (省算力); kv_kernel=1 -> 与 Q 同分辨率。
    """

    def __init__(self, dim, num_heads=4, grid=(11, 11), kv_kernel=2, dep_norm=False,
                 dep_on_v=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid = grid
        self.kv_kernel = kv_kernel
        # dep_norm=True: 建一个 LayerNorm, 把外部注入的深度图归一化后再与特征相加
        self.dep_norm = nn.LayerNorm(dim) if dep_norm else None
        # dep_on_v=True: V 分支也加同一份归一化深度(否则只影响 K 的相似度)
        self.dep_on_v = dep_on_v
        self.q = nn.Linear(dim, dim)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=kv_kernel, stride=kv_kernel)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=kv_kernel, stride=kv_kernel)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _tokens_to_map(x, gh, gw):
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, gh, gw)

    def _ln_map(self, d):
        """(B,C,H,W) 逐像素 LayerNorm(按通道), 把深度支路拉到与 LN 后特征同量级。"""
        b, c, h, w = d.shape
        return self.dep_norm(d.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x, grid=None, dep=None):
        b, n, c = x.shape
        gh, gw = grid if grid is not None else self.grid
        assert n == gh * gw, f"grid {gh}x{gw} needs {gh * gw} tokens, got {n}"

        q = self.q(x)                                        # (B, N, C)
        feat = self._tokens_to_map(x, gh, gw)                # (B, C, gh, gw)
        feat_k = feat
        feat_v = feat
        if dep is not None:
            # 深度先归一化, 再与特征相加, 然后才进 k_conv (先加后投影)
            dep_n = self._ln_map(dep) if self.dep_norm is not None else dep
            feat_k = feat + dep_n
            if self.dep_on_v:
                feat_v = feat + dep_n
        k = self.k_conv(feat_k).flatten(2).transpose(1, 2)   # (B, N/4, C)
        v = self.v_conv(feat_v).flatten(2).transpose(1, 2)   # (B, N/4, C)
        n_kv = k.shape[1]

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)      # (B,H,N,Dh)
        k = k.view(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,N/4,Dh)
        v = v.view(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)        # (B,H,N,Dh)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Block(nn.Module):
    """pre-norm 残差块: x + Attn(LN(x)), x + MLP(LN(x))。"""

    def __init__(self, dim, num_heads=4, grid=(11, 11), mlp_ratio=4.0, drop=0.0,
                 same_kv=False, dep_norm=False, dep_on_v=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DownKVAttention(dim=dim, num_heads=num_heads, grid=grid,
                                    kv_kernel=1 if same_kv else 2,
                                    dep_norm=dep_norm, dep_on_v=dep_on_v)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, grid=None, dep=None):
        x = x + self.attn(self.norm1(x), grid=grid, dep=dep)
        x = x + self.mlp(self.norm2(x))
        return x


class DepthStem(nn.Module):
    """深度图 (B,1,H,W) -> 1/2 -> 1/4 -> 切块平均池化 -> (B, C, gh, gw)。

    out_bn 用来对齐两条分支的幅度, 否则 depth 分支的幅度只有 rgb 的 3% 左右。
    """

    def __init__(self, widths=(16, 16), out_channels=128, grid=(11, 11)):
        super().__init__()
        c1, c2 = widths
        self.body = nn.Sequential(
            ConvBNReLU(1, c1, k=3, s=2, p=1),    # H/2
            ConvBNReLU(c1, c2, k=3, s=2, p=1),   # H/4
        )
        self.proj = nn.Conv2d(c2, out_channels, kernel_size=1)
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.grid = grid
        self.out_channels = out_channels

    def forward(self, d, grid=None):
        f = self.body(d)                               # (B, c2, H/4, W/4)
        gh, gw = grid if grid is not None else self.grid
        h, w = f.shape[-2:]
        if h % gh == 0 and w % gw == 0:
            kh, kw = h // gh, w // gw
            f = F.avg_pool2d(f, kernel_size=(kh, kw), stride=(kh, kw))   # 切块平均池化
        else:
            f = F.adaptive_avg_pool2d(f, (gh, gw))
        return self.out_bn(self.proj(f))                # (B, C, gh, gw)


def _laplacian_weight(channels):
    """固定 3x3 拉普拉斯算子, 复制到每个通道 -> (C, 1, 3, 3), 用于 depthwise conv。"""
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    return k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)


def _ds_conv(ch, k, p):
    """depthwise(k x k) + pointwise(1x1), 空间尺寸与通道数都不变。"""
    return nn.Sequential(
        nn.Conv2d(ch, ch, k, 1, p, groups=ch, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 1, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
    )


class Fusion(nn.Module):
    """(f1 + depth) -> 3x3 -> 1x1 升通道 -> 双分支 -> concat -> 3x3 -> 1x1 降通道。

    分支 A: 固定 3x3 拉普拉斯算子(depthwise, 无参数, 不参与训练) -> 3x3 卷积, 尺寸不变
    分支 B: 1xk0 -> k0x1 -> 1xk1 -> k1x1, 每级 depthwise + 1x1, 尺寸不变 (默认 7/15)
    concat 后 3x3(depthwise) 保持 2C, 最后 1x1 降回 C。

    use_depth=False 时直接走 rgb, 不做加法(depth 通路整体移除)。
    simple=True 时只留 3x3 + 1x1 的消融版本。
    """

    def __init__(self, dim=128, out_channels=128, use_depth=True, simple=False,
                 strip_k=(7, 15)):
        super().__init__()
        self.use_depth = use_depth
        self.simple = simple
        self.strip_k = tuple(strip_k)
        # (f1 + depth) -> 3x3 尺寸不变 -> 1x1 升通道
        self.stem = nn.Sequential(
            ConvBNReLU(dim, dim, 3, 1, 1),
            nn.Conv2d(dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if simple:
            # 消融: 去掉拉普拉斯与长条分支, 只留 3x3 + 1x1
            self.tail = nn.Sequential(
                ConvBNReLU(out_channels, out_channels, 3, 1, 1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            )
            return
        # 分支 A: 固定拉普拉斯 (不参与训练) + 3x3
        self.register_buffer("lap_kernel", _laplacian_weight(out_channels), persistent=False)
        self.lap_conv = ConvBNReLU(out_channels, out_channels, 3, 1, 1)
        # 分支 B: 横竖长条卷积, 每级 depthwise + 1x1
        k0, k1 = self.strip_k
        self.strip = nn.Sequential(
            _ds_conv(out_channels, (1, k0), (0, k0 // 2)),
            _ds_conv(out_channels, (k0, 1), (k0 // 2, 0)),
            _ds_conv(out_channels, (1, k1), (0, k1 // 2)),
            _ds_conv(out_channels, (k1, 1), (k1 // 2, 0)),
        )
        # concat -> 3x3(depthwise) 尺寸不变 -> 1x1 降通道
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 3, 1, 1,
                      groups=out_channels * 2, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
        )

    def forward(self, rgb, dep=None):
        x = rgb + dep if (self.use_depth and dep is not None) else rgb
        x = self.stem(x)                                                  # 3x3 -> 1x1 升通道
        if self.simple:
            return self.tail(x)                                            # 消融: 3x3 + 1x1
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=x.shape[1])   # 固定拉普拉斯算子
        a = self.lap_conv(lap)                                            # 3x3
        b = self.strip(x)                                                 # 1x7 -> 7x1 -> 1x15 -> 15x1
        return self.merge(torch.cat([a, b], dim=1))                        # concat -> 3x3 -> 1x1


class DepthInject(nn.Module):
    """深度注入 + 只保留 Fusion 的 stem(3×3 → 1×1 升通道 → BN → ReLU)。

    DSS 搬到 1/4 之后, 1/16 这一级只剩"把深度加进来并投影"。
    stem 的键名与 Fusion 完全一致, 因此 v6.1 的 fuse.stem.* 权重可直接载入,
    起点严格等于改动前; 唯一区别是少了 lap_conv / strip / merge 三支。
    """

    def __init__(self, dim=128, out_channels=128, use_depth=True):
        super().__init__()
        self.use_depth = use_depth
        self.stem = nn.Sequential(
            ConvBNReLU(dim, dim, 3, 1, 1),
            nn.Conv2d(dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, dep=None):
        x = rgb + dep if (self.use_depth and dep is not None) else rgb
        return self.stem(x)


class LearnableHighPass(nn.Module):
    """可学习高通: 3x3 depthwise -> BN -> ReLU -> 1x1 -> BN -> ReLU (原论文写法)。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DSSBlock(nn.Module):
    """动态结构特化块 (SASC 的 DSS): 双分支 + 逐像素软路由。

    RC 区域一致性: 1×7 → 7×1 → 1×15 → 15×1 (depthwise), 加残差
    BS 边界敏感  : 固定拉普拉斯(无参数) 与 x 拼接 → 1×1 → 3×3, 另吐 1 通道 edge logits
    SC 软路由    : [x, sigmoid(edge), mean_c|f_bs − f_rc|] → 1×1 → 3×3 → 1×1 → 1

    out = w · f_bs + (1 − w) · f_rc,  w = sigmoid(router(...))
    """

    def __init__(self, channels=32, mid=None, residual=True, use_edge_head=True,
                 use_hp=True):
        super().__init__()
        self.channels = channels
        self.residual = residual
        # 两支都产出"增量", 末位 1×1 置零(后面不接 ReLU/BN, 否则 0 点梯度为 0 会锁死整支)
        self.rc = nn.Sequential(
            _ds_conv(channels, (1, 7), (0, 3)),
            _ds_conv(channels, (7, 1), (3, 0)),
            _ds_conv(channels, (1, 15), (0, 7)),
            _ds_conv(channels, (15, 1), (7, 0)),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.register_buffer("lap_kernel", _laplacian_weight(channels),
                             persistent=False)
        # use_hp=False: BS 分支只吃固定拉普拉斯, 与 v7 权重严格对齐(无多余 hp 参数)
        self.hp = LearnableHighPass(channels) if use_hp else None
        # 末尾 1×1 置零 -> 起点 f_bs = x, 但 BN/3×3 前段仍能拿到梯度
        # (如果直接把第一层置零, 整支永远梯度为 0, 实测 20 轮后 bs.0 仍是全 0)
        self.bs = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            ConvBNReLU(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.edge_head = nn.Conv2d(channels, 1, kernel_size=1) if use_edge_head else None
        m = mid or max(channels // 4, 8)
        self.router = nn.Sequential(
            nn.Conv2d(channels + 2, m, kernel_size=1, bias=False),
            nn.BatchNorm2d(m), nn.ReLU(inplace=True),
            nn.Conv2d(m, m, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(m), nn.ReLU(inplace=True),
            nn.Conv2d(m, 1, kernel_size=1),
        )

    def forward(self, x, return_aux=False):
        d_rc = self.rc(x)                        # 末层置零 -> 起点 0
        f_rc = x + d_rc if self.residual else d_rc
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=x.shape[1])
        # 可学习高通与固定拉普拉斯并联(use_hp=False 时只有固定拉普拉斯)
        hf = lap + self.hp(x) if self.hp is not None else lap
        d_bs = self.bs(torch.cat([x, hf], dim=1))         # 末层置零 -> 起点 0
        f_bs = x + d_bs
        if self.edge_head is not None:
            edge_logits = self.edge_head(f_bs)
            edge = torch.sigmoid(edge_logits)
        else:
            edge_logits = None
            edge = lap.abs().mean(dim=1, keepdim=True)
        diff = (f_bs - f_rc).abs().mean(dim=1, keepdim=True)
        logits = self.router(torch.cat([x, edge, diff], dim=1))
        w = torch.sigmoid(logits)
        # w 起点 0.5: 两支各拿一半梯度。若把 router bias 设成 -4 (w≈0.018),
        # 两支的有效梯度被压 55 倍, 20 轮内根本醒不过来。
        out = w * f_bs + (1.0 - w) * f_rc
        if return_aux:
            return out, {"w": w, "edge": edge, "router_logits": logits,
                         "edge_logits": edge_logits,
                         "f_rc": f_rc, "f_bs": f_bs}
        return out


class AttnUpSample(nn.Module):
    """用注意力把 token 序列翻倍两次: L -> 2L -> 4L, 空间 (H,W) -> (2H,2W)。

    每一级:
        f2 = DownKV(LN(f))     长度不变, 但每个 token 用下采样的 KV 重新聚合了全局信息
        f3 = cat[f, f2]        长度 2L
        2D 化                  (2H, W) 或 (H, 2W)
        depthwise separable    3x3 dw + 1x1, 把拼接接缝两侧混合起来

    两级分别沿 height、width 方向翻倍, 所以最终正好是空间 2x 上采样。
    要求输入的 gh、gw 都是偶数(DownKV 的 KV 要能整除 2)。

    两级各自带一个独立的 DownKVAttention 和一个 LayerNorm, 参数不共享。

    两次拼接是"原地堆叠"而不是插值: 前 L 个 token 是原图, 后 L 个是注意力重采样图,
    形状上像上下/左右两张图拼在一起, 靠后面的 depthwise 3x3 把接缝抹平。
    """

    def __init__(self, dim=128, num_heads=4, kv_kernel=2, first_axis="h",
                 use_attn=True):
        super().__init__()
        assert first_axis in ("h", "w"), "first_axis must be 'h' or 'w'"
        self.first_axis = first_axis
        self.use_attn = use_attn
        if use_attn:
            # 两级各有独立的注意力块: 第一级 grid=(gh,gw), 第二级 grid=(h1,w1)
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = DownKVAttention(dim, num_heads, kv_kernel=kv_kernel)
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = DownKVAttention(dim, num_heads, kv_kernel=kv_kernel)
        else:
            # 消融: 纯 nearest 上采样, 每 token 复制成 2x2 块, 无注意力
            self.norm1 = self.attn1 = self.norm2 = self.attn2 = None
        self.dw1 = _ds_conv(dim, 3, 1)
        self.dw2 = _ds_conv(dim, 3, 1)

    @staticmethod
    def _to_2d(t, h, w):
        b, n, c = t.shape
        return t.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x, grid):
        """x: (B, N, C) tokens, grid=(gh,gw) -> (B, C, 2gh, 2gw)"""
        b, n, c = x.shape
        gh, gw = grid
        assert n == gh * gw, f"grid {gh}x{gw} needs {gh * gw} tokens, got {n}"
        assert gh % 2 == 0 and gw % 2 == 0, "grid must be even for KV downsample"

        # --- 第一级: 沿 first_axis 翻倍 ---
        a1 = self.attn1(self.norm1(x), grid=(gh, gw)) if self.use_attn else x
        if self.first_axis == "h":
            h1, w1 = gh * 2, gw
        else:
            h1, w1 = gh, gw * 2
        s1 = self._to_2d(torch.cat([x, a1], dim=1), h1, w1)      # (B, C, h1, w1)
        s1 = self.dw1(s1)
        t1 = s1.flatten(2).transpose(1, 2)                       # (B, 2L, C)

        # --- 第二级: 沿另一个轴翻倍 ---
        a2 = self.attn2(self.norm2(t1), grid=(h1, w1)) if self.use_attn else t1
        if self.first_axis == "h":
            h2, w2 = h1, w1 * 2
        else:
            h2, w2 = h1 * 2, w1
        s2 = self._to_2d(torch.cat([t1, a2], dim=1), h2, w2)     # (B, C, 2gh, 2gw)
        return self.dw2(s2)


class RefineUpBlock(nn.Module):
    """bilinear x2 + 上采样前在 token 域做一次注意力残差细化 (合理版)。

        t = t + Attn(LN(t))     长度不变, 每个 token 用下采样的 KV 聚合全局信息
        2D 化 -> bilinear 2x   空间语义与普通 UpBlock 完全一致
        3x3 -> 可选 skip 融合

    与 AttnUpBlock 的区别: 不把 bilinear 换成"token 堆叠"(那等价于 nearest),
    所以保留了插值的平滑性, 注意力只补长程上下文。
    参数比 AttnUpBlock 少一半 (一组 attn 而不是两组)。
    """

    def __init__(self, in_ch, out_ch, skip_ch=0, num_heads=4, kv_kernel=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.attn = DownKVAttention(in_ch, num_heads, kv_kernel=kv_kernel)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if skip_ch > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_fuse = None

    def forward(self, x, grid=None, skip=None):
        b, c, h, w = x.shape
        g = grid if grid is not None else (h, w)
        t = x.flatten(2).transpose(1, 2)                       # (B, N, C)
        t = t + self.attn(self.norm(t), grid=g)                # 残差细化, 长度不变
        x = t.transpose(1, 2).reshape(b, c, h, w)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        if skip is not None and self.skip_fuse is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)
            x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return x


class AttnUpBlock(nn.Module):
    """把 UpBlock 里的 bilinear x2 换成 AttnUpSample, 之后照旧 3x3 + 可选 skip。

    AttnUpSample 在 token 域把 L 拆成 4L, 输出已经是空间 2x; 剩下的 3x3 / skip 融合
    与普通 UpBlock 完全一致, 所以两种块可以直接互换。
    """

    def __init__(self, in_ch, out_ch, skip_ch=0, num_heads=4, kv_kernel=2,
                 use_attn=True):
        super().__init__()
        self.up = AttnUpSample(in_ch, num_heads, kv_kernel, use_attn=use_attn)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if skip_ch > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_fuse = None

    def forward(self, x, grid=None, skip=None):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)                    # (B, N, C)
        x = self.up(t, grid if grid is not None else (h, w))
        x = self.conv(x)
        if skip is not None and self.skip_fuse is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)
            x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return x


class CARAFE(nn.Module):
    """内容自适应上采样 (CARAFE, ICCV 2019 的精简实现)。

    x (B,C,H,W) -> 每个输出子像素位置预测一组 k x k 重组核 -> 邻域加权
                -> (B, C, up*H, up*W)

    权重图分辨率与输入相同, 通道数 up^2 * k^2: 前 up^2 组对应输出 2x2 里的
    四个子像素位置, 每组 k^2 个通道在该位置的 k x k 邻域内做 softmax。
    成本 O(H*W*k^2), 与 token 数的平方无关, 所以高分辨率也能跑。
    """

    def __init__(self, channels, up=2, k=5, cm=64, enc_k=3):
        super().__init__()
        self.up = up
        self.k = k
        self.comp = nn.Conv2d(channels, cm, 1)                    # 通道压缩
        self.enc = nn.Conv2d(cm, up * up * k * k, enc_k, 1, enc_k // 2,
                             bias=False)                          # 核预测
        nn.init.zeros_(self.enc.weight)

    def forward(self, x):
        b, c, h, w = x.shape
        k, up = self.k, self.up

        wgt = self.enc(F.relu(self.comp(x)))                      # (B, up^2*k^2, H, W)
        wgt = wgt.view(b, up * up, k * k, h * w)
        wgt = F.softmax(wgt, dim=2)                               # 每个子像素位置 k^2 归一化

        pat = F.unfold(x, k, padding=k // 2)                      # (B, C*k^2, H*W)
        pat = pat.view(b, c, k * k, h * w)

        # 逐位置做 (up^2, k^2) x (k^2, C), 比 4 维 einsum 快
        wm = wgt.permute(0, 3, 1, 2).reshape(b * h * w, up * up, k * k)
        pm = pat.permute(0, 3, 1, 2).reshape(b * h * w, c, k * k)
        out = torch.bmm(wm, pm.transpose(1, 2))                   # (B*H*W, up^2, C)

        out = out.view(b, h, w, up, up, c)
        # -> (B, C, H, up_a, W, up_b) 即 pixel-shuffle 排列
        return out.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h * up, w * up)


class CarafeUpBlock(nn.Module):
    """把 UpBlock 的 bilinear 换成 CARAFE, 其余 (3x3+BN+ReLU, skip concat) 完全一致。

    与 AttnUpBlock 的区别: 不增加 token 数, 不做全局聚合, 只按局部内容重组邻域。
    成本线性, 可以用在高分辨率级。
    """

    def __init__(self, in_ch, out_ch, skip_ch=0, up=2, k=5, cm=64):
        super().__init__()
        self.carafe = CARAFE(in_ch, up=up, k=k, cm=cm)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if skip_ch > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_fuse = None

    def forward(self, x, skip=None):
        x = self.conv(self.carafe(x))
        if skip is not None and self.skip_fuse is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)
            x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return x


class UpBlock(nn.Module):
    """bilinear x2 -> 3x3+BN+ReLU -> 可选 skip concat + 3x3+BN+ReLU。

    用插值而不是转置卷积: 上采样本身无参数, 也不会产生棋盘格伪影。
    """

    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if skip_ch > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_fuse = None

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        if skip is not None and self.skip_fuse is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)
            x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return x


class MaskDecoder(nn.Module):
    """上采样 levels 次, 在 1/4 与 1/2 分辨率处接 skip, 最后 1x1 输出 mask。

    patch=32 + grid=11 -> 5 级: 11→22→44→88(+1/4)→176(+1/2)→352
    patch=16 + grid=22 -> 4 级: 22→44→88(+1/4)→176(+1/2)→352
    patch=8  + grid=44 -> 3 级: 44→88(+1/4)→176(+1/2)→352
    """

    def __init__(self, in_ch=128, skip_chs=(16, 32), num_classes=1, levels=5,
                 quarter_idx=2, half_idx=3, attn_up_idx=(), num_heads=4,
                 attn_up_plain=False, attn_up_mode="stack",
                 carafe_idx=(), carafe_k=5, carafe_cm=64,
                 refine_quarter=False, dss_mid=None, dss_hp=True):
        super().__init__()
        self.levels = levels
        self.q_idx = quarter_idx
        self.h_idx = half_idx
        # attn_up_idx: 哪几级用 AttnUpSample 代替 bilinear, 例如 (0,) 只改第一级
        self.attn_up_idx = tuple(attn_up_idx) if attn_up_idx is not None else ()
        # carafe_idx: 哪几级用 CARAFE 代替 bilinear, 优先级高于 attn_up_idx
        self.carafe_idx = tuple(carafe_idx) if carafe_idx is not None else ()
        widths = [max(128 >> i, 16) for i in range(levels)]

        self.ups = nn.ModuleList()
        c = in_ch
        for i in range(levels):
            sk = skip_chs[1] if i == quarter_idx else (skip_chs[0] if i == half_idx else 0)
            if i in self.carafe_idx:
                self.ups.append(CarafeUpBlock(c, widths[i], skip_ch=sk,
                                              k=carafe_k, cm=carafe_cm))
            elif i in self.attn_up_idx:
                if attn_up_mode == "refine":
                    self.ups.append(RefineUpBlock(c, widths[i], skip_ch=sk,
                                                  num_heads=num_heads))
                else:
                    self.ups.append(AttnUpBlock(c, widths[i], skip_ch=sk,
                                                num_heads=num_heads,
                                                use_attn=not attn_up_plain))
            else:
                self.ups.append(UpBlock(c, widths[i], skip_ch=sk))
            c = widths[i]
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)
        # DSS 在 1/4 分辨率(88×88)上做逐像素软路由; 放在 quarter 那一级之后
        self.refine = DSSBlock(widths[quarter_idx], mid=dss_mid, use_hp=dss_hp) \
            if (refine_quarter and quarter_idx >= 0) else None

    def forward(self, x, skip_half=None, skip_quarter=None, grid=None,
                return_aux=False):
        h, w = grid if grid is not None else x.shape[-2:]
        aux = None
        for i, up in enumerate(self.ups):
            sk = skip_quarter if i == self.q_idx else (skip_half if i == self.h_idx else None)
            if isinstance(up, (AttnUpBlock, RefineUpBlock)):
                x = up(x, grid=(h, w), skip=sk)
            else:
                x = up(x, sk)
            if self.refine is not None and i == self.q_idx:
                if return_aux:
                    x, aux = self.refine(x, return_aux=True)
                else:
                    x = self.refine(x)
            h, w = x.shape[-2:]
        if return_aux:
            return self.head(x), aux
        return self.head(x)


# ----------------------------------------------------------------------------
# 网络
# ----------------------------------------------------------------------------
class Net(nn.Module):
    """MSF -> PatchEmbed -> depth × Block(DownKV) -> LayerNorm -> fuse.stem -> decoder。

    size=352 时三种 patch:
        patch=32 -> grid 11x11 (121 tok),  5 级上采样, KV 25 个 (11 奇数, 丢 1 行/列)
        patch=16 -> grid 22x22 (484 tok),  4 级上采样, KV 121 个 (整除)  ← 默认
        patch=8  -> grid 44x44 (1936 tok), 3 级上采样, KV 484 个 (整除)
    """

    def __init__(self, dim=128, num_heads=4, depth=8, size=352, grid=None,
                 widths=(16, 16), mlp_ratio=4.0, out_channels=128, num_classes=1,
                 use_depth=True, msf_widths=(8, 16, 32), msf_out=32, patch=16,
                 dense_patch=False, same_kv=False, drop=0.0, fuse_simple=False,
                 attn_up_idx=(), attn_up_plain=False, attn_up_mode="stack",
                 carafe_idx=(), carafe_k=5, carafe_cm=64,
                 dss_quarter=False, dss_add=False, dss_mid=None, dss_hp=True,
                 dep_blocks=(), dep_on_v=False, keep_fuse_dep=None,
                 dep_zero_init=False, strip_k=(7, 15)):
        super().__init__()
        assert size % patch == 0, f"size {size} must be divisible by patch {patch}"
        assert patch in (4, 8, 16, 32), "patch must be one of 4/8/16/32"
        self.patch_size = patch
        if grid is None:
            g = (size // patch, size // patch)
        else:
            g = tuple(grid)
            assert g[0] * patch == size and g[1] * patch == size, \
                f"grid {g} does not match size {size} with patch {patch}"
        self.grid = g
        self.dim = dim
        self.use_depth = use_depth
        # 深度注入到哪几个 Block(0 起算): 空 = 不走注意力注入(旧行为)
        self.dep_blocks = tuple(i for i in dep_blocks if 0 <= i < depth)
        self.dep_on_v = dep_on_v
        # keep_fuse_dep=None -> 自动: 没有注意力注入时保留 fuse 相加(旧行为),
        # 有注入时默认把 fuse 的深度拿掉(避免两处重复)
        self.keep_fuse_dep = (not self.dep_blocks) if keep_fuse_dep is None \
            else bool(keep_fuse_dep)

        # decoder 级数 = log2(patch); 1/4 与 1/2 分辨率各接一次 skip
        levels = int(math.log2(patch))
        quarter_idx = int(math.log2(patch // 4)) - 1 if patch >= 8 else -1
        half_idx = int(math.log2(patch // 2)) - 1 if patch >= 4 else -1

        self.msf = MultiScaleFusion(3, msf_widths, msf_out)
        self.patch_embed = PatchEmbed(
            in_ch=msf_out, dim=dim, grid=g, patch=patch, dense=dense_patch,
        )
        self.depth_stem = DepthStem(widths=widths, out_channels=dim, grid=g) \
            if use_depth else None
        self.blocks = nn.ModuleList(
            [Block(dim=dim, num_heads=num_heads, grid=g, mlp_ratio=mlp_ratio,
                   drop=drop, same_kv=same_kv,
                   dep_norm=(i in self.dep_blocks), dep_on_v=dep_on_v)
             for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.dep_zero_init = bool(dep_zero_init and self.dep_blocks)
        # dss_quarter=True: 1/16 只做深度注入+1×1, DSS 移到 1/4 (见 decoder.refine)
        _inject = dss_quarter and not dss_add
        _fuse_depth = use_depth and self.keep_fuse_dep
        self.fuse = DepthInject(dim=dim, out_channels=out_channels,
                                use_depth=_fuse_depth) if _inject else \
            Fusion(dim=dim, out_channels=out_channels,
                   use_depth=_fuse_depth, simple=fuse_simple,
                   strip_k=strip_k)
        self.decoder = MaskDecoder(
            in_ch=out_channels, skip_chs=(msf_widths[1], msf_widths[2]),
            num_classes=num_classes, levels=levels,
            quarter_idx=quarter_idx, half_idx=half_idx,
            attn_up_idx=attn_up_idx, num_heads=num_heads,
            attn_up_plain=attn_up_plain, attn_up_mode=attn_up_mode,
            carafe_idx=carafe_idx, carafe_k=carafe_k, carafe_cm=carafe_cm,
            refine_quarter=(dss_quarter or dss_add), dss_mid=dss_mid, dss_hp=dss_hp,
        )
        self.num_blocks = depth
        self.apply(self._init_weights)
        # dep_zero_init: 注入支的 LN weight 置 0 -> 初始化时深度贡献恒为 0,
        # 模型与"未注入深度"的旧权重数学上完全等价, 深度强度从 0 自己长出来。
        # 必须放在 _init_weights 之后, 否则会被 LayerNorm 的 ones_ 覆盖。
        if self.dep_zero_init:
            with torch.no_grad():
                for _b in self.blocks:
                    if _b.attn.dep_norm is not None:
                        _b.attn.dep_norm.weight.zero_()
                        _b.attn.dep_norm.bias.zero_()
        # DSS 起点做成恒等 (两支增量都置零), 但 router 保持 w=0.5 让两支都能拿梯度
        if self.decoder.refine is not None:
            rf = self.decoder.refine
            nn.init.zeros_(rf.rc[-1].weight)
            nn.init.zeros_(rf.bs[-1].weight)
            nn.init.zeros_(rf.router[-1].bias)
        # 分割头后面没有 BN 兜底: 按 0.02 初始化时 logits std 只有 0.06,
        # sigmoid 全挤在 0.5, BCE/Dice 梯度被压平。单独放大到 std=0.4。
        nn.init.trunc_normal_(self.decoder.head.weight, std=0.4)
        nn.init.zeros_(self.decoder.head.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, CARAFE):
            # 重组核输出置 0 -> softmax 均匀 -> 初始等价于邻域均值(box)上采样,
            # 比随机核稳定; 训练中再学出内容相关的形状。
            nn.init.zeros_(m.enc.weight)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # 冻结
    # ------------------------------------------------------------------
    def freeze(self, *names):
        """冻结指定顶层模块, 返回被冻结的参数个数。

        model.freeze("msf", "patch_embed", "depth_stem", "blocks", "norm", "fuse")
        只留 decoder 可训练。BN 的 running stats 不受 requires_grad 影响,
        训练时还需要把冻结区切到 eval, 见 train.py 的 freeze_bn_eval。
        """
        n = 0
        for pname, p in self.named_parameters():
            if p.requires_grad and any(pname == x or pname.startswith(x + ".")
                                       for x in names):
                p.requires_grad = False
                n += p.numel()
        return n

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def _features_full(self, x, dep=None):
        """返回 f1_2d (B,C,gh,gw)、skips (1/2, 1/4) 和 grid。"""
        h, (_, g2, f3) = self.msf(x, return_scales=True)
        feat = self.patch_embed(h)                         # (B, dim, gh, gw) + pos embed
        skips = (g2, f3)

        b, c, gh, gw = feat.shape
        t = feat.flatten(2).transpose(1, 2)                # (B, N, C)
        for i, blk in enumerate(self.blocks):
            # 只在指定块把深度注入注意力 K 分支(先归一化 -> 与特征相加 -> 再 k_conv)
            d = dep if (dep is not None and i in self.dep_blocks) else None
            t = blk(t, grid=(gh, gw), dep=d)
        t = self.norm(t)
        return self._to_2d(t, gh, gw), skips, (gh, gw)

    @staticmethod
    def _to_2d(t, gh, gw):
        b, n, c = t.shape
        return t.transpose(1, 2).reshape(b, c, gh, gw)

    def forward(self, x, depth=None, return_features=False, return_aux=False):
        gh, gw = self.grid
        # 深度要先进注意力块, 所以 dep 必须在 _features_full 之前算出来
        if not self.use_depth:
            dep = None
        elif depth is None:
            dep = torch.zeros(x.shape[0], self.dim, gh, gw,
                              device=x.device, dtype=x.dtype)
        else:
            dep = self.depth_stem(depth, grid=(gh, gw))     # (B, C, gh, gw)

        f1, skips, (gh, gw) = self._features_full(x, dep=dep)

        if return_features:
            return f1

        fused = self.fuse(f1, dep)                        # (B, 128, gh, gw)
        out, aux = self.decoder(fused, skip_half=skips[0], skip_quarter=skips[1],
                                grid=(gh, gw), return_aux=True)
        assert out.shape[-2:] == x.shape[-2:], (
            f"input {tuple(x.shape[-2:])} -> mask {tuple(out.shape[-2:])}; "
            f"size must be divisible by patch {self.patch_size}")
        if return_aux:
            return out, aux
        return out


# ----------------------------------------------------------------------------
# 损失 / 指标
# ----------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    """BCEWithLogits + soft Dice。输入是 logits, target 在 {0,1}。"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = logits.sigmoid()
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        union = prob.sum(dims) + target.sum(dims)
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def dice_score(logits, target, threshold=0.5, eps=1e-6):
    """监控用的二值 Dice。"""
    pred = (logits.sigmoid() > threshold).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    return ((2.0 * inter + eps) / (pred.sum(dims) + target.sum(dims) + eps)).mean().item()


# ----------------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    size = 352
    rgb, dep = torch.randn(2, 3, size, size), torch.randn(2, 1, size, size)
    tgt = (torch.rand(2, 1, size, size) > 0.5).float()

    model = Net(size=size)
    mask = model(rgb, dep)
    print(f"input  : rgb {tuple(rgb.shape)} + depth {tuple(dep.shape)}")
    print(f"grid   : {model.grid} ({model.grid[0] * model.grid[1]} tokens, "
          f"patch {model.patch_size})")
    print(f"mask   : {tuple(mask.shape)}")
    print(f"params : {sum(p.numel() for p in model.parameters()):,}")
    for name in ("msf", "patch_embed", "depth_stem", "blocks", "norm",
                 "fuse", "decoder"):
        mod = getattr(model, name, None)
        if mod is not None:
            print(f"  {name:<12}: {sum(p.numel() for p in mod.parameters()):>9,}")

    loss = DiceBCELoss()(mask, tgt)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"train  : loss {loss.item():.4f}  dice {dice_score(mask, tgt):.4f}  "
          f"缺梯度参数 {len(no_grad)}")

    # 冻结: 只留 decoder
    model2 = Net(size=size)
    n_frozen = model2.freeze("msf", "patch_embed", "depth_stem", "blocks", "norm", "fuse")
    n_all = sum(p.numel() for p in model2.parameters())
    print(f"freeze : {n_frozen:,} / {n_all:,} frozen, "
          f"trainable {model2.trainable_params():,} "
          f"({model2.trainable_params() / n_all * 100:.1f}%)")

    # 与 nn_net.Net 的权重兼容性
    try:
        import nn_net
        old = nn_net.Net(size=size).state_dict()
        new = Net(size=size).state_dict()
        hit = [k for k in new if k in old and old[k].shape == new[k].shape]
        miss = [k for k in new if k not in hit]
        num = sum(new[k].numel() for k in hit)
        tot = sum(v.numel() for v in new.values())
        print(f"compat : {len(hit)}/{len(new)} tensors, {num:,}/{tot:,} 参数 "
              f"({num / tot * 100:.1f}%) 可从 nn_net 载入")
        if miss:
            print(f"         缺: {miss[:5]}")
    except Exception as e:                                    # noqa: BLE001
        print("compat : 跳过 (", e, ")")

    model.eval()
    with torch.no_grad():
        for s in (256, 320, 512):
            m = Net(size=s).eval()
            _x, _d = torch.randn(1, 3, s, s), torch.randn(1, 1, s, s)
            print(f"  size {s:>3} -> {tuple(m(_x, _d).shape)}  grid {m.grid}")
        for p in (8, 32):
            m = Net(size=size, patch=p).eval()
            print(f"  patch {p:>2} -> {m.grid[0] * m.grid[1]:>4} tok, "
                  f"{m.decoder.levels} 级, {tuple(m(rgb, dep).shape)}")
