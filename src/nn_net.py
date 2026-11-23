"""RGB-D 显著性/分割网络: 多尺度融合前端 + 下采样 KV 注意力编码器 + 细粒度解码器.

整体结构 (默认 size=352, patch=16):

    rgb ── MultiScaleFusion ── 1/1·1/2·1/4 三层卷积, 由小到大上采样融合 → 32ch @1/1
        ── PatchEmbed ──────── depthwise patch(k=patch,s=patch) + 1x1 → dim + sincos 位置嵌入
        ── 8 × Block ───────── pre-norm 残差块, 每块内是 DownKVAttention (KV 2x2 步长卷积下采样)
        ── LayerNorm ───────── f1 (B, dim, gh, gw)

    depth ─ DepthStem ──────── 1/4 下采样 + 切块平均池化 → dim @grid

    f1 + depth ── Fusion ──── 3x3+BN+ReLU → 1x1 → 256ch
              ── MaskDecoder ─ 逐级 ×2 上采样, 在 1/4 与 1/2 处接 MSF 的 skip → mask

设计要点:
  * K/V 不做线性投影, 而是还原成 2D 后用步长卷积改变分辨率 —— 用卷积的空间
    邻域聚合替代全连接投影, 同时把 KV token 数压到 1/4。
  * 位置嵌入是固定 2D sincos (persistent=False), 不占 checkpoint, 换 grid 自动插值。
  * patch 必须是 2 的幂: 决定 token 数与解码器级数; 偶数 grid 保证 DownKV 整除。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Fixed (non-learnable) 2D sine-cosine embedding, returns (1, C, gh, gw)."""
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
        c2 : w1   -> w2    k3 s2   (H/2)
        c3 : w2   -> w3    k3 s2   (H/4)
        g2 = fuse2(cat[up(c3), c2])   -> (H/2, w2)   1/4 融进 1/2
        g1 = fuse1(cat[up(g2), c1))   -> (H,   w1)   1/2 融进 1/1
        out = 3x3(g1)                 -> (H,   out_ch)

    1/1 分辨率上做 3x3 很贵, 所以两级融合都用 1x1, 只在最后做一次 3x3。
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
    """1/1 分辨率的融合特征直接切 patch -> tokens, 加固定 2D sincos 位置嵌入。

    352 输入 + patch 32 -> 11x11 tokens (与 stem 路径的 token 数一致), 区别在于
    感受野来自 MSF 融合后的全分辨率特征, 而不是 1/4 下采样后的特征。

    投影用 depthwise patch + 1x1 (每组通道独立在 patch 内做线性组合, 再跨通道混合),
    参数量比 dense patch 小两个数量级以上, 但每个输出通道仍能看到全部输入通道。
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
        out = self.proj(x)
        pe = self.pos_embed
        if pe.shape[-2:] != out.shape[-2:]:
            pe = F.interpolate(pe, size=out.shape[-2:], mode="bilinear", align_corners=False)
        return out + pe


class Stem(nn.Module):
    """RGB image -> 1/2 -> 1/4 -> patch conv (k=8, s=8, p=0) -> + fixed pos embed.

    352 输入: 352 -> 176 -> 88 -> 11x11 tokens (88/8 = 11)
    """

    PATCH = 8

    def __init__(self, widths=(16, 16), out_channels=128, grid=(11, 11), in_ch=3):
        super().__init__()
        c1, c2 = widths
        self.body = nn.Sequential(
            ConvBNReLU(in_ch, c1, k=3, s=2, p=1),   # H/2
            ConvBNReLU(c1, c2, k=3, s=2, p=1),      # H/4
        )
        self.patch = nn.Conv2d(c2, out_channels, kernel_size=self.PATCH, stride=self.PATCH, padding=0)
        # fixed 2D sine-cosine positional embedding, not trained, not saved in state_dict
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(out_channels, *grid),
            persistent=False,
        )
        self.out_channels = out_channels

    def forward(self, x, return_feats=False):
        h = x
        feats = []
        for layer in self.body:
            h = layer(h)
            feats.append(h)                        # [1/2, 1/4]

        out = self.patch(h)
        pe = self.pos_embed
        if pe.shape[-2:] != out.shape[-2:]:
            pe = F.interpolate(pe, size=out.shape[-2:], mode="bilinear", align_corners=False)
        out = out + pe

        return (out, feats) if return_feats else out


class DownKVAttention(nn.Module):
    """Multi-head attention where K/V skip the linear projection.

    Q: normal linear projection on tokens.
    K/V: tokens are reshaped back to a 2D map (B, C, gh, gw), then a stride-2
    conv downsamples 1/2 spatially -> (B, C, gh/2, gw/2) -> flattened to N/4 tokens.
    """

    def __init__(self, dim, num_heads=4, grid=(11, 11), kv_kernel=2, dep_norm=False,
                 dep_on_v=False, latent_rank=0):
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
        # no linear projection for K/V, only a downsample conv
        # kv_kernel=2 -> KV 数量 1/4 (省算力); kv_kernel=1 -> 同分辨率 (SameKV 对照)
        # latent_rank>0: MLA 式低秩压缩, K/V 先共享一份 r 维 latent 再解压
        self.latent_rank = latent_rank
        if latent_rank:
            self.kv_down = nn.Conv2d(dim, latent_rank, kernel_size=1, bias=False)
            self.k_conv = nn.Conv2d(latent_rank, dim, kernel_size=kv_kernel, stride=kv_kernel)
            self.v_conv = nn.Conv2d(latent_rank, dim, kernel_size=kv_kernel, stride=kv_kernel)
        else:
            self.kv_down = None
            self.k_conv = nn.Conv2d(dim, dim, kernel_size=kv_kernel, stride=kv_kernel)
            self.v_conv = nn.Conv2d(dim, dim, kernel_size=kv_kernel, stride=kv_kernel)
        self.proj = nn.Linear(dim, dim)

    def _tokens_to_map(self, x, gh, gw):
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, gh, gw)

    def _ln_map(self, d):
        """(B,C,H,W) 逐像素 LayerNorm(按通道), 用于把深度支路拉到与 LN 后特征同量级。"""
        b, c, h, w = d.shape
        return self.dep_norm(d.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x, grid=None, num_extra=0, dep=None):
        b, n, c = x.shape
        gh, gw = grid if grid is not None else self.grid
        n_base = gh * gw
        assert n == n_base + num_extra, (
            f"grid {gh}x{gw} needs {n_base} spatial tokens, "
            f"got {n} (= {n_base} + {num_extra} extra)")

        q = self.q(x)                                        # (B, N+extra, C)

        base = x[:, num_extra:]                              # K/V 只取空间 token
        feat = self._tokens_to_map(base, gh, gw)             # (B, C, gh, gw)
        feat_k = feat
        feat_v = feat
        if dep is not None:
            # 深度先归一化, 再与特征相加, 然后才进 k_conv (先加后投影)
            dep_n = self._ln_map(dep) if self.dep_norm is not None else dep
            feat_k = feat + dep_n
            if self.dep_on_v:
                feat_v = feat + dep_n
        if self.kv_down is not None:
            # MLA: 一份低维 latent 共享给 K/V, 再从 r 维解压回 dim 并下采样
            c_kv = self.kv_down(feat_k)
            k = self.k_conv(c_kv).flatten(2).transpose(1, 2)
            v = self.v_conv(c_kv).flatten(2).transpose(1, 2)
        else:
            k = self.k_conv(feat_k).flatten(2).transpose(1, 2)   # (B, N/4, C)
            v = self.v_conv(feat_v).flatten(2).transpose(1, 2)   # (B, N/4, C)
        n_kv = k.shape[1]

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)      # (B,H,N,Dh)
        k = k.view(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,N/4,Dh)
        v = v.view(b, n_kv, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)         # (B,H,N,Dh)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), x + MLP(LN(x))."""

    def __init__(self, dim, num_heads=4, grid=(11, 11), mlp_ratio=4.0, drop=0.0,
                 same_kv=False, dep_norm=False, dep_on_v=False, latent_rank=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DownKVAttention(dim=dim, num_heads=num_heads, grid=grid,
                                    kv_kernel=1 if same_kv else 2, dep_norm=dep_norm,
                                    dep_on_v=dep_on_v, latent_rank=latent_rank)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, grid=None, num_extra=0, dep=None):
        x = x + self.attn(self.norm1(x), grid=grid, num_extra=num_extra, dep=dep)
        x = x + self.mlp(self.norm2(x))
        return x


class DepthStem(nn.Module):
    """Depth map (B,1,H,W) -> 1/2 -> 1/4 -> block avg pool -> (B, C, gh, gw).

    Same spatial size / channels as the RGB branch output, so the two can be fused.
    """

    def __init__(self, widths=(16, 16), out_channels=128, grid=(11, 11)):
        super().__init__()
        c1, c2 = widths
        self.body = nn.Sequential(
            ConvBNReLU(1, c1, k=3, s=2, p=1),    # H/2
            ConvBNReLU(c1, c2, k=3, s=2, p=1),   # H/4
        )
        self.proj = nn.Conv2d(c2, out_channels, kernel_size=1)
        # 输出归一化: 否则 depth 分支幅度只有 rgb 分支的 3%, 融合时几乎不起作用
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.grid = grid
        self.out_channels = out_channels

    def forward(self, d, grid=None):
        f = self.body(d)                               # (B, 16, H/4, W/4)
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
    """F(+depth) -> 3x3 -> 1x1 升通道 -> 双分支 -> concat -> 3x3 -> 1x1 降通道。

    分支 A: 固定 3x3 拉普拉斯算子(depthwise, 无参数) -> 3x3 卷积, 尺寸不变
    分支 B: 1x7 -> 7x1 -> 1x15 -> 15x1, 每级 depthwise + 1x1, 尺寸不变
    concat 后 3x3(depthwise) 保持 2C, 最后 1x1 降回 C。

    use_depth=False 时直接走 rgb, 不做加法(depth 通路整体移除)。
    """

    def __init__(self, dim=128, out_channels=256, use_depth=True, simple=False):
        super().__init__()
        self.use_depth = use_depth
        self.simple = simple
        # (F + depth) -> 3x3 尺寸不变 -> 1x1 升通道
        self.stem = nn.Sequential(
            ConvBNReLU(dim, dim, 3, 1, 1),
            nn.Conv2d(dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if simple:
            # 消融: 去掉 Laplace 与长短条分支, 只留 3x3 + 1x1 (v5 的融合方式)
            self.tail = nn.Sequential(
                ConvBNReLU(out_channels, out_channels, 3, 1, 1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
            )
            return
        # 分支 A: 固定拉普拉斯 (不参与训练) + 3x3
        self.register_buffer("lap_kernel", _laplacian_weight(out_channels), persistent=False)
        self.lap_conv = ConvBNReLU(out_channels, out_channels, 3, 1, 1)
        # 分支 B: 横竖长条卷积, 每级 depthwise + 1x1
        self.strip = nn.Sequential(
            _ds_conv(out_channels, (1, 7), (0, 3)),
            _ds_conv(out_channels, (7, 1), (3, 0)),
            _ds_conv(out_channels, (1, 15), (0, 7)),
            _ds_conv(out_channels, (15, 1), (7, 0)),
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
        return self.merge(torch.cat([a, b], dim=1))                        # concat -> 3x3 -> 1x1 降通道


class UpBlock(nn.Module):
    """bilinear x2 upsample -> 3x3 conv -> BN -> ReLU, optional skip concat."""

    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if skip_ch > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
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
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return x


class MaskDecoder(nn.Module):
    """tokens -> size x size mask, 上采样 levels 次, 在 1/4 与 1/2 分辨率处接 skip。

    第 k 次上采样后分辨率 = grid * 2^k, 而 1/4 = size/4、1/2 = size/2, 所以
    skip 接入的级索引由 patch 大小直接决定(见 Net 里 quarter_idx / half_idx)。

    patch=32 + grid=11 -> 5 级: 11→22→44→88(+1/4)→176(+1/2)→352
    patch=16 + grid=22 -> 4 级: 22→44→88(+1/4)→176(+1/2)→352
    patch=8  + grid=44 -> 3 级: 44→88(+1/4)→176(+1/2)→352
    """

    def __init__(self, in_ch=256, skip_chs=(16, 32), num_classes=1, levels=5,
                 quarter_idx=2, half_idx=3):
        super().__init__()
        self.levels = levels
        self.q_idx = quarter_idx
        self.h_idx = half_idx
        widths = [max(128 >> i, 16) for i in range(levels)]

        self.ups = nn.ModuleList()
        c = in_ch
        for i in range(levels):
            sk = skip_chs[1] if i == quarter_idx else (skip_chs[0] if i == half_idx else 0)
            self.ups.append(UpBlock(c, widths[i], skip_ch=sk))
            c = widths[i]
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x, skip_half=None, skip_quarter=None):
        for i, up in enumerate(self.ups):
            sk = skip_quarter if i == self.q_idx else (skip_half if i == self.h_idx else None)
            x = up(x, sk)
        return self.head(x)


class Net(nn.Module):
    """MSF -> PatchEmbed -> depth x pre-norm DownKV Block -> LayerNorm -> decoder.

    size=352 时三种 patch:
        patch=32 -> grid 11x11 (121 tok), 5 级上采样, KV 25 个 (11 是奇数, 丢 1 行/列)
        patch=16 -> grid 22x22 (484 tok), 4 级上采样, KV 121 个 (整除, 无丢弃)  ← 默认
        patch=8  -> grid 44x44 (1936 tok), 3 级上采样, KV 484 个 (整除, 无丢弃)

    消融开关: use_msf=False 回到 stem 路径 (1/4 下采样 + patch8, 无多尺度融合);
              use_depth=False 去掉深度分支。
    """

    def __init__(self, dim=128, num_heads=4, depth=8, size=352, grid=None, widths=(16, 16),
                 mlp_ratio=4.0, out_channels=256, num_classes=1, use_depth=True,
                 fuse_simple=False,
                 use_msf=True, msf_widths=(8, 16, 32), msf_out=32, patch=16, dense_patch=False,
                 same_kv=False):
        super().__init__()
        assert size % patch == 0, f"size {size} must be divisible by patch {patch}"
        assert patch in (4, 8, 16, 32), "patch must be one of 4/8/16/32"
        self.patch_size = patch
        # 关掉 MSF 时走 stem 路径: stem 固定 1/4 下采样 + patch 8, 所以 grid 按 32 算
        eff_patch = patch if use_msf else 32
        if grid is None:
            g = (size // eff_patch, size // eff_patch)
        else:
            g = tuple(grid)
            assert g[0] * eff_patch == size and g[1] * eff_patch == size, \
                f"grid {g} does not match size {size} with patch {eff_patch}"
        self.grid = g
        self.use_depth = use_depth
        self.use_msf = use_msf
        # decoder: log2(eff_patch) 次上采样; 1/4 与 1/2 分辨率各接一次 skip
        levels = int(math.log2(eff_patch))
        quarter_idx = int(math.log2(eff_patch // 4)) - 1 if eff_patch >= 8 else -1
        half_idx = int(math.log2(eff_patch // 2)) - 1 if eff_patch >= 4 else -1

        if use_msf:
            # RGB -> 1/1、1/2、1/4 -> 由小到大上采样融合 -> 1/1
            self.msf = MultiScaleFusion(3, msf_widths, msf_out)
            # 1/1 特征直接切 patch (352 / 16 = 22), 不再经过 stem 下采样
            self.patch_embed = PatchEmbed(
                in_ch=msf_out, dim=dim, grid=g, patch=patch, dense=dense_patch,
            )
            self.stem = None
            skip_chs = (msf_widths[1], msf_widths[2])     # 1/2, 1/4 来自 MSF
        else:
            self.msf = None
            self.patch_embed = None
            self.stem = Stem(widths=widths, out_channels=dim, grid=g)
            skip_chs = (widths[0], widths[1])

        self.depth_stem = DepthStem(widths=widths, out_channels=dim, grid=g) if use_depth else None
        self.blocks = nn.ModuleList(
            [Block(dim=dim, num_heads=num_heads, grid=g, mlp_ratio=mlp_ratio,
                   same_kv=same_kv) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.fuse = Fusion(dim=dim, out_channels=out_channels, use_depth=use_depth,
                           simple=fuse_simple)
        self.decoder = MaskDecoder(
            in_ch=out_channels, skip_chs=skip_chs, num_classes=num_classes,
            levels=levels, quarter_idx=quarter_idx, half_idx=half_idx,
        )
        self.num_blocks = depth
        self.apply(self._init_weights)
        # 分割头后面没有 BN 兜底: 若也按 0.02 初始化, logits std 只有 0.06,
        # sigmoid 全挤在 0.5 附近, BCE/Dice 梯度被压平(实测前 40 步 loss 不降反升).
        # 单独放大到 std=0.4, logits std ≈1.5, 第一步就能正常出梯度.
        nn.init.trunc_normal_(self.decoder.head.weight, std=0.4)
        nn.init.zeros_(self.decoder.head.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Returns f1_2d, the stem skips and the grid."""
        f1_2d, skips, grid = self._features_full(x)
        return f1_2d, skips, grid

    def _features_full(self, x):
        """Returns f1_2d (B,C,gh,gw), skips (1/2, 1/4) and the grid."""
        if self.msf is not None:
            h, (_, g2, f3) = self.msf(x, return_scales=True)   # 1/1, (1/1, 1/2, 1/4)
            feat = self.patch_embed(h)                         # (B, dim, gh, gw) + pos embed
            skips = [g2, f3]                                   # 1/2 (w2), 1/4 (w3)
        else:
            feat, skips = self.stem(x, return_feats=True)       # skips = [1/2, 1/4]

        b, c, gh, gw = feat.shape
        t = feat.flatten(2).transpose(1, 2)              # (B, N, C)

        for blk in self.blocks:
            t = blk(t, grid=(gh, gw))
        t = self.norm(t)
        return self._to_2d(t, gh, gw), skips, (gh, gw)

    @staticmethod
    def _to_2d(t, gh, gw):
        b, n, c = t.shape
        return t.transpose(1, 2).reshape(b, c, gh, gw)

    def forward(self, x, depth=None, return_features=False):
        rgb, skips, (gh, gw) = self._features_full(x)

        if return_features:
            return rgb

        if not self.use_depth:
            dep = None                             # 消融: depth 通路整体移除
        elif depth is None:
            dep = torch.zeros_like(rgb)
        else:
            dep = self.depth_stem(depth, grid=(gh, gw))   # (B, C, gh, gw)
            assert dep.shape == rgb.shape, f"{dep.shape} != {rgb.shape}"

        fused = self.fuse(rgb, dep)                # (B, 256, gh, gw)
        out = self.decoder(fused, skip_half=skips[0], skip_quarter=skips[1])
        assert out.shape[-2:] == x.shape[-2:], (
            f"input {tuple(x.shape[-2:])} -> mask {tuple(out.shape[-2:])}; "
            f"size must be divisible by patch {self.patch_size} and grid must be {self.grid}")
        return out


class DiceBCELoss(nn.Module):
    """BCEWithLogits + soft Dice. Inputs are logits, target in {0,1}."""

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
    """Binary Dice coefficient for monitoring."""
    pred = (logits.sigmoid() > threshold).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    return ((2.0 * inter + eps) / (pred.sum(dims) + target.sum(dims) + eps)).mean().item()


def _selftest(size=352):
    torch.manual_seed(0)
    rgb, dep = torch.randn(2, 3, size, size), torch.randn(2, 1, size, size)
    tgt = (torch.rand(2, 1, size, size) > 0.5).float()

    model = Net(size=size)
    mask = model(rgb, dep)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"input  : rgb {tuple(rgb.shape)} + depth {tuple(dep.shape)}")
    print(f"grid   : {model.grid}  ({model.grid[0] * model.grid[1]} tokens, patch {model.patch_size})")
    print(f"mask   : {tuple(mask.shape)}")
    print(f"params : {n_par:,}")
    for name, mod in (("msf", model.msf), ("patch", model.patch_embed), ("stem", model.stem),
                      ("depth", model.depth_stem), ("blocks", model.blocks),
                      ("fuse", model.fuse), ("decoder", model.decoder)):
        if mod is not None:
            print(f"  {name:<8}: {sum(p.numel() for p in mod.parameters()):>9,}")

    with torch.no_grad():
        h, (f1, g2, f3) = model.msf(rgb, return_scales=True)
        tok = model.patch_embed(h)
    print(f"scales : 1/1 {tuple(f1.shape)}  1/2 {tuple(g2.shape)}  1/4 {tuple(f3.shape)}")
    print(f"patch  : {tuple(h.shape)} -> {tuple(tok.shape)}")

    loss = DiceBCELoss()(mask, tgt)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"train  : loss {loss.item():.4f}  dice {dice_score(mask, tgt):.4f}  "
          f"缺梯度参数 {len(no_grad)}")

    model.eval()
    with torch.no_grad():
        for s in (256, 320, 512):
            m = Net(size=s).eval()
            y = m(torch.randn(1, 3, s, s), torch.randn(1, 1, s, s))
            print(f"  size {s:>3} -> mask {tuple(y.shape)}  grid {m.grid}")
        for p in (8, 32):
            m = Net(size=size, patch=p).eval()
            y = m(rgb, dep)
            print(f"  patch {p:>2} -> {m.grid[0] * m.grid[1]:>4} tok, "
                  f"{m.decoder.levels} 级, mask {tuple(y.shape)}")
        for kw in ({"use_msf": False}, {"use_depth": False}):
            m = Net(size=size, **kw).eval()
            print(f"  {list(kw)[0]}=False -> mask {tuple(m(rgb, dep).shape)}  grid {m.grid}")


if __name__ == "__main__":
    _selftest()
