"""v1 (net30) 架构的精确复刻, 用于 Same-KV 消融。

与 nn_net.Net 的差别(全部由 net30.pt 的 state_dict 反推):
  * 无 MSF, 走 Stem 路径: 2 x ConvBNReLU(stride2) -> 1/4 (88x88) -> Conv 8x8/s8 -> 11x11 (121 token)
  * 6 层 Block, dim 128, heads 4, mlp_ratio 4
  * DepthStem: 同结构 (1->16->16) -> 1x1 proj -> adaptive pool 到 11x11
  * Fusion(simple): Conv3x3(128->128) + Conv1x1(128->256), 无 Laplace/长短条
  * Decoder: up1..up5 宽度 [128,64,32,32,16], 1/4 与 1/2 skip 接在 up3/up4
  * 无位置编码

用法:
    from nn_v1 import NetV1
    m = NetV1(depth=6, same_kv=True)     # same_kv=True -> KV 与 Q 同分辨率 (25 -> 121)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_net import Block, ConvBNReLU


class StemV1(nn.Module):
    def __init__(self, widths=(16, 16), out_channels=128, grid=(11, 11)):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(3, widths[0], 3, 2, 1),
            ConvBNReLU(widths[0], widths[1], 3, 2, 1),
        )
        self.patch = nn.Conv2d(widths[1], out_channels, kernel_size=8, stride=8)

    def forward(self, x, return_feats=False):
        half = self.body[0](x)                 # 1/2 (176x176)
        quarter = self.body[1](half)           # 1/4 (88x88)
        x = self.patch(quarter)                # 1/32 -> 11x11
        return (x, [half, quarter]) if return_feats else x


class DepthStemV1(nn.Module):
    def __init__(self, widths=(16, 16), out_channels=128, grid=(11, 11)):
        super().__init__()
        self.grid = grid
        self.body = nn.Sequential(
            ConvBNReLU(1, widths[0], 3, 2, 1),
            ConvBNReLU(widths[0], widths[1], 3, 2, 1),
        )
        self.proj = nn.Conv2d(widths[1], out_channels, kernel_size=1)

    def forward(self, d, grid=None):
        gh, gw = grid if grid is not None else self.grid
        x = self.proj(self.body(d))
        return F.adaptive_avg_pool2d(x, (gh, gw))


class FusionV1(nn.Module):
    """simple fuse: 3x3 -> 1x1 (v1 的融合方式, 无 Laplace/长短条)。"""

    def __init__(self, dim=128, out_channels=256):
        super().__init__()
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(dim, out_channels, 1)

    def forward(self, rgb, dep=None):
        x = rgb + dep if dep is not None else rgb
        x = F.relu(self.conv3x3(x), inplace=True)
        return F.relu(self.conv1x1(x), inplace=True)


class UpV1(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if skip_ch > 0:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_fuse = None

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        if skip is not None and self.skip_fuse is not None:
            x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return x


class DecoderV1(nn.Module):
    """up1..up5, 宽度 [128,64,32,32,16], skip(1/4) 接 up3, skip(1/2) 接 up4。"""

    def __init__(self, in_ch=256, skip_half=16, skip_quarter=16, num_classes=1):
        super().__init__()
        self.up1 = UpV1(in_ch, 128)
        self.up2 = UpV1(128, 64)
        self.up3 = UpV1(64, 32, skip_ch=skip_quarter)
        self.up4 = UpV1(32, 32, skip_ch=skip_half)
        self.up5 = UpV1(32, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x, skip_half=None, skip_quarter=None):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x, skip_quarter)
        x = self.up4(x, skip_half)
        x = self.up5(x)
        return self.head(x)


class NetV1(nn.Module):
    def __init__(self, dim=128, num_heads=4, depth=6, size=352, widths=(16, 16),
                 mlp_ratio=4.0, out_channels=256, num_classes=1, use_depth=True,
                 same_kv=False, grid=None, dep_blocks=(1, 3, 5),
                 keep_fuse_dep=False, dep_on_v=False, latent_rank=0):
        super().__init__()
        g = (size // 32, size // 32) if grid is None else tuple(grid)
        self.grid = g
        self.patch_size = 32
        self.use_depth = use_depth
        self.same_kv = same_kv
        # 深度注入到哪几个 Block(0 起算): 1/3/5 即第 2/4/6 个块
        self.dep_blocks = tuple(i for i in dep_blocks if 0 <= i < depth)
        self.keep_fuse_dep = keep_fuse_dep
        self.stem = StemV1(widths=widths, out_channels=dim, grid=g)
        self.depth_stem = DepthStemV1(widths=widths, out_channels=dim, grid=g) if use_depth else None
        self.blocks = nn.ModuleList(
            [Block(dim=dim, num_heads=num_heads, grid=g, mlp_ratio=mlp_ratio,
                   same_kv=same_kv, dep_norm=(i in self.dep_blocks),
                   dep_on_v=dep_on_v, latent_rank=latent_rank)
             for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.fuse = FusionV1(dim=dim, out_channels=out_channels)
        self.decoder = DecoderV1(in_ch=out_channels, skip_half=widths[0],
                                 skip_quarter=widths[1], num_classes=num_classes)
        self.num_blocks = depth
        self.apply(self._init_weights)
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

    def _features_full(self, x, dep=None):
        feat, skips = self.stem(x, return_feats=True)      # (B,C,11,11), [1/2, 1/4]
        b, c, gh, gw = feat.shape
        t = feat.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.blocks):
            # 只在指定的块把深度注入注意力 K 分支(先归一化 -> 与特征相加 -> 再 k_conv)
            d = dep if (dep is not None and i in self.dep_blocks) else None
            t = blk(t, grid=(gh, gw), dep=d)
        t = self.norm(t)
        return t.transpose(1, 2).reshape(b, c, gh, gw), skips, (gh, gw)

    def forward(self, x, depth=None, return_features=False):
        gh, gw = self.grid
        dep = None
        if self.use_depth and depth is not None:
            dep = self.depth_stem(depth, grid=(gh, gw))     # 与 11x11 特征同尺寸
        rgb, skips, _ = self._features_full(x, dep=dep)
        if return_features:
            return rgb
        # fuse 不再吃深度(深度已改由注意力块注入); keep_fuse_dep=True 可恢复旧行为
        fused = self.fuse(rgb, dep if self.keep_fuse_dep else None)
        return self.decoder(fused, skip_half=skips[0], skip_quarter=skips[1])


if __name__ == "__main__":
    import sys
    ck = torch.load("E:/WorkBuddy-Outputs/net30.pt", map_location="cpu")
    for sk in (False, True):
        m = NetV1(depth=6, same_kv=sk)
        n = sum(p.numel() for p in m.parameters())
        msg = ""
        if not sk:
            try:
                m.load_state_dict(ck["model"], strict=True)
                msg = "  strict load OK"
            except Exception as e:
                msg = f"  load FAIL: {str(e)[:200]}"
        print(f"same_kv={sk} params {n:,}{msg}")
