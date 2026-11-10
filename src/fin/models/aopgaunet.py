"""AoP-GAU-Net: the proposed attention graph U-Net with a direct AoP regression head.

Architecture as released by the authors (repository commit 631c725, ``AoP-GAU-Net/model.py``): base width 32,
depthwise-separable convolutions, a graph bottleneck (noise-aware attention, dynamic graph learning, sparse
k-nearest-neighbour graph attention with 2-D positional encoding, convolutional feed-forward), attention-gated
skip connections and an attention-pooling head that predicts the AoP as 180 * sigmoid(.). Here
``in_channels = 1`` and ``num_classes = 3``; module names are unchanged so trained checkpoints load strictly.
The training loss is ``fin.training.losses.CombinedLoss``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """k x k depthwise convolution followed by a 1x1 pointwise convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ConvBNReLU(nn.Module):
    """Convolution (plain or depthwise-separable), batch norm, ReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_dw: bool = False, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        conv = (
            DepthwiseSeparableConv(in_ch, out_ch, kernel_size, padding)
            if use_dw
            else nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        )
        self.block = nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Two ConvBNReLU layers; returns (pooled output, skip features)."""

    def __init__(self, in_ch: int, out_ch: int, use_dw: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_ch, out_ch, use_dw), ConvBNReLU(out_ch, out_ch, use_dw))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class PositionalEncoding2D(nn.Module):
    """Learned projection of normalised (x, y) token coordinates, appended to the token features."""

    def __init__(self, coord_dim: int = 2, feat_dim: int = 128) -> None:
        super().__init__()
        max_cd = max(1, int(math.floor(0.25 * feat_dim)))
        self.coord_dim = min(coord_dim, max_cd)
        self.proj = nn.Linear(2, self.coord_dim, bias=False)
        self.coord_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(max(feat_dim, 1))))
        self._cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _get_coords(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._cache:
            ys = torch.linspace(0, 1, h, device=device)
            xs = torch.linspace(0, 1, w, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._cache[key] = torch.stack([gx, gy], dim=-1).reshape(h * w, 2)
        return self._cache[key]

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pe = self.proj(self._get_coords(h, w, tokens.device)) * self.coord_scale
        return torch.cat([tokens, pe.unsqueeze(0).expand(tokens.size(0), -1, -1)], dim=-1)


class NoiseAwareAttention(nn.Module):
    """Product of a depthwise spatial gate and a pointwise channel gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.spatial(h) * self.channel(h) * h


class DynamicGraphLearning(nn.Module):
    """Dense cosine-similarity graph over (pooled) positions; messages added with a learned weight."""

    def __init__(self, channels: int, reduction: int = 4, max_nodes: int = 256) -> None:
        super().__init__()
        self.d = max(channels // reduction, 16)
        self.max_nodes = max_nodes
        self.proj = nn.Conv2d(channels, self.d, 1, bias=False)
        self.back = nn.Conv2d(self.d, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.attn_drop = nn.Dropout(p=0.1)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> torch.Tensor:
        """Message weight in (0.05, 0.95)."""
        return torch.sigmoid(self.alpha_raw) * 0.9 + 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        needs_pool = h * w > self.max_nodes
        if needs_pool:
            side = int(math.isqrt(self.max_nodes))
            x_pool = F.adaptive_avg_pool2d(x, (side, side))
        else:
            x_pool, side = x, h
        feat = self.proj(x_pool).flatten(2)
        feat_norm = F.normalize(feat, dim=1)
        adj = torch.bmm(feat_norm.transpose(1, 2), feat_norm)
        adj = self.attn_drop(F.softmax(adj / math.sqrt(self.d), dim=-1))
        msg = self.back(torch.bmm(feat, adj).reshape(b, self.d, side, side))
        if needs_pool:
            msg = F.interpolate(msg, size=(h, w), mode="bilinear", align_corners=False)
        return self.norm(x + self.alpha * msg)


class SparseGraphAttention(nn.Module):
    """Multi-head attention restricted to the k nearest neighbours of each token in feature space."""

    def __init__(
        self,
        in_dim: int,
        graph_dim: int = 128,
        d_head: int = 32,
        k: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        coord_dim: int = 2,
    ) -> None:
        super().__init__()
        total = d_head * num_heads
        self.k, self.num_heads, self.d_head, self.scale = k, num_heads, d_head, math.sqrt(d_head)
        self.pos_enc = PositionalEncoding2D(coord_dim=coord_dim, feat_dim=in_dim)
        self.down = nn.Linear(in_dim + self.pos_enc.coord_dim, graph_dim, bias=False)
        self.up = nn.Linear(graph_dim, in_dim, bias=False)
        self.proj_q = nn.Linear(graph_dim, total, bias=False)
        self.proj_k = nn.Linear(graph_dim, total, bias=False)
        self.proj_v = nn.Linear(graph_dim, total, bias=False)
        self.out_proj = nn.Linear(total, graph_dim, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)
        self.ff_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(in_dim)

    @torch.no_grad()
    def _knn_indices(self, x: torch.Tensor) -> torch.Tensor:
        sq = (x * x).sum(-1, keepdim=True)
        dist = sq + sq.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
        dist.diagonal(dim1=1, dim2=2).fill_(1e9)
        return dist.topk(self.k, dim=-1, largest=False).indices

    @staticmethod
    def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, n, k = idx.shape
        d = x.shape[-1]
        return torch.gather(x, 1, idx.reshape(b, -1).unsqueeze(-1).expand(-1, -1, d)).reshape(b, n, k, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        tokens = x.flatten(2).transpose(1, 2)
        feat = self.down(self.pos_enc(tokens, h, w))
        q, k_feat, v_feat = self.proj_q(feat), self.proj_k(feat), self.proj_v(feat)
        knn_idx = self._knn_indices(feat.detach())
        k_nbr, v_nbr = self._gather(k_feat, knn_idx), self._gather(v_feat, knn_idx)
        nh, d = self.num_heads, self.d_head
        q = q.view(b, n, nh, d)
        k_nbr = k_nbr.view(b, n, self.k, nh, d).permute(0, 1, 3, 2, 4)
        v_nbr = v_nbr.view(b, n, self.k, nh, d).permute(0, 1, 3, 2, 4)
        attn = self.attn_drop(F.softmax((q.unsqueeze(3) * k_nbr).sum(-1) / self.scale, dim=-1))
        out = (attn.unsqueeze(-1) * v_nbr).sum(-2)
        out = self.up(self.ff_drop(self.out_proj(out.reshape(b, n, nh * d))))
        return self.norm(out + tokens).transpose(1, 2).reshape(b, c, h, w)


class GraphBottleneck(nn.Module):
    """Noise-aware attention, dynamic graph, sparse graph attention and a convolutional feed-forward block."""

    def __init__(
        self,
        channels: int,
        k: int = 6,
        graph_dim: int = 128,
        d_head: int = 32,
        num_heads: int = 4,
        dyn_max_nodes: int = 256,
    ) -> None:
        super().__init__()
        self.noise = NoiseAwareAttention(channels)
        self.dyngraph = DynamicGraphLearning(channels, max_nodes=dyn_max_nodes)
        self.gat = SparseGraphAttention(in_dim=channels, graph_dim=graph_dim, d_head=d_head, k=k, num_heads=num_heads)
        self.ff = nn.Sequential(ConvBNReLU(channels, channels), ConvBNReLU(channels, channels))
        self.ff_norm = nn.BatchNorm2d(channels)
        self.ff_drop = nn.Dropout2d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gat(self.dyngraph(self.noise(x)))
        return self.ff_norm(self.ff_drop(self.ff(x)) + x)


class AttentionGate(nn.Module):
    """Additive attention gate on a skip connection, gated by the decoder features."""

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int | None = None) -> None:
        super().__init__()
        inter_ch = inter_ch or max(x_ch // 2, 1)
        self.W_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.psi(F.relu(self.W_x(x) + self.W_g(g), inplace=True)) * x


class DecoderBlock(nn.Module):
    """Bilinear up-sampling + convolution, attention-gated skip concatenation, two ConvBNReLU layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_dw: bool = False) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ag = AttentionGate(x_ch=skip_ch, g_ch=out_ch)
        self.conv = nn.Sequential(ConvBNReLU(out_ch + skip_ch, out_ch, use_dw), ConvBNReLU(out_ch, out_ch, use_dw))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([x, self.ag(skip, x)], dim=1))


class AttentionPoolingHead(nn.Module):
    """Attention-weighted spatial pooling, then an MLP with a sigmoid output in (0, 1)."""

    def __init__(self, in_ch: int, hidden: int = 64, use_attention: bool = True) -> None:
        super().__init__()
        self.use_att = use_attention
        if use_attention:
            self.att_w = nn.Conv2d(in_ch, 1, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        if self.use_att:
            w = F.softmax(self.att_w(x).reshape(b, 1, -1), dim=-1)
            pooled = (w * x.reshape(b, c, -1)).sum(-1)
        else:
            pooled = x.mean(dim=(2, 3))
        return self.fc(pooled)


class AoPGAUNet(nn.Module):
    """Returns (segmentation logits (B, C, H, W), AoP in degrees (B, 1) = 180 * sigmoid(.))."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_ch: int = 32,
        k_neighbors: int = 6,
        graph_dim: int = 128,
        d_head: int = 32,
        num_heads: int = 4,
        use_dw: bool = True,
        use_att_pool: bool = True,
    ) -> None:
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        bn_ch = c[3] * 2
        self.enc1 = EncoderBlock(in_channels, c[0], use_dw)
        self.enc2 = EncoderBlock(c[0], c[1], use_dw)
        self.enc3 = EncoderBlock(c[1], c[2], use_dw)
        self.enc4 = EncoderBlock(c[2], c[3], use_dw)
        self.bottleneck_conv = nn.Sequential(ConvBNReLU(c[3], bn_ch, use_dw), ConvBNReLU(bn_ch, bn_ch, use_dw))
        self.graph_bottleneck = GraphBottleneck(
            channels=bn_ch,
            k=k_neighbors,
            graph_dim=min(graph_dim, bn_ch),
            d_head=d_head,
            num_heads=num_heads,
            dyn_max_nodes=256,
        )
        self.mc_drop = nn.Identity()
        self.dec1 = DecoderBlock(bn_ch, c[3], c[3], use_dw)
        self.dec2 = DecoderBlock(c[3], c[2], c[2], use_dw)
        self.dec3 = DecoderBlock(c[2], c[1], c[1], use_dw)
        self.dec4 = DecoderBlock(c[1], c[0], c[0], use_dw)
        self.seg_head = nn.Conv2d(c[0], num_classes, 1)
        self.aop_head = AttentionPoolingHead(in_ch=c[0], hidden=64, use_attention=use_att_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.mc_drop(self.graph_bottleneck(self.bottleneck_conv(x)))
        x = self.dec1(x, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)
        return self.seg_head(x), self.aop_head(x) * 180.0
