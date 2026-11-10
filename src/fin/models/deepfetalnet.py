"""DeepFetalNet: the proposed lightweight standard-plane classifier (659,433 parameters for two classes).

Lightweight CNN backbone (depthwise-separable convolutions with squeeze-and-excitation, output stride 16),
then an adaptive multi-scale fusion module (AMFAM, dilations 1/2/3) as the local branch, a self-attention
global-context branch, a learned softmax fusion of both, an attention refinement module (channel and spatial
attention with a residual) and a classification head. Architecture as released by the authors (repository
commit 631c725, ``DeepFetalNet/model.py``); module names are unchanged so trained checkpoints load strictly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution, 1x1 pointwise convolution, batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class LightweightCNNBackbone(nn.Module):
    """Stem (3 -> 32, stride 2) and four stages 64, 128, 256, 256 (strides 2, 2, 2, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(DepthwiseSeparableConv(32, 64, stride=2), SEBlock(64, reduction=8))
        self.stage2 = nn.Sequential(DepthwiseSeparableConv(64, 128, stride=2), SEBlock(128, reduction=16))
        self.stage3 = nn.Sequential(DepthwiseSeparableConv(128, 256, stride=2), SEBlock(256, reduction=16))
        self.stage4 = nn.Sequential(DepthwiseSeparableConv(256, 256, stride=1), SEBlock(256, reduction=16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)


class GlobalContextBranch(nn.Module):
    """1x1 projection to ``embed_dim``, multi-head self-attention over the spatial tokens, residual + norm."""

    def __init__(self, in_channels: int = 256, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        tokens = self.proj_in(x).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(tokens + attn_out)
        out = out.transpose(1, 2).view(b, -1, h, w)
        return self.proj_out(out)


class AdaptiveFusion(nn.Module):
    """Softmax-weighted sum of the local and global feature maps, weights predicted from pooled statistics."""

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels * 2, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        b = local_feat.size(0)
        stats = torch.cat([self.gap(local_feat).view(b, -1), self.gap(global_feat).view(b, -1)], dim=1)
        weights = self.softmax(self.fc2(self.relu(self.fc1(stats))))
        alpha_local = weights[:, 0].view(b, 1, 1, 1)
        alpha_global = weights[:, 1].view(b, 1, 1, 1)
        return alpha_local * local_feat + alpha_global * global_feat


class AMFAM(nn.Module):
    """Adaptive multi-scale fusion: three dilated depthwise-separable branches mixed by learned weights."""

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.dw1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, dilation=1, groups=in_channels, bias=False
        )
        self.pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dw2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False
        )
        self.pw2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=3, dilation=3, groups=in_channels, bias=False
        )
        self.pw3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels * 3, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        f1 = self.pw1(self.dw1(x))
        f2 = self.pw2(self.dw2(x))
        f3 = self.pw3(self.dw3(x))
        y = self.gap(torch.cat([f1, f2, f3], dim=1)).view(b, -1)
        weights = self.softmax(self.fc2(self.relu(self.fc1(y))))
        alpha1 = weights[:, 0].view(b, 1, 1, 1)
        alpha2 = weights[:, 1].view(b, 1, 1, 1)
        alpha3 = weights[:, 2].view(b, 1, 1, 1)
        return self.fusion(alpha1 * f1 + alpha2 * f2 + alpha3 * f3)


class AttentionRefinementModule(nn.Module):
    """Channel attention plus spatial attention, added to the input as a residual."""

    def __init__(self, in_channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        a_c = self.sigmoid_channel(self.fc2(self.relu(self.fc1(self.gap(x).view(b, c))))).view(b, c, 1, 1)
        f_c = x * a_c.expand_as(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a_s = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        f_s = x * a_s.expand_as(x)
        return x + (f_c + f_s)


class ClassificationHead(nn.Module):
    """Global average pooling, FC 256 -> 128, ReLU, dropout 0.5, FC 128 -> classes."""

    def __init__(self, in_features: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.gap(x), 1)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class DeepFetalNet(nn.Module):
    """DeepFetalNet; each component can be switched off for ablations (all on in every reported run)."""

    def __init__(
        self,
        num_classes: int = 2,
        use_global: bool = True,
        use_fusion: bool = True,
        use_amfam: bool = True,
        use_arm: bool = True,
    ) -> None:
        super().__init__()
        self.use_global = use_global
        self.use_fusion = use_fusion
        self.use_amfam = use_amfam
        self.use_arm = use_arm
        self.backbone = LightweightCNNBackbone()
        if self.use_amfam:
            self.amfam = AMFAM(in_channels=256)
        if self.use_global:
            self.global_branch = GlobalContextBranch(in_channels=256, embed_dim=64, num_heads=4)
        if self.use_fusion and self.use_global:
            self.adaptive_fusion = AdaptiveFusion(in_channels=256)
        if self.use_arm:
            self.arm = AttentionRefinementModule(in_channels=256, reduction=16)
        self.head = ClassificationHead(in_features=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        local_features = self.amfam(features) if self.use_amfam else features
        if self.use_global:
            global_features = self.global_branch(features)
            fused = (
                self.adaptive_fusion(local_features, global_features)
                if self.use_fusion
                else local_features + global_features
            )
        else:
            fused = local_features
        if self.use_arm:
            fused = self.arm(fused)
        return self.head(fused)
