"""U-Net (Ronneberger, Fischer and Brox, MICCAI 2015), written from the paper for this study.

Four-level encoder/decoder (64-128-256-512, bottleneck 1024), two 3x3 convolution + batch norm + ReLU per
level, transposed-convolution up-sampling and padding 1, so the output size equals the input size. Batch norm
is the common modern addition (not in the 2015 paper). Input 1 x H x W in [0, 1]; output (B, 3, H, W) logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """Two 3x3 convolution + batch norm + ReLU layers."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """U-Net with ``base`` channels at the first level."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3, base: int = 64) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList([DoubleConv(in_channels, c[0])] + [DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec, strict=True):
            x = dec(torch.cat([skips.pop(), up(x)], 1))
        return self.head(x)
