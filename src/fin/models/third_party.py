"""FAT-Net and MALUNet built from their published source, fetched at a pinned commit (``fin.third_party``).

The adaptations applied to the upstream files are listed in ``fin.third_party``. This module supplies the
weight loaders injected into FAT-Net and the grayscale wrapper with ImageNet normalisation.
"""

from __future__ import annotations

import functools
import types

import timm
import torch
import torch.nn as nn
from torchvision.models import resnet34

from fin.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from fin.third_party import load_module


def _deit_tiny_distilled(pretrained: bool) -> nn.Module:
    model = timm.create_model("deit_tiny_distilled_patch16_224.fb_in1k", pretrained=pretrained)
    model.patch_embed.strict_img_size = False
    return model


def _resnet34(pretrained: bool) -> nn.Module:
    return resnet34(weights="IMAGENET1K_V1" if pretrained else None)


@functools.cache
def _module(name: str) -> types.ModuleType:
    """Adapted upstream module (cached per process); FAT-Net gets the timm/torchvision weight loaders."""
    namespace = {"_fin_deit_tiny_distilled": _deit_tiny_distilled, "_fin_resnet34": _resnet34}
    return load_module(name, namespace if name == "fatnet" else None)


class FATNetGray(nn.Module):
    """FAT-Net for 1-channel input: replicate to 3 channels and apply ImageNet normalisation."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.net = _module("fatnet").FAT_Net(3, num_classes, pretrained=pretrained)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x.expand(-1, 3, -1, -1) - self.mean) / self.std)


def build_malunet(num_classes: int = 3, input_channels: int = 1) -> nn.Module:
    """Published MALUNet (with the SC_Att_Bridge) returning logits."""
    return _module("malunet").MALUNet(num_classes=num_classes, input_channels=input_channels)
