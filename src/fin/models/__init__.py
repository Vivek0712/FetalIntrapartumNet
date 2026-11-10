"""Model registry: seven classifiers and four segmenters.

Classifiers take a 3 x 224 x 224 ImageNet-normalised input and return two logits (index 1 = Standard).
Segmenters take a 1 x 256 x 256 grayscale input in [0, 1] and return three-class logits (0 bg, 1 FH, 2 PS);
AoP-GAU-Net also returns a direct AoP estimate in degrees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from fin.models.aopgaunet import AoPGAUNet
from fin.models.classifiers import (
    MobileNetV2Classifier,
    MobileNetV3LargeClassifier,
    PSFHSPNet,
    ResNet152Classifier,
    ShuffleNetV2Classifier,
    SqueezeNet11Classifier,
    init_psfhsp_from_resnet18,
)
from fin.models.deepfetalnet import DeepFetalNet
from fin.models.third_party import FATNetGray, build_malunet
from fin.models.unet import UNet

NUM_CLASSES_CLS = 2
NUM_CLASSES_SEG = 3


@dataclass(frozen=True)
class ClassifierSpec:
    """How to build one classifier: constructor(pretrained tree, load ImageNet weights), default init."""

    ctor: Callable[[bool, bool], nn.Module]
    default_init: str
    weights: str | None
    display: str


def _psfhsp(pretrained: bool, load_weights: bool) -> nn.Module:
    model = PSFHSPNet(num_classes=NUM_CLASSES_CLS)
    if pretrained and load_weights:
        init_psfhsp_from_resnet18(model)
    return model


CLASSIFIERS: dict[str, ClassifierSpec] = {
    "deepfetalnet": ClassifierSpec(
        lambda pt, lw: DeepFetalNet(num_classes=NUM_CLASSES_CLS), "scratch", None, "DeepFetalNet"
    ),
    "psfhspnet": ClassifierSpec(
        _psfhsp, "pretrained", "ResNet18_Weights.IMAGENET1K_V1 (partial: conv1, bn1, layer1, layer2)", "PSFHSP-Net"
    ),
    "resnet152": ClassifierSpec(
        lambda pt, lw: ResNet152Classifier(NUM_CLASSES_CLS, pretrained=pt, load_weights=lw),
        "pretrained",
        "ResNet152_Weights.IMAGENET1K_V1",
        "ResNet-152",
    ),
    "squeezenet11": ClassifierSpec(
        lambda pt, lw: SqueezeNet11Classifier(NUM_CLASSES_CLS, pretrained=pt, load_weights=lw),
        "pretrained",
        "SqueezeNet1_1_Weights.IMAGENET1K_V1",
        "SqueezeNet 1.1",
    ),
    "shufflenetv2": ClassifierSpec(
        lambda pt, lw: ShuffleNetV2Classifier(NUM_CLASSES_CLS, pretrained=pt, load_weights=lw),
        "pretrained",
        "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1",
        "ShuffleNetV2",
    ),
    "mobilenetv2": ClassifierSpec(
        lambda pt, lw: MobileNetV2Classifier(NUM_CLASSES_CLS, pretrained=pt, load_weights=lw),
        "pretrained",
        "MobileNet_V2_Weights.IMAGENET1K_V1",
        "MobileNetV2",
    ),
    "mobilenetv3": ClassifierSpec(
        lambda pt, lw: MobileNetV3LargeClassifier(NUM_CLASSES_CLS, pretrained=pt, load_weights=lw),
        "pretrained",
        "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        "MobileNetV3",
    ),
}
SEGMENTERS: dict[str, str] = {"unet": "U-Net", "fatnet": "FAT-Net", "malunet": "MALUNet", "aopgaunet": "AoP-GAU-Net"}
THIRD_PARTY_SEGMENTERS = {"fatnet": "fatnet", "malunet": "malunet"}


def build_classifier(name: str, init: str = "default", load_weights: bool = True) -> tuple[nn.Module, str, str | None]:
    """Build a classifier; ``init`` is ``default``, ``pretrained`` or ``scratch``.

    Returns (model, resolved init, ImageNet weights identifier or None). ``load_weights=False`` builds the
    module tree of the resolved init without downloading ImageNet weights.
    """
    spec = CLASSIFIERS[name]
    resolved = spec.default_init if init == "default" else init
    if name == "deepfetalnet" and resolved == "pretrained":
        raise ValueError("DeepFetalNet has no ImageNet weights")
    pretrained = resolved == "pretrained"
    return spec.ctor(pretrained, load_weights), resolved, (spec.weights if pretrained else None)


def build_segmenter(name: str, pretrained: bool = True) -> nn.Module:
    """Build a segmenter (``pretrained`` only affects FAT-Net's ImageNet encoders)."""
    if name == "unet":
        return UNet(1, NUM_CLASSES_SEG)
    if name == "fatnet":
        return FATNetGray(NUM_CLASSES_SEG, pretrained=pretrained)
    if name == "malunet":
        return build_malunet(NUM_CLASSES_SEG, 1)
    if name == "aopgaunet":
        return AoPGAUNet(1, NUM_CLASSES_SEG)
    raise ValueError(f"unknown segmenter {name}")


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of (unique) parameters; FAT-Net's shared FAM blocks are counted once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def split_output(out: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(segmentation logits, AoP in degrees or None) for any segmenter output."""
    if isinstance(out, tuple):
        return out[0], out[1].squeeze(1)
    return out, None
