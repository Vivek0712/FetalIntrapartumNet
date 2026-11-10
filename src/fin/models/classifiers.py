"""The six baseline classifiers (two-class heads), as released by the authors (repository commit 631c725).

PSFHSP-Net        re-implementation of the compressed ResNet-18 of Qiu et al. (stem, two 64-channel basic
                  blocks, one 128-channel basic block with stride 2, average pooling, FC); ImageNet ResNet-18
                  weights are transferred into conv1, bn1, layer1 and layer2 (48 of 50 tensors; fc is new).
ResNet-152, SqueezeNet 1.1, MobileNetV3-Large
                  torchvision models, last layer replaced by a two-way layer.
ShuffleNetV2 1.0x, MobileNetV2
                  pretrained: torchvision models with a two-way last layer; scratch: the authors' own module
                  trees (adaptive pooling instead of a fixed 7x7 pool). The MobileNetV2 scratch tree follows
                  d-li14/mobilenetv2.pytorch (Apache-2.0); the ShuffleNetV2 unit (``channel_shuffle``,
                  ``ShuffleBlock``) is adapted from torchvision (BSD-3-Clause); licence texts in
                  ``third_party_licenses/``, details in THIRD_PARTY.md. The two trees have different module names,
                  so a checkpoint loads only into the tree it was trained with.

``pretrained`` selects the tree; ``load_weights`` controls whether ImageNet weights are downloaded into it
(False builds the same tree with random weights, e.g. before loading a trained checkpoint).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet152_Weights,
    ShuffleNet_V2_X1_0_Weights,
    SqueezeNet1_1_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
    resnet18,
    resnet152,
    shufflenet_v2_x1_0,
    squeezenet1_1,
)


def _weights(enum_value: object, pretrained: bool, load_weights: bool) -> object | None:
    return enum_value if (pretrained and load_weights) else None


def _init_scratch(module: nn.Module) -> None:
    """He-normal convolutions, unit batch norm, N(0, 0.01) linear layers (the authors' scratch init)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


# ------------------------------------------------------------------------------------------------ PSFHSP-Net
class BasicBlock(nn.Module):
    """ResNet basic block with a projection shortcut when the shape changes."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class PSFHSPNet(nn.Module):
    """PSFHSP-Net: 7x7 stem, two 64-channel blocks, one 128-channel block (stride 2), pooling, FC."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def init_psfhsp_from_resnet18(model: PSFHSPNet) -> list[str]:
    """Copy ImageNet ResNet-18 weights into PSFHSP-Net (conv1, bn1, layer1, layer2); returns the copied keys."""
    src = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    dst = model.state_dict()
    copied = []
    for k in dst:
        sk = k.replace("shortcut.0", "downsample.0").replace("shortcut.1", "downsample.1")
        if sk in src and src[sk].shape == dst[k].shape:
            dst[k] = src[sk].clone()
            copied.append(k)
    model.load_state_dict(dst)
    return copied


# ------------------------------------------------------------------------------------------------ torchvision wraps
class ResNet152Classifier(nn.Module):
    """torchvision ResNet-152 with a two-way FC layer."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, load_weights: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = resnet152(weights=_weights(ResNet152_Weights.IMAGENET1K_V1, pretrained, load_weights))
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SqueezeNet11Classifier(nn.Module):
    """torchvision SqueezeNet 1.1 with a two-way final 1x1 convolution."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, load_weights: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = squeezenet1_1(weights=_weights(SqueezeNet1_1_Weights.IMAGENET1K_V1, pretrained, load_weights))
        self.backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.backbone.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetV3LargeClassifier(nn.Module):
    """torchvision MobileNetV3-Large with a two-way last linear layer."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, load_weights: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = mobilenet_v3_large(
            weights=_weights(MobileNet_V3_Large_Weights.IMAGENET1K_V1, pretrained, load_weights)
        )
        self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ------------------------------------------------------------------------------------------------ ShuffleNetV2
# ``channel_shuffle`` and ``ShuffleBlock`` are adapted from torchvision.models.shufflenetv2 (``channel_shuffle``,
# ``InvertedResidual``). Copyright (c) Soumith Chintala 2016, all rights reserved; BSD-3-Clause, full text in
# third_party_licenses/torchvision-BSD-3-Clause.txt. Adapted here: names, type hints, docstrings, error messages.
def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Interleave the channels of ``groups`` groups."""
    batch_size, num_channels, height, width = x.size()
    x = x.view(batch_size, groups, num_channels // groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class ShuffleBlock(nn.Module):
    """ShuffleNetV2 unit: channel split (stride 1) or two down-sampling branches (stride 2), then shuffle."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.stride = stride
        branch_features = out_channels // 2
        if stride == 1 and in_channels != branch_features << 1:
            raise ValueError("stride-1 units need in_channels == out_channels")
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels if self.stride > 1 else branch_features, branch_features, 1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_features, branch_features, 3, stride=stride, padding=1, groups=branch_features, bias=False
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, groups=2)


class ShuffleNetV2Classifier(nn.Module):
    """ShuffleNetV2 1.0x (stage channels 24, 116, 232, 464, 1024) with a two-way head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, load_weights: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        if pretrained:
            self.backbone = shufflenet_v2_x1_0(
                weights=_weights(ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1, pretrained, load_weights)
            )
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            return
        stage_repeats = [4, 8, 4]
        stage_out_channels = [24, 116, 232, 464, 1024]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, stage_out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        input_channel = stage_out_channels[0]
        stages = []
        for stage_idx, num_repeat in enumerate(stage_repeats):
            output_channel = stage_out_channels[stage_idx + 1]
            for i in range(num_repeat):
                stages.append(ShuffleBlock(input_channel, output_channel, stride=2 if i == 0 else 1))
                input_channel = output_channel
        self.stages = nn.Sequential(*stages)
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channel, stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_out_channels[-1], num_classes)
        _init_scratch(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            return self.backbone(x)
        x = self.conv5(self.stages(self.maxpool(self.conv1(x))))
        x = self.adaptive_pool(x).squeeze(3).squeeze(2)
        return self.fc(x)


# ------------------------------------------------------------------------------------------------ MobileNetV2
# The scratch tree (``_make_divisible``, ``InvertedResidual``, the scratch path of ``MobileNetV2Classifier``) follows
# d-li14/mobilenetv2.pytorch (Apache-2.0, full text in third_party_licenses/mobilenetv2.pytorch-Apache-2.0.txt).
# Adapted here: two-way head, type hints and docstrings.
def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    min_value = divisor if min_value is None else min_value
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_3x3_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def _conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual: 1x1 expansion, 3x3 depthwise, linear 1x1 projection."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(_conv_1x1_bn(inp, hidden_dim))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 (width 1.0) with a two-way head."""

    def __init__(
        self, num_classes: int = 2, pretrained: bool = False, width_mult: float = 1.0, load_weights: bool = True
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        if pretrained:
            self.backbone = mobilenet_v2(weights=_weights(MobileNet_V2_Weights.IMAGENET1K_V1, pretrained, load_weights))
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
            return
        setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(32 * width_mult, 8)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult), 8)
        features: list[nn.Module] = [_conv_3x3_bn(3, input_channel, 2)]
        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                features.append(InvertedResidual(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t))
                input_channel = output_channel
        features.append(_conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        _init_scratch(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            return self.backbone(x)
        return self.classifier(self.features(x).mean([2, 3]))
