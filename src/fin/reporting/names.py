"""Model identifiers and the display names used in the paper, tables and figures."""

from __future__ import annotations

# classifier ids in the order of the paper's tables (proposed model last)
CLS = ["psfhspnet", "resnet152", "squeezenet11", "shufflenetv2", "mobilenetv2", "mobilenetv3", "deepfetalnet"]
CLS_NAME = {
    "deepfetalnet": "DeepFetalNet",
    "psfhspnet": "PSFHSP-Net",
    "resnet152": "ResNet-152",
    "squeezenet11": "SqueezeNet 1.1",
    "shufflenetv2": "ShuffleNetV2 1.0$\\times$",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3": "MobileNetV3-Large",
}
CLS_PLAIN = {
    "deepfetalnet": "DeepFetalNet",
    "psfhspnet": "PSFHSP-Net",
    "resnet152": "ResNet-152",
    "squeezenet11": "SqueezeNet 1.1",
    "shufflenetv2": "ShuffleNetV2 1.0x",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3": "MobileNetV3-Large",
}
SEG = ["unet", "fatnet", "malunet", "aopgaunet"]
SEG_NAME = {"unet": "U-Net", "fatnet": "FAT-Net", "malunet": "MALUNet", "aopgaunet": "AoP-GAU-Net"}
# efficiency results.json "name" -> id
EFF_ID = {
    "DeepFetalNet": "deepfetalnet",
    "PSFHSP-Net": "psfhspnet",
    "ResNet-152": "resnet152",
    "SqueezeNet 1.1": "squeezenet11",
    "ShuffleNetV2": "shufflenetv2",
    "ShuffleNetV2 1.0x": "shufflenetv2",
    "MobileNetV2": "mobilenetv2",
    "MobileNetV3": "mobilenetv3",
    "MobileNetV3-Large": "mobilenetv3",
    "U-Net": "unet",
    "FAT-Net": "fatnet",
    "MALUNet": "malunet",
    "AoP-GAU-Net": "aopgaunet",
}
# key prefix -> evaluation set name of fin.evaluation.segmentation
SETS = {"jnu": "jnu_test", "iugc": "iugc"}
