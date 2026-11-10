"""Configuration: YAML defaults equal the documented values of the paper; device selection."""

import pytest
import torch

from fin.config import load_config, pick_device, require_setting
from fin.data.datasets import DEFAULT_AUGMENTATION, Augmentation
from fin.robustness.corruptions import base_seed, conditions

AUGMENTATION = {"rotation_deg": 10.0, "scale": [0.9, 1.1], "shift": 0.04, "contrast": [0.9, 1.1], "brightness": 0.1}

EXPECTED = {
    "split": {"seed": 42, "draws": 20000, "fractions": [0.8, 0.1, 0.1]},
    "classification": {
        "seeds": [0, 1, 2],
        "input_size": 224,
        "epochs": 30,
        "patience": 7,
        "batch_size": 32,
        "val_batch_size": 64,
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-2},
        "scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-6},
        "loss": "CrossEntropyLoss",
        "augmentation": AUGMENTATION,
        "selection_threshold": 0.5,
        "evaluation": {"threshold": 0.5, "batch_size": 64, "n_boot": 2000, "bootstrap_seed": 0},
    },
    "cv": {
        "k": 5,
        "seeds": [0],
        "fold_search": {"seed": 42, "draws": 20000, "val_fraction": 0.10},
        "pooled_comparison": {"n_boot": 2000, "seed": 0},
    },
    "segmentation": {
        "input_size": 256,
        "epochs": 60,
        "patience": 10,
        "batch_size": 8,
        "val_batch_size": 16,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "scheduler": {"name": "CosineAnnealingLR", "eta_min": 0.0},
        "augmentation": AUGMENTATION,
        "train_set": "seg",
        "reference_column": "aop_ref_challenge_code",
        "seeds": {"malunet": [0, 1, 2], "aopgaunet": [0, 1, 2], "unet": [0, 1, 2], "fatnet": [0, 1, 2]},
        "loss": {
            "baseline": {"dice": 1.0, "ce": 1.0},
            "aopgaunet": {"dice": 1.0, "ce": 1.0, "boundary": 0.5, "aop": 0.2},
        },
        "evaluation": {"batch_size": 16, "n_boot": 2000, "bootstrap_seed": 0},
    },
    "robustness": {"base_seed": 20261107, "threshold": 0.5, "batch_size": 64},
    "efficiency": {
        "seed": 0,
        "latency": {"warmup": 20, "reps": 100, "batch": 1},
        "throughput": {"batch": 16, "warmup": 3, "reps": 30, "budget_s": 20.0},
    },
}


def _subset(actual, expected, path=""):
    for k, v in expected.items():
        assert k in actual, f"{path}{k} missing"
        if isinstance(v, dict):
            _subset(actual[k], v, f"{path}{k}.")
        else:
            assert actual[k] == v, f"{path}{k}: {actual[k]!r} != {v!r}"


def test_config_defaults_equal_documented_values():
    for name, expected in EXPECTED.items():
        _subset(load_config(name), expected, f"{name}.")


def test_model_lists():
    models = ["deepfetalnet", "psfhspnet", "squeezenet11", "shufflenetv2", "mobilenetv2", "mobilenetv3", "resnet152"]
    assert load_config("classification")["models"] == models
    assert load_config("cv")["models"] == models
    assert load_config("segmentation")["models"] == ["malunet", "aopgaunet", "unet", "fatnet"]


def test_augmentation_defaults_equal_configs():
    for name in ("classification", "segmentation"):
        assert Augmentation.from_config(load_config(name)["augmentation"]) == DEFAULT_AUGMENTATION == Augmentation()


def test_require_setting():
    cfg = load_config("classification")
    require_setting(cfg, "optimizer.name", "AdamW")
    with pytest.raises(ValueError, match="only 'SGD'"):
        require_setting(cfg, "optimizer.name", "SGD")


def test_robustness_conditions():
    assert base_seed() == 20261107
    conds = conditions()
    assert [c[0] for c in conds] == [
        "clean",
        "speckle-0.05",
        "speckle-0.10",
        "speckle-0.20",
        "blur-5",
        "blur-9",
        "blur-15",
        "gauss-0.01",
        "gauss-0.05",
        "gauss-0.10",
    ]
    assert conds[0] == ("clean", "none", None, "None")
    assert conds[4][2] == 5 and isinstance(conds[4][2], int)


def test_pick_device():
    assert pick_device("cpu") == torch.device("cpu")
    assert pick_device("auto").type in ("cpu", "cuda", "mps")
