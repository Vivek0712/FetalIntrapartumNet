"""One training step of each pipeline on synthetic tensors, and seeded augmentation."""

import numpy as np
import torch
import torch.nn as nn

from fin.config import load_config, seed_everything
from fin.data.datasets import CachedSegmentationDataset, augment_pair
from fin.models import build_classifier, build_segmenter, split_output
from fin.training.losses import BoundaryLoss, DiceLoss, build_loss


def test_classification_step_on_synthetic_batch():
    seed_everything(0)
    cfg = load_config("classification")
    model, _, _ = build_classifier("deepfetalnet", "scratch")
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    x, y = torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_segmentation_step_with_combined_loss():
    seed_everything(0, python=False)
    cfg = load_config("segmentation")
    model = build_segmenter("aopgaunet")
    crit = build_loss("aopgaunet", cfg["loss"])
    img = (np.random.default_rng(0).random((2, 64, 64)) * 255).astype(np.uint8)
    msk = np.zeros((2, 64, 64), np.uint8)
    msk[:, 10:30, 10:30], msk[:, 40:50, 40:55] = 1, 2
    ds = CachedSegmentationDataset(img, msk, np.array([90.0, np.nan]), train=True, seed=0)
    x, m, r = map(torch.stack, zip(*[ds[i] for i in range(2)], strict=True))
    seg, aop = split_output(model(x))
    loss, parts = crit(seg, m, aop, r)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"dice", "ce", "bnd", "aop_mse_norm"}


def test_baseline_loss_and_dice_of_perfect_prediction():
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[0, 4:8, 4:8] = 1
    logits = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    assert DiceLoss()(logits, target) < 1e-3
    assert BoundaryLoss()(logits, target) < 1e-3
    loss, parts = build_loss("unet", load_config("segmentation")["loss"])(logits, target)
    assert loss < 1e-2 and set(parts) == {"dice", "ce"}


def test_augmentation_is_seeded():
    x = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    m = (x > 0.5).astype(np.uint8)
    a = augment_pair(x, m, np.random.default_rng(3))
    b = augment_pair(x, m, np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert set(np.unique(a[1])) <= {0, 1}
