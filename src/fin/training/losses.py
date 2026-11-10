"""Segmentation losses.

Baselines (U-Net, FAT-Net, MALUNet): cross-entropy + soft Dice (mean over all three classes including
background, smooth = 1).

AoP-GAU-Net: the authors' combined loss, 1.0 Dice + 1.0 CE + 0.5 boundary + 0.2 AoP, with three corrections
relative to the released code:
  1. Boundary term: Laplacian-edge MSE computed per foreground class on one-hot targets (FH and PS) and
     averaged. The released version compared the FH probability channel with the Laplacian of the label-valued
     target (0/1/2), so PS edges were penalised against an FH-only prediction.
  2. AoP term on a normalised scale, MSE(pred / 180, ref / 180), so it has the order of the other terms.
  3. Frames without a reference AoP (NaN) contribute no AoP loss; a batch without any gives 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """1 - mean over classes of the soft Dice coefficient."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nc = logits.shape[1]
        probs = F.softmax(logits.float(), dim=1)
        oh = F.one_hot(targets.long(), nc).permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class BoundaryLoss(nn.Module):
    """Laplacian-edge MSE between predicted foreground probabilities and one-hot targets."""

    def __init__(self) -> None:
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nc = logits.shape[1]
        probs = F.softmax(logits.float(), dim=1)[:, 1:]
        oh = F.one_hot(targets.long(), nc).permute(0, 3, 1, 2).float()[:, 1:]
        k = self.laplacian.expand(nc - 1, 1, 3, 3)
        pe = F.conv2d(probs, k, padding=1, groups=nc - 1).abs()
        te = F.conv2d(oh, k, padding=1, groups=nc - 1).abs()
        return F.mse_loss(pe, te)


class SegLoss(nn.Module):
    """Cross-entropy + Dice (baselines); returns (loss, components)."""

    def __init__(self, dice_w: float = 1.0, ce_w: float = 1.0) -> None:
        super().__init__()
        self.dice, self.ce = DiceLoss(), nn.CrossEntropyLoss()
        self.w = (dice_w, ce_w)

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        aop_pred: torch.Tensor | None = None,
        aop_ref: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        d, c = self.dice(seg_logits, seg_targets), self.ce(seg_logits.float(), seg_targets)
        return self.w[0] * d + self.w[1] * c, {"dice": d.item(), "ce": c.item()}


class CombinedLoss(nn.Module):
    """AoP-GAU-Net loss: weighted Dice, CE, boundary and normalised AoP terms; returns (loss, components)."""

    def __init__(self, dice_w: float = 1.0, bnd_w: float = 0.5, ce_w: float = 1.0, aop_w: float = 0.2) -> None:
        super().__init__()
        self.dice, self.bnd, self.ce = DiceLoss(), BoundaryLoss(), nn.CrossEntropyLoss()
        self.w = (dice_w, ce_w, bnd_w, aop_w)

    def forward(
        self, seg_logits: torch.Tensor, seg_targets: torch.Tensor, aop_pred: torch.Tensor, aop_ref: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        d = self.dice(seg_logits, seg_targets)
        c = self.ce(seg_logits.float(), seg_targets)
        b = self.bnd(seg_logits, seg_targets)
        ok = torch.isfinite(aop_ref)
        if ok.any():
            a = F.mse_loss(aop_pred[ok].float() / 180.0, aop_ref[ok].float() / 180.0)
        else:
            a = seg_logits.new_zeros(())
        loss = self.w[0] * d + self.w[1] * c + self.w[2] * b + self.w[3] * a
        return loss, {"dice": d.item(), "ce": c.item(), "bnd": b.item(), "aop_mse_norm": a.item()}


def build_loss(model_name: str, weights: dict[str, dict[str, float]]) -> nn.Module:
    """Loss for a segmenter; ``weights`` is the ``loss`` block of ``configs/segmentation.yaml``."""
    if model_name == "aopgaunet":
        w = weights["aopgaunet"]
        return CombinedLoss(dice_w=w["dice"], bnd_w=w["boundary"], ce_w=w["ce"], aop_w=w["aop"])
    w = weights["baseline"]
    return SegLoss(dice_w=w["dice"], ce_w=w["ce"])
