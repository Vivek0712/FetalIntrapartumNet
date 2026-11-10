"""Segmentation and AoP agreement metrics. Classes: 1 FH, 2 PS; masks are evaluated on the 256x256 model grid.

Dice / IoU
    per-image: Dice_i = 2|P n G| / (|P| + |G|) per frame and class, averaged over frames (primary). If both P and
    G are empty the frame is excluded for that class; if exactly one is empty, Dice = IoU = 0.
    pooled: intersections and sums accumulated over all frames, then one ratio.
HD95 (symmetric)
    surface = mask XOR binary_erosion(mask) (8-connectivity); HD95 = max(P95 of d(surface_P -> surface_G),
    P95 of d(surface_G -> surface_P)), Euclidean distance transform. In pixels of the 256 grid, and in native
    image pixels through anisotropic sampling (JNU: rows 1026/256, columns 1295/256; IUGC: 2, 2). No pixel
    spacing in mm is published for either dataset. Frames with an empty P or G have no HD95 (NaN, counted).
Boundary F1 (BF1)
    precision = fraction of predicted-surface pixels within ``tol`` of the GT surface, recall likewise;
    BF1 = 2PR / (P + R); tol = 2 px (also 1 and 3 px). One empty -> 0; both empty -> NaN.
AoP agreement (predicted - reference, degrees)
    n, n_fail (no ellipse could be fitted), MAE, RMSE, median AE, bias, SD of the differences, Bland-Altman 95%
    limits bias +- 1.96 SD, Pearson r, fraction within 5 and 10 degrees, maximum AE; 95% CI of the MAE by a
    cluster bootstrap over videos (2000 draws). Failures are excluded from the error statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

FG = {1: "FH", 2: "PS"}
_ST = ndi.generate_binary_structure(2, 2)


def surface(m: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (8-connected erosion)."""
    m = m.astype(bool)
    return m & ~ndi.binary_erosion(m, structure=_ST, border_value=0)


def _dist_to(surf: np.ndarray, sampling: tuple[float, float] | None) -> np.ndarray:
    return ndi.distance_transform_edt(~surf, sampling=sampling)


def dice(p: np.ndarray, g: np.ndarray) -> float:
    """Dice of two binary masks; NaN when both are empty (the per-image rule above)."""
    s = p.sum() + g.sum()
    return float(2 * (p & g).sum() / s) if s else float("nan")


def hd95(p: np.ndarray, g: np.ndarray, sampling: tuple[float, float] = (1.0, 1.0)) -> float:
    """Symmetric 95th-percentile Hausdorff distance; NaN when either mask is empty."""
    if not p.any() or not g.any():
        return np.nan
    sp, sg = surface(p), surface(g)
    d_pg = _dist_to(sg, sampling)[sp]
    d_gp = _dist_to(sp, sampling)[sg]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def bf1(p: np.ndarray, g: np.ndarray, tols: tuple[int, ...] = (1, 2, 3)) -> dict[int, float]:
    """Boundary F1 at each pixel tolerance."""
    if not p.any() and not g.any():
        return {t: np.nan for t in tols}
    if not p.any() or not g.any():
        return {t: 0.0 for t in tols}
    sp, sg = surface(p), surface(g)
    dg, dp = _dist_to(sg, None), _dist_to(sp, None)
    out = {}
    for t in tols:
        prec = float((dg[sp] <= t).mean())
        rec = float((dp[sg] <= t).mean())
        out[t] = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return out


def per_image(pred: np.ndarray, gt: np.ndarray, native_sampling: tuple[float, float] = (1.0, 1.0)) -> dict[str, Any]:
    """Per-frame metric arrays of (N, H, W) label maps, plus pooled counts per class."""
    n = len(pred)
    r: dict[str, Any] = {}
    for c, name in FG.items():
        p_c, g_c = pred == c, gt == c
        inter = (p_c & g_c).reshape(n, -1).sum(1).astype(np.float64)
        sp, sg = p_c.reshape(n, -1).sum(1).astype(np.float64), g_c.reshape(n, -1).sum(1).astype(np.float64)
        both_empty = (sp == 0) & (sg == 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[f"dice_{name}"] = np.where(both_empty, np.nan, 2 * inter / (sp + sg))
            r[f"iou_{name}"] = np.where(both_empty, np.nan, inter / (sp + sg - inter))
        r[f"pooled_{name}"] = (inter.sum(), sp.sum(), sg.sum())
        h, hn, b1, b2, b3 = [], [], [], [], []
        for i in range(n):
            h.append(hd95(p_c[i], g_c[i]))
            hn.append(hd95(p_c[i], g_c[i], native_sampling))
            b = bf1(p_c[i], g_c[i])
            b1.append(b[1])
            b2.append(b[2])
            b3.append(b[3])
        r[f"hd95_{name}"], r[f"hd95native_{name}"] = np.array(h), np.array(hn)
        r[f"bf1t1_{name}"], r[f"bf1_{name}"], r[f"bf1t3_{name}"] = np.array(b1), np.array(b2), np.array(b3)
        r[f"pred_empty_{name}"] = sp == 0
        r[f"gt_empty_{name}"] = sg == 0
    return r


def _mean_sd_n(x: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, 0
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan, int(len(x))


def summarise_seg(r: dict[str, Any]) -> dict[str, Any]:
    """Per-image means/SDs/counts, pooled Dice/IoU and empty-prediction counts of a ``per_image`` result."""
    s: dict[str, Any] = {}
    for name in FG.values():
        for k in ("dice", "iou", "hd95", "hd95native", "bf1", "bf1t1", "bf1t3"):
            m, sd, n = _mean_sd_n(r[f"{k}_{name}"])
            s[f"{k}_{name}_mean"], s[f"{k}_{name}_sd"], s[f"{k}_{name}_n"] = m, sd, n
        i, p, g = r[f"pooled_{name}"]
        s[f"pooled_dice_{name}"] = float(2 * i / (p + g)) if p + g else np.nan
        s[f"pooled_iou_{name}"] = float(i / (p + g - i)) if p + g - i else np.nan
        s[f"n_pred_empty_{name}"] = int(r[f"pred_empty_{name}"].sum())
        s[f"n_hd_undefined_{name}"] = int(np.isnan(r[f"hd95_{name}"]).sum())
    s["mean_fg_dice"] = float(np.nanmean([s["dice_FH_mean"], s["dice_PS_mean"]]))
    s["mean_fg_iou"] = float(np.nanmean([s["iou_FH_mean"], s["iou_PS_mean"]]))
    return s


def pixel_accuracy(lab: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame fraction of correctly labelled pixels."""
    n = len(lab)
    return (lab.reshape(n, -1) == gt.reshape(n, -1)).mean(1)


def aop_stats(
    pred: np.ndarray, ref: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """AoP agreement statistics over frames with a reference; failures (NaN predictions) counted separately."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    ok_ref = np.isfinite(ref)
    fail = ok_ref & ~np.isfinite(pred)
    ok = ok_ref & np.isfinite(pred)
    d = pred[ok] - ref[ok]
    ad = np.abs(d)
    s: dict[str, Any] = {"n": int(ok_ref.sum()), "n_fail": int(fail.sum()), "n_used": int(ok.sum())}
    if not ok.any():
        return s
    sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    s.update(
        mae=float(ad.mean()),
        rmse=float(np.sqrt((d**2).mean())),
        median_ae=float(np.median(ad)),
        bias=float(d.mean()),
        sd_diff=sd,
        loa_low=float(d.mean() - 1.96 * sd),
        loa_high=float(d.mean() + 1.96 * sd),
        pearson_r=float(np.corrcoef(pred[ok], ref[ok])[0, 1]) if len(d) > 2 else np.nan,
        within5=float((ad <= 5).mean()),
        within10=float((ad <= 10).mean()),
        max_ae=float(ad.max()),
    )
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(ref))
    g = np.asarray(groups)[ok]
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        boots.append(ad[np.concatenate([idx[u] for u in pick])].mean())
    s["mae_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    s["ci_unit"] = "cluster bootstrap over videos" if len(ug) < len(ad) else "frame bootstrap"
    return s
