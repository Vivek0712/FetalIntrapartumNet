"""Binary classification metrics.

Conventions
    y : 1 = Standard (SPHead), 0 = Non-Standard (``binary_label``)
    p : predicted probability of Standard (softmax column 1)
    The positive class for sensitivity and AUC is Standard. Per-class precision, recall and specificity are
    reported for both classes; the recall of one class equals the specificity of the other.
    Bootstrap: percentile CI from a cluster bootstrap over videos (resample videos with replacement and keep
    every frame of a drawn video). Resamples with a single class are skipped for AUC and counted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve

CLASS_NAMES = {1: "standard", 0: "non_standard"}
CATEGORY_NAMES = {3: "None", 4: "OnlySP", 5: "OnlyHead", 6: "SPHead"}


def confusion(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """2x2 matrix; rows = true [non_standard, standard], columns = predicted [non_standard, standard]."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    return np.array([[np.sum((y == t) & (yhat == q)) for q in (0, 1)] for t in (0, 1)], dtype=int)


def _safe(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/specificity/F1, macro-F1, AUC and the confusion matrix."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    cm = confusion(y, (p >= thr).astype(int))
    out: dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(thr),
        "confusion_matrix": cm.tolist(),
        "accuracy": _safe(np.trace(cm), cm.sum()),
    }
    f1s = []
    for c in (1, 0):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        f1 = _safe(2 * tp, 2 * tp + fp + fn)
        f1s.append(f1)
        out[CLASS_NAMES[c]] = {
            "precision": _safe(tp, tp + fp),
            "recall": _safe(tp, tp + fn),
            "specificity": _safe(tn, tn + fp),
            "f1": f1,
            "support": int(cm[c].sum()),
        }
    out["macro_f1"] = float(np.mean(f1s))
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def summary_metrics(m: dict[str, Any]) -> dict[str, float]:
    """Accuracy, macro-F1, AUC, sensitivity and specificity (Standard positive) of a ``point_metrics`` result."""
    return {
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "auc": m["auc"],
        "sensitivity": m["standard"]["recall"],
        "specificity": m["standard"]["specificity"],
    }


def fast_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    """Vectorised acc, prec, sens, spec, f1 (Standard), mf1 and rank-based AUC, for bootstrap loops."""
    yh = p >= thr
    pos, neg = y == 1, y == 0
    tp, fn = np.sum(yh & pos), np.sum(~yh & pos)
    fp, tn = np.sum(yh & neg), np.sum(~yh & neg)
    n1, n0 = pos.sum(), neg.sum()

    def sd(a: float, b: float) -> float:
        return a / b if b > 0 else np.nan

    f1s, f1n = sd(2 * tp, 2 * tp + fp + fn), sd(2 * tn, 2 * tn + fn + fp)
    if n1 and n0:
        r = rankdata(p)
        auc = (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    else:
        auc = np.nan
    return {
        "acc": (tp + tn) / len(y),
        "prec": sd(tp, tp + fp),
        "sens": sd(tp, tp + fn),
        "spec": sd(tn, tn + fp),
        "f1": f1s,
        "mf1": (f1s + f1n) / 2,
        "auc": auc,
    }


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (Standard positive)."""
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), np.asarray(p, dtype=float))
    t = float(thr[int(np.argmax(tpr - fpr))])
    return min(t, 1.0) if np.isfinite(t) else 1.0


def cluster_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    thr: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile CIs of accuracy, macro-F1, AUC, sensitivity and specificity, resampling whole clusters."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "macro_f1", "auc", "sensitivity", "specificity")
    vals: dict[str, list[float]] = {k: [] for k in keys}
    skipped = 0
    for _ in range(n_boot):
        ii = np.concatenate([idx_by[j] for j in rng.integers(0, len(uniq), len(uniq))])
        yy, pp = y[ii], p[ii]
        m = point_metrics(yy, pp, thr)
        vals["accuracy"].append(m["accuracy"])
        vals["macro_f1"].append(m["macro_f1"])
        vals["sensitivity"].append(m["standard"]["recall"])
        vals["specificity"].append(m["standard"]["specificity"])
        if len(np.unique(yy)) == 2:
            vals["auc"].append(m["auc"])
        else:
            skipped += 1
    ci = {}
    for k in keys:
        v = np.asarray(vals[k], dtype=float)
        v = v[np.isfinite(v)]
        ci[k] = (
            [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))] if len(v) else [float("nan")] * 2
        )
    return {
        "method": "video-level cluster bootstrap, percentile",
        "n_boot": n_boot,
        "seed": seed,
        "n_clusters": int(len(uniq)),
        "n_skipped_single_class_auc": skipped,
        "ci95": ci,
    }


def per_category_recall(category: np.ndarray, y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, Any]:
    """Fraction of frames of each JNU-IFM category predicted with their correct binary label."""
    category = np.asarray(category)
    yhat = (np.asarray(p) >= thr).astype(int)
    out = {}
    for c, name in CATEGORY_NAMES.items():
        mask = category == c
        if mask.any():
            out[name] = {"n": int(mask.sum()), "correct": float(np.mean(yhat[mask] == np.asarray(y)[mask]))}
    return out
