"""Headline results read directly from the committed run summaries (``results/runs``) and ``splits/frames.csv``.

Used by the README results table (``fin readme-table``) and the README charts (``fin repo-figures``). Nothing here
needs the datasets, the checkpoints or a manuscript folder, so both commands run on a fresh checkout.

Sources
    cls_cv/summary_cv.json      patient-grouped cross-validation: per-fold macro-F1 and AUC of every classifier
    cls/summary.json            locked test split: accuracy, macro-F1 and AUC over seeds
    seg/summary.json            Dice (FH, PS) and AoP MAE of every segmenter on JNU-IFM test and IUGC 2024
    seg/ceiling.json            AoP from the ground-truth masks (the floor of the geometric pipeline)
    seg/<model>/seed0/per_image_iugc.csv   IUGC 2024 reference angles (column ``ref_primary``)
    ext469/summary.json         IUGC 2025 release, frames unseen in training or validation
    splits/frames.csv           constant baseline: mean training reference AoP and the JNU-IFM test references
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fin.io import read_json_if_exists
from fin.reporting.names import CLS, SEG

# README order: the two proposed models first, then the baselines in the order of the paper's tables
CLS_ORDER = ["deepfetalnet"] + [m for m in CLS if m != "deepfetalnet"]
SEG_ORDER = ["aopgaunet"] + [m for m in SEG if m != "aopgaunet"]
AOP_SETS = ("jnu", "iugc", "unseen")
AOP_SET_NAME = {"jnu": "JNU-IFM test", "iugc": "IUGC 2024 test", "unseen": "IUGC 2025, unseen frames"}


@dataclass
class Stat:
    """Mean and sample SD of one metric over seeds or folds, with the individual values."""

    values: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def sd(self) -> float:
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else float("nan")


@dataclass
class Results:
    """Headline numbers of the paper, keyed by model id."""

    cv_k: int = 0
    cv_mf1: dict[str, Stat] = field(default_factory=dict)
    cv_auc: dict[str, Stat] = field(default_factory=dict)
    test_n: int = 0
    test_acc: dict[str, Stat] = field(default_factory=dict)
    test_mf1: dict[str, Stat] = field(default_factory=dict)
    test_auc: dict[str, Stat] = field(default_factory=dict)
    seeds_cls: list[int] = field(default_factory=list)
    dice: dict[tuple[str, str, str], Stat] = field(default_factory=dict)  # (model, set, "fh"|"ps")
    aop: dict[tuple[str, str], Stat] = field(default_factory=dict)  # (source, set); source = model or "<m>.head"
    seeds_seg: dict[str, int] = field(default_factory=dict)
    const_value: float = float("nan")
    const: dict[str, float] = field(default_factory=dict)  # set -> MAE of the constant prediction
    floor: dict[str, float] = field(default_factory=dict)  # set -> MAE of the ground-truth-mask construction
    n_unseen: int = 0


def _stat(d: dict[str, Any] | None) -> Stat | None:
    if not d or "values" not in d:
        return None
    vals = [float(x) for x in d["values"] if x is not None and np.isfinite(x)]
    return Stat(vals) if vals else None


def _ext_stat(runs: dict[str, Any], model: str, head: str) -> Stat | None:
    seeds = runs.get(model, {})
    vals = [
        float(r[head]["unseen_by_ours"]["mae"])
        for r in seeds.values()
        if head in r and "unseen_by_ours" in r[head] and np.isfinite(r[head]["unseen_by_ours"].get("mae", np.nan))
    ]
    return Stat(vals) if vals else None


def load(runs: Path, frames_csv: Path) -> Results:
    """Collect the headline numbers from ``runs`` (a ``results/runs`` folder) and the frame table."""
    r = Results()
    cv = read_json_if_exists(runs / "cls_cv" / "summary_cv.json")
    if cv:
        r.cv_k = int(cv["k"])
        for m, d in cv["models"].items():
            if (s := _stat(d.get("macro_f1"))) is not None:
                r.cv_mf1[m] = s
            if (s := _stat(d.get("auc"))) is not None:
                r.cv_auc[m] = s
    cls = read_json_if_exists(runs / "cls" / "summary.json")
    if cls:
        for m, d in cls.items():
            if not isinstance(d, dict) or "accuracy" not in d:
                continue
            r.test_n = int(d["n_test"])
            r.seeds_cls = [int(s) for s in d["seeds"]]
            for attr, k in (("test_acc", "accuracy"), ("test_mf1", "macro_f1"), ("test_auc", "auc")):
                if (s := _stat(d.get(k))) is not None:
                    getattr(r, attr)[m] = s
    seg = read_json_if_exists(runs / "seg" / "summary.json")
    if seg:
        for m, d in seg["models"].items():
            r.seeds_seg[m] = int(d["n_seeds"])
            for k, s in (("jnu", "jnu_test"), ("iugc", "iugc")):
                if s not in d:
                    continue
                for c, col in (("fh", "dice_FH_mean"), ("ps", "dice_PS_mean")):
                    if (st := _stat(d[s]["seg"].get(col))) is not None:
                        r.dice[(m, k, c)] = st
                for src, head in ((m, "aop_mask"), (f"{m}.head", "aop_head")):
                    if head in d[s] and (st := _stat(d[s][head]["primary"].get("mae"))) is not None:
                        r.aop[(src, k)] = st
    ceil = read_json_if_exists(runs / "seg" / "ceiling.json")
    if ceil:
        for k, s in (("jnu", "jnu_test"), ("iugc", "iugc")):
            if s in ceil:
                r.floor[k] = float(ceil[s]["primary"]["mae"])
    ext = read_json_if_exists(runs / "ext469" / "summary.json")
    if ext:
        r.n_unseen = int(ext["subset_sizes"]["unseen_by_ours"])
        r.const["unseen"] = float(ext["constant"]["unseen_by_ours"]["mae"])
        r.floor["unseen"] = float(ext["gt_mask_floor"]["unseen_by_ours"]["mae"])
        for m in SEG:
            for src, head in ((m, "mask"), (f"{m}.head", "head")):
                if (st := _ext_stat(ext.get("runs", {}), m, head)) is not None:
                    r.aop[(src, "unseen")] = st
    _constant(r, runs, frames_csv)
    return r


def _constant(r: Results, runs: Path, frames_csv: Path) -> None:
    """Constant baseline: the mean training reference AoP, predicted for every test frame."""
    if not frames_csv.exists():
        return
    fr = pd.read_csv(frames_csv, keep_default_na=False, na_values=[""])
    sp = fr[fr.category == 6]
    c = float(sp[sp.split == "train"].aop_ref_challenge_code.mean())
    r.const_value = c
    jnu = sp[sp.split == "test"].aop_ref_challenge_code.to_numpy(float)
    r.const["jnu"] = float(np.nanmean(np.abs(c - jnu)))
    for m in SEG:
        p = runs / "seg" / m / "seed0" / "per_image_iugc.csv"
        if p.exists():
            ref = pd.read_csv(p).ref_primary.to_numpy(float)
            r.const["iugc"] = float(np.nanmean(np.abs(c - ref[np.isfinite(ref)])))
            break
