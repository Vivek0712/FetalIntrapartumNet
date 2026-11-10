"""Patient-disjoint splits: the locked train/val/test split and the patient-grouped cross-validation folds.

Locked split (``configs/split.yaml``)
    Target 80/10/10 of frames; every split must contain all four categories. ``draws`` random permutations of
    the women from ``np.random.default_rng(seed)``; each permutation is filled greedily test -> val -> train
    (a woman goes to test while test stays within half her frames of its target, then likewise val, else
    train). Cost = |frame deviation| / N + L1 distance of each split's category mix from the overall mix; the
    lowest-cost draw is kept.

Cross-validation folds (``configs/cv.yaml``)
    The women are partitioned into K test folds: each of ``draws`` permutations assigns every woman to the
    currently smallest fold; cost = sum |fold frames - N/K| / (N/K) + sum |fold Standard share - overall share|.
    Per fold, validation is about 10% of all frames drawn from the women outside the fold (same cost against a
    target of 10%, ``draws`` permutations from ``default_rng(seed + 1 + k)``); training is every other woman.

    fin splits cv [--frames splits/frames.csv] [--out splits/cv]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fin.config import SPLITS, load_config
from fin.io import read_frames_csv, setup_logging, write_csv, write_json

log = logging.getLogger(__name__)

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
CATEGORIES = [3, 4, 5, 6]


def search_patient_split(
    df: pd.DataFrame, draws: int, seed: int, fractions: tuple[float, float, float]
) -> dict[str, Any]:
    """Search the locked split; returns cost, women -> split index, frame counts and the selected draw."""
    n = len(df)
    target = np.array(fractions) * n
    pats = np.array(sorted(df.patient_id.unique()))
    pc = df.groupby(["patient_id", "category"]).size().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0)
    glob_frac = pc.sum(0).values / n
    pcv = {p: pc.loc[p].values for p in pats}
    rng = np.random.default_rng(seed)
    best = None
    for it in range(draws):
        perm = rng.permutation(pats)
        asg, cnt = {}, np.zeros(3)
        for p in perm:
            k_frames = pcv[p].sum()
            k = 2 if cnt[2] + k_frames / 2 <= target[2] else (1 if cnt[1] + k_frames / 2 <= target[1] else 0)
            asg[p] = k
            cnt[k] += k_frames
        cats = np.zeros((3, 4))
        for p, k in asg.items():
            cats[k] += pcv[p]
        if (cats == 0).any():
            continue
        frac = cats / cats.sum(1, keepdims=True)
        cost = np.abs(cnt - target).sum() / n + np.abs(frac - glob_frac).sum()
        if best is None or cost < best[0]:
            best = (cost, asg, cnt.copy(), it)
    cost, asg, cnt, it = best
    return {"cost": float(cost), "assignment": asg, "frames": cnt, "draw": int(it)}


def check_disjoint(df: pd.DataFrame, key: str = "patient_id", split_col: str = "split") -> None:
    """Raise ValueError if any value of ``key`` (woman or video) appears in two splits."""
    groups = {s: set(d[key]) for s, d in df.groupby(split_col)}
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if groups[a] & groups[b]:
                raise ValueError(f"{key} in both {a} and {b}: {sorted(groups[a] & groups[b])[:5]}")


def make_cv_folds(
    frames: pd.DataFrame, k: int, draws: int, seed: int, val_fraction: float
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Patient-grouped K-fold split; returns one (frame_id, video_id, patient_id, split) table per fold."""
    std = frames.category == 6
    per = frames.assign(std=std).groupby("patient_id").agg(n=("frame_id", "size"), s=("std", "sum"))
    n_all, share = len(frames), std.mean()
    pids, n, s = per.index.to_numpy(), per.n.to_numpy(), per.s.to_numpy()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(draws):
        order = rng.permutation(len(pids))
        tot, sd, asg = np.zeros(k), np.zeros(k), np.zeros(len(pids), int)
        for i in order:
            f = int(np.argmin(tot))
            asg[i], tot[f], sd[f] = f, tot[f] + n[i], sd[f] + s[i]
        cost = np.abs(tot - n_all / k).sum() / (n_all / k) + np.abs(sd / tot - share).sum()
        if best is None or cost < best[0]:
            best = (cost, asg.copy())
    fold_of = dict(zip(pids, best[1], strict=True))

    meta: dict[str, Any] = {"k": k, "draws": draws, "seed": seed, "test_cost": float(best[0]), "folds": []}
    tables = []
    for f in range(k):
        rest = [p for p in pids if fold_of[p] != f]
        rn, rs = per.loc[rest, "n"].to_numpy(), per.loc[rest, "s"].to_numpy()
        vr, vbest = np.random.default_rng(seed + 1 + f), None
        for _ in range(draws):
            o, tn, ts, pick = vr.permutation(len(rest)), 0, 0, []
            for i in o:
                if tn >= val_fraction * n_all:
                    break
                pick.append(i)
                tn += rn[i]
                ts += rs[i]
            c = abs(tn - val_fraction * n_all) / (val_fraction * n_all) + abs(ts / tn - share)
            if vbest is None or c < vbest[0]:
                vbest = (c, [rest[i] for i in pick])
        val = set(vbest[1])
        role = {p: ("test" if fold_of[p] == f else "val" if p in val else "train") for p in pids}
        out = frames[["frame_id", "video_id", "patient_id"]].assign(split=frames.patient_id.map(role))
        tables.append(out)
        g = out.assign(std=std).groupby("split")
        meta["folds"].append(
            {
                "fold": f,
                "val_cost": float(vbest[0]),
                **{
                    sp: {
                        "patients": int(d.patient_id.nunique()),
                        "videos": int(d.video_id.nunique()),
                        "frames": int(len(d)),
                        "standard_share": round(float(d["std"].mean()), 3),
                    }
                    for sp, d in g
                },
            }
        )
        check_disjoint(out, "patient_id")
    tests = pd.concat([t[t.split == "test"] for t in tables])
    if len(tests) != n_all or not tests.frame_id.is_unique:
        raise RuntimeError(f"fold test sets hold {len(tests)} frames ({tests.frame_id.nunique()} unique), not {n_all}")
    return tables, meta


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: write the cross-validation fold files."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    c = sub.add_parser("cv", help="patient-grouped cross-validation folds")
    c.add_argument("--frames", type=Path, default=SPLITS / "frames.csv")
    c.add_argument("--out", type=Path, default=SPLITS / "cv")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config("cv")
    tables, meta = make_cv_folds(
        read_frames_csv(a.frames),
        cfg["k"],
        cfg["fold_search"]["draws"],
        cfg["fold_search"]["seed"],
        cfg["fold_search"]["val_fraction"],
    )
    for f, t in enumerate(tables):
        write_csv(a.out / f"fold{f}.csv", t)
    write_json(a.out / "folds.json", meta)
    for m in meta["folds"]:
        log.info(
            "fold %d %s",
            m["fold"],
            {
                sp: (m[sp]["patients"], m[sp]["videos"], m[sp]["frames"], m[sp]["standard_share"])
                for sp in ("train", "val", "test")
            },
        )


if __name__ == "__main__":
    main()
