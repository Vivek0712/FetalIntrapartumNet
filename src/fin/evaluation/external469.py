"""Second external AoP test: the IUGC 2025 release (Zenodo doi 10.5281/zenodo.17356295), 469 images.

Folder layout as released
    <dir>/label.csv   Filename, PS1, PS2, FH1, AOP   (landmarks "(x, y)" on the 512x512 image; AOP filled for 300
                      rows, empty for 169)
    <dir>/*.jpg       469 images, 512x512

Reference for every row: the landmark AoP (``fin.geometry.landmark_aop``), which reproduces the AOP column where it
is given; ``labelled300`` restricts to rows whose AOP is filled. Every trained run (``<runs>/<model>/seed*/best.pt``)
segments the grayscale image resized to 256x256 (``INTER_AREA``) and the AoP is measured on the predicted mask with
the challenge geometry; failures are counted separately. Also reported: the constant baseline (mean
``aop_ref_challenge_code`` of the JNU-IFM training SPHead frames) and, with ``--iugc-root``, the ground-truth-mask floor
on the labelled rows. ``--sources`` (Filename, source, landmark_match, iugc_key, jnu_split and the JNU overlap
columns; committed as ``splits/ext469_sources.csv``) adds per-source subsets and ``unseen_by_ours`` (rows whose image
is not a frame of a JNU-IFM training or validation woman).

    fin ext469 FOLDER [--sources splits/ext469_sources.csv] [--iugc-root DIR] [--out DIR]
Outputs: <out>/<model>_<seed>.csv, <out>/per_image_all.csv, <out>/summary.json.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from fin.config import SPLITS, jnu_frames_csv, load_config, pick_device, runs_dir
from fin.data.datasets import read_gray
from fin.data.labels import FH, IUGC, JNU_RAW, PS, remap
from fin.evaluation.segmentation import predict
from fin.geometry import aop_from_mask, aop_from_masks, landmark_aop
from fin.io import read_frames_csv, read_json, setup_logging, write_csv, write_json
from fin.models import build_segmenter

log = logging.getLogger(__name__)

SIZE = 256
SUMMARY_METRICS = ("mae", "rmse", "median_ae", "bias", "success", "within5", "within10")


def load_folder(d: Path) -> tuple[pd.DataFrame, np.ndarray, float | None]:
    """Label table with the landmark reference, the 256x256 images and the max |landmark - AOP column|."""
    lab = pd.read_csv(d / "label.csv")
    pts = {c: lab[c].map(ast.literal_eval) for c in ("PS1", "PS2", "FH1")}
    lab["ref_landmark"] = [landmark_aop(a, b, c) for a, b, c in zip(pts["PS1"], pts["PS2"], pts["FH1"], strict=True)]
    lab["label_aop"] = pd.to_numeric(lab["AOP"], errors="coerce")
    given = lab.label_aop.notna()
    dmax = float((lab.ref_landmark[given] - lab.label_aop[given]).abs().max()) if given.any() else None
    img = np.stack([cv2.resize(read_gray(d / f), (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in lab.Filename])
    return lab[["Filename", "ref_landmark", "label_aop"]], img, dmax


def stats(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Success rate and error statistics of predictions against references."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    okr = np.isfinite(ref)
    ok = okr & np.isfinite(pred)
    s: dict[str, Any] = {
        "n": int(okr.sum()),
        "n_fail": int((okr & ~ok).sum()),
        "success": float(ok.sum() / max(okr.sum(), 1)),
    }
    if ok.any():
        d = pred[ok] - ref[ok]
        a = np.abs(d)
        s.update(
            mae=float(a.mean()),
            rmse=float(np.sqrt((d**2).mean())),
            median_ae=float(np.median(a)),
            bias=float(d.mean()),
            within5=float((a <= 5).mean()),
            within10=float((a <= 10).mean()),
            pearson_r=float(np.corrcoef(pred[ok], ref[ok])[0, 1]) if ok.sum() > 2 else None,
        )
    return s


def subsets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean row masks of every reported subset."""
    sub = {
        "all469": np.ones(len(df), bool),
        "labelled300": df.label_aop.notna().to_numpy(),
        "unlabelled169": df.label_aop.isna().to_numpy(),
    }
    if "source" in df:
        for s in sorted(df.source.dropna().unique()):
            sub[f"source:{s}"] = (df.source == s).to_numpy()
        if "jnu_split" in df:
            sub["unseen_by_ours"] = (~df.jnu_split.isin(["train", "val"])).to_numpy()
            sub["jnu_trainval_patient"] = df.jnu_split.isin(["train", "val"]).to_numpy()
    return sub


def gt_mask_aop(df: pd.DataFrame, root: Path) -> np.ndarray:
    """Challenge geometry on the IUGC 2024 ground-truth mask of each labelled row (needs landmark_match, iugc_key)."""
    out = np.full(len(df), np.nan)
    for i, r in df.iterrows():
        if not isinstance(r.get("iugc_key"), str):
            continue
        k, sp = r.iugc_key, r.landmark_match
        p = root / "train" / "seg" / k.rsplit("_", 2)[0] / "mask" / k if sp == "train" else root / sp / "seg" / k
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # None: the release has no mask for this frame (skipped)
        if m is None:
            continue
        m = m[..., 0] if m.ndim == 3 else m
        if set(np.unique(m)) <= {0, JNU_RAW["ps"], JNU_RAW["fh"]}:
            y = remap(m, JNU_RAW["fh"], JNU_RAW["ps"])  # training masks use the JNU-IFM convention
        else:
            y = remap(m, IUGC["fh"], IUGC["ps"])
        out[i] = aop_from_mask(cv2.resize(y, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST), FH, PS)
    return out


def _mean_sd(v: list[float | None]) -> dict[str, Any]:
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return {
        "mean": float(a.mean()) if len(a) else None,
        "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "n_runs": int(len(a)),
    }


def run(
    folder: Path,
    seg_runs: Path,
    out: Path,
    sources: Path | None,
    iugc_root: Path | None,
    device: str,
    frames_csv: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every segmentation checkpoint on the folder; write per-run CSVs and the summary."""
    out.mkdir(parents=True, exist_ok=True)
    df, img, dmax = load_folder(folder)
    if sources:
        df = df.merge(pd.read_csv(sources), on="Filename", how="left")
    sub = subsets(df)
    ref = df.ref_landmark.to_numpy(float)
    summ: dict[str, Any] = {
        "folder": os.path.basename(os.path.abspath(folder)),
        "n": len(df),
        "n_label_aop": int(df.label_aop.notna().sum()),
        "landmark_vs_AOP_column_max_absdiff": dmax,
        "reference": "landmark AoP (angle at PS1 between PS1->PS2 and PS1->FH1)",
        "subset_sizes": {k: int(v.sum()) for k, v in sub.items()},
    }
    fr = read_frames_csv(frames_csv or jnu_frames_csv())
    c = float(fr[(fr.category == 6) & (fr.split == "train")].aop_ref_challenge_code.mean())
    summ["constant"] = {"value": c, **{k: stats(np.full(v.sum(), c), ref[v]) for k, v in sub.items()}}
    df["const"] = c
    if iugc_root and "iugc_key" in df:
        df["aop_gtmask"] = gt_mask_aop(df, iugc_root)
        has = np.array([isinstance(k, str) for k in df.iugc_key])
        summ["gt_mask_floor"] = {
            k: stats(df.aop_gtmask.to_numpy()[v & has], ref[v & has]) for k, v in sub.items() if (v & has).any()
        }
    dev = pick_device(device)
    bs = load_config("segmentation")["evaluation"]["batch_size"]
    per_run: dict[str, dict[str, Any]] = {}
    allp = df.copy()
    for ck in sorted(seg_runs.glob("*/seed*/best.pt")):
        rd = ck.parent
        name, seed = read_json(rd / "meta.json")["model"], rd.name
        model = build_segmenter(name, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        model.to(dev)
        lab, head, _ = predict(model, img, dev, bs)
        am = aop_from_masks(lab)
        pi = df[["Filename", "ref_landmark", "label_aop"]].copy()
        pi["aop_mask"] = am
        pi["fail"] = ~np.isfinite(am)
        pi["has_fh"] = (lab == FH).reshape(len(lab), -1).any(1)
        pi["has_ps"] = (lab == PS).reshape(len(lab), -1).any(1)
        if head is not None:
            pi["aop_head"] = head
        write_csv(out / f"{name}_{seed}.csv", pi)
        allp[f"{name}_{seed}"] = am
        r = {"mask": {k: stats(am[v], ref[v]) for k, v in sub.items()}}
        if head is not None:
            r["head"] = {k: stats(head[v], ref[v]) for k, v in sub.items()}
        per_run.setdefault(name, {})[seed] = r
        log.info(
            "%s %s MAE all %.2f fail %d",
            name,
            seed,
            r["mask"]["all469"].get("mae", np.nan),
            r["mask"]["all469"]["n_fail"],
        )
    agg: dict[str, Any] = {}
    for name, runs in per_run.items():
        agg[name] = {}
        for kind in ("mask", "head"):
            if kind not in next(iter(runs.values())):
                continue
            agg[name][kind] = {
                k: {m: _mean_sd([r[kind][k].get(m) for r in runs.values()]) for m in SUMMARY_METRICS} for k in sub
            }
        agg[name]["seeds"] = sorted(runs)
    summ["models"], summ["runs"] = agg, per_run
    write_csv(out / "per_image_all.csv", allp)
    write_json(out / "summary.json", summ, default=float)
    return summ


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=os.environ.get("FIN_EXT469"),
        help="IUGC 2025 release folder (default: $FIN_EXT469)",
    )
    ap.add_argument("--seg-runs", type=Path, default=None, help="default: $FIN_RUNS/seg")
    ap.add_argument("--sources", type=Path, default=SPLITS / "ext469_sources.csv")
    ap.add_argument("--iugc-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: $FIN_RUNS/ext469")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    setup_logging()
    if not a.folder:
        ap.error("give the folder as argument or set FIN_EXT469")
    summ = run(
        Path(a.folder),
        a.seg_runs or runs_dir() / "seg",
        a.out or runs_dir() / "ext469",
        a.sources if a.sources and a.sources.exists() else None,
        a.iugc_root,
        a.device,
    )
    for m, r in summ["models"].items():
        log.info("%s MAE all %s labelled %s", m, r["mask"]["all469"]["mae"], r["mask"]["labelled300"]["mae"])


if __name__ == "__main__":
    main()
