"""End-to-end evaluation on the JNU-IFM test split: classifier gate -> segmenter -> geometric AoP.

The gate (threshold 0.5 on p(Standard)) forwards frames; the segmenter's geometric AoP is scored on forwarded frames
that have a reference (SPHead). Classifier seed k is paired with segmenter seed k. Forwarded Non-Standard frames
have no reference and are counted as leaks; SPHead frames the gate rejects are counted as missed. No new inference
is needed: the segmenter was already run on every SPHead test frame (``per_image_jnu_test.csv``). The paper values
(``aop.e2e.*``) are formatted from these statistics by ``fin fill-results``.

    fin e2e [--cls DIR] [--seg DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fin.config import runs_dir
from fin.data.labels import video_of
from fin.evaluation.seg_metrics import aop_stats
from fin.io import setup_logging, write_json
from fin.runs import seed_dirs

log = logging.getLogger(__name__)

GATE_THRESHOLD = 0.5
E2E_N_BOOT = 200


def end_to_end(
    cls_root: Path, seg_root: Path, gate: str = "deepfetalnet", segmenter: str = "aopgaunet"
) -> tuple[list[int], list[dict[str, Any]]]:
    """Seed-paired end-to-end AoP statistics; returns (seeds, per-seed statistics)."""
    cr = dict(seed_dirs(cls_root, gate, "preds_test.npz"))
    sr = dict(seed_dirs(seg_root, segmenter, "per_image_jnu_test.csv"))
    seeds = sorted(set(cr) & set(sr))
    out = []
    for k in seeds:
        z = np.load(cr[k] / "preds_test.npz")
        fwd = dict(zip(z["frame_id"].astype(str), z["p_standard"] >= GATE_THRESHOLD, strict=True))
        y = dict(zip(z["frame_id"].astype(str), z["y"].astype(int), strict=True))
        t = pd.read_csv(sr[k] / "per_image_jnu_test.csv")
        keep = np.array([fwd.get(str(x), False) for x in t.frame_id])
        grp = [video_of(x, "jnu") for x in t.frame_id[keep]]
        r = aop_stats(t.aop_mask.to_numpy(float)[keep], t.ref_primary.to_numpy(float)[keep], grp, n_boot=E2E_N_BOOT)
        r["n_leak"] = int(sum(1 for f, v in fwd.items() if v and y[f] == 0))
        r["n_missed"] = int(len(t) - keep.sum())
        out.append(r)
    return seeds, out


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: write the per-seed end-to-end statistics as JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls", type=Path, default=None, help="default: $FIN_RUNS/cls")
    ap.add_argument("--seg", type=Path, default=None, help="default: $FIN_RUNS/seg")
    ap.add_argument("--out", type=Path, default=None, help="default: $FIN_RUNS/e2e/summary.json")
    a = ap.parse_args(argv)
    setup_logging()
    seeds, stats = end_to_end(a.cls or runs_dir() / "cls", a.seg or runs_dir() / "seg")
    out = a.out or runs_dir() / "e2e" / "summary.json"
    write_json(
        out,
        {
            "gate": "deepfetalnet",
            "segmenter": "aopgaunet",
            "threshold": GATE_THRESHOLD,
            "seeds": seeds,
            "per_seed": stats,
        },
    )
    for k, r in zip(seeds, stats, strict=True):
        log.info(
            "seed %d: n %d fail %d MAE %.2f leak %d missed %d",
            k,
            r["n"],
            r["n_fail"],
            r.get("mae", float("nan")),
            r["n_leak"],
            r["n_missed"],
        )
