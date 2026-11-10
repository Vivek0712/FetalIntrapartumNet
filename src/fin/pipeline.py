"""Resumable run schedules. Every run is a separate process (as in the reported runs); finished runs are skipped.

cls          train + evaluate every classifier and seed of ``configs/classification.yaml``, then aggregate
cv           the same for every fold of ``splits/cv`` (one seed), then the cross-validation summary
authorsplit  DeepFetalNet and MobileNetV3-Large on the video-level split, then the split check
seg          segmentation cache, AoP ceiling, train + evaluate every segmenter and seed (seed-major), then aggregate

    fin schedule cls [--runs DIR] [--models ...] [--seeds ...]
    fin schedule cv | authorsplit | seg
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from fin.config import SPLITS, load_config, runs_dir, seg_cache_dir
from fin.evaluation.aggregate import aggregate_cls, aggregate_cv, aggregate_seg, author_split_check
from fin.io import setup_logging

log = logging.getLogger(__name__)

AUTHOR_SPLIT_MODELS = ["deepfetalnet", "mobilenetv3"]


def _fin(args: list[str], env: dict[str, str] | None = None) -> int:
    """Run ``fin <args>`` in a child process; returns its exit code."""
    return subprocess.call([sys.executable, "-m", "fin.cli", *args], env={**os.environ, **(env or {})})


def cls_schedule(runs: Path, models: list[str], seeds: list[int], split_file: Path | None = None) -> None:
    """Train and evaluate every (model, seed) under ``runs``; then write the summary."""
    env = {"FIN_SPLIT_FILE": str(split_file)} if split_file else {}
    if _fin(["cache", "cls"]) != 0:
        raise RuntimeError("classification cache failed")
    for m in models:
        for s in seeds:
            out = runs / m / f"seed{s}"
            log.info("train %s seed %d -> %s", m, s, out)
            _fin(["train-cls", "--model", m, "--seed", str(s), "--out", str(out)], env)
            if not (out / "done.json").exists():
                log.error("train failed (no done.json): %s seed %d", m, s)
                continue
            _fin(["eval-cls", "--run", str(out)], env)
    aggregate_cls(runs)


def seg_schedule(runs: Path, models: list[str], seeds: dict[str, list[int]]) -> None:
    """Segmentation cache, ceiling, then train + evaluate (seed-major); then write the summary."""
    cache = seg_cache_dir()
    cached = (cache / "jnu_img.npy").exists() and (cache / "iugc_img.npy").exists()
    if not cached and _fin(["cache", "seg"]) != 0:
        raise RuntimeError("segmentation cache failed")
    if not (runs / "ceiling.json").exists():
        _fin(["eval-seg", "--ceiling", str(runs / "ceiling.json")])
    for s in sorted({k for v in seeds.values() for k in v}):
        for m in models:
            if s not in seeds[m]:
                continue
            out = runs / m / f"seed{s}"
            if not (out / "train_done.json").exists():
                log.info("train %s seed%d", m, s)
                _fin(["train-seg", "--model", m, "--seed", str(s), "--out", str(out)])
                if not (out / "train_done.json").exists():
                    log.error("train %s seed%d failed", m, s)
                    continue
            metrics = out / "metrics.json"
            if not metrics.exists() or (out / "best.pt").stat().st_mtime > metrics.stat().st_mtime:
                log.info("eval %s seed%d", m, s)
                _fin(["eval-seg", "--run", str(out)])
        aggregate_seg(runs)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["cls", "cv", "authorsplit", "seg"])
    ap.add_argument("--runs", type=Path, default=None, help="default: $FIN_RUNS/<cls|cls_cv|cls_authorsplit|seg>")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    setup_logging()
    if a.kind == "cls":
        cfg = load_config("classification")
        cls_schedule(a.runs or runs_dir() / "cls", a.models or cfg["models"], a.seeds or cfg["seeds"])
    elif a.kind == "cv":
        cfg = load_config("cv")
        root = a.runs or runs_dir() / "cls_cv"
        for m in a.models or cfg["models"]:
            for k in range(cfg["k"]):
                cls_schedule(root / f"fold{k}", [m], a.seeds or cfg["seeds"], SPLITS / "cv" / f"fold{k}.csv")
        aggregate_cv(root)
    elif a.kind == "authorsplit":
        root = a.runs or runs_dir() / "cls_authorsplit"
        cls_schedule(
            root,
            a.models or AUTHOR_SPLIT_MODELS,
            a.seeds or load_config("classification")["seeds"],
            SPLITS / "video_level_split.csv",
        )
        author_split_check(root, runs_dir() / "cls" / "summary.json")
    else:
        cfg = load_config("segmentation")
        models = a.models or cfg["models"]
        seeds = {m: (a.seeds or cfg["seeds"][m]) for m in models}
        seg_schedule(a.runs or runs_dir() / "seg", models, seeds)


if __name__ == "__main__":
    main()
