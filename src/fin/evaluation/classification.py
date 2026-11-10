"""Evaluate a trained classifier on the test split (and save validation predictions for the threshold choice).

Operating points
    primary   threshold 0.5 on p(Standard) (= argmax of the two logits), fixed before seeing the test set
    secondary Youden-J threshold chosen on the validation predictions of the same run, applied unchanged to test
AUC is threshold-free. 95% CIs: video-level cluster bootstrap, 2000 resamples, percentile; the test split has
only 8 videos from 7 women, so the intervals are wide by construction.

Outputs in the run directory
    preds_val.npz, preds_test.npz   frame_id, video_id, category, y (1 = Standard), p_standard, logits
    metrics_test.json               all metrics, both operating points, CIs, per-category correctness

    fin eval-cls --run RUNS/cls/deepfetalnet/seed0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fin.config import load_config, pick_device
from fin.data.cache import attach_cls_cache
from fin.data.datasets import ClassificationDataset, load_frames
from fin.evaluation.cls_metrics import cluster_bootstrap, per_category_recall, point_metrics, youden_threshold
from fin.io import read_json, read_json_if_exists, setup_logging, write_json, write_npz
from fin.models import build_classifier
from fin.training.classification import subset

log = logging.getLogger(__name__)


def load_run(run: Path, device: torch.device) -> tuple[dict[str, Any], nn.Module]:
    """Reconstruct a run's classifier with its own init tree and load ``best.pt`` strictly."""
    cfg = read_json(run / "config.json")
    model, _, _ = build_classifier(cfg["model"], cfg["init"], load_weights=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"), strict=True)
    return cfg, model.to(device).eval()


@torch.no_grad()
def predict_split(
    model: nn.Module, df: pd.DataFrame, dev: torch.device, size: int, batch_size: int = 64
) -> dict[str, np.ndarray]:
    """Logits and p(Standard) for every frame of a split, with ids, categories and labels."""
    ds = attach_cls_cache(ClassificationDataset(df, size, train=False, cache=True), size)
    logits = [
        model(x.to(dev)).float().cpu() for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ]
    lo = torch.cat(logits)
    return {
        "frame_id": df.frame_id.to_numpy().astype(str),
        "video_id": df.video_id.to_numpy().astype(str),
        "category": df.category.to_numpy().astype(int),
        "y": df.binary_label.to_numpy().astype(int),
        "p_standard": torch.softmax(lo, 1)[:, 1].numpy(),
        "logits": lo.numpy(),
    }


def evaluate(
    run: Path, device: str = "auto", n_boot: int | None = None, frac: float | None = None, force: bool = False
) -> dict[str, Any] | None:
    """Evaluate one run; returns the metrics (None when ``metrics_test.json`` exists and not ``force``)."""
    ecfg = load_config("classification")["evaluation"]
    n_boot = ecfg["n_boot"] if n_boot is None else n_boot
    out = run / "metrics_test.json"
    if out.exists() and not force:
        log.info("[skip] %s exists", out)
        return None
    dev = pick_device(device)
    cfg, model = load_run(run, dev)
    frac = cfg.get("subset", 1.0) if frac is None else frac
    size = int(cfg["input"].split("x")[0])
    thr, bseed = ecfg["threshold"], ecfg["bootstrap_seed"]

    t0 = time.time()
    pv = predict_split(model, subset(load_frames("val", "cls"), frac, cfg["seed"]), dev, size, ecfg["batch_size"])
    pt = predict_split(model, subset(load_frames("test", "cls"), frac, cfg["seed"]), dev, size, ecfg["batch_size"])
    write_npz(run / "preds_val.npz", **pv)
    write_npz(run / "preds_test.npz", **pt)

    thr_y = youden_threshold(pv["y"], pv["p_standard"])
    res = {
        "model": cfg["model"],
        "display_name": cfg["display_name"],
        "seed": cfg["seed"],
        "init": cfg["init"],
        "params": cfg["params"],
        "subset": frac,
        "positive_class": "Standard (SPHead)",
        "primary_threshold": thr,
        "test": point_metrics(pt["y"], pt["p_standard"], thr),
        "test_ci": cluster_bootstrap(pt["y"], pt["p_standard"], pt["video_id"], thr, n_boot, seed=bseed),
        "test_per_category": per_category_recall(pt["category"], pt["y"], pt["p_standard"], thr),
        "youden_val": {
            "threshold": thr_y,
            "val": point_metrics(pv["y"], pv["p_standard"], thr_y),
            "test": point_metrics(pt["y"], pt["p_standard"], thr_y),
            "test_ci": cluster_bootstrap(pt["y"], pt["p_standard"], pt["video_id"], thr_y, n_boot, seed=bseed),
        },
        "val": point_metrics(pv["y"], pv["p_standard"], thr),
        "train_summary": read_json_if_exists(run / "done.json"),
        "eval_seconds": time.time() - t0,
    }
    write_json(out, res, indent=2)
    t, ci = res["test"], res["test_ci"]["ci95"]
    log.info(
        "%s seed%s TEST n=%d acc %.4f mF1 %.4f AUC %.4f [%.4f, %.4f] sens %.4f spec %.4f | youden thr %.3f",
        cfg["model"],
        cfg["seed"],
        t["n"],
        t["accuracy"],
        t["macro_f1"],
        t["auc"],
        ci["auc"][0],
        ci["auc"][1],
        t["standard"]["recall"],
        t["standard"]["specificity"],
        thr_y,
    )
    return res


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run directory written by fin train-cls")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-boot", type=int, default=None, help="default: configs/classification.yaml")
    ap.add_argument("--subset", type=float, default=None, help="default: the run's own subset fraction")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    evaluate(a.run, a.device, a.n_boot, a.subset, a.force)


if __name__ == "__main__":
    main()
