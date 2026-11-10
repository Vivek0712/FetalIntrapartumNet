"""Classification robustness on the corrupted JNU-IFM test split.

Every trained run ``<cls_runs>/<model>/seed<k>/`` is reconstructed with its own init tree and loaded strictly, then
evaluated on the 630 test frames under each condition of ``configs/robustness.yaml`` (clean included). Input
pipeline as in the clean evaluation: cached 224x224 uint8 frame -> [0, 1] -> corruption -> 3-channel replication ->
ImageNet mean/std. Threshold 0.5 on p(Standard); Standard is the positive class. The clean condition must reproduce
the run's ``metrics_test.json`` AUC within 1e-3 (recorded as ``clean_check``).

Outputs
    <out>/<model>/seed<k>/preds.npz      frame_id, video_id, y and p_<condition> for every condition
    <out>/<model>/seed<k>/metrics.json   point metrics per condition
    <out>/summary.json                   per model and condition: mean/SD over seeds and per-seed values

    fin robustness [--cls-runs DIR] [--out DIR] [--models ...] [--seeds ...] [--limit N]
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

from fin.config import empty_cache, load_config, pick_device, runs_dir
from fin.data.cache import load_cls_images
from fin.data.datasets import load_frames, to_classifier_input
from fin.evaluation.classification import load_run
from fin.evaluation.cls_metrics import point_metrics
from fin.io import portable_path, read_json, setup_logging, write_json, write_npz
from fin.models import CLASSIFIERS
from fin.robustness.corruptions import base_seed, conditions, corrupt

log = logging.getLogger(__name__)

ORDER = ["deepfetalnet", "psfhspnet", "resnet152", "squeezenet11", "shufflenetv2", "mobilenetv2", "mobilenetv3"]


def test_images(size: int = 224, limit: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Test-split frame table and the cached uint8 frames in the same order (the clean evaluation input)."""
    df = load_frames("test", "cls")
    if limit:
        df = (
            df.sort_values("frame_id")
            .groupby("binary_label", group_keys=False)
            .head(max(1, limit // 2))
            .reset_index(drop=True)
        )
    return df, load_cls_images(df.frame_id.astype(str).to_numpy(), size)


@torch.no_grad()
def predict_probs(model: nn.Module, imgs_float: np.ndarray, dev: torch.device, bs: int = 64) -> np.ndarray:
    """p(Standard) for a stack of [0, 1] grayscale images."""
    ps = []
    for i in range(0, len(imgs_float), bs):
        lo = model(torch.from_numpy(to_classifier_input(imgs_float[i : i + bs])).to(dev)).float().cpu()
        ps.append(torch.softmax(lo, 1)[:, 1].numpy())
    return np.concatenate(ps)


def summarise(vals: list[float | None]) -> dict[str, Any]:
    """Mean, sample SD, count and values of the finite entries."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": None, "sd": None, "n": 0, "values": []}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
        "n": int(len(v)),
        "values": [float(x) for x in v],
    }


def run(
    cls_runs: Path,
    out: Path,
    models: list[str] | None,
    seeds: list[int] | None,
    only_conditions: list[str] | None,
    device: str,
    limit: int,
    force: bool,
) -> dict[str, Any]:
    """Evaluate every finished run under every condition and write the summary."""
    cfg = load_config("robustness")
    thr, bs = cfg["threshold"], cfg["batch_size"]
    dev = pick_device(device)
    conds = [c for c in conditions() if only_conditions is None or c[0] in only_conditions]
    df, img_u8 = test_images(224, limit or None)
    fids = df.frame_id.astype(str).to_numpy()
    y = df.binary_label.to_numpy().astype(int)
    t0 = time.time()
    corrupted = {cid: np.stack([corrupt(im, cid, f) for im, f in zip(img_u8, fids, strict=True)]) for cid, *_ in conds}
    log.info("[corrupt] %d conditions x %d frames in %.1fs", len(conds), len(fids), time.time() - t0)

    per_model: dict[str, list[dict[str, Any]]] = {}
    for ck in sorted(cls_runs.glob("*/seed*/best.pt")):
        rd = ck.parent
        m, seed = rd.parent.name, int(rd.name[4:])
        if (models and m not in models) or (seeds is not None and seed not in seeds):
            continue
        if not (rd / "done.json").exists():
            log.info("[skip] %s: training not finished", rd)
            continue
        od = out / m / f"seed{seed}"
        mfile = od / "metrics.json"
        if mfile.exists() and not force:
            res = read_json(mfile)
            if res.get("n_frames") == len(fids) and set(res["conditions"]) >= {c[0] for c in conds}:
                per_model.setdefault(m, []).append(res)
                log.info("[cached] %s seed%d", m, seed)
                continue
        run_cfg, model = load_run(rd, dev)
        t1 = time.time()
        preds: dict[str, np.ndarray] = {"frame_id": fids, "video_id": df.video_id.astype(str).to_numpy(), "y": y}
        res = {
            "model": m,
            "display_name": CLASSIFIERS[m].display if m in CLASSIFIERS else m,
            "seed": seed,
            "init": run_cfg["init"],
            "run": portable_path(rd),
            "n_frames": int(len(fids)),
            "limit": limit,
            "threshold": thr,
            "conditions": {},
        }
        for cid, *_ in conds:
            p = predict_probs(model, corrupted[cid], dev, bs)
            preds[f"p_{cid}"] = p
            res["conditions"][cid] = point_metrics(y, p, thr)
        mt = rd / "metrics_test.json"
        if "clean" in res["conditions"] and mt.exists() and not limit:
            ref = read_json(mt)["test"]["auc"]
            got = res["conditions"]["clean"]["auc"]
            res["clean_check"] = {
                "metrics_test_auc": ref,
                "robustness_clean_auc": got,
                "ok": bool(abs(ref - got) < 1e-3),
            }
            if not res["clean_check"]["ok"]:
                log.warning("%s seed%d: clean AUC %.4f != metrics_test %.4f", m, seed, got, ref)
        res["seconds"] = time.time() - t1
        write_npz(od / "preds.npz", **preds)
        write_json(mfile, res)
        log.info(
            "%s seed%d: %s (%.1fs)",
            m,
            seed,
            " ".join(f"{c}={res['conditions'][c]['auc']:.3f}" for c in res["conditions"]),
            res["seconds"],
        )
        per_model.setdefault(m, []).append(res)
        del model
        empty_cache(dev)

    summ: dict[str, Any] = {
        "protocol": {
            "split": "JNU-IFM test (cls)",
            "n_frames": int(len(fids)),
            "limit": limit,
            "positive_class": "Standard (SPHead)",
            "threshold": thr,
            "base_seed": base_seed(),
            "conditions": [{"id": c[0], "corruption": c[1], "param": c[2], "label": c[3]} for c in conds],
            "definitions": "see fin/robustness/corruptions.py",
        },
        "models": {},
    }
    for m in [x for x in ORDER if x in per_model] + [x for x in per_model if x not in ORDER]:
        rs = sorted(per_model[m], key=lambda r: r["seed"])
        o: dict[str, Any] = {
            "display_name": rs[0]["display_name"],
            "seeds": [r["seed"] for r in rs],
            "conditions": {},
            "clean_check": [r.get("clean_check") for r in rs],
        }
        for cid, *_ in conds:
            ms = [r["conditions"][cid] for r in rs]
            o["conditions"][cid] = {
                "accuracy": summarise([x["accuracy"] for x in ms]),
                "sensitivity": summarise([x["standard"]["recall"] for x in ms]),
                "specificity": summarise([x["standard"]["specificity"] for x in ms]),
                "precision": summarise([x["standard"]["precision"] for x in ms]),
                "macro_f1": summarise([x["macro_f1"] for x in ms]),
                "auc": summarise([x["auc"] for x in ms]),
                "confusion_matrices": [x["confusion_matrix"] for x in ms],
            }
        summ["models"][m] = o
    write_json(out / "summary.json", summ)
    log.info("wrote %s (%d models)", out / "summary.json", len(summ["models"]))
    return summ


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls-runs", type=Path, default=None, help="default: $FIN_RUNS/cls")
    ap.add_argument("--out", type=Path, default=None, help="default: $FIN_RUNS/robustness")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of condition ids (default: all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="use only this many test frames (quick test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    run(
        a.cls_runs or runs_dir() / "cls",
        a.out or runs_dir() / "robustness",
        a.models,
        a.seeds,
        a.conditions,
        a.device,
        a.limit,
        a.force,
    )


if __name__ == "__main__":
    main()
