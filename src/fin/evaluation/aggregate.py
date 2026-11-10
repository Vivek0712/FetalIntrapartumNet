"""Aggregate per-run outputs into summaries.

cls           <root>/<model>/seed<k>/metrics_test.json -> <root>/summary.{json,md}: mean +/- sample SD over seeds
              of the test metrics (threshold 0.5 and the validation-chosen Youden threshold), per-seed AUC CIs, best
              epoch and epochs run.
seg           <root>/<model>/seed<k>/metrics.json -> <root>/summary.{json,md}: mean and SD over seeds of every
              per-run segmentation and AoP statistic.
cv            patient-grouped K-fold cross-validation -> <cv_root>/summary_cv.{json,md}: per-fold metrics and their
              mean/SD/range, ranks per fold, pooled out-of-fold metrics (every frame tested once) and the paired
              difference to DeepFetalNet with a 95% CI from a bootstrap over women.
author-split  the video-level split of the earlier manuscript version (splits/video_level_split.csv) against the
              locked patient-disjoint split: test metrics overall and for frames of women who also have videos in
              training ("seen") or not ("unseen") -> <root>/author_split_check.{json,md}.

    fin aggregate cls [ROOT]       fin aggregate seg [ROOT]
    fin aggregate cv [--cv-root DIR]
    fin aggregate author-split [--root DIR] [--fixed-summary FILE]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fin.config import SPLITS, load_config, runs_dir
from fin.evaluation.cls_metrics import fast_metrics, point_metrics, summary_metrics
from fin.io import portable_path, read_frames_csv, read_json, setup_logging, write_json, write_text

log = logging.getLogger(__name__)

CLS_ORDER = ["deepfetalnet", "psfhspnet", "resnet152", "squeezenet11", "shufflenetv2", "mobilenetv2", "mobilenetv3"]
CV_MODELS = ["deepfetalnet", "psfhspnet", "squeezenet11", "shufflenetv2", "mobilenetv2", "mobilenetv3", "resnet152"]
CV_NAMES = {
    "deepfetalnet": "DeepFetalNet",
    "psfhspnet": "PSFHSP-Net",
    "squeezenet11": "SqueezeNet 1.1",
    "shufflenetv2": "ShuffleNetV2",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3": "MobileNetV3-Large",
    "resnet152": "ResNet-152",
}
SEG_MODELS = [("unet", "U-Net"), ("fatnet", "FAT-Net"), ("malunet", "MALUNet"), ("aopgaunet", "AoP-GAU-Net")]
SEG_SETS = [
    ("jnu_test", "JNU-IFM test (SPHead, 379 frames, 7 patients)"),
    ("iugc", "IUGC 2024 test (external, 300 frames)"),
]
SEG_KEYS = [
    "dice_FH_mean",
    "dice_PS_mean",
    "mean_fg_dice",
    "iou_FH_mean",
    "iou_PS_mean",
    "pooled_dice_FH",
    "pooled_dice_PS",
    "pooled_iou_FH",
    "pooled_iou_PS",
    "hd95_FH_mean",
    "hd95_PS_mean",
    "hd95native_FH_mean",
    "hd95native_PS_mean",
    "bf1_FH_mean",
    "bf1_PS_mean",
    "n_pred_empty_FH",
    "n_pred_empty_PS",
]
AOP_KEYS = [
    "n_fail",
    "mae",
    "rmse",
    "median_ae",
    "bias",
    "sd_diff",
    "loa_low",
    "loa_high",
    "pearson_r",
    "within5",
    "within10",
]
SPLIT_CHECK_MODELS = {"deepfetalnet": "DeepFetalNet", "mobilenetv3": "MobileNetV3-Large"}


def _seed_dirs(root: Path, marker: str) -> list[Path]:
    return sorted(root.glob(f"*/seed*/{marker}"))


# ------------------------------------------------------------------------------------------------ classification
def _ms_cls(v: list[float]) -> dict[str, Any]:
    a = np.asarray(v, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
        "n": int(len(a)),
        "values": a.tolist(),
    }


def _fmt_cls(d: dict[str, Any], pct: bool = True) -> str:
    k = 100 if pct else 1
    if d["n"] > 1:
        return f"{d['mean'] * k:.2f} +/- {d['sd'] * k:.2f}" if pct else f"{d['mean']:.4f} +/- {d['sd']:.4f}"
    return f"{d['mean'] * k:.2f}" if pct else f"{d['mean']:.4f}"


def aggregate_cls(root: Path) -> dict[str, Any]:
    """Write ``summary.json`` and ``summary.md`` for a classification run root."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for f in _seed_dirs(root, "metrics_test.json"):
        r = read_json(f)
        rows.setdefault(r["model"], []).append(r)
    summ: dict[str, Any] = {}
    for m in [x for x in CLS_ORDER if x in rows] + [x for x in rows if x not in CLS_ORDER]:
        rs = sorted(rows[m], key=lambda r: r["seed"])
        t = [r["test"] for r in rs]
        y = [r["youden_val"]["test"] for r in rs]
        ts = [r["train_summary"] for r in rs if r["train_summary"]]
        summ[m] = {
            "display_name": rs[0]["display_name"],
            "init": rs[0]["init"],
            "params": rs[0]["params"],
            "seeds": [r["seed"] for r in rs],
            "n_test": t[0]["n"],
            "accuracy": _ms_cls([x["accuracy"] for x in t]),
            "macro_f1": _ms_cls([x["macro_f1"] for x in t]),
            "auc": _ms_cls([x["auc"] for x in t]),
            "sensitivity_standard": _ms_cls([x["standard"]["recall"] for x in t]),
            "specificity_standard": _ms_cls([x["standard"]["specificity"] for x in t]),
            "precision_standard": _ms_cls([x["standard"]["precision"] for x in t]),
            "precision_non_standard": _ms_cls([x["non_standard"]["precision"] for x in t]),
            "auc_ci95_per_seed": [r["test_ci"]["ci95"]["auc"] for r in rs],
            "n_boot": rs[0]["test_ci"]["n_boot"],
            "n_test_videos": rs[0]["test_ci"]["n_clusters"],
            "youden": {
                "thresholds": [r["youden_val"]["threshold"] for r in rs],
                "accuracy": _ms_cls([x["accuracy"] for x in y]),
                "macro_f1": _ms_cls([x["macro_f1"] for x in y]),
                "sensitivity_standard": _ms_cls([x["standard"]["recall"] for x in y]),
                "specificity_standard": _ms_cls([x["standard"]["specificity"] for x in y]),
            },
            "val_macro_f1_best": _ms_cls([s["best_val_macro_f1"] for s in ts]),
            "best_epoch": [s["best_epoch"] for s in ts],
            "epochs_run": [s["epochs_run"] for s in ts],
            "confusion_matrices": [x["confusion_matrix"] for x in t],
        }
    write_json(root / "summary.json", summ, indent=2)
    lines = [
        "# Classification, JNU-IFM test split (Standard = SPHead positive, threshold 0.5)",
        "",
        "Mean +/- SD across seeds (sample SD). AUC CI = video-level cluster bootstrap per seed "
        f"({sorted({s['n_boot'] for s in summ.values()})} resamples, "
        f"{sorted({s['n_test_videos'] for s in summ.values()})} test videos).",
        "",
        "| Model | Init | Params | Seeds | Accuracy (%) | Macro-F1 (%) | AUC | Sensitivity (%) | Specificity (%) "
        "| AUC 95% CI per seed | Best epoch | Epochs run |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summ.values():
        ci = "; ".join(f"{a:.3f}-{b:.3f}" for a, b in s["auc_ci95_per_seed"])
        lines.append(
            f"| {s['display_name']} | {s['init']} | {s['params']:,} | {len(s['seeds'])} | "
            f"{_fmt_cls(s['accuracy'])} | {_fmt_cls(s['macro_f1'])} | {_fmt_cls(s['auc'], False)} | "
            f"{_fmt_cls(s['sensitivity_standard'])} | {_fmt_cls(s['specificity_standard'])} | {ci} | "
            f"{s['best_epoch']} | {s['epochs_run']} |"
        )
    lines += [
        "",
        "## Youden threshold chosen on validation, applied to test",
        "",
        "| Model | Thresholds | Accuracy (%) | Macro-F1 (%) | Sensitivity (%) | Specificity (%) |",
        "|---|---|---|---|---|---|",
    ]
    for s in summ.values():
        yd = s["youden"]
        lines.append(
            f"| {s['display_name']} | {', '.join(f'{t:.3f}' for t in yd['thresholds'])} | "
            f"{_fmt_cls(yd['accuracy'])} | {_fmt_cls(yd['macro_f1'])} | "
            f"{_fmt_cls(yd['sensitivity_standard'])} | {_fmt_cls(yd['specificity_standard'])} |"
        )
    write_text(root / "summary.md", "\n".join(lines) + "\n")
    return summ


# ------------------------------------------------------------------------------------------------ segmentation
def _ms_seg(vals: list[float | None]) -> dict[str, Any]:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": None, "sd": None, "n_runs": 0}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
        "n_runs": int(len(v)),
        "values": [float(x) for x in v],
    }


def _fmt_seg(d: dict[str, Any], scale: float = 1.0, nd: int = 2) -> str:
    if d["mean"] is None:
        return "n/a"
    s = f"{d['mean'] * scale:.{nd}f}"
    return s + (f" ± {d['sd'] * scale:.{nd}f}" if d.get("sd") is not None else "")


def aggregate_seg(root: Path) -> dict[str, Any]:
    """Write ``summary.json`` and ``summary.md`` for a segmentation run root."""
    summ: dict[str, Any] = {"runs": {}, "models": {}}
    ceil = root / "ceiling.json"
    if ceil.exists():
        summ["ceiling_gt_masks"] = read_json(ceil)
    for m, _ in SEG_MODELS:
        runs = sorted(root.glob(f"{m}/seed*/metrics.json"))
        res = [read_json(p) for p in runs]
        summ["runs"][m] = [portable_path(p) for p in runs]
        if not res:
            continue
        out: dict[str, Any] = {
            "params": res[0]["params"],
            "n_seeds": len(res),
            "best_epoch": _ms_seg([r.get("train", {}).get("best_epoch") for r in res]),
            "val_best_mean_fg_dice": _ms_seg([r.get("train", {}).get("best") for r in res]),
        }
        for s, _ in SEG_SETS:
            out[s] = {"seg": {k: _ms_seg([r[s]["seg"][k] for r in res]) for k in SEG_KEYS}}
            for head in ("aop_mask", "aop_head"):
                if head in res[0][s]:
                    out[s][head] = {
                        ref: {k: _ms_seg([r[s][head][ref].get(k) for r in res]) for k in AOP_KEYS}
                        for ref in ("primary", "secondary")
                    }
        summ["models"][m] = out
    write_json(root / "summary.json", summ)
    write_text(root / "summary.md", _seg_markdown(summ))
    return summ


def _seg_markdown(summ: dict[str, Any]) -> str:
    lines = [
        "# Segmentation and AoP results (mean ± SD over seeds)",
        "",
        "Dice/IoU/BF1 in %, per-image mean over test frames (pooled values in separate columns). HD95 in pixels on "
        "the 256x256 grid (native-pixel HD95 in summary.json; no mm spacing is published). BF1 tolerance 2 px. "
        "AoP from predicted masks with the FH-PS-AoP challenge geometry; MAE etc. in degrees; failures = frames "
        "where no ellipse could be fitted (excluded from the error statistics).",
        "",
    ]
    for s, title in SEG_SETS:
        lines += [
            f"## {title}",
            "",
            "### Segmentation",
            "",
            "| Model | Params (M) | Seeds | Dice FH | Dice PS | IoU FH | IoU PS | pooled Dice FH | pooled Dice PS | "
            "HD95 FH | HD95 PS | BF1 FH | BF1 PS | empty PS preds |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m, name in SEG_MODELS:
            if m not in summ["models"]:
                continue
            o = summ["models"][m]
            g = o[s]["seg"]
            lines.append(
                f"| {name} | {o['params'] / 1e6:.3f} | {o['n_seeds']} | {_fmt_seg(g['dice_FH_mean'], 100)} | "
                f"{_fmt_seg(g['dice_PS_mean'], 100)} | {_fmt_seg(g['iou_FH_mean'], 100)} | "
                f"{_fmt_seg(g['iou_PS_mean'], 100)} | "
                f"{_fmt_seg(g['pooled_dice_FH'], 100)} | {_fmt_seg(g['pooled_dice_PS'], 100)} | "
                f"{_fmt_seg(g['hd95_FH_mean'])} | {_fmt_seg(g['hd95_PS_mean'])} | {_fmt_seg(g['bf1_FH_mean'], 100)} | "
                f"{_fmt_seg(g['bf1_PS_mean'], 100)} | {_fmt_seg(g['n_pred_empty_PS'], 1, 1)} |"
            )
        for ref in ("primary", "secondary"):
            refname = None
            lines += [
                "",
                f"### AoP vs {ref} reference",
                "",
                "| Model | AoP source | failures | MAE | RMSE | median AE | bias | LoA low | LoA high | r "
                "| within 5° |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            if "ceiling_gt_masks" in summ:
                c = summ["ceiling_gt_masks"][s][ref]
                refname = c["ref"]
                lines.append(
                    f"| GT masks (ceiling) | mask geometry | {c['n_fail']} | {c['mae']:.2f} | {c['rmse']:.2f} | "
                    f"{c['median_ae']:.2f} | {c['bias']:.2f} | {c['loa_low']:.2f} | {c['loa_high']:.2f} | "
                    f"{c['pearson_r']:.3f} | {100 * c['within5']:.1f} |"
                )
            for m, name in SEG_MODELS:
                if m not in summ["models"]:
                    continue
                for head, lab in (("aop_mask", "predicted mask"), ("aop_head", "direct head")):
                    if head not in summ["models"][m][s]:
                        continue
                    a = summ["models"][m][s][head][ref]
                    lines.append(
                        f"| {name} | {lab} | {_fmt_seg(a['n_fail'], 1, 1)} | {_fmt_seg(a['mae'])} | "
                        f"{_fmt_seg(a['rmse'])} | "
                        f"{_fmt_seg(a['median_ae'])} | {_fmt_seg(a['bias'])} | {_fmt_seg(a['loa_low'])} | "
                        f"{_fmt_seg(a['loa_high'])} | {_fmt_seg(a['pearson_r'], 1, 3)} | "
                        f"{_fmt_seg(a['within5'], 100, 1)} |"
                    )
            if refname:
                lines.append(f"\nReference: {refname}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ cross-validation
def aggregate_cv(cv_root: Path, folds_dir: Path = SPLITS / "cv") -> dict[str, Any]:
    """Write ``summary_cv.json`` and ``summary_cv.md`` for the cross-validation runs."""
    cfg = load_config("cv")
    k_folds, n_boot = cfg["k"], cfg["pooled_comparison"]["n_boot"]
    pat: dict[str, int] = {}
    for k in range(k_folds):
        d = pd.read_csv(folds_dir / f"fold{k}.csv")
        pat.update(dict(zip(d.frame_id, d.patient_id, strict=True)))
    folds: dict[str, list[dict[str, Any]]] = {}
    pooled: dict[str, dict[str, np.ndarray]] = {}
    for m in CV_MODELS:
        per, ys, ps, fids = [], [], [], []
        for k in range(k_folds):
            f = cv_root / f"fold{k}" / m / "seed0" / "preds_test.npz"
            if not f.exists():
                continue
            t = np.load(f)
            per.append({"fold": k, **summary_metrics(point_metrics(t["y"], t["p_standard"]))})
            ys.append(t["y"])
            ps.append(t["p_standard"])
            fids.append(t["frame_id"])
        if per:
            folds[m] = per
        if len(per) == k_folds:
            pooled[m] = {"y": np.concatenate(ys), "p": np.concatenate(ps), "fid": np.concatenate(fids)}

    out: dict[str, Any] = {"k": k_folds, "n_boot": n_boot, "models": {}}
    complete = [m for m in CV_MODELS if m in pooled]
    ranks = {m: {"macro_f1": [], "auc": []} for m in complete}
    for k in range(k_folds):
        for met in ("macro_f1", "auc"):
            vals = {m: next(r[met] for r in folds[m] if r["fold"] == k) for m in complete}
            order = sorted(vals, key=lambda x: -vals[x])
            for m in complete:
                ranks[m][met].append(order.index(m) + 1)
    base = "deepfetalnet" if "deepfetalnet" in pooled else None
    boots: list[np.ndarray] = []
    if base:
        ref = pooled[base]
        women = np.array([pat[f] for f in ref["fid"]])
        uw = np.unique(women)
        idx_of = {w: np.where(women == w)[0] for w in uw}
        for m in complete:
            if not (pooled[m]["fid"] == ref["fid"]).all():
                if set(pooled[m]["fid"]) != set(ref["fid"]):
                    raise ValueError(f"{m}: pooled frames differ from DeepFetalNet's")
                pos = {f: i for i, f in enumerate(pooled[m]["fid"])}
                o = np.array([pos[f] for f in ref["fid"]])
                pooled[m] = {kk: v[o] for kk, v in pooled[m].items()}
        rng = np.random.default_rng(cfg["pooled_comparison"]["seed"])
        boots = [np.concatenate([idx_of[w] for w in rng.choice(uw, len(uw))]) for _ in range(n_boot)]

    def ci(a: list[float]) -> list[float]:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    for m in CV_MODELS:
        if m not in folds:
            continue
        per = folds[m]
        e: dict[str, Any] = {"name": CV_NAMES[m], "n_folds": len(per), "per_fold": per}
        for met in ("accuracy", "macro_f1", "auc", "sensitivity", "specificity"):
            v = np.array([r[met] for r in per])
            e[met] = {
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()),
                "max": float(v.max()),
                "values": v.tolist(),
            }
        if m in pooled:
            pm = pooled[m]
            e["pooled"] = {**summary_metrics(point_metrics(pm["y"], pm["p"])), "n": int(len(pm["y"]))}
            e["rank"] = {
                met: {"values": r, "mean": float(np.mean(r)), "best": int(min(r)), "worst": int(max(r))}
                for met, r in ranks[m].items()
            }
            if base and m != base:
                y, p, q = pm["y"], pm["p"], pooled[base]["p"]
                d_f1 = fast_metrics(y, p)["mf1"] - fast_metrics(y, q)["mf1"]
                d_auc = roc_auc_score(y, p) - roc_auc_score(y, q)
                bf, ba = [], []
                for b in boots:
                    if len(np.unique(y[b])) < 2:
                        continue
                    bf.append(fast_metrics(y[b], p[b])["mf1"] - fast_metrics(y[b], q[b])["mf1"])
                    ba.append(roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], q[b]))
                e["vs_deepfetalnet"] = {
                    "macro_f1": {"diff": float(d_f1), "ci": ci(bf)},
                    "auc": {"diff": float(d_auc), "ci": ci(ba)},
                    "folds_better_macro_f1": int(
                        sum(r["macro_f1"] > s["macro_f1"] for r, s in zip(per, folds[base], strict=True))
                    ),
                }
        out["models"][m] = e
    write_json(cv_root / "summary_cv.json", out)
    write_text(cv_root / "summary_cv.md", _cv_markdown(out))
    return out


def _cv_markdown(out: dict[str, Any]) -> str:
    lines = [
        "# Classification, patient-grouped 5-fold cross-validation (1 seed per fold, threshold 0.5)",
        "",
        "| Model | Folds | Macro-F1 mean ± SD (min–max) | AUC mean ± SD (min–max) | Pooled macro-F1 | Pooled AUC | "
        "Rank macro-F1 (best–worst) | Δ macro-F1 vs DFN [95% CI] | Δ AUC vs DFN [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in out["models"].values():
        a, u = e["macro_f1"], e["auc"]
        po = e.get("pooled", {})
        rk = e.get("rank", {}).get("macro_f1")
        vs = e.get("vs_deepfetalnet")
        if po:
            row = (
                f"| {e['name']} | {e['n_folds']} | {100 * a['mean']:.1f} ± {100 * a['sd']:.1f} "
                f"({100 * a['min']:.1f}–{100 * a['max']:.1f}) | {u['mean']:.3f} ± {u['sd']:.3f} "
                f"({u['min']:.3f}–{u['max']:.3f}) | {100 * po['macro_f1']:.1f} | {po['auc']:.3f} | "
            )
        else:
            row = (
                f"| {e['name']} | {e['n_folds']} | {100 * a['mean']:.1f} ± {100 * a['sd']:.1f} | "
                f"{u['mean']:.3f} ± {u['sd']:.3f} | – | – | "
            )
        row += f"{rk['mean']:.1f} ({rk['best']}–{rk['worst']}) | " if rk else "– | "
        if vs:
            mf, au = vs["macro_f1"], vs["auc"]
            row += (
                f"{100 * mf['diff']:+.1f} [{100 * mf['ci'][0]:+.1f}, {100 * mf['ci'][1]:+.1f}] | "
                f"{au['diff']:+.3f} [{au['ci'][0]:+.3f}, {au['ci'][1]:+.3f}] |"
            )
        else:
            row += "– | – |"
        lines.append(row)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ split sensitivity
def author_split_check(
    root: Path,
    fixed_summary: Path,
    split_file: Path = SPLITS / "video_level_split.csv",
    frames_file: Path = SPLITS / "frames.csv",
) -> dict[str, Any]:
    """Compare models trained on the video-level split with the locked split."""
    split = pd.read_csv(split_file)
    frames = read_frames_csv(frames_file)
    pat = dict(zip(frames.video_id, frames.patient_id, strict=False))
    train_women = set(split[split.split == "train"].video_id.map(pat))
    ours = read_json(fixed_summary)
    res: dict[str, Any] = {}
    rows = []

    def pick(m: dict[str, Any]) -> dict[str, Any]:
        return {**summary_metrics(m), "n": m["n"]}

    def fmt(d: dict[str, Any], m: str) -> str:
        if m == "auc":
            return f"{d[m]['mean']:.3f} ± {d[m]['sd']:.3f}"
        return f"{100 * d[m]['mean']:.1f} ± {100 * d[m]['sd']:.1f}"

    for model, name in SPLIT_CHECK_MODELS.items():
        per: dict[str, list[dict[str, Any]]] = {"all": [], "seen": [], "unseen": [], "val": []}
        for run in sorted((root / model).glob("seed*")):
            if not (run / "preds_test.npz").exists():
                continue
            t, v = np.load(run / "preds_test.npz"), np.load(run / "preds_val.npz")
            seen = np.array([pat[x] in train_women for x in t["video_id"]])
            per["all"].append(pick(point_metrics(t["y"], t["p_standard"])))
            per["seen"].append(pick(point_metrics(t["y"][seen], t["p_standard"][seen])))
            per["unseen"].append(pick(point_metrics(t["y"][~seen], t["p_standard"][~seen])))
            per["val"].append(pick(point_metrics(v["y"], v["p_standard"])))
        agg: dict[str, Any] = {
            k: {
                m: {
                    "mean": float(np.mean([r[m] for r in lst])),
                    "sd": float(np.std([r[m] for r in lst], ddof=1)) if len(lst) > 1 else 0.0,
                    "n": lst[0]["n"] if lst else 0,
                }
                for m in ("accuracy", "macro_f1", "auc", "sensitivity", "specificity")
            }
            for k, lst in per.items()
            if lst
        }
        agg["seeds"] = len(per["all"])
        o = ours[model]
        agg["patient_split_test"] = {
            m: {"mean": o[m]["mean"], "sd": o[m].get("sd", 0.0), "n": o["n_test"]}
            for m in ("accuracy", "macro_f1", "auc")
        }
        res[model] = agg
        for lab, key in (
            ("Video-level split, validation", "val"),
            ("Video-level split, test (all 622)", "all"),
            ("Video-level split, test, women also in train", "seen"),
            ("Video-level split, test, women not in train", "unseen"),
            ("Patient-disjoint split, test (630)", "patient_split_test"),
        ):
            d = agg.get(key)
            if d:
                rows.append(
                    f"| {name} | {lab} | {d['accuracy'].get('n', '')} | {fmt(d, 'accuracy')} | "
                    f"{fmt(d, 'macro_f1')} | {fmt(d, 'auc')} |"
                )
    write_json(root / "author_split_check.json", res)
    md = ["| Model | Evaluation | Frames | Accuracy (%) | Macro-F1 (%) | AUC |", "|---|---|---|---|---|---|", *rows]
    write_text(root / "author_split_check.md", "\n".join(md) + "\n")
    return res


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    for kind in ("cls", "seg"):
        p = sub.add_parser(kind)
        p.add_argument("root", type=Path, nargs="?", default=None, help=f"default: $FIN_RUNS/{kind}")
    p = sub.add_parser("cv")
    p.add_argument("--cv-root", type=Path, default=None, help="default: $FIN_RUNS/cls_cv")
    p.add_argument("--folds", type=Path, default=SPLITS / "cv")
    p = sub.add_parser("author-split")
    p.add_argument("--root", type=Path, default=None, help="default: $FIN_RUNS/cls_authorsplit")
    p.add_argument("--fixed-summary", type=Path, default=None, help="default: $FIN_RUNS/cls/summary.json")
    a = ap.parse_args(argv)
    setup_logging()
    if a.kind == "cls":
        aggregate_cls(a.root or runs_dir() / "cls")
    elif a.kind == "seg":
        aggregate_seg(a.root or runs_dir() / "seg")
    elif a.kind == "cv":
        aggregate_cv(a.cv_root or runs_dir() / "cls_cv", a.folds)
    else:
        author_split_check(
            a.root or runs_dir() / "cls_authorsplit", a.fixed_summary or runs_dir() / "cls" / "summary.json"
        )
    log.info("aggregated %s", a.kind)


if __name__ == "__main__":
    main()
