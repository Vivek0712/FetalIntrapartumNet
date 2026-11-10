"""Compute every paper value from the saved run outputs into a ``Store`` (keys documented in docs/RESULTS.md).

Conventions (stated in the paper)
    classification: Standard (SPHead) positive, threshold 0.5, test split; mean +/- sample SD over seeds; 95% CI =
        percentile cluster bootstrap over test videos of the seed-averaged metric (same video draw for every seed)
    segmentation: per-image mean over test frames (primary), then mean +/- SD over seeds
    AoP: predicted mask -> challenge geometry; error statistics over successful frames; failures counted
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fin.config import SPLITS
from fin.data.labels import video_of
from fin.evaluation.cls_metrics import fast_metrics
from fin.evaluation.e2e import end_to_end
from fin.evaluation.seg_metrics import aop_stats, per_image, pixel_accuracy, summarise_seg
from fin.io import read_frames_csv, read_json, read_json_if_exists
from fin.reporting.fmt import Store, ci_str, f, fin, fint, stats
from fin.reporting.names import CLS, CLS_NAME, EFF_ID, SEG, SEG_NAME, SETS
from fin.runs import seed_dirs

WARNINGS: list[str] = []


def warn(msg: str) -> None:
    """Record a warning (printed by ``fill_results``)."""
    WARNINGS.append(msg)


# ------------------------------------------------------------------------------------------------ data counts
def fill_data(v: Store, frames_csv: Path, iugc_csv: Path) -> pd.DataFrame | None:
    """Dataset and split counts, reference-AoP statistics and the IUGC 2024 summary."""
    if not frames_csv.exists():
        warn(f"frames.csv not found: {frames_csv}")
        return None
    df = read_frames_csv(frames_csv)
    cats = {3: "none", 4: "onlysp", 5: "onlyhead", 6: "sphead"}

    def block(prefix: str, d: pd.DataFrame) -> None:
        v.put(f"{prefix}.frames", fint(len(d)))
        v.put(f"{prefix}.videos", fint(d.video_id.nunique()))
        v.put(f"{prefix}.women", fint(d.patient_id.nunique()))
        for c, nm in cats.items():
            v.put(f"{prefix}.{nm}", fint((d.category == c).sum()))
        v.put(f"{prefix}.standard", fint((d.binary_label == 1).sum()))
        v.put(f"{prefix}.nonstandard", fint((d.binary_label == 0).sum()))
        v.put(f"{prefix}.standard.pct", f(100 * (d.binary_label == 1).mean(), 1))
        sp = d[d.category == 6]
        v.put(f"{prefix}.seg.frames", fint(len(sp)))
        v.put(f"{prefix}.seg.videos", fint(sp.video_id.nunique()))
        v.put(f"{prefix}.seg.women", fint(sp.patient_id.nunique()))
        a = sp.aop_ref_challenge_code.dropna()
        v.put(f"{prefix}.aop.frames", fint(len(a)))
        v.put(f"{prefix}.aop.mean", f(a.mean(), 2))
        v.put(f"{prefix}.aop.sd", f(a.std(ddof=1), 2))
        v.put(f"{prefix}.aop.min", f(a.min(), 1))
        v.put(f"{prefix}.aop.max", f(a.max(), 1))

    block("data.jnu", df)
    for s in ("train", "val", "test"):
        block(f"data.split.{s}", df[df.split == s])
    sp = df[df.category == 6]
    d = (sp.aop_ref - sp.aop_ref_challenge_code).abs().dropna()
    v.put("data.jnu.refdiff.mae", f(d.mean(), 2))
    v.put("data.jnu.refdiff.p95", f(d.quantile(0.95), 2))
    v.put("data.jnu.refdiff.max", f(d.max(), 2))
    v.put(
        "data.jnu.mismatch",
        fint(((df.category == 3) & (df.has_ps == 1)).sum() + ((df.category == 4) & (df.has_fh == 1)).sum()),
    )
    v.put("data.jnu.width", fint(df.width.iloc[0]))
    v.put("data.jnu.height", fint(df.height.iloc[0]))
    if iugc_csv.exists():
        ig = pd.read_csv(iugc_csv)
        v.put("data.iugc.frames", fint(len(ig)))
        v.put("data.iugc.aop.mean", f(ig.aop_official.mean(), 2))
        v.put("data.iugc.aop.sd", f(ig.aop_official.std(ddof=1), 2))
        dd = (ig.aop_official - ig.aop_mask_challenge_code).abs()
        v.put("data.iugc.refdiff.mae", f(dd.mean(), 2))
        v.put("data.iugc.refdiff.p95", f(dd.quantile(0.95), 2))
    else:
        warn(f"IUGC aop.csv not found: {iugc_csv}")
    return df


# ------------------------------------------------------------------------------------------------ classification
CLS_METRICS = [
    ("acc", "accuracy", 2, 100),
    ("prec", "precision", 2, 100),
    ("sens", "sensitivity", 2, 100),
    ("spec", "specificity", 2, 100),
    ("f1", "F1", 3, 1),
    ("mf1", "macro-F1", 3, 1),
    ("auc", "AUC", 3, 1),
]


def seed_avg_bootstrap(
    ys: list[np.ndarray], ps: list[np.ndarray], groups: np.ndarray, n_boot: int, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Percentile CI of the seed-averaged metric, resampling test videos (same draw for every seed)."""
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = [np.flatnonzero(g == u) for u in ug]
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {k: [] for k in ("acc", "prec", "sens", "spec", "f1", "mf1", "auc")}
    for _ in range(n_boot):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(ug), len(ug))])
        per = [fast_metrics(y[ii], p[ii]) for y, p in zip(ys, ps, strict=True)]
        for k in acc:
            acc[k].append(np.nanmean([m[k] for m in per]) if any(np.isfinite(m[k]) for m in per) else np.nan)
    out = {}
    for k, vals in acc.items():
        a = np.asarray(vals, float)
        a = a[np.isfinite(a)]
        out[k] = (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) if len(a) else (np.nan, np.nan)
    return out


def _gate_values(v: Store, m: str, ys: list[np.ndarray], ps: list[np.ndarray], cat: np.ndarray) -> None:
    fwd = [p >= 0.5 for p in ps]
    v.ms(f"cls.{m}.gate.forwarded", [x.sum() for x in fwd], 1)
    v.ms(
        f"cls.{m}.gate.purity",
        [np.mean(y[x] == 1) if x.any() else np.nan for y, x in zip(ys, fwd, strict=True)],
        2,
        100,
    )
    v.ms(f"cls.{m}.gate.leak", [np.sum(x & (y == 0)) for y, x in zip(ys, fwd, strict=True)], 1)
    v.ms(f"cls.{m}.gate.missed", [np.sum(~x & (y == 1)) for y, x in zip(ys, fwd, strict=True)], 1)
    for c, nm in ((3, "none"), (4, "onlysp"), (5, "onlyhead")):
        v.ms(f"cls.{m}.gate.leak.{nm}", [np.sum(x & (cat == c)) for x in fwd], 1)
        v.ms(f"cls.{m}.gate.leakrate.{nm}", [np.mean(x[cat == c]) if (cat == c).any() else np.nan for x in fwd], 2, 100)


def fill_cls(v: Store, root: Path, n_boot: int) -> dict[str, dict[str, Any]]:
    """Classification test metrics, CIs, gate analysis, paired differences to DeepFetalNet and rankings."""
    res: dict[str, dict[str, Any]] = {}
    for m in CLS:
        runs = seed_dirs(root, m, "metrics_test.json")
        if not runs:
            continue
        js = [read_json(d / "metrics_test.json") for _, d in runs]
        per = [
            {
                "acc": j["test"]["accuracy"],
                "prec": j["test"]["standard"]["precision"],
                "sens": j["test"]["standard"]["recall"],
                "spec": j["test"]["standard"]["specificity"],
                "f1": j["test"]["standard"]["f1"],
                "mf1": j["test"]["macro_f1"],
                "auc": j["test"]["auc"],
            }
            for j in js
        ]
        r: dict[str, Any] = {
            "seeds": [s for s, _ in runs],
            "per": per,
            "val_auc": stats([j["val"]["auc"] for j in js])[0],
        }
        v.put(f"cls.{m}.name", CLS_NAME[m])
        v.put(f"cls.{m}.init", js[0]["init"])
        v.put(f"cls.{m}.params", fint(js[0]["params"]))
        v.put(f"cls.{m}.nseeds", fint(len(js)))
        v.put(f"cls.{m}.seeds", ", ".join(str(s) for s in r["seeds"]))
        for k, _, nd, sc in CLS_METRICS:
            r[k] = v.ms(f"cls.{m}.{k}", [p[k] for p in per], nd, sc)
        v.ms(f"cls.{m}.val.auc", [j["val"]["auc"] for j in js], 3)
        v.ms(f"cls.{m}.val.mf1", [j["val"]["macro_f1"] for j in js], 3)
        ts = [j.get("train_summary") or {} for j in js]
        v.put(f"cls.{m}.bestepoch", ", ".join(str(t.get("best_epoch")) for t in ts if t))
        v.put(f"cls.{m}.epochsrun", ", ".join(str(t.get("epochs_run")) for t in ts if t))
        yv = [j["youden_val"] for j in js]
        v.ms(f"cls.{m}.youden.thr", [y["threshold"] for y in yv], 3)
        v.ms(f"cls.{m}.youden.acc", [y["test"]["accuracy"] for y in yv], 2, 100)
        v.ms(f"cls.{m}.youden.sens", [y["test"]["standard"]["recall"] for y in yv], 2, 100)
        v.ms(f"cls.{m}.youden.spec", [y["test"]["standard"]["specificity"] for y in yv], 2, 100)
        v.ms(f"cls.{m}.youden.mf1", [y["test"]["macro_f1"] for y in yv], 3)
        v.put(f"cls.{m}.auc.ci.seed0", ci_str(*js[0]["test_ci"]["ci95"]["auc"], 3))
        preds = [np.load(d / "preds_test.npz", allow_pickle=False) for _, d in runs if (d / "preds_test.npz").exists()]
        if len(preds) == len(runs):
            fid0 = preds[0]["frame_id"]
            if all(np.array_equal(fid0, z["frame_id"]) for z in preds):
                ys, ps = [z["y"].astype(int) for z in preds], [z["p_standard"].astype(float) for z in preds]
                if n_boot > 0:
                    ci = seed_avg_bootstrap(ys, ps, preds[0]["video_id"], n_boot)
                    for k, _, nd, sc in CLS_METRICS:
                        v.put(f"cls.{m}.{k}.ci", ci_str(*ci[k], nd, sc))
                    r["ci"] = ci
                _gate_values(v, m, ys, ps, preds[0]["category"].astype(int))
                r["roc"] = (ys, ps)
                r["groups"], r["fids"] = preds[0]["video_id"], fid0
            else:
                warn(f"cls {m}: preds_test.npz frame order differs across seeds; no seed-averaged CI")
        else:
            warn(f"cls {m}: preds_test.npz missing for some seeds")
        res[m] = r
        if m == CLS[0] or "cls.ntest" not in v:
            t = js[0]["test"]
            v.put("cls.ntest", fint(t["n"]))
            v.put("cls.ntest.standard", fint(t["standard"]["support"]))
            v.put("cls.ntest.nonstandard", fint(t["non_standard"]["support"]))
            v.put("cls.nboot", fint(n_boot))
            v.put("cls.nboot.seed", fint(js[0]["test_ci"]["n_boot"]))
            v.put("cls.ntestvideos", fint(js[0]["test_ci"]["n_clusters"]))
    _paired_vs_deepfetalnet(v, res, n_boot)
    if res:
        _rankings(v, res)
    return res


def _paired_vs_deepfetalnet(v: Store, res: dict[str, dict[str, Any]], n_boot: int) -> None:
    """Paired cluster bootstrap of the difference (model - DeepFetalNet) in seed-averaged metrics."""
    ref = res.get("deepfetalnet", {}).get("roc")
    if ref is None or n_boot <= 0:
        return
    g = np.asarray(res["deepfetalnet"]["groups"])
    ug = np.unique(g)
    idx = [np.flatnonzero(g == u) for u in ug]
    for m, r in res.items():
        if m == "deepfetalnet" or r.get("roc") is None or not np.array_equal(r["fids"], res["deepfetalnet"]["fids"]):
            continue
        rng = np.random.default_rng(1)
        diffs: dict[str, list[float]] = {k: [] for k in ("acc", "mf1", "auc")}
        for _ in range(n_boot):
            ii = np.concatenate([idx[j] for j in rng.integers(0, len(ug), len(ug))])
            a = [fast_metrics(y[ii], p[ii]) for y, p in zip(*r["roc"], strict=True)]
            b = [fast_metrics(y[ii], p[ii]) for y, p in zip(*ref, strict=True)]
            for k in diffs:
                diffs[k].append(np.nanmean([x[k] for x in a]) - np.nanmean([x[k] for x in b]))
        for k, _, nd, sc in [c for c in CLS_METRICS if c[0] in diffs]:
            d0 = stats([q[k] for q in r["per"]])[0] - stats([q[k] for q in res["deepfetalnet"]["per"]])[0]
            a = np.asarray(diffs[k], float)
            a = a[np.isfinite(a)]
            v.put(f"cls.{m}.vsdfn.{k}", f(d0, nd, sc, sign=True))
            if len(a):
                lo, hi = np.quantile(a, [0.025, 0.975])
                v.put(f"cls.{m}.vsdfn.{k}.ci", ci_str(lo, hi, nd, sc))
                v.put(f"cls.{m}.vsdfn.{k}.excl0", "yes" if lo > 0 or hi < 0 else "no")


def _rankings(v: Store, res: dict[str, dict[str, Any]]) -> None:
    v.put("cls.nmodels", fint(len(res)))
    v.put("cls.seeds", ", ".join(str(s) for s in sorted({s for r in res.values() for s in r["seeds"]})))
    for k, _, _, _ in CLS_METRICS:
        cand = [(res[m][k], m) for m in res if fin(res[m].get(k))]
        if cand:
            hi = max(cand)[1]
            v.put(f"cls.rank.{k}.first", CLS_NAME[hi])
            v.put(f"cls.rank.{k}.first.id", hi)
            order = sorted(cand, reverse=True)
            v.put(
                f"cls.rank.{k}.deepfetalnet",
                fint(1 + [mm for _, mm in order].index("deepfetalnet"))
                if "deepfetalnet" in res and fin(res["deepfetalnet"].get(k))
                else None,
            )
    base = [(res[m]["val_auc"], m) for m in res if m != "deepfetalnet" and fin(res[m]["val_auc"])]
    if base:
        b = max(base)[1]
        v.put("cls.baseline.valbest", CLS_NAME[b])
        v.put("cls.baseline.valbest.id", b)


# ------------------------------------------------------------------------------------------------ efficiency
# Devices of the reported benchmark (Table 2). A run on another GPU backend stores its timings under that
# backend's name (e.g. ``eff.<model>.lat.cuda``) and leaves the MPS keys empty.
PAPER_DEVICES = ("mps", "cpu")


def fill_eff(v: Store, path: Path) -> dict[str, Any]:
    """Parameters, GMACs, latency, throughput and weight memory of every model; benchmark machine."""
    out: dict[str, Any] = {}
    d = read_json_if_exists(path)
    if d is None:
        warn(f"efficiency results not found: {path}")
        return out
    hw = d.get("hardware", {})
    v.put("eff.hw.cpu", hw.get("cpu"))
    v.put("eff.hw.torch", hw.get("torch"))
    v.put("eff.hw.macos", hw.get("macos"))
    v.put("eff.hw.ncpu", hw.get("ncpu"))
    if hw.get("mem_bytes"):
        v.put("eff.hw.memgb", fint(int(hw["mem_bytes"]) / 2**30))
    for e in d["models"]:
        m = EFF_ID.get(e["name"])
        if m is None:
            warn(f"efficiency: unknown model name {e['name']}")
            continue
        out[m] = e
        nd = 3 if e["task"] == "classification" else 2
        v.put(f"eff.{m}.params", fint(e["params"]))
        v.put(f"eff.{m}.params.m", f(e["params"] / 1e6, 2))
        v.put(f"eff.{m}.gmacs", f(e.get("percall_GMACs"), nd))
        v.put(f"eff.{m}.gmacs.thop", f(e.get("thop_GMACs"), nd))
        v.put(f"eff.{m}.gmacs.fvcore", f(e.get("fvcore_GMACs"), nd))
        latency, throughput = e.get("latency") or {}, e.get("throughput_b16") or {}
        missing = [d for d in PAPER_DEVICES if d not in latency]
        if missing:
            found = ", ".join(latency) or "none"
            warn(
                f"efficiency: {e['name']} has no {'/'.join(missing)} timings (devices: {found}); "
                f"Table 2 reports {'/'.join(PAPER_DEVICES)}, so those cells stay empty"
            )
        for dev, lat in latency.items():
            v.put(f"eff.{m}.lat.{dev}", f(lat.get("median_ms"), 2))
            v.put(f"eff.{m}.lat.{dev}.iqr", ci_str(lat.get("q1_ms"), lat.get("q3_ms"), 2))
            v.put(f"eff.{m}.lat.{dev}.mean", f(lat.get("mean_ms"), 2))
            v.put(f"eff.{m}.lat.{dev}.sd", f(lat.get("sd_ms"), 2))
            v.put(f"eff.{m}.thr.{dev}", f((throughput.get(dev) or {}).get("images_per_s"), 1))
        v.put(f"eff.{m}.mem.weights", f((e.get("mps_memory_b1") or {}).get("weights_MB"), 1))
        v.put(f"eff.{m}.input", "$\\times$".join(str(x) for x in e["input_shape"][1:]))
    return out


# ------------------------------------------------------------------------------------------------ segmentation
SEG_FIELDS = [
    ("dice.fh", "dice_FH_mean", 100),
    ("dice.ps", "dice_PS_mean", 100),
    ("meandice", "mean_fg_dice", 100),
    ("iou.fh", "iou_FH_mean", 100),
    ("iou.ps", "iou_PS_mean", 100),
    ("meaniou", "mean_fg_iou", 100),
    ("hd95.fh", "hd95_FH_mean", 1),
    ("hd95.ps", "hd95_PS_mean", 1),
    ("hd95n.fh", "hd95native_FH_mean", 1),
    ("hd95n.ps", "hd95native_PS_mean", 1),
    ("bf1.fh", "bf1_FH_mean", 100),
    ("bf1.ps", "bf1_PS_mean", 100),
    ("pdice.fh", "pooled_dice_FH", 100),
    ("pdice.ps", "pooled_dice_PS", 100),
    ("piou.fh", "pooled_iou_FH", 100),
    ("piou.ps", "pooled_iou_PS", 100),
]
COUNT_FIELDS = [
    ("empty.fh", "n_pred_empty_FH"),
    ("empty.ps", "n_pred_empty_PS"),
    ("hdundef.fh", "n_hd_undefined_FH"),
    ("hdundef.ps", "n_hd_undefined_PS"),
]


def load_cache_set(cache: Path, s: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Frame table, cache rows and (memory-mapped) GT masks of one evaluation set, in evaluation order."""
    if s == "jnu_test":
        idx = read_frames_csv(cache / "jnu_index.csv")
        sel = np.where((idx.split == "test") & (idx.category == 6))[0]
        return idx.iloc[sel].reset_index(drop=True), sel, np.load(cache / "jnu_mask.npy", mmap_mode="r")
    d = pd.read_csv(cache / "iugc_index.csv")
    return d, np.arange(len(d)), np.load(cache / "iugc_mask.npy", mmap_mode="r")


def mean_shape_prior(cache: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Area-matched mean-shape prior from the training SPHead masks (image-blind reference).

    For each class the per-pixel frequency map is thresholded at the level whose area equals the mean training area
    of that class (PS wins overlaps); the same mask is predicted for every test frame.
    """
    idx = read_frames_csv(cache / "jnu_index.csv")
    sel = np.where((idx.split == "train") & (idx.category == 6))[0]
    msk = np.load(cache / "jnu_mask.npy", mmap_mode="r")
    h, w = msk.shape[1:]
    freq = np.zeros((3, h, w), np.float64)
    area = np.zeros(3)
    for i in range(0, len(sel), 256):
        b = np.asarray(msk[sel[i : i + 256]])
        for c in (1, 2):
            freq[c] += (b == c).sum(0)
            area[c] += (b == c).sum()
    freq /= len(sel)
    area /= len(sel)
    prior = np.zeros((h, w), np.uint8)
    for c in (1, 2):
        order = np.sort(freq[c].ravel())[::-1]
        thr = order[int(round(area[c])) - 1] if area[c] >= 1 else 1.1
        prior[freq[c] >= thr] = c
    return prior, {"n_train": int(len(sel)), "area_fh": float(area[1]), "area_ps": float(area[2])}


def fill_seg(v: Store, root: Path, cache: Path) -> dict[Any, Any]:
    """Segmentation metrics of every model and set, pixel accuracy and the two image-blind reference rows."""
    out: dict[Any, Any] = {}
    caches = {}
    for s in SETS.values():
        if (cache / "jnu_index.csv").exists():
            try:
                caches[s] = load_cache_set(cache, s)
            except FileNotFoundError:
                continue
    for m in SEG:
        runs = seed_dirs(root, m, "metrics.json")
        if not runs:
            continue
        js = [read_json(d / "metrics.json") for _, d in runs]
        out[m] = {"seeds": [s for s, _ in runs], "dirs": [d for _, d in runs], "J": js}
        v.put(f"seg.{m}.name", SEG_NAME[m])
        v.put(f"seg.{m}.nseeds", fint(len(js)))
        v.put(f"seg.{m}.seeds", ", ".join(str(s) for s, _ in runs))
        v.put(f"seg.{m}.params.m", f(js[0]["params"] / 1e6, 2))
        v.put(f"seg.{m}.params", fint(js[0]["params"]))
        v.put(f"seg.{m}.bestepoch", ", ".join(str((j.get("train") or {}).get("best_epoch")) for j in js))
        for k, s in SETS.items():
            if not all(s in j for j in js):
                continue
            v.put(f"seg.nframes.{k}", fint(js[0][s]["n_frames"]))
            for key, fld, sc in SEG_FIELDS:
                out[m][(k, key)] = v.ms(f"seg.{m}.{k}.{key}", [j[s]["seg"].get(fld) for j in js], 2, sc)
            for key, fld in COUNT_FIELDS:
                v.ms(f"seg.{m}.{k}.{key}", [j[s]["seg"].get(fld) for j in js], 1)
            if s in caches:
                d, sel, msk = caches[s]
                gt = np.asarray(msk[sel])
                pa = []
                for _, rd in runs:
                    z = rd / f"preds_{s}.npz"
                    if not z.exists():
                        continue
                    zz = np.load(z)
                    if not np.array_equal(zz["frame_id"].astype(str), d.frame_id.astype(str).to_numpy()):
                        warn(f"seg {m} {rd}: preds_{s}.npz frame order differs from cache; pixel accuracy skipped")
                        continue
                    pa.append(pixel_accuracy(zz["labels"], gt).mean())
                out[m][(k, "pixacc")] = v.ms(f"seg.{m}.{k}.pixacc", pa, 2, 100)
    if (cache / "jnu_mask.npy").exists():
        prior, info = mean_shape_prior(cache)
        v.put("seg.prior.ntrain", fint(info["n_train"]))
        for k, s in SETS.items():
            if s not in caches:
                continue
            d, sel, msk = caches[s]
            gt = np.asarray(msk[sel])
            for ref, pred in (("prior", np.broadcast_to(prior, gt.shape)), ("bg", np.zeros_like(gt))):
                pred = np.ascontiguousarray(pred)
                sm = summarise_seg(per_image(pred, gt))
                for key, fld, sc in SEG_FIELDS:
                    v.put(f"seg.{ref}.{k}.{key}", f(sm.get(fld), 2, sc))
                v.put(f"seg.{ref}.{k}.pixacc", f(pixel_accuracy(pred, gt).mean(), 2, 100))
                out[(ref, k)] = sm
                if ref == "prior":
                    out[("prior_mask",)] = prior
    else:
        warn(f"segmentation cache not found: {cache}")
    return out


# ------------------------------------------------------------------------------------------------ AoP
AOP_FIELDS = [
    ("mae", 2),
    ("rmse", 2),
    ("median_ae", 2),
    ("bias", 2),
    ("sd_diff", 2),
    ("loa_low", 2),
    ("loa_high", 2),
    ("pearson_r", 3),
    ("within5", 1),
    ("within10", 1),
    ("max_ae", 2),
]
AOP_KEY = {
    "median_ae": "medae",
    "sd_diff": "sd",
    "loa_low": "loalow",
    "loa_high": "loahigh",
    "pearson_r": "r",
    "within5": "w5",
    "within10": "w10",
    "max_ae": "maxae",
}


def aop_seed_boot(errs: list[np.ndarray], groups: list[str], n_boot: int, seed: int = 0) -> tuple[float, float]:
    """CI of the seed-averaged MAE; ``errs`` = per-seed |error| arrays (NaN = failure) over the same frames."""
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = [np.flatnonzero(g == u) for u in ug]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(ug), len(ug))])
        vals.append(np.nanmean([np.nanmean(e[ii]) for e in errs]))
    a = np.asarray(vals)
    a = a[np.isfinite(a)]
    return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) if len(a) else (np.nan, np.nan)


def put_aop(v: Store, base: str, runs_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Store the AoP statistics of one source and set (list over seeds of ``aop_stats`` results)."""
    n = [r.get("n") for r in runs_stats]
    nf = [r.get("n_fail") for r in runs_stats]
    nu = [r.get("n_used") for r in runs_stats]
    v.ms(f"{base}.n", n, 0)
    v.put(f"{base}.n", fint(stats(n)[0]))
    v.ms(f"{base}.nfail", nf, 1 if len(runs_stats) > 1 else 0)
    v.ms(f"{base}.succ", [u / t if t else np.nan for u, t in zip(nu, n, strict=True)], 1, 100)
    out: dict[str, Any] = {}
    for fld, nd in AOP_FIELDS:
        key = AOP_KEY.get(fld, fld)
        sc = 100 if fld.startswith("within") else 1
        out[key] = v.ms(f"{base}.{key}", [r.get(fld) for r in runs_stats], nd, sc, sign=(fld == "bias"))
    lo, hi = out.get("loalow"), out.get("loahigh")
    v.put(f"{base}.loa", ci_str(lo, hi, 2) if fin(lo) and fin(hi) else None)
    for w in ("within5", "within10"):
        v.ms(
            f"{base}.{AOP_KEY[w]}fi",
            [r.get(w) * r["n_used"] / r["n"] if fin(r.get(w)) and r.get("n") else np.nan for r in runs_stats],
            1,
            100,
        )
    out["nfail"] = stats(nf)[0]
    return out


def _aop_mae_ci(v: Store, src: str, k: str, s: str, dirs: list[Path], col: str, n_boot: int) -> None:
    """Seed-averaged cluster-bootstrap CI of the MAE from the per-frame CSVs."""
    errs, fids = [], None
    for d in dirs:
        p = d / f"per_image_{s}.csv"
        if not p.exists():
            return
        t = pd.read_csv(p)
        if col not in t:
            return
        if fids is None:
            fids = t.frame_id.astype(str).to_numpy()
        elif not np.array_equal(fids, t.frame_id.astype(str).to_numpy()):
            return
        errs.append(np.abs(t[col].to_numpy(float) - t.ref_primary.to_numpy(float)))
    if errs and n_boot > 0:
        lo, hi = aop_seed_boot(errs, [video_of(x, k) for x in fids], n_boot)
        v.put(f"aop.{src}.{k}.maeci", ci_str(lo, hi, 2))


def fill_aop(
    v: Store, seg_root: Path, seg: dict[Any, Any], frames_df: pd.DataFrame | None, iugc_csv: Path, n_boot: int
) -> dict[Any, Any]:
    """AoP agreement of the ground-truth masks, every segmenter (geometry and head) and the constant baseline."""
    out: dict[Any, Any] = {}
    ceil = read_json_if_exists(seg_root / "ceiling.json")
    if ceil is None:
        warn(f"ceiling.json not found in {seg_root}")
    for k, s in SETS.items():
        if ceil and s in ceil:
            out[("gt", k)] = put_aop(v, f"aop.gt.{k}", [ceil[s]["primary"]])
            v.put(f"aop.gt.{k}.sec.mae", f(ceil[s]["secondary"].get("mae"), 2))
            v.put(f"aop.ref.{k}", ceil[s]["primary"]["ref"].replace("_", r"\_"))
        for m in SEG:
            if m not in seg:
                continue
            js = seg[m]["J"]
            if not all(s in j for j in js):
                continue
            for src, head in ((m, "aop_mask"), (f"{m}.head", "aop_head")):
                if head not in js[0][s]:
                    continue
                out[(src, k)] = put_aop(v, f"aop.{src}.{k}", [j[s][head]["primary"] for j in js])
                v.ms(f"aop.{src}.{k}.sec.mae", [j[s][head]["secondary"].get("mae") for j in js], 2)
                _aop_mae_ci(v, src, k, s, seg[m]["dirs"], head, n_boot)
    if frames_df is not None:
        sp = frames_df[frames_df.category == 6]
        c = float(sp[sp.split == "train"].aop_ref_challenge_code.mean())
        v.put("aop.const.value", f(c, 2))
        te = sp[sp.split == "test"].sort_values("frame_id")
        refs = {"jnu": (te.aop_ref_challenge_code.to_numpy(float), te.video_id.astype(str).to_numpy())}
        if iugc_csv.exists():
            ig = pd.read_csv(iugc_csv)
            refs["iugc"] = (ig.aop_official.to_numpy(float), ig.iloc[:, 0].astype(str).to_numpy())
        for k, (ref, grp) in refs.items():
            st = aop_stats(np.full(len(ref), c), ref, grp, n_boot=max(n_boot, 1))
            out[("const", k)] = put_aop(v, f"aop.const.{k}", [st])
            v.put(f"aop.const.{k}.maeci", ci_str(*st["mae_ci95"], 2))
    for k in SETS:
        if ("gt", k) in out and ceil:
            v.put(f"aop.gt.{k}.maeci", ci_str(*ceil[SETS[k]]["primary"].get("mae_ci95", [None, None]), 2))
    return out


def fill_e2e(
    v: Store, cls_root: Path, seg_root: Path, gate: str = "deepfetalnet", segmenter: str = "aopgaunet"
) -> None:
    """End-to-end AoP on the JNU-IFM test split (``fin.evaluation.e2e``)."""
    seeds, st = end_to_end(cls_root, seg_root, gate, segmenter)
    if not seeds:
        return
    put_aop(v, "aop.e2e.jnu", st)
    v.ms("aop.e2e.jnu.nleak", [r["n_leak"] for r in st], 1)
    v.ms("aop.e2e.jnu.nmissed", [r["n_missed"] for r in st], 1)
    v.put("aop.e2e.seeds", ", ".join(str(k) for k in seeds))
    v.put("aop.e2e.gate", CLS_NAME[gate])
    v.put("aop.e2e.segmenter", SEG_NAME[segmenter])


# ------------------------------------------------------------------------------------------------ second external set
EXT469_SUBSETS = {"all": "all469", "labelled": "labelled300", "unseen": "unseen_by_ours"}


def fill_ext469(v: Store, path: Path) -> dict[str, Any] | None:
    """IUGC 2025 release (469 frames), landmark AoP reference: counts, constant, GT floor and every model."""
    if not path.exists():
        warn("ext469 summary.json not found (run fin ext469)")
        return None
    d = read_json(path)
    ss = d["subset_sizes"]
    v.put("ext469.n.all", fint(ss["all469"]))
    v.put("ext469.n.labelled", fint(ss["labelled300"]))
    v.put("ext469.n.landmarkonly", fint(ss["unlabelled169"]))
    v.put("ext469.n.unseen", fint(ss["unseen_by_ours"]))
    v.put("ext469.n.iugctest.labelled", fint(ss["source:iugc24_test_labelled"]))
    v.put("ext469.n.iugctrain.labelled", fint(ss["source:iugc24_train_labelled_jnu_recording"]))
    v.put("ext469.n.iugcval.labelled", fint(ss["source:iugc24_val_labelled"]))
    pi = path.parent / "per_image_all.csv"
    if pi.exists():
        t = read_frames_csv(pi)
        j = t[t.jnu_overlap.astype(str) == "True"]
        v.put("ext469.n.jnu", fint(len(j)))
        v.put("ext469.n.jnu.women", fint(j.jnu_patient.nunique()))
        v.put("ext469.n.jnu.videos", fint(j.jnu_video.nunique()))
        for s in ("train", "val", "test"):
            v.put(f"ext469.n.jnu.{s}", fint((j.jnu_split == s).sum()))
        seen = (t.jnu_overlap.astype(str) == "True") & t.jnu_split.isin(["train", "val"])
        u = t[~seen]
        if len(u) != ss["unseen_by_ours"]:
            warn(f"ext469: unseen subset from per_image_all.csv ({len(u)}) != summary.json ({ss['unseen_by_ours']})")
        v.put("ext469.n.unseen.iugctest", fint(u.source.astype(str).str.startswith("iugc24_test").sum()))
        v.put("ext469.n.unseen.iugcval", fint(u.source.astype(str).str.startswith("iugc24_val").sum()))
        v.put("ext469.n.unseen.jnutest", fint(((u.jnu_overlap.astype(str) == "True") & (u.jnu_split == "test")).sum()))
    x = d.get("landmark_vs_AOP_column_max_absdiff")
    if fin(x):
        e = int(np.ceil(np.log10(x))) if x > 0 else -16
        v.put("ext469.refdiff.max", f"10^{{{e}}}")
    v.put("ext469.const.value", f(d["constant"]["value"], 2))
    nfail: list[int] = []
    for k, s in EXT469_SUBSETS.items():
        v.put(f"ext469.const.{k}.mae", f(d["constant"][s]["mae"], 2))
        g = d["gt_mask_floor"].get(s)
        if g:
            v.put(f"ext469.gt.{k}.mae", f(g["mae"], 2))
            v.put(f"ext469.gt.{k}.n", fint(g["n"]))
        for m in SEG:
            runs = d["runs"].get(m, {})
            seeds = sorted(runs, key=lambda z: int(re.sub(r"\D", "", z) or 0))
            v.put(f"ext469.{m}.nseeds", fint(len(seeds)))
            for head, src in (("mask", m), ("head", f"{m}.head")):
                st = [runs[sd][head][s] for sd in seeds if head in runs[sd] and s in runs[sd][head]]
                if not st:
                    continue
                v.ms(f"ext469.{src}.{k}.mae", [r["mae"] for r in st], 2)
                v.put(
                    f"ext469.{src}.{k}.mae.seeds",
                    " and ".join(f(r["mae"], 2) for r in st) if len(st) == 2 else ", ".join(f(r["mae"], 2) for r in st),
                )
                nfail += [r.get("n_fail", 0) for r in st]
    v.put("ext469.nfail.max", fint(max(nfail)) if nfail else None)
    return d


# ------------------------------------------------------------------------------------------------ robustness
ROB_FIELDS = [
    ("acc", "accuracy", 2, 100),
    ("sens", "sensitivity", 2, 100),
    ("spec", "specificity", 2, 100),
    ("auc", "auc", 3, 1),
    ("mf1", "macro_f1", 3, 1),
]


def fill_rob(v: Store, path: Path, cls_res: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Corruption results of every classifier and condition, and the change of AUC from clean."""
    d = read_json_if_exists(path)
    if d is None:
        warn(f"robustness summary not found: {path}")
        return None
    if cls_res:
        cand = [
            (cls_res[m]["val_auc"], m)
            for m in d["models"]
            if m != "deepfetalnet" and m in cls_res and fin(cls_res[m]["val_auc"])
        ]
        if cand:
            b = max(cand)[1]
            v.put("cls.baseline.valbest", CLS_NAME[b])
            v.put("cls.baseline.valbest.id", b)
    v.put("rob.nframes", fint(d["protocol"]["n_frames"]))
    if d["protocol"].get("limit"):
        warn(f"robustness summary used --limit {d['protocol']['limit']} (test run, not the full split)")
    for m, o in d["models"].items():
        v.put(f"rob.{m}.nseeds", fint(len(o["seeds"])))
        for cid, c in o["conditions"].items():
            for key, fld, nd, sc in ROB_FIELDS:
                v.ms(f"rob.{m}.{cid}.{key}", c.get(fld, {}).get("values", []), nd, sc)
        if "clean" in o["conditions"]:
            for cid, c in o["conditions"].items():
                if cid == "clean":
                    continue
                dv = [a - b for a, b in zip(c["auc"]["values"], o["conditions"]["clean"]["auc"]["values"], strict=True)]
                v.ms(f"rob.{m}.{cid}.dauc", dv, 3, sign=True)
    return d


# ------------------------------------------------------------------------------------------------ cross-validation
def fill_cv(v: Store, cv_json: Path, asplit_json: Path, folds_json: Path = SPLITS / "cv" / "folds.json") -> None:
    """Cross-validation results and the video-level split sensitivity check."""
    if cv_json.exists():
        cv = read_json(cv_json)
        v.put("cv.k", fint(cv["k"]))
        v.put("cv.nboot", fint(cv["n_boot"]))
        folds = read_json(folds_json)["folds"]
        tw = [x["test"]["patients"] for x in folds]
        v.put("cv.test.women.min", fint(min(tw)))
        v.put("cv.test.women.max", fint(max(tw)))
        v.put("cv.val.frames", fint(folds[0]["val"]["frames"]))
        tf = [x["test"]["frames"] for x in folds]
        v.put("cv.test.frames.min", fint(min(tf)))
        v.put("cv.test.frames.max", fint(max(tf)))
        sig: dict[str, list[str]] = {"mf1": [], "auc": []}
        means = {}
        for m, e in cv["models"].items():
            for k, src, nd, sc in (
                ("acc", "accuracy", 1, 100),
                ("mf1", "macro_f1", 1, 100),
                ("auc", "auc", 3, 1),
                ("sens", "sensitivity", 1, 100),
                ("spec", "specificity", 1, 100),
            ):
                v.ms(f"cv.{m}.{k}", e[src]["values"], nd, sc)
                v.put(f"cv.{m}.{k}.range", f"{f(e[src]['min'], nd, sc)}--{f(e[src]['max'], nd, sc)}")
            means[m] = e["macro_f1"]["mean"]
            v.put(f"cv.{m}.nfolds", fint(e["n_folds"]))
            if "pooled" in e:
                v.put(f"cv.{m}.pooled.mf1", f(e["pooled"]["macro_f1"], 1, 100))
                v.put(f"cv.{m}.pooled.auc", f(e["pooled"]["auc"], 3))
                v.put(f"cv.{m}.pooled.spec", f(e["pooled"]["specificity"], 1, 100))
                v.put(f"cv.{m}.pooled.sens", f(e["pooled"]["sensitivity"], 1, 100))
            for k, src in (("mf1", "macro_f1"), ("auc", "auc")):
                r = e.get("rank", {}).get(src)
                if r:
                    v.put(f"cv.{m}.rank.{k}.best", fint(r["best"]))
                    v.put(f"cv.{m}.rank.{k}.worst", fint(r["worst"]))
                    v.put(f"cv.{m}.rank.{k}.mean", f(r["mean"], 1))
                    v.put(f"cv.{m}.rank.{k}", f"{f(r['mean'], 1)} ({r['best']}--{r['worst']})")
                vs = e.get("vs_deepfetalnet", {}).get(src)
                if vs:
                    nd, sc = (1, 100) if k == "mf1" else (3, 1)
                    lo, hi = vs["ci"]
                    v.put(f"cv.{m}.vsdfn.{k}", f(vs["diff"], nd, sc, sign=True))
                    v.put(f"cv.{m}.vsdfn.{k}.ci", ci_str(lo, hi, nd, sc))
                    v.put(f"cv.{m}.vsdfn.{k}.excl0", "yes" if lo > 0 or hi < 0 else "no")
                    if lo > 0:
                        sig[k].append(m)
            if "vs_deepfetalnet" in e:
                v.put(f"cv.{m}.vsdfn.foldsbetter", fint(e["vs_deepfetalnet"]["folds_better_macro_f1"]))
        for k in sig:
            v.put(f"cv.nsigbetter.{k}", fint(len(sig[k])))
            v.put(f"cv.sigbetter.{k}.names", ", ".join(CLS_NAME[m] for m in CLS if m in sig[k]) or "none")
        if means:
            b = max(means, key=means.get)
            v.put("cv.best.mf1.name", CLS_NAME[b])
            v.put("cv.best.mf1.id", b)
            v.put("cv.best.mf1.mean", f(means[b], 1, 100))
    if asplit_json.exists():
        for m, e in read_json(asplit_json).items():
            for st in ("val", "all", "seen", "unseen", "patient_split_test"):
                for k, src, nd, sc in (("acc", "accuracy", 1, 100), ("mf1", "macro_f1", 1, 100), ("auc", "auc", 3, 1)):
                    if st in e and src in e[st]:
                        d = e[st][src]
                        v.put(f"asplit.{m}.{st}.{k}", f"{f(d['mean'], nd, sc)} $\\pm$ {f(d['sd'], nd, sc)}")
                        v.put(f"asplit.{m}.{st}.{k}.mean", f(d["mean"], nd, sc))
                        if "n" in d:
                            v.put(f"asplit.{m}.{st}.n", fint(d["n"]))
