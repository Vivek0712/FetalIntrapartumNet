"""Result figures, drawn only from dataset files and saved model outputs.

    Fig 6  fig6-gradcam.png         Grad-CAM of the seven classifiers (seed-0 checkpoints) on two JNU test frames,
                                    clean and corrupted (``fin.robustness.corruptions``)
    Fig 7  fig7-roc.pdf             test ROC curves of the seven classifiers (seed 0), legend AUC mean +/- SD over seeds
    Fig 8  fig8-segmentation.png    the four segmenters (seed 0) on 3 JNU + 3 IUGC test frames
    Fig 9  fig9-aop-cases.png       six JNU test frames: GT vs predicted masks and AoP construction (seed 0)
    Supp   figS-bland-altman.pdf    Bland-Altman plots of the geometric AoP (each segmenter) and the regression head
    figure_frames.json              every selected frame id, the selection rule and the printed values

Selection rules (fixed in advance, no manual choice)
    Fig 6: per class, the test frame whose predicted probability of its true class, averaged over the seven seed-0
           models, is the median of that class (ties: lowest frame id). Rows: clean, speckle var 0.10, motion blur
           9, Gaussian var 0.05.
    Fig 8: per set, the frames whose mean foreground Dice ((FH + PS) / 2, per image), averaged over the four seed-0
           models, is closest to the 10th, 50th and 90th percentile of that set.
    Fig 9: from the ``--aop-model`` seed-0 per-frame output on JNU test, among frames with a valid angle: the frames
           whose absolute error is closest to the 10/25/50/75/90th percentile, plus the largest error.
Ultrasound pixels are shown as stored in the evaluation caches (256x256 for segmentation/AoP, 224x224 for
classification); nothing else is applied to them. Every angle is recomputed with ``fin.geometry`` and checked
against the saved per-frame value.

    fin figures [--paper-dir DIR | --out DIR] [--runs DIR] [--checkpoints DIR] [--only roc gradcam seg aop ba]
                [--device cpu]
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from fin.config import RESULT_RUNS, paper_dir, pick_device, runs_dir, seg_cache_dir
from fin.data.cache import build_cls_cache
from fin.data.datasets import to_classifier_input
from fin.evaluation.classification import load_run
from fin.evaluation.seg_metrics import dice
from fin.geometry import Construction, aop_from_mask, mask_construction
from fin.io import portable_path, read_json_if_exists, setup_logging, write_json
from fin.reporting.draw import ellipse_xy, overlay
from fin.reporting.names import CLS, CLS_PLAIN, SEG, SEG_NAME
from fin.reporting.values import load_cache_set
from fin.robustness.corruptions import CORRUPTION_NAME, condition, corrupt
from fin.runs import seed_dirs

log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
FH_C, PS_C = (0.20, 0.55, 1.00), (1.00, 0.55, 0.10)
MODEL_C = {
    "psfhspnet": "#1f77b4",
    "resnet152": "#8c564b",
    "squeezenet11": "#2ca02c",
    "shufflenetv2": "#9467bd",
    "mobilenetv2": "#e377c2",
    "mobilenetv3": "#17becf",
    "deepfetalnet": "#d62728",
}
SEL = {}


def _check_same_frames(saved: pd.DataFrame, cache: pd.DataFrame, source: str) -> None:
    """Raise if a saved per-frame table and the cache set do not list the same frames in the same order."""
    if not np.array_equal(saved.frame_id.astype(str).to_numpy(), cache.frame_id.astype(str).to_numpy()):
        raise RuntimeError(f"{source}: frame order differs from the segmentation cache")


def short(fid: object) -> str:
    """Frame id without the .png suffix."""
    return str(fid).replace(".png", "")


# ------------------------------------------------------------------------------------------------ Fig 7: ROC
def fig_roc(cls_root: Path, out: Path) -> None:
    """Figure 7: test ROC curves (seed 0) with the AUC mean +/- SD over seeds in the legend."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], ls=":", c="0.6", lw=0.8)
    used = {}
    for m in CLS:
        runs = seed_dirs(cls_root, m, "preds_test.npz")
        if not runs:
            continue
        aucs = []
        for _, d in runs:
            z = np.load(d / "preds_test.npz")
            aucs.append(roc_auc_score(z["y"], z["p_standard"]))
        s0, d0 = runs[0]
        z = np.load(d0 / "preds_test.npz")
        fpr, tpr, _ = roc_curve(z["y"], z["p_standard"])
        lab = f"{CLS_PLAIN[m]}  {np.mean(aucs):.3f}" + (f" ± {np.std(aucs, ddof=1):.3f}" if len(aucs) > 1 else "")
        ax.plot(fpr, tpr, c=MODEL_C[m], lw=1.8 if m == "deepfetalnet" else 1.0, label=lab)
        used[m] = {"seed_plotted": s0, "seeds": [s for s, _ in runs], "auc": aucs, "n": int(len(z["y"]))}
    ax.set_xlabel("1 − specificity (Non-Standard frames accepted)")
    ax.set_ylabel("Sensitivity (Standard frames accepted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(title="Model, AUC (mean ± SD over seeds)", loc="lower right", fontsize=6, title_fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig7-roc.pdf")
    plt.close(fig)
    SEL["fig7"] = used
    log.info("fig7 -> %s %s", out / "fig7-roc.pdf", list(used))


# ------------------------------------------------------------------------------------------------ Fig 6: Grad-CAM
TARGET = {  # model id -> candidate attribute paths of the last convolutional stage
    "deepfetalnet": ["arm"],
    "psfhspnet": ["layer2"],
    "resnet152": ["backbone.layer4"],
    "squeezenet11": ["backbone.features"],
    "shufflenetv2": ["backbone.conv5", "conv5"],
    "mobilenetv2": ["backbone.features", "features"],
    "mobilenetv3": ["backbone.features"],
}


def _getattr_path(obj: object, path: str) -> object:
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017): ReLU(sum_k mean(dy_c/dA_k) A_k) at the target layer, bilinearly
    upsampled to the input size and scaled to [0, 1] per map."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model, self.acts, self.grads = model, None, None
        for mod in model.modules():  # in-place activations break tensor hooks
            if hasattr(mod, "inplace"):
                mod.inplace = False
        layer.register_forward_hook(self._fwd)

    def _fwd(self, mod: torch.nn.Module, inp: object, out: torch.Tensor) -> None:
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float, int]:
        F = torch.nn.functional
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        p = torch.softmax(logits.float(), 1)[0, 1].item()
        c = int(logits.argmax(1).item())
        logits[0, c].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        return cam / cam.max() if cam.max() > 0 else cam, p, c


def fig_gradcam(cls_root: Path, out: Path, device: str = "cpu") -> None:
    """Figure 6: Grad-CAM of the seven seed-0 classifiers on two test frames, clean and corrupted."""
    runs = {m: seed_dirs(cls_root, m, "best.pt") for m in CLS}
    runs = {m: r[0][1] for m, r in runs.items() if r}
    if not runs:
        log.info("fig6: no classifier checkpoints")
        return
    # frame selection from the seed-0 test predictions
    ptrue, fids, ys = [], None, None
    for d in runs.values():
        if not (d / "preds_test.npz").exists():
            continue
        z = np.load(d / "preds_test.npz")
        if fids is None:
            fids, ys = z["frame_id"].astype(str), z["y"].astype(int)
        elif not np.array_equal(fids, z["frame_id"].astype(str)):
            continue
        ptrue.append(np.where(ys == 1, z["p_standard"], 1 - z["p_standard"]))
    score = np.mean(ptrue, 0)
    chosen = []
    for cls_y, name in ((1, "Standard"), (0, "Non-Standard")):
        idx = np.flatnonzero(ys == cls_y)
        med = np.median(score[idx])
        order = sorted(idx, key=lambda i: (abs(score[i] - med), fids[i]))
        chosen.append((name, fids[order[0]], float(score[order[0]])))
    z = np.load(build_cls_cache(224))
    pos = {f: i for i, f in enumerate(z["frame_id"])}
    conds = ["clean", "speckle-0.10", "blur-9", "gauss-0.05"]
    rows = [(name, fid, c) for name, fid, _ in chosen for c in conds]
    models = [m for m in CLS if m in runs]
    dev = pick_device(device)
    cams = {}
    layers = {}
    for m in models:
        _, net = load_run(runs[m], dev)
        path = next(p for p in TARGET[m] if _has(net, p))
        layers[m] = path
        gc = GradCAM(net, _getattr_path(net, path))
        for r, (_, fid, c) in enumerate(rows):
            x = corrupt(z["img"][pos[fid]], c, fid)
            cams[(r, m)] = gc(torch.from_numpy(to_classifier_input(x)[None]).to(dev))
        del net
    nr, nc = len(rows), len(models) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(6.9, 6.9 * nr / nc * 1.04))
    for r, (name, fid, c) in enumerate(rows):
        x = corrupt(z["img"][pos[fid]], c, fid)
        ax = axes[r, 0]
        ax.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        lab = "clean" if c == "clean" else f"{CORRUPTION_NAME[condition(c)[1]]},\n{condition(c)[3].lower()}"
        ax.set_ylabel(f"{name}\n{lab}", fontsize=6)
        for j, m in enumerate(models):
            cam, p, k = cams[(r, m)]
            a = axes[r, j + 1]
            a.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a.imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1, interpolation="bilinear")
            a.text(
                3,
                220,
                f"{'Std' if k == 1 else 'Non-Std'}  p={p:.2f}",
                color="w",
                fontsize=5,
                va="bottom",
                bbox=dict(fc="k", alpha=0.5, lw=0, pad=1),
            )
            if r == 0:
                a.set_title(CLS_PLAIN[m].replace(" 1.0x", ""), fontsize=6)
        if r == 0:
            axes[0, 0].set_title("Input", fontsize=6)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.subplots_adjust(left=0.07, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.03)
    fig.savefig(out / "fig6-gradcam.png", dpi=220)
    plt.close(fig)
    SEL["fig6"] = {
        "rule": "per class, frame whose mean p(true class) over seed-0 models is the class median",
        "frames": [{"class": n, "frame_id": f, "mean_p_true": s} for n, f, s in chosen],
        "conditions": conds,
        "target_layers": layers,
        "checkpoints": {m: portable_path(p) for m, p in runs.items()},
        "panels": {
            f"{rows[r][1]}|{rows[r][2]}|{m}": {"p_standard": cams[(r, m)][1], "pred": cams[(r, m)][2]}
            for r in range(nr)
            for m in models
        },
    }
    log.info("fig6 -> %s %s", out / "fig6-gradcam.png", layers)


def _has(obj: object, path: str) -> bool:
    try:
        _getattr_path(obj, path)
        return True
    except AttributeError:
        return False


# ------------------------------------------------------------------------------------------------ seg helpers
def load_cache(cache: Path, s: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Frame table, images, masks (memory-mapped) and rows of one evaluation set of the segmentation cache."""
    d, sel, msk = load_cache_set(cache, s)
    name = "jnu" if s == "jnu_test" else "iugc"
    return d, np.load(cache / f"{name}_img.npy", mmap_mode="r"), msk, sel


# ------------------------------------------------------------------------------------------------ Fig 8
def fig_seg(seg_root: Path, cache: Path, out: Path) -> None:
    """Figure 8: the four seed-0 segmenters on frames at the 10th/50th/90th percentile of mean Dice."""
    runs = {m: seed_dirs(seg_root, m, "preds_jnu_test.npz") for m in SEG}
    runs = {m: r[0][1] for m, r in runs.items() if r}
    if not runs:
        log.info("fig8: no segmentation outputs")
        return
    rows, info = [], []
    for s, tag in (("jnu_test", "JNU-IFM"), ("iugc", "IUGC 2024")):
        d, img, msk, sel = load_cache(cache, s)
        per = []
        for rd in runs.values():
            t = pd.read_csv(rd / f"per_image_{s}.csv")
            _check_same_frames(t, d, f"{rd}/per_image_{s}.csv")
            per.append(np.nanmean(np.c_[t.dice_FH.to_numpy(float), t.dice_PS.to_numpy(float)], 1))
        score = np.nanmean(per, 0)
        for q in (0.1, 0.5, 0.9):
            v = np.nanquantile(score, q)
            i = int(np.nanargmin(np.abs(score - v)))
            rows.append((s, tag, i, q))
            info.append(
                {"set": s, "quantile": q, "frame_id": str(d.frame_id[i]), "mean_fg_dice_over_models": float(score[i])}
            )
    models = [m for m in SEG if m in runs]
    preds = {(m, s): np.load(runs[m] / f"preds_{s}.npz")["labels"] for m in models for s in ("jnu_test", "iugc")}
    nc = 2 + len(models)
    fig, axes = plt.subplots(len(rows), nc, figsize=(6.9, 6.9 * len(rows) / nc * 1.02))
    for r, (s, tag, i, q) in enumerate(rows):
        d, img, msk, sel = load_cache(cache, s)
        im, gt = np.asarray(img[sel[i]]), np.asarray(msk[sel[i]])
        axes[r, 0].imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[r, 0].set_ylabel(f"{tag}, P{int(q * 100)}\n{short(d.frame_id[i])}", fontsize=5)
        overlay(axes[r, 1], im, gt, (FH_C, PS_C), alpha=0.35)
        for j, m in enumerate(models):
            lab = preds[(m, s)][i]
            overlay(axes[r, 2 + j], im, lab, (FH_C, PS_C), alpha=0.35)
            dfh, dps = dice(lab == 1, gt == 1), dice(lab == 2, gt == 2)
            axes[r, 2 + j].text(
                3,
                252,
                f"FH {dfh:.2f}  PS {dps:.2f}",
                color="w",
                fontsize=5,
                va="bottom",
                bbox=dict(fc="k", alpha=0.5, lw=0, pad=1),
            )
            info[r][f"dice_{m}"] = [float(dfh), float(dps)]
    for j, t in enumerate(["Image", "Ground truth"] + [SEG_NAME[m] for m in models]):
        axes[0, j].set_title(t, fontsize=6)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.subplots_adjust(left=0.07, right=0.995, top=0.965, bottom=0.005, wspace=0.03, hspace=0.03)
    fig.savefig(out / "fig8-segmentation.png", dpi=220)
    plt.close(fig)
    SEL["fig8"] = {
        "rule": "per set, frames closest to P10/P50/P90 of mean FG Dice averaged over seed-0 models",
        "runs": {m: portable_path(p) for m, p in runs.items()},
        "frames": info,
    }
    log.info("fig8 -> %s", out / "fig8-segmentation.png")


# ------------------------------------------------------------------------------------------------ Fig 9
def draw_case(ax: plt.Axes, img: np.ndarray, lab: np.ndarray, con: Construction | None, col: str) -> None:
    """One Figure 9 panel: masks, the fitted FH ellipse, the PS axis, the tangent and the angle arc."""
    overlay(ax, img, lab, (FH_C, PS_C), alpha=0.25)
    if con is None:
        ax.text(128, 128, "no valid angle", color="r", ha="center", fontsize=6)
        return
    ex, ey = ellipse_xy(con.fh_ellipse, n=200)
    ax.plot(ex, ey, c=col, lw=0.7, ls="--")
    P1, P2, T = map(np.array, (con.p1, con.p2, con.t))
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], c=col, lw=1.0)
    ext = P2 + 1.25 * (T - P2)
    ax.plot([P2[0], ext[0]], [P2[1], ext[1]], c=col, lw=1.0)
    a1, a2 = math.atan2(*(P1 - P2)[::-1]), math.atan2(*(T - P2)[::-1])
    dlt = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    r = 14
    s = np.linspace(0, 1, 30)
    ax.plot(P2[0] + r * np.cos(a1 + s * dlt), P2[1] + r * np.sin(a1 + s * dlt), c="yellow", lw=0.9)
    for name, P in (("P1", P1), ("P2", P2), ("T", T)):
        ax.plot(*P, "o", ms=2.2, c="yellow")
        ax.text(
            P[0] + 3,
            P[1] - 3,
            name.replace("P1", "$P_1$").replace("P2", "$P_2$").replace("T", "$T$"),
            color="yellow",
            fontsize=5,
            clip_on=True,
        )


def fig_aop(seg_root: Path, cache: Path, out: Path, model: str = "aopgaunet") -> None:
    """Figure 9: six JNU test frames (error percentiles and the maximum) with GT and predicted constructions."""
    runs = seed_dirs(seg_root, model, "per_image_jnu_test.csv")
    if not runs:
        log.info("fig9: no outputs for %s", model)
        return
    s0, rd = runs[0]
    t = pd.read_csv(rd / "per_image_jnu_test.csv")
    lab = np.load(rd / "preds_jnu_test.npz")["labels"]
    d, img, msk, sel = load_cache(cache, "jnu_test")
    _check_same_frames(t, d, f"{rd}/per_image_jnu_test.csv")
    err = np.abs(t.aop_mask.to_numpy(float) - t.ref_primary.to_numpy(float))
    ok = np.flatnonzero(np.isfinite(err))
    if len(ok) == 0:
        log.info("fig9: %s has no valid angle on JNU test", model)
        return
    picks, used = [], set()
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        v = np.quantile(err[ok], q)
        cand = sorted(ok, key=lambda i: (abs(err[i] - v), str(t.frame_id[i])))
        i = next((c for c in cand if c not in used), cand[0])
        used.add(i)
        picks.append((q, int(i)))
    fig, axes = plt.subplots(3, 4, figsize=(6.6, 5.1))
    info = []
    for n, (q, i) in enumerate(picks):
        im, gt, pr = np.asarray(img[sel[i]]), np.asarray(msk[sel[i]]), lab[i]
        cg, cp = mask_construction(gt), mask_construction(pr)
        a_gt, a_pr = aop_from_mask(gt, 1, 2), aop_from_mask(pr, 1, 2)
        # the drawn construction must give exactly the evaluated angles
        if cg is None or abs(cg.angle - a_gt) >= 1e-6 or abs(a_gt - t.ref_primary[i]) >= 1e-3:
            raise RuntimeError(f"fig9: frame {t.frame_id[i]}: GT construction {a_gt} != saved {t.ref_primary[i]}")
        if cp is None or abs(cp.angle - a_pr) >= 1e-6 or abs(a_pr - t.aop_mask[i]) >= 1e-3:
            raise RuntimeError(f"fig9: frame {t.frame_id[i]}: predicted construction {a_pr} != saved {t.aop_mask[i]}")
        ax_g, ax_p = axes[n // 2, 2 * (n % 2)], axes[n // 2, 2 * (n % 2) + 1]
        draw_case(ax_g, im, gt, cg, "lime")
        draw_case(ax_p, im, pr, cp, "red")
        lbl = "max" if q == 1.0 else f"P{int(q * 100)}"
        ax_g.set_title(f"{short(d.frame_id[i])} ({lbl})\nGT masks: reference {a_gt:.1f}°", fontsize=5.5)
        ax_p.set_title(f"\n{SEG_NAME[model]}: {a_pr:.1f}°, |error| {abs(a_pr - a_gt):.1f}°", fontsize=5.5)
        info.append(
            {
                "quantile": q,
                "frame_id": str(d.frame_id[i]),
                "aop_ref": float(a_gt),
                "aop_pred": float(a_pr),
                "abs_error": float(abs(a_pr - a_gt)),
            }
        )
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(0, 255)
        a.set_ylim(255, 0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04, hspace=0.25)
    fig.savefig(out / "fig9-aop-cases.png", dpi=240)
    plt.close(fig)
    SEL["fig9"] = {
        "rule": "abs error closest to P10/P25/P50/P75/P90 of successful frames, plus the maximum",
        "model": model,
        "seed": s0,
        "run": portable_path(rd),
        "n_valid": int(len(ok)),
        "n_frames": int(len(t)),
        "frames": info,
    }
    log.info("fig9 -> %s", out / "fig9-aop-cases.png")


# ------------------------------------------------------------------------------------------------ Bland-Altman
def fig_ba(seg_root: Path, out: Path) -> None:
    """Supplementary figure: Bland-Altman plots of every segmenter's geometric AoP and the regression head."""
    srcs = [(m, "aop_mask", f"{SEG_NAME[m]} (geometry)") for m in SEG] + [
        ("aopgaunet", "aop_head", "AoP-GAU-Net (head)")
    ]
    panels = []
    for m, col, name in srcs:
        r = seed_dirs(seg_root, m, "per_image_jnu_test.csv")
        if r:
            panels.append((m, col, name, r[0][1]))
    if not panels:
        log.info("figS: no outputs")
        return
    fig, axes = plt.subplots(2, len(panels), figsize=(2.0 * len(panels) + 0.6, 4.6), squeeze=False)
    for j, (_, col, name, rd) in enumerate(panels):
        for i, (s, tag) in enumerate((("jnu_test", "JNU-IFM test"), ("iugc", "IUGC 2024"))):
            ax = axes[i, j]
            p = rd / f"per_image_{s}.csv"
            if not p.exists():
                ax.axis("off")
                continue
            t = pd.read_csv(p)
            if col not in t:
                ax.axis("off")
                continue
            pr, rf = t[col].to_numpy(float), t.ref_primary.to_numpy(float)
            ok = np.isfinite(pr) & np.isfinite(rf)
            if ok.sum() < 3:
                ax.text(0.5, 0.5, f"{(~ok).sum()} failures", ha="center", transform=ax.transAxes)
            else:
                dd, mm = pr[ok] - rf[ok], (pr[ok] + rf[ok]) / 2
                b, sd = dd.mean(), dd.std(ddof=1)
                ax.scatter(mm, dd, s=2, c="k", alpha=0.4, lw=0)
                for y, ls in ((b, "-"), (b - 1.96 * sd, "--"), (b + 1.96 * sd, "--")):
                    ax.axhline(y, c="r", ls=ls, lw=0.7)
                ax.set_title(
                    f"{name}\n{tag}\nbias {b:+.1f}°, LoA [{b - 1.96 * sd:.1f}, {b + 1.96 * sd:.1f}]°", fontsize=6
                )
            ax.tick_params(labelsize=6)
    fig.supylabel("predicted − reference (°)", fontsize=8)
    fig.supxlabel("mean of predicted and reference (°)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figS-bland-altman.pdf")
    plt.close(fig)
    log.info("figS -> %s", out / "figS-bland-altman.pdf")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULT_RUNS, help="root of the saved outputs (cls/, seg/)")
    ap.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="classification runs with best.pt for Grad-CAM (default: $FIN_RUNS/cls)",
    )
    ap.add_argument("--seg-cache", type=Path, default=None, help="default: $FIN_SEG_CACHE")
    ap.add_argument("--out", type=Path, default=None, help="default: <paper-dir>/figs")
    ap.add_argument("--paper-dir", type=Path, default=None, help="manuscript folder (default: $FIN_PAPER)")
    ap.add_argument("--only", nargs="*", default=["roc", "gradcam", "seg", "aop", "ba"])
    ap.add_argument("--aop-model", default="aopgaunet")
    ap.add_argument("--device", default="cpu", help="Grad-CAM device")
    a = ap.parse_args(argv)
    a.out = a.out or paper_dir(a.paper_dir) / "figs"
    setup_logging()
    cls_root, seg_root = a.runs / "cls", a.runs / "seg"
    cache = a.seg_cache or seg_cache_dir()
    a.out.mkdir(parents=True, exist_ok=True)
    if "roc" in a.only:
        fig_roc(cls_root, a.out)
    if "gradcam" in a.only:
        fig_gradcam(a.checkpoints or runs_dir() / "cls", a.out, a.device)
    if "seg" in a.only:
        fig_seg(seg_root, cache, a.out)
    if "aop" in a.only:
        fig_aop(seg_root, cache, a.out, a.aop_model)
    if "ba" in a.only:
        fig_ba(seg_root, a.out)
    p = a.out / "figure_frames.json"
    old = read_json_if_exists(p) or {}
    old.update(SEL)
    write_json(p, old, default=float)


if __name__ == "__main__":
    main()
