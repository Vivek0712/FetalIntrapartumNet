"""Print ``key | meaning | source`` lines for the ``\\res{}`` keys of .tex files that are not yet registered in
<paper-dir>/RESULT_KEYS.md, following the key scheme of ``fin.reporting.values``. Append the output to the registry:

    fin describe-keys --paper-dir DIR DIR/sections/04_results_discussion.tex >> DIR/RESULT_KEYS.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from fin.config import paper_dir
from fin.reporting.keys import registered_keys, strip_comments
from fin.reporting.names import CLS_PLAIN, SEG_NAME

MET = {
    "acc": "accuracy (%)",
    "prec": "precision, Standard positive (%)",
    "sens": "sensitivity, Standard positive (%)",
    "spec": "specificity, Standard positive (%)",
    "f1": "F1 of the Standard class",
    "mf1": "macro-F1",
    "auc": "ROC-AUC (p(Standard))",
}
SEGM = {
    "dice.fh": "FH Dice (%)",
    "dice.ps": "PS Dice (%)",
    "meandice": "mean foreground Dice (%)",
    "iou.fh": "FH IoU (%)",
    "iou.ps": "PS IoU (%)",
    "meaniou": "mean foreground IoU (%)",
    "hd95.fh": "FH HD95 (px, 256 grid)",
    "hd95.ps": "PS HD95 (px, 256 grid)",
    "bf1.fh": "FH boundary F1, 2 px (%)",
    "bf1.ps": "PS boundary F1, 2 px (%)",
    "pixacc": "pixel accuracy (%)",
    "empty.ps": "frames with empty PS prediction",
}
AOPM = {
    "mae": "MAE (deg)",
    "rmse": "RMSE (deg)",
    "bias": "bias pred-ref (deg, signed)",
    "loa": "Bland-Altman 95% LoA (deg)",
    "w5": "% within 5 deg (successes)",
    "w10": "% within 10 deg (successes)",
    "nfail": "number of failed frames",
    "maeci": "MAE 95% CI, video cluster bootstrap",
    "w5fi": "% within 5 deg, failures as misses",
    "nleak": "accepted Non-Standard frames (no reference)",
    "nmissed": "Standard frames rejected by the gate",
}
SETN = {"jnu": "JNU-IFM test SPHead", "iugc": "IUGC 2024 test (300)"}
DATA = "splits/frames.csv"


def describe(k: str) -> tuple[str, str]:
    """(meaning, source) of a key following the store's key scheme."""
    p = k.split(".")
    tail = ""
    if p[-1] in ("mean", "sd"):
        tail = f", {p[-1]} over seeds only"
        p = p[:-1]
    kk = ".".join(p)
    if p[0] == "data":
        if p[1] == "jnu" and len(p) == 3:
            return f"JNU-IFM {p[2]} (count over all frames)", DATA
        if p[1] == "split":
            return f"{p[2]} split: {' '.join(p[3:])} (count; seg = SPHead frames)", DATA + " (split column)"
        if p[1] == "iugc":
            return f"IUGC 2024 test: {' '.join(p[2:])}", "$FIN_DATA/iugc/processed/aop.csv"
        if "refdiff" in p:
            return f"abs(aop_ref - aop_ref_challenge_code) over SPHead frames: {p[-1]} (deg)", DATA
        return f"dataset fact {kk}", DATA
    if p[0] == "cls":
        if p[1] in CLS_PLAIN:
            m = CLS_PLAIN[p[1]]
            if p[2] == "gate":
                return (
                    f"{m} as gate on JNU test at thr 0.5: {' '.join(p[3:])} (mean over seeds)",
                    f"results/runs/cls/{p[1]}/seed*/preds_test.npz",
                )
            if p[2] in MET:
                ci = " 95% CI (video cluster bootstrap of seed mean)" if p[-1] == "ci" else " mean +/- SD over seeds"
                return (
                    f"{m} JNU test {MET[p[2]]},{ci}{tail}",
                    f"results/runs/cls/{p[1]}/seed*/metrics_test.json, preds_test.npz",
                )
            return f"{m} {' '.join(p[2:])}", f"results/runs/cls/{p[1]}/seed*/metrics_test.json"
        if p[1] == "rank":
            return (
                f"display name of the classifier ranked first by mean test {p[2]}",
                "results/runs/cls/*/seed*/metrics_test.json",
            )
        if p[1] == "baseline":
            return (
                "baseline (not DeepFetalNet) with the highest mean validation AUC (name)",
                "results/runs/cls/*/seed*/metrics_test.json -> val.auc",
            )
        return f"classification {' '.join(p[1:])}", "results/runs/cls/*/seed*/metrics_test.json"
    if p[0] == "seg":
        if p[1] in ("prior", "bg"):
            what = "image-blind area-matched mean-shape prior" if p[1] == "prior" else "all-background prediction"
            if p[2] == "ntrain":
                return "number of training SPHead masks used for the mean-shape prior", "$FIN_SEG_CACHE/jnu_mask.npy"
            return (
                f"{what}: {SETN.get(p[2], p[2])} {SEGM.get('.'.join(p[3:]), '.'.join(p[3:]))}",
                "$FIN_SEG_CACHE (computed by fin.reporting.values)",
            )
        if p[1] in SEG_NAME:
            m = SEG_NAME[p[1]]
            if p[2] in ("seeds", "nseeds", "params", "params.m", "name"):
                return f"{m} {p[2]}", f"results/runs/seg/{p[1]}/seed*/metrics.json"
            return (
                f"{m} {SETN.get(p[2], p[2])} {SEGM.get('.'.join(p[3:]), '.'.join(p[3:]))}, per-image mean, mean +/- SD over seeds{tail}",
                f"results/runs/seg/{p[1]}/seed*/metrics.json",
            )
        return f"segmentation {' '.join(p[1:])}", "results/runs/seg/*/seed*/metrics.json"
    if p[0] == "aop":
        src = p[1]
        if src == "const":
            if p[2] == "value":
                return "constant AoP baseline: mean aop_ref_challenge_code of JNU training SPHead frames (deg)", DATA
            return f"constant-AoP baseline vs reference, {SETN.get(p[2], p[2])}: {AOPM.get(p[3], p[3])}", DATA
        if src == "gt":
            return (
                f"challenge geometry on GT masks vs reference (floor), {SETN.get(p[2], p[2])}: {AOPM.get(p[3], p[3])}",
                "results/runs/seg/ceiling.json",
            )
        if src == "e2e":
            if len(p) == 2 or p[2] in ("gate", "segmenter", "seeds"):
                return f"end-to-end evaluation: {p[-1]}", "fin.evaluation.e2e"
            return (
                f"end-to-end JNU test (DeepFetalNet gate thr 0.5 -> AoP-GAU-Net geometry), seed-paired: {AOPM.get(p[3], p[3])}",
                "results/runs/cls/deepfetalnet/seed*/preds_test.npz + runs/seg/aopgaunet/seed*/per_image_jnu_test.csv",
            )
        if src in SEG_NAME:
            hd = "direct regression head" if len(p) > 2 and p[2] == "head" else "geometry from predicted masks"
            q = p[3:] if p[2] == "head" else p[2:]
            return (
                f"{SEG_NAME[src]} AoP ({hd}) vs primary reference, {SETN.get(q[0], q[0])}: {AOPM.get(q[1], q[1])}, mean +/- SD over seeds{tail}",
                f"results/runs/seg/{src}/seed*/metrics.json, per_image_*.csv",
            )
    if p[0] == "eff":
        if p[1] == "hw":
            return f"benchmark machine/software: {p[2]}", "results/runs/efficiency/results.json -> hardware"
        return f"{p[1]} efficiency: {' '.join(p[2:])}", "results/runs/efficiency/results.json"
    if p[0] == "rob":
        return (
            f"robustness, {CLS_PLAIN.get(p[1], p[1])} on corrupted JNU test, condition {p[2] if len(p) > 2 else ''}: {' '.join(p[3:])}",
            "results/runs/robustness/summary.json",
        )
    return "(describe)", "(source)"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex", nargs="+", type=Path)
    ap.add_argument("--paper-dir", type=Path, default=None, help="manuscript folder (default: $FIN_PAPER)")
    a = ap.parse_args(argv)
    reg = set(registered_keys(paper_dir(a.paper_dir) / "RESULT_KEYS.md"))
    seen: list[str] = []
    for path in a.tex:
        for k in re.findall(r"\\res\{([^}]+)\}", strip_comments(path.read_text())):
            if k not in reg and k not in seen:
                seen.append(k)
    for k in seen:
        m, s = describe(k)
        sys.stdout.write(f"{k} | {m} | {s}\n")


if __name__ == "__main__":
    main()
