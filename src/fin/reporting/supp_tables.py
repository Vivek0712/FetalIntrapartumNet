"""Supplementary tables generated from the saved run summaries.

Writes <paper-dir>/sections/tables/supp_results.tex with the Youden-threshold table, the video-level split check, the
pooled Dice/IoU table, the second external AoP set (IUGC 2025 release), the Bland-Altman figure float and the list
of frames shown in the figures (<paper-dir>/figs/figure_frames.json, written by ``fin figures``).

    fin supp-tables --paper-dir DIR [--runs DIR]     (or FIN_PAPER=DIR)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fin.config import RESULT_RUNS, paper_dir
from fin.io import read_json, read_json_if_exists, setup_logging, write_text
from fin.reporting.names import CLS, CLS_NAME

log = logging.getLogger(__name__)

CLS_ORDER = list(CLS)
SEG_ORDER = [("unet", "U-Net"), ("fatnet", "FAT-Net"), ("malunet", "MALUNet"), ("aopgaunet", "AoP-GAU-Net")]


def pm(d: dict[str, Any] | None, scale: float = 100.0, nd: int = 1) -> str:
    """``mean $\\pm$ SD`` of a summary entry (``--`` when missing)."""
    if d is None:
        return "--"
    m, s = d["mean"] * scale, d.get("sd", 0.0) * scale
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def tex_id(s: object) -> str:
    """Escape underscores for LaTeX."""
    return str(s).replace("_", "\\_")


def youden_table(cls: dict[str, Any]) -> str:
    """Supplementary table: the validation-chosen Youden threshold applied to test."""
    rows = []
    for k in CLS_ORDER:
        if k not in cls:
            continue
        y = cls[k]["youden"]
        thr = ", ".join(f"{t:.2f}" for t in y["thresholds"])
        rows.append(
            f"{CLS_NAME[k]} & {thr} & {pm(y['accuracy'])} & {pm(y['macro_f1'])} & "
            f"{pm(y['sensitivity_standard'])} & {pm(y['specificity_standard'])} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\small",
            "\\caption{Classification on the JNU-IFM test split at the secondary operating point: for each run, the "
            "threshold that maximises Youden's $J$ on that run's validation predictions, applied unchanged to the test "
            "set. Thresholds per seed; metrics in \\% as mean $\\pm$ SD over seeds (Standard positive).}",
            "\\label{tab:youden}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Model & Thresholds & Accuracy & Macro-F1 & Sensitivity & Specificity \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def pooled_table(seg: dict[str, Any]) -> str:
    """Supplementary table: pooled Dice and IoU."""
    rows = []
    for k, name in SEG_ORDER:
        if k not in seg:
            continue
        cells = []
        for st in ("jnu_test", "iugc"):
            x = seg[k][st]["seg"]
            cells += [
                pm(x.get("pooled_dice_FH")),
                pm(x.get("pooled_dice_PS")),
                pm(x.get("pooled_iou_FH")),
                pm(x.get("pooled_iou_PS")),
            ]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    return "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\caption{Pooled segmentation scores (\\%): intersections and unions summed over all test frames before the "
            "ratio, mean $\\pm$ SD over seeds. Main-text Table~4 gives the per-image means.}",
            "\\label{tab:pooled}",
            "\\begin{tabular}{lcccccccc}",
            "\\toprule",
            " & \\multicolumn{4}{c}{JNU-IFM test} & \\multicolumn{4}{c}{IUGC 2024} \\\\",
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
            "Model & Dice FH & Dice PS & IoU FH & IoU PS & Dice FH & Dice PS & IoU FH & IoU PS \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def frames_text(sel: dict[str, Any]) -> str:
    """Paragraph listing the frames shown in the figures."""
    out = ["{\\sloppy", "\\paragraph{Frames shown in the figures.}"]
    f6 = sel.get("fig6", {}).get("frames", [])
    if f6:
        out.append(
            "Figure~\\ref{fig:clsqual} (Grad-CAM): "
            + "; ".join(f"{x['class']} \\texttt{{{tex_id(x['frame_id'])}}}" for x in f6)
            + "."
        )

    def fmt(x: dict[str, Any]) -> str:
        q = x.get("quantile", x.get("pct"))
        lab = x.get("label") or (f"P{round(q * 100)}" if isinstance(q, (int, float)) else str(q or ""))
        st = {"jnu_test": "JNU-IFM", "iugc": "IUGC 2024"}.get(x.get("set", ""), x.get("set", ""))
        return f"{st} {lab} \\texttt{{{tex_id(x.get('frame_id', x.get('id', '')))}}}".strip()

    for key, title in (("fig8", "Main-text Figure~7 (segmentation)"), ("fig9", "Main-text Figure~8 (AoP cases)")):
        fr = sel.get(key, {}).get("frames") or []
        if fr:
            out.append(f"{title}: " + "; ".join(fmt(x) for x in fr if isinstance(x, dict)) + ".")
    out.append(
        "The selection rule for each figure is stated in its caption; \\texttt{figs/figure_frames.json} "
        "in the code release records the selections.".replace("_", "\\_")
    )
    out.append("\\par}")
    return "\n".join(out)


def asplit_table(runs: Path) -> str:
    """Supplementary table: video-level split versus the locked split."""
    p = runs / "cls_authorsplit" / "author_split_check.json"
    if not p.exists():
        return ""
    A = read_json(p)
    names = {"deepfetalnet": "DeepFetalNet", "mobilenetv3": "MobileNetV3-Large"}
    labels = [
        ("val", "Video-level split, validation"),
        ("all", "Video-level split, test"),
        ("seen", "\\quad test frames of women with videos in training"),
        ("unseen", "\\quad test frames of women without"),
        ("patient_split_test", "Fixed patient-disjoint split, test"),
    ]
    rows = []
    for m, nm in names.items():
        for k, lab in labels:
            d = A[m].get(k)
            if not d:
                continue
            n = d["accuracy"].get("n", "")
            rows.append(
                f"{nm if k == 'val' else ''} & {lab} & {n} & {pm(d['accuracy'])} & {pm(d['macro_f1'])} & "
                f"{d['auc']['mean']:.3f} $\\pm$ {d['auc']['sd']:.3f} \\\\"
            )
    return "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\footnotesize",
            "\\caption{Sensitivity of the classification result to the split. DeepFetalNet and MobileNetV3-Large were retrained "
            "(seeds 0--2, same protocol) on a video-level split of JNU-IFM with 61/9/8 videos (4979/623/622 frames), in which five of the "
            "eight test videos come from women who also have videos in training. Mean $\\pm$ SD over seeds, threshold 0.5; accuracy and "
            "macro-F1 in \\%. The last row of each model repeats its result on the fixed patient-disjoint split.}",
            "\\label{tab:asplit}",
            "\\begin{tabular}{llrccc}",
            "\\toprule",
            "Model & Evaluation & Frames & Accuracy & Macro-F1 & AUC \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def ext469_table(runs: Path) -> str:
    """Supplementary table: AoP MAE against the landmark AoP on the IUGC 2025 release (all, labelled, unseen)."""
    p = runs / "ext469" / "summary.json"
    if not p.exists():
        return ""
    d = read_json(p)
    subs = [("all469", "All"), ("labelled300", "Labelled"), ("unseen_by_ours", "Unseen$^{a}$")]
    ss = d["subset_sizes"]

    def one(x: float | None) -> str:
        return f"{x:.2f}" if x is not None else "--"

    def seeds_pm(rr: dict[str, Any], head: str, sub: str) -> str:
        v = [rr[k][head][sub]["mae"] for k in sorted(rr) if head in rr[k] and sub in rr[k][head]]
        if not v:
            return "--"
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        return f"{m:.2f} $\\pm$ {sd:.2f}"

    rows = [
        f"Constant ({d['constant']['value']:.2f}$^\\circ$)$^{{b}}$ & -- & "
        + " & ".join(one(d["constant"][s]["mae"]) for s, _ in subs)
        + " \\\\",
        "Ground-truth masks$^{c}$ & -- & "
        + " & ".join(
            (f"{d['gt_mask_floor'][s]['mae']:.2f} ({d['gt_mask_floor'][s]['n']})" if s in d["gt_mask_floor"] else "--")
            for s, _ in subs
        )
        + " \\\\",
        "\\midrule",
    ]
    nfail = []
    for k, name in SEG_ORDER:
        r = d["runs"].get(k)
        if not r:
            continue
        rows.append(f"{name} & {len(r)} & " + " & ".join(seeds_pm(r, "mask", s) for s, _ in subs) + " \\\\")
        nfail += [r[x]["mask"][s].get("n_fail", 0) for x in r for s, _ in subs]
    r = d["runs"].get("aopgaunet", {})
    if any("head" in r[x] for x in r):
        rows.append(
            f"AoP-GAU-Net regression head$^{{d}}$ & {len(r)} & "
            + " & ".join(seeds_pm(r, "head", s) for s, _ in subs)
            + " \\\\"
        )
    head = " & ".join(f"{lab} ($n={ss[s]}$)" for s, lab in subs)
    jnu_tv = ss["all469"] - ss["unseen_by_ours"]
    nseeds = [len(d["runs"][k]) for k, _ in SEG_ORDER if d["runs"].get(k)]
    weak_sd = " (with two checkpoints the SD is a weak spread estimate)" if nseeds and min(nseeds) < 3 else ""
    maxfail = max(nfail) if nfail else 0
    fail_txt = f"at most {maxfail} failure{'' if maxfail == 1 else 's'} per run and subset"
    return "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\footnotesize",
            "\\caption{Second external AoP set: the public IUGC 2025 release (Zenodo 17356295) \\cite{iugc2025zenodo}, "
            f"{ss['all469']} frames ($512\\times512$ pixels) of IUGC 2024 videos, of which {ss['labelled300']} carry the official "
            f"AoP and {ss['unlabelled169']} only landmarks. The {ss['labelled300']} labelled frames are "
            f"{ss['source:iugc24_test_labelled']} IUGC 2024 test frames, {ss['source:iugc24_train_labelled_jnu_recording']} "
            f"training frames (JNU-IFM recordings) and {ss['source:iugc24_val_labelled']} validation frames; only the "
            f"{ss['source:iugc24_test_labelled']} test frames belong to the IUGC 2024 test set of main-text Table~5. Reference for every frame: the landmark AoP (vertex at the inferior "
            "PS end). MAE in degrees of the geometric AoP from each network's masks (seed checkpoints of main-text "
            f"Tables~4 and~5, no retraining), mean $\\pm$ SD over the checkpoints in column Seeds{weak_sd}. "
            f"Errors over frames with a valid angle; {fail_txt}. $^{{a}}$Frames not from a woman in our JNU-IFM training or validation split "
            f"(the other {jnu_tv} frames are JNU-IFM recordings of such women). "
            "$^{b}$Mean reference AoP of the JNU-IFM training frames, predicted for every frame. "
            "$^{c}$Challenge geometry on the IUGC 2024 ground-truth masks, available for the labelled frames only "
            "(number of frames in brackets): the error floor of mask-based AoP for this reference. The masks of "
            f"{ss['source:iugc24_train_labelled_jnu_recording']} frames come from the IUGC 2024 training set and those of "
            f"{ss['source:iugc24_val_labelled']} from its validation set; they are used only for this floor, never for training "
            "or model selection. "
            "$^{d}$Direct output of the regression head, not computed from the masks.}",
            "\\label{tab:ext469}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            f"Source & Seeds & {head} \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def build(runs: Path, figs: Path) -> str:
    """Text of supp_results.tex."""
    cls = read_json(runs / "cls" / "summary.json")
    seg = read_json(runs / "seg" / "summary.json")["models"]
    sel = read_json_if_exists(figs / "figure_frames.json") or {}
    parts = [
        "%% generated by fin.reporting.supp_tables (src/fin/reporting/supp_tables.py) -- do not edit by hand",
        youden_table(cls),
        "",
        asplit_table(runs),
        "",
        pooled_table(seg),
        "",
        ext469_table(runs),
        "",
        "\\begin{figure}[h]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{figS-bland-altman}",
        "\\caption{Bland--Altman plots (predicted minus reference AoP, degrees) on the JNU-IFM test split and IUGC 2024 "
        "(seed-0 models): the geometric AoP from the masks of the four segmenters and, in the rightmost column, the "
        "direct estimate of the AoP-GAU-Net regression head; solid line, bias; dashed lines, 95\\% limits of agreement.}",
        "\\label{fig:ba}",
        "\\end{figure}",
        "",
        frames_text(sel),
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULT_RUNS)
    ap.add_argument("--paper-dir", type=Path, default=None, help="manuscript folder (default: $FIN_PAPER)")
    ap.add_argument("--out", type=Path, default=None, help="default: <paper-dir>/sections/tables/supp_results.tex")
    a = ap.parse_args(argv)
    paper = paper_dir(a.paper_dir)
    a.out = a.out or paper / "sections" / "tables" / "supp_results.tex"
    setup_logging()
    write_text(a.out, build(a.runs, paper / "figs"))
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
