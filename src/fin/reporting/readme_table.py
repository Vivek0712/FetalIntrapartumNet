"""Results tables of README.md, generated from the committed run summaries (``results/runs``).

The block between ``<!-- BEGIN RESULTS TABLE -->`` and ``<!-- END RESULTS TABLE -->`` is replaced; no number in it is
typed by hand (sources: ``fin.reporting.summary``). ``--check`` exits non-zero when the README block differs from the
generated one (used by the tests).

    fin readme-table [--runs DIR] [--check]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from fin.config import RESULT_RUNS, ROOT, SPLITS
from fin.io import write_text
from fin.reporting import summary
from fin.reporting.names import CLS_PLAIN, SEG_NAME
from fin.reporting.summary import CLS_ORDER, SEG_ORDER, Results, Stat

BEGIN, END = "<!-- BEGIN RESULTS TABLE -->", "<!-- END RESULTS TABLE -->"


def fmt(s: Stat | float | None, nd: int, scale: float = 1.0) -> str:
    """``mean ± SD`` (or a single value) with ``nd`` decimals; empty when missing."""
    if s is None:
        return ""
    if isinstance(s, Stat):
        if not s.values:
            return ""
        m, sd = s.mean * scale, s.sd * scale
        return f"{m:.{nd}f}" if math.isnan(sd) else f"{m:.{nd}f} ± {sd:.{nd}f}"
    return "" if math.isnan(s) else f"{s * scale:.{nd}f}"


def _bold_best(cells: list[str], stats: list[Stat | None], lower: bool) -> list[str]:
    """Bold the cell with the best mean (lowest when ``lower``)."""
    means = [s.mean if s is not None and s.values else math.nan for s in stats]
    ok = [i for i, v in enumerate(means) if not math.isnan(v)]
    if not ok:
        return cells
    best = (min if lower else max)(ok, key=lambda i: means[i])
    return [f"**{c}**" if i == best else c for i, c in enumerate(cells)]


def build(r: Results) -> str:
    """Markdown of the results block."""
    seeds = ", ".join(str(s) for s in r.seeds_cls)
    lines = [
        BEGIN,
        "",
        f"**Standard-plane classification.** Patient-grouped {r.cv_k}-fold cross-validation over all JNU-IFM women "
        f"(mean ± SD over folds) and the locked test split ({r.test_n} frames; mean ± SD over seeds {seeds}); "
        "threshold 0.5. Best mean in bold.",
        "",
        "| Model | CV macro-F1 (%) | CV AUC | Test accuracy (%) | Test macro-F1 (%) | Test AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    cols = [
        (r.cv_mf1, 1, 100.0),
        (r.cv_auc, 3, 1.0),
        (r.test_acc, 1, 100.0),
        (r.test_mf1, 1, 100.0),
        (r.test_auc, 3, 1.0),
    ]
    table = [[fmt(src.get(m), nd, sc) for m in CLS_ORDER] for src, nd, sc in cols]
    table = [_bold_best(c, [src.get(m) for m in CLS_ORDER], False) for c, (src, _, _) in zip(table, cols, strict=True)]
    for i, m in enumerate(CLS_ORDER):
        lines.append(f"| {CLS_PLAIN[m]} | " + " | ".join(c[i] for c in table) + " |")
    n_seeds = sorted(set(r.seeds_seg.values()))
    seed_txt = (
        f"{n_seeds[0]} seeds"
        if len(n_seeds) == 1
        else "seeds; " + ", ".join(f"{SEG_NAME[m]} {r.seeds_seg[m]}" for m in SEG_ORDER if m in r.seeds_seg)
    )
    lines += [
        "",
        "**Segmentation and angle of progression.** Per-image Dice (%) of the fetal head (FH) and pubic symphysis "
        "(PS), and the mean absolute error (degrees) of the AoP measured from the predicted masks with the FH-PS-AoP "
        f"challenge geometry (mean ± SD over {seed_txt}). IUGC 2025 unseen: the {r.n_unseen} frames of the IUGC 2025 "
        "release whose recordings are in neither our training nor our validation split. Best mean in bold.",
        "",
        "| Model | Dice FH, JNU | Dice PS, JNU | Dice FH, IUGC 2024 | Dice PS, IUGC 2024 | "
        "AoP MAE, JNU | AoP MAE, IUGC 2024 | AoP MAE, IUGC 2025 unseen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dice_cols = [(k, c) for k in ("jnu", "iugc") for c in ("fh", "ps")]
    seg_cells = [[fmt(r.dice.get((m, k, c)), 1, 100.0) for m in SEG_ORDER] for k, c in dice_cols]
    seg_cells = [
        _bold_best(cells, [r.dice.get((m, k, c)) for m in SEG_ORDER], False)
        for cells, (k, c) in zip(seg_cells, dice_cols, strict=True)
    ]
    aop_cells = [[fmt(r.aop.get((m, k)), 2) for m in SEG_ORDER] for k in summary.AOP_SETS]
    aop_cells = [
        _bold_best(cells, [r.aop.get((m, k)) for m in SEG_ORDER], True)
        for cells, k in zip(aop_cells, summary.AOP_SETS, strict=True)
    ]
    for i, m in enumerate(SEG_ORDER):
        lines.append(f"| {SEG_NAME[m]} | " + " | ".join(c[i] for c in seg_cells + aop_cells) + " |")
    head = [fmt(r.aop.get(("aopgaunet.head", k)), 2) for k in summary.AOP_SETS]
    lines.append("| AoP-GAU-Net regression head | | | | | " + " | ".join(head) + " |")
    const = [fmt(r.const.get(k), 2) for k in summary.AOP_SETS]
    lines.append(f"| Constant prediction ({r.const_value:.2f}°, training mean) | | | | | " + " | ".join(const) + " |")
    # on JNU-IFM the reference itself is the construction on the ground-truth masks, so its floor is 0 by definition
    floor = ["0 (reference)" if k == "jnu" else fmt(r.floor.get(k), 2) for k in summary.AOP_SETS]
    lines.append("| Ground-truth masks (floor of the geometry) | | | | | " + " | ".join(floor) + " |")
    lines += [
        "",
        "Generated by `fin readme-table` from `results/runs` and `splits/frames.csv`; definitions in "
        "[docs/RESULTS.md](docs/RESULTS.md).",
        "",
        END,
    ]
    return "\n".join(lines)


def replace_block(readme: str, block: str) -> str:
    """README text with the results block replaced."""
    i, j = readme.index(BEGIN), readme.index(END) + len(END)
    return readme[:i] + block + readme[j:]


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULT_RUNS, help="results folder (default: results/runs)")
    ap.add_argument("--frames", type=Path, default=SPLITS / "frames.csv")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--check", action="store_true", help="fail if the README block is not up to date")
    a = ap.parse_args(argv)
    readme = a.readme.read_text()
    new = replace_block(readme, build(summary.load(a.runs, a.frames)))
    if a.check:
        if new != readme:
            sys.exit("README results table is out of date: run `fin readme-table`")
        return
    write_text(a.readme, new)


if __name__ == "__main__":
    main()
