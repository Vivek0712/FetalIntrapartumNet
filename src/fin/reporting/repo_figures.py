"""Figures of the repository README, written to ``docs/assets``.

    pipeline.svg            the three-stage pipeline (hand-drawn diagram, no data)
    cv_macro_f1.svg         classification: patient-grouped cross-validation macro-F1 of every classifier
    seg_dice.svg            segmentation: FH and PS Dice of every segmenter on JNU-IFM test and IUGC 2024
    aop_mae.svg             AoP mean absolute error on JNU-IFM test, IUGC 2024 and the IUGC 2025 unseen frames,
                            with the constant prediction and the ground-truth-mask floor
    example_aop.png         one JNU-IFM test frame: ground-truth and predicted masks with the AoP construction
                            (needs the segmentation cache, ``fin cache seg``)

The charts are pure functions of ``results/runs`` and ``splits/frames.csv`` (``fin.reporting.summary``), so they
change only when the results change. The SVGs carry their own light and dark colours (``prefers-color-scheme``) on a
transparent background, so they read on both GitHub themes; every mark has a ``<title>`` tooltip.

Example frame rule: the segmenter with the lowest mean AoP MAE on JNU-IFM test; from its seed-0 run, the frame whose
absolute AoP error is the median of the frames with a valid angle (ties: lowest frame id).

    fin repo-figures [--runs DIR] [--out DIR] [--only pipeline charts example]
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np

from fin.config import DOCS_ASSETS, RESULT_RUNS, SPLITS, seg_cache_dir
from fin.io import setup_logging, write_text
from fin.reporting import summary
from fin.reporting.names import CLS_PLAIN, SEG_NAME
from fin.reporting.summary import CLS_ORDER, SEG_ORDER, Results, Stat

log = logging.getLogger(__name__)

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
# Reference data-viz palette (categorical slots 1-3, validated light and dark) and ink/grid tokens. The surface is
# GitHub's page colour, used only for the 2px rings that separate overlapping marks.
STYLE = """
svg { --surface: #ffffff; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
      --card: #f6f8fa; --line: #d0d7de; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; }
@media (prefers-color-scheme: dark) {
  svg { --surface: #0d1117; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
        --card: #161b22; --line: #30363d; --s1: #3987e5; --s2: #d95926; --s3: #199e70; }
}
text { font-family: FONT; fill: var(--ink2); font-size: 12px; }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.sub { fill: var(--ink2); font-size: 12px; }
.lab { fill: var(--ink); font-size: 12.5px; }
.val { fill: var(--ink2); font-size: 11.5px; font-variant-numeric: tabular-nums;
       paint-order: stroke; stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.ref { stroke: var(--ink2); stroke-width: 1.5; }
.floor { stroke: var(--muted); stroke-width: 1.5; }
.ring { stroke: var(--surface); stroke-width: 2; }
""".replace("FONT", FONT)


@dataclass
class Svg:
    """Minimal SVG document builder."""

    width: int
    height: int
    label: str
    parts: list[str] = field(default_factory=list)

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, cls: str, anchor: str = "start", extra: str = "") -> None:
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str = "", style: str = "") -> None:
        st = f' style="{style}"' if style else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"{st}/>')

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{escape(self.label)}">'
        )
        return "\n".join(
            [head, f"<title>{escape(self.label)}</title>", f"<style>{STYLE}</style>", *self.parts, "</svg>"]
        )


def nice_range(lo: float, hi: float, step: float) -> tuple[float, float]:
    """Extend [lo, hi] outward to multiples of ``step``."""
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def x_axis(
    svg: Svg, x0: float, x1: float, y0: float, y1: float, lo: float, hi: float, step: float, unit: str = ""
) -> None:
    """Vertical hairline grid with tick labels below the plot."""
    v = lo
    while v <= hi + 1e-9:
        x = x0 + (v - lo) / (hi - lo) * (x1 - x0)
        svg.line(x, y0, x, y1, "grid")
        svg.text(x, y1 + 16, f"{v:g}{unit}", "tick", "middle")
        v += step


def dot(svg: Svg, x: float, y: float, r: float, color: str, tip: str, opacity: float = 1.0) -> None:
    """A marker with a 2px ring in the page colour and a tooltip."""
    svg.add(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="ring" style="fill: var({color}); fill-opacity: {opacity}">'
        f"<title>{escape(tip)}</title></circle>"
    )


def hbar(svg: Svg, x0: float, x1: float, yc: float, h: float, color: str, tip: str) -> None:
    """Horizontal bar from the baseline ``x0`` to ``x1``: square at the baseline, 4px rounded data end."""
    r = min(4.0, (x1 - x0) / 2, h / 2)
    y0, y1 = yc - h / 2, yc + h / 2
    d = (
        f"M{x0:.1f},{y0:.1f} H{x1 - r:.1f} Q{x1:.1f},{y0:.1f} {x1:.1f},{y0 + r:.1f} V{y1 - r:.1f} "
        f"Q{x1:.1f},{y1:.1f} {x1 - r:.1f},{y1:.1f} H{x0:.1f} Z"
    )
    svg.add(f'<path d="{d}" style="fill: var({color})"><title>{escape(tip)}</title></path>')


def pm(s: Stat, nd: int, scale: float = 1.0) -> str:
    """``mean ± SD`` text (the mean alone for a single value)."""
    return f"{s.mean * scale:.{nd}f}" if math.isnan(s.sd) else f"{s.mean * scale:.{nd}f} ± {s.sd * scale:.{nd}f}"


def legend(svg: Svg, x: float, y: float, items: list[tuple[str, str]]) -> None:
    """A one-row legend of coloured dots and names."""
    for color, name in items:
        svg.add(f'<circle cx="{x + 5:.1f}" cy="{y - 4:.1f}" r="5" style="fill: var({color})"/>')
        svg.text(x + 15, y, name, "lab")
        x += 15 + 7.2 * len(name) + 22


# ------------------------------------------------------------------------------------------------ pipeline
def pipeline_svg() -> str:
    """The three-stage pipeline: frame -> standard-plane classifier -> FH/PS segmenter -> AoP geometry."""
    bw, bh, gap, top = 222, 112, 44, 48
    W, H = 2 * 20 + 4 * bw + 3 * gap, 250
    svg = Svg(W, H, "FetalIntrapartumNet pipeline: standard-plane selection, FH/PS segmentation and AoP measurement")
    nodes = [
        ("Ultrasound frame", ["Intrapartum transperineal", "JNU-IFM · IUGC 2024 · IUGC 2025"], None),
        ("1  Standard-plane selection", ["Standard vs non-standard", "DeepFetalNet + 6 baselines"], "--s1"),
        ("2  FH / PS segmentation", ["Fetal head and pubic symphysis", "AoP-GAU-Net + 3 baselines"], "--s2"),
        ("3  Angle of progression", ["Ellipse–tangent geometry", "(FH-PS-AoP challenge code)"], "--s3"),
    ]
    x = 20
    centers = []
    for title, lines, color in nodes:
        svg.add(
            f'<rect x="{x}" y="{top}" width="{bw}" height="{bh}" rx="10" '
            'style="fill: var(--card); stroke: var(--line); stroke-width: 1"/>'
        )
        if color:
            svg.add(f'<rect x="{x}" y="{top}" width="{bw}" height="5" rx="2.5" style="fill: var({color})"/>')
        svg.text(x + 16, top + 34, title, "title", extra=' style="font-size: 14px"')
        for j, ln in enumerate(lines):
            svg.text(x + 16, top + 60 + 20 * j, ln, "sub")
        centers.append(x)
        x += bw + gap
    for i in range(3):
        x0, x1, y = centers[i] + bw + 6, centers[i + 1] - 8, top + bh / 2
        svg.line(x0, y, x1, y, style="stroke: var(--ink2); stroke-width: 2")
        svg.add(
            f'<path d="M{x1 - 8},{y - 5} L{x1},{y} L{x1 - 8},{y + 5}" '
            'style="fill: none; stroke: var(--ink2); stroke-width: 2"/>'
        )
    # rejection branch under stage 1 and the regression head under stage 2
    xs = centers[1] + bw / 2
    svg.line(xs, top + bh + 2, xs, top + bh + 28, style="stroke: var(--muted); stroke-width: 1.5")
    svg.text(xs, top + bh + 46, "non-standard frames are not measured", "sub", "middle")
    xs = centers[2] + bw / 2
    svg.line(xs, top + bh + 2, xs, top + bh + 28, style="stroke: var(--muted); stroke-width: 1.5")
    svg.text(xs, top + bh + 46, "AoP-GAU-Net also regresses the AoP", "sub", "middle")
    svg.text(20, 26, "Pipeline", "title")
    svg.text(110, 26, "each stage is trained and evaluated separately, then chained end to end", "sub")
    return svg.render()


# ------------------------------------------------------------------------------------------------ charts
def cv_chart(r: Results) -> str:
    """Dot plot: macro-F1 of every fold (small dots) and the mean ± SD over folds (large dot and whisker)."""
    models = sorted((m for m in CLS_ORDER if m in r.cv_mf1), key=lambda m: -r.cv_mf1[m].mean)
    vals = [v * 100 for m in models for v in r.cv_mf1[m].values]
    lo, hi = nice_range(min(vals) - 1, max(vals) + 1, 5)
    W, row, left, right, top = 820, 34, 190, 120, 70
    H = top + row * len(models) + 44
    svg = Svg(W, H, f"Cross-validation macro-F1 of the {len(models)} classifiers")
    svg.text(20, 26, "Standard-plane classification", "title")
    svg.text(
        20, 46, f"Macro-F1 (%) in {r.cv_k}-fold patient-grouped cross-validation · dots: folds · bar: mean ± SD", "sub"
    )
    x0, x1 = left, W - right
    y_end = top + row * len(models)
    x_axis(svg, x0, x1, top - 8, y_end, lo, hi, 5)

    def X(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    for i, m in enumerate(models):
        y = top + row * i + row / 2
        s = r.cv_mf1[m]
        name = CLS_PLAIN[m] + (" (proposed)" if m == "deepfetalnet" else "")
        svg.text(left - 12, y + 4, name, "lab", "end", ' style="font-weight: 600"' if m == "deepfetalnet" else "")
        mean, sd = s.mean * 100, s.sd * 100
        if not math.isnan(sd):
            svg.line(
                X(mean - sd), y, X(mean + sd), y, style="stroke: var(--s1); stroke-width: 2; stroke-linecap: round"
            )
        for k, v in enumerate(s.values):
            dot(svg, X(v * 100), y, 4, "--s1", f"{CLS_PLAIN[m]}, fold {k}: {v * 100:.1f}%", 0.45)
        dot(svg, X(mean), y, 6.5, "--s1", f"{CLS_PLAIN[m]}: mean {pm(s, 1, 100)}%")
        svg.text(x1 + 12, y + 4, f"{pm(s, 1, 100)}", "val")
    return svg.render()


def dice_chart(r: Results) -> str:
    """Two panels (JNU-IFM test, IUGC 2024): FH and PS Dice per segmenter, seeds (small) and mean ± SD (large)."""
    models = [m for m in SEG_ORDER if (m, "jnu", "fh") in r.dice]
    sets = [("jnu", "JNU-IFM test"), ("iugc", "IUGC 2024 test")]
    vals = [v * 100 for (m, k, c), s in r.dice.items() for v in s.values]
    lo, hi = nice_range(min(vals) - 1, max(vals) + 1, 5)
    W, row, left, top = 860, 44, 120, 112
    pw, pgap = (W - left - 30 - 40) / 2, 40
    H = top + row * len(models) + 44
    svg = Svg(W, H, "Segmentation Dice of the fetal head and pubic symphysis")
    svg.text(20, 26, "FH / PS segmentation", "title")
    svg.text(20, 46, "Per-image Dice (%) · small dots: seeds · large dot and bar: mean ± SD", "sub")
    legend(svg, 20, 72, [("--s1", "Fetal head (FH)"), ("--s2", "Pubic symphysis (PS)")])
    y_end = top + row * len(models)
    for p, (k, tag) in enumerate(sets):
        x0 = left + p * (pw + pgap)
        x1 = x0 + pw
        svg.text(x0, top - 10, tag, "lab", extra=' style="font-weight: 600"')
        x_axis(svg, x0, x1, top, y_end, lo, hi, 5)

        def X(v: float, x0: float = x0, x1: float = x1) -> float:
            return x0 + (v - lo) / (hi - lo) * (x1 - x0)

        for i, m in enumerate(models):
            yc = top + row * i + row / 2
            if p == 0:
                svg.text(left - 12, yc + 4, SEG_NAME[m], "lab", "end")
            for c, color, dy in (("fh", "--s1", -7), ("ps", "--s2", 7)):
                s = r.dice.get((m, k, c))
                if s is None:
                    continue
                y = yc + dy
                if not math.isnan(s.sd):
                    svg.line(
                        X(100 * (s.mean - s.sd)),
                        y,
                        X(100 * (s.mean + s.sd)),
                        y,
                        style=f"stroke: var({color}); stroke-width: 2; stroke-linecap: round",
                    )
                for j, v in enumerate(s.values):
                    dot(
                        svg,
                        X(v * 100),
                        y,
                        3.5,
                        color,
                        f"{SEG_NAME[m]}, {tag}, {c.upper()}, seed {j}: {v * 100:.1f}%",
                        0.45,
                    )
                dot(svg, X(s.mean * 100), y, 5.5, color, f"{SEG_NAME[m]}, {tag}, {c.upper()}: {pm(s, 1, 100)}%")
    return svg.render()


def aop_chart(r: Results) -> str:
    """Three stacked panels of AoP MAE bars (mean ± SD over seeds) with the constant and ground-truth-mask lines."""
    rows = [(m, SEG_NAME[m], "--s1") for m in SEG_ORDER if (m, "jnu") in r.aop]
    if ("aopgaunet.head", "jnu") in r.aop:
        rows.append(("aopgaunet.head", "AoP-GAU-Net head", "--s2"))
    hi_v = [s.mean + (0 if math.isnan(s.sd) else s.sd) for s in r.aop.values()] + list(r.const.values())
    hi = nice_range(0, max(hi_v) * 1.05, 5)[1]
    W, row, left, right = 860, 26, 160, 110
    head = 48
    panel_h = head + row * len(rows) + 34
    top0 = 84
    H = top0 + panel_h * len(summary.AOP_SETS) + 4
    svg = Svg(W, H, "AoP mean absolute error of every segmenter on three test sets")
    svg.text(20, 26, "Angle of progression", "title")
    svg.text(20, 46, "Mean absolute error (degrees, lower is better) · bar: mean over seeds · whisker: ± SD", "sub")
    legend(svg, 20, 70, [("--s1", "Geometry on predicted masks"), ("--s2", "Direct regression head")])
    x0, x1 = left, W - right

    def X(v: float) -> float:
        return x0 + v / hi * (x1 - x0)

    for p, k in enumerate(summary.AOP_SETS):
        top = top0 + p * panel_h
        tag = summary.AOP_SET_NAME[k] + (f" ({r.n_unseen} frames)" if k == "unseen" else "")
        svg.text(20, top + 20, tag, "lab", extra=' style="font-weight: 600"')
        y0, y1 = top + head, top + head + row * len(rows)
        x_axis(svg, x0, x1, y0, y1, 0, hi, 5, "°")
        # reference lines first, so bars and value labels sit on top of them
        c = r.const.get(k)
        if c is not None:
            svg.line(X(c), y0 - 10, X(c), y1 + 2, "ref")
            svg.text(X(c) + 5, y0 - 12, f"constant prediction {c:.2f}°", "val")
        f = r.floor.get(k)
        if f is not None and f > 0:
            svg.line(X(f), y0 - 10, X(f), y1 + 2, "floor", "stroke-dasharray: 4 3")
            svg.text(X(f) - 5, y0 - 12, f"ground-truth masks {f:.2f}°", "val", "end")
        svg.line(x0, y0, x0, y1, "axis")
        for i, (src, name, color) in enumerate(rows):
            s = r.aop.get((src, k))
            yc = y0 + row * i + row / 2
            svg.text(left - 12, yc + 4, name, "lab", "end")
            if s is None:
                continue
            hbar(svg, x0, X(s.mean), yc, 16, color, f"{name}, {tag}: {pm(s, 2)}°")
            if not math.isnan(s.sd):
                svg.line(X(s.mean - s.sd), yc, X(s.mean + s.sd), yc, style="stroke: var(--ink2); stroke-width: 1.5")
            svg.text(X(s.mean + (0 if math.isnan(s.sd) else s.sd)) + 8, yc + 4, f"{pm(s, 2)}°", "val")
    return svg.render()


# ------------------------------------------------------------------------------------------------ example
def example_png(r: Results, runs: Path, cache: Path, out: Path) -> None:
    """One JNU-IFM test frame (median error of the best segmenter, seed 0): ground truth vs prediction."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from fin.geometry import aop_from_mask, mask_construction
    from fin.reporting.figures import _check_same_frames, draw_case, load_cache

    model = min((m for m in SEG_ORDER if (m, "jnu") in r.aop), key=lambda m: r.aop[(m, "jnu")].mean)
    rd = runs / "seg" / model / "seed0"
    t = pd.read_csv(rd / "per_image_jnu_test.csv")
    lab = np.load(rd / "preds_jnu_test.npz")["labels"]
    d, img, msk, sel = load_cache(cache, "jnu_test")
    _check_same_frames(t, d, f"{rd.name}/per_image_jnu_test.csv")
    err = np.abs(t.aop_mask.to_numpy(float) - t.ref_primary.to_numpy(float))
    ok = np.flatnonzero(np.isfinite(err))
    med = np.median(err[ok])
    i = int(sorted(ok, key=lambda j: (abs(err[j] - med), str(t.frame_id[j])))[0])
    im, gt, pr = np.asarray(img[sel[i]]), np.asarray(msk[sel[i]]), lab[i]
    cg, cp = mask_construction(gt), mask_construction(pr)
    a_gt, a_pr = aop_from_mask(gt, 1, 2), aop_from_mask(pr, 1, 2)
    if cg is None or cp is None or abs(a_pr - t.aop_mask[i]) >= 1e-3 or abs(a_gt - t.ref_primary[i]) >= 1e-3:
        raise RuntimeError(f"example: construction of frame {t.frame_id[i]} does not reproduce the saved angles")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75))
    draw_case(axes[0], im, gt, cg, "lime")
    draw_case(axes[1], im, pr, cp, "red")
    titles = [f"Ground-truth masks · reference {a_gt:.1f}°", f"{SEG_NAME[model]} (seed 0) · {a_pr:.1f}°"]
    for a, tt in zip(axes, titles, strict=True):
        a.set_title(tt, fontsize=10, color="#898781")
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(0, 255)
        a.set_ylim(255, 0)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.03)
    fig.savefig(out / "example_aop.png", dpi=160, transparent=True)
    plt.close(fig)
    log.info("example: %s, frame %s, |error| %.2f°", model, t.frame_id[i], abs(a_pr - a_gt))


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULT_RUNS, help="results folder (default: results/runs)")
    ap.add_argument("--frames", type=Path, default=SPLITS / "frames.csv")
    ap.add_argument("--out", type=Path, default=DOCS_ASSETS, help="default: docs/assets")
    ap.add_argument("--seg-cache", type=Path, default=None, help="default: $FIN_SEG_CACHE")
    ap.add_argument("--only", nargs="*", default=["pipeline", "charts", "example"])
    a = ap.parse_args(argv)
    setup_logging()
    a.out.mkdir(parents=True, exist_ok=True)
    r = summary.load(a.runs, a.frames)
    if "pipeline" in a.only:
        write_text(a.out / "pipeline.svg", pipeline_svg() + "\n")
    if "charts" in a.only:
        write_text(a.out / "cv_macro_f1.svg", cv_chart(r) + "\n")
        write_text(a.out / "seg_dice.svg", dice_chart(r) + "\n")
        write_text(a.out / "aop_mae.svg", aop_chart(r) + "\n")
    if "example" in a.only:
        example_png(r, a.runs, a.seg_cache or seg_cache_dir(), a.out)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
