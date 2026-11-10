"""Figures 1-5 of the paper (method figures), drawn from the dataset files and the geometry and corruption code.

    Fig 1  fig1_quality      a test frame, clean and under the nine corruption conditions (``fin.robustness``)
    Fig 2  fig2_categories   one test frame per JNU-IFM category with its masks
    Fig 3  fig3_framework    pipeline diagram (gate, segmenter, geometry) with real frames
    Fig 4  fig4_aopgaunet    AoP-GAU-Net architecture diagram
    Fig 5  fig5_aop_geometry the AoP construction on a test frame's ground-truth masks (``fin.geometry``)
Output: <paper-dir>/figs/fig{N}_*.pdf and .png (300 dpi) and figs_1-5_log.json (frame ids and values).

Rules: every ultrasound panel is a JNU-IFM frame read from disk and resampled only with ``cv2.INTER_AREA``
(down-sizing) or nearest neighbour; no enhancement, denoising, sharpening or generative tool. Frames are picked by
a stated rule from the locked test split (``pick_frames``).

    fin method-figures [1 2 3 4 5] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle

from fin.config import paper_dir
from fin.data.datasets import load_frames, read_gray, read_mask
from fin.geometry import mask_construction
from fin.io import read_json_if_exists, setup_logging, write_json
from fin.reporting.draw import ellipse_xy, overlay
from fin.robustness import corruptions

log = logging.getLogger(__name__)
OUT = Path("figs")  # replaced by make(); the manuscript figure folder

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
    }
)
LOG = {}
FH_C = "#1fbf3f"  # fetal head
PS_C = "#e8302a"  # pubic symphysis
T_C = "#2f7bff"  # tangency point
ARC_C = "#f2b705"  # angle


# ----------------------------------------------------------------------------------------------- data
def load_df() -> pd.DataFrame:
    """JNU-IFM frame table with resolved image and mask paths."""
    return load_frames("all", "seg_all")


def read_label_map(p: str) -> np.ndarray:
    """Processed label map from disk; raises ValueError unless its values are 0 (background), 1 (FH), 2 (PS)."""
    m = read_mask(p)
    if not set(np.unique(m).tolist()) <= {0, 1, 2}:
        raise ValueError(f"{p}: label values {sorted(np.unique(m).tolist())} outside {{0, 1, 2}}")
    return m


def pick_frames(df: pd.DataFrame) -> dict[str, pd.Series]:
    """One test-split frame per JNU-IFM category, by a fixed rule.

    Only frames whose mask content agrees with the label are eligible (has_fh/has_ps).
      SPHead  : AoP reference (challenge code) closest to the test-split SPHead median; ties -> frame_id.
      OnlyHead: FH area closest to the median FH area of test OnlyHead frames.
      OnlySP  : PS area closest to the median PS area of test OnlySP frames.
      None    : mean grey level closest to the median of test None frames.
    """
    t = df[df.split == "test"]
    want = {"None": (0, 0), "OnlySP": (0, 1), "OnlyHead": (1, 0), "SPHead": (1, 1)}
    picks = {}
    for cat, (fh, ps) in want.items():
        c = t[(t.category_name == cat) & (t.has_fh == fh) & (t.has_ps == ps)].copy()
        if cat == "SPHead":
            key = c.aop_ref_challenge_code
        elif cat == "OnlyHead":
            key = c.fh_px.astype(float)
        elif cat == "OnlySP":
            key = c.ps_px.astype(float)
        else:
            key = c.image_path.map(lambda p: float(read_gray(p).mean()))
        c["d"] = (key - key.median()).abs()
        r = c.sort_values(["d", "frame_id"]).iloc[0]
        picks[cat] = r
    LOG["picks"] = {
        k: {"frame_id": v.frame_id, "video_id": v.video_id, "patient_id": int(v.patient_id), "split": v.split}
        for k, v in picks.items()
    }
    LOG["picks"]["SPHead"]["aop_ref_challenge_code"] = float(picks["SPHead"].aop_ref_challenge_code)
    return picks


def save(fig: plt.Figure, name: str) -> None:
    """Save as PDF and PNG (300 dpi) into the output folder."""
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    log.info("wrote %s", name)


def scale_bar(
    ax: plt.Axes, x0: float, y0: float, px: float, label: str, color: str = "white", lw: float = 1.2, fs: float = 5.5
) -> None:
    """Horizontal scale bar with a label above it."""
    ax.plot([x0, x0 + px], [y0, y0], color=color, lw=lw, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - 12, label, color=color, ha="center", va="bottom", fontsize=fs)


# ----------------------------------------------------------------------------------------------- Fig 1
def fig1(picks: dict[str, pd.Series]) -> None:
    """Clean frame (left) and the nine Table 3 conditions, generated by robustness/corruptions.py."""
    r = picks["SPHead"]
    x = cv2.resize(read_gray(r.image_path), (224, 224), interpolation=cv2.INTER_AREA)
    W, H = 436.5 / 72, 4.25
    fig = plt.figure(figsize=(W, H))
    pw_in, gap_in = 1.22, 0.24
    x0_in = (W - 4 * pw_in - 3 * gap_in) / 2
    rows = [("speckle", "Speckle noise"), ("motion", "Motion blur"), ("gaussian", "Gaussian noise")]
    used = []

    def panel(col: int, row_y: float, arr: np.ndarray, label: str) -> None:
        ax = fig.add_axes([(x0_in + col * (pw_in + gap_in)) / W, row_y / H, pw_in / W, pw_in / H])
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.text(0.5, -0.035, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.8)

    ys = [H - 0.08 - pw_in - i * (pw_in + 0.18) for i in range(3)]
    panel(0, ys[1], corruptions.corrupt(x, "clean", r.frame_id), "(a) Clean")
    used.append("clean")
    letter = 1
    for i, (kind, name) in enumerate(rows):
        conds = [c for c in corruptions.conditions() if c[1] == kind]
        for j, (cid, _, p, _label) in enumerate(conds):
            unit = "variance" if kind != "motion" else "kernel"
            val = f"{p:.2f}" if kind != "motion" else f"{int(p)} px"
            panel(j + 1, ys[i], corruptions.corrupt(x, cid, r.frame_id), f"({chr(97 + letter)}) {name}, {unit} {val}")
            used.append(cid)
            letter += 1
    LOG["fig1"] = {
        "frame_id": r.frame_id,
        "size": "224x224 classifier input (cv2.INTER_AREA)",
        "generator": "fin.robustness.corruptions.corrupt",
        "base_seed": corruptions.base_seed(),
        "conditions": used,
    }
    save(fig, "fig1_quality")


# ----------------------------------------------------------------------------------------------- Fig 2
def fig2(picks: dict[str, pd.Series]) -> None:
    """Figure 2: one test frame per JNU-IFM category with its masks."""
    W = 406.8 / 72
    fig = plt.figure(figsize=(W, 1.85))
    order = [("None", "Non-Standard"), ("OnlySP", "Non-Standard"), ("OnlyHead", "Non-Standard"), ("SPHead", "Standard")]
    pw = 0.235
    gx = (1 - 4 * pw) / 5
    for i, (cat, cls) in enumerate(order):
        r = picks[cat]
        img, m = read_gray(r.image_path), read_label_map(r.mask_path)
        h, w = img.shape
        ax = fig.add_axes([gx + i * (pw + gx), 0.37, pw, pw * W / 1.85 * h / w])
        overlay(ax, img, m, (FH_C, PS_C), alpha=0.28, outline="trace", lw=0.7, fill_dtype=np.float32)
        scale_bar(ax, w - 60 - 200, h - 45, 200, "200 px")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            -0.07,
            f"({chr(97 + i)}) {cat}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.22,
            f"{cls}\nvideo {r.video_id}, frame {int(r.frame_index)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.6,
            linespacing=1.2,
        )
    # legend
    lx = 0.30
    for lab, col in (("Fetal head (FH) mask", FH_C), ("Pubic symphysis (PS) mask", PS_C)):
        fig.patches.append(
            Rectangle(
                (lx, 0.035),
                0.022,
                0.055,
                transform=fig.transFigure,
                facecolor=matplotlib.colors.to_rgba(col, 0.45),
                edgecolor=col,
                lw=0.7,
            )
        )
        fig.text(lx + 0.03, 0.062, lab, va="center", fontsize=6.5)
        lx += 0.24
    LOG["fig2"] = {k: picks[k].frame_id for k, _ in order}
    save(fig, "fig2_categories")


# ----------------------------------------------------------------------------------------------- geometry
def construction(mask256: np.ndarray) -> dict:
    """Every point of the challenge construction (256 grid) plus two checks: T on the FH ellipse, P2-T tangent."""
    con = mask_construction(mask256)
    if con is None:
        raise ValueError("the selected SPHead frame has no valid AoP construction")
    # checks in the frame of the FH ellipse: T on the ellipse and P2-T orthogonal to its normal
    on = (con.q1 / con.a) ** 2 + (con.q2 / con.b) ** 2
    grad = np.array([con.q1 / con.a**2, con.q2 / con.b**2])
    dirv = np.array([con.q1 - con.x, con.q2 - con.y])
    tang = float(abs(grad @ dirv) / (np.linalg.norm(grad) * np.linalg.norm(dirv)))
    return dict(
        fe=con.fh_ellipse,
        pe=con.ps_ellipse,
        P1=con.p1,
        P2=con.p2,
        T=con.t,
        theta=con.angle,
        on_ellipse=float(on),
        cos_grad_dir=tang,
        v=float(con.disc),
        ps_long=con.ps_long,
        ps_short=con.ps_short,
    )


def draw_geometry(
    ax: plt.Axes,
    img_sq: np.ndarray,
    g: dict,
    s: float,
    lw: float = 1.0,
    ms: float = 5,
    labels: bool = True,
    fs: float = 8,
) -> tuple:
    """Draw on a square image of side 256*s; g from construction() in 256-grid coordinates."""
    off = (s - 1) / 2  # pixel-centre mapping x' = (x + 0.5) s - 0.5
    P = lambda p: (p[0] * s + off, p[1] * s + off)  # noqa: E731
    ax.imshow(img_sq, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for e, col in ((g["fe"], FH_C), (g["pe"], PS_C)):
        x, y = ellipse_xy(e, scale=s, offset=off)
        ax.plot(x, y, color=col, lw=lw)
        ax.plot(*P(e[0]), "+", color=col, ms=ms, mew=0.8)
    P1, P2, T = P(g["P1"]), P(g["P2"]), P(g["T"])
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color=PS_C, lw=lw * 1.6, solid_capstyle="round")
    d = np.array(T) - np.array(P2)
    end = np.array(P2) + d * 1.35
    ax.plot([P2[0], end[0]], [P2[1], end[1]], color="white", lw=lw, ls=(0, (4, 2)))
    # interior angle at P2 between P2->P1 and P2->T
    a1 = math.degrees(math.atan2(P1[1] - P2[1], P1[0] - P2[0]))
    a2 = math.degrees(math.atan2(T[1] - P2[1], T[0] - P2[0]))
    lo, hi = sorted([a1, a2])
    if hi - lo > 180:
        lo, hi = hi, lo + 360
    rad = 0.28 * min(math.hypot(*d), math.hypot(P1[0] - P2[0], P1[1] - P2[1])) * 2
    ax.add_patch(Arc(P2, rad, rad, theta1=lo, theta2=hi, color=ARC_C, lw=lw * 1.4))
    for p, col in ((P1, PS_C), (P2, PS_C), (T, T_C)):
        ax.plot(*p, "o", ms=ms, mfc="white", mec=col, mew=1.2, zorder=6)
    if labels:
        mid = math.radians((lo + hi) / 2)
        ax.text(
            P2[0] + rad * 0.80 * math.cos(mid),
            P2[1] + rad * 0.80 * math.sin(mid),
            rf"$\theta$ = {g['theta']:.1f}$\degree$",
            color=ARC_C,
            fontsize=fs,
            ha="right" if math.cos(mid) < 0 else "left",
            va="top" if math.sin(mid) > 0 else "bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, ec="none"),
        )
        for p, name, dx, dy in ((P1, r"$P_1$", -14, -8), (P2, r"$P_2$", 16, -6), (T, r"$T$", 14, 0)):
            ax.text(
                p[0] + dx * s / 4,
                p[1] + dy * s / 4,
                name,
                color="white",
                fontsize=fs + 1,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.6, ec="none"),
            )
    return P1, P2, T


def fig5(picks: dict[str, pd.Series]) -> dict:
    """Figure 5: the AoP construction on the ground-truth masks of the selected SPHead test frame."""
    r = picks["SPHead"]
    S = 4  # display grid 1024 = 4 x 256 (isotropic, so angles are preserved)
    m256 = cv2.resize(read_label_map(r.mask_path), (256, 256), interpolation=cv2.INTER_NEAREST)
    g = construction(m256)
    if abs(g["theta"] - r.aop_ref_challenge_code) >= 1e-6:
        raise RuntimeError(f"fig5: AoP {g['theta']} != aop_ref_challenge_code {r.aop_ref_challenge_code} of frames.csv")
    img = cv2.resize(read_gray(r.image_path), (256 * S, 256 * S), interpolation=cv2.INTER_AREA)
    W = 426.6 / 72
    fig = plt.figure(figsize=(W, W * 0.66))
    ax = fig.add_axes([0.01, 0.02, 0.62, 0.62 * W / (W * 0.66)])
    # outlines of the mask contours actually fitted (largest external contour)
    for lab, col in ((1, FH_C), (2, PS_C)):
        cs, _ = cv2.findContours((m256 == lab).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) * S + (S - 1) / 2
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, lw=0.6, ls=(0, (1, 1.2)))
    P1, P2, T = draw_geometry(ax, img, g, S, lw=1.0, ms=5.5, fs=8)
    xs = [P1[0], P2[0], T[0]] + list(ellipse_xy(g["fe"], scale=S, offset=(S - 1) / 2)[0])
    ys = [P1[1], P2[1], T[1]] + list(ellipse_xy(g["fe"], scale=S, offset=(S - 1) / 2)[1])
    pad = 70
    x0, x1 = max(0, min(xs) - pad), min(256 * S, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(256 * S, max(ys) + pad)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(cx - side / 2, cx + side / 2)
    ax.set_ylim(cy + side / 2, cy - side / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    sb = 25 * S  # 25 px of the 256 grid
    scale_bar(ax, cx + side / 2 - sb - 30, cy + side / 2 - 30, sb, "25 px (256 grid)", fs=6)
    # legend / key
    lx, ly = 0.66, 0.93
    items = [
        ("line", FH_C, "-", "FH ellipse (cv2.fitEllipse on the largest FH contour)"),
        ("line", PS_C, "-", "PS ellipse; thick segment: its long axis $P_1P_2$"),
        ("line", "0.35", (0, (1, 1.2)), "Mask contours that were fitted (dotted)"),
        ("pt", PS_C, None, "$P_1$, $P_2$: end-points of the PS long axis"),
        ("pt", T_C, None, "$T$: tangency point on the FH ellipse"),
        ("line", "0.35", (0, (4, 2)), "Tangent from $P_2$ through $T$"),
        ("line", ARC_C, "-", r"AoP $\theta$ = angle $P_1P_2T$ (interior)"),
    ]
    for kind, col, ls, txt in items:
        if kind == "line":
            fig.lines.append(
                matplotlib.lines.Line2D([lx, lx + 0.035], [ly, ly], color=col, ls=ls, lw=1.3, transform=fig.transFigure)
            )
        else:
            fig.lines.append(
                matplotlib.lines.Line2D(
                    [lx + 0.0175], [ly], marker="o", ms=5, mfc="white", mec=col, mew=1.2, transform=fig.transFigure
                )
            )
        fig.text(lx + 0.045, ly, txt, va="center", fontsize=6.4, wrap=True)
        ly -= 0.075
    ly -= 0.02
    note = (
        f"Video {r.video_id}, frame {int(r.frame_index)}\n(JNU-IFM test split, SPHead),\n"
        f"ground-truth masks resized to 256$\\times$256\n(nearest); display grid 1024$\\times$1024.\n"
        f"$\\theta$ = {g['theta']:.2f}$\\degree$ (challenge geometry code)."
    )
    fig.text(lx, ly, note, va="top", fontsize=6.4, linespacing=1.35)
    LOG["fig5"] = {
        k: (
            list(map(float, v))
            if isinstance(v, tuple) and len(v) == 2 and not isinstance(v[0], tuple)
            else (str(v) if isinstance(v, tuple) else v)
        )
        for k, v in g.items()
    }
    LOG["fig5"]["frame_id"] = r.frame_id
    save(fig, "fig5_aop_geometry")
    return g


# ----------------------------------------------------------------------------------------------- diagrams
EDGE = "#222222"


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    sub: str | None = None,
    fc: str = "white",
    ec: str = EDGE,
    fs: float = 6.6,
    sfs: float = 5.6,
    bold: bool = True,
    ls: str = "-",
    lw: float = 0.6,
    tc: str = "black",
) -> tuple:
    """Box with lower-left (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025", fc=fc, ec=ec, lw=lw, ls=ls))
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=tc,
        )
        ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=sfs, color=tc, linespacing=1.1)
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=tc,
            linespacing=1.1,
        )
    return (x, y, w, h)


def L(b: tuple) -> tuple:
    """Midpoint of the left edge of a box."""
    return (b[0], b[1] + b[3] / 2)


def R(b: tuple) -> tuple:
    """Midpoint of the right edge of a box."""
    return (b[0] + b[2], b[1] + b[3] / 2)


def T_(b: tuple) -> tuple:
    """Midpoint of the top edge of a box."""
    return (b[0] + b[2] / 2, b[1] + b[3])


def B(b: tuple) -> tuple:
    """Midpoint of the bottom edge of a box."""
    return (b[0] + b[2] / 2, b[1])


def arrow(ax: plt.Axes, p: tuple, q: tuple, lw: float = 0.6, color: str = EDGE, ls: str = "-") -> None:
    """Arrow from p to q."""
    ax.annotate(
        "",
        xy=q,
        xytext=p,
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.35,head_width=0.16", lw=lw, color=color, shrinkA=0, shrinkB=0, ls=ls
        ),
    )


def path(ax: plt.Axes, pts: list, lw: float = 0.6, color: str = EDGE, ls: str = "-", head: bool = True) -> None:
    """Poly-line through pts, with an arrow head on the last segment."""
    xs, ys = zip(*pts, strict=True)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, solid_capstyle="butt")
    if head:
        arrow(ax, pts[-2], pts[-1], lw=lw, color=color)


def op(ax: plt.Axes, c: tuple, sym: str, r: float = 0.065, fs: float = 7) -> tuple:
    """Circled operator symbol at c."""
    ax.add_patch(matplotlib.patches.Circle(c, r, fc="white", ec=EDGE, lw=0.6))
    ax.text(c[0], c[1] - 0.004, sym, ha="center", va="center", fontsize=fs)
    return c


def frame(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    fs: float = 7.6,
    ls: object = (0, (3, 2)),
    ec: str = "#555555",
    title_left: bool = True,
) -> None:
    """Dashed rounded frame with a title."""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc="none", ec=ec, lw=0.6, ls=ls)
    )
    if title:
        if title_left:
            ax.text(x + 0.08, y + h - 0.07, title, ha="left", va="top", fontsize=fs, fontweight="bold")
        else:
            ax.text(
                x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=fs, fontweight="bold", linespacing=1.05
            )


def image(
    ax: plt.Axes,
    arr: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    cmap: str = "gray",
    vmin: float = 0,
    vmax: float = 255,
) -> None:
    """Image placed in data coordinates with a thin border."""
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + w, y, y + h), interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=EDGE, lw=0.4, zorder=4))


def canvas(W: float, H: float) -> tuple:
    """Figure with one axis spanning it, in inch coordinates."""
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


TOP_FC = {"cls": "#eef3fb", "seg": "#eefaf0", "geo": "#fdf4e6", "io": "#f4f4f4"}


def fig3(picks: dict[str, pd.Series], g: dict) -> None:
    """Figure 3: pipeline diagram with real frames of the selected test video."""
    W, H = 486.1 / 72, 9.05
    fig, ax = canvas(W, H)
    df = load_df()
    r = picks["SPHead"]
    # ---------------- upper workflow
    yt, yb = H - 0.05, 5.42
    cols = [
        (0.05, 1.27, "Video input and\nframe extraction", "io"),
        (1.40, 1.72, "Standard-plane classification\n(DeepFetalNet)", "cls"),
        (3.20, 1.72, "FH and PS segmentation\n(AoP-GAU-Net)", "seg"),
        (5.00, 1.70, "Geometric AoP estimation\n(challenge geometry code)", "geo"),
    ]
    for x, w, t, _ in cols:
        frame(ax, x, yb, w, yt - yb, t, fs=7.4, title_left=False, ls="-", ec="#888888")
    bh, step = 0.30, 0.375
    # col 1: three real frames of the same video, then extraction box
    vid = df[df.video_id == r.video_id].sort_values("frame_index")
    idx = np.linspace(0, len(vid) - 1, 4).astype(int)
    tw, th = 0.62, 0.62 * 1026 / 1295
    film_x = 0.13
    ax.add_patch(Rectangle((film_x - 0.05, 6.22), tw + 0.10, 2.16, fc="#111111", ec=EDGE, lw=0.4))
    for k, j in enumerate(idx[:3]):
        im = cv2.resize(read_gray(vid.iloc[j].image_path), (259, 205), interpolation=cv2.INTER_AREA)
        image(ax, im, film_x, 8.30 - th - k * (th + 0.13), tw, th)
    ax.text(film_x + tw / 2, 6.38, r"$\vdots$", color="white", ha="center", va="center", fontsize=9)
    ax.text(film_x + tw / 2, 6.03, "B-mode video\n(input)", ha="center", va="center", fontsize=5.8)
    fx = box(ax, 0.86, 6.95, 0.42, 0.62, "Frame\nextrac-\ntion", fs=5.6)
    arrow(ax, (film_x + tw + 0.05, R(fx)[1]), L(fx))
    ax.text(0.68, 5.55, "8-bit grey frames,\n1295$\\times$1026 px", ha="center", va="center", fontsize=5.6)
    # col 2: DeepFetalNet
    x2, w2 = 1.50, 1.52
    yy = 8.33
    b_in = box(ax, x2, yy, w2, bh, "Input frame", "224$\\times$224, grey replicated to 3 channels", fc=TOP_FC["cls"])
    yy -= step
    b_bb = box(
        ax,
        x2,
        yy,
        w2,
        bh,
        "Lightweight CNN backbone",
        "stem + 4 DS-conv/SE stages; 14$\\times$14$\\times$256",
        fc=TOP_FC["cls"],
    )
    yy -= step
    b_am = box(ax, x2, yy, 0.72, bh, "AMFAM", "local, multi-scale", fc=TOP_FC["cls"])
    b_gc = box(ax, x2 + 0.80, yy, 0.72, bh, "GCB", "global self-attention", fc=TOP_FC["cls"])
    yy -= step
    b_af = box(
        ax, x2, yy, w2, bh, "AFM", "adaptive fusion: $\\beta_L F_{local} + \\beta_G F_{global}$", fc=TOP_FC["cls"]
    )
    yy -= step
    b_ar = box(ax, x2, yy, w2, bh, "ARM", "$F + F\\odot A_c + F\\odot A_s$", fc=TOP_FC["cls"])
    yy -= step
    b_hd = box(
        ax,
        x2,
        yy,
        w2,
        bh,
        "Classification head",
        "GAP, FC 256$\\rightarrow$128$\\rightarrow$2, dropout 0.5",
        fc=TOP_FC["cls"],
    )
    yy -= step
    b_sm = box(ax, x2, yy, w2, bh, "Softmax", "$p$ = P(Standard)", fc=TOP_FC["cls"])
    yy -= step + 0.08
    b_ns = box(ax, x2, yy, 0.72, bh + 0.08, "Non-Standard", "$p < \\tau$; not\nforwarded", fc="white")
    b_st = box(ax, x2 + 0.80, yy, 0.72, bh + 0.08, "Standard", "$p \\geq \\tau$; to\nsegmentation", fc="white")
    path(ax, [R(fx), (1.39, R(fx)[1]), (1.39, L(b_in)[1]), L(b_in)])
    for a_, b_ in ((b_in, b_bb), (b_af, b_ar), (b_ar, b_hd), (b_hd, b_sm)):
        arrow(ax, B(a_), T_(b_))
    for b_ in (b_am, b_gc):
        arrow(ax, (T_(b_)[0], b_bb[1]), T_(b_))
        arrow(ax, B(b_), (B(b_)[0], b_af[1] + bh))
    for b_ in (b_ns, b_st):
        arrow(ax, (T_(b_)[0], b_sm[1]), T_(b_))
    # col 3: AoP-GAU-Net
    x3, w3 = 3.30, 1.52
    yy = 8.33
    s_in = box(ax, x3, yy, w3, bh, "Standard frame", "resized to 256$\\times$256$\\times$1", fc=TOP_FC["seg"])
    yy -= step
    s_en = box(ax, x3, yy, w3, bh, "Encoder E1$-$E4", "DS-conv blocks, max-pool; 32$-$256 ch", fc=TOP_FC["seg"])
    yy -= step
    s_gb = box(
        ax,
        x3,
        yy,
        w3,
        bh,
        "Graph bottleneck",
        "16$\\times$16$\\times$512; $k$-NN graph attention ($k$ = 6)",
        fc=TOP_FC["seg"],
    )
    yy -= step
    s_de = box(ax, x3, yy, w3, bh, "Decoder D1$-$D4", "attention-gated skip connections", fc=TOP_FC["seg"])
    yy -= step
    s_hd = box(ax, x3, yy, w3, bh, "1$\\times$1 conv + softmax", "3 classes", fc=TOP_FC["seg"])
    yy -= step
    s_out = box(ax, x3, yy, w3, bh, "Segmentation mask", "256$\\times$256: BG 0, FH 1, PS 2", fc=TOP_FC["seg"])
    path(ax, [R(b_st), (R(b_st)[0] + 0.09, R(b_st)[1]), (R(b_st)[0] + 0.09, L(s_in)[1]), L(s_in)])
    for a_, b_ in ((s_in, s_en), (s_en, s_gb), (s_gb, s_de), (s_de, s_hd), (s_hd, s_out)):
        arrow(ax, B(a_), T_(b_))
    m256 = cv2.resize(read_label_map(r.mask_path), (256, 256), interpolation=cv2.INTER_NEAREST)
    tsz = 0.42
    for k, (lab, name, col) in enumerate(((0, "BG", "#bbbbbb"), (1, "FH", FH_C), (2, "PS", PS_C))):
        rgb = np.zeros((256, 256, 3), np.float32)
        rgb[m256 == lab] = matplotlib.colors.to_rgb(col)
        xx = x3 + 0.08 + k * (tsz + 0.12)
        image(ax, rgb, xx, 5.78, tsz, tsz, cmap=None, vmin=None, vmax=None)
        ax.text(xx + tsz / 2, 5.66, name, ha="center", va="center", fontsize=6)
    arrow(ax, B(s_out), (B(s_out)[0], 5.78 + tsz))
    # col 4: geometry
    x4, w4 = 5.10, 1.50
    yy = 8.33
    gsteps = [
        ("FH and PS masks", "256$\\times$256 grid"),
        ("Largest external contour", "one per structure"),
        ("Ellipse fitting", "FH and PS (least squares)"),
        ("PS long-axis end-points", "$P_1$, $P_2$"),
        ("Tangent from $P_2$", "to the FH ellipse; point $T$"),
        ("AoP output", "$\\theta = \\angle P_1P_2T$ (degrees)"),
    ]
    prev = None
    gb = []
    for t1, t2 in gsteps:
        b_ = box(ax, x4, yy, w4, bh, t1, t2, fc=TOP_FC["geo"])
        if prev:
            arrow(ax, B(prev), T_(b_))
        prev = b_
        gb.append(b_)
        yy -= step
    path(ax, [R(s_out), (4.94, R(s_out)[1]), (4.94, L(gb[0])[1]), L(gb[0])])
    # inset: the construction on the same frame (as Fig 5)
    S = 2
    img = cv2.resize(read_gray(r.image_path), (512, 512), interpolation=cv2.INTER_AREA)
    iax = fig.add_axes([(x4 + 0.31) / W, 5.50 / H, 0.88 / W, 0.88 / H])
    P1, P2, T = draw_geometry(iax, img, g, S, lw=0.6, ms=2.5, labels=False)
    fe_x, fe_y = ellipse_xy(g["fe"], scale=S, offset=(S - 1) / 2)
    xs, ys = list(fe_x) + [P1[0], P2[0]], list(fe_y) + [P1[1], P2[1]]
    c_x, c_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 30
    iax.set_xlim(c_x - side / 2, c_x + side / 2)
    iax.set_ylim(c_y + side / 2, c_y - side / 2)
    iax.set_xticks([])
    iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_linewidth(0.4)
    iax.text(
        0.03,
        0.04,
        rf"$\theta$ = {g['theta']:.1f}$\degree$",
        transform=iax.transAxes,
        color=ARC_C,
        fontsize=5.5,
        fontweight="bold",
    )
    arrow(ax, B(gb[-1]), (B(gb[-1])[0], 5.50 + 0.88))

    # ---------------- lower panels: DeepFetalNet modules (14x14x256 at 224 input)
    sub = dict(fs=6.2, sfs=5.4)
    # AMFAM
    y0, h0 = 3.72, 1.58
    frame(ax, 0.05, y0, W - 0.10, h0, "Adaptive multi-scale feature aggregation module (AMFAM)")
    fin = box(ax, 0.15, y0 + 0.62, 0.62, 0.36, "Input $F$", "14$\\times$14$\\times$256", **sub)
    brs = []
    for k, d in enumerate((1, 2, 3)):
        yk = y0 + 1.12 - k * 0.30
        b1 = box(
            ax,
            1.00,
            yk,
            1.10,
            0.27,
            f"Scale {k + 1} ($d$ = {d})",
            f"DW 3$\\times$3 (dilation {d}) + PW 1$\\times$1",
            **sub,
        )
        b2 = box(ax, 2.28, yk, 0.66, 0.27, f"$F_{k + 1}$", "14$\\times$14$\\times$256", **sub)
        path(ax, [R(fin), (0.88, R(fin)[1]), (0.88, L(b1)[1]), L(b1)])
        arrow(ax, R(b1), L(b2))
        brs.append(b2)
    agg = box(ax, 3.30, y0 + 0.72, 1.20, 0.40, "Weighted sum", "$\\alpha_1F_1 + \\alpha_2F_2 + \\alpha_3F_3$", **sub)
    for b2 in brs:
        path(ax, [R(b2), (3.12, R(b2)[1]), (3.12, L(agg)[1]), L(agg)])
    fus = box(ax, 4.72, y0 + 0.72, 0.92, 0.40, "Channel fusion", "1$\\times$1 conv 256$\\rightarrow$256", **sub)
    out = box(ax, 5.84, y0 + 0.72, 0.78, 0.40, "$F_{AMFAM}$", "14$\\times$14$\\times$256", **sub)
    arrow(ax, R(agg), L(fus))
    arrow(ax, R(fus), L(out))
    yw = y0 + 0.08
    wbs = [
        ("Concatenate", "$[F_1|F_2|F_3]$ 14$\\times$14$\\times$768"),
        ("GAP", "768"),
        ("FC", "768$\\rightarrow$128"),
        ("ReLU", None),
        ("FC", "128$\\rightarrow$3"),
        ("Softmax", "$\\alpha_1,\\alpha_2,\\alpha_3$"),
    ]
    xw = 1.00
    widths = [1.30, 0.55, 0.70, 0.50, 0.70, 0.82]
    prev = None
    for (t1, t2), ww in zip(wbs, widths, strict=True):
        b_ = box(ax, xw, yw, ww, 0.30, t1, t2, **sub)
        if prev:
            arrow(ax, R(prev), L(b_))
        prev = b_
        xw += ww + 0.16
    path(
        ax,
        [(R(brs[2])[0] + 0.06, R(brs[2])[1]), (R(brs[2])[0] + 0.06, yw + 0.37), (1.65, yw + 0.37), (1.65, yw + 0.30)],
        ls=":",
    )
    path(
        ax,
        [R(prev), (R(prev)[0] + 0.1, R(prev)[1]), (R(prev)[0] + 0.1, y0 + 0.50), (3.90, y0 + 0.50), (3.90, y0 + 0.72)],
        ls=":",
    )

    # GCB
    y1, h1 = 2.78, 0.87
    frame(ax, 0.05, y1, W - 0.10, h1, "Global context branch (GCB): self-attention")
    seq = [
        ("Input $F$", "14$\\times$14$\\times$256", 0.62),
        ("1$\\times$1 conv", "256$\\rightarrow$64", 0.58),
        ("Flatten", "196 tokens $\\times$ 64", 0.72),
        ("Multi-head\nself-attention", "64-d, 4 heads", 0.78),
        ("+", None, 0),
        ("LayerNorm", None, 0.56),
        ("Reshape", "14$\\times$14$\\times$64", 0.66),
        ("1$\\times$1 conv", "64$\\rightarrow$256", 0.58),
        ("$F_{GCB}$", "14$\\times$14$\\times$256", 0.70),
    ]
    xs_ = 0.15
    yb_ = y1 + 0.07
    prev = None
    tok = None
    for t1, t2, ww in seq:
        if t1 == "+":
            c = op(ax, (xs_ + 0.07, yb_ + 0.19), "+")
            arrow(ax, R(prev), (c[0] - 0.065, c[1]))
            prev = (c[0] - 0.065, c[1] - 0.065, 0.13, 0.13)
            path(
                ax,
                [
                    (tok[0] + tok[2] / 2 + 0.2, tok[1] + tok[3]),
                    (tok[0] + tok[2] / 2 + 0.2, y1 + 0.58),
                    (c[0], y1 + 0.58),
                    (c[0], c[1] + 0.065),
                ],
            )
            ax.text((tok[0] + c[0]) / 2 + 0.35, y1 + 0.60, "residual", ha="center", va="bottom", fontsize=5.4)
            xs_ += 0.30
            continue
        if "\n" in t1:
            b_ = box(ax, xs_, yb_, ww, 0.38, t1.replace("\n", " ") if False else t1, None, fs=6.0)
            ax.texts[-1].set_position((xs_ + ww / 2, yb_ + 0.24))
            ax.text(xs_ + ww / 2, yb_ + 0.07, t2, ha="center", va="center", fontsize=5.4)
        else:
            b_ = box(ax, xs_, yb_, ww, 0.38, t1, t2, **sub)
        if prev:
            arrow(ax, R(prev), L(b_))
        if t1 == "Flatten":
            tok = b_
        prev = b_
        xs_ += ww + 0.14

    # AFM
    y2, h2 = 1.40, 1.32
    frame(ax, 0.05, y2, W - 0.10, h2, "Adaptive fusion module (AFM)")
    fl = box(ax, 0.15, y2 + 0.62, 0.80, 0.34, "$F_{local}$ (AMFAM)", "14$\\times$14$\\times$256", **sub)
    fg = box(ax, 0.15, y2 + 0.12, 0.80, 0.34, "$F_{global}$ (GCB)", "14$\\times$14$\\times$256", **sub)
    gl = box(ax, 1.12, y2 + 0.66, 0.50, 0.26, "GAP", "256", **sub)
    gg = box(ax, 1.12, y2 + 0.16, 0.50, 0.26, "GAP", "256", **sub)
    cc = box(ax, 1.80, y2 + 0.40, 0.66, 0.30, "Concatenate", "512", **sub)
    arrow(ax, R(fl), L(gl))
    arrow(ax, R(fg), L(gg))
    path(ax, [R(gl), (1.70, R(gl)[1]), (1.70, L(cc)[1] + 0.05), (1.80, L(cc)[1] + 0.05)])
    path(ax, [R(gg), (1.70, R(gg)[1]), (1.70, L(cc)[1] - 0.05), (1.80, L(cc)[1] - 0.05)])
    prev = cc
    xw = 2.62
    for t1, t2, ww in (
        ("FC", "512$\\rightarrow$128", 0.62),
        ("ReLU", None, 0.46),
        ("FC", "128$\\rightarrow$2", 0.58),
        ("Softmax", "$\\beta_L$, $\\beta_G$", 0.66),
    ):
        b_ = box(ax, xw, y2 + 0.40, ww, 0.30, t1, t2, **sub)
        arrow(ax, R(prev), L(b_))
        prev = b_
        xw += ww + 0.14
    m1 = op(ax, (5.72, y2 + 0.79), "$\\times$")
    m2 = op(ax, (5.72, y2 + 0.29), "$\\times$")
    pl = op(ax, (6.00, y2 + 0.54), "+")
    path(ax, [(R(fl)[0] + 0.05, R(fl)[1]), (R(fl)[0] + 0.05, y2 + 1.06), (5.72, y2 + 1.06), (5.72, m1[1] + 0.065)])
    path(ax, [(R(fg)[0] + 0.05, R(fg)[1]), (R(fg)[0] + 0.05, y2 + 0.04), (5.72, y2 + 0.04), (5.72, m2[1] - 0.065)])
    path(ax, [(R(prev)[0], R(prev)[1] + 0.05), (5.50, R(prev)[1] + 0.05), (5.50, m1[1]), (m1[0] - 0.065, m1[1])])
    path(ax, [(R(prev)[0], R(prev)[1] - 0.05), (5.50, R(prev)[1] - 0.05), (5.50, m2[1]), (m2[0] - 0.065, m2[1])])
    ax.text(5.53, m1[1] + 0.09, "$\\beta_L$", fontsize=5.6)
    ax.text(5.53, m2[1] - 0.14, "$\\beta_G$", fontsize=5.6)
    path(ax, [(m1[0] + 0.065, m1[1]), (pl[0], m1[1]), (pl[0], pl[1] + 0.065)])
    path(ax, [(m2[0] + 0.065, m2[1]), (pl[0], m2[1]), (pl[0], pl[1] - 0.065)])
    fo = box(ax, 6.14, y2 + 0.36, 0.50, 0.36, "$F_{fused}$", "14$\\times$14\n$\\times$256", fs=6.2, sfs=5.0)
    arrow(ax, (pl[0] + 0.065, pl[1]), L(fo))

    # ARM
    y3, h3 = 0.05, 1.28
    frame(ax, 0.05, y3, W - 0.10, h3, "Attention refinement module (ARM): channel and spatial attention")
    fi = box(ax, 0.15, y3 + 0.45, 0.62, 0.36, "Input $F$", "14$\\times$14$\\times$256", **sub)
    ych, ysp = y3 + 0.78, y3 + 0.30
    chan = [
        ("GAP", "256", 0.46),
        ("FC", "256$\\rightarrow$16", 0.56),
        ("ReLU", None, 0.44),
        ("FC", "16$\\rightarrow$256", 0.56),
        ("Sigmoid", None, 0.52),
        ("$A_c$", "1$\\times$1$\\times$256", 0.62),
    ]
    spat = [
        ("Mean and max", "over channels", 0.80),
        ("Concatenate", "14$\\times$14$\\times$2", 0.76),
        ("Conv 7$\\times$7", "2$\\rightarrow$1", 0.58),
        ("Sigmoid", None, 0.52),
        ("$A_s$", "14$\\times$14$\\times$1", 0.62),
    ]
    ends = []
    for row, yr in ((chan, ych), (spat, ysp)):
        xw = 0.98
        prev = None
        for t1, t2, ww in row:
            b_ = box(ax, xw, yr - 0.13, ww, 0.27, t1, t2, **sub)
            if prev:
                arrow(ax, R(prev), L(b_))
            else:
                path(ax, [R(fi), (0.88, R(fi)[1]), (0.88, yr), L(b_)])
            prev = b_
            xw += ww + 0.12
        ends.append(prev)
    mc = op(ax, (5.05, ych), "$\\times$")
    ms_ = op(ax, (5.05, ysp), "$\\times$")
    arrow(ax, R(ends[0]), (mc[0] - 0.065, mc[1]))
    path(ax, [R(ends[1]), (mc[0] - 0.065, ysp)])
    ax.text(5.12, ych + 0.08, "$F\\odot A_c$", fontsize=5.6)
    ax.text(5.12, ysp - 0.15, "$F\\odot A_s$", fontsize=5.6)
    sm = op(ax, (5.55, (ych + ysp) / 2), "+")
    path(ax, [(mc[0] + 0.065, ych), (sm[0], ych), (sm[0], sm[1] + 0.065)])
    path(ax, [(ms_[0] + 0.065, ysp), (sm[0], ysp), (sm[0], sm[1] - 0.065)])
    # F feeds both multiplications and the residual
    path(ax, [(0.46, fi[1]), (0.46, y3 + 0.10), (5.05, y3 + 0.10), (5.05, ms_[1] - 0.065)])
    path(ax, [(0.46, fi[1] + fi[3]), (0.46, y3 + 1.00), (5.05, y3 + 1.00), (5.05, mc[1] + 0.065)])
    path(ax, [(0.46, y3 + 0.10), (5.85, y3 + 0.10), (5.85, sm[1] - 0.14)], head=False)
    rs = op(ax, (5.85, sm[1]), "+")
    arrow(ax, (sm[0] + 0.065, sm[1]), (rs[0] - 0.065, rs[1]))
    arrow(ax, (5.85, sm[1] - 0.14), (5.85, rs[1] - 0.065))
    ax.text(3.3, y3 + 0.085, "residual", fontsize=5.0, va="top")
    fo2 = box(ax, 6.02, sm[1] - 0.25, 0.62, 0.50, "$F_{ARM}$", "$F + F\\odot A_c$\n$+ F\\odot A_s$", fs=6.2, sfs=5.0)
    arrow(ax, (rs[0] + 0.065, rs[1]), L(fo2))
    save(fig, "fig3_framework")


def fig4(picks: dict[str, pd.Series], g: dict) -> None:
    """Figure 4: AoP-GAU-Net architecture diagram."""
    W, H = 446.5 / 72, 5.25
    fig, ax = canvas(W, H)
    r = picks["SPHead"]
    sub = dict(fs=6.2, sfs=5.2)
    # main frame
    ym = 2.05
    frame(ax, 0.75, ym, 4.13, H - ym - 0.05, "")
    ax.text(
        0.75 + 4.13 / 2,
        H - 0.13,
        "AoP-GAU-Net (input 256$\\times$256$\\times$1, output 3 classes)",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
    )
    img = cv2.resize(read_gray(r.image_path), (256, 256), interpolation=cv2.INTER_AREA)
    image(ax, img, 0.04, 4.31, 0.62, 0.62)
    ax.text(0.35, 4.23, "Standard frame\n256$\\times$256$\\times$1", ha="center", va="top", fontsize=5.4)
    ex, ew, bh = 0.85, 1.30, 0.40
    dx, dw = 3.30, 1.42
    agx, agw = 2.52, 0.50
    lv = [(256, 32), (128, 64), (64, 128), (32, 256)]
    ys = [4.42, 3.84, 3.26, 2.68]
    enc, dec, ags = [], [], []
    for i, ((s, c), y) in enumerate(zip(lv, ys, strict=True)):
        cin = 1 if i == 0 else lv[i - 1][1]
        e = box(
            ax,
            ex,
            y,
            ew,
            bh,
            f"E{i + 1}: {s}$\\times${s}$\\times${c}",
            f"2$\\times$(DS-conv 3$\\times$3+BN+ReLU), {cin}$\\rightarrow${c}\n"
            f"max-pool $\\rightarrow$ {s // 2}$\\times${s // 2}",
            fc=TOP_FC["seg"],
            **sub,
        )
        enc.append(e)
        j = 4 - i  # decoder index whose skip this is
        cprev = 512 if i == 3 else lv[i + 1][1]
        d = box(
            ax,
            dx,
            y,
            dw,
            bh,
            f"D{j}: {s}$\\times${s}$\\times${c}",
            f"$g$ = up$\\times$2 + conv 3$\\times$3 ({cprev}$\\rightarrow${c});\n"
            f"AG, concat, 2$\\times$DS-conv ({2 * c}$\\rightarrow${c})",
            fc=TOP_FC["seg"],
            **sub,
        )
        dec.append(d)
        a = box(ax, agx, y + 0.04, agw, bh - 0.08, f"AG{j}", None, fs=6.4)
        ags.append(a)
        arrow(ax, (R(e)[0], a[1] + 0.22), (agx, a[1] + 0.22))
        ax.text((R(e)[0] + agx) / 2, a[1] + 0.24, "$x$", ha="center", va="bottom", fontsize=6)
        arrow(ax, (agx + agw, a[1] + 0.22), (dx, a[1] + 0.22))
        ax.text((agx + agw + dx) / 2, a[1] + 0.24, "$\\hat{x}$", ha="center", va="bottom", fontsize=6)
        path(ax, [(dx, a[1] + 0.09), (agx + agw, a[1] + 0.09)], ls=(0, (2, 1.5)))
        ax.text((agx + agw + dx) / 2, a[1] + 0.07, "$g$", ha="center", va="top", fontsize=6)
    arrow(ax, (0.66, L(enc[0])[1]), L(enc[0]))
    for i in range(3):
        arrow(ax, B(enc[i]), T_(enc[i + 1]))
        arrow(ax, T_(dec[i + 1]), B(dec[i]))
    bc = box(
        ax,
        ex,
        2.13,
        ew,
        0.40,
        "Bottleneck conv: 16$\\times$16$\\times$512",
        "2$\\times$(DS-conv + BN + ReLU), 256$\\rightarrow$512",
        fc=TOP_FC["seg"],
        **sub,
    )
    gb = box(
        ax,
        2.40,
        2.13,
        1.30,
        0.40,
        "Graph bottleneck",
        "16$\\times$16$\\times$512 (256 nodes)",
        fc="#e3f1e6",
        ls=(0, (3, 2)),
        **sub,
    )
    arrow(ax, B(enc[3]), T_(bc))
    arrow(ax, R(bc), L(gb))
    path(ax, [R(gb), (dx + dw / 2, R(gb)[1]), B(dec[3])])
    # heads
    hx = 5.00
    sh = box(
        ax,
        hx,
        4.42,
        1.13,
        0.40,
        "Segmentation head",
        "1$\\times$1 conv 32$\\rightarrow$3, softmax",
        fc=TOP_FC["seg"],
        **sub,
    )
    arrow(ax, R(dec[0]), L(sh))
    m256 = cv2.resize(read_label_map(r.mask_path), (256, 256), interpolation=cv2.INTER_NEAREST)
    rgb = np.zeros((256, 256, 3), np.float32)
    rgb[m256 == 1] = matplotlib.colors.to_rgb(FH_C)
    rgb[m256 == 2] = matplotlib.colors.to_rgb(PS_C)
    image(ax, rgb, hx + 0.30, 3.68, 0.55, 0.55, cmap=None, vmin=None, vmax=None)
    arrow(ax, B(sh), (hx + 0.575, 4.23))
    geo = box(
        ax,
        hx,
        2.92,
        1.13,
        0.58,
        "Geometric AoP",
        "largest contours, ellipse fits,\n$P_1$, $P_2$, tangent point $T$\n(Fig. 5)",
        fc=TOP_FC["geo"],
        **sub,
    )
    geo_t = ax.texts[-2]
    geo_t.set_position((hx + 0.565, 2.92 + 0.47))
    ax.texts[-1].set_position((hx + 0.565, 2.92 + 0.21))
    arrow(ax, (hx + 0.575, 3.68), T_(geo))
    ao = box(ax, hx, 2.36, 1.13, 0.36, "AoP (geometry)", "$\\theta$ in degrees", fc=TOP_FC["geo"], **sub)
    arrow(ax, B(geo), T_(ao))
    # auxiliary regression head (in the code, trained with the combined loss)
    ah = box(
        ax,
        hx,
        1.62,
        1.13,
        0.40,
        "AoP head (auxiliary)",
        "attention pooling, FC 32$\\rightarrow$64$\\rightarrow$1,\n180$\\cdot\\sigma(\\cdot)$; direct AoP",
        fc="#f4f4f4",
        ls=(0, (3, 2)),
        **sub,
    )
    path(ax, [(R(dec[0])[0] + 0.09, R(dec[0])[1]), (R(dec[0])[0] + 0.09, 1.82), L(ah)], ls=(0, (3, 2)))

    # detail panels
    yA, hA = 0.98, 0.97
    frame(
        ax,
        0.05,
        yA,
        4.68,
        hA,
        "Attention gate (AG): $\\hat{x} = x\\odot\\sigma(\\psi(\\mathrm{ReLU}(W_x x + W_g g)))$",
        fs=6.8,
    )
    bx = box(ax, 0.15, yA + 0.42, 0.88, 0.28, "Skip feature $x$", "$H\\times W\\times C_x$", **sub)
    bg = box(ax, 0.15, yA + 0.07, 0.88, 0.28, "Gating feature $g$", "decoder, after up-sampling", **sub)
    wx = box(ax, 1.17, yA + 0.42, 0.78, 0.28, "1$\\times$1 conv $W_x$", "+ BN, $C_x$/2 ch", **sub)
    wg = box(ax, 1.17, yA + 0.07, 0.78, 0.28, "1$\\times$1 conv $W_g$", "+ BN, $C_x$/2 ch", **sub)
    arrow(ax, R(bx), L(wx))
    arrow(ax, R(bg), L(wg))
    pc = op(ax, (2.13, yA + 0.385), "+", r=0.06)
    path(ax, [R(wx), (pc[0], R(wx)[1]), (pc[0], pc[1] + 0.06)])
    path(ax, [R(wg), (pc[0], R(wg)[1]), (pc[0], pc[1] - 0.06)])
    prev = (pc[0] - 0.06, pc[1] - 0.06, 0.12, 0.12)
    xw = 2.31
    for t1, t2, ww in (("ReLU", None, 0.40), ("1$\\times$1 conv $\\psi$", "+ BN, 1 ch", 0.66), ("Sigmoid", None, 0.50)):
        b_ = box(ax, xw, yA + 0.25, ww, 0.28, t1, t2, **sub)
        arrow(ax, R(prev), L(b_))
        prev = b_
        xw += ww + 0.12
    mm = op(ax, (xw + 0.02, yA + 0.39), "$\\times$", r=0.06)
    arrow(ax, R(prev), (mm[0] - 0.06, mm[1]))
    path(ax, [(0.59, bx[1] + bx[3]), (0.59, yA + 0.79), (mm[0], yA + 0.79), (mm[0], mm[1] + 0.06)])
    ax.text(mm[0] + 0.14, mm[1], "$\\hat{x}$", fontsize=7, va="center")
    arrow(ax, (mm[0] + 0.06, mm[1]), (mm[0] + 0.13, mm[1]))

    yG, hG = 0.03, 0.90
    frame(ax, 0.05, yG, W - 0.10, hG, "Graph bottleneck (16$\\times$16$\\times$512; 256 nodes)", fs=6.8)
    steps = [
        ("Noise-aware\nattention", "$\\sigma$(DW 3$\\times$3)$\\odot\\sigma$(1$\\times$1)$\\odot h$", 1.05),
        ("Dynamic graph", "1$\\times$1 512$\\rightarrow$128, cosine\naffinity, $x + \\lambda\\,$msg, BN", 1.13),
        (
            "Sparse graph attention",
            "2-D coords, linear$\\rightarrow$128,\n$k$-NN ($k$ = 6), 4 heads $\\times$ 32,\n"
            "linear$\\rightarrow$512, residual + LN",
            1.48,
        ),
        ("Feed-forward", "2$\\times$(conv 3$\\times$3 + BN\n+ ReLU), residual, BN", 1.10),
    ]
    xw = 0.15
    prev = None
    for t1, t2, ww in steps:
        b_ = box(ax, xw, yG + 0.07, ww, 0.60, "", None, fc="#e3f1e6")
        ax.text(
            xw + ww / 2, yG + 0.55, t1.replace("\n", " "), ha="center", va="center", fontsize=6.2, fontweight="bold"
        )
        ax.text(xw + ww / 2, yG + 0.30, t2, ha="center", va="center", fontsize=5.2, linespacing=1.15)
        if prev:
            arrow(ax, R(prev), L(b_))
        prev = b_
        xw += ww + 0.20
    LOG["fig4_frame"] = r.frame_id
    save(fig, "fig4_aopgaunet")


# ----------------------------------------------------------------------------------------------- main
def make(which: list[int], out: Path) -> None:
    """Draw the selected figures into ``out`` and update its figs_1-5_log.json."""
    global OUT
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_df()
    picks = pick_frames(df)
    r = picks["SPHead"]
    m256 = cv2.resize(read_label_map(r.mask_path), (256, 256), interpolation=cv2.INTER_NEAREST)
    g = construction(m256)
    if 1 in which:
        fig1(picks)
    if 2 in which:
        fig2(picks)
    if 3 in which:
        fig3(picks, g)
    if 4 in which:
        fig4(picks, g)
    if 5 in which:
        fig5(picks)
    LOG["fig5_construction"] = {
        k: (
            list(map(float, v))
            if isinstance(v, tuple) and len(v) == 2 and not isinstance(v[0], tuple)
            else str(v)
            if isinstance(v, tuple)
            else v
        )
        for k, v in g.items()
    }
    LOG.pop("fig5", None)
    lp = OUT / "figs_1-5_log.json"
    prev = read_json_if_exists(lp) or {}
    prev.update(LOG)  # keep entries of figures not regenerated in this run
    write_json(lp, prev, indent=2, default=str)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", type=int, choices=[1, 2, 3, 4, 5], help="default: all")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <paper-dir>/figs)")
    ap.add_argument("--paper-dir", type=Path, default=None, help="manuscript folder (default: $FIN_PAPER)")
    a = ap.parse_args(argv)
    out = a.out or paper_dir(a.paper_dir) / "figs"
    setup_logging()
    make(a.figures or [1, 2, 3, 4, 5], out)


if __name__ == "__main__":
    main()
