"""Drawing primitives shared by the method figures (``fin.reporting.method_figures``) and the result figures
(``fin.reporting.figures``): label-map overlays and points on a fitted ellipse.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from fin.geometry import Ellipse

Color = str | tuple[float, float, float]


def ellipse_xy(e: Ellipse, n: int = 400, scale: float = 1.0, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """x and y of ``n`` points on a ``cv2.fitEllipse`` ellipse, mapped by ``v * scale + offset``."""
    (cx, cy), (w, h), angle = e
    tt = np.linspace(0, 2 * np.pi, n)
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    x = w / 2 * np.cos(tt)
    y = h / 2 * np.sin(tt)
    return (cx + ca * x - sa * y) * scale + offset, (cy + sa * x + ca * y) * scale + offset


def overlay(
    ax: plt.Axes,
    img: np.ndarray,
    lab: np.ndarray,
    colors: tuple[Color, Color],
    alpha: float,
    outline: Literal["contour", "trace"] = "contour",
    lw: float = 0.6,
    fill_dtype: type[np.floating] = np.float64,
) -> None:
    """Grayscale image with a semi-transparent fill of FH (label 1) and PS (label 2) and their outlines.

    ``outline="contour"`` draws the 0.5 iso-line of each class mask (``ax.contour``); ``outline="trace"`` draws the
    external pixel contours found by ``cv2.findContours``, closed. ``fill_dtype`` is the precision of the RGBA fill
    layer; it changes the rendered pixels slightly, so each figure keeps the precision it was published with.
    """
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    rgba = np.zeros(lab.shape + (4,), fill_dtype)
    for value, col in zip((1, 2), colors, strict=True):
        sel = lab == value
        rgba[sel, :3] = matplotlib.colors.to_rgb(col)
        rgba[sel, 3] = alpha
    ax.imshow(rgba, interpolation="nearest")
    for value, col in zip((1, 2), colors, strict=True):
        sel = lab == value
        if not sel.any():
            continue
        if outline == "contour":
            ax.contour(sel, levels=[0.5], colors=[col], linewidths=lw)
        else:
            cs, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                c = c[:, 0, :]
                ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, lw=lw)
