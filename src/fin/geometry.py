"""Angle of progression (AoP) geometry.

Primary geometry (used for every reported mask-derived AoP)
    The ellipse--tangent construction of the FH-PS-AoP challenge (MICCAI 2023), function ``drawline_AOD`` of
    ``ellipse.py``, taken from the ``maskoffs/PS-FH-MICCAI23`` collection of released challenge-team code at a
    pinned commit. That code carries no licence, so it is not part of this repository: ``fin fetch-third-party``
    downloads it and ``fin.third_party`` applies three documented changes in memory (``math.hypot``/``math.degrees``
    arithmetic, a clipped cosine, all construction points returned). This module is our driver around it: for each
    structure the largest external contour of its own class is fitted with ``cv2.fitEllipse``; the long axis of the
    pubic symphysis (PS) ellipse gives the end points P1 and P2 (P2 is taken as the inferior end); the tangent from
    P2 to the fetal-head (FH) ellipse touches it at T; the AoP is the interior angle P1-P2-T. A failure (missing
    structure, contour under five points, degenerate ellipse, undefined angle) gives NaN instead of an exception
    or 0. ``fin check-geometry`` compares the result with the unchanged challenge function on ground-truth masks.

Landmark AoP
    The angle at PS1 (inferior PS end) between PS1->PS2 and PS1->FH1, used for the IUGC 2025 release.

Secondary JNU-IFM reference (``aop_ref``)
    An independent ellipse--tangent implementation: Halir--Flusser direct least-squares ellipse fit to the
    8-connected boundary pixels, tangent points by mapping the FH ellipse to the unit circle, P2 = the PS
    long-axis end point with the larger column coordinate, and the larger of the two tangent angles.

Coordinates are (column, row) pixels of the mask grid.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi

from fin.third_party import load_module

Ellipse = tuple[tuple[float, float], tuple[float, float], float]


@functools.cache
def _drawline_aod() -> Callable[[Ellipse, Ellipse], dict[str, Any]]:
    """The adapted challenge function (``fin.third_party``, source ``aop_geometry``), loaded once per process."""
    return load_module("aop_geometry").drawline_AOD


def largest_contour_ellipse(mask: np.ndarray) -> Ellipse | None:
    """Fit an ellipse to the largest external contour of a binary mask; None if impossible."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if m.sum() == 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return None
    return cv2.fitEllipse(c)


@dataclass(frozen=True)
class Construction:
    """Every point of the challenge AoP construction (mask-grid coordinates).

    ``x``, ``y`` are P2 and ``q1``, ``q2`` the tangency point T in the frame of the FH ellipse, whose
    semi-axes are ``a`` and ``b``; ``disc`` is the discriminant of the tangent equation (negative when P2
    lies inside the FH ellipse, in which case the published code uses slope 0).
    """

    fh_ellipse: Ellipse
    ps_ellipse: Ellipse
    p1: tuple[float, float]
    p2: tuple[float, float]
    t: tuple[float, float]
    angle: float
    a: float
    b: float
    x: float
    y: float
    q1: float
    q2: float
    disc: float
    ps_long: float
    ps_short: float


def construct(fh_e: Ellipse, ps_e: Ellipse) -> Construction | None:
    """Run the challenge construction on two fitted ellipses; None when the angle is undefined.

    The challenge function raises ZeroDivisionError for a degenerate ellipse or a zero-length side and
    ValueError for a negative square root; both mean that no angle exists.
    """
    try:
        r = _drawline_aod()(fh_e, ps_e)
    except (ValueError, ZeroDivisionError):
        return None
    a, b, x, y = r["a"], r["b"], r["x"], r["y"]
    return Construction(
        fh_ellipse=fh_e,
        ps_ellipse=ps_e,
        p1=r["p1"],
        p2=r["p2"],
        t=r["t"],
        angle=r["angle"],
        a=a,
        b=b,
        x=x,
        y=y,
        q1=r["q1"],
        q2=r["q2"],
        disc=b**2 * x**2 + a**2 * y**2 - a**2 * b**2,
        ps_long=float(r["ps_long"]),
        ps_short=float(r["ps_short"]),
    )


def aop_from_ellipses(fh_e: Ellipse, ps_e: Ellipse) -> float:
    """AoP in degrees from the fitted FH and PS ellipses (NaN when undefined)."""
    con = construct(fh_e, ps_e)
    return float("nan") if con is None else con.angle


def mask_construction(mask: np.ndarray, fh: int = 1, ps: int = 2) -> Construction | None:
    """Fit both structures of a label map and run the construction; None on any failure."""
    fe = largest_contour_ellipse(mask == fh)
    pe = largest_contour_ellipse(mask == ps)
    if fe is None or pe is None:
        return None
    return construct(fe, pe)


def aop_from_mask(mask: np.ndarray, fh: int = 1, ps: int = 2) -> float:
    """AoP in degrees from a label map with FH value ``fh`` and PS value ``ps``; NaN on failure."""
    con = mask_construction(mask, fh, ps)
    return float("nan") if con is None else con.angle


def aop_from_masks(labels: np.ndarray, fh: int = 1, ps: int = 2) -> np.ndarray:
    """AoP of every label map in an (N, H, W) stack."""
    return np.array([aop_from_mask(m, fh, ps) for m in labels], float)


def landmark_aop(ps1: tuple[float, float], ps2: tuple[float, float], fh1: tuple[float, float]) -> float:
    """Landmark AoP: angle at PS1 between PS1->PS2 and PS1->FH1, in degrees."""
    u, w = np.subtract(ps2, ps1), np.subtract(fh1, ps1)
    cos = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


# ------------------------------------------------------------------------------- secondary JNU reference
def _boundary(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner = ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    y, x = np.nonzero(mask & ~inner)
    return x.astype(float), y.astype(float)


def _fit_ellipse_direct(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halir--Flusser direct least-squares fit: (centre, semi-axes a >= b, axis directions as columns)."""
    mx, my = x.mean(), y.mean()
    sc = max(x.std(), y.std(), 1.0)
    xs, ys = (x - mx) / sc, (y - my) / sc
    d1 = np.c_[xs * xs, xs * ys, ys * ys]
    d2 = np.c_[xs, ys, np.ones_like(xs)]
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    t = -np.linalg.solve(s3, s2.T)
    m = s1 + s2 @ t
    c = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0.0]])
    m2 = np.linalg.solve(c, m)
    _, vec = np.linalg.eig(m2)
    vec = np.real(vec)
    cond = 4 * vec[0] * vec[2] - vec[1] ** 2
    a1 = vec[:, cond > 0][:, 0]
    coef = np.r_[a1, t @ a1]
    qa, qb, qc, qd, qe, qf = coef
    m0 = np.array([[qa, qb / 2], [qb / 2, qc]])
    cen = np.linalg.solve(-2 * m0, [qd, qe])
    f0 = qf + 0.5 * (qd * cen[0] + qe * cen[1])
    w, axes_dir = np.linalg.eigh(m0)
    ax = np.sqrt(-f0 / w)
    order = np.argsort(-ax)
    return np.array([cen[0] * sc + mx, cen[1] * sc + my]), ax[order] * sc, axes_dir[:, order]


def _tangent_points(p: np.ndarray, cen: np.ndarray, ax: np.ndarray, axes_dir: np.ndarray) -> list[np.ndarray] | None:
    q = axes_dir.T @ (p - cen) / ax
    r = np.hypot(*q)
    if r <= 1:
        return None
    th = np.arctan2(q[1], q[0])
    da = np.arccos(1 / r)
    return [cen + axes_dir @ (np.array([np.cos(s), np.sin(s)]) * ax) for s in (th + da, th - da)]


def reference_aop(ps_mask: np.ndarray, fh_mask: np.ndarray) -> float:
    """Secondary JNU-IFM reference AoP (degrees) from binary PS and FH masks; NaN when undefined."""
    c, a, v = _fit_ellipse_direct(*_boundary(ps_mask))
    cf, af, vf = _fit_ellipse_direct(*_boundary(fh_mask))
    e1, e2 = c + v[:, 0] * a[0], c - v[:, 0] * a[0]
    p2, p1 = (e1, e2) if e1[0] > e2[0] else (e2, e1)
    tangents = _tangent_points(p2, cf, af, vf)
    if tangents is None:
        return float("nan")
    angles = []
    for t in tangents:
        u, w = p1 - p2, t - p2
        angles.append(np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
    return float(max(angles))
