"""Number formatting for the paper and the value store behind ``\\res{key}``."""

from __future__ import annotations

from typing import Any

import numpy as np

PM = r"\ensuremath{\pm}"


def fin(x: Any) -> bool:
    """True for a finite real number."""
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x)


def f(x: Any, nd: int = 2, scale: float = 1.0, sign: bool = False) -> str | None:
    """Fixed-point LaTeX number (math minus sign), or None when not finite."""
    if not fin(x):
        return None
    s = f"{x * scale:+.{nd}f}" if sign else f"{x * scale:.{nd}f}"
    return s.replace("-", "\\ensuremath{-}") if s.startswith("-") or "-" in s else s


def fint(x: Any) -> str | None:
    """Rounded integer; thousands separated by a LaTeX thin comma from 10,000 on."""
    if not fin(x):
        return None
    x = int(round(float(x)))
    return f"{x:,}".replace(",", "{,}") if abs(x) >= 10000 else str(x)


def stats(vals: list[Any]) -> tuple[float | None, float | None, int]:
    """(mean, sample SD or None for one value, count) of the finite values."""
    v = np.array([x for x in vals if fin(x)], float)
    if not len(v):
        return None, None, 0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None), int(len(v))


def pm(vals: list[Any], nd: int = 2, scale: float = 1.0, sign: bool = False) -> str | None:
    """``mean \\pm SD`` (SD omitted for a single value)."""
    m, s, _ = stats(vals)
    if m is None:
        return None
    out = f(m, nd, scale, sign)
    return out + (f" {PM} {f(s, nd, scale)}" if s is not None else "")


def ci_str(lo: Any, hi: Any, nd: int = 2, scale: float = 1.0) -> str | None:
    """``[lo, hi]``."""
    if not (fin(lo) and fin(hi)):
        return None
    return f"[{f(lo, nd, scale)}, {f(hi, nd, scale)}]"


class Store(dict):
    """key -> LaTeX string; None values are dropped.

    ``raw`` keeps the unformatted per-seed values of every metric stored with ``ms``, so registered keys that ask for
    another unit or precision can be re-formatted.
    """

    def __init__(self) -> None:
        super().__init__()
        self.raw: dict[str, tuple[list[float], int, float, bool]] = {}

    def put(self, key: str, val: Any) -> None:
        """Store ``str(val)`` unless it is None."""
        if val is not None:
            self[key] = str(val)

    def ms(self, base: str, vals: list[Any], nd: int = 2, scale: float = 1.0, sign: bool = False) -> float | None:
        """Store ``base`` (mean +/- SD), ``base.mean`` and ``base.sd``; return the mean."""
        vals = [float(x) for x in vals if fin(x)]
        self.raw[base] = (vals, nd, scale, sign)
        m, s, _ = stats(vals)
        self.put(base, pm(vals, nd, scale, sign))
        self.put(base + ".mean", f(m, nd, scale, sign))
        self.put(base + ".sd", f(s, nd, scale))
        return m


def fmt_raw(
    raw: tuple[list[float], int, float, bool], want: str | None, nd_hint: int | None, pct_hint: bool
) -> str | None:
    """Re-format stored per-seed values for a requested part (mean/sd/both), precision and unit."""
    vals, nd, scale, sign = raw
    if nd_hint is not None:
        nd = nd_hint
    if pct_hint and scale == 1 and vals and max(abs(v) for v in vals) <= 1.0:
        scale = 100
    m, s, _ = stats(vals)
    if want == "sd":
        return f(s, nd, scale)
    if want == "mean":
        return f(m, nd, scale, sign)
    return pm(vals, nd, scale, sign)
