"""Verify ``fin.geometry`` (the adapted challenge function) against the unchanged ``drawline_AOD``.

The challenge ``ellipse.py`` is fetched at a pinned commit (``fin.third_party``). The adapted function, driven by
``fin.geometry``, and the unchanged function are applied to the ground-truth masks at 256x256 of all JNU-IFM SPHead
frames and of the 300 IUGC 2024 test frames (segmentation cache); the JNU angles are also compared with
``aop_ref_challenge_code`` in ``frames.csv``.

    fin check-geometry [--out FILE]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from fin.config import seg_cache_dir
from fin.data.labels import FH, PS
from fin.geometry import aop_from_mask, largest_contour_ellipse
from fin.io import read_frames_csv, setup_logging, write_json
from fin.third_party import load_module

log = logging.getLogger(__name__)


def challenge_drawline_aod() -> Callable[[tuple, tuple], float]:
    """The challenge function ``drawline_AOD(fh_ellipse, ps_ellipse)``, unchanged."""
    return load_module("challenge_ellipse").drawline_AOD


def challenge_aop(mask: np.ndarray, func: Callable[[tuple, tuple], float] | None = None) -> float:
    """AoP of the unchanged challenge function on one label map (NaN if a structure cannot be fitted)."""
    fe, pe = largest_contour_ellipse(mask == FH), largest_contour_ellipse(mask == PS)
    if fe is None or pe is None:
        return float("nan")
    try:
        return float((func or challenge_drawline_aod())(fe, pe))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def compare(masks: np.ndarray) -> dict[str, Any]:
    """Maximum and mean absolute difference between ``fin.geometry`` and the challenge function."""
    func = challenge_drawline_aod()
    ours = np.array([aop_from_mask(np.asarray(m)) for m in masks])
    theirs = np.array([challenge_aop(np.asarray(m), func) for m in masks])
    d = np.abs(ours - theirs)
    return {
        "n": len(ours),
        "nan_ours": int(np.isnan(ours).sum()),
        "nan_challenge": int(np.isnan(theirs).sum()),
        "max_abs_diff": float(np.nanmax(d)),
        "mean_abs_diff": float(np.nanmean(d)),
        "ours": ours,
    }


def check(cache: Path | None = None) -> dict[str, Any]:
    """Run the comparison on both ground-truth mask sets of the segmentation cache."""
    cache = Path(cache or seg_cache_dir())
    idx = read_frames_csv(cache / "jnu_index.csv")
    sel = np.where(idx.category == 6)[0]
    jnu = compare(np.load(cache / "jnu_mask.npy", mmap_mode="r")[sel])
    ref = idx.aop_ref_challenge_code.to_numpy()[sel]
    jnu["max_abs_diff_vs_aop_ref_challenge_code"] = float(np.nanmax(np.abs(jnu.pop("ours") - ref)))
    iugc = compare(np.load(cache / "iugc_mask.npy", mmap_mode="r"))
    iugc.pop("ours")
    return {"jnu_sphead": jnu, "iugc": iugc}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=None, help="default: $FIN_SEG_CACHE")
    ap.add_argument("--out", type=Path, default=None, help="write the result as JSON")
    a = ap.parse_args(argv)
    setup_logging()
    res = check(a.cache)
    if a.out:
        write_json(a.out, res)
    for k, v in res.items():
        log.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
