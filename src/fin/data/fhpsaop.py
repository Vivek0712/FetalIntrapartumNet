"""Match the JNU-IFM SPHead frames to the FH-PS-AoP (MICCAI 2023) challenge release and attach reference AoPs.

The challenge images 00001-03743 are exactly the 3743 JNU-IFM SPHead frames: challenge image k is the k-th
SPHead frame when the JNU file names ``<video>_<frame>.png`` are sorted as strings (so ``_109`` < ``_19``). The
mapping is verified on every frame by image MAE at 256x256, nearest-neighbour image search and per-class mask
Dice against the remapped challenge label.

Reference AoPs written to ``frames.csv`` (computed on the challenge ground-truth label at 256x256)
    aop_ref_challenge_code   challenge geometry (``fin.geometry.aop_from_mask``); the primary JNU reference
    aop_ref                  independent ellipse--tangent implementation (``fin.geometry.reference_aop``)

    fin match-aop PROCESSED_DIR CHALLENGE_DIR
CHALLENGE_DIR holds ``image_mha/`` and ``label_mha/`` of the Zenodo release (doi 10.5281/zenodo.7851339).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from fin.data.datasets import read_gray, read_mask, resolve_path
from fin.data.labels import CHALLENGE, FH, PS, remap_challenge
from fin.geometry import aop_from_mask, reference_aop
from fin.io import read_frames_csv, setup_logging, write_csv, write_json

log = logging.getLogger(__name__)

N_CHALLENGE_JNU = 3743
MATCH_COLUMNS = ["challenge_id", "sim", "mae256", "dice_fh", "dice_ps", "aop_ref", "aop_ref_challenge_code"]


def read_mha(path: Path) -> np.ndarray:
    """Read one challenge ``.mha`` file as an array; 3-channel images are returned as H x W x 3."""
    import SimpleITK as sitk

    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    return a


def load_challenge(root: Path, n: int = N_CHALLENGE_JNU) -> tuple[np.ndarray, np.ndarray]:
    """First channel of challenge images 1..n and their labels (1 PS, 2 FH), each (n, 256, 256)."""
    img = np.stack([read_mha(root / "image_mha" / f"{i:05d}.mha")[..., 0] for i in range(1, n + 1)])
    lab = np.stack([read_mha(root / "label_mha" / f"{i:05d}.mha") for i in range(1, n + 1)])
    return img, lab


def thumbnail(a: np.ndarray) -> np.ndarray:
    """32x32 zero-mean unit-norm thumbnail for nearest-neighbour image search."""
    t = cv2.resize(a.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def match(df: pd.DataFrame, cimg: np.ndarray, clab: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach challenge ids, match statistics and the two reference AoPs; returns the table and a log."""
    n = cimg.shape[0]
    thumbs = np.stack([thumbnail(cimg[i]) for i in range(n)])
    fn = df.video_id + "_" + df.frame_index.astype(str) + ".png"
    sph = fn[df.category == 6].sort_values().index.to_numpy()
    if len(sph) != n:
        raise ValueError(f"{len(sph)} SPHead frames but {n} challenge images")
    recs = []
    for j, i in enumerate(sph):
        im = read_gray(df.image_path[i])
        r = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        sims = thumbs @ thumbnail(r)
        nn = int(sims.argmax())
        m = read_mask(df.mask_path[i])
        m256 = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
        cl = remap_challenge(clab[j])
        d = []
        for c in (FH, PS):
            a, b = m256 == c, cl == c
            d.append(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        recs.append(
            {
                "idx": i,
                "challenge_id": j + 1,
                "sim": float(sims[j]),
                "nn_is_mapped": int(nn == j),
                "nn_sim": float(sims[nn]),
                "mae256": float(np.abs(r.astype(int) - cimg[j].astype(int)).mean()),
                "dice_fh": d[0],
                "dice_ps": d[1],
                "aop_ref": reference_aop(clab[j] == CHALLENGE["ps"], clab[j] == CHALLENGE["fh"]),
                "aop_ref_challenge_code": aop_from_mask(cl, fh=FH, ps=PS),
            }
        )
    res = pd.DataFrame(recs).set_index("idx")
    ok = (res.mae256 < 8) & (res.dice_fh > 0.99) & (res.dice_ps > 0.99)
    dd = (res.aop_ref - res.aop_ref_challenge_code).dropna()
    info = {
        "n_sphead": len(sph),
        "n_challenge": n,
        "mapping": "lexicographic filename order of JNU SPHead frames",
        "nn_agrees_with_mapping": int(res.nn_is_mapped.sum()),
        "nn_disagree_sim_gap_max": float((res.nn_sim - res.sim).max()),
        "sim_min": float(res.sim.min()),
        "mae256_mean": float(res.mae256.mean()),
        "mae256_max": float(res.mae256.max()),
        "mae256_p99": float(res.mae256.quantile(0.99)),
        "matched_ok": int(ok.sum()),
        "dice_fh_mean": float(res.dice_fh.mean()),
        "dice_fh_min": float(res.dice_fh.min()),
        "dice_ps_mean": float(res.dice_ps.mean()),
        "dice_ps_min": float(res.dice_ps.min()),
        "n_dice_fh_lt_0_9": int((res.dice_fh < 0.9).sum()),
        "n_dice_ps_lt_0_8": int((res.dice_ps < 0.8).sum()),
        "aop_ref_nan": int(res.aop_ref.isna().sum()),
        "aop_ref_challenge_code_nan": int(res.aop_ref_challenge_code.isna().sum()),
        "aop_ref_mean": float(res.aop_ref.mean()),
        "aop_ref_sd": float(res.aop_ref.std()),
        "aop_two_methods_absdiff_mean": float(dd.abs().mean()),
        "aop_two_methods_absdiff_median": float(dd.abs().median()),
        "aop_two_methods_absdiff_p95": float(dd.abs().quantile(0.95)),
        "aop_two_methods_absdiff_max": float(dd.abs().max()),
    }
    df = df.copy()
    for c in MATCH_COLUMNS:
        df[c] = np.nan
        df.loc[res.index, c] = res[c]
    df.loc[res.index[~ok], ["aop_ref", "aop_ref_challenge_code"]] = np.nan
    df["challenge_id"] = df["challenge_id"].astype("Int64")
    df["paper_class"] = 1 - df["binary_label"]
    return df, info


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: write ``frames.csv`` and ``aop_match_log.json``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed", type=Path, help="folder with frames_base.csv (normally $FIN_DATA/jnu/processed)")
    ap.add_argument("challenge", type=Path, help="challenge release folder with image_mha/ and label_mha/")
    a = ap.parse_args(argv)
    setup_logging()
    base = read_frames_csv(a.processed / "frames_base.csv")
    resolved = base.assign(image_path=base.image_path.map(resolve_path), mask_path=base.mask_path.map(resolve_path))
    cimg, clab = load_challenge(a.challenge)
    matched, info = match(resolved, cimg, clab)
    for c in ("image_path", "mask_path"):
        matched[c] = base[c]
    write_csv(a.processed / "frames.csv", matched)
    write_json(a.processed / "aop_match_log.json", info)
    log.info("matched %d/%d SPHead frames", info["matched_ok"], info["n_sphead"])


if __name__ == "__main__":
    main()
