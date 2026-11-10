"""Pre-resized image caches, so training does not decode 1295x1026 PNGs every epoch.

Classification: all 6224 JNU-IFM frames at SIZE x SIZE (grayscale, ``cv2.INTER_AREA``), stored as
``$FIN_DATA/jnu/cache/cls<SIZE>.npz`` (frame_id, img); identical to ``ClassificationDataset`` decoding.

Segmentation: 256 x 256 uint8 images (``INTER_AREA``) and masks (``INTER_NEAREST``) of the JNU-IFM OnlySP,
OnlyHead and SPHead frames of every split and of the 300 IUGC 2024 test frames, stored in ``$FIN_SEG_CACHE``
as ``{jnu,iugc}_{img,mask}.npy`` and ``{jnu,iugc}_index.csv``.

    fin cache cls [--size 224]
    fin cache seg [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fin.config import data_dir, seg_cache_dir
from fin.data.datasets import ClassificationDataset, load_frames, read_gray, read_mask
from fin.io import read_frames_csv, setup_logging, write_csv

log = logging.getLogger(__name__)

SEG_SIZE = 256
SEG_JNU_COLUMNS = [
    "frame_id",
    "video_id",
    "patient_id",
    "split",
    "category",
    "category_name",
    "has_fh",
    "has_ps",
    "aop_ref",
    "aop_ref_challenge_code",
    "image_path",
    "mask_path",
]
SEG_IUGC_COLUMNS = ["frame_id", "aop_official", "aop_mask_challenge_code", "image_path", "mask_path"]


# ------------------------------------------------------------------------------------------------ classification
def _load_resized(args: tuple[str, int]) -> np.ndarray:
    path, size = args
    return cv2.resize(read_gray(path), (size, size), interpolation=cv2.INTER_AREA)


def cls_cache_path(size: int = 224) -> Path:
    """Location of the classification cache for one input size."""
    return data_dir() / "jnu" / "cache" / f"cls{size}.npz"


def build_cls_cache(size: int = 224, workers: int = 8) -> Path:
    """Build the classification cache once; return its path."""
    p = cls_cache_path(size)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    df = load_frames("all", "cls")
    with Pool(workers) as pool:
        imgs = pool.map(_load_resized, [(q, size) for q in df.image_path], chunksize=32)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, frame_id=df.frame_id.to_numpy().astype(str), img=np.stack(imgs))
    os.replace(tmp, p)
    return p


def load_cls_images(frame_ids: np.ndarray, size: int = 224) -> np.ndarray:
    """Cached uint8 images of the given frames, in that order."""
    z = np.load(build_cls_cache(size))
    pos = {f: i for i, f in enumerate(z["frame_id"])}
    return z["img"][[pos[f] for f in np.asarray(frame_ids).astype(str)]]


def attach_cls_cache(ds: ClassificationDataset, size: int = 224) -> ClassificationDataset:
    """Fill a ClassificationDataset's in-memory cache from the npz cache (by frame_id)."""
    z = np.load(build_cls_cache(size))
    pos = {f: i for i, f in enumerate(z["frame_id"])}
    img = z["img"]
    ds.set_cache({i: img[pos[f]] for i, f in enumerate(ds.df.frame_id.astype(str))})
    return ds


# ------------------------------------------------------------------------------------------------ segmentation
def _load_pair(args: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    ip, mp = args
    s = (SEG_SIZE, SEG_SIZE)
    im = cv2.resize(read_gray(ip), s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(read_mask(mp), s, interpolation=cv2.INTER_NEAREST)
    return im, m


def _build_seg_set(df: pd.DataFrame, out: Path, name: str, workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_load_pair, zip(df.image_path, df.mask_path, strict=True), chunksize=32))
    img = np.stack([r[0] for r in res])
    msk = np.stack([r[1] for r in res])
    if not set(np.unique(msk)) <= {0, 1, 2}:
        raise ValueError(f"{name}: unexpected mask values {np.unique(msk)}")
    np.save(out / f"{name}_img.npy", img)
    np.save(out / f"{name}_mask.npy", msk)
    write_csv(out / f"{name}_index.csv", df)
    log.info("%s %s class fractions %s", name, img.shape, np.bincount(msk.ravel(), minlength=3) / msk.size)


def build_seg_cache(out: Path | None = None, workers: int = 12) -> Path:
    """Build the segmentation cache (JNU-IFM OnlySP/OnlyHead/SPHead frames and IUGC 2024 test)."""
    out = Path(out or seg_cache_dir())
    out.mkdir(parents=True, exist_ok=True)
    j = load_frames("all", "seg_all")
    j = j[j.category.isin([4, 5, 6])].reset_index(drop=True)
    _build_seg_set(j[SEG_JNU_COLUMNS], out, "jnu", workers)
    i = load_frames("iugc", "seg")
    _build_seg_set(i[SEG_IUGC_COLUMNS], out, "iugc", workers)
    return out


def load_seg_cache(cache: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """JNU-IFM index, images and masks of the segmentation cache (arrays memory-mapped)."""
    cache = Path(cache or seg_cache_dir())
    idx = read_frames_csv(cache / "jnu_index.csv")
    return idx, np.load(cache / "jnu_img.npy", mmap_mode="r"), np.load(cache / "jnu_mask.npy", mmap_mode="r")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    c = sub.add_parser("cls", help="classification cache")
    c.add_argument("--size", type=int, default=224)
    s = sub.add_parser("seg", help="segmentation cache")
    s.add_argument("--out", type=Path, default=None, help="default: $FIN_SEG_CACHE")
    a = ap.parse_args(argv)
    setup_logging()
    if a.kind == "cls":
        log.info("classification cache: %s", build_cls_cache(a.size))
        return
    out = Path(a.out or seg_cache_dir())
    if (out / "jnu_img.npy").exists() and (out / "iugc_img.npy").exists():
        log.info("segmentation cache exists: %s", out)
        return
    log.info("segmentation cache: %s", build_seg_cache(out))


if __name__ == "__main__":
    main()
