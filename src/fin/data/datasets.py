"""Frame tables and PyTorch datasets.

All tasks use one locked patient-disjoint split of JNU-IFM (column ``split`` of ``frames.csv``: train, val,
test). The IUGC 2024 test set (300 frames) is the external set, split name ``iugc``.

Normalisation
    ClassificationDataset: grayscale uint8 -> float [0, 1] -> replicated to 3 channels -> ImageNet mean/std,
    so ImageNet-pretrained baselines and the proposed model see identical inputs.
    SegmentationDataset: grayscale float in [0, 1], one channel, no mean/std normalisation.
    Images are resized directly (no padding) to a square; JNU frames (1295 x 1026) are therefore scaled
    anisotropically.

Augmentation (train=True only; ``augmentation`` block of ``configs/classification.yaml`` and
``configs/segmentation.yaml``): random rotation +-10 degrees, isotropic scale 0.9-1.1, translation +-4% (one
affine shared by image and mask; the mask uses nearest neighbour), brightness +-0.1 and contrast x[0.9, 1.1].
No flips: a horizontal flip moves the PS to the other side and reverses the AoP construction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fin.config import data_dir, iugc_csv, jnu_frames_csv
from fin.data.labels import STANDARD_CATEGORY
from fin.io import read_frames_csv

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
TASKS = ("cls", "seg", "seg_all", "aop")
SPLITS = ("train", "val", "test")


def resolve_path(p: str) -> str:
    """Resolve a stored image or mask path: absolute paths are kept, relative ones are under ``FIN_DATA``."""
    return p if os.path.isabs(p) else str(data_dir() / p)


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("image_path", "mask_path"):
        if col in df:
            df[col] = df[col].map(resolve_path)
    return df


def apply_split_file(df: pd.DataFrame, split_file: str | os.PathLike) -> pd.DataFrame:
    """Replace the ``split`` column from a CSV (frame_id, split) that assigns every frame exactly once."""
    m = pd.read_csv(split_file).set_index("frame_id")["split"]
    if set(m.index) != set(df.frame_id):
        raise ValueError(f"{split_file} must assign every JNU-IFM frame exactly once")
    df = df.copy()
    df["split"] = df.frame_id.map(m)
    return df


def load_frames(split: str, task: str = "cls") -> pd.DataFrame:
    """Return the frame table of one split and task.

    split: ``train`` | ``val`` | ``test`` | ``all`` | ``iugc``.
    task:  ``cls`` all frames; ``seg`` SPHead frames (both structures annotated); ``seg_all`` all frames;
           ``aop`` SPHead frames with a reference AoP. ``iugc`` has no classification labels.
    ``FIN_SPLIT_FILE`` replaces the locked split (sensitivity analyses only).
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if split == "iugc":
        if task == "cls":
            raise ValueError("IUGC external set has only standard (SPHead) frames; no classification labels here")
        df = pd.read_csv(iugc_csv()).rename(columns={"name": "frame_id"})
        df["aop_ref"] = df["aop_official"]
        df["category"], df["category_name"], df["binary_label"], df["paper_class"] = 6, "SPHead", 1, 0
        df["split"] = "iugc"
        return _resolve_columns(df.reset_index(drop=True))
    df = read_frames_csv(jnu_frames_csv())
    alt = os.environ.get("FIN_SPLIT_FILE")
    if alt:
        df = apply_split_file(df, alt)
    if split != "all":
        if split not in SPLITS:
            raise ValueError("split must be train/val/test/all/iugc")
        df = df[df.split == split]
    if task == "seg":
        df = df[df.category == STANDARD_CATEGORY]
    elif task == "aop":
        df = df[(df.category == STANDARD_CATEGORY) & df.aop_ref.notna()]
    return _resolve_columns(df.reset_index(drop=True))


def read_image(path: str | Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """Read an image file with ``cv2.imread``; FileNotFoundError names the file if it is missing or unreadable."""
    im = cv2.imread(str(path), flags)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return im


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as 8-bit grayscale."""
    return read_image(path, cv2.IMREAD_GRAYSCALE)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a label map as stored (no conversion)."""
    return read_image(path, cv2.IMREAD_UNCHANGED)


@dataclass(frozen=True)
class Augmentation:
    """Train-only augmentation parameters (``augmentation`` block of the classification and segmentation configs).

    The defaults are the values of both configs; ``tests/test_config.py`` checks that they agree.
    """

    rotation_deg: float = 10.0
    scale: tuple[float, float] = (0.9, 1.1)
    shift: float = 0.04
    contrast: tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.1

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Augmentation:
        """Build from a config block with the field names as keys."""
        return cls(
            rotation_deg=float(cfg["rotation_deg"]),
            scale=(float(cfg["scale"][0]), float(cfg["scale"][1])),
            shift=float(cfg["shift"]),
            contrast=(float(cfg["contrast"][0]), float(cfg["contrast"][1])),
            brightness=float(cfg["brightness"]),
        )


DEFAULT_AUGMENTATION = Augmentation()


def random_affine(size: int, rng: np.random.Generator, aug: Augmentation = DEFAULT_AUGMENTATION) -> np.ndarray:
    """2x3 affine matrix: rotation +-rotation_deg and isotropic scale about the centre, then a shift."""
    c = (size / 2, size / 2)
    m = cv2.getRotationMatrix2D(c, rng.uniform(-aug.rotation_deg, aug.rotation_deg), rng.uniform(*aug.scale))
    m[:, 2] += rng.uniform(-aug.shift, aug.shift, 2) * size
    return m


def photometric(x: np.ndarray, rng: np.random.Generator, aug: Augmentation = DEFAULT_AUGMENTATION) -> np.ndarray:
    """Contrast about 0.5 and brightness shift, clipped to [0, 1]."""
    x = (x - 0.5) * rng.uniform(*aug.contrast) + 0.5 + rng.uniform(-aug.brightness, aug.brightness)
    return np.clip(x, 0.0, 1.0)


def augment_pair(
    x: np.ndarray,
    m: np.ndarray | None,
    rng: np.random.Generator,
    aug: Augmentation = DEFAULT_AUGMENTATION,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply one random affine to image (bilinear) and mask (nearest), then the photometric change."""
    size = x.shape[-1]
    a = random_affine(size, rng, aug)
    s = (size, size)
    x = cv2.warpAffine(x, a, s, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if m is not None:
        m = cv2.warpAffine(m, a, s, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return photometric(x, rng, aug), m


def to_classifier_input(x: np.ndarray) -> np.ndarray:
    """(..., H, W) float image in [0, 1] -> (..., 3, H, W) ImageNet-normalised float32."""
    x = np.repeat(x[..., None, :, :], 3, -3)
    shape = (3, 1, 1)
    return np.ascontiguousarray((x - IMAGENET_MEAN.reshape(shape)) / IMAGENET_STD.reshape(shape), np.float32)


class ClassificationDataset(Dataset):
    """(image 3xHxW float32, label long); label = ``binary_label`` (1 Standard) unless ``label_col`` is given."""

    def __init__(
        self,
        df: pd.DataFrame,
        size: int = 224,
        train: bool = False,
        label_col: str = "binary_label",
        imagenet_norm: bool = True,
        cache: bool = True,
        seed: int = 42,
        aug: Augmentation = DEFAULT_AUGMENTATION,
    ) -> None:
        self.df, self.size, self.train, self.aug = df.reset_index(drop=True), size, train, aug
        self.labels = self.df[label_col].to_numpy().astype(np.int64)
        self.norm = imagenet_norm
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, np.ndarray] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.df)

    def set_cache(self, images: dict[int, np.ndarray]) -> None:
        """Provide pre-resized uint8 images by row index (see ``fin.data.cache``)."""
        self._cache = images

    def _img(self, i: int) -> np.ndarray:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        im = cv2.resize(read_gray(self.df.image_path[i]), (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self._cache is not None:
            self._cache[i] = im
        return im

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._img(i).astype(np.float32) / 255.0
        if self.train:
            x, _ = augment_pair(x, None, self.rng, self.aug)
        if self.norm:
            out = to_classifier_input(x)
        else:
            out = np.ascontiguousarray(np.repeat(x[None], 3, 0), np.float32)
        return torch.from_numpy(out), torch.tensor(self.labels[i])


class SegmentationDataset(Dataset):
    """(image 1xHxW float32 in [0, 1], mask HxW long with 0 bg / 1 FH / 2 PS)."""

    def __init__(
        self,
        df: pd.DataFrame,
        size: int = 256,
        train: bool = False,
        cache: bool = True,
        seed: int = 42,
        aug: Augmentation = DEFAULT_AUGMENTATION,
    ) -> None:
        self.df, self.size, self.train, self.aug = df.reset_index(drop=True), size, train, aug
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.df)

    def _pair(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        s = (self.size, self.size)
        im = cv2.resize(read_gray(self.df.image_path[i]), s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(read_mask(self.df.mask_path[i]), s, interpolation=cv2.INTER_NEAREST)
        if self._cache is not None:
            self._cache[i] = (im, m)
        return im, m

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        im, m = self._pair(i)
        x = im.astype(np.float32) / 255.0
        if self.train:
            x, m = augment_pair(x, m, self.rng, self.aug)
        return torch.from_numpy(np.ascontiguousarray(x[None], np.float32)), torch.from_numpy(m.astype(np.int64))


class CachedSegmentationDataset(Dataset):
    """Segmentation training set over pre-resized arrays: (image, mask, reference AoP in degrees or NaN)."""

    def __init__(
        self,
        img: np.ndarray,
        msk: np.ndarray,
        aop: np.ndarray,
        train: bool,
        seed: int,
        aug: Augmentation = DEFAULT_AUGMENTATION,
    ) -> None:
        self.img, self.msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)
        self.aop = aop.astype(np.float32)
        self.train, self.rng, self.aug = train, np.random.default_rng(seed), aug

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, m = self.img[i].astype(np.float32) / 255.0, self.msk[i]
        if self.train:
            x, m = augment_pair(x, m, self.rng, self.aug)
        return (
            torch.from_numpy(np.ascontiguousarray(x[None], np.float32)),
            torch.from_numpy(m.astype(np.int64)),
            torch.tensor(self.aop[i]),
        )
