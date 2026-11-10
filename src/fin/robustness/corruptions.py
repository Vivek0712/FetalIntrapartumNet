"""Deterministic, parameterised image corruptions for the classification robustness study.

Conditions are listed in ``configs/robustness.yaml`` (speckle noise variance 0.05/0.10/0.20, motion blur kernel
5/9/15, Gaussian noise variance 0.01/0.05/0.10, plus the clean condition).

    input     x: grayscale image in [0, 1] at the model input size (224 x 224; the same uint8 frame the clean
              evaluation uses)
    speckle   y = x + x * n, n ~ N(0, var)   (multiplicative; scikit-image ``random_noise(mode="speckle")`` form)
    gaussian  y = x + n,     n ~ N(0, var)   (additive; ``random_noise(mode="gaussian")`` form)
    motion    y = x * k, k = horizontal line kernel of length L (all ones / L), cv2.filter2D, border reflect-101;
              horizontal = the lateral transducer direction in the stored frame
    output    clip(y, 0, 1); then the usual 3-channel replication and ImageNet mean/std

The noise for (condition, frame) is drawn from ``np.random.default_rng([base_seed, condition index,
crc32(frame_id)])``, so every model, seed and repeated run sees exactly the same corrupted pixels, independent of the
batch order. The blur is deterministic by construction.
"""

from __future__ import annotations

import functools
import zlib

import cv2
import numpy as np

from fin.config import load_config

Condition = tuple[str, str, float | int | None, str]
CORRUPTION_NAME = {"none": "Clean", "speckle": "Speckle noise", "motion": "Motion blur", "gaussian": "Gaussian noise"}


@functools.cache
def base_seed() -> int:
    """Base seed of the corruption noise (``configs/robustness.yaml``)."""
    return int(load_config("robustness")["base_seed"])


@functools.cache
def conditions() -> tuple[Condition, ...]:
    """Every condition as (id, corruption kind, parameter, label), in the order of ``configs/robustness.yaml``."""
    return tuple(tuple(c) for c in load_config("robustness")["conditions"])


def condition(cond_id: str) -> Condition:
    """The condition with id ``cond_id``."""
    for c in conditions():
        if c[0] == cond_id:
            return c
    raise KeyError(cond_id)


def _rng(cond_id: str, frame_id: str, seed: int) -> np.random.Generator:
    index = [c[0] for c in conditions()].index(cond_id)
    return np.random.default_rng([seed, index, zlib.crc32(str(frame_id).encode())])


def speckle(x: np.ndarray, var: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative Gaussian noise of the given variance."""
    return np.clip(x + x * rng.normal(0.0, np.sqrt(var), x.shape).astype(np.float32), 0.0, 1.0)


def gaussian(x: np.ndarray, var: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise of the given variance."""
    return np.clip(x + rng.normal(0.0, np.sqrt(var), x.shape).astype(np.float32), 0.0, 1.0)


def motion_blur(x: np.ndarray, length: int) -> np.ndarray:
    """Horizontal motion blur with a line kernel of ``length`` pixels."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    return np.clip(cv2.filter2D(x.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT_101), 0.0, 1.0)


def corrupt(x: np.ndarray, cond_id: str, frame_id: str, seed: int | None = None) -> np.ndarray:
    """Corrupt an H x W image (float in [0, 1] or uint8); returns float32 in [0, 1].

    ``seed`` replaces the configured base seed (``base_seed()``) of the noise generator.
    """
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    _, kind, p, _ = condition(cond_id)
    if kind == "none":
        return x.copy()
    if kind == "motion":
        return motion_blur(x, int(p))
    rng = _rng(cond_id, frame_id, base_seed() if seed is None else seed)
    if kind == "speckle":
        return speckle(x, float(p), rng)
    if kind == "gaussian":
        return gaussian(x, float(p), rng)
    raise ValueError(kind)
