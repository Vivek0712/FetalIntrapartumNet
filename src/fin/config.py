"""Paths, configuration files, device selection and seeding.

Environment variables (all optional)
    FIN_WORK          working folder for data, runs and the third-party cache (default: ``<repo>/work``)
    FIN_DATA          prepared datasets (default: ``$FIN_WORK/data``)
    FIN_RUNS          training and evaluation outputs (default: ``$FIN_WORK/runs``)
    FIN_THIRD_PARTY   cache of fetched third-party source files (default: ``$FIN_WORK/third_party``)
    FIN_SEG_CACHE     256x256 segmentation cache (default: ``$FIN_RUNS/seg/cache``)
    FIN_SPLIT_FILE    CSV (frame_id, split) that replaces the locked split, for sensitivity runs only
    FIN_REPO          repository checkout, when the package is not installed from it (see below)
    FIN_PAPER         LaTeX manuscript folder that the paper-reporting commands write into (``--paper-dir``)

Repository resources (``configs/``, ``splits/``, ``results/``) are located relative to the
repository root: ``FIN_REPO`` if set, else the checkout this file lives in (editable install, ``pip install -e .``),
else the checkout that contains the current folder. A regular (non-editable) install outside a checkout has no
access to these resources, so importing this module then raises ``RepositoryNotFoundError``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fin import PaperDirectoryError, RepositoryNotFoundError


def _is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "fin").is_dir() and (path / "configs").is_dir()


def repo_root() -> Path:
    """Return the repository root: ``FIN_REPO``, else the checkout holding this file or the current folder."""
    env = os.environ.get("FIN_REPO")
    if env:
        if not _is_checkout(Path(env)):
            raise RepositoryNotFoundError(f"FIN_REPO={env} is not a FetalIntrapartumNet checkout")
        return Path(env).resolve()
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (start, *start.parents):
            if _is_checkout(parent):
                return parent
    raise RepositoryNotFoundError(
        "cannot find the FetalIntrapartumNet repository (configs/, splits/, results/). Install the package "
        "from a checkout with `pip install -e .`, run from inside the checkout, or set FIN_REPO to its path."
    )


ROOT = repo_root()
CONFIGS = ROOT / "configs"
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
RESULT_RUNS = RESULTS / "runs"
DOCS_ASSETS = ROOT / "docs" / "assets"


def work_dir() -> Path:
    """Working folder for data, runs and caches (``FIN_WORK``)."""
    return Path(os.environ.get("FIN_WORK", ROOT / "work"))


def data_dir() -> Path:
    """Prepared datasets (``FIN_DATA``)."""
    return Path(os.environ.get("FIN_DATA", work_dir() / "data"))


def runs_dir() -> Path:
    """Training and evaluation outputs (``FIN_RUNS``)."""
    return Path(os.environ.get("FIN_RUNS", work_dir() / "runs"))


def paper_dir(arg: Path | None = None) -> Path:
    """LaTeX manuscript folder written by ``fin fill-results``, ``supp-tables``, ``figures`` and ``method-figures``.

    The manuscript is not part of this repository: pass ``--paper-dir`` or set ``FIN_PAPER`` to the folder that
    holds ``RESULT_KEYS.md``, ``sections/`` and ``figs/``.
    """
    path = arg if arg is not None else os.environ.get("FIN_PAPER")
    if not path:
        raise PaperDirectoryError(
            "no manuscript folder given: pass --paper-dir DIR or set FIN_PAPER to the folder holding "
            "RESULT_KEYS.md, sections/ and figs/"
        )
    return Path(path)


def third_party_dir() -> Path:
    """Cache of third-party source files fetched at pinned commits (``FIN_THIRD_PARTY``)."""
    return Path(os.environ.get("FIN_THIRD_PARTY", work_dir() / "third_party"))


def seg_cache_dir() -> Path:
    """Segmentation image/mask cache (``FIN_SEG_CACHE``)."""
    return Path(os.environ.get("FIN_SEG_CACHE", runs_dir() / "seg" / "cache"))


def jnu_frames_csv() -> Path:
    """Frame table of the prepared JNU-IFM dataset."""
    return data_dir() / "jnu" / "processed" / "frames.csv"


def iugc_csv() -> Path:
    """Frame table of the prepared IUGC 2024 test set."""
    return data_dir() / "iugc" / "processed" / "aop.csv"


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dictionary."""
    path = CONFIGS / f"{name}.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def require_setting(cfg: dict[str, Any], key: str, expected: Any) -> None:
    """Raise ValueError unless the dotted config ``key`` equals ``expected``.

    Used for settings that name a component the code implements in exactly one way (optimizer, scheduler, loss), so
    editing them in a config file cannot silently have no effect.
    """
    value: Any = cfg
    for part in key.split("."):
        value = value[part]
    if value != expected:
        raise ValueError(f"config {key} = {value!r}: only {expected!r} is implemented")


def pick_device(name: str = "auto") -> torch.device:
    """Return a torch device; ``auto`` picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels on CUDA or MPS (no-op on CPU)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: torch.device) -> None:
    """Release cached allocator blocks on CUDA or MPS (no-op on CPU)."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def seed_everything(seed: int, python: bool = True) -> None:
    """Seed NumPy's global generator and torch (and Python's ``random`` unless ``python`` is False)."""
    if python:
        random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
