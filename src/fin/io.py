"""The single I/O layer for result files: JSON, text, CSV and compressed arrays, portable paths, logging setup.

Every writer creates the parent folder and writes atomically (temporary file, then rename), so an
interrupted run never leaves a truncated result behind. Paths recorded inside result files go through
``portable_path`` (or ``portable_text`` for whole files), so no machine-specific absolute path reaches ``results/``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fin.config import ROOT, data_dir, runs_dir, work_dir

PathLike = str | os.PathLike


def _path_prefixes() -> list[tuple[str, str]]:
    """(absolute folder, placeholder) pairs in order of preference; ``.`` marks the repository root.

    Each folder appears in its absolute and its symlink-resolved spelling; within one folder the longer spelling
    comes first, so a text replacement never leaves part of a prefix behind.
    """
    pairs: list[tuple[str, str]] = []
    for folder, label in ((work_dir(), "$FIN_WORK"), (runs_dir(), "$FIN_RUNS"), (data_dir(), "$FIN_DATA"), (ROOT, ".")):
        forms = {str(Path(folder).absolute()), str(Path(folder).resolve())}
        pairs += [(f, label) for f in sorted(forms, key=len, reverse=True)]
    return pairs


def portable_path(path: PathLike) -> str:
    """``path`` with the working folders written as ``$FIN_WORK``/``$FIN_RUNS``/``$FIN_DATA`` and repository paths
    made relative; other paths are returned unchanged.

    ``$FIN_WORK`` is tried first, so with the default layout every run path reads ``$FIN_WORK/runs/...``.
    """
    p = Path(path)
    for prefix, label in _path_prefixes():
        for c in (p.absolute(), p.resolve()):
            if c.is_relative_to(prefix):
                rel = str(c.relative_to(prefix))
                if label == ".":
                    return rel
                return label if rel == "." else f"{label}/{rel}"
    return str(path)


def portable_text(text: str) -> str:
    """Replace every absolute working-folder or repository path inside ``text`` (a log or JSON file) as
    ``portable_path`` would."""
    for prefix, label in _path_prefixes():
        text = text.replace(prefix + "/", "" if label == "." else label + "/")
        text = re.sub(re.escape(prefix) + r'(?=["\s,;:)\]]|$)', label, text, flags=re.MULTILINE)
    return text


def _atomic(path: PathLike, write: Callable[[Path], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)
    return path


def write_json(path: PathLike, obj: Any, indent: int = 1, default: Callable[[Any], Any] | None = None) -> Path:
    """Write ``obj`` as JSON."""

    def _w(p: Path) -> None:
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=indent, default=default)

    return _atomic(path, _w)


def read_json(path: PathLike) -> Any:
    """Read a JSON file."""
    with open(path) as fh:
        return json.load(fh)


def read_json_if_exists(path: PathLike | None) -> Any:
    """Read a JSON file, or return None when ``path`` is None or missing."""
    if path is None or not os.path.exists(path):
        return None
    return read_json(path)


def append_jsonl(path: PathLike, record: Mapping[str, Any]) -> None:
    """Append one JSON line to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def write_text(path: PathLike, text: str) -> Path:
    """Write a text file."""

    def _w(p: Path) -> None:
        with open(p, "w") as fh:
            fh.write(text)

    return _atomic(path, _w)


def write_csv(path: PathLike, df: pd.DataFrame) -> Path:
    """Write a data frame as CSV without the index."""
    return _atomic(path, lambda p: df.to_csv(p, index=False))


def read_frames_csv(path: PathLike) -> pd.DataFrame:
    """Read a JNU-IFM frame table; category_name ``None`` must stay a string, not NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_npz(path: PathLike, **arrays: np.ndarray) -> Path:
    """Write arrays to a compressed ``.npz`` file."""

    def _w(p: Path) -> None:
        with open(p, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    return _atomic(path, _w)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a plain ``time level name: message`` log format on stderr."""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
