"""Copy the small outputs of the reported run families from ``$FIN_RUNS`` into ``results/runs``.

Copied: ``*.json``, ``*.jsonl``, ``*.csv``, ``*.log`` and ``*.npz`` files of the families below, at the same relative
paths; never checkpoints (``*.pt``), caches (``cache/`` folders, ``*.npy``) or anything else. Text files pass through
``fin.io.portable_text``, so the working-folder prefix is written as ``$FIN_WORK`` (``$FIN_RUNS``, ``$FIN_DATA``) and
repository paths become relative: the committed ``results/`` never holds a machine-specific absolute path. Files
already in ``results/runs`` that have no source are kept.

    fin collect-results [--runs DIR] [--dest DIR] [--families cls seg ...]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from fin.config import RESULT_RUNS, runs_dir
from fin.io import portable_text, setup_logging, write_text

log = logging.getLogger(__name__)

FAMILIES = ("cls", "cls_cv", "cls_authorsplit", "seg", "ext469", "e2e", "robustness", "efficiency")
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".log"}
BINARY_SUFFIXES = {".npz"}


def collect(runs: Path, dest: Path, families: tuple[str, ...] = FAMILIES) -> int:
    """Copy every selected file of every family present under ``runs``; returns the number of files written."""
    n = 0
    for fam in families:
        src_root = runs / fam
        if not src_root.is_dir():
            log.info("[skip] %s: no %s", fam, src_root)
            continue
        for src in sorted(src_root.rglob("*")):
            rel = src.relative_to(runs)
            if not src.is_file() or "cache" in rel.parts[:-1]:
                continue
            out = dest / rel
            if src.suffix in TEXT_SUFFIXES:
                write_text(out, portable_text(src.read_text()))
            elif src.suffix in BINARY_SUFFIXES:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, out)
            else:
                continue
            n += 1
        log.info("%s -> %s", src_root, dest / fam)
    return n


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=None, help="default: $FIN_RUNS")
    ap.add_argument("--dest", type=Path, default=RESULT_RUNS, help="default: results/runs")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), choices=FAMILIES)
    a = ap.parse_args(argv)
    setup_logging()
    n = collect(a.runs or runs_dir(), a.dest, tuple(a.families))
    log.info("%d files written", n)


if __name__ == "__main__":
    main()
