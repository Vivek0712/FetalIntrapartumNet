"""Discovery of run directories (``<root>/<model>/seed<k>/``)."""

from __future__ import annotations

from pathlib import Path


def seed_dirs(root: Path, model: str, marker: str) -> list[tuple[int, Path]]:
    """Sorted (seed, run directory) pairs for ``<root>/<model>/seed<k>/<marker>``.

    A flat ``<root>/<model>/<marker>`` (single-run layout) counts as seed 0.
    """
    root = Path(root)
    out = []
    for p in root.glob(f"{model}/seed*/{marker}"):
        try:
            out.append((int(p.parent.name[4:]), p.parent))
        except ValueError:
            continue
    if not out and (root / model / marker).exists():
        out.append((0, root / model))
    return sorted(out)
