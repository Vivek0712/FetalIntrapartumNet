"""Repository hygiene: no unfinished markers, no local paths, no manuscript files and consistent naming."""

import re

from fin.config import ROOT

TEXT = {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".cff", ".txt", ".json", ".csv", ".jsonl", ".svg", ".log"}
SKIP_DIRS = {".git", "work", "build", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "env"}
MARKERS = re.compile("|".join(["TO" + "DO", "T" + "BD", "X" + "XX", "FIX" + "ME", r"AUTHORS TO " + "CONFIRM"]))
LOCAL = re.compile("/" + "Users/|/private/" + "tmp/")
# this repository is the official code of the paper: it is never described as a reconstruction of earlier work
NAMING = re.compile(r"(?i)\bre" + r"buil(d|t)|\bre-?" + r"run|first " + r"version|\bv" + r"6\b|earlier " + r"version")
MANUSCRIPT = {".tex", ".cls", ".bib", ".pdf"}


def _files():
    for p in ROOT.rglob("*"):
        parts = p.relative_to(ROOT).parts
        if p.is_file() and not SKIP_DIRS & set(parts) and not any(x.endswith(".egg-info") for x in parts):
            yield p


def test_no_unfinished_markers_or_local_paths():
    bad = []
    for p in _files():
        if p.suffix in TEXT:
            text = p.read_text(errors="ignore")
            if MARKERS.search(text) or LOCAL.search(text):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_official_naming():
    bad = []
    for p in _files():
        if p.suffix in TEXT and NAMING.search(p.read_text(errors="ignore")):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_no_manuscript_files():
    assert not (ROOT / "paper").exists()
    bad = [str(p.relative_to(ROOT)) for p in _files() if p.suffix in MANUSCRIPT]
    assert not bad, bad
