"""The ``\\res{key}`` registry (``<paper-dir>/RESULT_KEYS.md``) and the resolver for registered keys.

Each registry line is ``key | meaning | source``. A meaning may ask for a precision (``(..., 1 dp)``), a percentage
(``(%``, ``in %``) or only the seed mean (``mean over seeds``); the resolver re-formats the stored per-seed values
accordingly. Keys spelled in another writer's style are mapped onto the store's scheme where unambiguous.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fin.reporting.fmt import Store, f, fmt_raw, stats
from fin.reporting.names import CLS, CLS_NAME, SEG, SEG_NAME

SYN = {
    "accuracy": "acc",
    "macrof1": "mf1",
    "macro_f1": "mf1",
    "sensitivity": "sens",
    "specificity": "spec",
    "precision": "prec",
    "rocauc": "auc",
    "mean_fg_dice": "meandice",
    "mean_fg_iou": "meaniou",
    "meanfgdice": "meandice",
    "meanfgiou": "meaniou",
    "median_ae": "medae",
    "within5": "w5",
    "within10": "w10",
}


def normalise(key: str) -> str:
    """Map another spelling onto the store's scheme (drop ``.test`` in cls keys, ``_fh`` -> ``.fh``, synonyms)."""
    parts = key.split(".")
    if parts[0] == "cls" and len(parts) > 1 and parts[1] == "gate":
        parts.insert(1, "deepfetalnet")
    if parts[0] == "cls" and len(parts) > 2 and parts[2] == "test":
        parts.pop(2)
    if parts[0] == "cls" and len(parts) > 3 and parts[2] == "gate" and parts[3] == "test":
        parts.pop(3)
    out = []
    for p in parts:
        p = SYN.get(p, p)
        m = re.match(r"^(dice|iou|hd95|hd95n|bf1|pdice|piou|empty)_(fh|ps)$", p)
        out += [m.group(1), m.group(2)] if m else [p]
    return ".".join(out)


def format_hint(meaning: str) -> tuple[int | None, bool, str | None]:
    """(decimal places, percentage, part) requested by a registry meaning."""
    nd = re.search(r"(\d)\s*dp", meaning)
    pct = bool(re.search(r"\(\s*%|%\s*,|in %|\(%\)", meaning))
    want = None
    if re.search(r"mean over seeds|seed[- ]mean|mean over seed", meaning) and not re.search(r"\bSD\b|\\pm|±", meaning):
        want = "mean"
    return (int(nd.group(1)) if nd else None), pct, want


def _best_of(v: Store, prefix_fmt: str, candidates: list[str], pick_max: bool = True) -> tuple[float, str] | None:
    cand = [
        (stats(v.raw[prefix_fmt.format(c)][0])[0], c)
        for c in candidates
        if prefix_fmt.format(c) in v.raw and stats(v.raw[prefix_fmt.format(c)][0])[0] is not None
    ]
    if not cand:
        return None
    return max(cand) if pick_max else min(cand)


def resolve(v: Store, key: str, meaning: str, frames_df: pd.DataFrame | None) -> str | None:
    """Value of a registered key in the unit and precision its meaning asks for (None if unknown)."""
    nd_h, pct_h, want_h = format_hint(meaning)
    base, want = normalise(key), want_h
    for suf in (".mean", ".sd"):
        if base.endswith(suf):
            base, want = base[: -len(suf)], suf[1:]
    if base in v.raw:
        return fmt_raw(v.raw[base], want, nd_h, pct_h)
    if normalise(key) in v and nd_h is None and not pct_h:
        return v[normalise(key)]
    m = re.match(r"^cls\.best\.(test|val)\.(\w+)\.(name|id|mean|sd)$", key)
    if m:
        met = SYN.get(m.group(2), m.group(2))
        pre = "" if m.group(1) == "test" else "val."
        best = _best_of(v, f"cls.{{}}.{pre}{met}", [c for c in CLS if c != "deepfetalnet"])
        if best is None:
            return None
        if m.group(3) == "name":
            return CLS_NAME[best[1]]
        if m.group(3) == "id":
            return best[1]
        return fmt_raw(v.raw[f"cls.{best[1]}.{pre}{met}"], m.group(3), nd_h, pct_h)
    m = re.match(r"^aop\.(jnu|iugc)\.(\w+)\.(min|max)(\.name)?$", key)
    if m:
        met = SYN.get(m.group(2), m.group(2))
        best = _best_of(v, f"aop.{{}}.{m.group(1)}.{met}", SEG, pick_max=m.group(3) == "max")
        if best is None:
            return None
        if m.group(4):
            return SEG_NAME[best[1]]
        return fmt_raw(v.raw[f"aop.{best[1]}.{m.group(1)}.{met}"], "mean", nd_h, pct_h)
    m = re.match(r"^data\.jnu\.(all|none|onlysp|onlyhead|sphead)\.(fh|ps)_px_pct$", key)
    if m and frames_df is not None:
        cat = {"all": None, "none": 3, "onlysp": 4, "onlyhead": 5, "sphead": 6}[m.group(1)]
        d = frames_df if cat is None else frames_df[frames_df.category == cat]
        return f((d[f"{m.group(2)}_px"] / (d.height * d.width)).mean(), nd_h if nd_h is not None else 2, 100)
    return None


def registered_keys(path: Path) -> dict[str, str]:
    """{key: meaning} from registry lines ``key | meaning | source`` (headers and comments skipped)."""
    keys: dict[str, str] = {}
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "|", "<!--", "-")) or "|" not in s:
            continue
        parts = [x.strip() for x in s.split("|")]
        k = parts[0].strip("`")
        if k and " " not in k and k != "key":
            keys[k] = parts[1] if len(parts) > 1 else ""
    return keys


def strip_comments(text: str) -> str:
    """Remove LaTeX comments (unescaped ``%`` to end of line)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.splitlines())


def used_keys(sections_dir: Path) -> dict[str, set[str]]:
    """{key: files} of every ``\\res{key}`` in the hand-written section files (generated files excluded)."""
    used: dict[str, set[str]] = {}
    for p in sorted(sections_dir.glob("*.tex")) + sorted(sections_dir.glob("*/*.tex")):
        if p.name.startswith("results_") or p.parent.name == "tables":
            continue
        for k in re.findall(r"\\res\{([^}]+)\}", strip_comments(p.read_text())):
            used.setdefault(k, set()).add(str(p.relative_to(sections_dir)))
    return used
