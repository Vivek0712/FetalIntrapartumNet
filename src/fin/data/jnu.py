"""Build the JNU-IFM frame table, the remapped masks and the locked patient-disjoint split.

Inputs
    RAW   ``us_data`` folder of the JNU-IFM release (78 video folders, each with ``frame_label.csv``,
          ``image/<video>_<frame>.png`` and ``mask/<video>_<frame>_mask.png``)
    T1    patient -> video map parsed from Table 1 of the JNU-IFM data descriptor (Lu et al., Data in Brief 41
          (2022) 107904); committed as ``splits/descriptor_table1_patients.json``
Outputs in OUT
    ``frames_base.csv`` (one row per frame, split included), ``split_videos.json``,
    ``masks/<video>/<frame_id>.png`` (0 bg, 1 FH, 2 PS) and ``build_log.json``.
    ``fin.data.fhpsaop`` then adds the reference AoP and writes ``frames.csv``.

    fin jnu build RAW OUT [--t1 splits/descriptor_table1_patients.json]
    fin jnu parse-descriptor DESCRIPTOR_TEXT OUT_JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from fin.config import SPLITS, data_dir, load_config
from fin.data.datasets import read_image, read_mask
from fin.data.labels import CATEGORY_CONTENT, CATEGORY_NAMES, JNU_RAW, binary_label, remap_jnu
from fin.data.splits import SPLIT_NAMES, check_disjoint, search_patient_split
from fin.io import setup_logging, write_csv, write_json

log = logging.getLogger(__name__)


def stored_path(p: str | Path) -> str:
    """Path to store in a frame table: relative to ``FIN_DATA`` when inside it, else absolute."""
    p = Path(p).resolve()
    root = data_dir().resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


def parse_descriptor_table(text: str) -> list[dict[str, Any]]:
    """Parse Table 1 of the data descriptor (text dump) into rows p, none, c2, c3, sph, files."""
    rows: list[dict[str, Any]] = []
    cur = None
    for line in text.split("\n"):
        m = re.match(r"\s+(\d{1,2})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(20\d{6}T\d{6})\s*$", line)
        if m:
            cur = {
                "p": int(m[1]),
                "none": int(m[2]),
                "c2": int(m[3]),
                "c3": int(m[4]),
                "sph": int(m[5]),
                "files": [m[6]],
            }
            rows.append(cur)
            continue
        m = re.match(r"\s+(20\d{6}T\d{6})\s*$", line)
        if m and cur:
            cur["files"].append(m[1])
    return rows


def read_frames(raw: Path, out: Path, problems: list[str]) -> pd.DataFrame:
    """Read every labelled frame, write its remapped mask and return the frame table."""
    rows = []
    for v in sorted(d for d in os.listdir(raw) if (raw / d).is_dir()):
        lab = pd.read_csv(raw / v / "frame_label.csv")
        imgs = {f for f in os.listdir(raw / v / "image") if f.endswith(".png")}
        seen = set()
        for _, r in lab.iterrows():
            fid, c = int(r["frame_id"]), int(r["frame_label"])
            ip = raw / v / "image" / f"{v}_{fid}.png"
            mp = raw / v / "mask" / f"{v}_{fid}_mask.png"
            seen.add(f"{v}_{fid}.png")
            if not (ip.exists() and mp.exists()):
                problems.append(f"missing file {ip} or {mp}")
                continue
            m = read_mask(mp)
            im = read_image(ip, cv2.IMREAD_UNCHANGED)
            if m.ndim != 2:
                problems.append(f"mask not 2D {mp} {m.shape}")
                m = m[..., 0]
            vals = set(np.unique(m).tolist())
            if not vals <= {0, JNU_RAW["ps"], JNU_RAW["fh"]}:
                problems.append(f"unexpected mask values {sorted(vals)} {mp}")
            has_fh, has_ps = bool((m == JNU_RAW["fh"]).any()), bool((m == JNU_RAW["ps"]).any())
            if (has_fh, has_ps) != CATEGORY_CONTENT[c]:
                problems.append(f"category {c} vs mask (fh={has_fh},ps={has_ps}) {v}_{fid}")
            op = out / "masks" / v / f"{v}_{fid}.png"
            op.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(op), remap_jnu(m))
            rows.append(
                {
                    "frame_id": f"{v}_{fid}",
                    "video_id": v,
                    "frame_index": fid,
                    "image_path": stored_path(ip),
                    "mask_path": stored_path(op),
                    "category": c,
                    "category_name": CATEGORY_NAMES[c],
                    "binary_label": binary_label(c),
                    "has_fh": int(has_fh),
                    "has_ps": int(has_ps),
                    "height": im.shape[0],
                    "width": im.shape[1],
                    "img_channels": 1 if im.ndim == 2 else im.shape[2],
                    "fh_px": int((m == JNU_RAW["fh"]).sum()),
                    "ps_px": int((m == JNU_RAW["ps"]).sum()),
                }
            )
        extra = imgs - seen
        if extra:
            problems.append(f"{v}: {len(extra)} images without label row")
    return pd.DataFrame(rows)


def attach_patients(df: pd.DataFrame, t1: list[dict[str, Any]], problems: list[str]) -> pd.DataFrame:
    """Add ``patient_id`` from the descriptor table and check the per-woman category counts against it."""
    v2p = {f: r["p"] for r in t1 for f in r["files"]}
    missing = sorted(set(df.video_id) - set(v2p))
    if missing:
        problems.append(f"videos without patient id: {missing}")
    df["patient_id"] = df.video_id.map(v2p)
    for r in t1:
        sub = df[df.patient_id == r["p"]]
        if (
            (sub.category == 6).sum() != r["sph"]
            or (sub.category == 3).sum() != r["none"]
            or sorted({(sub.category == 4).sum(), (sub.category == 5).sum()}) != sorted({r["c2"], r["c3"]})
        ):
            problems.append(
                f"patient {r['p']} counts differ from Table 1: data {dict(Counter(sub.category))} table {r}"
            )
    return df


def build(raw: Path, out: Path, t1_path: Path) -> dict[str, Any]:
    """Build the frame table and the locked split; returns the build log."""
    cfg = load_config("split")
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    df = read_frames(raw, out, problems)
    with open(t1_path) as fh:
        df = attach_patients(df, json.load(fh), problems)

    res = search_patient_split(df, cfg["draws"], cfg["seed"], tuple(cfg["fractions"]))
    df["split"] = df.patient_id.map(lambda p: SPLIT_NAMES[res["assignment"][p]])
    check_disjoint(df, "patient_id")
    check_disjoint(df, "video_id")
    names = ["train", "val", "test"]
    sv = {s: sorted(df[df.split == s].video_id.unique().tolist()) for s in names}
    sp = {s: sorted(int(x) for x in df[df.split == s].patient_id.unique()) for s in names}
    write_json(
        out / "split_videos.json",
        {
            "seed": cfg["seed"],
            "method": f"patient-disjoint random search ({cfg['draws']} draws, "
            f"rng=np.random.default_rng({cfg['seed']})), "
            "target 80/10/10 of frames, cost = |frame deviation|/N + L1 distance of category mix; "
            "every split must contain all four categories",
            "selected_draw": res["draw"],
            "cost": res["cost"],
            "videos": sv,
            "patients": sp,
        },
    )
    write_csv(out / "frames_base.csv", df)
    build_log: dict[str, Any] = {
        "problems": problems,
        "n_frames": len(df),
        "counts": {CATEGORY_NAMES[k]: int(v) for k, v in df.category.value_counts().sort_index().items()},
        "split_frames": df.split.value_counts().to_dict(),
        "split_videos": {k: len(v) for k, v in sv.items()},
        "split_patients": {k: len(v) for k, v in sp.items()},
        "split_by_category": pd.crosstab(df.split, df.category_name).to_dict(),
        "image_shapes": {
            str(k): int(v) for k, v in Counter(zip(df.height, df.width, df.img_channels, strict=True)).items()
        },
    }
    write_json(out / "build_log.json", build_log, default=int)
    return build_log


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="frame table, masks and locked split")
    b.add_argument("raw", type=Path, help="JNU-IFM us_data folder")
    b.add_argument("out", type=Path, help="output folder, normally $FIN_DATA/jnu/processed")
    b.add_argument("--t1", type=Path, default=SPLITS / "descriptor_table1_patients.json")
    p = sub.add_parser("parse-descriptor", help="patient -> video map from the descriptor's Table 1 text")
    p.add_argument("text", type=Path)
    p.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    setup_logging()
    if a.cmd == "parse-descriptor":
        rows = parse_descriptor_table(a.text.read_text())
        write_json(a.out, rows, indent=None)
        log.info("%d patients, %d files", len(rows), sum(len(r["files"]) for r in rows))
        return
    res = build(a.raw, a.out, a.t1)
    log.info("frames %d, splits %s, problems %d", res["n_frames"], res["split_frames"], len(res["problems"]))
    for problem in res["problems"]:
        log.warning("%s", problem)


if __name__ == "__main__":
    main()
