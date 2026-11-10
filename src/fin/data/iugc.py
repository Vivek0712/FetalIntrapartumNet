"""Build the IUGC 2024 external test set: 300 labelled frames (one per video), masks and the official AoP.

IUGC_ROOT is the folder into which the release archives of Zenodo doi 10.5281/zenodo.17655183 were unzipped; the test
archive gives ``test/videos/*.avi``, ``test/seg/*.png`` (masks: 0 bg, 127 PS, 255 FH), ``test/seg/landmarks.json``
(official landmark AoP), ``test/seg/seg_info.csv`` and ``test/test_info.csv``. The labelled frame is decoded from
each video at ``labeled_frame_index`` (OpenCV); with ``--frames-dir`` it is also compared with pre-extracted frames.

Outputs in OUT: ``images/<name>.png`` (grayscale), ``masks/<name>.png`` (0 bg, 1 FH, 2 PS), ``aop.csv``,
``build_log.json``.

    fin build-iugc IUGC_ROOT OUT [--frames-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from fin.data.datasets import read_gray, read_mask
from fin.data.jnu import stored_path
from fin.data.labels import FH, IUGC, PS, remap_iugc
from fin.geometry import aop_from_mask
from fin.io import setup_logging, write_csv, write_json

log = logging.getLogger(__name__)


def decode_frame(video: Path, index: int) -> tuple[np.ndarray | None, int]:
    """Grayscale frame ``index`` of a video and the video's frame count (None if it cannot be decoded)."""
    cap = cv2.VideoCapture(str(video))
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, fr = cap.read()
    cap.release()
    return (cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if ok else None), nfr


def build(root: Path, out: Path, frames_dir: Path | None = None) -> dict[str, Any]:
    """Decode, remap and tabulate the 300 labelled test frames; returns the build log."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    with open(root / "test" / "seg" / "landmarks.json") as fh:
        landmarks = json.load(fh)
    info = pd.read_csv(root / "test" / "test_info.csv")
    seg = pd.read_csv(root / "test" / "seg" / "seg_info.csv")
    problems: list[str] = []
    rows = []
    for _, r in seg.iterrows():
        stem, fi = r.filename[:-4], int(r.labeled_frame_index)
        name = f"{stem}_{fi}.png"
        if name not in landmarks:
            problems.append(f"no landmark for {name}")
            continue
        g, nfr = decode_frame(root / "test" / "videos" / r.filename, fi)
        if g is None:
            problems.append(f"cannot decode frame {fi} of {r.filename}")
            continue
        pre = frames_dir / name if frames_dir else None
        diff = np.nan
        if pre is not None and pre.exists():
            p = read_gray(pre)
            diff = float(np.abs(p.astype(int) - g.astype(int)).mean()) if p.shape == g.shape else np.inf
        m = read_mask(root / "test" / "seg" / name)
        if m.ndim == 3:
            m = m[..., 0]
        vals = set(np.unique(m).tolist())
        if not vals <= {0, IUGC["ps"], IUGC["fh"]}:
            problems.append(f"mask values {sorted(vals)} {name}")
        y = remap_iugc(m)
        if g.shape != y.shape:
            problems.append(f"image {g.shape} vs mask {y.shape} {name}")
        cv2.imwrite(str(out / "images" / name), g)
        cv2.imwrite(str(out / "masks" / name), y)
        ir = info[info.filename == r.filename].iloc[0]
        sp_idx = json.loads(ir.SP_index) if isinstance(ir.SP_index, str) else []
        lm = landmarks[name]
        rows.append(
            {
                "name": name,
                "video": r.filename,
                "frame_index": fi,
                "video_frames": nfr,
                "labelled_frame_is_SP": int(fi in sp_idx),
                "SP_count": int(ir.SP_count),
                "NSP_count": int(ir.NSP_count),
                "aop_official": float(lm["aop"]),
                "hsd_official": float(lm["hsd"]),
                "aop_mask_challenge_code": aop_from_mask(y, fh=FH, ps=PS),
                "has_fh": int((y == FH).any()),
                "has_ps": int((y == PS).any()),
                "height": g.shape[0],
                "width": g.shape[1],
                "prextracted_mae": diff,
                "image_path": stored_path(out / "images" / name),
                "mask_path": stored_path(out / "masks" / name),
            }
        )
    df = pd.DataFrame(rows)
    write_csv(out / "aop.csv", df)
    d = (df.aop_official - df.aop_mask_challenge_code).dropna()
    build_log = {
        "problems": problems,
        "n": len(df),
        "sizes": repr(df.groupby(["height", "width"]).size().to_dict()),
        "labelled_frame_is_SP": int(df.labelled_frame_is_SP.sum()),
        "has_both": int((df.has_fh & df.has_ps).sum()),
        "preextracted_identical": int((df.prextracted_mae == 0).sum()),
        "preextracted_mae_max": float(df.prextracted_mae.max()),
        "aop_official_mean": float(df.aop_official.mean()),
        "aop_official_sd": float(df.aop_official.std()),
        "aop_mask_vs_official_absdiff_mean": float(d.abs().mean()),
        "aop_mask_vs_official_absdiff_median": float(d.abs().median()),
        "aop_mask_vs_official_absdiff_p95": float(d.abs().quantile(0.95)),
        "aop_mask_nan": int(df.aop_mask_challenge_code.isna().sum()),
        "ps_pixel_fraction": float(np.mean([(read_mask(out / "masks" / n) == PS).mean() for n in df.name])),
        "mask_label_map": "source 0 bg, 127 PS, 255 FH (PS landmarks fall on 127 in 300/300 frames) "
        "-> 0 bg, 1 FH, 2 PS",
    }
    write_json(out / "build_log.json", build_log)
    return build_log


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="IUGC 2024 release root")
    ap.add_argument("out", type=Path, help="output folder, normally $FIN_DATA/iugc/processed")
    ap.add_argument("--frames-dir", type=Path, default=None, help="pre-extracted frames to compare with (optional)")
    a = ap.parse_args(argv)
    setup_logging()
    res = build(a.root, a.out, a.frames_dir)
    log.info("IUGC 2024 test frames: %d, problems: %d", res["n"], len(res["problems"]))


if __name__ == "__main__":
    main()
