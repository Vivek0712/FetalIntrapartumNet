"""Write the paper's values and Tables 1-5 from the saved run outputs. Nothing here trains a model.

Inputs (``--runs`` defaults to the committed ``results/runs``)
    cls/<model>/seed<k>/{metrics_test.json, preds_test.npz}     fin eval-cls
    seg/<model>/seed<k>/{metrics.json, per_image_*.csv, preds_*.npz}, seg/ceiling.json   fin eval-seg
    efficiency/results.json, robustness/summary.json, ext469/summary.json + per_image_all.csv
    cls_cv/summary_cv.json, cls_authorsplit/author_split_check.json
    $FIN_SEG_CACHE/{jnu,iugc}_{mask.npy,index.csv}   ground-truth masks (pixel accuracy, image-blind reference rows)
    splits/frames.csv (dataset counts, constant AoP), $FIN_DATA/iugc/processed/aop.csv (IUGC 2024 references)
Outputs
    <paper-dir>/sections/results_values.tex   one ``\\resdef{key}{value}`` per key (read by results_macros.tex)
    <paper-dir>/sections/tables/table{1,1_cv,2,3,3_auc_all,4,5}.tex
    --dump FILE                          every computed key and value as JSON
Every key registered in <paper-dir>/RESULT_KEYS.md that cannot be resolved, and every ``\\res{}`` in the paper that
is not
registered or has no value, is reported as a warning.

    fin fill-results --paper-dir DIR [--runs DIR] [--out-dir DIR] [--dump FILE]     (or FIN_PAPER=DIR)
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

from fin.config import RESULT_RUNS, SPLITS, iugc_csv, paper_dir, seg_cache_dir
from fin.io import portable_path, read_frames_csv, setup_logging, write_json, write_text
from fin.reporting import tables, values
from fin.reporting.fmt import Store
from fin.reporting.keys import registered_keys, resolve, used_keys
from fin.reporting.names import SEG

log = logging.getLogger(__name__)


def compute(
    runs: Path,
    frames: Path,
    iugc: Path,
    cache: Path,
    n_boot: int,
    cls: Path | None = None,
    seg: Path | None = None,
    eff: Path | None = None,
    rob: Path | None = None,
    ext469: Path | None = None,
) -> tuple[Store, dict[str, Path]]:
    """Fill a value store from every result source; returns it and the sources used."""
    values.WARNINGS.clear()
    src = {
        "cls": cls or runs / "cls",
        "seg": seg or runs / "seg",
        "eff": eff or runs / "efficiency" / "results.json",
        "rob": rob or runs / "robustness" / "summary.json",
    }
    v = Store()
    frames_df = values.fill_data(v, frames, iugc)
    cls_res = values.fill_cls(v, src["cls"], n_boot)
    values.fill_eff(v, src["eff"])
    seg_res = values.fill_seg(v, src["seg"], cache)
    values.fill_aop(v, src["seg"], seg_res, frames_df, iugc, n_boot)
    values.fill_rob(v, src["rob"], cls_res)
    values.fill_ext469(v, ext469 or runs / "ext469" / "summary.json")
    if not cls_res:
        values.warn(f"no classification runs under {src['cls']}")
    if not [m for m in SEG if m in seg_res]:
        values.warn(f"no segmentation runs under {src['seg']}")
    values.fill_e2e(v, src["cls"], src["seg"])
    values.fill_cv(v, runs / "cls_cv" / "summary_cv.json", runs / "cls_authorsplit" / "author_split_check.json")
    return v, src


def apply_registry(v: Store, keys_file: Path, sections: Path, frames: Path) -> dict[str, str]:
    """Resolve registered keys into the store and record unresolved, unregistered and missing keys as warnings."""
    frames_df = read_frames_csv(frames) if frames.exists() else None
    reg = registered_keys(keys_file)
    for k, meaning in reg.items():
        val = resolve(v, k, meaning, frames_df)
        if val is not None:
            v.put(k, val)
        elif k not in v:
            values.warn(f"registered key unresolved: {k}")
    for k, files in sorted(used_keys(sections).items()):
        if k not in reg:
            values.warn(f"\\res{{{k}}} used in {', '.join(sorted(files))} but not registered in RESULT_KEYS.md")
        if k not in v:
            values.warn(f"\\res{{{k}}} used in {', '.join(sorted(files))} has no value yet")
    return reg


def write_outputs(v: Store, out_dir: Path, src: dict[str, Path], keys: list[str]) -> None:
    """Write results_values.tex and the table files."""
    lines = [
        tables.HDR.strip(),
        f"%% sources: {json.dumps({k: portable_path(p) for k, p in src.items()})}",
        f"%% {len(keys)} keys; {len(values.WARNINGS)} warnings",
        "",
    ]
    lines += [r"\resdef{" + k + "}{" + v[k] + "}" for k in keys]
    write_text(out_dir / "results_values.tex", "\n".join(lines) + "\n")
    t = out_dir / "tables"
    write_text(t / "table1.tex", tables.table1(v))
    write_text(t / "table1_cv.tex", tables.table1cv(v))
    write_text(t / "table2.tex", tables.table2(v))
    write_text(t / "table3.tex", tables.table3(v))
    write_text(t / "table3_auc_all.tex", tables.table3_all(v))
    write_text(t / "table4.tex", tables.table4(v))
    write_text(t / "table5.tex", tables.table5(v))


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULT_RUNS, help="root holding cls/, seg/, efficiency/, ...")
    ap.add_argument("--cls", type=Path, default=None)
    ap.add_argument("--seg", type=Path, default=None)
    ap.add_argument("--seg-cache", type=Path, default=None, help="default: $FIN_SEG_CACHE")
    ap.add_argument("--eff", type=Path, default=None)
    ap.add_argument("--rob", type=Path, default=None)
    ap.add_argument("--ext469", type=Path, default=None, help="summary.json of fin ext469")
    ap.add_argument("--frames", type=Path, default=SPLITS / "frames.csv")
    ap.add_argument("--iugc", type=Path, default=None, help="default: $FIN_DATA/iugc/processed/aop.csv")
    ap.add_argument("--paper-dir", type=Path, default=None, help="manuscript folder (default: $FIN_PAPER)")
    ap.add_argument("--keys", type=Path, default=None, help="default: <paper-dir>/RESULT_KEYS.md")
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="results_values.tex and tables/ (default: <paper-dir>/sections)"
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--only-registered", action="store_true", help="write only keys listed in RESULT_KEYS.md")
    ap.add_argument("--dump", type=Path, default=None, help="write every computed key/value to this JSON file")
    a = ap.parse_args(argv)
    paper = paper_dir(a.paper_dir)
    a.keys = a.keys or paper / "RESULT_KEYS.md"
    a.out_dir = a.out_dir or paper / "sections"
    setup_logging()
    v, src = compute(
        a.runs,
        a.frames,
        a.iugc or iugc_csv(),
        a.seg_cache or seg_cache_dir(),
        a.n_boot,
        a.cls,
        a.seg,
        a.eff,
        a.rob,
        a.ext469,
    )
    reg = apply_registry(v, a.keys, paper / "sections", a.frames)
    keys = [k for k in sorted(v) if (k in reg or not a.only_registered)]
    write_outputs(v, a.out_dir, src, keys)
    if a.dump:
        write_json(a.dump, {k: v[k] for k in sorted(v)})
    log.info("wrote %d keys -> %s; tables -> %s", len(keys), a.out_dir / "results_values.tex", a.out_dir / "tables")
    for w in values.WARNINGS:
        log.warning("%s", w)


if __name__ == "__main__":
    main()
