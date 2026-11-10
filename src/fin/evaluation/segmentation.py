"""Evaluate a trained segmenter on the JNU-IFM SPHead test split (379 frames, 7 women, 8 videos) and on the external
IUGC 2024 test set (300 frames, one per video). Metric definitions: ``fin.evaluation.seg_metrics``.

The AoP is measured on the 256x256 predicted masks with the challenge geometry (``fin.geometry``).
References
    JNU  : aop_ref_challenge_code (primary; challenge geometry on the challenge label) and aop_ref (secondary,
           independent implementation). Both mask-derived; JNU-IFM has no sonographer AoP.
    IUGC : aop_official (landmark AoP of the release; primary) and aop_mask_challenge_code (secondary).
AoP-GAU-Net's regression head is evaluated separately from its mask-derived AoP.

Outputs in the run directory: metrics.json, per_image_{jnu_test,iugc}.csv, preds_{jnu_test,iugc}.npz.

    fin eval-seg --run RUNS/seg/unet/seed0
    fin eval-seg --ceiling RUNS/seg/ceiling.json        (challenge geometry on the GT masks vs the references)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fin.config import load_config, pick_device, seg_cache_dir
from fin.evaluation.seg_metrics import aop_stats, per_image, summarise_seg
from fin.geometry import aop_from_masks
from fin.io import (
    portable_path,
    read_frames_csv,
    read_json,
    read_json_if_exists,
    setup_logging,
    write_csv,
    write_json,
    write_npz,
)
from fin.models import build_segmenter, split_output

log = logging.getLogger(__name__)

EVAL_SETS = ("jnu_test", "iugc")
NATIVE_SAMPLING = {"jnu_test": (1026 / 256, 1295 / 256), "iugc": (512 / 256, 512 / 256)}


@torch.no_grad()
def predict(
    model: nn.Module, img: np.ndarray, dev: torch.device, bs: int = 16
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(N, H, W) uint8 images -> (uint8 label maps, AoP head in degrees or None, fp16 softmax probabilities)."""
    model.eval()
    labs, aops, probs = [], [], []
    for i in range(0, len(img), bs):
        x = torch.from_numpy(np.asarray(img[i : i + bs], np.float32) / 255.0)[:, None].to(dev)
        seg, aop = split_output(model(x))
        p = torch.softmax(seg.float(), 1)
        labs.append(p.argmax(1).to(torch.uint8).cpu().numpy())
        probs.append(p.to(torch.float16).cpu().numpy())
        if aop is not None:
            aops.append(aop.float().cpu().numpy())
    return np.concatenate(labs), (np.concatenate(aops) if aops else None), np.concatenate(probs)


def jnu_test_rows(idx: pd.DataFrame) -> np.ndarray:
    """Cache rows of the JNU-IFM SPHead test frames, in cache order."""
    return np.where((idx.split == "test") & (idx.category == 6))[0]


def load_set(name: str, cache: Path | None = None) -> dict[str, Any]:
    """Frame table, images, masks, primary/secondary references and bootstrap groups of one evaluation set."""
    cache = Path(cache or seg_cache_dir())
    if name == "jnu_test":
        idx = read_frames_csv(cache / "jnu_index.csv")
        sel = jnu_test_rows(idx)
        img, msk = np.load(cache / "jnu_img.npy", mmap_mode="r"), np.load(cache / "jnu_mask.npy", mmap_mode="r")
        d = idx.iloc[sel].reset_index(drop=True)
        refs = {"primary": d.aop_ref_challenge_code.to_numpy(float), "secondary": d.aop_ref.to_numpy(float)}
        refnames = {"primary": "aop_ref_challenge_code", "secondary": "aop_ref"}
        groups = d.video_id.to_numpy()
    else:
        d = pd.read_csv(cache / "iugc_index.csv")
        sel = np.arange(len(d))
        img, msk = np.load(cache / "iugc_img.npy", mmap_mode="r"), np.load(cache / "iugc_mask.npy", mmap_mode="r")
        refs = {"primary": d.aop_official.to_numpy(float), "secondary": d.aop_mask_challenge_code.to_numpy(float)}
        refnames = {"primary": "aop_official (IUGC 2024 landmarks)", "secondary": "aop_mask_challenge_code"}
        groups = d.frame_id.to_numpy()
    return {
        "df": d,
        "rows": sel,
        "img": np.ascontiguousarray(img[sel]),
        "mask": np.ascontiguousarray(msk[sel]),
        "refs": refs,
        "refnames": refnames,
        "groups": groups,
    }


def ceiling(out: Path, cache: Path | None = None) -> dict[str, Any]:
    """AoP from the ground-truth masks against both references, for both sets."""
    n_boot = load_config("segmentation")["evaluation"]["n_boot"]
    res: dict[str, Any] = {}
    for s in EVAL_SETS:
        e = load_set(s, cache)
        a = aop_from_masks(e["mask"])
        res[s] = {k: {"ref": e["refnames"][k], **aop_stats(a, e["refs"][k], e["groups"], n_boot)} for k in e["refs"]}
        res[s]["n_frames"] = len(e["df"])
    write_json(out, res)
    return res


def evaluate(run: Path, device: str = "auto", cache: Path | None = None) -> dict[str, Any]:
    """Evaluate one segmentation run on both sets and write its outputs."""
    ecfg = load_config("segmentation")["evaluation"]
    meta = read_json(run / "meta.json")
    model = build_segmenter(meta["model"], pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    dev = pick_device(device)
    model.to(dev)
    out: dict[str, Any] = {"model": meta["model"], "params": meta["params"], "run": portable_path(run)}
    train_done = read_json_if_exists(run / "train_done.json")
    if train_done is not None:
        out["train"] = train_done
    for s in EVAL_SETS:
        e = load_set(s, cache)
        lab, head, _ = predict(model, e["img"], dev, ecfg["batch_size"])
        r = per_image(lab, e["mask"], NATIVE_SAMPLING[s])
        seg = summarise_seg(r)
        am = aop_from_masks(lab)
        refs, rn, groups = e["refs"], e["refnames"], e["groups"]
        res: dict[str, Any] = {
            "n_frames": len(e["df"]),
            "seg": seg,
            "aop_mask": {
                k: {"ref": rn[k], **aop_stats(am, refs[k], groups, ecfg["n_boot"], ecfg["bootstrap_seed"])}
                for k in refs
            },
        }
        if head is not None:
            res["aop_head"] = {
                k: {"ref": rn[k], **aop_stats(head, refs[k], groups, ecfg["n_boot"], ecfg["bootstrap_seed"])}
                for k in refs
            }
        out[s] = res
        pi = pd.DataFrame(
            {
                "frame_id": e["df"].frame_id,
                **{k: v for k, v in r.items() if not k.startswith("pooled")},
                "aop_mask": am,
                "ref_primary": refs["primary"],
                "ref_secondary": refs["secondary"],
            }
        )
        if head is not None:
            pi["aop_head"] = head
        write_csv(run / f"per_image_{s}.csv", pi)
        write_npz(
            run / f"preds_{s}.npz",
            labels=lab,
            aop_mask=am,
            aop_head=head if head is not None else np.full(len(e["df"]), np.nan),
            frame_id=e["df"].frame_id.to_numpy().astype(str),
        )
        log.info(
            "%s Dice FH %.4f PS %.4f | AoP MAE %.2f fail %d",
            s,
            seg["dice_FH_mean"],
            seg["dice_PS_mean"],
            res["aop_mask"]["primary"].get("mae", np.nan),
            res["aop_mask"]["primary"]["n_fail"],
        )
    write_json(run / "metrics.json", out, default=float)
    return out


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--ceiling", type=Path)
    ap.add_argument("--cache", type=Path, default=None, help="default: $FIN_SEG_CACHE")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    setup_logging()
    if not (a.run or a.ceiling):
        ap.error("give --run and/or --ceiling")
    if a.ceiling:
        ceiling(a.ceiling, a.cache)
    if a.run:
        evaluate(a.run, a.device, a.cache)


if __name__ == "__main__":
    main()
