"""Train one segmentation model on the locked JNU-IFM split (``configs/segmentation.yaml``).

Identical budget for every model: AdamW (lr 3e-4, weight decay 1e-4), cosine annealing over at most 60 epochs,
batch 8 (drop_last), early stopping with patience 10 on the validation per-image mean foreground Dice (mean of
FH and PS), best checkpoint kept. Train-only augmentation from ``fin.data.datasets``. Input 1 x 256 x 256 in
[0, 1]; fp32. Loss: CE + Dice for the baselines; AoP-GAU-Net: ``CombinedLoss`` with the AoP term against
``aop_ref_challenge_code`` (frames without a reference contribute no AoP loss).
Training frames: ``seg`` = SPHead training frames (2994); ``seg_plus`` adds OnlySP and OnlyHead frames with their
single-structure masks. Validation = SPHead validation frames (370), used only for selection and early stopping.
Resumable: running the same command again continues from ``last.pt``; ``train_done.json`` marks completion.

    fin train-seg --model unet --seed 0 --out RUNS/seg/unet/seed0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fin.config import load_config, pick_device, require_setting, seed_everything, synchronize
from fin.data.cache import load_seg_cache
from fin.data.datasets import Augmentation, CachedSegmentationDataset
from fin.evaluation.segmentation import predict
from fin.io import append_jsonl, portable_path, setup_logging, write_json
from fin.models import SEGMENTERS, build_segmenter, count_params, split_output
from fin.training.losses import build_loss

log = logging.getLogger(__name__)


def select_rows(idx: pd.DataFrame, split: str, train_set: str = "seg") -> np.ndarray:
    """Cache rows of one split: SPHead frames, plus OnlySP/OnlyHead training frames for ``seg_plus``."""
    cats = [6] if (split != "train" or train_set == "seg") else [4, 5, 6]
    return np.where((idx.split == split) & idx.category.isin(cats))[0]


def val_score(
    model: nn.Module, img: np.ndarray, msk: np.ndarray, aop_ref: np.ndarray, dev: torch.device, bs: int = 16
) -> dict[str, float]:
    """Validation per-image Dice of FH and PS, their mean and (AoP-GAU-Net) the head's AoP MAE."""
    lab, aop, _ = predict(model, img, dev, bs)
    n = len(lab)
    d = []
    for c in (1, 2):
        p_c, g_c = lab == c, np.asarray(msk) == c
        inter = (p_c & g_c).reshape(n, -1).sum(1)
        s = p_c.reshape(n, -1).sum(1) + g_c.reshape(n, -1).sum(1)
        d.append(np.where(s > 0, 2 * inter / np.maximum(s, 1), np.nan))
    out = {"val_dice_FH": float(np.nanmean(d[0])), "val_dice_PS": float(np.nanmean(d[1]))}
    out["val_mean_fg_dice"] = (out["val_dice_FH"] + out["val_dice_PS"]) / 2
    if aop is not None:
        ok = np.isfinite(aop_ref)
        out["val_aophead_mae"] = float(np.abs(aop[ok] - aop_ref[ok]).mean())
    return out


def train(
    model_name: str, seed: int, out: Path, cfg: dict[str, Any], device: str, n_subset: int = 0, time_epochs: int = 0
) -> dict[str, Any] | None:
    """Train one run; returns ``train_done.json`` (None if already done or after a timing run)."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_done.json").exists() and not time_epochs:
        log.info("already done %s", out)
        return None
    require_setting(cfg, "optimizer.name", "AdamW")
    require_setting(cfg, "scheduler.name", "CosineAnnealingLR")
    seed_everything(seed, python=False)
    dev = pick_device(device)
    idx, img_all, msk_all = load_seg_cache()
    tr, va = select_rows(idx, "train", cfg["train_set"]), select_rows(idx, "val")
    if n_subset:
        rs = np.random.default_rng(0)
        tr, va = np.sort(rs.choice(tr, min(n_subset, len(tr)), replace=False)), va[: max(16, n_subset // 4)]
    aop_all = idx[cfg["reference_column"]].to_numpy(np.float64)
    ds = CachedSegmentationDataset(
        img_all[tr], msk_all[tr], aop_all[tr], True, seed, Augmentation.from_config(cfg["augmentation"])
    )
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=0, generator=g)
    vimg, vmsk, vaop = np.ascontiguousarray(img_all[va]), np.ascontiguousarray(msk_all[va]), aop_all[va]

    model = build_segmenter(model_name).to(dev)
    crit = build_loss(model_name, cfg["loss"]).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["scheduler"]["eta_min"])
    state: dict[str, Any] = {"epoch": 0, "best": -1.0, "best_epoch": -1, "bad": 0}
    if (out / "last.pt").exists() and not time_epochs:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        state = ck["state"]
        ds.rng.bit_generator.state = ck["aug_rng"]
        g.set_state(ck["loader_rng"])
        torch.set_rng_state(ck["torch_rng"])
        log.info("resumed at epoch %d best %.4f", state["epoch"], state["best"])
    args = {
        "model": model_name,
        "seed": seed,
        "out": portable_path(out),
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "bs": cfg["batch_size"],
        "lr": cfg["optimizer"]["lr"],
        "wd": cfg["optimizer"]["weight_decay"],
        "train_set": cfg["train_set"],
        "subset": n_subset,
        "time_epochs": time_epochs,
        "device": str(dev),
    }
    meta = {
        "model": model_name,
        "params": count_params(model),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "args": args,
        "ref_col": cfg["reference_column"],
    }
    write_json(out / "meta.json", meta)
    log.info("%s", meta)

    times: list[dict[str, float]] = []
    while state["epoch"] < cfg["epochs"] and state["bad"] < cfg["patience"]:
        model.train()
        t0 = time.time()
        tot, nb, parts = 0.0, 0, {}
        for x, y, r in dl:
            x, y, r = x.to(dev), y.to(dev), r.to(dev)
            seg, aop = split_output(model(x))
            loss, p = crit(seg, y, aop, r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v
        synchronize(dev)
        t_train = time.time() - t0
        sch.step()
        state["epoch"] += 1
        t1 = time.time()
        v = val_score(model, vimg, vmsk, vaop, dev, cfg["val_batch_size"])
        t_val = time.time() - t1
        times.append({"train_s": t_train, "val_s": t_val})
        rec = {
            "epoch": state["epoch"],
            "loss": tot / max(nb, 1),
            **{k: x / max(nb, 1) for k, x in parts.items()},
            **v,
            "lr": opt.param_groups[0]["lr"],
            "train_s": round(t_train, 1),
            "val_s": round(t_val, 1),
        }
        if not np.isfinite(rec["loss"]):
            raise RuntimeError(f"non-finite loss {rec}")
        if v["val_mean_fg_dice"] > state["best"]:
            state.update(best=v["val_mean_fg_dice"], best_epoch=state["epoch"], bad=0)
            torch.save(model.state_dict(), out / "best.pt")
        else:
            state["bad"] += 1
        rec.update(best=state["best"], best_epoch=state["best_epoch"])
        log.info("%s", rec)
        append_jsonl(out / "log.jsonl", rec)
        if time_epochs:
            if len(times) >= time_epochs:
                write_json(
                    out / "timing.json",
                    {
                        "model": model_name,
                        "n_train": int(len(tr)),
                        "bs": cfg["batch_size"],
                        "epochs": times,
                        "params": meta["params"],
                    },
                )
                return None
            continue
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sch": sch.state_dict(),
                "state": state,
                "aug_rng": ds.rng.bit_generator.state,
                "loader_rng": g.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            out / "last.pt",
        )
    done = {**state, "stopped": "patience" if state["bad"] >= cfg["patience"] else "max_epochs"}
    write_json(out / "train_done.json", done)
    if (out / "last.pt").exists():
        (out / "last.pt").unlink()
    return done


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; defaults come from ``configs/segmentation.yaml``."""
    cfg = load_config("segmentation")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(SEGMENTERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--patience", type=int, default=cfg["patience"])
    ap.add_argument("--bs", type=int, default=cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["optimizer"]["lr"])
    ap.add_argument("--wd", type=float, default=cfg["optimizer"]["weight_decay"])
    ap.add_argument("--train-set", default=cfg["train_set"], choices=["seg", "seg_plus"])
    ap.add_argument("--subset", type=int, default=0, help="smoke test: first N train / N/4 val frames")
    ap.add_argument("--time-epochs", type=int, default=0, help="timing: run this many epochs, write timing.json")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    setup_logging()
    cfg.update(epochs=a.epochs, patience=a.patience, batch_size=a.bs, train_set=a.train_set)
    cfg["optimizer"] = {**cfg["optimizer"], "lr": a.lr, "weight_decay": a.wd}
    train(a.model, a.seed, a.out, cfg, a.device, a.subset, a.time_epochs)


if __name__ == "__main__":
    main()
