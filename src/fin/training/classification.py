"""Train one Standard vs Non-Standard classifier on the locked JNU-IFM split.

Protocol (identical for all seven models; ``configs/classification.yaml``)
  * data: task ``cls`` of the locked split (train 4958 / val 636 / test 630 frames), 224x224, 3-channel
    ImageNet-normalised input, train-only augmentation (``fin.data.datasets``)
  * loss: unweighted cross-entropy on two logits
  * AdamW (lr 1e-4, weight decay 1e-2); CosineAnnealingLR(T_max = epochs = 30, eta_min = 1e-6) stepped once per
    epoch
  * at most 30 epochs, early stopping with patience 7 on validation macro-F1 for every model
  * best checkpoint = highest validation macro-F1 (strict improvement, first epoch wins ties); the test split is
    never read here
  * one JSON line per epoch in ``epochs.jsonl``, full history in ``history.json``, final state in ``done.json``
  * resumable: ``last.pt`` is written every epoch and removed when the run finishes; a run directory with
    ``done.json`` is skipped

    fin train-cls --model deepfetalnet --seed 0 --out RUNS/cls/deepfetalnet/seed0
    fin train-cls ... --subset 0.05 --stop-after 1        (smoke test)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fin.config import load_config, pick_device, require_setting, seed_everything, synchronize
from fin.data.cache import attach_cls_cache
from fin.data.datasets import Augmentation, ClassificationDataset, load_frames
from fin.evaluation.cls_metrics import point_metrics
from fin.io import append_jsonl, setup_logging, write_json, write_text
from fin.models import CLASSIFIERS, build_classifier, count_params

log = logging.getLogger(__name__)


def _sci(x: float) -> str:
    """1e-06 -> 1e-6 (the form recorded in config.json)."""
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def subset(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Category-stratified fraction of a frame table (``frac >= 1`` returns it unchanged)."""
    if frac >= 1:
        return df
    return df.groupby("category", group_keys=False).sample(frac=frac, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, dev: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """(labels, p(Standard), mean cross-entropy) over a loader."""
    model.eval()
    ps, ys, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        lo = model(x)
        loss_sum += float(ce(lo, y))
        n += len(y)
        ps.append(torch.softmax(lo.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(n, 1)


def train(
    model_name: str, seed: int, out: Path, cfg: dict[str, Any], init: str, device: str, frac: float, stop_after: int
) -> dict[str, Any] | None:
    """Train one run; returns the ``done.json`` record (None when the run had already finished)."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        log.info("[skip] %s already finished", out)
        return None
    require_setting(cfg, "optimizer.name", "AdamW")
    require_setting(cfg, "scheduler.name", "CosineAnnealingLR")
    require_setting(cfg, "loss", "CrossEntropyLoss")
    seed_everything(seed)
    dev = pick_device(device)
    size, epochs = cfg["input_size"], cfg["epochs"]
    aug = Augmentation.from_config(cfg["augmentation"])

    tr_df, va_df = subset(load_frames("train", "cls"), frac, seed), subset(load_frames("val", "cls"), frac, seed)
    tr = attach_cls_cache(ClassificationDataset(tr_df, size, train=True, cache=True, seed=seed, aug=aug), size)
    va = attach_cls_cache(ClassificationDataset(va_df, size, train=False, cache=True), size)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=g, drop_last=False)
    vl = DataLoader(va, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=0)

    model, resolved_init, weights = build_classifier(model_name, init)
    model.to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    eta_min = cfg["scheduler"]["eta_min"]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    ce = nn.CrossEntropyLoss()
    thr = cfg["selection_threshold"]

    write_json(
        out / "config.json",
        {
            "model": model_name,
            "display_name": CLASSIFIERS[model_name].display,
            "seed": seed,
            "init": resolved_init,
            "weights": weights,
            "params": count_params(model),
            "epochs_max": epochs,
            "patience": cfg["patience"],
            "batch_size": cfg["batch_size"],
            "optimizer": cfg["optimizer"]["name"],
            "lr": cfg["optimizer"]["lr"],
            "weight_decay": cfg["optimizer"]["weight_decay"],
            "scheduler": f"CosineAnnealingLR(T_max={epochs}, eta_min={_sci(eta_min)}), per epoch",
            "loss": "CrossEntropyLoss (unweighted)",
            "input": f"{size}x{size}x3, ImageNet mean/std",
            "selection": f"best validation macro-F1 (threshold {thr})",
            "device": str(dev),
            "n_train": len(tr),
            "n_val": len(va),
            "subset": frac,
            "stop_after": stop_after,
            "torch": torch.__version__,
        },
        indent=2,
    )

    hist: list[dict[str, Any]] = []
    best, best_ep, bad, start = -1.0, 0, 0, 1
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        hist, best, best_ep, bad, start = ck["hist"], ck["best"], ck["best_ep"], ck["bad"], ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        g.set_state(ck["gen_rng"])
        tr.rng = ck["aug_rng"]
        write_text(out / "epochs.jsonl", "".join(json.dumps(h) + "\n" for h in hist))
        log.info("[resume] from epoch %d", ck["epoch"])

    stop_reason = "max_epochs"
    t_run = time.time()
    for ep in range(start, epochs + 1):
        model.train()
        synchronize(dev)
        t0 = time.time()
        tot = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev, dtype=torch.long)
        n = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            lo = model(x)
            loss = ce(lo, y)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(y)
            n += len(y)
            correct += (lo.detach().argmax(1) == y).sum()
        tot_f, correct_i = float(tot), int(correct)
        synchronize(dev)
        t_train = time.time() - t0
        lr_used = opt.param_groups[0]["lr"]
        sch.step()
        t1 = time.time()
        yv, pv, vloss = predict(model, vl, dev)
        t_val = time.time() - t1
        m = point_metrics(yv, pv, thr)
        improved = m["macro_f1"] > best
        if improved:
            best, best_ep, bad = m["macro_f1"], ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        rec = {
            "epoch": ep,
            "lr": lr_used,
            "train_loss": tot_f / n,
            "train_acc": correct_i / n,
            "val_loss": vloss,
            "val_acc": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_auc": m["auc"],
            "val_sens_standard": m["standard"]["recall"],
            "val_spec_standard": m["standard"]["specificity"],
            "best": improved,
            "t_train_s": t_train,
            "t_val_s": t_val,
        }
        hist.append(rec)
        append_jsonl(out / "epochs.jsonl", rec)
        write_json(out / "history.json", hist)
        log.info(
            "ep %02d loss %.4f acc %.3f | val loss %.4f acc %.3f mF1 %.4f AUC %.4f%s | train %.1fs val %.1fs",
            ep,
            rec["train_loss"],
            rec["train_acc"],
            vloss,
            m["accuracy"],
            m["macro_f1"],
            m["auc"],
            " *" if improved else "",
            t_train,
            t_val,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sch": sch.state_dict(),
                "epoch": ep,
                "hist": hist,
                "best": best,
                "best_ep": best_ep,
                "bad": bad,
                "torch_rng": torch.get_rng_state(),
                "gen_rng": g.get_state(),
                "aug_rng": tr.rng,
            },
            out / "last.pt.tmp",
        )
        os.replace(out / "last.pt.tmp", last)
        if cfg["patience"] > 0 and bad >= cfg["patience"]:
            stop_reason = f"early_stopping(patience={cfg['patience']})"
            break
        if stop_after and ep >= stop_after:
            stop_reason = f"stop_after={stop_after}"
            break

    done = {
        "best_epoch": best_ep,
        "best_val_macro_f1": best,
        "epochs_run": len(hist),
        "stop_reason": stop_reason,
        "wall_s_this_session": time.time() - t_run,
        "mean_train_epoch_s": float(np.mean([h["t_train_s"] for h in hist])),
        "mean_val_s": float(np.mean([h["t_val_s"] for h in hist])),
    }
    write_json(out / "done.json", done, indent=2)
    last.unlink()
    return done


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; defaults come from ``configs/classification.yaml``."""
    cfg = load_config("classification")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(CLASSIFIERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", default=cfg["init"], choices=["default", "pretrained", "scratch"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"], help="max epochs and cosine T_max")
    ap.add_argument("--patience", type=int, default=cfg["patience"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["optimizer"]["lr"])
    ap.add_argument("--weight-decay", type=float, default=cfg["optimizer"]["weight_decay"])
    ap.add_argument("--size", type=int, default=cfg["input_size"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of train and val frames (smoke test)")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N epochs (smoke/timing); 0 = off")
    a = ap.parse_args(argv)
    setup_logging()
    cfg.update(epochs=a.epochs, patience=a.patience, batch_size=a.batch_size, input_size=a.size)
    cfg["optimizer"] = {**cfg["optimizer"], "lr": a.lr, "weight_decay": a.weight_decay}
    done = train(a.model, a.seed, a.out, cfg, a.init, a.device, a.subset, a.stop_after)
    if done:
        log.info("done: %s", done)


if __name__ == "__main__":
    main()
