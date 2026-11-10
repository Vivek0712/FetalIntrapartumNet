"""Efficiency benchmark of the seven classifiers and four segmenters (``configs/efficiency.yaml``).

Per model: trainable parameters; GMACs from thop (batch 1), fvcore (cross-check; one fused multiply-add counted
as one op, i.e. also MACs) and the exact per-call counter ``fin.efficiency.opcount`` (the figure the paper reports);
batch-1 latency on the GPU backend (MPS or CUDA) and on the CPU (20 warm-up + 100 timed passes, median and IQR,
device synchronised around every timed call); throughput at batch 16; MPS weight memory and the driver-allocated
increase after one forward pass. Models are random-initialised (weights change neither counts nor timing);
classifiers take 1x3x224x224, segmenters 1x1x256x256.

    fin efficiency --out RUNS/efficiency/results.json [--only NAME ...]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import logging
import os
import platform
import statistics
import subprocess
import time
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fin.config import empty_cache, load_config, pick_device, runs_dir, synchronize
from fin.efficiency.opcount import count_macs
from fin.io import setup_logging, write_json
from fin.models import CLASSIFIERS, SEGMENTERS, build_classifier, build_segmenter

log = logging.getLogger(__name__)


def registry(cfg: dict[str, Any]) -> list[tuple[str, str, Callable[[], nn.Module], tuple[int, ...]]]:
    """(task, display name, constructor, input shape) of every benchmarked model."""
    entries: list[tuple[str, str, Callable[[], nn.Module], tuple[int, ...]]] = []
    for key, spec in CLASSIFIERS.items():
        entries.append(
            (
                "classification",
                spec.display,
                (lambda k=key: build_classifier(k, init="scratch")[0]),
                tuple(cfg["classifier_input"]),
            )
        )
    for key, name in SEGMENTERS.items():
        entries.append(
            ("segmentation", name, (lambda k=key: build_segmenter(k, pretrained=False)), tuple(cfg["segmenter_input"]))
        )
    return entries


def n_params(m: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def thop_macs(m: nn.Module, x: torch.Tensor) -> float:
    """MACs counted by thop (known artefacts: transposed convolutions and shared modules are over-counted)."""
    from thop import profile

    mm = copy.deepcopy(m).eval()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        macs, _ = profile(mm, inputs=(x,), verbose=False)
    return float(macs)


def fvcore_macs(m: nn.Module, x: torch.Tensor) -> tuple[float, dict[str, int]]:
    """fvcore FlopCountAnalysis total (one multiply-add = one op) and its unsupported operations."""
    from fvcore.nn import FlopCountAnalysis

    mm = copy.deepcopy(m).eval()
    with torch.no_grad():
        fca = FlopCountAnalysis(mm, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = float(fca.total())
        unsupported = dict(fca.unsupported_ops())
    return total, {k: int(v) for k, v in unsupported.items()}


def time_model(
    m: nn.Module, shape: tuple[int, ...], dev: torch.device, warm: int = 20, reps: int = 100
) -> dict[str, Any]:
    """Batch latency in ms: median, quartiles, IQR, mean, SD and minimum over ``reps`` timed passes."""
    x = torch.randn(*shape, device=dev)
    ts = []
    with torch.inference_mode():
        for _ in range(warm):
            m(x)
        synchronize(dev)
        for _ in range(reps):
            synchronize(dev)
            t0 = time.perf_counter()
            m(x)
            synchronize(dev)
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    q = statistics.quantiles(ts, n=4)
    return {
        "median_ms": statistics.median(ts),
        "q1_ms": q[0],
        "q3_ms": q[2],
        "iqr_ms": q[2] - q[0],
        "mean_ms": statistics.fmean(ts),
        "sd_ms": statistics.stdev(ts),
        "min_ms": ts[0],
        "n": reps,
        "warmup": warm,
    }


def throughput(
    m: nn.Module,
    shape: tuple[int, ...],
    dev: torch.device,
    batch: int = 16,
    warm: int = 3,
    reps: int = 30,
    budget_s: float = 20.0,
) -> dict[str, Any]:
    """Images per second at ``batch``: up to ``reps`` passes, stopping once ``budget_s`` has elapsed (min 3)."""
    x = torch.randn(batch, *shape[1:], device=dev)
    with torch.inference_mode():
        for _ in range(warm):
            m(x)
        synchronize(dev)
        t0 = time.perf_counter()
        n = 0
        while n < reps:
            m(x)
            synchronize(dev)
            n += 1
            if n >= 3 and time.perf_counter() - t0 > budget_s:
                break
        dt = time.perf_counter() - t0
    return {"images_per_s": batch * n / dt, "batch": batch, "reps": n, "warmup": warm}


def mps_peak_mem(m: nn.Module, shape: tuple[int, ...], batch: int = 1) -> dict[str, Any] | None:
    """MPS weight memory and driver-allocated increase (MB) after one forward pass (approximate peak)."""
    if not torch.backends.mps.is_available():
        return None
    torch.mps.empty_cache()
    torch.mps.synchronize()
    base_cur = torch.mps.current_allocated_memory()
    base_drv = torch.mps.driver_allocated_memory()
    mm = copy.deepcopy(m).to("mps").eval()
    torch.mps.synchronize()
    w_cur = torch.mps.current_allocated_memory()
    x = torch.randn(batch, *shape[1:], device="mps")
    with torch.inference_mode():
        y = mm(x)
    torch.mps.synchronize()
    drv = torch.mps.driver_allocated_memory()
    out = {"weights_MB": (w_cur - base_cur) / 2**20, "driver_alloc_delta_MB": (drv - base_drv) / 2**20, "batch": batch}
    del mm, x, y
    torch.mps.empty_cache()
    return out


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"n/a ({e})"


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "n/a"


def hardware() -> dict[str, Any]:
    """Machine and software description recorded with the benchmark."""
    hw: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "thop": _ver("thop"),
        "fvcore": _ver("fvcore"),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_num_threads": torch.get_num_threads(),
    }
    if platform.system() == "Darwin":
        hw.update(
            cpu=_sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
            ncpu=_sh(["sysctl", "-n", "hw.ncpu"]),
            perf_cores=_sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
            eff_cores=_sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"]),
            mem_bytes=_sh(["sysctl", "-n", "hw.memsize"]),
            macos=_sh(["sw_vers", "-productVersion"]) + " (" + _sh(["sw_vers", "-buildVersion"]) + ")",
        )
    else:
        hw.update(
            cpu=platform.processor() or platform.machine(),
            ncpu=str(os.cpu_count()),
            mem_bytes=str(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
            os=platform.platform(),
        )
    return hw


def benchmark(out: Path, only: list[str] | None = None) -> dict[str, Any]:
    """Run the benchmark and write ``results.json``."""
    cfg = load_config("efficiency")
    lat, thr = cfg["latency"], cfg["throughput"]
    torch.manual_seed(cfg["seed"])
    best = pick_device("auto")
    gpu = best.type if best.type != "cpu" else None
    res: dict[str, Any] = {
        "hardware": hardware(),
        "protocol": {
            "latency": f"batch 1, {lat['warmup']} warm-up + {lat['reps']} timed forward passes, torch.inference_mode, "
            "eval mode, fp32, random input, device synchronised before and after each timed call; "
            "median and IQR",
            "throughput": f"batch {thr['batch']}, {thr['warmup']} warm-up + up to {thr['reps']} timed passes (stops "
            f"after {thr['budget_s']:g} s, min 3), images/s = {thr['batch']}*n/elapsed",
            "macs": "thop.profile at batch 1; fvcore FlopCountAnalysis at batch 1 (one multiply-add = one op, i.e. "
            "MACs); per-call counter fin.efficiency.opcount (conv, transposed conv on its input grid, linear, "
            "matmul, attention; every executed call counted once); FLOPs = 2 x MACs",
            "params": "trainable parameters (requires_grad)",
            "memory": "MPS: weights (current_allocated delta) and driver_allocated delta after one batch-1 / batch-16 "
            "forward pass; approximate peak",
        },
        "models": [],
    }
    warnings.filterwarnings("ignore")
    for task, name, ctor, shape in registry(cfg):
        if only and name not in only:
            continue
        log.info("[%s] %s %s", task, name, shape)
        m = ctor().eval()
        x = torch.randn(*shape)
        r: dict[str, Any] = {"task": task, "name": name, "input_shape": list(shape), "loadavg_start": os.getloadavg()}
        r["params"] = n_params(m)
        r["params_total"] = sum(p.numel() for p in m.parameters())
        r["thop_MACs"] = thop_macs(m, x)
        r["thop_GMACs"] = r["thop_MACs"] / 1e9
        r["GFLOPs_2xMACs"] = 2 * r["thop_GMACs"]
        ex, exd = count_macs(copy.deepcopy(m), x)
        r["percall_GMACs"] = ex / 1e9
        r["percall_breakdown_GMACs"] = {k: v / 1e9 for k, v in exd.items()}
        try:
            fv, unsup = fvcore_macs(m, x)
            r["fvcore_GMACs"] = fv / 1e9
            r["fvcore_unsupported_ops"] = unsup
        except (RuntimeError, ValueError, TypeError) as e:
            r["fvcore_GMACs"] = None
            r["fvcore_error"] = repr(e)[:300]
        r["latency"], r["throughput_b16"] = {}, {}
        for dev_name in ([gpu] if gpu else []) + ["cpu"]:
            dev = torch.device(dev_name)
            md = copy.deepcopy(m).to(dev).eval()
            r["latency"][dev_name] = time_model(md, shape, dev, lat["warmup"], lat["reps"])
            r["throughput_b16"][dev_name] = throughput(
                md, shape, dev, thr["batch"], thr["warmup"], thr["reps"], thr["budget_s"]
            )
            del md
            empty_cache(dev)
        r["loadavg_end"] = os.getloadavg()
        r["mps_memory_b1"] = mps_peak_mem(m, shape, 1)
        r["mps_memory_b16"] = mps_peak_mem(m, shape, 16)
        log.info(
            "%s params %d GMACs thop %.3f per-call %.3f latency %s",
            name,
            r["params"],
            r["thop_GMACs"],
            r["percall_GMACs"],
            {d: round(v["median_ms"], 2) for d, v in r["latency"].items()},
        )
        res["models"].append(r)
    write_json(out, res, indent=2)
    return res


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="default: $FIN_RUNS/efficiency/results.json")
    ap.add_argument("--only", nargs="*", help="display names of the models to benchmark")
    a = ap.parse_args(argv)
    setup_logging()
    benchmark(a.out or runs_dir() / "efficiency" / "results.json", a.only)


if __name__ == "__main__":
    main()
