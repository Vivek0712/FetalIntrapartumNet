# Environment

## Reported runs

Training and evaluation (every stage except `efficiency`):

| Item | Value |
|---|---|
| Machine | 4 x NVIDIA GeForce RTX 3090 (24 GB each, driver 595.71.05) on one Linux host with an Intel Xeon E5-2673 v4 CPU; one run per GPU |
| Python | 3.12.3 |
| PyTorch / torchvision | 2.15.0+cu132 / 0.29.2 (CUDA 13.2, cuDNN 9.26) |
| Device | CUDA for training, evaluation, the end-to-end pipeline and the robustness benchmark |
| Other packages | exact versions in `requirements.txt` (numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1, opencv-python-headless 5.0.0.93, timm 1.0.31, matplotlib 3.11.2, SimpleITK 2.5.7, thop 0.1.1.post2209072238, fvcore 0.1.5.post20221221) |

Efficiency benchmark (Table 2 of the paper; `results/runs/efficiency/results.json`):

| Item | Value |
|---|---|
| Machine | Apple M4 Pro (10 performance + 4 efficiency cores), 24 GB unified memory, macOS 15.7 (24G222) |
| Python | 3.12.11 |
| PyTorch | 2.15.0, 10 CPU threads |
| Device | Apple MPS for the accelerator columns; CPU for the CPU latency column |

Grad-CAM figures run on the CPU. Several runs sharing one host need a thread limit (`OMP_NUM_THREADS`, see
[REPRODUCE.md](REPRODUCE.md#several-gpus-on-one-host)).

## macOS (Apple silicon, MPS)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

## Linux (NVIDIA GPU, CUDA)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
CUDA_TAG=cu132            # the reported runs; use the wheel tag of your CUDA version, as listed on pytorch.org
pip install --index-url https://download.pytorch.org/whl/$CUDA_TAG torch==2.15.0 torchvision==0.29.2
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Choose the wheel tag that matches the installed CUDA driver. For a CPU-only machine (tests, reporting) use
`https://download.pytorch.org/whl/cpu`.

## Device, seeds and determinism

- `--device auto` (the default of the training, evaluation and benchmark commands) picks CUDA, then MPS, then CPU
  (`fin.config.pick_device`); Grad-CAM figures default to the CPU.
- Seeds: the split and fold searches use `np.random.default_rng(42)`; each run seeds Python, NumPy and torch with
  its seed (`fin.config.seed_everything`; segmentation runs seed NumPy and torch only), the data loader with a
  `torch.Generator` and the augmentation with its own `np.random.default_rng(seed)`. Bootstrap CIs use fixed seeds (0, and 1 for the paired differences).
- On CPU, training is bit-for-bit reproducible. CUDA and MPS kernels are not bit-exact (cuDNN is not set to
  deterministic mode), so retraining on a GPU, including the RTX 3090 of the reported runs, gives slightly different
  checkpoints and metrics than the reported ones; evaluating a given checkpoint is exact.
- `cv2.fitEllipse` perturbs a degenerate (nearly collinear) contour with OpenCV's process-wide random generator,
  so the AoP of such a mask depends on how many such fits ran earlier in the same process. Every evaluation
  command runs in its own process over a fixed frame order, so its outputs are reproducible; evaluating the same
  frames in another order or process can change the angle of such a frame.

## Paths

| Variable | Default | Content |
|---|---|---|
| `FIN_WORK` | `<repo>/work` | working folder |
| `FIN_DATA` | `$FIN_WORK/data` | prepared datasets |
| `FIN_RUNS` | `$FIN_WORK/runs` | one folder per run (checkpoints, metrics, predictions) |
| `FIN_SEG_CACHE` | `$FIN_RUNS/seg/cache` | 256 x 256 segmentation cache |
| `FIN_THIRD_PARTY` | `$FIN_WORK/third_party` | fetched third-party source files |
| `FIN_SPLIT_FILE` | unset | CSV (frame_id, split) replacing the locked split (cross-validation and sensitivity runs) |
| `FIN_EXT469` | unset | IUGC 2025 release folder |
| `FIN_REPO` | the checkout of the installed package | repository root (configs, splits, results) when `fin` is not installed in editable mode from it |
| `FIN_PAPER` | unset | LaTeX manuscript folder written by `fin fill-results`, `supp-tables`, `figures`, `method-figures` (same as `--paper-dir`) |

## Development

```bash
pip install -r requirements-dev.txt     # pytest, ruff, pre-commit (pinned)
ruff check . && ruff format --check .
fin fetch-third-party                   # lets the geometry and FAT-Net/MALUNet tests run instead of being skipped
pytest -q
pre-commit install                      # optional: ruff and whitespace hooks on commit
```

The package reads `configs/`, `splits/` and `results/` from the repository checkout, so install it in
editable mode (`pip install -e .`). A regular `pip install .` works only when `fin` runs inside the checkout or
with `FIN_REPO` set; otherwise every command stops with a message saying so.
