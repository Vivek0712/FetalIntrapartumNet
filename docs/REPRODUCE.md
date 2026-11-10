# Reproducing the paper

Three levels, from minutes to about 16 GPU-hours of training (5 h 13 min of wall-clock time on four GPUs).

## 1. Results summary and figures from the committed outputs (minutes)

Needs only the installed package:

```bash
fin readme-table                            # results tables of README.md from results/runs and splits/frames.csv
fin repo-figures --only pipeline charts     # docs/assets/*.svg: pipeline diagram and result charts
fin repo-figures --only example             # docs/assets/example_aop.png (needs the segmentation cache, fin cache seg)
```

Expected: `git diff --stat README.md docs/assets` is empty.

The manuscript itself is not part of this repository. Its values (one LaTeX macro per number), tables and figures
are written by the same code into a manuscript folder given by `--paper-dir` or `FIN_PAPER` (a folder holding
`RESULT_KEYS.md`, `sections/` and `figs/`):

```bash
export FIN_PAPER=/path/to/manuscript
fin fill-results     # $FIN_PAPER/sections/results_values.tex and tables/table*.tex from results/runs
fin supp-tables      # $FIN_PAPER/sections/tables/supp_results.tex
fin figures          # result figures (Grad-CAM needs the classification checkpoints)
fin method-figures   # method figures (needs the prepared JNU-IFM data)
```

`fin fill-results` also needs the prepared segmentation cache (`fin cache seg`, docs/DATA.md) for the pixel-accuracy
and image-blind reference rows.

## 2. Re-evaluate checkpoints (an hour)

With checkpoints in `$FIN_RUNS/<cls|seg>/<model>/seed<k>/best.pt` (from your own training run), every evaluation is
deterministic for a given checkpoint:

```bash
fin eval-cls --run $FIN_RUNS/cls/deepfetalnet/seed0 --force
fin eval-seg --run $FIN_RUNS/seg/fatnet/seed0
fin eval-seg --ceiling $FIN_RUNS/seg/ceiling.json
fin robustness --force
fin ext469 $FIN_EXT469 --iugc-root $IUGC24_ROOT
fin e2e
fin check-geometry                       # adapted vs unchanged challenge drawline_AOD on all GT masks
```

Compare the outputs with `results/runs/` (e.g. `metrics_test.json`, `metrics.json`, `per_image_*.csv`).

## 3. Full reproduction (about 16 GPU-hours of training; 5 h 13 min on four RTX 3090)

```bash
export FIN_WORK=$PWD/work
export JNU_RAW=/path/to/us_data FHPSAOP_DIR=/path/to/fhpsaop IUGC24_ROOT=/path/to/iugc24 FIN_EXT469=/path/to/iugc25
scripts/run_pipeline.sh                     # all stages; or name stages, e.g. scripts/run_pipeline.sh cls seg
```

The reported runs were trained and evaluated on one host with four NVIDIA GeForce RTX 3090 GPUs (CUDA 13.2,
Intel Xeon E5-2673 v4; [ENVIRONMENT.md](ENVIRONMENT.md)), one run per GPU. The efficiency stage ran separately on an
Apple M4 Pro, where Table 2 of the paper reports MPS and CPU timings.

| Stage | Command(s) | Output | Recorded time |
|---|---|---|---|
| data | `fin fetch-third-party`, `fin jnu build`, `fin match-aop`, `fin build-iugc`, `fin cache cls|seg` | `$FIN_DATA` | 9 min (CPU) |
| cls | `fin schedule cls` (7 models x seeds 0-2, 21 runs) | `$FIN_RUNS/cls` | 2.3 GPU-hours of epochs; 3 to 12 min per run |
| cls_cv | `fin schedule cv` (7 models x 5 folds, seed 0, 35 runs) | `$FIN_RUNS/cls_cv` | 4.7 GPU-hours of epochs; 1 to 26 min per run |
| cls_authorsplit | `fin schedule authorsplit` (2 models x 3 seeds, 6 runs) | `$FIN_RUNS/cls_authorsplit` | 1.9 GPU-hours of epochs; 5 to 29 min per run |
| seg | `fin schedule seg` (4 segmenters x seeds 0-2, 12 runs) | `$FIN_RUNS/seg` | 6.8 GPU-hours of epochs; 19 to 51 min per run |
| ext469 | `fin ext469` | `$FIN_RUNS/ext469` | under 1 min |
| e2e | `fin e2e` | `$FIN_RUNS/e2e` | seconds |
| robustness | `fin robustness` | `$FIN_RUNS/robustness` | 2 min (one GPU) |
| efficiency | `fin efficiency` | `$FIN_RUNS/efficiency/results.json` | under 20 min on the Apple M4 Pro (at most 20 s of throughput passes per device and model, 120 latency calls per device and model) |
| report | `fin collect-results` (small outputs to `results/runs`, local paths written as `$FIN_WORK/...`), `fin readme-table`, `fin repo-figures`; with `FIN_PAPER` set, also the manuscript values, tables and figures | `results/`, `README.md`, `docs/assets/` | minutes |

The GPU-hours are the training and validation epoch times summed from the per-epoch records in `results/runs`
(`epochs.jsonl` and `history.json` for classification, `log.jsonl` for segmentation); test-set evaluation adds
under a minute per run. Per run and model (minutes of epochs, minimum to maximum over the runs of that model):

| Model | cls (3 runs) | cls_cv (5 runs) | cls_authorsplit (3 runs) |
|---|---:|---:|---:|
| DeepFetalNet | 2.7-4.1 | 8.3-26.3 | 4.8-28.8 |
| PSFHSP-Net | 3.2-5.9 | 1.3-2.9 | |
| ResNet-152 | 7.1-8.4 | 4.2-8.0 | |
| SqueezeNet 1.1 | 2.6-4.7 | 4.0-16.6 | |
| ShuffleNetV2 1.0x | 6.7-10.8 | 7.4-15.2 | |
| MobileNetV2 | 6.4-12.1 | 4.5-12.9 | |
| MobileNetV3-Large | 6.5-9.5 | 2.7-4.8 | 20.3-25.7 |

| Segmenter (3 runs each) | Minutes of epochs per run | Epochs per run |
|---|---:|---:|
| AoP-GAU-Net | 25.5-37.6 | 26-36 |
| U-Net | 18.6-37.3 | 14-32 |
| FAT-Net | 23.7-51.3 | 17-36 |
| MALUNet | 24.1-47.2 | 18-42 |

Run times vary with early stopping (number of epochs) and with the load on the shared host: the runs of the four
GPUs shared one CPU, and the mean epoch time of the same classifier differed by a factor of 2.2 to 6.4 between its
runs (`cls`, `cls_cv` and `cls_authorsplit` together). The four-GPU wall-clock time, from the start of the `data` stage to the end of `robustness`, was 5 h 13 min.

### Several GPUs on one host

`fin schedule` runs one training process at a time. To use several GPUs, start one schedule per GPU with disjoint
models or seeds, and give each process a share of the CPU threads. Training does its data handling and
augmentation in the main process (`num_workers=0`), so concurrent runs without a thread limit compete for the CPU
and slow each other down. Set `OMP_NUM_THREADS` (and `MKL_NUM_THREADS`) to the
number of CPU cores divided by the number of concurrent runs:

```bash
export FIN_WORK=$PWD/work FIN_RUNS=$PWD/work/runs
export OMP_NUM_THREADS=$(( $(nproc) / 4 )) MKL_NUM_THREADS=$(( $(nproc) / 4 ))
scripts/run_pipeline.sh data                                  # caches first, so the schedules do not build them in parallel
fin eval-seg --ceiling $FIN_RUNS/seg/ceiling.json             # written once, before the parallel segmentation schedules
CUDA_VISIBLE_DEVICES=0 fin schedule seg --models unet   > seg0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 fin schedule seg --models fatnet > seg1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 fin schedule seg --models aopgaunet > seg2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 fin schedule seg --models malunet   > seg3.log 2>&1 &
wait
fin schedule seg      # skips the finished runs and writes the summary over all models
```

The same pattern works for `fin schedule cls`, `cv` and `authorsplit` (`--models`, `--seeds`). Each process writes
its own run folders; the final sequential call writes the summary that a single schedule would write. On Linux/CUDA every command
runs unchanged (`--device auto` selects CUDA).

### Apple silicon (MPS)

The same commands run on Apple MPS (`--device auto` selects MPS when CUDA is absent), one run after another. The efficiency benchmark records
accelerator timings under the device it finds: on MPS they fill the MPS columns of Table 2. On a CUDA machine it
records them under `cuda` instead: `fin fill-results` stores them as `eff.<model>.lat.cuda` /
`eff.<model>.thr.cuda`, warns that the MPS columns of Table 2 have no values, and leaves the MPS memory keys empty
(MPS only). To reproduce Table 2, run `fin efficiency` on Apple silicon.

### Single runs

```bash
fin train-cls --model deepfetalnet --seed 0 --out $FIN_RUNS/cls/deepfetalnet/seed0
fin eval-cls  --run $FIN_RUNS/cls/deepfetalnet/seed0
fin train-seg --model aopgaunet --seed 0 --out $FIN_RUNS/seg/aopgaunet/seed0
fin eval-seg  --run $FIN_RUNS/seg/aopgaunet/seed0
FIN_SPLIT_FILE=splits/cv/fold0.csv fin train-cls --model mobilenetv3 --seed 0 --out $FIN_RUNS/cls_cv/fold0/mobilenetv3/seed0
fin aggregate cls && fin aggregate seg && fin aggregate cv && fin aggregate author-split
```

Every hyperparameter default comes from `configs/*.yaml`; command-line options override them for experiments.
Runs are resumable (`last.pt`) and skipped once finished (`done.json`, `train_done.json`).

### What to expect when retraining

CUDA and MPS kernels are not bit-exact, so retrained checkpoints differ slightly from the reported ones (trained
with CUDA on RTX 3090 GPUs), and the seed-level metrics move within roughly their reported seed spread. On CPU, training is bit-for-bit reproducible.
