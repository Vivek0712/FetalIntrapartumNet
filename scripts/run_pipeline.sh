#!/bin/sh
# End-to-end reproduction of every result of the paper. Every stage is resumable: finished runs are skipped.
#
#   scripts/run_pipeline.sh [stage ...]        default: all stages in order
#
# Stages
#   data             fetch the pinned third-party code; prepare JNU-IFM, the FH-PS-AoP matching and IUGC 2024
#                    (needs the downloads, docs/DATA.md)
#   cls              7 classifiers x 3 seeds on the locked split
#   cls_cv           7 classifiers x 5 patient-grouped folds (1 seed)
#   cls_authorsplit  DeepFetalNet and MobileNetV3-Large on the video-level split
#   seg              4 segmenters (seeds per configs/segmentation.yaml), AoP ceiling
#   ext469           second external AoP set (IUGC 2025 release; needs FIN_EXT469)
#   e2e              end-to-end gate -> segmenter -> AoP on the JNU-IFM test split
#   robustness       corruption benchmark of every classifier run
#   efficiency       parameters, GMACs, latency and throughput of the 11 models
#   report           copy small outputs into results/ (portable paths), then the README tables and figures
#                    (docs/assets); with FIN_PAPER set, also the manuscript values, tables and figures
#
# Environment
#   FIN_WORK         working folder for data, runs and caches (default: ./work)
#   PYTHON           interpreter with the package installed (default: python)
#   JNU_RAW          JNU-IFM us_data folder              (data stage)
#   FHPSAOP_DIR      FH-PS-AoP release with image_mha/ and label_mha/   (data stage)
#   IUGC24_ROOT      IUGC 2024 release root               (data stage, ext469 mask floor)
#   FIN_EXT469       IUGC 2025 release folder (label.csv + images)       (ext469 stage)
#   FIN_PAPER        LaTeX manuscript folder (RESULT_KEYS.md, sections/, figs/); optional (report stage)
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
export FIN_WORK="${FIN_WORK:-$REPO/work}"
export FIN_DATA="${FIN_DATA:-$FIN_WORK/data}"
export FIN_RUNS="${FIN_RUNS:-$FIN_WORK/runs}"
PY="${PYTHON:-python}"
fin() { "$PY" -m fin "$@"; }
say() { echo "== $(date '+%F %T') $*"; }

stage_data() {
  fin fetch-third-party
  P="$FIN_DATA/jnu/processed"
  if [ ! -f "$P/frames.csv" ]; then
    : "${JNU_RAW:?set JNU_RAW to the JNU-IFM us_data folder}" "${FHPSAOP_DIR:?set FHPSAOP_DIR to the FH-PS-AoP release}"
    fin jnu build "$JNU_RAW" "$P"
    fin match-aop "$P" "$FHPSAOP_DIR"
  fi
  if [ ! -f "$FIN_DATA/iugc/processed/aop.csv" ]; then
    : "${IUGC24_ROOT:?set IUGC24_ROOT to the IUGC 2024 release root}"
    fin build-iugc "$IUGC24_ROOT" "$FIN_DATA/iugc/processed"
  fi
  fin cache cls
  fin cache seg
}

stage_cls() { fin schedule cls; }
stage_cls_cv() { fin schedule cv; }
stage_cls_authorsplit() { fin schedule authorsplit; }
stage_seg() { fin schedule seg; }

stage_ext469() {
  : "${FIN_EXT469:?set FIN_EXT469 to the IUGC 2025 release folder}"
  if [ -n "${IUGC24_ROOT:-}" ]; then fin ext469 "$FIN_EXT469" --iugc-root "$IUGC24_ROOT"; else fin ext469 "$FIN_EXT469"; fi
}

stage_e2e() { fin e2e; }
stage_robustness() { fin robustness; }
stage_efficiency() { fin efficiency; }

stage_report() {
  # small outputs of the reported run families only (metrics, logs, per-frame CSVs, predictions; no checkpoints or
  # caches), with machine-specific paths replaced by $FIN_WORK; then the README tables and figures
  fin collect-results
  fin readme-table
  fin repo-figures
  if [ -n "${FIN_PAPER:-}" ]; then
    fin fill-results
    fin supp-tables
    fin figures
    fin method-figures
  fi
}

STAGES="${*:-data cls cls_cv cls_authorsplit seg ext469 e2e robustness efficiency report}"
for s in $STAGES; do
  case "$s" in
    data|cls|cls_cv|cls_authorsplit|seg|ext469|e2e|robustness|efficiency|report) say "$s"; "stage_$s" ;;
    *) echo "unknown stage: $s" >&2; exit 2 ;;
  esac
done
say "done"
