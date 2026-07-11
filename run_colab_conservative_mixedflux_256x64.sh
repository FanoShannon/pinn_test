#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${WORK_DIR}/runs_conservative_mixedflux_256x64/${RUN_TAG}}"
PYTHON="${PYTHON:-python}"

GAMMA="${GAMMA:-1.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"

CLEAN_BEST="${CLEAN_BEST:-}"
FDM_PKL="${FDM_PKL:-/content/gdrive/MyDrive/FDM_parameter_scale_0711/gamma1_k1_v42_thin_layer_catalytic_v42.pkl}"
SKIP_CONSERVATIVE="${SKIP_CONSERVATIVE:-0}"
SKIP_MIXED="${SKIP_MIXED:-0}"
CONSERVATIVE_BEST_OVERRIDE="${CONSERVATIVE_BEST_OVERRIDE:-}"

CONSERVATIVE_ARCH="multiscale_film_tracegreen_clean_conservative"
MIXED_ARCH="multiscale_film_tracegreen_clean_mixedflux"
CONSERVATIVE_DIR="${RUN_ROOT}/conservative/checkpoints"
MIXED_DIR="${RUN_ROOT}/mixedflux/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
FINAL_DIR="${RUN_ROOT}/final_compare"
mkdir -p "$CONSERVATIVE_DIR" "$MIXED_DIR" "$LOG_DIR" "$FINAL_DIR"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$WORK_DIR"

if [[ "$SKIP_CONSERVATIVE" != "1" ]]; then
    if [[ -z "$CLEAN_BEST" || ! -f "$CLEAN_BEST" ]]; then
        echo "ERROR: set CLEAN_BEST=/absolute/path/to/clean_best.pth" >&2
        exit 1
    fi
    "$PYTHON" -u pinn_thin_layer_v9_6.py \
        --arch "$CONSERVATIVE_ARCH" \
        --gamma "$GAMMA" \
        --k-cat-star "$K_CAT_STAR" \
        --epochs 500 \
        --resume-checkpoint "$CLEAN_BEST" \
        --reset-optimizer-state \
        --reset-best-score \
        --checkpoint-dir "$CONSERVATIVE_DIR" \
        --learning-rate 1e-6 \
        --save-every 50 \
        --progress-every 25 \
        --empty-cache-every 100 \
        --abort-on-nan \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --base-train-points "$BASE_TRAIN_POINTS" \
        --max-train-points "$MAX_TRAIN_POINTS" \
        --train-point-growth 0 \
        --current-balance-weight 500 \
        --current-balance-ramp-epochs 100 \
        --current-balance-samples 256 \
        --early-stop-check-every 50 \
        --early-stop-min-epochs 200 \
        --early-stop-patience-checks 3 \
        --early-stop-min-relative-improvement 0.005 \
        --pde-thin-weight 10 \
        --pde-ext-weight 10 \
        --thin-interface-weight 300 \
        --ext-interface-weight 200 \
        --bounds-weight 30 \
        2>&1 | tee "$LOG_DIR/conservative_training.log"
fi

CONSERVATIVE_BEST="${CONSERVATIVE_BEST_OVERRIDE:-${CONSERVATIVE_DIR}/pinn_thin_layer_catalytic_v9_6_${CONSERVATIVE_ARCH}_best.pth}"
if [[ ! -f "$CONSERVATIVE_BEST" ]]; then
    echo "ERROR: conservative checkpoint not found: $CONSERVATIVE_BEST" >&2
    exit 1
fi

if [[ "$SKIP_MIXED" != "1" ]]; then
    "$PYTHON" -u pinn_thin_layer_v9_6.py \
        --arch "$MIXED_ARCH" \
        --gamma "$GAMMA" \
        --k-cat-star "$K_CAT_STAR" \
        --epochs 1000 \
        --resume-checkpoint "$CONSERVATIVE_BEST" \
        --reset-optimizer-state \
        --reset-best-score \
        --checkpoint-dir "$MIXED_DIR" \
        --learning-rate 1e-6 \
        --save-every 50 \
        --progress-every 25 \
        --empty-cache-every 100 \
        --abort-on-nan \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --base-train-points "$BASE_TRAIN_POINTS" \
        --max-train-points "$MAX_TRAIN_POINTS" \
        --train-point-growth 0 \
        --current-balance-weight 0 \
        --early-stop-check-every 50 \
        --early-stop-min-epochs 200 \
        --early-stop-patience-checks 3 \
        --early-stop-min-relative-improvement 0.005 \
        --pde-thin-weight 10 \
        --pde-ext-weight 10 \
        --thin-interface-weight 300 \
        --ext-interface-weight 200 \
        --bounds-weight 30 \
        2>&1 | tee "$LOG_DIR/mixedflux_training.log"
else
    echo "SKIP_MIXED=1: stopping after conservative stage."
fi

compare_checkpoint() {
    local arch="$1"
    local checkpoint="$2"
    local label="$3"
    if [[ ! -f "$FDM_PKL" ]]; then
        echo "FDM posterior comparison skipped; missing: $FDM_PKL"
        return
    fi
    "$PYTHON" -u compare_concentration_fields.py \
        --arch "$arch" \
        --gamma "$GAMMA" \
        --k-cat-star "$K_CAT_STAR" \
        --input-mode normalized \
        --checkpoint "$checkpoint" \
        --fdm-pkl "$FDM_PKL" \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --n-time 160 \
        --n-x-in 120 \
        --n-x-out 160 \
        --batch-size 4096 \
        --current-mode both \
        --output-json "$FINAL_DIR/${label}_metrics.json" \
        --output-npz "$FINAL_DIR/${label}_fields.npz" \
        --output-figure "$FINAL_DIR/${label}_residual.png" \
        --output-current-figure "$FINAL_DIR/${label}_current.png"
}

MIXED_BEST="${MIXED_DIR}/pinn_thin_layer_catalytic_v9_6_${MIXED_ARCH}_best.pth"
compare_checkpoint "$CONSERVATIVE_ARCH" "$CONSERVATIVE_BEST" "conservative_best"
if [[ "$SKIP_MIXED" != "1" && -f "$MIXED_BEST" ]]; then
    compare_checkpoint "$MIXED_ARCH" "$MIXED_BEST" "mixedflux_best"
fi

echo "DONE: $RUN_ROOT"
