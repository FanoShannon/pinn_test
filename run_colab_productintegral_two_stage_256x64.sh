#!/usr/bin/env bash
set -euo pipefail

# Final two-stage ProductIntegral-TraceGreen workflow.
#
# Stage 1: multiscale_green_grid_dynamic_stage1 from scratch.
# Stage 2: multiscale_film_tracegreen_productintegral from Stage 1 best.
# Final: zero-training Inventory-Hermite lift applied to Stage 2 physics-best.
#
# The Stage 2 interface trace is a fixed physical product-integral operator.
# FDM is optional and posterior-only: it is never used in a loss, early stop,
# or checkpoint selection.

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96/runs_productintegral_two_stage/${RUN_TAG}}"

GAMMA="${GAMMA:-10.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"
FDM_PKL="${FDM_PKL:-}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
LIFT_TIME_GRID="${LIFT_TIME_GRID:-1024}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"

STAGE1_ARCH="multiscale_green_grid_dynamic_stage1"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5000}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE1_MIN_EPOCHS="${STAGE1_MIN_EPOCHS:-1500}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
STAGE1_BEST="${STAGE1_BEST:-}"

STAGE2_ARCH="multiscale_film_tracegreen_productintegral"
FINAL_ARCH="multiscale_film_tracegreen_productintegral_lift"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1000}"
STAGE2_LR="${STAGE2_LR:-2e-6}"
STAGE2_MIN_EPOCHS="${STAGE2_MIN_EPOCHS:-500}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
STAGE2_BEST="${STAGE2_BEST:-}"

FDM_COMPARE_EVERY_STAGE1="${FDM_COMPARE_EVERY_STAGE1:-250}"
FDM_COMPARE_EVERY_STAGE2="${FDM_COMPARE_EVERY_STAGE2:-50}"
FINAL_COMPARE_BATCH_SIZE="${FINAL_COMPARE_BATCH_SIZE:-4096}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR"

STAGE1_DIR="$RUN_ROOT/stage1_dynamic"
STAGE2_DIR="$RUN_ROOT/stage2_productintegral"
ZERO_DIR="$RUN_ROOT/stage2_zero_training_compare"
TRAINED_DIR="$RUN_ROOT/stage2_trained_compare"
FINAL_DIR="$RUN_ROOT/final_compare"
LOG_DIR="$RUN_ROOT/logs"
MASTER_LOG="$LOG_DIR/productintegral_two_stage_master.log"
mkdir -p "$STAGE1_DIR/checkpoints" "$STAGE2_DIR/checkpoints" \
    "$ZERO_DIR" "$TRAINED_DIR" "$FINAL_DIR" "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

validate_fdm() {
    if [[ -z "$FDM_PKL" ]]; then
        log "No FDM_PKL supplied; posterior comparisons are disabled."
        return 0
    fi
    if [[ ! -f "$FDM_PKL" ]]; then
        log "ERROR: FDM_PKL does not exist: $FDM_PKL"
        exit 1
    fi
    "$PYTHON" - "$FDM_PKL" "$GAMMA" "$K_CAT_STAR" <<'PY'
import pickle
import sys
from math import isclose

path, gamma, k_cat = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with open(path, "rb") as handle:
    params = pickle.load(handle).get("params", {})
fdm_gamma = params.get("gamma")
fdm_k = params.get("k_cat", params.get("k_cat_star"))
if fdm_gamma is None or not isclose(float(fdm_gamma), gamma, rel_tol=1e-7, abs_tol=1e-10):
    raise SystemExit(f"FDM gamma mismatch: file={fdm_gamma}, run={gamma}")
if fdm_k is None or not isclose(float(fdm_k), k_cat, rel_tol=1e-7, abs_tol=1e-10):
    raise SystemExit(f"FDM k_cat mismatch: file={fdm_k}, run={k_cat}")
print(f"Validated posterior-only FDM: gamma={fdm_gamma}, k_cat={fdm_k}")
PY
}

add_common_train_args() {
    local -n command="$1"
    command+=(
        --gamma "$GAMMA"
        --k-cat-star "$K_CAT_STAR"
        --green-time-grid "$GREEN_TIME_GRID"
        --green-kernel-points "$GREEN_KERNEL_POINTS"
        --lift-time-grid "$LIFT_TIME_GRID"
        --base-train-points "$BASE_TRAIN_POINTS"
        --max-train-points "$MAX_TRAIN_POINTS"
        --train-point-growth 0
        --pde-thin-weight 10.0
        --pde-ext-weight 10.0
        --thin-interface-weight 300.0
        --ext-interface-weight 200.0
        --bounds-weight 30.0
        --progress-every 25
        --empty-cache-every 100
        --abort-on-nan
        --early-stop-check-every 100
        --early-stop-patience-checks 4
        --early-stop-min-relative-improvement 0.005
        --early-stop-ema-alpha 0.5
        --early-stop-validation-points 96
    )
}

add_fdm_args() {
    local -n command="$1"
    local every="$2"
    local output_dir="$3"
    if [[ -n "$FDM_PKL" && "$every" != "0" ]]; then
        command+=(
            --fdm-compare-pkl "$FDM_PKL"
            --fdm-compare-every "$every"
            --fdm-compare-dir "$output_dir"
            --fdm-compare-batch-size 8192
            --fdm-compare-save-figure
        )
    fi
}

compare_checkpoint() {
    local arch="$1"
    local checkpoint="$2"
    local output_dir="$3"
    local label="$4"
    if [[ -z "$FDM_PKL" ]]; then
        return 0
    fi
    mkdir -p "$output_dir"
    "$PYTHON" -u compare_concentration_fields.py \
        --arch "$arch" \
        --gamma "$GAMMA" \
        --k-cat-star "$K_CAT_STAR" \
        --input-mode normalized \
        --checkpoint "$checkpoint" \
        --fdm-pkl "$FDM_PKL" \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --lift-time-grid "$LIFT_TIME_GRID" \
        --n-time 160 \
        --n-x-in 120 \
        --n-x-out 160 \
        --cv-points 800 \
        --batch-size "$FINAL_COMPARE_BATCH_SIZE" \
        --current-mode both \
        --output-json "$output_dir/${label}_metrics.json" \
        --output-npz "$output_dir/${label}_fields.npz" \
        --output-figure "$output_dir/${label}_residual.png" \
        --output-current-figure "$output_dir/${label}_current.png"
}

validate_fdm
log "Two-stage ProductIntegral run: gamma=$GAMMA, k_cat=$K_CAT_STAR"
log "Stage 1=$STAGE1_ARCH; Stage 2=$STAGE2_ARCH; final zero-lift=$FINAL_ARCH."
log "FDM is posterior-only; Inventory-Hermite lift never enters Stage 2 training."

if [[ "$SKIP_STAGE1" == "0" ]]; then
    STAGE1_CMD=(
        "$PYTHON" -u pinn_thin_layer_v9_6.py
        --arch "$STAGE1_ARCH"
        --epochs "$STAGE1_EPOCHS"
        --reset-optimizer-state
        --reset-best-score
        --checkpoint-dir "$STAGE1_DIR/checkpoints"
        --learning-rate "$STAGE1_LR"
        --early-stop-min-epochs "$STAGE1_MIN_EPOCHS"
        --save-every 250
    )
    add_common_train_args STAGE1_CMD
    add_fdm_args STAGE1_CMD "$FDM_COMPARE_EVERY_STAGE1" "$STAGE1_DIR/fdm_compare"
    log "Stage 1 command: ${STAGE1_CMD[*]}"
    "${STAGE1_CMD[@]}" 2>&1 | tee "$LOG_DIR/stage1_dynamic.log"
    STAGE1_BEST="$STAGE1_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth"
fi

if [[ -z "$STAGE1_BEST" || ! -f "$STAGE1_BEST" ]]; then
    log "ERROR: Stage 1 best checkpoint does not exist: $STAGE1_BEST"
    exit 1
fi

log "Zero-training ProductIntegral evaluation from Stage 1 best."
compare_checkpoint "$STAGE2_ARCH" "$STAGE1_BEST" "$ZERO_DIR" \
    "productintegral_from_stage1_zero_training" \
    2>&1 | tee "$LOG_DIR/stage2_zero_training_compare.log"

if [[ "$SKIP_STAGE2" == "0" ]]; then
    STAGE2_CMD=(
        "$PYTHON" -u pinn_thin_layer_v9_6.py
        --arch "$STAGE2_ARCH"
        --epochs "$STAGE2_EPOCHS"
        --resume-checkpoint "$STAGE1_BEST"
        --reset-optimizer-state
        --reset-best-score
        --checkpoint-dir "$STAGE2_DIR/checkpoints"
        --learning-rate "$STAGE2_LR"
        --early-stop-min-epochs "$STAGE2_MIN_EPOCHS"
        --save-every 100
        --clean-residual-initial-scale 0.0
        --clean-residual-decay-epochs 0
    )
    add_common_train_args STAGE2_CMD
    add_fdm_args STAGE2_CMD "$FDM_COMPARE_EVERY_STAGE2" "$STAGE2_DIR/fdm_compare"
    log "Stage 2 command: ${STAGE2_CMD[*]}"
    "${STAGE2_CMD[@]}" 2>&1 | tee "$LOG_DIR/stage2_productintegral.log"
    STAGE2_BEST="$STAGE2_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth"
fi

if [[ -z "$STAGE2_BEST" || ! -f "$STAGE2_BEST" ]]; then
    log "ERROR: Stage 2 best checkpoint does not exist: $STAGE2_BEST"
    exit 1
fi

log "Stage 2 physics-best posterior evaluation before Inventory lift."
compare_checkpoint "$STAGE2_ARCH" "$STAGE2_BEST" "$TRAINED_DIR" \
    "productintegral_stage2_best_before_lift" \
    2>&1 | tee "$LOG_DIR/stage2_trained_compare.log"

log "Applying Inventory-Hermite lift to Stage 2 best with zero training."
compare_checkpoint "$FINAL_ARCH" "$STAGE2_BEST" "$FINAL_DIR" \
    "productintegral_stage2_best_zero_lift" \
    2>&1 | tee "$LOG_DIR/final_zero_lift_compare.log"

log "Completed."
log "Stage 1 best: $STAGE1_BEST"
log "Stage 2 best: $STAGE2_BEST"
log "Zero-training report: $ZERO_DIR"
log "Stage 2 pre-lift report: $TRAINED_DIR"
log "Final zero-lift report: $FINAL_DIR"
