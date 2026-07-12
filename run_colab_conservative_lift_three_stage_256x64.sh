#!/usr/bin/env bash
set -euo pipefail

# Fixed-parameter, from-scratch pipeline for the conservative Inventory-Hermite
# lift.  It deliberately separates the two architecture transitions:
#
# Stage 1: dynamic Green warm-up learns a stable concentration solution.
# Stage 2: Film-TraceGreen clean learns the causal interface/external structure.
# Stage 3: Inventory-Hermite lift is applied as a zero-training posterior
# transform.  Optional Stage 3 fine-tuning is retained only as an ablation.
#
# FDM is never used in a training loss, early stopping, or checkpoint selection.
# When FDM_PKL is supplied it is used only for posterior reports.

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96/runs_conservative_lift_three_stage/${RUN_TAG}}"

GAMMA="${GAMMA:-10.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"
FDM_PKL="${FDM_PKL:-}"

GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
LIFT_TIME_GRID="${LIFT_TIME_GRID:-1024}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-100}"

STAGE1_ARCH="multiscale_green_grid_dynamic_stage1"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5000}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE1_MIN_EPOCHS="${STAGE1_MIN_EPOCHS:-1500}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
STAGE1_RESUME_CKPT="${STAGE1_RESUME_CKPT:-}"

STAGE2_ARCH="multiscale_film_tracegreen_clean"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1500}"
STAGE2_LR="${STAGE2_LR:-2e-6}"
STAGE2_MIN_EPOCHS="${STAGE2_MIN_EPOCHS:-1100}"
CLEAN_RESIDUAL_INITIAL_SCALE="${CLEAN_RESIDUAL_INITIAL_SCALE:-1.0}"
CLEAN_RESIDUAL_DECAY_EPOCHS="${CLEAN_RESIDUAL_DECAY_EPOCHS:-1000}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-}"

STAGE3_ARCH="multiscale_film_tracegreen_conservative_lift"
TRAIN_LIFT="${TRAIN_LIFT:-0}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-300}"
STAGE3_LR="${STAGE3_LR:-1e-6}"
STAGE3_MIN_EPOCHS="${STAGE3_MIN_EPOCHS:-100}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"

EARLY_STOP_CHECK_EVERY="${EARLY_STOP_CHECK_EVERY:-100}"
EARLY_STOP_PATIENCE_CHECKS="${EARLY_STOP_PATIENCE_CHECKS:-4}"
EARLY_STOP_MIN_RELATIVE_IMPROVEMENT="${EARLY_STOP_MIN_RELATIVE_IMPROVEMENT:-0.005}"
EARLY_STOP_EMA_ALPHA="${EARLY_STOP_EMA_ALPHA:-0.5}"
EARLY_STOP_VALIDATION_POINTS="${EARLY_STOP_VALIDATION_POINTS:-96}"
DISABLE_EARLY_STOP="${DISABLE_EARLY_STOP:-0}"

FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-0}"
FDM_COMPARE_BATCH_SIZE="${FDM_COMPARE_BATCH_SIZE:-8192}"
FINAL_COMPARE_BATCH_SIZE="${FINAL_COMPARE_BATCH_SIZE:-4096}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR"

STAGE1_DIR="$RUN_ROOT/stage1_dynamic"
STAGE2_DIR="$RUN_ROOT/stage2_tracegreen_clean"
STAGE3_DIR="$RUN_ROOT/stage3_inventory_lift"
LOG_DIR="$RUN_ROOT/logs"
FINAL_DIR="$RUN_ROOT/final_compare"
MASTER_LOG="$LOG_DIR/conservative_lift_three_stage_master.log"
mkdir -p "$STAGE1_DIR/checkpoints" "$STAGE2_DIR/checkpoints" \
    "$STAGE3_DIR/checkpoints" "$LOG_DIR" "$FINAL_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

check_fdm_reference() {
    if [[ -z "$FDM_PKL" ]]; then
        log "No FDM_PKL provided: training and physics-only selection remain enabled."
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
if fdm_gamma is None or fdm_k is None:
    raise SystemExit(f"FDM metadata is incomplete in {path}: {params}")
if not isclose(float(fdm_gamma), gamma, rel_tol=1e-7, abs_tol=1e-10):
    raise SystemExit(f"FDM gamma mismatch: file={fdm_gamma}, run={gamma}")
if not isclose(float(fdm_k), k_cat, rel_tol=1e-7, abs_tol=1e-10):
    raise SystemExit(f"FDM k_cat mismatch: file={fdm_k}, run={k_cat}")
print(f"Validated FDM posterior reference: gamma={fdm_gamma}, k_cat={fdm_k}")
PY
}

add_common_args() {
    local -n cmd_ref="$1"
    cmd_ref+=(
        --gamma "$GAMMA"
        --k-cat-star "$K_CAT_STAR"
        --save-every "$SAVE_EVERY"
        --progress-every "$PROGRESS_EVERY"
        --empty-cache-every "$EMPTY_CACHE_EVERY"
        --abort-on-nan
        --early-stop-check-every "$EARLY_STOP_CHECK_EVERY"
        --early-stop-patience-checks "$EARLY_STOP_PATIENCE_CHECKS"
        --early-stop-min-relative-improvement "$EARLY_STOP_MIN_RELATIVE_IMPROVEMENT"
        --early-stop-ema-alpha "$EARLY_STOP_EMA_ALPHA"
        --early-stop-validation-points "$EARLY_STOP_VALIDATION_POINTS"
        --green-time-grid "$GREEN_TIME_GRID"
        --green-kernel-points "$GREEN_KERNEL_POINTS"
        --base-train-points "$BASE_TRAIN_POINTS"
        --max-train-points "$MAX_TRAIN_POINTS"
        --train-point-growth 0
        --pde-thin-weight 10.0
        --pde-ext-weight 10.0
        --thin-interface-weight 300.0
        --ext-interface-weight 200.0
        --bounds-weight 30.0
    )
}

add_fdm_compare_args() {
    local -n cmd_ref="$1"
    local output_dir="$2"
    if [[ "$FDM_COMPARE_EVERY" != "0" && -n "$FDM_PKL" ]]; then
        cmd_ref+=(
            --fdm-compare-pkl "$FDM_PKL"
            --fdm-compare-every "$FDM_COMPARE_EVERY"
            --fdm-compare-dir "$output_dir/fdm_compare"
            --fdm-compare-batch-size "$FDM_COMPARE_BATCH_SIZE"
            --fdm-compare-save-figure
        )
    fi
}

run_stage() {
    local label="$1"
    local log_file="$2"
    shift 2
    log "$label command: $*"
    "$@" 2>&1 | tee "$log_file"
}

compare_one() {
    local arch="$1"
    local checkpoint="$2"
    local label="$3"
    if [[ -z "$FDM_PKL" || ! -f "$checkpoint" ]]; then
        return 0
    fi
    run_stage "Posterior compare $label" "$LOG_DIR/${label}_compare.log" \
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
        --cv-points 2000 \
        --batch-size "$FINAL_COMPARE_BATCH_SIZE" \
        --current-mode both \
        --output-json "$FINAL_DIR/${label}_metrics.json" \
        --output-npz "$FINAL_DIR/${label}_fields.npz" \
        --output-figure "$FINAL_DIR/${label}_residual.png" \
        --output-current-figure "$FINAL_DIR/${label}_current.png"
}

check_fdm_reference
log "=================================================="
log "Conservative-lift three-stage fixed-parameter run"
log "gamma=$GAMMA, k_cat_star=$K_CAT_STAR, green=$GREEN_TIME_GRID/$GREEN_KERNEL_POINTS, lift_grid=$LIFT_TIME_GRID"
log "Stage1=$STAGE1_ARCH; Stage2=$STAGE2_ARCH; Stage3=$STAGE3_ARCH (train_lift=$TRAIN_LIFT)"
log "FDM is posterior-only: ${FDM_PKL:-not supplied}"
log "=================================================="

STAGE1_BEST="$STAGE1_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_${STAGE1_ARCH}_best.pth"
if [[ "$SKIP_STAGE1" == "1" ]]; then
    if [[ ! -f "$STAGE1_RESUME_CKPT" ]]; then
        log "ERROR: SKIP_STAGE1=1 requires STAGE1_RESUME_CKPT."
        exit 1
    fi
    STAGE1_BEST="$STAGE1_RESUME_CKPT"
else
    stage1_cmd=("$PYTHON" -u pinn_thin_layer_v9_6.py --arch "$STAGE1_ARCH" --epochs "$STAGE1_EPOCHS" --reset-optimizer-state --reset-best-score --checkpoint-dir "$STAGE1_DIR/checkpoints" --learning-rate "$STAGE1_LR" --early-stop-min-epochs "$STAGE1_MIN_EPOCHS")
    add_common_args stage1_cmd
    add_fdm_compare_args stage1_cmd "$STAGE1_DIR"
    if [[ "$DISABLE_EARLY_STOP" == "1" ]]; then stage1_cmd+=(--no-early-stop); fi
    run_stage "Stage 1" "$LOG_DIR/stage1_dynamic_training.log" "${stage1_cmd[@]}"
    [[ -f "$STAGE1_BEST" ]] || { log "ERROR: stage1 best checkpoint is missing."; exit 1; }
fi

STAGE2_BEST="$STAGE2_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_${STAGE2_ARCH}_best.pth"
if [[ "$SKIP_STAGE2" == "1" ]]; then
    if [[ ! -f "$STAGE2_RESUME_CKPT" ]]; then
        log "ERROR: SKIP_STAGE2=1 requires STAGE2_RESUME_CKPT."
        exit 1
    fi
    STAGE2_BEST="$STAGE2_RESUME_CKPT"
else
    stage2_cmd=("$PYTHON" -u pinn_thin_layer_v9_6.py --arch "$STAGE2_ARCH" --epochs "$STAGE2_EPOCHS" --resume-checkpoint "$STAGE1_BEST" --reset-optimizer-state --reset-best-score --checkpoint-dir "$STAGE2_DIR/checkpoints" --learning-rate "$STAGE2_LR" --early-stop-min-epochs "$STAGE2_MIN_EPOCHS" --clean-residual-initial-scale "$CLEAN_RESIDUAL_INITIAL_SCALE" --clean-residual-decay-epochs "$CLEAN_RESIDUAL_DECAY_EPOCHS")
    add_common_args stage2_cmd
    add_fdm_compare_args stage2_cmd "$STAGE2_DIR"
    if [[ "$DISABLE_EARLY_STOP" == "1" ]]; then stage2_cmd+=(--no-early-stop); fi
    run_stage "Stage 2" "$LOG_DIR/stage2_tracegreen_clean_training.log" "${stage2_cmd[@]}"
    [[ -f "$STAGE2_BEST" ]] || { log "ERROR: stage2 best checkpoint is missing."; exit 1; }
fi

STAGE3_CURRENT="$STAGE3_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_${STAGE3_ARCH}.pth"
STAGE3_BEST="$STAGE3_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_${STAGE3_ARCH}_best.pth"
if [[ "$TRAIN_LIFT" == "1" && "$SKIP_STAGE3" != "1" ]]; then
    log "Stage 3 training is enabled as an explicit ablation."
    stage3_cmd=("$PYTHON" -u pinn_thin_layer_v9_6.py --arch "$STAGE3_ARCH" --epochs "$STAGE3_EPOCHS" --resume-checkpoint "$STAGE2_BEST" --reset-optimizer-state --reset-best-score --checkpoint-dir "$STAGE3_DIR/checkpoints" --learning-rate "$STAGE3_LR" --early-stop-min-epochs "$STAGE3_MIN_EPOCHS" --lift-time-grid "$LIFT_TIME_GRID")
    add_common_args stage3_cmd
    add_fdm_compare_args stage3_cmd "$STAGE3_DIR"
    if [[ "$DISABLE_EARLY_STOP" == "1" ]]; then stage3_cmd+=(--no-early-stop); fi
    run_stage "Stage 3 training ablation" "$LOG_DIR/stage3_inventory_lift_training.log" "${stage3_cmd[@]}"
else
    log "Stage 3 uses zero-training Inventory-Hermite lift on Stage 2 best."
fi

compare_one "$STAGE1_ARCH" "$STAGE1_BEST" "stage1_dynamic_best"
compare_one "$STAGE2_ARCH" "$STAGE2_BEST" "stage2_clean_best"
compare_one "$STAGE3_ARCH" "$STAGE2_BEST" "stage3_lift_zero"
if [[ "$TRAIN_LIFT" == "1" ]]; then
    compare_one "$STAGE3_ARCH" "$STAGE3_CURRENT" "stage3_lift_current"
    compare_one "$STAGE3_ARCH" "$STAGE3_BEST" "stage3_lift_best"
fi

log "DONE: $RUN_ROOT"
