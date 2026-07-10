#!/usr/bin/env bash
set -euo pipefail

# Reproducible paper-facing clean pipeline.
#
# Stage 1:
#   multiscale_green_grid_dynamic_stage1
#   - standard interface state
#   - signed full-field Green history
#   - fixed three-mode residual + fixed three-mode dynamic physical correction
#   - no Film-Abel, KernelMix, matched Abel, interface residual, or hybrid extra modes
#
# Stage 2:
#   multiscale_film_tracegreen_clean
#   - Film-Abel interface chain
#   - erfc trace-preserving TraceGreen external lift
#   - optional R_smooth scaffold that decays to zero

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./runs_film_tracegreen_clean_two_stage_256x64/${RUN_TAG}}"

GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-100}"
GAMMA="${GAMMA:-10.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"

STAGE1_ARCH="multiscale_green_grid_dynamic_stage1"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5000}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"

STAGE2_ARCH="multiscale_film_tracegreen_clean"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1500}"
STAGE2_LR="${STAGE2_LR:-2e-6}"
CLEAN_RESIDUAL_INITIAL_SCALE="${CLEAN_RESIDUAL_INITIAL_SCALE:-1.0}"
CLEAN_RESIDUAL_DECAY_EPOCHS="${CLEAN_RESIDUAL_DECAY_EPOCHS:-1000}"

FDM_PKL="${FDM_PKL:-/content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl}"
FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-250}"
FDM_COMPARE_BATCH_SIZE="${FDM_COMPARE_BATCH_SIZE:-8192}"
FINAL_COMPARE_BATCH_SIZE="${FINAL_COMPARE_BATCH_SIZE:-4096}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR"

STAGE1_CKPT_DIR="${RUN_ROOT}/stage1_dynamic_fixed/checkpoints"
STAGE2_CKPT_DIR="${RUN_ROOT}/stage2_tracegreen_clean/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
FDM_STAGE1_DIR="${RUN_ROOT}/stage1_dynamic_fixed/fdm_compare"
FDM_STAGE2_DIR="${RUN_ROOT}/stage2_tracegreen_clean/fdm_compare"
FINAL_DIR="${RUN_ROOT}/final_compare"
MASTER_LOG="${LOG_DIR}/clean_two_stage_master.log"
STAGE1_LOG="${LOG_DIR}/stage1_dynamic_fixed_training.log"
STAGE2_LOG="${LOG_DIR}/stage2_tracegreen_clean_training.log"

mkdir -p "$STAGE1_CKPT_DIR" "$STAGE2_CKPT_DIR" "$LOG_DIR" "$FDM_STAGE1_DIR" "$FDM_STAGE2_DIR" "$FINAL_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

add_fdm_compare_args() {
    local -n cmd_ref="$1"
    local out_dir="$2"
    if [[ "$FDM_COMPARE_EVERY" != "0" && -f "$FDM_PKL" ]]; then
        cmd_ref+=(
            --fdm-compare-pkl "$FDM_PKL"
            --fdm-compare-every "$FDM_COMPARE_EVERY"
            --fdm-compare-dir "$out_dir"
            --fdm-compare-batch-size "$FDM_COMPARE_BATCH_SIZE"
            --fdm-compare-save-figure
        )
    else
        log "In-training FDM compare disabled or FDM missing: ${FDM_PKL}"
    fi
}

compare_one() {
    local arch="$1"
    local ckpt="$2"
    local label="$3"
    if [[ ! -f "$ckpt" || ! -f "$FDM_PKL" ]]; then
        return 0
    fi

    log "Running final posterior concentration compare for ${label}: ${ckpt}"
    "$PYTHON" -u compare_concentration_fields.py \
        --arch "$arch" \
        --gamma "$GAMMA" \
        --k-cat-star "$K_CAT_STAR" \
        --input-mode normalized \
        --checkpoint "$ckpt" \
        --fdm-pkl "$FDM_PKL" \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --n-time 160 \
        --n-x-in 120 \
        --n-x-out 160 \
        --batch-size "$FINAL_COMPARE_BATCH_SIZE" \
        --output-json "${FINAL_DIR}/${label}_concentration_metrics.json" \
        --output-npz "${FINAL_DIR}/${label}_concentration_fields.npz" \
        --output-figure "${FINAL_DIR}/${label}_residual_summary.png" \
        2>&1 | tee "${LOG_DIR}/${label}_final_compare.log"
}

log "=================================================="
log "PINN v9.6 clean two-stage run"
log "work_dir=${WORK_DIR}"
log "run_root=${RUN_ROOT}"
log "stage1=${STAGE1_ARCH}, epochs=${STAGE1_EPOCHS}, lr=${STAGE1_LR}"
log "stage2=${STAGE2_ARCH}, epochs=${STAGE2_EPOCHS}, lr=${STAGE2_LR}"
log "green M=${GREEN_TIME_GRID}, K=${GREEN_KERNEL_POINTS}"
log "physical parameters: gamma=${GAMMA}, k_cat_star=${K_CAT_STAR}"
log "clean residual scale=${CLEAN_RESIDUAL_INITIAL_SCALE} -> 0 over ${CLEAN_RESIDUAL_DECAY_EPOCHS} epochs"
log "python=$($PYTHON --version 2>&1)"
log "=================================================="

"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER_LOG"
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("GPU memory GB:", round(props.total_memory / 1024**3, 2))
PY

STAGE1_BEST="${STAGE1_CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_${STAGE1_ARCH}_best.pth"

if [[ "$SKIP_STAGE1" == "1" ]]; then
    if [[ -z "${STAGE1_RESUME_CKPT:-}" || ! -f "${STAGE1_RESUME_CKPT:-}" ]]; then
        log "ERROR: SKIP_STAGE1=1 requires STAGE1_RESUME_CKPT=/path/to/stage1_best.pth"
        exit 1
    fi
    STAGE1_BEST="$STAGE1_RESUME_CKPT"
    log "Skipping stage1. Using stage1 checkpoint: ${STAGE1_BEST}"
else
    stage1_cmd=(
        "$PYTHON" -u pinn_thin_layer_v9_6.py
        --arch "$STAGE1_ARCH"
        --gamma "$GAMMA"
        --k-cat-star "$K_CAT_STAR"
        --epochs "$STAGE1_EPOCHS"
        --reset-optimizer-state
        --reset-best-score
        --checkpoint-dir "$STAGE1_CKPT_DIR"
        --save-every "$SAVE_EVERY"
        --progress-every "$PROGRESS_EVERY"
        --empty-cache-every "$EMPTY_CACHE_EVERY"
        --no-early-stop
        --abort-on-nan
        --learning-rate "$STAGE1_LR"
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
    add_fdm_compare_args stage1_cmd "$FDM_STAGE1_DIR"

    log "Stage1 command: ${stage1_cmd[*]}"
    "${stage1_cmd[@]}" 2>&1 | tee "$STAGE1_LOG"

    if [[ ! -f "$STAGE1_BEST" ]]; then
        log "ERROR: stage1 best checkpoint not found: ${STAGE1_BEST}"
        exit 1
    fi
fi

log "Stage1 best checkpoint for stage2: ${STAGE1_BEST}"

stage2_cmd=(
    "$PYTHON" -u pinn_thin_layer_v9_6.py
    --arch "$STAGE2_ARCH"
    --gamma "$GAMMA"
    --k-cat-star "$K_CAT_STAR"
    --epochs "$STAGE2_EPOCHS"
    --resume-checkpoint "$STAGE1_BEST"
    --reset-optimizer-state
    --reset-best-score
    --checkpoint-dir "$STAGE2_CKPT_DIR"
    --save-every "$SAVE_EVERY"
    --progress-every "$PROGRESS_EVERY"
    --empty-cache-every "$EMPTY_CACHE_EVERY"
    --no-early-stop
    --abort-on-nan
    --learning-rate "$STAGE2_LR"
    --green-time-grid "$GREEN_TIME_GRID"
    --green-kernel-points "$GREEN_KERNEL_POINTS"
    --base-train-points "$BASE_TRAIN_POINTS"
    --max-train-points "$MAX_TRAIN_POINTS"
    --train-point-growth 0
    --clean-residual-initial-scale "$CLEAN_RESIDUAL_INITIAL_SCALE"
    --clean-residual-decay-epochs "$CLEAN_RESIDUAL_DECAY_EPOCHS"
    --pde-thin-weight 10.0
    --pde-ext-weight 10.0
    --thin-interface-weight 300.0
    --ext-interface-weight 200.0
    --bounds-weight 30.0
)
add_fdm_compare_args stage2_cmd "$FDM_STAGE2_DIR"

log "Stage2 command: ${stage2_cmd[*]}"
"${stage2_cmd[@]}" 2>&1 | tee "$STAGE2_LOG"

STAGE2_CURRENT="${STAGE2_CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_${STAGE2_ARCH}.pth"
STAGE2_BEST="${STAGE2_CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_${STAGE2_ARCH}_best.pth"

compare_one "$STAGE1_ARCH" "$STAGE1_BEST" "stage1_best"
compare_one "$STAGE2_ARCH" "$STAGE2_CURRENT" "stage2_current"
compare_one "$STAGE2_ARCH" "$STAGE2_BEST" "stage2_best"

log "DONE"
log "Run root: ${RUN_ROOT}"
log "Stage1 log: ${STAGE1_LOG}"
log "Stage2 log: ${STAGE2_LOG}"
log "Final compare dir: ${FINAL_DIR}"
