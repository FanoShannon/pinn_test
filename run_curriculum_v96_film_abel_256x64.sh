#!/usr/bin/env bash
set -euo pipefail

# Two-stage from-zero curriculum for the Film-Abel branch.
#
# Stage 1 trains the softer dynamic Green model from scratch.
# Stage 2 switches only the interface architecture to Film-Abel while keeping
# the same Green history resolution, M=256 and K=64.  This avoids the previous
# instability where the architecture, quadrature resolution, and optimizer
# scale all changed at the same time.
#
# Colab usage:
#   %cd /content/gdrive/MyDrive/pinn_v96
#   !bash run_curriculum_v96_film_abel_256x64.sh
#
# Optional overrides:
#   WORK_DIR=/content/gdrive/MyDrive/pinn_v96 \
#   STAGE1_EPOCHS=10000 STAGE2_EPOCHS=8000 \
#   STAGE1_LR=5e-5 STAGE2_LR=5e-6 \
#   bash run_curriculum_v96_film_abel_256x64.sh

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./runs_film_abel_256x64_two_stage/${RUN_TAG}}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-10000}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-8000}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE2_LR="${STAGE2_LR:-5e-6}"

GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"

FDM_PKL="${FDM_PKL:-/content/gdrive/MyDrive/FDM/thin_layer_catalytic_v41_fixed.pkl}"
FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-0}"

cd "$WORK_DIR"

STAGE1_DIR="${RUN_ROOT}/checkpoints_dynamic_256x64"
STAGE2_DIR="${RUN_ROOT}/checkpoints_film_abel_256x64"
STAGE1_SNAPSHOT_DIR="${RUN_ROOT}/stage1_finished_snapshot"
LOG_DIR="${RUN_ROOT}/logs"
FDM_DIR="${RUN_ROOT}/fdm_compare"
MASTER_LOG="${LOG_DIR}/curriculum_master.log"
STAGE1_LOG="${LOG_DIR}/stage1_dynamic.log"
STAGE2_LOG="${LOG_DIR}/stage2_film_abel.log"

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE1_SNAPSHOT_DIR" "$LOG_DIR" "$FDM_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

checkpoint_for() {
    local dir="$1"
    local arch="$2"
    local best="${dir}/pinn_thin_layer_catalytic_v9_6_${arch}_best.pth"
    local current="${dir}/pinn_thin_layer_catalytic_v9_6_${arch}.pth"

    if [[ -f "$best" ]]; then
        echo "$best"
    elif [[ -f "$current" ]]; then
        echo "$current"
    else
        return 1
    fi
}

check_checkpoint() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        log "ERROR: ${label} checkpoint missing: ${path}"
        return 1
    fi
    local size
    size=$(wc -c < "$path" | tr -d ' ')
    log "${label} checkpoint: ${path} (${size} bytes)"
}

snapshot_stage1() {
    local source_dir="$1"
    local checkpoint="$2"
    local target_dir="$3"
    mkdir -p "$target_dir"

    log ""
    log "Saving Stage 1 snapshot before Stage 2..."
    log "Snapshot dir: ${target_dir}"

    local arch="multiscale_green_grid_dynamic"
    local best="${source_dir}/pinn_thin_layer_catalytic_v9_6_${arch}_best.pth"
    local current="${source_dir}/pinn_thin_layer_catalytic_v9_6_${arch}.pth"

    if [[ -f "$best" ]]; then
        cp -f "$best" "${target_dir}/$(basename "$best")"
        log "  copied best: ${target_dir}/$(basename "$best")"
    fi
    if [[ -f "$current" ]]; then
        cp -f "$current" "${target_dir}/$(basename "$current")"
        log "  copied current: ${target_dir}/$(basename "$current")"
    fi
    if [[ -f "$STAGE1_LOG" ]]; then
        cp -f "$STAGE1_LOG" "${target_dir}/stage1_dynamic.log"
        log "  copied log: ${target_dir}/stage1_dynamic.log"
    fi

    {
        echo "stage1_source_dir=${source_dir}"
        echo "stage1_selected_checkpoint=${checkpoint}"
        echo "stage1_snapshot_time=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "resume_with_same_run_tag=RUN_TAG=${RUN_TAG} bash run_curriculum_v96_film_abel_256x64.sh"
    } > "${target_dir}/README_stage1_snapshot.txt"

    local snapshot_checkpoint="${target_dir}/$(basename "$checkpoint")"
    check_checkpoint "$snapshot_checkpoint" "Stage 1 snapshot"
}

fdm_args_for() {
    local out_dir="$1"
    if [[ "$FDM_COMPARE_EVERY" != "0" && -f "$FDM_PKL" ]]; then
        printf '%s\n' \
            --fdm-compare-pkl "$FDM_PKL" \
            --fdm-compare-every "$FDM_COMPARE_EVERY" \
            --fdm-compare-dir "$out_dir" \
            --fdm-compare-save-figure \
            --fdm-compare-save-fields
    fi
}

run_stage() {
    local label="$1"
    local arch="$2"
    local epochs="$3"
    local lr="$4"
    local out_dir="$5"
    local log_file="$6"
    shift 6

    log ""
    log ">>> ${label}: ${arch}"
    log ">>> epochs=${epochs}, lr=${lr}, M=${GREEN_TIME_GRID}, K=${GREEN_KERNEL_POINTS}"
    log ">>> output=${out_dir}"

    local cmd=(
        "$PYTHON" -u pinn_thin_layer_v9_6.py
        --arch "$arch"
        --epochs "$epochs"
        --checkpoint-dir "$out_dir"
        --save-every 500
        --progress-every 25
        --empty-cache-every 100
        --no-early-stop
        --abort-on-nan
        --learning-rate "$lr"
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
        "$@"
    )

    log "Command: ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "$log_file"
}

log "=================================================="
log "PINN v9.6 Film-Abel 256x64 two-stage curriculum"
log "work_dir=${WORK_DIR}"
log "run_root=${RUN_ROOT}"
log "python=$($PYTHON --version 2>&1)"
log "=================================================="

"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER_LOG"
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

stage1_fdm_args=()
if [[ "$FDM_COMPARE_EVERY" != "0" && -f "$FDM_PKL" ]]; then
    while IFS= read -r item; do stage1_fdm_args+=("$item"); done < <(fdm_args_for "${FDM_DIR}/stage1_dynamic")
fi

run_stage \
    "STAGE 1" \
    "multiscale_green_grid_dynamic" \
    "$STAGE1_EPOCHS" \
    "$STAGE1_LR" \
    "$STAGE1_DIR" \
    "$STAGE1_LOG" \
    --reset-optimizer-state \
    --reset-best-score \
    "${stage1_fdm_args[@]}"

STAGE1_CKPT="$(checkpoint_for "$STAGE1_DIR" "multiscale_green_grid_dynamic")"
check_checkpoint "$STAGE1_CKPT" "Stage 1"
snapshot_stage1 "$STAGE1_DIR" "$STAGE1_CKPT" "$STAGE1_SNAPSHOT_DIR"
STAGE1_CKPT="${STAGE1_SNAPSHOT_DIR}/$(basename "$STAGE1_CKPT")"
check_checkpoint "$STAGE1_CKPT" "Stage 1 resume snapshot"

stage2_fdm_args=()
if [[ "$FDM_COMPARE_EVERY" != "0" && -f "$FDM_PKL" ]]; then
    while IFS= read -r item; do stage2_fdm_args+=("$item"); done < <(fdm_args_for "${FDM_DIR}/stage2_film_abel")
fi

run_stage \
    "STAGE 2" \
    "multiscale_green_grid_film_abel" \
    "$STAGE2_EPOCHS" \
    "$STAGE2_LR" \
    "$STAGE2_DIR" \
    "$STAGE2_LOG" \
    --resume-checkpoint "$STAGE1_CKPT" \
    --reset-optimizer-state \
    --reset-best-score \
    "${stage2_fdm_args[@]}"

STAGE2_CKPT="$(checkpoint_for "$STAGE2_DIR" "multiscale_green_grid_film_abel")"
check_checkpoint "$STAGE2_CKPT" "Stage 2"

log ""
log "DONE"
log "Stage 1 checkpoint: ${STAGE1_CKPT}"
log "Stage 2 checkpoint: ${STAGE2_CKPT}"
log "Logs: ${LOG_DIR}"
log ""
log "To enable posterior FDM compare next time, run with:"
log "  FDM_COMPARE_EVERY=500 bash run_curriculum_v96_film_abel_256x64.sh"
