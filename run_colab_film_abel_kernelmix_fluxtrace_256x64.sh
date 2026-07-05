#!/usr/bin/env bash
set -euo pipefail

# Colab run for the theory-first FluxTrace external Green ablation.
#
# Main change versus causalhybrid/intmemory:
#   C_D(y,t) = G_flux[J](y,t)
#            + h00(z) * (D_int(t) - G_flux[J](0,t))
#            + h01(z) * (D_far - G_flux[J](L,t))
#            + R_smooth(y,t)
#
# The external boundary flux is treated as the analytic trace of G_flux[J].
# Therefore this branch removes the old Hermite h10 slope correction that was
# over-constraining the near-interface external field.
#
# Colab usage:
#   %cd /content/gdrive/MyDrive/pinn_v96
#   !bash run_colab_film_abel_kernelmix_fluxtrace_256x64.sh

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./runs_film_abel_kernelmix_fluxtrace_256x64/${RUN_TAG}}"

ARCH="multiscale_green_grid_film_abel_kernelmix_fluxtrace"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-2e-6}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-100}"

FDM_PKL="${FDM_PKL:-/content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl}"
FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-500}"
FDM_COMPARE_BATCH_SIZE="${FDM_COMPARE_BATCH_SIZE:-8192}"
FINAL_COMPARE_BATCH_SIZE="${FINAL_COMPARE_BATCH_SIZE:-4096}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORK_DIR"

CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
FDM_DIR="${RUN_ROOT}/fdm_compare"
FINAL_DIR="${RUN_ROOT}/final_compare"
MASTER_LOG="${LOG_DIR}/film_abel_kernelmix_fluxtrace_master.log"
TRAIN_LOG="${LOG_DIR}/film_abel_kernelmix_fluxtrace_training.log"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$FDM_DIR" "$FINAL_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

latest_checkpoint_in() {
    local root="$1"
    local pattern="$2"
    if [[ -d "$root" ]]; then
        find "$root" -path "*/checkpoints/${pattern}" -type f 2>/dev/null | sort | tail -n 1
    fi
}

resolve_resume_ckpt() {
    if [[ -n "${RESUME_CKPT:-}" && -f "${RESUME_CKPT}" ]]; then
        echo "${RESUME_CKPT}"
        return 0
    fi

    local patterns=(
        "./runs_film_abel_kernelmix_causalhybrid_intmemory_256x64 pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory.pth"
        "./runs_film_abel_kernelmix_causalhybrid_intmemory_256x64 pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory_best.pth"
        "./runs_film_abel_kernelmix_causalhybrid_smooth_256x64 pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth.pth"
        "./runs_film_abel_kernelmix_causalhybrid_smooth_256x64 pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth_best.pth"
        "./runs_film_abel_kernelmix_causalhybrid_256x64 pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_best.pth"
        "./runs_film_abel_kernelmix_256x64 pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_best.pth"
    )

    local entry root pattern found
    for entry in "${patterns[@]}"; do
        root="${entry%% *}"
        pattern="${entry#* }"
        found="$(latest_checkpoint_in "$root" "$pattern")"
        if [[ -n "$found" && -f "$found" ]]; then
            echo "$found"
            return 0
        fi
    done

    local candidates=(
        "./checkpoints_film_abel_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_best.pth"
        "./pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_best.pth"
    )
    for path in "${candidates[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    log "ERROR: warm-start checkpoint not found."
    log "Set RESUME_CKPT=/path/to/kernelmix_or_intmemory_checkpoint.pth"
    return 1
}

compare_one() {
    local ckpt="$1"
    local label="$2"
    if [[ ! -f "$ckpt" || ! -f "$FDM_PKL" ]]; then
        return 0
    fi

    log "Running final posterior concentration compare for ${label}: ${ckpt}"
    "$PYTHON" -u compare_concentration_fields.py \
        --arch "$ARCH" \
        --input-mode normalized \
        --checkpoint "$ckpt" \
        --fdm-pkl "$FDM_PKL" \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --n-time 160 \
        --n-x-in 120 \
        --n-x-out 160 \
        --batch-size "$FINAL_COMPARE_BATCH_SIZE" \
        --output-json "${FINAL_DIR}/fluxtrace_${label}_concentration_metrics.json" \
        --output-npz "${FINAL_DIR}/fluxtrace_${label}_concentration_fields.npz" \
        --output-figure "${FINAL_DIR}/fluxtrace_${label}_residual_summary.png" \
        2>&1 | tee "${LOG_DIR}/fluxtrace_${label}_final_compare.log"
}

log "=================================================="
log "PINN v9.6 Film-Abel KernelMix FluxTrace run"
log "work_dir=${WORK_DIR}"
log "run_root=${RUN_ROOT}"
log "arch=${ARCH}"
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

RESUME_PATH="$(resolve_resume_ckpt)"
log "Resume checkpoint: ${RESUME_PATH}"

train_cmd=(
    "$PYTHON" -u pinn_thin_layer_v9_6.py
    --arch "$ARCH"
    --epochs "$EPOCHS"
    --resume-checkpoint "$RESUME_PATH"
    --reset-optimizer-state
    --reset-best-score
    --checkpoint-dir "$CKPT_DIR"
    --save-every "$SAVE_EVERY"
    --progress-every "$PROGRESS_EVERY"
    --empty-cache-every "$EMPTY_CACHE_EVERY"
    --no-early-stop
    --abort-on-nan
    --learning-rate "$LR"
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

if [[ "$FDM_COMPARE_EVERY" != "0" && -f "$FDM_PKL" ]]; then
    train_cmd+=(
        --fdm-compare-pkl "$FDM_PKL"
        --fdm-compare-every "$FDM_COMPARE_EVERY"
        --fdm-compare-dir "$FDM_DIR"
        --fdm-compare-batch-size "$FDM_COMPARE_BATCH_SIZE"
        --fdm-compare-save-figure
    )
else
    log "In-training FDM compare disabled or FDM missing: ${FDM_PKL}"
fi

log "Training command: ${train_cmd[*]}"
"${train_cmd[@]}" 2>&1 | tee "$TRAIN_LOG"

CURRENT_CKPT="${CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_${ARCH}.pth"
BEST_CKPT="${CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_${ARCH}_best.pth"
compare_one "$CURRENT_CKPT" "current"
if [[ "$BEST_CKPT" != "$CURRENT_CKPT" ]]; then
    compare_one "$BEST_CKPT" "best"
fi

log "DONE"
log "Checkpoint dir: ${CKPT_DIR}"
log "Training log: ${TRAIN_LOG}"
log "Final compare dir: ${FINAL_DIR}"
