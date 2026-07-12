#!/usr/bin/env bash
set -euo pipefail

# Train the causal inventory-constrained Hermite lift from the clean physics-best
# checkpoint.  The lift has no FDM term in the loss; FDM comparison is optional
# posterior diagnostics only.

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96/runs_inventory_hermite_lift_train/$(date +%Y%m%d_%H%M%S)}"
CLEAN_BEST="${CLEAN_BEST:-}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/gdrive/MyDrive/pinn_v96_parameter_scale_0711}"
FDM_PKL="${FDM_PKL:-/content/gdrive/MyDrive/FDM_parameter_scale_0711/gamma1_k1_v42_thin_layer_catalytic_v42.pkl}"

GAMMA="${GAMMA:-1.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-1e-6}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
LIFT_TIME_GRID="${LIFT_TIME_GRID:-1024}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-4000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-4000}"
SAVE_EVERY="${SAVE_EVERY:-50}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-50}"

FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-0}"
FDM_COMPARE_BATCH_SIZE="${FDM_COMPARE_BATCH_SIZE:-4096}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$CLEAN_BEST" || ! -f "$CLEAN_BEST" ]]; then
    echo "CLEAN_BEST path is missing or invalid; searching under $DRIVE_ROOT ..."
    CLEAN_BEST="$(find "$DRIVE_ROOT" -type f \
        -name 'pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth' \
        -print 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "$CLEAN_BEST" || ! -f "$CLEAN_BEST" ]]; then
    echo "ERROR: no clean physics-best checkpoint found." >&2
    echo "Set CLEAN_BEST=/absolute/path/to/multiscale_film_tracegreen_clean_best.pth" >&2
    exit 1
fi
if [[ ! -f "$FDM_PKL" && "$FDM_COMPARE_EVERY" != "0" ]]; then
    echo "ERROR: FDM_COMPARE_EVERY is enabled but FDM file is missing: $FDM_PKL" >&2
    exit 1
fi

CKPT_DIR="$RUN_ROOT/checkpoints"
LOG_DIR="$RUN_ROOT/logs"
FDM_DIR="$RUN_ROOT/fdm_compare"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$FDM_DIR"
cd "$WORK_DIR"

echo "Using clean checkpoint: $CLEAN_BEST"
echo "Run root: $RUN_ROOT"
echo "Architecture: multiscale_film_tracegreen_conservative_lift"
echo "Lift grid: $LIFT_TIME_GRID; Green grid/kernel: $GREEN_TIME_GRID/$GREEN_KERNEL_POINTS"
echo "Training: epochs=$EPOCHS, lr=$LR, points=$BASE_TRAIN_POINTS, progress_every=$PROGRESS_EVERY"

train_cmd=(
    "$PYTHON" -u pinn_thin_layer_v9_6.py
    --arch multiscale_film_tracegreen_conservative_lift
    --gamma "$GAMMA"
    --k-cat-star "$K_CAT_STAR"
    --epochs "$EPOCHS"
    --resume-checkpoint "$CLEAN_BEST"
    --reset-optimizer-state
    --reset-best-score
    --checkpoint-dir "$CKPT_DIR"
    --save-every "$SAVE_EVERY"
    --progress-every "$PROGRESS_EVERY"
    --empty-cache-every "$EMPTY_CACHE_EVERY"
    --abort-on-nan
    --learning-rate "$LR"
    --green-time-grid "$GREEN_TIME_GRID"
    --green-kernel-points "$GREEN_KERNEL_POINTS"
    --lift-time-grid "$LIFT_TIME_GRID"
    --base-train-points "$BASE_TRAIN_POINTS"
    --max-train-points "$MAX_TRAIN_POINTS"
    --train-point-growth 0
    --early-stop-check-every 50
    --early-stop-patience-checks 3
    --early-stop-min-epochs 100
    --early-stop-min-relative-improvement 0.005
    --pde-thin-weight 10.0
    --pde-ext-weight 10.0
    --thin-interface-weight 300.0
    --ext-interface-weight 200.0
    --bounds-weight 30.0
)

if [[ "$FDM_COMPARE_EVERY" != "0" ]]; then
    train_cmd+=(
        --fdm-compare-pkl "$FDM_PKL"
        --fdm-compare-every "$FDM_COMPARE_EVERY"
        --fdm-compare-dir "$FDM_DIR"
        --fdm-compare-batch-size "$FDM_COMPARE_BATCH_SIZE"
        --fdm-compare-save-figure
    )
fi

"${train_cmd[@]}" 2>&1 | tee "$LOG_DIR/inventory_hermite_lift_training.log"

echo "DONE"
echo "Current checkpoint: $CKPT_DIR/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_conservative_lift.pth"
echo "Best checkpoint:    $CKPT_DIR/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_conservative_lift_best.pth"
