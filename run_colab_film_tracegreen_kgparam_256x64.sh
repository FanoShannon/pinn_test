#!/usr/bin/env bash
set -euo pipefail

# Physics-only joint k-gamma training. This runner does not accept or read FDM data.
WORK_DIR="${WORK_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-kgparam_productintegral_run01}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_kg_runs/${RUN_TAG}}"
WARM_START_CKPT="${WARM_START_CKPT:-}"
EPOCHS="${EPOCHS:-1500}"
LR="${LR:-2e-6}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"

if [[ -z "$WARM_START_CKPT" || ! -f "$WARM_START_CKPT" ]]; then
    echo "Set WARM_START_CKPT to the fixed k=1, gamma=10 ProductIntegral checkpoint." >&2
    exit 1
fi
if (( EPOCHS <= 0 )); then
    echo "EPOCHS must be positive. Use the separate posterior runner for zero-shot evaluation." >&2
    exit 1
fi

cd "$WORK_DIR"
CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" -u pinn_thin_layer_v9_6.py \
    --arch multiscale_film_tracegreen_kgparam \
    --gamma 10.0 --k-cat-star 1.0 \
    --epochs "$EPOCHS" \
    --warm-start-checkpoint "$WARM_START_CKPT" \
    --checkpoint-dir "$CKPT_DIR" \
    --learning-rate "$LR" \
    --green-time-grid "$GREEN_TIME_GRID" \
    --green-kernel-points "$GREEN_KERNEL_POINTS" \
    --kparam-anchor-epochs 1000 \
    --kparam-anchor-probability 0.5 \
    --kparam-log10-std 1.25 \
    --gammaparam-anchor-probability 0.6 \
    --base-train-points 8000 --max-train-points 9000 --train-point-growth 0 \
    --save-every 250 --progress-every 25 --empty-cache-every 100 \
    --abort-on-nan --reset-best-score \
    --early-stop-check-every 100 --early-stop-patience-checks 4 \
    --early-stop-min-epochs 1000 --early-stop-validation-points 96 \
    --pde-thin-weight 10 --pde-ext-weight 10 \
    --thin-interface-weight 300 --ext-interface-weight 200 --bounds-weight 30 \
    2>&1 | tee "${LOG_DIR}/training.log"

echo "Physics-only training complete: ${RUN_ROOT}"
echo "Best checkpoint: ${CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_kgparam_best.pth"
