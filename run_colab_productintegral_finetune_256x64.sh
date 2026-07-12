#!/usr/bin/env bash
set -euo pipefail

# Apply the fixed, singularity-matched interface trace to a clean Stage 2
# checkpoint, report the zero-training transform, then fine-tune the remaining
# thin/external parameters. FDM is posterior-only and never enters a loss,
# early stopping, or checkpoint selection.

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
CLEAN_BEST="${CLEAN_BEST:-}"
FDM_PKL="${FDM_PKL:-}"
GAMMA="${GAMMA:-10.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96/runs_productintegral/${RUN_TAG}_gamma${GAMMA}}"

GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
EPOCHS="${EPOCHS:-300}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"
FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-50}"
COMPARE_BATCH_SIZE="${COMPARE_BATCH_SIZE:-4096}"

ARCH="multiscale_film_tracegreen_productintegral"
CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
ZERO_DIR="$RUN_ROOT/zero_training_compare"
FINAL_DIR="$RUN_ROOT/final_compare"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$CHECKPOINT_DIR" "$ZERO_DIR" "$FINAL_DIR" "$LOG_DIR"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$CLEAN_BEST" || ! -f "$CLEAN_BEST" ]]; then
    echo "ERROR: set CLEAN_BEST to an existing Stage 2 clean physics-best checkpoint."
    exit 1
fi
if [[ -z "$FDM_PKL" || ! -f "$FDM_PKL" ]]; then
    echo "ERROR: set FDM_PKL to the matching posterior-only FDM pickle."
    exit 1
fi

cd "$WORK_DIR"

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

compare_checkpoint() {
    local checkpoint="$1"
    local output_dir="$2"
    local label="$3"
    mkdir -p "$output_dir"
    "$PYTHON" -u compare_concentration_fields.py \
        --arch "$ARCH" \
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
        --cv-points 2000 \
        --batch-size "$COMPARE_BATCH_SIZE" \
        --current-mode both \
        --output-json "$output_dir/${label}_metrics.json" \
        --output-npz "$output_dir/${label}_fields.npz" \
        --output-figure "$output_dir/${label}_residual.png" \
        --output-current-figure "$output_dir/${label}_current.png"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Zero-training product-integral evaluation"
compare_checkpoint "$CLEAN_BEST" "$ZERO_DIR" "productintegral_zero_training" \
    2>&1 | tee "$LOG_DIR/zero_training_compare.log"

TRAIN_CMD=(
    "$PYTHON" -u pinn_thin_layer_v9_6.py
    --arch "$ARCH"
    --gamma "$GAMMA"
    --k-cat-star "$K_CAT_STAR"
    --epochs "$EPOCHS"
    --resume-checkpoint "$CLEAN_BEST"
    --reset-optimizer-state
    --reset-best-score
    --checkpoint-dir "$CHECKPOINT_DIR"
    --learning-rate "$LEARNING_RATE"
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
    --save-every 50
    --progress-every 25
    --empty-cache-every 100
    --abort-on-nan
    --early-stop-check-every 50
    --early-stop-patience-checks 4
    --early-stop-min-epochs 100
    --early-stop-min-relative-improvement 0.005
    --early-stop-ema-alpha 0.5
    --early-stop-validation-points 96
    --fdm-compare-pkl "$FDM_PKL"
    --fdm-compare-every "$FDM_COMPARE_EVERY"
    --fdm-compare-dir "$RUN_ROOT/fdm_compare"
    --fdm-compare-batch-size 8192
    --fdm-compare-save-figure
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fine-tuning: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/productintegral_finetune.log"

BEST="$CHECKPOINT_DIR/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth"
if [[ ! -f "$BEST" ]]; then
    echo "ERROR: product-integral best checkpoint was not created: $BEST"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final physics-best posterior evaluation"
compare_checkpoint "$BEST" "$FINAL_DIR" "productintegral_finetuned_best" \
    2>&1 | tee "$LOG_DIR/final_compare.log"

echo "Completed."
echo "Zero-training metrics: $ZERO_DIR/productintegral_zero_training_metrics.json"
echo "Best checkpoint:       $BEST"
echo "Final metrics:         $FINAL_DIR/productintegral_finetuned_best_metrics.json"
