#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
CLEAN_BEST="${CLEAN_BEST:-}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/gdrive/MyDrive/pinn_v96}"
FDM_PKL="${FDM_PKL:-/content/gdrive/MyDrive/FDM_parameter_scale_0711/gamma1_k1_v42_thin_layer_catalytic_v42.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96/runs_inventory_hermite_lift/$(date +%Y%m%d_%H%M%S)}"

GAMMA="${GAMMA:-1.0}"
K_CAT_STAR="${K_CAT_STAR:-1.0}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
LIFT_TIME_GRID="${LIFT_TIME_GRID:-1024}"

if [[ -z "$CLEAN_BEST" || ! -f "$CLEAN_BEST" ]]; then
    echo "CLEAN_BEST path is missing or invalid; searching under $DRIVE_ROOT ..."
    CLEAN_BEST="$(find "$DRIVE_ROOT" -type f \
        -name 'pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth' \
        -print | sort | tail -n 1)"
fi
if [[ -z "$CLEAN_BEST" || ! -f "$CLEAN_BEST" ]]; then
    echo "ERROR: no clean physics-best checkpoint found under $DRIVE_ROOT." >&2
    echo "Run: find $DRIVE_ROOT -type f -name '*multiscale_film_tracegreen_clean*best.pth'" >&2
    exit 1
fi
if [[ ! -f "$FDM_PKL" ]]; then
    echo "ERROR: FDM reference not found: $FDM_PKL" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$WORK_DIR"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Using clean checkpoint: $CLEAN_BEST"
echo "Using FDM reference: $FDM_PKL"

"$PYTHON" -u compare_concentration_fields.py \
    --arch multiscale_film_tracegreen_conservative_lift \
    --input-mode normalized \
    --gamma "$GAMMA" \
    --k-cat-star "$K_CAT_STAR" \
    --checkpoint "$CLEAN_BEST" \
    --fdm-pkl "$FDM_PKL" \
    --green-time-grid "$GREEN_TIME_GRID" \
    --green-kernel-points "$GREEN_KERNEL_POINTS" \
    --lift-time-grid "$LIFT_TIME_GRID" \
    --n-time 160 \
    --n-x-in 120 \
    --n-x-out 160 \
    --cv-points 2000 \
    --batch-size 4096 \
    --current-mode both \
    --output-json "$OUTPUT_DIR/inventory_lift_metrics.json" \
    --output-npz "$OUTPUT_DIR/inventory_lift_fields.npz" \
    --output-figure "$OUTPUT_DIR/inventory_lift_residual.png" \
    --output-current-figure "$OUTPUT_DIR/inventory_lift_current.png" \
    2>&1 | tee "$OUTPUT_DIR/inventory_lift_eval.log"

echo "DONE: $OUTPUT_DIR"
