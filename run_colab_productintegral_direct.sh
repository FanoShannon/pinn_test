#!/usr/bin/env bash
set -euo pipefail

# Single supported workflow:
# Dynamic Stage 1 -> ProductIntegral 300 -> posterior inventory lift.
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
K_CAT="${K_CAT:-}"
GAMMA="${GAMMA:-}"
FDM_PKL="${FDM_PKL:-}"
STAGE1_BEST="${STAGE1_BEST:-}"

[[ -n "$K_CAT" ]] || { echo "Set K_CAT, for example K_CAT=1.0" >&2; exit 2; }
[[ -n "$GAMMA" ]] || { echo "Set GAMMA, for example GAMMA=10.0" >&2; exit 2; }
[[ -f "$FDM_PKL" ]] || { echo "Set FDM_PKL to the matching posterior FDM." >&2; exit 2; }

RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_productintegral_direct/direct_k${K_CAT}_gamma${GAMMA}}"
STAGE1_DIR="$RUN_ROOT/stage1_dynamic"
PRODUCT_DIR="$RUN_ROOT/productintegral_300"
FINAL_DIR="$RUN_ROOT/final_inventory_lift"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$STAGE1_DIR/checkpoints" "$PRODUCT_DIR/checkpoints" "$FINAL_DIR" "$LOG_DIR"

GREEN_TIME_GRID=256
GREEN_KERNEL_POINTS=64
BASE_TRAIN_POINTS=8000
MAX_TRAIN_POINTS=9000
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5000}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
PRODUCT_EPOCHS="${PRODUCT_EPOCHS:-300}"
PRODUCT_LR="${PRODUCT_LR:-2e-6}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

if [[ -z "$STAGE1_BEST" ]]; then
    STAGE1_BEST="$STAGE1_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth"
    echo "Phase 1/3: training Dynamic Stage 1"
    "$PYTHON" -u pinn_thin_layer_v9_6.py \
      --arch multiscale_green_grid_dynamic_stage1 \
      --gamma "$GAMMA" --k-cat-star "$K_CAT" \
      --epochs "$STAGE1_EPOCHS" \
      --reset-optimizer-state --reset-best-score \
      --checkpoint-dir "$STAGE1_DIR/checkpoints" \
      --learning-rate "$STAGE1_LR" \
      --green-time-grid "$GREEN_TIME_GRID" \
      --green-kernel-points "$GREEN_KERNEL_POINTS" \
      --base-train-points "$BASE_TRAIN_POINTS" \
      --max-train-points "$MAX_TRAIN_POINTS" \
      --train-point-growth 0 \
      --pde-thin-weight 10.0 --pde-ext-weight 10.0 \
      --thin-interface-weight 300.0 --ext-interface-weight 200.0 \
      --bounds-weight 30.0 \
      --save-every 100 --progress-every 25 --empty-cache-every 100 \
      --abort-on-nan \
      --early-stop-check-every 100 \
      --early-stop-patience-checks 4 \
      --early-stop-min-epochs 1500 \
      --early-stop-min-relative-improvement 0.005 \
      --early-stop-ema-alpha 0.5 \
      --early-stop-validation-points 96 \
      2>&1 | tee "$LOG_DIR/stage1_dynamic.log"
else
    echo "Phase 1/3: reusing Dynamic Stage 1 checkpoint"
fi

[[ -f "$STAGE1_BEST" ]] || {
    echo "Missing Dynamic Stage 1 best: $STAGE1_BEST" >&2
    exit 1
}

PRODUCT_BEST="$PRODUCT_DIR/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth"
echo "Phase 2/3: Dynamic Stage 1 -> ProductIntegral ${PRODUCT_EPOCHS}"
"$PYTHON" -u pinn_thin_layer_v9_6.py \
  --arch multiscale_film_tracegreen_productintegral \
  --gamma "$GAMMA" --k-cat-star "$K_CAT" \
  --epochs "$PRODUCT_EPOCHS" \
  --resume-checkpoint "$STAGE1_BEST" \
  --reset-optimizer-state --reset-best-score \
  --checkpoint-dir "$PRODUCT_DIR/checkpoints" \
  --learning-rate "$PRODUCT_LR" \
  --green-time-grid "$GREEN_TIME_GRID" \
  --green-kernel-points "$GREEN_KERNEL_POINTS" \
  --base-train-points "$BASE_TRAIN_POINTS" \
  --max-train-points "$MAX_TRAIN_POINTS" \
  --train-point-growth 0 \
  --pde-thin-weight 10.0 --pde-ext-weight 10.0 \
  --thin-interface-weight 300.0 --ext-interface-weight 200.0 \
  --bounds-weight 30.0 \
  --clean-residual-initial-scale 0.0 \
  --clean-residual-decay-epochs 0 \
  --save-every 100 --progress-every 25 --empty-cache-every 100 \
  --abort-on-nan \
  --early-stop-check-every 100 \
  --early-stop-patience-checks 4 \
  --early-stop-min-epochs 100 \
  --early-stop-min-relative-improvement 0.005 \
  --early-stop-ema-alpha 0.5 \
  --early-stop-validation-points 96 \
  2>&1 | tee "$LOG_DIR/productintegral_300.log"

[[ -f "$PRODUCT_BEST" ]] || {
    echo "Missing ProductIntegral best: $PRODUCT_BEST" >&2
    exit 1
}

echo "Phase 3/3: posterior-only inventory lift"
FINAL_METRICS="$FINAL_DIR/metrics.json"
"$PYTHON" -u compare_concentration_fields.py \
  --arch multiscale_film_tracegreen_productintegral_lift \
  --input-mode normalized \
  --gamma "$GAMMA" --k-cat-star "$K_CAT" \
  --checkpoint "$PRODUCT_BEST" --fdm-pkl "$FDM_PKL" \
  --green-time-grid "$GREEN_TIME_GRID" \
  --green-kernel-points "$GREEN_KERNEL_POINTS" \
  --lift-time-grid 4096 \
  --n-time 160 --n-x-in 120 --n-x-out 160 --cv-points 2000 \
  --batch-size 4096 --current-mode both \
  --output-json "$FINAL_METRICS" \
  --output-npz "$FINAL_DIR/fields.npz" \
  --output-figure "$FINAL_DIR/residual.png" \
  --output-current-figure "$FINAL_DIR/current.png" \
  2>&1 | tee "$LOG_DIR/final_inventory_lift.log"

COMPARISON="$FINAL_DIR/reproduction_vs_historical_best.json"
"$PYTHON" - "$FINAL_METRICS" "$COMPARISON" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
candidate = {
    "overall_dimensionless_rmse": float(metrics["overall_dimensionless"]["rmse"]),
    "CV_J_over_J_ref_rmse": float(metrics["CV_J_over_J_ref"]["rmse"]),
}
reference = {
    "overall_dimensionless_rmse": 6.522669005881162e-4,
    "CV_J_over_J_ref_rmse": 1.501318934104662e-3,
}
comparison = {}
for name in candidate:
    comparison[name] = {
        "candidate": candidate[name],
        "historical_best": reference[name],
        "relative_change_percent": 100.0 * (
            candidate[name] / reference[name] - 1.0
        ),
    }
worst = max(abs(item["relative_change_percent"]) for item in comparison.values())
result = {
    "comparison": comparison,
    "within_1_percent": worst <= 1.0,
    "within_3_percent": worst <= 3.0,
    "within_5_percent": worst <= 5.0,
    "fdm_role": "posterior_only",
}
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("\n=== Reproduction vs historical epoch-2400 best ===")
for name, item in comparison.items():
    print(
        f"{name}: {item['candidate']:.9e} vs "
        f"{item['historical_best']:.9e} "
        f"({item['relative_change_percent']:+.3f}%)"
    )
print(
    "Status:",
    f"within1={result['within_1_percent']}, "
    f"within3={result['within_3_percent']}, "
    f"within5={result['within_5_percent']}",
)
PY

echo "Complete"
echo "Stage 1 best:         $STAGE1_BEST"
echo "ProductIntegral best: $PRODUCT_BEST"
echo "Final metrics:        $FINAL_METRICS"
echo "Reproduction report:  $COMPARISON"
