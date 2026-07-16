#!/usr/bin/env bash
set -euo pipefail

# Reuse a frozen Dynamic Stage 1 checkpoint, train ProductIntegral directly
# for 300 epochs, then apply the posterior-only inventory lift.
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
K_CAT="${K_CAT:-1.0}"
GAMMA="${GAMMA:-10.0}"
FDM_PKL="${FDM_PKL:-}"

REFERENCE_ROOT="${REFERENCE_ROOT:-/content/gdrive/MyDrive/pinn_v96_reference/reference_k${K_CAT}_gamma${GAMMA}}"
STAGE1_BEST="${STAGE1_BEST:-${REFERENCE_ROOT}/clean_two_stage/stage1_dynamic_fixed/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_reference_direct/reference_k${K_CAT}_gamma${GAMMA}}"
PRODUCT_ROOT="$RUN_ROOT/productintegral_300_from_stage1"
FINAL_ROOT="$RUN_ROOT/final_productintegral_lift"
THREE_STAGE_METRICS="${THREE_STAGE_METRICS:-${REFERENCE_ROOT}/final_productintegral_lift/reference_lift_metrics.json}"

[[ -f "$STAGE1_BEST" ]] || {
    echo "Missing Dynamic Stage 1 best: $STAGE1_BEST" >&2
    exit 1
}
[[ -f "$FDM_PKL" ]] || {
    echo "Set FDM_PKL to the matching posterior FDM." >&2
    exit 1
}
mkdir -p "$PRODUCT_ROOT" "$FINAL_ROOT"

echo "Direct reference phase 1/2: Dynamic Stage 1 -> ProductIntegral 300"
WORK_DIR="$CODE_DIR" PYTHON="$PYTHON" RUN_ROOT="$PRODUCT_ROOT" \
CLEAN_BEST="$STAGE1_BEST" GAMMA="$GAMMA" K_CAT_STAR="$K_CAT" \
FDM_PKL="$FDM_PKL" EPOCHS=300 LEARNING_RATE=2e-6 \
SAVE_EVERY=100 EARLY_STOP_CHECK_EVERY=100 FDM_COMPARE_EVERY=0 \
bash "$CODE_DIR/run_colab_productintegral_finetune_256x64.sh"

PRODUCT_BEST="$PRODUCT_ROOT/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth"
[[ -f "$PRODUCT_BEST" ]] || {
    echo "Missing ProductIntegral best: $PRODUCT_BEST" >&2
    exit 1
}

echo "Direct reference phase 2/2: zero-training inventory lift posterior"
"$PYTHON" -u "$CODE_DIR/compare_concentration_fields.py" \
  --arch multiscale_film_tracegreen_productintegral_lift \
  --input-mode normalized --gamma "$GAMMA" --k-cat-star "$K_CAT" \
  --checkpoint "$PRODUCT_BEST" --fdm-pkl "$FDM_PKL" \
  --green-time-grid 256 --green-kernel-points 64 --lift-time-grid 4096 \
  --n-time 160 --n-x-in 120 --n-x-out 160 --cv-points 2000 \
  --batch-size 4096 --current-mode both \
  --output-json "$FINAL_ROOT/reference_direct_lift_metrics.json" \
  --output-npz "$FINAL_ROOT/reference_direct_lift_fields.npz" \
  --output-figure "$FINAL_ROOT/reference_direct_lift_residual.png" \
  --output-current-figure "$FINAL_ROOT/reference_direct_lift_current.png"

DIRECT_METRICS="$FINAL_ROOT/reference_direct_lift_metrics.json"
COMPARISON_JSON="$FINAL_ROOT/reference_direct_vs_best.json"
"$PYTHON" - "$DIRECT_METRICS" "$THREE_STAGE_METRICS" "$COMPARISON_JSON" <<'PY'
import json
import sys
from pathlib import Path

direct_path = Path(sys.argv[1])
three_stage_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

direct = json.loads(direct_path.read_text(encoding="utf-8"))
direct_overall = float(direct["overall_dimensionless"]["rmse"])
direct_cv = float(direct["CV_J_over_J_ref"]["rmse"])

historical = {
    "overall_dimensionless_rmse": 6.522669005881162e-4,
    "CV_J_over_J_ref_rmse": 1.501318934104662e-3,
}


def compare(reference):
    return {
        "overall_dimensionless_rmse": {
            "candidate": direct_overall,
            "reference": reference["overall_dimensionless_rmse"],
            "relative_change_percent": 100.0 * (
                direct_overall / reference["overall_dimensionless_rmse"] - 1.0
            ),
        },
        "CV_J_over_J_ref_rmse": {
            "candidate": direct_cv,
            "reference": reference["CV_J_over_J_ref_rmse"],
            "relative_change_percent": 100.0 * (
                direct_cv / reference["CV_J_over_J_ref_rmse"] - 1.0
            ),
        },
    }


result = {
    "direct_metrics": str(direct_path),
    "historical_epoch2400_best": compare(historical),
}

if three_stage_path.is_file():
    three_stage = json.loads(three_stage_path.read_text(encoding="utf-8"))
    three_stage_reference = {
        "overall_dimensionless_rmse": float(
            three_stage["overall_dimensionless"]["rmse"]
        ),
        "CV_J_over_J_ref_rmse": float(
            three_stage["CV_J_over_J_ref"]["rmse"]
        ),
    }
    result["three_stage_reference"] = {
        "metrics": str(three_stage_path),
        **compare(three_stage_reference),
    }

changes = [
    abs(item["relative_change_percent"])
    for item in result["historical_epoch2400_best"].values()
]
result["reproduction_status"] = {
    "within_1_percent": max(changes) <= 1.0,
    "within_3_percent": max(changes) <= 3.0,
    "within_5_percent": max(changes) <= 5.0,
}
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("\n=== Direct Stage1 -> ProductIntegral vs historical epoch-2400 best ===")
for name, item in result["historical_epoch2400_best"].items():
    print(
        f"{name}: {item['candidate']:.9e} vs {item['reference']:.9e} "
        f"({item['relative_change_percent']:+.3f}%)"
    )
if "three_stage_reference" in result:
    print("\n=== Direct chain vs three-stage reproduction ===")
    for name in ("overall_dimensionless_rmse", "CV_J_over_J_ref_rmse"):
        item = result["three_stage_reference"][name]
        print(
            f"{name}: {item['candidate']:.9e} vs {item['reference']:.9e} "
            f"({item['relative_change_percent']:+.3f}%)"
        )
print("Reproduction status:", result["reproduction_status"])
print("Saved comparison:", output_path)
PY

echo "Direct Stage1 -> ProductIntegral reproduction complete"
echo "Stage 1 best:         $STAGE1_BEST"
echo "ProductIntegral best: $PRODUCT_BEST"
echo "Final posterior:      $DIRECT_METRICS"
echo "Best comparison:      $COMPARISON_JSON"
