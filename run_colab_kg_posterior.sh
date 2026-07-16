#!/usr/bin/env bash
set -euo pipefail

# Joint zero-training posterior. It never trains or modifies the checkpoint.
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-}"
FDM_DIR="${FDM_DIR:-/content/gdrive/MyDrive/FDM_kg_v42}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96_productintegral_direct/kg_zero_shot_lift}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT to the fixed ProductIntegral best checkpoint." >&2
    exit 1
fi

cd "$CODE_DIR"
CASES=()
MISSING_CASES=()
add_case() {
    local k_value="$1"
    local gamma_value="$2"
    local prefix="$3"
    local path="${FDM_DIR}/${prefix}_thin_layer_catalytic_v42.pkl"
    if [[ -f "$path" ]]; then
        CASES+=(--case "${k_value},${gamma_value}=${path}")
    else
        MISSING_CASES+=("$path")
    fi
}

add_case 0.1 0.1 kg_k0p1_g0p1_v42
add_case 0.1 10 kg_k0p1_g10_v42
add_case 0.1 100 kg_k0p1_g100_v42
add_case 1 0.1 kg_k1_g0p1_v42
add_case 1 10 kg_k1_g10_v42
add_case 1 100 kg_k1_g100_v42
add_case 10 0.1 kg_k10_g0p1_v42
add_case 10 10 kg_k10_g10_v42
add_case 10 100 kg_k10_g100_v42
add_case 0.316227766 0.316227766 kg_k0p316_g0p316_v42
add_case 0.316227766 3.16227766 kg_k0p316_g3p162_v42
add_case 0.316227766 31.6227766 kg_k0p316_g31p62_v42
add_case 3.16227766 0.316227766 kg_k3p162_g0p316_v42
add_case 3.16227766 3.16227766 kg_k3p162_g3p162_v42
add_case 3.16227766 31.6227766 kg_k3p162_g31p62_v42
add_case 0.01 1 kg_k0p01_g1_v42
add_case 100 1 kg_k100_g1_v42

if [[ ${#MISSING_CASES[@]} -ne 0 ]]; then
    echo "Missing required joint FDM cases:" >&2
    printf '  %s\n' "${MISSING_CASES[@]}" >&2
    exit 1
fi

CMD=(
    "$PYTHON" -u compare_kg_parameter_cases.py
    --checkpoint "$CHECKPOINT"
    --output-dir "$OUTPUT_DIR"
    --apply-inventory-lift
    --zero-shot-fixed-reference
    --green-time-grid 256
    --green-kernel-points 64
    --lift-time-grid 4096
    "${CASES[@]}"
)
"${CMD[@]}"

"$PYTHON" - "$OUTPUT_DIR/kg_parameter_summary.json" "$OUTPUT_DIR/kg_vs_historical.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
aggregate = summary["aggregate"]
reference = {
    "overall_dimensionless_rmse": {
        "mean": 7.035943227751474e-4,
        "worst": 1.1049047967171505e-3,
    },
    "CV_J_over_J_ref_rmse": {
        "mean": 1.1992785514117572e-2,
        "worst": 8.45022662526531e-2,
    },
}
comparison = {}
for metric, baseline in reference.items():
    current = aggregate[metric]
    comparison[metric] = {
        "current": {"mean": current["mean"], "worst": current["worst"]},
        "historical": baseline,
        "mean_change_percent": 100.0 * (current["mean"] / baseline["mean"] - 1.0),
        "worst_change_percent": 100.0 * (current["worst"] / baseline["worst"] - 1.0),
    }
output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print("\n=== Joint KG vs historical zero-shot + lift ===")
for metric, values in comparison.items():
    print(
        f"{metric}: mean {values['mean_change_percent']:+.3f}%, "
        f"worst {values['worst_change_percent']:+.3f}%"
    )
print(f"Comparison: {output_path}")
PY

echo "Results: $OUTPUT_DIR"
