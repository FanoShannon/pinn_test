#!/usr/bin/env bash
set -euo pipefail

# Separate posterior process. It never trains or modifies the checkpoint.
WORK_DIR="${WORK_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-}"
FDM_DIR="${FDM_DIR:-/content/gdrive/MyDrive/FDM_kg_v42}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96_kg_runs/posterior}"
ZERO_SHOT_FIXED_REFERENCE="${ZERO_SHOT_FIXED_REFERENCE:-0}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT to a frozen fixed-reference or kgparam checkpoint." >&2
    exit 1
fi

cd "$WORK_DIR"
CASES=()
add_case() {
    local k_value="$1"
    local gamma_value="$2"
    local prefix="$3"
    local path="${FDM_DIR}/${prefix}_thin_layer_catalytic_v42.pkl"
    [[ -f "$path" ]] && CASES+=(--case "${k_value},${gamma_value}=${path}")
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

if [[ ${#CASES[@]} -eq 0 ]]; then
    echo "No joint FDM cases found in ${FDM_DIR}." >&2
    exit 1
fi

CMD=(
    "$PYTHON" -u compare_kg_parameter_cases.py
    --checkpoint "$CHECKPOINT"
    --output-dir "$OUTPUT_DIR"
    --apply-inventory-lift
    --green-time-grid 256
    --green-kernel-points 64
    --lift-time-grid 4096
    "${CASES[@]}"
)
[[ "$ZERO_SHOT_FIXED_REFERENCE" == "1" ]] && CMD+=(--zero-shot-fixed-reference)
"${CMD[@]}"
