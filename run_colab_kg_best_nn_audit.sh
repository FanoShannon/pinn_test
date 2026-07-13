#!/usr/bin/env bash
set -euo pipefail

# Posterior-only four-way ablation of the fixed k=1,gamma=10 best checkpoint.
WORK_DIR="${WORK_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-}"
FDM_DIR="${FDM_DIR:-/content/gdrive/MyDrive/FDM_kg_v42}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96_kg_nn_audit}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT to the epoch-2400 fixed k=1,gamma=10 ProductIntegral best checkpoint." >&2
    exit 1
fi

cd "$WORK_DIR"
CASES=()
add_case() {
    local k_value="$1" gamma_value="$2" prefix="$3"
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
    echo "No FDM cases found in ${FDM_DIR}." >&2
    exit 1
fi

run_variant() {
    local name="$1" lift="$2" zero_nn="$3"
    local cmd=(
        "$PYTHON" -u compare_kg_parameter_cases.py
        --checkpoint "$CHECKPOINT"
        --output-dir "$OUTPUT_DIR/$name"
        --zero-shot-fixed-reference
        --skip-figures
        --green-time-grid 256
        --green-kernel-points 64
        --lift-time-grid 4096
        "${CASES[@]}"
    )
    [[ "$lift" == "1" ]] && cmd+=(--apply-inventory-lift)
    [[ "$zero_nn" == "1" ]] && cmd+=(--disable-thin-network)
    echo "=== ${name} ==="
    "${cmd[@]}"
}

run_variant trained_nn_no_lift 0 0
run_variant zero_nn_no_lift 0 1
run_variant trained_nn_lift 1 0
run_variant zero_nn_lift 1 1

"$PYTHON" -u summarize_kg_nn_audit.py --audit-dir "$OUTPUT_DIR"
echo "Results: $OUTPUT_DIR/kg_nn_contribution_summary.json"
