#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96_delta_research}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-all}"
HISTORY_BACKEND="${HISTORY_BACKEND:-soe}"
TRUE_K="${TRUE_K:-1}"
TRUE_GAMMA="${TRUE_GAMMA:-10}"
TRUE_DELTA="${TRUE_DELTA:-0.035}"
SCAN_RATES="${SCAN_RATES:-5,40,320}"
INVERSE_MODES="${INVERSE_MODES:-all,fix-gamma}"
NOISE_LEVELS="${NOISE_LEVELS:-0,0.001}"
N_STARTS="${N_STARTS:-3}"
LBFGS_ITERATIONS="${LBFGS_ITERATIONS:-36}"
TARGET_TIME_GRID="${TARGET_TIME_GRID:-2049}"
TARGET_MODES="${TARGET_MODES:-256}"
OBSERVATION_POINTS="${OBSERVATION_POINTS:-257}"
OPERATOR_MODES="${OPERATOR_MODES:-96}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"

run_forward() {
    "$PYTHON" analyze_delta_forward.py \
        --output-dir "${OUTPUT_DIR}/forward_convergence"
    "$PYTHON" analyze_kgdelta_forward.py \
        --history-backend "$HISTORY_BACKEND" \
        --output-dir "${OUTPUT_DIR}/joint_stress"
}

run_inverse() {
    "$PYTHON" invert_delta_from_cv.py \
        --history-backend "$HISTORY_BACKEND" \
        --output-dir "${OUTPUT_DIR}/delta_inversion"
}

run_identifiability() {
    "$PYTHON" analyze_inverse_identifiability.py \
        --sigmas 40 \
        --history-backend "$HISTORY_BACKEND" \
        --output "${OUTPUT_DIR}/single_scan_identifiability.json"
    "$PYTHON" analyze_inverse_identifiability.py \
        --sigmas 5,40,320 \
        --history-backend "$HISTORY_BACKEND" \
        --output "${OUTPUT_DIR}/wide_scan_identifiability.json"
}

run_benchmark() {
    "$PYTHON" benchmark_fast_abel_history.py \
        --output "${OUTPUT_DIR}/fast_history_benchmark.json"
}

run_multiscan_design() {
    "$PYTHON" design_multiscan_rates.py \
        --history-backend "$HISTORY_BACKEND" \
        --parameter-cases "nominal:${TRUE_K}:${TRUE_GAMMA}:${TRUE_DELTA}" \
        --output "${OUTPUT_DIR}/multiscan_rate_design.json"
}

run_multiscan_inverse() {
    local arguments=(
        --true-k "$TRUE_K"
        --true-gamma "$TRUE_GAMMA"
        --true-delta "$TRUE_DELTA"
        --sigmas "$SCAN_RATES"
        --inverse-modes "$INVERSE_MODES"
        --noise-levels "$NOISE_LEVELS"
        --n-starts "$N_STARTS"
        --lbfgs-iterations "$LBFGS_ITERATIONS"
        --target-time-grid "$TARGET_TIME_GRID"
        --target-modes "$TARGET_MODES"
        --observation-points "$OBSERVATION_POINTS"
        --operator-modes "$OPERATOR_MODES"
        --history-backend "$HISTORY_BACKEND"
        --device "$DEVICE"
        --output-dir "${OUTPUT_DIR}/multiscan_parameter_inverse"
    )
    if [[ -n "${STARTS:-}" ]]; then
        arguments+=(--starts "$STARTS")
    fi
    "$PYTHON" invert_multiscan_parameters.py "${arguments[@]}"
}

case "$MODE" in
    forward)
        run_forward
        ;;
    inverse)
        run_inverse
        ;;
    identifiability)
        run_identifiability
        ;;
    benchmark)
        run_benchmark
        ;;
    multiscan-design)
        run_multiscan_design
        ;;
    multiscan-inverse)
        run_multiscan_inverse
        ;;
    multiscan)
        run_multiscan_design
        run_multiscan_inverse
        ;;
    all)
        run_forward
        run_inverse
        run_identifiability
        run_benchmark
        ;;
    *)
        echo "MODE must be forward, inverse, identifiability, benchmark, multiscan-design, multiscan-inverse, multiscan, or all" >&2
        exit 2
        ;;
esac

echo "Results: ${OUTPUT_DIR}"
