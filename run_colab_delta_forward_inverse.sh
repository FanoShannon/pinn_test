#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96_delta_research}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-all}"
HISTORY_BACKEND="${HISTORY_BACKEND:-soe}"

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
    all)
        run_forward
        run_inverse
        run_identifiability
        run_benchmark
        ;;
    *)
        echo "MODE must be forward, inverse, identifiability, benchmark, or all" >&2
        exit 2
        ;;
esac

echo "Results: ${OUTPUT_DIR}"
