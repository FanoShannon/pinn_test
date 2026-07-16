#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/pinn_v96_delta_research}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-all}"

mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"

run_forward() {
    "$PYTHON" analyze_delta_forward.py \
        --output-dir "${OUTPUT_DIR}/forward_convergence"
    "$PYTHON" analyze_kgdelta_forward.py \
        --output-dir "${OUTPUT_DIR}/joint_stress"
}

run_inverse() {
    "$PYTHON" invert_delta_from_cv.py \
        --output-dir "${OUTPUT_DIR}/delta_inversion"
}

run_identifiability() {
    "$PYTHON" analyze_inverse_identifiability.py \
        --sigmas 40 \
        --output "${OUTPUT_DIR}/single_scan_identifiability.json"
    "$PYTHON" analyze_inverse_identifiability.py \
        --sigmas 5,40,320 \
        --output "${OUTPUT_DIR}/wide_scan_identifiability.json"
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
    all)
        run_forward
        run_inverse
        run_identifiability
        ;;
    *)
        echo "MODE must be forward, inverse, identifiability, or all" >&2
        exit 2
        ;;
esac

echo "Results: ${OUTPUT_DIR}"
