#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/curved_film_3d/prototype}"

cd "$ROOT_DIR"
python -m pip install -q -r requirements.txt
python -m unittest -v test_curved_film_3d.py
python run_curved_film_3d.py \
  --output "$OUTPUT_DIR" \
  --steps "${STEPS:-33}" \
  --duration "${DURATION:-0.05}" \
  --nx "${NX:-10}" \
  --ny "${NY:-9}" \
  --nz "${NZ:-24}" \
  --cap-profile "${CAP_PROFILE:-spherical}" \
  --gamma "${GAMMA:-10}" \
  --k-cat "${K_CAT:-1}" \
  --heterogeneity-x "${HETERO_X:-0.55}" \
  --heterogeneity-xy "${HETERO_XY:-0.25}"

if [[ "${RUN_CONVERGENCE:-1}" == "1" ]]; then
  python analyze_curved_film_3d_convergence.py \
    --output "$OUTPUT_DIR/convergence" \
    --steps "${CONVERGENCE_STEPS:-13}" \
    --duration "${CONVERGENCE_DURATION:-0.015}"
fi

if [[ "${RUN_FDM_VALIDATION:-1}" == "1" ]]; then
  python validate_curved_film_3d_fdm.py \
    --output "$OUTPUT_DIR/fdm_validation" \
    --operator-grid "${OPERATOR_GRID:-8,7,18}" \
    --operator-steps "${OPERATOR_STEPS:-65}" \
    --fdm-grid "${FDM_GRID:-14,13,36}" \
    --fdm-steps "${FDM_STEPS:-193}"
fi
