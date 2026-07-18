#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/curved_pi_dtn_bie}"

cd "$ROOT_DIR"
python -m pip install -q -r requirements.txt
python -m unittest -v test_surface_heat_bem.py test_curved_pi_dtn_bie.py

python run_curved_pi_dtn_bie.py \
  --output "$OUTPUT_DIR/forward" \
  --steps "${STEPS:-65}" \
  --nx "${SURFACE_NX:-4}" \
  --ny "${SURFACE_NY:-4}" \
  --film-modes "${FILM_MODES:-64}" \
  --history-quadrature "${HISTORY_QUADRATURE:-6}"

if [[ "${RUN_FDM_POSTERIOR:-1}" == "1" ]]; then
  python validate_curved_pi_dtn_bie_fdm.py \
    --output "$OUTPUT_DIR/fdm_posterior" \
    --steps "${POSTERIOR_STEPS:-65}" \
    --surface-grid "${POSTERIOR_SURFACE_GRID:-4,4}" \
    --fdm-grid "${POSTERIOR_FDM_GRID:-8,7,108}" \
    --fdm-length-z "${POSTERIOR_FDM_LENGTH_Z:-3}" \
    --periodic-images "${PERIODIC_IMAGES:-3}" \
    --film-modes "${FILM_MODES:-64}" \
    --history-quadrature "${HISTORY_QUADRATURE:-6}"
fi
