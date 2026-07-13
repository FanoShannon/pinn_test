#!/usr/bin/env bash
set -euo pipefail

# One entry point, three explicit jobs. FDM is accepted only by posterior modes.
MODE="${MODE:-kg}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_TAG="${RUN_TAG:-minimal_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$RUN_ROOT"
cd "$CODE_DIR"

case "$MODE" in
  base)
    BASE_K="${BASE_K:?Set BASE_K to the fixed base k_cat}"
    BASE_GAMMA="${BASE_GAMMA:?Set BASE_GAMMA to the fixed base gamma}"
    EPOCHS="${EPOCHS:-1500}"
    LR="${LR:-5e-5}"
    POINTS="${POINTS:-4096}"
    echo "Training physics-only base k=${BASE_K}, gamma=${BASE_GAMMA}"
    "$PYTHON" -u train_base.py \
      --k "$BASE_K" --gamma "$BASE_GAMMA" --epochs "$EPOCHS" \
      --lr "$LR" --points "$POINTS" --device "$DEVICE" \
      --output-dir "$RUN_ROOT/base"
    ;;
  lift)
    BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to base_best.pth}"
    FDM_PKL="${FDM_PKL:?Set FDM_PKL for posterior validation}"
    echo "Zero-training posterior with mandatory inventory lift"
    "$PYTHON" -u posterior.py \
      --checkpoint "$BASE_CKPT" --fdm-pkl "$FDM_PKL" \
      --device "$DEVICE" --output-dir "$RUN_ROOT/lift_posterior"
    ;;
  kg)
    BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to base_best.pth}"
    FDM_MANIFEST="${FDM_MANIFEST:?Set FDM_MANIFEST to kg_cases_v42.tsv}"
    echo "Joint k/gamma zero-training posterior with mandatory inventory lift"
    "$PYTHON" -u posterior.py \
      --checkpoint "$BASE_CKPT" --manifest "$FDM_MANIFEST" \
      --device "$DEVICE" --output-dir "$RUN_ROOT/kg_lift_posterior"
    ;;
  *)
    echo "MODE must be base, lift, or kg; got ${MODE}" >&2
    exit 2
    ;;
esac

echo "Results: $RUN_ROOT"
