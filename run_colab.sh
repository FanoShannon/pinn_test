#!/usr/bin/env bash
set -euo pipefail

# One entry point, three explicit jobs. FDM is accepted only by posterior modes.
MODE="${MODE:-kg}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
cd "$CODE_DIR"

case "$MODE" in
  base)
    K_CAT="${K_CAT:-}"
    GAMMA="${GAMMA:-}"
    [[ -n "$K_CAT" ]] || { echo "Set K_CAT, for example K_CAT=1" >&2; exit 2; }
    [[ -n "$GAMMA" ]] || { echo "Set GAMMA, for example GAMMA=10" >&2; exit 2; }
    RUN_TAG="${RUN_TAG:-base_k${K_CAT}_gamma${GAMMA}}"
    RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
    mkdir -p "$RUN_ROOT"
    EPOCHS="${EPOCHS:-1500}"
    LR="${LR:-5e-5}"
    POINTS="${POINTS:-4096}"
    echo "Training physics-only base k=${K_CAT}, gamma=${GAMMA}"
    "$PYTHON" -u train_base.py \
      --k "$K_CAT" --gamma "$GAMMA" --epochs "$EPOCHS" \
      --lr "$LR" --points "$POINTS" --device "$DEVICE" \
      --output-dir "$RUN_ROOT/base"
    ;;
  lift)
    RUN_TAG="${RUN_TAG:-lift_$(date +%Y%m%d_%H%M%S)}"
    RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
    mkdir -p "$RUN_ROOT"
    BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to base_best.pth}"
    FDM_PKL="${FDM_PKL:?Set FDM_PKL for posterior validation}"
    echo "Zero-training posterior with mandatory inventory lift"
    "$PYTHON" -u posterior.py \
      --checkpoint "$BASE_CKPT" --fdm-pkl "$FDM_PKL" \
      --device "$DEVICE" --output-dir "$RUN_ROOT/lift_posterior"
    ;;
  kg)
    RUN_TAG="${RUN_TAG:-kg_$(date +%Y%m%d_%H%M%S)}"
    RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
    mkdir -p "$RUN_ROOT"
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
