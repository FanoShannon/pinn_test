#!/usr/bin/env bash
set -euo pipefail

# One entry point. FDM is accepted only by posterior modes.
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
    RUN_TAG="${RUN_TAG:-base2_k${K_CAT}_gamma${GAMMA}}"
    RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
    mkdir -p "$RUN_ROOT"
    POINTS="${POINTS:-8000}"
    STAGE1_EPOCHS="${STAGE1_EPOCHS:-5000}"
    STAGE1_LR="${STAGE1_LR:-5e-5}"
    STAGE1_MIN_EPOCHS="${STAGE1_MIN_EPOCHS:-1500}"
    STAGE2_EPOCHS="${STAGE2_EPOCHS:-300}"
    STAGE2_LR="${STAGE2_LR:-1e-6}"
    STAGE2_MIN_EPOCHS="${STAGE2_MIN_EPOCHS:-100}"
    echo "Stage 1/2: direct ProductIntegral k=${K_CAT}, gamma=${GAMMA}"
    "$PYTHON" -u train_base.py \
      --stage-name stage1_productintegral \
      --k "$K_CAT" --gamma "$GAMMA" --epochs "$STAGE1_EPOCHS" \
      --lr "$STAGE1_LR" --points "$POINTS" --device "$DEVICE" \
      --early-stop-check-every 100 --early-stop-min-epochs "$STAGE1_MIN_EPOCHS" \
      --early-stop-patience 4 --validation-points 96 \
      --output-dir "$RUN_ROOT/stage1"
    STAGE1_BEST="$RUN_ROOT/stage1/base_best.pth"
    [[ -f "$STAGE1_BEST" ]] || { echo "Missing Stage 1 best: $STAGE1_BEST" >&2; exit 1; }
    echo "Stage 2/2: ProductIntegral low-LR fine-tune"
    "$PYTHON" -u train_base.py \
      --stage-name stage2_productintegral \
      --k "$K_CAT" --gamma "$GAMMA" --epochs "$STAGE2_EPOCHS" \
      --lr "$STAGE2_LR" --points "$POINTS" --device "$DEVICE" \
      --warm-start "$STAGE1_BEST" \
      --early-stop-check-every 50 --early-stop-min-epochs "$STAGE2_MIN_EPOCHS" \
      --early-stop-patience 4 --validation-points 96 \
      --output-dir "$RUN_ROOT/stage2"
    echo "Final two-stage base: $RUN_ROOT/stage2/base_best.pth"
    ;;
  base1)
    K_CAT="${K_CAT:-}"
    GAMMA="${GAMMA:-}"
    [[ -n "$K_CAT" ]] || { echo "Set K_CAT, for example K_CAT=1" >&2; exit 2; }
    [[ -n "$GAMMA" ]] || { echo "Set GAMMA, for example GAMMA=10" >&2; exit 2; }
    RUN_TAG="${RUN_TAG:-base1_k${K_CAT}_gamma${GAMMA}}"
    RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
    mkdir -p "$RUN_ROOT"
    EPOCHS="${EPOCHS:-1500}"
    LR="${LR:-5e-5}"
    POINTS="${POINTS:-4096}"
    echo "Single-stage ProductIntegral ablation k=${K_CAT}, gamma=${GAMMA}"
    "$PYTHON" -u train_base.py \
      --stage-name single_stage_ablation --k "$K_CAT" --gamma "$GAMMA" \
      --epochs "$EPOCHS" --lr "$LR" --points "$POINTS" --device "$DEVICE" \
      --no-early-stop --output-dir "$RUN_ROOT/base"
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
  audit)
    RUN_TAG="${RUN_TAG:-network_audit_$(date +%Y%m%d_%H%M%S)}"
    RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_minimal/${RUN_TAG}}"
    mkdir -p "$RUN_ROOT"
    BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to base_best.pth}"
    FDM_MANIFEST="${FDM_MANIFEST:?Set FDM_MANIFEST to kg_cases_v42.tsv}"
    echo "Network contribution audit: trained vs strict zero, both with inventory lift"
    "$PYTHON" -u posterior.py \
      --checkpoint "$BASE_CKPT" --manifest "$FDM_MANIFEST" \
      --network-mode both --device "$DEVICE" \
      --output-dir "$RUN_ROOT/network_audit"
    ;;
  *)
    echo "MODE must be base, base1, lift, kg, or audit; got ${MODE}" >&2
    exit 2
    ;;
esac

echo "Results: $RUN_ROOT"
