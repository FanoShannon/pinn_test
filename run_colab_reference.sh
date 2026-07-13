#!/usr/bin/env bash
set -euo pipefail

# Exact highgamma-product-integral reference chain. FDM is posterior-only.
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
K_CAT="${K_CAT:-}"
GAMMA="${GAMMA:-}"
FDM_PKL="${FDM_PKL:-}"
[[ -n "$K_CAT" ]] || { echo "Set K_CAT, for example K_CAT=1" >&2; exit 2; }
[[ -n "$GAMMA" ]] || { echo "Set GAMMA, for example GAMMA=10" >&2; exit 2; }
[[ -f "$FDM_PKL" ]] || { echo "Set FDM_PKL to the matching posterior FDM" >&2; exit 2; }

RUN_TAG="${RUN_TAG:-reference_k${K_CAT}_gamma${GAMMA}}"
RUN_ROOT="${RUN_ROOT:-/content/gdrive/MyDrive/pinn_v96_reference/${RUN_TAG}}"
CLEAN_ROOT="$RUN_ROOT/clean_two_stage"
PRODUCT_ROOT="$RUN_ROOT/productintegral_300"
FINAL_ROOT="$RUN_ROOT/final_productintegral_lift"
FDM_COMPARE_EVERY="${FDM_COMPARE_EVERY:-0}"
mkdir -p "$RUN_ROOT" "$FINAL_ROOT"

echo "Reference phase 1/3: Dynamic Stage 1 -> clean TraceGreen Stage 2"
WORK_DIR="$CODE_DIR" PYTHON="$PYTHON" RUN_ROOT="$CLEAN_ROOT" \
GAMMA="$GAMMA" K_CAT_STAR="$K_CAT" FDM_PKL="$FDM_PKL" \
FDM_COMPARE_EVERY="$FDM_COMPARE_EVERY" \
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5000}" \
STAGE2_EPOCHS="${CLEAN_STAGE2_EPOCHS:-1500}" \
bash "$CODE_DIR/run_colab_clean_two_stage_256x64.sh"

CLEAN_BEST="$CLEAN_ROOT/stage2_tracegreen_clean/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth"
[[ -f "$CLEAN_BEST" ]] || { echo "Missing clean best: $CLEAN_BEST" >&2; exit 1; }

echo "Reference phase 2/3: ProductIntegral 300-epoch fine-tune"
WORK_DIR="$CODE_DIR" PYTHON="$PYTHON" RUN_ROOT="$PRODUCT_ROOT" CLEAN_BEST="$CLEAN_BEST" \
GAMMA="$GAMMA" K_CAT_STAR="$K_CAT" FDM_PKL="$FDM_PKL" \
EPOCHS="${PRODUCT_EPOCHS:-300}" LEARNING_RATE="${PRODUCT_LR:-1e-6}" \
FDM_COMPARE_EVERY="$FDM_COMPARE_EVERY" \
bash "$CODE_DIR/run_colab_productintegral_finetune_256x64.sh"

PRODUCT_BEST="$PRODUCT_ROOT/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth"
[[ -f "$PRODUCT_BEST" ]] || { echo "Missing ProductIntegral best: $PRODUCT_BEST" >&2; exit 1; }

echo "Reference phase 3/3: zero-training inventory lift posterior"
"$PYTHON" -u "$CODE_DIR/compare_concentration_fields.py" \
  --arch multiscale_film_tracegreen_productintegral_lift \
  --input-mode normalized --gamma "$GAMMA" --k-cat-star "$K_CAT" \
  --checkpoint "$PRODUCT_BEST" --fdm-pkl "$FDM_PKL" \
  --green-time-grid 256 --green-kernel-points 64 --lift-time-grid 4096 \
  --n-time 160 --n-x-in 120 --n-x-out 160 --cv-points 2000 \
  --batch-size 4096 --current-mode both \
  --output-json "$FINAL_ROOT/reference_lift_metrics.json" \
  --output-npz "$FINAL_ROOT/reference_lift_fields.npz" \
  --output-figure "$FINAL_ROOT/reference_lift_residual.png" \
  --output-current-figure "$FINAL_ROOT/reference_lift_current.png"

echo "Reference reproduction complete"
echo "Clean best:          $CLEAN_BEST"
echo "ProductIntegral best: $PRODUCT_BEST"
echo "Final posterior:      $FINAL_ROOT/reference_lift_metrics.json"
