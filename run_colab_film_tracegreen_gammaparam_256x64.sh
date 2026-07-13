#!/usr/bin/env bash
set -euo pipefail

# Gamma generalization with k_cat=1 and delta=0.035. FDM is posterior-only.
# Run from the checked-out code directory by default; keep outputs in RUN_ROOT.
WORK_DIR="${WORK_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-gamma_productintegral_run01}"
RUN_ROOT="${RUN_ROOT:-${WORK_DIR}/runs_film_tracegreen_gammaparam_256x64/${RUN_TAG}}"
WARM_START_CKPT="${WARM_START_CKPT:-}"
EPOCHS="${EPOCHS:-0}"
LR="${LR:-2e-6}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
LIFT_TIME_GRID="${LIFT_TIME_GRID:-4096}"
ANCHOR_PROBABILITY="${ANCHOR_PROBABILITY:-0.6}"

FDM_DIR="${FDM_DIR:-/content/gdrive/MyDrive/FDM}"
FDM_G01="${FDM_G01:-${FDM_DIR}/gamma0p1_k1_v42_thin_layer_catalytic_v42.pkl}"
FDM_G0316="${FDM_G0316:-${FDM_DIR}/gamma0p316_k1_v42_thin_layer_catalytic_v42.pkl}"
FDM_G1="${FDM_G1:-${FDM_DIR}/gamma1_k1_v42_thin_layer_catalytic_v42.pkl}"
FDM_G3162="${FDM_G3162:-${FDM_DIR}/gamma3p162_k1_v42_thin_layer_catalytic_v42.pkl}"
FDM_G10="${FDM_G10:-${FDM_DIR}/kcat1_v42_thin_layer_catalytic_v42.pkl}"
FDM_G3162_HI="${FDM_G3162_HI:-${FDM_DIR}/gamma31p62_k1_v42_thin_layer_catalytic_v42.pkl}"
FDM_G100="${FDM_G100:-${FDM_DIR}/gamma100_k1_v42_thin_layer_catalytic_v42.pkl}"

if [[ -z "$WARM_START_CKPT" || ! -f "$WARM_START_CKPT" ]]; then
    echo "Set WARM_START_CKPT to the fixed gamma=10, k_cat=1 ProductIntegral checkpoint." >&2
    exit 1
fi

cd "$WORK_DIR"
CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
ZERO_DIR="${RUN_ROOT}/zero_shot_compare"
POST_DIR="${RUN_ROOT}/posterior_compare"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$ZERO_DIR" "$POST_DIR"

CASES=()
[[ -f "$FDM_G01" ]] && CASES+=(--case "0.1=$FDM_G01")
[[ -f "$FDM_G0316" ]] && CASES+=(--case "0.316227766=$FDM_G0316")
[[ -f "$FDM_G1" ]] && CASES+=(--case "1=$FDM_G1")
[[ -f "$FDM_G3162" ]] && CASES+=(--case "3.16227766=$FDM_G3162")
[[ -f "$FDM_G10" ]] && CASES+=(--case "10=$FDM_G10")
[[ -f "$FDM_G3162_HI" ]] && CASES+=(--case "31.6227766=$FDM_G3162_HI")
[[ -f "$FDM_G100" ]] && CASES+=(--case "100=$FDM_G100")

if [[ ${#CASES[@]} -gt 0 ]]; then
    "$PYTHON" -u compare_gamma_parameter_cases.py \
        --checkpoint "$WARM_START_CKPT" \
        --output-dir "$ZERO_DIR" \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --apply-inventory-lift \
        --lift-time-grid "$LIFT_TIME_GRID" \
        --zero-shot-fixed-reference \
        "${CASES[@]}" 2>&1 | tee "${LOG_DIR}/zero_shot_compare.log"
fi

if [[ "$EPOCHS" == "0" && ${#CASES[@]} -eq 0 ]]; then
    echo "EPOCHS=0 requested, but no matching gamma FDM files were found in $FDM_DIR." >&2
    exit 1
fi

if [[ "$EPOCHS" == "0" ]]; then
    echo "EPOCHS=0: zero-shot posterior complete; training skipped."
    echo "Summary: ${ZERO_DIR}/gamma_parameter_summary.json"
    exit 0
fi

"$PYTHON" -u pinn_thin_layer_v9_6.py \
    --arch multiscale_film_tracegreen_gammaparam \
    --gamma 10.0 --k-cat-star 1.0 \
    --epochs "$EPOCHS" \
    --warm-start-checkpoint "$WARM_START_CKPT" \
    --checkpoint-dir "$CKPT_DIR" \
    --learning-rate "$LR" \
    --green-time-grid "$GREEN_TIME_GRID" \
    --green-kernel-points "$GREEN_KERNEL_POINTS" \
    --gammaparam-anchor-probability "$ANCHOR_PROBABILITY" \
    --base-train-points 8000 --max-train-points 9000 --train-point-growth 0 \
    --save-every 250 --progress-every 25 --empty-cache-every 100 \
    --abort-on-nan --reset-best-score \
    --early-stop-check-every 100 --early-stop-patience-checks 4 \
    --early-stop-min-epochs 1000 --early-stop-validation-points 96 \
    --pde-thin-weight 10 --pde-ext-weight 10 \
    --thin-interface-weight 300 --ext-interface-weight 200 --bounds-weight 30 \
    2>&1 | tee "${LOG_DIR}/training.log"

BEST_CKPT="${CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_gammaparam_best.pth"
if [[ ${#CASES[@]} -gt 0 ]]; then
    "$PYTHON" -u compare_gamma_parameter_cases.py \
        --checkpoint "$BEST_CKPT" \
        --output-dir "$POST_DIR" \
        --green-time-grid "$GREEN_TIME_GRID" \
        --green-kernel-points "$GREEN_KERNEL_POINTS" \
        --apply-inventory-lift \
        --lift-time-grid "$LIFT_TIME_GRID" \
        "${CASES[@]}" 2>&1 | tee "${LOG_DIR}/posterior_compare.log"
fi

echo "Run complete: $RUN_ROOT"
