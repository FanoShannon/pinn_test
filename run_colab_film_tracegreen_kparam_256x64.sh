#!/usr/bin/env bash
set -euo pipefail

# ProductIntegral-TraceGreen k_cat generalization with gamma=10 and delta=0.035 fixed.
# FDM files below are posterior-only and are never passed to the training loss.

WORK_DIR="${WORK_DIR:-/content/gdrive/MyDrive/pinn_v96}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./runs_film_tracegreen_kparam_256x64/${RUN_TAG}}"
WARM_START_CKPT="${WARM_START_CKPT:-}"

EPOCHS="${EPOCHS:-3000}"
LR="${LR:-2e-6}"
ANCHOR_EPOCHS="${ANCHOR_EPOCHS:-1000}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-1000}"
ANCHOR_PROBABILITY="${ANCHOR_PROBABILITY:-0.5}"
KPARAM_LOG10_STD="${KPARAM_LOG10_STD:-1.25}"
GREEN_TIME_GRID="${GREEN_TIME_GRID:-256}"
GREEN_KERNEL_POINTS="${GREEN_KERNEL_POINTS:-64}"
BASE_TRAIN_POINTS="${BASE_TRAIN_POINTS:-8000}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-9000}"
LIFT_TIME_GRID="${LIFT_TIME_GRID:-4096}"
ZERO_SHOT_APPLY_LIFT="${ZERO_SHOT_APPLY_LIFT:-1}"

FDM_K01="${FDM_K01:-/content/gdrive/MyDrive/FDM/kcat0p1_v42_thin_layer_catalytic_v42.pkl}"
FDM_K001="${FDM_K001:-/content/gdrive/MyDrive/FDM/kcat0p01_v42_thin_layer_catalytic_v42.pkl}"
FDM_K0316="${FDM_K0316:-/content/gdrive/MyDrive/FDM/kcat0p316_v42_thin_layer_catalytic_v42.pkl}"
FDM_K1="${FDM_K1:-/content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl}"
FDM_K3162="${FDM_K3162:-/content/gdrive/MyDrive/FDM/kcat3p162_v42_thin_layer_catalytic_v42.pkl}"
FDM_K10="${FDM_K10:-/content/gdrive/MyDrive/FDM/kcat10_v42_thin_layer_catalytic_v42.pkl}"
FDM_K100="${FDM_K100:-/content/gdrive/MyDrive/FDM/kcat100_v42_thin_layer_catalytic_v42.pkl}"
K1_BASELINE_METRICS="${K1_BASELINE_METRICS:-}"
K1_HISTORICAL_METRICS="${K1_HISTORICAL_METRICS:-}"

if [[ -z "$WARM_START_CKPT" || ! -f "$WARM_START_CKPT" ]]; then
    echo "Set WARM_START_CKPT to an existing fixed or kparam checkpoint." >&2
    exit 1
fi

cd "$WORK_DIR"
CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
POSTERIOR_DIR="${RUN_ROOT}/posterior_compare"
ZERO_SHOT_DIR="${RUN_ROOT}/zero_shot_compare"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$POSTERIOR_DIR" "$ZERO_SHOT_DIR"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CASES=()
[[ -f "$FDM_K001" ]] && CASES+=(--case "0.01=$FDM_K001")
[[ -f "$FDM_K01" ]] && CASES+=(--case "0.1=$FDM_K01")
[[ -f "$FDM_K0316" ]] && CASES+=(--case "0.316227766=$FDM_K0316")
[[ -f "$FDM_K1" ]] && CASES+=(--case "1=$FDM_K1")
[[ -f "$FDM_K3162" ]] && CASES+=(--case "3.16227766=$FDM_K3162")
[[ -f "$FDM_K10" ]] && CASES+=(--case "10=$FDM_K10")
[[ -f "$FDM_K100" ]] && CASES+=(--case "100=$FDM_K100")

if [[ ${#CASES[@]} -gt 0 ]]; then
    ZERO_SHOT_CMD=(
        "$PYTHON" -u compare_k_parameter_cases.py
        --checkpoint "$WARM_START_CKPT"
        --output-dir "$ZERO_SHOT_DIR"
        --green-time-grid "$GREEN_TIME_GRID"
        --green-kernel-points "$GREEN_KERNEL_POINTS"
        --zero-shot-fixed-reference
        "${CASES[@]}"
    )
    if [[ "$ZERO_SHOT_APPLY_LIFT" == "1" ]]; then
        ZERO_SHOT_CMD+=(--apply-inventory-lift --lift-time-grid "$LIFT_TIME_GRID")
    fi
    "${ZERO_SHOT_CMD[@]}" 2>&1 | tee "${LOG_DIR}/zero_shot_compare.log"
fi

if [[ "$EPOCHS" == "0" && ${#CASES[@]} -eq 0 ]]; then
    echo "EPOCHS=0 requested, but no FDM case files were found." >&2
    exit 1
fi

if [[ "$EPOCHS" == "0" ]]; then
    echo "EPOCHS=0: posterior-only evaluation completed; training skipped."
    echo "Summary: ${ZERO_SHOT_DIR}/k_parameter_summary.json"
    exit 0
fi

"$PYTHON" -u pinn_thin_layer_v9_6.py \
    --arch multiscale_film_tracegreen_kparam \
    --gamma 10.0 \
    --k-cat-star 1.0 \
    --epochs "$EPOCHS" \
    --warm-start-checkpoint "$WARM_START_CKPT" \
    --checkpoint-dir "$CKPT_DIR" \
    --reset-best-score \
    --save-every 250 \
    --progress-every 25 \
    --empty-cache-every 100 \
    --abort-on-nan \
    --learning-rate "$LR" \
    --green-time-grid "$GREEN_TIME_GRID" \
    --green-kernel-points "$GREEN_KERNEL_POINTS" \
    --base-train-points "$BASE_TRAIN_POINTS" \
    --max-train-points "$MAX_TRAIN_POINTS" \
    --train-point-growth 0 \
    --kparam-anchor-epochs "$ANCHOR_EPOCHS" \
    --kparam-anchor-probability "$ANCHOR_PROBABILITY" \
    --kparam-log10-std "$KPARAM_LOG10_STD" \
    --kparam-freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
    --early-stop-check-every 100 \
    --early-stop-patience-checks 4 \
    --early-stop-min-epochs 1200 \
    --early-stop-validation-points 96 \
    --pde-thin-weight 10.0 \
    --pde-ext-weight 10.0 \
    --thin-interface-weight 300.0 \
    --ext-interface-weight 200.0 \
    --bounds-weight 30.0 \
    2>&1 | tee "${LOG_DIR}/training.log"

BEST_CKPT="${CKPT_DIR}/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_kparam_best.pth"
if [[ ${#CASES[@]} -gt 0 ]]; then
    COMPARE_CMD=(
        "$PYTHON" -u compare_k_parameter_cases.py
        --checkpoint "$BEST_CKPT"
        --output-dir "$POSTERIOR_DIR"
        --green-time-grid "$GREEN_TIME_GRID"
        --green-kernel-points "$GREEN_KERNEL_POINTS"
        --apply-inventory-lift
        --lift-time-grid "$LIFT_TIME_GRID"
        "${CASES[@]}"
    )
    [[ -f "$K1_BASELINE_METRICS" ]] && COMPARE_CMD+=(--k1-baseline-metrics "$K1_BASELINE_METRICS")
    [[ -f "$K1_HISTORICAL_METRICS" ]] && COMPARE_CMD+=(--k1-historical-metrics "$K1_HISTORICAL_METRICS")
    "${COMPARE_CMD[@]}" 2>&1 | tee "${LOG_DIR}/posterior_compare.log"
else
    echo "No matching FDM cases found; posterior comparison skipped."
fi
