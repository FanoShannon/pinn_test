# Reproduction Guide

## 1. Environment

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

The current mainline depends only on NumPy, PyTorch, and Matplotlib. It does not
need a checkpoint.

## 2. Tests

```bash
python -m unittest -v \
  test_soe_history.py \
  test_productintegral_dtn.py \
  test_differentiable_operator.py \
  test_multiscan_inversion.py
```

The tests cover direct/SOE history agreement, physical root closure,
TraceGreen boundaries, runtime thickness, NumPy/PyTorch agreement, automatic
gradients, and multi-scan objective construction.

## 3. Forward and Stability Studies

Thickness convergence:

```bash
python analyze_delta_forward.py --output-dir outputs/delta_forward
```

Joint positive-parameter stress audit:

```bash
python analyze_kgdelta_forward.py \
  --history-backend soe \
  --output-dir outputs/joint_stress
```

Direct-versus-SOE benchmark:

```bash
python benchmark_soe_history.py --output outputs/soe_benchmark.json
```

These commands use no FDM.

## 4. Independent FDM Posterior

FDM is accepted only by the NumPy reference CLI:

```bash
python productintegral_dtn.py \
  --fdm-pkl /path/to/matching_case.pkl \
  --output-dir outputs/fdm_posterior \
  --k-cat 1 \
  --gamma 10 \
  --delta 0.035 \
  --operator-time-grid 4096 \
  --mode-counts 256 \
  --history-backend soe
```

The command validates FDM metadata before comparison. No FDM value is passed to
the forward recurrence.

## 5. Differentiable Inversion

Delta-only inversion:

```bash
python invert_delta_from_cv.py \
  --history-backend soe \
  --output-dir outputs/delta_inverse
```

Local single- and multi-scan identifiability:

```bash
python analyze_inverse_identifiability.py \
  --sigmas 40 \
  --history-backend soe \
  --output outputs/single_scan.json

python analyze_inverse_identifiability.py \
  --sigmas 5,40,320 \
  --history-backend soe \
  --output outputs/wide_scan.json
```

Scan-rate design:

```bash
python design_multiscan_rates.py \
  --history-backend soe \
  --output outputs/rate_design.json
```

Joint multi-scan inversion:

```bash
python invert_multiscan_parameters.py \
  --true-k 1 \
  --true-gamma 10 \
  --true-delta 0.035 \
  --sigmas 2.5,40,640 \
  --inverse-modes all,fix-gamma \
  --noise-levels 0,0.01 \
  --history-backend soe \
  --output-dir outputs/multiscan_inverse
```

## 6. Colab

Mount Drive, then run:

```bash
git clone --branch codex/productintegral-dtn-soe-mainline \
  https://github.com/FanoShannon/pinn_test.git /content/pi_dtn_soe

MODE=all \
OUTPUT_DIR=/content/gdrive/MyDrive/pi_dtn_soe_mainline \
bash /content/pi_dtn_soe/run_colab_mainline.sh
```

Available modes are `forward`, `posterior`, `inverse`, `identifiability`,
`benchmark`, `multiscan-design`, `multiscan-inverse`, `multiscan`, `all`, and
`full`.

For posterior-only validation:

```bash
MODE=posterior \
TRUE_K=1 \
TRUE_GAMMA=10 \
TRUE_DELTA=0.035 \
FDM_PKL=/content/gdrive/MyDrive/FDM_kg_v42/kg_k1_g10_v42_thin_layer_catalytic_v42.pkl \
OUTPUT_DIR=/content/gdrive/MyDrive/pi_dtn_soe_mainline \
bash /content/pi_dtn_soe/run_colab_mainline.sh
```

