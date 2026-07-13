# Minimal ProductIntegral k-gamma model

This branch is a standalone clean project. It does not import the old
`pinn_thin_layer_v9_6.py`, old model classes, or old runners.

It keeps only:

- ProductIntegral Abel interface closure in `d=C_D/gamma`
- Hermite thin-layer field with one trainable correction network
- trace-preserving TraceGreen external field
- zero-training causal inventory lift
- FDM posterior comparison

FDM is never accepted by `train_base.py`. It is read only by `posterior.py`.
Both `MODE=lift` and `MODE=kg` force inventory lift; there is no unlifted KG
posterior option.

## Colab setup

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```bash
!git clone -b codex/kg-minimal-product-integral \
  https://github.com/FanoShannon/pinn_test.git /content/pinn_minimal
%cd /content/pinn_minimal
```

## 1. Train any fixed base

You only fill in these two physical parameters:

```python
K_CAT = 1.0
GAMMA = 10.0
```

Then run:

```bash
!MODE=base K_CAT={K_CAT} GAMMA={GAMMA} EPOCHS=1500 bash ./run_colab.sh
```

The checkpoint is:

```text
/content/gdrive/MyDrive/pinn_v96_minimal/base_k1.0_gamma10.0/base/base_best.pth
```

Change only `K_CAT` and `GAMMA` to create another positive fixed base.
The checkpoint records `physics_only` and `fdm_used_for_training=false`.

## 2. Zero-training lift at one case

```bash
!MODE=lift \
  BASE_CKPT=/content/gdrive/MyDrive/pinn_v96_minimal/base_k1.0_gamma10.0/base/base_best.pth \
  FDM_PKL=/content/gdrive/MyDrive/FDM_kg_v42/kg_k1_g10_v42_thin_layer_catalytic_v42.pkl \
  RUN_TAG=lift_k1_g10 bash ./run_colab.sh
```

This loads the base and applies inventory lift at inference. It performs no
optimizer step and writes `posterior_summary.json` under `lift_posterior/`.

## 3. Joint k-gamma zero-training generalization with lift

Generate or validate the posterior-only FDM inventory:

```bash
!git clone -b codex/kg-grid-v42 \
  https://github.com/FanoShannon/FDM.git /content/FDM
%cd /content/FDM
!CASE_SET=full FDM_OUTPUT_DIR=/content/gdrive/MyDrive/FDM_kg_v42 \
  bash ./run_colab_fdm_kg17_v42.sh
```

Then evaluate one base over the whole manifest:

```bash
%cd /content/pinn_minimal
!MODE=kg \
  BASE_CKPT=/content/gdrive/MyDrive/pinn_v96_minimal/base_k1.0_gamma10.0/base/base_best.pth \
  FDM_MANIFEST=/content/gdrive/MyDrive/FDM_kg_v42/kg_cases_v42.tsv \
  RUN_TAG=kg_from_k1_g10 bash ./run_colab.sh
```

`MODE=kg` changes the runtime physical `k,gamma`, recomputes every causal
cache, and always recomputes inventory lift for that pair. FDM remains outside
the model and loss; it is used only after inference to report posterior error.

## Outputs

- `base/base_best.pth`: best physics-only fixed base
- `base/base_current.pth`: latest base snapshot
- `base/training.jsonl`: physics loss history
- `lift_posterior/posterior_summary.json`: one lifted posterior
- `kg_lift_posterior/posterior_summary.json`: aggregate mean and worst pair
