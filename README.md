# ProductIntegral reference reproduction

This branch freezes the original `codex/highgamma-product-integral` model code
and reproduces the checkpoint family that produced the strong zero-shot k/gamma
posterior results. It is deliberately separate from the minimal ablation branch.

The actual training chain has three trainable phases, followed by a zero-training
lift:

1. `multiscale_green_grid_dynamic_stage1`, up to 5000 epochs.
2. `multiscale_film_tracegreen_clean`, up to 1500 epochs.
3. `multiscale_film_tracegreen_productintegral`, 300 added epochs.
4. `multiscale_film_tracegreen_productintegral_lift`, zero optimizer steps.

FDM is used only for posterior reports. It is not part of any loss, physics early
stop, or checkpoint selection.

## Colab

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```bash
!git clone -b codex/productintegral-reference-reproduction --single-branch \
  https://github.com/FanoShannon/pinn_test.git /content/pinn_reference
%cd /content/pinn_reference
```

Fill only these values:

```python
K_CAT = 1.0
GAMMA = 10.0
FDM_PKL = "/content/gdrive/MyDrive/FDM_kg_v42/kg_k1_g10_v42_thin_layer_catalytic_v42.pkl"
```

Run the complete reference chain:

```bash
!K_CAT={K_CAT} GAMMA={GAMMA} FDM_PKL={FDM_PKL} \
  bash ./run_colab_reference.sh
```

Final ProductIntegral checkpoint:

```text
/content/gdrive/MyDrive/pinn_v96_reference/reference_k1.0_gamma10.0/productintegral_300/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth
```

Final fixed-case lift metrics:

```text
/content/gdrive/MyDrive/pinn_v96_reference/reference_k1.0_gamma10.0/final_productintegral_lift/reference_lift_metrics.json
```

The default disables periodic FDM comparisons to save time. This does not alter
training because those comparisons were posterior-only. Set
`FDM_COMPARE_EVERY=250` only when intermediate posterior figures are desired.
