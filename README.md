# ProductIntegral Direct

This branch contains two supported workflows:

```text
Dynamic Stage 1
-> ProductIntegral 300 epochs
-> zero-training inventory lift

fixed ProductIntegral checkpoint
-> joint k/gamma physical conditioning
-> zero-training inventory lift
-> posterior-only FDM comparison
```

There is no trainable Clean stage. FDM is used only by the final posterior
comparison and never enters a loss, early stopping, or checkpoint selection.

## Colab

```python
from google.colab import drive
drive.mount("/content/gdrive")
```

```bash
!git clone -b codex/productintegral-direct-clean --single-branch \
  https://github.com/FanoShannon/pinn_test.git \
  /content/pinn_productintegral_direct
%cd /content/pinn_productintegral_direct
```

Set the physical parameters and matching posterior FDM:

```python
K_CAT = 1.0
GAMMA = 10.0
FDM_PKL = (
    "/content/gdrive/MyDrive/FDM_kg_v42/"
    "kg_k1_g10_v42_thin_layer_catalytic_v42.pkl"
)
```

Train the complete direct chain:

```bash
!K_CAT={K_CAT} GAMMA={GAMMA} FDM_PKL={FDM_PKL} \
  bash ./run_colab_productintegral_direct.sh
```

To reuse an existing Dynamic Stage 1 checkpoint, add only `STAGE1_BEST`:

```bash
!K_CAT={K_CAT} GAMMA={GAMMA} FDM_PKL={FDM_PKL} \
STAGE1_BEST="/content/gdrive/MyDrive/pinn_v96_reference/reference_k1.0_gamma10.0/clean_two_stage/stage1_dynamic_fixed/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth" \
bash ./run_colab_productintegral_direct.sh
```

## Outputs

Default root:

```text
/content/gdrive/MyDrive/pinn_v96_productintegral_direct/direct_k1.0_gamma10.0
```

Final ProductIntegral checkpoint:

```text
productintegral_300/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth
```

Final posterior metrics:

```text
final_inventory_lift/metrics.json
```

Automatic comparison with the historical epoch-2400 best:

```text
final_inventory_lift/reproduction_vs_historical_best.json
```

The retained internal `FilmTraceClean` class names are checkpoint-compatible
physical base implementations used by ProductIntegral. They are not exposed as
a trainable architecture or workflow.

## Joint k/gamma zero-training posterior

No parameter training is required. The fixed ProductIntegral network is loaded
as the reference backbone, while the ProductIntegral/TraceGreen physical chain
is evaluated at each positive `k` and `gamma`. Inventory lift is mandatory.
FDM files are read only after the frozen prediction is produced.

```python
CHECKPOINT = (
    "/content/gdrive/MyDrive/pinn_v96_productintegral_direct/"
    "direct_k1.0_gamma10.0/productintegral_300/checkpoints/"
    "pinn_thin_layer_catalytic_v9_6_"
    "multiscale_film_tracegreen_productintegral_best.pth"
)
```

```bash
!CHECKPOINT="{CHECKPOINT}" \
FDM_DIR="/content/gdrive/MyDrive/FDM_kg_v42" \
bash ./run_colab_kg_posterior.sh
```

The script discovers the 17 established FDM cases, writes per-case metrics and
`kg_parameter_summary.json`, then compares aggregate mean/worst errors with the
historical zero-shot + lift result in `kg_vs_historical.json`.
