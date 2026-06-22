# PINN v9.6 Thin-Layer Catalytic Experiments

This repository contains the v9.6 PINN experiments for the thin-layer catalytic
electrochemical PDE problem.  FDM concentration data is used only for posterior
evaluation and plotting.  It is not used in the training loss.

## Current Branch

`codex/film-abel-interface`

Main code files:

- `pinn_thin_layer_v9_6.py`: training, architectures, checkpointing, in-training posterior compare.
- `compare_concentration_fields.py`: FDM-vs-PINN posterior metrics and residual summary plots.
- `decompose_concentration_error.py`: spatial/time-group error decomposition from saved NPZ fields.
- `plot_direct_concentration_comparison.py`: direct FDM/PINN field and curve plots from saved NPZ/JSON outputs.

## Experiment History

| Stage | Architecture | Main change | Representative checkpoint | Key observation |
|---|---|---|---|---|
| Baseline legacy | `legacy` | Direct field PINN | `baseline/pinn_thin_layer_catalytic_v9_6_best.pth` | Useful as historical baseline, but weak for concentration fields. |
| Multiscale | `multiscale` | Larger multiscale MLP | `checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_best.pth` | Improves `C_A/C_B`; `C_C/C_D` still weak. |
| HardBC | `multiscale_hardbc` | Hard conservation for thin film | `checkpoints_hardbc_v2/..._best.pth` | Stabilizes `C_A/C_B`, but does not solve external `C_C/C_D`. |
| Hermite/extbasis | `multiscale_hermite_extbasis` | Hermite thin layer and external basis | `checkpoints_hermite_extbasis_v1/...pth` | Better interface handling, but `C_C_int` remains limited. |
| Green grid hybrid | `multiscale_green_grid_hybrid` | Signed full-field Green history plus residual modes | `checkpoints_green_grid_hybrid_v1/..._best.pth` | Improves full-field `C_C`, but `C_C_int` remains a bottleneck. |
| Dynamic Green | `multiscale_green_grid_dynamic` | Adds dynamic inputs: `theta`, `dtheta/dt`, `J`, `dJ/dt`, `Q` | `checkpoints_green_grid_dynamic_v1/..._best.pth` | Stronger reverse-scan behavior; representative metrics below. |
| Interface memory | `multiscale_green_grid_interface_memory` | Adds memory correction to the interface state | `checkpoints_green_grid_interface_memory_v1/...pth` | Modest `C_C/C_D` gain, still not enough for `C_C_int`. |
| Film-Abel interface | `multiscale_green_grid_film_abel` | Replaces black-box interface source with quasi-steady film transfer plus Abel memory | `checkpoints_film_abel_v1/..._best.pth` and current `.pth` | Large `C_C_int` and CV improvement. Full-field `C_C` becomes limited by spatial propagation. |

## Representative Posterior Metrics

All metrics below use FDM posterior evaluation only.  FDM data is not used in
training.

| Checkpoint | `C_C` R2 | `C_C` RMSE | `C_C_int` R2 | `C_C_int` RMSE | `CV_J` R2 | Overall RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `dynamic_best_54500` | 0.7419 | 0.5072 | 0.8669 | 0.4978 | 0.9969 | 0.3834 |
| `interface_memory_current_56500` | 0.7587 | 0.4904 | 0.8738 | 0.4848 | 0.9968 | 0.3707 |
| `film_abel_best_55000` | 0.7951 | 0.4520 | 0.9624 | 0.2646 | 0.9955 | 0.3417 |
| `film_abel_current_56000` | 0.7793 | 0.4690 | 0.9770 | 0.2070 | 0.9992 | 0.3545 |

Interpretation:

- Film-Abel confirms that the upstream source chain was the main `C_C_int`
  bottleneck.
- Continuing film-Abel from epoch 55000 to 56000 improves `C_C_int` and `CV_J`,
  but degrades full-field `C_C`.
- Direct FDM/PINN profiles show the new bottleneck is spatial propagation:
  `C_C(x,t)` returns to bulk too quickly, especially in the mid external region.
- Error decomposition for film-Abel current shows the dominant error is
  `mid_0.10_0.30L`, reverse scan (`dtheta_positive`), and negative `dJ`.

## Required Data and Checkpoints

Minimum files needed to reproduce the current comparison:

- FDM posterior data:
  - `../FDM/thin_layer_catalytic_v41_fixed.pkl`
  - `../FDM/v41_cv_data_fixed.csv`
- Dynamic warm start:
  - `checkpoints_green_grid_dynamic_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic.pth`
  - `checkpoints_green_grid_dynamic_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_best.pth`
- Film-Abel branch:
  - `checkpoints_film_abel_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_best.pth`
  - `checkpoints_film_abel_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel.pth`
- Optional comparison branch:
  - `checkpoints_green_grid_interface_memory_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_interface_memory.pth`
  - `checkpoints_green_grid_hybrid_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_hybrid_best.pth`

Do not commit checkpoint files to git.  Keep them in Google Drive or local
experiment folders.

## Colab Training Commands

Use:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
```

### Dynamic Green baseline

```bash
!mkdir -p checkpoints_green_grid_dynamic_v1 fdm_compare_green_grid_dynamic_v1

!PYTHONIOENCODING=utf-8 python -u pinn_thin_layer_v9_6.py \
  --arch multiscale_green_grid_dynamic \
  --epochs 4000 \
  --no-early-stop \
  --resume-checkpoint /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_hybrid_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_hybrid.pth \
  --reset-optimizer-state \
  --reset-best-score \
  --checkpoint-dir /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_dynamic_v1 \
  --save-every 500 \
  --progress-every 25 \
  --empty-cache-every 100 \
  --green-time-grid 256 \
  --green-kernel-points 64 \
  --base-train-points 8000 \
  --max-train-points 9000 \
  --train-point-growth 0 \
  --fdm-compare-pkl /content/gdrive/MyDrive/FDM/thin_layer_catalytic_v41_fixed.pkl \
  --fdm-compare-every 500 \
  --fdm-compare-dir /content/gdrive/MyDrive/pinn_v96/fdm_compare_green_grid_dynamic_v1 \
  2>&1 | tee -a /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_dynamic_v1/v96_green_grid_dynamic_training.log
```

### Interface-memory branch

```bash
!mkdir -p checkpoints_green_grid_interface_memory_v1 fdm_compare_green_grid_interface_memory_v1

!PYTHONIOENCODING=utf-8 python -u pinn_thin_layer_v9_6.py \
  --arch multiscale_green_grid_interface_memory \
  --epochs 4000 \
  --no-early-stop \
  --resume-checkpoint /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_dynamic_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic.pth \
  --reset-optimizer-state \
  --reset-best-score \
  --checkpoint-dir /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_interface_memory_v1 \
  --save-every 500 \
  --progress-every 25 \
  --empty-cache-every 100 \
  --green-time-grid 256 \
  --green-kernel-points 64 \
  --base-train-points 8000 \
  --max-train-points 9000 \
  --train-point-growth 0 \
  --fdm-compare-pkl /content/gdrive/MyDrive/FDM/thin_layer_catalytic_v41_fixed.pkl \
  --fdm-compare-every 500 \
  --fdm-compare-dir /content/gdrive/MyDrive/pinn_v96/fdm_compare_green_grid_interface_memory_v1 \
  2>&1 | tee -a /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_interface_memory_v1/v96_green_grid_interface_memory_training.log
```

### Film-Abel interface branch

```bash
!mkdir -p checkpoints_film_abel_v1 fdm_compare_film_abel_v1

!PYTHONIOENCODING=utf-8 python -u pinn_thin_layer_v9_6.py \
  --arch multiscale_green_grid_film_abel \
  --epochs 4000 \
  --no-early-stop \
  --resume-checkpoint /content/gdrive/MyDrive/pinn_v96/checkpoints_green_grid_dynamic_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic.pth \
  --reset-optimizer-state \
  --reset-best-score \
  --checkpoint-dir /content/gdrive/MyDrive/pinn_v96/checkpoints_film_abel_v1 \
  --save-every 500 \
  --progress-every 25 \
  --empty-cache-every 100 \
  --green-time-grid 256 \
  --green-kernel-points 64 \
  --base-train-points 8000 \
  --max-train-points 9000 \
  --train-point-growth 0 \
  --fdm-compare-pkl /content/gdrive/MyDrive/FDM/thin_layer_catalytic_v41_fixed.pkl \
  --fdm-compare-every 500 \
  --fdm-compare-dir /content/gdrive/MyDrive/pinn_v96/fdm_compare_film_abel_v1 \
  2>&1 | tee -a /content/gdrive/MyDrive/pinn_v96/checkpoints_film_abel_v1/v96_film_abel_training.log
```

To continue an existing Film-Abel run, resume from the Film-Abel current
checkpoint instead of the dynamic checkpoint:

```bash
--resume-checkpoint /content/gdrive/MyDrive/pinn_v96/checkpoints_film_abel_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel.pth
```

## Posterior Evaluation

```bash
PYTHONIOENCODING=utf-8 python -u compare_concentration_fields.py \
  --arch multiscale_green_grid_film_abel \
  --input-mode normalized \
  --checkpoint checkpoints_film_abel_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_best.pth \
  --fdm-pkl ../FDM/thin_layer_catalytic_v41_fixed.pkl \
  --green-time-grid 256 \
  --green-kernel-points 64 \
  --n-time 160 \
  --n-x-in 120 \
  --n-x-out 160 \
  --output-json analysis_film_abel_direct_v1/film_abel_best_55000_metrics.json \
  --output-npz analysis_film_abel_direct_v1/film_abel_best_55000_fields.npz \
  --output-figure analysis_film_abel_direct_v1/film_abel_best_55000_residual_summary.png
```

## Direct FDM/PINN Plots

After producing `*_fields.npz` and `*_metrics.json` files, run:

```bash
PYTHONIOENCODING=utf-8 python -u plot_direct_concentration_comparison.py \
  --output-dir analysis_film_abel_direct_v1 \
  --item dynamic_best_54500:analysis_film_abel_direct_v1/dynamic_best_54500_fields.npz:analysis_film_abel_direct_v1/dynamic_best_54500_metrics.json \
  --item interface_memory_current_56500:analysis_film_abel_direct_v1/interface_memory_current_56500_fields.npz:analysis_film_abel_direct_v1/interface_memory_current_56500_metrics.json \
  --item film_abel_best_55000:analysis_film_abel_direct_v1/film_abel_best_55000_fields.npz:analysis_film_abel_direct_v1/film_abel_best_55000_metrics.json \
  --item film_abel_current_56000:analysis_film_abel_direct_v1/film_abel_current_56000_fields.npz:analysis_film_abel_direct_v1/film_abel_current_56000_metrics.json
```

This creates:

- direct field plots: FDM, PINN, and residual columns.
- direct interface/CV curve plots.
- combined `C_C(x)` profiles across checkpoints.
- combined metrics CSV and bar plots.

## Error Decomposition

```bash
PYTHONIOENCODING=utf-8 python -u decompose_concentration_error.py \
  --input-npz analysis_film_abel_direct_v1/film_abel_current_56000_fields.npz \
  --field C_C \
  --output-json analysis_film_abel_direct_v1/film_abel_current_56000_C_C_decomposition.json \
  --output-csv analysis_film_abel_direct_v1/film_abel_current_56000_C_C_decomposition.csv \
  --output-figure analysis_film_abel_direct_v1/film_abel_current_56000_C_C_decomposition.png
```

## Next Research Direction

The current best interpretation is:

1. Film-Abel solves much of the interface-source error.
2. The remaining limitation is external spatial propagation, not `C_C_int`.
3. The next model should keep the Film-Abel interface and improve
   `C_C_int -> C_C(x,t)` propagation with learnable full-field Green diffusion
   length, multi-scale Green kernels, or stronger coordinate-safe spatial
   correction modes.
