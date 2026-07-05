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

## Final Clean Architecture

The current paper-facing architecture is:

```text
multiscale_film_tracegreen_clean
```

It is a cleaned-up version of the best TraceGreen branch.  It keeps the pieces
that produced clear gains and removes exploratory correction paths that were
useful for diagnosis but make the final story hard to defend.

### Mathematical Form

Thin-layer conservation is hard constrained:

```text
C_A + C_B = 1
```

External-region conservation is also hard constrained:

```text
C_C + C_D = gamma
```

The interface state is represented by a Film-Abel/KernelMix chain:

```text
C_B_int(t) = C_B_surface(t) / (1 + k_cat*delta*C_C_int(t)/D_B)

J_rxn(t) = k_cat*C_B_int(t)*C_C_int(t)

C_D_int(t) =
    alpha*A[J_rxn](t)
  + sum_i beta_i*M_i[J_rxn](t)
  - dt_phase(t)*A[dJ_rxn/dt](t)
  + r_int(t)

C_C_int(t) = gamma - C_D_int(t)
```

where `A[...]` is the Abel memory term and `M_i[...]` are finite-memory
exponential kernels.  The external field is then driven by a single boundary
trace:

```text
C_D(y,t) =
    G_trace_erfc[C_D_int](y,t)
  + h01(y)*(0 - D_far(t))
  + R_smooth(y,t)

C_C(y,t) = gamma - C_D(y,t)
```

The key numerical/mathematical contribution is the trace-preserving erfc
quadrature used in `G_trace_erfc`:

```text
lim_{y -> 0+} G_trace_erfc[C_D_int](y,t) = C_D_int(t)
```

This removed the nonphysical discontinuity between the interface point and the
first external grid point that appeared with ordinary time quadrature.

### What Is Kept

| Component | Status | Reason |
|---|---|---|
| hard conservation `C_A+C_B=1`, `C_C+C_D=gamma` | final | physical invariant and stable |
| Film relation for `C_B_int` | final | explains the accurate `C_B_int` behavior |
| Abel + finite-memory KernelMix | final | compact causal interface memory |
| bounded interface residual `r_int(t)` | final | small model-discrepancy correction |
| TraceGreen external field | final | propagates `C_D_int` causally into the external region |
| erfc trace-preserving quadrature | final | fixes the near-interface discontinuity |
| smooth external residual `R_smooth(y,t)` | final | endpoint-compatible correction for finite-domain effects |

### What Is Moved to Ablation

| Component | Status | Reason |
|---|---|---|
| direct dynamic external correction | ablation | can create time-switching artifacts |
| causal gate / causal convolution variants | ablation | useful diagnosis, not needed in final TraceGreen path |
| FluxTrace | ablation | less stable than single-trace TraceGreen |
| EMA branch | ablation | helped analyze memory drift but not final |
| matched Abel | failed ablation | direct replacement worsened `C_C_int` |
| mixed Abel | failed ablation | learned lambda stayed near zero |
| extra reversal-jump / smoothness penalties | removed from final | diagnostic losses, not core physics |

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
| Film-Abel KernelMix | `multiscale_green_grid_film_abel_kernelmix` | Lets Abel and finite-memory kernels compete inside the interface prior | `runs_film_abel_kernelmix_256x64/<timestamp>/...` | Targets the reverse-scan `C_C_int` peak without starting from the EMA branch. |
| KernelMix causal | `multiscale_green_grid_film_abel_kernelmix_causal` | Multiplies the dynamic external correction by a diffusion reachability gate | `runs_film_abel_kernelmix_causal_256x64/<timestamp>/...` | Tests whether the pre-reversal far-field blue residual is caused by anti-causal dynamic correction. |
| KernelMix causal-conv | `multiscale_green_grid_film_abel_kernelmix_causalconv` | Replaces direct dynamic field correction with a learned source convolved through the heat kernel | `runs_film_abel_kernelmix_causalconv_256x64/<timestamp>/...` | Tests whether the reversal band is caused by dynamic correction changing too sharply in time. |
| KernelMix causal-hybrid | `multiscale_green_grid_film_abel_kernelmix_causalhybrid` | Adds causal-convolution correction plus a small diffusion-gated direct residual | `runs_film_abel_kernelmix_causalhybrid_256x64/<timestamp>/...` | Combines causalconv continuity with causal-gate local flexibility. |
| KernelMix causal-hybrid smooth | `multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth` | Reduces the direct residual and adds direct/static/phase temporal smoothness losses | `runs_film_abel_kernelmix_causalhybrid_smooth_256x64/<timestamp>/...` | Tests whether the remaining reversal line is caused by direct residual time switching. |
| TraceGreen erfc | `multiscale_green_grid_film_abel_kernelmix_tracegreen` | Uses single Film-Abel boundary trace and erfc trace-preserving Dirichlet Green lift | `runs_film_abel_kernelmix_tracegreen_256x64/20260705_081613/...` | Best single-parameter result; removes near-interface discontinuity. |
| Matched Abel | `multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel` | Replaces midpoint Abel with singularity-matched Abel quadrature | `runs_film_abel_kernelmix_tracegreen_matchedabel_256x64/...` | Worse `C_C_int`; interface error is not simple Abel endpoint quadrature error. |
| Mixed Abel | `multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel` | Learnable mixture of midpoint Abel and matched Abel | `runs_film_abel_kernelmix_tracegreen_mixedabel_256x64/...` | Lambda stayed near zero; keep old effective Abel/KernelMix memory. |
| Clean final | `multiscale_film_tracegreen_clean` | Finalized Film-Abel + erfc TraceGreen, with exploratory debug losses disabled | `runs_film_tracegreen_clean_256x64/<timestamp>/...` | Paper-facing architecture and baseline for parameter generalization. |

## Representative Posterior Metrics

All metrics below use FDM posterior evaluation only.  FDM data is not used in
training.

The historical table below was produced against the older v4.1 FDM reference.
Keep it only for branch history.  New Colab runs and final evaluation should use
the finite-volume v4.2 reference:

- Colab: `/content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl`
- Local: `../FDM/kcat1_v42_thin_layer_catalytic_v42.pkl`

Current v4.2 posterior checkpoints:

| Checkpoint | `C_C` R2 | `C_C` RMSE | `C_C_int` R2 | `C_C_int` RMSE | `CV_J` R2 | Overall RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `film_abel_256x64_best` | 0.9813 | 0.1462 | 0.9976 | 0.0697 | 0.9983 | 0.1105 |
| `film_abel_ema_500ep_best` | 0.9821 | 0.1428 | 0.9971 | 0.0768 | 0.9983 | 0.1080 |
| `film_abel_kernelmix_latest` | 0.9765 | 0.1638 | 0.9973 | 0.0735 | 0.9983 | 0.1239 |
| `kernelmix_causal_preview_no_retrain` | 0.9961 | 0.0671 | 0.9973 | 0.0735 | 0.9983 | 0.0507 |
| `kernelmix_causalconv_preview_no_retrain` | 0.9949 | 0.0763 | 0.9973 | 0.0735 | 0.9983 | 0.0577 |
| `kernelmix_causalhybrid_preview_no_retrain` | 0.9969 | 0.0596 | 0.9973 | 0.0735 | 0.9983 | 0.0451 |
| `kernelmix_causalhybrid_500ep` | 0.9967 | 0.0611 | 0.9974 | 0.0724 | 0.9983 | 0.0462 |

The `*_preview_no_retrain` rows use the existing KernelMix checkpoint with the
new correction structure applied at evaluation time.  They are direction checks,
not completed training runs.

Historical v4.1 posterior checkpoints:

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
  - `../FDM/kcat1_v42_thin_layer_catalytic_v42.pkl`
  - `../FDM/kcat1_v42_cv_data_v42.csv`
  - Google Drive mirror: `/content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl`
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
  --fdm-compare-pkl /content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl \
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
  --fdm-compare-pkl /content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl \
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
  --fdm-compare-pkl /content/gdrive/MyDrive/FDM/kcat1_v42_thin_layer_catalytic_v42.pkl \
  --fdm-compare-every 500 \
  --fdm-compare-dir /content/gdrive/MyDrive/pinn_v96/fdm_compare_film_abel_v1 \
  2>&1 | tee -a /content/gdrive/MyDrive/pinn_v96/checkpoints_film_abel_v1/v96_film_abel_training.log
```

To continue an existing Film-Abel run, resume from the Film-Abel current
checkpoint instead of the dynamic checkpoint:

```bash
--resume-checkpoint /content/gdrive/MyDrive/pinn_v96/checkpoints_film_abel_v1/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel.pth
```

### From-zero Film-Abel 256x64 curriculum

For a clean from-zero run that keeps the Green history resolution fixed across
the architecture switch, use:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_curriculum_v96_film_abel_256x64.sh
```

This runs:

- Stage 1: `multiscale_green_grid_dynamic`, `M=256`, `K=64`, from scratch.
- Stage 2: `multiscale_green_grid_film_abel`, `M=256`, `K=64`, from the Stage 1 best checkpoint.

Outputs are isolated under:

```text
runs_film_abel_256x64_two_stage/<timestamp>/
```

Stage 1 is also copied to:

```text
runs_film_abel_256x64_two_stage/<timestamp>/stage1_finished_snapshot/
```

so a Stage 2 NaN does not obscure the completed Stage 1 checkpoint and log.
The script uses `--abort-on-nan` and a smaller Stage 2 learning rate by default.

### Film-Abel KernelMix warm start

To test whether the Abel long-tail prior is causing the `C_C_int` reverse-scan
peak, warm start from a Film-Abel checkpoint, not from EMA:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_colab_film_abel_kernelmix_256x64.sh
```

This runs `multiscale_green_grid_film_abel_kernelmix` with `M=256`, `K=64`,
`LR=2e-6`, and v4.2 FDM posterior comparison every 500 epochs by default.

### Film-Abel KernelMix causal warm start

To test the diffusion-causality fix for the pre-reversal far-field blue
residual, run:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_colab_film_abel_kernelmix_causal_256x64.sh
```

This runs `multiscale_green_grid_film_abel_kernelmix_causal`.  It keeps the
Film-Abel/KernelMix interface chain but gates only the dynamic external
correction:

```text
g(y,t) = exp(-y^2 / (4*D*t*alpha))
```

The intent is to stop the residual correction from adding `C_D` at far external
positions before diffusion from the interface can plausibly reach them.

### Film-Abel KernelMix causal-convolution warm start

To replace the direct dynamic external correction by a causal source history,
run:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_colab_film_abel_kernelmix_causalconv_256x64.sh
```

This runs `multiscale_green_grid_film_abel_kernelmix_causalconv`.  It keeps the
Film-Abel/KernelMix interface chain, but changes the dynamic correction from a
direct field

```text
NN_corr(x,t,theta,dtheta,J,dJ,Q)
```

to a scalar source propagated by the heat kernel:

```text
C_corr(y,t) = int_0^t S_corr(tau) K(y,t-tau) dtau
```

with endpoint subtraction so it does not directly overwrite the interface or
far-field values.  This is the stricter follow-up after the causal gate: it
targets the remaining reversal-line discontinuity-like band.

### Film-Abel KernelMix causal-hybrid warm start

To combine causal-convolution continuity with a small diffusion-gated direct
residual, run:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_colab_film_abel_kernelmix_causalhybrid_256x64.sh
```

This runs `multiscale_green_grid_film_abel_kernelmix_causalhybrid`:

```text
C_dyn = C_causalconv + 0.25 * C_direct_gated
```

The preview check gives the best no-retrain full-field score so far while
keeping the reversal jump close to the pure causal-convolution version.

### Film-Abel KernelMix causal-hybrid-smooth warm start

The causal-hybrid 500-epoch diagnostic showed that the remaining vertical
reversal band is dominated by the direct dynamic residual, not by the diffusion
gate or the causal-convolution branch.  To reduce that switch-like behavior, run:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_colab_film_abel_kernelmix_causalhybrid_smooth_256x64.sh
```

This runs `multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth`:

```text
C_dyn = C_causalconv + 0.05 * C_direct_gated
```

and adds lightweight finite-difference smoothness penalties for:

- direct dynamic reversal jump,
- direct dynamic temporal curvature near reversal,
- static residual temporal curvature near reversal,
- Film-Abel phase temporal curvature near reversal.

### Clean Film-TraceGreen final run

Use this for the final paper-facing architecture and future parameter
generalization experiments:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!bash run_colab_film_tracegreen_clean_256x64.sh
```

By default the script warms from the best TraceGreen checkpoint if available:

```text
runs_film_abel_kernelmix_tracegreen_256x64/20260705_081613/checkpoints/
```

For a short check:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!EPOCHS=100 SAVE_EVERY=100 FDM_COMPARE_EVERY=100 \
  bash run_colab_film_tracegreen_clean_256x64.sh
```

## Posterior Evaluation

```bash
PYTHONIOENCODING=utf-8 python -u compare_concentration_fields.py \
  --arch multiscale_film_tracegreen_clean \
  --input-mode normalized \
  --checkpoint runs_film_tracegreen_clean_256x64/<timestamp>/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth \
  --fdm-pkl ../FDM/kcat1_v42_thin_layer_catalytic_v42.pkl \
  --green-time-grid 256 \
  --green-kernel-points 64 \
  --n-time 160 \
  --n-x-in 120 \
  --n-x-out 160 \
  --output-json analysis_film_tracegreen_clean/metrics.json \
  --output-npz analysis_film_tracegreen_clean/fields.npz \
  --output-figure analysis_film_tracegreen_clean/residual_summary.png
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
