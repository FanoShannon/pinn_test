# PINN v9.6 Thin-Layer Catalytic Experiments

This repository contains the v9.6 PINN experiments for the thin-layer catalytic
electrochemical PDE problem.  FDM concentration data is used only for posterior
evaluation and plotting.  It is not used in the training loss.

## Current Branch

`codex/fluxtrace-green`

Main code files:

- `pinn_thin_layer_v9_6.py`: training, architectures, checkpointing, in-training posterior compare.
- `compare_concentration_fields.py`: FDM-vs-PINN posterior metrics and residual summary plots.
- `decompose_concentration_error.py`: spatial/time-group error decomposition from saved NPZ fields.
- `plot_direct_concentration_comparison.py`: direct FDM/PINN field and curve plots from saved NPZ/JSON outputs.

## Final Clean Architecture

The current paper-facing workflow is a fixed two-stage clean pipeline:

```text
Stage 1: multiscale_green_grid_dynamic_stage1
Stage 2: multiscale_film_tracegreen_clean
```

Stage 1 is a reproducible warm-start model.  It prepares stable `C_A/C_B`,
`C_B_int`, `J_rxn`, and coarse external fields without carrying over the many
exploratory correction branches.  Stage 2 is the paper-facing model: Film-Abel
interface memory plus erfc trace-preserving TraceGreen external propagation.

### Stage 1: Fixed Dynamic Green Warm Start

The fixed warm-start architecture is:

```text
multiscale_green_grid_dynamic_stage1
```

It keeps the original learnable interface-state network:

```text
C_B_int(t), C_C_int(t), J_rxn(t) = InterfaceStateNet(t)
J_rxn(t) = k_cat*C_B_int(t)*C_C_int(t)
```

The external field is initialized by signed full-field Green history:

```text
C_D^G(y,t) = int_0^t J_rxn(tau) K(y,t-tau) dtau
```

and a small fixed residual scaffold:

```text
C_D(y,t) =
    C_D^G(y,t)
  + Hermite endpoint corrections
  + R_3(y,t)
  + R_dyn,3(y,t,theta,dtheta/dt,J_rxn,dJ_rxn/dt,Q_rxn)

C_C(y,t) = gamma - C_D(y,t)
```

where `R_3` and `R_dyn,3` each use only three endpoint-compatible modes.  This
removes the old Hybrid-lite extra spatial modes and keeps Stage 1 from becoming
a competing final `C_C/C_D` solver.

Stage 1 removes:

```text
Film-Abel interface memory
KernelMix beta terms
matched/mixed Abel
interface residual r_int
hybrid extra spatial modes
TraceGreen boundary lift
```

Stage 1 should be judged mainly by:

```text
C_A/C_B, C_B_int, CV_J, and stability of J_rxn(t)
```

not by final `C_C/C_D` accuracy.

### Stage 2: Clean Film-TraceGreen

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
  + sum_i g_i*M_i[J_rxn](t)
  - dt_phase(t)*A[dJ_rxn/dt](t)

C_C_int(t) = gamma - C_D_int(t)
```

where `A[...]` is the Abel memory term and `M_i[...]` are positive finite-memory
exponential kernels.  The small interface residual and old KernelMix beta terms
are disabled in the clean final path because posterior contribution auditing
showed they do not improve the trained TraceGreen checkpoint.  The external
field is then driven by a single boundary trace:

```text
C_D(y,t) =
    G_trace_erfc[C_D_int](y,t)
  + h01(y)*(0 - D_far(t))
  + s_train(e)*R_smooth(y,t)

C_C(y,t) = gamma - C_D(y,t)
```

`R_smooth` is treated as a training scaffold.  It can be enabled early when the
interface trace is still inaccurate, then linearly decayed to zero.  For final
posterior evaluation from the best TraceGreen checkpoint, the default clean
scale is `s_train=0`.

The key numerical/mathematical contribution is the trace-preserving erfc
quadrature used in `G_trace_erfc`:

```text
lim_{y -> 0+} G_trace_erfc[C_D_int](y,t) = C_D_int(t)
```

This removed the nonphysical discontinuity between the interface point and the
first external grid point that appeared with ordinary time quadrature.

### Recommended Clean Training Schedule

The recommended clean run is:

```text
1. Train Stage 1 from scratch.
2. Save Stage 1 current and best checkpoints.
3. Warm-start Stage 2 from Stage 1 best.
4. Enable R_smooth early with s_train=1.
5. Linearly decay s_train to 0.
6. Use the Stage 2 best checkpoint for posterior FDM evaluation.
```

This keeps the whole experiment "from zero" while avoiding random-initialized
Film-Abel/TraceGreen instability.

### What Is Kept

| Component | Status | Reason |
|---|---|---|
| hard conservation `C_A+C_B=1`, `C_C+C_D=gamma` | final | physical invariant and stable |
| Film relation for `C_B_int` | final | explains the accurate `C_B_int` behavior |
| Abel + positive finite-memory terms | final | compact causal interface memory |
| bounded interface residual `r_int(t)` | removed from final | posterior audit showed no benefit |
| TraceGreen external field | final | propagates `C_D_int` causally into the external region |
| erfc trace-preserving quadrature | final | fixes the near-interface discontinuity |
| smooth external residual `R_smooth(y,t)` | training scaffold | useful early, decayed to zero for final clean evaluation |

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
| old KernelMix `beta_i` terms | removed from final | posterior audit showed negligible or negative contribution |

## Experiment History

| Stage | Architecture | Main change | Representative checkpoint | Key observation |
|---|---|---|---|---|
| Baseline legacy | `legacy` | Direct field PINN | `baseline/pinn_thin_layer_catalytic_v9_6_best.pth` | Useful as historical baseline, but weak for concentration fields. |
| Multiscale | `multiscale` | Larger multiscale MLP | `checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_best.pth` | Improves `C_A/C_B`; `C_C/C_D` still weak. |
| HardBC | `multiscale_hardbc` | Hard conservation for thin film | `checkpoints_hardbc_v2/..._best.pth` | Stabilizes `C_A/C_B`, but does not solve external `C_C/C_D`. |
| Hermite/extbasis | `multiscale_hermite_extbasis` | Hermite thin layer and external basis | `checkpoints_hermite_extbasis_v1/...pth` | Better interface handling, but `C_C_int` remains limited. |
| Green grid hybrid | `multiscale_green_grid_hybrid` | Signed full-field Green history plus residual modes | `checkpoints_green_grid_hybrid_v1/..._best.pth` | Improves full-field `C_C`, but `C_C_int` remains a bottleneck. |
| Dynamic Green | `multiscale_green_grid_dynamic` | Adds dynamic inputs: `theta`, `dtheta/dt`, `J`, `dJ/dt`, `Q` | `checkpoints_green_grid_dynamic_v1/..._best.pth` | Stronger reverse-scan behavior; representative metrics below. |
| Dynamic Green fixed Stage1 | `multiscale_green_grid_dynamic_stage1` | Fixed warm-start version: signed Green plus three residual modes and three dynamic modes | `runs_film_tracegreen_clean_two_stage_256x64/<timestamp>/stage1_dynamic_fixed/...` | Reproducible Stage1 for clean Film-TraceGreen training. |
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

Use this for the complete paper-facing cleaned two-stage architecture and future
parameter generalization experiments:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!git fetch origin
!git checkout codex/parameter-scale-consistency
!git pull --ff-only origin codex/parameter-scale-consistency

!bash run_colab_clean_two_stage_256x64.sh
```

Useful overrides:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!STAGE1_EPOCHS=5000 STAGE2_EPOCHS=1500 \
  CLEAN_RESIDUAL_INITIAL_SCALE=1.0 CLEAN_RESIDUAL_DECAY_EPOCHS=1000 \
  SAVE_EVERY=250 FDM_COMPARE_EVERY=250 \
  bash run_colab_clean_two_stage_256x64.sh
```

For an independent fixed-parameter run, generate a matching FDM file on the
same external domain, then pass the physical parameters to both stages:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!GAMMA=1.0 K_CAT_STAR=1.0 \
  FDM_PKL=/content/gdrive/MyDrive/FDM/gamma1_k1_v42_thin_layer_catalytic_v42.pkl \
  STAGE1_EPOCHS=5000 STAGE2_EPOCHS=1500 \
  CLEAN_RESIDUAL_INITIAL_SCALE=1.0 CLEAN_RESIDUAL_DECAY_EPOCHS=1000 \
  SAVE_EVERY=250 FDM_COMPARE_EVERY=250 \
  bash run_colab_clean_two_stage_256x64.sh
```

`gamma` and `k_cat_star` are fixed run configuration values here; they are not
learned and are not conditional neural-network inputs.  Checkpoints and FDM
metadata are validated before loading, so a `gamma=10, k=1` checkpoint cannot be
silently reused for another physical problem.

Parameter-scale consistency in this branch:

- FDM and PINN both use `L_ext=6*sqrt(T_sim)`, independent of concentration amplitude.
- External PDE/IC/far-field/bounds residuals use `C/gamma` scaling while retaining
  the original `gamma=10` numerical weight calibration.
- Interface flux residuals use
  `J_ref=k*gamma/(1+k*delta*gamma/D_B)` scaling.
- Stage 1 predicts fractional, rather than absolute, `C_C_int` depletion and its
  external correction amplitudes scale with `gamma`.
- Clean Stage 2 maps the Film-Abel prior into `(0,gamma)` with a differentiable
  softplus-ratio map instead of a hard clamp.

### k_cat parameter generalization

`multiscale_film_tracegreen_kparam` generalizes only `k_cat` over `[0.1,10]`;
`gamma=10` and `delta=0.035` remain fixed.  One scalar `k_cat` is sampled per
optimizer step, so the Film-Abel history and every collocation point in that
step share one physical problem.  The first stage samples the anchors
`{0.1,1,10}` with probabilities `{0.25,0.50,0.25}`.  The second stage mixes at
least 50% anchor samples with log-uniform continuous samples.

The new path must warm-start from a parameter-scale-consistency clean `k=1`
checkpoint; a fixed checkpoint cannot be resumed as if it already contained a
parameterized optimizer state:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!WARM_START_CKPT=/content/gdrive/MyDrive/pinn_v96/path/to/clean_k1_best.pth \
  bash run_colab_film_tracegreen_kparam_256x64.sh
```

Multi-case FDM comparison remains posterior-only.  The Colab script evaluates
available anchor files plus the log-midpoint holdouts `0.316` and `3.162`, then
writes per-case metrics and `k_parameter_summary.json` with mean and worst-case
dimensionless errors.  A zero-shot physics audit can use the same comparison
driver with `--zero-shot-fixed-reference` and a fixed clean checkpoint.

Physics-only early stopping is enabled by default.  It uses a fixed deterministic
collocation set and never reads FDM data.  Every 100 epochs it evaluates the
scale-consistent PDE, surface, initial, far-field, interface, bounds, and reversal
residuals; an EMA (`alpha=0.5`) must improve by at least `0.5%` to reset four-check
patience.  Stage 1 cannot stop before 1500 added epochs.  Stage 2 cannot stop until
the `R_smooth` decay has completed plus one validation interval.  `STAGE1_EPOCHS`
and `STAGE2_EPOCHS` are therefore maximum budgets rather than mandatory lengths.

Useful overrides are `EARLY_STOP_CHECK_EVERY`, `EARLY_STOP_PATIENCE_CHECKS`,
`EARLY_STOP_STAGE1_MIN_EPOCHS`, `EARLY_STOP_STAGE2_MIN_EPOCHS`, and
`EARLY_STOP_MIN_RELATIVE_IMPROVEMENT`.  `EARLY_STOP_VALIDATION_POINTS` defaults
to 96 and can be reduced to 64 for a cheaper diagnostic.  Set
`DISABLE_EARLY_STOP=1` only for a deliberate fixed-length ablation.

For a short check:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!STAGE1_EPOCHS=500 STAGE2_EPOCHS=500 \
  CLEAN_RESIDUAL_INITIAL_SCALE=1.0 CLEAN_RESIDUAL_DECAY_EPOCHS=300 \
  SAVE_EVERY=100 FDM_COMPARE_EVERY=100 \
  bash run_colab_clean_two_stage_256x64.sh
```

If Stage 1 already finished and you only want to rerun Stage 2:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!SKIP_STAGE1=1 \
  STAGE1_RESUME_CKPT=/content/gdrive/MyDrive/pinn_v96/runs_film_tracegreen_clean_two_stage_256x64/<timestamp>/stage1_dynamic_fixed/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth \
  STAGE2_EPOCHS=1500 CLEAN_RESIDUAL_INITIAL_SCALE=1.0 CLEAN_RESIDUAL_DECAY_EPOCHS=1000 \
  bash run_colab_clean_two_stage_256x64.sh
```

If you only want to run Stage 2 against an explicit checkpoint, use:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!EPOCHS=1500 SAVE_EVERY=250 FDM_COMPARE_EVERY=250 \
  CLEAN_RESIDUAL_INITIAL_SCALE=1.0 CLEAN_RESIDUAL_DECAY_EPOCHS=1000 \
  RESUME_CKPT=/content/gdrive/MyDrive/pinn_v96/path/to/stage1_or_tracegreen_checkpoint.pth \
  bash run_colab_film_tracegreen_clean_256x64.sh
```

## Posterior Evaluation

```bash
PYTHONIOENCODING=utf-8 python -u compare_concentration_fields.py \
  --arch multiscale_film_tracegreen_clean \
  --gamma 1.0 \
  --k-cat-star 1.0 \
  --input-mode normalized \
  --checkpoint runs_film_tracegreen_clean_256x64/<timestamp>/checkpoints/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth \
  --fdm-pkl ../FDM/gamma1_k1_v42_thin_layer_catalytic_v42.pkl \
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

## Conservative Thin Current and Mixed Flux

For the `gamma=1, k=1` clean checkpoint, the concentration field implies an
accurate CV through the thin-film inventory identity

```text
J_conservative = -J_rxn - d/dt integral_0^delta C_B(x,t) dx
```

even though the raw surface-gradient current is inaccurate.  The two new
architectures isolate and repair that derivative inconsistency without using
FDM in the training loss:

- `multiscale_film_tracegreen_clean_conservative`: unchanged concentration
  forward plus a ramped inventory-current consistency loss.
- `multiscale_film_tracegreen_clean_mixedflux`: explicit `q_B` with hard surface
  and interface flux states and first-order conservation/constitutive losses.

Run both Colab stages from the clean physics-best checkpoint:

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!CLEAN_BEST=/content/gdrive/MyDrive/pinn_v96/path/to/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth \
  GAMMA=1.0 K_CAT_STAR=1.0 \
  FDM_PKL=/content/gdrive/MyDrive/FDM_parameter_scale_0711/gamma1_k1_v42_thin_layer_catalytic_v42.pkl \
  bash run_colab_conservative_mixedflux_256x64.sh
```

FDM is read only after training.  Model selection and early stopping use fixed
physics validation.  Posterior comparison now reports `CV_J_surface`,
`CV_J_conservative`, and `CV_J_surface_vs_conservative`; the legacy `CV_J`
field remains the surface-gradient metric.

Set `SKIP_MIXED=1` to stop after the conservative stage and run only the
recommended current-correction model.

## Inventory-Constrained Hermite Lift

`multiscale_film_tracegreen_conservative_lift` is a zero-training hard
conservation transform for the clean Film-TraceGreen checkpoint.  It adds

```text
C_B = C_B_base + delta * a(t) * h10(x/delta)
h10(s) = s * (1-s)^2
```

and solves the causal inventory equation

```text
(delta^2 / 12) * da/dt + D_A * a
    = J_conservative_base - J_surface_base.
```

The lift preserves both endpoint concentrations and the interface derivative.
It changes only the electrode-side derivative and therefore makes the surface
current satisfy the thin-film inventory identity without FDM supervision.

Run the posterior test directly from the clean physics-best checkpoint:

```bash
CLEAN_BEST=/absolute/path/to/pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth \
WORK_DIR=/content/pinn_v96_inventory_lift_code \
FDM_PKL=/content/gdrive/MyDrive/FDM_parameter_scale_0711/gamma1_k1_v42_thin_layer_catalytic_v42.pkl \
bash /content/pinn_v96_inventory_lift_code/run_colab_inventory_hermite_lift_eval.sh
```

FDM is used only by the posterior comparison.  A local zero-training check at
`gamma=1`, `k=1`, and a 1024-point causal lift grid produced surface-current
RMSE `3.095e-3`, R2 `0.99991`, and surface-versus-conservative RMSE
`3.82e-5`.

## Conservative-Lift Three-Stage Sweep

The inventory lift is intentionally a short third stage, rather than a
replacement for the original clean Stage 1 or Stage 2:

```text
Stage 1: multiscale_green_grid_dynamic_stage1
         Learn a stable concentration field from scratch.

Stage 2: multiscale_film_tracegreen_clean
         Learn the Film-Abel interface state and TraceGreen external field.

Stage 3: multiscale_film_tracegreen_conservative_lift
         Apply the causal Hermite inventory lift and fine-tune the thin PDE.
```

Stage 3 preserves the Stage 2 checkpoint as a separate posterior baseline.
It does not use FDM in its loss, its early stopping, or its checkpoint choice.
This separation avoids introducing the Film-Abel/TraceGreen transition and the
surface-current lift in the same optimizer transition.

Run one fixed-parameter experiment per gamma.  Do not reuse a gamma=10
checkpoint for gamma=100.

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!WORK_DIR=/content/pinn_v96_conservative_lift_code \
  RUN_ROOT=/content/gdrive/MyDrive/pinn_v96/runs_conservative_lift_gamma10 \
  GAMMA=10.0 K_CAT_STAR=1.0 \
  FDM_PKL=/content/gdrive/MyDrive/FDM_parameter_scale_0711/kcat1_v42_thin_layer_catalytic_v42.pkl \
  FDM_COMPARE_EVERY=0 \
  bash /content/pinn_v96_conservative_lift_code/run_colab_conservative_lift_three_stage_256x64.sh
```

For gamma=100, first produce an FDM reference with matching metadata.  The
larger flux warrants a finer time discretization than the default reference:

```bash
%cd /content/gdrive/MyDrive/FDM_parameter_scale_0711
!PYTHONIOENCODING=utf-8 python -u run_fdm_kcat1_v42.py \
  --gamma 100.0 --k-cat 1.0 \
  --n-x-in 400 --n-x-out 1000 --n-t 16000 \
  --reaction-iterations 6 \
  --output-prefix gamma100_k1_v42
```

Then run a separate PINN experiment.  `LIFT_TIME_GRID=2048` is recommended for
this high-flux case so the causal lift resolves faster inventory transients.

```bash
%cd /content/gdrive/MyDrive/pinn_v96
!WORK_DIR=/content/pinn_v96_conservative_lift_code \
  RUN_ROOT=/content/gdrive/MyDrive/pinn_v96/runs_conservative_lift_gamma100 \
  GAMMA=100.0 K_CAT_STAR=1.0 \
  FDM_PKL=/content/gdrive/MyDrive/FDM_parameter_scale_0711/gamma100_k1_v42_thin_layer_catalytic_v42.pkl \
  LIFT_TIME_GRID=2048 \
  FDM_COMPARE_EVERY=0 \
  bash /content/pinn_v96_conservative_lift_code/run_colab_conservative_lift_three_stage_256x64.sh
```

The script checks the FDM `gamma` and `k_cat` metadata before starting.  It
writes Stage 1, Stage 2, Stage 3, and posterior direct-comparison artifacts
into the selected `RUN_ROOT`.
