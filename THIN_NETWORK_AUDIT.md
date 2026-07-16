# Thin-network contribution audit

This branch evaluates whether the remaining neural thin-layer correction is
needed after the ProductIntegral interface and inventory lift have been fixed.
It does not change the default ProductIntegral architecture.

## Data-use boundary

- The trained checkpoint is used only by the network ablation and the one-way
  diagnostic.
- The coupled ProductIntegral-DtN operator uses no checkpoint and performs zero
  optimization steps.
- FDM concentration and current values enter only `evaluate()` and plotting
  functions after the physical solution has been computed.
- The coupled solver receives only the known CV protocol, diffusivities,
  `delta`, `k_cat`, `gamma`, the temporal resolution, and the modal resolution.
- The conservation reductions require `D_A=D_B` and `D_C=D_D`; posterior
  metadata is checked before evaluation.

## Existing thin model

The checkpoint-compatible thin field is

```text
C_B = cubic Hermite physical field
    + surface-slope neural bubble
    + endpoint-preserving interior neural bubble
    + posterior inventory lift.
```

The four posterior ablations independently disable the two learned bubbles:

| Inventory lift | Thin correction | Overall dimensionless RMSE | C_B RMSE | CV/Jref RMSE |
|---|---:|---:|---:|---:|
| off | full network | 8.048689e-4 | 1.208463e-3 | 1.365070e-2 |
| off | physics only | 8.046440e-4 | 1.208113e-3 | 1.408891e-2 |
| on | full network | 6.522835e-4 | 9.703548e-4 | 1.502166e-3 |
| on | physics only | 6.483041e-4 | 9.641121e-4 | 1.502031e-3 |

At `k_cat=1`, `gamma=10`, the learned bubbles do not improve the final lifted
solution. Their total correction has RMS `3.74e-6`; 96.8% of its spatial modal
energy lies in the first four Dirichlet modes.

## Finite-slab DtN operator

Inside the film,

```math
\frac{\partial C_B}{\partial t}
=D_B\frac{\partial^2 C_B}{\partial x^2},
\qquad 0<x<\delta ,
```

with

```math
C_B(0,t)=g(t),
\qquad
-D_B\frac{\partial C_B}{\partial x}(\delta,t)=J(t).
```

Use

```math
C_B(x,t)=g(t)-\frac{x}{D_B}J(t)
+\sum_{m=0}^{M-1}a_m(t)\sin(\mu_m x),
\qquad
\mu_m=\frac{(m+\tfrac12)\pi}{\delta}.
```

The transformed modes satisfy

```math
\dot a_m
=-D_B\mu_m^2a_m
-\frac{2\dot g}{\delta\mu_m}
+\frac{2(-1)^m\dot J}{\delta D_B\mu_m^2}.
```

For piecewise-linear `g` and `J`, every mode is advanced exactly by an
exponential time-differencing cell update.

The external interface trace uses the singularity-matched ProductIntegral cell:

```math
C_{D,i,n}
=I_{\mathrm{completed},n}
+2\sqrt{\frac{\Delta t}{\pi D_D}}
\left(\frac13J_{n-1}+\frac23J_n\right).
```

For a fixed history at the start of cell `n`, the two interface concentrations
are affine in the unknown current:

```math
C_{B,i,n}=b_0+b_1J_n,
\qquad
C_{C,i,n}=c_0+c_1J_n.
```

Therefore the fully coupled closure is one scalar equation:

```math
F(J_n)
=J_n-k_{\mathrm{cat}}
\left(b_0+b_1J_n\right)
\left(c_0+c_1J_n\right)=0.
```

A bracketed analytic Newton update solves this equation causally in each time
cell. The reference run closes it to `4.16e-13`.

The solved `C_D` interface history is propagated through the existing
trace-preserving erfc heat potential. No spatial grid is used by either the
finite-slab DtN operator or the external TraceGreen operator.

## Reference result

The fair headline current is the direct causal surface flux. Centered inventory
differentiation is retained only as a posterior diagnostic because it reads both
neighboring time points.

At `k_cat=1`, `gamma=10`, with 256 time points, 64 thin modes, and 64 TraceGreen
quadrature points:

| Method | Checkpoint | Overall dimensionless RMSE | C_B RMSE | C_B interface RMSE | CV/Jref RMSE |
|---|---:|---:|---:|---:|---:|
| ProductIntegral + NN + lift | yes | 6.522835e-4 | 9.703548e-4 | 1.560192e-3 | 1.502166e-3 |
| Coupled ProductIntegral-DtN | no | 9.324963e-5 | 3.100714e-5 | 3.982946e-5 | 3.877808e-4 |

The zero-training coupled operator reduces:

- overall dimensionless RMSE by 85.7%;
- thin `C_B` RMSE by 96.8%;
- thin interface RMSE by 97.4%;
- causal surface-current RMSE by 74.2%.

The causal backward-inventory current gives `4.939502e-4`. The noncausal
centered diagnostic gives `5.710728e-5` and must not be used as the headline
result.

## Resolution study

The following table uses 64 thin modes at `k_cat=1`, `gamma=10`.

| Time points | C_B RMSE | C_B interface RMSE | Surface CV/Jref | Backward-inventory CV/Jref |
|---:|---:|---:|---:|---:|
| 64 | 1.632375e-4 | 2.133365e-4 | 1.895242e-3 | 1.997093e-3 |
| 128 | 7.424873e-5 | 9.613767e-5 | 8.846065e-4 | 9.913503e-4 |
| 256 | 3.100714e-5 | 3.982946e-5 | 3.877808e-4 | 4.939502e-4 |
| 512 | 1.054691e-5 | 1.346874e-5 | 1.523021e-4 | 2.462602e-4 |
| 1024 | 2.945109e-6 | 4.158100e-6 | 6.473109e-5 | 1.203770e-4 |

The old one-way diagnostic showed a vertical line at `theta=10`. It came from
evaluating the truncated spectral derivative at the incompatible `t=0`
Dirichlet switch. The corrected plots enforce the exact initial state
`C_B(x,0)=0` and `J_surface(0)=0`; the artifact is not a physical CV peak.

## Parameter stress tests

All cases below use zero training and no checkpoint.

| k_cat | gamma | Time points | Modes | Overall dimensionless RMSE | Surface CV/Jref |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 256 | 64 | 1.108014e-4 | 3.471451e-3 |
| 1 | 1 | 1024 | 64 | 1.074214e-4 | 5.331970e-4 |
| 1 | 100 | 256 | 64 | 3.283164e-5 | 7.677204e-5 |
| 100 | 10 | 256 | 64 | 3.523919e-4 | 8.658770e-5 |
| 0.1 | 0.1 | 256 | 64 | 3.211144e-5 | 3.421819e-1 |
| 0.1 | 0.1 | 1024 | 64 | 1.433198e-5 | 5.240649e-2 |
| 0.1 | 0.1 | 4096 | 256 | 1.427932e-5 | 1.299820e-2 |
| 0.01 | 1 | 256 | 64 | 2.885232e-5 | 3.421768e-1 |
| 0.01 | 1 | 1024 | 64 | 2.310012e-6 | 5.240481e-2 |
| 0.01 | 1 | 4096 | 256 | 1.856515e-6 | 1.299652e-2 |

Low `gamma` exposes the main remaining numerical limitation: normalizing by the
smaller `J_ref` amplifies temporal surface-derivative error. Increasing the
operator grid from 256 to 1024 points reduces the `gamma=1` surface-current
error by 84.6%.

The two low-product cases have nearly the same thin field because both have
`k_cat * gamma = 0.01`. Their external normalized errors remain different, so
the solver has not collapsed the two physical parameters into one condition.

For these cases, `J_ref` is approximately `0.01`, while the total reversible CV
span is approximately `0.70`. At 4096 time points and 256 modes:

| k_cat | gamma | CV absolute RMSE | CV amplitude NRMSE | CV/Jref |
|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 1.299365e-4 | 1.856025e-4 | 1.299820e-2 |
| 0.01 | 1 | 1.299197e-4 | 1.856468e-4 | 1.299652e-2 |

Thus `CV/Jref` is deliberately stringent but becomes ill-conditioned as the
catalytic scale approaches zero. Absolute CV error and CV-span NRMSE must be
reported beside it.

At equal 256-time/64-mode cost, the low-product surface current is worse than
the previous ProductIntegral zero-shot result. The physical operator becomes
better only after temporal and modal refinement. A production replacement
therefore needs an adaptive grid or an accelerated analytic surface-DtN current,
not one fixed resolution for the whole positive parameter domain.

## Reproduction

Network contribution audit:

```powershell
python analyze_thin_network_contribution.py `
  --checkpoint <productintegral-checkpoint.pth> `
  --fdm-pkl <posterior-case.pkl> `
  --output-dir <audit-output>
```

Fully coupled zero-training operator:

```powershell
python prototype_coupled_productintegral_dtn.py `
  --fdm-pkl <posterior-case.pkl> `
  --output-dir <operator-output> `
  --k-cat 1 `
  --gamma 10 `
  --operator-time-grid 256 `
  --mode-counts 64
```

Unit tests:

```powershell
python -m unittest -v test_thin_dtn_operator.py
```

## Research decision

The reference result supports replacing the remaining thin neural field with a
coupled physical operator. It does not yet justify deleting the neural path from
the production branch. Before that decision:

1. run the full joint `k_cat/gamma` posterior grid;
2. formalize modal and temporal convergence bounds;
3. benchmark wall time and memory against FDM and grid-free CV;
4. compare the direct surface flux and a strictly causal high-order inventory
   derivative;
5. retain a small learned residual only if a parameter-region audit shows a
   repeatable physical-model discrepancy.
