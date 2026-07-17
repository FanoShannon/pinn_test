# Verified Results

All values below were produced by scripts in this branch or by their direct
pre-cleanup equivalents. Forward/inverse operators use no FDM data. Rows marked
"FDM posterior" compare an already-computed operator trajectory against an
independent FDM v4.2 file.

## 1. Reference Forward Case

At $k_{\mathrm{cat}}=1$, $\gamma=10$, $\delta=0.035$, with 256 time points,
64 film modes, and 64 TraceGreen quadrature points:

| Metric | Zero-training ProductIntegral-DtN |
|---|---:|
| Overall dimensionless RMSE | $9.324963\times10^{-5}$ |
| Film $C_B$ RMSE | $3.100714\times10^{-5}$ |
| Interface $C_B$ RMSE | $3.982946\times10^{-5}$ |
| Surface-current RMSE divided by $J_{\mathrm{ref}}$ | $3.877808\times10^{-4}$ |

The maximum nonlinear closure error was approximately $4.2\times10^{-13}$.

## 2. Temporal Convergence

Using 64 film modes at the same reference parameters:

| Time points | $C_B$ RMSE | Interface $C_B$ RMSE | Surface CV/$J_{\mathrm{ref}}$ |
|---:|---:|---:|---:|
| 64 | $1.6324\times10^{-4}$ | $2.1334\times10^{-4}$ | $1.8952\times10^{-3}$ |
| 128 | $7.4249\times10^{-5}$ | $9.6138\times10^{-5}$ | $8.8461\times10^{-4}$ |
| 256 | $3.1007\times10^{-5}$ | $3.9829\times10^{-5}$ | $3.8778\times10^{-4}$ |
| 512 | $1.0547\times10^{-5}$ | $1.3469\times10^{-5}$ | $1.5230\times10^{-4}$ |
| 1024 | $2.9451\times10^{-6}$ | $4.1581\times10^{-6}$ | $6.4731\times10^{-5}$ |

The remaining low-flux current error is primarily temporal/modal derivative
error rather than reaction-closure error.

## 3. Joint Positive-Parameter Stress Audit

A 100-case grid spanning

$$
10^{-5}\leq\mathrm{Da}\leq1.4\times10^3
$$

produced no NaN, invalid concentration, or wrong nonlinear root. Newton used at
most five safeguarded iterations and the closure residual remained below
$6.2\times10^{-13}$.

Representative zero-training cases include:

| $k_{\mathrm{cat}}$ | $\gamma$ | Time x modes | Overall RMSE | Surface CV/$J_{\mathrm{ref}}$ |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1024 x 64 | $1.0742\times10^{-4}$ | $5.3320\times10^{-4}$ |
| 1 | 100 | 256 x 64 | $3.2832\times10^{-5}$ | $7.6772\times10^{-5}$ |
| 100 | 10 | 256 x 64 | $3.5239\times10^{-4}$ | $8.6588\times10^{-5}$ |
| 0.1 | 0.1 | 4096 x 256 | $1.4279\times10^{-5}$ | $1.2998\times10^{-2}$ |
| 0.01 | 1 | 4096 x 256 | $1.8565\times10^{-6}$ | $1.2997\times10^{-2}$ |

For the last two cases, $J_{\mathrm{ref}}\approx0.01$ while the reversible CV
span is approximately 0.70. Their CV-span NRMSE is about
$1.86\times10^{-4}$; the larger CV/$J_{\mathrm{ref}}$ value is a scale effect.

## 4. Thickness FDM Posterior

The operator uses 4096 time points, 256 film modes, and 64-point
Gauss-Legendre TraceGreen quadrature. FDM uses 400 film cells, 1000 external
cells, and at least 8000 time points.

| $\delta$ | Overall dimensionless RMSE | Interface $C_B$ RMSE | CV/$J_{\mathrm{ref}}$ RMSE |
|---:|---:|---:|---:|
| 0.0100 | $8.844\times10^{-6}$ | $3.350\times10^{-6}$ | $3.388\times10^{-5}$ |
| 0.0175 | $8.933\times10^{-6}$ | $5.185\times10^{-6}$ | $2.903\times10^{-5}$ |
| 0.0350 | $1.004\times10^{-5}$ | $2.995\times10^{-6}$ | $1.015\times10^{-5}$ |
| 0.0700 | $9.337\times10^{-6}$ | $1.240\times10^{-5}$ | $2.168\times10^{-5}$ |
| 0.1400 | $1.110\times10^{-5}$ | $1.683\times10^{-5}$ | $8.763\times10^{-5}$ |

Five independent joint $(k_{\mathrm{cat}},\gamma,\delta)$ posterior cases had
overall dimensionless RMSE between $1.255\times10^{-6}$ and
$3.394\times10^{-5}$.

## 5. SOE Accuracy and Cost

The completed far history is accelerated while the singular current cell and
16 recent cells remain direct.

| Time points | NumPy speedup | PyTorch forward/backward speedup | Max $|\Delta J|$ |
|---:|---:|---:|---:|
| 257 | 1.53x | 1.19x | $2.93\times10^{-11}$ |
| 1025 | 2.46x | 1.72x | $3.21\times10^{-11}$ |
| 4097 | 3.85x | not measured | $3.64\times10^{-11}$ |
| 8193 | 5.87x | not measured | $4.33\times10^{-11}$ |

At 1025 points, the relative SOE/direct difference in the automatic derivative
with respect to $\log\delta$ was $1.50\times10^{-11}$.

## 6. Delta Inversion

Targets use a 4097 x 512 NumPy operator. Inversion uses 257 current samples and
a 257 x 96 differentiable operator, starting from $\delta=0.035$.

| True $\delta$ | No-noise estimate | Relative error | 0.1% noise estimate | Relative error |
|---:|---:|---:|---:|---:|
| 0.0175 | 0.01753295 | 0.188% | 0.01752657 | 0.152% |
| 0.0350 | 0.03505570 | 0.159% | 0.03507680 | 0.219% |
| 0.0700 | 0.07007069 | 0.101% | 0.07006476 | 0.093% |
| 0.1400 | 0.14011487 | 0.082% | 0.14012304 | 0.088% |

Six initial values from 0.006 to 0.18 converged to the same $\delta=0.07$
solution. With only 65 CV samples, the worst error was 1.01% at 0.5% noise and
1.42% at 1% noise.

## 7. Multi-Scan Joint Inversion

Blind synthetic inversions used target resolution 2049 x 256, inverse
resolution 257 x 96, three scan rates $(2.5,40,640)$, four starts, and 36 LBFGS
iterations.

| True $(k,\gamma,\delta)$ | Noise | Estimated parameters | Maximum relative error |
|---|---:|---|---:|
| (1, 10, 0.035) | 0 | (1.00012, 9.99647, 0.0350132) | 0.035% |
| (1, 10, 0.035) | 1% | (0.97325, 10.4314, 0.0336244) | 4.31% |
| (0.3, 30, 0.07) | 0 | (0.294667, 30.2424, 0.0691146) | 1.78% |
| (3, 3, 0.07) | 0 | (2.98118, 2.99835, 0.0692193) | 1.12% |

The last two cases share $k\gamma=9$, so their separation demonstrates use of
the independent external inventory/history effect of $\gamma$.

## 8. Honest Failure Regimes

- At $(0.1,0.1,0.035)$, unrestricted three-parameter inversion admits a
  low-objective false branch because the chemistry is close to the product
  $k\gamma$. Fixing $\gamma$ reduced the maximum error to 4.28%.
- At $(10,100,0.14)$, transport saturation destroys kinetic information.
  Different starts return substantially different $k$ and $\gamma$ with nearly
  identical objectives. Delta-only inversion remains accurate when $k$ and
  $\gamma$ are known.

These failures are part of the result: forward parameter generality does not
imply global inverse identifiability.

