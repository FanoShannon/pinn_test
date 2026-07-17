# Stable ProductIntegral-DtN-SOE Algorithm

## 1. Finite-Slab DtN Representation

Let $g(t)=C_B(0,t)$. The mixed Dirichlet/Neumann film solution is represented as

$$
C_B(x,t)
=g(t)-\frac{x}{D_B}J(t)
+\sum_{m=0}^{M-1}a_m(t)\sin(\mu_mx),
$$

with

$$
\mu_m=\frac{(m+1/2)\pi}{\delta}.
$$

The half-integer spectrum makes the sine modes zero at the electrode and gives
zero modal derivative at the interface. The modal equations are

$$
\dot a_m
=-D_B\mu_m^2a_m
-\frac{2\dot g}{\delta\mu_m}
+\frac{2(-1)^m\dot J}{\delta D_B\mu_m^2}.
$$

For piecewise-linear $g$ and $J$, each mode is advanced by an exponential
time-differencing cell update. No film spatial grid is introduced.

At the end of time cell $n$, this update gives

$$
C_{B,i,n}=b_{0,n}+b_{1,n}J_n.
$$

All completed film history is contained in $b_{0,n}$; $b_{1,n}$ is the exact
current-cell DtN response of the truncated spectral system.

## 2. Singularity-Matched ProductIntegration

The Abel kernel diverges as $(t-\tau)^{-1/2}$ near the current time. Applying a
regular quadrature rule directly to the last cell loses the dominant local
mass. The algorithm instead interpolates $J$ linearly and integrates the
singular kernel analytically.

At $t_n$,

$$
C_{D,i,n}
=I_{n}^{\mathrm{completed}}
+2\sqrt{\frac{\Delta t}{\pi D_D}}
\left(\frac13J_{n-1}+\frac23J_n\right).
$$

Therefore

$$
C_{C,i,n}=c_{0,n}+c_{1,n}J_n,
$$

where the completed causal history is in $c_{0,n}$ and

$$
c_{1,n}=-\frac43\sqrt{\frac{\Delta t}{\pi D_D}}.
$$

The $1/3,2/3$ weights are part of the nonlinear closure; they are never
approximated by the SOE acceleration.

## 3. Scalar Coupled Closure

The two affine interface states reduce the full current cell to

$$
F(J_n)
=J_n
-k_{\mathrm{cat}}
(b_{0,n}+b_{1,n}J_n)
(c_{0,n}+c_{1,n}J_n)
=0.
$$

The NumPy reference brackets the physical root using positivity bounds for the
two interface concentrations. Each Newton proposal is accepted only inside the
active bracket; otherwise the method takes a bisection step. This prevents
high-$\mathrm{Da}$ Newton excursions from selecting a nonphysical quadratic
root.

The differentiable PyTorch recurrence uses a fixed number of analytic Newton
updates. Tests compare it with the safeguarded NumPy reference and verify
finite gradients over the calibrated parameter range.

## 4. Direct and SOE Abel History

The direct backend evaluates every completed ProductIntegral cell and costs
$O(N_t^2)$. It remains the numerical reference.

The fast backend splits the completed history into a direct near part and an
SOE far part:

$$
u^{-1/2}\approx\sum_{q=1}^{P}w_qe^{-\lambda_qu}.
$$

Every exponential contributes one recursive causal state. With
$N_{\mathrm{near}}$ exact cells and $P$ exponential terms, the memory cost is
$O(P)$ and the history work is $O((N_{\mathrm{near}}+P)N_t)$.

The implementation uses positive quadrature weights, caches plans by grid and
diffusion parameters, and preserves the exact singular current cell. The SOE
backend is currently restricted to uniform time grids.

## 5. Trace-Preserving External Field

After solving the interface trace, the external field is reconstructed with
the half-space heat potential

$$
C_D(y,t)
=\int_0^t
\frac{y\,e^{-y^2/[4D_D(t-\tau)]}}
{2\sqrt{\pi D_D(t-\tau)^3}}
C_{D,i}(\tau)\,\mathrm d\tau.
$$

The implementation transforms the integration variable to the complementary
error-function mass coordinate. Gauss-Legendre quadrature in that coordinate
preserves the $y\to0^+$ trace far more accurately than uniform time
quadrature. A cubic endpoint correction enforces the finite numerical far
boundary without changing the interface value.

## 6. Differentiation

The PyTorch recurrence is written entirely in float64 tensor operations. For a
generic closure $F(J_n,\mathbf p)=0$, the local implicit sensitivity is

$$
\frac{\partial J_n}{\partial\mathbf p}
=-
\left(\frac{\partial F}{\partial J_n}\right)^{-1}
\frac{\partial F}{\partial\mathbf p}.
$$

Automatic differentiation through the unrolled recurrence also includes the
parameter response of all previous DtN and Abel states. No artificial PDE term
such as $\partial C/\partial k$ is introduced.

## 7. Stability Invariants

The implementation and tests enforce:

- finite positive runtime parameters;
- a uniform, increasing time grid;
- exact initial concentration and zero initial current;
- physical root bracketing in the NumPy reference;
- reaction closure residual monitoring;
- direct-versus-SOE output and gradient agreement;
- FDM parameter matching before any posterior comparison.
