# Physical Model and Operator Factorization

## 1. Scope

The current model contains a finite reactive film, $0<x<\delta$, and a
semi-infinite external diffusion region, $y=x-\delta\geq0$. The electrode is at
$x=0$ and the film/external interface is at $x=\delta$.

The implementation is dimensionless. Its runtime physical parameters are

$$
\mathbf p=(k_{\mathrm{cat}},\gamma,\delta),
\qquad
k_{\mathrm{cat}}>0,
\quad \gamma>0,
\quad \delta>0.
$$

No parameter is treated as a new spatial coordinate. A parameter change means
solving a new causal physical trajectory.

## 2. Conservation Reduction

The present mainline assumes

$$
D_A=D_B,
\qquad
D_C=D_D.
$$

The two conserved totals are therefore

$$
C_A+C_B=1,
\qquad
C_C+C_D=\gamma.
$$

Only $C_B$ in the film and $C_D$ in the external region need to be propagated:

$$
C_A=1-C_B,
\qquad
C_C=\gamma-C_D.
$$

This reduction is exact under the equal-diffusivity assumptions. Unequal
diffusivities require separate DtN states and are listed as an extension, not
silently approximated.

## 3. Electrode Protocol

For a triangular scan with positive scan rate $\sigma$,

$$
T=\frac{2|\theta_i-\theta_{\mathrm{sw}}|}{\sigma}.
$$

The default protocol uses $\theta_i=10$, $\theta_{\mathrm{sw}}=-10$, and a
switch at $T/2$. Reversible Nernst equilibrium at $x=0$ gives

$$
C_A(0,t)=e^{\theta(t)}C_B(0,t),
$$

and conservation gives the exact surface state

$$
C_B(0,t)=g(t)=\frac{1}{1+e^{\theta(t)}},
\qquad
C_A(0,t)=\frac{1}{1+e^{-\theta(t)}}.
$$

## 4. Film and External Diffusion

The independent concentrations satisfy

$$
\frac{\partial C_B}{\partial t}
=D_B\frac{\partial^2C_B}{\partial x^2},
\qquad 0<x<\delta,
$$

$$
\frac{\partial C_D}{\partial t}
=D_D\frac{\partial^2C_D}{\partial y^2},
\qquad y>0.
$$

The initial and far-field conditions are

$$
C_B(x,0)=0,
\qquad
C_D(y,0)=0,
\qquad
C_D(y,t)\longrightarrow0 \text{ as } y\longrightarrow\infty.
$$

## 5. Nonlinear Interface Closure

At the film/external interface,

$$
C_{B,i}(t)=C_B(\delta,t),
\qquad
C_{C,i}(t)=\gamma-C_D(0,t).
$$

The catalytic flux is

$$
J(t)=k_{\mathrm{cat}}C_{B,i}(t)C_{C,i}(t).
$$

It imposes the film Neumann condition

$$
-D_B\frac{\partial C_B}{\partial x}(\delta,t)=J(t).
$$

The half-space Neumann-to-trace map is the Abel history

$$
C_{D,i}(t)
=\frac{1}{\sqrt{\pi D_D}}
\int_0^t\frac{J(\tau)}{\sqrt{t-\tau}}\,\mathrm d\tau,
$$

so that $C_{C,i}=\gamma-C_{D,i}$. This closes the reaction, finite-film
diffusion, and external diffusion in one causal nonlinear history problem.

## 6. Electrode Current and Inventory

The film inventory is

$$
M_B(t)=\int_0^\delta C_B(x,t)\,\mathrm dx.
$$

Integrating the film diffusion equation gives the exact balance

$$
J_{\mathrm e}(t)=-J(t)-\frac{\mathrm dM_B}{\mathrm dt}.
$$

The implementation computes the headline current directly from the spectral
surface derivative. Backward and centered inventory derivatives are retained
as causal and noncausal diagnostics, respectively.

## 7. Dimensionless Regimes

The principal runtime groups are

$$
\mathrm{Da}=\frac{k_{\mathrm{cat}}\gamma\delta}{D_B},
\qquad
\mathrm{Fo}_T=\frac{D_BT}{\delta^2}.
$$

A useful flux scale is

$$
J_{\mathrm{ref}}
=\frac{k_{\mathrm{cat}}\gamma}
{1+k_{\mathrm{cat}}\gamma\delta/D_B}.
$$

$\mathrm{Da}$ controls kinetic-to-film transport competition, while
$\mathrm{Fo}_T$ controls how much of the film can equilibrate during one scan.
At very small catalytic flux, $J/J_{\mathrm{ref}}$ becomes an intentionally
strict but ill-conditioned error normalization; absolute current RMSE and CV
span NRMSE must be reported beside it.

## 8. General Operator Form

The reusable structure is broader than the present rate law. Linear transport
operators make the current-cell interface state affine in an unknown flux:

$$
\mathbf c_{i,n}=\mathbf a_n+\mathbf B_n\mathbf J_n.
$$

A local rate law $\mathbf R$ then produces a low-dimensional closure

$$
\mathbf F(\mathbf J_n;\mathbf p)
=\mathbf J_n
-\mathbf R(\mathbf a_n+\mathbf B_n\mathbf J_n;\mathbf p)
=\mathbf0.
$$

The current repository uses a scalar $J_n$ and a bimolecular $R$. Other local
rate laws can reuse the transport operators; multiple reactions lead to a
small vector closure. New geometries or migration physics require new transport
operators and are not covered by the current implementation.
