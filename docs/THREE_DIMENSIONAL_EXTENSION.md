# True-3D Curved-Film Extension

## 1. Scope of the First Prototype

The first three-dimensional problem is a planar electrode beneath a curved,
strictly positive-thickness film. The film occupies

$$
\Omega_f
=\left\{(x,y,z):(x,y)\in A,\ 0<z<h(x,y)\right\},
$$

and the external electrolyte occupies the remainder of the finite reference
box above the curved interface. The default cap is a spherical cap on a flat
positive-thickness base:

$$
h(x,y)=\delta_0+
\begin{cases}
\sqrt{R^2-r^2}-\sqrt{R^2-a^2},&r<a,\\
0,&r\ge a,
\end{cases}
\qquad r=\sqrt{x^2+y^2}.
$$

For cap height $H$ and base radius $a$,

$$
R=\frac{a^2+H^2}{2H}.
$$

The positive base thickness $\delta_0$ avoids a zero-thickness contact line.
A literal hemisphere touching the electrode at its rim is deferred because
that geometry introduces a corner singularity and a degenerate film thickness.
The code also retains a cosine-cap option as a smooth-profile numerical control.

No polar or azimuthal symmetry is imposed. All fields are represented as
$C(x,y,z,t)$ on a Cartesian discretization. The default interface kinetics are
deliberately non-axisymmetric:

$$
k(x,y,z)
=k_0\exp\left(
\epsilon_x\frac{x}{L_x/2}
+\epsilon_{xy}\frac{xy}{(L_x/2)(L_y/2)}
\right).
$$

Consequently, the interface flux cannot be represented as $J(r,t)$.

## 2. Coupled Three-Dimensional Physics

Using the equal-diffusivity conservation reduction, the prototype propagates
$C_B$ in the film and $C_D$ in the external electrolyte:

$$
\partial_t C_B=D_B\nabla^2C_B,\qquad \mathbf x\in\Omega_f,
$$

$$
\partial_t C_D=D_D\nabla^2C_D,\qquad \mathbf x\in\Omega_o,
$$

with

$$
C_A=1-C_B,\qquad C_C=\gamma-C_D.
$$

The local interface closure is

$$
J(\mathbf s,t)
=k(\mathbf s)C_{B,i}(\mathbf s,t)C_{C,i}(\mathbf s,t),
\qquad \mathbf s\in\Gamma_i.
$$

The planar electrode is driven by the same reversible triangular-potential
boundary state as the one-dimensional mainline. The top of the finite external
reference box uses $C_D=0$; lateral boundaries are no-flux.

## 3. Discrete DtN Condensation

This prototype first discretizes both volumes with a cell-centered finite-volume
operator. Backward Euler gives

$$
M_f\mathbf b_n
=\mathbf r_{f,n}-\Delta t\,S_f\mathbf J_n,
$$

$$
M_o\mathbf d_n
=\mathbf r_{o,n}+\Delta t\,S_o\mathbf J_n.
$$

The volume states are eliminated before the nonlinear solve. Interface traces
therefore have the affine form

$$
\mathbf b_{i,n}=\mathbf b_{0,n}-R_f\mathbf J_n,
$$

$$
\mathbf c_{i,n}=\mathbf c_{0,n}-R_o\mathbf J_n.
$$

The current-cell closure is the surface-vector equation

$$
\mathbf F(\mathbf J_n)
=\mathbf J_n
-\mathbf k\odot
(\mathbf b_{0,n}-R_f\mathbf J_n)
\odot
(\mathbf c_{0,n}-R_o\mathbf J_n)
=\mathbf0.
$$

A positivity-preserving line-searched Newton method solves this system. This is
the direct three-dimensional analogue of the scalar closure in the planar
ProductIntegral-DtN solver.

## 4. What the Prototype Does and Does Not Establish

It establishes that:

- the scalar interface state can be promoted to a two-dimensional surface field;
- full three-dimensional diffusion can be condensed to an interface-only
  nonlinear solve;
- a non-axisymmetric $k(x,y)$ produces a measurable non-axisymmetric flux field;
- the flat, uniform limit preserves lateral uniformity exactly on the grid.

It does not yet establish a volume-grid-free three-dimensional algorithm. The
sparse volume matrices are currently used to construct $R_f$ and $R_o$. This is
a verification scaffold and does not use FDM data, neural-network training, or
posterior fitting.

## 5. Route to ProductIntegral-DtN-SOE in Curved Geometry

The final method must replace the volume-generated responses by heat-potential
boundary operators on the planar electrode and curved interface. In schematic
form, a curved-boundary heat representation contains surface-time operators

$$
\left(\frac12I+K_D\right)c_i=V_Dj+c_{\mathrm{initial}},
$$

where $V_D$ and $K_D$ are heat single- and double-layer operators. Unlike the
planar Abel kernel, these operators couple temporal singularity, panel distance,
surface normal, and curvature.

The planned sequence is:

1. compare the condensed prototype with a direct sparse volume solve;
2. implement direct-history heat boundary elements on triangular panels;
3. add singular local-panel ProductIntegration;
4. compress completed history with operator-valued SOE or convolution
   quadrature;
5. add matrix-free Newton-Krylov and FMM or hierarchical-matrix acceleration;
6. validate symmetric caps, then non-axisymmetric kinetics and geometry against
   an independent three-dimensional FDM or FEM posterior.

Only after step 4 should the three-dimensional method be described as a
ProductIntegral-DtN-SOE solver.

The current implementation already checks the algebraic reconstruction by
substituting each condensed solution into both sparse volume equations. A
three-level grid audit compares the integrated electrode current and mean
interface flux, but it is not yet a continuum convergence certificate because
the curved interface remains voxelized.

## 6. Reproduction

Local smoke test:

```bash
python -m unittest -v test_curved_film_3d.py
python run_curved_film_3d.py --output runs_curved_film_3d/prototype
python analyze_curved_film_3d_convergence.py
```

Colab:

```bash
git clone --branch codex/true-3d-curved-film-prototype \
  https://github.com/FanoShannon/pinn_test.git /content/curved_film_3d

OUTPUT_DIR=/content/gdrive/MyDrive/curved_film_3d/prototype \
bash /content/curved_film_3d/run_colab_curved_film_3d.sh
```

The summary explicitly records `symmetry_reduction=false` and
`volume_grid_free=false` so that the exploratory result cannot be confused with
the mature one-dimensional mainline claim.

## 7. Initial Result

For the default spherical cap with a non-axisymmetric kinetic map, an
$8\times7\times18$ grid produced 112 interface-face unknowns. The nonlinear
closure required at most four Newton iterations, with maximum closure residual
$7.62\times10^{-11}$ and sparse-volume reconstruction residual
$3.55\times10^{-15}$. The reflected-flux asymmetry index was 0.108, whereas the
flat uniform control was symmetric to machine precision.

The medium-to-fine grid difference was 0.91% for integrated electrode current
but 4.90% for mean interface flux. This supports the boundary-condensation
architecture while showing that voxelized surface geometry is not accurate
enough for final local-flux claims. The frozen machine-readable audit is in
[`results/curved_film_3d_prototype_results.json`](../results/curved_film_3d_prototype_results.json).

## 8. Independent Monolithic FDM Posterior

The posterior reference solves the film volume, external volume, and all
interface fluxes in one sparse block Newton system. It does not use the
precomputed DtN response or the condensed solution as an initial condition.
On the same grid, the two formulations agree to machine precision:

$$
\max_t|J_{\mathrm{CV}}^{\mathrm{DtN}}-J_{\mathrm{CV}}^{\mathrm{FDM}}|
=6.22\times10^{-15}.
$$

The accuracy posterior then compares the $8\times7\times18$, 65-step condensed
operator against a $14\times13\times36$, 193-step monolithic FDM. The main
results are:

- CV NRMSE: 2.98%;
- mean curved-interface flux NRMSE: 3.96%;
- mean film-side interface concentration NRMSE: 4.00%;
- mean external $C_C/\gamma$ interface concentration NRMSE: 0.37%;
- forward peak-current relative error: 0.48%;
- reverse peak-current relative error: 6.77%.

The FDM reference itself changes by 0.74% in CV NRMSE between the
$12\times11\times30$, 129-step and $14\times13\times36$, 193-step levels. Thus
the approximately 3% global CV discrepancy is primarily coarse-operator
geometry/time error rather than an unconverged reference, while the reverse
peak remains the clearest target for improved curved-panel discretization.

The frozen metrics are in
[`results/curved_film_3d_fdm_validation.json`](../results/curved_film_3d_fdm_validation.json).
