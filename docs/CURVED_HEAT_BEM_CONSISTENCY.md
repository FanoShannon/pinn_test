# Abel-Consistent Curved Heat BEM

## Purpose

The one-dimensional ProductIntegral model and a triangular surface method must
be two discretizations of the same diffusion theory. Replacing a line grid by
triangles is not, by itself, a physical generalization. The required link is
the flat, translation-invariant limit: the three-dimensional surface history
operator must reduce to the Abel history used by the planar model.

This prototype establishes that link for the **external diffusion single-layer
operator**. It does not yet claim a complete curved-film electrochemical
solver.

## Common Surface Operator

For diffusivity \(D\), the free-space heat kernel is

$$
G_3(\mathbf{x},t)
=\frac{\exp[-|\mathbf{x}|^2/(4Dt)]}{(4\pi Dt)^{3/2}},
\qquad t>0.
$$

The half-space image factor gives the causal surface single layer

$$
(V_\Gamma j)(\mathbf{x},t)
=2\int_0^t\int_\Gamma
G_3(\mathbf{x}-\mathbf{y},t-\tau)
j(\mathbf{y},\tau)\,dS_{\mathbf{y}}\,d\tau.
$$

On a flat infinite plane with spatially uniform flux, integration over the two
tangential coordinates is exact:

$$
2\int_{\mathbb{R}^2}G_3((\boldsymbol{\rho},0),u)
\,d\boldsymbol{\rho}
=\frac{1}{\sqrt{\pi D u}}.
$$

Consequently,

$$
(V_\Gamma j)(t)
=\frac{1}{\sqrt{\pi D}}
\int_0^t\frac{j(\tau)}{\sqrt{t-\tau}}\,d\tau,
$$

which is exactly the Abel operator already discretized by ProductIntegration
in the one-dimensional mainline. Thus the 1D method is the zero tangential
wavenumber, flat-interface member of the surface theory.

## Abel Self Split

A direct time quadrature of a triangle's self interaction is inaccurate near
\(u=t-\tau=0\). The implementation writes

$$
V_\Gamma=V_{\mathrm{Abel}}+V_{\mathrm{regular}}.
$$

For triangle \(T_i\), the universal singular part is

$$
K_{ii}^{\mathrm{sing}}(u)=\frac{1}{\sqrt{\pi D u}},
$$

and it is integrated in time by the same piecewise-linear ProductIntegral as
the 1D solver. The finite-panel correction uses an equal-area disk
\(\pi a_i^2=A_i\):

$$
K_{ii}^{\mathrm{disk}}(u)
=\frac{1}{\sqrt{\pi D u}}
\left[1-\exp\left(-\frac{A_i}{4\pi D u}\right)\right].
$$

Therefore the regular diagonal contribution is approximated by

$$
K_{ii}^{\mathrm{regular}}(u)
=-\frac{1}{\sqrt{\pi D u}}
\exp\left(-\frac{A_i}{4\pi D u}\right).
$$

Off-diagonal triangle integrals use a symmetric three-point area quadrature.
The regular history is integrated with Gauss-Legendre quadrature on each time
cell. This split makes the singular limit exact while leaving geometry in a
smooth correction.

## What the Prototype Tests

1. A periodic triangulated plane with uniform flux is compared with the exact
   1D Abel ProductIntegral trace.
2. The same connectivity is used for a spherical cap and for its flattened
   disk, isolating the geometry-dependent correction.
3. A non-axisymmetric flux is applied so the result cannot collapse to a
   radial one-dimensional calculation.
4. Both flat and curved meshes are refined to expose spatial discretization
   error.

The frozen default audit gives:

- flat 128-panel NRMSE versus 1D Abel: \(3.270\times10^{-3}\);
- flat maximum relative discrepancy: \(5.180\times10^{-3}\);
- curved 80-panel mean-trace error versus 140 panels: \(5.391\times10^{-3}\);
- curved-versus-flat geometry correction: \(3.823\times10^{-2}\);
- non-axisymmetric trace index: \(2.507\times10^{-1}\).

The flat error decreases under mesh refinement, while curvature produces a
larger, resolved signal. This is evidence of operator consistency, not a full
electrochemical validation.

## Exact Scope and Missing Blocks

The implemented object is \(V_\Gamma j\), not the complete boundary integral
equation. On a general curved boundary, a heat representation normally leads
to a trace equation containing both single- and double-layer terms, schematically

$$
\left(\frac{1}{2}I+K_\Gamma\right)c_\Gamma
=V_\Gamma j_\Gamma+\text{initial-data contribution}.
$$

The following blocks remain before claiming a complete 3D
ProductIntegral-DtN-SOE solver:

1. stable time-domain assembly of the full heat boundary-integral equation;
2. a two-surface thin-film DtN block coupling the electrode and curved
   film/external interface;
3. the nonlinear catalytic closure and current reconstruction on every panel;
4. SOE or convolution-quadrature acceleration of all regular matrix histories;
5. posterior comparison with an independently refined three-dimensional FDM.

FDM data are not read by this consistency experiment and must remain excluded
from forward construction, training, and inversion objectives.

## Reproduction

```bash
python -m unittest -v test_surface_heat_bem.py
python analyze_surface_heat_bem_consistency.py \
  --output runs_curved_heat_bem/consistency
```

The JSON output records `full_heat_bie_solved=false` and
`film_dtn_included=false` so downstream reports cannot silently promote this
prototype into a stronger claim.

## Numerical Context

The full-BIE roadmap follows established time-domain heat boundary-integral
and convolution-quadrature theory rather than treating the prototype split as
a replacement for that theory:

- T. Qiu, A. Rieder, F.-J. Sayas, and S. Zhang, *Time-domain boundary integral
  equation modeling of heat transmission problems*, Numerische Mathematik
  143 (2019), 223-259,
  https://doi.org/10.1007/s00211-019-01040-y.
- C. Lubich, *Convolution quadrature and discretized operational calculus*,
  Numerische Mathematik 52 (1988), 129-145,
  https://doi.org/10.1007/BF01398686.
- J. Tausch, *A fast method for solving the heat equation by layer
  potentials*, Journal of Computational Physics 224 (2007), 956-969,
  https://doi.org/10.1016/j.jcp.2006.11.001.
