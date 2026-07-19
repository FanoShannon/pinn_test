# Surface-Modal Film DtN Prototype

## 1. Purpose

The first curved solver assigns an independent finite vertical slab to every
surface panel. That construction preserves the mature one-dimensional normal
DtN, but omits tangential diffusion inside the film. This prototype promotes
the normal modal amplitudes to fields on the curved interface.

It is an ablation-quality thin-shell model, not yet an exact variable-thickness
three-dimensional film solver.

## 2. Surface-Modal Representation

For local normal coordinate \(\xi\in[0,h(\mathbf s)]\), the film is represented
as

$$
C_B(\mathbf s,\xi,t)
=g(t)-\frac{\xi}{D_B}q(\mathbf s,t)
+\sum_{m=0}^{M-1}a_m(\mathbf s,t)\sin(\mu_m\xi),
$$

with

$$
\mu_m(\mathbf s)=\frac{(m+1/2)\pi}{h(\mathbf s)}.
$$

Each amplitude is discretized with periodic P1 surface finite elements. With
mass-lumped surface mass \(M_\Gamma\) and stiffness \(K_\Gamma\), the prototype
advances

$$
\dot{\mathbf a}_m
=-D_B\left(M_\Gamma^{-1}K_\Gamma
+\mathrm{diag}(\mu_m^2)\right)\mathbf a_m
+\mathbf f_m(\dot g,\dot q).
$$

The matrix exponential and its integrated response are precomputed for every
normal mode. Projection from panel fluxes to nodes is conservative and exactly
preserves a constant field. Projection back to panel centroids produces a
non-diagonal current-cell film response,

$$
\mathbf C_{B,i,n}
=\mathbf b_n^{(0)}+R_{B,n}^{\mathrm{surface}}\mathbf J_n.
$$

The external BIE and nonlinear reaction closure are unchanged.

For spatially varying flux on a flat constant-thickness film, the lifting
term also contributes

$$
-\frac{2(-1)^m}{h\mu_m^2}\Delta_\Gamma q.
$$

The implementation integrates this forcing exactly for a piecewise-linear
time history. It is enabled only for constant thickness. For variable
thickness, adding this flat term without the associated basis-gradient and
curvature terms is inconsistent and can make the current-cell response
nonphysical.

## 3. Exact Flat Limit

For a flat film of constant thickness, let \(\lambda_\ell\) be an eigenvalue of
the discrete Laplace--Beltrami operator. The modal decay rate is

$$
\alpha_{m\ell}=D_B(\mu_m^2+\lambda_\ell).
$$

The constant surface mode has \(\lambda_0=0\), so it exactly recovers the
one-dimensional finite-slab DtN. Automated tests verify both the constant mode
and a nonzero tangential eigenmode.

## 4. Curved-Film Approximation Boundary

When \(h=h(\mathbf s)\), differentiating the normal basis also produces terms
containing \(\nabla_\Gamma h\), curvature, and coupling between normal modes.
The first curved prototype does not include those terms. It uses the local
thickness inside \(\mu_m\) while propagating amplitudes with the top-interface
Laplace--Beltrami operator. Result files therefore record:

- `film_variable_thickness_basis_derivatives_included=false`;
- `film_higher_order_curvature_terms_included=false`.

The implementation is best interpreted as a frozen-normal-basis thin-shell
DtN. A complete version requires either the missing geometric coupling matrices
or a two-boundary film heat BIE.

The later matched-geometry audit showed that this frozen curved model should
not be promoted as the general correction. Its flat lifting term improves a
flat heterogeneous prescribed-flux film trace only from 7.55% to 7.31%, and it
cannot consistently represent variable-thickness geometric coupling.

## 5. First Posterior Ablation

The default 32-panel spherical-cap problem was evaluated against the same
independent \(8\times7\times108\), length-3 voxel FDM posterior used by the
local-column method.

| Film model | CV NRMSE | Total-flux NRMSE | Mean \(C_{B,i}\) NRMSE |
|---|---:|---:|---:|
| Local column | 1.6955% | 1.6339% | 10.1749% |
| Surface modal, \(\alpha=0.1\) | 1.6913% | 1.6402% | 10.1662% |
| Surface modal, \(\alpha=0.25\) | 1.6949% | 1.6404% | 10.1693% |
| Surface modal, \(\alpha=1\) | 1.7072% | 1.6436% | 10.1787% |

Here \(\alpha\) multiplies the tangential stiffness for the ablation; the
physical prototype uses \(\alpha=1\). The surface-modal CV differs from the
local-column CV by only 0.21% NRMSE at \(\alpha=1\). Tangential modal coupling
therefore has a resolved but small effect for this geometry and parameter set.
It does not explain the approximately 10% mean film-interface discrepancy.

That discrepancy cannot be assigned to film physics alone because the
posterior compares a smooth triangular cap with a voxel staircase whose area is
21% larger. A smooth-geometry body-fitted FEM or cut-cell reference is required
before judging the accuracy of the new film block.

## 6. Reproduction

```bash
python -m unittest -v test_curved_pi_dtn_bie.py

python run_curved_pi_dtn_bie.py \
  --film-model surface_modal \
  --output runs_curved_heat_bem/surface_modal

python validate_curved_pi_dtn_bie_fdm.py \
  --film-model surface_modal \
  --output runs_curved_heat_bem/surface_modal_fdm

python analyze_surface_modal_film_dtn.py \
  --strengths 0,1
```

Additional strengths can be run in separate calls, for example
`--strengths 0.1` and `--strengths 0.25`.

FDM is read only by the posterior scripts and is not used to construct,
calibrate, or select the forward operator.
