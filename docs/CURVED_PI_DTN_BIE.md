# End-to-End Curved PI-DtN-BIE Solver

## 1. Coupled Problem

The solver advances a catalytic film bounded by a planar electrode and a
periodic curved graph \(\Gamma\). The curved graph is triangulated without an
axisymmetric reduction. On every panel,

$$
J_i(t)=k_i C_{B,i}(t)C_{C,i}(t),
\qquad C_{C,i}=\gamma-C_{D,i}.
$$

No FDM field, neural network, checkpoint, or fitted correction enters this
forward recurrence.

## 2. External Neumann Heat BIE

The physical reaction flux is not identified directly with an arbitrary
single-layer density. Instead, a density \(\lambda\) is obtained from the
Neumann boundary equation

$$
\left(\frac12 I-D_DK_\Gamma'\right)\lambda=J,
$$

and the external interface concentration is

$$
C_D|_\Gamma=V_\Gamma\lambda.
$$

Here \(V_\Gamma\) is the heat single layer and \(K_\Gamma'\) is its
target-normal derivative. Piecewise-linear time cells give the affine current
step

$$
\lambda_n=\lambda_n^{(0)}+L_nJ_n,
$$

$$
C_{D,n}=d_n^{(0)}+R_{D,n}J_n.
$$

The diagonal Abel singularity in \(V_\Gamma\) is integrated analytically; only
the regular geometry correction is passed to Gaussian time quadrature. For a
flat uniform surface, \(K_\Gamma'=0\), \(\lambda=2J\), and this equation
reduces to the original one-dimensional Abel ProductIntegral.

## 3. Finite-Film Two-Boundary DtN

The first complete film block assigns one vertical finite slab to every
projected triangle. If \(h_i\) is its local thickness, its half-integer modes
are

$$
\mu_{im}=\frac{(m+1/2)\pi}{h_i}.
$$

The reaction flux is defined per curved area. Conservation converts it to a
column flux per projected area,

$$
q_i=\frac{J_i}{n_{z,i}},
$$

because \(dA_{\mathrm{proj}}=n_{z,i}dA_\Gamma\). Exact modal propagation over
the current time cell produces

$$
C_{B,i,n}=b_{i,n}^{(0)}+(R_{B,n}J_n)_i.
$$

This is a genuine two-boundary finite-thickness DtN in the normal/vertical
direction and reproduces the planar finite-slab solver. It is currently local
between columns: tangential diffusion inside the film is not included. The
external BIE is fully surface-coupled.

An optional surface-modal research block promotes the normal-mode amplitudes
to P1 fields on the interface and couples them with a Laplace--Beltrami
operator. It exactly preserves the flat constant mode, but its current
variable-thickness implementation freezes derivatives of the local normal
basis. See [the dedicated ablation](SURFACE_MODAL_FILM_DTN.md).

## 4. Interface-Only Nonlinear Solve

Combining both affine traces gives

$$
\mathbf F(\mathbf J_n)
=\mathbf J_n
-\mathbf k\odot
\left(\mathbf b_n^{(0)}+R_{B,n}\mathbf J_n\right)
\odot
\left(\mathbf c_n^{(0)}-R_{D,n}\mathbf J_n\right)=0.
$$

A positivity-preserving line-searched Newton method solves this surface-vector
equation. The electrode current is reconstructed from the finite-film modal
gradient and integrated with projected triangle areas.

## 5. Verified Limits and Posterior

With 65 time points, a flat uniform periodic surface reproduces the mature 1D
mainline with:

- reaction-flux NRMSE: \(2.872\times10^{-4}\);
- CV NRMSE: \(2.820\times10^{-4}\).

For the spherical cap, the fair FDM posterior uses a length-3 external domain,
not the earlier length-0.5 box whose far boundary lies inside the diffusion
layer. The default 32-panel BIE versus an \(8\times7\times108\) monolithic FDM
gives:

- CV NRMSE: \(1.695\times10^{-2}\);
- total reaction flux per projected area NRMSE: \(1.634\times10^{-2}\);
- normalized external interface concentration NRMSE: \(1.033\times10^{-2}\);
- forward and reverse peak-current errors: 2.64% and 2.58%;
- BIE Neumann residual: \(2.66\times10^{-15}\);
- nonlinear closure residual: \(1.04\times10^{-10}\).

The voxel staircase has area 0.7156 while the smooth triangle surface has area
0.5895. Therefore the per-own-surface-area mean reaction flux is retained only
as a geometry-sensitive diagnostic. The physically comparable total flux per
projected electrode area is the primary metric. Under a prescribed equal-total
flux, the isolated external BIE trace differs from the long-domain FDM by
1.34%.

## 6. Honest Current Boundary

This branch is an end-to-end curved CV solver, but not yet the final fast 3D
method:

1. film tangential diffusion still requires a coupled two-surface film BIE or
   a geometrically complete surface-modal response; the first surface-modal
   ablation omits variable-basis and higher-order curvature terms;
2. direct matrix history costs \(O(N_t^2N_\Gamma^2)\);
3. periodic images must be replaced by an Ewald/FMM or periodic-kernel backend;
4. the voxel FDM should be supplemented by cut-cell FVM or FEM for local-field
   validation on the same smooth geometry;
5. operator-valued SOE or convolution quadrature is required before making a
   speed claim.

The result dictionaries explicitly record
`film_tangential_diffusion_included=false`, `volume_grid_used=false`, and
`fdm_data_used=false`.

## 7. Reproduction

```bash
python -m unittest -v test_surface_heat_bem.py test_curved_pi_dtn_bie.py
python run_curved_pi_dtn_bie.py
python validate_curved_pi_dtn_bie_fdm.py
```

Colab:

```bash
git clone --branch codex/full-curved-pi-dtn-bie \
  https://github.com/FanoShannon/pinn_test.git /content/curved_pi_dtn_bie

OUTPUT_DIR=/content/gdrive/MyDrive/curved_pi_dtn_bie \
bash /content/curved_pi_dtn_bie/run_colab_curved_pi_dtn_bie.sh
```
