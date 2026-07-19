# Matched-Geometry Error Audit

## 1. Question

The original curved posterior compared a smooth triangular PI-DtN-BIE solver
with a voxel-staircase FDM. Its 1.70% CV discrepancy and 10.17% mean film-trace
discrepancy mixed operator error, geometry mismatch, and reference
discretization. This audit separates those effects without changing or fitting
the main operator.

## 2. Independent Smooth Reference

The posterior reference uses periodic layered tetrahedra. Its film/external
interface is exactly the same piecewise-planar triangular graph used by the
BIE. The film is mapped between the planar electrode and the graph; the
external mesh is mapped between the graph and a length-3 far plane.

P1 finite elements give the backward-Euler system

$$
\left(\frac{M}{\Delta t}+DK\right)\mathbf u_n
=\frac{M}{\Delta t}\mathbf u_{n-1}+B\mathbf J_n.
$$

Eliminating volume unknowns gives the same interface contract as the main
solver,

$$
\mathbf c_{i,n}=\mathbf a_n+\mathbf B_n\mathbf J_n.
$$

Thus film and external responses can be compared before closing the nonlinear
reaction. The FEM imports no BIE matrices, ProductIntegral weights, FDM files,
network outputs, or fitted corrections.

## 3. Reference Qualification

- Flat uniform 33-step FEM versus mature 1D: reaction flux 0.237%, CV 0.214%.
- Normal-layer refinement from 9/36 to 18/72: CV change 0.0022%.
- Time refinement from 65 to 129 points: CV change 0.292%, flux change 0.160%.
- FEM nonlinear closure residual: below \(1.21\times10^{-10}\).

The remaining matched comparison includes backward-Euler temporal error and
the finite length-3 external boundary, but no smooth/voxel interface mismatch.

## 4. Geometry and Heterogeneity Factorial

| Geometry | Kinetics | CV NRMSE | Local flux RMS | Prescribed film trace | Prescribed external trace |
|---|---|---:|---:|---:|---:|
| Flat | Uniform | 0.114% | 0.182% | 0.073% | 0.982% |
| Curved | Uniform | 0.181% | 3.613% | 4.726% | 1.889% |
| Flat | Heterogeneous | 0.623% | 8.116% | 7.553% | 1.764% |
| Curved | Heterogeneous | 0.814% | 10.530% | 10.179% | 2.351% |

Curvature/variable thickness and heterogeneous kinetics independently activate
spatial film error. Their combination is the most demanding regime. The
integrated CV remains much less sensitive than the local panel field.

## 5. Nonlinear Feedback Decomposition

For

$$
\mathbf F(\mathbf J)
=\mathbf J-\mathbf k\odot\mathbf C_B\odot\mathbf C_C,
$$

the frozen-history perturbation is

$$
\delta\mathbf J
\simeq A^{-1}\left[
\mathrm{diag}(\mathbf k\odot\mathbf C_C)\delta\mathbf C_B
+\mathrm{diag}(\mathbf k\odot\mathbf C_B)\delta\mathbf C_C
\right],
$$

where \(A=\partial\mathbf F/\partial\mathbf J\). In the curved heterogeneous
case:

- film-trace contribution / flux RMS: 6.97%;
- external-trace contribution / flux RMS: 0.49%;
- combined one-step contribution / flux RMS: 7.20%;
- closure condition number: mean 1.60, maximum 1.70.

The closure is not close to singular. It moderately amplifies transport error;
it is not the source of the discrepancy.

## 6. Surface Spectral Interpretation

Raw P0 face-wise errors also contain representation mismatch against the
continuous P1 FEM trace. Projection onto common Laplace--Beltrami modes gives:

- film trace: 3.60% over all projected modes, 0.19% in the constant mode;
- reaction flux: 4.80% over all projected modes, 1.15% in the constant mode;
- 24% of projected film error energy lies above the first four surface modes.

The main operator therefore predicts mean inventory and integrated CV much
more accurately than local nonuniform surface patterns.

## 7. Conclusion

The original voxel posterior overstated mean film-trace error: 10.17% becomes
0.123% when the geometry is matched. The CV discrepancy falls from 1.70% to
0.814%. A real structural limitation remains in local spatial fields, driven
by curvature/variable thickness together with heterogeneous reaction.

This does not justify an empirical curved-film correction. The production
local-column operator remains appropriate for integrated CV in the tested
regime. Applications requiring local surface maps need a geometry-complete
film transport block satisfying the same affine interface contract, such as a
generic FEM Schur complement or a fully derived curved-shell DtN.

## 8. Reproduction

```bash
python -m unittest -v test_matched_geometry_fem.py

python analyze_matched_geometry_error.py \
  --output runs_curved_heat_bem/matched_geometry_audit

# Factorial controls
python analyze_matched_geometry_error.py --cap-height 0 \
  --heterogeneity-x 0 --heterogeneity-xy 0
python analyze_matched_geometry_error.py --cap-height 0
python analyze_matched_geometry_error.py \
  --heterogeneity-x 0 --heterogeneity-xy 0
```

All comparisons are posterior-only.

