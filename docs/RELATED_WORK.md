# Relationship to FDM and Prior Work

## 1. Positioning Statement

This repository does not claim that Abel convolution, ProductIntegration,
Dirichlet-to-Neumann maps, Green functions, SOE kernel compression, or
automatic differentiation were invented here. They are established ideas.

The methodological contribution under study is their structure-preserving
composition for a nonlinear finite-film/half-space electrochemical system:

- exact finite-slab modal transport;
- an analytically integrated singular current cell;
- a safeguarded low-dimensional nonlinear closure;
- fast causal memory that leaves the local closure unchanged;
- trace-preserving external-field reconstruction;
- end-to-end parameter derivatives and identifiability analysis.

The novelty claim must be assessed for this coupled construction and its
electrochemical consequences, not for each ingredient in isolation.

## 2. Conventional FDM

| Aspect | Conventional FDM | Current operator mainline |
|---|---|---|
| Spatial representation | Film and external spatial grids | No spatial grid; film modes and boundary histories |
| Time history | Local time stepping in the enlarged state | Explicit causal Abel memory |
| Interface reaction | Coupled to grid boundary unknowns | Scalar safeguarded closure per cell |
| External field | Solved at all external nodes | Reconstructed only when requested |
| Parameter gradients | Finite differences or differentiable discretization | Native float64 automatic differentiation |
| Geometry/general physics | Easier to extend to irregular geometry, migration, nonlinear transport | Currently planar, linear diffusion, local reaction |
| Role in this repository | Independent posterior reference only | Forward, sensitivity, design, and inverse engine |

The operator is not categorically superior to FDM. It is advantageous when the
linear layered diffusion structure applies and many repeated forward/gradient
evaluations are needed. FDM remains more flexible for multidimensional
geometry and additional nonlinear transport physics.

The posterior FDM and operator have independent spatial representations and
different discretization errors. Disagreement in a low-flux electrode gradient
can therefore be decomposed using the FDM thin-layer mass-balance current rather
than assigning all error to one method.

## 3. Grid-Free Cyclic Voltammetry

Coffman, Lu, and Subotnik introduced a grid-free sweep/CV algorithm based on a
Green-function solution coupled to an implicit ODE solver. Their paper
establishes that eliminating the spatial diffusion grid in voltammetry is not,
by itself, a new claim:

- A. J. Coffman, J. Lu, and J. E. Subotnik, "A Grid-Free Approach for
  Simulating Sweep and Cyclic Voltammetry," *J. Chem. Phys.* **154**, 161101
  (2021). [DOI: 10.1063/5.0044156](https://doi.org/10.1063/5.0044156)

The present research question is narrower and more structural: how to couple a
finite reactive film, a half-space Abel memory, and a nonlinear two-sided
interface while retaining a stable singular cell, fast history, concentration
reconstruction, and parameter gradients. A publication comparison should use
matched mechanisms and measured wall time before making a speed claim.

## 4. Differentiable Electrochemistry

Chen and co-workers frame differentiable electrochemistry as end-to-end
differentiable coupling of thermodynamics, kinetics, and transport for
gradient-based mechanistic inference across several electrochemical examples:

- H. Chen et al., "Differentiable Electrochemistry: A Paradigm Characterizing
  Physical Laws in Electrochemical Systems," *ACS Energy Lett.* **11**,
  2005-2018 (2026).
  [DOI: 10.1021/acsenergylett.5c03761](https://doi.org/10.1021/acsenergylett.5c03761)

The relationship is complementary:

| Differentiable Electrochemistry | Current operator mainline |
|---|---|
| Broad differentiable-programming paradigm and multiple applications | Deep structure exploitation for one layered reaction-diffusion class |
| Focus on connecting physical theories to parameter discovery | Focus on removing spatial grids and compressing singular causal memory |
| Demonstrates gradient-based inference over several mechanisms | Adds explicit ProductIntegral closure, DtN reduction, SOE acceleration, and regime-wise identifiability |

The defensible claim is that this solver can serve as a specialized,
structure-aware differentiable backend. It should not be presented as replacing
the broader paradigm.

## 5. ProductIntegration and SOE Literature

ProductIntegration for weakly singular Volterra equations is classical. One
relevant convergence study is:

- A. Orsi, "Product Integration for Volterra Integral Equations of the Second
  Kind with Weakly Singular Kernels," *Math. Comp.* **65**, 1201-1212 (1996).
  [DOI: 10.1090/S0025-5718-96-00736-3](https://doi.org/10.1090/S0025-5718-96-00736-3)

SOE acceleration of power-law histories is likewise established:

- S. Jiang, J. Zhang, Q. Zhang, and Z. Zhang, "Fast Evaluation of the Caputo
  Fractional Derivative and Its Applications to Fractional Diffusion
  Equations," *Commun. Comput. Phys.* **21**, 650-678 (2017).
  [DOI: 10.4208/cicp.OA-2016-0136](https://doi.org/10.4208/cicp.OA-2016-0136)
- Z. Gao, J. Liang, and Z. Xu, "A Kernel-Independent Sum-of-Exponentials
  Method," *J. Sci. Comput.* **93**, 40 (2022).
  [DOI: 10.1007/s10915-022-01999-1](https://doi.org/10.1007/s10915-022-01999-1)

Our SOE backend is a practical near-direct/far-SOE construction specialized to
the Abel ProductIntegral history. A paper should include its own approximation
and propagation error analysis rather than relying only on empirical agreement.

## 6. JCTC Fit

JCTC states that it welcomes advances in theory, methodology, data science,
and innovative computational chemistry packages, while excluding
straightforward single-class applications of established methods:

- [JCTC Aims and Scope](https://pubs.acs.org/page/jctcce/about.html)

For that reason, a JCTC manuscript should lead with the general causal operator
factorization and validate at least one additional electrochemical mechanism.
A single catalytic CV application, however accurate, is a weaker fit than a
transferable computational methodology with convergence, gradient, and
identifiability evidence.

