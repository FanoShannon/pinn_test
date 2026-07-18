# Research and Publication Roadmap

## Current Scientific Claim

The present code supports the following bounded claim:

> A nonlinear planar finite-film/half-space reaction-diffusion model can be
> reduced to a stable, spatial-grid-free, causal and differentiable operator.
> It provides accurate positive-parameter forward solutions and enables
> regime-aware multi-scan inversion without neural-network or FDM training.

It does not yet support a universal electrochemistry-solver claim.

## Phase 1: Publication Hardening

1. Derive a temporal/modal/SOE error decomposition.
2. Add adaptive time and mode selection using current and trace tolerances.
3. Benchmark matched accuracy, wall time, peak memory, and gradient cost
   against FDM and a grid-free CV implementation.
4. Repeat noisy inversion over many random noise realizations and report bias,
   variance, profile likelihoods, and confidence intervals.
5. Freeze machine-readable reference outputs and continuous-integration tests.

Exit criterion: every headline numerical claim is reproducible from one command
and accompanied by a resolution/error audit.

## Phase 2: Demonstrate Transferability

### Priority A: finite electron-transfer kinetics

Replace the exact Nernst surface state with Butler-Volmer or
Marcus-Hush-Chidsey kinetics. The electrode flux becomes an additional unknown,
turning the scalar current-cell closure into a small coupled nonlinear system.

Scientific value: demonstrates that the method is not tied to a reversible
surface boundary and connects directly to modern differentiable
electrochemistry.

### Priority B: break the conservation reduction

Allow $D_A\ne D_B$ and/or $D_C\ne D_D$ by propagating separate modal/history
states.

Scientific value: demonstrates that exact pairwise conservation is an
optimization, not a hidden requirement of the operator concept.

### Priority C: second local reaction law

Add a pseudo-first-order EC-prime limit or another chemically justified local
rate law through the generic closure $\mathbf R(\mathbf c_i;\mathbf p)$.

Scientific value: isolates reusable transport operators from mechanism-specific
kinetics.

Exit criterion: at least two mechanisms share the same transport/memory code
and pass independent posterior validation.

## Phase 3: Experimental Inference

1. Select an experimental system whose planar diffusion and film assumptions
   are defensible.
2. Calibrate instrument nuisance parameters separately from chemical
   parameters.
3. Acquire or use several widely separated scan rates.
4. Perform sensitivity-qualified inversion and posterior predictive checks.
5. Report failure when the data lie in weak-reaction or saturated regimes.

An experimental case is not required to establish the numerical algorithm, but
it substantially strengthens chemical relevance and prevents a purely
self-consistent synthetic inverse study.

## Phase 4: Broader Operators

Later extensions include finite external domains, cylindrical/spherical DtN
maps, adsorption states, migration/Poisson coupling, and multidimensional
corner corrections. Each requires a new transport operator or a hybrid domain
decomposition; none should be implied by the current planar code.

The first multidimensional scaffold is documented in
[`THREE_DIMENSIONAL_EXTENSION.md`](THREE_DIMENSIONAL_EXTENSION.md). It uses a
planar electrode, a positive-thickness curved film, full Cartesian fields, and
non-axisymmetric interface kinetics. Its sparse-volume DtN condensation is a
verification stage. The publication-grade target replaces those responses by
curved heat-layer boundary operators with singular ProductIntegration and
operator-valued fast history.

## Recommended JCTC Narrative

1. General low-dimensional causal operator factorization.
2. Stable ProductIntegral-DtN-SOE algorithm and convergence.
3. End-to-end gradients and experiment design.
4. Two or more electrochemical mechanisms.
5. Independent FDM and matched-method benchmarks.
6. Interior success and asymptotic non-identifiability map.

A suitable working title is:

> Differentiable Causal Operators for Grid-Free Simulation and Inversion of
> Multiscale Voltammetry
