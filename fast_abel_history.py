from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class SoeHistoryPlan:
    """Near-exact, sum-of-exponentials plan for a uniform Abel history."""

    n_steps: int
    near_cells: int
    n_terms: int
    tolerance: float
    near_weights: np.ndarray
    decay: np.ndarray
    tail_amplitudes: np.ndarray


def _positive_float(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _freeze(values):
    values = np.asarray(values, dtype=np.float64)
    values.setflags(write=False)
    return values


def _dimensionless_pi_weights(count):
    """Completed-history nodal weights before sqrt(dt / (pi D))."""
    lag = np.arange(1, count + 1, dtype=np.float64)
    ascending = (
        2.0 * (lag + 1.0) * (np.sqrt(lag + 1.0) - np.sqrt(lag))
        -(2.0 / 3.0) * ((lag + 1.0) ** 1.5 - lag ** 1.5)
    )
    weights = ascending.copy()
    if count > 1:
        interior_lag = lag[1:]
        descending = (
            (2.0 / 3.0)
            *(
                interior_lag ** 1.5
                -(interior_lag - 1.0) ** 1.5
            )
            -2.0
            *(interior_lag - 1.0)
            *(
                np.sqrt(interior_lag)
                -np.sqrt(interior_lag - 1.0)
            )
        )
        weights[1:] += descending
    return weights


@lru_cache(maxsize=64)
def build_soe_history_plan(
    n_steps,
    dt,
    diffusion,
    near_cells=16,
    n_terms=128,
    tolerance=1e-10,
):
    """Build positive SOE weights for the completed ProductIntegral tail.

    The singular current cell remains outside this plan and is integrated
    exactly by the coupled nonlinear closure.
    """
    n_steps = int(n_steps)
    near_cells = int(near_cells)
    n_terms = int(n_terms)
    dt = _positive_float("dt", dt)
    diffusion = _positive_float("diffusion", diffusion)
    tolerance = _positive_float("tolerance", tolerance)
    if n_steps < 3:
        raise ValueError("n_steps must be at least three")
    if near_cells < 1:
        raise ValueError("near_cells must be positive")
    if n_terms < 8:
        raise ValueError("n_terms must be at least eight")
    if tolerance >= 0.1:
        raise ValueError("tolerance must be less than 0.1")

    effective_near = min(near_cells, n_steps - 2)
    scale = np.sqrt(dt / (np.pi * diffusion))
    near_weights = scale * _dimensionless_pi_weights(effective_near)

    max_lag = max(n_steps - 2, effective_near + 1)
    log_inverse_tolerance = np.log(1.0 / tolerance)
    lower_rate = (
        (tolerance * np.sqrt(np.pi) / 4.0) ** 2 / max_lag
    )
    upper_rate = (
        log_inverse_tolerance
        +2.0 * np.log(log_inverse_tolerance)
    ) / effective_near

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(n_terms)
    log_lower = np.log(lower_rate)
    log_upper = np.log(upper_rate)
    log_rates = (
        0.5 * (log_upper - log_lower) * nodes
        +0.5 * (log_upper + log_lower)
    )
    rates = np.exp(log_rates)
    laplace_weights = (
        0.5
        *(log_upper - log_lower)
        *quadrature_weights
        *np.exp(0.5 * log_rates)
        /np.sqrt(np.pi)
    )
    decay = np.exp(-rates)

    # A linear nodal basis is a convolution of two unit boxes. Its Laplace
    # transform is ((1 - exp(-rate)) / rate)^2.
    basis_transform = (-np.expm1(-rates) / rates) ** 2
    tail_amplitudes = (
        scale
        *laplace_weights
        *basis_transform
        *np.exp(-rates * effective_near)
    )
    return SoeHistoryPlan(
        n_steps=n_steps,
        near_cells=effective_near,
        n_terms=n_terms,
        tolerance=tolerance,
        near_weights=_freeze(near_weights),
        decay=_freeze(decay),
        tail_amplitudes=_freeze(tail_amplitudes),
    )


class NumpySoeHistory:
    """Online SOE state for one causal ProductIntegral trajectory."""

    def __init__(self, plan):
        self.plan = plan
        self.tail_state = np.zeros(plan.n_terms, dtype=np.float64)

    def completed(self, current, step_index):
        if step_index <= 1:
            return 0.0
        count = min(self.plan.near_cells, step_index - 1)
        lag = np.arange(1, count + 1)
        near = np.dot(
            self.plan.near_weights[:count],
            np.asarray(current)[step_index - lag],
        )
        tail = np.dot(self.plan.tail_amplitudes, self.tail_state)
        return float(near + tail)

    def advance(self, current, step_index):
        if step_index >= self.plan.near_cells:
            entering = current[step_index - self.plan.near_cells]
            self.tail_state = self.plan.decay * self.tail_state + entering
