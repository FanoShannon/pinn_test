"""Dimensionless physical constants and voltage protocols for the mainline solver."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransportParameters:
    """Constant diffusivities used by the current conservation reduction."""

    d_a: float = 1.0
    d_b: float = 1.0
    d_c: float = 1.0
    d_d: float = 1.0

    def validate_reduction(self):
        if not np.isclose(self.d_a, self.d_b, rtol=0.0, atol=1e-14):
            raise NotImplementedError("C_A = 1 - C_B requires D_A = D_B")
        if not np.isclose(self.d_c, self.d_d, rtol=0.0, atol=1e-14):
            raise NotImplementedError("C_C = gamma - C_D requires D_C = D_D")


THETA_INITIAL = 10.0
THETA_SWITCH = -10.0
SWITCH_FRACTION = 0.5
DEFAULT_SCAN_RATE = 40.0
DEFAULT_DELTA = 0.035
EXTERNAL_LENGTH_FACTOR = 6.0
TRANSPORT = TransportParameters()


def protocol_duration(scan_rate=DEFAULT_SCAN_RATE):
    scan_rate = float(scan_rate)
    if not np.isfinite(scan_rate) or scan_rate <= 0.0:
        raise ValueError("scan_rate must be finite and positive")
    return 2.0 * abs(THETA_INITIAL - THETA_SWITCH) / scan_rate


DEFAULT_SIMULATION_TIME = protocol_duration()


def triangular_protocol_numpy(time):
    """Return theta(t) and the reversible Nernst surface concentration C_B."""
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or time[-1] <= 0.0:
        raise ValueError("time must be a one-dimensional positive-duration grid")
    switch_time = SWITCH_FRACTION * float(time[-1])
    theta = np.where(
        time <= switch_time,
        THETA_INITIAL
        -2.0 * (THETA_INITIAL - THETA_SWITCH) * time / time[-1],
        THETA_SWITCH
        +2.0
        * (THETA_INITIAL - THETA_SWITCH)
        * (time - switch_time)
        / time[-1],
    )
    return theta, 1.0 / (1.0 + np.exp(theta))


def characteristic_reaction_flux(gamma, k_cat, delta, diffusion_b=None):
    """Film-limited reference scale k*gamma/(1+k*gamma*delta/D_B)."""
    diffusion_b = TRANSPORT.d_b if diffusion_b is None else float(diffusion_b)
    validate_positive_parameters(gamma=gamma, k_cat=k_cat, delta=delta)
    return float(k_cat) * float(gamma) / (
        1.0 + float(k_cat) * float(gamma) * float(delta) / diffusion_b
    )


def validate_positive_parameters(**parameters):
    for name, value in parameters.items():
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
