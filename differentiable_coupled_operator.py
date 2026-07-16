import numpy as np
import torch

import pinn_thin_layer_v9_6 as pinn


def triangular_protocol(time):
    simulation_time = time[-1]
    switch_time = float(pinn.T_switch) * simulation_time
    theta = torch.where(
        time <= switch_time,
        float(pinn.theta_i)
        -2.0
        *float(pinn.theta_i - pinn.theta_switch)
        *time
        /simulation_time,
        float(pinn.theta_switch)
        +2.0
        *float(pinn.theta_i - pinn.theta_switch)
        *(time - switch_time)
        /simulation_time,
    )
    surface = torch.sigmoid(-theta)
    return theta, surface


def completed_product_integral(current, step_index, dt, diffusion):
    if step_index <= 1:
        return current[0] * 0.0

    values = torch.stack(current)
    j_left = values[:step_index - 1]
    j_right = values[1:step_index]
    delta_j = j_right - j_left
    cells = torch.arange(
        step_index - 1,
        dtype=values.dtype,
        device=values.device,
    )
    lag_hi = (float(step_index) - cells) * dt
    lag_lo = lag_hi - dt
    sqrt_hi = torch.sqrt(lag_hi)
    sqrt_lo = torch.sqrt(lag_lo)
    intercept = j_left + delta_j * lag_hi / dt
    cell_integral = (
        2.0 * intercept * (sqrt_hi - sqrt_lo)
        -(2.0 / 3.0)
        *(delta_j / dt)
        *(lag_hi ** 1.5 - lag_lo ** 1.5)
    )
    return torch.sum(cell_integral) / np.sqrt(np.pi * diffusion)


def reaction_closure_newton(
    b_intercept,
    b_slope,
    c_intercept,
    c_slope,
    k_cat,
    initial,
    iterations,
):
    value = initial
    for _ in range(iterations):
        b_value = b_intercept + b_slope * value
        c_value = c_intercept + c_slope * value
        residual = value - k_cat * b_value * c_value
        derivative = 1.0 - k_cat * (
            b_slope * c_value + c_slope * b_value
        )
        value = value - residual / derivative
    return value


def solve_coupled_operator(
    time,
    gamma,
    k_cat,
    delta,
    n_modes,
    newton_iterations=10,
):
    """Differentiable ProductIntegral-DtN recurrence for scalar parameters."""
    if not torch.is_tensor(time):
        time = torch.as_tensor(time, dtype=torch.float64)
    if not torch.is_tensor(delta):
        delta = torch.as_tensor(
            delta,
            dtype=time.dtype,
            device=time.device,
        )
    gamma = torch.as_tensor(gamma, dtype=time.dtype, device=time.device)
    k_cat = torch.as_tensor(k_cat, dtype=time.dtype, device=time.device)
    if time.ndim != 1 or len(time) < 3:
        raise ValueError("time must be a one-dimensional grid with >=3 points")
    if delta.ndim != 0 or gamma.ndim != 0 or k_cat.ndim != 0:
        raise ValueError("gamma, k_cat, and delta must be scalar tensors")
    if (
        not torch.isfinite(delta).item()
        or not torch.isfinite(gamma).item()
        or not torch.isfinite(k_cat).item()
        or delta.detach().item() <= 0.0
        or gamma.detach().item() <= 0.0
        or k_cat.detach().item() <= 0.0
    ):
        raise ValueError("gamma, k_cat, and delta must be finite and positive")

    dt_values = torch.diff(time)
    dt = dt_values[0]
    if not torch.allclose(dt_values, dt, rtol=1e-10, atol=1e-14):
        raise ValueError("The differentiable operator requires uniform time")

    diffusion_b = float(pinn.D_rel_B)
    diffusion_d = float(pinn.D_rel_D)
    theta, surface = triangular_protocol(time)
    mode_index = torch.arange(
        n_modes,
        dtype=time.dtype,
        device=time.device,
    )
    mu = (mode_index + 0.5) * np.pi / delta
    boundary_sign = torch.where(
        (torch.arange(n_modes, device=time.device) % 2) == 0,
        torch.ones(n_modes, dtype=time.dtype, device=time.device),
        -torch.ones(n_modes, dtype=time.dtype, device=time.device),
    )
    decay_rate = diffusion_b * mu ** 2
    decay = torch.exp(-decay_rate * dt)
    gain = -torch.expm1(-decay_rate * dt) / decay_rate
    current_cell_factor = 2.0 * torch.sqrt(
        dt / (np.pi * diffusion_d)
    )
    flux_response = (
        2.0
        *boundary_sign
        *gain
        /(dt * delta * diffusion_b * mu ** 2)
    )
    current_c_slope = -(2.0 / 3.0) * current_cell_factor

    amplitudes = [-2.0 * surface[0] / (delta * mu)]
    current = [torch.zeros((), dtype=time.dtype, device=time.device)]
    c_b_int = [torch.zeros((), dtype=time.dtype, device=time.device)]
    c_c_int = [gamma]
    c_d_int = [torch.zeros((), dtype=time.dtype, device=time.device)]

    for step in range(1, len(time)):
        completed = completed_product_integral(
            current,
            step,
            dt,
            diffusion_d,
        )
        previous_current = current[-1]
        surface_slope = (surface[step] - surface[step - 1]) / dt
        amplitude_intercept = (
            decay * amplitudes[-1]
            +gain * (-2.0 * surface_slope / (delta * mu))
            -flux_response * previous_current
        )
        b_intercept = (
            surface[step] + torch.sum(amplitude_intercept * boundary_sign)
        )
        b_slope = (
            -delta / diffusion_b
            +torch.sum(flux_response * boundary_sign)
        )
        c_intercept = (
            gamma
            -completed
            -(1.0 / 3.0) * current_cell_factor * previous_current
        )
        value = reaction_closure_newton(
            b_intercept,
            b_slope,
            c_intercept,
            current_c_slope,
            k_cat,
            previous_current,
            newton_iterations,
        )
        amplitude = amplitude_intercept + flux_response * value
        current.append(value)
        amplitudes.append(amplitude)
        c_b_int.append(b_intercept + b_slope * value)
        c_c_value = c_intercept + current_c_slope * value
        c_c_int.append(c_c_value)
        c_d_int.append(gamma - c_c_value)

    amplitudes = torch.stack(amplitudes)
    current = torch.stack(current)
    c_b_int = torch.stack(c_b_int)
    c_c_int = torch.stack(c_c_int)
    c_d_int = torch.stack(c_d_int)
    surface_current = (
        -current
        +diffusion_b * torch.sum(amplitudes * mu[None, :], dim=1)
    )
    surface_current = torch.cat((
        torch.zeros(
            1,
            dtype=surface_current.dtype,
            device=surface_current.device,
        ),
        surface_current[1:],
    ))
    inventory = (
        delta * surface
        -0.5 * delta ** 2 * current / diffusion_b
        +torch.sum(amplitudes / mu[None, :], dim=1)
    )
    inventory = torch.cat((
        torch.zeros(1, dtype=inventory.dtype, device=inventory.device),
        inventory[1:],
    ))
    return {
        "time": time,
        "theta": theta,
        "C_B_surface": surface,
        "C_B_int": c_b_int,
        "C_C_int": c_c_int,
        "C_D_int": c_d_int,
        "J_rxn": current,
        "J_surface": surface_current,
        "M_B": inventory,
        "amplitudes": amplitudes,
        "mu": mu,
    }
