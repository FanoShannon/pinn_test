import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import physical_model as physics
import posterior_metrics
import soe_abel_history


def triangular_protocol(time):
    return physics.triangular_protocol_numpy(time)


def characteristic_reaction_flux(gamma, k_cat, delta, diffusion_b):
    return physics.characteristic_reaction_flux(
        gamma,
        k_cat,
        delta,
        diffusion_b,
    )


def validate_positive_parameters(gamma, k_cat, delta):
    physics.validate_positive_parameters(
        gamma=gamma,
        k_cat=k_cat,
        delta=delta,
    )


def validate_fdm_parameters(fdm, gamma, k_cat, delta):
    parameters = fdm.get("params", fdm.get("parameters", {}))
    expected = {
        "gamma": float(gamma),
        "k_cat": float(k_cat),
        "delta": float(delta),
        "D_A": physics.TRANSPORT.d_a,
        "D_B": physics.TRANSPORT.d_b,
        "D_C": physics.TRANSPORT.d_c,
        "D_D": physics.TRANSPORT.d_d,
    }
    aliases = {
        "k_cat": ("k_cat", "k_cat_star"),
        "gamma": ("gamma",),
        "delta": ("delta",),
        "D_A": ("D_A",),
        "D_B": ("D_B",),
        "D_C": ("D_C",),
        "D_D": ("D_D",),
    }
    for name, target in expected.items():
        stored = None
        for key in aliases[name]:
            if key in parameters:
                stored = float(parameters[key])
                break
        if stored is None:
            continue
        if not np.isclose(stored, target, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"FDM parameter mismatch for {name}: "
                f"file={stored}, operator={target}"
            )


def completed_product_integral(current, step_index, dt, diffusion):
    """Completed-cell Abel integral at t[step_index]."""
    if step_index <= 1:
        return 0.0

    j_left = current[:step_index - 1]
    j_right = current[1:step_index]
    delta_j = j_right - j_left
    cells = np.arange(step_index - 1, dtype=np.float64)
    lag_hi = (float(step_index) - cells) * dt
    lag_lo = lag_hi - dt
    sqrt_hi = np.sqrt(lag_hi)
    sqrt_lo = np.sqrt(lag_lo)
    intercept = j_left + delta_j * lag_hi / dt
    cell_integral = (
        2.0 * intercept * (sqrt_hi - sqrt_lo)
        -(2.0 / 3.0) * (delta_j / dt)
        * (lag_hi ** 1.5 - lag_lo ** 1.5)
    )
    return float(np.sum(cell_integral) / np.sqrt(np.pi * diffusion))


def reaction_closure_root(
    b_intercept,
    b_slope,
    c_intercept,
    c_slope,
    k_cat,
    initial,
    max_iterations,
):
    """Solve J = k (b0 + b1 J) (c0 + c1 J) with safeguarded Newton."""
    if b_intercept <= 0.0 or c_intercept <= 0.0:
        return 0.0, 0, 0.0

    upper_candidates = []
    if b_slope < 0.0:
        upper_candidates.append(b_intercept / -b_slope)
    if c_slope < 0.0:
        upper_candidates.append(c_intercept / -c_slope)
    if upper_candidates:
        upper = max(min(upper_candidates), 1e-15)
    else:
        upper = max(
            2.0 * k_cat * b_intercept * c_intercept,
            2.0 * initial,
            1.0,
        )

    def residual(value):
        b_value = b_intercept + b_slope * value
        c_value = c_intercept + c_slope * value
        return value - k_cat * b_value * c_value

    lower = 0.0
    f_upper = residual(upper)
    expansion_count = 0
    while f_upper < 0.0 and not upper_candidates and expansion_count < 20:
        upper *= 2.0
        f_upper = residual(upper)
        expansion_count += 1
    if f_upper < 0.0:
        raise RuntimeError("Could not bracket the coupled reaction root")

    value = float(np.clip(initial, lower, upper))
    if value <= lower or value >= upper:
        value = 0.5 * (lower + upper)

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        b_value = b_intercept + b_slope * value
        c_value = c_intercept + c_slope * value
        f_value = value - k_cat * b_value * c_value
        if abs(f_value) <= 1e-13 * max(1.0, abs(value)):
            break

        if f_value > 0.0:
            upper = value
        else:
            lower = value

        derivative = 1.0 - k_cat * (
            b_slope * c_value + c_slope * b_value
        )
        if np.isfinite(derivative) and abs(derivative) > 1e-14:
            candidate = value - f_value / derivative
        else:
            candidate = np.nan
        if not np.isfinite(candidate) or candidate <= lower or candidate >= upper:
            candidate = 0.5 * (lower + upper)
        value = float(candidate)

    return value, iterations, float(abs(residual(value)))


def solve_coupled_operator(
    time,
    gamma,
    k_cat,
    n_modes,
    newton_iterations,
    delta=None,
    history_backend="direct",
    history_near_cells=16,
    history_soe_terms=128,
    history_soe_tolerance=1e-10,
):
    """Couple finite-slab DtN and external Abel ProductIntegral cell by cell."""
    delta = float(physics.DEFAULT_DELTA if delta is None else delta)
    validate_positive_parameters(gamma, k_cat, delta)
    physics.TRANSPORT.validate_reduction()
    time = np.asarray(time, dtype=np.float64)
    if len(time) < 3:
        raise ValueError("At least three time points are required")
    dt_values = np.diff(time)
    dt = float(dt_values[0])
    if not np.allclose(dt_values, dt, rtol=1e-10, atol=1e-14):
        raise ValueError("The coupled operator requires a uniform time grid")

    diffusion_b = physics.TRANSPORT.d_b
    diffusion_d = physics.TRANSPORT.d_d
    if history_backend not in ("direct", "soe"):
        raise ValueError("history_backend must be 'direct' or 'soe'")
    soe_history = None
    if history_backend == "soe":
        soe_plan = soe_abel_history.build_soe_history_plan(
            len(time),
            dt,
            diffusion_d,
            near_cells=history_near_cells,
            n_terms=history_soe_terms,
            tolerance=history_soe_tolerance,
        )
        soe_history = soe_abel_history.NumpySoeHistory(soe_plan)
    thickness = delta
    theta, surface = triangular_protocol(time)

    mode_index = np.arange(n_modes, dtype=np.float64)
    mu = (mode_index + 0.5) * np.pi / thickness
    boundary_sign = (-1.0) ** mode_index
    decay_rate = diffusion_b * mu ** 2
    decay = np.exp(-decay_rate * dt)
    gain = -np.expm1(-decay_rate * dt) / decay_rate
    current_cell_factor = 2.0 * np.sqrt(dt / (np.pi * diffusion_d))

    amplitudes = np.empty((len(time), n_modes), dtype=np.float64)
    current = np.zeros(len(time), dtype=np.float64)
    c_d_int = np.zeros(len(time), dtype=np.float64)
    c_c_int = np.empty(len(time), dtype=np.float64)
    c_b_int = np.empty(len(time), dtype=np.float64)
    root_residual = np.zeros(len(time), dtype=np.float64)
    root_iterations = np.zeros(len(time), dtype=np.int64)

    c_c_int[0] = gamma
    amplitudes[0] = -2.0 * surface[0] / (thickness * mu)
    c_b_int[0] = 0.0
    current[0] = 0.0

    flux_response = (
        2.0 * boundary_sign * gain
        /(dt * thickness * diffusion_b * mu ** 2)
    )
    current_c_slope = -(2.0 / 3.0) * current_cell_factor

    for step in range(1, len(time)):
        if soe_history is None:
            completed = completed_product_integral(
                current,
                step,
                dt,
                diffusion_d,
            )
        else:
            completed = soe_history.completed(current, step)
        previous_current = current[step - 1]
        surface_slope = (surface[step] - surface[step - 1]) / dt
        amplitude_intercept = (
            decay * amplitudes[step - 1]
            +gain * (-2.0 * surface_slope / (thickness * mu))
            -flux_response * previous_current
        )

        b_intercept = surface[step] + amplitude_intercept @ boundary_sign
        b_slope = (
            -thickness / diffusion_b
            +flux_response @ boundary_sign
        )
        c_intercept = (
            gamma - completed
            -(1.0 / 3.0) * current_cell_factor * previous_current
        )

        value, iterations, residual = reaction_closure_root(
            b_intercept,
            b_slope,
            c_intercept,
            current_c_slope,
            k_cat,
            previous_current,
            newton_iterations,
        )
        current[step] = value
        root_iterations[step] = iterations
        root_residual[step] = residual
        amplitudes[step] = amplitude_intercept + flux_response * value
        c_b_int[step] = b_intercept + b_slope * value
        c_c_int[step] = c_intercept + current_c_slope * value
        c_d_int[step] = gamma - c_c_int[step]
        if soe_history is not None:
            soe_history.advance(current, step)

    return {
        "gamma": float(gamma),
        "k_cat": float(k_cat),
        "delta": thickness,
        "D_B": diffusion_b,
        "D_D": diffusion_d,
        "history_backend": history_backend,
        "history_near_cells": (
            None if soe_history is None else soe_history.plan.near_cells
        ),
        "history_soe_terms": (
            None if soe_history is None else soe_history.plan.n_terms
        ),
        "history_soe_tolerance": (
            None if soe_history is None else soe_history.plan.tolerance
        ),
        "external_length": float(
            physics.EXTERNAL_LENGTH_FACTOR * np.sqrt(float(time[-1]))
        ),
        "time": time,
        "theta": theta,
        "C_B_surface": surface,
        "C_B_int": c_b_int,
        "C_C_int": c_c_int,
        "C_D_int": c_d_int,
        "J_rxn": current,
        "mu": mu,
        "boundary_sign": boundary_sign,
        "amplitudes": amplitudes,
        "root_residual": root_residual,
        "root_iterations": root_iterations,
    }


def reconstruct_state(history, x_eval):
    diffusion = float(history["D_B"])
    thickness = float(history["delta"])
    x_eval = np.asarray(x_eval, dtype=np.float64)
    if np.min(x_eval) < -1e-12 or np.max(x_eval) > thickness + 1e-12:
        raise ValueError(
            f"Thin evaluation coordinates must lie in [0, {thickness}]"
        )
    basis = np.sin(np.outer(x_eval, history["mu"]))
    field = (
        history["C_B_surface"][:, None]
        -x_eval[None, :] * history["J_rxn"][:, None] / diffusion
        +history["amplitudes"] @ basis.T
    )
    surface_current = (
        -history["J_rxn"]
        +diffusion * (
            history["amplitudes"]
            @ history["mu"]
        )
    )
    inventory = (
        thickness * history["C_B_surface"]
        -0.5 * thickness ** 2 * history["J_rxn"] / diffusion
        +history["amplitudes"] @ (1.0 / history["mu"])
    )
    field[0, :] = 0.0
    surface_current[0] = 0.0
    inventory[0] = 0.0
    inventory_dt_centered = np.gradient(
        inventory,
        history["time"],
        edge_order=2,
    )
    centered_inventory_current = (
        -history["J_rxn"] - inventory_dt_centered
    )
    centered_inventory_current[0] = 0.0
    inventory_dt_backward = np.zeros_like(inventory)
    inventory_dt_backward[1:] = (
        inventory[1:] - inventory[:-1]
    ) / np.diff(history["time"])
    backward_inventory_current = (
        -history["J_rxn"] - inventory_dt_backward
    )
    backward_inventory_current[0] = 0.0
    return {
        "C_B": field,
        "J_surface": surface_current,
        "M_B": inventory,
        "J_inventory_centered": centered_inventory_current,
        "J_inventory_backward": backward_inventory_current,
    }


def interpolate_time_series(source_time, values, target_time):
    source_time = np.asarray(source_time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(target_time, source_time, values)
    flat = values.reshape(values.shape[0], -1)
    interpolated = np.column_stack([
        np.interp(target_time, source_time, flat[:, column])
        for column in range(flat.shape[1])
    ])
    return interpolated.reshape((len(target_time),) + values.shape[1:])


def fdm_thin_balance_current(fdm):
    """Recover electrode current from the FDM thin-layer mass balance."""
    time = np.asarray(fdm["t"], dtype=np.float64)
    x_in = np.asarray(fdm["x_in"], dtype=np.float64)
    c_b = np.asarray(
        fdm["concentrations"]["C_B"],
        dtype=np.float64,
    )
    reaction = np.asarray(fdm["R"], dtype=np.float64)
    inventory = np.trapezoid(c_b, x_in, axis=1)
    inventory_dt = np.gradient(inventory, time, edge_order=2)
    current = -reaction - inventory_dt
    current[0] = 0.0
    return current


def tracegreen_external_field(
    history,
    time_eval,
    x_eval,
    kernel_points,
    batch_size,
    quadrature="gauss-legendre",
):
    """Propagate the solved interface trace with the erfc heat potential."""
    time_eval = np.asarray(time_eval, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    tt, xx = np.meshgrid(time_eval, x_eval, indexing="ij")
    flat_time = tt.reshape(-1)
    flat_y = np.clip(
        xx.reshape(-1) - float(history["delta"]),
        0.0,
        float(history["external_length"]),
    )
    history_grid = torch.as_tensor(
        history["C_D_int"],
        dtype=torch.float64,
    )
    simulation_time = float(history["time"][-1])
    if quadrature == "midpoint":
        nodes = (
            torch.arange(kernel_points, dtype=torch.float64) + 0.5
        ) / float(kernel_points)
        weights = torch.full(
            (kernel_points,),
            1.0 / float(kernel_points),
            dtype=torch.float64,
        )
    elif quadrature == "gauss-legendre":
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(kernel_points)
        nodes = torch.from_numpy(0.5 * (raw_nodes + 1.0))
        weights = torch.from_numpy(0.5 * raw_weights)
    else:
        raise ValueError(
            "quadrature must be 'gauss-legendre' or 'midpoint'"
        )
    diffusion = float(history["D_D"])
    ext_length = float(history["external_length"])
    output = np.empty_like(flat_time)

    def interpolate_history(tau):
        tau = torch.clamp(tau, 0.0, simulation_time)
        coordinate = tau * (len(history_grid) - 1) / simulation_time
        left = torch.floor(coordinate).to(torch.long)
        left = torch.clamp(left, 0, len(history_grid) - 2)
        right = left + 1
        fraction = coordinate - left.to(coordinate.dtype)
        return history_grid[left] + fraction * (
            history_grid[right] - history_grid[left]
        )

    def heat_trace(t_value, y_value):
        t_positive = torch.clamp(
            t_value,
            min=1e-6 * simulation_time,
        )
        trace_mass = torch.erfc(
            y_value / torch.sqrt(4.0 * diffusion * t_positive)
        )
        u = torch.clamp(
            trace_mass[:, None] * nodes[None, :],
            min=1e-12,
            max=1.0 - 1e-12,
        )
        eta = torch.erfinv(1.0 - u)
        delay = (
            y_value[:, None] ** 2
            /(4.0 * diffusion * torch.clamp(eta ** 2, min=1e-16))
        )
        tau = torch.clamp(
            t_value[:, None] - delay,
            0.0,
            simulation_time,
        )
        trace = trace_mass * torch.sum(
            interpolate_history(tau) * weights[None, :],
            dim=1,
        )
        interface = interpolate_history(t_value)
        return torch.where(y_value <= 1e-12, interface, trace)

    for start in range(0, len(flat_time), batch_size):
        stop = min(start + batch_size, len(flat_time))
        t_batch = torch.from_numpy(flat_time[start:stop])
        y_batch = torch.from_numpy(flat_y[start:stop])
        trace = heat_trace(t_batch, y_batch)
        far_trace = heat_trace(
            t_batch,
            torch.full_like(y_batch, ext_length),
        )
        r = torch.clamp(y_batch / ext_length, 0.0, 1.0)
        h01 = -2.0 * r ** 3 + 3.0 * r ** 2
        corrected = trace - h01 * far_trace
        output[start:stop] = corrected.numpy()

    c_d = output.reshape(len(time_eval), len(x_eval))
    c_c = float(history["gamma"]) - c_d
    return c_c, c_d


def evaluate(
    fdm,
    history,
    n_time,
    n_x,
    n_x_out,
    cv_points,
    green_kernel_points,
    trace_batch_size,
    green_quadrature="gauss-legendre",
):
    time = history["time"]
    fdm_time = np.asarray(fdm["t"], dtype=np.float64)
    x_fdm = np.asarray(fdm["x_in"], dtype=np.float64)
    x_out_fdm = np.asarray(fdm["x_out"], dtype=np.float64)
    time_indices = posterior_metrics.select_indices(len(time), n_time)
    x_indices = posterior_metrics.select_indices(len(x_fdm), n_x)
    x_out_indices = posterior_metrics.select_indices(len(x_out_fdm), n_x_out)
    cv_indices = posterior_metrics.select_indices(len(time), cv_points)
    x_eval = x_fdm[x_indices]
    x_out_eval = x_out_fdm[x_out_indices]
    state = reconstruct_state(history, x_eval)

    field_times = time[time_indices]
    current_times = time[cv_indices]
    fdm_b = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["concentrations"]["C_B"], dtype=np.float64)[:, x_indices],
        field_times,
    )
    fdm_b_int = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["concentrations"]["C_B"], dtype=np.float64)[:, -1],
        field_times,
    )
    fdm_c_c_int = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["concentrations"]["C_C"], dtype=np.float64)[:, 0],
        field_times,
    )
    fdm_current = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["J"], dtype=np.float64),
        current_times,
    )
    fdm_balance_current = interpolate_time_series(
        fdm_time,
        fdm_thin_balance_current(fdm),
        current_times,
    )
    fdm_reaction = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["R"], dtype=np.float64),
        field_times,
    )
    fdm_c = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["concentrations"]["C_C"], dtype=np.float64)[
            :, x_out_indices
        ],
        field_times,
    )
    fdm_d = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["concentrations"]["C_D"], dtype=np.float64)[
            :, x_out_indices
        ],
        field_times,
    )
    predicted_c, predicted_d = tracegreen_external_field(
        history,
        field_times,
        x_out_eval,
        green_kernel_points,
        trace_batch_size,
        quadrature=green_quadrature,
    )

    predicted_current = state["J_surface"][cv_indices].copy()
    predicted_centered = state["J_inventory_centered"][cv_indices].copy()
    predicted_backward = state["J_inventory_backward"][cv_indices].copy()
    predicted_current[0] = 0.0
    predicted_centered[0] = 0.0
    predicted_backward[0] = 0.0
    gamma = float(history["gamma"])
    k_cat = float(history["k_cat"])
    j_ref = characteristic_reaction_flux(
        gamma,
        k_cat,
        history["delta"],
        history["D_B"],
    )

    metrics = {
        "C_A": posterior_metrics.field_metrics(1.0 - state["C_B"][time_indices], 1.0 - fdm_b),
        "C_B": posterior_metrics.field_metrics(state["C_B"][time_indices], fdm_b),
        "C_B_int": posterior_metrics.field_metrics(
            history["C_B_int"][time_indices],
            fdm_b_int,
        ),
        "C_C_int": posterior_metrics.field_metrics(
            history["C_C_int"][time_indices],
            fdm_c_c_int,
        ),
        "C_C_int_over_gamma": posterior_metrics.field_metrics(
            history["C_C_int"][time_indices] / gamma,
            fdm_c_c_int / gamma,
        ),
        "C_C": posterior_metrics.field_metrics(predicted_c, fdm_c),
        "C_D": posterior_metrics.field_metrics(predicted_d, fdm_d),
        "C_C_over_gamma": posterior_metrics.field_metrics(
            predicted_c / gamma,
            fdm_c / gamma,
        ),
        "C_D_over_gamma": posterior_metrics.field_metrics(
            predicted_d / gamma,
            fdm_d / gamma,
        ),
        "J_rxn": posterior_metrics.field_metrics(
            history["J_rxn"][time_indices],
            fdm_reaction,
        ),
        "CV_J_surface": posterior_metrics.field_metrics(
            predicted_current,
            fdm_current,
        ),
        "CV_J_surface_vs_FDM_thin_balance": posterior_metrics.field_metrics(
            predicted_current,
            fdm_balance_current,
        ),
        "CV_J_inventory_centered": posterior_metrics.field_metrics(
            predicted_centered,
            fdm_current,
        ),
        "CV_J_inventory_centered_vs_FDM_thin_balance": (
            posterior_metrics.field_metrics(
                predicted_centered,
                fdm_balance_current,
            )
        ),
        "CV_J_inventory_backward": posterior_metrics.field_metrics(
            predicted_backward,
            fdm_current,
        ),
        "CV_J_surface_over_J_ref": posterior_metrics.field_metrics(
            predicted_current / j_ref,
            fdm_current / j_ref,
        ),
        "CV_J_surface_vs_FDM_thin_balance_over_J_ref": (
            posterior_metrics.field_metrics(
                predicted_current / j_ref,
                fdm_balance_current / j_ref,
            )
        ),
        "CV_J_inventory_centered_over_J_ref": posterior_metrics.field_metrics(
            predicted_centered / j_ref,
            fdm_current / j_ref,
        ),
        "CV_J_inventory_backward_over_J_ref": posterior_metrics.field_metrics(
            predicted_backward / j_ref,
            fdm_current / j_ref,
        ),
        "surface_vs_centered_inventory": posterior_metrics.field_metrics(
            predicted_current / j_ref,
            predicted_centered / j_ref,
        ),
        "surface_vs_backward_inventory": posterior_metrics.field_metrics(
            predicted_current / j_ref,
            predicted_backward / j_ref,
        ),
        "FDM_surface_vs_thin_balance": posterior_metrics.field_metrics(
            fdm_current,
            fdm_balance_current,
        ),
        "FDM_surface_vs_thin_balance_over_J_ref": (
            posterior_metrics.field_metrics(
                fdm_current / j_ref,
                fdm_balance_current / j_ref,
            )
        ),
        "reaction_closure_max_abs": float(np.max(np.abs(
            history["J_rxn"]
            -k_cat
            *history["C_B_int"]
            *history["C_C_int"]
        ))),
        "root_residual_max_abs": float(np.max(history["root_residual"])),
        "root_iterations_max": int(np.max(history["root_iterations"])),
        "C_B_min": float(np.min(state["C_B"])),
        "C_B_max": float(np.max(state["C_B"])),
        "C_C_int_min": float(np.min(history["C_C_int"])),
        "C_C_int_max": float(np.max(history["C_C_int"])),
        "overall_dimensionless": {
            "rmse": float(np.sqrt(
                (
                    np.sum((state["C_B"][time_indices] - fdm_b) ** 2)
                    +np.sum(
                        (
                            (1.0 - state["C_B"][time_indices])
                            -(1.0 - fdm_b)
                        ) ** 2
                    )
                    +np.sum(((predicted_c - fdm_c) / gamma) ** 2)
                    +np.sum(((predicted_d - fdm_d) / gamma) ** 2)
                )
                /(
                    2 * state["C_B"][time_indices].size
                    +predicted_c.size
                    +predicted_d.size
                )
            ))
        },
        "_state": state,
        "_external_C_C": predicted_c,
        "_external_C_D": predicted_d,
        "_time_indices": time_indices,
        "_x_indices": x_indices,
        "_x_out_indices": x_out_indices,
        "_cv_indices": cv_indices,
        "_fdm_balance_current": fdm_balance_current,
    }
    return metrics


def serializable_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("_")
    }


def plot_diagnostics(output_dir, fdm, history, metrics):
    state = metrics["_state"]
    time_indices = metrics["_time_indices"]
    x_indices = metrics["_x_indices"]
    cv_indices = metrics["_cv_indices"]
    time = history["time"]
    fdm_time = np.asarray(fdm["t"], dtype=np.float64)
    x_eval = np.asarray(fdm["x_in"], dtype=np.float64)[x_indices]
    fdm_b = interpolate_time_series(
        fdm_time,
        np.asarray(fdm["concentrations"]["C_B"], dtype=np.float64)[:, x_indices],
        time[time_indices],
    )
    residual = state["C_B"][time_indices] - fdm_b

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    vmax = max(float(np.percentile(np.abs(residual), 99.0)), 1e-8)
    image = axes[0, 0].imshow(
        residual.T,
        origin="lower",
        aspect="auto",
        extent=[
            time[time_indices[0]],
            time[time_indices[-1]],
            x_eval[0],
            x_eval[-1],
        ],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 0].set_title("Coupled ProductIntegral-DtN C_B residual")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("x")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.85)

    axes[0, 1].plot(
        time,
        history["C_B_int"],
        label="coupled physical operator",
    )
    axes[0, 1].plot(
        time,
        interpolate_time_series(
            fdm_time,
            np.asarray(fdm["concentrations"]["C_B"])[:, -1],
            time,
        ),
        label="FDM posterior",
        alpha=0.8,
    )
    axes[0, 1].set_title("Thin interface concentration")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        history["theta"][cv_indices],
        interpolate_time_series(
            fdm_time,
            np.asarray(fdm["J"]),
            time[cv_indices],
        ),
        label="FDM",
        linewidth=2,
    )
    axes[1, 0].plot(
        history["theta"][cv_indices],
        metrics["_fdm_balance_current"],
        label="FDM thin-balance current",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[1, 0].plot(
        history["theta"][cv_indices],
        state["J_surface"][cv_indices],
        label="coupled DtN surface",
        linestyle="--",
    )
    axes[1, 0].plot(
        history["theta"][cv_indices],
        state["J_inventory_backward"][cv_indices],
        label="causal backward-inventory",
        linestyle=":",
    )
    axes[1, 0].set_title("CV current")
    axes[1, 0].set_xlabel("theta")
    axes[1, 0].set_ylabel("J")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(
        time,
        history["J_rxn"]
        -float(history["k_cat"])
        *history["C_B_int"]
        *history["C_C_int"],
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title("Reaction closure residual")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].grid(alpha=0.3)

    figure_path = output_dir / "coupled_productintegral_dtn.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def parse_mode_counts(text):
    values = sorted({
        int(item.strip())
        for item in text.split(",")
        if item.strip()
    })
    if not values or values[0] < 1:
        raise ValueError("mode counts must be positive integers")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Zero-training coupled ProductIntegral-finite-slab DtN solver. "
            "FDM is used only for posterior metrics."
        )
    )
    parser.add_argument("--fdm-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=physics.DEFAULT_DELTA)
    parser.add_argument("--operator-time-grid", type=int, default=256)
    parser.add_argument("--mode-counts", default="32,64,128,256")
    parser.add_argument("--newton-iterations", type=int, default=12)
    parser.add_argument(
        "--history-backend",
        choices=("direct", "soe"),
        default="direct",
    )
    parser.add_argument("--history-near-cells", type=int, default=16)
    parser.add_argument("--history-soe-terms", type=int, default=128)
    parser.add_argument("--history-soe-tolerance", type=float, default=1e-10)
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=120)
    parser.add_argument("--cv-points", type=int, default=2000)
    parser.add_argument("--green-kernel-points", type=int, default=64)
    parser.add_argument(
        "--green-quadrature",
        choices=("gauss-legendre", "midpoint"),
        default="gauss-legendre",
    )
    parser.add_argument("--trace-batch-size", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    args.fdm_pkl = args.fdm_pkl.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_positive_parameters(args.gamma, args.k_cat, args.delta)

    with args.fdm_pkl.open("rb") as handle:
        fdm = pickle.load(handle)
    validate_fdm_parameters(fdm, args.gamma, args.k_cat, args.delta)
    if args.operator_time_grid < 3:
        raise ValueError("operator-time-grid must be at least 3")
    time = np.linspace(
        0.0,
        physics.DEFAULT_SIMULATION_TIME,
        args.operator_time_grid,
    )

    all_metrics = {}
    retained_history = None
    retained_metrics = None
    for n_modes in parse_mode_counts(args.mode_counts):
        history = solve_coupled_operator(
            time,
            args.gamma,
            args.k_cat,
            n_modes,
            args.newton_iterations,
            delta=args.delta,
            history_backend=args.history_backend,
            history_near_cells=args.history_near_cells,
            history_soe_terms=args.history_soe_terms,
            history_soe_tolerance=args.history_soe_tolerance,
        )
        metrics = evaluate(
            fdm,
            history,
            args.n_time,
            args.n_x,
            args.n_x_out,
            args.cv_points,
            args.green_kernel_points,
            args.trace_batch_size,
            args.green_quadrature,
        )
        all_metrics[str(n_modes)] = serializable_metrics(metrics)
        retained_history = history
        retained_metrics = metrics
        print(
            f"modes={n_modes:4d} "
            f"C_B={metrics['C_B']['rmse']:.6e} "
            f"C_B_int={metrics['C_B_int']['rmse']:.6e} "
            f"C_C_int/gamma={metrics['C_C_int_over_gamma']['rmse']:.6e} "
            f"overall={metrics['overall_dimensionless']['rmse']:.6e} "
            f"CVsurf/Jref={metrics['CV_J_surface_over_J_ref']['rmse']:.6e} "
            f"CVback/Jref={metrics['CV_J_inventory_backward_over_J_ref']['rmse']:.6e} "
            f"CVcenter/Jref={metrics['CV_J_inventory_centered_over_J_ref']['rmse']:.6e} "
            f"closure={metrics['reaction_closure_max_abs']:.3e}"
        )

    report = {
        "solver": "coupled_productintegral_finite_slab_dtn",
        "training_steps": 0,
        "fdm_role": "posterior_only",
        "spatial_grid_in_operator": False,
        "operator_time_grid": int(args.operator_time_grid),
        "history_backend": args.history_backend,
        "history_near_cells": int(args.history_near_cells),
        "history_soe_terms": int(args.history_soe_terms),
        "history_soe_tolerance": float(args.history_soe_tolerance),
        "green_kernel_points": int(args.green_kernel_points),
        "green_quadrature": args.green_quadrature,
        "parameters": {
            "gamma": float(args.gamma),
            "k_cat": float(args.k_cat),
            "delta": float(args.delta),
            "D_B": physics.TRANSPORT.d_b,
            "D_D": physics.TRANSPORT.d_d,
        },
        "mode_results": all_metrics,
    }
    report_path = args.output_dir / "coupled_productintegral_dtn_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure_path = plot_diagnostics(
        args.output_dir,
        fdm,
        retained_history,
        retained_metrics,
    )
    np.savez_compressed(
        args.output_dir / "coupled_productintegral_dtn_state.npz",
        time=retained_history["time"],
        theta=retained_history["theta"],
        C_B_surface=retained_history["C_B_surface"],
        C_B_int=retained_history["C_B_int"],
        C_C_int=retained_history["C_C_int"],
        C_D_int=retained_history["C_D_int"],
        J_rxn=retained_history["J_rxn"],
        J_surface=retained_metrics["_state"]["J_surface"],
        J_inventory_backward=retained_metrics["_state"]["J_inventory_backward"],
        J_inventory_centered=retained_metrics["_state"]["J_inventory_centered"],
    )
    print(f"Saved metrics: {report_path}")
    print(f"Saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
