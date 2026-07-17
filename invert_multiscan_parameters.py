import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import differentiable_coupled_operator as differentiable
import pinn_thin_layer_v9_6 as pinn
import prototype_coupled_productintegral_dtn as numpy_operator


PARAMETER_NAMES = ("k_cat", "gamma", "delta")
FREE_PARAMETERS = {
    "all": PARAMETER_NAMES,
    "fix-gamma": ("k_cat", "delta"),
    "fix-k": ("gamma", "delta"),
    "fix-delta": ("k_cat", "gamma"),
    "k-only": ("k_cat",),
    "gamma-only": ("gamma",),
    "delta-only": ("delta",),
}


@dataclass
class ScanCurve:
    sigma: float
    time: np.ndarray
    clean_current: np.ndarray
    observed_current: np.ndarray
    current_span: float


def parse_positive_values(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one positive value is required")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("All values must be finite and positive")
    return values


def parse_modes(text):
    modes = [item.strip() for item in text.split(",") if item.strip()]
    if not modes:
        raise ValueError("At least one inverse mode is required")
    invalid = [mode for mode in modes if mode not in FREE_PARAMETERS]
    if invalid:
        raise ValueError(f"Unknown inverse modes: {invalid}")
    return modes


def scan_duration(sigma):
    return 2.0 * abs(float(pinn.theta_i - pinn.theta_switch)) / float(sigma)


def parameter_bounds(args):
    return {
        "k_cat": (float(args.k_min), float(args.k_max)),
        "gamma": (float(args.gamma_min), float(args.gamma_max)),
        "delta": (float(args.delta_min), float(args.delta_max)),
    }


def validate_bounds(bounds):
    for name, (lower, upper) in bounds.items():
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or lower >= upper
        ):
            raise ValueError(f"Invalid positive bounds for {name}: {lower}, {upper}")


def encode_start(start, free_names, bounds, dtype=torch.float64, device="cpu"):
    raw_values = []
    for name in free_names:
        value = float(start[name])
        lower, upper = bounds[name]
        if not lower < value < upper:
            raise ValueError(
                f"Initial {name}={value:g} must lie strictly inside "
                f"({lower:g}, {upper:g})"
            )
        fraction = (
            (np.log(value) - np.log(lower))
            /(np.log(upper) - np.log(lower))
        )
        fraction = float(np.clip(fraction, 1e-10, 1.0 - 1e-10))
        raw_values.append(np.log(fraction / (1.0 - fraction)))
    return torch.tensor(
        raw_values,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )


def decode_parameters(raw, free_names, fixed_values, bounds):
    values = {
        name: torch.as_tensor(
            fixed_values[name],
            dtype=raw.dtype,
            device=raw.device,
        )
        for name in PARAMETER_NAMES
    }
    for index, name in enumerate(free_names):
        lower, upper = bounds[name]
        log_value = (
            np.log(lower)
            +(np.log(upper) - np.log(lower)) * torch.sigmoid(raw[index])
        )
        values[name] = torch.exp(log_value)
    return values


def parse_explicit_starts(text):
    if not text:
        return []
    starts = []
    for item in text.split(";"):
        fields = [float(value.strip()) for value in item.split(":")]
        if len(fields) != 3:
            raise ValueError(
                "Each explicit start must be k:gamma:delta; "
                f"got {item!r}"
            )
        starts.append(dict(zip(PARAMETER_NAMES, fields)))
    return starts


def generate_starts(bounds, count, seed, explicit_starts=None):
    explicit_starts = list(explicit_starts or [])
    if count < len(explicit_starts):
        raise ValueError("n-starts cannot be smaller than explicit starts")
    starts = explicit_starts
    if not starts and count:
        starts.append({
            name: float(np.sqrt(lower * upper))
            for name, (lower, upper) in bounds.items()
        })
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        fractions = rng.uniform(0.12, 0.88, size=len(PARAMETER_NAMES))
        start = {}
        for fraction, name in zip(fractions, PARAMETER_NAMES):
            lower, upper = bounds[name]
            start[name] = float(np.exp(
                np.log(lower)
                +fraction * (np.log(upper) - np.log(lower))
            ))
        starts.append(start)
    return starts


def high_resolution_curve(
    sigma,
    k_cat,
    gamma,
    delta,
    target_time_grid,
    target_modes,
    observation_points,
):
    duration = scan_duration(sigma)
    target_time = np.linspace(0.0, duration, target_time_grid)
    history = numpy_operator.solve_coupled_operator(
        target_time,
        gamma=gamma,
        k_cat=k_cat,
        delta=delta,
        n_modes=target_modes,
        newton_iterations=16,
        history_backend="direct",
    )
    state = numpy_operator.reconstruct_state(
        history,
        np.array([0.0, delta]),
    )
    observation_time = np.linspace(0.0, duration, observation_points)
    current = np.interp(
        observation_time,
        target_time,
        state["J_surface"],
    )
    return observation_time, current


def make_scan_curves(
    sigmas,
    true_parameters,
    target_time_grid,
    target_modes,
    observation_points,
):
    curves = []
    for sigma in sigmas:
        time, clean = high_resolution_curve(
            sigma,
            true_parameters["k_cat"],
            true_parameters["gamma"],
            true_parameters["delta"],
            target_time_grid,
            target_modes,
            observation_points,
        )
        span = max(float(np.ptp(clean)), 1e-12)
        curves.append(ScanCurve(
            sigma=float(sigma),
            time=time,
            clean_current=clean,
            observed_current=clean.copy(),
            current_span=span,
        ))
    return curves


def add_noise(clean_curves, noise_level, rng):
    curves = []
    for clean_curve in clean_curves:
        observed = clean_curve.clean_current.copy()
        if noise_level:
            observed += rng.normal(
                0.0,
                noise_level * clean_curve.current_span,
                size=len(observed),
            )
            observed[0] = clean_curve.clean_current[0]
        curves.append(ScanCurve(
            sigma=clean_curve.sigma,
            time=clean_curve.time,
            clean_current=clean_curve.clean_current,
            observed_current=observed,
            current_span=clean_curve.current_span,
        ))
    return curves


def torch_curves(curves, device):
    result = []
    for curve in curves:
        result.append({
            "sigma": curve.sigma,
            "time": torch.as_tensor(
                curve.time,
                dtype=torch.float64,
                device=device,
            ),
            "observed": torch.as_tensor(
                curve.observed_current,
                dtype=torch.float64,
                device=device,
            ),
            "span": curve.current_span,
        })
    return result


def multiscan_loss(
    raw,
    free_names,
    fixed_values,
    bounds,
    observations,
    inverse_modes,
    newton_iterations,
    history_backend,
    history_near_cells,
    history_soe_terms,
    history_soe_tolerance,
):
    parameters = decode_parameters(raw, free_names, fixed_values, bounds)
    losses = []
    for observation in observations:
        state = differentiable.solve_coupled_operator(
            observation["time"],
            gamma=parameters["gamma"],
            k_cat=parameters["k_cat"],
            delta=parameters["delta"],
            n_modes=inverse_modes,
            newton_iterations=newton_iterations,
            history_backend=history_backend,
            history_near_cells=history_near_cells,
            history_soe_terms=history_soe_terms,
            history_soe_tolerance=history_soe_tolerance,
        )
        residual = (
            state["J_surface"][1:] - observation["observed"][1:]
        ) / observation["span"]
        losses.append(torch.mean(residual ** 2))
    return torch.stack(losses).mean(), parameters


def detached_parameters(parameters):
    return {
        name: float(value.detach().cpu())
        for name, value in parameters.items()
    }


def invert_one_start(
    mode,
    start,
    true_parameters,
    fixed_values,
    bounds,
    observations,
    inverse_modes,
    newton_iterations,
    lbfgs_iterations,
    history_backend,
    history_near_cells,
    history_soe_terms,
    history_soe_tolerance,
    device,
):
    free_names = FREE_PARAMETERS[mode]
    effective_start = dict(start)
    for name in PARAMETER_NAMES:
        if name not in free_names:
            effective_start[name] = float(fixed_values[name])
    raw = encode_start(effective_start, free_names, bounds, device=device)
    optimizer = torch.optim.LBFGS(
        [raw],
        lr=0.8,
        max_iter=lbfgs_iterations,
        max_eval=2 * lbfgs_iterations,
        tolerance_grad=1e-10,
        tolerance_change=1e-13,
        history_size=15,
        line_search_fn="strong_wolfe",
    )
    trajectory = []

    def closure():
        optimizer.zero_grad()
        loss, parameters = multiscan_loss(
            raw,
            free_names,
            fixed_values,
            bounds,
            observations,
            inverse_modes,
            newton_iterations,
            history_backend,
            history_near_cells,
            history_soe_terms,
            history_soe_tolerance,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite multi-scan objective")
        loss.backward()
        trajectory.append({
            "evaluation": len(trajectory),
            "loss": float(loss.detach().cpu()),
            "raw_gradient_norm": float(torch.linalg.vector_norm(raw.grad).cpu()),
            "parameters": detached_parameters(parameters),
        })
        return loss

    failure = None
    try:
        optimizer.step(closure)
    except (FloatingPointError, RuntimeError) as error:
        failure = f"{type(error).__name__}: {error}"

    with torch.no_grad():
        final_loss, final_parameters = multiscan_loss(
            raw,
            free_names,
            fixed_values,
            bounds,
            observations,
            inverse_modes,
            newton_iterations,
            history_backend,
            history_near_cells,
            history_soe_terms,
            history_soe_tolerance,
        )
    estimated = detached_parameters(final_parameters)
    relative_errors = {
        name: abs(estimated[name] - true_parameters[name]) / true_parameters[name]
        for name in PARAMETER_NAMES
    }
    log_errors = {
        name: abs(np.log(estimated[name] / true_parameters[name]))
        for name in PARAMETER_NAMES
    }
    per_scan = {}
    with torch.no_grad():
        for observation in observations:
            state = differentiable.solve_coupled_operator(
                observation["time"],
                gamma=final_parameters["gamma"],
                k_cat=final_parameters["k_cat"],
                delta=final_parameters["delta"],
                n_modes=inverse_modes,
                newton_iterations=newton_iterations,
                history_backend=history_backend,
                history_near_cells=history_near_cells,
                history_soe_terms=history_soe_terms,
                history_soe_tolerance=history_soe_tolerance,
            )
            residual = (
                state["J_surface"][1:] - observation["observed"][1:]
            ).detach().cpu().numpy()
            per_scan[f"{observation['sigma']:.12g}"] = {
                "current_rmse": float(np.sqrt(np.mean(residual ** 2))),
                "current_span_nrmse": float(
                    np.sqrt(np.mean(residual ** 2)) / observation["span"]
                ),
            }
    return {
        "mode": mode,
        "free_parameters": list(free_names),
        "fixed_parameters": {
            name: float(fixed_values[name])
            for name in PARAMETER_NAMES
            if name not in free_names
        },
        "initial_parameters": {
            name: float(effective_start[name]) for name in PARAMETER_NAMES
        },
        "estimated_parameters": estimated,
        "relative_errors": relative_errors,
        "absolute_log_errors": log_errors,
        "max_free_relative_error": float(max(
            relative_errors[name] for name in free_names
        )),
        "objective": float(final_loss.detach().cpu()),
        "function_evaluations": len(trajectory),
        "failure": failure,
        "per_scan": per_scan,
        "trajectory": trajectory,
    }


def fixed_values_for_mode(mode, true_parameters, args):
    fixed_values = dict(true_parameters)
    overrides = {
        "k_cat": args.fixed_k,
        "gamma": args.fixed_gamma,
        "delta": args.fixed_delta,
    }
    free_names = FREE_PARAMETERS[mode]
    for name, override in overrides.items():
        if name not in free_names and override is not None:
            fixed_values[name] = float(override)
    return fixed_values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FDM-free multi-scan inversion of k_cat, gamma, and delta with "
            "the differentiable ProductIntegral-DtN operator."
        )
    )
    parser.add_argument("--true-k", type=float, default=1.0)
    parser.add_argument("--true-gamma", type=float, default=10.0)
    parser.add_argument("--true-delta", type=float, default=0.035)
    parser.add_argument("--sigmas", default="5,40,320")
    parser.add_argument(
        "--inverse-modes",
        default="all,fix-gamma,fix-k,fix-delta",
    )
    parser.add_argument("--fixed-k", type=float, default=None)
    parser.add_argument("--fixed-gamma", type=float, default=None)
    parser.add_argument("--fixed-delta", type=float, default=None)
    parser.add_argument("--k-min", type=float, default=0.01)
    parser.add_argument("--k-max", type=float, default=100.0)
    parser.add_argument("--gamma-min", type=float, default=0.1)
    parser.add_argument("--gamma-max", type=float, default=100.0)
    parser.add_argument("--delta-min", type=float, default=0.005)
    parser.add_argument("--delta-max", type=float, default=0.2)
    parser.add_argument("--noise-levels", default="0,0.001")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument(
        "--starts",
        default=None,
        help="Optional semicolon-separated k:gamma:delta starts.",
    )
    parser.add_argument("--target-time-grid", type=int, default=2049)
    parser.add_argument("--target-modes", type=int, default=256)
    parser.add_argument("--observation-points", type=int, default=257)
    parser.add_argument("--operator-modes", type=int, default=96)
    parser.add_argument("--newton-iterations", type=int, default=10)
    parser.add_argument("--lbfgs-iterations", type=int, default=36)
    parser.add_argument(
        "--history-backend",
        choices=("direct", "soe"),
        default="soe",
    )
    parser.add_argument("--history-near-cells", type=int, default=16)
    parser.add_argument("--history-soe-terms", type=int, default=128)
    parser.add_argument("--history-soe-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sigmas = parse_positive_values(args.sigmas)
    modes = parse_modes(args.inverse_modes)
    noise_levels = [
        float(value.strip())
        for value in args.noise_levels.split(",")
        if value.strip()
    ]
    if any(not np.isfinite(level) or level < 0.0 for level in noise_levels):
        raise ValueError("Noise levels must be finite and nonnegative")
    if args.n_starts < 1:
        raise ValueError("n-starts must be positive")
    true_parameters = {
        "k_cat": float(args.true_k),
        "gamma": float(args.true_gamma),
        "delta": float(args.true_delta),
    }
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in true_parameters.values()
    ):
        raise ValueError("True parameters must be finite and positive")
    bounds = parameter_bounds(args)
    validate_bounds(bounds)
    for name, value in true_parameters.items():
        lower, upper = bounds[name]
        if not lower < value < upper:
            raise ValueError(f"True {name}={value:g} is outside inversion bounds")

    explicit_starts = parse_explicit_starts(args.starts)
    starts = generate_starts(
        bounds,
        args.n_starts,
        args.seed,
        explicit_starts,
    )
    for start in starts:
        encode_start(start, PARAMETER_NAMES, bounds)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print("Generating high-resolution direct targets", flush=True)
    clean_curves = make_scan_curves(
        sigmas,
        true_parameters,
        args.target_time_grid,
        args.target_modes,
        args.observation_points,
    )
    all_results = {}
    rows = []
    for noise_level in noise_levels:
        print(f"Preparing observations for noise={noise_level:g}", flush=True)
        curves = add_noise(clean_curves, noise_level, rng)
        observations = torch_curves(curves, device)
        for mode in modes:
            fixed_values = fixed_values_for_mode(mode, true_parameters, args)
            case_results = []
            for start_index, start in enumerate(starts):
                print(
                    f"mode={mode} noise={noise_level:g} "
                    f"start={start_index + 1}/{len(starts)} "
                    f"k={start['k_cat']:.4g} gamma={start['gamma']:.4g} "
                    f"delta={start['delta']:.4g}",
                    flush=True,
                )
                result = invert_one_start(
                    mode,
                    start,
                    true_parameters,
                    fixed_values,
                    bounds,
                    observations,
                    args.operator_modes,
                    args.newton_iterations,
                    args.lbfgs_iterations,
                    args.history_backend,
                    args.history_near_cells,
                    args.history_soe_terms,
                    args.history_soe_tolerance,
                    device,
                )
                case_results.append(result)
                rows.append({
                    "noise_level": noise_level,
                    "mode": mode,
                    "start_index": start_index,
                    "objective": result["objective"],
                    "max_free_relative_error": result["max_free_relative_error"],
                    "estimated_k": result["estimated_parameters"]["k_cat"],
                    "estimated_gamma": result["estimated_parameters"]["gamma"],
                    "estimated_delta": result["estimated_parameters"]["delta"],
                    "relative_k_error": result["relative_errors"]["k_cat"],
                    "relative_gamma_error": result["relative_errors"]["gamma"],
                    "relative_delta_error": result["relative_errors"]["delta"],
                    "function_evaluations": result["function_evaluations"],
                    "failure": result["failure"],
                })
                estimate = result["estimated_parameters"]
                print(
                    f"  objective={result['objective']:.3e} "
                    f"estimate=({estimate['k_cat']:.6g}, "
                    f"{estimate['gamma']:.6g}, {estimate['delta']:.6g}) "
                    f"max_free_error={result['max_free_relative_error']:.3e}",
                    flush=True,
                )
            best = min(case_results, key=lambda item: item["objective"])
            key = f"noise={noise_level:.12g},mode={mode}"
            all_results[key] = {
                "best_by_objective": best,
                "all_starts": case_results,
                "parameter_spread": {
                    name: {
                        "min": float(min(
                            item["estimated_parameters"][name]
                            for item in case_results
                        )),
                        "max": float(max(
                            item["estimated_parameters"][name]
                            for item in case_results
                        )),
                    }
                    for name in PARAMETER_NAMES
                },
            }

    csv_path = args.output_dir / "multiscan_inversion_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "study": "fdm_free_multiscan_parameter_inversion",
        "fdm_used": False,
        "neural_network_used": False,
        "target_source": "high_resolution_direct_physical_operator",
        "inverse_source": "differentiable_physical_operator",
        "true_parameters": true_parameters,
        "parameter_bounds": bounds,
        "scan_rates": sigmas,
        "noise_levels": noise_levels,
        "inverse_modes": modes,
        "target_resolution": {
            "time_grid": args.target_time_grid,
            "modes": args.target_modes,
            "history_backend": "direct",
        },
        "inverse_resolution": {
            "observation_points_per_scan": args.observation_points,
            "modes": args.operator_modes,
            "history_backend": args.history_backend,
            "history_near_cells": args.history_near_cells,
            "history_soe_terms": args.history_soe_terms,
            "history_soe_tolerance": args.history_soe_tolerance,
            "device": device,
        },
        "loss": (
            "equal-weight mean over scans of current MSE divided by each "
            "observed clean-current span squared; t=0 excluded"
        ),
        "results": all_results,
    }
    json_path = args.output_dir / "multiscan_inversion_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
