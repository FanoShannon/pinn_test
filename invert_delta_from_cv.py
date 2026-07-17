import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import differentiable_coupled_operator as differentiable
import pinn_thin_layer_v9_6 as pinn
import prototype_coupled_productintegral_dtn as numpy_operator


def parse_values(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required")
    for value in values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Values must be finite and positive, got {value}")
    return values


def bounded_delta(raw, delta_min, delta_max):
    log_min = np.log(delta_min)
    log_max = np.log(delta_max)
    return torch.exp(
        log_min + (log_max - log_min) * torch.sigmoid(raw)
    )


def raw_from_delta(delta, delta_min, delta_max):
    fraction = (
        np.log(delta) - np.log(delta_min)
    ) / (
        np.log(delta_max) - np.log(delta_min)
    )
    fraction = float(np.clip(fraction, 1e-8, 1.0 - 1e-8))
    return float(np.log(fraction / (1.0 - fraction)))


def high_resolution_target(
    delta,
    gamma,
    k_cat,
    target_time_grid,
    target_modes,
    observation_time,
):
    time = np.linspace(0.0, float(pinn.T_sim), target_time_grid)
    history = numpy_operator.solve_coupled_operator(
        time,
        gamma=gamma,
        k_cat=k_cat,
        n_modes=target_modes,
        newton_iterations=16,
        delta=delta,
    )
    state = numpy_operator.reconstruct_state(
        history,
        np.array([0.0, delta]),
    )
    return np.interp(
        observation_time,
        time,
        state["J_surface"],
    )


def invert_one_case(
    true_delta,
    gamma,
    k_cat,
    initial_delta,
    delta_min,
    delta_max,
    observation_time,
    observed_current,
    inverse_modes,
    newton_iterations,
    lbfgs_iterations,
    history_backend,
    history_near_cells,
    history_soe_terms,
    history_soe_tolerance,
):
    time = torch.from_numpy(observation_time)
    target = torch.from_numpy(observed_current)
    current_scale = max(float(np.ptp(observed_current)), 1e-12)
    mask = time > 0.0
    raw = torch.tensor(
        raw_from_delta(initial_delta, delta_min, delta_max),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [raw],
        lr=0.8,
        max_iter=lbfgs_iterations,
        max_eval=2 * lbfgs_iterations,
        tolerance_grad=1e-11,
        tolerance_change=1e-13,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    trajectory = []

    def closure():
        optimizer.zero_grad()
        delta = bounded_delta(raw, delta_min, delta_max)
        state = differentiable.solve_coupled_operator(
            time,
            gamma=gamma,
            k_cat=k_cat,
            delta=delta,
            n_modes=inverse_modes,
            newton_iterations=newton_iterations,
            history_backend=history_backend,
            history_near_cells=history_near_cells,
            history_soe_terms=history_soe_terms,
            history_soe_tolerance=history_soe_tolerance,
        )
        residual = (
            state["J_surface"][mask] - target[mask]
        ) / current_scale
        loss = torch.mean(residual ** 2)
        loss.backward()
        trajectory.append({
            "evaluation": len(trajectory),
            "delta": float(delta.detach()),
            "loss": float(loss.detach()),
            "raw_gradient": float(raw.grad.detach()),
        })
        return loss

    optimizer.step(closure)
    estimated_delta = float(
        bounded_delta(raw, delta_min, delta_max).detach()
    )
    final_state = differentiable.solve_coupled_operator(
        time,
        gamma=gamma,
        k_cat=k_cat,
        delta=torch.tensor(estimated_delta, dtype=torch.float64),
        n_modes=inverse_modes,
        newton_iterations=newton_iterations,
        history_backend=history_backend,
        history_near_cells=history_near_cells,
        history_soe_terms=history_soe_terms,
        history_soe_tolerance=history_soe_tolerance,
    )
    prediction = final_state["J_surface"].detach().numpy()
    residual = prediction[1:] - observed_current[1:]
    return {
        "true_delta": float(true_delta),
        "initial_delta": float(initial_delta),
        "estimated_delta": estimated_delta,
        "relative_delta_error": float(
            abs(estimated_delta - true_delta) / true_delta
        ),
        "current_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "current_span_nrmse": float(
            np.sqrt(np.mean(residual ** 2)) / current_scale
        ),
        "objective": float(np.mean((residual / current_scale) ** 2)),
        "function_evaluations": len(trajectory),
        "trajectory": trajectory,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FDM-free differentiable inversion of thin-layer thickness from "
            "synthetic CV current generated by the high-resolution physical "
            "operator."
        )
    )
    parser.add_argument(
        "--true-deltas",
        default="0.0175,0.035,0.07,0.14",
    )
    parser.add_argument("--initial-delta", type=float, default=0.035)
    parser.add_argument(
        "--initial-deltas",
        default=None,
        help="Optional comma-separated multi-start values.",
    )
    parser.add_argument("--delta-min", type=float, default=0.005)
    parser.add_argument("--delta-max", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--noise-levels", default="0,0.001")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--target-time-grid", type=int, default=4097)
    parser.add_argument("--target-modes", type=int, default=512)
    parser.add_argument("--observation-points", type=int, default=257)
    parser.add_argument("--inverse-modes", type=int, default=96)
    parser.add_argument("--newton-iterations", type=int, default=10)
    parser.add_argument("--lbfgs-iterations", type=int, default=24)
    parser.add_argument(
        "--history-backend",
        choices=("direct", "soe"),
        default="soe",
    )
    parser.add_argument("--history-near-cells", type=int, default=16)
    parser.add_argument("--history-soe-terms", type=int, default=128)
    parser.add_argument("--history-soe-tolerance", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    true_deltas = parse_values(args.true_deltas)
    initial_deltas = (
        parse_values(args.initial_deltas)
        if args.initial_deltas
        else [args.initial_delta]
    )
    noise_levels = [
        float(value)
        for value in args.noise_levels.split(",")
        if value.strip()
    ]
    if any(level < 0.0 or not np.isfinite(level) for level in noise_levels):
        raise ValueError("noise levels must be finite and nonnegative")
    if not args.delta_min < args.delta_max:
        raise ValueError("delta-min must be less than delta-max")
    for value in true_deltas + initial_deltas:
        if not args.delta_min < value < args.delta_max:
            raise ValueError(
                f"delta={value} must lie inside the inversion bounds"
            )

    observation_time = np.linspace(
        0.0,
        float(pinn.T_sim),
        args.observation_points,
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    cases = {}
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for true_delta in true_deltas:
        clean_current = high_resolution_target(
            true_delta,
            args.gamma,
            args.k_cat,
            args.target_time_grid,
            args.target_modes,
            observation_time,
        )
        current_span = max(float(np.ptp(clean_current)), 1e-12)
        for noise_level in noise_levels:
            noise = rng.normal(
                loc=0.0,
                scale=noise_level * current_span,
                size=len(clean_current),
            )
            noise[0] = 0.0
            observed = clean_current + noise
            for initial_delta in initial_deltas:
                print(
                    f"Inverting delta={true_delta:g}, "
                    f"noise={noise_level:.3g}, initial={initial_delta:g}"
                )
                result = invert_one_case(
                    true_delta,
                    args.gamma,
                    args.k_cat,
                    initial_delta,
                    args.delta_min,
                    args.delta_max,
                    observation_time,
                    observed,
                    args.inverse_modes,
                    args.newton_iterations,
                    args.lbfgs_iterations,
                    args.history_backend,
                    args.history_near_cells,
                    args.history_soe_terms,
                    args.history_soe_tolerance,
                )
                key = (
                    f"delta={true_delta:.12g},"
                    f"noise={noise_level:.12g},"
                    f"initial={initial_delta:.12g}"
                )
                cases[key] = result
                rows.append({
                    key_name: value
                    for key_name, value in result.items()
                    if key_name != "trajectory"
                } | {"noise_level": noise_level})
                print(
                    f"  estimated={result['estimated_delta']:.8f} "
                    f"relative_error={result['relative_delta_error']:.3e} "
                    f"current_nrmse={result['current_span_nrmse']:.3e} "
                    f"evaluations={result['function_evaluations']}"
                )

    csv_path = args.output_dir / "delta_inversion_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "study": "differentiable_delta_inversion",
        "fdm_used": False,
        "training_data": "high_resolution_physical_operator_only",
        "parameters": {
            "gamma": float(args.gamma),
            "k_cat": float(args.k_cat),
            "delta_bounds": [args.delta_min, args.delta_max],
        },
        "target_resolution": {
            "n_time": args.target_time_grid,
            "n_modes": args.target_modes,
            "history_backend": "direct",
        },
        "inverse_resolution": {
            "n_time": args.observation_points,
            "n_modes": args.inverse_modes,
            "history_backend": args.history_backend,
            "history_near_cells": args.history_near_cells,
            "history_soe_terms": args.history_soe_terms,
            "history_soe_tolerance": args.history_soe_tolerance,
        },
        "cases": cases,
    }
    json_path = args.output_dir / "delta_inversion_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
