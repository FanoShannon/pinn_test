import argparse
import json
from pathlib import Path

import numpy as np
import torch

import differentiable_coupled_operator as differentiable
import pinn_thin_layer_v9_6 as pinn


PARAMETER_NAMES = ("log_k_cat", "log_gamma", "log_delta")


def parse_values(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one positive value is required")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("All values must be finite and positive")
    return values


def sensitivity_matrix(
    time,
    k_cat,
    gamma,
    delta,
    n_modes,
    history_backend,
    history_near_cells,
    history_soe_terms,
    history_soe_tolerance,
):
    parameters = torch.tensor(
        [np.log(k_cat), np.log(gamma), np.log(delta)],
        dtype=torch.float64,
        requires_grad=True,
    )

    def current_from_log_parameters(values):
        state = differentiable.solve_coupled_operator(
            time,
            gamma=torch.exp(values[1]),
            k_cat=torch.exp(values[0]),
            delta=torch.exp(values[2]),
            n_modes=n_modes,
            newton_iterations=10,
            history_backend=history_backend,
            history_near_cells=history_near_cells,
            history_soe_terms=history_soe_terms,
            history_soe_tolerance=history_soe_tolerance,
        )
        return state["J_surface"][1:]

    columns = []
    current = None
    for index in range(len(PARAMETER_NAMES)):
        direction = torch.zeros_like(parameters)
        direction[index] = 1.0
        current, derivative = torch.autograd.functional.jvp(
            current_from_log_parameters,
            parameters,
            direction,
            create_graph=False,
        )
        columns.append(derivative.detach().numpy())
    return current.detach().numpy(), np.column_stack(columns)


def identifiability_metrics(scaled_jacobian, scan_spans):
    scaled = np.asarray(scaled_jacobian, dtype=np.float64)
    column_norms = np.linalg.norm(scaled, axis=0)
    normalized = scaled / np.maximum(column_norms[None, :], 1e-15)
    correlation = normalized.T @ normalized
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    condition = float(
        singular_values[0] / max(singular_values[-1], 1e-15)
    )
    fisher = scaled.T @ scaled
    fisher_eigenvalues = np.linalg.eigvalsh(fisher)
    return {
        "scan_current_spans": [float(value) for value in scan_spans],
        "relative_sensitivity_norms": {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, column_norms)
        },
        "sensitivity_correlation": {
            row_name: {
                column_name: float(correlation[row, column])
                for column, column_name in enumerate(PARAMETER_NAMES)
            }
            for row, row_name in enumerate(PARAMETER_NAMES)
        },
        "singular_values": [float(value) for value in singular_values],
        "jacobian_condition_number": condition,
        "fisher_eigenvalues": [
            float(value) for value in fisher_eigenvalues
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FDM-free local identifiability audit for joint k/gamma/delta "
            "inversion from one CV current."
        )
    )
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument(
        "--sigmas",
        default="40",
        help="Comma-separated triangular scan rates.",
    )
    parser.add_argument(
        "--deltas",
        default="0.0175,0.035,0.07,0.14",
    )
    parser.add_argument("--time-grid", type=int, default=257)
    parser.add_argument("--modes", type=int, default=96)
    parser.add_argument(
        "--history-backend",
        choices=("direct", "soe"),
        default="soe",
    )
    parser.add_argument("--history-near-cells", type=int, default=16)
    parser.add_argument("--history-soe-terms", type=int, default=128)
    parser.add_argument("--history-soe-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    deltas = parse_values(args.deltas)
    sigmas = parse_values(args.sigmas)
    cases = {}
    for delta in deltas:
        scaled_blocks = []
        spans = []
        for sigma in sigmas:
            simulation_time = (
                2.0
                *abs(float(pinn.theta_i - pinn.theta_switch))
                /sigma
            )
            time = torch.linspace(
                0.0,
                simulation_time,
                args.time_grid,
                dtype=torch.float64,
            )
            current, jacobian = sensitivity_matrix(
                time,
                args.k_cat,
                args.gamma,
                delta,
                args.modes,
                args.history_backend,
                args.history_near_cells,
                args.history_soe_terms,
                args.history_soe_tolerance,
            )
            current_span = max(float(np.ptp(current)), 1e-15)
            spans.append(current_span)
            scaled_blocks.append(jacobian / current_span)
        metrics = identifiability_metrics(
            np.vstack(scaled_blocks),
            spans,
        )
        cases[f"{delta:.12g}"] = metrics
        correlation = metrics["sensitivity_correlation"]
        print(
            f"delta={delta:g} condition="
            f"{metrics['jacobian_condition_number']:.3e} "
            f"corr(k,gamma)={correlation['log_k_cat']['log_gamma']:.5f} "
            f"corr(k,delta)={correlation['log_k_cat']['log_delta']:.5f} "
            f"corr(gamma,delta)="
            f"{correlation['log_gamma']['log_delta']:.5f}"
        )

    report = {
        "study": "joint_inverse_local_identifiability",
        "fdm_used": False,
        "observation": (
            "single_scan_surface_current"
            if len(sigmas) == 1
            else "multi_scan_surface_current"
        ),
        "parameters": {
            "k_cat": float(args.k_cat),
            "gamma": float(args.gamma),
            "sigmas": sigmas,
        },
        "resolution": {
            "n_time": int(args.time_grid),
            "n_modes": int(args.modes),
            "history_backend": args.history_backend,
            "history_near_cells": args.history_near_cells,
            "history_soe_terms": args.history_soe_terms,
            "history_soe_tolerance": args.history_soe_tolerance,
        },
        "cases": cases,
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
