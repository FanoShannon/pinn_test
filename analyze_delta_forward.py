import argparse
import csv
import json
from pathlib import Path

import numpy as np

import pinn_thin_layer_v9_6 as pinn
import prototype_coupled_productintegral_dtn as coupled


def parse_float_list(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one positive value is required")
    for value in values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Values must be finite and positive, got {value}")
    return values


def parse_int_list(text):
    values = sorted({
        int(item.strip())
        for item in text.split(",")
        if item.strip()
    })
    if not values or values[0] < 3:
        raise ValueError("Resolution values must be integers of at least three")
    return values


def rmse(predicted, reference):
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def relative_l2(values, scale_values):
    values = np.asarray(values, dtype=np.float64)
    scale_values = np.asarray(scale_values, dtype=np.float64)
    denominator = float(np.linalg.norm(scale_values))
    return float(np.linalg.norm(values) / max(denominator, 1e-15))


def solve_state(delta, gamma, k_cat, n_time, n_modes, newton_iterations):
    time = np.linspace(0.0, float(pinn.T_sim), int(n_time))
    history = coupled.solve_coupled_operator(
        time,
        gamma=gamma,
        k_cat=k_cat,
        n_modes=n_modes,
        newton_iterations=newton_iterations,
        delta=delta,
    )
    x_fraction = np.linspace(0.0, 1.0, 129)
    state = coupled.reconstruct_state(history, delta * x_fraction)
    return history, state


def convergence_metrics(candidate, candidate_state, reference, reference_state):
    time = candidate["time"]
    gamma = float(candidate["gamma"])
    j_ref = coupled.characteristic_reaction_flux(
        gamma,
        candidate["k_cat"],
        candidate["delta"],
        candidate["D_B"],
    )
    reference_b = coupled.interpolate_time_series(
        reference["time"],
        reference_state["C_B"],
        time,
    )
    reference_b_int = coupled.interpolate_time_series(
        reference["time"],
        reference["C_B_int"],
        time,
    )
    reference_c_int = coupled.interpolate_time_series(
        reference["time"],
        reference["C_C_int"],
        time,
    )
    reference_reaction = coupled.interpolate_time_series(
        reference["time"],
        reference["J_rxn"],
        time,
    )
    reference_surface = coupled.interpolate_time_series(
        reference["time"],
        reference_state["J_surface"],
        time,
    )
    current_error = candidate_state["J_surface"] - reference_surface
    current_span = float(np.ptp(reference_surface))
    closure = (
        candidate["J_rxn"]
        -candidate["k_cat"] * candidate["C_B_int"] * candidate["C_C_int"]
    )
    return {
        "C_B_rmse": rmse(candidate_state["C_B"], reference_b),
        "C_B_int_rmse": rmse(candidate["C_B_int"], reference_b_int),
        "C_C_int_over_gamma_rmse": rmse(
            candidate["C_C_int"] / gamma,
            reference_c_int / gamma,
        ),
        "J_rxn_over_J_ref_rmse": rmse(
            candidate["J_rxn"] / j_ref,
            reference_reaction / j_ref,
        ),
        "J_surface_abs_rmse": rmse(
            candidate_state["J_surface"],
            reference_surface,
        ),
        "J_surface_over_J_ref_rmse": rmse(
            candidate_state["J_surface"] / j_ref,
            reference_surface / j_ref,
        ),
        "J_surface_span_nrmse": float(
            np.sqrt(np.mean(current_error ** 2)) / max(current_span, 1e-15)
        ),
        "reaction_closure_max_abs": float(np.max(np.abs(closure))),
        "root_residual_max_abs": float(np.max(candidate["root_residual"])),
        "C_B_min": float(np.min(candidate_state["C_B"])),
        "C_B_max": float(np.max(candidate_state["C_B"])),
        "C_C_int_min": float(np.min(candidate["C_C_int"])),
        "C_C_int_max": float(np.max(candidate["C_C_int"])),
    }


def logarithmic_delta_sensitivity(
    delta,
    gamma,
    k_cat,
    n_time,
    n_modes,
    newton_iterations,
    log_step,
):
    delta_minus = float(delta * np.exp(-log_step))
    delta_plus = float(delta * np.exp(log_step))
    minus_history, minus_state = solve_state(
        delta_minus,
        gamma,
        k_cat,
        n_time,
        n_modes,
        newton_iterations,
    )
    plus_history, plus_state = solve_state(
        delta_plus,
        gamma,
        k_cat,
        n_time,
        n_modes,
        newton_iterations,
    )
    derivative_surface = (
        plus_state["J_surface"] - minus_state["J_surface"]
    ) / (2.0 * log_step)
    derivative_reaction = (
        plus_history["J_rxn"] - minus_history["J_rxn"]
    ) / (2.0 * log_step)
    midpoint_surface = 0.5 * (
        plus_state["J_surface"] + minus_state["J_surface"]
    )
    midpoint_reaction = 0.5 * (
        plus_history["J_rxn"] + minus_history["J_rxn"]
    )
    peak_index = int(np.argmax(np.abs(derivative_surface)))
    return {
        "log_step": float(log_step),
        "surface_relative_l2": relative_l2(
            derivative_surface,
            midpoint_surface,
        ),
        "reaction_relative_l2": relative_l2(
            derivative_reaction,
            midpoint_reaction,
        ),
        "surface_derivative_rms": float(
            np.sqrt(np.mean(derivative_surface ** 2))
        ),
        "surface_derivative_max_abs": float(
            np.max(np.abs(derivative_surface))
        ),
        "surface_derivative_peak_time": float(
            plus_history["time"][peak_index]
        ),
        "surface_derivative_peak_theta": float(
            plus_history["theta"][peak_index]
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FDM-free forward convergence and sensitivity audit for runtime "
            "thin-layer thickness."
        )
    )
    parser.add_argument(
        "--deltas",
        default="0.01,0.0175,0.035,0.07,0.14",
    )
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--time-grids", default="257,513,1025")
    parser.add_argument("--mode-counts", default="32,64,128")
    parser.add_argument("--reference-time-grid", type=int, default=4097)
    parser.add_argument("--reference-modes", type=int, default=256)
    parser.add_argument("--newton-iterations", type=int, default=16)
    parser.add_argument("--sensitivity-time-grid", type=int, default=1025)
    parser.add_argument("--sensitivity-modes", type=int, default=128)
    parser.add_argument("--sensitivity-log-step", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    deltas = parse_float_list(args.deltas)
    time_grids = parse_int_list(args.time_grids)
    mode_counts = parse_int_list(args.mode_counts)
    coupled.validate_positive_parameters(args.gamma, args.k_cat, min(deltas))
    if args.reference_time_grid < max(time_grids):
        raise ValueError("reference-time-grid must cover the tested time grids")
    if args.reference_modes < max(mode_counts):
        raise ValueError("reference-modes must cover the tested mode counts")
    if args.sensitivity_log_step <= 0.0:
        raise ValueError("sensitivity-log-step must be positive")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = {}

    for delta in deltas:
        print(f"Reference solve: delta={delta:g}")
        reference, reference_state = solve_state(
            delta,
            args.gamma,
            args.k_cat,
            args.reference_time_grid,
            args.reference_modes,
            args.newton_iterations,
        )
        delta_rows = []
        for n_time in time_grids:
            for n_modes in mode_counts:
                candidate, candidate_state = solve_state(
                    delta,
                    args.gamma,
                    args.k_cat,
                    n_time,
                    n_modes,
                    args.newton_iterations,
                )
                metrics = convergence_metrics(
                    candidate,
                    candidate_state,
                    reference,
                    reference_state,
                )
                row = {
                    "delta": float(delta),
                    "n_time": int(n_time),
                    "n_modes": int(n_modes),
                    **metrics,
                }
                rows.append(row)
                delta_rows.append(row)
                print(
                    f"  nt={n_time:4d} modes={n_modes:3d} "
                    f"C_B={metrics['C_B_rmse']:.3e} "
                    f"Jspan={metrics['J_surface_span_nrmse']:.3e} "
                    f"J/Jref={metrics['J_surface_over_J_ref_rmse']:.3e}"
                )

        sensitivity = logarithmic_delta_sensitivity(
            delta,
            args.gamma,
            args.k_cat,
            args.sensitivity_time_grid,
            args.sensitivity_modes,
            args.newton_iterations,
            args.sensitivity_log_step,
        )
        cases[f"{delta:.12g}"] = {
            "dimensionless_groups": {
                "Da": float(
                    args.k_cat
                    *args.gamma
                    *delta
                    /float(pinn.D_rel_B)
                ),
                "Fo_T": float(
                    pinn.D_rel_B * pinn.T_sim / delta ** 2
                ),
                "K_external": float(
                    args.k_cat * np.sqrt(pinn.T_sim / pinn.D_rel_D)
                ),
            },
            "sensitivity": sensitivity,
            "convergence": delta_rows,
        }

    csv_path = args.output_dir / "delta_forward_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "study": "runtime_delta_forward_convergence",
        "fdm_used": False,
        "training_steps": 0,
        "parameters": {
            "gamma": float(args.gamma),
            "k_cat": float(args.k_cat),
            "D_B": float(pinn.D_rel_B),
            "D_D": float(pinn.D_rel_D),
            "T_sim": float(pinn.T_sim),
        },
        "reference_resolution": {
            "n_time": int(args.reference_time_grid),
            "n_modes": int(args.reference_modes),
        },
        "cases": cases,
    }
    json_path = args.output_dir / "delta_forward_convergence.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
