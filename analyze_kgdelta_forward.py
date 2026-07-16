import argparse
import csv
import json
from pathlib import Path

import numpy as np

import pinn_thin_layer_v9_6 as pinn
import prototype_coupled_productintegral_dtn as coupled


def parse_values(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required")
    for value in values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Values must be finite and positive, got {value}")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FDM-free numerical stress test for the joint k/gamma/delta "
            "coupled physical operator."
        )
    )
    parser.add_argument("--k-values", default="0.01,0.1,1,10,100")
    parser.add_argument("--gamma-values", default="0.1,1,10,100")
    parser.add_argument(
        "--delta-values",
        default="0.01,0.0175,0.035,0.07,0.14",
    )
    parser.add_argument("--time-grid", type=int, default=513)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--newton-iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    k_values = parse_values(args.k_values)
    gamma_values = parse_values(args.gamma_values)
    delta_values = parse_values(args.delta_values)
    if args.time_grid < 3 or args.modes < 1:
        raise ValueError("time-grid and modes must be positive")

    time = np.linspace(0.0, float(pinn.T_sim), args.time_grid)
    rows = []
    failures = []
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for delta in delta_values:
        for gamma in gamma_values:
            for k_cat in k_values:
                label = f"k={k_cat:g},gamma={gamma:g},delta={delta:g}"
                try:
                    history = coupled.solve_coupled_operator(
                        time,
                        gamma=gamma,
                        k_cat=k_cat,
                        n_modes=args.modes,
                        newton_iterations=args.newton_iterations,
                        delta=delta,
                    )
                    state = coupled.reconstruct_state(
                        history,
                        delta * np.linspace(0.0, 1.0, 65),
                    )
                    closure = (
                        history["J_rxn"]
                        -k_cat
                        *history["C_B_int"]
                        *history["C_C_int"]
                    )
                    finite = all(np.isfinite(history[name]).all() for name in (
                        "C_B_int",
                        "C_C_int",
                        "C_D_int",
                        "J_rxn",
                        "amplitudes",
                    ))
                    finite = finite and all(
                        np.isfinite(state[name]).all()
                        for name in ("C_B", "J_surface", "M_B")
                    )
                    bounds_error = max(
                        max(0.0, -float(np.min(state["C_B"]))),
                        max(0.0, float(np.max(state["C_B"])) - 1.0),
                        max(0.0, -float(np.min(history["C_C_int"])) / gamma),
                        max(
                            0.0,
                            float(np.max(history["C_C_int"])) / gamma - 1.0,
                        ),
                    )
                    row = {
                        "k_cat": k_cat,
                        "gamma": gamma,
                        "delta": delta,
                        "Da": k_cat * gamma * delta / pinn.D_rel_B,
                        "Fo_T": pinn.D_rel_B * pinn.T_sim / delta ** 2,
                        "J_ref": coupled.characteristic_reaction_flux(
                            gamma,
                            k_cat,
                            delta,
                            pinn.D_rel_B,
                        ),
                        "finite": bool(finite),
                        "bounds_error": float(bounds_error),
                        "closure_max_abs": float(np.max(np.abs(closure))),
                        "root_residual_max_abs": float(
                            np.max(history["root_residual"])
                        ),
                        "root_iterations_max": int(
                            np.max(history["root_iterations"])
                        ),
                        "C_B_int_min": float(np.min(history["C_B_int"])),
                        "C_B_int_max": float(np.max(history["C_B_int"])),
                        "C_C_int_over_gamma_min": float(
                            np.min(history["C_C_int"]) / gamma
                        ),
                        "C_C_int_over_gamma_max": float(
                            np.max(history["C_C_int"]) / gamma
                        ),
                        "J_rxn_max": float(np.max(history["J_rxn"])),
                        "J_surface_min": float(np.min(state["J_surface"])),
                        "J_surface_max": float(np.max(state["J_surface"])),
                    }
                    rows.append(row)
                    if (
                        not finite
                        or bounds_error > 1e-8
                        or row["closure_max_abs"] > 1e-9
                    ):
                        failures.append({"case": label, **row})
                    print(
                        f"{label} Da={row['Da']:.3e} "
                        f"closure={row['closure_max_abs']:.2e} "
                        f"iterations={row['root_iterations_max']:2d} "
                        f"bounds={row['bounds_error']:.2e}"
                    )
                except Exception as exc:
                    failure = {
                        "case": label,
                        "exception": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"{label} FAILED: {failure['exception']}")

    csv_path = args.output_dir / "kgdelta_forward_stress.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "study": "joint_k_gamma_delta_forward_stress",
        "fdm_used": False,
        "training_steps": 0,
        "resolution": {
            "n_time": int(args.time_grid),
            "n_modes": int(args.modes),
        },
        "case_count": len(rows) + len([
            failure for failure in failures if "exception" in failure
        ]),
        "accepted_count": len(rows) - len([
            failure for failure in failures if "exception" not in failure
        ]),
        "failures": failures,
        "rows": rows,
    }
    json_path = args.output_dir / "kgdelta_forward_stress.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Failures: {len(failures)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
