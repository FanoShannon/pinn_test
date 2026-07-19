"""Posterior ablation of local-column and surface-modal curved-film DtN."""

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_pi_dtn_bie as curved


def nrmse(candidate, reference):
    return float(
        np.sqrt(np.mean((candidate - reference) ** 2))
        /max(float(np.max(np.abs(reference))), 1e-15)
    )


def area_mean(values, area):
    return np.sum(values * area[None, :], axis=1) / np.sum(area)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate tangential strength in the surface-modal film DtN"
    )
    parser.add_argument(
        "--output", default="runs_curved_heat_bem/surface_modal_ablation"
    )
    parser.add_argument(
        "--fdm-curves",
        default=(
            "runs_curved_heat_bem/local_fdm_recheck/"
            "curved_pi_dtn_bie_fdm_curves.npz"
        ),
    )
    parser.add_argument("--strengths", default="0,1")
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--film-modes", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    strengths = [float(value) for value in args.strengths.split(",")]
    fdm_path = Path(args.fdm_curves)
    if not fdm_path.exists():
        raise FileNotFoundError(
            "Run validate_curved_pi_dtn_bie_fdm.py once before this ablation: "
            f"{fdm_path}"
        )
    fdm = np.load(fdm_path)
    time = np.linspace(0.0, 1.0, args.steps)
    config = curved.CurvedSurfaceConfig()
    parameters = curved.CurvedSurfacePhysics()

    cases = [("local", None)] + [
        ("surface_modal", strength) for strength in strengths
    ]
    records = []
    curves = {}
    local_current = None
    for film_model, strength in cases:
        result = curved.simulate_curved_pi_dtn_bie(
            time,
            config,
            parameters,
            n_film_modes=args.film_modes,
            history_quadrature=6,
            max_iterations=30,
            film_model=film_model,
            film_tangential_strength=1.0 if strength is None else strength,
        )
        b_mean = area_mean(result["C_B_int"], result["face_area"])
        c_mean = area_mean(result["C_C_int"], result["face_area"])
        total_flux = np.sum(
            result["J_face"] * result["face_area"][None, :], axis=1
        ) /(config.length_x * config.length_y)
        if local_current is None:
            local_current = result["electrode_current_density"]
        label = "local" if strength is None else f"alpha={strength:g}"
        curves[label] = {
            "current": result["electrode_current_density"],
            "b_mean": b_mean,
        }
        records.append({
            "label": label,
            "film_model": film_model,
            "tangential_strength": strength,
            "CV_nrmse_vs_fdm": nrmse(
                result["electrode_current_density"], fdm["fdm_current"]
            ),
            "CV_nrmse_vs_local": nrmse(
                result["electrode_current_density"], local_current
            ),
            "total_flux_nrmse_vs_fdm": nrmse(
                total_flux, fdm["fdm_total_flux_per_projected_area"]
            ),
            "C_B_mean_nrmse_vs_fdm": nrmse(
                b_mean, fdm["fdm_C_B_interface_mean"]
            ),
            "C_C_over_gamma_mean_nrmse_vs_fdm": nrmse(
                c_mean / parameters.gamma,
                fdm["fdm_C_C_interface_mean"] / parameters.gamma,
            ),
            "max_closure_residual": float(
                np.max(result["closure_residual"])
            ),
            "constant_mode_residual": result["film_constant_mode_residual"],
        })
        del result
        gc.collect()

    summary = {
        "status": "posterior_ablation_no_fdm_feedback",
        "fdm_data_used_in_forward": False,
        "surface_grid": [config.n_x, config.n_y],
        "time_steps": args.steps,
        "records": records,
        "interpretation_boundary": (
            "The curved variable-thickness model uses a frozen normal basis. "
            "The ablation can identify sensitivity to tangential coupling, but "
            "cannot separate smooth-triangle/voxel geometry mismatch."
        ),
    }
    (output / "surface_modal_film_dtn_ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(fdm["theta"], fdm["fdm_current"], color="black", label="FDM")
    for label, values in curves.items():
        axes[0].plot(fdm["theta"], values["current"], label=label)
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("electrode current density")
    axes[0].set_title("Film-DtN ablation: CV")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(time, fdm["fdm_C_B_interface_mean"], color="black", label="FDM")
    for label, values in curves.items():
        axes[1].plot(time, values["b_mean"], label=label)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("mean film-side interface C_B")
    axes[1].set_title("Film-DtN ablation: interface state")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    figure.savefig(output / "surface_modal_film_dtn_ablation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
