import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_film_3d as curved


LEVELS = (
    (6, 5, 15),
    (8, 7, 20),
    (10, 9, 25),
)


def normalized_rmse(candidate, reference):
    scale = max(float(np.max(np.abs(reference))), 1e-15)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale)


def main():
    parser = argparse.ArgumentParser(
        description="Resolution audit for the true-3D curved-film prototype"
    )
    parser.add_argument(
        "--output", default="runs_curved_film_3d/convergence"
    )
    parser.add_argument("--steps", type=int, default=13)
    parser.add_argument("--duration", type=float, default=0.015)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, args.duration, args.steps)

    runs = []
    for nx, ny, nz in LEVELS:
        result = curved.simulate_curved_film_3d(
            time,
            grid=curved.CurvedFilmGrid(nx=nx, ny=ny, nz=nz),
            store_fields=False,
        )
        runs.append(result)
        print(
            f"grid={nx}x{ny}x{nz} faces={result['J_face'].shape[1]} "
            f"closure={np.max(result['closure_residual']):.3e} "
            f"linear={np.max(result['linear_residual']):.3e}"
        )

    reference = runs[-1]
    levels = []
    for specification, result in zip(LEVELS, runs):
        levels.append({
            "grid": list(specification),
            "n_interface_faces": int(result["J_face"].shape[1]),
            "electrode_current_nrmse_vs_finest": normalized_rmse(
                result["electrode_current_density"],
                reference["electrode_current_density"],
            ),
            "mean_reaction_flux_nrmse_vs_finest": normalized_rmse(
                result["mean_reaction_flux"],
                reference["mean_reaction_flux"],
            ),
            "max_closure_residual": float(
                np.max(result["closure_residual"])
            ),
            "max_linear_residual": float(
                np.max(result["linear_residual"])
            ),
            "nonaxisymmetric_index": result["nonaxisymmetric_index"],
        })
    summary = {
        "status": "resolution_audit_not_continuum_certificate",
        "time_steps": args.steps,
        "duration": args.duration,
        "levels": levels,
    }
    (output / "curved_film_3d_convergence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for specification, result in zip(LEVELS, runs):
        label = "x".join(str(value) for value in specification)
        axes[0].plot(time, result["electrode_current_density"], label=label)
        axes[1].plot(time, result["mean_reaction_flux"], label=label)
    axes[0].set_title("Electrode current-density convergence")
    axes[1].set_title("Mean interface-flux convergence")
    for axis in axes:
        axis.set_xlabel("time")
        axis.legend()
    figure.savefig(output / "curved_film_3d_convergence.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
