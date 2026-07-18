import argparse
import json
import time as wall_time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_pi_dtn_bie as curved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end curved PI-DtN heat-BIE solver"
    )
    parser.add_argument("--output", default="runs_curved_heat_bem/full_coupled")
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--periodic-images", type=int, default=3)
    parser.add_argument("--base-thickness", type=float, default=0.075)
    parser.add_argument("--cap-height", type=float, default=0.12)
    parser.add_argument("--cap-radius", type=float, default=0.26)
    parser.add_argument("--cap-profile", choices=("spherical", "cosine"), default="spherical")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--heterogeneity-x", type=float, default=0.55)
    parser.add_argument("--heterogeneity-xy", type=float, default=0.25)
    parser.add_argument("--film-modes", type=int, default=64)
    parser.add_argument("--history-quadrature", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, args.duration, args.steps)
    config = curved.CurvedSurfaceConfig(
        n_x=args.nx,
        n_y=args.ny,
        base_thickness=args.base_thickness,
        cap_height=args.cap_height,
        cap_radius=args.cap_radius,
        cap_profile=args.cap_profile,
        periodic_images=args.periodic_images,
    )
    parameters = curved.CurvedSurfacePhysics(
        gamma=args.gamma,
        k_cat=args.k_cat,
        heterogeneity_x=args.heterogeneity_x,
        heterogeneity_xy=args.heterogeneity_xy,
    )
    start = wall_time.perf_counter()
    result = curved.simulate_curved_pi_dtn_bie(
        time,
        config,
        parameters,
        n_film_modes=args.film_modes,
        history_quadrature=args.history_quadrature,
        max_iterations=30,
    )
    elapsed = wall_time.perf_counter() - start
    area = result["face_area"]
    b_mean = np.sum(result["C_B_int"] * area[None, :], axis=1) / np.sum(area)
    c_mean = np.sum(result["C_C_int"] * area[None, :], axis=1) / np.sum(area)
    summary = {
        "method": result["method"],
        "status": "complete_interface_only_research_solver",
        "full_external_heat_bie": result["full_external_heat_bie"],
        "film_two_boundary_dtn": result["film_two_boundary_dtn"],
        "film_tangential_diffusion_included": result[
            "film_tangential_diffusion_included"
        ],
        "volume_grid_used": result["volume_grid_used"],
        "fdm_data_used": result["fdm_data_used"],
        "neural_network_used": result["neural_network_used"],
        "runtime_seconds": elapsed,
        "time_steps": len(time),
        "surface_panels": int(len(area)),
        "max_closure_residual": float(np.max(result["closure_residual"])),
        "max_bie_boundary_residual": float(
            np.max(result["bie_boundary_residual"])
        ),
        "max_newton_iterations": int(np.max(result["newton_iterations"])),
        "nonaxisymmetric_index": result["nonaxisymmetric_index"],
        "current_min": float(np.min(result["electrode_current_density"])),
        "current_max": float(np.max(result["electrode_current_density"])),
        "C_B_interface_range": [
            float(np.min(result["C_B_int"])),
            float(np.max(result["C_B_int"])),
        ],
        "C_C_interface_range": [
            float(np.min(result["C_C_int"])),
            float(np.max(result["C_C_int"])),
        ],
        "parameters": {
            "gamma": parameters.gamma,
            "k_cat": parameters.k_cat,
            "base_thickness": config.base_thickness,
            "cap_height": config.cap_height,
            "cap_radius": config.cap_radius,
        },
    }
    (output / "curved_pi_dtn_bie_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    peak_flux_index = int(np.argmax(result["mean_reaction_flux"]))
    np.savez_compressed(
        output / "curved_pi_dtn_bie_state.npz",
        time=time,
        theta=result["theta"],
        current=result["electrode_current_density"],
        mean_flux=result["mean_reaction_flux"],
        C_B_interface_mean=b_mean,
        C_C_interface_mean=c_mean,
        face_centers=result["face_centers"],
        peak_flux=result["J_face"][peak_flux_index],
    )

    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    axis = figure.add_subplot(2, 2, 1)
    axis.plot(result["theta"], result["electrode_current_density"])
    axis.set_title("Curved-film cyclic voltammogram")
    axis.set_xlabel("theta")
    axis.set_ylabel("electrode current density")
    axis.grid(alpha=0.25)
    axis = figure.add_subplot(2, 2, 2)
    axis.plot(time, result["mean_reaction_flux"])
    axis.set_title("Area-mean catalytic flux")
    axis.set_xlabel("time")
    axis.grid(alpha=0.25)
    axis = figure.add_subplot(2, 2, 3)
    axis.plot(time, b_mean, label="C_B interface")
    axis.plot(time, c_mean / parameters.gamma, label="C_C/gamma interface")
    axis.set_title("Interface states")
    axis.set_xlabel("time")
    axis.legend()
    axis.grid(alpha=0.25)
    axis = figure.add_subplot(2, 2, 4, projection="3d")
    centers = result["face_centers"]
    scatter = axis.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=result["J_face"][peak_flux_index],
        cmap="viridis",
        s=28,
    )
    axis.set_title("Peak-time curved-interface flux")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    figure.colorbar(scatter, ax=axis, shrink=0.7)
    figure.savefig(output / "curved_pi_dtn_bie.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
