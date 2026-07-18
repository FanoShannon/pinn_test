import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_film_3d as curved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the true-3D curved-film DtN research prototype"
    )
    parser.add_argument("--output", default="runs_curved_film_3d/prototype")
    parser.add_argument("--steps", type=int, default=33)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument(
        "--cap-profile", choices=("spherical", "cosine"), default="spherical"
    )
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--heterogeneity-x", type=float, default=0.55)
    parser.add_argument("--heterogeneity-xy", type=float, default=0.25)
    parser.add_argument("--max-newton", type=int, default=20)
    return parser.parse_args()


def plot_result(result, output):
    geometry = result["geometry"]
    operator = result["operator"]
    film = result["film_history"][-1]
    external = result["external_history"][-1]
    b_field, d_field = operator.expand_fields(film, external)
    y_index = len(geometry["y"]) // 2
    face = result["face_centers"]
    final_flux = result["J_face"][-1]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    image = axes[0, 0].imshow(
        geometry["height"].T,
        origin="lower",
        extent=(
            geometry["x"][0], geometry["x"][-1],
            geometry["y"][0], geometry["y"][-1],
        ),
        aspect="auto",
    )
    axes[0, 0].set_title("Curved film height h(x,y)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    figure.colorbar(image, ax=axes[0, 0])

    scatter = axes[0, 1].scatter(
        face[:, 0], face[:, 1], c=final_flux, s=22, cmap="viridis"
    )
    axes[0, 1].set_title("Final interface flux J(x,y,z)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    figure.colorbar(scatter, ax=axes[0, 1])

    combined = np.where(
        geometry["film_mask"][:, y_index, :],
        b_field[:, y_index, :],
        (result["parameters"].gamma - d_field[:, y_index, :])
        /result["parameters"].gamma,
    )
    image = axes[1, 0].imshow(
        combined.T,
        origin="lower",
        extent=(
            geometry["x"][0], geometry["x"][-1],
            geometry["z"][0], geometry["z"][-1],
        ),
        aspect="auto",
        cmap="plasma",
    )
    axes[1, 0].set_title("Mid-y dimensionless section: B, C/gamma")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("z")
    figure.colorbar(image, ax=axes[1, 0])

    axes[1, 1].plot(
        result["time"], result["electrode_current_density"], label="electrode"
    )
    axes[1, 1].plot(
        result["time"], result["mean_reaction_flux"], label="interface"
    )
    axes[1, 1].set_title("Spatially averaged fluxes")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].locator_params(axis="x", nbins=5)
    axes[1, 1].legend()
    figure.savefig(output / "curved_film_3d_summary.png", dpi=180)
    plt.close(figure)

    xx, yy = np.meshgrid(geometry["x"], geometry["y"], indexing="ij")
    figure = plt.figure(figsize=(11, 5), constrained_layout=True)
    surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
    surface = surface_axis.plot_surface(
        xx,
        yy,
        geometry["height"],
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )
    surface_axis.set_title("Spherical-cap film interface")
    surface_axis.set_xlabel("x")
    surface_axis.set_ylabel("y")
    surface_axis.set_zlabel("z")
    surface_axis.view_init(elev=28, azim=-58)
    figure.colorbar(surface, ax=surface_axis, shrink=0.7, label="height")

    flux_axis = figure.add_subplot(1, 2, 2, projection="3d")
    scatter = flux_axis.scatter(
        face[:, 0],
        face[:, 1],
        face[:, 2],
        c=final_flux,
        s=28,
        cmap="plasma",
    )
    flux_axis.set_title("Non-axisymmetric interface flux")
    flux_axis.set_xlabel("x")
    flux_axis.set_ylabel("y")
    flux_axis.set_zlabel("z")
    flux_axis.view_init(elev=28, azim=-58)
    figure.colorbar(scatter, ax=flux_axis, shrink=0.7, label="J")
    figure.savefig(output / "curved_film_3d_surface.png", dpi=180)
    plt.close(figure)

    switch = int(np.argmin(result["theta"]))
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.plot(
        result["theta"][:switch + 1],
        result["electrode_current_density"][:switch + 1],
        color="#1261a0",
        linewidth=2.2,
        label="forward scan",
    )
    axis.plot(
        result["theta"][switch:],
        result["electrode_current_density"][switch:],
        color="#c43c39",
        linewidth=2.2,
        label="reverse scan",
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axis.set_xlabel("dimensionless potential theta")
    axis.set_ylabel("mean electrode current density")
    axis.set_title("True-3D curved-film cyclic voltammogram")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output / "curved_film_3d_cv.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grid = curved.CurvedFilmGrid(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        cap_profile=args.cap_profile,
    )
    parameters = curved.CurvedFilmPhysics(
        gamma=args.gamma,
        k_cat=args.k_cat,
        heterogeneity_x=args.heterogeneity_x,
        heterogeneity_xy=args.heterogeneity_xy,
    )
    result = curved.simulate_curved_film_3d(
        np.linspace(0.0, args.duration, args.steps),
        grid=grid,
        parameters=parameters,
        max_iterations=args.max_newton,
        store_fields=True,
    )
    np.savez_compressed(
        output / "curved_film_3d_results.npz",
        time=result["time"],
        theta=result["theta"],
        electrode_current_density=result["electrode_current_density"],
        mean_reaction_flux=result["mean_reaction_flux"],
        J_face=result["J_face"],
        C_B_int=result["C_B_int"],
        C_C_int=result["C_C_int"],
        C_D_int=result["C_D_int"],
        face_centers=result["face_centers"],
        face_area=result["face_area"],
        k_face=result["k_face"],
    )
    summary = {
        "status": "research_prototype",
        "coordinates": "full_cartesian_xyz",
        "symmetry_reduction": False,
        "volume_grid_free": False,
        "grid": asdict(grid),
        "parameters": asdict(parameters),
        "n_interface_faces": int(result["J_face"].shape[1]),
        "max_closure_residual": float(np.max(result["closure_residual"])),
        "max_linear_reconstruction_residual": float(
            np.max(result["linear_residual"])
        ),
        "max_newton_iterations": int(np.max(result["newton_iterations"])),
        "nonaxisymmetric_index": result["nonaxisymmetric_index"],
        "flux_spatial_cv": result["flux_spatial_cv"],
        "final_mean_reaction_flux": float(result["mean_reaction_flux"][-1]),
        "final_electrode_current_density": float(
            result["electrode_current_density"][-1]
        ),
    }
    (output / "curved_film_3d_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_result(result, output)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
