import argparse
import json
import time as wall_time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_film_3d as volume
import curved_film_3d_fdm as direct_fdm
import curved_pi_dtn_bie as boundary


def nrmse(candidate, reference):
    return float(
        np.sqrt(np.mean((candidate - reference) ** 2))
        /max(np.max(np.abs(reference)), 1e-15)
    )


def area_mean(values, area):
    return np.sum(values * area[None, :], axis=1) / np.sum(area)


def peak_metrics(candidate, reference, theta, section):
    index = np.arange(len(theta))[section]
    candidate_index = index[np.argmax(candidate[section])]
    reference_index = index[np.argmax(reference[section])]
    reference_peak = float(reference[reference_index])
    return {
        "current_relative_error": float(
            abs(candidate[candidate_index] - reference_peak)
            /max(abs(reference_peak), 1e-15)
        ),
        "theta_absolute_error": float(
            abs(theta[candidate_index] - theta[reference_index])
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Posterior-only curved PI-DtN-BIE versus monolithic FDM"
    )
    parser.add_argument("--output", default="runs_curved_heat_bem/fdm_posterior")
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--surface-grid", default="4,4")
    parser.add_argument("--fdm-grid", default="8,7,108")
    parser.add_argument("--fdm-length-z", type=float, default=3.0)
    parser.add_argument("--periodic-images", type=int, default=3)
    parser.add_argument("--history-quadrature", type=int, default=6)
    parser.add_argument("--film-modes", type=int, default=64)
    return parser.parse_args()


def parse_grid(text, count):
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != count or min(values) < 2:
        raise ValueError(f"Expected {count} positive grid dimensions")
    return values


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    surface_nx, surface_ny = parse_grid(args.surface_grid, 2)
    fdm_nx, fdm_ny, fdm_nz = parse_grid(args.fdm_grid, 3)
    time = np.linspace(0.0, 1.0, args.steps)
    surface_config = boundary.CurvedSurfaceConfig(
        n_x=surface_nx,
        n_y=surface_ny,
        periodic_images=args.periodic_images,
    )
    surface_physics = boundary.CurvedSurfacePhysics()
    start = wall_time.perf_counter()
    bie = boundary.simulate_curved_pi_dtn_bie(
        time,
        surface_config,
        surface_physics,
        n_film_modes=args.film_modes,
        history_quadrature=args.history_quadrature,
        max_iterations=30,
    )
    bie_seconds = wall_time.perf_counter() - start

    fdm_grid = volume.CurvedFilmGrid(
        nx=fdm_nx,
        ny=fdm_ny,
        nz=fdm_nz,
        length_x=surface_config.length_x,
        length_y=surface_config.length_y,
        length_z=args.fdm_length_z,
        base_thickness=surface_config.base_thickness,
        cap_height=surface_config.cap_height,
        cap_radius=surface_config.cap_radius,
        cap_profile=surface_config.cap_profile,
    )
    fdm_physics = volume.CurvedFilmPhysics()
    start = wall_time.perf_counter()
    fdm = direct_fdm.simulate_direct_fdm(
        time,
        grid=fdm_grid,
        parameters=fdm_physics,
        max_iterations=20,
    )
    fdm_seconds = wall_time.perf_counter() - start

    bie_b = area_mean(bie["C_B_int"], bie["face_area"])
    bie_c = area_mean(bie["C_C_int"], bie["face_area"])
    fdm_b = area_mean(fdm["C_B_int"], fdm["face_area"])
    fdm_c = area_mean(fdm["C_C_int"], fdm["face_area"])
    projected_domain_area = surface_config.length_x * surface_config.length_y
    bie_total_flux = np.sum(
        bie["J_face"] * bie["face_area"][None, :], axis=1
    ) / projected_domain_area
    fdm_total_flux = np.sum(
        fdm["J_face"] * fdm["face_area"][None, :], axis=1
    ) / projected_domain_area

    # Linear posterior: prescribe equal total external flux on both geometries.
    amplitude = np.sin(np.pi * time / time[-1]) ** 2
    external_bie = boundary.ExternalNeumannHeatBie(
        bie["surface"],
        surface_physics.diffusion_d,
        len(time),
        time[1] - time[0],
        periodic_images=args.periodic_images,
        quadrature_order=args.history_quadrature,
    )
    density = np.zeros_like(bie["J_face"])
    bie_trace = np.zeros_like(density)
    for step in range(1, len(time)):
        affine = external_bie.affine_step(density, step)
        prescribed = np.full(density.shape[1], amplitude[step])
        density[step] = (
            affine["density_intercept"]
            +affine["density_response"] @ prescribed
        )
        bie_trace[step] = (
            affine["trace_intercept"]
            +affine["trace_response"] @ prescribed
        )
    bie_trace_mean = area_mean(bie_trace, bie["face_area"])
    fdm_operator = volume.CurvedFilm3DOperator(
        fdm_grid, fdm_physics, time[1] - time[0]
    )
    area_ratio = float(
        np.sum(bie["face_area"]) / np.sum(fdm_operator.face_area)
    )
    external_state = np.zeros(
        fdm_operator.external_system.shape[0], dtype=np.float64
    )
    fdm_trace_mean = np.zeros(len(time), dtype=np.float64)
    for step in range(1, len(time)):
        external_free = fdm_operator.external_lu.solve(external_state)
        prescribed = np.full(
            fdm_operator.n_faces, area_ratio * amplitude[step]
        )
        external_state = (
            external_free + fdm_operator.external_volume_response @ prescribed
        )
        trace = (
            np.asarray(fdm_operator.p_external @ external_free).ravel()
            +fdm_operator.external_trace_response @ prescribed
        )
        fdm_trace_mean[step] = np.sum(
            trace * fdm_operator.face_area
        ) / np.sum(fdm_operator.face_area)
    switch = len(time) // 2
    metrics = {
        "status": "posterior_only_no_fdm_feedback",
        "surface_method": bie["method"],
        "film_tangential_diffusion_included": bie[
            "film_tangential_diffusion_included"
        ],
        "surface_grid": [surface_nx, surface_ny],
        "surface_panels": int(bie["J_face"].shape[1]),
        "fdm_grid": [fdm_nx, fdm_ny, fdm_nz],
        "fdm_length_z": args.fdm_length_z,
        "time_steps": len(time),
        "runtime_seconds": {"surface_bie": bie_seconds, "fdm": fdm_seconds},
        "CV_nrmse": nrmse(
            bie["electrode_current_density"],
            fdm["electrode_current_density"],
        ),
        "mean_reaction_flux_nrmse": nrmse(
            bie["mean_reaction_flux"], fdm["mean_reaction_flux"]
        ),
        "total_reaction_flux_per_projected_area_nrmse": nrmse(
            bie_total_flux, fdm_total_flux
        ),
        "C_B_interface_mean_nrmse": nrmse(bie_b, fdm_b),
        "C_C_over_gamma_interface_mean_nrmse": nrmse(
            bie_c / surface_physics.gamma,
            fdm_c / surface_physics.gamma,
        ),
        "forward_peak": peak_metrics(
            bie["electrode_current_density"],
            fdm["electrode_current_density"],
            bie["theta"],
            slice(0, switch + 1),
        ),
        "reverse_peak": peak_metrics(
            bie["electrode_current_density"],
            fdm["electrode_current_density"],
            bie["theta"],
            slice(switch, len(time)),
        ),
        "max_closure_residual": float(np.max(bie["closure_residual"])),
        "max_bie_boundary_residual": float(
            np.max(bie["bie_boundary_residual"])
        ),
        "geometry_area": {
            "smooth_triangle_surface": float(np.sum(bie["face_area"])),
            "voxel_staircase_surface": float(np.sum(fdm["face_area"])),
            "smooth_over_voxel": area_ratio,
        },
        "external_prescribed_equal_total_flux_nrmse": nrmse(
            bie_trace_mean, fdm_trace_mean
        ),
        "interpretation": (
            "The posterior compares a smooth periodic triangular interface and "
            "local-column film DtN against a finite-box voxel FDM. The remaining "
            "difference includes FDM boundary/time error, geometry mismatch, and "
            "the omitted film tangential diffusion."
        ),
    }
    (output / "curved_pi_dtn_bie_fdm_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "curved_pi_dtn_bie_fdm_curves.npz",
        time=time,
        theta=bie["theta"],
        bie_current=bie["electrode_current_density"],
        fdm_current=fdm["electrode_current_density"],
        bie_flux=bie["mean_reaction_flux"],
        fdm_flux=fdm["mean_reaction_flux"],
        bie_C_B_interface_mean=bie_b,
        fdm_C_B_interface_mean=fdm_b,
        bie_C_C_interface_mean=bie_c,
        fdm_C_C_interface_mean=fdm_c,
        bie_total_flux_per_projected_area=bie_total_flux,
        fdm_total_flux_per_projected_area=fdm_total_flux,
        bie_prescribed_external_trace=bie_trace_mean,
        fdm_prescribed_external_trace=fdm_trace_mean,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(bie["theta"], fdm["electrode_current_density"], label="FDM")
    axes[0, 0].plot(bie["theta"], bie["electrode_current_density"], "--", label="surface BIE")
    axes[0, 0].set_title("Cyclic voltammogram")
    axes[0, 0].set_xlabel("theta")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(time, fdm_total_flux, label="FDM")
    axes[0, 1].plot(time, bie_total_flux, "--", label="surface BIE")
    axes[0, 1].set_title("Total reaction flux / projected area")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(time, fdm_b, label="FDM C_B")
    axes[1, 0].plot(time, bie_b, "--", label="BIE C_B")
    axes[1, 0].set_title("Mean film-side interface state")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(time, fdm_c / surface_physics.gamma, label="FDM C_C/gamma")
    axes[1, 1].plot(time, bie_c / surface_physics.gamma, "--", label="BIE C_C/gamma")
    axes[1, 1].set_title("Mean external interface state")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    figure.savefig(output / "curved_pi_dtn_bie_fdm.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
