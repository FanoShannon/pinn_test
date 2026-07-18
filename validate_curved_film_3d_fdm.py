import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_film_3d as curved
import curved_film_3d_fdm as fdm


def nrmse(candidate, reference):
    scale = max(float(np.max(np.abs(reference))), 1e-15)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale)


def weighted_interface_mean(result, name):
    area = result["face_area"]
    return np.sum(result[name] * area[None, :], axis=1) / np.sum(area)


def peak_metrics(theta, current):
    switch = int(np.argmin(theta))
    forward = int(np.argmax(current[:switch + 1]))
    reverse = switch + int(np.argmin(current[switch:]))
    return {
        "forward_theta": float(theta[forward]),
        "forward_current": float(current[forward]),
        "reverse_theta": float(theta[reverse]),
        "reverse_current": float(current[reverse]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Posterior validation against a monolithic true-3D FDM"
    )
    parser.add_argument(
        "--output", default="runs_curved_film_3d/fdm_validation"
    )
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--operator-steps", type=int, default=65)
    parser.add_argument("--fdm-steps", type=int, default=129)
    parser.add_argument("--operator-grid", default="8,7,18")
    parser.add_argument("--fdm-grid", default="12,11,30")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    operator_shape = tuple(int(value) for value in args.operator_grid.split(","))
    fdm_shape = tuple(int(value) for value in args.fdm_grid.split(","))
    if len(operator_shape) != 3 or len(fdm_shape) != 3:
        raise ValueError("Grid specifications must contain nx,ny,nz")
    parameters = curved.CurvedFilmPhysics()
    operator_grid = curved.CurvedFilmGrid(
        nx=operator_shape[0], ny=operator_shape[1], nz=operator_shape[2]
    )
    fdm_grid = curved.CurvedFilmGrid(
        nx=fdm_shape[0], ny=fdm_shape[1], nz=fdm_shape[2]
    )

    same_time = np.linspace(0.0, args.duration, args.operator_steps)
    print("Running condensed true-3D operator...")
    operator = curved.simulate_curved_film_3d(
        same_time,
        grid=operator_grid,
        parameters=parameters,
        store_fields=True,
    )
    print("Running same-grid monolithic FDM audit...")
    same_fdm = fdm.simulate_direct_fdm(
        same_time,
        grid=operator_grid,
        parameters=parameters,
    )

    fine_time = np.linspace(0.0, args.duration, args.fdm_steps)
    print("Running refined monolithic 3D FDM posterior reference...")
    fine_fdm = fdm.simulate_direct_fdm(
        fine_time,
        grid=fdm_grid,
        parameters=parameters,
    )

    fine_current = np.interp(
        operator["time"], fine_fdm["time"], fine_fdm["electrode_current_density"]
    )
    fine_flux = np.interp(
        operator["time"], fine_fdm["time"], fine_fdm["mean_reaction_flux"]
    )
    fine_b_int = np.interp(
        operator["time"],
        fine_fdm["time"],
        weighted_interface_mean(fine_fdm, "C_B_int"),
    )
    fine_c_int = np.interp(
        operator["time"],
        fine_fdm["time"],
        weighted_interface_mean(fine_fdm, "C_C_int") / parameters.gamma,
    )
    operator_b_int = weighted_interface_mean(operator, "C_B_int")
    operator_c_int = (
        weighted_interface_mean(operator, "C_C_int") / parameters.gamma
    )
    operator_b_inventory = np.mean(operator["film_history"], axis=1)
    operator_d_inventory = (
        np.mean(operator["external_history"], axis=1) / parameters.gamma
    )
    fine_b_inventory = np.interp(
        operator["time"],
        fine_fdm["time"],
        fine_fdm["C_B_inventory_mean"],
    )
    fine_d_inventory = np.interp(
        operator["time"],
        fine_fdm["time"],
        fine_fdm["C_D_over_gamma_inventory_mean"],
    )

    operator_peaks = peak_metrics(
        operator["theta"], operator["electrode_current_density"]
    )
    fdm_peaks = peak_metrics(fine_fdm["theta"], fine_fdm["electrode_current_density"])
    metrics = {
        "status": "posterior_only_no_training",
        "geometry": "positive-base spherical cap",
        "parameters": asdict(parameters),
        "operator_grid": asdict(operator_grid),
        "fdm_grid": asdict(fdm_grid),
        "operator_steps": args.operator_steps,
        "fdm_steps": args.fdm_steps,
        "same_grid_algebraic_audit": {
            "CV_max_abs": float(np.max(np.abs(
                operator["electrode_current_density"]
                -same_fdm["electrode_current_density"]
            ))),
            "mean_flux_max_abs": float(np.max(np.abs(
                operator["mean_reaction_flux"]
                -same_fdm["mean_reaction_flux"]
            ))),
            "J_face_max_abs": float(np.max(np.abs(
                operator["J_face"] - same_fdm["J_face"]
            ))),
            "FDM_max_residual": float(np.max(same_fdm["residual"])),
        },
        "refined_fdm_posterior": {
            "CV_nrmse": nrmse(
                operator["electrode_current_density"], fine_current
            ),
            "mean_reaction_flux_nrmse": nrmse(
                operator["mean_reaction_flux"], fine_flux
            ),
            "C_B_interface_mean_nrmse": nrmse(operator_b_int, fine_b_int),
            "C_C_over_gamma_interface_mean_nrmse": nrmse(
                operator_c_int, fine_c_int
            ),
            "C_B_inventory_mean_nrmse": nrmse(
                operator_b_inventory, fine_b_inventory
            ),
            "C_D_over_gamma_inventory_mean_nrmse": nrmse(
                operator_d_inventory, fine_d_inventory
            ),
            "operator_peaks": operator_peaks,
            "fdm_peaks": fdm_peaks,
            "forward_peak_current_relative_error": float(
                abs(operator_peaks["forward_current"] - fdm_peaks["forward_current"])
                /max(abs(fdm_peaks["forward_current"]), 1e-15)
            ),
            "reverse_peak_current_relative_error": float(
                abs(operator_peaks["reverse_current"] - fdm_peaks["reverse_current"])
                /max(abs(fdm_peaks["reverse_current"]), 1e-15)
            ),
            "forward_peak_theta_abs_error": float(abs(
                operator_peaks["forward_theta"] - fdm_peaks["forward_theta"]
            )),
            "reverse_peak_theta_abs_error": float(abs(
                operator_peaks["reverse_theta"] - fdm_peaks["reverse_theta"]
            )),
            "FDM_max_residual": float(np.max(fine_fdm["residual"])),
        },
    }
    (output / "curved_film_3d_fdm_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "curved_film_3d_fdm_curves.npz",
        operator_time=operator["time"],
        operator_theta=operator["theta"],
        operator_current=operator["electrode_current_density"],
        operator_mean_flux=operator["mean_reaction_flux"],
        fdm_time=fine_fdm["time"],
        fdm_theta=fine_fdm["theta"],
        fdm_current=fine_fdm["electrode_current_density"],
        fdm_mean_flux=fine_fdm["mean_reaction_flux"],
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(
        operator["theta"], operator["electrode_current_density"],
        label="condensed operator", linewidth=2.0
    )
    axes[0, 0].plot(
        fine_fdm["theta"], fine_fdm["electrode_current_density"],
        "--", label="refined monolithic FDM", linewidth=1.8
    )
    axes[0, 0].set_title("CV posterior")
    axes[0, 0].set_xlabel("theta")
    axes[0, 0].set_ylabel("electrode current density")
    axes[0, 0].legend()

    axes[0, 1].plot(operator["time"], operator["mean_reaction_flux"], label="operator")
    axes[0, 1].plot(fine_fdm["time"], fine_fdm["mean_reaction_flux"], "--", label="FDM")
    axes[0, 1].set_title("Mean curved-interface flux")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].legend()

    axes[1, 0].plot(operator["time"], operator_b_int, label="operator B")
    axes[1, 0].plot(operator["time"], fine_b_int, "--", label="FDM B")
    axes[1, 0].plot(operator["time"], operator_c_int, label="operator C/gamma")
    axes[1, 0].plot(operator["time"], fine_c_int, "--", label="FDM C/gamma")
    axes[1, 0].set_title("Area-mean interface concentrations")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].legend(ncol=2)

    axes[1, 1].plot(operator["time"], operator_b_inventory, label="operator B")
    axes[1, 1].plot(operator["time"], fine_b_inventory, "--", label="FDM B")
    axes[1, 1].plot(operator["time"], operator_d_inventory, label="operator D/gamma")
    axes[1, 1].plot(operator["time"], fine_d_inventory, "--", label="FDM D/gamma")
    axes[1, 1].set_title("Volume-mean inventories")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].legend(ncol=2)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / "curved_film_3d_fdm_validation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
