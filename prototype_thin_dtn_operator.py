import argparse
import json
import pickle
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import compare_concentration_fields as compare_fields
import pinn_thin_layer_v9_6 as pinn


def advance_mixed_boundary_modes(time, surface_value, reaction_flux, n_modes):
    diffusion = float(pinn.D_rel_B)
    thickness = float(pinn.delta)
    mode_index = np.arange(n_modes, dtype=np.float64)
    mu = (mode_index + 0.5) * np.pi / thickness
    decay_rate = diffusion * mu**2
    boundary_sign = (-1.0) ** mode_index

    amplitudes = np.empty((len(time), n_modes), dtype=np.float64)
    amplitudes[0] = (
        -2.0 * surface_value[0] / (thickness * mu)
        +2.0 * reaction_flux[0] * boundary_sign /
        (thickness * diffusion * mu**2)
    )

    for step in range(len(time) - 1):
        dt = float(time[step + 1] - time[step])
        surface_slope = (surface_value[step + 1] - surface_value[step]) / dt
        flux_slope = (reaction_flux[step + 1] - reaction_flux[step]) / dt
        forcing = (
            -2.0 * surface_slope / (thickness * mu)
            +2.0 * boundary_sign * flux_slope /
            (thickness * diffusion * mu**2)
        )
        decay = np.exp(-decay_rate * dt)
        gain = -np.expm1(-decay_rate * dt) / decay_rate
        amplitudes[step + 1] = decay * amplitudes[step] + gain * forcing
    return mu, boundary_sign, amplitudes


def reconstruct_thin_state(
    time,
    x_eval,
    surface_value,
    reaction_flux,
    mu,
    boundary_sign,
    amplitudes,
    n_modes,
):
    diffusion = float(pinn.D_rel_B)
    thickness = float(pinn.delta)
    modes = slice(0, n_modes)
    mu_use = mu[modes]
    amplitudes_use = amplitudes[:, modes]
    basis = np.sin(np.outer(x_eval, mu_use))
    field = (
        surface_value[:, None]
        -x_eval[None, :] * reaction_flux[:, None] / diffusion
        +amplitudes_use @ basis.T
    )
    interface = (
        surface_value
        -thickness * reaction_flux / diffusion
        +amplitudes_use @ boundary_sign[modes]
    )
    surface_current = (
        -reaction_flux
        +diffusion * (amplitudes_use @ mu_use)
    )
    inventory = (
        thickness * surface_value
        -0.5 * thickness**2 * reaction_flux / diffusion
        +amplitudes_use @ (1.0 / mu_use)
    )
    inventory_dt = np.gradient(inventory, time, edge_order=2)
    conservative_current = -reaction_flux - inventory_dt
    return {
        "C_B": field,
        "C_B_int": interface,
        "J_surface": surface_current,
        "M_B": inventory,
        "J_conservative": conservative_current,
        "balance_residual": surface_current - conservative_current,
    }


def load_product_integral_history(args, fdm, device):
    config_args = Namespace(
        checkpoint=str(args.checkpoint),
        gamma=args.gamma,
        k_cat_star=args.k_cat,
        arch="multiscale_film_tracegreen_productintegral",
    )
    compare_fields.configure_evaluation_parameters(config_args, fdm)
    model_thin, model_ext = compare_fields.build_models(
        "multiscale_film_tracegreen_productintegral",
        True,
        device,
        green_time_grid=args.green_time_grid,
        green_kernel_points=args.green_kernel_points,
        green_history_grad=True,
    )
    compare_fields.load_checkpoint(model_thin, model_ext, args.checkpoint)
    model_thin.eval()
    interface = model_thin.interface_state

    time = np.asarray(fdm["t"], dtype=np.float64)
    time_tensor = torch.from_numpy(time.astype("float32")).reshape(-1, 1).to(device)
    with torch.no_grad():
        state = interface(time_tensor)
    return {
        "time": time,
        "C_B_surface": state["C_B_surface"].cpu().numpy().reshape(-1).astype(np.float64),
        "C_B_int_film": state["C_B_int"].cpu().numpy().reshape(-1).astype(np.float64),
        "C_C_int": state["C_C_int"].cpu().numpy().reshape(-1).astype(np.float64),
        "J_rxn": state["J_rxn"].cpu().numpy().reshape(-1).astype(np.float64),
    }


def evaluate_mode_count(
    fdm,
    history,
    mu,
    boundary_sign,
    amplitudes,
    n_modes,
    n_time,
    n_x,
    cv_points,
):
    time = history["time"]
    x_fdm = np.asarray(fdm["x_in"], dtype=np.float64)
    time_indices = compare_fields.select_indices(len(time), n_time)
    x_indices = compare_fields.select_indices(len(x_fdm), n_x)
    cv_indices = compare_fields.select_indices(len(time), cv_points)
    x_eval = x_fdm[x_indices]

    state = reconstruct_thin_state(
        time,
        x_eval,
        history["C_B_surface"],
        history["J_rxn"],
        mu,
        boundary_sign,
        amplitudes,
        n_modes,
    )
    fdm_b = np.asarray(
        fdm["concentrations"]["C_B"],
        dtype=np.float64,
    )[np.ix_(time_indices, x_indices)]
    predicted_b = state["C_B"][time_indices]
    fdm_b_int = np.asarray(
        fdm["concentrations"]["C_B"],
        dtype=np.float64,
    )[time_indices, -1]
    predicted_b_int = state["C_B_int"][time_indices]

    fdm_current = np.asarray(fdm["J"], dtype=np.float64)[cv_indices]
    predicted_current = state["J_surface"][cv_indices].copy()
    predicted_conservative = state["J_conservative"][cv_indices].copy()
    if cv_indices[0] == 0:
        predicted_current[0] = 0.0
        predicted_conservative[0] = 0.0

    reaction_reconstructed = (
        float(pinn.k_cat_star) *
        state["C_B_int"] *
        history["C_C_int"]
    )
    closure_defect = history["J_rxn"] - reaction_reconstructed
    j_ref = float(pinn.characteristic_reaction_flux())
    tau0 = 4.0 * float(pinn.delta) ** 2 / (np.pi**2 * float(pinn.D_rel_B))
    settled = time[cv_indices] >= 5.0 * tau0

    return {
        "n_modes": int(n_modes),
        "C_A": compare_fields.field_metrics(1.0 - predicted_b, 1.0 - fdm_b),
        "C_B": compare_fields.field_metrics(predicted_b, fdm_b),
        "C_B_int": compare_fields.field_metrics(predicted_b_int, fdm_b_int),
        "CV_J_surface": compare_fields.field_metrics(predicted_current, fdm_current),
        "CV_J_conservative": compare_fields.field_metrics(
            predicted_conservative,
            fdm_current,
        ),
        "CV_J_over_J_ref": compare_fields.field_metrics(
            predicted_current / j_ref,
            fdm_current / j_ref,
        ),
        "CV_J_over_J_ref_after_5tau0": compare_fields.field_metrics(
            predicted_current[settled] / j_ref,
            fdm_current[settled] / j_ref,
        ),
        "surface_vs_conservative": compare_fields.field_metrics(
            predicted_current,
            predicted_conservative,
        ),
        "reaction_closure_defect_rmse": float(
            np.sqrt(np.mean(closure_defect**2))
        ),
        "reaction_closure_defect_over_J_ref_rmse": float(
            np.sqrt(np.mean((closure_defect / j_ref) ** 2))
        ),
        "film_vs_dtn_interface_rmse": float(np.sqrt(np.mean(
            (history["C_B_int_film"] - state["C_B_int"]) ** 2
        ))),
        "_state": state,
        "_time_indices": time_indices,
        "_x_indices": x_indices,
        "_cv_indices": cv_indices,
        "_closure_defect": closure_defect,
    }


def serializable_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("_")
    }


def plot_diagnostics(output_dir, fdm, history, metrics):
    state = metrics["_state"]
    time_indices = metrics["_time_indices"]
    x_indices = metrics["_x_indices"]
    cv_indices = metrics["_cv_indices"]
    closure_defect = metrics["_closure_defect"]
    time = history["time"]
    x_eval = np.asarray(fdm["x_in"], dtype=np.float64)[x_indices]
    fdm_b = np.asarray(fdm["concentrations"]["C_B"], dtype=np.float64)[
        np.ix_(time_indices, x_indices)
    ]
    predicted_b = state["C_B"][time_indices]
    surface_current = state["J_surface"][cv_indices].copy()
    conservative_current = state["J_conservative"][cv_indices].copy()
    if cv_indices[0] == 0:
        surface_current[0] = 0.0
        conservative_current[0] = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    residual = predicted_b - fdm_b
    vmax = max(float(np.percentile(np.abs(residual), 99.0)), 1e-8)
    image = axes[0, 0].imshow(
        residual.T,
        origin="lower",
        aspect="auto",
        extent=[
            time[time_indices[0]],
            time[time_indices[-1]],
            x_eval[0],
            x_eval[-1],
        ],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 0].set_title("One-way thin-DtN C_B residual")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("x")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.85)

    axes[0, 1].plot(
        time,
        history["C_B_int_film"],
        label="quasi-steady Film",
        linewidth=1.5,
    )
    axes[0, 1].plot(
        time,
        state["C_B_int"],
        label="one-way thin-DtN",
        linewidth=1.5,
    )
    axes[0, 1].plot(
        time,
        np.asarray(fdm["concentrations"]["C_B"])[:, -1],
        label="FDM posterior",
        linewidth=1.2,
        alpha=0.8,
    )
    axes[0, 1].set_title("Thin interface concentration")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        np.asarray(fdm["theta"])[cv_indices],
        np.asarray(fdm["J"])[cv_indices],
        label="FDM",
        linewidth=2,
    )
    axes[1, 0].plot(
        np.asarray(fdm["theta"])[cv_indices],
        surface_current,
        label="one-way thin-DtN",
        linestyle="--",
    )
    axes[1, 0].plot(
        np.asarray(fdm["theta"])[cv_indices],
        conservative_current,
        label="DtN conservative",
        linestyle=":",
    )
    axes[1, 0].set_title("CV current")
    axes[1, 0].set_xlabel("theta")
    axes[1, 0].set_ylabel("J")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(time, closure_defect)
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title("J_film - k C_B,DtN C_C")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("reaction closure defect")
    axes[1, 1].grid(alpha=0.3)

    path = output_dir / "thin_dtn_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_mode_counts(text, max_modes):
    values = sorted({
        int(item.strip())
        for item in text.split(",")
        if item.strip()
    })
    values = [value for value in values if 1 <= value <= max_modes]
    if max_modes not in values:
        values.append(max_modes)
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-way finite-slab Dirichlet-to-Neumann prototype. ProductIntegral "
            "provides J(t); no FDM values enter the operator."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fdm-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--green-time-grid", type=int, default=256)
    parser.add_argument("--green-kernel-points", type=int, default=64)
    parser.add_argument("--max-modes", type=int, default=256)
    parser.add_argument("--mode-counts", default="16,32,64,128,256")
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x", type=int, default=120)
    parser.add_argument("--cv-points", type=int, default=2000)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.fdm_pkl = args.fdm_pkl.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    with args.fdm_pkl.open("rb") as handle:
        fdm = pickle.load(handle)
    history = load_product_integral_history(args, fdm, device)
    mu, boundary_sign, amplitudes = advance_mixed_boundary_modes(
        history["time"],
        history["C_B_surface"],
        history["J_rxn"],
        args.max_modes,
    )

    all_metrics = {}
    retained = None
    for n_modes in parse_mode_counts(args.mode_counts, args.max_modes):
        metrics = evaluate_mode_count(
            fdm,
            history,
            mu,
            boundary_sign,
            amplitudes,
            n_modes,
            args.n_time,
            args.n_x,
            args.cv_points,
        )
        all_metrics[str(n_modes)] = serializable_metrics(metrics)
        retained = metrics
        print(
            f"modes={n_modes:4d} "
            f"C_B={metrics['C_B']['rmse']:.6e} "
            f"C_B_int={metrics['C_B_int']['rmse']:.6e} "
            f"CV/Jref={metrics['CV_J_over_J_ref']['rmse']:.6e} "
            f"closure/Jref={metrics['reaction_closure_defect_over_J_ref_rmse']:.6e}"
        )

    report = {
        "prototype": "one_way_productintegral_flux_to_finite_slab_dtn",
        "self_consistent_reaction": False,
        "training_steps": 0,
        "fdm_role": "posterior_only",
        "checkpoint": str(args.checkpoint),
        "parameters": {
            "gamma": float(pinn.gamma),
            "k_cat": float(pinn.k_cat_star),
            "delta": float(pinn.delta),
            "D_B": float(pinn.D_rel_B),
        },
        "mode_results": all_metrics,
        "next_step": (
            "Replace the one-way ProductIntegral flux with a coupled current-cell "
            "Newton solve for C_B_int, C_D_int, and J."
        ),
    }
    report_path = args.output_dir / "thin_dtn_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure_path = plot_diagnostics(args.output_dir, fdm, history, retained)
    np.savez_compressed(
        args.output_dir / "thin_dtn_state.npz",
        time=history["time"],
        C_B_surface=history["C_B_surface"],
        C_B_int_film=history["C_B_int_film"],
        C_C_int=history["C_C_int"],
        J_rxn=history["J_rxn"],
        C_B_int_dtn=retained["_state"]["C_B_int"],
        J_surface_dtn=retained["_state"]["J_surface"],
        J_conservative_dtn=retained["_state"]["J_conservative"],
        reaction_closure_defect=retained["_closure_defect"],
    )
    print(f"Saved metrics: {report_path}")
    print(f"Saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
