import argparse
import csv
import gc
import json
import pickle
import time
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import compare_concentration_fields as compare_fields
import pinn_thin_layer_v9_6 as pinn


VARIANTS = {
    "full": {
        "correction_scale": 1.5,
        "surface_bubble_scale": 0.25,
    },
    "interior_only": {
        "correction_scale": 1.5,
        "surface_bubble_scale": 0.0,
    },
    "surface_only": {
        "correction_scale": 0.0,
        "surface_bubble_scale": 0.25,
    },
    "physics_only": {
        "correction_scale": 0.0,
        "surface_bubble_scale": 0.0,
    },
}

ARCHITECTURES = {
    "no_lift": "multiscale_film_tracegreen_productintegral",
    "lift": "multiscale_film_tracegreen_productintegral_lift",
}

SUMMARY_METRICS = {
    "C_A_rmse": ("C_A", "rmse"),
    "C_B_rmse": ("C_B", "rmse"),
    "C_B_int_rmse": ("C_B_int", "rmse"),
    "overall_dimensionless_rmse": ("overall_dimensionless", "rmse"),
    "CV_J_surface_rmse": ("CV_J_surface", "rmse"),
    "CV_J_conservative_rmse": ("CV_J_conservative", "rmse"),
    "CV_J_over_J_ref_rmse": ("CV_J_over_J_ref", "rmse"),
    "current_balance_rmse": ("CV_J_surface_vs_conservative", "rmse"),
}


def metric_value(metrics, path):
    value = metrics
    for key in path:
        value = value[key]
    return float(value)


def compare_namespace(args, architecture, variant, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_case_outputs = bool(args.save_case_outputs)
    return Namespace(
        fdm_pkl=str(args.fdm_pkl),
        checkpoint=str(args.checkpoint),
        gamma=args.gamma,
        k_cat_star=args.k_cat,
        zero_shot_fixed_reference=False,
        arch=architecture,
        green_time_grid=args.green_time_grid,
        green_kernel_points=args.green_kernel_points,
        lift_time_grid=args.lift_time_grid,
        green_detach_history=False,
        thin_correction_scale=variant["correction_scale"],
        thin_surface_bubble_scale=variant["surface_bubble_scale"],
        input_mode="normalized",
        n_time=args.n_time,
        n_x_in=args.n_x_in,
        n_x_out=args.n_x_out,
        batch_size=args.batch_size,
        device=args.device,
        output_json=str(output_dir / "metrics.json"),
        output_npz=str(output_dir / "fields.npz") if save_case_outputs else "",
        output_figure=str(output_dir / "residual.png") if save_case_outputs else "",
        output_current_figure=(
            str(output_dir / "current.png") if save_case_outputs else ""
        ),
        current_mode="both",
        cv_points=args.cv_points,
    )


def run_posterior_audit(args):
    results = {}
    for lift_name, architecture in ARCHITECTURES.items():
        results[lift_name] = {}
        for variant_name, variant in VARIANTS.items():
            case_dir = args.output_dir / lift_name / variant_name
            metrics_path = case_dir / "metrics.json"
            if args.reuse_existing and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                elapsed = 0.0
                print(f"Reusing {lift_name}/{variant_name}: {metrics_path}")
            else:
                print(f"\n=== {lift_name}/{variant_name} ===")
                started = time.perf_counter()
                metrics = compare_fields.compare(
                    compare_namespace(args, architecture, variant, case_dir)
                )
                elapsed = time.perf_counter() - started

            row = {
                "elapsed_seconds": elapsed,
                "architecture": architecture,
                **variant,
            }
            for name, path in SUMMARY_METRICS.items():
                row[name] = metric_value(metrics, path)
            results[lift_name][variant_name] = row

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for lift_name, variants in results.items():
        reference = variants["full"]
        for variant_name, row in variants.items():
            row["relative_to_full_percent"] = {
                metric: 100.0 * (row[metric] / reference[metric] - 1.0)
                for metric in SUMMARY_METRICS
                if reference[metric] != 0.0
            }

    return results


def write_summary(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "thin_network_ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = output_dir / "thin_network_ablation_summary.csv"
    fieldnames = [
        "lift",
        "variant",
        "architecture",
        "correction_scale",
        "surface_bubble_scale",
        "elapsed_seconds",
        *SUMMARY_METRICS.keys(),
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for lift_name, variants in results.items():
            for variant_name, row in variants.items():
                writer.writerow({
                    "lift": lift_name,
                    "variant": variant_name,
                    **{name: row[name] for name in fieldnames if name in row},
                })

    labels = list(VARIANTS)
    x = np.arange(len(labels))
    width = 0.36
    figure_metrics = [
        ("C_B_rmse", "Thin-field C_B RMSE", False),
        ("overall_dimensionless_rmse", "Overall dimensionless RMSE", False),
        ("CV_J_over_J_ref_rmse", "CV / Jref RMSE", True),
        ("current_balance_rmse", "Surface-conservative RMSE", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (metric, title, log_scale) in zip(axes.flat, figure_metrics):
        no_lift = [results["no_lift"][name][metric] for name in labels]
        lift = [results["lift"][name][metric] for name in labels]
        axis.bar(x - width / 2, no_lift, width, label="no lift", color="#4C78A8")
        axis.bar(x + width / 2, lift, width, label="inventory lift", color="#F58518")
        axis.set_xticks(x, labels, rotation=18)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        if log_scale:
            axis.set_yscale("log")
        axis.legend()
    figure_path = output_dir / "thin_network_ablation_summary.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_path, csv_path, figure_path


def load_full_model(args, device):
    with args.fdm_pkl.open("rb") as handle:
        fdm = pickle.load(handle)
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
    compare_fields.configure_thin_network_scales(
        model_thin,
        correction_scale=VARIANTS["full"]["correction_scale"],
        surface_bubble_scale=VARIANTS["full"]["surface_bubble_scale"],
    )
    model_thin.eval()
    return model_thin


def evaluate_network_components(model_thin, time_grid, s_grid, device, batch_size):
    tt, ss = np.meshgrid(time_grid, s_grid, indexing="ij")
    points = np.column_stack([
        tt.reshape(-1),
        (ss * float(pinn.delta)).reshape(-1),
    ]).astype("float32")
    raw_chunks = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.from_numpy(points[start:start + batch_size]).to(device)
            t_raw = batch[:, 0:1]
            x_raw = batch[:, 1:2]
            x_net = model_thin._network_inputs(t_raw, x_raw)
            raw_chunks.append(
                model_thin._raw_field(x_net, t_raw, x_raw).cpu().numpy()
            )
    raw = np.vstack(raw_chunks).reshape(len(time_grid), len(s_grid), 2)

    time_gate = (
        1.0 -
        np.exp(-np.maximum(time_grid, 0.0) / (0.05 * float(pinn.T_sim)))
    )[:, None]
    s = s_grid[None, :]
    endpoint_shape = s**2 * (1.0 - s)**2
    surface_shape = s * (1.0 - s)**2
    interior = (
        time_gate *
        float(model_thin.correction_scale) *
        endpoint_shape *
        np.tanh(raw[:, :, 0])
    )
    surface = (
        time_gate *
        float(model_thin.surface_bubble_scale) *
        surface_shape *
        np.tanh(raw[:, :, 1])
    )
    return raw, interior, surface, interior + surface


def modal_projection(field, s_grid, n_modes):
    coefficients = []
    for mode in range(1, n_modes + 1):
        basis = np.sqrt(2.0) * np.sin(mode * np.pi * s_grid)
        coefficients.append(np.trapezoid(field * basis[None, :], s_grid, axis=1))
    coefficients = np.stack(coefficients, axis=1)
    mode_energy = np.mean(coefficients**2, axis=0)
    total_energy = float(np.mean(np.trapezoid(field**2, s_grid, axis=1)))
    fractions = mode_energy / max(total_energy, 1e-30)
    return coefficients, mode_energy, fractions


def component_stats(field, time_grid, s_grid):
    rms_by_time = np.sqrt(np.trapezoid(field**2, s_grid, axis=1))
    peak_index = int(np.argmax(rms_by_time))
    reversal_index = int(np.argmin(np.abs(time_grid - 0.5 * float(pinn.T_sim))))
    return {
        "max_abs": float(np.max(np.abs(field))),
        "rms": float(np.sqrt(np.mean(field**2))),
        "peak_spatial_rms": float(rms_by_time[peak_index]),
        "peak_time": float(time_grid[peak_index]),
        "reversal_spatial_rms": float(rms_by_time[reversal_index]),
    }, rms_by_time


def run_modal_analysis(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_thin = load_full_model(args, device)
    time_grid = np.linspace(
        0.0,
        float(pinn.T_sim),
        args.mode_time_points,
        dtype=np.float64,
    )
    s_grid = np.linspace(0.0, 1.0, args.mode_x_points, dtype=np.float64)
    raw, interior, surface, total = evaluate_network_components(
        model_thin,
        time_grid,
        s_grid,
        device,
        args.batch_size,
    )
    coefficients, mode_energy, fractions = modal_projection(
        total,
        s_grid,
        args.mode_count,
    )
    stats = {}
    rms_curves = {}
    for name, field in {
        "interior_bubble": interior,
        "surface_bubble": surface,
        "total_correction": total,
    }.items():
        stats[name], rms_curves[name] = component_stats(field, time_grid, s_grid)

    cumulative = np.cumsum(fractions)
    checkpoints = {}
    for count in (1, 2, 4, 8, 16, 32):
        if count <= len(cumulative):
            checkpoints[str(count)] = float(cumulative[count - 1])

    tau0 = 4.0 * float(pinn.delta) ** 2 / (np.pi**2 * float(pinn.D_rel_B))
    report = {
        "checkpoint": str(args.checkpoint),
        "gamma": float(pinn.gamma),
        "k_cat": float(pinn.k_cat_star),
        "delta": float(pinn.delta),
        "thin_first_mode_timescale": tau0,
        "time_step": float(time_grid[1] - time_grid[0]),
        "components": stats,
        "dirichlet_sine_mode_energy_fraction": fractions.tolist(),
        "cumulative_mode_energy_fraction": cumulative.tolist(),
        "cumulative_checkpoints": checkpoints,
        "interpretation": (
            "Projection is diagnostic only. The learned correction vanishes at both "
            "thin-layer endpoints, so a Dirichlet sine basis is used."
        ),
    }
    output_dir = args.output_dir / "mode_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "thin_network_mode_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "thin_network_components.npz",
        time=time_grid,
        s=s_grid,
        raw=raw,
        interior=interior,
        surface=surface,
        total=total,
        coefficients=coefficients,
        mode_energy=mode_energy,
        mode_energy_fraction=fractions,
        interior_rms=rms_curves["interior_bubble"],
        surface_rms=rms_curves["surface_bubble"],
        total_rms=rms_curves["total_correction"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (name, field) in zip(
        axes.flat[:3],
        [
            ("Interior curvature bubble", interior),
            ("Surface-slope bubble", surface),
            ("Total learned correction", total),
        ],
    ):
        vmax = max(float(np.percentile(np.abs(field), 99.5)), 1e-8)
        image = axis.imshow(
            field.T,
            origin="lower",
            aspect="auto",
            extent=[time_grid[0], time_grid[-1], 0.0, 1.0],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axis.axvline(0.5 * float(pinn.T_sim), color="black", linestyle="--", alpha=0.5)
        axis.set_title(name)
        axis.set_xlabel("time")
        axis.set_ylabel("s = x / delta")
        fig.colorbar(image, ax=axis, shrink=0.85)

    axis = axes[1, 1]
    axis.plot(time_grid, rms_curves["interior_bubble"], label="interior")
    axis.plot(time_grid, rms_curves["surface_bubble"], label="surface")
    axis.plot(time_grid, rms_curves["total_correction"], label="total", linewidth=2)
    axis.axvline(0.5 * float(pinn.T_sim), color="black", linestyle="--", alpha=0.5)
    axis.set_title("Spatial RMS of learned correction")
    axis.set_xlabel("time")
    axis.set_ylabel("RMS")
    axis.grid(alpha=0.3)
    axis.legend()
    figure_path = output_dir / "thin_network_components.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    modes = np.arange(1, len(fractions) + 1)
    axis.bar(modes, fractions, color="#4C78A8", label="per-mode energy")
    axis.plot(modes, cumulative, color="#E45756", marker="o", label="cumulative")
    axis.set_xlabel("Dirichlet sine mode")
    axis.set_ylabel("Fraction of total correction energy")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    mode_figure_path = output_dir / "thin_network_mode_energy.png"
    fig.savefig(mode_figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return report_path, figure_path, mode_figure_path


def print_key_results(results):
    print("\n=== Thin-network contribution audit ===")
    for lift_name in ("no_lift", "lift"):
        print(f"\n{lift_name}")
        print(
            "variant | overall_dim | C_B | CV/Jref | "
            "surface-conservative"
        )
        for variant_name in VARIANTS:
            row = results[lift_name][variant_name]
            print(
                f"{variant_name:13s} | "
                f"{row['overall_dimensionless_rmse']:.6e} | "
                f"{row['C_B_rmse']:.6e} | "
                f"{row['CV_J_over_J_ref_rmse']:.6e} | "
                f"{row['current_balance_rmse']:.6e}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Posterior-only audit of the two learned thin-layer Hermite bubbles. "
            "FDM is used only after the checkpoint is frozen."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fdm-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--green-time-grid", type=int, default=256)
    parser.add_argument("--green-kernel-points", type=int, default=64)
    parser.add_argument("--lift-time-grid", type=int, default=4096)
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x-in", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=160)
    parser.add_argument("--cv-points", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="")
    parser.add_argument("--mode-time-points", type=int, default=2001)
    parser.add_argument("--mode-x-points", type=int, default=257)
    parser.add_argument("--mode-count", type=int, default=32)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--save-case-outputs", action="store_true")
    parser.add_argument("--skip-posterior", action="store_true")
    parser.add_argument("--skip-mode-analysis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.fdm_pkl = args.fdm_pkl.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.fdm_pkl.is_file():
        raise FileNotFoundError(args.fdm_pkl)

    results = None
    if not args.skip_posterior:
        results = run_posterior_audit(args)
        summary_paths = write_summary(results, args.output_dir)
        print_key_results(results)
        print("Summary files:")
        for path in summary_paths:
            print(f"  {path}")

    if not args.skip_mode_analysis:
        mode_paths = run_modal_analysis(args)
        print("Mode-analysis files:")
        for path in mode_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
