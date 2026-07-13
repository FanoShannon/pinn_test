"""Posterior-only FDM comparison for one joint k-gamma checkpoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_case(value):
    try:
        pair_text, path = value.split("=", 1)
        k_text, gamma_text = pair_text.split(",", 1)
        k_value = float(k_text)
        gamma_value = float(gamma_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected --case K,GAMMA=PATH") from exc
    if k_value <= 0.0 or gamma_value <= 0.0 or not path:
        raise argparse.ArgumentTypeError(
            "Expected positive K,GAMMA and a non-empty FDM path"
        )
    return k_value, gamma_value, path


def number_label(value):
    return f"{value:g}".replace(".", "p")


def case_label(k_value, gamma_value):
    return f"k_{number_label(k_value)}_gamma_{number_label(gamma_value)}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one kgparam checkpoint against posterior FDM cases."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--apply-inventory-lift", action="store_true")
    parser.add_argument("--zero-shot-fixed-reference", action="store_true")
    parser.add_argument("--disable-thin-network", action="store_true")
    parser.add_argument("--green-time-grid", type=int, default=256)
    parser.add_argument("--green-kernel-points", type=int, default=64)
    parser.add_argument("--lift-time-grid", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x-in", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=160)
    parser.add_argument("--cv-points", type=int, default=2000)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    architecture = (
        "multiscale_film_tracegreen_kgparam_lift"
        if args.apply_inventory_lift
        else "multiscale_film_tracegreen_kgparam"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_script = Path(__file__).with_name("compare_concentration_fields.py")
    all_metrics = {}

    for k_value, gamma_value, fdm_path in args.case:
        label = case_label(k_value, gamma_value)
        output_json = output_dir / f"{label}_metrics.json"
        print(
            f"Running posterior case k={k_value:g}, gamma={gamma_value:g} "
            f"with architecture={architecture}",
            flush=True,
        )
        command = [
            sys.executable, "-u", str(compare_script),
            "--arch", architecture,
            "--gamma", str(gamma_value),
            "--k-cat-star", str(k_value),
            "--input-mode", "normalized",
            "--checkpoint", args.checkpoint,
            "--fdm-pkl", fdm_path,
            "--green-time-grid", str(args.green_time_grid),
            "--green-kernel-points", str(args.green_kernel_points),
            "--lift-time-grid", str(args.lift_time_grid),
            "--batch-size", str(args.batch_size),
            "--n-time", str(args.n_time),
            "--n-x-in", str(args.n_x_in),
            "--n-x-out", str(args.n_x_out),
            "--cv-points", str(args.cv_points),
            "--output-json", str(output_json),
            "--output-figure", (
                "" if args.skip_figures
                else str(output_dir / f"{label}_residual_summary.png")
            ),
            "--output-npz", str(output_dir / f"{label}_fields.npz") if args.save_fields else "",
        ]
        if args.zero_shot_fixed_reference:
            command.append("--zero-shot-fixed-reference")
        if args.disable_thin_network:
            command.append("--disable-thin-network")
        subprocess.run(command, check=True)
        all_metrics[f"k={k_value:g},gamma={gamma_value:g}"] = json.loads(
            output_json.read_text(encoding="utf-8")
        )

    metric_paths = {
        "C_A_rmse": ("C_A", "rmse"),
        "C_B_rmse": ("C_B", "rmse"),
        "C_C_over_gamma_rmse": ("C_C_over_gamma", "rmse"),
        "C_D_over_gamma_rmse": ("C_D_over_gamma", "rmse"),
        "C_B_int_rmse": ("C_B_int", "rmse"),
        "C_C_int_over_gamma_rmse": ("C_C_int_over_gamma", "rmse"),
        "overall_dimensionless_rmse": ("overall_dimensionless", "rmse"),
        "CV_J_over_J_ref_rmse": ("CV_J_over_J_ref", "rmse"),
    }
    aggregate = {}
    for name, (group, field) in metric_paths.items():
        values = {
            key: float(metrics[group][field])
            for key, metrics in all_metrics.items()
            if group in metrics and field in metrics[group]
        }
        if values:
            worst_pair = max(values, key=values.get)
            aggregate[name] = {
                "mean": float(np.mean(list(values.values()))),
                "worst": values[worst_pair],
                "worst_pair": worst_pair,
            }

    summary = {
        "checkpoint": args.checkpoint,
        "architecture": architecture,
        "posterior_inventory_lift": bool(args.apply_inventory_lift),
        "thin_neural_correction": not bool(args.disable_thin_network),
        "cases": all_metrics,
        "aggregate": aggregate,
        "fdm_usage": "frozen_checkpoint_posterior_evaluation_only",
    }
    summary_path = output_dir / "kg_parameter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Saved joint summary to {summary_path}")


if __name__ == "__main__":
    main()
