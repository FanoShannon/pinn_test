"""Posterior-only FDM comparison for one k-parameterized checkpoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_case(value):
    try:
        k_text, path = value.split("=", 1)
        k_value = float(k_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected --case K=PATH") from exc
    if k_value <= 0.0 or not path:
        raise argparse.ArgumentTypeError("Expected positive K and a non-empty FDM path")
    return k_value, path


def case_label(k_value):
    return f"k_{k_value:g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one kparam checkpoint against multiple posterior FDM cases."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="multiscale_film_tracegreen_kparam")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--green-time-grid", type=int, default=256)
    parser.add_argument("--green-kernel-points", type=int, default=64)
    parser.add_argument("--apply-inventory-lift", action="store_true",
                        help="Apply the posterior-only inventory Hermite current lift.")
    parser.add_argument("--lift-time-grid", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x-in", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=160)
    parser.add_argument("--cv-points", type=int, default=2000)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--zero-shot-fixed-reference", action="store_true")
    parser.add_argument("--k1-baseline-metrics", default="",
                        help="Optional parameter-scale-consistency clean k=1 metrics JSON.")
    parser.add_argument("--k1-historical-metrics", default="",
                        help="Optional older clean k=1 metrics JSON for historical regression reporting.")
    args = parser.parse_args()

    eval_arch = (
        "multiscale_film_tracegreen_kparam_lift"
        if args.apply_inventory_lift
        else args.arch
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_script = Path(__file__).with_name("compare_concentration_fields.py")
    all_metrics = {}

    for k_value, fdm_path in args.case:
        label = case_label(k_value)
        output_json = output_dir / f"{label}_metrics.json"
        print(
            f"Running posterior case k={k_value:g} "
            f"with architecture={eval_arch}",
            flush=True,
        )
        command = [
            sys.executable,
            "-u",
            str(compare_script),
            "--arch", eval_arch,
            "--gamma", str(args.gamma),
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
            "--output-figure", str(output_dir / f"{label}_residual_summary.png"),
            "--output-npz", str(output_dir / f"{label}_fields.npz") if args.save_fields else "",
        ]
        if args.zero_shot_fixed_reference:
            command.append("--zero-shot-fixed-reference")
        subprocess.run(command, check=True)
        all_metrics[f"{k_value:g}"] = json.loads(output_json.read_text(encoding="utf-8"))

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
            worst_k = max(values, key=values.get)
            aggregate[name] = {
                "mean": float(np.mean(list(values.values()))),
                "worst": values[worst_k],
                "worst_k": float(worst_k),
            }

    k1_regression = {}
    current_k1 = all_metrics.get("1")
    for label, path in (
        ("parameter_scale_clean", args.k1_baseline_metrics),
        ("historical_clean", args.k1_historical_metrics),
    ):
        if current_k1 is None or not path:
            continue
        baseline = json.loads(Path(path).read_text(encoding="utf-8"))
        comparisons = {}
        for name, (group, field) in metric_paths.items():
            if group not in current_k1 or group not in baseline:
                continue
            current_value = float(current_k1[group][field])
            baseline_value = float(baseline[group][field])
            ratio = current_value / max(abs(baseline_value), 1e-12)
            comparisons[name] = {
                "current": current_value,
                "baseline": baseline_value,
                "ratio": ratio,
                "within_5_percent": ratio <= 1.05,
            }
        k1_regression[label] = comparisons

    summary = {
        "checkpoint": args.checkpoint,
        "architecture": eval_arch,
        "posterior_inventory_lift": bool(args.apply_inventory_lift),
        "gamma": args.gamma,
        "cases": all_metrics,
        "aggregate": aggregate,
        "k1_regression": k1_regression,
        "fdm_usage": "posterior_evaluation_only",
    }
    summary_path = output_dir / "k_parameter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved multi-k posterior summary: {summary_path}")


if __name__ == "__main__":
    main()
