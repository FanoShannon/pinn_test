"""Summarize a four-way posterior NN/lift ablation over joint k-gamma cases."""

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "trained_nn_no_lift",
    "zero_nn_no_lift",
    "trained_nn_lift",
    "zero_nn_lift",
)


def percent_improvement(reference, candidate):
    if reference == 0.0:
        return None
    return 100.0 * (reference - candidate) / reference


def parse_pair(label):
    pieces = dict(piece.split("=", 1) for piece in label.split(","))
    return float(pieces["k"]), float(pieces["gamma"])


def metric(metrics, group):
    return float(metrics[group]["rmse"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    args = parser.parse_args()
    root = Path(args.audit_dir)
    summaries = {}
    for variant in VARIANTS:
        path = root / variant / "kg_parameter_summary.json"
        summaries[variant] = json.loads(path.read_text(encoding="utf-8"))

    labels = set(summaries[VARIANTS[0]]["cases"])
    for variant in VARIANTS[1:]:
        labels &= set(summaries[variant]["cases"])
    if not labels:
        raise RuntimeError("The four audit variants have no common cases.")

    rows = {}
    for label in sorted(labels, key=lambda item: np.prod(parse_pair(item))):
        k_value, gamma_value = parse_pair(label)
        values = {}
        for variant in VARIANTS:
            case = summaries[variant]["cases"][label]
            values[variant] = {
                "overall_dimensionless_rmse": metric(case, "overall_dimensionless"),
                "CV_J_rmse": metric(case, "CV_J"),
                "CV_J_over_J_ref_rmse": metric(case, "CV_J_over_J_ref"),
                "CV_conservative_rmse": metric(case, "CV_J_conservative"),
                "surface_vs_conservative_rmse": metric(
                    case, "CV_J_surface_vs_conservative"
                ),
            }
        trained_lift = values["trained_nn_lift"]
        zero_lift = values["zero_nn_lift"]
        trained_raw = values["trained_nn_no_lift"]
        zero_raw = values["zero_nn_no_lift"]
        rows[label] = {
            "k": k_value,
            "gamma": gamma_value,
            "k_gamma": k_value * gamma_value,
            "J_ref": k_value * gamma_value
            / (1.0 + k_value * gamma_value * 0.035),
            "variants": values,
            "effects_percent": {
                "nn_with_lift_overall": percent_improvement(
                    zero_lift["overall_dimensionless_rmse"],
                    trained_lift["overall_dimensionless_rmse"],
                ),
                "nn_with_lift_CV_absolute": percent_improvement(
                    zero_lift["CV_J_rmse"], trained_lift["CV_J_rmse"]
                ),
                "nn_with_lift_CV_over_J_ref": percent_improvement(
                    zero_lift["CV_J_over_J_ref_rmse"],
                    trained_lift["CV_J_over_J_ref_rmse"],
                ),
                "nn_without_lift_CV_absolute": percent_improvement(
                    zero_raw["CV_J_rmse"], trained_raw["CV_J_rmse"]
                ),
                "lift_with_nn_CV_absolute": percent_improvement(
                    trained_raw["CV_J_rmse"], trained_lift["CV_J_rmse"]
                ),
                "lift_without_nn_CV_absolute": percent_improvement(
                    zero_raw["CV_J_rmse"], zero_lift["CV_J_rmse"]
                ),
            },
        }

    aggregate_all = {}
    for variant in VARIANTS:
        for metric_name in (
            "overall_dimensionless_rmse",
            "CV_J_rmse",
            "CV_J_over_J_ref_rmse",
        ):
            values = {
                label: row["variants"][variant][metric_name]
                for label, row in rows.items()
            }
            worst_label = max(values, key=values.get)
            aggregate_all[f"{variant}.{metric_name}"] = {
                "mean": float(np.mean(list(values.values()))),
                "worst": float(values[worst_label]),
                "worst_pair": worst_label,
            }

    low_rows = {
        label: row for label, row in rows.items() if row["k_gamma"] <= 0.1 + 1e-12
    }
    low_summary = {}
    for effect_name in next(iter(rows.values()))["effects_percent"]:
        values = [
            row["effects_percent"][effect_name]
            for row in low_rows.values()
            if row["effects_percent"][effect_name] is not None
        ]
        low_summary[effect_name] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    result = {
        "cases": rows,
        "aggregate_all_cases": aggregate_all,
        "low_k_gamma_threshold": 0.1,
        "low_k_gamma_effects_percent": low_summary,
        "interpretation": "Positive effect means the named component reduced posterior error.",
        "fdm_usage": "posterior_only_never_training_or_checkpoint_selection",
    }
    output = root / "kg_nn_contribution_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("k,gamma,k_gamma,NN_lift_CV_abs_pct,NN_lift_CV_Jref_pct,lift_noNN_CV_abs_pct")
    for row in rows.values():
        effect = row["effects_percent"]
        print(
            f"{row['k']:g},{row['gamma']:g},{row['k_gamma']:g},"
            f"{effect['nn_with_lift_CV_absolute']:.6g},"
            f"{effect['nn_with_lift_CV_over_J_ref']:.6g},"
            f"{effect['lift_without_nn_CV_absolute']:.6g}"
        )
    print(json.dumps(low_summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
