import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from analyze_inverse_identifiability import sensitivity_matrix
import physical_model as physics


PARAMETER_NAMES = ("k_cat", "gamma", "delta")
MODE_INDICES = {
    "all": (0, 1, 2),
    "fix-gamma": (0, 2),
    "fix-k": (1, 2),
    "fix-delta": (0, 1),
    "k-only": (0,),
    "gamma-only": (1,),
    "delta-only": (2,),
}


def parse_positive_values(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one positive value is required")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("All values must be finite and positive")
    return values


def parse_counts(text):
    counts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not counts or any(count < 1 for count in counts):
        raise ValueError("Combination sizes must be positive integers")
    return counts


def parse_modes(text):
    modes = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in MODE_INDICES]
    if not modes or invalid:
        raise ValueError(f"Invalid inverse modes: {invalid}")
    return modes


def parse_parameter_cases(text):
    cases = {}
    for item in text.split(";"):
        fields = [field.strip() for field in item.split(":")]
        if len(fields) != 4:
            raise ValueError(
                "Each case must be label:k:gamma:delta; "
                f"got {item!r}"
            )
        label = fields[0]
        values = tuple(float(value) for value in fields[1:])
        if not label or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError(f"Invalid parameter case: {item!r}")
        cases[label] = values
    if not cases:
        raise ValueError("At least one parameter case is required")
    return cases


def matrix_metrics(matrix, indices):
    selected = np.asarray(matrix, dtype=np.float64)[:, indices]
    singular_values = np.linalg.svd(selected, compute_uv=False)
    raw_condition = float(
        singular_values[0] / max(singular_values[-1], 1e-15)
    )
    norms = np.linalg.norm(selected, axis=0)
    normalized = selected / np.maximum(norms[None, :], 1e-15)
    shape_singular_values = np.linalg.svd(normalized, compute_uv=False)
    shape_condition = float(
        shape_singular_values[0] / max(shape_singular_values[-1], 1e-15)
    )
    correlation = normalized.T @ normalized
    if len(indices) > 1:
        off_diagonal = correlation[np.triu_indices(len(indices), k=1)]
        max_correlation = float(np.max(np.abs(off_diagonal)))
    else:
        max_correlation = 0.0
    return {
        "raw_condition_number": raw_condition,
        "column_normalized_condition_number": shape_condition,
        "max_absolute_sensitivity_correlation": max_correlation,
        "singular_values": [float(value) for value in singular_values],
        "column_norms": [float(value) for value in norms],
        "correlation": correlation.tolist(),
    }


def rank_combinations(blocks_by_case, candidates, count, mode, top_count):
    indices = MODE_INDICES[mode]
    ranked = []
    for combination in itertools.combinations(candidates, count):
        case_metrics = {}
        for label, blocks in blocks_by_case.items():
            matrix = np.vstack([blocks[sigma] for sigma in combination])
            case_metrics[label] = matrix_metrics(matrix, indices)
        shape_conditions = [
            value["column_normalized_condition_number"]
            for value in case_metrics.values()
        ]
        raw_conditions = [
            value["raw_condition_number"]
            for value in case_metrics.values()
        ]
        if len(indices) == 1:
            inverse_sensitivities = [
                1.0 / max(value["column_norms"][0], 1e-15)
                for value in case_metrics.values()
            ]
            ranking_score = float(max(inverse_sensitivities))
            ranking_metric = "worst_inverse_relative_sensitivity"
        else:
            ranking_score = float(max(shape_conditions))
            ranking_metric = "worst_column_normalized_condition_number"
        entry = {
            "scan_rates": [float(value) for value in combination],
            "ranking_metric": ranking_metric,
            "ranking_score": ranking_score,
            "worst_column_normalized_condition_number": float(
                max(shape_conditions)
            ),
            "geometric_mean_column_normalized_condition_number": float(
                np.exp(np.mean(np.log(shape_conditions)))
            ),
            "worst_raw_condition_number": float(max(raw_conditions)),
            "cases": case_metrics,
        }
        ranked.append(entry)
    ranked.sort(key=lambda item: (
        item["ranking_score"],
        item["geometric_mean_column_normalized_condition_number"],
    ))
    return ranked[:top_count]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FDM-free scan-rate design for joint k_cat/gamma/delta inverse "
            "problems using local differentiable sensitivities."
        )
    )
    parser.add_argument(
        "--candidate-sigmas",
        default="2.5,5,10,20,40,80,160,320,640",
    )
    parser.add_argument("--combination-sizes", default="2,3,4")
    parser.add_argument(
        "--inverse-modes",
        default="all,fix-gamma,fix-k,fix-delta",
    )
    parser.add_argument(
        "--parameter-cases",
        default="nominal:1:10:0.035",
        help="Semicolon-separated label:k:gamma:delta cases.",
    )
    parser.add_argument("--time-grid", type=int, default=129)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--history-backend",
        choices=("direct", "soe"),
        default="soe",
    )
    parser.add_argument("--history-near-cells", type=int, default=16)
    parser.add_argument("--history-soe-terms", type=int, default=128)
    parser.add_argument("--history-soe-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = sorted(set(parse_positive_values(args.candidate_sigmas)))
    counts = parse_counts(args.combination_sizes)
    modes = parse_modes(args.inverse_modes)
    parameter_cases = parse_parameter_cases(args.parameter_cases)
    if any(count > len(candidates) for count in counts):
        raise ValueError("Combination size exceeds number of candidate rates")
    if args.top < 1:
        raise ValueError("top must be positive")

    blocks_by_case = {}
    spans_by_case = {}
    for label, (k_cat, gamma, delta) in parameter_cases.items():
        blocks = {}
        spans = {}
        for sigma in candidates:
            duration = (
                2.0
                * abs(physics.THETA_INITIAL - physics.THETA_SWITCH)
                /sigma
            )
            time = torch.linspace(
                0.0,
                duration,
                args.time_grid,
                dtype=torch.float64,
            )
            current, jacobian = sensitivity_matrix(
                time,
                k_cat,
                gamma,
                delta,
                args.modes,
                args.history_backend,
                args.history_near_cells,
                args.history_soe_terms,
                args.history_soe_tolerance,
            )
            span = max(float(np.ptp(current)), 1e-15)
            blocks[sigma] = jacobian / span
            spans[sigma] = span
            print(
                f"case={label} sigma={sigma:g} span={span:.6g}",
                flush=True,
            )
        blocks_by_case[label] = blocks
        spans_by_case[label] = {
            f"{sigma:.12g}": float(span)
            for sigma, span in spans.items()
        }

    rankings = {}
    for mode in modes:
        rankings[mode] = {}
        for count in counts:
            entries = rank_combinations(
                blocks_by_case,
                candidates,
                count,
                mode,
                args.top,
            )
            rankings[mode][str(count)] = entries
            best = entries[0]
            print(
                f"mode={mode} scans={count} best={best['scan_rates']} "
                f"{best['ranking_metric']}={best['ranking_score']:.3f}",
                flush=True,
            )

    report = {
        "study": "fdm_free_multiscan_rate_design",
        "fdm_used": False,
        "neural_network_used": False,
        "parameter_order": list(PARAMETER_NAMES),
        "candidate_scan_rates": candidates,
        "parameter_cases": {
            label: {
                "k_cat": values[0],
                "gamma": values[1],
                "delta": values[2],
            }
            for label, values in parameter_cases.items()
        },
        "current_spans": spans_by_case,
        "resolution": {
            "time_grid": args.time_grid,
            "modes": args.modes,
            "history_backend": args.history_backend,
        },
        "ranking_rule": (
            "For two or three free parameters, minimize the worst-case "
            "column-normalized Jacobian condition across requested cases. "
            "For one free parameter, maximize the worst-case relative "
            "sensitivity norm."
        ),
        "rankings": rankings,
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
