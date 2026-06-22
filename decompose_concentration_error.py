import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_r2(pred, true):
    diff = pred - true
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def field_metrics(pred, true):
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    true_range = float(np.max(true) - np.min(true))
    true_std = float(np.std(true))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "r2": safe_r2(pred, true),
        "nrmse_range": float(rmse / true_range) if true_range > 0 else float("nan"),
        "nrmse_std": float(rmse / true_std) if true_std > 0 else float("nan"),
        "mse": mse,
        "true_mean": float(np.mean(true)),
        "true_std": true_std,
        "true_range": true_range,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


def metric_from_diff(diff, true):
    pred_zero_based = true + diff
    return field_metrics(pred_zero_based, true)


def select_zone_masks(x, near_frac, mid_frac, far_boundary_frac):
    x = np.asarray(x, dtype=float)
    y = x - x[0]
    length = float(x[-1] - x[0])
    if length <= 0:
        raise ValueError("x grid must be increasing and have nonzero length")
    yn = y / length
    masks = {
        "interface_point": np.arange(len(x)) == 0,
        f"near_0_{near_frac:.2f}L": (yn > 0.0) & (yn <= near_frac),
        f"mid_{near_frac:.2f}_{mid_frac:.2f}L": (yn > near_frac) & (yn <= mid_frac),
        f"bulk_{mid_frac:.2f}_1L": yn > mid_frac,
        f"far_last_{far_boundary_frac:.2f}L": yn >= far_boundary_frac,
        "far_boundary_point": np.arange(len(x)) == len(x) - 1,
    }
    return masks, yn


def zone_table(pred, true, x, masks):
    rows = []
    diff = pred - true
    total_sse = float(np.sum(diff ** 2))
    total_points = diff.size
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        zone_pred = pred[:, mask]
        zone_true = true[:, mask]
        zone_diff = zone_pred - zone_true
        metrics = field_metrics(zone_pred, zone_true)
        sse = float(np.sum(zone_diff ** 2))
        metrics.update({
            "zone": name,
            "x_min": float(np.min(x[mask])),
            "x_max": float(np.max(x[mask])),
            "n_x": int(np.sum(mask)),
            "n_points": int(zone_diff.size),
            "point_share": float(zone_diff.size / total_points),
            "sse_share": float(sse / total_sse) if total_sse > 0 else float("nan"),
        })
        rows.append(metrics)
    return rows


def time_group_table(pred, true, time_masks):
    rows = []
    diff = pred - true
    total_sse = float(np.sum(diff ** 2))
    total_points = diff.size
    for name, mask in time_masks.items():
        if not np.any(mask):
            continue
        group_pred = pred[mask, :]
        group_true = true[mask, :]
        group_diff = group_pred - group_true
        metrics = field_metrics(group_pred, group_true)
        sse = float(np.sum(group_diff ** 2))
        metrics.update({
            "group": name,
            "n_t": int(np.sum(mask)),
            "n_points": int(group_diff.size),
            "point_share": float(group_diff.size / total_points),
            "sse_share": float(sse / total_sse) if total_sse > 0 else float("nan"),
        })
        rows.append(metrics)
    return rows


def dynamic_time_masks(data, t, high_quantile, t_switch, k_cat_star):
    masks = {}
    dtheta_sign = np.where(t <= t_switch, -1.0, 1.0)
    masks["dtheta_negative"] = dtheta_sign < 0.0
    masks["dtheta_positive"] = dtheta_sign > 0.0

    if "pinn_C_B" not in data or "pinn_C_C" not in data:
        return masks, {}

    c_b_int = np.asarray(data["pinn_C_B"], dtype=float)[:, -1]
    c_c_int = np.asarray(data["pinn_C_C"], dtype=float)[:, 0]
    j_rxn = k_cat_star * c_b_int * c_c_int
    if len(t) >= 3:
        d_j = np.gradient(j_rxn, t, edge_order=2)
    elif len(t) == 2:
        d_j = np.gradient(j_rxn, t)
    else:
        d_j = np.zeros_like(j_rxn)

    masks["J_negative"] = j_rxn < 0.0
    masks["J_positive"] = j_rxn >= 0.0
    masks["dJ_negative"] = d_j < 0.0
    masks["dJ_positive"] = d_j >= 0.0
    threshold = float(np.quantile(np.abs(d_j), high_quantile)) if len(d_j) else 0.0
    masks[f"high_abs_dJ_q{high_quantile:.2f}"] = np.abs(d_j) >= threshold

    signals = {
        "C_B_int_pred": c_b_int.tolist(),
        "C_C_int_pred": c_c_int.tolist(),
        "J_rxn_pred": j_rxn.tolist(),
        "dJ_rxn_dt_pred": d_j.tolist(),
        "high_abs_dJ_threshold": threshold,
    }
    return masks, signals


def gradient_diagnostics(pred, true, x, masks):
    if pred.shape[1] < 3:
        return {}, []
    grad_pred = np.gradient(pred, x, axis=1, edge_order=2)
    grad_true = np.gradient(true, x, axis=1, edge_order=2)
    overall = field_metrics(grad_pred, grad_true)
    rows = zone_table(grad_pred, grad_true, x, masks)
    if pred.shape[1] > 1:
        dx0 = float(x[1] - x[0])
        pred_slope0 = (pred[:, 1] - pred[:, 0]) / dx0
        true_slope0 = (true[:, 1] - true[:, 0]) / dx0
        overall["interface_first_cell_slope"] = field_metrics(pred_slope0, true_slope0)
    return overall, rows


def bias_shape_decomposition(pred, true):
    diff = pred - true
    full = field_metrics(pred, true)
    interface_diff = diff[:, [0]]
    decomposed = {
        "full": full,
        "global_bias_removed": metric_from_diff(diff - np.mean(diff), true),
        "time_bias_removed": metric_from_diff(diff - np.mean(diff, axis=1, keepdims=True), true),
        "space_bias_removed": metric_from_diff(diff - np.mean(diff, axis=0, keepdims=True), true),
        "interface_error_removed": metric_from_diff(diff - interface_diff, true),
    }

    centered_true = true.reshape(-1)
    centered_pred = pred.reshape(-1)
    var_pred = float(np.var(centered_pred))
    if var_pred > 0:
        cov = float(np.mean((centered_pred - np.mean(centered_pred)) * (centered_true - np.mean(centered_true))))
        scale = cov / var_pred
        offset = float(np.mean(centered_true) - scale * np.mean(centered_pred))
        calibrated = scale * pred + offset
        decomposed["global_affine_calibrated"] = field_metrics(calibrated, true)
        decomposed["global_affine_scale"] = float(scale)
        decomposed["global_affine_offset"] = float(offset)
    return decomposed


def by_axis_profiles(pred, true, x, t):
    diff = pred - true
    rmse_x = np.sqrt(np.mean(diff ** 2, axis=0))
    mae_x = np.mean(np.abs(diff), axis=0)
    bias_x = np.mean(diff, axis=0)
    true_std_x = np.std(true, axis=0)
    r2_x = np.array([safe_r2(pred[:, i], true[:, i]) for i in range(pred.shape[1])])

    rmse_t = np.sqrt(np.mean(diff ** 2, axis=1))
    bias_t = np.mean(diff, axis=1)
    true_std_t = np.std(true, axis=1)
    r2_t = np.array([safe_r2(pred[i, :], true[i, :]) for i in range(pred.shape[0])])

    return {
        "x": x,
        "t": t,
        "rmse_x": rmse_x,
        "mae_x": mae_x,
        "bias_x": bias_x,
        "true_std_x": true_std_x,
        "r2_x": r2_x,
        "rmse_t": rmse_t,
        "bias_t": bias_t,
        "true_std_t": true_std_t,
        "r2_t": r2_t,
    }


def diagnose(field, zones, decomposition, gradient, time_groups=None):
    messages = []
    full = decomposition["full"]
    int_zone = next((row for row in zones if row["zone"] == "interface_point"), None)
    if int_zone and int_zone["r2"] >= 0.75 and full["r2"] < 0.8:
        messages.append(
            f"{field}: interface time trace is reasonably accurate, but the 2D field is still below R2=0.8."
        )

    full_rmse = full["rmse"]
    if full_rmse > 0:
        int_removed_ratio = decomposition["interface_error_removed"]["rmse"] / full_rmse
        space_removed_ratio = decomposition["space_bias_removed"]["rmse"] / full_rmse
        time_removed_ratio = decomposition["time_bias_removed"]["rmse"] / full_rmse
        if int_removed_ratio > 0.8:
            messages.append(
                "Removing the interface residual as a time-dependent offset barely reduces RMSE; "
                "the main error is spatial propagation/shape, not only C_int."
            )
        else:
            messages.append(
                "A large fraction of RMSE follows the interface residual; improving C_int/J_rxn may still help."
            )
        if space_removed_ratio < 0.75:
            messages.append(
                "Removing the persistent spatial bias profile reduces RMSE strongly; finite-domain shape correction is important."
            )
        if time_removed_ratio < 0.75:
            messages.append(
                "Removing per-time spatial mean reduces RMSE strongly; a time-dependent amplitude/offset mode is missing."
            )

    if zones:
        worst = max(zones, key=lambda row: row.get("sse_share", 0.0))
        if worst["sse_share"] > max(0.35, 1.5 * worst["point_share"]):
            messages.append(
                f"Error is concentrated in {worst['zone']} "
                f"(SSE share {worst['sse_share']:.2f}, point share {worst['point_share']:.2f})."
            )

    if gradient:
        grad_r2 = gradient.get("r2", float("nan"))
        slope = gradient.get("interface_first_cell_slope", {})
        slope_r2 = slope.get("r2", float("nan"))
        if np.isfinite(grad_r2) and grad_r2 < 0.5:
            messages.append(
                "Spatial gradient agreement is poor; C_C_int can be right while dC/dx and bulk propagation are wrong."
            )
        if np.isfinite(slope_r2) and slope_r2 < 0.5:
            messages.append(
                "The first-cell/interface slope is weak; flux consistency may be the bottleneck for full-field C_C."
            )

    for group in time_groups or []:
        if group["point_share"] < 0.999 and group["sse_share"] > 1.25 * group["point_share"]:
            messages.append(
                f"Time group {group['group']} is over-represented in the error "
                f"(SSE share {group['sse_share']:.2f}, point share {group['point_share']:.2f})."
            )

    if not messages:
        messages.append("No single dominant failure mode detected; compare zone and gradient tables directly.")
    return messages


def write_csv(path, field_reports):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "field", "table", "zone_or_group", "rmse", "mae", "bias", "r2", "nrmse_range",
            "true_std", "true_range", "n_x", "point_share", "sse_share", "x_min", "x_max"
        ])
        for field, report in field_reports.items():
            for table_name in ("zones", "gradient_zones"):
                for row in report.get(table_name, []):
                    writer.writerow([
                        field,
                        table_name,
                        row["zone"],
                        row["rmse"],
                        row["mae"],
                        row["bias"],
                        row["r2"],
                        row["nrmse_range"],
                        row["true_std"],
                        row["true_range"],
                        row["n_x"],
                        row["point_share"],
                        row["sse_share"],
                        row["x_min"],
                        row["x_max"],
                    ])
            for row in report.get("time_groups", []):
                writer.writerow([
                    field,
                    "time_groups",
                    row["group"],
                    row["rmse"],
                    row["mae"],
                    row["bias"],
                    row["r2"],
                    row["nrmse_range"],
                    row["true_std"],
                    row["true_range"],
                    "",
                    row["point_share"],
                    row["sse_share"],
                    "",
                    "",
                ])


def plot_report(path, field, report):
    profiles = report["profiles"]
    zones = report["zones"]
    decomp = report["decomposition"]
    x = np.asarray(profiles["x"])
    t = np.asarray(profiles["t"])
    x_norm = (x - x[0]) / (x[-1] - x[0])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    ax = axes[0, 0]
    labels = [row["zone"] for row in zones]
    rmse_vals = [row["rmse"] for row in zones]
    share_vals = [row["sse_share"] for row in zones]
    ax.bar(np.arange(len(labels)) - 0.18, rmse_vals, width=0.36, label="RMSE")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(labels)) + 0.18, share_vals, width=0.36, color="#E45756", alpha=0.65, label="SSE share")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("SSE share")
    ax.set_title(f"{field} spatial zones")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.plot(x_norm, profiles["rmse_x"], label="RMSE(x)", linewidth=2)
    ax.plot(x_norm, profiles["true_std_x"], label="FDM std_t(x)", linewidth=2, alpha=0.8)
    ax.set_title("Error vs distance")
    ax.set_xlabel("(x - interface) / L")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(x_norm, profiles["bias_x"], label="bias(x)", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Persistent spatial bias")
    ax.set_xlabel("(x - interface) / L")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    r2_x = np.asarray(profiles["r2_x"])
    ax.plot(x_norm, r2_x, linewidth=2)
    ax.axhline(0.8, color="#54A24B", linestyle="--", linewidth=1)
    ax.set_ylim(min(-0.2, np.nanmin(r2_x) - 0.05), 1.02)
    ax.set_title("R2 across time at each x")
    ax.set_xlabel("(x - interface) / L")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, profiles["rmse_t"], label="RMSE(t)", linewidth=2)
    ax.plot(t, np.abs(profiles["bias_t"]), label="|bias(t)|", linewidth=2, alpha=0.8)
    ax.set_title("Error over time")
    ax.set_xlabel("T")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    decomp_names = [
        "full",
        "global_bias_removed",
        "time_bias_removed",
        "space_bias_removed",
        "interface_error_removed",
    ]
    vals = [decomp[name]["rmse"] for name in decomp_names]
    x_pos = np.arange(len(decomp_names))
    ax.bar(x_pos, vals, color=["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#54A24B"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(decomp_names, rotation=35, ha="right")
    ax.set_title("What removes the RMSE?")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_report(args):
    data = np.load(args.input_npz)
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["x_out"], dtype=float)
    masks, yn = select_zone_masks(x, args.near_frac, args.mid_frac, args.far_boundary_frac)
    time_masks, dynamic_signals = dynamic_time_masks(
        data,
        t,
        high_quantile=args.high_dj_quantile,
        t_switch=args.t_switch,
        k_cat_star=args.k_cat_star,
    )

    field_reports = {}
    for field in args.fields:
        pred_key = f"pinn_{field}"
        fdm_key = f"fdm_{field}"
        if pred_key not in data or fdm_key not in data:
            raise KeyError(f"{args.input_npz} does not contain {pred_key}/{fdm_key}")
        pred = np.asarray(data[pred_key], dtype=float)
        true = np.asarray(data[fdm_key], dtype=float)
        if pred.shape != true.shape:
            raise ValueError(f"{field} shape mismatch: {pred.shape} vs {true.shape}")
        if pred.shape != (len(t), len(x)):
            raise ValueError(f"{field} shape {pred.shape} does not match t/x_out {(len(t), len(x))}")

        zones = zone_table(pred, true, x, masks)
        gradient, gradient_zones = gradient_diagnostics(pred, true, x, masks)
        time_groups = time_group_table(pred, true, time_masks)
        decomposition = bias_shape_decomposition(pred, true)
        profiles = by_axis_profiles(pred, true, x, t)

        report = {
            "field": field,
            "overall": field_metrics(pred, true),
            "zones": zones,
            "time_groups": time_groups,
            "gradient": gradient,
            "gradient_zones": gradient_zones,
            "decomposition": decomposition,
            "diagnosis": diagnose(field, zones, decomposition, gradient, time_groups=time_groups),
            "profiles": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in profiles.items()},
        }
        field_reports[field] = report

    output = {
        "input_npz": str(args.input_npz),
        "zone_definition": {
            "interface_x": float(x[0]),
            "far_x": float(x[-1]),
            "near_frac": args.near_frac,
            "mid_frac": args.mid_frac,
            "far_boundary_frac": args.far_boundary_frac,
        },
        "dynamic_group_definition": {
            "t_switch": args.t_switch,
            "high_dj_quantile": args.high_dj_quantile,
            "k_cat_star": args.k_cat_star,
        },
        "dynamic_signals": dynamic_signals,
        "fields": field_reports,
    }
    return output


def print_summary(report):
    print(f"Input: {report['input_npz']}")
    for field, item in report["fields"].items():
        overall = item["overall"]
        int_zone = next((row for row in item["zones"] if row["zone"] == "interface_point"), None)
        print(f"\n{field}")
        print("overall_rmse,r2,bias,nrmse_range")
        print(f"{overall['rmse']:.8e},{overall['r2']:.8e},{overall['bias']:.8e},{overall['nrmse_range']:.8e}")
        if int_zone:
            print("interface_rmse,r2,bias,nrmse_range")
            print(f"{int_zone['rmse']:.8e},{int_zone['r2']:.8e},{int_zone['bias']:.8e},{int_zone['nrmse_range']:.8e}")
        print("zone,rmse,r2,bias,sse_share,point_share,true_std,true_range")
        for row in item["zones"]:
            print(
                f"{row['zone']},{row['rmse']:.8e},{row['r2']:.8e},"
                f"{row['bias']:.8e},{row['sse_share']:.8e},{row['point_share']:.8e},"
                f"{row['true_std']:.8e},{row['true_range']:.8e}"
            )
        if item.get("time_groups"):
            print("time_group,rmse,r2,bias,sse_share,point_share,true_std,true_range")
            for row in item["time_groups"]:
                print(
                    f"{row['group']},{row['rmse']:.8e},{row['r2']:.8e},"
                    f"{row['bias']:.8e},{row['sse_share']:.8e},{row['point_share']:.8e},"
                    f"{row['true_std']:.8e},{row['true_range']:.8e}"
                )
        print("decomposition,rmse,ratio_to_full")
        full_rmse = item["decomposition"]["full"]["rmse"]
        for name in ["global_bias_removed", "time_bias_removed", "space_bias_removed", "interface_error_removed"]:
            rmse = item["decomposition"][name]["rmse"]
            ratio = rmse / full_rmse if full_rmse > 0 else float("nan")
            print(f"{name},{rmse:.8e},{ratio:.8e}")
        print("diagnosis")
        for message in item["diagnosis"]:
            print(f"- {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Decompose posterior PINN-vs-FDM concentration field errors by region, bias, and spatial gradients."
    )
    parser.add_argument("--input-npz", required=True, help="NPZ produced by compare_concentration_fields.py --output-npz")
    parser.add_argument("--fields", nargs="+", default=["C_C", "C_D"], help="External fields to analyze.")
    parser.add_argument("--near-frac", type=float, default=0.10, help="Near-interface region upper bound in normalized external length.")
    parser.add_argument("--mid-frac", type=float, default=0.30, help="Middle diffusion region upper bound in normalized external length.")
    parser.add_argument("--far-boundary-frac", type=float, default=0.90, help="Far-boundary region lower bound in normalized external length.")
    parser.add_argument("--t-switch", type=float, default=0.5, help="Current theta(t) switch time used for dtheta/dt grouping.")
    parser.add_argument("--k-cat-star", type=float, default=1.0, help="Reaction prefactor used for posterior J_rxn grouping.")
    parser.add_argument("--high-dj-quantile", type=float, default=0.75, help="Quantile threshold for high |dJ_rxn/dt| time group.")
    parser.add_argument("--output-json", default="", help="Optional JSON report path.")
    parser.add_argument("--output-csv", default="", help="Optional CSV zone table path.")
    parser.add_argument("--output-figure", default="", help="Optional PNG summary. Uses the first requested field.")
    args = parser.parse_args()

    report = make_report(args)
    print_summary(report)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, report["fields"])
    if args.output_figure:
        first_field = args.fields[0]
        plot_report(args.output_figure, first_field, report["fields"][first_field])


if __name__ == "__main__":
    main()
