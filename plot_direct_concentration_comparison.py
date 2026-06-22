import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


FIELD_SPECS = [
    ("C_A", "x_in"),
    ("C_B", "x_in"),
    ("C_C", "x_out"),
    ("C_D", "x_out"),
]


def parse_item(raw):
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(
            "--item must use label:fields.npz:metrics.json format, "
            f"got {raw!r}"
        )
    label, npz_path, metrics_path = parts
    return label, Path(npz_path), Path(metrics_path)


def load_item(raw):
    label, npz_path, metrics_path = parse_item(raw)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    data = np.load(npz_path)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "label": label,
        "npz_path": npz_path,
        "metrics_path": metrics_path,
        "data": data,
        "metrics": metrics,
    }


def nearest_indices(values, targets):
    values = np.asarray(values, dtype=float)
    indices = []
    for target in targets:
        indices.append(int(np.argmin(np.abs(values - target))))
    return indices


def metric_text(metrics, field):
    item = metrics.get(field, {})
    rmse = item.get("rmse", float("nan"))
    r2 = item.get("r2", float("nan"))
    bias = item.get("bias", float("nan"))
    return f"RMSE={rmse:.3g}, R2={r2:.3f}, bias={bias:.3g}"


def field_extent(t, x):
    return [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]


def plot_single_direct_fields(item, output_dir):
    data = item["data"]
    metrics = item["metrics"]
    label = item["label"]
    t = data["t"]

    fig, axes = plt.subplots(
        len(FIELD_SPECS),
        3,
        figsize=(15, 12),
        constrained_layout=True,
        sharey=False,
    )
    fig.suptitle(f"{label}: direct FDM/PINN concentration fields", fontsize=14)

    for row, (field, x_key) in enumerate(FIELD_SPECS):
        x = data[x_key]
        fdm = np.asarray(data[f"fdm_{field}"], dtype=float)
        pinn = np.asarray(data[f"pinn_{field}"], dtype=float)
        residual = pinn - fdm
        vmin = min(float(np.nanmin(fdm)), float(np.nanmin(pinn)))
        vmax = max(float(np.nanmax(fdm)), float(np.nanmax(pinn)))
        rmax = float(np.nanmax(np.abs(residual)))
        rnorm = TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax) if rmax > 0 else None

        panels = [
            ("FDM", fdm, "viridis", None, vmin, vmax),
            ("PINN", pinn, "viridis", None, vmin, vmax),
            ("PINN - FDM", residual, "coolwarm", rnorm, None, None),
        ]
        for col, (title, values, cmap, norm, local_vmin, local_vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                values,
                aspect="auto",
                origin="lower",
                extent=field_extent(t, x),
                cmap=cmap,
                norm=norm,
                vmin=local_vmin,
                vmax=local_vmax,
            )
            ax.set_title(f"{field} {title}")
            ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("t")
            cb = fig.colorbar(im, ax=ax, shrink=0.88)
            cb.ax.tick_params(labelsize=8)
        axes[row, 1].text(
            0.02,
            1.04,
            metric_text(metrics, field),
            transform=axes[row, 1].transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    path = output_dir / f"{label}_direct_fields.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_single_direct_curves(item, output_dir):
    data = item["data"]
    metrics = item["metrics"]
    label = item["label"]
    t = data["t"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    fig.suptitle(f"{label}: direct interface and CV curves", fontsize=14)

    curve_specs = [
        ("C_B_int", data["fdm_C_B"][:, -1], data["pinn_C_B"][:, -1], t, "t"),
        ("C_C_int", data["fdm_C_C"][:, 0], data["pinn_C_C"][:, 0], t, "t"),
    ]
    for ax, (name, fdm, pinn, xaxis, xlabel) in zip(axes[:2], curve_specs):
        ax.plot(xaxis, fdm, color="black", lw=2.0, label="FDM")
        ax.plot(xaxis, pinn, color="#d62728", lw=1.8, label="PINN")
        ax.set_title(f"{name}\n{metric_text(metrics, name)}")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    if {"cv_theta", "cv_fdm_J", "cv_pinn_J"}.issubset(set(data.files)):
        ax.plot(data["cv_theta"], data["cv_fdm_J"], color="black", lw=2.0, label="FDM")
        ax.plot(data["cv_theta"], data["cv_pinn_J"], color="#1f77b4", lw=1.8, label="PINN")
        ax.set_xlabel("theta")
        ax.set_title(f"CV J\n{metric_text(metrics, 'CV_J')}")
    else:
        ax.text(0.5, 0.5, "No CV data in NPZ", ha="center", va="center")
        ax.set_title("CV J")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    path = output_dir / f"{label}_direct_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_combined_c_c_profiles(items, output_dir, times):
    first = items[0]["data"]
    t = first["t"]
    x = first["x_out"]
    idx = nearest_indices(t, times)

    fig, axes = plt.subplots(
        len(idx),
        1,
        figsize=(9, max(3.0 * len(idx), 4.0)),
        constrained_layout=True,
        sharex=True,
    )
    if len(idx) == 1:
        axes = [axes]
    fig.suptitle("Direct C_C profiles: FDM vs checkpoint predictions", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(items), 1)))
    for ax, i_t in zip(axes, idx):
        ax.plot(x, first["fdm_C_C"][i_t], color="black", lw=2.4, label="FDM")
        for color, item in zip(colors, items):
            label = item["label"]
            c_c = item["data"]["pinn_C_C"][i_t]
            r2 = item["metrics"].get("C_C", {}).get("r2", float("nan"))
            ax.plot(x, c_c, lw=1.6, color=color, label=f"{label} (R2={r2:.3f})")
        ax.set_ylabel("C_C")
        ax.set_title(f"t={float(t[i_t]):.4f}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("x")
    axes[0].legend(loc="best", fontsize=8)

    path = output_dir / "combined_C_C_profiles_direct.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_metrics_csv(items, output_dir):
    fields = ["C_A", "C_B", "C_C", "C_D", "C_B_int", "C_C_int", "CV_J"]
    metric_names = ["rmse", "mae", "max_abs", "bias", "r2", "nrmse"]
    path = output_dir / "combined_metrics_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "field", *metric_names])
        for item in items:
            metrics = item["metrics"]
            for field in fields:
                entry = metrics.get(field)
                if not entry:
                    continue
                writer.writerow([item["label"], field, *[entry.get(name, "") for name in metric_names]])
    return path


def plot_metric_bars(items, output_dir):
    labels = [item["label"] for item in items]
    fields = ["C_C", "C_C_int", "CV_J"]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for i, field in enumerate(fields):
        r2_values = [item["metrics"].get(field, {}).get("r2", np.nan) for item in items]
        rmse_values = [item["metrics"].get(field, {}).get("rmse", np.nan) for item in items]
        axes[0].bar(x + (i - 1) * width, r2_values, width=width, label=field)
        axes[1].bar(x + (i - 1) * width, rmse_values, width=width, label=field)

    axes[0].set_title("R2 comparison")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_title("RMSE comparison")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    path = output_dir / "combined_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Create direct FDM/PINN comparison plots from compare_concentration_fields outputs."
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Comparison item as label:fields.npz:metrics.json. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated plots and CSV.")
    parser.add_argument(
        "--times",
        nargs="*",
        type=float,
        default=[0.25, 0.45, 0.55, 0.65, 0.85],
        help="Times used for combined C_C profile plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = [load_item(raw) for raw in args.item]
    generated = []
    for item in items:
        generated.append(plot_single_direct_fields(item, output_dir))
        generated.append(plot_single_direct_curves(item, output_dir))
    generated.append(plot_combined_c_c_profiles(items, output_dir, args.times))
    generated.append(plot_metric_bars(items, output_dir))
    generated.append(write_metrics_csv(items, output_dir))

    print("Generated:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
