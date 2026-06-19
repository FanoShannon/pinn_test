import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pinn_thin_layer_v9_6 as pinn


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_models(arch, normalize_inputs, device):
    if arch == "legacy":
        model_thin = pinn.ThinLayerNet_v9_6(normalize_inputs=normalize_inputs)
        model_ext = pinn.ExternalNet_v9_6(pinn.gamma, normalize_inputs=normalize_inputs)
    elif arch == "multiscale":
        model_thin = pinn.ThinLayerNet_v9_6_Multiscale(normalize_inputs=normalize_inputs)
        model_ext = pinn.ExternalNet_v9_6_Multiscale(pinn.gamma, normalize_inputs=normalize_inputs)
    elif arch == "multiscale_hardbc":
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHardBC(normalize_inputs=normalize_inputs)
        model_ext = pinn.ExternalNet_v9_6_Multiscale(pinn.gamma, normalize_inputs=normalize_inputs)
    elif arch == "multiscale_hermite":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleHermite(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model_thin.to(device), model_ext.to(device)


def load_checkpoint(model_thin, model_ext, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_thin.load_state_dict(state["model_thin_state_dict"], strict=True)
    model_ext.load_state_dict(state["model_ext_state_dict"], strict=True)
    return state.get("epoch", None)


def select_indices(size, n_select):
    n_select = min(size, n_select)
    return np.unique(np.linspace(0, size - 1, n_select, dtype=int))


def predict_pair(model, t_eval, x_eval, device, batch_size):
    tt, xx = np.meshgrid(t_eval, x_eval, indexing="ij")
    points = np.column_stack([tt.ravel(), xx.ravel()]).astype("float32")
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.from_numpy(points[start:start + batch_size]).to(device)
            chunks.append(torch.cat(model(batch), dim=1).cpu().numpy())
    out = np.vstack(chunks)
    return (
        out[:, 0].reshape(len(t_eval), len(x_eval)),
        out[:, 1].reshape(len(t_eval), len(x_eval)),
    )


def field_metrics(pinn_field, fdm_field):
    diff = pinn_field - fdm_field
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((fdm_field - np.mean(fdm_field)) ** 2))
    fdm_range = float(np.max(fdm_field) - np.min(fdm_field))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "nrmse": float(rmse / fdm_range) if fdm_range > 0 else float("nan"),
        "fdm_min": float(np.min(fdm_field)),
        "fdm_max": float(np.max(fdm_field)),
        "pinn_min": float(np.min(pinn_field)),
        "pinn_max": float(np.max(pinn_field)),
    }


def compare(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalize_inputs = args.input_mode == "normalized"

    model_thin, model_ext = build_models(args.arch, normalize_inputs, device)
    epoch = load_checkpoint(model_thin, model_ext, args.checkpoint)

    with open(args.fdm_pkl, "rb") as handle:
        fdm = pickle.load(handle)

    t_fdm = np.asarray(fdm["t"], dtype=float)
    x_in_fdm = np.asarray(fdm["x_in"], dtype=float)
    x_out_fdm = np.asarray(fdm["x_out"], dtype=float)
    fdm_conc = fdm["concentrations"]

    t_idx = select_indices(len(t_fdm), args.n_time)
    x_in_idx = select_indices(len(x_in_fdm), args.n_x_in)
    x_out_idx = select_indices(len(x_out_fdm), args.n_x_out)

    t_eval = t_fdm[t_idx]
    x_in_eval = x_in_fdm[x_in_idx]
    x_out_eval = x_out_fdm[x_out_idx]

    fdm_eval = {
        "C_A": np.asarray(fdm_conc["C_A"], dtype=float)[np.ix_(t_idx, x_in_idx)],
        "C_B": np.asarray(fdm_conc["C_B"], dtype=float)[np.ix_(t_idx, x_in_idx)],
        "C_C": np.asarray(fdm_conc["C_C"], dtype=float)[np.ix_(t_idx, x_out_idx)],
        "C_D": np.asarray(fdm_conc["C_D"], dtype=float)[np.ix_(t_idx, x_out_idx)],
    }

    pinn_a, pinn_b = predict_pair(model_thin, t_eval, x_in_eval, device, args.batch_size)
    pinn_c, pinn_d = predict_pair(model_ext, t_eval, x_out_eval, device, args.batch_size)
    pinn_eval = {"C_A": pinn_a, "C_B": pinn_b, "C_C": pinn_c, "C_D": pinn_d}

    metrics = {name: field_metrics(pinn_eval[name], fdm_eval[name]) for name in ["C_A", "C_B", "C_C", "C_D"]}
    total_sq = sum(float(np.sum((pinn_eval[name] - fdm_eval[name]) ** 2)) for name in metrics)
    total_n = sum(pinn_eval[name].size for name in metrics)
    metrics["overall"] = {"rmse": float(np.sqrt(total_sq / total_n))}

    c_b_int_fdm = np.asarray(fdm_conc["C_B"], dtype=float)[np.ix_(t_idx, [len(x_in_fdm) - 1])].reshape(-1)
    c_c_int_fdm = np.asarray(fdm_conc["C_C"], dtype=float)[np.ix_(t_idx, [0])].reshape(-1)
    metrics["C_B_int"] = field_metrics(pinn_b[:, -1], c_b_int_fdm)
    metrics["C_C_int"] = field_metrics(pinn_c[:, 0], c_c_int_fdm)
    metrics["C_C_int_history_note"] = (
        "C_C_int is evaluated from the FDM concentration field C_C[:,0], "
        "not from the stored C_C_int history array, whose first element is uninitialized."
    )

    if args.output_figure:
        plot_residual_summary(
            args.output_figure,
            t_eval,
            x_in_eval,
            x_out_eval,
            fdm_eval,
            pinn_eval,
            metrics,
        )

    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            t=t_eval,
            x_in=x_in_eval,
            x_out=x_out_eval,
            **{f"fdm_{k}": v for k, v in fdm_eval.items()},
            **{f"pinn_{k}": v for k, v in pinn_eval.items()},
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {epoch}")
    print(f"Architecture: {args.arch}")
    print(f"Input mode: {args.input_mode}")
    print(f"Device: {device}")
    print("field,rmse,mae,max_abs,bias,r2,nrmse,fdm_min,fdm_max,pinn_min,pinn_max")
    for name in ["C_A", "C_B", "C_C", "C_D", "C_B_int", "C_C_int"]:
        row = metrics[name]
        print(
            f"{name},{row['rmse']:.8e},{row['mae']:.8e},{row['max_abs']:.8e},"
            f"{row['bias']:.8e},{row['r2']:.8e},{row['nrmse']:.8e},"
            f"{row['fdm_min']:.8e},{row['fdm_max']:.8e},"
            f"{row['pinn_min']:.8e},{row['pinn_max']:.8e}"
        )
    print(f"overall_rmse,{metrics['overall']['rmse']:.8e}")
    return metrics


def plot_residual_summary(path, t_eval, x_in_eval, x_out_eval, fdm_eval, pinn_eval, metrics):
    names = ["C_A", "C_B", "C_C", "C_D"]
    residuals = {name: pinn_eval[name] - fdm_eval[name] for name in names}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    ax = axes[0, 0]
    metric_names = ["C_A", "C_B", "C_C", "C_D", "C_B_int", "C_C_int"]
    rmse_values = [metrics[name]["rmse"] for name in metric_names]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2"]
    ax.bar(metric_names, rmse_values, color=colors)
    ax.set_ylabel("RMSE")
    ax.set_title("PINN - FDM residual size")
    ax.grid(axis="y", alpha=0.3)

    for axis, name in zip(axes.flat[1:5], names):
        x_eval = x_in_eval if name in ("C_A", "C_B") else x_out_eval
        vmax = np.nanpercentile(np.abs(residuals[name]), 99)
        vmax = max(vmax, 1e-8)
        image = axis.imshow(
            residuals[name].T,
            origin="lower",
            aspect="auto",
            extent=[t_eval[0], t_eval[-1], x_eval[0], x_eval[-1]],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axis.set_title(f"{name} residual")
        axis.set_xlabel("T")
        axis.set_ylabel("X")
        fig.colorbar(image, ax=axis, shrink=0.85)

    ax = axes[1, 1]
    c_b_int_residual = residuals["C_B"][:, -1]
    ax.plot(t_eval, c_b_int_residual, color="#54A24B", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("C_B_int residual")
    ax.set_xlabel("T")
    ax.set_ylabel("PINN - FDM")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    c_c_int_residual = residuals["C_C"][:, 0]
    ax.plot(t_eval, c_c_int_residual, color="#B279A2", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("C_C_int residual")
    ax.set_xlabel("T")
    ax.set_ylabel("PINN - FDM")
    ax.grid(alpha=0.3)

    axes[1, 3].axis("off")

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PINN C_A/C_B/C_C/C_D against exact downsampled FDM grid points."
    )
    parser.add_argument("--fdm-pkl", default="../FDM/thin_layer_catalytic_v41_fixed.pkl")
    parser.add_argument("--checkpoint", default="./pinn_thin_layer_catalytic_v9_6_best.pth")
    parser.add_argument("--arch", choices=["legacy", "multiscale", "multiscale_hardbc", "multiscale_hermite"], default="legacy")
    parser.add_argument("--input-mode", choices=["legacy", "normalized"], default="legacy")
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x-in", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-json", default="./concentration_compare_metrics.json")
    parser.add_argument("--output-npz", default="./concentration_compare_fields.npz")
    parser.add_argument("--output-figure", default="./concentration_residual_summary.png")
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
