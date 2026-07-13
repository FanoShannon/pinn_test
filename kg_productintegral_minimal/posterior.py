#!/usr/bin/env python3
"""FDM posterior only. Every evaluation uses the zero-training inventory lift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .core import DELTA, X_EXT_MAX, load_base_checkpoint
except ImportError:
    from core import DELTA, X_EXT_MAX, load_base_checkpoint


def indices(size: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, size - 1, min(size, count), dtype=int))


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    span = float(np.ptp(truth))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(error))),
            "max_abs": float(np.max(np.abs(error))), "bias": float(np.mean(error)),
            "nrmse": rmse / span if span > 0 else math.nan}


def predict_grid(model, t: np.ndarray, x: np.ndarray, external: bool,
                 device: torch.device, batch_size: int) -> np.ndarray:
    tt, xx = np.meshgrid(t, x, indexing="ij")
    points = np.column_stack((tt.ravel(), xx.ravel())).astype("float32")
    first, second = [], []
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(points[start:start + batch_size]).to(device)
        tb, xb = batch[:, :1], batch[:, 1:2]
        with torch.no_grad():
            a, b = model.forward_external(tb, xb) if external else model.forward_thin(tb, xb)
        first.append(a.cpu().numpy().reshape(-1))
        second.append(b.cpu().numpy().reshape(-1))
    shape = (len(t), len(x))
    return np.concatenate(first).reshape(shape), np.concatenate(second).reshape(shape)


def infer_parameters(fdm: dict[str, Any], requested_k: float | None,
                     requested_gamma: float | None, fallback_k: float,
                     fallback_gamma: float) -> tuple[float, float]:
    params = fdm.get("params", {})
    stored_k = params.get("k_cat", params.get("k_cat_star"))
    stored_gamma = params.get("gamma")
    k = requested_k if requested_k is not None else (stored_k if stored_k is not None else fallback_k)
    gamma = requested_gamma if requested_gamma is not None else (stored_gamma if stored_gamma is not None else fallback_gamma)
    if stored_k is not None and not np.isclose(float(stored_k), float(k), rtol=1e-7, atol=1e-10):
        raise ValueError(f"FDM k mismatch: metadata={stored_k}, requested={k}")
    if stored_gamma is not None and not np.isclose(float(stored_gamma), float(gamma), rtol=1e-7, atol=1e-10):
        raise ValueError(f"FDM gamma mismatch: metadata={stored_gamma}, requested={gamma}")
    return float(k), float(gamma)


def evaluate_case(model, fdm_path: Path, requested_k: float | None,
                  requested_gamma: float | None, args: argparse.Namespace) -> dict[str, Any]:
    with fdm_path.open("rb") as handle:
        fdm = pickle.load(handle)
    k, gamma = infer_parameters(
        fdm, requested_k, requested_gamma, model.base_k, model.base_gamma
    )
    model.set_conditions(k, gamma)
    model.eval()

    t_all = np.asarray(fdm["t"], dtype=float)
    xin_all = np.asarray(fdm["x_in"], dtype=float)
    xout_all = np.asarray(fdm["x_out"], dtype=float)
    if not np.isclose(xout_all[-1], X_EXT_MAX, rtol=1e-6, atol=1e-8):
        raise ValueError(f"External domain mismatch: FDM x_max={xout_all[-1]}, expected {X_EXT_MAX}")
    ti, xi, xo = indices(len(t_all), args.n_time), indices(len(xin_all), args.n_x_in), indices(len(xout_all), args.n_x_out)
    t, xin, xout = t_all[ti], xin_all[xi], xout_all[xo]
    conc = fdm["concentrations"]
    truth = {
        "C_A": np.asarray(conc["C_A"])[np.ix_(ti, xi)],
        "C_B": np.asarray(conc["C_B"])[np.ix_(ti, xi)],
        "C_C": np.asarray(conc["C_C"])[np.ix_(ti, xo)],
        "C_D": np.asarray(conc["C_D"])[np.ix_(ti, xo)],
    }
    ca, cb = predict_grid(model, t, xin, False, args.device_obj, args.batch_size)
    cc, cd = predict_grid(model, t, xout, True, args.device_obj, args.batch_size)
    pred = {"C_A": ca, "C_B": cb, "C_C": cc, "C_D": cd}
    result: dict[str, Any] = {
        "fdm": str(fdm_path), "k": k, "gamma": gamma,
        "inventory_lift": True, "fdm_used_for_training": False,
    }
    for name in ("C_A", "C_B"):
        result[name] = metrics(pred[name], truth[name])
    for name in ("C_C", "C_D"):
        result[name + "_over_gamma"] = metrics(pred[name] / gamma, truth[name] / gamma)
    result["C_B_interface"] = metrics(cb[:, -1], np.asarray(conc["C_B"])[np.ix_(ti, [len(xin_all) - 1])].reshape(-1))
    result["C_C_interface_over_gamma"] = metrics(
        cc[:, 0] / gamma, np.asarray(conc["C_C"])[np.ix_(ti, [0])].reshape(-1) / gamma)
    errors = [ca - truth["C_A"], cb - truth["C_B"],
              (cc - truth["C_C"]) / gamma, (cd - truth["C_D"]) / gamma]
    result["overall_dimensionless_rmse"] = float(np.sqrt(sum(np.sum(e**2) for e in errors) / sum(e.size for e in errors)))

    if "J" in fdm:
        ji = indices(len(t_all), args.cv_points)
        jt = torch.from_numpy(t_all[ji].reshape(-1, 1).astype("float32")).to(args.device_obj)
        with torch.no_grad():
            j_pred = model.surface_current(jt).cpu().numpy().reshape(-1)
        j_truth = np.asarray(fdm["J"], dtype=float)[ji]
        j_ref = k * gamma / (1.0 + k * gamma * DELTA)
        result["CV_J"] = metrics(j_pred, j_truth)
        result["CV_J_over_J_ref"] = metrics(j_pred / j_ref, j_truth / j_ref)
    return result


def read_manifest(path: Path) -> list[tuple[float | None, float | None, Path]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Empty manifest")
        for row in reader:
            pkl = row.get("pkl") or row.get("fdm_pkl") or row.get("path")
            if not pkl:
                raise ValueError("Manifest needs a pkl, fdm_pkl, or path column")
            p = Path(pkl)
            if not p.is_absolute():
                p = path.parent / p
            kval = row.get("k_cat") or row.get("k") or row.get("k_cat_star")
            gval = row.get("gamma")
            rows.append((float(kval) if kval else None, float(gval) if gval else None, p))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fdm-pkl", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--k", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x-in", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=160)
    parser.add_argument("--cv-points", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.device_obj = torch.device(args.device)
    model, checkpoint = load_base_checkpoint(args.checkpoint, args.device_obj)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = ([(args.k, args.gamma, args.fdm_pkl)] if args.fdm_pkl
             else read_manifest(args.manifest))
    results = []
    for k, gamma, path in cases:
        print(f"Posterior with mandatory lift: k={k}, gamma={gamma}, FDM={path}", flush=True)
        result = evaluate_case(model, path, k, gamma, args)
        results.append(result)
        tag = f"k{result['k']:.8g}_gamma{result['gamma']:.8g}".replace(".", "p")
        (args.output_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  overall={result['overall_dimensionless_rmse']:.6e} "
              f"CV/Jref={result.get('CV_J_over_J_ref', {}).get('rmse', math.nan):.6e}", flush=True)

    tracked = ["overall_dimensionless_rmse", "CV_J_over_J_ref", "C_A", "C_B",
               "C_C_over_gamma", "C_D_over_gamma", "C_B_interface", "C_C_interface_over_gamma"]
    aggregate: dict[str, Any] = {}
    for name in tracked:
        values = [(r[name]["rmse"] if isinstance(r.get(name), dict) else r.get(name), r) for r in results if r.get(name) is not None]
        if values:
            worst = max(values, key=lambda pair: pair[0])
            aggregate[name] = {"mean": float(np.mean([v for v, _ in values])), "worst": float(worst[0]),
                               "worst_pair": {"k": worst[1]["k"], "gamma": worst[1]["gamma"]}}
    summary = {
        "checkpoint": str(args.checkpoint), "base_parameters": checkpoint["parameters"],
        "mode": "zero_training_productintegral_tracegreen_inventory_lift",
        "fdm_role": "posterior_only", "cases": results, "aggregate": aggregate,
    }
    output = args.output_dir / "posterior_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
