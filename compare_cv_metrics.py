import argparse

import numpy as np
import pandas as pd


def load_cv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df["theta"].to_numpy(dtype=float), df["J"].to_numpy(dtype=float)


def compare_cv(fdm_csv, pinn_csv):
    fdm_theta, fdm_j = load_cv(fdm_csv)
    pinn_theta, pinn_j = load_cv(pinn_csv)

    if len(pinn_j) != len(fdm_j):
        src = np.linspace(0.0, 1.0, len(pinn_j))
        dst = np.linspace(0.0, 1.0, len(fdm_j))
        pinn_j = np.interp(dst, src, pinn_j)
        pinn_theta = np.interp(dst, src, pinn_theta)

    diff = pinn_j - fdm_j
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((fdm_j - np.mean(fdm_j))**2))

    fdm_peak_idx = int(np.argmin(fdm_j))
    pinn_peak_idx = int(np.argmin(pinn_j))

    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "fdm_peak_j": float(fdm_j[fdm_peak_idx]),
        "fdm_peak_theta": float(fdm_theta[fdm_peak_idx]),
        "pinn_peak_j": float(pinn_j[pinn_peak_idx]),
        "pinn_peak_theta": float(pinn_theta[pinn_peak_idx]),
        "peak_j_error": float(pinn_j[pinn_peak_idx] - fdm_j[fdm_peak_idx]),
        "peak_theta_error": float(pinn_theta[pinn_peak_idx] - fdm_theta[fdm_peak_idx]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PINN CV output with FDM CV data for evaluation only.")
    parser.add_argument("--fdm", default="../FDM/kcat1_v42_cv_data_v42.csv")
    parser.add_argument("--pinn", default="./cv_theta_J_v9_6.csv")
    args = parser.parse_args()

    metrics = compare_cv(args.fdm, args.pinn)
    print("PINN vs FDM CV metrics (evaluation only)")
    print(f"R^2:             {metrics['r2']:.6f}")
    print(f"RMSE:            {metrics['rmse']:.6e}")
    print(f"MAE:             {metrics['mae']:.6e}")
    print(f"Max abs error:   {metrics['max_abs_error']:.6e}")
    print(f"FDM peak:        {metrics['fdm_peak_j']:.6f} @ theta={metrics['fdm_peak_theta']:.4f}")
    print(f"PINN peak:       {metrics['pinn_peak_j']:.6f} @ theta={metrics['pinn_peak_theta']:.4f}")
    print(f"Peak J error:    {metrics['peak_j_error']:.6f}")
    print(f"Peak theta err:  {metrics['peak_theta_error']:.6f}")


if __name__ == "__main__":
    main()
