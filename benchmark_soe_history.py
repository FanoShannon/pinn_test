import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import differentiable_productintegral_dtn as differentiable
import physical_model as physics
import productintegral_dtn as numpy_operator


def parse_int_list(text):
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or min(values) < 3:
        raise ValueError("Grid sizes must be integers of at least three")
    return values


def timed(callable_, repeats, synchronize=None):
    callable_()
    samples = []
    for _ in range(repeats):
        gc.collect()
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        result = callable_()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)
    return result, float(np.median(samples)), samples


def numpy_solve(n_time, backend, args):
    time_grid = np.linspace(0.0, physics.DEFAULT_SIMULATION_TIME, n_time)
    return numpy_operator.solve_coupled_operator(
        time_grid,
        gamma=args.gamma,
        k_cat=args.k_cat,
        n_modes=args.modes,
        newton_iterations=args.newton_iterations,
        delta=args.delta,
        history_backend=backend,
        history_near_cells=args.history_near_cells,
        history_soe_terms=args.history_soe_terms,
        history_soe_tolerance=args.history_soe_tolerance,
    )


def torch_solve(n_time, backend, args, device):
    time_grid = torch.linspace(
        0.0,
        physics.DEFAULT_SIMULATION_TIME,
        n_time,
        dtype=torch.float64,
        device=device,
    )
    log_delta = torch.tensor(
        np.log(args.delta),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    state = differentiable.solve_coupled_operator(
        time_grid,
        gamma=args.gamma,
        k_cat=args.k_cat,
        delta=torch.exp(log_delta),
        n_modes=args.modes,
        newton_iterations=args.newton_iterations,
        history_backend=backend,
        history_near_cells=args.history_near_cells,
        history_soe_terms=args.history_soe_terms,
        history_soe_tolerance=args.history_soe_tolerance,
    )
    objective = torch.mean(state["J_surface"][1:] ** 2)
    gradient = torch.autograd.grad(objective, log_delta)[0]
    return state, gradient


def parse_args():
    parser = argparse.ArgumentParser(
        description="FDM-free direct versus fast Abel history benchmark."
    )
    parser.add_argument("--numpy-grids", default="257,1025,4097,8193")
    parser.add_argument("--torch-grids", default="129,257,513,1025")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--k-cat", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.035)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--newton-iterations", type=int, default=12)
    parser.add_argument("--history-near-cells", type=int, default=16)
    parser.add_argument("--history-soe-terms", type=int, default=128)
    parser.add_argument("--history-soe-tolerance", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else None
    report = {
        "study": "soe_abel_history_benchmark",
        "fdm_used": False,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "configuration": {
            "gamma": args.gamma,
            "k_cat": args.k_cat,
            "delta": args.delta,
            "modes": args.modes,
            "history_near_cells": args.history_near_cells,
            "history_soe_terms": args.history_soe_terms,
            "history_soe_tolerance": args.history_soe_tolerance,
            "repeats": args.repeats,
        },
        "numpy": {},
        "torch_forward_backward": {},
    }

    for n_time in parse_int_list(args.numpy_grids):
        direct, direct_time, direct_samples = timed(
            lambda: numpy_solve(n_time, "direct", args),
            args.repeats,
        )
        fast, fast_time, fast_samples = timed(
            lambda: numpy_solve(n_time, "soe", args),
            args.repeats,
        )
        current_error = fast["J_rxn"] - direct["J_rxn"]
        current_span = max(float(np.ptp(direct["J_rxn"])), 1e-15)
        metrics = {
            "direct_seconds": direct_time,
            "soe_seconds": fast_time,
            "speedup": direct_time / fast_time,
            "direct_samples": direct_samples,
            "soe_samples": fast_samples,
            "J_max_abs_difference": float(np.max(np.abs(current_error))),
            "J_span_nrmse": float(
                np.sqrt(np.mean(current_error ** 2)) / current_span
            ),
            "C_C_int_max_abs_difference": float(np.max(np.abs(
                fast["C_C_int"] - direct["C_C_int"]
            ))),
        }
        report["numpy"][str(n_time)] = metrics
        print(
            f"NumPy n={n_time}: speedup={metrics['speedup']:.3f} "
            f"Jmax={metrics['J_max_abs_difference']:.3e}"
        )

    for n_time in parse_int_list(args.torch_grids):
        direct, direct_time, direct_samples = timed(
            lambda: torch_solve(n_time, "direct", args, device),
            args.repeats,
            synchronize,
        )
        fast, fast_time, fast_samples = timed(
            lambda: torch_solve(n_time, "soe", args, device),
            args.repeats,
            synchronize,
        )
        direct_state, direct_gradient = direct
        fast_state, fast_gradient = fast
        direct_current = direct_state["J_rxn"].detach()
        current_error = (
            fast_state["J_rxn"].detach() - direct_current
        ).detach()
        current_span = max(
            float(torch.max(direct_current) - torch.min(direct_current)),
            1e-15,
        )
        metrics = {
            "direct_seconds": direct_time,
            "soe_seconds": fast_time,
            "speedup": direct_time / fast_time,
            "direct_samples": direct_samples,
            "soe_samples": fast_samples,
            "J_max_abs_difference": float(torch.max(torch.abs(current_error))),
            "J_span_nrmse": float(
                torch.sqrt(torch.mean(current_error ** 2)) / current_span
            ),
            "log_delta_gradient_relative_difference": float(
                torch.abs(fast_gradient - direct_gradient)
                /torch.clamp(torch.abs(direct_gradient), min=1e-15)
            ),
        }
        report["torch_forward_backward"][str(n_time)] = metrics
        print(
            f"Torch n={n_time}: speedup={metrics['speedup']:.3f} "
            f"gradrel="
            f"{metrics['log_delta_gradient_relative_difference']:.3e}"
        )

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
