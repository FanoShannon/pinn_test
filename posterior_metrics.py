"""Small posterior-only utilities with no dependency on the forward solver."""

import numpy as np


def select_indices(size, n_select):
    n_select = min(int(size), int(n_select))
    if n_select < 1:
        raise ValueError("n_select must be positive")
    return np.unique(np.linspace(0, size - 1, n_select, dtype=int))


def field_metrics(predicted, reference):
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: predicted={predicted.shape}, reference={reference.shape}"
        )
    difference = predicted - reference
    squared_residual = float(np.sum(difference ** 2))
    squared_total = float(np.sum((reference - np.mean(reference)) ** 2))
    reference_range = float(np.max(reference) - np.min(reference))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "bias": float(np.mean(difference)),
        "r2": (
            float(1.0 - squared_residual / squared_total)
            if squared_total > 0.0
            else float("nan")
        ),
        "nrmse": (
            float(rmse / reference_range)
            if reference_range > 0.0
            else float("nan")
        ),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "predicted_min": float(np.min(predicted)),
        "predicted_max": float(np.max(predicted)),
    }
