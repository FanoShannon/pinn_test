"""
PINN for Thin-Layer Electrocatalytic Model - v9.6 (界面耦合增强版)
在 v9.5 基础上：
1. 保持通量符号不变（经分析，符号与 FDM 等价，只是定义不同）
2. 大幅增加界面耦合权重: 50 → 200
3. 增加界面附近采样点: max(8000, n_points)
4. 增加远场采样: n_points/5
5. 监控界面浓度 C_B(δ) 和 C_C(δ)
"""

import matplotlib
import pandas as pd
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json
import pickle
import subprocess
from typing import Tuple, Dict
import time

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

# ==================== 模型参数 ====================
sigma = 40
theta_i = 10
theta_switch = -10
T_sim = 2 * abs(theta_i - theta_switch) / sigma
T_switch = 0.5
lambda_factor = 0.035
delta = lambda_factor * np.sqrt(T_sim)

X_ext_factor = 6
X_ext_max = delta + X_ext_factor * np.sqrt(T_sim)

REFERENCE_K_CAT_STAR = 1.0
REFERENCE_GAMMA = 10.0
KPARAM_REFERENCE = 1.0
KPARAM_ANCHORS = (0.1, 1.0, 10.0)
KPARAM_VALIDATION_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
KPARAM_LOG10_STD = 1.25
KPARAM_LOG10_SCALE = 1.0
KPARAM_SUPPORT = "positive_finite"
KPARAM_PARAMETERIZATION = "k_cat_log10"
KPARAM_INTERFACE_OPERATOR = "product_integral_piecewise_linear_newton_v2"
GAMMAPARAM_REFERENCE = 10.0
GAMMAPARAM_ANCHORS = (0.1, 1.0, 100.0)
GAMMAPARAM_VALIDATION_VALUES = (0.1, 0.316227766, 1.0, 3.16227766, 10.0, 31.6227766, 100.0)
GAMMAPARAM_CALIBRATION_RANGE = (0.1, 100.0)
GAMMAPARAM_SUPPORT = "positive_finite"
GAMMAPARAM_PARAMETERIZATION = "gamma_log10"
GAMMAPARAM_INTERFACE_OPERATOR = "normalized_product_integral_piecewise_linear_newton_v1"

k_cat_star = REFERENCE_K_CAT_STAR
gamma = REFERENCE_GAMMA

D_rel_A = 1.0
D_rel_B = 1.0
D_rel_C = 1.0
D_rel_D = 1.0


def configure_physical_parameters(gamma_value=REFERENCE_GAMMA,
                                  k_cat_value=REFERENCE_K_CAT_STAR):
    """Set one fixed physical parameter pair before model construction."""
    global gamma, k_cat_star
    gamma_value = float(gamma_value)
    k_cat_value = float(k_cat_value)
    if not np.isfinite(gamma_value) or gamma_value <= 0.0:
        raise ValueError(f"gamma must be finite and positive, got {gamma_value}")
    if not np.isfinite(k_cat_value) or k_cat_value <= 0.0:
        raise ValueError(f"k_cat_star must be finite and positive, got {k_cat_value}")
    gamma = gamma_value
    k_cat_star = k_cat_value


def characteristic_reaction_flux(gamma_value=None, k_cat_value=None):
    """Film-limited characteristic flux for scale-consistent residuals."""
    gamma_value = gamma if gamma_value is None else float(gamma_value)
    k_cat_value = k_cat_star if k_cat_value is None else float(k_cat_value)
    return (k_cat_value * gamma_value) / (
        1.0 + k_cat_value * delta * gamma_value / D_rel_B
    )


def external_residual_scale(gamma_value=None):
    """Reference-calibrated 1/gamma scale; equals one at gamma=10."""
    gamma_value = gamma if gamma_value is None else float(gamma_value)
    return REFERENCE_GAMMA / max(gamma_value, 1e-12)


def flux_residual_scale(gamma_value=None, k_cat_value=None):
    """Reference-calibrated 1/J_ref scale; equals one at the v9.6 baseline."""
    current = characteristic_reaction_flux(gamma_value, k_cat_value)
    reference = characteristic_reaction_flux(REFERENCE_GAMMA, REFERENCE_K_CAT_STAR)
    return reference / max(current, 1e-12)


def parameter_condition_cache_key(interface_state):
    if interface_state is None:
        return None
    getter = getattr(interface_state, "condition_cache_key", None)
    return getter() if getter is not None else None


def model_active_k_cat(model_ext=None, model_thin=None):
    for model in (model_ext, model_thin):
        if model is None:
            continue
        getter = getattr(model, "active_k_cat", None)
        if getter is not None:
            return float(getter())
        interface_state = getattr(model, "interface_state", None)
        getter = getattr(interface_state, "active_k_cat", None)
        if getter is not None:
            return float(getter())
    return float(k_cat_star)


def set_model_k_cat(model_thin, model_ext, value):
    setter = getattr(model_ext, "set_k_cat", None)
    if setter is None:
        setter = getattr(model_thin, "set_k_cat", None)
    if setter is None:
        raise TypeError("This architecture does not support runtime k_cat conditioning")
    setter(value)


def model_active_gamma(model_ext=None, model_thin=None):
    for model in (model_ext, model_thin):
        if model is None:
            continue
        getter = getattr(model, "active_gamma", None)
        if getter is not None:
            return float(getter())
        interface_state = getattr(model, "interface_state", None)
        getter = getattr(interface_state, "active_gamma", None)
        if getter is not None:
            return float(getter())
    return float(gamma)


def set_model_gamma(model_thin, model_ext, value):
    setter = getattr(model_ext, "set_gamma", None)
    if setter is None:
        setter = getattr(model_thin, "set_gamma", None)
    if setter is None:
        raise TypeError("This architecture does not support runtime gamma conditioning")
    setter(value)
    clear_lift = getattr(model_thin, "clear_lift_cache", None)
    if clear_lift is not None:
        clear_lift()


def activate_k_from_input(interface_state, x_input):
    """Read one physical k_cat condition from [T, X, k_cat]."""
    if x_input.shape[1] < 3:
        return x_input[:, :2]
    k_values = x_input[:, 2:3]
    if not torch.isfinite(k_values).all():
        raise ValueError("k_cat input contains NaN or Inf")
    k_value = float(k_values[0].detach().cpu())
    if not torch.allclose(
        k_values,
        torch.ones_like(k_values) * k_values[0:1],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "Product-integral kparam batches must contain one shared k_cat value"
        )
    interface_state.set_k_cat(k_value)
    return x_input[:, :2]


def activate_gamma_from_input(interface_state, x_input):
    """Read one physical gamma condition from [T, X, gamma]."""
    if x_input.shape[1] < 3:
        return x_input[:, :2]
    gamma_values = x_input[:, 2:3]
    if not torch.isfinite(gamma_values).all():
        raise ValueError("gamma input contains NaN or Inf")
    gamma_value = float(gamma_values[0].detach().cpu())
    if not torch.allclose(
        gamma_values,
        torch.ones_like(gamma_values) * gamma_values[0:1],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "Product-integral gammaparam batches must contain one shared gamma value"
        )
    interface_state.set_gamma(gamma_value)
    return x_input[:, :2]


def conditioned_model_inputs(model, T_raw, X_raw, condition_value=None):
    inputs = torch.cat([T_raw, X_raw], dim=1)
    if getattr(model, "k_parameterized", False):
        if condition_value is None:
            condition_value = model_active_k_cat(model_thin=model)
        return torch.cat([inputs, torch.ones_like(T_raw) * float(condition_value)], dim=1)
    if getattr(model, "gamma_parameterized", False):
        if condition_value is None:
            condition_value = model_active_gamma(model_thin=model)
        return torch.cat([inputs, torch.ones_like(T_raw) * float(condition_value)], dim=1)
    return inputs


def is_k_parameterized(model_ext=None, model_thin=None):
    return bool(
        getattr(model_ext, "k_parameterized", False) or
        getattr(model_thin, "k_parameterized", False)
    )


def is_gamma_parameterized(model_ext=None, model_thin=None):
    return bool(
        getattr(model_ext, "gamma_parameterized", False) or
        getattr(model_thin, "gamma_parameterized", False)
    )


def parameterization_metadata(model_ext):
    if is_gamma_parameterized(model_ext=model_ext):
        interface_state = getattr(model_ext, "interface_state", None)
        metadata = {
            "parameterization": GAMMAPARAM_PARAMETERIZATION,
            "gamma_support": GAMMAPARAM_SUPPORT,
            "gamma_reference": float(getattr(model_ext, "gamma_reference", GAMMAPARAM_REFERENCE)),
            "gamma_calibration_range": list(GAMMAPARAM_CALIBRATION_RANGE),
            "gamma_condition_transform": "tanh(log10(gamma/gamma_reference))",
            "gamma_sampling": "one_gamma_per_step_60pct_anchors_40pct_log_uniform",
            "gamma_interface_operator": getattr(
                interface_state, "gammaparam_interface_operator", GAMMAPARAM_INTERFACE_OPERATOR
            ),
            "product_integral_fixed_point_iterations": int(
                getattr(interface_state, "fixed_point_iterations", 2)
            ),
            "product_integral_newton_projection_iterations": int(
                getattr(interface_state, "newton_projection_iterations", 4)
            ),
        }
        metadata.update(getattr(model_ext, "gamma_sampling_metadata", {}))
        return metadata
    if not is_k_parameterized(model_ext=model_ext):
        return {}
    metadata = {
        "parameterization": KPARAM_PARAMETERIZATION,
        "k_support": KPARAM_SUPPORT,
        "k_reference": float(getattr(model_ext, "k_reference", KPARAM_REFERENCE)),
        "k_condition_transform": "tanh(log10(k/k_reference))",
        "k_sampling": "one_k_per_step_log10_normal_with_anchors",
    }
    interface_state = getattr(model_ext, "interface_state", None)
    operator = getattr(interface_state, "kparam_interface_operator", None)
    if operator is not None:
        metadata["k_interface_operator"] = operator
        metadata["product_integral_fixed_point_iterations"] = int(
            getattr(interface_state, "fixed_point_iterations", 2)
        )
        metadata["product_integral_newton_projection_iterations"] = int(
            getattr(interface_state, "newton_projection_iterations", 1)
        )
    metadata.update(getattr(model_ext, "k_sampling_metadata", {}))
    return metadata


def smooth_bounded_concentration(value, upper, transition_fraction=0.01,
                                 transition_gate=None):
    """Map a concentration smoothly to ``(0, upper)`` without a hard clamp.

    For values well inside the interval, softplus(value/tau) and
    softplus((upper-value)/tau) reduce to value/tau and
    (upper-value)/tau, so the map is approximately the identity.  Outside the
    interval it approaches the nearest bound while retaining a useful gradient.
    """
    upper_t = torch.as_tensor(upper, device=value.device, dtype=value.dtype)
    gate = 1.0 if transition_gate is None else transition_gate
    tau = torch.clamp(
        upper_t * float(transition_fraction) * gate,
        min=upper_t * 1e-7,
    )
    positive_value = F.softplus(value / tau)
    positive_remaining = F.softplus((upper_t - value) / tau)
    fraction = positive_value / torch.clamp(
        positive_value + positive_remaining,
        min=torch.finfo(value.dtype).eps,
    )
    return upper_t * fraction


def print_physical_configuration():
    print("=== PINN v9.6 - Interface Coupling Enhancement ===")
    print(f"Thin layer thickness delta = {delta:.4f}")
    print(f"External region length = {X_ext_max - delta:.4f}")
    print(f"Catalytic constant k_cat* = {k_cat_star}, gamma = {gamma}")
    print(f"Characteristic film flux J_ref = {characteristic_reaction_flux():.6g}")
    print(f"Film Damkohler number = {k_cat_star * gamma * delta / D_rel_B:.6g}")
    print(f"Simulation time T_sim = {T_sim:.4f}")

# ==================== 模型路径 ====================
MODEL_V95_BEST_PATH = './pinn_thin_layer_catalytic_v9_5_best.pth'
MODEL_V96_PATH = './pinn_thin_layer_catalytic_v9_6.pth'
MODEL_V96_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_best.pth'
MODEL_V96_MULTISCALE_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale.pth'
MODEL_V96_MULTISCALE_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_best.pth'
MODEL_V96_MULTISCALE_HARDBC_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_hardbc.pth'
MODEL_V96_MULTISCALE_HARDBC_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_hardbc_best.pth'
MODEL_V96_MULTISCALE_HERMITE_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_hermite.pth'
MODEL_V96_MULTISCALE_HERMITE_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_hermite_best.pth'
MODEL_V96_MULTISCALE_HERMITE_EXTBASIS_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_hermite_extbasis.pth'
MODEL_V96_MULTISCALE_HERMITE_EXTBASIS_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_hermite_extbasis_best.pth'
MODEL_V96_MULTISCALE_GREEN_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green.pth'
MODEL_V96_MULTISCALE_GREEN_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_HYBRID_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_hybrid.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_HYBRID_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_hybrid_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_STAGE1_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_STAGE1_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_dynamic_stage1_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_interface_memory.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_interface_memory_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_memory.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_memory_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSAL_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causal.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSAL_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causal_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALCONV_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalconv.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALCONV_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalconv_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_SMOOTH_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_SMOOTH_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_INTMEMORY_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_INTMEMORY_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_FLUXTRACE_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_fluxtrace.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_FLUXTRACE_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_fluxtrace_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_tracegreen.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_tracegreen_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MATCHEDABEL_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MATCHEDABEL_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MIXEDABEL_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MIXEDABEL_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_LIFT_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_lift.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_LIFT_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_productintegral_lift_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_CONSERVATIVE_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_conservative.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_CONSERVATIVE_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_conservative_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_MIXEDFLUX_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_mixedflux.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_MIXEDFLUX_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_clean_mixedflux_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CONSERVATIVE_LIFT_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_conservative_lift.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CONSERVATIVE_LIFT_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_conservative_lift_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_KPARAM_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_kparam.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_KPARAM_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_kparam_best.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_GAMMAPARAM_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_gammaparam.pth'
MODEL_V96_MULTISCALE_FILM_TRACEGREEN_GAMMAPARAM_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_film_tracegreen_gammaparam_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_EMA_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_ema.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_EMA_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_ema_best.pth'
MODEL_V96_MULTISCALE_BUFFER_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_buffer.pth'
MODEL_V96_MULTISCALE_BUFFER_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_buffer_best.pth'
MODEL_V96_MULTISCALE_FLUXBUFFER_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_fluxbuffer.pth'
MODEL_V96_MULTISCALE_FLUXBUFFER_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_fluxbuffer_best.pth'
MODEL_V96_MULTISCALE_HIS_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_his.pth'
MODEL_V96_MULTISCALE_HIS_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_his_best.pth'
MODEL_V96_HIS_PINN_PATH = './pinn_thin_layer_catalytic_v9_6_his_pinn.pth'
MODEL_V96_HIS_PINN_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_his_pinn_best.pth'
MODEL_V96_HIS_PINN_EXT_PATH = './pinn_thin_layer_catalytic_v9_6_his_pinn_ext.pth'
MODEL_V96_HIS_PINN_EXT_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_his_pinn_ext_best.pth'

USE_NORMALIZED_COORDS = True


def resolve_checkpoint_paths(arch, checkpoint_dir=None):
    if arch == "multiscale":
        current_path = MODEL_V96_MULTISCALE_PATH
        best_path = MODEL_V96_MULTISCALE_BEST_PATH
    elif arch == "multiscale_hardbc":
        current_path = MODEL_V96_MULTISCALE_HARDBC_PATH
        best_path = MODEL_V96_MULTISCALE_HARDBC_BEST_PATH
    elif arch == "multiscale_hermite":
        current_path = MODEL_V96_MULTISCALE_HERMITE_PATH
        best_path = MODEL_V96_MULTISCALE_HERMITE_BEST_PATH
    elif arch == "multiscale_hermite_extbasis":
        current_path = MODEL_V96_MULTISCALE_HERMITE_EXTBASIS_PATH
        best_path = MODEL_V96_MULTISCALE_HERMITE_EXTBASIS_BEST_PATH
    elif arch == "multiscale_green":
        current_path = MODEL_V96_MULTISCALE_GREEN_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_BEST_PATH
    elif arch == "multiscale_green_grid":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_BEST_PATH
    elif arch == "multiscale_green_grid_hybrid":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_HYBRID_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_HYBRID_BEST_PATH
    elif arch == "multiscale_green_grid_dynamic":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_BEST_PATH
    elif arch == "multiscale_green_grid_dynamic_stage1":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_STAGE1_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_DYNAMIC_STAGE1_BEST_PATH
    elif arch == "multiscale_green_grid_interface_memory":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_BEST_PATH
    elif arch == "multiscale_green_grid_memory":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causal":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSAL_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSAL_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalconv":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALCONV_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALCONV_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_SMOOTH_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_SMOOTH_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_INTMEMORY_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_CAUSALHYBRID_INTMEMORY_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_fluxtrace":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_FLUXTRACE_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_FLUXTRACE_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MATCHEDABEL_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MATCHEDABEL_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MIXEDABEL_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_KERNELMIX_TRACEGREEN_MIXEDABEL_BEST_PATH
    elif arch == "multiscale_film_tracegreen_clean":
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_BEST_PATH
    elif arch == "multiscale_film_tracegreen_productintegral":
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_BEST_PATH
    elif arch == "multiscale_film_tracegreen_productintegral_lift":
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_LIFT_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_PRODUCTINTEGRAL_LIFT_BEST_PATH
    elif arch == "multiscale_film_tracegreen_clean_conservative":
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_CONSERVATIVE_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_CONSERVATIVE_BEST_PATH
    elif arch == "multiscale_film_tracegreen_clean_mixedflux":
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_MIXEDFLUX_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CLEAN_MIXEDFLUX_BEST_PATH
    elif arch == "multiscale_film_tracegreen_conservative_lift":
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CONSERVATIVE_LIFT_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_CONSERVATIVE_LIFT_BEST_PATH
    elif arch in {
        "multiscale_film_tracegreen_kparam",
        "multiscale_film_tracegreen_kparam_lift",
    }:
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_KPARAM_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_KPARAM_BEST_PATH
    elif arch in {
        "multiscale_film_tracegreen_gammaparam",
        "multiscale_film_tracegreen_gammaparam_lift",
    }:
        current_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_GAMMAPARAM_PATH
        best_path = MODEL_V96_MULTISCALE_FILM_TRACEGREEN_GAMMAPARAM_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel_ema":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_EMA_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_EMA_BEST_PATH
    elif arch == "multiscale_buffer":
        current_path = MODEL_V96_MULTISCALE_BUFFER_PATH
        best_path = MODEL_V96_MULTISCALE_BUFFER_BEST_PATH
    elif arch == "multiscale_fluxbuffer":
        current_path = MODEL_V96_MULTISCALE_FLUXBUFFER_PATH
        best_path = MODEL_V96_MULTISCALE_FLUXBUFFER_BEST_PATH
    elif arch == "multiscale_his":
        current_path = MODEL_V96_MULTISCALE_HIS_PATH
        best_path = MODEL_V96_MULTISCALE_HIS_BEST_PATH
    elif arch == "his_pinn":
        current_path = MODEL_V96_HIS_PINN_PATH
        best_path = MODEL_V96_HIS_PINN_BEST_PATH
    elif arch == "his_pinn_ext":
        current_path = MODEL_V96_HIS_PINN_EXT_PATH
        best_path = MODEL_V96_HIS_PINN_EXT_BEST_PATH
    else:
        current_path = './pinn_thin_layer_catalytic_v9_6.pth'
        best_path = './pinn_thin_layer_catalytic_v9_6_best.pth'

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        current_path = os.path.join(checkpoint_dir, os.path.basename(current_path))
        best_path = os.path.join(checkpoint_dir, os.path.basename(best_path))

    return current_path, best_path


def normalize_time(T):
    return 2.0 * T / T_sim - 1.0


def normalize_thin_x(X):
    return 2.0 * X / delta - 1.0


def normalize_ext_x(X):
    return 2.0 * (X - delta) / (X_ext_max - delta) - 1.0


def sample_external_x(n_points, device, near_fraction=0.50, mid_fraction=0.35):
    n_near = int(n_points * near_fraction)
    n_mid = int(n_points * mid_fraction)
    n_uniform = n_points - n_near - n_mid

    boundary_layer = min(0.7, 20.0 * delta)
    diffusion_layer = min(X_ext_max - delta, 3.0)
    x_near = delta + boundary_layer * torch.rand(n_near, 1, device=device) ** 2
    x_mid = delta + diffusion_layer * torch.rand(n_mid, 1, device=device) ** 1.5
    x_parts = [x_near, x_mid]
    if n_uniform > 0:
        x_parts.append(delta + torch.rand(n_uniform, 1, device=device) * (X_ext_max - delta))
    x_all = torch.cat(x_parts, dim=0)
    return x_all[torch.randperm(n_points, device=device)]


def sample_thin_x(n_points, device, boundary_fraction=0.70):
    n_boundary = int(n_points * boundary_fraction)
    n_surface = n_boundary // 2
    n_interface = n_boundary - n_surface
    n_uniform = n_points - n_boundary

    x_surface = delta * torch.rand(n_surface, 1, device=device) ** 2
    x_interface = delta * (1.0 - torch.rand(n_interface, 1, device=device) ** 2)
    x_parts = [x_surface, x_interface]
    if n_uniform > 0:
        x_parts.append(torch.rand(n_uniform, 1, device=device) * delta)
    x_all = torch.cat(x_parts, dim=0)
    return x_all[torch.randperm(n_points, device=device)]


def sample_interface_t(n_points, device, focus_fraction=0.65):
    """Bias interface samples toward the potential-turning and theta=0 windows."""
    n_focus = int(n_points * focus_fraction)
    n_uniform = n_points - n_focus

    if n_focus > 0:
        centers = torch.tensor(
            [0.25 * T_sim, T_switch * T_sim, 0.75 * T_sim],
            dtype=torch.float32,
            device=device,
        )
        center_idx = torch.randint(0, len(centers), (n_focus, 1), device=device)
        t_focus = centers[center_idx] + 0.055 * T_sim * torch.randn(n_focus, 1, device=device)
        t_focus = torch.clamp(t_focus, 0.0, T_sim)
    else:
        t_focus = torch.empty(0, 1, device=device)

    if n_uniform > 0:
        t_uniform = torch.rand(n_uniform, 1, device=device) * T_sim
        t_all = torch.cat([t_focus, t_uniform], dim=0)
    else:
        t_all = t_focus
    return t_all[torch.randperm(n_points, device=device)]

# ==================== 网络架构（与 v9.5 相同）====================
class ThinLayerNet_v9_6(nn.Module):
    def __init__(self, normalize_inputs=True):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, x_input):
        if self.normalize_inputs:
            x_input = torch.cat([
                normalize_time(x_input[:, 0:1]),
                normalize_thin_x(x_input[:, 1:2])
            ], dim=1)
        raw = self.net(x_input)
        conc_normalized = F.softmax(raw, dim=1)
        C_A = conc_normalized[:, 0:1]
        C_B = conc_normalized[:, 1:2]
        return C_A, C_B


class ExternalNet_v9_6(nn.Module):
    def __init__(self, gamma_val, normalize_inputs=True):
        super().__init__()
        self.gamma = gamma_val
        self.normalize_inputs = normalize_inputs
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x_input):
        if self.normalize_inputs:
            x_input = torch.cat([
                normalize_time(x_input[:, 0:1]),
                normalize_ext_x(x_input[:, 1:2])
            ], dim=1)
        raw = self.net(x_input)
        C_C = torch.sigmoid(raw) * self.gamma
        C_D = self.gamma - C_C
        return C_C, C_D


class FourierFeatureLayer(nn.Module):
    def __init__(self, in_features=2, frequencies=(1.0, 2.0, 4.0, 8.0, 16.0)):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))
        self.out_features = in_features * (1 + 2 * len(frequencies))

    def forward(self, x):
        features = [x]
        for freq in self.frequencies:
            scaled = np.pi * freq * x
            features.append(torch.sin(scaled))
            features.append(torch.cos(scaled))
        return torch.cat(features, dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.block(x))


class MultiscaleResidualHead(nn.Module):
    def __init__(self, out_features, width=256, depth=5, in_features=2):
        super().__init__()
        self.features = FourierFeatureLayer(in_features=in_features)
        layers = [nn.Linear(self.features.out_features, width), nn.Tanh()]
        for _ in range(depth):
            layers.append(ResidualMLPBlock(width))
        final_layer = nn.Linear(width, out_features)
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.features(x))


def hermite_cubic_basis(s):
    """Cubic Hermite basis on s in [0, 1].

    h00/h01 interpolate endpoint values; h10/h11 interpolate endpoint
    derivatives after multiplication by the interval length.
    """
    s = torch.clamp(s, 0.0, 1.0)
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00, h10, h01, h11


def hermite_reconstruct(x, xL, xR, yL, dyL, yR, dyR):
    """Reconstruct y(x) from endpoint values and physical x-derivatives."""
    length = torch.as_tensor(xR - xL, dtype=x.dtype, device=x.device)
    length = torch.clamp(length, min=torch.finfo(x.dtype).eps)
    s = (x - xL) / length
    h00, h10, h01, h11 = hermite_cubic_basis(s)
    return h00 * yL + h10 * length * dyL + h01 * yR + h11 * length * dyR


class ThinLayerNet_v9_6_Multiscale(nn.Module):
    def __init__(self, normalize_inputs=True):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.net = MultiscaleResidualHead(out_features=2, width=256, depth=5)

    def forward(self, x_input):
        if self.normalize_inputs:
            x_input = torch.cat([
                normalize_time(x_input[:, 0:1]),
                normalize_thin_x(x_input[:, 1:2])
            ], dim=1)
        raw = self.net(x_input)
        conc_normalized = F.softmax(raw, dim=1)
        C_A = conc_normalized[:, 0:1]
        C_B = conc_normalized[:, 1:2]
        return C_A, C_B


class ThinLayerNet_v9_6_MultiscaleHardBC(nn.Module):
    def __init__(self, normalize_inputs=True):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.net = MultiscaleResidualHead(out_features=2, width=256, depth=5)

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            x_net = torch.cat([
                normalize_time(T_raw),
                normalize_thin_x(X_raw)
            ], dim=1)
        else:
            x_net = x_input

        c_b_surface = torch.sigmoid(-potential_theta(T_raw))
        raw = self.net(x_net)[:, 1:2]
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))
        # Physics prior only: the interface B tendency follows the electrode Nernst state.
        c_b_prior_logit = torch.logit(torch.clamp(c_b_surface, 1e-6, 1.0 - 1e-6))
        c_b_free = time_gate * torch.sigmoid(raw + c_b_prior_logit)
        x_gate = torch.clamp(X_raw / delta, 0.0, 1.0)
        C_B = (1.0 - x_gate) * c_b_surface + x_gate * c_b_free
        C_A = 1.0 - C_B
        return C_A, C_B


class InterfaceStateNet_v9_6(nn.Module):
    def __init__(self, normalize_inputs=True):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.net = MultiscaleResidualHead(out_features=3, width=192, depth=4, in_features=1)
        # At raw=0 this gives the old gamma=10 initialization C_C_int/gamma=0.8,
        # but unlike the old fixed 4.0-unit depletion it can represent the full
        # physically admissible fractional range for every gamma.
        self.c_c_depletion_prior_logit = float(np.log(0.2 / 0.8))
        # Thin-layer gradients follow the characteristic reaction flux.  Scale
        # the old baseline range without changing gamma=10, k=1 behavior.
        self.surface_slope_scale = 6.0 / flux_residual_scale()

    def forward(self, T_raw):
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
        else:
            t_net = T_raw

        theta = potential_theta(T_raw)
        c_b_surface = torch.sigmoid(-theta)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))

        raw = self.net(t_net)
        c_b_prior_logit = torch.logit(torch.clamp(c_b_surface, 1e-6, 1.0 - 1e-6))
        C_B_int = time_gate * torch.sigmoid(c_b_prior_logit + raw[:, 0:1])
        C_C_int = gamma * (
            1.0 - time_gate * torch.sigmoid(
                self.c_c_depletion_prior_logit + raw[:, 1:2]
            )
        )
        J_rxn = k_cat_star * C_B_int * C_C_int
        surface_slope = self.surface_slope_scale * time_gate * torch.tanh(raw[:, 2:3])
        return {
            "C_B_surface": c_b_surface,
            "C_B_int": C_B_int,
            "C_C_int": C_C_int,
            "J_rxn": J_rxn,
            "surface_slope": surface_slope,
        }


class InterfaceStateNet_v9_6_Memory(InterfaceStateNet_v9_6):
    """Interface-state network with causal reaction memory.

    The standard interface state is a direct time-to-state map.  That makes the
    reverse-scan region hard: C_C_int is a slow history variable, while C_B_int
    responds quickly to the potential.  This experimental block keeps the
    original ``net`` intact for checkpoint compatibility, then adds a
    zero-initialized correction head driven by theta, dtheta/dt, base reaction
    flux, dJ/dt, cumulative charge, and exponential memories of the base flux.
    """

    def __init__(self, normalize_inputs=True, time_grid_points=256):
        super().__init__(normalize_inputs=normalize_inputs)
        self.interface_memory = True
        self.time_grid_points = int(time_grid_points)
        if self.time_grid_points < 4:
            raise ValueError("time_grid_points must be at least 4")
        self.register_buffer(
            "time_grid",
            torch.linspace(0.0, float(T_sim), self.time_grid_points, dtype=torch.float32).reshape(-1, 1),
        )
        self.register_buffer(
            "interface_memory_lambdas",
            torch.tensor([0.02, 0.05, 0.10, 0.20], dtype=torch.float32).reshape(1, -1),
        )
        self.interface_corr_net = MultiscaleResidualHead(
            out_features=3,
            width=160,
            depth=3,
            in_features=12,
        )
        nn.init.zeros_(self.interface_corr_net.net[-1].weight)
        nn.init.zeros_(self.interface_corr_net.net[-1].bias)
        self.register_buffer(
            "interface_corr_scales",
            torch.tensor([1.25, 1.25, 0.75], dtype=torch.float32).reshape(1, 3),
        )
        self._interface_memory_cache = None

    def clear_step_cache(self):
        self._interface_memory_cache = None

    def _base_state(self, T_raw):
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
        else:
            t_net = T_raw

        theta = potential_theta(T_raw)
        c_b_surface = torch.sigmoid(-theta)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))

        raw = self.net(t_net)
        c_b_prior_logit = torch.logit(torch.clamp(c_b_surface, 1e-6, 1.0 - 1e-6))
        C_B_int = time_gate * torch.sigmoid(c_b_prior_logit + raw[:, 0:1])
        C_C_int = gamma * (
            1.0 - time_gate * torch.sigmoid(
                self.c_c_depletion_prior_logit + raw[:, 1:2]
            )
        )
        J_rxn = k_cat_star * C_B_int * C_C_int
        surface_slope = self.surface_slope_scale * time_gate * torch.tanh(raw[:, 2:3])
        return {
            "raw": raw,
            "theta": theta,
            "C_B_surface": c_b_surface,
            "C_B_int": C_B_int,
            "C_C_int": C_C_int,
            "J_rxn": J_rxn,
            "surface_slope": surface_slope,
            "time_gate": time_gate,
            "c_b_prior_logit": c_b_prior_logit,
        }

    def _interp_history_multi(self, tau, value_grid):
        tau = torch.clamp(tau, 0.0, float(T_sim))
        u = tau * (self.time_grid_points - 1) / float(T_sim)
        idx0 = torch.floor(u).clamp(0, self.time_grid_points - 2).long().reshape(-1)
        idx1 = idx0 + 1
        alpha = (u.reshape(-1, 1) - idx0.reshape(-1, 1).to(dtype=u.dtype)).clamp(0.0, 1.0)
        values = value_grid.reshape(self.time_grid_points, -1)
        v0 = values[idx0]
        v1 = values[idx1]
        return v0 + alpha * (v1 - v0)

    def _base_history(self, T_ref):
        use_cache = self.training and torch.is_grad_enabled()
        cache_key = (T_ref.device, T_ref.dtype, torch.is_grad_enabled())
        if use_cache and self._interface_memory_cache is not None and self._interface_memory_cache[0] == cache_key:
            return self._interface_memory_cache[1]

        t_grid = self.time_grid.to(device=T_ref.device, dtype=T_ref.dtype)
        base = self._base_state(t_grid)
        j_grid = base["J_rxn"].reshape(self.time_grid_points, 1)
        dt = float(T_sim) / float(self.time_grid_points - 1)

        segments = 0.5 * (j_grid[:-1] + j_grid[1:]) * dt
        q_grid = torch.cat([torch.zeros_like(j_grid[:1]), torch.cumsum(segments, dim=0)], dim=0)

        lambdas = self.interface_memory_lambdas.to(device=T_ref.device, dtype=T_ref.dtype)
        decay = torch.exp(-dt / torch.clamp(lambdas, min=1e-6))
        memory = torch.zeros(1, lambdas.shape[1], device=T_ref.device, dtype=T_ref.dtype)
        rows = [memory]
        for idx in range(1, self.time_grid_points):
            j_prev = j_grid[idx - 1:idx]
            j_cur = j_grid[idx:idx + 1]
            memory = decay * memory + 0.5 * dt * (decay * j_prev + j_cur)
            rows.append(memory)
        m_grid = torch.cat(rows, dim=0)

        if self.time_grid_points > 1:
            d_j = torch.zeros_like(j_grid)
            d_j[1:-1] = (j_grid[2:] - j_grid[:-2]) / (2.0 * dt)
            d_j[0] = (j_grid[1] - j_grid[0]) / dt
            d_j[-1] = (j_grid[-1] - j_grid[-2]) / dt
        else:
            d_j = torch.zeros_like(j_grid)

        history = {
            "J": j_grid,
            "dJ": d_j,
            "Q": q_grid,
            "M": m_grid,
        }
        if use_cache:
            self._interface_memory_cache = (cache_key, history)
        return history

    def _memory_features(self, T_raw, base):
        history = self._base_history(T_raw)
        theta = base["theta"]
        theta_dot = potential_theta_dot(T_raw)
        j_rxn = base["J_rxn"]
        d_j = self._interp_history_multi(T_raw, history["dJ"])
        q_rxn = self._interp_history_multi(T_raw, history["Q"])
        memories = self._interp_history_multi(T_raw, history["M"])

        theta_scale = max(abs(float(theta_i)), abs(float(theta_switch)), 1.0)
        theta_dot_scale = theta_scale / max(float(T_sim), 1e-12)
        j_scale = max(float(k_cat_star * gamma), 1.0)
        q_scale = max(j_scale * float(T_sim), 1.0)
        lambdas = self.interface_memory_lambdas.to(device=T_raw.device, dtype=T_raw.dtype)
        memory_scale = torch.clamp(j_scale * lambdas, min=1e-6)
        t_net = normalize_time(T_raw) if self.normalize_inputs else T_raw

        return torch.cat([
            t_net,
            theta / theta_scale,
            theta_dot / theta_dot_scale,
            base["C_B_int"],
            base["C_C_int"] / gamma,
            j_rxn / j_scale,
            d_j / max(j_scale / max(float(T_sim), 1e-12), 1e-12),
            q_rxn / q_scale,
            memories / memory_scale,
        ], dim=1)

    def forward(self, T_raw):
        base = self._base_state(T_raw)
        features = self._memory_features(T_raw, base)
        corr = self.interface_corr_scales.to(device=T_raw.device, dtype=T_raw.dtype) * torch.tanh(
            self.interface_corr_net(features)
        )

        time_gate = base["time_gate"]
        C_B_int = time_gate * torch.sigmoid(
            base["c_b_prior_logit"] + base["raw"][:, 0:1] + corr[:, 0:1]
        )
        C_C_int = gamma * (
            1.0 - time_gate * torch.sigmoid(
                self.c_c_depletion_prior_logit + base["raw"][:, 1:2] + corr[:, 1:2]
            )
        )
        J_rxn = k_cat_star * C_B_int * C_C_int
        surface_slope = self.surface_slope_scale * time_gate * torch.tanh(
            base["raw"][:, 2:3] + corr[:, 2:3]
        )
        return {
            "C_B_surface": base["C_B_surface"],
            "C_B_int": C_B_int,
            "C_C_int": C_C_int,
            "J_rxn": J_rxn,
            "surface_slope": surface_slope,
        }


class InterfaceStateNet_v9_6_FilmAbel(nn.Module):
    """Physics-guided interface chain for the film/Abel ablation.

    The standard interface state maps T -> (C_B_int, C_C_int, J) with a neural
    network.  This prototype instead makes the interface source causal:

    C_B_surface --quasi-steady film--> C_B_int
    J_rxn history --Abel/Green memory--> C_C_int
    C_B_int, C_C_int --reaction law--> J_rxn

    The small residual head only corrects the Abel interface concentration; it
    is zero-initialized so the first forward pass is the pure film+Abel model.
    """

    def __init__(self, normalize_inputs=True, time_grid_points=256, kernel_points=64):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.film_abel_interface = True
        self.singularity_matched_abel = False
        self.time_grid_points = int(time_grid_points)
        self.kernel_points = int(kernel_points)
        if self.time_grid_points < 4:
            raise ValueError("time_grid_points must be at least 4")
        if self.kernel_points < 2:
            raise ValueError("kernel_points must be at least 2")

        self.register_buffer(
            "time_grid",
            torch.linspace(0.0, float(T_sim), self.time_grid_points, dtype=torch.float32).reshape(-1, 1),
        )
        self.register_buffer(
            "kernel_nodes",
            (torch.arange(self.kernel_points, dtype=torch.float32) + 0.5) / self.kernel_points,
        )
        self.register_buffer(
            "film_abel_memory_lambdas",
            torch.tensor([0.02, 0.05, 0.10, 0.20], dtype=torch.float32).reshape(1, -1),
        )
        self.residual_net = MultiscaleResidualHead(out_features=1, width=128, depth=3, in_features=13)
        nn.init.zeros_(self.residual_net.net[-1].weight)
        nn.init.zeros_(self.residual_net.net[-1].bias)

        self.abel_gain_raw = nn.Parameter(torch.zeros(1))
        self.abel_diffusion_raw = nn.Parameter(torch.zeros(1))
        self.phase_shift_raw = nn.Parameter(torch.zeros(1))
        self.residual_logit_scale = 1.25
        self._film_abel_cache = None

    def clear_step_cache(self):
        self._film_abel_cache = None

    def active_k_cat(self):
        return float(k_cat_star)

    def _active_k_tensor(self, reference):
        return torch.as_tensor(
            self.active_k_cat(), device=reference.device, dtype=reference.dtype
        )

    def active_gamma(self):
        return float(gamma)

    def _active_gamma_tensor(self, reference):
        return torch.as_tensor(
            self.active_gamma(), device=reference.device, dtype=reference.dtype
        )

    def condition_cache_key(self):
        return None

    def _history_feature_flux_scale(self):
        return max(float(k_cat_star * gamma), 1.0)

    def _abel_gain(self):
        return torch.exp(0.5 * torch.tanh(self.abel_gain_raw))

    def _abel_diffusion(self):
        return D_rel_D * torch.exp(0.5 * torch.tanh(self.abel_diffusion_raw))

    def _phase_shift(self):
        return 0.03 * T_sim * torch.tanh(self.phase_shift_raw)

    def _feature_theta_dot(self, T_raw):
        return potential_theta_dot(T_raw)

    def _surface_state(self, T_raw):
        theta = potential_theta(T_raw)
        theta_dot = self._feature_theta_dot(T_raw)
        c_b_surface = torch.sigmoid(-theta)
        dc_b_surface_dt = -c_b_surface * (1.0 - c_b_surface) * theta_dot
        return theta, theta_dot, c_b_surface, dc_b_surface_dt

    def _interp_scalar_grid(self, tau, value_grid):
        tau = torch.clamp(tau, 0.0, float(T_sim))
        original_shape = tau.shape
        flat_tau = tau.reshape(-1)
        u = flat_tau * (self.time_grid_points - 1) / float(T_sim)
        idx0 = torch.floor(u).clamp(0, self.time_grid_points - 2).long()
        idx1 = idx0 + 1
        alpha = (u - idx0.to(dtype=u.dtype)).clamp(0.0, 1.0)
        values = value_grid.reshape(self.time_grid_points)
        out = values[idx0] + alpha * (values[idx1] - values[idx0])
        return out.reshape(original_shape)

    def _interp_multi_grid(self, tau, value_grid):
        tau = torch.clamp(tau, 0.0, float(T_sim))
        original_prefix = tau.shape[:-1]
        flat_tau = tau.reshape(-1)
        u = flat_tau * (self.time_grid_points - 1) / float(T_sim)
        idx0 = torch.floor(u).clamp(0, self.time_grid_points - 2).long()
        idx1 = idx0 + 1
        alpha = (u.reshape(-1, 1) - idx0.reshape(-1, 1).to(dtype=u.dtype)).clamp(0.0, 1.0)
        values = value_grid.reshape(self.time_grid_points, -1)
        out = values[idx0] + alpha * (values[idx1] - values[idx0])
        return out.reshape(*original_prefix, values.shape[1])

    def _film_reaction(self, c_b_surface, c_c_int):
        c_c_pos = torch.clamp(c_c_int, min=1e-8)
        k_value = self._active_k_tensor(c_c_pos)
        denom = 1.0 + (k_value * delta / D_rel_B) * c_c_pos
        c_b_int = c_b_surface / torch.clamp(denom, min=1e-8)
        j_rxn = k_value * c_c_pos * c_b_int
        surface_slope = -j_rxn / D_rel_B
        return c_b_int, j_rxn, surface_slope

    def _bounded_c_d_int(self, c_d_prior, corr_raw, T_raw):
        eps = 1e-6
        frac = torch.clamp(c_d_prior / gamma, eps, 1.0 - eps)
        prior_logit = torch.logit(frac)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))
        corr_logit = self.residual_logit_scale * time_gate * torch.tanh(corr_raw)
        return gamma * torch.sigmoid(prior_logit + corr_logit)

    def _c_d_correction_raw(self, features):
        return self.residual_net(features)

    def _feature_tensor(self, T_raw, theta, theta_dot, c_b_surface, dc_b_surface_dt,
                        c_d_prior, j_hist, d_j_hist, q_hist, memories):
        theta_scale = max(abs(float(theta_i)), abs(float(theta_switch)), 1.0)
        theta_dot_scale = theta_scale / max(float(T_sim), 1e-12)
        j_scale = self._history_feature_flux_scale()
        q_floor = 1e-8 if getattr(self, "k_parameterized", False) else 1.0
        q_scale = max(j_scale * float(T_sim), q_floor)
        cb_dt_scale = max(theta_dot_scale, 1.0)
        lambdas = self.film_abel_memory_lambdas.to(device=T_raw.device, dtype=T_raw.dtype)
        memory_scale = torch.clamp(j_scale * lambdas, min=1e-6)
        t_net = normalize_time(T_raw) if self.normalize_inputs else T_raw

        return torch.cat([
            t_net,
            theta / theta_scale,
            theta_dot / theta_dot_scale,
            c_b_surface,
            dc_b_surface_dt / cb_dt_scale,
            c_d_prior / gamma,
            j_hist / j_scale,
            d_j_hist / max(j_scale / max(float(T_sim), 1e-12), 1e-12),
            q_hist / q_scale,
            memories / memory_scale,
        ], dim=1)

    def _abel_convolution_from_grid(self, T_raw, value_grid):
        """Abel/Neumann-to-trace convolution.

        The interface memory maps flux history to boundary concentration with

            int_0^t q(tau) / sqrt(t - tau) d tau.

        The default path preserves legacy checkpoints.  The matched-Abel
        ablation uses t-tau=t*u^2, so the singular weight is absorbed into the
        Jacobian and constant histories are integrated exactly.
        """
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        diffusion = torch.clamp(self._abel_diffusion().to(device=T_raw.device, dtype=T_raw.dtype), min=1e-8)
        gain = self._abel_gain().to(device=T_raw.device, dtype=T_raw.dtype)
        if not getattr(self, "singularity_matched_abel", False):
            tau = t_pos * nodes
            dtau = torch.clamp(t_pos * (1.0 - nodes), min=1e-8 * T_sim)
            value_tau = self._interp_scalar_grid(tau, value_grid)
            kernel = gain / torch.sqrt(torch.clamp(np.pi * diffusion * dtau, min=1e-12))
            return T_raw * torch.mean(value_tau * kernel, dim=1, keepdim=True)

        tau = t_pos * (1.0 - nodes ** 2)
        value_tau = self._interp_scalar_grid(tau, value_grid)
        prefactor = 2.0 * gain * torch.sqrt(t_pos) / torch.sqrt(torch.clamp(np.pi * diffusion, min=1e-12))
        return prefactor * torch.mean(value_tau, dim=1, keepdim=True)

    def _abel_memory_from_grid(self, T_raw, j_grid, d_j_grid=None):
        c_d = self._abel_convolution_from_grid(T_raw, j_grid)
        if d_j_grid is not None:
            # First-order phase correction: G*J(t-dt_phase) ~= G*J - dt_phase*G*dJ.
            # The learnable signed dt_phase starts at zero, so this branch cannot
            # change warm-start behavior until training finds it useful.
            c_d = c_d - self._phase_shift().to(device=T_raw.device, dtype=T_raw.dtype) * self._abel_convolution_from_grid(T_raw, d_j_grid)
        return c_d

    def _history_grid(self, T_ref):
        use_cache = self.training and torch.is_grad_enabled()
        cache_key = (
            T_ref.device,
            T_ref.dtype,
            torch.is_grad_enabled(),
            self.condition_cache_key(),
        )
        if use_cache and self._film_abel_cache is not None and self._film_abel_cache[0] == cache_key:
            return self._film_abel_cache[1]

        device = T_ref.device
        dtype = T_ref.dtype
        t_grid = self.time_grid.to(device=device, dtype=dtype)
        dt = float(T_sim) / float(self.time_grid_points - 1)
        lambdas = self.film_abel_memory_lambdas.to(device=device, dtype=dtype)
        decay = torch.exp(-dt / torch.clamp(lambdas, min=1e-6))

        j_rows = []
        cb_surface_rows = []
        cb_int_rows = []
        cc_int_rows = []
        cd_int_rows = []
        cd_prior_rows = []
        slope_rows = []
        q_rows = []
        memory_rows = []

        q_hist = torch.zeros(1, 1, device=device, dtype=dtype)
        memory = torch.zeros(1, lambdas.shape[1], device=device, dtype=dtype)
        last_j = torch.zeros(1, 1, device=device, dtype=dtype)
        last_d_j = torch.zeros(1, 1, device=device, dtype=dtype)

        for idx in range(self.time_grid_points):
            t = t_grid[idx:idx + 1]
            theta, theta_dot, c_b_surface, dc_b_surface_dt = self._surface_state(t)
            if idx == 0:
                c_d_prior = torch.zeros(1, 1, device=device, dtype=dtype)
            else:
                j_prev = torch.cat(j_rows, dim=0)
                steps = torch.arange(idx, device=device, dtype=dtype).reshape(-1, 1)
                lags = (float(idx) - steps - 0.5) * dt
                diffusion = torch.clamp(self._abel_diffusion().to(device=device, dtype=dtype), min=1e-8)
                gain = self._abel_gain().to(device=device, dtype=dtype)
                if getattr(self, "singularity_matched_abel", False):
                    s_hi = torch.clamp(lags + 0.5 * dt, min=0.0)
                    s_lo = torch.clamp(lags - 0.5 * dt, min=0.0)
                    weights = 2.0 * gain * (
                        torch.sqrt(s_hi) - torch.sqrt(s_lo)
                    ) / torch.sqrt(torch.clamp(np.pi * diffusion, min=1e-12))
                    c_d_prior = torch.sum(j_prev * weights, dim=0, keepdim=True)
                else:
                    kernel = gain / torch.sqrt(torch.clamp(np.pi * diffusion * lags, min=1e-12))
                    c_d_prior = torch.sum(j_prev * kernel, dim=0, keepdim=True) * dt

            features = self._feature_tensor(
                t,
                theta,
                theta_dot,
                c_b_surface,
                dc_b_surface_dt,
                c_d_prior,
                last_j,
                last_d_j,
                q_hist,
                memory,
            )
            c_d_int = self._bounded_c_d_int(c_d_prior, self._c_d_correction_raw(features), t)
            c_c_int = gamma - c_d_int
            c_b_int, j_rxn, surface_slope = self._film_reaction(c_b_surface, c_c_int)

            q_next = q_hist + j_rxn * dt
            memory_next = decay * memory + j_rxn * dt

            cb_surface_rows.append(c_b_surface)
            cb_int_rows.append(c_b_int)
            cc_int_rows.append(c_c_int)
            cd_int_rows.append(c_d_int)
            cd_prior_rows.append(c_d_prior)
            j_rows.append(j_rxn)
            slope_rows.append(surface_slope)
            q_rows.append(q_next)
            memory_rows.append(memory_next)

            q_hist = q_next
            memory = memory_next
            last_j = j_rxn
            if idx == 0:
                last_d_j = torch.zeros_like(j_rxn)
            else:
                last_d_j = (j_rxn - j_rows[-2]) / dt

        j_grid = torch.cat(j_rows, dim=0)
        if self.time_grid_points > 1:
            d_j_grid = torch.zeros_like(j_grid)
            d_j_grid[1:-1] = (j_grid[2:] - j_grid[:-2]) / (2.0 * dt)
            d_j_grid[0] = (j_grid[1] - j_grid[0]) / dt
            d_j_grid[-1] = (j_grid[-1] - j_grid[-2]) / dt
        else:
            d_j_grid = torch.zeros_like(j_grid)

        history = {
            "C_B_surface": torch.cat(cb_surface_rows, dim=0),
            "C_B_int": torch.cat(cb_int_rows, dim=0),
            "C_C_int": torch.cat(cc_int_rows, dim=0),
            "C_D_int": torch.cat(cd_int_rows, dim=0),
            "C_D_prior": torch.cat(cd_prior_rows, dim=0),
            "J": j_grid,
            "dJ": d_j_grid,
            "surface_slope": torch.cat(slope_rows, dim=0),
            "Q": torch.cat(q_rows, dim=0),
            "M": torch.cat(memory_rows, dim=0),
        }
        if use_cache:
            self._film_abel_cache = (cache_key, history)
        return history

    def forward(self, T_raw):
        history = self._history_grid(T_raw)
        theta, theta_dot, c_b_surface, dc_b_surface_dt = self._surface_state(T_raw)
        c_d_prior = self._abel_memory_from_grid(T_raw, history["J"], history["dJ"])
        j_hist = self._interp_multi_grid(T_raw, history["J"])
        d_j_hist = self._interp_multi_grid(T_raw, history["dJ"])
        q_hist = self._interp_multi_grid(T_raw, history["Q"])
        memories = self._interp_multi_grid(T_raw, history["M"])
        features = self._feature_tensor(
            T_raw,
            theta,
            theta_dot,
            c_b_surface,
            dc_b_surface_dt,
            c_d_prior,
            j_hist,
            d_j_hist,
            q_hist,
            memories,
        )
        c_d_int = self._bounded_c_d_int(c_d_prior, self._c_d_correction_raw(features), T_raw)
        c_c_int = gamma - c_d_int
        c_b_int, j_rxn, surface_slope = self._film_reaction(c_b_surface, c_c_int)
        return {
            "C_B_surface": c_b_surface,
            "C_B_int": c_b_int,
            "C_C_int": c_c_int,
            "J_rxn": j_rxn,
            "dJ_rxn_dt": d_j_hist,
            "Q_rxn": q_hist,
            "surface_slope": surface_slope,
        }


class InterfaceStateNet_v9_6_FilmAbelEMA(InterfaceStateNet_v9_6_FilmAbel):
    """Film-Abel interface state with explicit stable-memory anti-drift.

    The base Film-Abel chain computes C_C_int from a causal Abel prior plus a
    small residual.  That is physically meaningful, but Abel's long tail can
    carry a reverse-scan phase error to late times.  This ablation keeps the
    Abel prior, then adds a zero-initialized residual head driven by the same
    normalized multi-exponential memories.  These memory features behave like a
    bank of EMA filters, so the correction can learn controlled forgetting
    instead of only accumulating the signed flux history.
    """

    def __init__(self, normalize_inputs=True, time_grid_points=256, kernel_points=64):
        super().__init__(
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
        )
        self.film_abel_ema_interface = True
        self.ema_drift_net = MultiscaleResidualHead(out_features=1, width=128, depth=3, in_features=13)
        nn.init.zeros_(self.ema_drift_net.net[-1].weight)
        nn.init.zeros_(self.ema_drift_net.net[-1].bias)
        self.ema_drift_scale = 0.75

    def _c_d_correction_raw(self, features):
        base_raw = super()._c_d_correction_raw(features)
        ema_raw = self.ema_drift_net(features)
        return base_raw + self.ema_drift_scale * torch.tanh(ema_raw)


class InterfaceStateNet_v9_6_FilmAbelKernelMix(InterfaceStateNet_v9_6_FilmAbel):
    """Film-Abel interface with a learnable memory-kernel mixture.

    EMA used finite-memory features only inside the residual head.  This variant
    changes the interface prior itself:

        C_D_int prior = alpha * Abel[J] + sum beta_i * ExpMemory_i[J]
                      - dt_phase(state) * Abel[dJ]

    New parameters are zero-initialized so loading a Film-Abel checkpoint starts
    near the v1 solution.  Training can then reduce Abel long-tail drift by
    moving weight into short/mid/long finite-memory kernels instead of asking a
    residual network to undo the whole accumulated error.
    """

    def __init__(self, normalize_inputs=True, time_grid_points=256, kernel_points=64):
        super().__init__(
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
        )
        self.film_abel_kernelmix_interface = True
        self.kernelmix_alpha_raw = nn.Parameter(torch.zeros(1))
        self.kernelmix_beta_raw = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.kernelmix_beta_gain = 0.45
        self.dynamic_phase_net = MultiscaleResidualHead(out_features=1, width=64, depth=2, in_features=4)
        nn.init.zeros_(self.dynamic_phase_net.net[-1].weight)
        nn.init.zeros_(self.dynamic_phase_net.net[-1].bias)
        self.dynamic_phase_scale = 0.025 * T_sim
        self.continuous_time_features = True

    def _feature_theta_dot(self, T_raw):
        return potential_theta_dot_smooth(T_raw)

    def _kernelmix_alpha(self):
        return 1.0 + 0.20 * torch.tanh(self.kernelmix_alpha_raw)

    def _kernelmix_beta(self):
        return self.kernelmix_beta_gain * torch.tanh(self.kernelmix_beta_raw).reshape(1, -1)

    def _dynamic_phase_features(self, theta, theta_dot, j_hist, d_j_hist):
        theta_scale = max(abs(float(theta_i)), abs(float(theta_switch)), 1.0)
        theta_dot_scale = theta_scale / max(float(T_sim), 1e-12)
        j_scale = self._history_feature_flux_scale()
        dj_scale = max(j_scale / max(float(T_sim), 1e-12), 1e-12)
        return torch.cat([
            theta / theta_scale,
            theta_dot / theta_dot_scale,
            j_hist / j_scale,
            d_j_hist / dj_scale,
        ], dim=1)

    def _dynamic_phase_shift(self, theta, theta_dot, j_hist, d_j_hist):
        raw = self.dynamic_phase_net(self._dynamic_phase_features(theta, theta_dot, j_hist, d_j_hist))
        return self._phase_shift().to(device=raw.device, dtype=raw.dtype) + self.dynamic_phase_scale * torch.tanh(raw)

    def _finite_memory_from_grid(self, T_raw, value_grid):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * nodes
        dtau = torch.clamp(t_pos * (1.0 - nodes), min=1e-8 * T_sim)
        value_tau = self._interp_scalar_grid(tau, value_grid).unsqueeze(-1)
        lambdas = self.film_abel_memory_lambdas.to(device=T_raw.device, dtype=T_raw.dtype).reshape(1, 1, -1)
        diffusion = torch.clamp(self._abel_diffusion().to(device=T_raw.device, dtype=T_raw.dtype), min=1e-8)
        kernel = torch.exp(-dtau.unsqueeze(-1) / torch.clamp(lambdas, min=1e-6))
        kernel = kernel / torch.sqrt(torch.clamp(np.pi * diffusion * lambdas, min=1e-12))
        return T_raw * torch.mean(value_tau * kernel, dim=1)

    def _finite_memory_from_previous(self, value_prev, lags, dt, device, dtype):
        lambdas = self.film_abel_memory_lambdas.to(device=device, dtype=dtype)
        diffusion = torch.clamp(self._abel_diffusion().to(device=device, dtype=dtype), min=1e-8)
        kernel = torch.exp(-lags / torch.clamp(lambdas, min=1e-6))
        kernel = kernel / torch.sqrt(torch.clamp(np.pi * diffusion * lambdas, min=1e-12))
        return torch.sum(value_prev * kernel, dim=0, keepdim=True) * dt

    def _abel_from_previous(self, value_prev, lags, dt, device, dtype):
        diffusion = torch.clamp(self._abel_diffusion().to(device=device, dtype=dtype), min=1e-8)
        gain = self._abel_gain().to(device=device, dtype=dtype)
        if getattr(self, "singularity_matched_abel", False):
            s_hi = torch.clamp(lags + 0.5 * dt, min=0.0)
            s_lo = torch.clamp(lags - 0.5 * dt, min=0.0)
            weights = 2.0 * gain * (
                torch.sqrt(s_hi) - torch.sqrt(s_lo)
            ) / torch.sqrt(torch.clamp(np.pi * diffusion, min=1e-12))
            return torch.sum(value_prev * weights, dim=0, keepdim=True)

        kernel = gain / torch.sqrt(torch.clamp(np.pi * diffusion * lags, min=1e-12))
        return torch.sum(value_prev * kernel, dim=0, keepdim=True) * dt

    def _kernelmix_prior(self, abel_prior, finite_terms):
        beta = self._kernelmix_beta().to(device=abel_prior.device, dtype=abel_prior.dtype)
        alpha = self._kernelmix_alpha().to(device=abel_prior.device, dtype=abel_prior.dtype)
        return alpha * abel_prior + torch.sum(beta * finite_terms, dim=1, keepdim=True)

    def _abel_memory_from_grid(self, T_raw, j_grid, d_j_grid=None):
        abel_j = self._abel_convolution_from_grid(T_raw, j_grid)
        finite_j = self._finite_memory_from_grid(T_raw, j_grid)
        c_d = self._kernelmix_prior(abel_j, finite_j)
        if d_j_grid is not None:
            theta, theta_dot, _, _ = self._surface_state(T_raw)
            j_hist = self._interp_multi_grid(T_raw, j_grid)
            d_j_hist = self._interp_multi_grid(T_raw, d_j_grid)
            phase = self._dynamic_phase_shift(theta, theta_dot, j_hist, d_j_hist)
            c_d = c_d - phase * self._abel_convolution_from_grid(T_raw, d_j_grid)
        return c_d

    def _history_grid(self, T_ref):
        use_cache = self.training and torch.is_grad_enabled()
        cache_key = (
            T_ref.device,
            T_ref.dtype,
            torch.is_grad_enabled(),
            self.condition_cache_key(),
        )
        if use_cache and self._film_abel_cache is not None and self._film_abel_cache[0] == cache_key:
            return self._film_abel_cache[1]

        device = T_ref.device
        dtype = T_ref.dtype
        t_grid = self.time_grid.to(device=device, dtype=dtype)
        dt = float(T_sim) / float(self.time_grid_points - 1)
        lambdas = self.film_abel_memory_lambdas.to(device=device, dtype=dtype)
        decay = torch.exp(-dt / torch.clamp(lambdas, min=1e-6))

        j_rows = []
        d_j_rows = []
        cb_surface_rows = []
        cb_int_rows = []
        cc_int_rows = []
        cd_int_rows = []
        cd_prior_rows = []
        slope_rows = []
        q_rows = []
        memory_rows = []

        q_hist = torch.zeros(1, 1, device=device, dtype=dtype)
        memory = torch.zeros(1, lambdas.shape[1], device=device, dtype=dtype)
        last_j = torch.zeros(1, 1, device=device, dtype=dtype)
        last_d_j = torch.zeros(1, 1, device=device, dtype=dtype)

        for idx in range(self.time_grid_points):
            t = t_grid[idx:idx + 1]
            theta, theta_dot, c_b_surface, dc_b_surface_dt = self._surface_state(t)
            if idx == 0:
                c_d_prior = torch.zeros(1, 1, device=device, dtype=dtype)
            else:
                j_prev = torch.cat(j_rows, dim=0)
                d_j_prev = torch.cat(d_j_rows, dim=0)
                steps = torch.arange(idx, device=device, dtype=dtype).reshape(-1, 1)
                lags = (float(idx) - steps - 0.5) * dt
                abel_j = self._abel_from_previous(j_prev, lags, dt, device, dtype)
                finite_j = self._finite_memory_from_previous(j_prev, lags, dt, device, dtype)
                c_d_prior = self._kernelmix_prior(abel_j, finite_j)
                abel_d_j = self._abel_from_previous(d_j_prev, lags, dt, device, dtype)
                phase = self._dynamic_phase_shift(theta, theta_dot, last_j, last_d_j)
                c_d_prior = c_d_prior - phase * abel_d_j

            features = self._feature_tensor(
                t,
                theta,
                theta_dot,
                c_b_surface,
                dc_b_surface_dt,
                c_d_prior,
                last_j,
                last_d_j,
                q_hist,
                memory,
            )
            c_d_int = self._bounded_c_d_int(c_d_prior, self._c_d_correction_raw(features), t)
            c_c_int = gamma - c_d_int
            c_b_int, j_rxn, surface_slope = self._film_reaction(c_b_surface, c_c_int)

            if idx == 0:
                d_j_cur = torch.zeros_like(j_rxn)
            else:
                d_j_cur = (j_rxn - j_rows[-1]) / dt

            q_next = q_hist + j_rxn * dt
            memory_next = decay * memory + j_rxn * dt

            cb_surface_rows.append(c_b_surface)
            cb_int_rows.append(c_b_int)
            cc_int_rows.append(c_c_int)
            cd_int_rows.append(c_d_int)
            cd_prior_rows.append(c_d_prior)
            j_rows.append(j_rxn)
            d_j_rows.append(d_j_cur)
            slope_rows.append(surface_slope)
            q_rows.append(q_next)
            memory_rows.append(memory_next)

            q_hist = q_next
            memory = memory_next
            last_j = j_rxn
            last_d_j = d_j_cur

        j_grid = torch.cat(j_rows, dim=0)
        if self.time_grid_points > 1:
            d_j_grid = torch.zeros_like(j_grid)
            d_j_grid[1:-1] = (j_grid[2:] - j_grid[:-2]) / (2.0 * dt)
            d_j_grid[0] = (j_grid[1] - j_grid[0]) / dt
            d_j_grid[-1] = (j_grid[-1] - j_grid[-2]) / dt
        else:
            d_j_grid = torch.zeros_like(j_grid)

        history = {
            "C_B_surface": torch.cat(cb_surface_rows, dim=0),
            "C_B_int": torch.cat(cb_int_rows, dim=0),
            "C_C_int": torch.cat(cc_int_rows, dim=0),
            "C_D_int": torch.cat(cd_int_rows, dim=0),
            "C_D_prior": torch.cat(cd_prior_rows, dim=0),
            "J": j_grid,
            "dJ": d_j_grid,
            "surface_slope": torch.cat(slope_rows, dim=0),
            "Q": torch.cat(q_rows, dim=0),
            "M": torch.cat(memory_rows, dim=0),
        }
        if use_cache:
            self._film_abel_cache = (cache_key, history)
        return history


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausal(InterfaceStateNet_v9_6_FilmAbelKernelMix):
    """KernelMix interface with a diffusion-causal external correction gate.

    Green/Abel/base terms already propagate interface flux causally.  This
    marker asks the external dynamic correction to obey the same boundary-origin
    diffusion reachability, so it cannot paint a large far-field correction
    before diffusion can carry interface information there.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_dynamic_correction = True
        self.causal_gate_alpha = 2.0
        self.causal_gate_time_floor = 0.005 * T_sim
        self.causal_jump_weight = 50.0


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausalConv(InterfaceStateNet_v9_6_FilmAbelKernelMixCausal):
    """KernelMix interface marker for causal-convolution dynamic correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_convolution_dynamic = True
        self.causal_source_scale = 0.75
        self.causal_jump_weight = 25.0


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybrid(InterfaceStateNet_v9_6_FilmAbelKernelMixCausalConv):
    """Marker for causal source convolution plus a small gated direct residual."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_hybrid_dynamic = True
        self.causal_direct_scale = 0.25
        self.causal_source_scale = 0.65
        self.causal_jump_weight = 35.0


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridSmooth(InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybrid):
    """Causal-hybrid marker with explicit smoothness regularization.

    Diagnostics showed the small direct dynamic residual can still switch
    sharply at the scan reversal.  This ablation keeps the causal-convolution
    branch as the main external correction, reduces the direct branch, and asks
    the training loop to penalize direct/static/phase temporal roughness.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_hybrid_smooth = True
        self.causal_direct_scale = 0.05
        self.causal_source_scale = 0.70
        self.causal_jump_weight = 20.0
        self.direct_jump_weight = 1500.0
        self.direct_smooth_weight = 250.0
        self.static_smooth_weight = 60.0
        self.phase_smooth_weight = 25.0
        self.smooth_time_window = 0.08 * T_sim


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
    InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridSmooth
):
    """Interface-memory v2 for C_C_int-dominated residuals.

    The smooth causal-hybrid branch fixed much of the external-field time jump,
    but posterior decomposition showed that C_C_int still inherits a biased Abel
    long-memory prior and a free logit residual that can drift after reversal.

    This branch keeps the same causal external field, but changes the interface
    state in two narrow ways:
      1. finite-memory kernels contribute positively to C_D_prior with fresh
         parameters, so old warm-start beta values near zero cannot lock them out;
      2. the residual correction is additive in concentration units, followed by
         a clamp for bounds, instead of being the primary logit/sigmoid pathway.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface_memory_v2 = True

        # The old residual_net may be loaded from a warm-start checkpoint.  It is
        # intentionally bypassed because decomposition showed it can push C_C_int
        # in the wrong direction after reversal.
        for param in self.residual_net.parameters():
            param.requires_grad_(False)

        self.interface_residual_net_v2 = MultiscaleResidualHead(
            out_features=1,
            width=96,
            depth=2,
            in_features=13,
        )
        nn.init.zeros_(self.interface_residual_net_v2.net[-1].weight)
        nn.init.zeros_(self.interface_residual_net_v2.net[-1].bias)

        # Fresh positive finite-memory gains.  Initial value is about 0.04 per
        # memory scale, enough to counter the observed Abel under-production of
        # C_D_int without making a large discontinuous architecture jump.
        self.intmemory_finite_gain_raw = nn.Parameter(torch.full((4,), -2.2, dtype=torch.float32))
        self.intmemory_finite_gain_scale = 0.40

        # Additive C_D correction bound.  The scale starts near 0.13 concentration
        # units and is learned, while the zero-initialized head starts neutral.
        self.intmemory_corr_scale_raw = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        self.intmemory_corr_scale = 0.35

        self.cint_reversal_jump_weight = 800.0
        self.cint_temporal_smooth_weight = 120.0
        self.phase_smooth_weight = 40.0

    def _intmemory_finite_gain(self):
        raw = self.intmemory_finite_gain_raw.reshape(1, -1)
        return self.intmemory_finite_gain_scale * torch.sigmoid(raw)

    def _intmemory_corr_amplitude(self):
        return self.intmemory_corr_scale * torch.sigmoid(self.intmemory_corr_scale_raw)

    def _kernelmix_prior(self, abel_prior, finite_terms):
        prior = super()._kernelmix_prior(abel_prior, finite_terms)
        gain = self._intmemory_finite_gain().to(device=finite_terms.device, dtype=finite_terms.dtype)
        return prior + torch.sum(gain * finite_terms, dim=1, keepdim=True)

    def _c_d_correction_raw(self, features):
        return self.interface_residual_net_v2(features)

    def _bounded_c_d_int(self, c_d_prior, corr_raw, T_raw):
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))
        amp = self._intmemory_corr_amplitude().to(device=T_raw.device, dtype=T_raw.dtype)
        c_d_candidate = c_d_prior + amp * time_gate * torch.tanh(corr_raw)
        eps = 1e-6 * gamma
        return torch.clamp(c_d_candidate, eps, gamma - eps)


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemoryMatchedAbel(
    InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory
):
    """Interface-memory v2 with singularity-matched Abel quadrature.

    This ablation keeps the same trainable interface-state parameters as
    IntMemory, but evaluates the Neumann-to-trace Abel transfer with a quadrature
    matched to the 1/sqrt(t-tau) singularity.  It is intentionally separated
    from the legacy IntMemory path because old checkpoints learned small
    compensations for the previous midpoint Abel rule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.singularity_matched_abel = True
        self.matched_abel_interface = True


class InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemoryMixedAbel(
    InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory
):
    """Interface-memory v2 with a learnable old/matched Abel mixture.

    Directly replacing the midpoint Abel rule with the singularity-matched rule
    changed the effective interface memory too abruptly.  This ablation keeps
    the old midpoint Abel path as the default and lets training decide whether
    to blend in the matched-Abel quadrature:

        A_mix[J] = (1 - lambda) A_midpoint[J] + lambda A_matched[J].

    The scalar lambda is initialized near zero so an IntMemory or TraceGreen
    checkpoint can warm-start this architecture without a large output jump.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixed_abel_interface = True
        self.mixed_abel_logit = nn.Parameter(torch.tensor([-6.0], dtype=torch.float32))

    def _abel_mix_lambda(self):
        return torch.sigmoid(self.mixed_abel_logit)

    def _legacy_abel_convolution_from_grid(self, T_raw, value_grid):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * nodes
        dtau = torch.clamp(t_pos * (1.0 - nodes), min=1e-8 * T_sim)
        value_tau = self._interp_scalar_grid(tau, value_grid)
        diffusion = torch.clamp(self._abel_diffusion().to(device=T_raw.device, dtype=T_raw.dtype), min=1e-8)
        gain = self._abel_gain().to(device=T_raw.device, dtype=T_raw.dtype)
        kernel = gain / torch.sqrt(torch.clamp(np.pi * diffusion * dtau, min=1e-12))
        return T_raw * torch.mean(value_tau * kernel, dim=1, keepdim=True)

    def _matched_abel_convolution_from_grid(self, T_raw, value_grid):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * (1.0 - nodes ** 2)
        value_tau = self._interp_scalar_grid(tau, value_grid)
        diffusion = torch.clamp(self._abel_diffusion().to(device=T_raw.device, dtype=T_raw.dtype), min=1e-8)
        gain = self._abel_gain().to(device=T_raw.device, dtype=T_raw.dtype)
        prefactor = 2.0 * gain * torch.sqrt(t_pos) / torch.sqrt(torch.clamp(np.pi * diffusion, min=1e-12))
        return prefactor * torch.mean(value_tau, dim=1, keepdim=True)

    def _abel_convolution_from_grid(self, T_raw, value_grid):
        lam = self._abel_mix_lambda().to(device=T_raw.device, dtype=T_raw.dtype)
        legacy = self._legacy_abel_convolution_from_grid(T_raw, value_grid)
        matched = self._matched_abel_convolution_from_grid(T_raw, value_grid)
        return (1.0 - lam) * legacy + lam * matched

    def _legacy_abel_from_previous(self, value_prev, lags, dt, device, dtype):
        diffusion = torch.clamp(self._abel_diffusion().to(device=device, dtype=dtype), min=1e-8)
        gain = self._abel_gain().to(device=device, dtype=dtype)
        kernel = gain / torch.sqrt(torch.clamp(np.pi * diffusion * lags, min=1e-12))
        return torch.sum(value_prev * kernel, dim=0, keepdim=True) * dt

    def _matched_abel_from_previous(self, value_prev, lags, dt, device, dtype):
        diffusion = torch.clamp(self._abel_diffusion().to(device=device, dtype=dtype), min=1e-8)
        gain = self._abel_gain().to(device=device, dtype=dtype)
        s_hi = torch.clamp(lags + 0.5 * dt, min=0.0)
        s_lo = torch.clamp(lags - 0.5 * dt, min=0.0)
        weights = 2.0 * gain * (
            torch.sqrt(s_hi) - torch.sqrt(s_lo)
        ) / torch.sqrt(torch.clamp(np.pi * diffusion, min=1e-12))
        return torch.sum(value_prev * weights, dim=0, keepdim=True)

    def _abel_from_previous(self, value_prev, lags, dt, device, dtype):
        lam = self._abel_mix_lambda().to(device=device, dtype=dtype)
        legacy = self._legacy_abel_from_previous(value_prev, lags, dt, device, dtype)
        matched = self._matched_abel_from_previous(value_prev, lags, dt, device, dtype)
        return (1.0 - lam) * legacy + lam * matched


class InterfaceStateNet_v9_6_FilmTraceClean(
    InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory
):
    """Clean final interface state for the Film-TraceGreen architecture.

    This is the paper-facing path: keep the physically interpretable interface
    chain and remove the exploratory dynamic/jump regularizers used during
    debugging.  The retained interface map is

        C_B_int = C_B_surface / (1 + k*delta*C_C_int/D_B)
        J_rxn   = k*C_B_int*C_C_int
        C_D_int = alpha*Abel[J] + positive finite-memory terms
                  - phase*Abel[dJ]
        C_C_int = gamma - C_D_int.

    The contribution audit showed that the small interface residual and the old
    KernelMix beta terms do not help the trained TraceGreen solution, while the
    positive finite-memory boost is essential.  Old TraceGreen checkpoints remain
    compatible because no parameter shapes are changed; this class disables
    ablation-specific training penalties and ignores the non-contributing terms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film_trace_clean_interface = True

        # The final architecture should not depend on auxiliary reversal-jump or
        # direct-dynamic regularizers that were only used to diagnose failures.
        self.continuous_time_features = False
        self.causal_dynamic_correction = False
        self.causal_hybrid_smooth = False
        self.causal_jump_weight = 0.0
        self.direct_jump_weight = 0.0
        self.direct_smooth_weight = 0.0
        self.static_smooth_weight = 0.0
        self.phase_smooth_weight = 0.0
        self.cint_reversal_jump_weight = 0.0
        self.cint_temporal_smooth_weight = 0.0

    def _kernelmix_beta(self):
        return torch.zeros(1, 4, device=self.kernelmix_beta_raw.device, dtype=self.kernelmix_beta_raw.dtype)

    def _c_d_correction_raw(self, features):
        return 0.0 * features[:, 0:1]

    def _bounded_c_d_int(self, c_d_prior, corr_raw, T_raw):
        del corr_raw
        transition_gate = 1.0 - torch.exp(
            -torch.clamp(T_raw, min=0.0) / (0.05 * T_sim)
        )
        mapped = smooth_bounded_concentration(
            c_d_prior,
            gamma,
            transition_fraction=0.02,
            transition_gate=transition_gate,
        )
        # The transition width vanishes continuously as t -> 0; enforce the
        # exact initial trace at the endpoint used by the IC loss/evaluator.
        return torch.where(T_raw <= 0.0, torch.zeros_like(mapped), mapped)


class InterfaceStateNet_v9_6_FilmTraceProductIntegral(
    InterfaceStateNet_v9_6_FilmTraceClean
):
    """Causal interface trace from singularity-matched product integration.

    The clean Film-Abel model used a midpoint Abel rule plus learned finite-memory
    and phase corrections.  That combination is poorly conditioned when the film
    reaction saturates at high gamma, and its smooth concentration bound adds an
    O(gamma) offset near the initial state.  This paper-facing ablation instead
    integrates a piecewise-linear reaction flux analytically on every time cell.

    The final cell contains the unknown current J_n.  Two unrolled fixed-point
    updates followed by analytic Newton projections solve the scalar
    film-reaction/Abel coupling.  Legacy memory
    parameters remain registered for strict checkpoint compatibility but are
    frozen and do not participate in this forward path.
    """

    def __init__(self, *args, fixed_point_iterations=2,
                 newton_projection_iterations=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.product_integral_interface = True
        self.fixed_point_iterations = max(1, int(fixed_point_iterations))
        self.newton_projection_iterations = max(
            1, int(newton_projection_iterations)
        )
        self._last_product_integral_diagnostics = None

        # The interface trace is now a fixed physical operator.  Retain the old
        # tensors only so a clean checkpoint can be loaded with strict=True.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _history_feature_flux_scale(self):
        return max(characteristic_reaction_flux(gamma, self.active_k_cat()), 1e-8)

    def _product_integral_d_j_d_c_d(self, c_b_surface, c_c_int):
        k_value = self._active_k_tensor(c_c_int)
        film_denom = 1.0 + (k_value * delta / D_rel_B) * c_c_int
        return -k_value * c_b_surface / torch.clamp(film_denom.pow(2), min=1e-12)

    def _completed_product_integral(self, j_known, step_index, dt, device, dtype):
        """Integrate completed cells with an exact linear/sqrt-kernel rule."""
        if step_index <= 1:
            return torch.zeros(1, 1, device=device, dtype=dtype)

        j_left = j_known[:-1]
        j_right = j_known[1:]
        delta_j = j_right - j_left
        cells = torch.arange(step_index - 1, device=device, dtype=dtype).reshape(-1, 1)
        lag_hi = (float(step_index) - cells) * dt
        lag_lo = torch.clamp(lag_hi - dt, min=0.0)
        sqrt_hi = torch.sqrt(lag_hi)
        sqrt_lo = torch.sqrt(lag_lo)

        intercept = j_left + delta_j * lag_hi / dt
        cell_integral = (
            2.0 * intercept * (sqrt_hi - sqrt_lo)
            - (2.0 / 3.0) * (delta_j / dt)
            * (lag_hi.pow(1.5) - lag_lo.pow(1.5))
        )
        diffusion = torch.as_tensor(D_rel_D, device=device, dtype=dtype)
        return torch.sum(cell_integral, dim=0, keepdim=True) / torch.sqrt(
            torch.clamp(np.pi * diffusion, min=1e-12)
        )

    def _history_grid(self, T_ref):
        use_cache = self.training and torch.is_grad_enabled()
        cache_key = (
            T_ref.device,
            T_ref.dtype,
            torch.is_grad_enabled(),
            self.condition_cache_key(),
            "product_integral",
            self.fixed_point_iterations,
            self.newton_projection_iterations,
        )
        if use_cache and self._film_abel_cache is not None and self._film_abel_cache[0] == cache_key:
            return self._film_abel_cache[1]

        device = T_ref.device
        dtype = T_ref.dtype
        t_grid = self.time_grid.to(device=device, dtype=dtype)
        dt = float(T_sim) / float(self.time_grid_points - 1)
        diffusion = torch.as_tensor(D_rel_D, device=device, dtype=dtype)
        gamma_t = self._active_gamma_tensor(t_grid).reshape(1, 1)
        normalized_gamma_state = bool(
            getattr(self, "normalized_gamma_product_integral", False)
        )
        current_cell_factor = 2.0 * torch.sqrt(
            torch.as_tensor(dt, device=device, dtype=dtype)
            / torch.clamp(np.pi * diffusion, min=1e-12)
        )
        lambdas = self.film_abel_memory_lambdas.to(device=device, dtype=dtype)
        decay = torch.exp(-dt / torch.clamp(lambdas, min=1e-6))

        j_rows = []
        fixed_point_residual_rows = []
        cb_surface_rows = []
        cb_int_rows = []
        cc_int_rows = []
        cd_int_rows = []
        slope_rows = []
        q_rows = []
        memory_rows = []
        q_hist = torch.zeros(1, 1, device=device, dtype=dtype)
        memory = torch.zeros(1, lambdas.shape[1], device=device, dtype=dtype)

        for idx in range(self.time_grid_points):
            t = t_grid[idx:idx + 1]
            _, _, c_b_surface, _ = self._surface_state(t)

            if idx == 0:
                c_d_int = torch.zeros(1, 1, device=device, dtype=dtype)
                d_int = torch.zeros_like(c_d_int)
                c_c_int = gamma_t
                c_b_int, j_rxn, surface_slope = self._film_reaction(c_b_surface, c_c_int)
                fixed_point_residual = torch.zeros_like(j_rxn)
            else:
                j_known = torch.cat(j_rows, dim=0)
                completed = self._completed_product_integral(
                    j_known, idx, dt, device, dtype
                )
                j_previous = j_known[-1:]
                j_guess = j_previous

                if normalized_gamma_state:
                    completed_state = completed / gamma_t
                    current_cell_state_factor = current_cell_factor / gamma_t
                else:
                    completed_state = completed
                    current_cell_state_factor = current_cell_factor

                for _ in range(self.fixed_point_iterations):
                    state_candidate = completed_state + current_cell_state_factor * (
                        (1.0 / 3.0) * j_previous + (2.0 / 3.0) * j_guess
                    )
                    c_c_candidate = (
                        gamma_t * (1.0 - state_candidate)
                        if normalized_gamma_state
                        else gamma_t - state_candidate
                    )
                    _, j_guess, _ = self._film_reaction(c_b_surface, c_c_candidate)

                state_int = completed_state + current_cell_state_factor * (
                    (1.0 / 3.0) * j_previous + (2.0 / 3.0) * j_guess
                )
                # Repeated analytic Newton projections close the stiff scalar
                # equation at high Damkohler number.  The fixed-parameter
                # baseline keeps one projection; kparam uses a few unrolled
                # projections without adding learned degrees of freedom.
                for _ in range(self.newton_projection_iterations):
                    c_c_int = (
                        gamma_t * (1.0 - state_int)
                        if normalized_gamma_state
                        else gamma_t - state_int
                    )
                    _, j_rxn, _ = self._film_reaction(c_b_surface, c_c_int)
                    d_j_d_c_d = self._product_integral_d_j_d_c_d(
                        c_b_surface, c_c_int
                    )
                    d_j_d_state = (
                        gamma_t * d_j_d_c_d
                        if normalized_gamma_state
                        else d_j_d_c_d
                    )
                    closure = state_int - completed_state - current_cell_state_factor * (
                        (1.0 / 3.0) * j_previous + (2.0 / 3.0) * j_rxn
                    )
                    closure_derivative = (
                        1.0
                        - current_cell_state_factor * (2.0 / 3.0) * d_j_d_state
                    )
                    state_int = state_int - closure / torch.clamp(
                        closure_derivative, min=1e-8
                    )
                if normalized_gamma_state:
                    d_int = state_int
                    c_d_int = gamma_t * d_int
                    c_c_int = gamma_t * (1.0 - d_int)
                else:
                    c_d_int = state_int
                    d_int = c_d_int / gamma_t
                    c_c_int = gamma_t - c_d_int
                c_b_int, j_rxn, surface_slope = self._film_reaction(c_b_surface, c_c_int)
                fixed_point_residual = state_int - completed_state - current_cell_state_factor * (
                    (1.0 / 3.0) * j_previous + (2.0 / 3.0) * j_rxn
                )

            q_next = q_hist + j_rxn * dt
            memory_next = decay * memory + j_rxn * dt
            cb_surface_rows.append(c_b_surface)
            cb_int_rows.append(c_b_int)
            cc_int_rows.append(c_c_int)
            cd_int_rows.append(c_d_int)
            j_rows.append(j_rxn)
            fixed_point_residual_rows.append(fixed_point_residual)
            slope_rows.append(surface_slope)
            q_rows.append(q_next)
            memory_rows.append(memory_next)
            q_hist = q_next
            memory = memory_next

        j_grid = torch.cat(j_rows, dim=0)
        c_d_grid = torch.cat(cd_int_rows, dim=0)
        d_grid = c_d_grid / gamma_t
        if self.time_grid_points > 1:
            d_j_grid = torch.zeros_like(j_grid)
            d_j_grid[1:-1] = (j_grid[2:] - j_grid[:-2]) / (2.0 * dt)
            d_j_grid[0] = (j_grid[1] - j_grid[0]) / dt
            d_j_grid[-1] = (j_grid[-1] - j_grid[-2]) / dt
        else:
            d_j_grid = torch.zeros_like(j_grid)

        fixed_point_residual = torch.cat(fixed_point_residual_rows, dim=0)
        diagnostic_state = d_grid if normalized_gamma_state else c_d_grid
        diagnostic_upper = torch.ones_like(gamma_t) if normalized_gamma_state else gamma_t
        lower_violation = torch.relu(-diagnostic_state)
        upper_violation = torch.relu(diagnostic_state - diagnostic_upper)
        self._last_product_integral_diagnostics = {
            "c_d_min": c_d_grid.detach().min(),
            "c_d_max": c_d_grid.detach().max(),
            "bounds_max": torch.maximum(lower_violation.max(), upper_violation.max()).detach(),
            "fixed_point_max": fixed_point_residual.detach().abs().max(),
            "all_finite": torch.isfinite(c_d_grid.detach()).all() & torch.isfinite(j_grid.detach()).all(),
        }

        history = {
            "C_B_surface": torch.cat(cb_surface_rows, dim=0),
            "C_B_int": torch.cat(cb_int_rows, dim=0),
            "C_C_int": torch.cat(cc_int_rows, dim=0),
            "C_D_int": c_d_grid,
            "C_D_fraction": d_grid,
            "C_D_prior": c_d_grid,
            "J": j_grid,
            "dJ": d_j_grid,
            "surface_slope": torch.cat(slope_rows, dim=0),
            "Q": torch.cat(q_rows, dim=0),
            "M": torch.cat(memory_rows, dim=0),
        }
        if use_cache:
            self._film_abel_cache = (cache_key, history)
        return history

    def forward(self, T_raw):
        # Interpolate the already solved causal trace.  Re-evaluating the old
        # continuous-time Abel/bounded path here would reintroduce its bias.
        history = self._history_grid(T_raw)
        gamma_t = self._active_gamma_tensor(T_raw)
        if getattr(self, "normalized_gamma_product_integral", False):
            d_int = self._interp_multi_grid(T_raw, history["C_D_fraction"])
            c_d_int = gamma_t * d_int
            c_c_int = gamma_t * (1.0 - d_int)
        else:
            c_d_int = self._interp_multi_grid(T_raw, history["C_D_int"])
            c_c_int = gamma_t - c_d_int
        _, _, c_b_surface, _ = self._surface_state(T_raw)
        c_b_int, j_rxn, surface_slope = self._film_reaction(c_b_surface, c_c_int)
        return {
            "C_B_surface": c_b_surface,
            "C_B_int": c_b_int,
            "C_C_int": c_c_int,
            "J_rxn": j_rxn,
            "dJ_rxn_dt": self._interp_multi_grid(T_raw, history["dJ"]),
            "Q_rxn": self._interp_multi_grid(T_raw, history["Q"]),
            "surface_slope": surface_slope,
        }


class InterfaceStateNet_v9_6_FilmTraceKParam(
    InterfaceStateNet_v9_6_FilmTraceProductIntegral
):
    """Product-integral interface with one runtime k condition per step.

    The active k enters both the film reaction and the implicit singular-cell
    closure.  Thus every condition solves its own causal Volterra trace instead
    of reusing the old learned Film-Abel correction calibrated at k=1.
    """

    def __init__(self, *args, k_reference=KPARAM_REFERENCE,
                 newton_projection_iterations=4, **kwargs):
        super().__init__(
            *args,
            newton_projection_iterations=newton_projection_iterations,
            **kwargs,
        )
        self.k_parameterized = True
        self.kparam_interface_operator = KPARAM_INTERFACE_OPERATOR
        self.k_reference = float(k_reference)
        if not np.isfinite(self.k_reference) or self.k_reference <= 0.0:
            raise ValueError("k_reference must be finite and positive")
        self.register_buffer(
            "_k_cat_condition",
            torch.tensor(self.k_reference, dtype=torch.float64),
            persistent=False,
        )

    def set_k_cat(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"k_cat must be finite and positive, got {value}")
        if not np.isclose(value, self.active_k_cat(), rtol=0.0, atol=1e-12):
            self._k_cat_condition.fill_(value)
            self.clear_step_cache()

    def active_k_cat(self):
        return float(self._k_cat_condition.detach().cpu())

    def _active_k_tensor(self, reference):
        return self._k_cat_condition.to(device=reference.device, dtype=reference.dtype)

    def condition_cache_key(self):
        return self.active_k_cat()

    def log_k_condition(self, reference):
        value = self._k_cat_condition.to(device=reference.device, dtype=reference.dtype)
        return torch.tanh(torch.log10(value / self.k_reference) / KPARAM_LOG10_SCALE)

    def _film_transfer_fraction(self, c_c_int):
        c_positive = torch.clamp(c_c_int, min=torch.finfo(c_c_int.dtype).tiny)
        k_value = self._active_k_tensor(c_positive)
        log_da = (
            torch.log(k_value)
            + torch.log(c_positive)
            + np.log(delta / D_rel_B)
        )
        return torch.sigmoid(log_da)

    def _film_reaction(self, c_b_surface, c_c_int):
        transfer = self._film_transfer_fraction(c_c_int)
        c_b_int = c_b_surface * (1.0 - transfer)
        j_rxn = (D_rel_B / delta) * c_b_surface * transfer
        surface_slope = -j_rxn / D_rel_B
        return c_b_int, j_rxn, surface_slope

    def _product_integral_d_j_d_c_d(self, c_b_surface, c_c_int):
        c_positive = torch.clamp(c_c_int, min=torch.finfo(c_c_int.dtype).tiny)
        transfer = self._film_transfer_fraction(c_positive)
        return -(
            (D_rel_B / delta)
            * c_b_surface
            * transfer
            * (1.0 - transfer)
            / c_positive
        )

    def _history_feature_flux_scale(self):
        return max(
            characteristic_reaction_flux(gamma, self.active_k_cat()),
            1e-8,
        )


class InterfaceStateNet_v9_6_FilmTraceGammaParam(
    InterfaceStateNet_v9_6_FilmTraceKParam
):
    """Product-integral interface with normalized C_D/gamma state."""

    def __init__(self, *args, gamma_reference=GAMMAPARAM_REFERENCE,
                 newton_projection_iterations=1, **kwargs):
        super().__init__(
            *args,
            k_reference=REFERENCE_K_CAT_STAR,
            newton_projection_iterations=newton_projection_iterations,
            **kwargs,
        )
        self.k_parameterized = False
        self.gamma_parameterized = True
        self.normalized_gamma_product_integral = True
        self.gammaparam_interface_operator = GAMMAPARAM_INTERFACE_OPERATOR
        self.gamma_reference = float(gamma_reference)
        if not np.isfinite(self.gamma_reference) or self.gamma_reference <= 0.0:
            raise ValueError("gamma_reference must be finite and positive")
        self.register_buffer(
            "_gamma_condition",
            torch.tensor(self.gamma_reference, dtype=torch.float64),
            persistent=False,
        )

    def set_gamma(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"gamma must be finite and positive, got {value}")
        if not np.isclose(value, self.active_gamma(), rtol=0.0, atol=1e-12):
            self._gamma_condition.fill_(value)
            self.clear_step_cache()

    def active_gamma(self):
        return float(self._gamma_condition.detach().cpu())

    def _active_gamma_tensor(self, reference):
        return self._gamma_condition.to(device=reference.device, dtype=reference.dtype)

    def condition_cache_key(self):
        return ("gamma", self.active_gamma())

    def log_gamma_condition(self, reference):
        value = self._active_gamma_tensor(reference)
        return torch.tanh(torch.log10(value / self.gamma_reference))

    def _history_feature_flux_scale(self):
        return max(
            characteristic_reaction_flux(self.active_gamma(), REFERENCE_K_CAT_STAR),
            1e-8,
        )


class HISInterfaceStateNet_v9_6(nn.Module):
    """Minimal interface-state block for the HIS-PINN prototype.

    Baseline PINN: predicts C(x,t) directly with an MLP.
    Hard-constrained window/buffer PINN: predicts a free field and corrects
    boundary/interface mismatch.
    HIS-PINN prototype: predicts physically meaningful endpoint states and
    reconstructs the thin-layer profile with Hermite interpolation.

    The current v9.6 equations do not include an electric potential PDE, so
    phi-related states are exposed for future extensions but are not used in the
    concentration residuals yet.
    """

    def __init__(self, normalize_inputs=True):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.net = MultiscaleResidualHead(out_features=7, width=160, depth=3, in_features=1)
        self.slope_scale = 8.0

    def forward(self, T_raw):
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
        else:
            t_net = T_raw

        raw = self.net(t_net)
        theta = potential_theta(T_raw)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))

        c_L = torch.sigmoid(-theta)
        c_R = time_gate * torch.sigmoid(raw[:, 0:1])
        dc_dx_L = self.slope_scale * time_gate * torch.tanh(raw[:, 1:2])
        dc_dx_R = self.slope_scale * time_gate * torch.tanh(raw[:, 2:3])

        phi_L = theta
        phi_R = phi_L + 0.1 * torch.tanh(raw[:, 3:4])
        dphi_dx_L = self.slope_scale * time_gate * torch.tanh(raw[:, 4:5])
        dphi_dx_R = self.slope_scale * time_gate * torch.tanh(raw[:, 5:6])

        # Optional interface reaction proxy for diagnostics/future losses.  The
        # current training loop still uses J_rxn = k*C_B_int*C_C_int.
        reaction_flux = -D_rel_B * dc_dx_R

        return {
            "c_L": c_L,
            "c_R": c_R,
            "dc_dx_L": dc_dx_L,
            "dc_dx_R": dc_dx_R,
            "phi_L": phi_L,
            "phi_R": phi_R,
            "dphi_dx_L": dphi_dx_L,
            "dphi_dx_R": dphi_dx_R,
            "reaction_flux": reaction_flux,
            "reaction_flux_free": time_gate * torch.tanh(raw[:, 6:7]),
        }


class ThinLayerNet_v9_6_HISPrototype(nn.Module):
    """Thin-layer Hermite Interface-State PINN prototype.

    This module is intentionally small: only the thin-layer B concentration is
    reconstructed with Hermite endpoint states.  The external/bulk region can
    remain a normal MLP so the baseline project structure stays intact.
    """

    def __init__(self, interface_state=None, normalize_inputs=True):
        super().__init__()
        self.interface_state = interface_state if interface_state is not None else HISInterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        state = self.interface_state(T_raw)
        C_B = hermite_reconstruct(
            X_raw,
            0.0,
            delta,
            state["c_L"],
            state["dc_dx_L"],
            state["c_R"],
            state["dc_dx_R"],
        )
        C_A = 1.0 - C_B
        return C_A, C_B


class ThinLayerNet_v9_6_MultiscaleHermite(nn.Module):
    def __init__(self, interface_state=None, normalize_inputs=True):
        super().__init__()
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        # Keep this name/shape compatible with MultiscaleHardBC so old checkpoints
        # can seed the correction network when strict=False loading is used.
        self.net = MultiscaleResidualHead(out_features=2, width=256, depth=5)
        self.correction_scale = 1.5
        self.surface_bubble_scale = 0.25

    def _raw_field(self, x_net, T_raw, X_raw):
        del T_raw, X_raw
        return self.net(x_net)

    def _network_inputs(self, T_raw, X_raw):
        if self.normalize_inputs:
            return torch.cat([
                normalize_time(T_raw),
                normalize_thin_x(X_raw),
            ], dim=1)
        return torch.cat([T_raw, X_raw], dim=1)

    def surface_current_state(self, T_raw):
        """Electrode current encoded by the Hermite left-end slope.

        ``surface_bubble = s(1-s)^2`` is exactly the cubic Hermite ``h10``
        basis.  Its coefficient can therefore be interpreted as a correction
        to the left-end B slope rather than an unrelated field residual.
        """
        X_surface = torch.zeros_like(T_raw)
        x_net = self._network_inputs(T_raw, X_surface)
        state = self.interface_state(T_raw)
        raw_surface = self._raw_field(x_net, T_raw, X_surface)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))
        b_slope = (
            state["surface_slope"] +
            (self.surface_bubble_scale / delta) * time_gate *
            torch.tanh(raw_surface[:, 1:2])
        )
        return D_rel_A * b_slope

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        x_net = self._network_inputs(T_raw, X_raw)

        state = self.interface_state(T_raw)
        c_b_surface = state["C_B_surface"]
        c_b_int = state["C_B_int"]
        j_rxn = state["J_rxn"]
        surface_slope = state["surface_slope"]
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))

        interface_slope = -j_rxn / D_rel_B

        s = torch.clamp(X_raw / delta, 0.0, 1.0)
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        c_b_base = (
            h00 * c_b_surface +
            h10 * delta * surface_slope +
            h01 * c_b_int +
            h11 * delta * interface_slope
        )

        raw_field = self._raw_field(x_net, T_raw, X_raw)
        endpoint_bubble = s2 * (1.0 - s) ** 2
        surface_bubble = s * (1.0 - s) ** 2
        correction = time_gate * (
            self.surface_bubble_scale * surface_bubble * torch.tanh(raw_field[:, 1:2]) +
            self.correction_scale * endpoint_bubble * torch.tanh(raw_field[:, 0:1])
        )
        C_B = c_b_base + correction
        C_A = 1.0 - C_B
        return C_A, C_B


class ThinLayerNet_v9_6_MultiscaleHermiteKParam(ThinLayerNet_v9_6_MultiscaleHermite):
    """Hermite thin field with a zero-at-k=1 conditional correction adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_parameterized = True
        self.k_adapter = MultiscaleResidualHead(
            out_features=2,
            width=64,
            depth=2,
            in_features=3,
        )
        nn.init.zeros_(self.k_adapter.net[-1].weight)
        nn.init.zeros_(self.k_adapter.net[-1].bias)

    def active_k_cat(self):
        return self.interface_state.active_k_cat()

    def set_k_cat(self, value):
        self.interface_state.set_k_cat(value)

    def forward(self, x_input):
        base_input = activate_k_from_input(self.interface_state, x_input)
        return super().forward(base_input)

    def _raw_field(self, x_net, T_raw, X_raw):
        base = self.net(x_net)
        kappa = self.interface_state.log_k_condition(T_raw)
        k_feature = torch.ones_like(T_raw) * kappa
        if self.normalize_inputs:
            adapter_input = torch.cat([
                normalize_time(T_raw),
                normalize_thin_x(X_raw),
                k_feature,
            ], dim=1)
        else:
            adapter_input = torch.cat([T_raw, X_raw, k_feature], dim=1)
        return base + k_feature * self.k_adapter(adapter_input)


class ThinLayerNet_v9_6_MultiscaleHermiteGammaParam(
    ThinLayerNet_v9_6_MultiscaleHermite
):
    """Hermite field with a zero-at-gamma=10 conditional adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_parameterized = True
        self.gamma_adapter = MultiscaleResidualHead(
            out_features=2,
            width=64,
            depth=2,
            in_features=3,
        )
        nn.init.zeros_(self.gamma_adapter.net[-1].weight)
        nn.init.zeros_(self.gamma_adapter.net[-1].bias)

    def active_gamma(self):
        return self.interface_state.active_gamma()

    def set_gamma(self, value):
        self.interface_state.set_gamma(value)

    def forward(self, x_input):
        base_input = activate_gamma_from_input(self.interface_state, x_input)
        return super().forward(base_input)

    def _raw_field(self, x_net, T_raw, X_raw):
        base = self.net(x_net)
        eta = self.interface_state.log_gamma_condition(T_raw)
        gamma_feature = torch.ones_like(T_raw) * eta
        if self.normalize_inputs:
            adapter_input = torch.cat([
                normalize_time(T_raw),
                normalize_thin_x(X_raw),
                gamma_feature,
            ], dim=1)
        else:
            adapter_input = torch.cat([T_raw, X_raw, gamma_feature], dim=1)
        return base + gamma_feature * self.gamma_adapter(adapter_input)


class ThinLayerNet_v9_6_MultiscaleHermiteConservative(ThinLayerNet_v9_6_MultiscaleHermite):
    """Checkpoint-compatible Hermite thin layer with inventory supervision."""

    conservative_thin_current = True


class ThinLayerNet_v9_6_InventoryHermiteLift(
    ThinLayerNet_v9_6_MultiscaleHermiteConservative
):
    """Causal Hermite lift that enforces the thin-film inventory identity.

    Starting from the checkpoint-compatible clean field C_B^0, the lift adds
    ``delta * a(t) * h10(x/delta)``.  It leaves both endpoint values and the
    right endpoint derivative unchanged while correcting the electrode slope.
    The amplitude solves

        (delta**2 / 12) * da/dt + D_A * a = J_cons^0 - J_surface^0,

    so the corrected surface current and corrected inventory current agree by
    construction, up to the causal time-grid approximation.
    """

    inventory_hermite_lift = True

    def __init__(
        self,
        *args,
        lift_time_grid_points=1024,
        lift_quadrature_points=16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lift_time_grid_points = max(64, int(lift_time_grid_points))
        self.lift_quadrature_points = max(4, int(lift_quadrature_points))
        self._lift_eval_cache = None

    @property
    def lift_inventory_coefficient(self):
        return float(delta ** 2 / 12.0)

    def clear_lift_cache(self):
        self._lift_eval_cache = None

    def train(self, mode=True):
        if mode:
            self.clear_lift_cache()
        return super().train(mode)

    def base_forward(self, x_input):
        return super().forward(x_input)

    def base_surface_current_state(self, T_raw):
        return super().surface_current_state(T_raw)

    @staticmethod
    def _causal_etd_step(a0, source0, source_slope, h, tau):
        decay_rate = D_rel_A / tau
        one_minus_decay = -torch.expm1(-decay_rate * h)
        decay = 1.0 - one_minus_decay
        return (
            decay * a0 +
            (one_minus_decay / D_rel_A) * source0 +
            (source_slope / D_rel_A) *
            (h - one_minus_decay / decay_rate)
        )

    def _compute_lift_grid(self, device, dtype, create_graph=False):
        if self._lift_eval_cache is not None:
            cache = self._lift_eval_cache
            if cache[0].device == device and cache[0].dtype == dtype:
                return cache

        with torch.enable_grad():
            T_grid = torch.linspace(
                0.0,
                float(T_sim),
                self.lift_time_grid_points,
                device=device,
                dtype=dtype,
            ).reshape(-1, 1)
            T_grid.requires_grad_(True)

            nodes_np, weights_np = np.polynomial.legendre.leggauss(
                self.lift_quadrature_points
            )
            nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
            weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
            X_quad = 0.5 * delta * (nodes + 1.0)
            T_quad = T_grid.expand(-1, self.lift_quadrature_points)
            X_quad_full = X_quad.reshape(1, -1).expand(T_grid.shape[0], -1)
            inputs = torch.stack(
                [T_quad.reshape(-1), X_quad_full.reshape(-1)], dim=1
            )
            _, C_B_base = self.base_forward(inputs)
            C_B_base = C_B_base.reshape(
                T_grid.shape[0], self.lift_quadrature_points
            )
            M_base = 0.5 * delta * torch.sum(
                C_B_base * weights.reshape(1, -1), dim=1, keepdim=True
            )
            dM_base_dt = torch.autograd.grad(
                M_base.sum(),
                T_grid,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]

            state = self.interface_state(T_grid)
            J_surface_base = self.base_surface_current_state(T_grid)
            J_conservative_base = -state["J_rxn"] - dM_base_dt
            source = J_conservative_base - J_surface_base

            dt = float(T_sim) / float(self.lift_time_grid_points - 1)
            source_slope = torch.cat([
                torch.zeros_like(source[:1]),
                (source[1:] - source[:-1]) / dt,
            ], dim=0)
            tau = self.lift_inventory_coefficient
            h_step = torch.as_tensor(dt, device=device, dtype=dtype)
            amplitudes = [torch.zeros_like(source[0])]
            for index in range(self.lift_time_grid_points - 1):
                amplitudes.append(self._causal_etd_step(
                    amplitudes[-1],
                    source[index],
                    source_slope[index],
                    h_step,
                    tau,
                ))
            amplitude_grid = torch.stack(amplitudes, dim=0).reshape(-1, 1)

        result = (T_grid, source, source_slope, amplitude_grid)
        if not create_graph:
            result = tuple(value.detach() for value in result)
        self._lift_eval_cache = result
        return result

    def lift_amplitude(self, T_raw):
        create_graph = bool(self.training and torch.is_grad_enabled())
        T_grid, source, source_slope, amplitude_grid = self._compute_lift_grid(
            T_raw.device,
            T_raw.dtype,
            create_graph=create_graph,
        )
        dt = float(T_sim) / float(self.lift_time_grid_points - 1)
        T_eval = torch.clamp(T_raw, 0.0, float(T_sim))
        index = torch.floor(T_eval / dt).to(torch.long)
        index = torch.clamp(index, 0, self.lift_time_grid_points - 1)
        T_left = index.to(T_raw.dtype) * dt
        h = T_eval - T_left
        a0 = amplitude_grid[index.reshape(-1)].reshape_as(T_raw)
        source0 = source[index.reshape(-1)].reshape_as(T_raw)
        slope0 = source_slope[index.reshape(-1)].reshape_as(T_raw)
        return self._causal_etd_step(
            a0,
            source0,
            slope0,
            h,
            self.lift_inventory_coefficient,
        )

    def surface_current_state(self, T_raw):
        return (
            self.base_surface_current_state(T_raw) +
            D_rel_A * self.lift_amplitude(T_raw)
        )

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        C_A_base, C_B_base = self.base_forward(x_input)
        s = torch.clamp(X_raw / delta, 0.0, 1.0)
        _, h10, _, _ = hermite_cubic_basis(s)
        correction = delta * self.lift_amplitude(T_raw) * h10
        C_B = C_B_base + correction
        C_A = C_A_base - correction
        return C_A, C_B


class ThinLayerNet_v9_6_InventoryHermiteLiftKParam(
    ThinLayerNet_v9_6_InventoryHermiteLift,
    ThinLayerNet_v9_6_MultiscaleHermiteKParam,
):
    """Posterior-only inventory lift over the trained k-conditional field."""

    k_parameterized = True
    kparam_posterior_lift = True

    def forward(self, x_input):
        base_input = activate_k_from_input(self.interface_state, x_input)
        return ThinLayerNet_v9_6_InventoryHermiteLift.forward(self, base_input)


class ThinLayerNet_v9_6_InventoryHermiteLiftGammaParam(
    ThinLayerNet_v9_6_InventoryHermiteLift,
    ThinLayerNet_v9_6_MultiscaleHermiteGammaParam,
):
    """Posterior-only inventory lift over the gamma-conditional field."""

    gamma_parameterized = True
    gammaparam_posterior_lift = True

    def forward(self, x_input):
        base_input = activate_gamma_from_input(self.interface_state, x_input)
        return ThinLayerNet_v9_6_InventoryHermiteLift.forward(self, base_input)


class ThinLayerNet_v9_6_MultiscaleHermiteMixedFlux(
    ThinLayerNet_v9_6_MultiscaleHermiteConservative
):
    """Hermite concentration plus an explicit conservative B-flux field."""

    mixed_flux_thin = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_net = MultiscaleResidualHead(
            out_features=1,
            width=128,
            depth=3,
            in_features=2,
        )
        nn.init.zeros_(self.flux_net.net[-1].weight)
        nn.init.zeros_(self.flux_net.net[-1].bias)

    def flux_b(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        s = torch.clamp(X_raw / delta, 0.0, 1.0)
        state = self.interface_state(T_raw)
        j_electrode = self.surface_current_state(T_raw)
        q_surface = -(D_rel_B / D_rel_A) * j_electrode
        q_interface = state["J_rxn"]
        raw_flux = self.flux_net(self._network_inputs(T_raw, X_raw))
        active_k = model_active_k_cat(model_thin=self)
        j_ref = characteristic_reaction_flux(gamma, active_k)
        interior = j_ref * s * (1.0 - s) * torch.tanh(raw_flux)
        return (1.0 - s) * q_surface + s * q_interface + interior


def thin_current_components(model_thin, T_raw, quadrature_points=16, create_graph=False):
    """Return surface, reaction, inventory, and conservative thin-film currents."""
    quadrature_points = max(4, int(quadrature_points))
    if not T_raw.requires_grad:
        T_raw = T_raw.detach().clone().requires_grad_(True)

    X_surface = torch.zeros_like(T_raw, requires_grad=True)
    active_k = model_active_k_cat(model_thin=model_thin)
    active_condition = (
        model_active_gamma(model_thin=model_thin)
        if is_gamma_parameterized(model_thin=model_thin) else active_k
    )
    C_A_surface, _ = model_thin(
        conditioned_model_inputs(model_thin, T_raw, X_surface, active_condition)
    )
    C_A_X_surface = torch.autograd.grad(
        C_A_surface.sum(),
        X_surface,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    J_surface = -D_rel_A * C_A_X_surface

    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_points)
    nodes = torch.as_tensor(nodes_np, device=T_raw.device, dtype=T_raw.dtype)
    weights = torch.as_tensor(weights_np, device=T_raw.device, dtype=T_raw.dtype)
    X_quad = (0.5 * delta * (nodes + 1.0)).reshape(1, -1)
    X_quad = X_quad.expand(T_raw.shape[0], -1)
    T_quad = T_raw.expand(-1, quadrature_points)
    quad_t = T_quad.reshape(-1, 1)
    quad_x = X_quad.reshape(-1, 1)
    quad_inputs = conditioned_model_inputs(model_thin, quad_t, quad_x, active_condition)
    _, C_B_quad = model_thin(quad_inputs)
    C_B_quad = C_B_quad.reshape(T_raw.shape[0], quadrature_points)
    M_B = 0.5 * delta * torch.sum(C_B_quad * weights.reshape(1, -1), dim=1, keepdim=True)
    dM_B_dt = torch.autograd.grad(
        M_B.sum(),
        T_raw,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    state = model_thin.interface_state(T_raw)
    J_reaction = -state["J_rxn"]
    J_inventory = -dM_B_dt
    J_conservative = J_reaction + J_inventory
    return {
        "J_surface": J_surface,
        "J_reaction": J_reaction,
        "J_inventory": J_inventory,
        "J_conservative": J_conservative,
        "balance_residual": J_surface - J_conservative,
        "M_B": M_B,
    }


class ExternalNet_v9_6_Multiscale(nn.Module):
    def __init__(self, gamma_val, normalize_inputs=True):
        super().__init__()
        self.gamma = gamma_val
        self.normalize_inputs = normalize_inputs
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            x_input = torch.cat([
                normalize_time(T_raw),
                normalize_ext_x(X_raw)
            ], dim=1)
        raw = self.net(x_input)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        farfield_gate = 1.0 - torch.clamp((X_raw - delta) / (X_ext_max - delta), 0.0, 1.0)
        C_D = self.gamma * time_gate * farfield_gate * torch.sigmoid(raw)
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleHermite(nn.Module):
    def __init__(self, gamma_val, interface_state=None, normalize_inputs=True):
        super().__init__()
        self.gamma = gamma_val
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)
        self.correction_scale = 2.0

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            x_net = torch.cat([
                normalize_time(T_raw),
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            x_net = x_input

        state = self.interface_state(T_raw)
        c_c_int = state["C_C_int"]
        j_rxn = state["J_rxn"]
        interface_slope = j_rxn / D_rel_C
        far_slope = torch.zeros_like(interface_slope)
        ext_length = X_ext_max - delta

        r = torch.clamp((X_raw - delta) / ext_length, 0.0, 1.0)
        r2 = r * r
        r3 = r2 * r
        h00 = 2.0 * r3 - 3.0 * r2 + 1.0
        h10 = r3 - 2.0 * r2 + r
        h01 = -2.0 * r3 + 3.0 * r2
        h11 = r3 - r2

        c_c_base = (
            h00 * c_c_int +
            h10 * ext_length * interface_slope +
            h01 * self.gamma +
            h11 * ext_length * far_slope
        )

        raw = self.net(x_net)
        endpoint_bubble = r2 * (1.0 - r) ** 2
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        correction = self.correction_scale * time_gate * endpoint_bubble * torch.tanh(raw)
        C_C = c_c_base + correction
        C_D = self.gamma - C_C
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleHermiteExtBasis(nn.Module):
    def __init__(self, gamma_val, interface_state=None, normalize_inputs=True):
        super().__init__()
        self.gamma = gamma_val
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        # Keep this head compatible with the first Hermite external net so its
        # checkpoint can seed the new architecture with strict=False loading.
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)
        self.shape_net = MultiscaleResidualHead(out_features=2, width=192, depth=4)
        self.beta_net = MultiscaleResidualHead(out_features=1, width=128, depth=3, in_features=1)
        self.register_buffer(
            "correction_scales",
            torch.tensor([18.0, 12.0, 12.0], dtype=torch.float32).reshape(1, 3),
        )

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([
                t_net,
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            t_net = T_raw
            x_net = x_input

        state = self.interface_state(T_raw)
        c_c_int = state["C_C_int"]
        j_rxn = state["J_rxn"]
        interface_slope = j_rxn / D_rel_C
        far_slope = torch.zeros_like(interface_slope)
        ext_length = X_ext_max - delta

        r = torch.clamp((X_raw - delta) / ext_length, 0.0, 1.0)
        beta = 0.5 + 12.0 * torch.sigmoid(self.beta_net(t_net))
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = (1.0 - torch.exp(-beta * r)) / denom
        z = torch.clamp(z, 0.0, 1.0)
        dz_dx_at_interface = beta / (ext_length * denom)

        z2 = z * z
        z3 = z2 * z
        h00 = 2.0 * z3 - 3.0 * z2 + 1.0
        h10 = z3 - 2.0 * z2 + z
        h01 = -2.0 * z3 + 3.0 * z2
        h11 = z3 - z2

        c_c_base = (
            h00 * c_c_int +
            h10 * interface_slope / dz_dx_at_interface +
            h01 * self.gamma +
            h11 * far_slope
        )

        raw_main = self.net(x_net)
        raw_shape = self.shape_net(x_net)
        raw = torch.cat([raw_main, raw_shape], dim=1)
        endpoint_bubble = z2 * (1.0 - z) ** 2
        modes = torch.cat([
            endpoint_bubble,
            endpoint_bubble * (2.0 * z - 1.0),
            endpoint_bubble * (6.0 * z2 - 6.0 * z + 1.0),
        ], dim=1)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        correction = time_gate * torch.sum(self.correction_scales * modes * torch.tanh(raw), dim=1, keepdim=True)
        C_C = c_c_base + correction
        C_D = self.gamma - C_C
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleGreenKernel(nn.Module):
    def __init__(self, gamma_val, interface_state=None, normalize_inputs=True, kernel_points=8,
                 detach_history=True):
        super().__init__()
        self.gamma = gamma_val
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        self.detach_history = detach_history
        # Names match ExtBasis where possible so an extbasis checkpoint can seed
        # the finite-domain residual nets with strict=False loading.
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)
        self.shape_net = MultiscaleResidualHead(out_features=2, width=192, depth=4)
        self.beta_net = MultiscaleResidualHead(out_features=1, width=128, depth=3, in_features=1)
        self.register_buffer(
            "kernel_nodes",
            (torch.arange(kernel_points, dtype=torch.float32) + 0.5) / kernel_points,
        )
        self.register_buffer(
            "correction_scales",
            torch.tensor([8.0, 6.0, 6.0], dtype=torch.float32).reshape(1, 3),
        )

    def green_flux_convolution(self, T_raw, y):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(T_raw)
        tau = t_pos * nodes
        dt = torch.clamp(t_pos * (1.0 - nodes), min=1e-6 * T_sim)

        tau_flat = tau.reshape(-1, 1)
        if self.detach_history:
            with torch.no_grad():
                state_tau = self.interface_state(tau_flat)
                q_tau = state_tau["J_rxn"].reshape(T_raw.shape[0], -1)
        else:
            state_tau = self.interface_state(tau_flat)
            q_tau = state_tau["J_rxn"].reshape(T_raw.shape[0], -1)

        y_pos = torch.clamp(y, min=0.0)
        y_mat = y_pos.expand_as(dt)
        kernel = torch.exp(-(y_mat ** 2) / (4.0 * D_rel_D * dt))
        kernel = kernel / torch.sqrt(torch.clamp(np.pi * D_rel_D * dt, min=1e-12))

        c_d = T_raw * torch.mean(q_tau * kernel, dim=1, keepdim=True)
        c_d_y = T_raw * torch.mean(
            q_tau * kernel * (-y_mat / (2.0 * D_rel_D * dt)),
            dim=1,
            keepdim=True,
        )
        return c_d, c_d_y

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([
                t_net,
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            t_net = T_raw
            x_net = x_input

        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        j_rxn = state["J_rxn"]
        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)

        r = torch.clamp(y / ext_length, 0.0, 1.0)
        beta = 0.5 + 12.0 * torch.sigmoid(self.beta_net(t_net))
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = (1.0 - torch.exp(-beta * r)) / denom
        z = torch.clamp(z, 0.0, 1.0)
        dz_dx_at_interface = beta / (ext_length * denom)

        c_d_green, c_d_green_y = self.green_flux_convolution(T_raw, y)
        zeros = torch.zeros_like(y)
        far_y = torch.ones_like(y) * ext_length
        c_d_green_0, c_d_green_y_0 = self.green_flux_convolution(T_raw, zeros)
        c_d_green_far, c_d_green_y_far = self.green_flux_convolution(T_raw, far_y)

        z2 = z * z
        z3 = z2 * z
        h00 = 2.0 * z3 - 3.0 * z2 + 1.0
        h10 = z3 - 2.0 * z2 + z
        h01 = -2.0 * z3 + 3.0 * z2

        target_interface_slope = -j_rxn / D_rel_D
        target_far_value = torch.zeros_like(d_int)
        c_d_base = (
            c_d_green +
            h00 * (d_int - c_d_green_0) +
            h10 * (target_interface_slope - c_d_green_y_0) / dz_dx_at_interface +
            h01 * (target_far_value - c_d_green_far)
        )

        raw_main = self.net(x_net)
        raw_shape = self.shape_net(x_net)
        raw = torch.cat([raw_main, raw_shape], dim=1)
        endpoint_bubble = z2 * (1.0 - z) ** 2
        modes = torch.cat([
            endpoint_bubble,
            endpoint_bubble * (2.0 * z - 1.0),
            endpoint_bubble * (6.0 * z2 - 6.0 * z + 1.0),
        ], dim=1)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        residual = time_gate * torch.sum(self.correction_scales * modes * torch.tanh(raw), dim=1, keepdim=True)

        C_D = c_d_base + residual
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleGreenGrid(nn.Module):
    """Differentiable Green-history external reconstruction.

    The first Green prototype evaluated J_rxn(tau) separately for every
    external collocation point and every quadrature node.  That is accurate but
    creates an N_ext x K_history interface-state graph.  This version evaluates
    J_rxn on a global time grid once per training step, linearly interpolates it
    at the quadrature nodes, and then applies the same endpoint correction and
    endpoint-vanishing residual basis as the previous Green model.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=32,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__()
        self.gamma = gamma_val
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        self.history_grad = history_grad
        self.cache_history = cache_history
        self.time_grid_points = int(time_grid_points)
        self.kernel_points = int(kernel_points)
        if self.time_grid_points < 4:
            raise ValueError("time_grid_points must be at least 4")
        if self.kernel_points < 2:
            raise ValueError("kernel_points must be at least 2")

        # Keep names compatible with ExtBasis/Green so checkpoints can seed this
        # architecture with strict=False loading.
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)
        self.shape_net = MultiscaleResidualHead(out_features=2, width=192, depth=4)
        self.beta_net = MultiscaleResidualHead(out_features=1, width=128, depth=3, in_features=1)
        self.register_buffer(
            "kernel_nodes",
            (torch.arange(self.kernel_points, dtype=torch.float32) + 0.5) / self.kernel_points,
        )
        self.register_buffer(
            "time_grid",
            torch.linspace(0.0, float(T_sim), self.time_grid_points, dtype=torch.float32).reshape(-1, 1),
        )
        self.register_buffer(
            "correction_scales",
            torch.tensor([8.0, 6.0, 6.0], dtype=torch.float32).reshape(1, 3),
        )
        self._history_cache = None

    def clear_step_cache(self):
        self._history_cache = None
        if hasattr(self.interface_state, "clear_step_cache"):
            self.interface_state.clear_step_cache()

    def _history_grid(self, T_ref):
        use_cache = self.cache_history and self.training and torch.is_grad_enabled()
        cache_key = (
            T_ref.device,
            T_ref.dtype,
            torch.is_grad_enabled(),
            self.history_grad,
            parameter_condition_cache_key(self.interface_state),
        )
        if use_cache and self._history_cache is not None and self._history_cache[0] == cache_key:
            return self._history_cache[1]

        t_grid = self.time_grid.to(device=T_ref.device, dtype=T_ref.dtype)
        wants_grad = self.history_grad and torch.is_grad_enabled()
        if wants_grad:
            state = self.interface_state(t_grid)
            j_grid = state["J_rxn"]
        else:
            with torch.no_grad():
                state = self.interface_state(t_grid)
                j_grid = state["J_rxn"]
            if torch.is_grad_enabled():
                j_grid = j_grid.detach()

        if use_cache:
            self._history_cache = (cache_key, j_grid)
        return j_grid

    def _interp_history(self, tau, j_grid):
        tau = torch.clamp(tau, 0.0, float(T_sim))
        u = tau * (self.time_grid_points - 1) / float(T_sim)
        idx0 = torch.floor(u).clamp(0, self.time_grid_points - 2).long()
        idx1 = idx0 + 1
        alpha = (u - idx0.to(dtype=u.dtype)).clamp(0.0, 1.0)
        values = j_grid.reshape(-1)
        j0 = values[idx0]
        j1 = values[idx1]
        return j0 + alpha * (j1 - j0)

    def green_flux_convolution_fused(self, T_raw, y):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * nodes
        dt = torch.clamp(t_pos * (1.0 - nodes), min=1e-6 * T_sim)

        j_grid = self._history_grid(T_raw)
        q_tau = self._interp_history(tau, j_grid)

        ext_length = X_ext_max - delta
        y_triplet = torch.cat([
            torch.clamp(y, min=0.0),
            torch.zeros_like(y),
            torch.ones_like(y) * ext_length,
        ], dim=1)
        y_mat = y_triplet.unsqueeze(-1).expand(-1, -1, self.kernel_points)
        dt_mat = dt.unsqueeze(1)
        q_mat = q_tau.unsqueeze(1)

        kernel = torch.exp(-(y_mat ** 2) / (4.0 * D_rel_D * dt_mat))
        kernel = kernel / torch.sqrt(torch.clamp(np.pi * D_rel_D * dt_mat, min=1e-12))
        c_d = T_raw.unsqueeze(1) * torch.mean(q_mat * kernel, dim=2, keepdim=True)
        c_d_y = T_raw.unsqueeze(1) * torch.mean(
            q_mat * kernel * (-y_mat / (2.0 * D_rel_D * dt_mat)),
            dim=2,
            keepdim=True,
        )
        return c_d.squeeze(-1), c_d_y.squeeze(-1)

    def residual_raw(self, x_net):
        raw_main = self.net(x_net)
        raw_shape = self.shape_net(x_net)
        return torch.cat([raw_main, raw_shape], dim=1)

    def residual_modes(self, z):
        z2 = z * z
        endpoint_bubble = z2 * (1.0 - z) ** 2
        return torch.cat([
            endpoint_bubble,
            endpoint_bubble * (2.0 * z - 1.0),
            endpoint_bubble * (6.0 * z2 - 6.0 * z + 1.0),
        ], dim=1)

    def residual_correction(self, T_raw, z, x_net):
        raw = self.residual_raw(x_net)
        modes = self.residual_modes(z)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        scales = self.correction_scales.to(device=raw.device, dtype=raw.dtype)
        return time_gate * torch.sum(scales * modes * torch.tanh(raw), dim=1, keepdim=True)

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([
                t_net,
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            t_net = T_raw
            x_net = x_input

        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        j_rxn = state["J_rxn"]
        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)

        r = torch.clamp(y / ext_length, 0.0, 1.0)
        beta = 0.5 + 12.0 * torch.sigmoid(self.beta_net(t_net))
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = (1.0 - torch.exp(-beta * r)) / denom
        z = torch.clamp(z, 0.0, 1.0)
        dz_dx_at_interface = beta / (ext_length * denom)

        c_d_all, c_d_y_all = self.green_flux_convolution_fused(T_raw, y)
        c_d_green = c_d_all[:, 0:1]
        c_d_green_0 = c_d_all[:, 1:2]
        c_d_green_far = c_d_all[:, 2:3]
        c_d_green_y_0 = c_d_y_all[:, 1:2]

        z2 = z * z
        z3 = z2 * z
        h00 = 2.0 * z3 - 3.0 * z2 + 1.0
        h10 = z3 - 2.0 * z2 + z
        h01 = -2.0 * z3 + 3.0 * z2

        target_interface_slope = -j_rxn / D_rel_D
        target_far_value = torch.zeros_like(d_int)
        c_d_base = (
            c_d_green +
            h00 * (d_int - c_d_green_0) +
            h10 * (target_interface_slope - c_d_green_y_0) / dz_dx_at_interface +
            h01 * (target_far_value - c_d_green_far)
        )

        residual = self.residual_correction(T_raw, z, x_net)

        C_D = c_d_base + residual
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleGreenGridHybrid(ExternalNet_v9_6_MultiscaleGreenGrid):
    """Hybrid-lite Green/external-basis reconstruction for C_C/C_D.

    Baseline PINNs predict the whole external field directly.  Green-grid
    supplies a physics-shaped history-flux base field, then adds a small
    endpoint-vanishing residual.  This hybrid keeps the original Green-grid
    residual scales for warm-start stability and adds two stronger interior
    shape modes.  The residual still has zero value and zero slope at the
    interface/far boundary, so it targets the C_C/C_D spatial shape without
    relaxing the hard interface/terminal structure.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.hybrid_lite = True
        # Existing Green/Grid checkpoints still seed net, shape_net and beta_net.
        # Only these two extra interior modes start from scratch.
        self.shape_extra_net = MultiscaleResidualHead(out_features=2, width=160, depth=3)
        nn.init.zeros_(self.shape_extra_net.net[-1].weight)
        nn.init.zeros_(self.shape_extra_net.net[-1].bias)
        self.correction_scales = torch.tensor(
            [8.0, 6.0, 6.0, 10.0, 8.0],
            dtype=torch.float32,
        ).reshape(1, 5)

    def residual_raw(self, x_net):
        raw_main = self.net(x_net)
        raw_shape = self.shape_net(x_net)
        raw_extra = self.shape_extra_net(x_net)
        return torch.cat([raw_main, raw_shape, raw_extra], dim=1)

    def residual_modes(self, z):
        z2 = z * z
        z3 = z2 * z
        z4 = z2 * z2
        endpoint_bubble = z2 * (1.0 - z) ** 2
        shifted_p3 = 20.0 * z3 - 30.0 * z2 + 12.0 * z - 1.0
        shifted_p4 = 70.0 * z4 - 140.0 * z3 + 90.0 * z2 - 20.0 * z + 1.0
        return torch.cat([
            endpoint_bubble,
            endpoint_bubble * (2.0 * z - 1.0),
            endpoint_bubble * (6.0 * z2 - 6.0 * z + 1.0),
            endpoint_bubble * shifted_p3,
            endpoint_bubble * shifted_p4,
        ], dim=1)


class ExternalNet_v9_6_MultiscaleGreenGridDynamic(ExternalNet_v9_6_MultiscaleGreenGridHybrid):
    """Unified Green-grid model with dynamic physical inputs.

    The current failure mode is concentrated near the current-reversal dynamics:
    C_C_int and the interface slope can be accurate while the middle external
    field has a time-dependent amplitude bias.  This class keeps one unified
    model and the full signed Green history, then augments the residual
    correction with theta, dtheta/dt, J_rxn, dJ_rxn/dt, and Q_rxn.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.dynamic_lite = True
        self.dynamic_net = MultiscaleResidualHead(out_features=5, width=192, depth=3, in_features=7)
        nn.init.zeros_(self.dynamic_net.net[-1].weight)
        nn.init.zeros_(self.dynamic_net.net[-1].bias)
        self.register_buffer(
            "dynamic_correction_scales",
            torch.tensor([6.0, 4.0, 4.0, 6.0, 4.0], dtype=torch.float32).reshape(1, 5),
        )

    def _reaction_charge(self, T_raw):
        j_grid = self._history_grid(T_raw)
        dt = float(T_sim) / float(self.time_grid_points - 1)
        segments = 0.5 * (j_grid[:-1] + j_grid[1:]) * dt
        q_grid = torch.cat([torch.zeros_like(j_grid[:1]), torch.cumsum(segments, dim=0)], dim=0)
        return self._interp_history(T_raw, q_grid)

    def _reaction_rate_derivative(self, T_raw):
        eps = 1e-3 * float(T_sim)
        t_plus = torch.clamp(T_raw + eps, 0.0, float(T_sim))
        t_minus = torch.clamp(T_raw - eps, 0.0, float(T_sim))
        denom = torch.clamp(t_plus - t_minus, min=eps)
        j_plus = self.interface_state(t_plus)["J_rxn"]
        j_minus = self.interface_state(t_minus)["J_rxn"]
        return (j_plus - j_minus) / denom

    def dynamic_features(self, T_raw, X_raw, r, x_net, state):
        theta = potential_theta(T_raw)
        if getattr(self.interface_state, "continuous_time_features", False):
            theta_dot = potential_theta_dot_smooth(T_raw)
        else:
            theta_dot = potential_theta_dot(T_raw)
        j_rxn = state["J_rxn"]
        if "dJ_rxn_dt" in state:
            dj_dt = state["dJ_rxn_dt"]
        else:
            dj_dt = self._reaction_rate_derivative(T_raw)
        if "Q_rxn" in state:
            q_rxn = state["Q_rxn"]
        else:
            q_rxn = self._reaction_charge(T_raw)

        theta_scale = max(abs(float(theta_i)), abs(float(theta_switch)), 1.0)
        theta_dot_scale = theta_scale / max(float(T_sim), 1e-12)
        j_scale = max(float(k_cat_star * gamma), 1.0)
        q_scale = max(j_scale * float(T_sim), 1.0)
        x_mid = 2.0 * r - 1.0
        return torch.cat([
            x_net[:, 0:1],
            x_mid,
            theta / theta_scale,
            theta_dot / theta_dot_scale,
            j_rxn / j_scale,
            dj_dt / max(j_scale / max(float(T_sim), 1e-12), 1e-12),
            q_rxn / q_scale,
        ], dim=1)

    def dynamic_modes(self, r):
        r2 = r * r
        r3 = r2 * r
        r4 = r2 * r2
        window = r * (1.0 - r)
        shifted_p3 = 20.0 * r3 - 30.0 * r2 + 12.0 * r - 1.0
        shifted_p4 = 70.0 * r4 - 140.0 * r3 + 90.0 * r2 - 20.0 * r + 1.0
        return torch.cat([
            window,
            window * (2.0 * r - 1.0),
            window * (6.0 * r2 - 6.0 * r + 1.0),
            window * shifted_p3,
            window * shifted_p4,
        ], dim=1)

    def diffusion_causal_gate(self, T_raw, X_raw):
        if not getattr(self.interface_state, "causal_dynamic_correction", False):
            return torch.ones_like(T_raw)

        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)
        alpha = max(float(getattr(self.interface_state, "causal_gate_alpha", 2.0)), 1e-6)
        t_floor = max(float(getattr(self.interface_state, "causal_gate_time_floor", 0.005 * T_sim)), 1e-8)
        t_eff = torch.clamp(T_raw, min=t_floor)
        denom = max(4.0 * float(D_rel_D) * alpha, 1e-8) * t_eff
        exponent = -(y ** 2) / denom
        return torch.exp(torch.clamp(exponent, min=-60.0, max=0.0))

    def dynamic_correction(self, T_raw, X_raw, r, x_net, state):
        dynamic_input = self.dynamic_features(T_raw, X_raw, r, x_net, state)
        raw = self.dynamic_net(dynamic_input)
        modes = self.dynamic_modes(r)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        scales = self.dynamic_correction_scales.to(device=raw.device, dtype=raw.dtype)
        correction = time_gate * torch.sum(scales * modes * torch.tanh(raw), dim=1, keepdim=True)
        return correction * self.diffusion_causal_gate(T_raw, X_raw)

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([
                t_net,
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            t_net = T_raw
            x_net = x_input

        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        j_rxn = state["J_rxn"]
        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)

        r = torch.clamp(y / ext_length, 0.0, 1.0)
        beta = 0.5 + 12.0 * torch.sigmoid(self.beta_net(t_net))
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = (1.0 - torch.exp(-beta * r)) / denom
        z = torch.clamp(z, 0.0, 1.0)
        dz_dx_at_interface = beta / (ext_length * denom)

        c_d_all, c_d_y_all = self.green_flux_convolution_fused(T_raw, y)
        c_d_green = c_d_all[:, 0:1]
        c_d_green_0 = c_d_all[:, 1:2]
        c_d_green_far = c_d_all[:, 2:3]
        c_d_green_y_0 = c_d_y_all[:, 1:2]

        z2 = z * z
        z3 = z2 * z
        h00 = 2.0 * z3 - 3.0 * z2 + 1.0
        h10 = z3 - 2.0 * z2 + z
        h01 = -2.0 * z3 + 3.0 * z2

        target_interface_slope = -j_rxn / D_rel_D
        target_far_value = torch.zeros_like(d_int)
        c_d_base = (
            c_d_green +
            h00 * (d_int - c_d_green_0) +
            h10 * (target_interface_slope - c_d_green_y_0) / dz_dx_at_interface +
            h01 * (target_far_value - c_d_green_far)
        )

        residual = self.residual_correction(T_raw, z, x_net)
        residual = residual + self.dynamic_correction(T_raw, X_raw, r, x_net, state)

        C_D = c_d_base + residual
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleGreenGridDynamicStage1(ExternalNet_v9_6_MultiscaleGreenGridDynamic):
    """Fixed clean warm-start external model for the two-stage workflow.

    Stage 1 should prepare a reproducible initial state for the final
    Film-TraceGreen model, not solve the paper-facing C_C/C_D field by adding
    many experiment-specific corrections.  This class keeps the useful signed
    full-field Green history and dynamic physical inputs, but removes the
    hybrid-only extra spatial modes and uses a smaller three-mode dynamic
    correction.  Stage 2 then takes over with the Film-Abel interface chain and
    the erfc trace-preserving TraceGreen lift.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.stage1_clean_warmstart = True
        self.hybrid_lite = False

        # The parent dynamic model was introduced after Hybrid-lite and still
        # constructs two extra spatial modes.  Keep checkpoint compatibility but
        # remove those modes from the stage1 forward path.
        if hasattr(self, "shape_extra_net"):
            for param in self.shape_extra_net.parameters():
                param.requires_grad_(False)

        self.dynamic_net = MultiscaleResidualHead(out_features=3, width=160, depth=3, in_features=7)
        nn.init.zeros_(self.dynamic_net.net[-1].weight)
        nn.init.zeros_(self.dynamic_net.net[-1].bias)
        concentration_factor = float(gamma_val) / REFERENCE_GAMMA
        self.correction_scales = concentration_factor * torch.tensor(
            [8.0, 6.0, 6.0], dtype=torch.float32
        ).reshape(1, 3)
        self.dynamic_correction_scales = concentration_factor * torch.tensor(
            [3.0, 2.0, 2.0], dtype=torch.float32
        ).reshape(1, 3)

    def residual_raw(self, x_net):
        return ExternalNet_v9_6_MultiscaleGreenGrid.residual_raw(self, x_net)

    def residual_modes(self, z):
        return ExternalNet_v9_6_MultiscaleGreenGrid.residual_modes(self, z)

    def dynamic_modes(self, r):
        r2 = r * r
        window = r * (1.0 - r)
        return torch.cat([
            window,
            window * (2.0 * r - 1.0),
            window * (6.0 * r2 - 6.0 * r + 1.0),
        ], dim=1)


class ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalConv(ExternalNet_v9_6_MultiscaleGreenGridDynamic):
    """Dynamic correction as a causal heat-kernel convolution.

    The direct dynamic residual can use the current scan state to paint a whole
    external concentration field.  This ablation instead learns a scalar
    correction source S_corr(t) and propagates it with the same Green kernel used
    for physical flux history.  Endpoint subtraction keeps the correction from
    changing the prescribed interface and far-field values directly.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.causal_convolution_dynamic = True
        self.dynamic_source_net = MultiscaleResidualHead(out_features=1, width=160, depth=3, in_features=7)
        nn.init.zeros_(self.dynamic_source_net.net[-1].weight)
        nn.init.zeros_(self.dynamic_source_net.net[-1].bias)
        self._dynamic_source_cache = None

    def clear_step_cache(self):
        super().clear_step_cache()
        self._dynamic_source_cache = None

    def _source_features(self, T_raw):
        X_src = torch.ones_like(T_raw) * delta
        r_src = torch.zeros_like(T_raw)
        if self.normalize_inputs:
            x_net = torch.cat([normalize_time(T_raw), normalize_ext_x(X_src)], dim=1)
        else:
            x_net = torch.cat([T_raw, X_src], dim=1)
        state = self.interface_state(T_raw)
        return self.dynamic_features(T_raw, X_src, r_src, x_net, state)

    def _dynamic_source(self, T_raw):
        raw = self.dynamic_source_net(self._source_features(T_raw))
        scale = float(getattr(self.interface_state, "causal_source_scale", 0.75))
        return scale * torch.tanh(raw)

    def _dynamic_source_grid(self, T_ref):
        use_cache = self.cache_history and self.training and torch.is_grad_enabled()
        cache_key = (
            T_ref.device,
            T_ref.dtype,
            torch.is_grad_enabled(),
            parameter_condition_cache_key(self.interface_state),
        )
        if use_cache and self._dynamic_source_cache is not None and self._dynamic_source_cache[0] == cache_key:
            return self._dynamic_source_cache[1]

        t_grid = self.time_grid.to(device=T_ref.device, dtype=T_ref.dtype)
        source_grid = self._dynamic_source(t_grid)
        if use_cache:
            self._dynamic_source_cache = (cache_key, source_grid)
        return source_grid

    def _causal_source_convolution(self, T_raw, X_raw, z):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * nodes
        dt = torch.clamp(t_pos * (1.0 - nodes), min=1e-6 * T_sim)

        source_grid = self._dynamic_source_grid(T_raw)
        source_tau = self._interp_history(tau, source_grid)

        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)
        y_triplet = torch.cat([
            y,
            torch.zeros_like(y),
            torch.ones_like(y) * ext_length,
        ], dim=1)
        y_mat = y_triplet.unsqueeze(-1).expand(-1, -1, self.kernel_points)
        dt_mat = dt.unsqueeze(1)
        source_mat = source_tau.unsqueeze(1)

        kernel = torch.exp(-(y_mat ** 2) / (4.0 * D_rel_D * dt_mat))
        kernel = kernel / torch.sqrt(torch.clamp(np.pi * D_rel_D * dt_mat, min=1e-12))
        conv_all = T_raw.unsqueeze(1) * torch.mean(source_mat * kernel, dim=2, keepdim=True)
        conv_all = conv_all.squeeze(-1)

        conv_y = conv_all[:, 0:1]
        conv_0 = conv_all[:, 1:2]
        conv_far = conv_all[:, 2:3]

        z2 = z * z
        z3 = z2 * z
        h00 = 2.0 * z3 - 3.0 * z2 + 1.0
        h01 = -2.0 * z3 + 3.0 * z2
        return conv_y - h00 * conv_0 - h01 * conv_far

    def dynamic_correction(self, T_raw, X_raw, r, x_net, state):
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
        else:
            t_net = T_raw
        beta = 0.5 + 12.0 * torch.sigmoid(self.beta_net(t_net))
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = torch.clamp((1.0 - torch.exp(-beta * r)) / denom, 0.0, 1.0)
        return self._causal_source_convolution(T_raw, X_raw, z)


class ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybrid(ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalConv):
    """Causal-convolution correction plus a small gated direct correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_hybrid_dynamic = True

    def direct_dynamic_correction(self, T_raw, X_raw, r, x_net, state):
        direct = ExternalNet_v9_6_MultiscaleGreenGridDynamic.dynamic_correction(
            self,
            T_raw,
            X_raw,
            r,
            x_net,
            state,
        )
        direct_scale = float(getattr(self.interface_state, "causal_direct_scale", 0.25))
        return direct_scale * direct

    def dynamic_correction(self, T_raw, X_raw, r, x_net, state):
        causal_conv = super().dynamic_correction(T_raw, X_raw, r, x_net, state)
        return causal_conv + self.direct_dynamic_correction(T_raw, X_raw, r, x_net, state)


class ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybridSmooth(ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybrid):
    """Causal-hybrid external field with smooth direct residual regularization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_hybrid_smooth_dynamic = True


class ExternalNet_v9_6_MultiscaleGreenGridFluxTrace(ExternalNet_v9_6_MultiscaleGreenGridHybrid):
    """Flux-trace-aware Green reconstruction for the external C/D field.

    The standard Green-grid branch evaluates the Green kernel derivative at
    y=0 with autograd, which gives zero for each quadrature node.  It then uses a
    Hermite h10 slope correction to impose the boundary flux.  Decomposition
    showed that this polynomial slope correction pollutes the near field during
    the forward scan.

    This branch treats the boundary flux as the analytic trace of the flux Green
    solution: the Green term carries J_rxn history and its boundary flux is not
    re-evaluated by autograd at y=0.  Therefore the explicit h10 slope
    correction is removed.  The remaining corrections preserve the analytic flux
    trace because they have zero value/slope at the interface except for the
    value-matching h00 term, whose derivative is zero at z=0.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.fluxtrace_green = True
        self.fluxtrace_analytic_boundary_flux = True
        # Start with a broad value-correction trace.  A sharp h00 correction
        # recreates the same near-interface PDE stiffness that FluxTrace is
        # trying to remove from the old h10 slope patch.
        self.fluxtrace_value_beta_raw = nn.Parameter(torch.tensor([-3.0], dtype=torch.float32))
        self.fluxtrace_trace_weight = 25.0
        self.register_buffer(
            "fluxtrace_residual_scales",
            torch.tensor([2.0, 1.25, 1.25, 1.5, 1.25], dtype=torch.float32).reshape(1, 5),
        )

    def fluxtrace_value_beta(self):
        return 2.0 + 8.0 * torch.sigmoid(self.fluxtrace_value_beta_raw)

    def fluxtrace_coordinate(self, T_raw, X_raw):
        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)
        r = torch.clamp(y / ext_length, 0.0, 1.0)
        beta = self.fluxtrace_value_beta().to(device=T_raw.device, dtype=T_raw.dtype)
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = torch.clamp((1.0 - torch.exp(-beta * r)) / denom, 0.0, 1.0)
        return y, r, z, beta

    def fluxtrace_residual_correction(self, T_raw, z, x_net):
        raw = self.residual_raw(x_net)
        modes = self.residual_modes(z)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        scales = self.fluxtrace_residual_scales.to(device=raw.device, dtype=raw.dtype)
        return time_gate * torch.sum(scales * modes * torch.tanh(raw), dim=1, keepdim=True)

    def fluxtrace_trace_consistency_loss(self, T_raw):
        """Neumann-to-Dirichlet consistency for the external diffusion trace.

        For a pure external diffusion region, the boundary flux history already
        determines the boundary concentration trace through the Abel/Green
        operator.  If the interface-state network proposes an independent
        C_D_int far away from G_flux[J](0,t), the h00 value correction becomes a
        hidden boundary source and the external PDE loss cannot settle.  This
        loss keeps the interface memory and the flux Green trace compatible
        without using FDM data.
        """
        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        c_d_all, _ = self.green_flux_convolution_fused(T_raw, torch.zeros_like(T_raw))
        c_d_green_0 = c_d_all[:, 0:1]
        return torch.mean((d_int - c_d_green_0) ** 2)

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([t_net, normalize_ext_x(X_raw)], dim=1)
        else:
            x_net = x_input

        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        y, _, z, _ = self.fluxtrace_coordinate(T_raw, X_raw)

        c_d_all, _ = self.green_flux_convolution_fused(T_raw, y)
        c_d_green = c_d_all[:, 0:1]
        c_d_green_0 = c_d_all[:, 1:2]
        c_d_green_far = c_d_all[:, 2:3]

        z2 = z * z
        z3 = z2 * z
        h00 = 2.0 * z3 - 3.0 * z2 + 1.0
        h01 = -2.0 * z3 + 3.0 * z2

        c_d_base = (
            c_d_green +
            h00 * (d_int - c_d_green_0) +
            h01 * (torch.zeros_like(d_int) - c_d_green_far)
        )
        residual = self.fluxtrace_residual_correction(T_raw, z, x_net)

        C_D = c_d_base + residual
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(ExternalNet_v9_6_MultiscaleGreenGridHybrid):
    """Film-Abel trace-driven external reconstruction.

    FluxTrace showed that using two independent traces,

        C_D_int^film(t) and G_flux[J](0,t),

    then sewing them together with an h00 correction makes the external PDE
    stiff.  This ablation keeps the successful Film-Abel/interface-memory trace
    and propagates it into the external region with the heat-equation Dirichlet
    boundary potential.  The external field therefore has one boundary source:

        C_D(0,t) = C_D_int^film(t).

    A far-boundary Hermite value correction and endpoint-vanishing residual are
    still allowed, but they cannot alter the interface value.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.tracegreen_external = True
        self.external_analytic_boundary_flux = True
        self.tracegreen_far_beta_raw = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32))
        self.register_buffer(
            "tracegreen_residual_scales",
            torch.tensor([2.0, 1.25, 1.25, 1.5, 1.25], dtype=torch.float32).reshape(1, 5),
        )

    def tracegreen_far_beta(self):
        return 1.0 + 7.0 * torch.sigmoid(self.tracegreen_far_beta_raw)

    def tracegreen_coordinate(self, T_raw, X_raw):
        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)
        r = torch.clamp(y / ext_length, 0.0, 1.0)
        beta = self.tracegreen_far_beta().to(device=T_raw.device, dtype=T_raw.dtype)
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = torch.clamp((1.0 - torch.exp(-beta * r)) / denom, 0.0, 1.0)
        return y, r, z, beta

    def _trace_history_grid(self, T_ref):
        wants_grad = self.history_grad and torch.is_grad_enabled()
        if hasattr(self.interface_state, "_history_grid"):
            if wants_grad:
                history = self.interface_state._history_grid(T_ref)
                c_d_grid = history.get("C_D_int")
                if c_d_grid is None:
                    c_d_grid = self.gamma - history["C_C_int"]
            else:
                with torch.no_grad():
                    history = self.interface_state._history_grid(T_ref)
                    c_d_grid = history.get("C_D_int")
                    if c_d_grid is None:
                        c_d_grid = self.gamma - history["C_C_int"]
                if torch.is_grad_enabled():
                    c_d_grid = c_d_grid.detach()
            return c_d_grid

        t_grid = self.time_grid.to(device=T_ref.device, dtype=T_ref.dtype)
        if wants_grad:
            state = self.interface_state(t_grid)
            return self.gamma - state["C_C_int"]
        with torch.no_grad():
            state = self.interface_state(t_grid)
            c_d_grid = self.gamma - state["C_C_int"]
        return c_d_grid.detach() if torch.is_grad_enabled() else c_d_grid

    def trace_boundary_convolution_fused(self, T_raw, y):
        """Dirichlet heat-potential propagation of Film-Abel C_D_int(t).

        The half-line Dirichlet Poisson kernel is an approximate identity:
        C_D(y,t) -> C_D_int(t) as y -> 0+.  Uniform quadrature in tau misses
        that limit because the kernel mass collapses into a very narrow window
        near tau=t.  Use u=erfc(y / (2*sqrt(D*(t-tau)))) instead; uniform
        quadrature in u preserves the boundary trace without an artificial
        jump between y=0 and the first exterior grid point.
        """
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, 1, -1).to(device=T_raw.device, dtype=T_raw.dtype)

        c_d_grid = self._trace_history_grid(T_raw)
        current_state = self.interface_state(T_raw)
        c_d_int = self.gamma - current_state["C_C_int"]

        ext_length = X_ext_max - delta
        y_triplet = torch.cat([
            torch.clamp(y, min=0.0),
            torch.zeros_like(y),
            torch.ones_like(y) * ext_length,
        ], dim=1)

        diffusion = torch.as_tensor(D_rel_D, device=T_raw.device, dtype=T_raw.dtype)
        z0 = y_triplet / torch.sqrt(torch.clamp(4.0 * diffusion * t_pos, min=1e-12))
        trace_mass = torch.erfc(z0)

        u = trace_mass.unsqueeze(-1) * nodes
        u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
        eta = torch.erfinv(1.0 - u)
        eta2 = torch.clamp(eta ** 2, min=1e-8)
        delay = y_triplet.unsqueeze(-1) ** 2 / (4.0 * diffusion * eta2)
        tau = torch.clamp(T_raw.unsqueeze(1) - delay, 0.0, float(T_sim))

        c_d_tau = self._interp_history(tau, c_d_grid)
        c_d = trace_mass * torch.mean(c_d_tau, dim=2)

        # The continuous Dirichlet kernel has trace C_D_int at y=0, but the
        # interpolated history can differ slightly from the current interface
        # state.  Restore the exact analytic trace for interface evaluation and
        # hard endpoint matching.
        c_d_int_triplet = c_d_int.expand(-1, y_triplet.shape[1])
        c_d = torch.where(y_triplet <= 1e-10, c_d_int_triplet, c_d)
        return c_d

    def tracegreen_residual_correction(self, T_raw, z, x_net):
        raw = self.residual_raw(x_net)
        modes = self.residual_modes(z)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        scales = self.tracegreen_residual_scales.to(device=raw.device, dtype=raw.dtype)
        return time_gate * torch.sum(scales * modes * torch.tanh(raw), dim=1, keepdim=True)

    def _x_net_and_z(self, T_raw, X_raw):
        if self.normalize_inputs:
            x_net = torch.cat([normalize_time(T_raw), normalize_ext_x(X_raw)], dim=1)
        else:
            x_net = torch.cat([T_raw, X_raw], dim=1)
        y, _, z, _ = self.tracegreen_coordinate(T_raw, X_raw)
        return x_net, y, z

    def tracegreen_lift_and_residual(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        x_net, y, z = self._x_net_and_z(T_raw, X_raw)
        c_d_all = self.trace_boundary_convolution_fused(T_raw, y)
        c_d_trace = c_d_all[:, 0:1]
        c_d_far = c_d_all[:, 2:3]

        z2 = z * z
        z3 = z2 * z
        h01 = -2.0 * z3 + 3.0 * z2

        c_d_base = c_d_trace + h01 * (torch.zeros_like(c_d_far) - c_d_far)
        residual = self.tracegreen_residual_correction(T_raw, z, x_net)
        return c_d_base, residual

    def pde_fields(self, x_input):
        """Return only the smooth correction for PDE residual training.

        The HeatDirichlet lift contains the endpoint-singular Green kernel and
        is treated as an analytic solution component.  Autograd PDE residuals
        should therefore act on the smooth neural correction, not on the lift.
        """
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        x_net, _, z = self._x_net_and_z(T_raw, X_raw)
        residual = self.tracegreen_residual_correction(T_raw, z, x_net)
        return -residual, residual

    def forward(self, x_input):
        c_d_base, residual = self.tracegreen_lift_and_residual(x_input)
        C_D = c_d_base + residual
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_FilmTraceGreenClean(ExternalNet_v9_6_MultiscaleGreenGridFilmTrace):
    """Clean final external field for the Film-TraceGreen architecture.

    The external concentration is reconstructed from a single causal boundary
    trace rather than predicted directly:

        C_D(y,t) = G_trace^erfc[C_D_int](y,t) + h01(y)*(0-D_far)
                   + s_train(e)*R_smooth(y,t)
        C_C(y,t) = gamma - C_D(y,t).

    The erfc variable transform in ``trace_boundary_convolution_fused`` is the
    structural part of the method: it preserves the y -> 0+ trace and removes
    the near-interface discontinuity seen with ordinary time quadrature.  The
    contribution audit showed that the learned external residual R_smooth is a
    post-training negative contributor for the best checkpoint.  Therefore it is
    treated as a training scaffold: useful early when C_D_int is still inaccurate,
    but decayed to zero for the final clean model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film_trace_clean_external = True

        # Keep only the trace-driven Green lift plus its smooth residual.  These
        # flags silence exploratory dynamic/hybrid logging for the clean model.
        self.hybrid_lite = False
        self.dynamic_green_residual = False
        self.causal_hybrid_dynamic = False
        self.causal_hybrid_smooth_dynamic = False
        self.register_buffer("clean_residual_scale", torch.tensor(0.0, dtype=torch.float32))

    def set_clean_residual_scale(self, scale):
        scale = float(max(0.0, scale))
        self.clean_residual_scale.fill_(scale)

    def tracegreen_residual_correction(self, T_raw, z, x_net):
        scale = self.clean_residual_scale.to(device=T_raw.device, dtype=T_raw.dtype)
        concentration_factor = float(self.gamma) / REFERENCE_GAMMA
        return concentration_factor * scale * super().tracegreen_residual_correction(T_raw, z, x_net)


class ExternalNet_v9_6_FilmTraceGreenKParam(ExternalNet_v9_6_FilmTraceGreenClean):
    """TraceGreen external field driven by a shared runtime k condition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_parameterized = True
        self.k_reference = float(self.interface_state.k_reference)

    def active_k_cat(self):
        return self.interface_state.active_k_cat()

    def set_k_cat(self, value):
        self.interface_state.set_k_cat(value)
        self.clear_step_cache()

    def pde_fields(self, x_input):
        base_input = activate_k_from_input(self.interface_state, x_input)
        return super().pde_fields(base_input)

    def forward(self, x_input):
        base_input = activate_k_from_input(self.interface_state, x_input)
        return super().forward(base_input)


class ExternalNet_v9_6_FilmTraceGreenGammaParam(
    ExternalNet_v9_6_FilmTraceGreenClean
):
    """TraceGreen external field driven by a shared runtime gamma condition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_parameterized = True
        self.gamma_reference = float(self.interface_state.gamma_reference)

    def active_gamma(self):
        return self.interface_state.active_gamma()

    def set_gamma(self, value):
        self.interface_state.set_gamma(value)
        self.gamma = self.interface_state.active_gamma()
        self.clear_step_cache()

    def tracegreen_residual_correction(self, T_raw, z, x_net):
        scale = self.clean_residual_scale.to(device=T_raw.device, dtype=T_raw.dtype)
        concentration_factor = self.interface_state._active_gamma_tensor(T_raw) / REFERENCE_GAMMA
        return concentration_factor * scale * ExternalNet_v9_6_MultiscaleGreenGridFilmTrace.tracegreen_residual_correction(
            self, T_raw, z, x_net
        )

    def pde_fields(self, x_input):
        base_input = activate_gamma_from_input(self.interface_state, x_input)
        return super().pde_fields(base_input)

    def forward(self, x_input):
        base_input = activate_gamma_from_input(self.interface_state, x_input)
        c_d_base, residual = self.tracegreen_lift_and_residual(base_input)
        C_D = c_d_base + residual
        gamma_t = self.interface_state._active_gamma_tensor(C_D)
        C_C = gamma_t - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleGreenGridMemory(ExternalNet_v9_6_MultiscaleGreenGridDynamic):
    """Dynamic Green-grid model with causal memory and coordinate-safe basis.

    This is the next ablation after ``multiscale_green_grid_dynamic``.  It keeps
    the unified model and signed full-field Green convolution, then adds three
    structure-level corrections aimed at the reverse-scan lag:

    1. causal exponential memories of J_rxn with several time constants;
    2. uniformly distributed, boundary-compatible local basis functions in the
       normalized external coordinate r=(x-delta)/L, not hand-placed at an FDM
       residual hot spot;
    3. small learnable multi-scale Green-kernel corrections around the nominal
       diffusion coefficient.

    New heads/Green amplitudes are zero-initialized so a dynamic checkpoint can
    seed this architecture without immediately changing its output.
    """

    def __init__(
        self,
        gamma_val,
        interface_state=None,
        normalize_inputs=True,
        time_grid_points=256,
        kernel_points=64,
        history_grad=True,
        cache_history=True,
    ):
        super().__init__(
            gamma_val,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=time_grid_points,
            kernel_points=kernel_points,
            history_grad=history_grad,
            cache_history=cache_history,
        )
        self.memory_multibasis = True
        self.register_buffer(
            "memory_lambdas",
            torch.tensor([0.02, 0.05, 0.10, 0.20], dtype=torch.float32).reshape(1, -1),
        )
        self.register_buffer(
            "local_basis_centers",
            torch.linspace(0.1, 0.9, 9, dtype=torch.float32).reshape(1, -1),
        )
        self.register_buffer(
            "local_basis_sigma",
            torch.tensor(0.11, dtype=torch.float32),
        )
        self.memory_mode_count = 12
        self.memory_net = MultiscaleResidualHead(
            out_features=self.memory_mode_count,
            width=224,
            depth=3,
            in_features=11,
        )
        nn.init.zeros_(self.memory_net.net[-1].weight)
        nn.init.zeros_(self.memory_net.net[-1].bias)
        self.register_buffer(
            "memory_mode_scales",
            torch.tensor([0.65, 0.65, 0.65] + [0.90] * 9, dtype=torch.float32).reshape(1, -1),
        )
        self.register_buffer(
            "green_extra_diffusion_scales",
            torch.tensor([0.5, 2.0], dtype=torch.float32).reshape(1, -1),
        )
        self.green_extra_alpha = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.green_extra_gain = 0.35

    def _interp_history_multi(self, tau, value_grid):
        tau = torch.clamp(tau, 0.0, float(T_sim))
        u = tau * (self.time_grid_points - 1) / float(T_sim)
        idx0 = torch.floor(u).clamp(0, self.time_grid_points - 2).long().reshape(-1)
        idx1 = idx0 + 1
        alpha = (u.reshape(-1, 1) - idx0.reshape(-1, 1).to(dtype=u.dtype)).clamp(0.0, 1.0)
        values = value_grid.reshape(self.time_grid_points, -1)
        v0 = values[idx0]
        v1 = values[idx1]
        return v0 + alpha * (v1 - v0)

    def _reaction_memory_grid(self, T_ref):
        j_grid = self._history_grid(T_ref).reshape(self.time_grid_points, 1)
        lambdas = self.memory_lambdas.to(device=T_ref.device, dtype=T_ref.dtype)
        dt = float(T_sim) / float(self.time_grid_points - 1)
        decay = torch.exp(-dt / torch.clamp(lambdas, min=1e-6))
        memory = torch.zeros(1, lambdas.shape[1], device=T_ref.device, dtype=T_ref.dtype)
        rows = [memory]
        for idx in range(1, self.time_grid_points):
            j_prev = j_grid[idx - 1:idx]
            j_cur = j_grid[idx:idx + 1]
            memory = decay * memory + 0.5 * dt * (decay * j_prev + j_cur)
            rows.append(memory)
        return torch.cat(rows, dim=0)

    def _reaction_memories(self, T_raw):
        memory_grid = self._reaction_memory_grid(T_raw)
        return self._interp_history_multi(T_raw, memory_grid)

    def memory_features(self, T_raw, X_raw, r, x_net, state):
        base_features = self.dynamic_features(T_raw, X_raw, r, x_net, state)
        memories = self._reaction_memories(T_raw)
        lambdas = self.memory_lambdas.to(device=T_raw.device, dtype=T_raw.dtype)
        j_scale = max(float(k_cat_star * gamma), 1.0)
        memory_scale = torch.clamp(j_scale * lambdas, min=1e-6)
        return torch.cat([base_features, memories / memory_scale], dim=1)

    def memory_modes(self, r):
        r2 = r * r
        bubble = r2 * (1.0 - r) ** 2
        bubble_norm = bubble / 0.0625
        global_modes = torch.cat([
            bubble_norm,
            bubble_norm * (2.0 * r - 1.0),
            bubble_norm * (6.0 * r2 - 6.0 * r + 1.0),
        ], dim=1)

        centers = self.local_basis_centers.to(device=r.device, dtype=r.dtype)
        sigma = torch.clamp(self.local_basis_sigma.to(device=r.device, dtype=r.dtype), min=1e-4)
        center_bubble = torch.clamp(centers ** 2 * (1.0 - centers) ** 2, min=1e-4)
        gaussians = torch.exp(-0.5 * ((r - centers) / sigma) ** 2)
        local_modes = bubble * gaussians / center_bubble
        return torch.cat([global_modes, local_modes], dim=1)

    def memory_correction(self, T_raw, X_raw, r, x_net, state):
        memory_input = self.memory_features(T_raw, X_raw, r, x_net, state)
        raw = self.memory_net(memory_input)
        modes = self.memory_modes(r)
        scales = self.memory_mode_scales.to(device=raw.device, dtype=raw.dtype)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        return time_gate * torch.sum(scales * modes * torch.tanh(raw), dim=1, keepdim=True)

    def dynamic_correction(self, T_raw, X_raw, r, x_net, state):
        return (
            super().dynamic_correction(T_raw, X_raw, r, x_net, state) +
            self.memory_correction(T_raw, X_raw, r, x_net, state)
        )

    def _green_flux_convolution_fused_with_diffusion(self, T_raw, y, diffusion_coeff):
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * nodes
        dt = torch.clamp(t_pos * (1.0 - nodes), min=1e-6 * T_sim)

        j_grid = self._history_grid(T_raw)
        q_tau = self._interp_history(tau, j_grid)

        ext_length = X_ext_max - delta
        y_triplet = torch.cat([
            torch.clamp(y, min=0.0),
            torch.zeros_like(y),
            torch.ones_like(y) * ext_length,
        ], dim=1)
        y_mat = y_triplet.unsqueeze(-1).expand(-1, -1, self.kernel_points)
        dt_mat = dt.unsqueeze(1)
        q_mat = q_tau.unsqueeze(1)
        diffusion = torch.as_tensor(diffusion_coeff, device=T_raw.device, dtype=T_raw.dtype)

        kernel = torch.exp(-(y_mat ** 2) / (4.0 * diffusion * dt_mat))
        kernel = kernel / torch.sqrt(torch.clamp(np.pi * diffusion * dt_mat, min=1e-12))
        c_d = T_raw.unsqueeze(1) * torch.mean(q_mat * kernel, dim=2, keepdim=True)
        c_d_y = T_raw.unsqueeze(1) * torch.mean(
            q_mat * kernel * (-y_mat / (2.0 * diffusion * dt_mat)),
            dim=2,
            keepdim=True,
        )
        return c_d.squeeze(-1), c_d_y.squeeze(-1)

    def green_flux_convolution_fused(self, T_raw, y):
        c_d, c_d_y = super().green_flux_convolution_fused(T_raw, y)
        scales = self.green_extra_diffusion_scales.to(device=T_raw.device, dtype=T_raw.dtype).reshape(-1)
        weights = self.green_extra_gain * torch.tanh(
            self.green_extra_alpha.to(device=T_raw.device, dtype=T_raw.dtype)
        )
        for scale, weight in zip(scales, weights):
            extra_c_d, extra_c_d_y = self._green_flux_convolution_fused_with_diffusion(
                T_raw,
                y,
                D_rel_D * scale,
            )
            c_d = c_d + weight * extra_c_d
            c_d_y = c_d_y + weight * extra_c_d_y
        return c_d, c_d_y


class ExternalNet_v9_6_MultiscaleBuffer(nn.Module):
    """Free external field plus an analytic 1D buffer correction.

    The free network is not endpoint-windowed.  At each forward pass the buffer
    removes its value/slope mismatch at the interface and far field, leaving the
    interior shape much less constrained than a global Hermite ansatz.
    """

    def __init__(self, gamma_val, interface_state=None, normalize_inputs=True):
        super().__init__()
        self.gamma = gamma_val
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        # Match ExtBasis/Green module names so their checkpoints can seed this
        # unrestricted free field with strict=False loading.
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)
        self.shape_net = MultiscaleResidualHead(out_features=2, width=192, depth=4)
        self.beta_net = MultiscaleResidualHead(out_features=1, width=128, depth=3, in_features=1)
        self.register_buffer(
            "free_scales",
            torch.tensor([5.0, 4.0, 4.0], dtype=torch.float32).reshape(1, 3),
        )

    def _inputs_and_warp(self, T_raw, X_raw):
        ext_length = X_ext_max - delta
        r = (X_raw - delta) / ext_length
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([
                t_net,
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            t_net = T_raw
            x_net = torch.cat([T_raw, X_raw], dim=1)

        beta = 0.5 + 12.0 * torch.sigmoid(self.beta_net(t_net))
        exp_neg_beta = torch.exp(-beta)
        denom = torch.clamp(1.0 - exp_neg_beta, min=1e-5)
        z = (1.0 - torch.exp(-beta * r)) / denom
        return x_net, z

    def _free_cd(self, T_raw, X_raw):
        x_net, z = self._inputs_and_warp(T_raw, X_raw)
        raw = torch.cat([self.net(x_net), self.shape_net(x_net)], dim=1)
        z2 = z * z
        modes = torch.cat([
            torch.ones_like(z),
            2.0 * z - 1.0,
            6.0 * z2 - 6.0 * z + 1.0,
        ], dim=1)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.04 * T_sim))
        return time_gate * torch.sum(self.free_scales * modes * torch.tanh(raw), dim=1, keepdim=True)

    def _boundary_free_terms(self, T_raw, x_value, create_graph):
        X_raw = torch.full_like(T_raw, float(x_value))
        X_raw.requires_grad_(True)
        C_D_free = self._free_cd(T_raw, X_raw)
        C_D_free_X = torch.autograd.grad(
            C_D_free.sum(),
            X_raw,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return C_D_free, C_D_free_X

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        ext_length = X_ext_max - delta
        r = (X_raw - delta) / ext_length

        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        j_rxn = state["J_rxn"]
        target_interface_slope = -j_rxn / D_rel_D
        target_far_value = torch.zeros_like(d_int)
        target_far_slope = torch.zeros_like(d_int)

        C_D_free = self._free_cd(T_raw, X_raw)

        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            T_boundary = T_raw if outer_grad_enabled else T_raw.detach()
            d_free_0, d_free_x_0 = self._boundary_free_terms(
                T_boundary,
                delta,
                create_graph=outer_grad_enabled,
            )
            d_free_1, d_free_x_1 = self._boundary_free_terms(
                T_boundary,
                X_ext_max,
                create_graph=outer_grad_enabled,
            )
        if not outer_grad_enabled:
            d_free_0 = d_free_0.detach()
            d_free_x_0 = d_free_x_0.detach()
            d_free_1 = d_free_1.detach()
            d_free_x_1 = d_free_x_1.detach()

        r2 = r * r
        r3 = r2 * r
        h00 = 2.0 * r3 - 3.0 * r2 + 1.0
        h10 = r3 - 2.0 * r2 + r
        h01 = -2.0 * r3 + 3.0 * r2
        h11 = r3 - r2

        value_error_0 = d_int - d_free_0
        slope_error_0 = target_interface_slope - d_free_x_0
        value_error_1 = target_far_value - d_free_1
        slope_error_1 = target_far_slope - d_free_x_1
        buffer = (
            h00 * value_error_0 +
            h10 * ext_length * slope_error_0 +
            h01 * value_error_1 +
            h11 * ext_length * slope_error_1
        )

        C_D = C_D_free + buffer
        C_C = self.gamma - C_D
        return C_C, C_D


class ExternalNet_v9_6_MultiscaleFluxBuffer(ExternalNet_v9_6_MultiscaleBuffer):
    """Buffer concentration field with an explicit conservative flux state.

    C_D is still projected onto the interface/farfield constraints by the
    analytic buffer.  The extra flux head gives the external PDE a mixed
    conservation-form target, which avoids relying only on second derivatives
    of the concentration in the stiff diffusion layer.
    """

    def __init__(self, gamma_val, interface_state=None, normalize_inputs=True):
        super().__init__(gamma_val, interface_state=interface_state, normalize_inputs=normalize_inputs)
        self.flux_net = MultiscaleResidualHead(out_features=2, width=192, depth=4)
        self.flux_residual_scale = 4.0

    def flux_d(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        x_net, z = self._inputs_and_warp(T_raw, X_raw)
        z = torch.clamp(z, 0.0, 1.0)

        state = self.interface_state(T_raw)
        j_rxn = state["J_rxn"]
        raw = self.flux_net(x_net)

        endpoint_bubble = z * (1.0 - z)
        # The multiplicative shape term preserves J_D(delta)=J_rxn and
        # J_D(Xmax)=0 while still letting the interior profile bend.
        shape = torch.exp(1.5 * endpoint_bubble * torch.tanh(raw[:, 0:1]))
        base_flux = j_rxn * (1.0 - z) * shape

        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.04 * T_sim))
        residual_flux = (
            self.flux_residual_scale *
            time_gate *
            endpoint_bubble *
            torch.tanh(raw[:, 1:2])
        )
        return base_flux + residual_flux


class ExternalNet_v9_6_MultiscaleHISLayer(nn.Module):
    """Hermite Interface-State external reconstruction.

    The interface-state network supplies the reactive boundary state
    (C_C(delta,T), J_rxn).  A local asymptotic layer reconstructs the steep
    external D profile near the interface, a farfield Hermite correction closes
    the finite domain, and endpoint-vanishing residual modes keep interior
    expressivity without breaking the hard interface/farfield constraints.
    """

    def __init__(self, gamma_val, interface_state=None, normalize_inputs=True):
        super().__init__()
        self.gamma = gamma_val
        self.interface_state = interface_state if interface_state is not None else InterfaceStateNet_v9_6(normalize_inputs)
        self.normalize_inputs = normalize_inputs
        # Keep these names compatible with ExtBasis/Green checkpoints.
        self.net = MultiscaleResidualHead(out_features=1, width=256, depth=5)
        self.shape_net = MultiscaleResidualHead(out_features=2, width=192, depth=4)
        self.layer_state_net = MultiscaleResidualHead(out_features=2, width=160, depth=3, in_features=1)
        self.register_buffer(
            "correction_scales",
            torch.tensor([8.0, 6.0, 6.0], dtype=torch.float32).reshape(1, 3),
        )

    def _net_inputs(self, T_raw, X_raw):
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
            x_net = torch.cat([
                t_net,
                normalize_ext_x(X_raw)
            ], dim=1)
        else:
            t_net = T_raw
            x_net = torch.cat([T_raw, X_raw], dim=1)
        return t_net, x_net

    def _layer_params(self, T_raw):
        if self.normalize_inputs:
            t_net = normalize_time(T_raw)
        else:
            t_net = T_raw
        raw = self.layer_state_net(t_net)
        t_scale = torch.sqrt(torch.clamp(T_raw / T_sim, min=1e-4))
        ell = 0.025 + 3.0 * t_scale * torch.sigmoid(raw[:, 0:1])
        curvature = 1.5 * torch.tanh(raw[:, 1:2])
        return ell, curvature

    def _local_layer_cd(self, T_raw, X_raw):
        ext_length = X_ext_max - delta
        y = torch.clamp(X_raw - delta, 0.0, ext_length)
        state = self.interface_state(T_raw)
        d_int = self.gamma - state["C_C_int"]
        j_rxn = state["J_rxn"]
        target_interface_slope = -j_rxn / D_rel_D
        ell, curvature = self._layer_params(T_raw)

        eta = y / torch.clamp(ell, min=1e-5)
        eta_safe = torch.clamp(eta, min=0.0, max=40.0)
        exp_eta = torch.exp(-eta_safe)
        slope_match = target_interface_slope + d_int / torch.clamp(ell, min=1e-5)
        local = d_int * exp_eta + slope_match * y * exp_eta
        local = local + curvature * d_int * eta_safe ** 2 * exp_eta
        return local

    def _local_boundary_terms(self, T_raw, x_value, create_graph):
        X_raw = torch.full_like(T_raw, float(x_value))
        X_raw.requires_grad_(True)
        C_D_local = self._local_layer_cd(T_raw, X_raw)
        C_D_local_X = torch.autograd.grad(
            C_D_local.sum(),
            X_raw,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return C_D_local, C_D_local_X

    def forward(self, x_input):
        T_raw = x_input[:, 0:1]
        X_raw = x_input[:, 1:2]
        _, x_net = self._net_inputs(T_raw, X_raw)
        ext_length = X_ext_max - delta
        r = torch.clamp((X_raw - delta) / ext_length, 0.0, 1.0)

        C_D_local = self._local_layer_cd(T_raw, X_raw)

        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            T_boundary = T_raw if outer_grad_enabled else T_raw.detach()
            d_far, d_far_x = self._local_boundary_terms(
                T_boundary,
                X_ext_max,
                create_graph=outer_grad_enabled,
            )
        if not outer_grad_enabled:
            d_far = d_far.detach()
            d_far_x = d_far_x.detach()

        r2 = r * r
        r3 = r2 * r
        h01 = -2.0 * r3 + 3.0 * r2
        h11 = r3 - r2
        C_D_base = C_D_local + h01 * (-d_far) + h11 * ext_length * (-d_far_x)

        raw_main = self.net(x_net)
        raw_shape = self.shape_net(x_net)
        raw = torch.cat([raw_main, raw_shape], dim=1)
        endpoint_bubble = r2 * (1.0 - r) ** 2
        modes = torch.cat([
            endpoint_bubble,
            endpoint_bubble * (2.0 * r - 1.0),
            endpoint_bubble * (6.0 * r2 - 6.0 * r + 1.0),
        ], dim=1)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.04 * T_sim))
        residual = time_gate * torch.sum(self.correction_scales * modes * torch.tanh(raw), dim=1, keepdim=True)

        C_D = C_D_base + residual
        C_C = self.gamma - C_D
        return C_C, C_D


def create_models_v96(
    arch="legacy",
    normalize_inputs=True,
    green_time_grid=256,
    green_kernel_points=32,
    green_history_grad=True,
    green_cache_history=True,
    lift_time_grid=1024,
):
    if arch == "legacy":
        return (
            ThinLayerNet_v9_6(normalize_inputs=normalize_inputs),
            ExternalNet_v9_6(gamma, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale":
        return (
            ThinLayerNet_v9_6_Multiscale(normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_Multiscale(gamma, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale_hardbc":
        return (
            ThinLayerNet_v9_6_MultiscaleHardBC(normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_Multiscale(gamma, normalize_inputs=normalize_inputs),
        )
    if arch == "his_pinn":
        interface_state = HISInterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_HISPrototype(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_Multiscale(gamma, normalize_inputs=normalize_inputs),
        )
    if arch == "his_pinn_ext":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleHISLayer(gamma, interface_state=interface_state, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale_hermite":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleHermite(gamma, interface_state=interface_state, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale_hermite_extbasis":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleHermiteExtBasis(gamma, interface_state=interface_state, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale_green":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenKernel(gamma, interface_state=interface_state, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale_green_grid":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGrid(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_hybrid":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridHybrid(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_dynamic":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamic(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_dynamic_stage1":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        interface_state.continuous_time_features = True
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamicStage1(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_interface_memory":
        interface_state = InterfaceStateNet_v9_6_Memory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamic(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_memory":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridMemory(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel":
        interface_state = InterfaceStateNet_v9_6_FilmAbel(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamic(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMix(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamic(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_causal":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausal(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamic(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_causalconv":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalConv(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalConv(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybrid(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybrid(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridSmooth(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybridSmooth(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybridSmooth(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_fluxtrace":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridFluxTrace(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemoryMatchedAbel(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel":
        interface_state = InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemoryMixedAbel(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_film_tracegreen_clean":
        interface_state = InterfaceStateNet_v9_6_FilmTraceClean(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_FilmTraceGreenClean(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_film_tracegreen_productintegral":
        interface_state = InterfaceStateNet_v9_6_FilmTraceProductIntegral(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
            ),
            ExternalNet_v9_6_FilmTraceGreenClean(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_film_tracegreen_productintegral_lift":
        interface_state = InterfaceStateNet_v9_6_FilmTraceProductIntegral(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_InventoryHermiteLift(
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                lift_time_grid_points=lift_time_grid,
            ),
            ExternalNet_v9_6_FilmTraceGreenClean(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch in {
        "multiscale_film_tracegreen_clean_conservative",
        "multiscale_film_tracegreen_clean_mixedflux",
        "multiscale_film_tracegreen_conservative_lift",
    }:
        interface_state = InterfaceStateNet_v9_6_FilmTraceClean(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        if arch.endswith("mixedflux"):
            thin_class = ThinLayerNet_v9_6_MultiscaleHermiteMixedFlux
        elif arch.endswith("conservative_lift"):
            thin_class = ThinLayerNet_v9_6_InventoryHermiteLift
        else:
            thin_class = ThinLayerNet_v9_6_MultiscaleHermiteConservative
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("conservative_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        return (
            thin_class(**thin_kwargs),
            ExternalNet_v9_6_FilmTraceGreenClean(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch in {
        "multiscale_film_tracegreen_kparam",
        "multiscale_film_tracegreen_kparam_lift",
    }:
        if not np.isclose(gamma, REFERENCE_GAMMA, rtol=0.0, atol=1e-12):
            raise ValueError(
                "multiscale_film_tracegreen_kparam fixes gamma=10; "
                f"received gamma={gamma}"
            )
        interface_state = InterfaceStateNet_v9_6_FilmTraceKParam(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        interface_state.set_k_cat(k_cat_star)
        thin_class = (
            ThinLayerNet_v9_6_InventoryHermiteLiftKParam
            if arch.endswith("_lift")
            else ThinLayerNet_v9_6_MultiscaleHermiteKParam
        )
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        return (
            thin_class(**thin_kwargs),
            ExternalNet_v9_6_FilmTraceGreenKParam(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch in {
        "multiscale_film_tracegreen_gammaparam",
        "multiscale_film_tracegreen_gammaparam_lift",
    }:
        if not np.isclose(k_cat_star, REFERENCE_K_CAT_STAR, rtol=0.0, atol=1e-12):
            raise ValueError(
                "multiscale_film_tracegreen_gammaparam fixes k_cat=1; "
                f"received k_cat={k_cat_star}"
            )
        interface_state = InterfaceStateNet_v9_6_FilmTraceGammaParam(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        interface_state.set_gamma(gamma)
        thin_class = (
            ThinLayerNet_v9_6_InventoryHermiteLiftGammaParam
            if arch.endswith("_lift")
            else ThinLayerNet_v9_6_MultiscaleHermiteGammaParam
        )
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        return (
            thin_class(**thin_kwargs),
            ExternalNet_v9_6_FilmTraceGreenGammaParam(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_green_grid_film_abel_ema":
        interface_state = InterfaceStateNet_v9_6_FilmAbelEMA(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleGreenGridMemory(
                gamma,
                interface_state=interface_state,
                normalize_inputs=normalize_inputs,
                time_grid_points=green_time_grid,
                kernel_points=green_kernel_points,
                history_grad=green_history_grad,
                cache_history=green_cache_history,
            ),
        )
    if arch == "multiscale_buffer":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleBuffer(gamma, interface_state=interface_state, normalize_inputs=normalize_inputs),
        )
    if arch == "multiscale_fluxbuffer":
        interface_state = InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        return (
            ThinLayerNet_v9_6_MultiscaleHermite(interface_state=interface_state, normalize_inputs=normalize_inputs),
            ExternalNet_v9_6_MultiscaleFluxBuffer(gamma, interface_state=interface_state, normalize_inputs=normalize_inputs),
        )
    raise ValueError(f"Unknown architecture: {arch}")


def potential_theta(T):
    return torch.where(T <= T_switch * T_sim,
                       theta_i - 2*(theta_i - theta_switch) * T / T_sim,
                       theta_switch + 2*(theta_i - theta_switch) * (T - T_switch * T_sim) / T_sim)


def potential_theta_dot(T):
    down = torch.ones_like(T) * (-2.0 * (theta_i - theta_switch) / T_sim)
    up = torch.ones_like(T) * (2.0 * (theta_i - theta_switch) / T_sim)
    return torch.where(T <= T_switch * T_sim, down, up)


def potential_theta_dot_smooth(T, width=0.015):
    """Continuous scan-direction feature for neural correction heads.

    The physical triangular potential is continuous while its derivative jumps
    at reversal.  Concentrations should remain continuous, so correction
    networks should not receive a step input that lets them create a state jump.
    This smoothed derivative is used only as a neural feature; theta(t) itself
    and the physical scan reversal remain unchanged.
    """
    slope = 2.0 * (theta_i - theta_switch) / T_sim
    width_t = max(float(width) * float(T_sim), 1e-6)
    return slope * torch.tanh((T - T_switch * T_sim) / width_t)


# ==================== 模型保存/加载 ====================
def checkpoint_parameters(model_ext):
    parameters = {
        'sigma': sigma, 'theta_i': theta_i, 'theta_switch': theta_switch,
        'T_sim': T_sim, 'delta': delta, 'X_ext_max': X_ext_max,
        'gamma': GAMMAPARAM_REFERENCE if is_gamma_parameterized(model_ext=model_ext) else gamma,
        'k_cat_star': KPARAM_REFERENCE if is_k_parameterized(model_ext=model_ext) else k_cat_star,
        'lambda_factor': lambda_factor,
        'normalize_inputs': USE_NORMALIZED_COORDS,
        'green_time_grid': getattr(model_ext, 'time_grid_points', None),
        'green_kernel_points': getattr(model_ext, 'kernel_points', None),
        'green_history_grad': getattr(model_ext, 'history_grad', None),
    }
    parameters.update(parameterization_metadata(model_ext))
    return parameters


def save_model_v96(model_thin, model_ext, optimizer, scheduler, epoch, path, 
                   loss_history=None, best_val_loss=None, suffix=''):
    save_path = path.replace('.pth', f'{suffix}.pth') if suffix else path
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    rng_state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save({
        'epoch': epoch,
        'model_thin_state_dict': model_thin.state_dict(),
        'model_ext_state_dict': model_ext.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_history': loss_history if loss_history else {},
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'rng_state': rng_state,
        'parameters': checkpoint_parameters(model_ext),
    }, save_path)
    print(f"✅ Model saved to {save_path} (epoch {epoch})")


def load_compatible_state_dict(module, state_dict, module_name):
    current_state = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
            continue
        compatible[key] = value

    result = module.load_state_dict(compatible, strict=False)
    if skipped:
        preview = ", ".join(
            f"{name}: {old_shape}->{new_shape}" for name, old_shape, new_shape in skipped[:5]
        )
        more = "" if len(skipped) <= 5 else f", ... +{len(skipped) - 5} more"
        print(f"{module_name} state entries skipped for shape mismatch: {preview}{more}")
    return result


def validate_checkpoint_physical_parameters(
    checkpoint,
    path,
    model_ext=None,
    evaluation_k=None,
    allow_fixed_reference=False,
):
    parameters = checkpoint.get('parameters', {})
    target_is_gammaparam = is_gamma_parameterized(model_ext=model_ext)
    for name, active in (("gamma", gamma), ("delta", delta)):
        if name == "gamma" and target_is_gammaparam:
            continue
        stored = parameters.get(name)
        if stored is None:
            continue
        if not np.isclose(float(stored), float(active), rtol=1e-7, atol=1e-10):
            raise ValueError(
                f"Checkpoint parameter mismatch for {name}: checkpoint={stored}, "
                f"active={active}. Select a checkpoint trained with matching physics: {path}"
            )

    checkpoint_kind = parameters.get("parameterization", "fixed")
    if target_is_gammaparam:
        if checkpoint_kind == GAMMAPARAM_PARAMETERIZATION:
            expected = {
                "gamma_support": GAMMAPARAM_SUPPORT,
                "gamma_condition_transform": "tanh(log10(gamma/gamma_reference))",
                "gamma_interface_operator": GAMMAPARAM_INTERFACE_OPERATOR,
            }
            for name, active in expected.items():
                if parameters.get(name) != active:
                    raise ValueError(
                        f"Checkpoint gamma-parameterization mismatch for {name}: "
                        f"checkpoint={parameters.get(name)}, active={active}: {path}"
                    )
            stored_reference = parameters.get("gamma_reference")
            if stored_reference is None or not np.isclose(
                float(stored_reference), GAMMAPARAM_REFERENCE, rtol=1e-7, atol=1e-10
            ):
                raise ValueError(
                    "Checkpoint gamma_reference mismatch: "
                    f"checkpoint={stored_reference}, active={GAMMAPARAM_REFERENCE}: {path}"
                )
        elif allow_fixed_reference and checkpoint_kind == "fixed":
            stored_gamma = float(parameters.get("gamma", GAMMAPARAM_REFERENCE))
            stored_k = float(parameters.get("k_cat_star", REFERENCE_K_CAT_STAR))
            if not np.isclose(stored_gamma, GAMMAPARAM_REFERENCE, rtol=1e-7, atol=1e-10):
                raise ValueError(
                    "Only a fixed gamma=10 checkpoint can warm-start the "
                    f"gamma-parameterized model; checkpoint gamma={stored_gamma}: {path}"
                )
            if not np.isclose(stored_k, REFERENCE_K_CAT_STAR, rtol=1e-7, atol=1e-10):
                raise ValueError(
                    "Gamma-parameterized warm-start requires fixed k_cat=1; "
                    f"checkpoint k_cat={stored_k}: {path}"
                )
        else:
            raise ValueError(
                "Gamma parameterization requires either a matching gammaparam resume "
                "checkpoint or --warm-start-checkpoint with the fixed gamma=10, "
                "k_cat=1 baseline."
            )
        return

    target_is_kparam = is_k_parameterized(model_ext=model_ext)
    if target_is_kparam:
        active_k = model_active_k_cat(model_ext=model_ext) if evaluation_k is None else float(evaluation_k)
        if not np.isfinite(active_k) or active_k <= 0.0:
            raise ValueError(f"Evaluation k_cat must be finite and positive, got {active_k}")
        if checkpoint_kind == KPARAM_PARAMETERIZATION:
            stored_support = parameters.get("k_support")
            if stored_support != KPARAM_SUPPORT:
                raise ValueError(
                    "Checkpoint k-support mismatch: "
                    f"checkpoint={stored_support}, active={KPARAM_SUPPORT}: {path}"
                )
            expected_transform = "tanh(log10(k/k_reference))"
            stored_transform = parameters.get("k_condition_transform")
            if stored_transform != expected_transform:
                raise ValueError(
                    "Checkpoint k-condition transform mismatch: "
                    f"checkpoint={stored_transform}, active={expected_transform}: {path}"
                )
            active_reference = float(getattr(model_ext, "k_reference", KPARAM_REFERENCE))
            stored_reference = parameters.get("k_reference")
            if stored_reference is None or not np.isclose(
                float(stored_reference), active_reference, rtol=1e-7, atol=1e-10
            ):
                raise ValueError(
                    "Checkpoint k-parameterization mismatch for k_reference: "
                    f"checkpoint={stored_reference}, active={active_reference}: {path}"
                )
            interface_state = getattr(model_ext, "interface_state", None)
            active_operator = getattr(interface_state, "kparam_interface_operator", None)
            stored_operator = parameters.get("k_interface_operator")
            if active_operator is not None and stored_operator != active_operator:
                raise ValueError(
                    "Checkpoint k-interface operator mismatch: "
                    f"checkpoint={stored_operator}, active={active_operator}. "
                    "Warm-start from a fixed k=1 ProductIntegral checkpoint or "
                    f"resume a matching kparam checkpoint: {path}"
                )
            active_iterations = int(getattr(interface_state, "fixed_point_iterations", 2))
            stored_iterations = parameters.get("product_integral_fixed_point_iterations")
            if stored_iterations is None or int(stored_iterations) != active_iterations:
                raise ValueError(
                    "Checkpoint product-integral closure mismatch: "
                    f"checkpoint iterations={stored_iterations}, active={active_iterations}: {path}"
                )
            active_newton_iterations = int(
                getattr(interface_state, "newton_projection_iterations", 1)
            )
            stored_newton_iterations = parameters.get(
                "product_integral_newton_projection_iterations"
            )
            if (
                stored_newton_iterations is None
                or int(stored_newton_iterations) != active_newton_iterations
            ):
                raise ValueError(
                    "Checkpoint product-integral Newton projection mismatch: "
                    f"checkpoint iterations={stored_newton_iterations}, "
                    f"active={active_newton_iterations}: {path}"
                )
        elif allow_fixed_reference:
            stored_k = float(parameters.get("k_cat_star", KPARAM_REFERENCE))
            if not np.isclose(stored_k, KPARAM_REFERENCE, rtol=1e-7, atol=1e-10):
                raise ValueError(
                    "Only a fixed k_cat=1 checkpoint can warm-start/zero-shot the "
                    f"k-parameterized model; checkpoint k_cat={stored_k}: {path}"
                )
        else:
            raise ValueError(
                "A fixed-parameter checkpoint cannot resume a k-parameterized run. "
                "Use --warm-start-checkpoint for the fixed k=1 clean baseline."
            )
    else:
        if checkpoint_kind == KPARAM_PARAMETERIZATION:
            raise ValueError(
                f"A k-parameterized checkpoint requires the kparam architecture: {path}"
            )
        if checkpoint_kind == GAMMAPARAM_PARAMETERIZATION:
            raise ValueError(
                f"A gamma-parameterized checkpoint requires the gammaparam architecture: {path}"
            )
        stored_k = parameters.get("k_cat_star")
        if stored_k is not None and not np.isclose(
            float(stored_k), float(k_cat_star), rtol=1e-7, atol=1e-10
        ):
            raise ValueError(
                f"Checkpoint parameter mismatch for k_cat_star: checkpoint={stored_k}, "
                f"active={k_cat_star}. Select a matching fixed-parameter checkpoint: {path}"
            )


def load_model_v96(model_thin, model_ext, optimizer, scheduler, path, load_optimizer_state=True):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint_physical_parameters(checkpoint, path, model_ext=model_ext)
        load_compatible_state_dict(model_thin, checkpoint['model_thin_state_dict'], "Thin model")
        load_compatible_state_dict(model_ext, checkpoint['model_ext_state_dict'], "External model")
        optimizer_loaded = False
        if optimizer and checkpoint.get('optimizer_state_dict') is not None and load_optimizer_state:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_loaded = True
            except ValueError as exc:
                print(f"Optimizer state skipped: {exc}")
        elif optimizer and checkpoint.get('optimizer_state_dict') is not None and not load_optimizer_state:
            print("Optimizer state skipped because --reset-optimizer-state is active.")
        if scheduler and checkpoint.get('scheduler_state_dict') is not None and optimizer_loaded and load_optimizer_state:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as exc:
                print(f"Scheduler state skipped: {exc}")
        elif scheduler and checkpoint.get('scheduler_state_dict') is not None and not optimizer_loaded:
            print("Scheduler state skipped because optimizer state was not loaded.")

        rng_state = checkpoint.get('rng_state')
        if rng_state:
            if rng_state.get('torch') is not None:
                torch.set_rng_state(rng_state['torch'])
            if rng_state.get('numpy') is not None:
                np.random.set_state(rng_state['numpy'])
            if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

        epoch = checkpoint['epoch']
        loss_history = checkpoint.get('loss_history', {})
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✅ Model loaded from {path} (epoch {epoch})")
        return epoch, loss_history, best_val_loss
    return 0, {}, float('inf')


def warm_start_model_v96(model_thin, model_ext, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_physical_parameters(
        checkpoint,
        path,
        model_ext=model_ext,
        evaluation_k=KPARAM_REFERENCE,
        allow_fixed_reference=True,
    )
    load_compatible_state_dict(
        model_thin, checkpoint['model_thin_state_dict'], "Thin warm-start model"
    )
    load_compatible_state_dict(
        model_ext, checkpoint['model_ext_state_dict'], "External warm-start model"
    )
    if is_k_parameterized(model_ext=model_ext):
        set_model_k_cat(model_thin, model_ext, KPARAM_REFERENCE)
    if is_gamma_parameterized(model_ext=model_ext):
        set_model_gamma(model_thin, model_ext, GAMMAPARAM_REFERENCE)
    print(f"Warm-started model weights from {path}; optimizer and epoch were not loaded")
    return checkpoint


def normalize_loss_history(loss_history):
    if not loss_history:
        return loss_history

    total_len = len(loss_history.get('total', []))
    for key in (
        'interface_thin', 'interface_ext', 'bounds', 'reversal_continuity',
        'dynamic_reversal_jump', 'direct_reversal_jump',
        'direct_temporal_smooth', 'static_temporal_smooth',
        'phase_temporal_smooth', 'cint_reversal_jump',
        'cint_temporal_smooth',
    ):
        values = list(loss_history.get(key, []))
        if len(values) < total_len:
            values = [float('nan')] * (total_len - len(values)) + values
        elif len(values) > total_len:
            values = values[-total_len:]
        loss_history[key] = values
    return loss_history


# ==================== 物理验证工具函数 ====================
def verify_interface_physics(model_thin, model_ext, device, n_test=50):
    model_thin.eval()
    model_ext.eval()

    T_test = torch.linspace(0.1, 0.9, n_test, device=device).reshape(-1, 1) * T_sim
    T_test.requires_grad_(True)
    X_int = torch.ones_like(T_test) * delta
    X_int.requires_grad_(True)

    with torch.enable_grad():
        inputs_thin = torch.cat([T_test, X_int], dim=1)
        C_A_int, C_B_int = model_thin(inputs_thin)
        C_A_X_int = torch.autograd.grad(C_A_int.sum(), X_int, create_graph=True, retain_graph=True)[0]
        C_B_X_int = torch.autograd.grad(C_B_int.sum(), X_int, create_graph=True, retain_graph=True)[0]

        inputs_ext = torch.cat([T_test, X_int], dim=1)
        C_C_int, C_D_int = model_ext(inputs_ext)
        C_C_X_int = torch.autograd.grad(C_C_int.sum(), X_int, create_graph=True, retain_graph=True)[0]
        C_D_X_int = torch.autograd.grad(C_D_int.sum(), X_int, create_graph=True, retain_graph=True)[0]

        active_k = model_active_k_cat(model_ext=model_ext, model_thin=model_thin)
        J_rxn = active_k * C_B_int * C_C_int
        J_A = -D_rel_A * C_A_X_int
        J_B = -D_rel_B * C_B_X_int
        if (
            getattr(model_ext, "fluxtrace_analytic_boundary_flux", False) or
            getattr(model_ext, "external_analytic_boundary_flux", False)
        ):
            # FluxTrace treats the external boundary flux as the analytic trace
            # of the Green flux solution.  Pointwise autograd at y=0 only sees
            # the regular quadrature part and misses the singular boundary
            # contribution, so report the trace-consistent flux here.
            J_C = -J_rxn
            J_D = J_rxn
        else:
            J_C = -D_rel_C * C_C_X_int
            J_D = -D_rel_D * C_D_X_int

        results = {
            'J_A': J_A.detach().cpu().numpy(),
            'J_B': J_B.detach().cpu().numpy(),
            'J_C': J_C.detach().cpu().numpy(),
            'J_D': J_D.detach().cpu().numpy(),
            'J_rxn': J_rxn.detach().cpu().numpy(),
            'C_B_int': C_B_int.detach().cpu().numpy(),
            'C_C_int': C_C_int.detach().cpu().numpy(),
            'err_A': torch.abs(J_A + J_rxn).mean().item(),
            'err_B': torch.abs(J_B - J_rxn).mean().item(),
            'err_C': torch.abs(J_C + J_rxn).mean().item(),
            'err_D': torch.abs(J_D - J_rxn).mean().item(),
        }
        return results


def print_physics_verification(results):
    print(f"\n{'='*60}")
    print("Interface Physics Verification (v9.6)")
    print(f"{'='*60}")
    print(f"Mean reaction rate J_rxn: {results['J_rxn'].mean():.4e}")
    print(f"Interface C_B mean: {results['C_B_int'].mean():.4e}")
    print(f"Interface C_C mean: {results['C_C_int'].mean():.4e}")
    print(f"\nFlux verification:")
    print(f"  J_A mean: {results['J_A'].mean():.4e}, error: {results['err_A']:.4e}")
    print(f"  J_B mean: {results['J_B'].mean():.4e}, error: {results['err_B']:.4e}")
    print(f"  J_C mean: {results['J_C'].mean():.4e}, error: {results['err_C']:.4e}")
    print(f"  J_D mean: {results['J_D'].mean():.4e}, error: {results['err_D']:.4e}")
    J_total_in = results['J_B'].mean() + results['J_C'].mean()
    J_total_out = results['J_A'].mean() + results['J_D'].mean()
    print(f"\nMass conservation:")
    print(f"  Net flux: {J_total_in + J_total_out:.4e} (should be ≈ 0)")
    print(f"{'='*60}")


# ==================== 验证函数 ====================
def validate_model(model_thin, model_ext, device, epoch, verbose=True):
    model_thin.eval()
    model_ext.eval()
    active_k = model_active_k_cat(model_ext=model_ext, model_thin=model_thin)
    active_gamma = model_active_gamma(model_ext=model_ext, model_thin=model_thin)
    active_condition = active_gamma if is_gamma_parameterized(
        model_ext=model_ext, model_thin=model_thin
    ) else active_k

    with torch.no_grad():
        # 1. Nernst误差
        T_test = torch.linspace(0, T_sim, 200, device=device).reshape(-1, 1)
        X_test = torch.zeros_like(T_test)
        C_A, C_B = model_thin(conditioned_model_inputs(
            model_thin, T_test, X_test, active_condition
        ))
        theta = potential_theta(T_test)
        nernst_err = torch.sqrt(torch.mean((C_A - C_B * torch.exp(theta))**2)).item()
        nernst_max_err = torch.max(torch.abs(C_A - C_B * torch.exp(theta))).item()
        C_A_eq = torch.sigmoid(theta)
        C_B_eq = torch.sigmoid(-theta)
        surface_state_err = torch.sqrt(torch.mean((C_A - C_A_eq)**2 + (C_B - C_B_eq)**2)).item()

        X_int = torch.ones_like(T_test) * delta
        _, C_B_int = model_thin(conditioned_model_inputs(
            model_thin, T_test, X_int, active_condition
        ))
        C_C_int, _ = model_ext(conditioned_model_inputs(
            model_ext, T_test, X_int, active_condition
        ))
        J_rxn_mean = torch.mean(active_k * C_B_int * C_C_int).item()
        C_B_int_mean = torch.mean(C_B_int).item()
        C_C_int_mean = torch.mean(C_C_int).item()

        # 2. 薄层守恒
        T_cons = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device) * T_sim
        X_cons = torch.linspace(0, delta, 50, device=device).reshape(-1, 1)
        err_AB_max = 0
        for t in T_cons:
            T_grid = torch.ones_like(X_cons) * t
            inputs = conditioned_model_inputs(
                model_thin, T_grid, X_cons, active_condition
            )
            C_A_test, C_B_test = model_thin(inputs)
            err = torch.abs(C_A_test + C_B_test - 1.0).max().item()
            err_AB_max = max(err_AB_max, err)

        # 3. 外部守恒
        X_cons_ext = torch.linspace(delta, X_ext_max, 50, device=device).reshape(-1, 1)
        err_CD_max = 0
        for t in T_cons:
            T_grid = torch.ones_like(X_cons_ext) * t
            inputs = conditioned_model_inputs(
                model_ext, T_grid, X_cons_ext, active_condition
            )
            C_C_test, C_D_test = model_ext(inputs)
            err = torch.abs(C_C_test + C_D_test - active_gamma).max().item()
            err_CD_max = max(err_CD_max, err)

        interface_state = getattr(model_ext, "interface_state", None)
        product_integral_diag = getattr(
            interface_state, "_last_product_integral_diagnostics", None
        )
        if product_integral_diag is not None:
            product_integral_diag = {
                key: (
                    bool(value.detach().cpu())
                    if key == "all_finite"
                    else float(value.detach().cpu())
                )
                for key, value in product_integral_diag.items()
            }

    # 4. CV峰电流
    T_surf = torch.linspace(0, T_sim, 300, device=device).reshape(-1, 1)
    X_surf = torch.zeros_like(T_surf)
    X_surf.requires_grad_(True)

    C_A_surf, _ = model_thin(conditioned_model_inputs(
        model_thin, T_surf, X_surf, active_condition
    ))
    C_A_X_surf = torch.autograd.grad(
        C_A_surf.sum(), X_surf, 
        create_graph=False, retain_graph=False
    )[0]

    J_total = -C_A_X_surf.detach().cpu().numpy().flatten()
    theta_values = potential_theta(T_surf).cpu().numpy().flatten()
    J_peak = np.min(J_total)
    peak_idx = np.argmin(J_total)
    theta_peak = theta_values[peak_idx]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validation @ Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Nernst RMSE: {nernst_err:.4e}")
        print(f"Nernst Max:  {nernst_max_err:.4e}")
        print(f"Surface state RMSE: {surface_state_err:.4e}")
        print(f"Interface means: C_B={C_B_int_mean:.4e}, C_C={C_C_int_mean:.4e}, J_rxn={J_rxn_mean:.4e}")
        print(f"Peak Current: {J_peak:.4f} @ θ={theta_peak:.2f}")
        print(f"Conservation A+B: {err_AB_max:.2e}")
        print(f"Conservation C+D: {err_CD_max:.2e}")
        if product_integral_diag is not None:
            print(
                "Product integral: "
                f"C_D=[{product_integral_diag['c_d_min']:.4e}, "
                f"{product_integral_diag['c_d_max']:.4e}], "
                f"bounds={product_integral_diag['bounds_max']:.2e}, "
                f"fixed-point={product_integral_diag['fixed_point_max']:.2e}, "
                f"finite={product_integral_diag['all_finite']}"
            )
        print(f"{'='*60}")

    return {
        'nernst_err': nernst_err,
        'nernst_max_err': nernst_max_err,
        'surface_state_err': surface_state_err,
        'product_integral': product_integral_diag,
        'C_B_int_mean': C_B_int_mean,
        'C_C_int_mean': C_C_int_mean,
        'J_rxn_mean': J_rxn_mean,
        'J_peak': J_peak,
        'theta_peak': theta_peak,
        'err_AB': err_AB_max,
        'err_CD': err_CD_max
    }


# ==================== v9.6 训练函数 - 界面耦合增强 ====================
def concentration_bounds_loss(C_A, C_B, C_C, C_D, gamma_value=None):
    gamma_value = gamma if gamma_value is None else float(gamma_value)
    thin_loss = (
        torch.mean(torch.relu(-C_A) ** 2) +
        torch.mean(torch.relu(C_A - 1.0) ** 2) +
        torch.mean(torch.relu(-C_B) ** 2) +
        torch.mean(torch.relu(C_B - 1.0) ** 2)
    )
    ext_scale = external_residual_scale(gamma_value)
    ext_loss = (
        torch.mean((ext_scale * torch.relu(-C_C)) ** 2) +
        torch.mean((ext_scale * torch.relu(C_C - gamma_value)) ** 2) +
        torch.mean((ext_scale * torch.relu(-C_D)) ** 2) +
        torch.mean((ext_scale * torch.relu(C_D - gamma_value)) ** 2)
    )
    return thin_loss + ext_loss


def fixed_physics_validation_score(
    model_thin,
    model_ext,
    device,
    pde_thin_weight,
    pde_ext_weight,
    farfield_weight,
    initial_weight,
    bounds_weight,
    thin_interface_weight,
    ext_interface_weight,
    n_points=96,
    k_value=None,
    gamma_value=None,
    current_balance_weight=0.0,
):
    """Deterministic, FDM-free physics score for checkpointing and early stop."""
    n_points = max(32, int(n_points))
    thin_was_training = model_thin.training
    ext_was_training = model_ext.training
    previous_k = model_active_k_cat(model_ext=model_ext, model_thin=model_thin)
    previous_gamma = model_active_gamma(model_ext=model_ext, model_thin=model_thin)
    if k_value is not None:
        set_model_k_cat(model_thin, model_ext, k_value)
    if gamma_value is not None:
        set_model_gamma(model_thin, model_ext, gamma_value)
    model_thin.eval()
    model_ext.eval()
    if hasattr(model_ext, "clear_step_cache"):
        model_ext.clear_step_cache()

    active_k = model_active_k_cat(model_ext=model_ext, model_thin=model_thin)
    active_gamma = model_active_gamma(model_ext=model_ext, model_thin=model_thin)
    active_condition = active_gamma if is_gamma_parameterized(
        model_ext=model_ext, model_thin=model_thin
    ) else active_k
    ext_scale = external_residual_scale(active_gamma)
    flux_scale = flux_residual_scale(active_gamma, active_k)
    is_flux_state = hasattr(model_ext, "flux_d")
    is_mixed_thin = bool(getattr(model_thin, "mixed_flux_thin", False))

    def fixed_pair(x_min, x_max, multiplier):
        index = torch.arange(n_points, device=device, dtype=torch.float32)
        time_fraction = (index + 0.5) / n_points
        space_fraction = ((index * multiplier) % n_points + 0.5) / n_points
        T_eval = (time_fraction * T_sim).reshape(-1, 1).requires_grad_(True)
        X_eval = (x_min + space_fraction * (x_max - x_min)).reshape(-1, 1)
        X_eval.requires_grad_(True)
        return T_eval, X_eval

    try:
        with torch.enable_grad():
            T_thin, X_thin = fixed_pair(0.0, delta, 37)
            thin_inputs = conditioned_model_inputs(model_thin, T_thin, X_thin, active_condition)
            C_A, C_B = model_thin(thin_inputs)
            C_B_T = torch.autograd.grad(C_B.sum(), T_thin, create_graph=True, retain_graph=True)[0]
            C_B_X = torch.autograd.grad(C_B.sum(), X_thin, create_graph=True, retain_graph=True)[0]
            if is_mixed_thin:
                q_B = model_thin.flux_b(thin_inputs)
                q_B_X = torch.autograd.grad(q_B.sum(), X_thin, create_graph=True)[0]
                loss_thin_conservation = torch.mean((C_B_T + q_B_X) ** 2)
                loss_thin_constitutive = torch.mean((q_B + D_rel_B * C_B_X) ** 2)
                loss_pde_thin = loss_thin_conservation + 2.0 * loss_thin_constitutive
            else:
                C_A_T = torch.autograd.grad(C_A.sum(), T_thin, create_graph=True, retain_graph=True)[0]
                C_A_X = torch.autograd.grad(C_A.sum(), X_thin, create_graph=True, retain_graph=True)[0]
                C_A_XX = torch.autograd.grad(C_A_X.sum(), X_thin, create_graph=True, retain_graph=True)[0]
                C_B_XX = torch.autograd.grad(C_B_X.sum(), X_thin, create_graph=True)[0]
                loss_pde_thin = (
                    torch.mean((C_A_T - D_rel_A * C_A_XX) ** 2) +
                    torch.mean((C_B_T - D_rel_B * C_B_XX) ** 2)
                )
                loss_thin_conservation = torch.zeros((), device=device)
                loss_thin_constitutive = torch.zeros((), device=device)

            j_ref = characteristic_reaction_flux(active_gamma, active_k)
            loss_current_balance = torch.zeros((), device=device)
            diagnose_current_balance = bool(
                current_balance_weight > 0.0 or
                getattr(model_thin, "conservative_thin_current", False)
            )
            if diagnose_current_balance:
                T_current = torch.linspace(
                    0.0, float(T_sim), n_points, device=device
                ).reshape(-1, 1).requires_grad_(True)
                current = thin_current_components(
                    model_thin, T_current, quadrature_points=16, create_graph=False
                )
                loss_current_balance = torch.mean(
                    (current["balance_residual"] / max(j_ref, 1e-8)) ** 2
                )

            T_ext, X_ext = fixed_pair(delta, X_ext_max, 53)
            ext_inputs = conditioned_model_inputs(model_ext, T_ext, X_ext, active_condition)
            C_C, C_D = model_ext(ext_inputs)
            if hasattr(model_ext, "pde_fields"):
                C_C_pde, C_D_pde = model_ext.pde_fields(ext_inputs)
            else:
                C_C_pde, C_D_pde = C_C, C_D

            if is_flux_state:
                C_D_T = torch.autograd.grad(C_D_pde.sum(), T_ext, create_graph=True, retain_graph=True)[0]
                C_D_X = torch.autograd.grad(C_D_pde.sum(), X_ext, create_graph=True, retain_graph=True)[0]
                J_D_ext = model_ext.flux_d(ext_inputs)
                J_D_X = torch.autograd.grad(J_D_ext.sum(), X_ext, create_graph=True)[0]
                loss_ext_conservation = torch.mean((ext_scale * (C_D_T + J_D_X)) ** 2)
                loss_ext_constitutive = torch.mean((ext_scale * (J_D_ext + D_rel_D * C_D_X)) ** 2)
                loss_pde_ext = loss_ext_conservation + 2.0 * loss_ext_constitutive
            else:
                C_C_T = torch.autograd.grad(C_C_pde.sum(), T_ext, create_graph=True, retain_graph=True)[0]
                C_C_X = torch.autograd.grad(C_C_pde.sum(), X_ext, create_graph=True, retain_graph=True)[0]
                C_C_XX = torch.autograd.grad(C_C_X.sum(), X_ext, create_graph=True)[0]
                loss_pde_ext = torch.mean((ext_scale * (C_C_T - D_rel_C * C_C_XX)) ** 2)

            T_state = torch.linspace(0.0, float(T_sim), n_points, device=device).reshape(-1, 1)
            X_surface = torch.zeros_like(T_state)
            C_A_surface, C_B_surface = model_thin(
                conditioned_model_inputs(model_thin, T_state, X_surface, active_condition)
            )
            theta_state = potential_theta(T_state)
            C_A_eq = torch.sigmoid(theta_state)
            C_B_eq = torch.sigmoid(-theta_state)
            loss_surface = torch.mean(
                (C_A_surface - C_A_eq) ** 2 + (C_B_surface - C_B_eq) ** 2
            )

            X_far = torch.ones_like(T_state) * X_ext_max
            C_C_far, C_D_far = model_ext(
                conditioned_model_inputs(model_ext, T_state, X_far, active_condition)
            )
            loss_farfield = (
                torch.mean((ext_scale * (C_C_far - active_gamma)) ** 2) +
                torch.mean((ext_scale * C_D_far) ** 2)
            )

            X_ini_thin = torch.linspace(0.0, float(delta), n_points, device=device).reshape(-1, 1)
            X_ini_ext = torch.linspace(float(delta), float(X_ext_max), n_points, device=device).reshape(-1, 1)
            T_ini = torch.zeros_like(X_ini_thin)
            C_A_ini, C_B_ini = model_thin(
                conditioned_model_inputs(model_thin, T_ini, X_ini_thin, active_condition)
            )
            C_C_ini, C_D_ini = model_ext(
                conditioned_model_inputs(model_ext, T_ini, X_ini_ext, active_condition)
            )
            loss_initial = (
                torch.mean((C_A_ini - 1.0) ** 2) + torch.mean(C_B_ini ** 2) +
                torch.mean((ext_scale * (C_C_ini - active_gamma)) ** 2) +
                torch.mean((ext_scale * C_D_ini) ** 2)
            )
            loss_bounds = concentration_bounds_loss(
                C_A, C_B, C_C, C_D, gamma_value=active_gamma
            )

            index = torch.arange(n_points, device=device, dtype=torch.float32)
            T_int = (((index + 0.5) / n_points) * T_sim).reshape(-1, 1)
            X_int_thin = torch.ones_like(T_int, requires_grad=True) * delta
            C_A_int, C_B_int = model_thin(
                conditioned_model_inputs(model_thin, T_int, X_int_thin, active_condition)
            )
            C_A_X_int = torch.autograd.grad(C_A_int.sum(), X_int_thin, create_graph=True, retain_graph=True)[0]
            C_B_X_int = torch.autograd.grad(C_B_int.sum(), X_int_thin, create_graph=True)[0]

            X_int_ext = torch.ones_like(T_int, requires_grad=True) * delta
            C_C_int, C_D_int = model_ext(
                conditioned_model_inputs(model_ext, T_int, X_int_ext, active_condition)
            )
            J_rxn = active_k * C_B_int * C_C_int
            loss_interface_thin = (
                torch.mean((flux_scale * (-D_rel_A * C_A_X_int + J_rxn)) ** 2) +
                torch.mean((flux_scale * (-D_rel_B * C_B_X_int - J_rxn)) ** 2)
            )

            if getattr(model_ext, "tracegreen_external", False):
                loss_interface_ext = torch.mean(
                    (ext_scale * (C_C_int + C_D_int - active_gamma)) ** 2
                )
            elif getattr(model_ext, "fluxtrace_analytic_boundary_flux", False):
                trace_loss = model_ext.fluxtrace_trace_consistency_loss(T_int)
                loss_interface_ext = (
                    torch.mean((ext_scale * (C_C_int + C_D_int - active_gamma)) ** 2) +
                    float(getattr(model_ext, "fluxtrace_trace_weight", 1.0)) * trace_loss
                )
            else:
                C_C_X_int = torch.autograd.grad(C_C_int.sum(), X_int_ext, create_graph=True, retain_graph=True)[0]
                C_D_X_int = torch.autograd.grad(C_D_int.sum(), X_int_ext, create_graph=True)[0]
                loss_interface_ext = (
                    torch.mean((flux_scale * (-D_rel_C * C_C_X_int + J_rxn)) ** 2) +
                    torch.mean((flux_scale * (-D_rel_D * C_D_X_int - J_rxn)) ** 2)
                )

            interface_state = getattr(model_ext, "interface_state", None)
            loss_reversal = torch.zeros((), device=device)
            if getattr(interface_state, "continuous_time_features", False):
                eps_t = 0.0025 * T_sim
                X_rev = torch.linspace(float(delta), float(X_ext_max), 64, device=device).reshape(-1, 1)
                T_minus = torch.ones_like(X_rev) * (T_switch * T_sim - eps_t)
                T_plus = torch.ones_like(X_rev) * (T_switch * T_sim + eps_t)
                C_C_minus, C_D_minus = model_ext(
                    conditioned_model_inputs(model_ext, T_minus, X_rev, active_condition)
                )
                C_C_plus, C_D_plus = model_ext(
                    conditioned_model_inputs(model_ext, T_plus, X_rev, active_condition)
                )
                loss_reversal = torch.mean(
                    (C_C_plus - C_C_minus) ** 2 + (C_D_plus - C_D_minus) ** 2
                )

            product_integral_bounds_max = 0.0
            product_integral_closure_max = 0.0
            product_integral_valid = True
            product_integral_diag = getattr(
                interface_state, "_last_product_integral_diagnostics", None
            )
            if getattr(interface_state, "product_integral_interface", False):
                if product_integral_diag is None:
                    product_integral_valid = False
                else:
                    product_integral_bounds_max = float(
                        product_integral_diag["bounds_max"].detach().cpu()
                    )
                    product_integral_closure_max = float(
                        product_integral_diag["fixed_point_max"].detach().cpu()
                    )
                    product_integral_valid = bool(
                        product_integral_diag["all_finite"].detach().cpu()
                    ) and (
                        product_integral_bounds_max <= 1e-6 * (
                            1.0 if getattr(interface_state, "normalized_gamma_product_integral", False)
                            else active_gamma
                        )
                    ) and (
                        product_integral_closure_max <= 1e-5 * (
                            1.0 if getattr(interface_state, "normalized_gamma_product_integral", False)
                            else active_gamma
                        )
                    )

            score = (
                pde_thin_weight * loss_pde_thin +
                pde_ext_weight * loss_pde_ext +
                100.0 * loss_surface +
                farfield_weight * loss_farfield +
                initial_weight * loss_initial +
                bounds_weight * loss_bounds +
                thin_interface_weight * loss_interface_thin +
                ext_interface_weight * loss_interface_ext +
                100.0 * loss_reversal +
                current_balance_weight * loss_current_balance
            )
            components = {
                "score": float(score.detach().cpu()),
                "k_cat": active_k,
                "gamma": active_gamma,
                "pde_thin": float(loss_pde_thin.detach().cpu()),
                "pde_ext": float(loss_pde_ext.detach().cpu()),
                "surface": float(loss_surface.detach().cpu()),
                "farfield": float(loss_farfield.detach().cpu()),
                "initial": float(loss_initial.detach().cpu()),
                "bounds": float(loss_bounds.detach().cpu()),
                "interface_thin": float(loss_interface_thin.detach().cpu()),
                "interface_ext": float(loss_interface_ext.detach().cpu()),
                "reversal": float(loss_reversal.detach().cpu()),
                "current_balance": float(loss_current_balance.detach().cpu()),
                "current_balance_rmse": float(
                    max(j_ref, 1e-8) * torch.sqrt(loss_current_balance).detach().cpu()
                ),
                "thin_conservation": float(loss_thin_conservation.detach().cpu()),
                "thin_constitutive": float(loss_thin_constitutive.detach().cpu()),
                "product_integral_bounds_max": product_integral_bounds_max,
                "product_integral_closure_max": product_integral_closure_max,
                "product_integral_valid": product_integral_valid,
            }
    finally:
        if hasattr(model_ext, "clear_step_cache"):
            model_ext.clear_step_cache()
        if k_value is not None:
            set_model_k_cat(model_thin, model_ext, previous_k)
        if gamma_value is not None:
            set_model_gamma(model_thin, model_ext, previous_gamma)
        model_thin.train(thin_was_training)
        model_ext.train(ext_was_training)

    return components


def parameterized_physics_validation_score(
    model_thin,
    model_ext,
    device,
    k_values=KPARAM_VALIDATION_VALUES,
    **kwargs,
):
    per_k = {}
    for value in k_values:
        result = fixed_physics_validation_score(
            model_thin,
            model_ext,
            device,
            k_value=float(value),
            **kwargs,
        )
        if not result.get("product_integral_valid", True):
            raise FloatingPointError(
                "Invalid product-integral trace during k validation: "
                f"k={float(value):g}, bounds={result['product_integral_bounds_max']:.3e}, "
                f"closure={result['product_integral_closure_max']:.3e}"
            )
        per_k[f"{float(value):g}"] = result

    scores = np.asarray([entry["score"] for entry in per_k.values()], dtype=float)
    aggregate = 0.5 * float(np.mean(scores)) + 0.5 * float(np.max(scores))
    component_names = [
        "pde_thin", "pde_ext", "surface", "farfield", "initial",
        "bounds", "interface_thin", "interface_ext", "reversal",
        "current_balance", "current_balance_rmse",
        "thin_conservation", "thin_constitutive",
        "product_integral_bounds_max", "product_integral_closure_max",
    ]
    result = {
        "score": aggregate,
        "score_mean": float(np.mean(scores)),
        "score_worst": float(np.max(scores)),
        "per_k": per_k,
        "product_integral_valid": True,
    }
    for name in component_names:
        result[name] = float(np.mean([entry[name] for entry in per_k.values()]))
    return result


def gamma_parameterized_physics_validation_score(
    model_thin,
    model_ext,
    device,
    gamma_values=GAMMAPARAM_VALIDATION_VALUES,
    **kwargs,
):
    per_gamma = {}
    for value in gamma_values:
        result = fixed_physics_validation_score(
            model_thin,
            model_ext,
            device,
            gamma_value=float(value),
            **kwargs,
        )
        if not result.get("product_integral_valid", True):
            raise FloatingPointError(
                "Invalid product-integral trace during gamma validation: "
                f"gamma={float(value):g}, bounds={result['product_integral_bounds_max']:.3e}, "
                f"closure={result['product_integral_closure_max']:.3e}"
            )
        per_gamma[f"{float(value):g}"] = result

    scores = np.asarray([entry["score"] for entry in per_gamma.values()], dtype=float)
    result = {
        "score": 0.5 * float(np.mean(scores)) + 0.5 * float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_worst": float(np.max(scores)),
        "per_gamma": per_gamma,
        "product_integral_valid": True,
    }
    component_names = [
        "pde_thin", "pde_ext", "surface", "farfield", "initial",
        "bounds", "interface_thin", "interface_ext", "reversal",
        "current_balance", "current_balance_rmse", "thin_conservation",
        "thin_constitutive", "product_integral_bounds_max",
        "product_integral_closure_max",
    ]
    for name in component_names:
        result[name] = float(np.mean([entry[name] for entry in per_gamma.values()]))
    return result


def run_fdm_posterior_compare(model_thin, model_ext, epoch, arch_name, fdm_pkl,
                              output_dir=None, n_time=160, n_x_in=120, n_x_out=160,
                              batch_size=65536, save_figure=False, save_fields=False):
    if not fdm_pkl:
        return
    if not os.path.exists(fdm_pkl):
        print(f"FDM posterior concentration compare skipped; file not found: {fdm_pkl}")
        return

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_concentration_fields.py")
    if not os.path.exists(script_path):
        print(f"FDM posterior concentration compare skipped; script not found: {script_path}")
        return

    if output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(MODEL_V96_PATH))
        output_dir = os.path.join(checkpoint_dir, "fdm_compare")
    os.makedirs(output_dir, exist_ok=True)

    eval_checkpoint = os.path.join(output_dir, "fdm_compare_latest_eval.pth")
    output_json = os.path.join(output_dir, f"concentration_compare_epoch_{epoch:06d}.json")
    output_npz = (
        os.path.join(output_dir, f"concentration_compare_fields_epoch_{epoch:06d}.npz")
        if save_fields else ""
    )
    output_figure = (
        os.path.join(output_dir, f"concentration_residual_summary_epoch_{epoch:06d}.png")
        if save_figure else ""
    )

    torch.save({
        "epoch": epoch,
        "model_thin_state_dict": model_thin.state_dict(),
        "model_ext_state_dict": model_ext.state_dict(),
        "parameters": checkpoint_parameters(model_ext),
    }, eval_checkpoint)

    cmd = [
        sys.executable, "-u", script_path,
        "--fdm-pkl", fdm_pkl,
        "--arch", arch_name,
        "--gamma", str(model_active_gamma(model_ext=model_ext, model_thin=model_thin)),
        "--input-mode", "normalized" if USE_NORMALIZED_COORDS else "legacy",
        "--checkpoint", eval_checkpoint,
        "--n-time", str(n_time),
        "--n-x-in", str(n_x_in),
        "--n-x-out", str(n_x_out),
        "--batch-size", str(batch_size),
        "--cv-points", "800",
        "--output-json", output_json,
        "--output-npz", output_npz,
        "--output-figure", output_figure,
    ]
    if not is_k_parameterized(model_ext=model_ext):
        cmd.extend(["--k-cat-star", str(k_cat_star)])
    if isinstance(model_ext, ExternalNet_v9_6_MultiscaleGreenGrid):
        cmd.extend([
            "--green-time-grid", str(model_ext.time_grid_points),
            "--green-kernel-points", str(model_ext.kernel_points),
        ])
        if not model_ext.history_grad:
            cmd.append("--green-detach-history")
    if getattr(model_thin, "inventory_hermite_lift", False):
        cmd.extend([
            "--lift-time-grid", str(model_thin.lift_time_grid_points),
        ])

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"\nFDM posterior concentration compare failed @ epoch {epoch}")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        return

    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception as exc:
        print(f"\nFDM posterior concentration compare completed, but metrics JSON could not be read: {exc}")
        if result.stdout:
            print(result.stdout.strip())
        return

    print(f"\n{'='*60}")
    print(f"FDM posterior concentration compare @ Epoch {epoch}")
    print("Evaluation only: FDM is not used in training loss.")
    print(f"Metrics JSON: {output_json}")
    print("field,rmse,mae,max_abs,bias,r2,nrmse")
    for name in [
        "C_A", "C_B", "C_C", "C_D", "C_B_int", "C_C_int",
        "CV_J_surface", "CV_J_conservative", "CV_J_surface_vs_conservative",
    ]:
        if name not in metrics:
            continue
        row = metrics[name]
        print(
            f"{name},{row['rmse']:.6e},{row['mae']:.6e},{row['max_abs']:.6e},"
            f"{row['bias']:.6e},{row['r2']:.6e},{row['nrmse']:.6e}"
        )
    print(f"overall_rmse,{metrics['overall']['rmse']:.6e}")
    print(f"{'='*60}")


def sample_kparam_value(
    local_epoch,
    anchor_epochs,
    anchor_probability=0.5,
    log10_std=KPARAM_LOG10_STD,
):
    anchor_probability = float(np.clip(anchor_probability, 0.5, 1.0))
    if local_epoch < int(anchor_epochs) or np.random.random() < anchor_probability:
        return float(np.random.choice(KPARAM_ANCHORS, p=(0.25, 0.50, 0.25)))
    log10_std = float(log10_std)
    if not np.isfinite(log10_std) or log10_std <= 0.0:
        raise ValueError("kparam_log10_std must be finite and positive")
    for _ in range(16):
        value = float(10.0 ** np.random.normal(0.0, log10_std))
        if np.isfinite(value) and value > 0.0:
            return value
    raise FloatingPointError("Could not sample a finite positive k_cat")


def sample_gammaparam_value(anchor_probability=0.6):
    """Sample one shared gamma for a complete causal optimizer step."""
    if np.random.random() < float(np.clip(anchor_probability, 0.0, 1.0)):
        return float(np.random.choice(GAMMAPARAM_ANCHORS))
    log_min, log_max = np.log10(GAMMAPARAM_CALIBRATION_RANGE)
    return float(10.0 ** np.random.uniform(log_min, log_max))


def train_model_v9_6(model_thin, model_ext, n_epochs=30000, start_epoch=0, resume=False,
                     early_stop=True, resume_checkpoint=None, resume_best=False,
                     save_every=2000, thin_interface_weight=None, ext_interface_weight=200.0,
                     reset_best_score=False, pde_thin_weight=10.0, pde_ext_weight=10.0,
                     farfield_weight=1.0, initial_weight=1.0, bounds_weight=None,
                     arch_name="legacy", fdm_compare_pkl="", fdm_compare_every=0,
                     fdm_compare_dir=None, fdm_compare_n_time=160,
                     fdm_compare_n_x_in=120, fdm_compare_n_x_out=160,
                     fdm_compare_batch_size=65536, fdm_compare_save_figure=False,
                     fdm_compare_save_fields=False, reset_optimizer_state=False,
                     base_train_points=8000, max_train_points=15000,
                     train_point_growth=40, progress_every=50,
                     empty_cache_every=0, learning_rate=5e-5,
                     abort_on_nan=False,
                     clean_residual_initial_scale=0.0,
                     clean_residual_decay_epochs=0,
                     early_stop_check_every=100,
                     early_stop_patience_checks=4,
                     early_stop_min_epochs=-1,
                     early_stop_min_relative_improvement=0.005,
                     early_stop_ema_alpha=0.5,
                     early_stop_validation_points=96,
                     kparam_anchor_epochs=1000,
                     kparam_anchor_probability=0.5,
                     kparam_log10_std=KPARAM_LOG10_STD,
                     kparam_freeze_backbone_epochs=1000,
                     gammaparam_anchor_probability=0.6,
                     gammaparam_freeze_backbone=True,
                     current_balance_weight=None,
                     current_balance_ramp_epochs=100,
                     current_balance_samples=256):
    if getattr(model_thin, "kparam_posterior_lift", False):
        raise ValueError(
            "multiscale_film_tracegreen_kparam_lift is posterior-only; "
            "train multiscale_film_tracegreen_kparam and apply the lift at evaluation"
        )
    if getattr(model_thin, "gammaparam_posterior_lift", False):
        raise ValueError(
            "multiscale_film_tracegreen_gammaparam_lift is posterior-only; "
            "train multiscale_film_tracegreen_gammaparam and apply the lift at evaluation"
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_thin.to(device)
    model_ext.to(device)

    params = []
    seen_params = set()
    for module in (model_thin, model_ext):
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen_params:
                params.append(param)
                seen_params.add(id(param))
    is_kparam = is_k_parameterized(model_ext=model_ext, model_thin=model_thin)
    is_gammaparam = is_gamma_parameterized(model_ext=model_ext, model_thin=model_thin)
    is_conservative_thin = bool(getattr(model_thin, "conservative_thin_current", False))
    is_mixed_thin = bool(getattr(model_thin, "mixed_flux_thin", False))
    is_inventory_lift = bool(getattr(model_thin, "inventory_hermite_lift", False))
    if current_balance_weight is None:
        current_balance_weight = (
            500.0
            if is_conservative_thin and not is_mixed_thin and not is_inventory_lift
            else 0.0
        )
    current_balance_weight = max(0.0, float(current_balance_weight))
    current_balance_ramp_epochs = max(0, int(current_balance_ramp_epochs))
    current_balance_samples = max(32, int(current_balance_samples))
    original_requires_grad = {id(param): param.requires_grad for param in params}
    adapter_module = (
        getattr(model_thin, "k_adapter", None)
        if is_kparam else getattr(model_thin, "gamma_adapter", None)
    )
    adapter_param_ids = {
        id(param) for param in (adapter_module or nn.Identity()).parameters()
    }
    backbone_frozen = False
    if is_kparam and int(kparam_freeze_backbone_epochs) > 0:
        for param in params:
            param.requires_grad_(id(param) in adapter_param_ids)
        backbone_frozen = True
    if is_gammaparam and bool(gammaparam_freeze_backbone):
        for param in params:
            param.requires_grad_(id(param) in adapter_param_ids)
        backbone_frozen = True
    if is_kparam or is_gammaparam:
        adapter_params = [param for param in params if id(param) in adapter_param_ids]
        backbone_params = [param for param in params if id(param) not in adapter_param_ids]
        if not adapter_params:
            raise RuntimeError("Parameterized architecture is missing its conditional adapter")
        optimizer = torch.optim.AdamW(
            [
                {"params": adapter_params, "lr": learning_rate},
                {"params": backbone_params, "lr": 0.1 * learning_rate},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-7)

    is_hermite = isinstance(model_thin, ThinLayerNet_v9_6_MultiscaleHermite)
    is_his = isinstance(model_thin, ThinLayerNet_v9_6_HISPrototype)
    is_ext_his = isinstance(model_ext, ExternalNet_v9_6_MultiscaleHISLayer)
    is_green = isinstance(model_ext, (ExternalNet_v9_6_MultiscaleGreenKernel, ExternalNet_v9_6_MultiscaleGreenGrid))
    is_buffer = isinstance(model_ext, ExternalNet_v9_6_MultiscaleBuffer)
    is_flux_state = hasattr(model_ext, "flux_d")
    interface_state = getattr(model_ext, "interface_state", None)
    is_extbasis = isinstance(model_ext, (
        ExternalNet_v9_6_MultiscaleHermiteExtBasis,
        ExternalNet_v9_6_MultiscaleGreenKernel,
        ExternalNet_v9_6_MultiscaleGreenGrid,
        ExternalNet_v9_6_MultiscaleBuffer,
        ExternalNet_v9_6_MultiscaleFluxBuffer,
        ExternalNet_v9_6_MultiscaleHISLayer,
    ))
    is_hardbc = isinstance(model_thin, (
        ThinLayerNet_v9_6_MultiscaleHardBC,
        ThinLayerNet_v9_6_MultiscaleHermite,
        ThinLayerNet_v9_6_HISPrototype,
    ))
    if thin_interface_weight is None:
        thin_interface_weight = 300.0 if (is_hermite or is_his) else (800.0 if is_hardbc else 200.0)
    if bounds_weight is None:
        bounds_weight = 30.0 if (is_extbasis or is_his or is_ext_his) else 0.0

    loss_history = {
        'total': [], 'pde_thin': [], 'pde_ext': [], 'surface': [],
        'farfield': [], 'initial': [], 'bounds': [], 'interface': [],
        'interface_thin': [], 'interface_ext': [], 'reversal_continuity': [],
        'dynamic_reversal_jump': [], 'direct_reversal_jump': [],
        'direct_temporal_smooth': [], 'static_temporal_smooth': [],
        'phase_temporal_smooth': [], 'cint_reversal_jump': [],
        'cint_temporal_smooth': [], 'lr': [], 'nernst_err': [],
        'surface_state': [], 'physics_score': [],
        'physics_validation_score': [], 'physics_validation_ema': [],
        'physics_validation_epoch': [], 'k_cat': [], 'gamma': [],
        'current_balance': [], 'current_balance_weight': [],
        'thin_conservation': [], 'thin_constitutive': []
    }

    best_score = float('inf')
    best_epoch = 0
    no_improve_checks = 0
    validation_ema = None
    best_val_results = None

    early_stop_check_every = max(1, int(early_stop_check_every))
    early_stop_patience_checks = max(1, int(early_stop_patience_checks))
    early_stop_ema_alpha = float(early_stop_ema_alpha)
    if not 0.0 < early_stop_ema_alpha <= 1.0:
        raise ValueError("early_stop_ema_alpha must be in (0, 1]")
    early_stop_min_relative_improvement = max(
        0.0, float(early_stop_min_relative_improvement)
    )
    early_stop_validation_points = max(32, int(early_stop_validation_points))
    kparam_anchor_epochs = max(0, int(kparam_anchor_epochs))
    kparam_freeze_backbone_epochs = max(0, int(kparam_freeze_backbone_epochs))
    kparam_anchor_probability = float(np.clip(kparam_anchor_probability, 0.5, 1.0))
    kparam_log10_std = float(kparam_log10_std)
    if not np.isfinite(kparam_log10_std) or kparam_log10_std <= 0.0:
        raise ValueError("kparam_log10_std must be finite and positive")
    if is_kparam:
        model_ext.k_sampling_metadata = {
            "k_anchor_values": list(KPARAM_ANCHORS),
            "k_anchor_probabilities": [0.25, 0.50, 0.25],
            "k_anchor_epochs": kparam_anchor_epochs,
            "k_anchor_probability_after_stage1": kparam_anchor_probability,
            "k_log10_normal_std": kparam_log10_std,
            "k_validation_values": list(KPARAM_VALIDATION_VALUES),
            "k_freeze_backbone_epochs": kparam_freeze_backbone_epochs,
        }
    if is_gammaparam:
        gammaparam_anchor_probability = float(np.clip(
            gammaparam_anchor_probability, 0.0, 1.0
        ))
        model_ext.gamma_sampling_metadata = {
            "gamma_anchor_values": list(GAMMAPARAM_ANCHORS),
            "gamma_anchor_probability": gammaparam_anchor_probability,
            "gamma_continuous_distribution": "log10_uniform[-1,2]",
            "gamma_validation_values": list(GAMMAPARAM_VALIDATION_VALUES),
            "gamma_backbone_frozen": bool(gammaparam_freeze_backbone),
        }
    if int(early_stop_min_epochs) < 0:
        if getattr(model_ext, "film_trace_clean_external", False):
            early_stop_min_epochs = max(
                500,
                int(clean_residual_decay_epochs) + early_stop_check_every,
            )
        elif getattr(model_ext, "stage1_clean_warmstart", False):
            early_stop_min_epochs = 1500
        else:
            early_stop_min_epochs = 1000
    early_stop_min_epochs = max(0, int(early_stop_min_epochs))

    resume_paths = []
    if resume_checkpoint:
        resume_paths.append(resume_checkpoint)
    elif resume_best:
        resume_paths.append(MODEL_V96_BEST_PATH)
    elif resume:
        resume_paths.extend([MODEL_V96_PATH, MODEL_V96_BEST_PATH])

    if resume_paths:
        loaded_path = None
        for candidate_path in resume_paths:
            if os.path.exists(candidate_path):
                loaded_path = candidate_path
                break
        if loaded_path is None:
            raise FileNotFoundError(
                "No resume checkpoint found. Tried: " + ", ".join(resume_paths)
            )

        start_epoch, loaded_history, loaded_best = load_model_v96(
            model_thin, model_ext, optimizer, scheduler, loaded_path,
            load_optimizer_state=not reset_optimizer_state,
        )
        if loaded_history:
            loss_history.update(normalize_loss_history(loaded_history))
            validation_history = loss_history.get('physics_validation_ema', [])
            if validation_history:
                validation_ema = float(validation_history[-1])
        if reset_best_score:
            best_score = float('inf')
            best_epoch = start_epoch
            validation_ema = None
            print("Best score reset because the active loss weights may differ from the checkpoint.")
        elif validation_ema is not None and loaded_best is not None:
            best_score = loaded_best
            best_epoch = start_epoch
        else:
            best_score = float('inf')
            print("Best score reset because the checkpoint predates fixed physics validation.")
        print(f"Resumed from {loaded_path} at epoch {start_epoch}")

    print(f"\n{'='*80}")
    print(f"Starting v9.6 training from epoch {start_epoch}")
    print(f"Training for {n_epochs} additional epochs")
    print(f"Current checkpoint: {MODEL_V96_PATH}")
    print(f"Best checkpoint:    {MODEL_V96_BEST_PATH}")
    print(f"Save every:         {save_every} epochs")
    print(
        "Physics early stop: "
        f"enabled={early_stop}, check_every={early_stop_check_every}, "
        f"min_epochs={early_stop_min_epochs}, "
        f"patience_checks={early_stop_patience_checks}, "
        f"min_relative_improvement={early_stop_min_relative_improvement:.4g}, "
        f"ema_alpha={early_stop_ema_alpha:.3f}, "
        f"validation_points={early_stop_validation_points}"
    )
    print(f"Key changes:")
    print(f"  1. Interface weight: 50 → 200 (4x increase)")
    print(f"  2. Interface sampling: focused T windows plus higher hardbc point count")
    print(f"  3. Farfield sampling: n_points/10 → n_points/5")
    print(f"  4. Flux sign: UNCHANGED (analysis shows equivalent to FDM)")
    print(f"  5. Thin/external interface weights: {thin_interface_weight:.1f}/{ext_interface_weight:.1f}")
    print(f"  6. PDE weights thin/ext: {pde_thin_weight:.1f}/{pde_ext_weight:.1f}; bounds weight: {bounds_weight:.1f}")
    print(
        "  6b. Parameter-consistent residual scales: "
        f"external={external_residual_scale():.6g}, "
        f"flux={flux_residual_scale(k_cat_value=model_active_k_cat(model_ext=model_ext)):.6g}, "
        f"J_ref={characteristic_reaction_flux(k_cat_value=model_active_k_cat(model_ext=model_ext)):.6g}"
    )
    if is_kparam:
        print(
            "  6c. k-parameterization: "
            f"support={KPARAM_SUPPORT}, anchors={KPARAM_ANCHORS}, "
            f"anchor_epochs={kparam_anchor_epochs}, anchor_probability={kparam_anchor_probability:.2f}, "
            f"log10_normal_std={kparam_log10_std:.3g}, validation={KPARAM_VALIDATION_VALUES}, "
            f"freeze_backbone_epochs={kparam_freeze_backbone_epochs}"
        )
    if fdm_compare_pkl and fdm_compare_every > 0:
        print(f"  7. FDM posterior compare every {fdm_compare_every} epochs: {fdm_compare_pkl}")
    if isinstance(model_ext, ExternalNet_v9_6_MultiscaleGreenGrid):
        print(
            "  8. Green-grid history: "
            f"M={model_ext.time_grid_points}, K={model_ext.kernel_points}, "
            f"history_grad={model_ext.history_grad}, step_cache={model_ext.cache_history}"
        )
        if getattr(model_ext, "hybrid_lite", False):
            print(
                "  9. Hybrid-lite residual: "
                f"modes={model_ext.correction_scales.numel()}, "
                f"scales={model_ext.correction_scales.detach().cpu().numpy().reshape(-1).tolist()}"
            )
        if getattr(model_ext, "stage1_clean_warmstart", False):
            print(
                "  9. Stage1 fixed warm-start: "
                "signed full-field Green + 3 smooth residual modes + "
                "3 dynamic physical-input modes; no Film-Abel, KernelMix, "
                "matched Abel, interface residual, or hybrid extra modes."
            )
        if getattr(model_ext, "dynamic_lite", False):
            print(
                " 10. Dynamic correction inputs: "
                "x,t,theta,dtheta_dt,J_rxn,dJ_rxn_dt,Q_rxn; "
                f"scales={model_ext.dynamic_correction_scales.detach().cpu().numpy().reshape(-1).tolist()}"
            )
            if getattr(getattr(model_ext, "interface_state", None), "causal_dynamic_correction", False):
                interface_state = model_ext.interface_state
                print(
                    " 10b. Diffusion-causal dynamic gate: "
                    "g=exp(-y^2/(4*D*t*alpha)); "
                    f"alpha={float(interface_state.causal_gate_alpha):.3f}, "
                    f"t_floor={float(interface_state.causal_gate_time_floor):.4f}, "
                    f"dyn_jump_w={float(interface_state.causal_jump_weight):.1f}"
                )
            if getattr(model_ext, "causal_convolution_dynamic", False):
                source_scale = float(getattr(model_ext.interface_state, "causal_source_scale", 0.75))
                print(
                    " 10c. Causal-convolution dynamic correction: "
                    "S_corr(t) -> int S_corr(tau)*K(y,t-tau) dtau; "
                    f"source_scale={source_scale:.3f}"
                )
            if getattr(model_ext, "causal_hybrid_dynamic", False):
                direct_scale = float(getattr(model_ext.interface_state, "causal_direct_scale", 0.25))
                print(
                    " 10d. Causal-hybrid dynamic correction: "
                    "causal convolution plus gated direct residual; "
                    f"direct_scale={direct_scale:.3f}"
                )
            if getattr(model_ext, "causal_hybrid_smooth_dynamic", False):
                interface_state = model_ext.interface_state
                print(
                    " 10e. Smooth causal-hybrid regularization: "
                    f"direct_jump_w={float(interface_state.direct_jump_weight):.1f}, "
                    f"direct_smooth_w={float(interface_state.direct_smooth_weight):.1f}, "
                    f"static_smooth_w={float(interface_state.static_smooth_weight):.1f}, "
                    f"phase_smooth_w={float(interface_state.phase_smooth_weight):.1f}, "
                    f"window={float(interface_state.smooth_time_window):.4f}"
                )
        if getattr(model_ext, "memory_multibasis", False):
            print(
                " 11. Memory/local-basis correction: "
                f"lambdas={model_ext.memory_lambdas.detach().cpu().numpy().reshape(-1).tolist()}, "
                f"local_centers={model_ext.local_basis_centers.detach().cpu().numpy().reshape(-1).tolist()}, "
                f"green_extra_D={model_ext.green_extra_diffusion_scales.detach().cpu().numpy().reshape(-1).tolist()}"
            )
        interface_state = getattr(model_ext, "interface_state", None)
        if getattr(interface_state, "interface_memory", False):
            print(
                " 12. Interface-state memory: "
                f"lambdas={interface_state.interface_memory_lambdas.detach().cpu().numpy().reshape(-1).tolist()}, "
                f"corr_scales={interface_state.interface_corr_scales.detach().cpu().numpy().reshape(-1).tolist()}"
            )
        if (
            getattr(interface_state, "film_abel_interface", False)
            and not getattr(interface_state, "product_integral_interface", False)
        ):
            print(
                " 12. Film-Abel interface chain: "
                f"M={interface_state.time_grid_points}, K={interface_state.kernel_points}, "
                f"memory_lambdas={interface_state.film_abel_memory_lambdas.detach().cpu().numpy().reshape(-1).tolist()}, "
                "C_B_int=C_B_surface/(1+k*delta*C_C_int/D_B), "
                "C_D_int=Abel[J]-dt_phase*Abel[dJ]+small_residual"
            )
        if getattr(interface_state, "product_integral_interface", False):
            diagnostics = interface_state._last_product_integral_diagnostics
            diagnostic_text = "not evaluated"
            if diagnostics is not None:
                diagnostic_text = (
                    f"fixed_point_max={float(diagnostics['fixed_point_max'].detach().cpu()):.3e}, "
                    f"bounds_max={float(diagnostics['bounds_max'].detach().cpu()):.3e}"
                )
            print(
                " 12b. Product-integral interface: "
                "piecewise-linear singular Abel rule with two fixed-point updates "
                f"and {int(interface_state.newton_projection_iterations)} "
                "analytic Newton closure projection(s); "
                "finite-memory, phase-Abel, residual, and smooth bound disabled; "
                f"{diagnostic_text}"
            )
        if (
            getattr(interface_state, "film_abel_kernelmix_interface", False)
            and not getattr(interface_state, "product_integral_interface", False)
        ):
            beta = interface_state._kernelmix_beta().detach().cpu().numpy().reshape(-1).tolist()
            print(
                " 13. Film-Abel KernelMix prior: "
                "C_D_prior=alpha*Abel[J]+sum(beta_i*ExpMemory_i[J])-dt_phase(state)*Abel[dJ]; "
                f"alpha={float(interface_state._kernelmix_alpha().detach().cpu()):.4f}, beta={beta}"
            )
        if (
            getattr(interface_state, "interface_memory_v2", False)
            and not getattr(interface_state, "product_integral_interface", False)
        ):
            finite_gain = interface_state._intmemory_finite_gain().detach().cpu().numpy().reshape(-1).tolist()
            corr_amp = float(interface_state._intmemory_corr_amplitude().detach().cpu())
            bounded_map = (
                "smooth_softplus_ratio(C_D_prior)"
                if getattr(interface_state, "film_trace_clean_interface", False)
                else "clamp(C_D_prior+DeltaC_D)"
            )
            print(
                " 14. Interface-memory v2: "
                "C_D_prior += positive finite-memory boost; "
                f"C_D_int={bounded_map}; "
                f"finite_gain={finite_gain}, corr_amp={corr_amp:.4f}, "
                f"cint_jump_w={float(interface_state.cint_reversal_jump_weight):.1f}, "
                f"cint_smooth_w={float(interface_state.cint_temporal_smooth_weight):.1f}"
            )
        if getattr(model_ext, "fluxtrace_green", False):
            beta = float(model_ext.fluxtrace_value_beta().detach().cpu())
            scales = model_ext.fluxtrace_residual_scales.detach().cpu().numpy().reshape(-1).tolist()
            print(
                " 15. FluxTrace external field: "
                "C_D=G_flux[J]+h00*(D_int-G0)+h01*(D_far-Gfar)+R_smooth; "
                "external interface flux uses analytic Green trace, no h10 slope correction; "
                f"beta={beta:.3f}, trace_w={float(model_ext.fluxtrace_trace_weight):.1f}, "
                f"residual_scales={scales}"
            )
        if getattr(model_ext, "tracegreen_external", False):
            beta = float(model_ext.tracegreen_far_beta().detach().cpu())
            scales = model_ext.tracegreen_residual_scales.detach().cpu().numpy().reshape(-1).tolist()
            trace_name = (
                "ProductIntegral"
                if getattr(interface_state, "product_integral_interface", False)
                else "Film-Abel"
            )
            print(
                f" 16. {trace_name} TraceGreen external field: "
                "C_D=HeatDirichlet[C_D_int](x,t)+h01*(0-D_far)+R_smooth; "
                "single causal boundary trace, PDE loss acts only on R_smooth; "
                f"far_beta={beta:.3f}, residual_scales={scales}"
            )
        if getattr(model_ext, "film_trace_clean_external", False):
            if getattr(interface_state, "product_integral_interface", False):
                print(
                    " 16b. ProductIntegral-TraceGreen final model: fixed physical "
                    "interface trace plus erfc trace-preserving Dirichlet Green lift."
                )
            else:
                print(
                    " 16b. Clean Film-TraceGreen final model: "
                    "Film-Abel/KernelMix interface state plus erfc trace-preserving "
                    "Dirichlet Green lift; exploratory dynamic, matched-Abel, and "
                    "reversal-jump ablation penalties are disabled."
                )
        if is_conservative_thin:
            print(
                " 18. Thin inventory current: "
                "J_surface=-J_rxn-d/dt integral(C_B dx); "
                f"target_weight={current_balance_weight:.1f}, "
                f"ramp={current_balance_ramp_epochs}, samples={current_balance_samples}, quadrature=16"
            )
        if is_mixed_thin:
            print(
                " 19. Thin mixed flux: q_B(0)=-J_electrode, q_B(delta)=J_rxn; "
                "L_thin=||C_B,t+q_B,x||^2+2||q_B+D_B*C_B,x||^2."
            )
        if getattr(getattr(model_ext, "interface_state", None), "matched_abel_interface", False):
            print(
                " 17. Matched Abel interface memory: Neumann-to-trace "
                "1/sqrt(t-tau) singularity uses t-tau=t*u^2 quadrature and "
                "exact piecewise-constant history weights."
            )
        if getattr(getattr(model_ext, "interface_state", None), "mixed_abel_interface", False):
            lam = float(model_ext.interface_state._abel_mix_lambda().detach().cpu())
            print(
                " 17. Mixed Abel interface memory: "
                "A_mix=(1-lambda)*A_midpoint+lambda*A_matched; "
                f"lambda={lam:.6f}"
            )
    print(f"{'='*80}")

    train_wall_start = time.time()
    last_progress_wall = train_wall_start

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model_thin.train()
        model_ext.train()
        local_epoch = epoch - start_epoch
        kparam_schedule_epoch = epoch if is_kparam else local_epoch
        if is_kparam:
            if backbone_frozen and kparam_schedule_epoch >= kparam_freeze_backbone_epochs:
                for param in params:
                    param.requires_grad_(original_requires_grad[id(param)])
                backbone_frozen = False
                print(
                    f"[kparam] clean backbone unfrozen at parameterized epoch {kparam_schedule_epoch}",
                    flush=True,
                )
            step_k = sample_kparam_value(
                kparam_schedule_epoch,
                anchor_epochs=kparam_anchor_epochs,
                anchor_probability=kparam_anchor_probability,
                log10_std=kparam_log10_std,
            )
            set_model_k_cat(model_thin, model_ext, step_k)
        else:
            step_k = float(k_cat_star)
        if is_gammaparam:
            step_gamma = sample_gammaparam_value(gammaparam_anchor_probability)
            set_model_gamma(model_thin, model_ext, step_gamma)
            step_condition = step_gamma
        else:
            step_gamma = float(gamma)
            step_condition = step_k
        if hasattr(model_ext, "clear_step_cache"):
            model_ext.clear_step_cache()

        if hasattr(model_ext, "set_clean_residual_scale"):
            if clean_residual_decay_epochs and clean_residual_decay_epochs > 0:
                frac = min(max(local_epoch / float(clean_residual_decay_epochs), 0.0), 1.0)
                clean_scale = float(clean_residual_initial_scale) * (1.0 - frac)
            else:
                clean_scale = float(clean_residual_initial_scale)
            model_ext.set_clean_residual_scale(clean_scale)

        n_points = min(
            int(base_train_points) + (epoch - start_epoch) // 10 * int(train_point_growth),
            int(max_train_points),
        )
        n_surface = max(5000, n_points)
        if is_green or is_buffer or is_ext_his:
            n_interface = max(2500, n_points // 2)
        else:
            n_interface = max(12000, int(1.25 * n_points)) if is_hardbc else max(8000, n_points)
        n_farfield = n_points // 5  # v9.6: 增加远场采样

        # ========== 1. Interior region sampling ==========
        T_thin = torch.rand(n_points, 1, device=device) * T_sim
        X_thin = sample_thin_x(n_points, device)
        T_thin.requires_grad_(True)
        X_thin.requires_grad_(True)

        # ========== 2. External region sampling ==========
        T_ext = torch.rand(n_points, 1, device=device) * T_sim
        X_ext = sample_external_x(n_points, device)
        T_ext.requires_grad_(True)
        X_ext.requires_grad_(True)

        # ========== 3. Loss computation ==========

        # 3.1 Thin layer PDE
        inputs_thin = conditioned_model_inputs(model_thin, T_thin, X_thin, step_condition)
        C_A, C_B = model_thin(inputs_thin)

        C_B_T = torch.autograd.grad(C_B.sum(), T_thin, create_graph=True)[0]
        C_B_X = torch.autograd.grad(C_B.sum(), X_thin, create_graph=True)[0]
        if is_mixed_thin:
            q_B_thin = model_thin.flux_b(inputs_thin)
            q_B_X = torch.autograd.grad(q_B_thin.sum(), X_thin, create_graph=True)[0]
            loss_thin_conservation = torch.mean((C_B_T + q_B_X) ** 2)
            loss_thin_constitutive = torch.mean((q_B_thin + D_rel_B * C_B_X) ** 2)
            loss_pde_thin = loss_thin_conservation + 2.0 * loss_thin_constitutive
        else:
            C_A_T = torch.autograd.grad(C_A.sum(), T_thin, create_graph=True)[0]
            C_A_X = torch.autograd.grad(C_A.sum(), X_thin, create_graph=True)[0]
            C_A_XX = torch.autograd.grad(C_A_X.sum(), X_thin, create_graph=True)[0]
            C_B_XX = torch.autograd.grad(C_B_X.sum(), X_thin, create_graph=True)[0]
            loss_pde_thin = (
                torch.mean((C_A_T - D_rel_A * C_A_XX) ** 2) +
                torch.mean((C_B_T - D_rel_B * C_B_XX) ** 2)
            )
            loss_thin_conservation = torch.zeros((), device=device)
            loss_thin_constitutive = torch.zeros((), device=device)

        loss_current_balance = torch.zeros((), device=device)
        added_epoch = epoch - start_epoch + 1
        if current_balance_weight > 0.0:
            T_current = torch.rand(
                current_balance_samples, 1, device=device
            ) * T_sim
            T_current.requires_grad_(True)
            current = thin_current_components(
                model_thin,
                T_current,
                quadrature_points=16,
                create_graph=True,
            )
            j_ref = characteristic_reaction_flux(step_gamma, step_k)
            loss_current_balance = torch.mean(
                (current["balance_residual"] / max(j_ref, 1e-8)) ** 2
            )

        # 3.2 External PDE
        inputs_ext = conditioned_model_inputs(model_ext, T_ext, X_ext, step_condition)
        C_C, C_D = model_ext(inputs_ext)
        if hasattr(model_ext, "pde_fields"):
            C_C_pde, C_D_pde = model_ext.pde_fields(inputs_ext)
        else:
            C_C_pde, C_D_pde = C_C, C_D

        ext_scale = external_residual_scale(step_gamma)

        if is_flux_state:
            C_D_T = torch.autograd.grad(C_D_pde.sum(), T_ext, create_graph=True)[0]
            C_D_X = torch.autograd.grad(C_D_pde.sum(), X_ext, create_graph=True)[0]
            J_D_ext = model_ext.flux_d(inputs_ext)
            J_D_X = torch.autograd.grad(J_D_ext.sum(), X_ext, create_graph=True)[0]
            loss_ext_conservation = torch.mean((ext_scale * (C_D_T + J_D_X))**2)
            loss_ext_constitutive = torch.mean((ext_scale * (J_D_ext + D_rel_D * C_D_X))**2)
            loss_pde_ext = loss_ext_conservation + 2.0 * loss_ext_constitutive
        else:
            C_C_T = torch.autograd.grad(C_C_pde.sum(), T_ext, create_graph=True)[0]
            C_C_X = torch.autograd.grad(C_C_pde.sum(), X_ext, create_graph=True)[0]
            C_C_XX = torch.autograd.grad(C_C_X.sum(), X_ext, create_graph=True)[0]

            loss_pde_ext = torch.mean((ext_scale * (C_C_T - D_rel_C * C_C_XX))**2)

        # 3.3 Surface boundary (Nernst) - 与 v9.5 相同
        T_surf = torch.rand(n_surface, 1, device=device) * T_sim

        n_near = int(n_surface * 0.8)
        n_uniform = n_surface - n_near

        X_near = delta * 0.05 * torch.exp(-8.0 * torch.rand(n_near, 1, device=device))
        X_uniform = torch.rand(n_uniform, 1, device=device) * delta
        X_surf = torch.cat([X_near, X_uniform], dim=0)

        perm = torch.randperm(n_surface)
        X_surf = X_surf[perm]

        inputs_surf = conditioned_model_inputs(model_thin, T_surf, X_surf, step_condition)
        C_A_surf, C_B_surf = model_thin(inputs_surf)
        theta_surf = potential_theta(T_surf)

        nernst_residual = C_A_surf - C_B_surf * torch.exp(theta_surf)

        char_length = delta * 0.005
        dist_weight = torch.exp(-X_surf / char_length)

        dist_norm = torch.sum(dist_weight) + 1e-12
        loss_nernst_l2 = torch.sum(dist_weight * nernst_residual**2) / dist_norm
        loss_nernst_l1 = torch.sum(dist_weight * torch.abs(nernst_residual)) / dist_norm
        loss_nernst_max = torch.max(dist_weight * torch.abs(nernst_residual))

        T_exact = torch.rand(n_surface // 3, 1, device=device) * T_sim
        X_exact = torch.zeros_like(T_exact)
        inputs_exact = conditioned_model_inputs(model_thin, T_exact, X_exact, step_condition)
        C_A_exact, C_B_exact = model_thin(inputs_exact)
        theta_exact = potential_theta(T_exact)
        C_A_eq = torch.sigmoid(theta_exact)
        C_B_eq = torch.sigmoid(-theta_exact)
        loss_surface_state = torch.mean((C_A_exact - C_A_eq)**2 + (C_B_exact - C_B_eq)**2)

        strong_red = (theta_exact < -6).float()
        loss_red = torch.mean(strong_red * C_A_exact**2)

        strong_ox = (theta_exact > 6).float()
        loss_ox = torch.mean(strong_ox * C_B_exact**2)

        near_zero = (torch.abs(theta_exact) < 0.3).float()
        loss_zero = torch.mean(near_zero * (C_A_exact - 0.5)**2)

        loss_surface = (loss_nernst_l2 + 2.0 * loss_nernst_l1 + 0.1 * loss_nernst_max +
                        5.0 * loss_surface_state +
                        0.5 * (loss_red + loss_ox) + loss_zero)

        # 3.4 Far-field
        T_far = torch.rand(n_farfield, 1, device=device) * T_sim
        X_far = torch.ones_like(T_far) * X_ext_max
        inputs_far = conditioned_model_inputs(model_ext, T_far, X_far, step_condition)
        C_C_far, C_D_far = model_ext(inputs_far)

        loss_farfield = (
            torch.mean((ext_scale * (C_C_far - step_gamma))**2) +
            torch.mean((ext_scale * C_D_far)**2)
        )

        # 3.5 Initial condition
        T_ini = torch.zeros(n_points // 5, 1, device=device)
        X_ini_thin = torch.rand(n_points // 5, 1, device=device) * delta
        X_ini_ext = delta + torch.rand(n_points // 5, 1, device=device) * (X_ext_max - delta)

        C_A_ini, C_B_ini = model_thin(
            conditioned_model_inputs(model_thin, T_ini, X_ini_thin, step_condition)
        )
        C_C_ini, C_D_ini = model_ext(
            conditioned_model_inputs(model_ext, T_ini, X_ini_ext, step_condition)
        )

        T_ini_int = torch.zeros(n_points // 10, 1, device=device)
        X_ini_int = torch.ones_like(T_ini_int) * delta
        C_C_ini_int, C_D_ini_int = model_ext(
            conditioned_model_inputs(model_ext, T_ini_int, X_ini_int, step_condition)
        )

        loss_initial = (
            torch.mean((C_A_ini - 1.0)**2) +
            torch.mean(C_B_ini**2) +
            torch.mean((ext_scale * (C_C_ini - step_gamma))**2) +
            torch.mean((ext_scale * C_D_ini)**2) +
            2.0 * torch.mean((ext_scale * (C_C_ini_int - step_gamma))**2) +
            2.0 * torch.mean((ext_scale * C_D_ini_int)**2)
        )

        loss_bounds = concentration_bounds_loss(
            C_A, C_B, C_C, C_D, gamma_value=step_gamma
        )

        # 3.6 Interface coupling - v9.6: 保持 v9.5 符号不变
        # 经分析，v9.5 的符号与 FDM 等价，只是定义不同
        T_int = sample_interface_t(n_interface, device)

        X_int_thin = torch.ones_like(T_int) * delta
        X_int_thin.requires_grad_(True)
        inputs_int_thin = conditioned_model_inputs(model_thin, T_int, X_int_thin, step_condition)
        C_A_int, C_B_int = model_thin(inputs_int_thin)
        C_A_X_int = torch.autograd.grad(C_A_int.sum(), X_int_thin, create_graph=True, retain_graph=True)[0]
        C_B_X_int = torch.autograd.grad(C_B_int.sum(), X_int_thin, create_graph=True, retain_graph=True)[0]

        X_int_ext = torch.ones_like(T_int) * delta
        X_int_ext.requires_grad_(True)
        inputs_int_ext = conditioned_model_inputs(model_ext, T_int, X_int_ext, step_condition)
        C_C_int, C_D_int = model_ext(inputs_int_ext)
        C_C_X_int = torch.autograd.grad(C_C_int.sum(), X_int_ext, create_graph=True, retain_graph=True)[0]
        C_D_X_int = torch.autograd.grad(C_D_int.sum(), X_int_ext, create_graph=True, retain_graph=True)[0]

        active_k = model_active_k_cat(model_ext=model_ext, model_thin=model_thin)
        J_rxn = active_k * C_B_int * C_C_int
        flux_scale = flux_residual_scale(step_gamma, active_k)

        # 保持 v9.5 的通量符号（经分析正确）
        flux_A_res = flux_scale * (-D_rel_A * C_A_X_int + J_rxn)
        flux_B_res = flux_scale * (-D_rel_B * C_B_X_int - J_rxn)
        flux_C_res = flux_scale * (-D_rel_C * C_C_X_int + J_rxn)
        flux_D_res = flux_scale * (-D_rel_D * C_D_X_int - J_rxn)

        loss_interface_thin = torch.mean(flux_A_res**2) + torch.mean(flux_B_res**2)
        if getattr(model_ext, "tracegreen_external", False):
            loss_interface_ext = torch.mean(
                (ext_scale * (C_C_int + C_D_int - step_gamma)) ** 2
            )
        elif getattr(model_ext, "fluxtrace_analytic_boundary_flux", False):
            # The flux Green term satisfies the boundary flux in analytic trace
            # sense.  Autograd at y=0 evaluates the regular part of the kernel
            # and misses the singular t-tau -> 0 contribution, so using it here
            # would incorrectly force a Hermite slope correction back in.
            trace_loss = model_ext.fluxtrace_trace_consistency_loss(T_int)
            loss_interface_ext = (
                torch.mean((ext_scale * (C_C_int + C_D_int - step_gamma)) ** 2) +
                float(getattr(model_ext, "fluxtrace_trace_weight", 1.0)) * trace_loss
            )
        else:
            loss_interface_ext = torch.mean(flux_C_res**2) + torch.mean(flux_D_res**2)
        loss_interface = loss_interface_thin + loss_interface_ext

        loss_reversal_continuity = torch.zeros((), device=device)
        loss_dynamic_reversal_jump = torch.zeros((), device=device)
        loss_direct_reversal_jump = torch.zeros((), device=device)
        loss_direct_temporal_smooth = torch.zeros((), device=device)
        loss_static_temporal_smooth = torch.zeros((), device=device)
        loss_phase_temporal_smooth = torch.zeros((), device=device)
        loss_cint_reversal_jump = torch.zeros((), device=device)
        loss_cint_temporal_smooth = torch.zeros((), device=device)
        if getattr(interface_state, "continuous_time_features", False):
            n_continuity = min(1024, max(128, n_points // 4))
            X_cont = sample_external_x(n_continuity, device)
            eps_t = 0.0025 * T_sim
            T_minus = torch.ones_like(X_cont) * (T_switch * T_sim - eps_t)
            T_plus = torch.ones_like(X_cont) * (T_switch * T_sim + eps_t)
            C_C_minus, C_D_minus = model_ext(
                conditioned_model_inputs(model_ext, T_minus, X_cont, step_condition)
            )
            C_C_plus, C_D_plus = model_ext(
                conditioned_model_inputs(model_ext, T_plus, X_cont, step_condition)
            )
            loss_reversal_continuity = torch.mean(
                (C_C_plus - C_C_minus) ** 2 +
                (C_D_plus - C_D_minus) ** 2
            )
            if getattr(interface_state, "causal_dynamic_correction", False) and hasattr(model_ext, "dynamic_correction"):
                ext_length = X_ext_max - delta
                y_cont = torch.clamp(X_cont - delta, 0.0, ext_length)
                r_cont = torch.clamp(y_cont / ext_length, 0.0, 1.0)
                if model_ext.normalize_inputs:
                    x_minus = torch.cat([normalize_time(T_minus), normalize_ext_x(X_cont)], dim=1)
                    x_plus = torch.cat([normalize_time(T_plus), normalize_ext_x(X_cont)], dim=1)
                else:
                    x_minus = torch.cat([T_minus, X_cont], dim=1)
                    x_plus = torch.cat([T_plus, X_cont], dim=1)
                state_minus = interface_state(T_minus)
                state_plus = interface_state(T_plus)
                dyn_minus = model_ext.dynamic_correction(T_minus, X_cont, r_cont, x_minus, state_minus)
                dyn_plus = model_ext.dynamic_correction(T_plus, X_cont, r_cont, x_plus, state_plus)
                loss_dynamic_reversal_jump = torch.mean((dyn_plus - dyn_minus) ** 2)
                if getattr(interface_state, "causal_hybrid_smooth", False) and hasattr(model_ext, "direct_dynamic_correction"):
                    direct_minus = model_ext.direct_dynamic_correction(T_minus, X_cont, r_cont, x_minus, state_minus)
                    direct_plus = model_ext.direct_dynamic_correction(T_plus, X_cont, r_cont, x_plus, state_plus)
                    loss_direct_reversal_jump = torch.mean((direct_plus - direct_minus) ** 2)

                    def external_regularizer_terms(T_eval, X_eval):
                        y_eval = torch.clamp(X_eval - delta, 0.0, X_ext_max - delta)
                        r_eval = torch.clamp(y_eval / (X_ext_max - delta), 0.0, 1.0)
                        if model_ext.normalize_inputs:
                            t_net_eval = normalize_time(T_eval)
                            x_net_eval = torch.cat([t_net_eval, normalize_ext_x(X_eval)], dim=1)
                        else:
                            t_net_eval = T_eval
                            x_net_eval = torch.cat([T_eval, X_eval], dim=1)
                        beta_eval = 0.5 + 12.0 * torch.sigmoid(model_ext.beta_net(t_net_eval))
                        exp_neg_beta_eval = torch.exp(-beta_eval)
                        denom_eval = torch.clamp(1.0 - exp_neg_beta_eval, min=1e-5)
                        z_eval = torch.clamp(
                            (1.0 - torch.exp(-beta_eval * r_eval)) / denom_eval,
                            0.0,
                            1.0,
                        )
                        state_eval = interface_state(T_eval)
                        direct_eval = model_ext.direct_dynamic_correction(
                            T_eval, X_eval, r_eval, x_net_eval, state_eval,
                        )
                        static_eval = model_ext.residual_correction(T_eval, z_eval, x_net_eval)
                        return direct_eval, static_eval

                    n_smooth = min(512, max(128, n_points // 8))
                    X_smooth = sample_external_x(n_smooth, device)
                    smooth_window = float(getattr(interface_state, "smooth_time_window", 0.08 * T_sim))
                    dt_smooth = 0.0025 * T_sim
                    T_mid = (
                        T_switch * T_sim +
                        (2.0 * torch.rand_like(X_smooth) - 1.0) * smooth_window
                    )
                    T_mid = torch.clamp(T_mid, dt_smooth, T_sim - dt_smooth)
                    direct_prev, static_prev = external_regularizer_terms(T_mid - dt_smooth, X_smooth)
                    direct_mid, static_mid = external_regularizer_terms(T_mid, X_smooth)
                    direct_next, static_next = external_regularizer_terms(T_mid + dt_smooth, X_smooth)
                    loss_direct_temporal_smooth = torch.mean(
                        (direct_next - 2.0 * direct_mid + direct_prev) ** 2
                    )
                    loss_static_temporal_smooth = torch.mean(
                        (static_next - 2.0 * static_mid + static_prev) ** 2
                    )

                    if hasattr(interface_state, "_dynamic_phase_shift"):
                        n_phase = 96
                        T_phase = torch.linspace(
                            max(0.0, T_switch * T_sim - smooth_window),
                            min(float(T_sim), T_switch * T_sim + smooth_window),
                            n_phase,
                            device=device,
                        ).reshape(-1, 1)
                        history_phase = interface_state._history_grid(T_phase)
                        theta_phase, theta_dot_phase, _, _ = interface_state._surface_state(T_phase)
                        j_phase = interface_state._interp_multi_grid(T_phase, history_phase["J"])
                        d_j_phase = interface_state._interp_multi_grid(T_phase, history_phase["dJ"])
                        phase_values = interface_state._dynamic_phase_shift(
                            theta_phase, theta_dot_phase, j_phase, d_j_phase,
                        )
                        loss_phase_temporal_smooth = torch.mean(
                            (phase_values[2:] - 2.0 * phase_values[1:-1] + phase_values[:-2]) ** 2
                        )

            if getattr(interface_state, "cint_reversal_jump_weight", 0.0) > 0:
                c_int_minus = interface_state(T_minus)["C_C_int"]
                c_int_plus = interface_state(T_plus)["C_C_int"]
                loss_cint_reversal_jump = torch.mean((c_int_plus - c_int_minus) ** 2)

            if getattr(interface_state, "cint_temporal_smooth_weight", 0.0) > 0:
                smooth_window = float(getattr(interface_state, "smooth_time_window", 0.08 * T_sim))
                n_cint = 128
                T_cint = torch.linspace(
                    max(0.0, T_switch * T_sim - smooth_window),
                    min(float(T_sim), T_switch * T_sim + smooth_window),
                    n_cint,
                    device=device,
                ).reshape(-1, 1)
                c_int_values = interface_state(T_cint)["C_C_int"]
                loss_cint_temporal_smooth = torch.mean(
                    (c_int_values[2:] - 2.0 * c_int_values[1:-1] + c_int_values[:-2]) ** 2
                )

        # ========== 4. Weight adjustment ==========
        if epoch < start_epoch + 3000:
            surface_weight = 20.0 * np.exp((epoch - start_epoch) / 3000)
        elif epoch < start_epoch + 10000:
            surface_weight = 60.0 * np.exp((epoch - start_epoch - 3000) / 2000)
        else:
            surface_weight = min(500.0, 150.0 * np.exp((epoch - start_epoch - 10000) / 1500))

        if current_balance_ramp_epochs > 0:
            current_ramp = min(1.0, added_epoch / float(current_balance_ramp_epochs))
        else:
            current_ramp = 1.0
        active_current_balance_weight = current_balance_weight * current_ramp

        # v9.6: 大幅增加界面权重
        base_weights = {
            'pde_thin': pde_thin_weight,
            'pde_ext': pde_ext_weight,
            'surface': surface_weight,
            'farfield': farfield_weight,
            'initial': initial_weight,
            'bounds': bounds_weight,
            'interface_thin': thin_interface_weight,
            'interface_ext': ext_interface_weight,
            'reversal_continuity': 100.0,
            'dynamic_reversal_jump': float(getattr(interface_state, "causal_jump_weight", 0.0)),
            'direct_reversal_jump': float(getattr(interface_state, "direct_jump_weight", 0.0)),
            'direct_temporal_smooth': float(getattr(interface_state, "direct_smooth_weight", 0.0)),
            'static_temporal_smooth': float(getattr(interface_state, "static_smooth_weight", 0.0)),
            'phase_temporal_smooth': float(getattr(interface_state, "phase_smooth_weight", 0.0)),
            'cint_reversal_jump': float(getattr(interface_state, "cint_reversal_jump_weight", 0.0)),
            'cint_temporal_smooth': float(getattr(interface_state, "cint_temporal_smooth_weight", 0.0)),
            'current_balance': active_current_balance_weight,
        }

        # 3.7 Total loss
        total_loss = (
            base_weights['pde_thin'] * loss_pde_thin +
            base_weights['pde_ext'] * loss_pde_ext +
            base_weights['surface'] * loss_surface +
            base_weights['farfield'] * loss_farfield +
            base_weights['initial'] * loss_initial +
            base_weights['bounds'] * loss_bounds +
            base_weights['interface_thin'] * loss_interface_thin +
            base_weights['interface_ext'] * loss_interface_ext +
            base_weights['reversal_continuity'] * loss_reversal_continuity +
            base_weights['dynamic_reversal_jump'] * loss_dynamic_reversal_jump +
            base_weights['direct_reversal_jump'] * loss_direct_reversal_jump +
            base_weights['direct_temporal_smooth'] * loss_direct_temporal_smooth +
            base_weights['static_temporal_smooth'] * loss_static_temporal_smooth +
            base_weights['phase_temporal_smooth'] * loss_phase_temporal_smooth +
            base_weights['cint_reversal_jump'] * loss_cint_reversal_jump +
            base_weights['cint_temporal_smooth'] * loss_cint_temporal_smooth +
            base_weights['current_balance'] * loss_current_balance
        )
        physics_score = (
            pde_thin_weight * loss_pde_thin +
            pde_ext_weight * loss_pde_ext +
            100.0 * loss_surface +
            farfield_weight * loss_farfield +
            initial_weight * loss_initial +
            bounds_weight * loss_bounds +
            thin_interface_weight * loss_interface_thin +
            ext_interface_weight * loss_interface_ext +
            base_weights['reversal_continuity'] * loss_reversal_continuity +
            base_weights['dynamic_reversal_jump'] * loss_dynamic_reversal_jump +
            base_weights['direct_reversal_jump'] * loss_direct_reversal_jump +
            base_weights['direct_temporal_smooth'] * loss_direct_temporal_smooth +
            base_weights['static_temporal_smooth'] * loss_static_temporal_smooth +
            base_weights['phase_temporal_smooth'] * loss_phase_temporal_smooth +
            base_weights['cint_reversal_jump'] * loss_cint_reversal_jump +
            base_weights['cint_temporal_smooth'] * loss_cint_temporal_smooth +
            current_balance_weight * loss_current_balance
        )

        # ========== 5. Optimization ==========
        if not torch.isfinite(total_loss):
            print(f"Non-finite total loss at epoch {epoch}: {total_loss.item()}")
            if abort_on_nan:
                raise FloatingPointError(f"Non-finite total loss at epoch {epoch}")
            if hasattr(model_ext, "clear_step_cache"):
                model_ext.clear_step_cache()
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"Non-finite gradient norm at epoch {epoch}: {grad_norm.item()}")
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model_ext, "clear_step_cache"):
                model_ext.clear_step_cache()
            if abort_on_nan:
                raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}")
            continue

        optimizer.step()
        scheduler.step()
        if hasattr(model_thin, "clear_lift_cache"):
            model_thin.clear_lift_cache()
        if hasattr(model_ext, "clear_step_cache"):
            model_ext.clear_step_cache()
        completed_step = epoch - start_epoch + 1
        if (
            empty_cache_every > 0 and
            completed_step % int(empty_cache_every) == 0 and
            torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        # ========== 6. Record ==========
        nernst_err = torch.sqrt(loss_nernst_l2).item()
        loss_history['total'].append(total_loss.item())
        loss_history['pde_thin'].append(loss_pde_thin.item())
        loss_history['pde_ext'].append(loss_pde_ext.item())
        loss_history['surface'].append(loss_surface.item())
        loss_history['farfield'].append(loss_farfield.item())
        loss_history['initial'].append(loss_initial.item())
        loss_history.setdefault('bounds', []).append(loss_bounds.item())
        loss_history['interface'].append(loss_interface.item())
        loss_history.setdefault('interface_thin', []).append(loss_interface_thin.item())
        loss_history.setdefault('interface_ext', []).append(loss_interface_ext.item())
        loss_history.setdefault('reversal_continuity', []).append(loss_reversal_continuity.item())
        loss_history.setdefault('dynamic_reversal_jump', []).append(loss_dynamic_reversal_jump.item())
        loss_history.setdefault('direct_reversal_jump', []).append(loss_direct_reversal_jump.item())
        loss_history.setdefault('direct_temporal_smooth', []).append(loss_direct_temporal_smooth.item())
        loss_history.setdefault('static_temporal_smooth', []).append(loss_static_temporal_smooth.item())
        loss_history.setdefault('phase_temporal_smooth', []).append(loss_phase_temporal_smooth.item())
        loss_history.setdefault('cint_reversal_jump', []).append(loss_cint_reversal_jump.item())
        loss_history.setdefault('cint_temporal_smooth', []).append(loss_cint_temporal_smooth.item())
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_history['nernst_err'].append(nernst_err)
        loss_history['surface_state'].append(loss_surface_state.item())
        loss_history.setdefault('physics_score', []).append(physics_score.item())
        loss_history.setdefault('k_cat', []).append(active_k)
        loss_history.setdefault('gamma', []).append(step_gamma)
        loss_history.setdefault('current_balance', []).append(loss_current_balance.item())
        loss_history.setdefault('current_balance_weight', []).append(active_current_balance_weight)
        loss_history.setdefault('thin_conservation', []).append(loss_thin_conservation.item())
        loss_history.setdefault('thin_constitutive', []).append(loss_thin_constitutive.item())
        interface_state_for_history = getattr(model_ext, "interface_state", None)
        if getattr(interface_state_for_history, "mixed_abel_interface", False):
            loss_history.setdefault('abel_mix_lambda', []).append(
                float(interface_state_for_history._abel_mix_lambda().detach().cpu())
            )

        # Deterministic physics-only validation.  FDM is deliberately excluded
        # from checkpoint selection and stopping to avoid posterior data leakage.
        if completed_step % early_stop_check_every == 0:
            validation_fn = (
                gamma_parameterized_physics_validation_score
                if is_gammaparam else (
                    parameterized_physics_validation_score
                    if is_kparam else fixed_physics_validation_score
                )
            )
            validation = validation_fn(
                model_thin,
                model_ext,
                device,
                pde_thin_weight=pde_thin_weight,
                pde_ext_weight=pde_ext_weight,
                farfield_weight=farfield_weight,
                initial_weight=initial_weight,
                bounds_weight=bounds_weight,
                thin_interface_weight=thin_interface_weight,
                ext_interface_weight=ext_interface_weight,
                n_points=early_stop_validation_points,
                current_balance_weight=current_balance_weight,
            )
            raw_validation_score = validation["score"]
            if validation_ema is None:
                validation_ema = raw_validation_score
            else:
                validation_ema = (
                    early_stop_ema_alpha * raw_validation_score +
                    (1.0 - early_stop_ema_alpha) * validation_ema
                )

            loss_history.setdefault('physics_validation_score', []).append(raw_validation_score)
            loss_history.setdefault('physics_validation_ema', []).append(validation_ema)
            loss_history.setdefault('physics_validation_epoch', []).append(epoch + 1)
            if is_kparam:
                loss_history.setdefault('physics_validation_per_k', []).append(
                    {key: value['score'] for key, value in validation['per_k'].items()}
                )
            if is_gammaparam:
                loss_history.setdefault('physics_validation_per_gamma', []).append(
                    {key: value['score'] for key, value in validation['per_gamma'].items()}
                )

            required_score = best_score * (1.0 - early_stop_min_relative_improvement)
            improved = not np.isfinite(best_score) or validation_ema < required_score
            if improved:
                best_score = validation_ema
                best_epoch = epoch + 1
                no_improve_checks = 0
                best_val_results = validation
                save_model_v96(
                    model_thin,
                    model_ext,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    MODEL_V96_BEST_PATH,
                    loss_history=loss_history,
                    best_val_loss=best_score,
                )
                status = "new best"
            elif completed_step >= early_stop_min_epochs:
                no_improve_checks += 1
                status = f"plateau {no_improve_checks}/{early_stop_patience_checks}"
            else:
                no_improve_checks = 0
                status = "warmup"

            print(
                "[physics-val] "
                f"epoch={epoch + 1} raw={raw_validation_score:.4e} "
                f"ema={validation_ema:.4e} best={best_score:.4e} "
                f"status={status} | "
                f"pde=({validation['pde_thin']:.2e},{validation['pde_ext']:.2e}) "
                f"iface=({validation['interface_thin']:.2e},{validation['interface_ext']:.2e}) "
                f"current={validation['current_balance']:.2e} "
                f"cv_self_rmse={validation['current_balance_rmse']:.3e} "
                f"mixed=({validation['thin_conservation']:.2e},{validation['thin_constitutive']:.2e})",
                flush=True,
            )

            if (
                early_stop and
                completed_step >= early_stop_min_epochs and
                no_improve_checks >= early_stop_patience_checks
            ):
                print(f"\n{'='*60}")
                print(f"Physics early stopping at epoch {epoch + 1}")
                print(f"Best validation EMA: {best_score:.4e} at epoch {best_epoch}")
                print("FDM was not used for checkpoint selection or stopping.")
                print(f"{'='*60}")
                break

        step_count = epoch - start_epoch + 1
        if progress_every > 0 and step_count % progress_every == 0:
            now = time.time()
            sec_per_epoch = (now - train_wall_start) / max(step_count, 1)
            eta_sec = sec_per_epoch * max(start_epoch + n_epochs - epoch - 1, 0)
            interval_sec = now - last_progress_wall
            last_progress_wall = now
            gpu_msg = ""
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
                peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                gpu_msg = f" | cuda={allocated_gb:.1f}/{reserved_gb:.1f} GB peak={peak_gb:.1f} GB"
            interface_state = getattr(model_ext, "interface_state", None)
            phase_msg = ""
            if (
                getattr(interface_state, "film_abel_interface", False)
                and not getattr(interface_state, "product_integral_interface", False)
            ):
                phase_msg = (
                    f" | dt_phase={float(interface_state._phase_shift().detach().cpu()):+.3e}"
                    f" | D_eff={float(interface_state._abel_diffusion().detach().cpu()):.3f}"
                    f" | abel_gain={float(interface_state._abel_gain().detach().cpu()):.3f}"
                )
            if getattr(interface_state, "product_integral_interface", False):
                diagnostics = interface_state._last_product_integral_diagnostics
                if diagnostics is not None:
                    phase_msg = (
                        f" | PI_closure={float(diagnostics['fixed_point_max'].detach().cpu()):.2e}"
                        f" | PI_bounds={float(diagnostics['bounds_max'].detach().cpu()):.2e}"
                    )
            if getattr(interface_state, "mixed_abel_interface", False):
                phase_msg += f" | abel_mix_lambda={float(interface_state._abel_mix_lambda().detach().cpu()):.6f}"
            if hasattr(model_ext, "clean_residual_scale"):
                phase_msg += f" | R_smooth_scale={float(model_ext.clean_residual_scale.detach().cpu()):.3f}"
            if is_kparam:
                phase_msg += (
                    f" | k={active_k:.6g}"
                    f" | J_ref={characteristic_reaction_flux(k_cat_value=active_k):.4g}"
                )
            if is_gammaparam:
                phase_msg += (
                    f" | gamma={step_gamma:.6g}"
                    f" | J_ref={characteristic_reaction_flux(step_gamma, active_k):.4g}"
                )
            print(
                f"[progress] epoch {epoch + 1}/{start_epoch + n_epochs} "
                f"({step_count}/{n_epochs}) | loss={total_loss.item():.3e} "
                f"pde=({loss_pde_thin.item():.2e},{loss_pde_ext.item():.2e}) "
                f"iface=({loss_interface_thin.item():.2e},{loss_interface_ext.item():.2e}) "
                f"rev={loss_reversal_continuity.item():.2e} "
                f"dynjump={loss_dynamic_reversal_jump.item():.2e} "
                f"direct=({loss_direct_reversal_jump.item():.2e},{loss_direct_temporal_smooth.item():.2e}) "
                f"static={loss_static_temporal_smooth.item():.2e} "
                f"phase={loss_phase_temporal_smooth.item():.2e} "
                f"cint=({loss_cint_reversal_jump.item():.2e},{loss_cint_temporal_smooth.item():.2e}) "
                f"current={loss_current_balance.item():.2e}@{active_current_balance_weight:.1f} "
                f"mixed=({loss_thin_conservation.item():.2e},{loss_thin_constitutive.item():.2e}) "
                f"| {sec_per_epoch:.2f}s/epoch | last {progress_every}={interval_sec:.1f}s "
                f"| ETA={eta_sec/60.0:.1f}min{gpu_msg}{phase_msg}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

        # ========== 7. Human-readable diagnostics ==========
        if completed_step % 500 == 0:
            if hasattr(model_ext, "clear_step_cache"):
                model_ext.clear_step_cache()
            val_results = validate_model(model_thin, model_ext, device, epoch + 1, verbose=True)

        # 进度输出
        if completed_step % 500 == 0:
            with torch.no_grad():
                T_test = torch.rand(100, 1, device=device) * T_sim
                X_test = torch.rand(100, 1, device=device) * delta
                C_A_test, C_B_test = model_thin(torch.cat([T_test, X_test], dim=1))
                err_AB = torch.abs(C_A_test + C_B_test - 1.0).max().item()

                X_test_ext = delta + torch.rand(100, 1, device=device) * (X_ext_max - delta)
                C_C_test, C_D_test = model_ext(torch.cat([T_test, X_test_ext], dim=1))
                err_CD = torch.abs(C_C_test + C_D_test - step_gamma).max().item()

                T_surf_test = torch.rand(50, 1, device=device) * T_sim
                X_surf_test = torch.zeros_like(T_surf_test)
                C_A_surf_test, C_B_surf_test = model_thin(torch.cat([T_surf_test, X_surf_test], dim=1))
                theta_surf_test = potential_theta(T_surf_test)
                nernst_err_test = torch.sqrt(torch.mean((C_A_surf_test - C_B_surf_test * torch.exp(theta_surf_test))**2)).item()

                # v9.6: 检查界面浓度
                T_int_test = torch.rand(50, 1, device=device) * T_sim
                X_int_test = torch.ones_like(T_int_test) * delta
                _, C_B_int_test = model_thin(torch.cat([T_int_test, X_int_test], dim=1))
                C_C_int_test, _ = model_ext(torch.cat([T_int_test, X_int_test], dim=1))
                active_k_test = model_active_k_cat(model_ext=model_ext, model_thin=model_thin)
                J_rxn_test = (active_k_test * C_B_int_test * C_C_int_test).mean().item()

            print(f"\nEpoch {epoch:5d} | Total: {total_loss.item():.4e}")
            print(f"  Physics score: {physics_score.item():.4e}")
            print(f"  PDE_thin: {loss_pde_thin.item():.4e} | PDE_ext: {loss_pde_ext.item():.4e}")
            if is_conservative_thin:
                print(
                    f"  Current balance: {loss_current_balance.item():.4e} "
                    f"(w={active_current_balance_weight:.1f})"
                )
            if is_mixed_thin:
                print(
                    f"  Thin mixed: conservation={loss_thin_conservation.item():.4e}, "
                    f"constitutive={loss_thin_constitutive.item():.4e}"
                )
            print(f"  Surface: {loss_surface.item():.4e} (w={base_weights['surface']:.2f})")
            print(f"  Surface state: {loss_surface_state.item():.4e}")
            print(f"  Bounds: {loss_bounds.item():.4e} (w={base_weights['bounds']:.1f})")
            print(
                f"  Interface: {loss_interface.item():.4e} "
                f"(thin={loss_interface_thin.item():.4e}, ext={loss_interface_ext.item():.4e}; "
                f"w={base_weights['interface_thin']:.1f}/{base_weights['interface_ext']:.1f})"
            )
            print(
                f"  Reversal continuity: {loss_reversal_continuity.item():.4e} "
                f"(w={base_weights['reversal_continuity']:.1f}); "
                f"dynamic jump: {loss_dynamic_reversal_jump.item():.4e} "
                f"(w={base_weights['dynamic_reversal_jump']:.1f})"
            )
            if (
                base_weights['direct_reversal_jump'] > 0 or
                base_weights['direct_temporal_smooth'] > 0 or
                base_weights['static_temporal_smooth'] > 0 or
                base_weights['phase_temporal_smooth'] > 0 or
                base_weights['cint_reversal_jump'] > 0 or
                base_weights['cint_temporal_smooth'] > 0
            ):
                print(
                    f"  Smooth diagnostics: direct_jump={loss_direct_reversal_jump.item():.4e} "
                    f"(w={base_weights['direct_reversal_jump']:.1f}); "
                    f"direct_smooth={loss_direct_temporal_smooth.item():.4e} "
                    f"(w={base_weights['direct_temporal_smooth']:.1f}); "
                    f"static_smooth={loss_static_temporal_smooth.item():.4e} "
                    f"(w={base_weights['static_temporal_smooth']:.1f}); "
                    f"phase_smooth={loss_phase_temporal_smooth.item():.4e} "
                    f"(w={base_weights['phase_temporal_smooth']:.1f}); "
                    f"cint_jump={loss_cint_reversal_jump.item():.4e} "
                    f"(w={base_weights['cint_reversal_jump']:.1f}); "
                    f"cint_smooth={loss_cint_temporal_smooth.item():.4e} "
                    f"(w={base_weights['cint_temporal_smooth']:.1f})"
                )
            print(f"  Hard: A+B={err_AB:.2e}, C+D={err_CD:.2e}")
            print(f"  Nernst={nernst_err_test:.2e} | J_rxn={J_rxn_test:.4e}")
            print(f"  C_B(δ)={C_B_int_test.mean().item():.4f}, C_C(δ)={C_C_int_test.mean().item():.4f}")
            print(f"  Best physics validation EMA: {best_score:.2e} @ {best_epoch}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            if hasattr(model_ext, "clean_residual_scale"):
                print(f"  R_smooth scaffold scale: {float(model_ext.clean_residual_scale.detach().cpu()):.4f}")
            interface_state = getattr(model_ext, "interface_state", None)
            if getattr(interface_state, "mixed_abel_interface", False):
                print(f"  Abel mix lambda: {float(interface_state._abel_mix_lambda().detach().cpu()):.6f}")

            if completed_step % 2000 == 0:
                phys_results = verify_interface_physics(model_thin, model_ext, device)
                print_physics_verification(phys_results)

            if fdm_compare_pkl and fdm_compare_every > 0 and completed_step % fdm_compare_every == 0:
                if hasattr(model_ext, "clear_step_cache"):
                    model_ext.clear_step_cache()
                run_fdm_posterior_compare(
                    model_thin, model_ext, epoch + 1, arch_name, fdm_compare_pkl,
                    output_dir=fdm_compare_dir,
                    n_time=fdm_compare_n_time,
                    n_x_in=fdm_compare_n_x_in,
                    n_x_out=fdm_compare_n_x_out,
                    batch_size=fdm_compare_batch_size,
                    save_figure=fdm_compare_save_figure,
                    save_fields=fdm_compare_save_fields,
                )
                if hasattr(model_ext, "clear_step_cache"):
                    model_ext.clear_step_cache()
                if empty_cache_every > 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_model_v96(model_thin, model_ext, optimizer, scheduler,
                         epoch + 1, MODEL_V96_PATH,
                         loss_history=loss_history, best_val_loss=best_score)

    # Very short runs may finish before the first scheduled validation check.
    # Still create a physically selected best checkpoint for downstream stages.
    if not np.isfinite(best_score):
        validation_fn = (
            gamma_parameterized_physics_validation_score
            if is_gammaparam else (
                parameterized_physics_validation_score
                if is_kparam else fixed_physics_validation_score
            )
        )
        validation = validation_fn(
            model_thin,
            model_ext,
            device,
            pde_thin_weight=pde_thin_weight,
            pde_ext_weight=pde_ext_weight,
            farfield_weight=farfield_weight,
            initial_weight=initial_weight,
            bounds_weight=bounds_weight,
            thin_interface_weight=thin_interface_weight,
            ext_interface_weight=ext_interface_weight,
            n_points=early_stop_validation_points,
            current_balance_weight=current_balance_weight,
        )
        best_score = validation["score"]
        best_epoch = epoch + 1
        best_val_results = validation
        loss_history.setdefault('physics_validation_score', []).append(best_score)
        loss_history.setdefault('physics_validation_ema', []).append(best_score)
        loss_history.setdefault('physics_validation_epoch', []).append(best_epoch)
        if is_kparam:
            loss_history.setdefault('physics_validation_per_k', []).append(
                {key: value['score'] for key, value in validation['per_k'].items()}
            )
        if is_gammaparam:
            loss_history.setdefault('physics_validation_per_gamma', []).append(
                {key: value['score'] for key, value in validation['per_gamma'].items()}
            )
        save_model_v96(
            model_thin,
            model_ext,
            optimizer,
            scheduler,
            epoch + 1,
            MODEL_V96_BEST_PATH,
            loss_history=loss_history,
            best_val_loss=best_score,
        )

    # Final save
    save_model_v96(model_thin, model_ext, optimizer, scheduler,
                 epoch + 1, MODEL_V96_PATH,
                 loss_history=loss_history, best_val_loss=best_score)

    # Final validation
    print(f"\n{'='*80}")
    print("Final Validation")
    final_val = validate_model(model_thin, model_ext, device, epoch + 1, verbose=True)

    print(f"\n{'='*80}")
    print("Final Physics Validation")
    final_phys = verify_interface_physics(model_thin, model_ext, device, n_test=100)
    print_physics_verification(final_phys)

    return loss_history, best_val_results


# ==================== 预测与可视化 ====================
def predict_and_visualize_v9_6(model_thin, model_ext, loss_history, gamma, n_cv=8000):
    device = next(model_thin.parameters()).device
    model_thin.eval()
    model_ext.eval()

    # CV curve
    T_grid = torch.linspace(0, T_sim, n_cv, device=device).reshape(-1, 1)
    T_surf = T_grid.clone()
    X_surf = torch.zeros_like(T_surf)
    X_surf.requires_grad_(True)

    C_A_surf, _ = model_thin(torch.cat([T_surf, X_surf], dim=1))
    C_A_X_surf = torch.autograd.grad(C_A_surf.sum(), X_surf, create_graph=False)[0]

    J_total = -C_A_X_surf.detach().cpu().numpy().flatten()
    T_np = T_surf.detach().cpu().numpy().flatten()
    theta_values = potential_theta(T_surf).detach().cpu().numpy().flatten()

    J_thin_pure = -sigma * delta / 4

    # Grid data
    X_in_grid = np.linspace(0, delta, 100)
    X_ext_grid = np.linspace(delta, X_ext_max, 100)
    T_plot = np.linspace(0, T_sim, 100)

    TT_in, XX_in = np.meshgrid(T_plot, X_in_grid)
    TT_ext, XX_ext = np.meshgrid(T_plot, X_ext_grid)

    with torch.no_grad():
        inputs_in = torch.tensor(np.stack([TT_in.flatten(), XX_in.flatten()], axis=1), 
                                dtype=torch.float32, device=device)
        C_A_grid, C_B_grid = model_thin(inputs_in)
        C_A_grid = C_A_grid.cpu().numpy().reshape(XX_in.shape)
        C_B_grid = C_B_grid.cpu().numpy().reshape(XX_in.shape)

        inputs_ext = torch.tensor(np.stack([TT_ext.flatten(), XX_ext.flatten()], axis=1), 
                                 dtype=torch.float32, device=device)
        C_C_grid, C_D_grid = model_ext(inputs_ext)
        C_C_grid = C_C_grid.cpu().numpy().reshape(XX_ext.shape)
        C_D_grid = C_D_grid.cpu().numpy().reshape(XX_ext.shape)

    phys_results = verify_interface_physics(model_thin, model_ext, device, n_test=100)

    # Visualization
    fig = plt.figure(figsize=(20, 16))

    # Row 1
    ax1 = plt.subplot(3, 4, 1)
    epochs = range(len(loss_history['total']))
    ax1.semilogy(epochs, loss_history['total'], 'k-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss (v9.6)')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 4, 2)
    ax2.semilogy(epochs, loss_history['pde_thin'], 'b-', alpha=0.7, label='PDE_thin')
    ax2.semilogy(epochs, loss_history['pde_ext'], 'r-', alpha=0.7, label='PDE_ext')
    ax2.semilogy(epochs, loss_history['surface'], 'g-', alpha=0.7, label='Surface')
    ax2.semilogy(epochs, loss_history['interface'], 'm-', alpha=0.7, label='Interface')
    if len(loss_history.get('interface_thin', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['interface_thin'], color='#7B3294', alpha=0.6, label='Interface thin')
    if len(loss_history.get('interface_ext', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['interface_ext'], color='#008837', alpha=0.6, label='Interface ext')
    if len(loss_history.get('reversal_continuity', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['reversal_continuity'], color='#E66101', alpha=0.6, label='Reversal cont.')
    if len(loss_history.get('dynamic_reversal_jump', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['dynamic_reversal_jump'], color='#5E3C99', alpha=0.6, label='Dynamic jump')
    if len(loss_history.get('direct_reversal_jump', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['direct_reversal_jump'], color='#1B9E77', alpha=0.6, label='Direct jump')
    if len(loss_history.get('direct_temporal_smooth', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['direct_temporal_smooth'], color='#D95F02', alpha=0.6, label='Direct smooth')
    if len(loss_history.get('static_temporal_smooth', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['static_temporal_smooth'], color='#7570B3', alpha=0.6, label='Static smooth')
    if len(loss_history.get('phase_temporal_smooth', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['phase_temporal_smooth'], color='#E7298A', alpha=0.6, label='Phase smooth')
    if len(loss_history.get('cint_reversal_jump', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['cint_reversal_jump'], color='#66A61E', alpha=0.6, label='Cint jump')
    if len(loss_history.get('cint_temporal_smooth', [])) == len(loss_history['total']):
        ax2.semilogy(epochs, loss_history['cint_temporal_smooth'], color='#A6761D', alpha=0.6, label='Cint smooth')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 4, 3)
    ax3.semilogy(epochs, loss_history['lr'], 'c-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(theta_values, J_total, 'b-', linewidth=3, label=f'Catalytic (γ={gamma})')
    ax4.axhline(J_thin_pure, color='r', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Pure thin-layer: {J_thin_pure:.3f}')

    peak_idx = np.argmin(J_total)
    ax4.scatter([theta_values[peak_idx]], [J_total[peak_idx]], 
                color='red', s=150, zorder=5, marker='*')
    ax4.annotate(f'Peak: {J_total[peak_idx]:.3f}\nθ={theta_values[peak_idx]:.2f}', 
                xy=(theta_values[peak_idx], J_total[peak_idx]),
                xytext=(theta_values[peak_idx]+3, J_total[peak_idx]+0.5))

    ax4.set_xlabel('Potential θ')
    ax4.set_ylabel('Flux J')
    ax4.set_title('Cyclic Voltammogram (v9.6)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 2: Concentration
    ax5 = plt.subplot(3, 4, 5)
    im5 = ax5.pcolormesh(T_plot, X_in_grid, C_A_grid, shading='auto', 
                         cmap='RdYlBu_r', vmin=0, vmax=1)
    ax5.set_xlabel('Time T')
    ax5.set_ylabel('X (thin layer)')
    ax5.set_title('C_A in Thin Layer')
    plt.colorbar(im5, ax=ax5)

    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.pcolormesh(T_plot, X_in_grid, C_B_grid, shading='auto', 
                         cmap='RdYlBu_r', vmin=0, vmax=1)
    ax6.set_xlabel('Time T')
    ax6.set_ylabel('X (thin layer)')
    ax6.set_title('C_B in Thin Layer')
    plt.colorbar(im6, ax=ax6)

    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.pcolormesh(T_plot, X_ext_grid, C_C_grid, shading='auto', 
                         cmap='viridis', vmin=0, vmax=gamma)
    ax7.axhline(delta, color='white', linestyle='--', alpha=0.7, linewidth=2)
    ax7.set_xlabel('Time T')
    ax7.set_ylabel('X (external)')
    ax7.set_title('C_C in External Region')
    plt.colorbar(im7, ax=ax7)

    ax8 = plt.subplot(3, 4, 8)
    im8 = ax8.pcolormesh(T_plot, X_ext_grid, C_D_grid, shading='auto', 
                         cmap='plasma', vmin=0, vmax=gamma)
    ax8.axhline(delta, color='white', linestyle='--', alpha=0.7, linewidth=2)
    ax8.set_xlabel('Time T')
    ax8.set_ylabel('X (external)')
    ax8.set_title('C_D in External Region')
    plt.colorbar(im8, ax=ax8)

    # Row 3: Validation
    ax9 = plt.subplot(3, 4, 9)
    with torch.no_grad():
        T_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device) * T_sim
        X_test = torch.linspace(0, delta, 50, device=device).reshape(-1, 1)
        for i, t in enumerate(T_test):
            T_grid = torch.ones_like(X_test) * t
            inputs = torch.cat([T_grid, X_test], dim=1)
            C_A_test, C_B_test = model_thin(inputs)
            sum_AB = C_A_test + C_B_test
            ax9.plot(X_test.cpu().numpy(), sum_AB.cpu().numpy(), 
                     linewidth=2, label=f'T={t/T_sim:.2f}T_sim')
        ax9.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.7)
        ax9.set_xlabel('X (thin layer)')
        ax9.set_ylabel('C_A + C_B')
        ax9.set_title('Thin Layer Conservation')
        ax9.legend(fontsize=7)
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim(0.99, 1.01)

    ax10 = plt.subplot(3, 4, 10)
    with torch.no_grad():
        T_test_ext = torch.tensor([0.0, 0.5, 1.0], device=device) * T_sim
        X_test_ext = torch.linspace(delta, X_ext_max, 50, device=device).reshape(-1, 1)
        for i, t in enumerate(T_test_ext):
            T_grid = torch.ones_like(X_test_ext) * t
            inputs = torch.cat([T_grid, X_test_ext], dim=1)
            C_C_test, C_D_test = model_ext(inputs)
            sum_CD = C_C_test + C_D_test
            ax10.plot(X_test_ext.cpu().numpy(), sum_CD.cpu().numpy()/gamma, 
                      linewidth=2, label=f'T={t/T_sim:.2f}T_sim')
        ax10.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.7)
        ax10.set_xlabel('X (external)')
        ax10.set_ylabel('(C_C + C_D) / γ')
        ax10.set_title('External Region Conservation')
        ax10.legend(fontsize=7)
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim(0.99, 1.01)

    ax11 = plt.subplot(3, 4, 11)
    T_phys = np.linspace(0.1, 0.9, 100) * T_sim
    ax11.plot(T_phys, phys_results['J_A'], 'b-', linewidth=2, label='J_A', alpha=0.7)
    ax11.plot(T_phys, phys_results['J_B'], 'r-', linewidth=2, label='J_B', alpha=0.7)
    ax11.plot(T_phys, phys_results['J_C'], 'g-', linewidth=2, label='J_C', alpha=0.7)
    ax11.plot(T_phys, phys_results['J_D'], 'm-', linewidth=2, label='J_D', alpha=0.7)
    ax11.plot(T_phys, phys_results['J_rxn'], 'k--', linewidth=3, label='J_rxn', alpha=0.9)
    ax11.plot(T_phys, -phys_results['J_rxn'], 'k:', linewidth=2, label='-J_rxn', alpha=0.5)
    ax11.set_xlabel('Time T')
    ax11.set_ylabel('Flux')
    ax11.set_title('Interface Fluxes (v9.6)', fontweight='bold')
    ax11.legend(fontsize=8, ncol=2)
    ax11.grid(True, alpha=0.3)
    ax11.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax12 = plt.subplot(3, 4, 12)
    peak_idx = np.argmin(J_total)
    J_peak = J_total[peak_idx]
    enhancement = J_peak / J_thin_pure if J_thin_pure != 0 else 0
    phys_err = (phys_results['err_A'] + phys_results['err_B'] + 
                phys_results['err_C'] + phys_results['err_D']) / 4

    final_nernst_err = loss_history['nernst_err'][-1] if 'nernst_err' in loss_history and len(loss_history['nernst_err']) > 0 else float('nan')

    results_text = f"v9.6 Interface Coupling Enhancement\n"
    results_text += f"="*40 + "\n"
    results_text += f"Peak current (catalytic): {J_peak:.4f}\n"
    results_text += f"Peak current (pure): {J_thin_pure:.4f}\n"
    results_text += f"Enhancement factor: {enhancement:.2f}x\n"
    results_text += f"Peak potential: {theta_values[peak_idx]:.2f}\n"
    results_text += f"\nPhysics validation:\n"
    results_text += f"  Interface flux: {phys_err:.2e}\n"
    results_text += f"  Nernst error: {final_nernst_err:.2e}\n"
    results_text += f"\nv9.6 Changes:\n"
    results_text += f"• Interface weight: 50→200\n"
    results_text += f"• Interface sampling: ↑\n"
    results_text += f"• Farfield sampling: ↑\n"
    results_text += f"• Flux sign: verified correct"

    ax12.text(0.05, 0.5, results_text, fontsize=9, 
              verticalalignment='center', transform=ax12.transAxes,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax12.axis('off')

    plt.tight_layout()
    plt.savefig('./thin_layer_catalytic_v9_6_results.png', 
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    plt.close('all')
    import gc
    gc.collect()

    print(f"\n{'='*80}")
    print(f"THIN-LAYER ELECTROCATALYTIC RESULTS (v9.6)")
    print(f"{'='*80}")
    print(f"Peak current (catalytic):     {J_peak:.6f}")
    print(f"Peak current (pure thin-layer): {J_thin_pure:.6f}")
    print(f"Catalytic enhancement factor:   {enhancement:.2f}x")
    print(f"Peak potential:                 {theta_values[peak_idx]:.4f}")
    print(f"Interface flux error:           {phys_err:.2e}")
    print(f"Final Nernst error:             {final_nernst_err:.2e}")
    print(f"{'='*80}")

    return {
        'theta': theta_values,
        'J': J_total,
        'T': T_np,
        'C_A_grid': C_A_grid,
        'C_B_grid': C_B_grid,
        'C_C_grid': C_C_grid,
        'C_D_grid': C_D_grid,
        'phys_results': phys_results,
        'results': {
            'J_peak': J_peak,
            'J_pure': J_thin_pure,
            'enhancement': enhancement,
            'peak_potential': theta_values[peak_idx],
            'phys_error': phys_err,
            'nernst_error': final_nernst_err
        }
    }


def save_cv_csv(results, path="./cv_theta_J_v9_6.csv"):
    df_cv = pd.DataFrame({
        "theta": results['theta'],
        "J": results['J']
    })
    df_cv.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CV theta-J data saved to: {path}")


def compare_with_fdm(results, fdm_csv_path="../FDM/kcat1_v42_cv_data_v42.csv"):
    if not os.path.exists(fdm_csv_path):
        print(f"FDM comparison skipped; file not found: {fdm_csv_path}")
        return None

    fdm = pd.read_csv(fdm_csv_path, encoding="utf-8-sig")
    fdm_theta = fdm["theta"].to_numpy(dtype=float)
    fdm_J = fdm["J"].to_numpy(dtype=float)
    pinn_theta = np.asarray(results["theta"], dtype=float)
    pinn_J = np.asarray(results["J"], dtype=float)

    if len(pinn_J) != len(fdm_J):
        src = np.linspace(0.0, 1.0, len(pinn_J))
        dst = np.linspace(0.0, 1.0, len(fdm_J))
        pinn_J_cmp = np.interp(dst, src, pinn_J)
        pinn_theta_cmp = np.interp(dst, src, pinn_theta)
    else:
        pinn_J_cmp = pinn_J
        pinn_theta_cmp = pinn_theta

    diff = pinn_J_cmp - fdm_J
    fdm_peak_idx = int(np.argmin(fdm_J))
    pinn_peak_idx = int(np.argmin(pinn_J_cmp))
    metrics = {
        "rmse_J": float(np.sqrt(np.mean(diff**2))),
        "mae_J": float(np.mean(np.abs(diff))),
        "max_abs_J": float(np.max(np.abs(diff))),
        "fdm_peak_J": float(fdm_J[fdm_peak_idx]),
        "fdm_peak_theta": float(fdm_theta[fdm_peak_idx]),
        "pinn_peak_J": float(pinn_J_cmp[pinn_peak_idx]),
        "pinn_peak_theta": float(pinn_theta_cmp[pinn_peak_idx]),
        "peak_J_error": float(pinn_J_cmp[pinn_peak_idx] - fdm_J[fdm_peak_idx]),
        "peak_theta_error": float(pinn_theta_cmp[pinn_peak_idx] - fdm_theta[fdm_peak_idx]),
    }

    print("\n" + "="*80)
    print("PINN vs FDM CV comparison")
    print("="*80)
    print(f"J RMSE:          {metrics['rmse_J']:.6e}")
    print(f"J MAE:           {metrics['mae_J']:.6e}")
    print(f"J max abs error: {metrics['max_abs_J']:.6e}")
    print(f"FDM peak:        {metrics['fdm_peak_J']:.6f} @ theta={metrics['fdm_peak_theta']:.4f}")
    print(f"PINN peak:       {metrics['pinn_peak_J']:.6f} @ theta={metrics['pinn_peak_theta']:.4f}")
    print(f"Peak J error:    {metrics['peak_J_error']:.6f}")
    print(f"Peak theta err:  {metrics['peak_theta_error']:.6f}")
    print("="*80)
    return metrics


def run_smoke_test_v9_6(model_thin, model_ext, device, arch_name="legacy"):
    """Tiny validation path for CI/Colab sanity checks.

    This is not a quality test and does not use FDM. It verifies tensor shapes,
    finite outputs, spatial gradients, and one optimizer step for both the
    baseline path and experimental architectures such as his_pinn.
    """
    model_thin.train()
    model_ext.train()

    params = []
    seen_params = set()
    for module in (model_thin, model_ext):
        for param in module.parameters():
            if id(param) not in seen_params:
                params.append(param)
                seen_params.add(id(param))
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)

    n = 8
    T = torch.linspace(0.05 * T_sim, 0.95 * T_sim, n, device=device).reshape(-1, 1)
    X_thin = torch.linspace(0.0, delta, n, device=device).reshape(-1, 1)
    X_ext = torch.linspace(delta, X_ext_max, n, device=device).reshape(-1, 1)
    T_thin = T.clone().detach().requires_grad_(True)
    X_thin = X_thin.clone().detach().requires_grad_(True)
    T_ext = T.clone().detach().requires_grad_(True)
    X_ext = X_ext.clone().detach().requires_grad_(True)

    C_A, C_B = model_thin(torch.cat([T_thin, X_thin], dim=1))
    C_C, C_D = model_ext(torch.cat([T_ext, X_ext], dim=1))

    for name, value in {
        "C_A": C_A,
        "C_B": C_B,
        "C_C": C_C,
        "C_D": C_D,
    }.items():
        if value.shape != (n, 1):
            raise RuntimeError(f"{arch_name} smoke test failed: {name} shape is {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{arch_name} smoke test failed: {name} contains NaN/Inf")

    C_B_X = torch.autograd.grad(C_B.sum(), X_thin, create_graph=True)[0]
    C_C_X = torch.autograd.grad(C_C.sum(), X_ext, create_graph=True)[0]
    if not torch.isfinite(C_B_X).all() or not torch.isfinite(C_C_X).all():
        raise RuntimeError(f"{arch_name} smoke test failed: spatial gradient contains NaN/Inf")

    loss = (
        torch.mean((C_A + C_B - 1.0) ** 2) +
        torch.mean((C_C + C_D - gamma) ** 2) +
        1e-4 * torch.mean(C_B_X ** 2) +
        1e-4 * torch.mean(C_C_X ** 2) +
        1e-6 * torch.mean(C_A ** 2 + C_B ** 2 + C_C ** 2 + C_D ** 2)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Smoke test passed for arch={arch_name}")
    print(f"  shapes: C_A/C_B/C_C/C_D = {tuple(C_A.shape)}")
    print("  finite: yes")
    print(f"  tiny loss after one backward path: {loss.item():.6e}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate PINN v9.6.")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--eval-only", action="store_true",
                        help="Load a checkpoint and export CV data without training.")
    parser.add_argument("--checkpoint", default=MODEL_V96_BEST_PATH,
                        help="Checkpoint used by --eval-only.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the current checkpoint, falling back to best.")
    parser.add_argument("--resume-best", action="store_true",
                        help="Resume training from the best checkpoint.")
    parser.add_argument("--resume-checkpoint", default=None,
                        help="Explicit checkpoint path to resume training from.")
    parser.add_argument("--warm-start-checkpoint", default=None,
                        help="Load model weights only; seeds a parameterized model from its fixed reference checkpoint.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for current/best checkpoints, e.g. a Google Drive folder on Colab.")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Save the current checkpoint every N epochs. Use 0 to disable periodic saves.")
    parser.add_argument("--legacy-inputs", action="store_true",
                        help="Use raw coordinates for old v9.6 checkpoints.")
    parser.add_argument("--cv-points", type=int, default=8000)
    parser.add_argument("--fdm-csv", default="",
                        help="Optional matching fixed-parameter FDM CV CSV.")
    parser.add_argument("--gamma", type=float, default=REFERENCE_GAMMA,
                        help="Fixed bulk C concentration ratio for this run.")
    parser.add_argument("--k-cat-star", type=float, default=REFERENCE_K_CAT_STAR,
                        help="Fixed catalytic reaction constant for this run.")
    parser.add_argument("--arch", choices=["legacy", "multiscale", "multiscale_hardbc", "his_pinn", "his_pinn_ext", "multiscale_hermite", "multiscale_hermite_extbasis", "multiscale_green", "multiscale_green_grid", "multiscale_green_grid_hybrid", "multiscale_green_grid_dynamic", "multiscale_green_grid_dynamic_stage1", "multiscale_green_grid_interface_memory", "multiscale_green_grid_memory", "multiscale_green_grid_film_abel", "multiscale_green_grid_film_abel_kernelmix", "multiscale_green_grid_film_abel_kernelmix_causal", "multiscale_green_grid_film_abel_kernelmix_causalconv", "multiscale_green_grid_film_abel_kernelmix_causalhybrid", "multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth", "multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory", "multiscale_green_grid_film_abel_kernelmix_fluxtrace", "multiscale_green_grid_film_abel_kernelmix_tracegreen", "multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel", "multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel", "multiscale_film_tracegreen_clean", "multiscale_film_tracegreen_productintegral", "multiscale_film_tracegreen_productintegral_lift", "multiscale_film_tracegreen_clean_conservative", "multiscale_film_tracegreen_clean_mixedflux", "multiscale_film_tracegreen_conservative_lift", "multiscale_film_tracegreen_kparam", "multiscale_film_tracegreen_kparam_lift", "multiscale_film_tracegreen_gammaparam", "multiscale_film_tracegreen_gammaparam_lift", "multiscale_green_grid_film_abel_ema", "multiscale_buffer"], default="legacy")
    parser.add_argument("--green-time-grid", type=int, default=256,
                        help="Global history time-grid size for multiscale_green_grid.")
    parser.add_argument("--green-kernel-points", type=int, default=32,
                        help="Green convolution quadrature points for multiscale_green_grid.")
    parser.add_argument("--lift-time-grid", type=int, default=1024,
                        help="Causal time-grid points for the inventory Hermite lift.")
    parser.add_argument("--green-detach-history", action="store_true",
                        help="Disable history-gradient backprop in multiscale_green_grid for a cheaper ablation.")
    parser.add_argument("--green-no-step-cache", action="store_true",
                        help="Disable per-iteration J_rxn(t_grid) graph cache in multiscale_green_grid.")
    parser.add_argument("--thin-interface-weight", type=float, default=None,
                        help="Weight for thin-layer interface flux residual. Default: 300 for hermite, 800 for hardbc, 200 otherwise.")
    parser.add_argument("--ext-interface-weight", type=float, default=200.0,
                        help="Weight for external-region interface flux residual.")
    parser.add_argument("--pde-thin-weight", type=float, default=10.0,
                        help="Weight for thin-layer PDE residual.")
    parser.add_argument("--pde-ext-weight", type=float, default=10.0,
                        help="Weight for external-region PDE residual.")
    parser.add_argument("--farfield-weight", type=float, default=1.0,
                        help="Weight for external farfield boundary residual.")
    parser.add_argument("--initial-weight", type=float, default=1.0,
                        help="Weight for initial-condition residual.")
    parser.add_argument("--bounds-weight", type=float, default=None,
                        help="Weight for concentration bounds. Default: 30 for multiscale_hermite_extbasis, 0 otherwise.")
    parser.add_argument("--fdm-compare-pkl", default="",
                        help="Optional FDM pkl for posterior concentration comparison during training. Evaluation only; not used in loss.")
    parser.add_argument("--fdm-compare-every", type=int, default=0,
                        help="Run compare_concentration_fields.py every N epochs during training. Use 0 to disable.")
    parser.add_argument("--fdm-compare-dir", default=None,
                        help="Directory for in-training FDM comparison JSON/temporary checkpoint.")
    parser.add_argument("--fdm-compare-n-time", type=int, default=160)
    parser.add_argument("--fdm-compare-n-x-in", type=int, default=120)
    parser.add_argument("--fdm-compare-n-x-out", type=int, default=160)
    parser.add_argument("--fdm-compare-batch-size", type=int, default=65536)
    parser.add_argument("--fdm-compare-save-figure", action="store_true",
                        help="Save residual summary PNG for each in-training FDM comparison.")
    parser.add_argument("--fdm-compare-save-fields", action="store_true",
                        help="Save residual field NPZ for each in-training FDM comparison.")
    parser.add_argument("--base-train-points", type=int, default=8000,
                        help="Initial interior collocation point count per epoch.")
    parser.add_argument("--max-train-points", type=int, default=15000,
                        help="Maximum interior collocation point count per epoch.")
    parser.add_argument("--train-point-growth", type=int, default=40,
                        help="Point-count increase every 10 epochs. Use 0 for fixed-size training.")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="Print a lightweight heartbeat every N epochs. Use 0 to disable.")
    parser.add_argument("--empty-cache-every", type=int, default=0,
                        help="Call torch.cuda.empty_cache() every N training steps. Use 0 to disable.")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Initial AdamW learning rate.")
    parser.add_argument("--clean-residual-initial-scale", type=float, default=0.0,
                        help="Initial R_smooth scaffold scale for multiscale_film_tracegreen_clean.")
    parser.add_argument("--clean-residual-decay-epochs", type=int, default=0,
                        help="Epochs over which the clean R_smooth scaffold decays linearly to zero.")
    parser.add_argument("--current-balance-weight", type=float, default=None,
                        help="Final thin inventory-current consistency weight; architecture-aware by default.")
    parser.add_argument("--current-balance-ramp-epochs", type=int, default=100,
                        help="Linear ramp length for the inventory-current consistency weight.")
    parser.add_argument("--current-balance-samples", type=int, default=256,
                        help="Random time samples used by the thin inventory-current loss each epoch.")
    parser.add_argument("--abort-on-nan", action="store_true",
                        help="Abort before optimizer.step if loss or gradient norm is non-finite.")
    parser.add_argument("--reset-best-score", action="store_true",
                        help="Ignore checkpoint best score when resuming after changing loss weights.")
    parser.add_argument("--reset-optimizer-state", action="store_true",
                        help="Load model weights from the resume checkpoint but restart optimizer/scheduler state.")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping and run exactly --epochs epochs.")
    parser.add_argument("--early-stop-check-every", type=int, default=100,
                        help="Evaluate the fixed FDM-free physics validation set every N epochs.")
    parser.add_argument("--early-stop-patience-checks", type=int, default=4,
                        help="Stop after this many validation checks without sufficient improvement.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=-1,
                        help="Minimum added epochs before stopping; -1 selects an architecture-aware default.")
    parser.add_argument("--early-stop-min-relative-improvement", type=float, default=0.005,
                        help="Relative EMA improvement required to reset early-stop patience.")
    parser.add_argument("--early-stop-ema-alpha", type=float, default=0.5,
                        help="EMA alpha for deterministic physics validation scores.")
    parser.add_argument("--early-stop-validation-points", type=int, default=96,
                        help="Deterministic validation points per physics component.")
    parser.add_argument("--kparam-anchor-epochs", type=int, default=1000,
                        help="Added epochs using only k={0.1,1,10} before continuous log-k sampling.")
    parser.add_argument("--kparam-anchor-probability", type=float, default=0.5,
                        help="Anchor sampling probability after the anchor-only stage; minimum 0.5.")
    parser.add_argument("--kparam-log10-std", type=float, default=KPARAM_LOG10_STD,
                        help="Std.dev. of full-support Normal sampling in log10(k/k_ref).")
    parser.add_argument("--kparam-freeze-backbone-epochs", type=int, default=1000,
                        help="Added epochs that train only the zero-initialized k adapter.")
    parser.add_argument("--gammaparam-anchor-probability", type=float, default=0.6,
                        help="Probability of sampling gamma from {0.1,1,100}; otherwise log-uniform on [0.1,100].")
    parser.add_argument("--gammaparam-unfreeze-backbone", action="store_true",
                        help="Also train the gamma=10 backbone. Default keeps it frozen to protect the reference model.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny forward/gradient/backward sanity check and exit.")
    args = parser.parse_args()

    configure_physical_parameters(args.gamma, args.k_cat_star)
    print_physical_configuration()

    checkpoint_was_default = args.checkpoint == parser.get_default("checkpoint")
    MODEL_V96_PATH, MODEL_V96_BEST_PATH = resolve_checkpoint_paths(
        args.arch, args.checkpoint_dir
    )
    if checkpoint_was_default:
        args.checkpoint = MODEL_V96_BEST_PATH

    USE_NORMALIZED_COORDS = not args.legacy_inputs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Normalized coordinates: {USE_NORMALIZED_COORDS}")

    model_thin, model_ext = create_models_v96(
        args.arch,
        normalize_inputs=USE_NORMALIZED_COORDS,
        green_time_grid=args.green_time_grid,
        green_kernel_points=args.green_kernel_points,
        green_history_grad=not args.green_detach_history,
        green_cache_history=not args.green_no_step_cache,
        lift_time_grid=args.lift_time_grid,
    )
    model_thin = model_thin.to(device)
    model_ext = model_ext.to(device)
    if is_k_parameterized(model_ext=model_ext):
        set_model_k_cat(model_thin, model_ext, args.k_cat_star)
    if is_gamma_parameterized(model_ext=model_ext):
        set_model_gamma(model_thin, model_ext, args.gamma)

    total_params = (sum(p.numel() for p in model_thin.parameters()) + 
                   sum(p.numel() for p in model_ext.parameters()))
    print(f"Total trainable parameters: {total_params:,}")

    if args.smoke_test:
        run_smoke_test_v9_6(model_thin, model_ext, device, arch_name=args.arch)
        sys.exit(0)

    if args.eval_only:
        params = list(model_thin.parameters()) + list(model_ext.parameters())
        optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-7)
        loaded_epoch, loss_history, _ = load_model_v96(
            model_thin, model_ext, optimizer, scheduler, args.checkpoint
        )
        loss_history = normalize_loss_history(loss_history)
        if loaded_epoch == 0 and not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        if not loss_history:
            loss_history = {
                'total': [], 'pde_thin': [], 'pde_ext': [], 'surface': [],
                'farfield': [], 'initial': [], 'bounds': [], 'interface': [],
                'interface_thin': [], 'interface_ext': [], 'reversal_continuity': [],
                'dynamic_reversal_jump': [], 'direct_reversal_jump': [],
                'direct_temporal_smooth': [], 'static_temporal_smooth': [],
                'phase_temporal_smooth': [], 'cint_reversal_jump': [],
                'cint_temporal_smooth': [], 'lr': [], 'nernst_err': [],
                'surface_state': [], 'physics_score': []
            }
        print(f"\nEvaluating checkpoint: {args.checkpoint}")
        validate_model(model_thin, model_ext, device, loaded_epoch, verbose=True)
        results = predict_and_visualize_v9_6(
            model_thin, model_ext, loss_history, gamma, n_cv=args.cv_points
        )
        save_cv_csv(results)
        compare_with_fdm(results, args.fdm_csv)
        sys.exit(0)

    # 加载 v9.5 best 作为初始权重
    start_epoch = 0
    loaded_from = ""

    if args.warm_start_checkpoint and (args.resume or args.resume_best or args.resume_checkpoint):
        raise ValueError("--warm-start-checkpoint cannot be combined with resume options")

    if args.warm_start_checkpoint:
        warm_start_model_v96(model_thin, model_ext, args.warm_start_checkpoint)
        loaded_from = f"warm-start {args.warm_start_checkpoint}"
    elif args.arch != "legacy":
        print(f"\nArchitecture '{args.arch}' starts from scratch.")
        print("Legacy checkpoints are not loaded into the multiscale network.")
    elif os.path.exists(MODEL_V95_BEST_PATH) and USE_NORMALIZED_COORDS:
        print("\nNormalized-coordinate training starts from scratch.")
        print("Old raw-coordinate checkpoints are not used as initial weights.")
    elif os.path.exists(MODEL_V95_BEST_PATH):
        checkpoint = torch.load(MODEL_V95_BEST_PATH, map_location='cpu', weights_only=False)
        model_thin.load_state_dict(checkpoint['model_thin_state_dict'], strict=False)
        model_ext.load_state_dict(checkpoint['model_ext_state_dict'])
        start_epoch = checkpoint['epoch']
        loaded_from = "v9.5 best"
        print(f"\nLoaded v9.5 best model from epoch {start_epoch}")
    else:
        print(f"\n⚠️ No pretrained model found")
        print("Starting from scratch")

    # 验证初始状态
    if args.resume or args.resume_best or args.resume_checkpoint:
        print("\nResume mode enabled; checkpoint will be loaded inside training.")
        print("Skipping initial validation before checkpoint load.")
    else:
        print(f"\n{'='*80}")
        print(f"Initial Model Status (loaded from {loaded_from})")
        current_val = validate_model(model_thin, model_ext, device, start_epoch, verbose=True)

    # 开始训练
    print(f"\n{'='*80}")
    print("Starting v9.6 training")
    print(f"{'='*80}")

    loss_history, best_results = train_model_v9_6(
        model_thin, model_ext, 
        n_epochs=args.epochs,
        start_epoch=start_epoch,
        resume=args.resume,
        early_stop=not args.no_early_stop,
        resume_checkpoint=args.resume_checkpoint,
        resume_best=args.resume_best,
        save_every=args.save_every,
        thin_interface_weight=args.thin_interface_weight,
        ext_interface_weight=args.ext_interface_weight,
        reset_best_score=args.reset_best_score,
        pde_thin_weight=args.pde_thin_weight,
        pde_ext_weight=args.pde_ext_weight,
        farfield_weight=args.farfield_weight,
        initial_weight=args.initial_weight,
        bounds_weight=args.bounds_weight,
        arch_name=args.arch,
        fdm_compare_pkl=args.fdm_compare_pkl,
        fdm_compare_every=args.fdm_compare_every,
        fdm_compare_dir=args.fdm_compare_dir,
        fdm_compare_n_time=args.fdm_compare_n_time,
        fdm_compare_n_x_in=args.fdm_compare_n_x_in,
        fdm_compare_n_x_out=args.fdm_compare_n_x_out,
        fdm_compare_batch_size=args.fdm_compare_batch_size,
        fdm_compare_save_figure=args.fdm_compare_save_figure,
        fdm_compare_save_fields=args.fdm_compare_save_fields,
        reset_optimizer_state=args.reset_optimizer_state,
        base_train_points=args.base_train_points,
        max_train_points=args.max_train_points,
        train_point_growth=args.train_point_growth,
        progress_every=args.progress_every,
        empty_cache_every=args.empty_cache_every,
        learning_rate=args.learning_rate,
        abort_on_nan=args.abort_on_nan,
        clean_residual_initial_scale=args.clean_residual_initial_scale,
        clean_residual_decay_epochs=args.clean_residual_decay_epochs,
        early_stop_check_every=args.early_stop_check_every,
        early_stop_patience_checks=args.early_stop_patience_checks,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_min_relative_improvement=args.early_stop_min_relative_improvement,
        early_stop_ema_alpha=args.early_stop_ema_alpha,
        early_stop_validation_points=args.early_stop_validation_points,
        kparam_anchor_epochs=args.kparam_anchor_epochs,
        kparam_anchor_probability=args.kparam_anchor_probability,
        kparam_log10_std=args.kparam_log10_std,
        kparam_freeze_backbone_epochs=args.kparam_freeze_backbone_epochs,
        gammaparam_anchor_probability=args.gammaparam_anchor_probability,
        gammaparam_freeze_backbone=not args.gammaparam_unfreeze_backbone,
        current_balance_weight=args.current_balance_weight,
        current_balance_ramp_epochs=args.current_balance_ramp_epochs,
        current_balance_samples=args.current_balance_samples,
    )

    # 可视化
    # if loss_history and len(loss_history['total']) > 0:
    #     results = predict_and_visualize_v9_6(model_thin, model_ext, loss_history, gamma)
    #
    #     with open('./results_thin_layer_catalytic_v9_6.pkl', 'wb') as f:
    #         pickle.dump({
    #             'loss_history': loss_history,
    #             'cv_results': {
    #                 'theta': results['theta'],
    #                 'J': results['J']
    #             },
    #             'key_results': results['results'],
    #             'phys_results': results['phys_results'],
    #             'parameters': {
    #                 'sigma': sigma, 'theta_i': theta_i, 'theta_switch': theta_switch,
    #                 'T_sim': T_sim, 'delta': delta, 'gamma': gamma,
    #                 'k_cat_star': k_cat_star, 'lambda_factor': lambda_factor
    #             }
    #         }, f)
    #     print(f"\nDetailed results saved to: ./results_thin_layer_catalytic_v9_6.pkl")
    if loss_history and len(loss_history['total']) > 0:
        results = predict_and_visualize_v9_6(
            model_thin, model_ext, loss_history, gamma, n_cv=args.cv_points
        )

        # 保存完整 pkl
        with open('./results_thin_layer_catalytic_v9_6.pkl', 'wb') as f:
            pickle.dump({
                'loss_history': loss_history,
                'cv_results': {
                    'theta': results['theta'],
                    'J': results['J']
                },
                'key_results': results['results'],
                'phys_results': results['phys_results'],
                'parameters': {
                    'sigma': sigma, 'theta_i': theta_i, 'theta_switch': theta_switch,
                    'T_sim': T_sim, 'delta': delta, 'gamma': gamma,
                    'k_cat_star': k_cat_star, 'lambda_factor': lambda_factor
                }
            }, f)

        print(f"\nDetailed results saved to: ./results_thin_layer_catalytic_v9_6.pkl")

        # 单独保存 theta 和 J 到 CSV
        save_cv_csv(results)
        compare_with_fdm(results, args.fdm_csv)

    print(f"\n{'='*80}")
    print("v9.6 model training completed")
    print(f"{'='*80}")
    print("✓ Interface weight increased to 200")
    print("✓ Interface sampling increased")
    print("✓ Farfield sampling increased")
    print("✓ Flux sign verified correct (unchanged)")
    print(f"\nModel file: {MODEL_V96_PATH}")
    print(f"Best model: {MODEL_V96_BEST_PATH}")
    print(f"Results figure: ./thin_layer_catalytic_v9_6_results.png")
    print(f"{'='*80}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    sys.exit(0)
