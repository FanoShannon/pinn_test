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

k_cat_star = 1.0
gamma = 10.0

D_rel_A = 1.0
D_rel_B = 1.0
D_rel_C = 1.0
D_rel_D = 1.0

print(f"=== PINN v9.6 - Interface Coupling Enhancement ===")
print(f"Thin layer thickness δ = {delta:.4f}")
print(f"External region length = {X_ext_max - delta:.4f}")
print(f"Catalytic constant k_cat* = {k_cat_star}, γ = {gamma}")
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
MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_interface_memory.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_interface_memory_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_memory.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_memory_best.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel.pth'
MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_BEST_PATH = './pinn_thin_layer_catalytic_v9_6_multiscale_green_grid_film_abel_best.pth'
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
    elif arch == "multiscale_green_grid_interface_memory":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_INTERFACE_MEMORY_BEST_PATH
    elif arch == "multiscale_green_grid_memory":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_MEMORY_BEST_PATH
    elif arch == "multiscale_green_grid_film_abel":
        current_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_PATH
        best_path = MODEL_V96_MULTISCALE_GREEN_GRID_FILM_ABEL_BEST_PATH
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
        self.c_c_depletion_scale = 4.0
        self.surface_slope_scale = 6.0

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
        C_C_int = gamma - self.c_c_depletion_scale * time_gate * torch.sigmoid(raw[:, 1:2])
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
        C_C_int = gamma - self.c_c_depletion_scale * time_gate * torch.sigmoid(raw[:, 1:2])
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
        C_C_int = gamma - self.c_c_depletion_scale * time_gate * torch.sigmoid(
            base["raw"][:, 1:2] + corr[:, 1:2]
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

    def _abel_gain(self):
        return torch.exp(0.5 * torch.tanh(self.abel_gain_raw))

    def _abel_diffusion(self):
        return D_rel_D * torch.exp(0.5 * torch.tanh(self.abel_diffusion_raw))

    def _phase_shift(self):
        return 0.03 * T_sim * torch.tanh(self.phase_shift_raw)

    def _surface_state(self, T_raw):
        theta = potential_theta(T_raw)
        theta_dot = potential_theta_dot(T_raw)
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
        denom = 1.0 + (k_cat_star * delta / D_rel_B) * c_c_pos
        c_b_int = c_b_surface / torch.clamp(denom, min=1e-8)
        j_rxn = k_cat_star * c_c_pos * c_b_int
        surface_slope = -j_rxn / D_rel_B
        return c_b_int, j_rxn, surface_slope

    def _bounded_c_d_int(self, c_d_prior, corr_raw, T_raw):
        eps = 1e-6
        frac = torch.clamp(c_d_prior / gamma, eps, 1.0 - eps)
        prior_logit = torch.logit(frac)
        time_gate = 1.0 - torch.exp(-torch.clamp(T_raw, min=0.0) / (0.05 * T_sim))
        corr_logit = self.residual_logit_scale * time_gate * torch.tanh(corr_raw)
        return gamma * torch.sigmoid(prior_logit + corr_logit)

    def _feature_tensor(self, T_raw, theta, theta_dot, c_b_surface, dc_b_surface_dt,
                        c_d_prior, j_hist, d_j_hist, q_hist, memories):
        theta_scale = max(abs(float(theta_i)), abs(float(theta_switch)), 1.0)
        theta_dot_scale = theta_scale / max(float(T_sim), 1e-12)
        j_scale = max(float(k_cat_star * gamma), 1.0)
        q_scale = max(j_scale * float(T_sim), 1.0)
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
        t_pos = torch.clamp(T_raw, min=1e-6 * T_sim)
        nodes = self.kernel_nodes.reshape(1, -1).to(device=T_raw.device, dtype=T_raw.dtype)
        tau = t_pos * nodes
        dtau = torch.clamp(t_pos * (1.0 - nodes), min=1e-8 * T_sim)
        value_tau = self._interp_scalar_grid(tau, value_grid)
        diffusion = torch.clamp(self._abel_diffusion().to(device=T_raw.device, dtype=T_raw.dtype), min=1e-8)
        gain = self._abel_gain().to(device=T_raw.device, dtype=T_raw.dtype)
        kernel = gain / torch.sqrt(torch.clamp(np.pi * diffusion * dtau, min=1e-12))
        return T_raw * torch.mean(value_tau * kernel, dim=1, keepdim=True)

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
        cache_key = (T_ref.device, T_ref.dtype, torch.is_grad_enabled())
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
            c_d_int = self._bounded_c_d_int(c_d_prior, self.residual_net(features), t)
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
        c_d_int = self._bounded_c_d_int(c_d_prior, self.residual_net(features), T_raw)
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

        raw_field = self.net(x_net)
        endpoint_bubble = s2 * (1.0 - s) ** 2
        surface_bubble = s * (1.0 - s) ** 2
        correction = time_gate * (
            self.surface_bubble_scale * surface_bubble * torch.tanh(raw_field[:, 1:2]) +
            self.correction_scale * endpoint_bubble * torch.tanh(raw_field[:, 0:1])
        )
        C_B = c_b_base + correction
        C_A = 1.0 - C_B
        return C_A, C_B


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
        cache_key = (T_ref.device, T_ref.dtype, torch.is_grad_enabled(), self.history_grad)
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

    def dynamic_correction(self, T_raw, X_raw, r, x_net, state):
        dynamic_input = self.dynamic_features(T_raw, X_raw, r, x_net, state)
        raw = self.dynamic_net(dynamic_input)
        modes = self.dynamic_modes(r)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        scales = self.dynamic_correction_scales.to(device=raw.device, dtype=raw.dtype)
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
        residual = residual + self.dynamic_correction(T_raw, X_raw, r, x_net, state)

        C_D = c_d_base + residual
        C_C = self.gamma - C_D
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


# ==================== 模型保存/加载 ====================
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
        'parameters': {
            'sigma': sigma, 'theta_i': theta_i, 'theta_switch': theta_switch,
            'T_sim': T_sim, 'delta': delta, 'X_ext_max': X_ext_max,
            'gamma': gamma, 'k_cat_star': k_cat_star, 'lambda_factor': lambda_factor,
            'normalize_inputs': USE_NORMALIZED_COORDS,
            'green_time_grid': getattr(model_ext, 'time_grid_points', None),
            'green_kernel_points': getattr(model_ext, 'kernel_points', None),
            'green_history_grad': getattr(model_ext, 'history_grad', None),
        }
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


def load_model_v96(model_thin, model_ext, optimizer, scheduler, path, load_optimizer_state=True):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
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


def normalize_loss_history(loss_history):
    if not loss_history:
        return loss_history

    total_len = len(loss_history.get('total', []))
    for key in ('interface_thin', 'interface_ext', 'bounds'):
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

        J_A = -D_rel_A * C_A_X_int
        J_B = -D_rel_B * C_B_X_int
        J_C = -D_rel_C * C_C_X_int
        J_D = -D_rel_D * C_D_X_int
        J_rxn = k_cat_star * C_B_int * C_C_int

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

    with torch.no_grad():
        # 1. Nernst误差
        T_test = torch.linspace(0, T_sim, 200, device=device).reshape(-1, 1)
        X_test = torch.zeros_like(T_test)
        C_A, C_B = model_thin(torch.cat([T_test, X_test], dim=1))
        theta = potential_theta(T_test)
        nernst_err = torch.sqrt(torch.mean((C_A - C_B * torch.exp(theta))**2)).item()
        nernst_max_err = torch.max(torch.abs(C_A - C_B * torch.exp(theta))).item()
        C_A_eq = torch.sigmoid(theta)
        C_B_eq = torch.sigmoid(-theta)
        surface_state_err = torch.sqrt(torch.mean((C_A - C_A_eq)**2 + (C_B - C_B_eq)**2)).item()

        X_int = torch.ones_like(T_test) * delta
        _, C_B_int = model_thin(torch.cat([T_test, X_int], dim=1))
        C_C_int, _ = model_ext(torch.cat([T_test, X_int], dim=1))
        J_rxn_mean = torch.mean(k_cat_star * C_B_int * C_C_int).item()
        C_B_int_mean = torch.mean(C_B_int).item()
        C_C_int_mean = torch.mean(C_C_int).item()

        # 2. 薄层守恒
        T_cons = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device) * T_sim
        X_cons = torch.linspace(0, delta, 50, device=device).reshape(-1, 1)
        err_AB_max = 0
        for t in T_cons:
            T_grid = torch.ones_like(X_cons) * t
            inputs = torch.cat([T_grid, X_cons], dim=1)
            C_A_test, C_B_test = model_thin(inputs)
            err = torch.abs(C_A_test + C_B_test - 1.0).max().item()
            err_AB_max = max(err_AB_max, err)

        # 3. 外部守恒
        X_cons_ext = torch.linspace(delta, X_ext_max, 50, device=device).reshape(-1, 1)
        err_CD_max = 0
        for t in T_cons:
            T_grid = torch.ones_like(X_cons_ext) * t
            inputs = torch.cat([T_grid, X_cons_ext], dim=1)
            C_C_test, C_D_test = model_ext(inputs)
            err = torch.abs(C_C_test + C_D_test - gamma).max().item()
            err_CD_max = max(err_CD_max, err)

    # 4. CV峰电流
    T_surf = torch.linspace(0, T_sim, 300, device=device).reshape(-1, 1)
    X_surf = torch.zeros_like(T_surf)
    X_surf.requires_grad_(True)

    C_A_surf, _ = model_thin(torch.cat([T_surf, X_surf], dim=1))
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
        print(f"{'='*60}")

    return {
        'nernst_err': nernst_err,
        'nernst_max_err': nernst_max_err,
        'surface_state_err': surface_state_err,
        'C_B_int_mean': C_B_int_mean,
        'C_C_int_mean': C_C_int_mean,
        'J_rxn_mean': J_rxn_mean,
        'J_peak': J_peak,
        'theta_peak': theta_peak,
        'err_AB': err_AB_max,
        'err_CD': err_CD_max
    }


# ==================== v9.6 训练函数 - 界面耦合增强 ====================
def concentration_bounds_loss(C_A, C_B, C_C, C_D):
    thin_loss = (
        torch.mean(torch.relu(-C_A) ** 2) +
        torch.mean(torch.relu(C_A - 1.0) ** 2) +
        torch.mean(torch.relu(-C_B) ** 2) +
        torch.mean(torch.relu(C_B - 1.0) ** 2)
    )
    ext_loss = (
        torch.mean(torch.relu(-C_C) ** 2) +
        torch.mean(torch.relu(C_C - gamma) ** 2) +
        torch.mean(torch.relu(-C_D) ** 2) +
        torch.mean(torch.relu(C_D - gamma) ** 2)
    )
    return thin_loss + ext_loss


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
        "parameters": {
            "sigma": sigma, "theta_i": theta_i, "theta_switch": theta_switch,
            "T_sim": T_sim, "delta": delta, "X_ext_max": X_ext_max,
            "gamma": gamma, "k_cat_star": k_cat_star, "lambda_factor": lambda_factor,
            "normalize_inputs": USE_NORMALIZED_COORDS,
            "green_time_grid": getattr(model_ext, "time_grid_points", None),
            "green_kernel_points": getattr(model_ext, "kernel_points", None),
            "green_history_grad": getattr(model_ext, "history_grad", None),
        },
    }, eval_checkpoint)

    cmd = [
        sys.executable, "-u", script_path,
        "--fdm-pkl", fdm_pkl,
        "--arch", arch_name,
        "--input-mode", "normalized" if USE_NORMALIZED_COORDS else "legacy",
        "--checkpoint", eval_checkpoint,
        "--n-time", str(n_time),
        "--n-x-in", str(n_x_in),
        "--n-x-out", str(n_x_out),
        "--batch-size", str(batch_size),
        "--output-json", output_json,
        "--output-npz", output_npz,
        "--output-figure", output_figure,
    ]
    if isinstance(model_ext, ExternalNet_v9_6_MultiscaleGreenGrid):
        cmd.extend([
            "--green-time-grid", str(model_ext.time_grid_points),
            "--green-kernel-points", str(model_ext.kernel_points),
        ])
        if not model_ext.history_grad:
            cmd.append("--green-detach-history")

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
    for name in ["C_A", "C_B", "C_C", "C_D", "C_B_int", "C_C_int", "CV_J"]:
        if name not in metrics:
            continue
        row = metrics[name]
        print(
            f"{name},{row['rmse']:.6e},{row['mae']:.6e},{row['max_abs']:.6e},"
            f"{row['bias']:.6e},{row['r2']:.6e},{row['nrmse']:.6e}"
        )
    print(f"overall_rmse,{metrics['overall']['rmse']:.6e}")
    print(f"{'='*60}")


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
                     abort_on_nan=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_thin.to(device)
    model_ext.to(device)

    params = []
    seen_params = set()
    for module in (model_thin, model_ext):
        for param in module.parameters():
            if id(param) not in seen_params:
                params.append(param)
                seen_params.add(id(param))
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-7)

    is_hermite = isinstance(model_thin, ThinLayerNet_v9_6_MultiscaleHermite)
    is_his = isinstance(model_thin, ThinLayerNet_v9_6_HISPrototype)
    is_ext_his = isinstance(model_ext, ExternalNet_v9_6_MultiscaleHISLayer)
    is_green = isinstance(model_ext, (ExternalNet_v9_6_MultiscaleGreenKernel, ExternalNet_v9_6_MultiscaleGreenGrid))
    is_buffer = isinstance(model_ext, ExternalNet_v9_6_MultiscaleBuffer)
    is_flux_state = hasattr(model_ext, "flux_d")
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
        'interface_thin': [], 'interface_ext': [], 'lr': [],
        'nernst_err': [], 'surface_state': [], 'physics_score': []
    }

    best_score = float('inf')
    best_epoch = 0
    patience = 5000
    no_improve = 0
    best_val_results = None

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
        if reset_best_score:
            best_score = float('inf')
            best_epoch = start_epoch
            print("Best score reset because the active loss weights may differ from the checkpoint.")
        elif loaded_best is not None:
            best_score = loaded_best
            best_epoch = start_epoch
        print(f"Resumed from {loaded_path} at epoch {start_epoch}")

    print(f"\n{'='*80}")
    print(f"Starting v9.6 training from epoch {start_epoch}")
    print(f"Training for {n_epochs} additional epochs")
    print(f"Current checkpoint: {MODEL_V96_PATH}")
    print(f"Best checkpoint:    {MODEL_V96_BEST_PATH}")
    print(f"Save every:         {save_every} epochs")
    print(f"Key changes:")
    print(f"  1. Interface weight: 50 → 200 (4x increase)")
    print(f"  2. Interface sampling: focused T windows plus higher hardbc point count")
    print(f"  3. Farfield sampling: n_points/10 → n_points/5")
    print(f"  4. Flux sign: UNCHANGED (analysis shows equivalent to FDM)")
    print(f"  5. Thin/external interface weights: {thin_interface_weight:.1f}/{ext_interface_weight:.1f}")
    print(f"  6. PDE weights thin/ext: {pde_thin_weight:.1f}/{pde_ext_weight:.1f}; bounds weight: {bounds_weight:.1f}")
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
        if getattr(model_ext, "dynamic_lite", False):
            print(
                " 10. Dynamic correction inputs: "
                "x,t,theta,dtheta_dt,J_rxn,dJ_rxn_dt,Q_rxn; "
                f"scales={model_ext.dynamic_correction_scales.detach().cpu().numpy().reshape(-1).tolist()}"
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
        if getattr(interface_state, "film_abel_interface", False):
            print(
                " 12. Film-Abel interface chain: "
                f"M={interface_state.time_grid_points}, K={interface_state.kernel_points}, "
                f"memory_lambdas={interface_state.film_abel_memory_lambdas.detach().cpu().numpy().reshape(-1).tolist()}, "
                "C_B_int=C_B_surface/(1+k*delta*C_C_int/D_B), "
                "C_D_int=Abel[J]-dt_phase*Abel[dJ]+small_residual"
            )
    print(f"{'='*80}")

    train_wall_start = time.time()
    last_progress_wall = train_wall_start

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model_thin.train()
        model_ext.train()
        if hasattr(model_ext, "clear_step_cache"):
            model_ext.clear_step_cache()

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
        inputs_thin = torch.cat([T_thin, X_thin], dim=1)
        C_A, C_B = model_thin(inputs_thin)

        C_A_T = torch.autograd.grad(C_A.sum(), T_thin, create_graph=True)[0]
        C_A_X = torch.autograd.grad(C_A.sum(), X_thin, create_graph=True)[0]
        C_A_XX = torch.autograd.grad(C_A_X.sum(), X_thin, create_graph=True)[0]

        C_B_T = torch.autograd.grad(C_B.sum(), T_thin, create_graph=True)[0]
        C_B_X = torch.autograd.grad(C_B.sum(), X_thin, create_graph=True)[0]
        C_B_XX = torch.autograd.grad(C_B_X.sum(), X_thin, create_graph=True)[0]

        loss_pde_thin = torch.mean((C_A_T - D_rel_A * C_A_XX)**2) +                         torch.mean((C_B_T - D_rel_B * C_B_XX)**2)

        # 3.2 External PDE
        inputs_ext = torch.cat([T_ext, X_ext], dim=1)
        C_C, C_D = model_ext(inputs_ext)

        if is_flux_state:
            C_D_T = torch.autograd.grad(C_D.sum(), T_ext, create_graph=True)[0]
            C_D_X = torch.autograd.grad(C_D.sum(), X_ext, create_graph=True)[0]
            J_D_ext = model_ext.flux_d(inputs_ext)
            J_D_X = torch.autograd.grad(J_D_ext.sum(), X_ext, create_graph=True)[0]
            loss_ext_conservation = torch.mean((C_D_T + J_D_X)**2)
            loss_ext_constitutive = torch.mean((J_D_ext + D_rel_D * C_D_X)**2)
            loss_pde_ext = loss_ext_conservation + 2.0 * loss_ext_constitutive
        else:
            C_C_T = torch.autograd.grad(C_C.sum(), T_ext, create_graph=True)[0]
            C_C_X = torch.autograd.grad(C_C.sum(), X_ext, create_graph=True)[0]
            C_C_XX = torch.autograd.grad(C_C_X.sum(), X_ext, create_graph=True)[0]

            loss_pde_ext = torch.mean((C_C_T - D_rel_C * C_C_XX)**2)

        # 3.3 Surface boundary (Nernst) - 与 v9.5 相同
        T_surf = torch.rand(n_surface, 1, device=device) * T_sim

        n_near = int(n_surface * 0.8)
        n_uniform = n_surface - n_near

        X_near = delta * 0.05 * torch.exp(-8.0 * torch.rand(n_near, 1, device=device))
        X_uniform = torch.rand(n_uniform, 1, device=device) * delta
        X_surf = torch.cat([X_near, X_uniform], dim=0)

        perm = torch.randperm(n_surface)
        X_surf = X_surf[perm]

        inputs_surf = torch.cat([T_surf, X_surf], dim=1)
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
        inputs_exact = torch.cat([T_exact, X_exact], dim=1)
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
        inputs_far = torch.cat([T_far, X_far], dim=1)
        C_C_far, C_D_far = model_ext(inputs_far)

        loss_farfield = torch.mean((C_C_far - gamma)**2) + torch.mean(C_D_far**2)

        # 3.5 Initial condition
        T_ini = torch.zeros(n_points // 5, 1, device=device)
        X_ini_thin = torch.rand(n_points // 5, 1, device=device) * delta
        X_ini_ext = delta + torch.rand(n_points // 5, 1, device=device) * (X_ext_max - delta)

        C_A_ini, C_B_ini = model_thin(torch.cat([T_ini, X_ini_thin], dim=1))
        C_C_ini, C_D_ini = model_ext(torch.cat([T_ini, X_ini_ext], dim=1))

        T_ini_int = torch.zeros(n_points // 10, 1, device=device)
        X_ini_int = torch.ones_like(T_ini_int) * delta
        C_C_ini_int, C_D_ini_int = model_ext(torch.cat([T_ini_int, X_ini_int], dim=1))

        loss_initial = (torch.mean((C_A_ini - 1.0)**2) + 
                       torch.mean(C_B_ini**2) + 
                       torch.mean((C_C_ini - gamma)**2) + 
                       torch.mean(C_D_ini**2) +
                       2.0 * torch.mean((C_C_ini_int - gamma)**2) +
                       2.0 * torch.mean(C_D_ini_int**2))

        loss_bounds = concentration_bounds_loss(C_A, C_B, C_C, C_D)

        # 3.6 Interface coupling - v9.6: 保持 v9.5 符号不变
        # 经分析，v9.5 的符号与 FDM 等价，只是定义不同
        T_int = sample_interface_t(n_interface, device)

        X_int_thin = torch.ones_like(T_int) * delta
        X_int_thin.requires_grad_(True)
        inputs_int_thin = torch.cat([T_int, X_int_thin], dim=1)
        C_A_int, C_B_int = model_thin(inputs_int_thin)
        C_A_X_int = torch.autograd.grad(C_A_int.sum(), X_int_thin, create_graph=True, retain_graph=True)[0]
        C_B_X_int = torch.autograd.grad(C_B_int.sum(), X_int_thin, create_graph=True, retain_graph=True)[0]

        X_int_ext = torch.ones_like(T_int) * delta
        X_int_ext.requires_grad_(True)
        inputs_int_ext = torch.cat([T_int, X_int_ext], dim=1)
        C_C_int, C_D_int = model_ext(inputs_int_ext)
        C_C_X_int = torch.autograd.grad(C_C_int.sum(), X_int_ext, create_graph=True, retain_graph=True)[0]
        C_D_X_int = torch.autograd.grad(C_D_int.sum(), X_int_ext, create_graph=True, retain_graph=True)[0]

        J_rxn = k_cat_star * C_B_int * C_C_int

        # 保持 v9.5 的通量符号（经分析正确）
        flux_A_res = -D_rel_A * C_A_X_int + J_rxn
        flux_B_res = -D_rel_B * C_B_X_int - J_rxn
        flux_C_res = -D_rel_C * C_C_X_int + J_rxn
        flux_D_res = -D_rel_D * C_D_X_int - J_rxn

        loss_interface_thin = torch.mean(flux_A_res**2) + torch.mean(flux_B_res**2)
        loss_interface_ext = torch.mean(flux_C_res**2) + torch.mean(flux_D_res**2)
        loss_interface = loss_interface_thin + loss_interface_ext

        # ========== 4. Weight adjustment ==========
        if epoch < start_epoch + 3000:
            surface_weight = 20.0 * np.exp((epoch - start_epoch) / 3000)
        elif epoch < start_epoch + 10000:
            surface_weight = 60.0 * np.exp((epoch - start_epoch - 3000) / 2000)
        else:
            surface_weight = min(500.0, 150.0 * np.exp((epoch - start_epoch - 10000) / 1500))

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
            base_weights['interface_ext'] * loss_interface_ext
        )
        physics_score = (
            pde_thin_weight * loss_pde_thin +
            pde_ext_weight * loss_pde_ext +
            100.0 * loss_surface +
            farfield_weight * loss_farfield +
            initial_weight * loss_initial +
            bounds_weight * loss_bounds +
            thin_interface_weight * loss_interface_thin +
            ext_interface_weight * loss_interface_ext
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
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_history['nernst_err'].append(nernst_err)
        loss_history['surface_state'].append(loss_surface_state.item())
        loss_history.setdefault('physics_score', []).append(physics_score.item())

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
            if getattr(interface_state, "film_abel_interface", False):
                phase_msg = (
                    f" | dt_phase={float(interface_state._phase_shift().detach().cpu()):+.3e}"
                    f" | D_eff={float(interface_state._abel_diffusion().detach().cpu()):.3f}"
                    f" | abel_gain={float(interface_state._abel_gain().detach().cpu()):.3f}"
                )
            print(
                f"[progress] epoch {epoch + 1}/{start_epoch + n_epochs} "
                f"({step_count}/{n_epochs}) | loss={total_loss.item():.3e} "
                f"pde=({loss_pde_thin.item():.2e},{loss_pde_ext.item():.2e}) "
                f"iface=({loss_interface_thin.item():.2e},{loss_interface_ext.item():.2e}) "
                f"| {sec_per_epoch:.2f}s/epoch | last {progress_every}={interval_sec:.1f}s "
                f"| ETA={eta_sec/60.0:.1f}min{gpu_msg}{phase_msg}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

        # ========== 7. Validation & Early stopping ==========
        if epoch % 500 == 0:
            if hasattr(model_ext, "clear_step_cache"):
                model_ext.clear_step_cache()
            val_results = validate_model(model_thin, model_ext, device, epoch, verbose=True)
            save_score = physics_score.item()

            if save_score < best_score:
                best_score = save_score
                best_epoch = epoch
                no_improve = 0
                best_val_results = val_results
                save_model_v96(model_thin, model_ext, optimizer, scheduler,
                             epoch, MODEL_V96_BEST_PATH,
                             loss_history=loss_history, best_val_loss=best_score)
                print(f"  ✓ New best model saved (physics score: {best_score:.4e})")
            else:
                no_improve += 500

            if early_stop and no_improve > patience and epoch > start_epoch + 8000:
                print(f"\n{'='*60}")
                print(f"Early stopping at epoch {epoch}")
                print(f"Best physics score: {best_score:.4e} at epoch {best_epoch}")
                print(f"{'='*60}")
                break

        # 进度输出
        if epoch % 500 == 0:
            with torch.no_grad():
                T_test = torch.rand(100, 1, device=device) * T_sim
                X_test = torch.rand(100, 1, device=device) * delta
                C_A_test, C_B_test = model_thin(torch.cat([T_test, X_test], dim=1))
                err_AB = torch.abs(C_A_test + C_B_test - 1.0).max().item()

                X_test_ext = delta + torch.rand(100, 1, device=device) * (X_ext_max - delta)
                C_C_test, C_D_test = model_ext(torch.cat([T_test, X_test_ext], dim=1))
                err_CD = torch.abs(C_C_test + C_D_test - gamma).max().item()

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
                J_rxn_test = (k_cat_star * C_B_int_test * C_C_int_test).mean().item()

            print(f"\nEpoch {epoch:5d} | Total: {total_loss.item():.4e}")
            print(f"  Physics score: {physics_score.item():.4e}")
            print(f"  PDE_thin: {loss_pde_thin.item():.4e} | PDE_ext: {loss_pde_ext.item():.4e}")
            print(f"  Surface: {loss_surface.item():.4e} (w={base_weights['surface']:.2f})")
            print(f"  Surface state: {loss_surface_state.item():.4e}")
            print(f"  Bounds: {loss_bounds.item():.4e} (w={base_weights['bounds']:.1f})")
            print(
                f"  Interface: {loss_interface.item():.4e} "
                f"(thin={loss_interface_thin.item():.4e}, ext={loss_interface_ext.item():.4e}; "
                f"w={base_weights['interface_thin']:.1f}/{base_weights['interface_ext']:.1f})"
            )
            print(f"  Hard: A+B={err_AB:.2e}, C+D={err_CD:.2e}")
            print(f"  Nernst={nernst_err_test:.2e} | J_rxn={J_rxn_test:.4e}")
            print(f"  C_B(δ)={C_B_int_test.mean().item():.4f}, C_C(δ)={C_C_int_test.mean().item():.4f}")
            print(f"  Best physics score: {best_score:.2e} @ {best_epoch}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

            if epoch % 2000 == 0 and epoch > start_epoch:
                phys_results = verify_interface_physics(model_thin, model_ext, device)
                print_physics_verification(phys_results)

            if fdm_compare_pkl and fdm_compare_every > 0 and epoch % fdm_compare_every == 0:
                if hasattr(model_ext, "clear_step_cache"):
                    model_ext.clear_step_cache()
                run_fdm_posterior_compare(
                    model_thin, model_ext, epoch, arch_name, fdm_compare_pkl,
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

    # FDM 参考
    ax4.axhline(-6.0628, color='g', linestyle=':', linewidth=2, 
                alpha=0.7, label=f'FDM ref: -6.063')

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
    results_text += f"FDM reference: 17.32x\n"
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
    print(f"FDM reference enhancement:      17.32x")
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


def compare_with_fdm(results, fdm_csv_path="../FDM/v41_cv_data_fixed.csv"):
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
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for current/best checkpoints, e.g. a Google Drive folder on Colab.")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Save the current checkpoint every N epochs. Use 0 to disable periodic saves.")
    parser.add_argument("--legacy-inputs", action="store_true",
                        help="Use raw coordinates for old v9.6 checkpoints.")
    parser.add_argument("--cv-points", type=int, default=8000)
    parser.add_argument("--fdm-csv", default="../FDM/v41_cv_data_fixed.csv")
    parser.add_argument("--arch", choices=["legacy", "multiscale", "multiscale_hardbc", "his_pinn", "his_pinn_ext", "multiscale_hermite", "multiscale_hermite_extbasis", "multiscale_green", "multiscale_green_grid", "multiscale_green_grid_hybrid", "multiscale_green_grid_dynamic", "multiscale_green_grid_interface_memory", "multiscale_green_grid_memory", "multiscale_green_grid_film_abel", "multiscale_buffer"], default="legacy")
    parser.add_argument("--green-time-grid", type=int, default=256,
                        help="Global history time-grid size for multiscale_green_grid.")
    parser.add_argument("--green-kernel-points", type=int, default=32,
                        help="Green convolution quadrature points for multiscale_green_grid.")
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
    parser.add_argument("--abort-on-nan", action="store_true",
                        help="Abort before optimizer.step if loss or gradient norm is non-finite.")
    parser.add_argument("--reset-best-score", action="store_true",
                        help="Ignore checkpoint best score when resuming after changing loss weights.")
    parser.add_argument("--reset-optimizer-state", action="store_true",
                        help="Load model weights from the resume checkpoint but restart optimizer/scheduler state.")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping and run exactly --epochs epochs.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny forward/gradient/backward sanity check and exit.")
    args = parser.parse_args()

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
    )
    model_thin = model_thin.to(device)
    model_ext = model_ext.to(device)

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
                'interface_thin': [], 'interface_ext': [], 'lr': [],
                'nernst_err': [], 'surface_state': [], 'physics_score': []
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

    if args.arch != "legacy":
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
