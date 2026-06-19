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
import pickle
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

USE_NORMALIZED_COORDS = True


def resolve_checkpoint_paths(arch, checkpoint_dir=None):
    if arch == "multiscale":
        current_path = MODEL_V96_MULTISCALE_PATH
        best_path = MODEL_V96_MULTISCALE_BEST_PATH
    elif arch == "multiscale_hardbc":
        current_path = MODEL_V96_MULTISCALE_HARDBC_PATH
        best_path = MODEL_V96_MULTISCALE_HARDBC_BEST_PATH
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


def sample_external_x(n_points, device, near_fraction=0.75):
    n_near = int(n_points * near_fraction)
    n_uniform = n_points - n_near
    boundary_layer = min(0.5, 12.0 * delta)
    x_near = delta + boundary_layer * torch.rand(n_near, 1, device=device) ** 2
    x_uniform = delta + torch.rand(n_uniform, 1, device=device) * (X_ext_max - delta)
    if n_uniform > 0:
        x_all = torch.cat([x_near, x_uniform], dim=0)
        return x_all[torch.randperm(n_points, device=device)]
    return x_near


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
    def __init__(self, out_features, width=256, depth=5):
        super().__init__()
        self.features = FourierFeatureLayer()
        layers = [nn.Linear(self.features.out_features, width), nn.Tanh()]
        for _ in range(depth):
            layers.append(ResidualMLPBlock(width))
        final_layer = nn.Linear(width, out_features)
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.features(x))


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

        raw = self.net(x_net)
        time_gate = torch.clamp(T_raw / T_sim, 0.0, 1.0)
        c_b_free = time_gate * F.softmax(raw, dim=1)[:, 1:2]
        c_b_surface = torch.sigmoid(-potential_theta(T_raw))
        x_gate = torch.clamp(X_raw / delta, 0.0, 1.0)
        C_B = (1.0 - x_gate) * c_b_surface + x_gate * c_b_free
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


def create_models_v96(arch="legacy", normalize_inputs=True):
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
    raise ValueError(f"Unknown architecture: {arch}")


def potential_theta(T):
    return torch.where(T <= T_switch * T_sim,
                       theta_i - 2*(theta_i - theta_switch) * T / T_sim,
                       theta_switch + 2*(theta_i - theta_switch) * (T - T_switch * T_sim) / T_sim)


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
        }
    }, save_path)
    print(f"✅ Model saved to {save_path} (epoch {epoch})")


def load_model_v96(model_thin, model_ext, optimizer, scheduler, path):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model_thin.load_state_dict(checkpoint['model_thin_state_dict'], strict=False)
        model_ext.load_state_dict(checkpoint['model_ext_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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
def train_model_v9_6(model_thin, model_ext, n_epochs=30000, start_epoch=0, resume=False,
                     early_stop=True, resume_checkpoint=None, resume_best=False,
                     save_every=2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_thin.to(device)
    model_ext.to(device)

    params = list(model_thin.parameters()) + list(model_ext.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-7)

    loss_history = {
        'total': [], 'pde_thin': [], 'pde_ext': [], 'surface': [],
        'farfield': [], 'initial': [], 'interface': [], 'lr': [],
        'nernst_err': [], 'surface_state': []
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
            model_thin, model_ext, optimizer, scheduler, loaded_path
        )
        if loaded_history:
            loss_history.update(loaded_history)
        if loaded_best is not None:
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
    print(f"  2. Interface sampling: n_points/2 → max(8000, n_points)")
    print(f"  3. Farfield sampling: n_points/10 → n_points/5")
    print(f"  4. Flux sign: UNCHANGED (analysis shows equivalent to FDM)")
    print(f"{'='*80}")

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model_thin.train()
        model_ext.train()

        n_points = min(8000 + (epoch - start_epoch) // 10 * 40, 15000)
        n_surface = max(5000, n_points)
        n_interface = max(8000, n_points)  # v9.6: 大幅增加界面采样
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

        # 3.6 Interface coupling - v9.6: 保持 v9.5 符号不变
        # 经分析，v9.5 的符号与 FDM 等价，只是定义不同
        T_int = torch.rand(n_interface, 1, device=device) * T_sim

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

        loss_interface = (torch.mean(flux_A_res**2) + 
                         torch.mean(flux_B_res**2) + 
                         torch.mean(flux_C_res**2) + 
                         torch.mean(flux_D_res**2))

        # ========== 4. Weight adjustment ==========
        if epoch < start_epoch + 3000:
            surface_weight = 20.0 * np.exp((epoch - start_epoch) / 3000)
        elif epoch < start_epoch + 10000:
            surface_weight = 60.0 * np.exp((epoch - start_epoch - 3000) / 2000)
        else:
            surface_weight = min(500.0, 150.0 * np.exp((epoch - start_epoch - 10000) / 1500))

        # v9.6: 大幅增加界面权重
        base_weights = {
            'pde_thin': 10.0,
            'pde_ext': 10.0,
            'surface': surface_weight,
            'farfield': 1.0,
            'initial': 1.0,
            'interface': 200.0,  # v9.6: 从 50 增加到 200
        }

        # 3.7 Total loss
        total_loss = (
            base_weights['pde_thin'] * loss_pde_thin +
            base_weights['pde_ext'] * loss_pde_ext +
            base_weights['surface'] * loss_surface +
            base_weights['farfield'] * loss_farfield +
            base_weights['initial'] * loss_initial +
            base_weights['interface'] * loss_interface
        )

        # ========== 5. Optimization ==========
        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ========== 6. Record ==========
        nernst_err = torch.sqrt(loss_nernst_l2).item()
        loss_history['total'].append(total_loss.item())
        loss_history['pde_thin'].append(loss_pde_thin.item())
        loss_history['pde_ext'].append(loss_pde_ext.item())
        loss_history['surface'].append(loss_surface.item())
        loss_history['farfield'].append(loss_farfield.item())
        loss_history['initial'].append(loss_initial.item())
        loss_history['interface'].append(loss_interface.item())
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_history['nernst_err'].append(nernst_err)
        loss_history['surface_state'].append(loss_surface_state.item())

        # ========== 7. Validation & Early stopping ==========
        if epoch % 500 == 0:
            val_results = validate_model(model_thin, model_ext, device, epoch, verbose=True)
            save_score = total_loss.item()

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
            print(f"  PDE_thin: {loss_pde_thin.item():.4e} | PDE_ext: {loss_pde_ext.item():.4e}")
            print(f"  Surface: {loss_surface.item():.4e} (w={base_weights['surface']:.2f})")
            print(f"  Surface state: {loss_surface_state.item():.4e}")
            print(f"  Interface: {loss_interface.item():.4e} (w={base_weights['interface']:.1f})")
            print(f"  Hard: A+B={err_AB:.2e}, C+D={err_CD:.2e}")
            print(f"  Nernst={nernst_err_test:.2e} | J_rxn={J_rxn_test:.4e}")
            print(f"  C_B(δ)={C_B_int_test.mean().item():.4f}, C_C(δ)={C_C_int_test.mean().item():.4f}")
            print(f"  Best physics score: {best_score:.2e} @ {best_epoch}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

            if epoch % 2000 == 0 and epoch > start_epoch:
                phys_results = verify_interface_physics(model_thin, model_ext, device)
                print_physics_verification(phys_results)

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
    parser.add_argument("--arch", choices=["legacy", "multiscale", "multiscale_hardbc"], default="legacy")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping and run exactly --epochs epochs.")
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

    model_thin, model_ext = create_models_v96(args.arch, normalize_inputs=USE_NORMALIZED_COORDS)
    model_thin = model_thin.to(device)
    model_ext = model_ext.to(device)

    total_params = (sum(p.numel() for p in model_thin.parameters()) + 
                   sum(p.numel() for p in model_ext.parameters()))
    print(f"Total trainable parameters: {total_params:,}")

    if args.eval_only:
        params = list(model_thin.parameters()) + list(model_ext.parameters())
        optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-7)
        loaded_epoch, loss_history, _ = load_model_v96(
            model_thin, model_ext, optimizer, scheduler, args.checkpoint
        )
        if loaded_epoch == 0 and not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        if not loss_history:
            loss_history = {
                'total': [], 'pde_thin': [], 'pde_ext': [], 'surface': [],
                'farfield': [], 'initial': [], 'interface': [], 'lr': [],
                'nernst_err': [], 'surface_state': []
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
        save_every=args.save_every
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
