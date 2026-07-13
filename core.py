"""Standalone ProductIntegral/Hermite/TraceGreen model.

This module intentionally has no dependency on pinn_thin_layer_v9_6.py.  FDM
data is not accepted anywhere in the model or its training path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
from torch import Tensor, nn


SIGMA = 40.0
THETA_I = 10.0
THETA_SWITCH = -10.0
T_SIM = 1.0
T_SWITCH = 0.5
DELTA = 0.035
X_EXT_MAX = 6.035
D_A = D_B = D_C = D_D = 1.0
FORMAT = "kg_productintegral_minimal_v1"


def potential(t: Tensor) -> Tensor:
    forward = THETA_I - 2.0 * (THETA_I - THETA_SWITCH) * t / T_SIM
    reverse = THETA_SWITCH + 2.0 * (THETA_I - THETA_SWITCH) * (t - T_SWITCH) / T_SIM
    return torch.where(t <= T_SWITCH, forward, reverse)


def surface_b(t: Tensor) -> Tensor:
    return torch.sigmoid(-potential(t))


def _interp(grid_t: Tensor, values: Tensor, query: Tensor) -> Tensor:
    q = query.reshape(-1).clamp(float(grid_t[0]), float(grid_t[-1]))
    idx = torch.searchsorted(grid_t, q, right=True).clamp(1, grid_t.numel() - 1)
    lo, hi = idx - 1, idx
    alpha = (q - grid_t[lo]) / (grid_t[hi] - grid_t[lo]).clamp_min(1e-12)
    out = values[lo] + alpha * (values[hi] - values[lo])
    return out.reshape(query.shape)


def hermite_basis(s: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00, h10, h01, h11


class FourierFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.tensor([1, 2, 4, 8, 16], dtype=torch.float32))

    @property
    def output_dim(self) -> int:
        return 22

    def forward(self, tx: Tensor) -> Tensor:
        phase = math.pi * tx[..., None] * self.frequencies
        return torch.cat((tx, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.norm(x + self.net(x)))


class ThinCorrection(nn.Module):
    def __init__(self, width: int = 256, depth: int = 5) -> None:
        super().__init__()
        self.features = FourierFeatures()
        self.input = nn.Linear(self.features.output_dim, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(depth))
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, t: Tensor, s: Tensor) -> Tensor:
        z = torch.tanh(self.input(self.features(torch.cat((t, s), dim=-1))))
        for block in self.blocks:
            z = block(z)
        return self.output(z)


@dataclass
class InterfaceState:
    c_b_surface: Tensor
    c_b_interface: Tensor
    c_c_interface: Tensor
    c_d_interface: Tensor
    flux: Tensor
    surface_slope: Tensor
    interface_slope: Tensor


class ProductIntegralInterface(nn.Module):
    """Causal Abel product-integration closure in normalized d=C_D/gamma."""

    def __init__(self, k: float, gamma: float, history_points: int = 256) -> None:
        super().__init__()
        self.k = self._positive(k, "k")
        self.gamma = self._positive(gamma, "gamma")
        self.history_points = max(32, int(history_points))
        self._cache: Dict[Tuple[Any, ...], Dict[str, Tensor]] = {}

    @staticmethod
    def _positive(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
        return value

    def set_conditions(self, k: float, gamma: float) -> None:
        self.k = self._positive(k, "k")
        self.gamma = self._positive(gamma, "gamma")
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _film(self, c_b_surface: Tensor, d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        c_c = (self.gamma * (1.0 - d)).clamp_min(torch.finfo(d.dtype).tiny)
        log_da = math.log(self.k * DELTA / D_B) + torch.log(c_c)
        transfer = torch.sigmoid(log_da)
        c_b_i = c_b_surface * (1.0 - transfer)
        flux = (D_B / DELTA) * c_b_surface * transfer
        return c_b_i, c_c, flux

    def _flux_derivative_d(self, c_b_surface: Tensor, d: Tensor) -> Tensor:
        c_c = (self.gamma * (1.0 - d)).clamp_min(torch.finfo(d.dtype).tiny)
        transfer = torch.sigmoid(math.log(self.k * DELTA / D_B) + torch.log(c_c))
        d_j_d_cc = (D_B / DELTA) * c_b_surface * transfer * (1.0 - transfer) / c_c
        return -self.gamma * d_j_d_cc

    def history(self, device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
        key = (device.type, device.index, str(dtype), self.k, self.gamma, self.history_points)
        if key in self._cache:
            return self._cache[key]

        with torch.no_grad():
            tg = torch.linspace(0.0, T_SIM, self.history_points, device=device, dtype=dtype)
            cb_s = surface_b(tg)
            d = torch.zeros_like(tg)
            flux = torch.zeros_like(tg)
            cb_i = cb_s.clone()
            c_c = torch.full_like(tg, self.gamma)
            sqrt_pi_d = math.sqrt(math.pi * D_D)

            cb_i[0], c_c[0], flux[0] = self._film(cb_s[0], d[0])
            for n in range(1, tg.numel()):
                tn = tg[n]
                completed = torch.zeros((), device=device, dtype=dtype)
                if n > 1:
                    a, b = tg[: n - 1], tg[1:n]
                    ja, jb = flux[: n - 1], flux[1:n]
                    h = (b - a).clamp_min(1e-12)
                    u0, u1 = torch.sqrt(tn - a), torch.sqrt(tn - b)
                    i0 = 2.0 * (u0 - u1)
                    i1 = 2.0 * ((tn - a) * (u0 - u1) - (u0**3 - u1**3) / 3.0) / h
                    completed = torch.sum(ja * i0 + (jb - ja) * i1)

                h_last = (tg[n] - tg[n - 1]).clamp_min(1e-12)
                a_last = 2.0 * torch.sqrt(h_last) / sqrt_pi_d
                fixed = completed / (self.gamma * sqrt_pi_d)
                dn = d[n - 1]
                for _ in range(2):
                    _, _, jn = self._film(cb_s[n], dn)
                    dn = (fixed + a_last * (flux[n - 1] / 3.0 + 2.0 * jn / 3.0) / self.gamma).clamp(0, 1)
                for _ in range(4):
                    _, _, jn = self._film(cb_s[n], dn)
                    f = dn - fixed - a_last * (flux[n - 1] / 3.0 + 2.0 * jn / 3.0) / self.gamma
                    fp = 1.0 - a_last * (2.0 / 3.0) * self._flux_derivative_d(cb_s[n], dn) / self.gamma
                    dn = (dn - f / fp.clamp_min(1e-8)).clamp(0, 1)
                d[n] = dn
                cb_i[n], c_c[n], flux[n] = self._film(cb_s[n], dn)

            c_d = self.gamma * d
            surface_slope = -flux / D_B
            interface_slope = -flux / D_B
            result = {
                "t": tg,
                "c_b_surface": cb_s,
                "c_b_interface": cb_i,
                "c_c_interface": c_c,
                "c_d_interface": c_d,
                "d": d,
                "flux": flux,
                "surface_slope": surface_slope,
                "interface_slope": interface_slope,
            }
        self._cache[key] = result
        return result

    def forward(self, t: Tensor) -> InterfaceState:
        h = self.history(t.device, t.dtype)
        get = lambda name: _interp(h["t"], h[name], t)
        return InterfaceState(
            get("c_b_surface"), get("c_b_interface"), get("c_c_interface"),
            get("c_d_interface"), get("flux"), get("surface_slope"), get("interface_slope")
        )


class HermiteThinLayer(nn.Module):
    def __init__(self, interface: ProductIntegralInterface) -> None:
        super().__init__()
        self.interface = interface
        self.correction = ThinCorrection()

    def forward_base(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        state = self.interface(t)
        s = (x / DELTA).clamp(0, 1)
        h00, h10, h01, h11 = hermite_basis(s)
        cb = (h00 * state.c_b_surface + h10 * DELTA * state.surface_slope
              + h01 * state.c_b_interface + h11 * DELTA * state.interface_slope)
        raw = self.correction(t, s)
        gate = (1.0 - torch.exp(-40.0 * t)).clamp(0, 1)
        cb = cb + gate * (0.25 * s * (1 - s) ** 2 * torch.tanh(raw[:, 1:2])
                          + 1.5 * s**2 * (1 - s) ** 2 * torch.tanh(raw[:, 0:1]))
        return 1.0 - cb, cb, raw

    def forward(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        ca, cb, _ = self.forward_base(t, x)
        return ca, cb

    def surface_current_base(self, t: Tensor) -> Tensor:
        state = self.interface(t)
        raw = self.correction(t, torch.zeros_like(t))
        gate = (1.0 - torch.exp(-40.0 * t)).clamp(0, 1)
        slope = state.surface_slope + (0.25 / DELTA) * gate * torch.tanh(raw[:, 1:2])
        return D_A * slope


class InventoryLift(nn.Module):
    """Physics-only causal inventory correction; it has no trainable parameters."""

    def __init__(self, thin: HermiteThinLayer, points: int = 4096) -> None:
        super().__init__()
        self.thin = thin
        self.points = max(256, int(points))
        self._cache: Dict[Tuple[Any, ...], Tuple[Tensor, Tensor, Tensor, Tensor]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _etd_step(a0: Tensor, source0: Tensor, source_slope: Tensor,
                  h: Tensor | float) -> Tensor:
        tau = DELTA**2 / 12.0
        rate = D_A / tau
        h_tensor = torch.as_tensor(h, device=a0.device, dtype=a0.dtype)
        one_minus_decay = -torch.expm1(-rate * h_tensor)
        decay = 1.0 - one_minus_decay
        return (decay * a0 + (one_minus_decay / D_A) * source0
                + (source_slope / D_A) * (h_tensor - one_minus_decay / rate))

    def _amplitude_grid(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        interface = self.thin.interface
        key = (device.type, device.index, str(dtype), interface.k, interface.gamma, self.points)
        if key in self._cache:
            return self._cache[key]
        with torch.enable_grad():
            tg = torch.linspace(0, T_SIM, self.points, device=device, dtype=dtype).reshape(-1, 1)
            tg.requires_grad_(True)
            nodes_np, weights_np = np.polynomial.legendre.leggauss(16)
            nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
            weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
            x_quad = 0.5 * DELTA * (nodes + 1.0)
            tq = tg.expand(-1, nodes.numel()).reshape(-1, 1)
            xq = x_quad.reshape(1, -1).expand(tg.shape[0], -1).reshape(-1, 1)
            _, cb_quad = self.thin(tq, xq)
            cb_quad = cb_quad.reshape(tg.shape[0], -1)
            inventory = 0.5 * DELTA * torch.sum(cb_quad * weights.reshape(1, -1), dim=1, keepdim=True)
            d_inventory_dt = torch.autograd.grad(inventory.sum(), tg, create_graph=False)[0]
            j_surface = self.thin.surface_current_base(tg)
            j_cons = -interface(tg).flux - d_inventory_dt
            source = j_cons - j_surface
            dt = float(T_SIM / (self.points - 1))
            source_slope = torch.cat((torch.zeros_like(source[:1]), (source[1:] - source[:-1]) / dt), dim=0)
            amp = torch.zeros_like(tg)
            for i in range(1, self.points):
                amp[i] = self._etd_step(amp[i - 1], source[i - 1], source_slope[i - 1], dt)
        self._cache[key] = (tg.detach().reshape(-1), source.detach().reshape(-1),
                            source_slope.detach().reshape(-1), amp.detach().reshape(-1))
        return self._cache[key]

    def amplitude(self, t: Tensor) -> Tensor:
        tg, source, source_slope, amp = self._amplitude_grid(t.device, t.dtype)
        q = t.clamp(0, T_SIM)
        dt = float(T_SIM / (self.points - 1))
        index = torch.floor(q / dt).long().clamp(0, self.points - 1)
        left = index.to(q.dtype) * dt
        flat = index.reshape(-1)
        a0 = amp[flat].reshape_as(q)
        s0 = source[flat].reshape_as(q)
        slope0 = source_slope[flat].reshape_as(q)
        return self._etd_step(a0, s0, slope0, q - left)

    def forward(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        ca, cb = self.thin(t, x)
        s = (x / DELTA).clamp(0, 1)
        _, h10, _, _ = hermite_basis(s)
        correction = DELTA * self.amplitude(t) * h10
        return ca - correction, cb + correction

    def surface_current(self, t: Tensor) -> Tensor:
        return self.thin.surface_current_base(t) + D_A * self.amplitude(t)


class TraceGreenExternal(nn.Module):
    """Dirichlet trace preserving half-line heat propagator."""

    def __init__(self, interface: ProductIntegralInterface, kernel_points: int = 64) -> None:
        super().__init__()
        self.interface = interface
        self.kernel_points = max(16, int(kernel_points))
        nodes = (torch.arange(self.kernel_points, dtype=torch.float64) + 0.5) / self.kernel_points
        self.register_buffer("u_nodes", nodes.to(torch.float32))

    def forward(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = (x - DELTA).clamp_min(0)
        state = self.interface(t)
        trace = state.c_d_interface
        eps = torch.finfo(t.dtype).eps
        u = self.u_nodes.to(device=t.device, dtype=t.dtype).reshape(1, -1)
        h = self.interface.history(t.device, t.dtype)

        def propagate(distance: Tensor) -> Tensor:
            t_pos = t.clamp_min(1e-6 * T_SIM)
            mass = torch.erfc(distance / torch.sqrt((4.0 * D_D * t_pos).clamp_min(eps)))
            transformed = (mass * u).clamp(1e-7, 1.0 - 1e-7)
            eta = torch.erfinv(1.0 - transformed)
            delay = distance**2 / (4.0 * D_D * eta**2).clamp_min(1e-8)
            tau = (t - delay).clamp(0, T_SIM)
            past = _interp(h["t"], h["c_d_interface"], tau)
            value = mass * past.mean(dim=1, keepdim=True)
            return torch.where(distance <= 1e-10, trace, value)

        cd_trace = propagate(y)
        far_distance = torch.full_like(y, X_EXT_MAX - DELTA)
        cd_far = propagate(far_distance)

        # A Hermite far-boundary patch preserves value and first derivative at y=0.
        length = X_EXT_MAX - DELTA
        beta = 1.0 + 7.0 * torch.sigmoid(torch.tensor(-2.0, device=t.device, dtype=t.dtype))
        z = (torch.expm1(beta * y / length) / torch.expm1(beta)).clamp(0, 1)
        _, _, h01, _ = hermite_basis(z)
        cd = cd_trace - h01 * cd_far
        return self.interface.gamma - cd, cd


class MinimalKGModel(nn.Module):
    """Minimal model bundle. Joint k/gamma inference always uses inventory lift."""

    def __init__(self, k: float, gamma: float, history_points: int = 256,
                 kernel_points: int = 64, lift_points: int = 4096) -> None:
        super().__init__()
        self.base_k = float(k)
        self.base_gamma = float(gamma)
        self.history_points = int(history_points)
        self.kernel_points = int(kernel_points)
        self.lift_points = int(lift_points)
        self.interface = ProductIntegralInterface(k, gamma, history_points)
        self.thin = HermiteThinLayer(self.interface)
        self.lift = InventoryLift(self.thin, lift_points)
        self.external = TraceGreenExternal(self.interface, kernel_points)

    def set_conditions(self, k: float, gamma: float) -> None:
        self.interface.set_conditions(k, gamma)
        self.lift.clear_cache()

    def forward_thin(self, t: Tensor, x: Tensor, lifted: bool = True) -> Tuple[Tensor, Tensor]:
        if not lifted:
            raise ValueError("Minimal KG inference requires inventory lift; lifted=False is disabled")
        return self.lift(t, x)

    def forward_external(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.external(t, x)

    def surface_current(self, t: Tensor) -> Tensor:
        return self.lift.surface_current(t)

    def training_thin(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Unlifted base field used only for fixed-condition physics training."""
        return self.thin(t, x)


def save_base_checkpoint(path: str | Path, model: MinimalKGModel, epoch: int,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         score: Optional[float] = None,
                         extra_parameters: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "format": FORMAT,
        "epoch": int(epoch),
        "score": score,
        "model_state_dict": model.state_dict(),
        "parameters": {
            "base_k": model.base_k,
            "base_gamma": model.base_gamma,
            "delta": DELTA,
            "training_data": "physics_only",
            "fdm_used_for_training": False,
            "inference": "productintegral_tracegreen_inventory_lift",
            "history_points": model.history_points,
            "kernel_points": model.kernel_points,
            "lift_points": model.lift_points,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_parameters:
        payload["parameters"].update(extra_parameters)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_base_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Tuple[MinimalKGModel, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT} checkpoint; old/dirty checkpoints are intentionally unsupported")
    params = payload.get("parameters", {})
    if params.get("fdm_used_for_training") is not False or params.get("training_data") != "physics_only":
        raise ValueError("Checkpoint does not certify physics-only training")
    model = MinimalKGModel(
        float(params["base_k"]),
        float(params["base_gamma"]),
        history_points=int(params.get("history_points", 256)),
        kernel_points=int(params.get("kernel_points", 64)),
        lift_points=int(params.get("lift_points", 4096)),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device), payload
