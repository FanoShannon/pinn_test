#!/usr/bin/env python3
"""Train one fixed-(k, gamma) base using physics residuals only."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

try:
    from .core import D_A, D_B, DELTA, MinimalKGModel, load_base_checkpoint, save_base_checkpoint
except ImportError:
    from core import D_A, D_B, DELTA, MinimalKGModel, load_base_checkpoint, save_base_checkpoint


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


def sample_points(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(count, 1, device=device, requires_grad=True)
    selector = torch.rand(count, 1, device=device)
    uniform = torch.rand(count, 1, device=device)
    near_surface = torch.rand(count, 1, device=device) ** 2
    near_interface = 1.0 - torch.rand(count, 1, device=device) ** 2
    s = torch.where(selector < 0.3, near_surface, torch.where(selector > 0.7, near_interface, uniform))
    x = (DELTA * s).requires_grad_(True)
    return t, x


def validation_points(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(count, device=device, dtype=torch.float32)
    t = ((index + 0.5) / count).reshape(-1, 1).requires_grad_(True)
    s = (((index * 37) % count + 0.5) / count).reshape(-1, 1)
    x = (DELTA * s).requires_grad_(True)
    return t, x


def physics_loss(model: MinimalKGModel, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    ca, cb = model.training_thin(t, x)
    ca_t, cb_t = derivative(ca, t), derivative(cb, t)
    ca_x, cb_x = derivative(ca, x), derivative(cb, x)
    ca_xx, cb_xx = derivative(ca_x, x), derivative(cb_x, x)
    # Equivalent diffusion-time residual in s=x/delta coordinates. This removes
    # the artificial delta^-4 loss amplification without changing the PDE.
    r_a = (DELTA**2 / D_A) * ca_t - DELTA**2 * ca_xx
    r_b = (DELTA**2 / D_B) * cb_t - DELTA**2 * cb_xx
    pde = r_a.square().mean() + r_b.square().mean()
    bounds = (torch.relu(-ca).square().mean() + torch.relu(ca - 1).square().mean()
              + torch.relu(-cb).square().mean() + torch.relu(cb - 1).square().mean())
    conservation = (ca + cb - 1).square().mean()
    total = pde + 30.0 * bounds + 10.0 * conservation
    metrics = {"loss": float(total.detach()), "pde": float(pde.detach()),
               "bounds": float(bounds.detach()), "conservation": float(conservation.detach())}
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--stage-name", default="base")
    parser.add_argument("--early-stop-check-every", type=int, default=100)
    parser.add_argument("--early-stop-min-epochs", type=int, default=1500)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--early-stop-ema-alpha", type=float, default=0.5)
    parser.add_argument("--validation-points", type=int, default=96)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("base training requires --epochs >= 1")
    if args.early_stop_check_every < 1 or args.validation_points < 2:
        raise ValueError("early-stop check interval must be >=1 and validation points >=2")
    if not 0.0 < args.early_stop_ema_alpha <= 1.0:
        raise ValueError("early-stop EMA alpha must be in (0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.warm_start:
        model, warm_payload = load_base_checkpoint(args.warm_start, device)
        if not math.isclose(model.base_k, args.k, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Warm-start k={model.base_k} does not match requested k={args.k}")
        if not math.isclose(model.base_gamma, args.gamma, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"Warm-start gamma={model.base_gamma} does not match requested gamma={args.gamma}"
            )
        print(f"Warm-start weights: {args.warm_start} (epoch={warm_payload.get('epoch')})", flush=True)
    else:
        model = MinimalKGModel(args.k, args.gamma).to(device)
    trainable = list(model.thin.correction.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    best = math.inf
    best_ema = math.inf
    score_ema = None
    stale_checks = 0
    log_path = args.output_dir / "training.jsonl"
    checkpoint_parameters = {
        "training_stage": args.stage_name,
        "epoch_budget": args.epochs,
        "learning_rate": args.lr,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "checkpoint_selection": "deterministic_physics_validation_v1",
        "validation_points": args.validation_points,
    }
    print(f"Physics-only {args.stage_name}: k={args.k:g}, gamma={args.gamma:g}, device={device}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t, x = sample_points(args.points, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = physics_loss(model, t, x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite physics loss at epoch {epoch}: {metrics}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}")
            optimizer.step()
            scheduler.step()
            metrics.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm))

            check_validation = (
                epoch == 1 or epoch % args.early_stop_check_every == 0 or epoch == args.epochs
            )
            should_stop = False
            if check_validation:
                model.eval()
                tv, xv = validation_points(args.validation_points, device)
                validation, validation_metrics = physics_loss(model, tv, xv)
                score = float(validation.detach())
                if not math.isfinite(score):
                    raise FloatingPointError(f"Non-finite deterministic physics score at epoch {epoch}")
                score_ema = score if score_ema is None else (
                    args.early_stop_ema_alpha * score
                    + (1.0 - args.early_stop_ema_alpha) * score_ema
                )
                if score < best:
                    best = score
                    save_base_checkpoint(
                        args.output_dir / "base_best.pth", model, epoch, optimizer, best,
                        extra_parameters=checkpoint_parameters,
                    )
                threshold = best_ema * (1.0 - args.early_stop_min_relative_improvement)
                if not math.isfinite(best_ema) or score_ema < threshold:
                    best_ema = score_ema
                    stale_checks = 0
                elif epoch >= args.early_stop_min_epochs:
                    stale_checks += 1
                metrics.update(validation_score=score, validation_ema=score_ema,
                               validation_pde=validation_metrics["pde"], stale_checks=stale_checks)
                print(f"  physics_validation={score:.4e} ema={score_ema:.4e} "
                      f"best={best:.4e} stale={stale_checks}/{args.early_stop_patience}", flush=True)
                should_stop = (
                    not args.no_early_stop
                    and epoch >= args.early_stop_min_epochs
                    and stale_checks >= args.early_stop_patience
                )

            log.write(json.dumps(metrics) + "\n")
            log.flush()

            if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs or should_stop:
                save_base_checkpoint(
                    args.output_dir / "base_current.pth", model, epoch, optimizer,
                    metrics["loss"], extra_parameters=checkpoint_parameters,
                )
                print(f"epoch={epoch:5d} loss={metrics['loss']:.4e} pde={metrics['pde']:.4e} "
                      f"bounds={metrics['bounds']:.3e} grad={metrics['grad_norm']:.3e}", flush=True)
            if should_stop:
                print(f"Physics early stop at epoch {epoch}", flush=True)
                break

    print(f"Best checkpoint: {args.output_dir / 'base_best.pth'}", flush=True)


if __name__ == "__main__":
    main()
