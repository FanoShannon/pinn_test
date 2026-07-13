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


def build_models(
    arch,
    normalize_inputs,
    device,
    green_time_grid=256,
    green_kernel_points=32,
    green_history_grad=True,
    lift_time_grid=1024,
):
    if arch == "legacy":
        model_thin = pinn.ThinLayerNet_v9_6(normalize_inputs=normalize_inputs)
        model_ext = pinn.ExternalNet_v9_6(pinn.gamma, normalize_inputs=normalize_inputs)
    elif arch == "multiscale":
        model_thin = pinn.ThinLayerNet_v9_6_Multiscale(normalize_inputs=normalize_inputs)
        model_ext = pinn.ExternalNet_v9_6_Multiscale(pinn.gamma, normalize_inputs=normalize_inputs)
    elif arch == "multiscale_hardbc":
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHardBC(normalize_inputs=normalize_inputs)
        model_ext = pinn.ExternalNet_v9_6_Multiscale(pinn.gamma, normalize_inputs=normalize_inputs)
    elif arch == "his_pinn":
        interface_state = pinn.HISInterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_HISPrototype(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_Multiscale(
            pinn.gamma,
            normalize_inputs=normalize_inputs,
        )
    elif arch == "his_pinn_ext":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleHISLayer(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
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
    elif arch == "multiscale_hermite_extbasis":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleHermiteExtBasis(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
    elif arch == "multiscale_green":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenKernel(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
    elif arch == "multiscale_green_grid":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGrid(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_hybrid":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridHybrid(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_dynamic":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamic(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_dynamic_stage1":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        interface_state.continuous_time_features = True
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamicStage1(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_interface_memory":
        interface_state = pinn.InterfaceStateNet_v9_6_Memory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamic(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_memory":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridMemory(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbel(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamic(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMix(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamic(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causal":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausal(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamic(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalconv":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalConv(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalConv(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybrid(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybrid(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridSmooth(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybridSmooth(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridDynamicCausalHybridSmooth(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_fluxtrace":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridFluxTrace(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemory(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemoryMatchedAbel(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelKernelMixCausalHybridIntMemoryMixedAbel(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridFilmTrace(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_film_tracegreen_clean":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceClean(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenClean(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_film_tracegreen_productintegral":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceProductIntegral(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenClean(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_film_tracegreen_productintegral_lift":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceProductIntegral(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_InventoryHermiteLift(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            lift_time_grid_points=lift_time_grid,
        )
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenClean(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch in {
        "multiscale_film_tracegreen_clean_conservative",
        "multiscale_film_tracegreen_clean_mixedflux",
        "multiscale_film_tracegreen_conservative_lift",
    }:
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceClean(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        if arch.endswith("mixedflux"):
            thin_class = pinn.ThinLayerNet_v9_6_MultiscaleHermiteMixedFlux
        elif arch.endswith("conservative_lift"):
            thin_class = pinn.ThinLayerNet_v9_6_InventoryHermiteLift
        else:
            thin_class = pinn.ThinLayerNet_v9_6_MultiscaleHermiteConservative
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("conservative_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        model_thin = thin_class(**thin_kwargs)
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenClean(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch in {
        "multiscale_film_tracegreen_kparam",
        "multiscale_film_tracegreen_kparam_lift",
    }:
        if not np.isclose(pinn.gamma, pinn.REFERENCE_GAMMA, rtol=0.0, atol=1e-12):
            raise ValueError("multiscale_film_tracegreen_kparam requires gamma=10")
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceKParam(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        thin_class = (
            pinn.ThinLayerNet_v9_6_InventoryHermiteLiftKParam
            if arch.endswith("_lift")
            else pinn.ThinLayerNet_v9_6_MultiscaleHermiteKParam
        )
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        model_thin = thin_class(**thin_kwargs)
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenKParam(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch in {
        "multiscale_film_tracegreen_gammaparam",
        "multiscale_film_tracegreen_gammaparam_lift",
    }:
        if not np.isclose(pinn.k_cat_star, pinn.REFERENCE_K_CAT_STAR, rtol=0.0, atol=1e-12):
            raise ValueError("multiscale_film_tracegreen_gammaparam requires k_cat=1")
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceGammaParam(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        interface_state.set_gamma(pinn.gamma)
        thin_class = (
            pinn.ThinLayerNet_v9_6_InventoryHermiteLiftGammaParam
            if arch.endswith("_lift")
            else pinn.ThinLayerNet_v9_6_MultiscaleHermiteGammaParam
        )
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        model_thin = thin_class(**thin_kwargs)
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenGammaParam(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch in {
        "multiscale_film_tracegreen_kgparam",
        "multiscale_film_tracegreen_kgparam_lift",
    }:
        interface_state = pinn.InterfaceStateNet_v9_6_FilmTraceKGParam(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        interface_state.set_conditions(pinn.k_cat_star, pinn.gamma)
        thin_class = (
            pinn.ThinLayerNet_v9_6_InventoryHermiteLiftKGParam
            if arch.endswith("_lift")
            else pinn.ThinLayerNet_v9_6_MultiscaleHermiteKGParam
        )
        thin_kwargs = {
            "interface_state": interface_state,
            "normalize_inputs": normalize_inputs,
        }
        if arch.endswith("_lift"):
            thin_kwargs["lift_time_grid_points"] = lift_time_grid
        model_thin = thin_class(**thin_kwargs)
        model_ext = pinn.ExternalNet_v9_6_FilmTraceGreenKGParam(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_green_grid_film_abel_ema":
        interface_state = pinn.InterfaceStateNet_v9_6_FilmAbelEMA(
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
        )
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleGreenGridMemory(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
            time_grid_points=green_time_grid,
            kernel_points=green_kernel_points,
            history_grad=green_history_grad,
            cache_history=False,
        )
    elif arch == "multiscale_buffer":
        interface_state = pinn.InterfaceStateNet_v9_6(normalize_inputs=normalize_inputs)
        model_thin = pinn.ThinLayerNet_v9_6_MultiscaleHermite(
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
        model_ext = pinn.ExternalNet_v9_6_MultiscaleBuffer(
            pinn.gamma,
            interface_state=interface_state,
            normalize_inputs=normalize_inputs,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model_thin.to(device), model_ext.to(device)


def load_checkpoint(model_thin, model_ext, checkpoint, allow_fixed_reference=False):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pinn.validate_checkpoint_physical_parameters(
        state,
        checkpoint,
        model_ext=model_ext,
        evaluation_k=pinn.model_active_k_cat(model_ext=model_ext),
        evaluation_gamma=pinn.model_active_gamma(model_ext=model_ext),
        allow_fixed_reference=allow_fixed_reference,
    )
    try:
        model_thin.load_state_dict(state["model_thin_state_dict"], strict=True)
        model_ext.load_state_dict(state["model_ext_state_dict"], strict=True)
    except RuntimeError:
        pinn.load_compatible_state_dict(model_thin, state["model_thin_state_dict"], "Thin model")
        pinn.load_compatible_state_dict(model_ext, state["model_ext_state_dict"], "External model")
    return state.get("epoch", None)


def configure_evaluation_parameters(args, fdm):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_params = checkpoint.get("parameters", {})
    fdm_params = fdm.get("params", {})
    is_kparam_checkpoint = (
        checkpoint_params.get("parameterization") == pinn.KPARAM_PARAMETERIZATION
    )
    is_gammaparam_checkpoint = (
        checkpoint_params.get("parameterization") == pinn.GAMMAPARAM_PARAMETERIZATION
    )
    is_kgparam_checkpoint = (
        checkpoint_params.get("parameterization") == pinn.KGPARAM_PARAMETERIZATION
    )
    target_kparam = "kparam" in args.arch
    target_gammaparam = "gammaparam" in args.arch
    target_kgparam = "kgparam" in args.arch

    gamma_value = args.gamma
    if gamma_value is None:
        if is_kgparam_checkpoint or target_kgparam or is_gammaparam_checkpoint or target_gammaparam:
            gamma_value = fdm_params.get("gamma", pinn.GAMMAPARAM_REFERENCE)
        else:
            gamma_value = checkpoint_params.get("gamma", fdm_params.get("gamma", pinn.REFERENCE_GAMMA))

    k_cat_value = args.k_cat_star
    if k_cat_value is None:
        fdm_k = fdm_params.get("k_cat", fdm_params.get("k_cat_star"))
        if is_kgparam_checkpoint or target_kgparam or is_kparam_checkpoint or target_kparam:
            k_cat_value = pinn.KPARAM_REFERENCE if fdm_k is None else fdm_k
        elif is_gammaparam_checkpoint or target_gammaparam:
            k_cat_value = pinn.REFERENCE_K_CAT_STAR
        else:
            k_cat_value = checkpoint_params.get(
                "k_cat_star",
                pinn.REFERENCE_K_CAT_STAR if fdm_k is None else fdm_k,
            )

    pinn.configure_physical_parameters(gamma_value, k_cat_value)

    expected = {
        "gamma": float(pinn.gamma),
        "k_cat": float(pinn.k_cat_star),
    }
    for name, active in expected.items():
        stored = fdm_params.get(name)
        if stored is None and name == "k_cat":
            stored = fdm_params.get("k_cat_star")
        if stored is not None and not np.isclose(float(stored), active, rtol=1e-7, atol=1e-10):
            raise ValueError(
                f"FDM/PINN parameter mismatch for {name}: FDM={stored}, PINN={active}."
            )
    return checkpoint


def select_indices(size, n_select):
    n_select = min(size, n_select)
    return np.unique(np.linspace(0, size - 1, n_select, dtype=int))


def predict_pair(model, t_eval, x_eval, device, batch_size):
    tt, xx = np.meshgrid(t_eval, x_eval, indexing="ij")
    points = np.column_stack([tt.ravel(), xx.ravel()]).astype("float32")
    if pinn.is_joint_parameterized(model_thin=model, model_ext=model):
        k_value = pinn.model_active_k_cat(model_thin=model, model_ext=model)
        gamma_value = pinn.model_active_gamma(model_thin=model, model_ext=model)
        points = np.column_stack([
            points,
            np.full(len(points), k_value, dtype="float32"),
            np.full(len(points), gamma_value, dtype="float32"),
        ])
    elif getattr(model, "k_parameterized", False):
        k_value = pinn.model_active_k_cat(model_thin=model, model_ext=model)
        points = np.column_stack([
            points,
            np.full(len(points), k_value, dtype="float32"),
        ])
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


def predict_surface_current(model_thin, t_eval, device, batch_size):
    chunks = []
    model_thin.eval()
    for start in range(0, len(t_eval), batch_size):
        t_batch = torch.from_numpy(t_eval[start:start + batch_size].reshape(-1, 1).astype("float32")).to(device)
        x_batch = torch.zeros_like(t_batch, requires_grad=True)
        with torch.enable_grad():
            inputs = pinn.conditioned_model_inputs(model_thin, t_batch, x_batch)
            c_a, _ = model_thin(inputs)
            c_a_x = torch.autograd.grad(c_a.sum(), x_batch, create_graph=False)[0]
        chunks.append((-c_a_x).detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def predict_current_components(model_thin, t_eval, device, batch_size, quadrature_points=16):
    outputs = {
        "surface": [],
        "reaction": [],
        "inventory": [],
        "conservative": [],
        "balance_residual": [],
    }
    model_thin.eval()
    for start in range(0, len(t_eval), batch_size):
        t_batch = torch.from_numpy(
            t_eval[start:start + batch_size].reshape(-1, 1).astype("float32")
        ).to(device)
        t_batch.requires_grad_(True)
        with torch.enable_grad():
            current = pinn.thin_current_components(
                model_thin,
                t_batch,
                quadrature_points=quadrature_points,
                create_graph=False,
            )
        outputs["surface"].append(current["J_surface"].detach().cpu().numpy().reshape(-1))
        outputs["reaction"].append(current["J_reaction"].detach().cpu().numpy().reshape(-1))
        outputs["inventory"].append(current["J_inventory"].detach().cpu().numpy().reshape(-1))
        outputs["conservative"].append(current["J_conservative"].detach().cpu().numpy().reshape(-1))
        outputs["balance_residual"].append(current["balance_residual"].detach().cpu().numpy().reshape(-1))
    return {name: np.concatenate(chunks) for name, chunks in outputs.items()}


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


def build_cv_eval(model_thin, fdm, device, batch_size, n_cv, current_mode="both"):
    if "theta" not in fdm or "J" not in fdm or "t" not in fdm:
        return None
    t_all = np.asarray(fdm["t"], dtype=float)
    theta_all = np.asarray(fdm["theta"], dtype=float)
    j_fdm_all = np.asarray(fdm["J"], dtype=float)
    cv_idx = select_indices(len(t_all), n_cv)
    t_cv = t_all[cv_idx]
    theta_cv = theta_all[cv_idx]
    j_fdm = j_fdm_all[cv_idx]
    current = predict_current_components(model_thin, t_cv, device, batch_size)
    result = {
        "t": t_cv,
        "theta": theta_cv,
        "fdm_J": j_fdm,
        "pinn_J": current["surface"],
        "pinn_J_surface": current["surface"],
        "pinn_J_reaction": current["reaction"],
        "pinn_J_inventory": current["inventory"],
        "pinn_J_conservative": current["conservative"],
        "pinn_J_balance_residual": current["balance_residual"],
        "current_mode": current_mode,
    }
    return result


def compare(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalize_inputs = args.input_mode == "normalized"

    with open(args.fdm_pkl, "rb") as handle:
        fdm = pickle.load(handle)
    configure_evaluation_parameters(args, fdm)

    model_thin, model_ext = build_models(
        args.arch,
        normalize_inputs,
        device,
        green_time_grid=args.green_time_grid,
        green_kernel_points=args.green_kernel_points,
        green_history_grad=not args.green_detach_history,
        lift_time_grid=args.lift_time_grid,
    )
    if pinn.is_k_parameterized(model_ext=model_ext):
        pinn.set_model_k_cat(model_thin, model_ext, pinn.k_cat_star)
    if pinn.is_gamma_parameterized(model_ext=model_ext):
        pinn.set_model_gamma(model_thin, model_ext, pinn.gamma)
    epoch = load_checkpoint(
        model_thin,
        model_ext,
        args.checkpoint,
        allow_fixed_reference=getattr(args, "zero_shot_fixed_reference", False),
    )

    t_fdm = np.asarray(fdm["t"], dtype=float)
    x_in_fdm = np.asarray(fdm["x_in"], dtype=float)
    x_out_fdm = np.asarray(fdm["x_out"], dtype=float)
    fdm_conc = fdm["concentrations"]
    if not np.isclose(x_out_fdm[-1], pinn.X_ext_max, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "FDM/PINN external-domain mismatch: "
            f"FDM x_max={x_out_fdm[-1]:.8g}, PINN x_max={pinn.X_ext_max:.8g}. "
            "Regenerate FDM with a gamma-independent external length."
        )

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

    concentration_names = ["C_A", "C_B", "C_C", "C_D"]
    metrics = {name: field_metrics(pinn_eval[name], fdm_eval[name]) for name in concentration_names}
    total_sq = sum(
        float(np.sum((pinn_eval[name] - fdm_eval[name]) ** 2))
        for name in concentration_names
    )
    total_n = sum(pinn_eval[name].size for name in concentration_names)
    metrics["overall"] = {"rmse": float(np.sqrt(total_sq / total_n))}
    dimensionless_errors = [
        pinn_eval["C_A"] - fdm_eval["C_A"],
        pinn_eval["C_B"] - fdm_eval["C_B"],
        (pinn_eval["C_C"] - fdm_eval["C_C"]) / pinn.gamma,
        (pinn_eval["C_D"] - fdm_eval["C_D"]) / pinn.gamma,
    ]
    metrics["overall_dimensionless"] = {
        "rmse": float(np.sqrt(
            sum(float(np.sum(error ** 2)) for error in dimensionless_errors) /
            sum(error.size for error in dimensionless_errors)
        ))
    }
    metrics["C_C_over_gamma"] = field_metrics(
        pinn_eval["C_C"] / pinn.gamma,
        fdm_eval["C_C"] / pinn.gamma,
    )
    metrics["C_D_over_gamma"] = field_metrics(
        pinn_eval["C_D"] / pinn.gamma,
        fdm_eval["C_D"] / pinn.gamma,
    )

    c_b_int_fdm = np.asarray(fdm_conc["C_B"], dtype=float)[np.ix_(t_idx, [len(x_in_fdm) - 1])].reshape(-1)
    c_c_int_fdm = np.asarray(fdm_conc["C_C"], dtype=float)[np.ix_(t_idx, [0])].reshape(-1)
    metrics["C_B_int"] = field_metrics(pinn_b[:, -1], c_b_int_fdm)
    metrics["C_C_int"] = field_metrics(pinn_c[:, 0], c_c_int_fdm)
    metrics["C_C_int_over_gamma"] = field_metrics(
        pinn_c[:, 0] / pinn.gamma,
        c_c_int_fdm / pinn.gamma,
    )
    metrics["C_C_int_history_note"] = (
        "C_C_int is evaluated from the FDM concentration field C_C[:,0], "
        "not from the stored C_C_int history array, whose first element is uninitialized."
    )
    cv_eval = build_cv_eval(
        model_thin,
        fdm,
        device,
        args.batch_size,
        args.cv_points,
        current_mode=args.current_mode,
    )
    if cv_eval is not None:
        metrics["CV_J"] = field_metrics(cv_eval["pinn_J"], cv_eval["fdm_J"])
        metrics["CV_J_surface"] = field_metrics(
            cv_eval["pinn_J_surface"], cv_eval["fdm_J"]
        )
        metrics["CV_J_conservative"] = field_metrics(
            cv_eval["pinn_J_conservative"], cv_eval["fdm_J"]
        )
        metrics["CV_J_surface_vs_conservative"] = field_metrics(
            cv_eval["pinn_J_surface"], cv_eval["pinn_J_conservative"]
        )
        j_ref = pinn.characteristic_reaction_flux()
        metrics["CV_J_over_J_ref"] = field_metrics(
            cv_eval["pinn_J"] / j_ref,
            cv_eval["fdm_J"] / j_ref,
        )
        fdm_cathodic = int(np.argmin(cv_eval["fdm_J"]))
        pinn_cathodic = int(np.argmin(cv_eval["pinn_J"]))
        fdm_anodic = int(np.argmax(cv_eval["fdm_J"]))
        pinn_anodic = int(np.argmax(cv_eval["pinn_J"]))
        metrics["CV_peaks"] = {
            "cathodic": {
                "fdm_J": float(cv_eval["fdm_J"][fdm_cathodic]),
                "pinn_J": float(cv_eval["pinn_J"][pinn_cathodic]),
                "J_error": float(cv_eval["pinn_J"][pinn_cathodic] - cv_eval["fdm_J"][fdm_cathodic]),
                "fdm_theta": float(cv_eval["theta"][fdm_cathodic]),
                "pinn_theta": float(cv_eval["theta"][pinn_cathodic]),
                "theta_error": float(cv_eval["theta"][pinn_cathodic] - cv_eval["theta"][fdm_cathodic]),
            },
            "anodic": {
                "fdm_J": float(cv_eval["fdm_J"][fdm_anodic]),
                "pinn_J": float(cv_eval["pinn_J"][pinn_anodic]),
                "J_error": float(cv_eval["pinn_J"][pinn_anodic] - cv_eval["fdm_J"][fdm_anodic]),
                "fdm_theta": float(cv_eval["theta"][fdm_anodic]),
                "pinn_theta": float(cv_eval["theta"][pinn_anodic]),
                "theta_error": float(cv_eval["theta"][pinn_anodic] - cv_eval["theta"][fdm_anodic]),
            },
        }

    if args.output_figure:
        plot_residual_summary(
            args.output_figure,
            t_eval,
            x_in_eval,
            x_out_eval,
            fdm_eval,
            pinn_eval,
            metrics,
            cv_eval=cv_eval,
        )
    if args.output_current_figure and cv_eval is not None:
        plot_current_decomposition(args.output_current_figure, cv_eval)

    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            t=t_eval,
            x_in=x_in_eval,
            x_out=x_out_eval,
            **{f"fdm_{k}": v for k, v in fdm_eval.items()},
            **{f"pinn_{k}": v for k, v in pinn_eval.items()},
            **(
                {f"cv_{k}": v for k, v in cv_eval.items() if k != "current_mode"}
                if cv_eval is not None else {}
            ),
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {epoch}")
    print(f"Architecture: {args.arch}")
    print(f"Input mode: {args.input_mode}")
    print(f"Parameters: gamma={pinn.gamma}, k_cat_star={pinn.k_cat_star}")
    print(f"Device: {device}")
    print("field,rmse,mae,max_abs,bias,r2,nrmse,fdm_min,fdm_max,pinn_min,pinn_max")
    for name in [
        "C_A", "C_B", "C_C", "C_D", "C_B_int", "C_C_int",
        "C_C_over_gamma", "C_D_over_gamma", "C_C_int_over_gamma",
    ]:
        row = metrics[name]
        print(
            f"{name},{row['rmse']:.8e},{row['mae']:.8e},{row['max_abs']:.8e},"
            f"{row['bias']:.8e},{row['r2']:.8e},{row['nrmse']:.8e},"
            f"{row['fdm_min']:.8e},{row['fdm_max']:.8e},"
            f"{row['pinn_min']:.8e},{row['pinn_max']:.8e}"
        )
    print(f"overall_rmse,{metrics['overall']['rmse']:.8e}")
    print(f"overall_dimensionless_rmse,{metrics['overall_dimensionless']['rmse']:.8e}")
    if "CV_J" in metrics:
        for name in [
            "CV_J", "CV_J_surface", "CV_J_conservative",
            "CV_J_surface_vs_conservative", "CV_J_over_J_ref",
        ]:
            row = metrics[name]
            print(
                f"{name},{row['rmse']:.8e},{row['mae']:.8e},{row['max_abs']:.8e},"
                f"{row['bias']:.8e},{row['r2']:.8e},{row['nrmse']:.8e},"
                f"{row['fdm_min']:.8e},{row['fdm_max']:.8e},"
                f"{row['pinn_min']:.8e},{row['pinn_max']:.8e}"
            )
    return metrics


def plot_residual_summary(
    path,
    t_eval,
    x_in_eval,
    x_out_eval,
    fdm_eval,
    pinn_eval,
    metrics,
    cv_eval=None,
):
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

    ax = axes[1, 3]
    if cv_eval is not None:
        ax.plot(cv_eval["theta"], cv_eval["fdm_J"], color="blue", linewidth=2.2, label="FDM")
        if cv_eval.get("current_mode") in {"surface", "both"}:
            ax.plot(
                cv_eval["theta"], cv_eval["pinn_J_surface"],
                color="#F58518", linewidth=1.8, linestyle="--", label="PINN surface"
            )
        if cv_eval.get("current_mode") in {"conservative", "both"}:
            ax.plot(
                cv_eval["theta"], cv_eval["pinn_J_conservative"],
                color="#54A24B", linewidth=1.8, linestyle=":", label="PINN conservative"
            )
        fdm_pc = int(np.argmin(cv_eval["fdm_J"]))
        fdm_pa = int(np.argmax(cv_eval["fdm_J"]))
        ax.scatter(cv_eval["theta"][fdm_pc], cv_eval["fdm_J"][fdm_pc], color="red", s=28, zorder=3)
        ax.scatter(cv_eval["theta"][fdm_pa], cv_eval["fdm_J"][fdm_pa], color="green", s=28, zorder=3)
        ax.set_title("CV from current model")
        ax.set_xlabel("Potential theta")
        ax.set_ylabel("Current J")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "CV data not found",
            ha="center",
            va="center",
            fontsize=12,
            color="0.35",
        )

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_current_decomposition(path, cv_eval):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = cv_eval["t"]
    theta = cv_eval["theta"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(theta, cv_eval["fdm_J"], color="blue", linewidth=2.2, label="FDM")
    ax.plot(theta, cv_eval["pinn_J_surface"], color="#F58518", linestyle="--", label="surface")
    ax.plot(theta, cv_eval["pinn_J_conservative"], color="#54A24B", linestyle=":", label="conservative")
    ax.set_title("CV current readouts")
    ax.set_xlabel("Potential theta")
    ax.set_ylabel("Current J")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, cv_eval["pinn_J_reaction"], label=r"$-J_{rxn}$")
    ax.plot(t, cv_eval["pinn_J_inventory"], label=r"$-dM_B/dt$")
    ax.plot(t, cv_eval["pinn_J_conservative"], color="black", linewidth=2, label="sum")
    ax.set_title("Conservative current decomposition")
    ax.set_xlabel("T")
    ax.set_ylabel("Current contribution")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, cv_eval["pinn_J_balance_residual"], color="#E45756")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Surface - conservative current")
    ax.set_xlabel("T")
    ax.set_ylabel("Balance residual")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, cv_eval["pinn_J_surface"] - cv_eval["fdm_J"], label="surface - FDM")
    ax.plot(t, cv_eval["pinn_J_conservative"] - cv_eval["fdm_J"], label="conservative - FDM")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Posterior current errors")
    ax.set_xlabel("T")
    ax.set_ylabel("PINN - FDM")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PINN C_A/C_B/C_C/C_D against exact downsampled FDM grid points."
    )
    parser.add_argument("--fdm-pkl", default="../FDM/kcat1_v42_thin_layer_catalytic_v42.pkl")
    parser.add_argument("--checkpoint", default="./pinn_thin_layer_catalytic_v9_6_best.pth")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Fixed gamma. Defaults to checkpoint metadata, then FDM metadata.")
    parser.add_argument("--k-cat-star", type=float, default=None,
                        help="Fixed k_cat*. Defaults to checkpoint metadata, then FDM metadata.")
    parser.add_argument("--zero-shot-fixed-reference", action="store_true",
                        help="Allow a fixed reference checkpoint to seed parameterized zero-shot evaluation.")
    parser.add_argument("--arch", choices=["legacy", "multiscale", "multiscale_hardbc", "his_pinn", "his_pinn_ext", "multiscale_hermite", "multiscale_hermite_extbasis", "multiscale_green", "multiscale_green_grid", "multiscale_green_grid_hybrid", "multiscale_green_grid_dynamic", "multiscale_green_grid_dynamic_stage1", "multiscale_green_grid_interface_memory", "multiscale_green_grid_memory", "multiscale_green_grid_film_abel", "multiscale_green_grid_film_abel_kernelmix", "multiscale_green_grid_film_abel_kernelmix_causal", "multiscale_green_grid_film_abel_kernelmix_causalconv", "multiscale_green_grid_film_abel_kernelmix_causalhybrid", "multiscale_green_grid_film_abel_kernelmix_causalhybrid_smooth", "multiscale_green_grid_film_abel_kernelmix_causalhybrid_intmemory", "multiscale_green_grid_film_abel_kernelmix_fluxtrace", "multiscale_green_grid_film_abel_kernelmix_tracegreen", "multiscale_green_grid_film_abel_kernelmix_tracegreen_matchedabel", "multiscale_green_grid_film_abel_kernelmix_tracegreen_mixedabel", "multiscale_film_tracegreen_clean", "multiscale_film_tracegreen_productintegral", "multiscale_film_tracegreen_productintegral_lift", "multiscale_film_tracegreen_clean_conservative", "multiscale_film_tracegreen_clean_mixedflux", "multiscale_film_tracegreen_conservative_lift", "multiscale_film_tracegreen_kparam", "multiscale_film_tracegreen_kparam_lift", "multiscale_film_tracegreen_gammaparam", "multiscale_film_tracegreen_gammaparam_lift", "multiscale_film_tracegreen_kgparam", "multiscale_film_tracegreen_kgparam_lift", "multiscale_green_grid_film_abel_ema", "multiscale_buffer"], default="legacy")
    parser.add_argument("--green-time-grid", type=int, default=256)
    parser.add_argument("--green-kernel-points", type=int, default=32)
    parser.add_argument("--lift-time-grid", type=int, default=1024)
    parser.add_argument("--green-detach-history", action="store_true")
    parser.add_argument("--input-mode", choices=["legacy", "normalized"], default="legacy")
    parser.add_argument("--n-time", type=int, default=160)
    parser.add_argument("--n-x-in", type=int, default=120)
    parser.add_argument("--n-x-out", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-json", default="./concentration_compare_metrics.json")
    parser.add_argument("--output-npz", default="./concentration_compare_fields.npz")
    parser.add_argument("--output-figure", default="./concentration_residual_summary.png")
    parser.add_argument("--output-current-figure", default="",
                        help="Optional current decomposition and conservation diagnostic PNG.")
    parser.add_argument("--current-mode", choices=["surface", "conservative", "both"], default="both",
                        help="Current curves shown in figures; CV_J remains the surface-current metric.")
    parser.add_argument("--cv-points", type=int, default=2000, help="Number of FDM/PINN CV points plotted in the summary figure.")
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
