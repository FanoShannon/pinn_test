import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import surface_heat_bem as bem


def nrmse(candidate, reference):
    scale = max(float(np.max(np.abs(reference))), 1e-15)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale)


def area_mean(surface, values):
    return np.sum(values * surface.areas[None, :], axis=1) / np.sum(surface.areas)


def reflection_asymmetry(surface, values):
    centers = surface.centroids
    reflected = centers.copy()
    reflected[:, 0] *= -1.0
    scale = max(np.ptp(centers[:, 0]), np.ptp(centers[:, 1]), 1e-15)
    distance = np.sum(
        ((reflected[:, None, :] - centers[None, :, :]) / scale) ** 2,
        axis=2,
    )
    partner = np.argmin(distance, axis=1)
    return float(
        np.sqrt(np.mean((values - values[partner]) ** 2))
        / max(np.sqrt(np.mean(values ** 2)), 1e-15)
    )


def run_flat_level(n_side, time, amplitude, images, quadrature):
    surface = bem.flat_periodic_square(n_side)
    operator = bem.AbelSplitHeatSingleLayer(
        surface, diffusion=1.0, periodic_images=images
    )
    flux = np.repeat(amplitude[:, None], operator.n_panels, axis=1)
    trace = bem.convolve_surface_history(
        operator, time, flux, quadrature_order=quadrature
    )
    reference = bem.abel_product_integral(time, flux, diffusion=1.0)[:, 0]
    mean_trace = area_mean(surface, trace)
    return surface, mean_trace, reference


def run_cap_level(n_radial, n_angular, time, amplitude, quadrature):
    sphere_radius = 0.35
    cap_radius = 0.25
    surface = bem.spherical_cap_surface(
        n_radial=n_radial,
        n_angular=n_angular,
        radius=sphere_radius,
        cap_radius=cap_radius,
    )
    flat = bem.flattened_surface(surface)
    sphere_operator = bem.AbelSplitHeatSingleLayer(surface)
    flat_operator = bem.AbelSplitHeatSingleLayer(flat)
    x = surface.centroids[:, 0] / cap_radius
    y = surface.centroids[:, 1] / cap_radius
    spatial_flux = np.exp(0.55 * x + 0.25 * x * y)
    flux = amplitude[:, None] * spatial_flux[None, :]
    sphere_trace = bem.convolve_surface_history(
        sphere_operator, time, flux, quadrature_order=quadrature
    )
    flat_trace = bem.convolve_surface_history(
        flat_operator, time, flux, quadrature_order=quadrature
    )
    return {
        "surface": surface,
        "flat_surface": flat,
        "flux": flux,
        "sphere_trace": sphere_trace,
        "flat_trace": flat_trace,
        "sphere_mean": area_mean(surface, sphere_trace),
        "flat_mean": area_mean(flat, flat_trace),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit the 1D-to-curved-3D Abel-split heat BEM consistency"
    )
    parser.add_argument(
        "--output", default="runs_curved_heat_bem/consistency"
    )
    parser.add_argument("--steps", type=int, default=17)
    parser.add_argument("--duration", type=float, default=0.03)
    parser.add_argument("--periodic-images", type=int, default=3)
    parser.add_argument("--quadrature", type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, args.duration, args.steps)
    amplitude = (
        np.sin(np.pi * time / args.duration) ** 2
        + 0.2 * time / args.duration
    )

    flat_results = []
    for n_side in (3, 4, 6, 8):
        surface, trace, reference = run_flat_level(
            n_side,
            time,
            amplitude,
            args.periodic_images,
            args.quadrature,
        )
        flat_results.append({
            "n_side": n_side,
            "n_panels": len(surface.triangles),
            "trace": trace,
            "reference": reference,
            "nrmse": nrmse(trace, reference),
            "max_relative": float(
                np.max(np.abs(trace - reference))
                / max(np.max(np.abs(reference)), 1e-15)
            ),
        })
        print(
            f"flat n={n_side} panels={len(surface.triangles)} "
            f"NRMSE={flat_results[-1]['nrmse']:.3e}"
        )

    cap_results = []
    for n_radial, n_angular in ((2, 12), (3, 16), (4, 20)):
        result = run_cap_level(
            n_radial, n_angular, time, amplitude, args.quadrature
        )
        result["n_radial"] = n_radial
        result["n_angular"] = n_angular
        result["n_panels"] = len(result["surface"].triangles)
        cap_results.append(result)
        print(
            f"cap radial={n_radial} angular={n_angular} "
            f"panels={result['n_panels']}"
        )

    cap_reference = cap_results[-1]["sphere_mean"]
    cap_levels = []
    for result in cap_results:
        cap_levels.append({
            "n_radial": result["n_radial"],
            "n_angular": result["n_angular"],
            "n_panels": result["n_panels"],
            "mean_trace_nrmse_vs_finest": nrmse(
                result["sphere_mean"], cap_reference
            ),
        })
    finest = cap_results[-1]
    geometry_correction = nrmse(finest["sphere_mean"], finest["flat_mean"])
    asymmetry = reflection_asymmetry(
        finest["surface"], finest["sphere_trace"][-1]
    )

    summary = {
        "status": "single_layer_consistency_prototype",
        "full_heat_bie_solved": False,
        "film_dtn_included": False,
        "time_steps": args.steps,
        "duration": args.duration,
        "periodic_images": args.periodic_images,
        "triangle_quadrature_points": 3,
        "time_quadrature_order": args.quadrature,
        "flat_uniform_abel_limit": [
            {
                "n_side": result["n_side"],
                "n_panels": result["n_panels"],
                "nrmse": result["nrmse"],
                "max_relative": result["max_relative"],
            }
            for result in flat_results
        ],
        "curved_cap_convergence": cap_levels,
        "curved_vs_flat_geometry_correction_nrmse": geometry_correction,
        "nonaxisymmetric_trace_index": asymmetry,
    }
    (output / "surface_heat_bem_consistency.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "surface_heat_bem_consistency.npz",
        time=time,
        amplitude=amplitude,
        abel_reference=flat_results[-1]["reference"],
        finest_flat_periodic=flat_results[-1]["trace"],
        cap_mean=finest["sphere_mean"],
        flat_disk_mean=finest["flat_mean"],
        cap_centroids=finest["surface"].centroids,
        cap_final_trace=finest["sphere_trace"][-1],
    )

    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    axis = figure.add_subplot(2, 2, 1)
    axis.plot(time, flat_results[-1]["reference"], label="1D Abel/ProductIntegral")
    axis.plot(time, flat_results[-1]["trace"], "--", label="triangle heat BEM")
    axis.set_title("Flat uniform limit")
    axis.set_xlabel("time")
    axis.set_ylabel("interface trace")
    axis.legend()
    axis.grid(alpha=0.2)

    axis = figure.add_subplot(2, 2, 2)
    axis.loglog(
        [result["n_panels"] for result in flat_results],
        [result["nrmse"] for result in flat_results],
        "o-",
    )
    axis.set_title("Convergence to the 1D Abel limit")
    axis.set_xlabel("triangle panels")
    axis.set_ylabel("NRMSE")
    axis.grid(alpha=0.2)

    axis = figure.add_subplot(2, 2, 3)
    axis.plot(time, finest["sphere_mean"], label="spherical cap")
    axis.plot(time, finest["flat_mean"], "--", label="flattened disk")
    axis.set_title("Geometry-dependent regular correction")
    axis.set_xlabel("time")
    axis.set_ylabel("area-mean trace")
    axis.legend()
    axis.grid(alpha=0.2)

    axis = figure.add_subplot(2, 2, 4, projection="3d")
    centers = finest["surface"].centroids
    scatter = axis.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=finest["sphere_trace"][-1],
        cmap="viridis",
        s=24,
    )
    axis.set_title("Curved non-axisymmetric trace")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    figure.colorbar(scatter, ax=axis, shrink=0.7)
    figure.savefig(output / "surface_heat_bem_consistency.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
