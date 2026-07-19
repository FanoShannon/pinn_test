"""Matched-geometry posterior and causal interface-operator error audit."""

import argparse
import json
import time as wall_time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curved_pi_dtn_bie as curved
import matched_geometry_fem as matched
import physical_model as physics
import surface_modal_film_dtn as surface_film


def nrmse(candidate, reference):
    return float(
        np.sqrt(np.mean((np.asarray(candidate) - np.asarray(reference)) ** 2))
        /max(float(np.max(np.abs(reference))), 1e-15)
    )


def area_mean(values, area):
    return np.sum(values * area[None, :], axis=1) / np.sum(area)


def area_time_norm(values, area):
    spatial = np.sum(np.asarray(values) ** 2 * area[None, :], axis=1) / np.sum(area)
    return float(np.sqrt(np.mean(spatial)))


def surface_spectral_error(candidate, reference, surface, low_mode_count=4):
    fem = surface_film.assemble_surface_fem(surface)
    lumped = fem["lumped_mass"]
    inverse_root = 1.0 / np.sqrt(lumped)
    symmetric_laplace = (
        inverse_root[:, None]
        *fem["stiffness"]
        *inverse_root[None, :]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_laplace)
    candidate_nodes = np.asarray(candidate) @ fem["face_to_node"].T
    reference_nodes = np.asarray(reference) @ fem["face_to_node"].T
    candidate_coefficients = (
        candidate_nodes * np.sqrt(lumped)[None, :]
    ) @ eigenvectors
    reference_coefficients = (
        reference_nodes * np.sqrt(lumped)[None, :]
    ) @ eigenvectors
    difference = candidate_coefficients - reference_coefficients
    error_energy = np.mean(difference ** 2, axis=0)
    reference_energy = np.mean(reference_coefficients ** 2, axis=0)
    count = min(int(low_mode_count), len(eigenvalues))

    def relative(indices):
        numerator = float(np.sum(error_energy[indices]))
        denominator = max(float(np.sum(reference_energy[indices])), 1e-30)
        return float(np.sqrt(numerator / denominator))

    total_error = max(float(np.sum(error_energy)), 1e-30)
    return {
        "surface_eigenvalues_first": [
            float(value) for value in eigenvalues[:count]
        ],
        "constant_mode_relative_rms": relative(slice(0, 1)),
        "low_modes_relative_rms": relative(slice(0, count)),
        "low_nonconstant_modes_relative_rms": (
            relative(slice(1, count)) if count > 1 else 0.0
        ),
        "all_projected_modes_relative_rms": relative(slice(None)),
        "error_energy_fraction_above_low_modes": float(
            np.sum(error_energy[count:]) / total_error
        ),
    }


def _face_kinetics(surface, config, parameters):
    centers = surface.centroids
    x_scale = max(0.5 * config.length_x, 1e-15)
    y_scale = max(0.5 * config.length_y, 1e-15)
    exponent = (
        parameters.heterogeneity_x * centers[:, 0] / x_scale
        +parameters.heterogeneity_xy * centers[:, 0] * centers[:, 1]
        /(x_scale * y_scale)
    )
    return parameters.k_cat * np.exp(exponent)


def run_bie_blocks_with_prescribed_flux(
    time, config, parameters, flux, film_model="local"
):
    surface = config.build_surface()
    dt = float(time[1] - time[0])
    _, electrode = physics.triangular_protocol_numpy(time)
    normal_z = surface.normals[:, 2]
    if film_model == "local":
        film = curved.LocalColumnFilmDtn(
            surface.centroids[:, 2],
            normal_z,
            parameters.diffusion_b,
            dt,
            n_modes=64,
        )
    elif film_model == "surface_modal":
        film = surface_film.SurfaceModalFilmDtn(
            surface,
            normal_z,
            parameters.diffusion_b,
            dt,
            n_modes=64,
        )
    else:
        raise ValueError("film_model must be 'local' or 'surface_modal'")
    external = curved.ExternalNeumannHeatBie(
        surface,
        parameters.diffusion_d,
        len(time),
        dt,
        periodic_images=config.periodic_images,
        quadrature_order=6,
    )
    amplitudes = film.initial_amplitudes(electrode[0])
    density = np.zeros_like(flux)
    b_value = np.zeros_like(flux)
    d_value = np.zeros_like(flux)
    current = np.zeros(len(time), dtype=np.float64)
    b_intercept = [None] * len(time)
    b_response = [None] * len(time)
    d_intercept = [None] * len(time)
    d_response = [None] * len(time)
    for step in range(1, len(time)):
        amplitude_free, b_free, b_matrix = film.affine_step(
            amplitudes,
            flux[step - 1],
            electrode[step - 1],
            electrode[step],
            dt,
        )
        external_free = external.affine_step(density, step)
        amplitudes = film.update_amplitudes(amplitude_free, flux[step])
        density[step] = (
            external_free["density_intercept"]
            +external_free["density_response"] @ flux[step]
        )
        b_value[step] = b_free + b_matrix @ flux[step]
        d_value[step] = (
            external_free["trace_intercept"]
            +external_free["trace_response"] @ flux[step]
        )
        current[step] = np.sum(
            film.electrode_current(amplitudes, flux[step])
            *surface.areas * normal_z
        ) / np.sum(surface.areas * normal_z)
        b_intercept[step] = b_free
        b_response[step] = b_matrix
        d_intercept[step] = external_free["trace_intercept"]
        d_response[step] = external_free["trace_response"]
    return {
        "b": b_value,
        "d": d_value,
        "c": parameters.gamma - d_value,
        "current": current,
        "b_intercept": b_intercept,
        "b_response": b_response,
        "d_intercept": d_intercept,
        "d_response": d_response,
    }


def run_fem_blocks_with_prescribed_flux(
    time, config, parameters, fem_config, flux
):
    _, electrode = physics.triangular_protocol_numpy(time)
    operator = matched.MatchedGeometryFemOperator(
        config, parameters, float(time[1] - time[0]), fem_config=fem_config
    )
    state = operator.initial_state(electrode[0])
    b_value = np.zeros_like(flux)
    d_value = np.zeros_like(flux)
    current = np.zeros(len(time), dtype=np.float64)
    b_intercept = [None] * len(time)
    b_response = [None] * len(time)
    d_intercept = [None] * len(time)
    d_response = [None] * len(time)
    for step in range(1, len(time)):
        affine = operator.affine_step(state, electrode[step])
        b_value[step] = (
            affine["b_intercept"] + affine["b_response"] @ flux[step]
        )
        d_value[step] = (
            affine["external"]["trace_intercept"]
            +affine["external"]["trace_response"] @ flux[step]
        )
        next_state = operator.update(
            affine, flux[step], electrode[step], previous_state=state
        )
        current[step] = next_state["electrode_current"]
        b_intercept[step] = affine["b_intercept"]
        b_response[step] = affine["b_response"]
        d_intercept[step] = affine["external"]["trace_intercept"]
        d_response[step] = affine["external"]["trace_response"]
        state = next_state
    return {
        "b": b_value,
        "d": d_value,
        "c": parameters.gamma - d_value,
        "current": current,
        "b_intercept": b_intercept,
        "b_response": b_response,
        "d_intercept": d_intercept,
        "d_response": d_response,
    }


def feedback_error_audit(main, bie_blocks, fem_blocks, k_face, area):
    n_time, n_faces = main["J_face"].shape
    film_effect = np.zeros_like(main["J_face"])
    external_effect = np.zeros_like(main["J_face"])
    combined_effect = np.zeros_like(main["J_face"])
    condition = np.ones(n_time, dtype=np.float64)
    b_response_error = []
    d_response_error = []
    for step in range(1, n_time):
        b_value = bie_blocks["b"][step]
        c_value = bie_blocks["c"][step]
        b_matrix = bie_blocks["b_response"][step]
        c_matrix = -bie_blocks["d_response"][step]
        jacobian = (
            np.eye(n_faces)
            -np.diag(k_face * c_value) @ b_matrix
            -np.diag(k_face * b_value) @ c_matrix
        )
        condition[step] = np.linalg.cond(jacobian)
        delta_b = fem_blocks["b"][step] - b_value
        delta_c = fem_blocks["c"][step] - c_value
        film_rhs = k_face * c_value * delta_b
        external_rhs = k_face * b_value * delta_c
        film_effect[step] = np.linalg.solve(jacobian, film_rhs)
        external_effect[step] = np.linalg.solve(jacobian, external_rhs)
        combined_effect[step] = film_effect[step] + external_effect[step]
        fem_b_matrix = fem_blocks["b_response"][step]
        fem_d_matrix = fem_blocks["d_response"][step]
        b_response_error.append(
            np.linalg.norm(b_matrix - fem_b_matrix)
            /max(np.linalg.norm(fem_b_matrix), 1e-15)
        )
        d_response_error.append(
            np.linalg.norm(bie_blocks["d_response"][step] - fem_d_matrix)
            /max(np.linalg.norm(fem_d_matrix), 1e-15)
        )
    scale = max(area_time_norm(main["J_face"], area), 1e-15)
    return {
        "film_effect": film_effect,
        "external_effect": external_effect,
        "combined_effect": combined_effect,
        "condition": condition,
        "summary": {
            "film_trace_effect_over_flux_rms": area_time_norm(
                film_effect, area
            ) / scale,
            "external_trace_effect_over_flux_rms": area_time_norm(
                external_effect, area
            ) / scale,
            "combined_one_step_effect_over_flux_rms": area_time_norm(
                combined_effect, area
            ) / scale,
            "feedback_condition_mean": float(np.mean(condition[1:])),
            "feedback_condition_max": float(np.max(condition[1:])),
            "film_current_cell_response_relative_fro_mean": float(
                np.mean(b_response_error)
            ),
            "external_current_cell_response_relative_fro_mean": float(
                np.mean(d_response_error)
            ),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched smooth-geometry operator error audit"
    )
    parser.add_argument(
        "--output", default="runs_curved_heat_bem/matched_geometry_audit"
    )
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--film-layers", type=int, default=18)
    parser.add_argument("--external-layers", type=int, default=72)
    parser.add_argument("--external-top", type=float, default=3.0)
    parser.add_argument("--cap-height", type=float, default=0.12)
    parser.add_argument("--heterogeneity-x", type=float, default=0.55)
    parser.add_argument("--heterogeneity-xy", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, 1.0, args.steps)
    config = curved.CurvedSurfaceConfig(cap_height=args.cap_height)
    parameters = curved.CurvedSurfacePhysics(
        heterogeneity_x=args.heterogeneity_x,
        heterogeneity_xy=args.heterogeneity_xy,
    )
    fem_config = matched.MatchedFemConfig(
        film_layers=args.film_layers,
        external_layers=args.external_layers,
        external_top=args.external_top,
    )

    started = wall_time.perf_counter()
    main_result = curved.simulate_curved_pi_dtn_bie(
        time, config, parameters, n_film_modes=64, max_iterations=30,
        film_model="local",
    )
    main_seconds = wall_time.perf_counter() - started
    started = wall_time.perf_counter()
    fem_result = matched.simulate_matched_geometry_fem(
        time, config, parameters, fem_config=fem_config, max_iterations=30
    )
    fem_seconds = wall_time.perf_counter() - started
    coarse_fem_config = matched.MatchedFemConfig(
        film_layers=max(4, args.film_layers // 2),
        external_layers=max(12, args.external_layers // 2),
        external_top=args.external_top,
    )
    coarse_fem = matched.simulate_matched_geometry_fem(
        time, config, parameters, fem_config=coarse_fem_config,
        max_iterations=30,
    )
    refined_time = np.linspace(0.0, 1.0, 2 * args.steps - 1)
    refined_time_fem = matched.simulate_matched_geometry_fem(
        refined_time, config, parameters, fem_config=fem_config,
        max_iterations=30,
    )
    common_flux = main_result["J_face"]
    bie_blocks = run_bie_blocks_with_prescribed_flux(
        time, config, parameters, common_flux
    )
    fem_blocks = run_fem_blocks_with_prescribed_flux(
        time, config, parameters, fem_config, common_flux
    )
    feedback = feedback_error_audit(
        main_result,
        bie_blocks,
        fem_blocks,
        main_result["k_face"],
        main_result["face_area"],
    )

    area = main_result["face_area"]
    main_b = area_mean(main_result["C_B_int"], area)
    main_c = area_mean(main_result["C_C_int"], area)
    fem_b = area_mean(fem_result["C_B_int"], area)
    fem_c = area_mean(fem_result["C_C_int"], area)
    actual_flux_difference = fem_result["J_face"] - main_result["J_face"]
    flux_scale = max(area_time_norm(main_result["J_face"], area), 1e-15)
    metrics = {
        "status": "posterior_only_matched_smooth_geometry",
        "main_operator_modified": False,
        "fdm_file_used": False,
        "fitted_correction_used": False,
        "surface_panels": int(len(area)),
        "time_steps": args.steps,
        "film_layers": args.film_layers,
        "external_layers": args.external_layers,
        "external_top": args.external_top,
        "cap_height": args.cap_height,
        "heterogeneity_x": args.heterogeneity_x,
        "heterogeneity_xy": args.heterogeneity_xy,
        "runtime_seconds": {"bie": main_seconds, "fem": fem_seconds},
        "reference_convergence": {
            "coarse_layers": [
                coarse_fem_config.film_layers,
                coarse_fem_config.external_layers,
            ],
            "fine_layers": [args.film_layers, args.external_layers],
            "coarse_to_fine_CV_nrmse": nrmse(
                coarse_fem["electrode_current_density"],
                fem_result["electrode_current_density"],
            ),
            "time_steps_coarse_fine": [args.steps, 2 * args.steps - 1],
            "time_coarse_to_fine_CV_nrmse": nrmse(
                fem_result["electrode_current_density"],
                refined_time_fem["electrode_current_density"][::2],
            ),
            "time_coarse_to_fine_flux_relative_rms": area_time_norm(
                fem_result["J_face"] - refined_time_fem["J_face"][::2],
                area,
            ) /max(
                area_time_norm(refined_time_fem["J_face"][::2], area),
                1e-15,
            ),
        },
        "closed_loop": {
            "CV_nrmse": nrmse(
                main_result["electrode_current_density"],
                fem_result["electrode_current_density"],
            ),
            "reaction_flux_area_time_relative_rms": area_time_norm(
                actual_flux_difference, area
            ) / flux_scale,
            "mean_C_B_interface_nrmse": nrmse(main_b, fem_b),
            "mean_C_C_over_gamma_interface_nrmse": nrmse(
                main_c / parameters.gamma, fem_c / parameters.gamma
            ),
            "max_fem_closure_residual": float(
                np.max(fem_result["closure_residual"])
            ),
            "C_B_surface_spectral_error": surface_spectral_error(
                main_result["C_B_int"],
                fem_result["C_B_int"],
                main_result["surface"],
            ),
            "reaction_flux_surface_spectral_error": surface_spectral_error(
                main_result["J_face"],
                fem_result["J_face"],
                main_result["surface"],
            ),
        },
        "prescribed_common_flux": {
            "film_trace_area_time_relative_rms": area_time_norm(
                bie_blocks["b"] - fem_blocks["b"], area
            ) /max(area_time_norm(fem_blocks["b"], area), 1e-15),
            "external_trace_area_time_relative_rms": area_time_norm(
                bie_blocks["d"] - fem_blocks["d"], area
            ) /max(area_time_norm(fem_blocks["d"], area), 1e-15),
            "electrode_current_nrmse": nrmse(
                bie_blocks["current"], fem_blocks["current"]
            ),
        },
        "linearized_feedback": feedback["summary"],
        "interpretation_boundary": (
            "The FEM shares the smooth triangular interface but uses backward "
            "Euler volume tetrahedra. Remaining differences include FEM temporal "
            "and normal-layer discretization and the BIE periodic-image truncation."
        ),
    }
    (output / "matched_geometry_error_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "matched_geometry_error_curves.npz",
        time=time,
        theta=main_result["theta"],
        bie_current=main_result["electrode_current_density"],
        fem_current=fem_result["electrode_current_density"],
        bie_C_B_mean=main_b,
        fem_C_B_mean=fem_b,
        bie_C_D_mean=parameters.gamma - main_c,
        fem_C_D_mean=parameters.gamma - fem_c,
        predicted_film_flux_effect=area_mean(feedback["film_effect"], area),
        predicted_external_flux_effect=area_mean(
            feedback["external_effect"], area
        ),
        actual_mean_flux_difference=area_mean(actual_flux_difference, area),
        feedback_condition=feedback["condition"],
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(main_result["theta"], fem_result["electrode_current_density"], label="matched FEM")
    axes[0, 0].plot(main_result["theta"], main_result["electrode_current_density"], "--", label="PI-DtN-BIE")
    axes[0, 0].set_title("Closed-loop CV")
    axes[0, 0].set_xlabel("theta")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(time, fem_b, label="matched FEM")
    axes[0, 1].plot(time, main_b, "--", label="PI-DtN-BIE")
    axes[0, 1].set_title("Mean film-side interface state")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(time, area_mean(actual_flux_difference, area), label="actual FEM - BIE")
    axes[1, 0].plot(time, area_mean(feedback["film_effect"], area), "--", label="film trace prediction")
    axes[1, 0].plot(time, area_mean(feedback["external_effect"], area), ":", label="external trace prediction")
    axes[1, 0].set_title("Reaction-flux error decomposition")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(time, feedback["condition"])
    axes[1, 1].set_title("Nonlinear closure condition number")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].grid(alpha=0.25)
    figure.savefig(output / "matched_geometry_error_audit.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
