"""End-to-end curved ProductIntegral-DtN heat-BIE research solver.

The external Neumann problem is reduced to triangular boundary unknowns with a
single-layer density equation. The film uses an exact finite-column spectral
DtN at every projected surface panel. The nonlinear catalytic solve is carried
out only for the curved-interface reaction flux.
"""

from dataclasses import dataclass

import numpy as np

import physical_model as physics
import surface_heat_bem as heat_bem


@dataclass(frozen=True)
class CurvedSurfaceConfig:
    n_x: int = 4
    n_y: int = 4
    length_x: float = 0.8
    length_y: float = 0.7
    base_thickness: float = 0.075
    cap_height: float = 0.12
    cap_radius: float = 0.26
    cap_profile: str = "spherical"
    periodic_images: int = 3

    def build_surface(self):
        return heat_bem.periodic_cap_surface(
            n_x=self.n_x,
            n_y=self.n_y,
            length_x=self.length_x,
            length_y=self.length_y,
            base_thickness=self.base_thickness,
            cap_height=self.cap_height,
            cap_radius=self.cap_radius,
            profile=self.cap_profile,
        )


@dataclass(frozen=True)
class CurvedSurfacePhysics:
    gamma: float = 10.0
    k_cat: float = 1.0
    diffusion_b: float = 1.0
    diffusion_d: float = 1.0
    heterogeneity_x: float = 0.55
    heterogeneity_xy: float = 0.25

    def validate(self):
        physics.validate_positive_parameters(
            gamma=self.gamma,
            k_cat=self.k_cat,
            diffusion_b=self.diffusion_b,
            diffusion_d=self.diffusion_d,
        )
        if not np.isfinite(self.heterogeneity_x):
            raise ValueError("heterogeneity_x must be finite")
        if not np.isfinite(self.heterogeneity_xy):
            raise ValueError("heterogeneity_xy must be finite")


def _abel_cell_weights(lag_cells, dt, diffusion):
    """Piecewise-linear weights for half of the image-doubled Abel kernel."""
    lag_hi = float(lag_cells) * dt
    lag_lo = float(lag_cells - 1) * dt
    root_difference = np.sqrt(lag_hi) - np.sqrt(lag_lo)
    power_difference = lag_hi ** 1.5 - lag_lo ** 1.5
    denominator = dt * np.sqrt(np.pi * diffusion)
    image_left = (
        (2.0 / 3.0) * power_difference
        -2.0 * lag_lo * root_difference
    ) / denominator
    image_right = (
        2.0 * lag_hi * root_difference
        -(2.0 / 3.0) * power_difference
    ) / denominator
    return 0.5 * image_left, 0.5 * image_right


class SurfaceHistoryWeights:
    """Precompute direct-history matrices for V and the adjoint double layer."""

    def __init__(self, operator, n_time, dt, quadrature_order=6):
        self.operator = operator
        self.n_time = int(n_time)
        self.dt = float(dt)
        nodes, raw_weights = np.polynomial.legendre.leggauss(
            int(quadrature_order)
        )
        fractions = 0.5 * (nodes + 1.0)
        time_weights = 0.5 * self.dt * raw_weights
        n_panels = operator.n_panels
        identity = np.eye(n_panels)
        self.v_left = [None] * self.n_time
        self.v_right = [None] * self.n_time
        self.k_left = [None] * self.n_time
        self.k_right = [None] * self.n_time

        for lag_cells in range(1, self.n_time):
            abel_left, abel_right = _abel_cell_weights(
                lag_cells, self.dt, operator.diffusion
            )
            v_left = abel_left * identity
            v_right = abel_right * identity
            k_left = np.zeros((n_panels, n_panels), dtype=np.float64)
            k_right = np.zeros_like(k_left)
            for fraction, weight in zip(fractions, time_weights):
                lag = (lag_cells - fraction) * self.dt
                regular = 0.5 * operator.regular_matrix(lag)
                normal = operator.normal_derivative_matrix(lag)
                v_left += weight * (1.0 - fraction) * regular
                v_right += weight * fraction * regular
                k_left += weight * (1.0 - fraction) * normal
                k_right += weight * fraction * normal
            self.v_left[lag_cells] = v_left
            self.v_right[lag_cells] = v_right
            self.k_left[lag_cells] = k_left
            self.k_right[lag_cells] = k_right


class ExternalNeumannHeatBie:
    """Causal Neumann-to-Dirichlet map on a periodic curved graph."""

    def __init__(
        self,
        surface,
        diffusion,
        n_time,
        dt,
        periodic_images=2,
        quadrature_order=6,
    ):
        self.diffusion = float(diffusion)
        self.operator = heat_bem.AbelSplitHeatSingleLayer(
            surface,
            diffusion=self.diffusion,
            periodic_images=periodic_images,
        )
        self.weights = SurfaceHistoryWeights(
            self.operator,
            n_time,
            dt,
            quadrature_order=quadrature_order,
        )
        self.identity = np.eye(self.operator.n_panels)
        self.current_matrix = (
            0.5 * self.identity
            -self.diffusion * self.weights.k_right[1]
        )
        self.density_response = np.linalg.solve(
            self.current_matrix, self.identity
        )

    @property
    def n_panels(self):
        return self.operator.n_panels

    def affine_step(self, density_history, step):
        density_history = np.asarray(density_history, dtype=np.float64)
        k_completed = np.zeros(self.n_panels, dtype=np.float64)
        v_completed = np.zeros(self.n_panels, dtype=np.float64)
        for cell in range(step - 1):
            lag_cells = step - cell
            left = density_history[cell]
            right = density_history[cell + 1]
            k_completed += (
                self.weights.k_left[lag_cells] @ left
                +self.weights.k_right[lag_cells] @ right
            )
            v_completed += (
                self.weights.v_left[lag_cells] @ left
                +self.weights.v_right[lag_cells] @ right
            )

        previous_density = density_history[step - 1]
        density_rhs_history = self.diffusion * (
            k_completed + self.weights.k_left[1] @ previous_density
        )
        density_intercept = np.linalg.solve(
            self.current_matrix, density_rhs_history
        )
        trace_intercept = (
            v_completed
            +self.weights.v_left[1] @ previous_density
            +self.weights.v_right[1] @ density_intercept
        )
        trace_response = (
            self.weights.v_right[1] @ self.density_response
        )
        return {
            "density_intercept": density_intercept,
            "density_response": self.density_response,
            "trace_intercept": trace_intercept,
            "trace_response": trace_response,
            "k_completed": k_completed,
        }

    def boundary_residual(self, density_history, physical_flux, step):
        value = np.zeros(self.n_panels, dtype=np.float64)
        for cell in range(step):
            lag_cells = step - cell
            value += (
                self.weights.k_left[lag_cells] @ density_history[cell]
                +self.weights.k_right[lag_cells] @ density_history[cell + 1]
            )
        recovered_flux = (
            0.5 * density_history[step] -self.diffusion * value
        )
        return recovered_flux - physical_flux


class LocalColumnFilmDtn:
    """Finite-thickness spectral DtN for vertical columns under each panel."""

    def __init__(
        self,
        thickness,
        normal_z,
        diffusion,
        dt,
        n_modes=64,
    ):
        self.thickness = np.asarray(thickness, dtype=np.float64)
        self.normal_z = np.asarray(normal_z, dtype=np.float64)
        self.diffusion = float(diffusion)
        self.n_modes = int(n_modes)
        if np.min(self.thickness) <= 0.0:
            raise ValueError("All local film thicknesses must be positive")
        if np.min(self.normal_z) <= 0.0:
            raise ValueError("The curved interface must remain an upward graph")
        mode = np.arange(self.n_modes, dtype=np.float64)
        self.sign = (-1.0) ** mode
        self.mu = (
            (mode[None, :] + 0.5) * np.pi / self.thickness[:, None]
        )
        decay_rate = self.diffusion * self.mu ** 2
        self.decay = np.exp(-decay_rate * dt)
        gain = -np.expm1(-decay_rate * dt) / decay_rate
        self.gain = gain
        self.flux_response = (
            2.0 * self.sign[None, :] * gain
            /(dt * self.thickness[:, None] * self.diffusion * self.mu ** 2)
        )

    def initial_amplitudes(self, electrode_value):
        return (
            -2.0 * float(electrode_value)
            /(self.thickness[:, None] * self.mu)
        )

    def affine_step(
        self,
        previous_amplitudes,
        previous_surface_flux,
        electrode_previous,
        electrode_current,
        dt,
    ):
        surface_slope = (
            float(electrode_current) - float(electrode_previous)
        ) / float(dt)
        previous_column_flux = (
            np.asarray(previous_surface_flux, dtype=np.float64) / self.normal_z
        )
        amplitude_intercept = (
            self.decay * previous_amplitudes
            +self.gain
            *(-2.0 * surface_slope /(self.thickness[:, None] * self.mu))
            -self.flux_response * previous_column_flux[:, None]
        )
        b_intercept = (
            float(electrode_current)
            +np.sum(amplitude_intercept * self.sign[None, :], axis=1)
        )
        column_slope = (
            -self.thickness / self.diffusion
            +np.sum(self.flux_response * self.sign[None, :], axis=1)
        )
        surface_slope_response = column_slope / self.normal_z
        return amplitude_intercept, b_intercept, np.diag(surface_slope_response)

    def update_amplitudes(self, amplitude_intercept, surface_flux):
        column_flux = np.asarray(surface_flux) / self.normal_z
        return amplitude_intercept + self.flux_response * column_flux[:, None]

    def electrode_current(self, amplitudes, surface_flux):
        column_flux = np.asarray(surface_flux) / self.normal_z
        # Positive current follows the existing true-3D FDM convention.
        return column_flux -self.diffusion * np.sum(
            amplitudes * self.mu, axis=1
        )

    def inventory(self, amplitudes, surface_flux, electrode_value):
        column_flux = np.asarray(surface_flux) / self.normal_z
        return (
            self.thickness * float(electrode_value)
            -0.5 * self.thickness ** 2 * column_flux / self.diffusion
            +np.sum(amplitudes / self.mu, axis=1)
        )


def _solve_reaction_flux(
    previous_flux,
    k_face,
    b_intercept,
    b_response,
    c_intercept,
    c_response,
    gamma,
    max_iterations,
):
    flux = np.maximum(np.asarray(previous_flux, dtype=np.float64), 0.0).copy()
    scale = max(1.0, float(np.max(k_face) * gamma))
    residual_norm = np.inf
    for iteration in range(1, int(max_iterations) + 1):
        b_value = b_intercept + b_response @ flux
        c_value = c_intercept + c_response @ flux
        residual = flux - k_face * b_value * c_value
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= 1e-11 * scale:
            return flux, iteration, residual_norm
        jacobian = (
            np.eye(len(flux))
            -np.diag(k_face * c_value) @ b_response
            -np.diag(k_face * b_value) @ c_response
        )
        direction = np.linalg.solve(jacobian, -residual)
        accepted = False
        alpha = 1.0
        for _ in range(20):
            candidate = flux + alpha * direction
            b_new = b_intercept + b_response @ candidate
            c_new = c_intercept + c_response @ candidate
            candidate_residual = candidate - k_face * b_new * c_new
            candidate_norm = float(np.max(np.abs(candidate_residual)))
            if (
                np.all(candidate >= -1e-13)
                and np.all(b_new >= -1e-13)
                and np.all(c_new >= -1e-13)
                and np.all(c_new <= gamma * (1.0 + 1e-10))
                and candidate_norm < residual_norm
            ):
                flux = np.maximum(candidate, 0.0)
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            target = k_face * np.maximum(b_value, 0.0) * np.clip(
                c_value, 0.0, gamma
            )
            flux = 0.5 * flux + 0.5 * target
    raise RuntimeError(
        "Curved PI-DtN-BIE reaction Newton did not converge; "
        f"last residual={residual_norm:.3e}"
    )


def simulate_curved_pi_dtn_bie(
    time,
    config=None,
    parameters=None,
    n_film_modes=64,
    history_quadrature=6,
    max_iterations=20,
):
    """Run the complete interface-only curved reaction-diffusion recurrence."""
    config = CurvedSurfaceConfig() if config is None else config
    parameters = CurvedSurfacePhysics() if parameters is None else parameters
    parameters.validate()
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be a strictly increasing 1D grid")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-10, atol=1e-14):
        raise ValueError("A uniform time grid is required")

    surface = config.build_surface()
    n_panels = len(surface.triangles)
    normal_z = surface.normals[:, 2]
    projected_area = surface.areas * normal_z
    if np.min(projected_area) <= 0.0:
        raise ValueError("The surface must be a single-valued upward graph")
    theta, electrode = physics.triangular_protocol_numpy(time)
    external = ExternalNeumannHeatBie(
        surface,
        parameters.diffusion_d,
        len(time),
        dt,
        periodic_images=config.periodic_images,
        quadrature_order=history_quadrature,
    )
    film = LocalColumnFilmDtn(
        surface.centroids[:, 2],
        normal_z,
        parameters.diffusion_b,
        dt,
        n_modes=n_film_modes,
    )

    x_scale = max(0.5 * config.length_x, 1e-15)
    y_scale = max(0.5 * config.length_y, 1e-15)
    centers = surface.centroids
    exponent = (
        parameters.heterogeneity_x * centers[:, 0] / x_scale
        +parameters.heterogeneity_xy
        *centers[:, 0] * centers[:, 1] /(x_scale * y_scale)
    )
    k_face = parameters.k_cat * np.exp(exponent)

    flux = np.zeros((len(time), n_panels), dtype=np.float64)
    density = np.zeros_like(flux)
    b_interface = np.zeros_like(flux)
    c_interface = np.full_like(flux, parameters.gamma)
    d_interface = np.zeros_like(flux)
    current_map = np.zeros_like(flux)
    inventory = np.zeros_like(flux)
    closure_residual = np.zeros(len(time), dtype=np.float64)
    bie_residual = np.zeros(len(time), dtype=np.float64)
    newton_iterations = np.zeros(len(time), dtype=np.int64)
    amplitudes = film.initial_amplitudes(electrode[0])

    for step in range(1, len(time)):
        amplitude_intercept, b_intercept, b_response = film.affine_step(
            amplitudes,
            flux[step - 1],
            electrode[step - 1],
            electrode[step],
            dt,
        )
        external_affine = external.affine_step(density, step)
        d_intercept = external_affine["trace_intercept"]
        d_response = external_affine["trace_response"]
        c_intercept = parameters.gamma - d_intercept
        c_response = -d_response
        value, iterations, residual = _solve_reaction_flux(
            flux[step - 1],
            k_face,
            b_intercept,
            b_response,
            c_intercept,
            c_response,
            parameters.gamma,
            max_iterations,
        )
        flux[step] = value
        density[step] = (
            external_affine["density_intercept"]
            +external_affine["density_response"] @ value
        )
        amplitudes = film.update_amplitudes(amplitude_intercept, value)
        b_interface[step] = b_intercept + b_response @ value
        d_interface[step] = d_intercept + d_response @ value
        c_interface[step] = parameters.gamma - d_interface[step]
        current_map[step] = film.electrode_current(amplitudes, value)
        inventory[step] = film.inventory(amplitudes, value, electrode[step])
        closure_residual[step] = residual
        bie_residual[step] = float(np.max(np.abs(
            external.boundary_residual(density, value, step)
        )))
        newton_iterations[step] = iterations

    electrode_current = np.sum(
        current_map * projected_area[None, :], axis=1
    ) / np.sum(projected_area)
    mean_flux = np.sum(
        flux * surface.areas[None, :], axis=1
    ) / np.sum(surface.areas)
    mean_inventory = np.sum(
        inventory * projected_area[None, :], axis=1
    ) / np.sum(projected_area)
    final_flux = flux[-1]
    reflected = centers.copy()
    reflected[:, 0] *= -1.0
    coordinate_scale = max(config.length_x, config.length_y)
    distance = np.sum(
        ((reflected[:, None, :] - centers[None, :, :]) / coordinate_scale) ** 2,
        axis=2,
    )
    partner = np.argmin(distance, axis=1)
    nonaxisymmetric_index = float(
        np.sqrt(np.mean((final_flux - final_flux[partner]) ** 2))
        /max(np.sqrt(np.mean(final_flux ** 2)), 1e-15)
    )
    return {
        "method": "curved_productintegral_local_film_dtn_full_heat_bie",
        "full_external_heat_bie": True,
        "film_two_boundary_dtn": True,
        "film_tangential_diffusion_included": False,
        "volume_grid_used": False,
        "fdm_data_used": False,
        "neural_network_used": False,
        "time": time,
        "theta": theta,
        "C_B_surface": electrode,
        "electrode_current_density": electrode_current,
        "mean_reaction_flux": mean_flux,
        "J_face": flux,
        "single_layer_density": density,
        "C_B_int": b_interface,
        "C_C_int": c_interface,
        "C_D_int": d_interface,
        "film_inventory": mean_inventory,
        "face_centers": centers,
        "face_area": surface.areas,
        "projected_area": projected_area,
        "k_face": k_face,
        "closure_residual": closure_residual,
        "bie_boundary_residual": bie_residual,
        "newton_iterations": newton_iterations,
        "nonaxisymmetric_index": nonaxisymmetric_index,
        "surface": surface,
        "config": config,
        "parameters": parameters,
    }
