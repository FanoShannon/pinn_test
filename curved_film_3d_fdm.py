"""Direct monolithic 3D finite-volume reference for posterior validation."""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

import curved_film_3d as curved
import physical_model as physics


def _max_abs(value):
    return float(np.max(np.abs(value))) if len(value) else 0.0


class DirectCurvedFilm3DFdm:
    """Solve volume states and interface flux together without DtN condensation."""

    def __init__(self, grid, parameters, dt):
        grid.validate()
        parameters.validate()
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.grid = grid
        self.parameters = parameters
        self.dt = float(dt)
        self.geometry = curved.build_geometry(grid)
        spacing = (
            self.geometry["dx"],
            self.geometry["dy"],
            self.geometry["dz"],
        )
        film_matrix, electrode_source, film_index = (
            curved._build_diffusion_matrix(
                self.geometry["film_mask"],
                parameters.diffusion_b,
                spacing,
                "bottom",
            )
        )
        external_matrix, _, external_index = curved._build_diffusion_matrix(
            self.geometry["external_mask"],
            parameters.diffusion_d,
            spacing,
            "top",
        )
        records = curved._enumerate_interface_faces(
            self.geometry["film_mask"],
            film_index,
            external_index,
            spacing,
        )
        (
            self.p_film,
            self.p_external,
            self.film_sink,
            self.external_source,
            self.face_centers,
            half_distance,
            self.face_area,
        ) = curved._face_operators(
            records,
            self.geometry,
            film_matrix.shape[0],
            external_matrix.shape[0],
        )
        self.film_index = film_index
        self.external_index = external_index
        self.electrode_source = electrode_source
        self.film_system = (
            sparse.eye(film_matrix.shape[0], format="csc")
            -self.dt * film_matrix.tocsc()
        )
        self.external_system = (
            sparse.eye(external_matrix.shape[0], format="csc")
            -self.dt * external_matrix.tocsc()
        )
        self.half_b = half_distance / parameters.diffusion_b
        self.half_d = half_distance / parameters.diffusion_d
        x_scale = max(0.5 * grid.length_x, 1e-15)
        y_scale = max(0.5 * grid.length_y, 1e-15)
        exponent = (
            parameters.heterogeneity_x * self.face_centers[:, 0] / x_scale
            +parameters.heterogeneity_xy
            *self.face_centers[:, 0] * self.face_centers[:, 1]
            /(x_scale * y_scale)
        )
        self.k_face = parameters.k_cat * np.exp(exponent)
        self.zero_film_external = sparse.csr_matrix(
            (self.film_system.shape[0], self.external_system.shape[0])
        )
        self.zero_external_film = sparse.csr_matrix(
            (self.external_system.shape[0], self.film_system.shape[0])
        )

    @property
    def n_faces(self):
        return len(self.k_face)

    def initial_state(self):
        return (
            np.zeros(self.film_system.shape[0], dtype=np.float64),
            np.zeros(self.external_system.shape[0], dtype=np.float64),
            np.zeros(self.n_faces, dtype=np.float64),
        )

    def interface_state(self, film, external, flux):
        b_interface = np.asarray(self.p_film @ film).ravel() - self.half_b * flux
        d_interface = (
            np.asarray(self.p_external @ external).ravel() + self.half_d * flux
        )
        return b_interface, self.parameters.gamma - d_interface, d_interface

    def residual(self, film, external, flux, film_rhs, external_rhs):
        b_interface, c_interface, _ = self.interface_state(
            film, external, flux
        )
        film_residual = (
            self.film_system @ film
            +self.dt * np.asarray(self.film_sink @ flux).ravel()
            -film_rhs
        )
        external_residual = (
            self.external_system @ external
            -self.dt * np.asarray(self.external_source @ flux).ravel()
            -external_rhs
        )
        closure_residual = flux - self.k_face * b_interface * c_interface
        return film_residual, external_residual, closure_residual

    def _jacobian(self, b_interface, c_interface):
        closure_film = (
            -sparse.diags(self.k_face * c_interface) @ self.p_film
        )
        closure_external = (
            sparse.diags(self.k_face * b_interface) @ self.p_external
        )
        closure_flux = sparse.diags(
            1.0
            +self.k_face
            *(self.half_b * c_interface + self.half_d * b_interface)
        )
        return sparse.bmat(
            (
                (
                    self.film_system,
                    self.zero_film_external,
                    self.dt * self.film_sink,
                ),
                (
                    self.zero_external_film,
                    self.external_system,
                    -self.dt * self.external_source,
                ),
                (closure_film, closure_external, closure_flux),
            ),
            format="csc",
        )

    def advance(self, film_previous, external_previous, previous_flux,
                electrode_value, max_iterations=16):
        film_rhs = (
            film_previous
            +self.dt * self.electrode_source * float(electrode_value)
        )
        external_rhs = external_previous
        film_predictor_rhs = (
            film_rhs
            -self.dt * np.asarray(self.film_sink @ previous_flux).ravel()
        )
        external_predictor_rhs = (
            external_rhs
            +self.dt * np.asarray(self.external_source @ previous_flux).ravel()
        )
        film = sparse_linalg.spsolve(self.film_system, film_predictor_rhs)
        external = sparse_linalg.spsolve(
            self.external_system, external_predictor_rhs
        )
        flux = np.maximum(previous_flux, 0.0).copy()
        scale = max(1.0, float(np.max(self.k_face) * self.parameters.gamma))

        for iteration in range(1, max_iterations + 1):
            residual_parts = self.residual(
                film, external, flux, film_rhs, external_rhs
            )
            residual_norm = max(_max_abs(part) for part in residual_parts)
            if residual_norm <= 1e-11 * scale:
                b_interface, c_interface, d_interface = self.interface_state(
                    film, external, flux
                )
                return {
                    "film": film,
                    "external": external,
                    "flux": flux,
                    "C_B_int": b_interface,
                    "C_C_int": c_interface,
                    "C_D_int": d_interface,
                    "iterations": iteration,
                    "residual": residual_norm,
                }
            b_interface, c_interface, _ = self.interface_state(
                film, external, flux
            )
            jacobian = self._jacobian(b_interface, c_interface)
            residual_vector = np.concatenate(residual_parts)
            direction = sparse_linalg.spsolve(jacobian, -residual_vector)
            n_film = len(film)
            n_external = len(external)
            d_film = direction[:n_film]
            d_external = direction[n_film:n_film + n_external]
            d_flux = direction[n_film + n_external:]

            accepted = False
            alpha = 1.0
            for _ in range(20):
                candidate_film = film + alpha * d_film
                candidate_external = external + alpha * d_external
                candidate_flux = flux + alpha * d_flux
                b_new, c_new, _ = self.interface_state(
                    candidate_film, candidate_external, candidate_flux
                )
                candidate_parts = self.residual(
                    candidate_film,
                    candidate_external,
                    candidate_flux,
                    film_rhs,
                    external_rhs,
                )
                candidate_norm = max(_max_abs(part) for part in candidate_parts)
                if (
                    np.all(candidate_flux >= -1e-12)
                    and np.all(b_new >= -1e-12)
                    and np.all(c_new >= -1e-12)
                    and candidate_norm < residual_norm
                ):
                    film = candidate_film
                    external = candidate_external
                    flux = np.maximum(candidate_flux, 0.0)
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                raise RuntimeError(
                    "Direct 3D FDM Newton line search failed at residual "
                    f"{residual_norm:.3e}"
                )
        raise RuntimeError(
            "Direct 3D FDM Newton did not converge; "
            f"last residual={residual_norm:.3e}"
        )

    def electrode_current_density(self, film, electrode_value):
        bottom = self.film_index[:, :, 0]
        bottom_values = film[bottom.ravel()].reshape(bottom.shape)
        return float(np.mean(
            2.0 * self.parameters.diffusion_b
            *(float(electrode_value) - bottom_values)
            /self.geometry["dz"]
        ))


def simulate_direct_fdm(time, grid=None, parameters=None, max_iterations=16):
    """Run the posterior-only monolithic three-dimensional FDM reference."""
    grid = curved.CurvedFilmGrid() if grid is None else grid
    parameters = curved.CurvedFilmPhysics() if parameters is None else parameters
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be a strictly increasing one-dimensional grid")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-10, atol=1e-14):
        raise ValueError("The direct 3D FDM currently requires a uniform grid")
    theta, electrode = physics.triangular_protocol_numpy(time)
    solver = DirectCurvedFilm3DFdm(grid, parameters, dt)
    film, external, flux = solver.initial_state()
    n_time = len(time)
    current = np.zeros(n_time)
    mean_flux = np.zeros(n_time)
    b_inventory = np.zeros(n_time)
    d_inventory = np.zeros(n_time)
    residual = np.zeros(n_time)
    iterations = np.zeros(n_time, dtype=np.int64)
    flux_history = np.zeros((n_time, solver.n_faces))
    b_interface = np.zeros_like(flux_history)
    c_interface = np.full_like(flux_history, parameters.gamma)

    for step in range(1, n_time):
        state = solver.advance(
            film,
            external,
            flux,
            electrode[step],
            max_iterations=max_iterations,
        )
        film = state["film"]
        external = state["external"]
        flux = state["flux"]
        flux_history[step] = flux
        b_interface[step] = state["C_B_int"]
        c_interface[step] = state["C_C_int"]
        current[step] = solver.electrode_current_density(film, electrode[step])
        mean_flux[step] = float(
            np.sum(flux * solver.face_area) / np.sum(solver.face_area)
        )
        b_inventory[step] = float(np.mean(film))
        d_inventory[step] = float(np.mean(external) / parameters.gamma)
        residual[step] = state["residual"]
        iterations[step] = state["iterations"]
    return {
        "time": time,
        "theta": theta,
        "C_B_surface": electrode,
        "electrode_current_density": current,
        "mean_reaction_flux": mean_flux,
        "C_B_inventory_mean": b_inventory,
        "C_D_over_gamma_inventory_mean": d_inventory,
        "J_face": flux_history,
        "C_B_int": b_interface,
        "C_C_int": c_interface,
        "face_centers": solver.face_centers,
        "face_area": solver.face_area,
        "residual": residual,
        "newton_iterations": iterations,
        "grid": grid,
        "parameters": parameters,
    }
