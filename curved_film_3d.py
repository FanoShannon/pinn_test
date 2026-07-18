"""True-3D curved-film research prototype.

The module keeps all three Cartesian coordinates. A sparse volume discretization
is condensed to an interface response, so the nonlinear solve is performed only
for the two-dimensional reaction-flux field. This is a verification scaffold
for a future boundary-integral DtN implementation, not the final grid-free
solver.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

import physical_model as physics


@dataclass(frozen=True)
class CurvedFilmGrid:
    nx: int = 10
    ny: int = 9
    nz: int = 24
    length_x: float = 0.8
    length_y: float = 0.7
    length_z: float = 0.5
    base_thickness: float = 0.075
    cap_height: float = 0.12
    cap_radius: float = 0.26
    cap_profile: str = "spherical"

    def validate(self):
        if min(self.nx, self.ny) < 4 or self.nz < 6:
            raise ValueError("The 3D grid requires nx, ny >= 4 and nz >= 6")
        for name in (
            "length_x",
            "length_y",
            "length_z",
            "base_thickness",
            "cap_radius",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.cap_height < 0.0:
            raise ValueError("cap_height must be nonnegative")
        if self.base_thickness + self.cap_height >= self.length_z:
            raise ValueError("The curved film must remain below the far boundary")
        if self.cap_radius >= 0.5 * min(self.length_x, self.length_y):
            raise ValueError("cap_radius must fit inside the lateral domain")
        if self.cap_profile not in ("spherical", "cosine"):
            raise ValueError("cap_profile must be 'spherical' or 'cosine'")
        if self.cap_profile == "spherical" and self.cap_height > self.cap_radius:
            raise ValueError("A spherical minor cap requires cap_height <= cap_radius")


@dataclass(frozen=True)
class CurvedFilmPhysics:
    gamma: float = 10.0
    k_cat: float = 1.0
    heterogeneity_x: float = 0.55
    heterogeneity_xy: float = 0.25
    diffusion_b: float = 1.0
    diffusion_d: float = 1.0

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


def cosine_cap_height(x, y, grid):
    """Smooth positive-thickness cap joined to a flat surrounding film."""
    radius = np.sqrt(x ** 2 + y ** 2)
    taper = np.zeros_like(radius)
    inside = radius < grid.cap_radius
    taper[inside] = 0.5 * (
        1.0 + np.cos(np.pi * radius[inside] / grid.cap_radius)
    )
    return grid.base_thickness + grid.cap_height * taper


def spherical_cap_height(x, y, grid):
    """A spherical cap on a positive flat base, without a zero-thickness rim."""
    if grid.cap_height == 0.0:
        return np.full_like(x, grid.base_thickness, dtype=np.float64)
    radius = np.sqrt(x ** 2 + y ** 2)
    sphere_radius = (
        grid.cap_radius ** 2 + grid.cap_height ** 2
    ) / (2.0 * grid.cap_height)
    rim_height = np.sqrt(sphere_radius ** 2 - grid.cap_radius ** 2)
    rise = np.zeros_like(radius)
    inside = radius < grid.cap_radius
    rise[inside] = (
        np.sqrt(sphere_radius ** 2 - radius[inside] ** 2) - rim_height
    )
    return grid.base_thickness + rise


def build_geometry(grid):
    grid.validate()
    dx = grid.length_x / grid.nx
    dy = grid.length_y / grid.ny
    dz = grid.length_z / grid.nz
    x = (np.arange(grid.nx) + 0.5) * dx - 0.5 * grid.length_x
    y = (np.arange(grid.ny) + 0.5) * dy - 0.5 * grid.length_y
    z = (np.arange(grid.nz) + 0.5) * dz
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if grid.cap_profile == "spherical":
        height = spherical_cap_height(xx, yy, grid)
    else:
        height = cosine_cap_height(xx, yy, grid)
    film_mask = z[None, None, :] < height[:, :, None]
    external_mask = ~film_mask
    if not np.all(film_mask[:, :, 0]):
        raise RuntimeError("The film must cover the complete planar electrode")
    if not np.all(external_mask[:, :, -1]):
        raise RuntimeError("The external domain must reach the far boundary")
    return {
        "x": x,
        "y": y,
        "z": z,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "height": height,
        "film_mask": film_mask,
        "external_mask": external_mask,
    }


def _index_map(mask):
    result = -np.ones(mask.shape, dtype=np.int64)
    result[mask] = np.arange(np.count_nonzero(mask), dtype=np.int64)
    return result


def _build_diffusion_matrix(mask, diffusion, spacing, dirichlet_side):
    """Cell-centered finite-volume diffusion with lateral no-flux walls."""
    nx, ny, nz = mask.shape
    index = _index_map(mask)
    rows = []
    cols = []
    data = []
    boundary_source = np.zeros(np.count_nonzero(mask), dtype=np.float64)
    directions = (
        (-1, 0, 0, spacing[0]),
        (1, 0, 0, spacing[0]),
        (0, -1, 0, spacing[1]),
        (0, 1, 0, spacing[1]),
        (0, 0, -1, spacing[2]),
        (0, 0, 1, spacing[2]),
    )
    for i, j, k in np.argwhere(mask):
        row = int(index[i, j, k])
        diagonal = 0.0
        for di, dj, dk, step in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            in_box = 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz
            if in_box and mask[ni, nj, nk]:
                coefficient = diffusion / step ** 2
                rows.append(row)
                cols.append(int(index[ni, nj, nk]))
                data.append(coefficient)
                diagonal -= coefficient
                continue

            is_dirichlet = (
                dirichlet_side == "bottom" and dk == -1 and k == 0
            ) or (
                dirichlet_side == "top" and dk == 1 and k == nz - 1
            )
            if is_dirichlet:
                coefficient = 2.0 * diffusion / step ** 2
                diagonal -= coefficient
                boundary_source[row] += coefficient

        rows.append(row)
        cols.append(row)
        data.append(diagonal)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(boundary_source), len(boundary_source)),
    )
    return matrix, boundary_source, index


def _enumerate_interface_faces(film_mask, film_index, external_index, spacing):
    nx, ny, nz = film_mask.shape
    records = []
    positive_directions = (
        (1, 0, 0, spacing[0], spacing[1] * spacing[2]),
        (0, 1, 0, spacing[1], spacing[0] * spacing[2]),
        (0, 0, 1, spacing[2], spacing[0] * spacing[1]),
    )
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for di, dj, dk, distance, area in positive_directions:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if ni >= nx or nj >= ny or nk >= nz:
                        continue
                    left_film = bool(film_mask[i, j, k])
                    right_film = bool(film_mask[ni, nj, nk])
                    if left_film == right_film:
                        continue
                    if left_film:
                        film_cell = (i, j, k)
                        external_cell = (ni, nj, nk)
                    else:
                        film_cell = (ni, nj, nk)
                        external_cell = (i, j, k)
                    records.append({
                        "film": int(film_index[film_cell]),
                        "external": int(external_index[external_cell]),
                        "film_ijk": film_cell,
                        "external_ijk": external_cell,
                        "center_index": (
                            0.5 * (i + ni + 1.0),
                            0.5 * (j + nj + 1.0),
                            0.5 * (k + nk + 1.0),
                        ),
                        "distance": distance,
                        "area": area,
                    })
    if not records:
        raise RuntimeError("No curved film/external interface faces were found")
    return records


def _face_operators(records, geometry, n_film, n_external):
    n_faces = len(records)
    film_rows = np.array([item["film"] for item in records])
    external_rows = np.array([item["external"] for item in records])
    faces = np.arange(n_faces)
    unit = np.ones(n_faces)
    p_film = sparse.csr_matrix(
        (unit, (faces, film_rows)), shape=(n_faces, n_film)
    )
    p_external = sparse.csr_matrix(
        (unit, (faces, external_rows)), shape=(n_faces, n_external)
    )
    film_volume = geometry["dx"] * geometry["dy"] * geometry["dz"]
    sink = sparse.csc_matrix(
        (
            np.array([item["area"] / film_volume for item in records]),
            (film_rows, faces),
        ),
        shape=(n_film, n_faces),
    )
    source = sparse.csc_matrix(
        (
            np.array([item["area"] / film_volume for item in records]),
            (external_rows, faces),
        ),
        shape=(n_external, n_faces),
    )
    x0 = -0.5 * geometry["x"].size * geometry["dx"]
    y0 = -0.5 * geometry["y"].size * geometry["dy"]
    centers = np.empty((n_faces, 3), dtype=np.float64)
    for face, item in enumerate(records):
        ix, iy, iz = item["center_index"]
        centers[face] = (
            x0 + ix * geometry["dx"],
            y0 + iy * geometry["dy"],
            iz * geometry["dz"],
        )
    half_distance = 0.5 * np.array([item["distance"] for item in records])
    area = np.array([item["area"] for item in records])
    return p_film, p_external, sink, source, centers, half_distance, area


class CurvedFilm3DOperator:
    """Backward-Euler discrete DtN operator for a curved three-dimensional film."""

    def __init__(self, grid, parameters, dt):
        grid.validate()
        parameters.validate()
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.grid = grid
        self.parameters = parameters
        self.dt = float(dt)
        self.geometry = build_geometry(grid)
        spacing = (
            self.geometry["dx"],
            self.geometry["dy"],
            self.geometry["dz"],
        )
        film_matrix, electrode_source, film_index = _build_diffusion_matrix(
            self.geometry["film_mask"],
            parameters.diffusion_b,
            spacing,
            "bottom",
        )
        external_matrix, _, external_index = _build_diffusion_matrix(
            self.geometry["external_mask"],
            parameters.diffusion_d,
            spacing,
            "top",
        )
        records = _enumerate_interface_faces(
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
        ) = _face_operators(
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
        self.film_lu = sparse_linalg.splu(self.film_system)
        self.external_lu = sparse_linalg.splu(self.external_system)

        film_volume_response = self.dt * self.film_lu.solve(
            self.film_sink.toarray()
        )
        external_volume_response = self.dt * self.external_lu.solve(
            self.external_source.toarray()
        )
        self.film_volume_response = film_volume_response
        self.external_volume_response = external_volume_response
        self.film_trace_response = (
            self.p_film @ film_volume_response
            +np.diag(half_distance / parameters.diffusion_b)
        )
        self.external_trace_response = (
            self.p_external @ external_volume_response
            +np.diag(half_distance / parameters.diffusion_d)
        )

        x_scale = max(0.5 * grid.length_x, 1e-15)
        y_scale = max(0.5 * grid.length_y, 1e-15)
        exponent = (
            parameters.heterogeneity_x * self.face_centers[:, 0] / x_scale
            +parameters.heterogeneity_xy
            *self.face_centers[:, 0] * self.face_centers[:, 1]
            /(x_scale * y_scale)
        )
        self.k_face = parameters.k_cat * np.exp(exponent)

    @property
    def n_faces(self):
        return len(self.k_face)

    def initial_state(self):
        return (
            np.zeros(self.film_system.shape[0], dtype=np.float64),
            np.zeros(self.external_system.shape[0], dtype=np.float64),
            np.zeros(self.n_faces, dtype=np.float64),
        )

    def _free_states(self, film_previous, external_previous, electrode_value):
        film_rhs = (
            film_previous
            +self.dt * self.electrode_source * float(electrode_value)
        )
        film_free = self.film_lu.solve(film_rhs)
        external_free = self.external_lu.solve(external_previous)
        return film_free, external_free

    def interface_state(self, film_free, external_free, flux):
        b_free = np.asarray(self.p_film @ film_free).ravel()
        d_free = np.asarray(self.p_external @ external_free).ravel()
        b_interface = b_free - self.film_trace_response @ flux
        d_interface = d_free + self.external_trace_response @ flux
        c_interface = self.parameters.gamma - d_interface
        return b_interface, c_interface, d_interface

    def solve_flux(self, film_free, external_free, initial, max_iterations=20):
        flux = np.maximum(np.asarray(initial, dtype=np.float64), 0.0).copy()
        scale = max(1.0, float(np.max(self.k_face) * self.parameters.gamma))
        residual_norm = np.inf
        for iteration in range(1, max_iterations + 1):
            b_interface, c_interface, _ = self.interface_state(
                film_free, external_free, flux
            )
            residual = flux - self.k_face * b_interface * c_interface
            residual_norm = float(np.max(np.abs(residual)))
            if residual_norm <= 1e-11 * scale:
                return flux, iteration, residual_norm
            jacobian = (
                np.eye(self.n_faces)
                +np.diag(self.k_face * c_interface)
                @self.film_trace_response
                +np.diag(self.k_face * b_interface)
                @self.external_trace_response
            )
            direction = np.linalg.solve(jacobian, -residual)
            accepted = False
            alpha = 1.0
            for _ in range(18):
                candidate = flux + alpha * direction
                b_new, c_new, _ = self.interface_state(
                    film_free, external_free, candidate
                )
                candidate_residual = (
                    candidate - self.k_face * b_new * c_new
                )
                candidate_norm = float(np.max(np.abs(candidate_residual)))
                if (
                    np.all(candidate >= -1e-13)
                    and np.all(b_new >= -1e-13)
                    and np.all(c_new >= -1e-13)
                    and candidate_norm < residual_norm
                ):
                    flux = np.maximum(candidate, 0.0)
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                target = self.k_face * np.maximum(b_interface, 0.0) * np.maximum(
                    c_interface, 0.0
                )
                flux = 0.5 * flux + 0.5 * target
        raise RuntimeError(
            "The curved-film interface Newton solve did not converge; "
            f"last residual={residual_norm:.3e}"
        )

    def advance(self, film_previous, external_previous, previous_flux,
                electrode_value, max_iterations=20):
        film_free, external_free = self._free_states(
            film_previous, external_previous, electrode_value
        )
        flux, iterations, residual = self.solve_flux(
            film_free,
            external_free,
            previous_flux,
            max_iterations=max_iterations,
        )
        film = film_free - self.film_volume_response @ flux
        external = external_free + self.external_volume_response @ flux
        b_interface, c_interface, d_interface = self.interface_state(
            film_free, external_free, flux
        )
        film_rhs = (
            film_previous
            +self.dt * self.electrode_source * float(electrode_value)
            -self.dt * np.asarray(self.film_sink @ flux).ravel()
        )
        external_rhs = (
            external_previous
            +self.dt * np.asarray(self.external_source @ flux).ravel()
        )
        linear_residual = max(
            float(np.max(np.abs(self.film_system @ film - film_rhs))),
            float(np.max(np.abs(
                self.external_system @ external - external_rhs
            ))),
        )
        return {
            "film": film,
            "external": external,
            "flux": flux,
            "C_B_int": b_interface,
            "C_C_int": c_interface,
            "C_D_int": d_interface,
            "iterations": iterations,
            "closure_residual": residual,
            "linear_residual": linear_residual,
        }

    def electrode_current_density(self, film, electrode_value):
        bottom = self.film_index[:, :, 0]
        bottom_values = film[bottom.ravel()].reshape(bottom.shape)
        return (
            2.0 * self.parameters.diffusion_b
            *(float(electrode_value) - bottom_values)
            /self.geometry["dz"]
        )

    def expand_fields(self, film, external):
        shape = self.geometry["film_mask"].shape
        b_field = np.full(shape, np.nan, dtype=np.float64)
        d_field = np.full(shape, np.nan, dtype=np.float64)
        b_field[self.geometry["film_mask"]] = film
        d_field[self.geometry["external_mask"]] = external
        return b_field, d_field


def simulate_curved_film_3d(
    time,
    grid=None,
    parameters=None,
    max_iterations=20,
    store_fields=True,
):
    """Run the full Cartesian curved-film interface-field closure."""
    grid = CurvedFilmGrid() if grid is None else grid
    parameters = CurvedFilmPhysics() if parameters is None else parameters
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 3 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be a strictly increasing one-dimensional grid")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-10, atol=1e-14):
        raise ValueError("The prototype currently requires a uniform time grid")
    _, electrode = physics.triangular_protocol_numpy(time)
    operator = CurvedFilm3DOperator(grid, parameters, dt)
    film, external, previous_flux = operator.initial_state()
    n_time = len(time)
    flux_history = np.zeros((n_time, operator.n_faces), dtype=np.float64)
    b_interface = np.zeros_like(flux_history)
    c_interface = np.full_like(flux_history, parameters.gamma)
    d_interface = np.zeros_like(flux_history)
    current = np.zeros(n_time, dtype=np.float64)
    closure = np.zeros(n_time, dtype=np.float64)
    linear_residual = np.zeros(n_time, dtype=np.float64)
    iterations = np.zeros(n_time, dtype=np.int64)
    film_history = [] if store_fields else None
    external_history = [] if store_fields else None
    if store_fields:
        film_history.append(film.copy())
        external_history.append(external.copy())

    for step in range(1, n_time):
        state = operator.advance(
            film,
            external,
            previous_flux,
            electrode[step],
            max_iterations=max_iterations,
        )
        film = state["film"]
        external = state["external"]
        previous_flux = state["flux"]
        flux_history[step] = previous_flux
        b_interface[step] = state["C_B_int"]
        c_interface[step] = state["C_C_int"]
        d_interface[step] = state["C_D_int"]
        closure[step] = state["closure_residual"]
        linear_residual[step] = state["linear_residual"]
        iterations[step] = state["iterations"]
        current_map = operator.electrode_current_density(film, electrode[step])
        current[step] = float(np.mean(current_map))
        if store_fields:
            film_history.append(film.copy())
            external_history.append(external.copy())

    weighted_mean_flux = (
        np.sum(flux_history * operator.face_area[None, :], axis=1)
        /np.sum(operator.face_area)
    )
    final_flux = flux_history[-1]
    reflected_centers = operator.face_centers.copy()
    reflected_centers[:, 0] *= -1.0
    coordinate_scale = np.array([
        operator.geometry["dx"],
        operator.geometry["dy"],
        operator.geometry["dz"],
    ])
    scaled_distance = (
        (reflected_centers[:, None, :] - operator.face_centers[None, :, :])
        /coordinate_scale[None, None, :]
    )
    reflected_face = np.argmin(np.sum(scaled_distance ** 2, axis=2), axis=1)
    nonaxisymmetric_index = float(
        np.sqrt(np.mean((final_flux - final_flux[reflected_face]) ** 2))
        /max(np.sqrt(np.mean(final_flux ** 2)), 1e-15)
    )
    flux_spatial_cv = float(
        np.std(final_flux) / max(abs(np.mean(final_flux)), 1e-15)
    )
    result = {
        "time": time,
        "C_B_surface": electrode,
        "electrode_current_density": current,
        "mean_reaction_flux": weighted_mean_flux,
        "J_face": flux_history,
        "C_B_int": b_interface,
        "C_C_int": c_interface,
        "C_D_int": d_interface,
        "face_centers": operator.face_centers,
        "face_area": operator.face_area,
        "k_face": operator.k_face,
        "closure_residual": closure,
        "linear_residual": linear_residual,
        "newton_iterations": iterations,
        "nonaxisymmetric_index": nonaxisymmetric_index,
        "flux_spatial_cv": flux_spatial_cv,
        "grid": grid,
        "parameters": parameters,
        "geometry": operator.geometry,
        "operator": operator,
    }
    if store_fields:
        result["film_history"] = np.asarray(film_history)
        result["external_history"] = np.asarray(external_history)
    return result
