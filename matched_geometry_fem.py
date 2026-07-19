"""Independent smooth-geometry FEM posterior for curved interface operators.

The reference uses periodic layered tetrahedra whose film/external interface is
exactly the triangular graph used by the surface BIE.  It is intentionally a
volume method: no BIE matrix, ProductIntegral weight, neural network, FDM file,
or fitted correction enters its construction.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

import curved_pi_dtn_bie as curved
import physical_model as physics


def _tetra_local_matrices(points):
    coordinates = np.ones((4, 4), dtype=np.float64)
    coordinates[:, 1:] = np.asarray(points, dtype=np.float64)
    determinant = np.linalg.det(coordinates)
    volume = abs(determinant) / 6.0
    if volume <= 1e-18:
        raise ValueError("Degenerate tetrahedron in matched-geometry FEM")
    gradients = np.linalg.inv(coordinates)[1:, :].T
    stiffness = volume * (gradients @ gradients.T)
    mass = np.full((4, 4), volume / 20.0, dtype=np.float64)
    np.fill_diagonal(mass, volume / 10.0)
    return mass, stiffness


def _structured_layer_coordinates(surface, n_x, n_y, fractions, region, top_z):
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.ndim != 1 or len(fractions) < 2:
        raise ValueError("fractions must contain at least two levels")
    if not np.isclose(fractions[0], 0.0) or not np.isclose(fractions[-1], 1.0):
        raise ValueError("fractions must span [0, 1]")
    if not np.all(np.diff(fractions) > 0.0):
        raise ValueError("fractions must be strictly increasing")
    expected = (int(n_x) + 1) * (int(n_y) + 1)
    if len(surface.vertices) != expected:
        raise ValueError("Matched FEM requires the structured periodic cap mesh")
    height = surface.vertices[:, 2]
    if region == "film":
        z = height[:, None] * fractions[None, :]
    elif region == "external":
        if float(top_z) <= np.max(height):
            raise ValueError("external top_z must lie above the curved interface")
        z = height[:, None] + (
            float(top_z) - height[:, None]
        ) * fractions[None, :]
    else:
        raise ValueError("region must be 'film' or 'external'")
    xy = surface.vertices[:, :2]
    coordinates = np.empty((expected, len(fractions), 3), dtype=np.float64)
    coordinates[:, :, :2] = xy[:, None, :]
    coordinates[:, :, 2] = z
    return coordinates


class PeriodicLayeredTetDomain:
    """P1 FEM domain between two periodic structured graph surfaces."""

    _cube_tetrahedra = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )

    def __init__(
        self,
        surface,
        n_x,
        n_y,
        fractions,
        region,
        diffusion,
        dt,
        top_z=3.0,
    ):
        self.surface = surface
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.fractions = np.asarray(fractions, dtype=np.float64)
        self.n_levels = len(self.fractions)
        self.region = str(region)
        self.diffusion = float(diffusion)
        self.dt = float(dt)
        if self.diffusion <= 0.0 or self.dt <= 0.0:
            raise ValueError("diffusion and dt must be positive")
        coordinates = _structured_layer_coordinates(
            surface,
            self.n_x,
            self.n_y,
            self.fractions,
            self.region,
            top_z,
        )
        self._coordinates = coordinates
        self.n_nodes = self.n_x * self.n_y * self.n_levels
        mass_rows = []
        mass_cols = []
        mass_data = []
        stiffness_rows = []
        stiffness_cols = []
        stiffness_data = []

        def reduced_node(i, j, level):
            return (
                ((i % self.n_x) * self.n_y + (j % self.n_y))
                *self.n_levels + level
            )

        def full_xy(i, j):
            return i * (self.n_y + 1) + j

        for i in range(self.n_x):
            for j in range(self.n_y):
                for level in range(self.n_levels - 1):
                    corners = (
                        (i, j, level),
                        (i + 1, j, level),
                        (i, j + 1, level),
                        (i + 1, j + 1, level),
                        (i, j, level + 1),
                        (i + 1, j, level + 1),
                        (i, j + 1, level + 1),
                        (i + 1, j + 1, level + 1),
                    )
                    local_points = np.array([
                        coordinates[full_xy(ii, jj), kk]
                        for ii, jj, kk in corners
                    ])
                    local_nodes = np.array([
                        reduced_node(ii, jj, kk) for ii, jj, kk in corners
                    ])
                    for tetrahedron in self._cube_tetrahedra:
                        tetrahedron = np.asarray(tetrahedron, dtype=np.int64)
                        nodes = local_nodes[tetrahedron]
                        local_mass, local_stiffness = _tetra_local_matrices(
                            local_points[tetrahedron]
                        )
                        for local_i, node_i in enumerate(nodes):
                            for local_j, node_j in enumerate(nodes):
                                mass_rows.append(node_i)
                                mass_cols.append(node_j)
                                mass_data.append(local_mass[local_i, local_j])
                                stiffness_rows.append(node_i)
                                stiffness_cols.append(node_j)
                                stiffness_data.append(
                                    local_stiffness[local_i, local_j]
                                )
        shape = (self.n_nodes, self.n_nodes)
        self.mass = sparse.coo_matrix(
            (mass_data, (mass_rows, mass_cols)), shape=shape
        ).tocsr()
        self.stiffness = sparse.coo_matrix(
            (stiffness_data, (stiffness_rows, stiffness_cols)), shape=shape
        ).tocsr()
        self.system = self.mass / self.dt + self.diffusion * self.stiffness

        fixed_level = 0 if self.region == "film" else self.n_levels - 1
        self.dirichlet = np.array([
            reduced_node(i, j, fixed_level)
            for i in range(self.n_x) for j in range(self.n_y)
        ], dtype=np.int64)
        fixed_mask = np.zeros(self.n_nodes, dtype=bool)
        fixed_mask[self.dirichlet] = True
        self.free = np.flatnonzero(~fixed_mask)
        self.system_ff = self.system[self.free][:, self.free].tocsc()
        self.system_fd = self.system[self.free][:, self.dirichlet].tocsr()
        self.lu = sparse_linalg.splu(self.system_ff)

        interface_level = self.n_levels - 1 if self.region == "film" else 0
        n_faces = len(surface.triangles)
        boundary = np.zeros((self.n_nodes, n_faces), dtype=np.float64)
        trace = np.zeros((n_faces, self.n_nodes), dtype=np.float64)
        for face, triangle in enumerate(surface.triangles):
            for full_vertex in triangle:
                i = int(full_vertex) // (self.n_y + 1)
                j = int(full_vertex) % (self.n_y + 1)
                node = reduced_node(i, j, interface_level)
                boundary[node, face] += surface.areas[face] / 3.0
                trace[face, node] += 1.0 / 3.0
        self.boundary = boundary
        self.trace = trace
        self.flux_sign = -1.0 if self.region == "film" else 1.0
        self.response_free = self.lu.solve(
            self.flux_sign * self.boundary[self.free]
        )
        self.trace_response = self.trace[:, self.free] @ self.response_free
        self.projected_area = float(np.prod(surface.periodic_lengths))

    def initial_state(self, dirichlet_value=0.0):
        state = np.zeros(self.n_nodes, dtype=np.float64)
        state[self.dirichlet] = float(dirichlet_value)
        return state

    def affine_step(self, previous_state, dirichlet_value=0.0):
        previous_state = np.asarray(previous_state, dtype=np.float64)
        dirichlet = np.full(len(self.dirichlet), float(dirichlet_value))
        rhs = (
            self.mass[self.free] @ previous_state / self.dt
            -self.system_fd @ dirichlet
        )
        free_intercept = self.lu.solve(np.asarray(rhs).ravel())
        state_intercept = np.zeros(self.n_nodes, dtype=np.float64)
        state_intercept[self.free] = free_intercept
        state_intercept[self.dirichlet] = dirichlet
        return {
            "state_intercept": state_intercept,
            "trace_intercept": self.trace @ state_intercept,
            "trace_response": self.trace_response,
        }

    def update(self, affine, flux):
        state = np.asarray(affine["state_intercept"], dtype=np.float64).copy()
        state[self.free] += self.response_free @ np.asarray(flux, dtype=np.float64)
        return state

    def dirichlet_boundary_flux(self, state, previous_state, interface_flux):
        residual = (
            self.mass @ (np.asarray(state) - np.asarray(previous_state)) / self.dt
            +self.diffusion * (self.stiffness @ np.asarray(state))
            -self.flux_sign * self.boundary @ np.asarray(interface_flux)
        )
        return float(np.sum(residual[self.dirichlet]) / self.projected_area)

    def inventory(self, state):
        return float(np.sum(self.mass @ np.asarray(state)))


@dataclass(frozen=True)
class MatchedFemConfig:
    film_layers: int = 10
    external_layers: int = 40
    external_top: float = 3.0
    external_grading_power: float = 2.0

    def validate(self):
        if self.film_layers < 2 or self.external_layers < 4:
            raise ValueError("Insufficient matched-FEM layers")
        if self.external_top <= 0.0 or self.external_grading_power <= 0.0:
            raise ValueError("External FEM dimensions must be positive")


class MatchedGeometryFemOperator:
    """Film/external affine transport maps on one smooth periodic geometry."""

    def __init__(self, surface_config, parameters, dt, fem_config=None):
        self.surface_config = surface_config
        self.parameters = parameters
        self.fem_config = MatchedFemConfig() if fem_config is None else fem_config
        self.fem_config.validate()
        self.surface = surface_config.build_surface()
        film_fraction = np.linspace(
            0.0, 1.0, self.fem_config.film_layers + 1
        )
        raw_external = np.linspace(
            0.0, 1.0, self.fem_config.external_layers + 1
        )
        external_fraction = raw_external ** self.fem_config.external_grading_power
        self.film = PeriodicLayeredTetDomain(
            self.surface,
            surface_config.n_x,
            surface_config.n_y,
            film_fraction,
            "film",
            parameters.diffusion_b,
            dt,
            top_z=self.fem_config.external_top,
        )
        self.external = PeriodicLayeredTetDomain(
            self.surface,
            surface_config.n_x,
            surface_config.n_y,
            external_fraction,
            "external",
            parameters.diffusion_d,
            dt,
            top_z=self.fem_config.external_top,
        )
        centers = self.surface.centroids
        x_scale = max(0.5 * surface_config.length_x, 1e-15)
        y_scale = max(0.5 * surface_config.length_y, 1e-15)
        exponent = (
            parameters.heterogeneity_x * centers[:, 0] / x_scale
            +parameters.heterogeneity_xy * centers[:, 0] * centers[:, 1]
            /(x_scale * y_scale)
        )
        self.k_face = parameters.k_cat * np.exp(exponent)

    def initial_state(self, electrode_value):
        return {
            "film": self.film.initial_state(electrode_value),
            "external": self.external.initial_state(0.0),
            "flux": np.zeros(len(self.surface.triangles), dtype=np.float64),
        }

    def affine_step(self, state, electrode_value):
        film = self.film.affine_step(state["film"], electrode_value)
        external = self.external.affine_step(state["external"], 0.0)
        return {
            "film": film,
            "external": external,
            "b_intercept": film["trace_intercept"],
            "b_response": film["trace_response"],
            "c_intercept": self.parameters.gamma - external["trace_intercept"],
            "c_response": -external["trace_response"],
        }

    def update(self, affine, flux, electrode_value, previous_state):
        del electrode_value
        film_state = self.film.update(affine["film"], flux)
        external_state = self.external.update(affine["external"], flux)
        current = self.film.dirichlet_boundary_flux(
            film_state, previous_state["film"], flux
        )
        return {
            "film": film_state,
            "external": external_state,
            "flux": np.asarray(flux, dtype=np.float64).copy(),
            "electrode_current": current,
        }


def simulate_matched_geometry_fem(
    time,
    surface_config=None,
    parameters=None,
    fem_config=None,
    max_iterations=30,
):
    """Run the smooth-volume posterior with the common nonlinear closure."""
    surface_config = (
        curved.CurvedSurfaceConfig() if surface_config is None else surface_config
    )
    parameters = curved.CurvedSurfacePhysics() if parameters is None else parameters
    parameters.validate()
    time = np.asarray(time, dtype=np.float64)
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-10, atol=1e-14):
        raise ValueError("Matched FEM requires a uniform time grid")
    theta, electrode = physics.triangular_protocol_numpy(time)
    operator = MatchedGeometryFemOperator(
        surface_config, parameters, dt, fem_config=fem_config
    )
    n_time = len(time)
    n_faces = len(operator.surface.triangles)
    state = operator.initial_state(electrode[0])
    flux = np.zeros((n_time, n_faces), dtype=np.float64)
    b_interface = np.zeros_like(flux)
    c_interface = np.full_like(flux, parameters.gamma)
    current = np.zeros(n_time, dtype=np.float64)
    closure = np.zeros(n_time, dtype=np.float64)
    iterations = np.zeros(n_time, dtype=np.int64)
    feedback_condition = np.ones(n_time, dtype=np.float64)
    for step in range(1, n_time):
        affine = operator.affine_step(state, electrode[step])
        value, count, residual = curved._solve_reaction_flux(
            flux[step - 1],
            operator.k_face,
            affine["b_intercept"],
            affine["b_response"],
            affine["c_intercept"],
            affine["c_response"],
            parameters.gamma,
            max_iterations,
        )
        b_value = affine["b_intercept"] + affine["b_response"] @ value
        c_value = affine["c_intercept"] + affine["c_response"] @ value
        jacobian = (
            np.eye(n_faces)
            -np.diag(operator.k_face * c_value) @ affine["b_response"]
            -np.diag(operator.k_face * b_value) @ affine["c_response"]
        )
        next_state = operator.update(
            affine, value, electrode[step], previous_state=state
        )
        flux[step] = value
        b_interface[step] = b_value
        c_interface[step] = c_value
        current[step] = next_state["electrode_current"]
        closure[step] = residual
        iterations[step] = count
        feedback_condition[step] = np.linalg.cond(jacobian)
        state = next_state
    area = operator.surface.areas
    mean_flux = np.sum(flux * area[None, :], axis=1) / np.sum(area)
    return {
        "method": "matched_geometry_periodic_layered_tetra_fem",
        "posterior_reference_only": True,
        "fdm_file_used": False,
        "surface_bie_used": False,
        "neural_network_used": False,
        "time": time,
        "theta": theta,
        "electrode": electrode,
        "electrode_current_density": current,
        "J_face": flux,
        "mean_reaction_flux": mean_flux,
        "C_B_int": b_interface,
        "C_C_int": c_interface,
        "face_area": area,
        "closure_residual": closure,
        "newton_iterations": iterations,
        "feedback_condition": feedback_condition,
        "operator": operator,
    }
