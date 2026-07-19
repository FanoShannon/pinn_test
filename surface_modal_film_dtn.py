"""Surface-coupled spectral DtN approximation for a curved finite film.

The normal coordinate retains the half-integer modes of the mature planar
solver.  Their amplitudes are P1 fields on the curved interface and diffuse
with a mass-lumped Laplace--Beltrami operator.  For a flat constant-thickness
film, the constant surface mode is exactly the original one-dimensional DtN.

For variable thickness this is a frozen-normal-basis thin-shell model: it
includes tangential diffusion of every normal mode, but omits derivatives of
the thickness-dependent basis and higher-order curvature coupling.
"""

import numpy as np
from scipy import linalg


def _periodic_vertex_map(surface):
    """Map duplicated periodic-edge vertices onto common surface DOFs."""
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    if surface.periodic_lengths is None:
        return np.arange(len(vertices), dtype=np.int64), vertices.copy()

    lengths = np.asarray(surface.periodic_lengths, dtype=np.float64)
    origin = np.min(vertices[:, :2], axis=0)
    wrapped = vertices.copy()
    wrapped[:, :2] = (
        np.mod(vertices[:, :2] - origin[None, :], lengths[None, :])
        +origin[None, :]
    )
    tolerance = 1e-11 * max(1.0, float(np.max(lengths)))
    for axis in range(2):
        upper = origin[axis] + lengths[axis]
        edge = np.abs(vertices[:, axis] - upper) <= tolerance
        wrapped[edge, axis] = origin[axis]

    full_to_reduced = np.empty(len(vertices), dtype=np.int64)
    key_to_index = {}
    reduced_sum = []
    reduced_count = []
    for full_index, point in enumerate(wrapped):
        key = tuple(np.round(point, decimals=12))
        reduced_index = key_to_index.get(key)
        if reduced_index is None:
            reduced_index = len(reduced_sum)
            key_to_index[key] = reduced_index
            reduced_sum.append(point.copy())
            reduced_count.append(1)
        else:
            reduced_sum[reduced_index] += point
            reduced_count[reduced_index] += 1
        full_to_reduced[full_index] = reduced_index
    reduced_vertices = np.asarray(reduced_sum) / np.asarray(
        reduced_count, dtype=np.float64
    )[:, None]
    return full_to_reduced, reduced_vertices


def assemble_surface_fem(surface):
    """Assemble periodic P1 surface mass/stiffness and face projections."""
    full_to_reduced, reduced_vertices = _periodic_vertex_map(surface)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    reduced_triangles = full_to_reduced[triangles]
    n_nodes = len(reduced_vertices)
    n_faces = len(triangles)
    mass = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    stiffness = np.zeros_like(mass)
    face_to_node_raw = np.zeros((n_nodes, n_faces), dtype=np.float64)
    node_to_face = np.zeros((n_faces, n_nodes), dtype=np.float64)

    reference_gradients = np.array(
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    local_mass_pattern = np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    triangle_points = surface.triangle_vertices
    for face, (full_nodes, nodes, points) in enumerate(
        zip(triangles, reduced_triangles, triangle_points)
    ):
        del full_nodes
        jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
        metric = jacobian.T @ jacobian
        area = 0.5 * np.sqrt(np.linalg.det(metric))
        physical_gradients = (
            jacobian @ np.linalg.solve(metric, reference_gradients.T)
        ).T
        local_mass = (area / 12.0) * local_mass_pattern
        local_stiffness = area * (physical_gradients @ physical_gradients.T)
        for local_i, node_i in enumerate(nodes):
            face_to_node_raw[node_i, face] += area / 3.0
            node_to_face[face, node_i] += 1.0 / 3.0
            for local_j, node_j in enumerate(nodes):
                mass[node_i, node_j] += local_mass[local_i, local_j]
                stiffness[node_i, node_j] += local_stiffness[local_i, local_j]

    lumped_mass = np.sum(mass, axis=1)
    if np.min(lumped_mass) <= 0.0:
        raise ValueError("Surface mesh contains an unsupported isolated node")
    face_to_node = face_to_node_raw / lumped_mass[:, None]

    node_height_sum = np.zeros(n_nodes, dtype=np.float64)
    node_height_count = np.zeros(n_nodes, dtype=np.float64)
    for full_index, reduced_index in enumerate(full_to_reduced):
        node_height_sum[reduced_index] += surface.vertices[full_index, 2]
        node_height_count[reduced_index] += 1.0
    node_height = node_height_sum / node_height_count
    return {
        "mass": mass,
        "lumped_mass": lumped_mass,
        "stiffness": stiffness,
        "laplace_beltrami": stiffness / lumped_mass[:, None],
        "face_to_node": face_to_node,
        "node_to_face": node_to_face,
        "node_height": node_height,
        "full_to_reduced": full_to_reduced,
        "reduced_vertices": reduced_vertices,
    }


class SurfaceModalFilmDtn:
    """Normal spectral modes coupled by surface tangential diffusion."""

    def __init__(
        self,
        surface,
        normal_z,
        diffusion,
        dt,
        n_modes=64,
        tangential_strength=1.0,
    ):
        self.surface = surface
        self.normal_z = np.asarray(normal_z, dtype=np.float64)
        self.diffusion = float(diffusion)
        self.dt = float(dt)
        self.n_modes = int(n_modes)
        self.tangential_strength = float(tangential_strength)
        if self.diffusion <= 0.0 or self.dt <= 0.0:
            raise ValueError("diffusion and dt must be positive")
        if self.n_modes < 1:
            raise ValueError("n_modes must be positive")
        if self.tangential_strength < 0.0:
            raise ValueError("tangential_strength must be nonnegative")
        if np.min(self.normal_z) <= 0.0:
            raise ValueError("The curved interface must remain an upward graph")

        fem = assemble_surface_fem(surface)
        self.fem = fem
        self.face_to_node = fem["face_to_node"]
        self.node_to_face = fem["node_to_face"]
        self.node_height = fem["node_height"]
        self.face_height = np.asarray(surface.centroids[:, 2], dtype=np.float64)
        if min(np.min(self.node_height), np.min(self.face_height)) <= 0.0:
            raise ValueError("All local film thicknesses must be positive")

        self.face_flux_to_node_column_flux = (
            self.face_to_node / self.normal_z[None, :]
        )
        mode = np.arange(self.n_modes, dtype=np.float64)
        self.sign = (-1.0) ** mode
        self.mu = (
            (mode[None, :] + 0.5) * np.pi / self.node_height[:, None]
        )
        laplace = self.tangential_strength * fem["laplace_beltrami"]
        identity = np.eye(len(self.node_height), dtype=np.float64)
        self.decay = []
        self.gain = []
        self.flux_response = []
        for mode_index, sign in enumerate(self.sign):
            mu = self.mu[:, mode_index]
            generator = self.diffusion * (laplace + np.diag(mu ** 2))
            decay = linalg.expm(-self.dt * generator)
            gain = np.linalg.solve(generator, identity - decay)
            flux_coefficient = 2.0 * sign /(
                self.node_height * self.diffusion * mu ** 2
            )
            response = (
                gain
                @ np.diag(flux_coefficient / self.dt)
                @ self.face_flux_to_node_column_flux
            )
            self.decay.append(decay)
            self.gain.append(gain)
            self.flux_response.append(response)

    @property
    def n_nodes(self):
        return len(self.node_height)

    def initial_amplitudes(self, electrode_value):
        return -2.0 * float(electrode_value) /(
            self.node_height[:, None] * self.mu
        )

    def affine_step(
        self,
        previous_amplitudes,
        previous_surface_flux,
        electrode_previous,
        electrode_current,
        dt,
    ):
        if not np.isclose(float(dt), self.dt, rtol=1e-12, atol=1e-15):
            raise ValueError("SurfaceModalFilmDtn requires its construction dt")
        previous_amplitudes = np.asarray(previous_amplitudes, dtype=np.float64)
        previous_surface_flux = np.asarray(previous_surface_flux, dtype=np.float64)
        surface_slope = (
            float(electrode_current) - float(electrode_previous)
        ) / self.dt
        amplitude_intercept = np.empty_like(previous_amplitudes)
        n_faces = len(previous_surface_flux)
        b_response = np.diag(
            -self.face_height /(self.diffusion * self.normal_z)
        )
        b_intercept = np.full(n_faces, float(electrode_current), dtype=np.float64)

        for mode_index, sign in enumerate(self.sign):
            mu = self.mu[:, mode_index]
            electrode_forcing = -2.0 * surface_slope /(
                self.node_height * mu
            )
            response = self.flux_response[mode_index]
            intercept = (
                self.decay[mode_index] @ previous_amplitudes[:, mode_index]
                +self.gain[mode_index] @ electrode_forcing
                -response @ previous_surface_flux
            )
            amplitude_intercept[:, mode_index] = intercept
            b_intercept += sign * (self.node_to_face @ intercept)
            b_response += sign * (self.node_to_face @ response)
        return amplitude_intercept, b_intercept, b_response

    def update_amplitudes(self, amplitude_intercept, surface_flux):
        amplitudes = np.asarray(amplitude_intercept, dtype=np.float64).copy()
        surface_flux = np.asarray(surface_flux, dtype=np.float64)
        for mode_index in range(self.n_modes):
            amplitudes[:, mode_index] += (
                self.flux_response[mode_index] @ surface_flux
            )
        return amplitudes

    def electrode_current(self, amplitudes, surface_flux):
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        column_flux = np.asarray(surface_flux, dtype=np.float64) / self.normal_z
        modal_gradient_nodes = np.sum(amplitudes * self.mu, axis=1)
        return column_flux -self.diffusion * (
            self.node_to_face @ modal_gradient_nodes
        )

    def inventory(self, amplitudes, surface_flux, electrode_value):
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        column_flux = np.asarray(surface_flux, dtype=np.float64) / self.normal_z
        modal_inventory = self.node_to_face @ np.sum(
            amplitudes / self.mu, axis=1
        )
        return (
            self.face_height * float(electrode_value)
            -0.5 * self.face_height ** 2 * column_flux / self.diffusion
            +modal_inventory
        )

    def constant_mode_residual(self):
        ones = np.ones(self.n_nodes, dtype=np.float64)
        return float(np.max(np.abs(self.fem["laplace_beltrami"] @ ones)))

