"""Triangular-panel heat single-layer prototype with an exact Abel self term."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriangleSurface:
    vertices: np.ndarray
    triangles: np.ndarray
    periodic_lengths: tuple | None = None

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n_vertices, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (n_triangles, 3)")
        if np.min(triangles) < 0 or np.max(triangles) >= len(vertices):
            raise ValueError("triangle connectivity is out of range")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        if self.periodic_lengths is not None:
            lengths = tuple(float(value) for value in self.periodic_lengths)
            if len(lengths) != 2 or min(lengths) <= 0.0:
                raise ValueError("periodic_lengths must contain positive Lx, Ly")
            object.__setattr__(self, "periodic_lengths", lengths)
        if np.min(self.areas) <= 0.0:
            raise ValueError("All triangles must have positive area")

    @property
    def triangle_vertices(self):
        return self.vertices[self.triangles]

    @property
    def centroids(self):
        return np.mean(self.triangle_vertices, axis=1)

    @property
    def area_vectors(self):
        points = self.triangle_vertices
        return 0.5 * np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])

    @property
    def areas(self):
        return np.linalg.norm(self.area_vectors, axis=1)

    @property
    def normals(self):
        return self.area_vectors / self.areas[:, None]


def flat_periodic_square(n_side, length=1.0):
    """Triangulate one periodic square cell in the z=0 plane."""
    n_side = int(n_side)
    if n_side < 2:
        raise ValueError("n_side must be at least 2")
    length = float(length)
    coordinates = np.linspace(-0.5 * length, 0.5 * length, n_side + 1)
    vertices = np.array(
        [(x, y, 0.0) for x in coordinates for y in coordinates],
        dtype=np.float64,
    )
    triangles = []
    stride = n_side + 1
    for i in range(n_side):
        for j in range(n_side):
            lower_left = i * stride + j
            lower_right = (i + 1) * stride + j
            upper_left = i * stride + j + 1
            upper_right = (i + 1) * stride + j + 1
            triangles.append((lower_left, lower_right, upper_right))
            triangles.append((lower_left, upper_right, upper_left))
    return TriangleSurface(
        vertices,
        np.asarray(triangles, dtype=np.int64),
        periodic_lengths=(length, length),
    )


def periodic_cap_surface(
    n_x=6,
    n_y=6,
    length_x=0.8,
    length_y=0.7,
    base_thickness=0.075,
    cap_height=0.12,
    cap_radius=0.26,
    profile="spherical",
):
    """Triangulate a periodic curved graph above a planar electrode."""
    n_x = int(n_x)
    n_y = int(n_y)
    if min(n_x, n_y) < 2:
        raise ValueError("n_x and n_y must be at least 2")
    length_x = float(length_x)
    length_y = float(length_y)
    base_thickness = float(base_thickness)
    cap_height = float(cap_height)
    cap_radius = float(cap_radius)
    if min(length_x, length_y, base_thickness, cap_radius) <= 0.0:
        raise ValueError("Surface lengths and base thickness must be positive")
    if cap_height < 0.0 or cap_height > cap_radius:
        raise ValueError("cap_height must lie in [0, cap_radius]")
    if cap_radius >= 0.5 * min(length_x, length_y):
        raise ValueError("cap_radius must fit inside the periodic cell")
    if profile not in ("spherical", "cosine"):
        raise ValueError("profile must be 'spherical' or 'cosine'")

    x = np.linspace(-0.5 * length_x, 0.5 * length_x, n_x + 1)
    y = np.linspace(-0.5 * length_y, 0.5 * length_y, n_y + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    radius = np.sqrt(xx ** 2 + yy ** 2)
    rise = np.zeros_like(radius)
    inside = radius < cap_radius
    if cap_height > 0.0 and profile == "spherical":
        sphere_radius = (
            cap_radius ** 2 + cap_height ** 2
        ) / (2.0 * cap_height)
        rim_height = np.sqrt(sphere_radius ** 2 - cap_radius ** 2)
        rise[inside] = (
            np.sqrt(sphere_radius ** 2 - radius[inside] ** 2) - rim_height
        )
    elif cap_height > 0.0:
        rise[inside] = 0.5 * cap_height * (
            1.0 + np.cos(np.pi * radius[inside] / cap_radius)
        )
    vertices = np.column_stack((
        xx.ravel(),
        yy.ravel(),
        (base_thickness + rise).ravel(),
    ))

    triangles = []
    stride = n_y + 1
    for i in range(n_x):
        for j in range(n_y):
            lower_left = i * stride + j
            lower_right = (i + 1) * stride + j
            upper_left = i * stride + j + 1
            upper_right = (i + 1) * stride + j + 1
            triangles.append((lower_left, lower_right, upper_right))
            triangles.append((lower_left, upper_right, upper_left))
    return TriangleSurface(
        vertices,
        np.asarray(triangles, dtype=np.int64),
        periodic_lengths=(length_x, length_y),
    )


def spherical_cap_surface(n_radial=4, n_angular=16, radius=0.35,
                          cap_radius=0.25, base_height=0.075):
    """Triangulate a spherical minor cap without imposing axisymmetry in use."""
    n_radial = int(n_radial)
    n_angular = int(n_angular)
    radius = float(radius)
    cap_radius = float(cap_radius)
    if n_radial < 2 or n_angular < 6:
        raise ValueError("The cap requires n_radial >= 2 and n_angular >= 6")
    if not 0.0 < cap_radius < radius:
        raise ValueError("cap_radius must lie inside the sphere radius")
    rim_height = np.sqrt(radius ** 2 - cap_radius ** 2)
    vertices = [(0.0, 0.0, base_height + radius - rim_height)]
    for radial_index in range(1, n_radial + 1):
        radial = cap_radius * radial_index / n_radial
        height = base_height + np.sqrt(radius ** 2 - radial ** 2) - rim_height
        for angular_index in range(n_angular):
            angle = 2.0 * np.pi * angular_index / n_angular
            vertices.append((
                radial * np.cos(angle),
                radial * np.sin(angle),
                height,
            ))
    triangles = []
    for angular_index in range(n_angular):
        current = 1 + angular_index
        following = 1 + (angular_index + 1) % n_angular
        triangles.append((0, current, following))
    for radial_index in range(1, n_radial):
        inner_start = 1 + (radial_index - 1) * n_angular
        outer_start = 1 + radial_index * n_angular
        for angular_index in range(n_angular):
            following = (angular_index + 1) % n_angular
            inner_current = inner_start + angular_index
            inner_following = inner_start + following
            outer_current = outer_start + angular_index
            outer_following = outer_start + following
            triangles.append((inner_current, outer_current, outer_following))
            triangles.append((inner_current, outer_following, inner_following))
    return TriangleSurface(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
    )


def flattened_surface(surface):
    """Project a surface mesh to a plane while preserving its connectivity."""
    vertices = surface.vertices.copy()
    vertices[:, 2] = np.min(vertices[:, 2])
    return TriangleSurface(vertices, surface.triangles.copy())


class AbelSplitHeatSingleLayer:
    """Heat single layer split into universal Abel self and regular geometry."""

    def __init__(self, surface, diffusion=1.0, periodic_images=0):
        self.surface = surface
        self.diffusion = float(diffusion)
        self.periodic_images = int(periodic_images)
        if self.diffusion <= 0.0:
            raise ValueError("diffusion must be positive")
        if self.periodic_images < 0:
            raise ValueError("periodic_images must be nonnegative")
        if self.periodic_images and surface.periodic_lengths is None:
            raise ValueError("Periodic images require a periodic surface")
        self.centroids = surface.centroids
        self.areas = surface.areas
        triangle_points = surface.triangle_vertices
        barycentric = np.array([
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        ])
        self.source_quadrature = np.einsum(
            "qv,pvc->pqc", barycentric, triangle_points
        )
        self.source_weights = np.repeat(
            (self.areas / 3.0)[:, None], 3, axis=1
        )
        self.distance_squared = self._distance_squared((0.0, 0.0, 0.0))
        self.equivalent_disk_beta = self.areas / (
            4.0 * np.pi * self.diffusion
        )

    @property
    def n_panels(self):
        return len(self.areas)

    def _distance_squared(self, shift):
        shifted_source = self.source_quadrature + np.asarray(shift)[None, None, :]
        difference = (
            self.centroids[:, None, None, :] - shifted_source[None, :, :, :]
        )
        return np.sum(difference ** 2, axis=3)

    def abel_kernel(self, lag):
        lag = float(lag)
        if lag <= 0.0:
            raise ValueError("lag must be positive")
        return 1.0 / np.sqrt(np.pi * self.diffusion * lag)

    def regular_matrix(self, lag):
        """Return K_Gamma(lag) minus the diagonal infinite-plane Abel term."""
        lag = float(lag)
        if lag <= 0.0:
            raise ValueError("lag must be positive")
        prefactor = 2.0 / (4.0 * np.pi * self.diffusion * lag) ** 1.5
        matrix = prefactor * np.sum(
            np.exp(-self.distance_squared / (4.0 * self.diffusion * lag))
            *self.source_weights[None, :, :],
            axis=2,
        )
        diagonal = np.arange(self.n_panels)
        matrix[diagonal, diagonal] = (
            -self.abel_kernel(lag)
            *np.exp(-self.equivalent_disk_beta / lag)
        )

        if self.surface.periodic_lengths is not None:
            lx, ly = self.surface.periodic_lengths
            for image_x in range(-self.periodic_images, self.periodic_images + 1):
                for image_y in range(-self.periodic_images, self.periodic_images + 1):
                    if image_x == 0 and image_y == 0:
                        continue
                    distance_squared = self._distance_squared(
                        (image_x * lx, image_y * ly, 0.0)
                    )
                    matrix += prefactor * np.sum(
                        np.exp(
                            -distance_squared / (4.0 * self.diffusion * lag)
                        ) * self.source_weights[None, :, :],
                        axis=2,
                    )
        return matrix

    def full_matrix(self, lag):
        return self.regular_matrix(lag) + np.eye(self.n_panels) * self.abel_kernel(lag)

    def single_layer_matrix(self, lag):
        """Return the standard heat single layer without the image factor two."""
        return 0.5 * self.full_matrix(lag)

    def normal_derivative_matrix(self, lag):
        """Return the target-normal derivative of the standard single layer."""
        lag = float(lag)
        if lag <= 0.0:
            raise ValueError("lag must be positive")
        prefactor = 1.0 / (4.0 * np.pi * self.diffusion * lag) ** 1.5

        def contribution(shift):
            shifted_source = (
                self.source_quadrature + np.asarray(shift)[None, None, :]
            )
            difference = (
                self.centroids[:, None, None, :]
                - shifted_source[None, :, :, :]
            )
            distance_squared = np.sum(difference ** 2, axis=3)
            normal_distance = np.einsum(
                "ipqc,ic->ipq", difference, self.surface.normals
            )
            kernel = (
                -normal_distance
                /(2.0 * self.diffusion * lag)
                *prefactor
                *np.exp(-distance_squared / (4.0 * self.diffusion * lag))
            )
            return np.sum(
                kernel * self.source_weights[None, :, :], axis=2
            )

        matrix = contribution((0.0, 0.0, 0.0))
        np.fill_diagonal(matrix, 0.0)
        if self.surface.periodic_lengths is not None:
            lx, ly = self.surface.periodic_lengths
            for image_x in range(-self.periodic_images, self.periodic_images + 1):
                for image_y in range(-self.periodic_images, self.periodic_images + 1):
                    if image_x == 0 and image_y == 0:
                        continue
                    matrix += contribution((image_x * lx, image_y * ly, 0.0))
        return matrix


def abel_product_integral(time, flux, diffusion):
    """Apply the exact piecewise-linear Abel ProductIntegral panel by panel."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if flux.shape[0] != len(time):
        raise ValueError("flux first dimension must match time")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-12, atol=1e-14):
        raise ValueError("A uniform time grid is required")
    result = np.zeros_like(flux)
    for step in range(1, len(time)):
        cells = np.arange(step, dtype=np.float64)
        lag_hi = (step - cells) * dt
        lag_lo = lag_hi - dt
        left = flux[:step]
        right = flux[1:step + 1]
        slope = (right - left) / dt
        intercept = left + slope * lag_hi[:, None]
        contribution = (
            2.0 * intercept * (np.sqrt(lag_hi) - np.sqrt(lag_lo))[:, None]
            -(2.0 / 3.0) * slope
            *(lag_hi ** 1.5 - lag_lo ** 1.5)[:, None]
        )
        result[step] = np.sum(contribution, axis=0) / np.sqrt(
            np.pi * diffusion
        )
    return result


def convolve_surface_history(operator, time, flux, quadrature_order=6):
    """Convolve panel flux with Abel self plus regular curved-surface history."""
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if flux.shape != (len(time), operator.n_panels):
        raise ValueError("flux must have shape (n_time, n_panels)")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-12, atol=1e-14):
        raise ValueError("A uniform time grid is required")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    cell_fraction = 0.5 * (nodes + 1.0)
    cell_weights = 0.5 * dt * weights
    result = abel_product_integral(time, flux, operator.diffusion)
    regular_cache = {}
    for lag_cells in range(1, len(time)):
        matrices = []
        for fraction in cell_fraction:
            lag = (lag_cells - fraction) * dt
            matrices.append(operator.regular_matrix(lag))
        regular_cache[lag_cells] = matrices

    for step in range(1, len(time)):
        regular_value = np.zeros(operator.n_panels, dtype=np.float64)
        for cell in range(step):
            lag_cells = step - cell
            left = flux[cell]
            right = flux[cell + 1]
            for fraction, weight, matrix in zip(
                cell_fraction, cell_weights, regular_cache[lag_cells]
            ):
                value = (1.0 - fraction) * left + fraction * right
                regular_value += weight * (matrix @ value)
        result[step] += regular_value
    return result
