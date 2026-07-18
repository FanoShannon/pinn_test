import unittest

import numpy as np

import surface_heat_bem as bem


class SurfaceHeatBemTests(unittest.TestCase):
    def test_local_self_kernel_has_exact_abel_limit(self):
        surface = bem.flat_periodic_square(3)
        operator = bem.AbelSplitHeatSingleLayer(surface, periodic_images=1)
        lag = 1e-8
        full = operator.full_matrix(lag)
        expected = operator.abel_kernel(lag)
        np.testing.assert_allclose(
            np.diag(full), expected, rtol=1e-12, atol=0.0
        )

    def test_constant_flux_abel_product_integral_is_exact(self):
        time = np.linspace(0.0, 0.1, 17)
        flux = np.ones((len(time), 3))
        result = bem.abel_product_integral(time, flux, diffusion=1.0)
        expected = 2.0 * np.sqrt(time / np.pi)
        np.testing.assert_allclose(result[:, 0], expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result[:, 1], expected, rtol=2e-14, atol=2e-14)

    def test_spherical_cap_is_a_true_triangle_surface(self):
        surface = bem.spherical_cap_surface(n_radial=3, n_angular=12)
        self.assertGreater(len(surface.triangles), 20)
        self.assertGreater(np.ptp(surface.centroids[:, 2]), 0.0)
        self.assertGreater(np.ptp(surface.normals[:, 0]), 0.0)
        self.assertGreater(np.ptp(surface.normals[:, 1]), 0.0)

    def test_periodic_flat_panels_recover_the_1d_abel_trace(self):
        time = np.linspace(0.0, 0.02, 9)
        amplitude = np.sin(np.pi * time / time[-1]) ** 2
        surface = bem.flat_periodic_square(4)
        operator = bem.AbelSplitHeatSingleLayer(
            surface, periodic_images=2
        )
        flux = np.repeat(amplitude[:, None], operator.n_panels, axis=1)
        trace = bem.convolve_surface_history(operator, time, flux, 5)
        mean_trace = np.sum(
            trace * surface.areas[None, :], axis=1
        ) / np.sum(surface.areas)
        reference = bem.abel_product_integral(time, flux, 1.0)[:, 0]
        error = np.sqrt(np.mean((mean_trace - reference) ** 2))
        scale = np.max(np.abs(reference))
        self.assertLess(error / scale, 0.012)


if __name__ == "__main__":
    unittest.main()
