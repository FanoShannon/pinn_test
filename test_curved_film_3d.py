import unittest

import numpy as np

import curved_film_3d as curved


class CurvedFilm3DTests(unittest.TestCase):
    def test_curved_geometry_is_a_three_dimensional_volume(self):
        grid = curved.CurvedFilmGrid(nx=6, ny=5, nz=12)
        geometry = curved.build_geometry(grid)
        self.assertEqual(geometry["film_mask"].shape, (6, 5, 12))
        self.assertGreater(np.ptp(geometry["height"]), 0.0)
        self.assertTrue(np.all(geometry["film_mask"][:, :, 0]))
        self.assertTrue(np.all(geometry["external_mask"][:, :, -1]))

    def test_nonaxisymmetric_interface_closure_is_finite(self):
        time = np.linspace(0.0, 0.01, 5)
        result = curved.simulate_curved_film_3d(
            time,
            grid=curved.CurvedFilmGrid(nx=5, ny=4, nz=10),
            parameters=curved.CurvedFilmPhysics(
                gamma=2.0,
                k_cat=0.5,
                heterogeneity_x=0.5,
                heterogeneity_xy=0.2,
            ),
            store_fields=False,
        )
        for name in (
            "J_face",
            "C_B_int",
            "C_C_int",
            "C_D_int",
            "electrode_current_density",
        ):
            self.assertTrue(np.isfinite(result[name]).all(), name)
        self.assertLess(np.max(result["closure_residual"]), 1e-9)
        self.assertLess(np.max(result["linear_residual"]), 1e-11)
        self.assertGreaterEqual(np.min(result["J_face"]), 0.0)
        self.assertGreaterEqual(np.min(result["C_B_int"]), -1e-11)
        self.assertGreaterEqual(np.min(result["C_C_int"]), -1e-11)
        self.assertGreater(np.ptp(result["k_face"]), 0.0)

    def test_flat_uniform_case_preserves_lateral_uniformity(self):
        time = np.linspace(0.0, 0.008, 4)
        result = curved.simulate_curved_film_3d(
            time,
            grid=curved.CurvedFilmGrid(
                nx=5,
                ny=4,
                nz=10,
                cap_height=0.0,
            ),
            parameters=curved.CurvedFilmPhysics(
                gamma=1.0,
                k_cat=0.25,
                heterogeneity_x=0.0,
                heterogeneity_xy=0.0,
            ),
            store_fields=False,
        )
        final_flux = result["J_face"][-1]
        np.testing.assert_allclose(
            final_flux,
            np.mean(final_flux),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertLess(result["nonaxisymmetric_index"], 1e-12)


if __name__ == "__main__":
    unittest.main()
