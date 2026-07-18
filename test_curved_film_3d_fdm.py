import unittest

import numpy as np

import curved_film_3d as curved
import curved_film_3d_fdm as fdm


class CurvedFilm3DFdmTests(unittest.TestCase):
    def test_monolithic_fdm_matches_condensed_operator_on_same_grid(self):
        time = np.linspace(0.0, 0.008, 5)
        grid = curved.CurvedFilmGrid(nx=5, ny=4, nz=10)
        parameters = curved.CurvedFilmPhysics(gamma=2.0, k_cat=0.5)
        operator = curved.simulate_curved_film_3d(
            time, grid=grid, parameters=parameters, store_fields=False
        )
        reference = fdm.simulate_direct_fdm(
            time, grid=grid, parameters=parameters
        )
        np.testing.assert_allclose(
            operator["electrode_current_density"],
            reference["electrode_current_density"],
            rtol=0.0,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            operator["J_face"], reference["J_face"], rtol=0.0, atol=2e-11
        )
        self.assertLess(np.max(reference["residual"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
