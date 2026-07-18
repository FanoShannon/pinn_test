import unittest

import numpy as np

import curved_pi_dtn_bie as curved
import productintegral_dtn as planar
import surface_heat_bem as heat_bem


class CurvedPiDtnBieTests(unittest.TestCase):
    def test_flat_adjoint_double_layer_is_zero(self):
        surface = heat_bem.periodic_cap_surface(
            n_x=3, n_y=3, cap_height=0.0
        )
        operator = heat_bem.AbelSplitHeatSingleLayer(
            surface, periodic_images=2
        )
        np.testing.assert_allclose(
            operator.normal_derivative_matrix(0.02), 0.0, atol=1e-14
        )

    def test_flat_uniform_coupled_limit_matches_planar_operator(self):
        time = np.linspace(0.0, 1.0, 33)
        config = curved.CurvedSurfaceConfig(
            n_x=3,
            n_y=3,
            base_thickness=0.075,
            cap_height=0.0,
            periodic_images=2,
        )
        parameters = curved.CurvedSurfacePhysics(
            heterogeneity_x=0.0,
            heterogeneity_xy=0.0,
        )
        result = curved.simulate_curved_pi_dtn_bie(
            time,
            config,
            parameters,
            n_film_modes=48,
            history_quadrature=5,
        )
        reference = planar.solve_coupled_operator(
            time,
            gamma=10.0,
            k_cat=1.0,
            n_modes=48,
            newton_iterations=20,
            delta=0.075,
        )
        state = planar.reconstruct_state(reference, np.linspace(0.0, 0.075, 8))

        def relative_rmse(candidate, expected):
            return np.sqrt(np.mean((candidate - expected) ** 2)) / np.max(
                np.abs(expected)
            )

        self.assertLess(
            relative_rmse(result["mean_reaction_flux"], reference["J_rxn"]),
            0.01,
        )
        self.assertLess(
            relative_rmse(
                result["electrode_current_density"], -state["J_surface"]
            ),
            0.01,
        )

    def test_curved_complete_recurrence_is_finite_and_closed(self):
        time = np.linspace(0.0, 0.1, 17)
        result = curved.simulate_curved_pi_dtn_bie(
            time,
            curved.CurvedSurfaceConfig(
                n_x=3, n_y=3, periodic_images=1
            ),
            curved.CurvedSurfacePhysics(),
            n_film_modes=32,
            history_quadrature=4,
            max_iterations=30,
        )
        for key in (
            "electrode_current_density",
            "mean_reaction_flux",
            "C_B_int",
            "C_C_int",
            "J_face",
        ):
            self.assertTrue(np.all(np.isfinite(result[key])))
        self.assertLess(np.max(result["closure_residual"]), 2e-9)
        self.assertLess(np.max(result["bie_boundary_residual"]), 1e-11)
        self.assertGreaterEqual(np.min(result["C_B_int"]), -1e-12)
        self.assertGreaterEqual(np.min(result["C_C_int"]), -1e-12)


if __name__ == "__main__":
    unittest.main()
