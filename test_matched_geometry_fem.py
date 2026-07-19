import unittest

import numpy as np

import curved_pi_dtn_bie as curved
import matched_geometry_fem as matched
import productintegral_dtn as planar


class MatchedGeometryFemTests(unittest.TestCase):
    def test_tetra_domain_is_conservative_and_symmetric(self):
        config = curved.CurvedSurfaceConfig(n_x=3, n_y=3)
        parameters = curved.CurvedSurfacePhysics()
        operator = matched.MatchedGeometryFemOperator(
            config,
            parameters,
            dt=0.02,
            fem_config=matched.MatchedFemConfig(
                film_layers=4, external_layers=12
            ),
        )
        for domain in (operator.film, operator.external):
            np.testing.assert_allclose(
                domain.mass.toarray(), domain.mass.toarray().T, atol=1e-14
            )
            np.testing.assert_allclose(
                domain.stiffness.toarray(),
                domain.stiffness.toarray().T,
                atol=1e-13,
            )
            residual = domain.stiffness @ np.ones(domain.n_nodes)
            self.assertLess(np.max(np.abs(residual)), 2e-12)

    def test_flat_uniform_limit_matches_planar_solver(self):
        time = np.linspace(0.0, 1.0, 33)
        config = curved.CurvedSurfaceConfig(
            n_x=3, n_y=3, cap_height=0.0, base_thickness=0.075
        )
        parameters = curved.CurvedSurfacePhysics(
            heterogeneity_x=0.0, heterogeneity_xy=0.0
        )
        result = matched.simulate_matched_geometry_fem(
            time,
            config,
            parameters,
            fem_config=matched.MatchedFemConfig(
                film_layers=6, external_layers=24
            ),
        )
        reference = planar.solve_coupled_operator(
            time,
            gamma=10.0,
            k_cat=1.0,
            delta=0.075,
            n_modes=48,
            newton_iterations=20,
        )
        state = planar.reconstruct_state(
            reference, np.linspace(0.0, 0.075, 8)
        )

        def nrmse(candidate, expected):
            return np.sqrt(np.mean((candidate - expected) ** 2)) / np.max(
                np.abs(expected)
            )

        self.assertLess(
            nrmse(result["mean_reaction_flux"], reference["J_rxn"]), 0.005
        )
        self.assertLess(
            nrmse(
                result["electrode_current_density"], -state["J_surface"]
            ),
            0.005,
        )
        self.assertFalse(result["fdm_file_used"])
        self.assertFalse(result["surface_bie_used"])


if __name__ == "__main__":
    unittest.main()
