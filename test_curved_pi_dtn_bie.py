import unittest

import numpy as np

import curved_pi_dtn_bie as curved
import productintegral_dtn as planar
import surface_heat_bem as heat_bem
import surface_modal_film_dtn as surface_film


class CurvedPiDtnBieTests(unittest.TestCase):
    def test_surface_modal_film_preserves_constant_mode(self):
        surface = heat_bem.periodic_cap_surface(
            n_x=3, n_y=3, cap_height=0.0, base_thickness=0.075
        )
        film = surface_film.SurfaceModalFilmDtn(
            surface,
            surface.normals[:, 2],
            diffusion=1.0,
            dt=0.01,
            n_modes=12,
        )
        self.assertLess(film.constant_mode_residual(), 1e-12)

        local = curved.LocalColumnFilmDtn(
            surface.centroids[:, 2],
            surface.normals[:, 2],
            diffusion=1.0,
            dt=0.01,
            n_modes=12,
        )
        uniform_flux = np.full(len(surface.triangles), 0.2)
        local_state = local.initial_amplitudes(0.1)
        modal_state = film.initial_amplitudes(0.1)
        local_step = local.affine_step(
            local_state, uniform_flux, 0.1, 0.12, 0.01
        )
        modal_step = film.affine_step(
            modal_state, uniform_flux, 0.1, 0.12, 0.01
        )
        np.testing.assert_allclose(modal_step[1], local_step[1], atol=2e-12)
        np.testing.assert_allclose(
            modal_step[2] @ np.ones(len(uniform_flux)),
            local_step[2] @ np.ones(len(uniform_flux)),
            atol=2e-12,
        )

    def test_surface_modal_film_response_couples_panels(self):
        surface = heat_bem.periodic_cap_surface(n_x=3, n_y=3)
        film = surface_film.SurfaceModalFilmDtn(
            surface,
            surface.normals[:, 2],
            diffusion=1.0,
            dt=0.02,
            n_modes=12,
        )
        amplitudes = film.initial_amplitudes(0.0)
        _, _, response = film.affine_step(
            amplitudes,
            np.zeros(len(surface.triangles)),
            0.0,
            0.1,
            0.02,
        )
        off_diagonal = response - np.diag(np.diag(response))
        self.assertGreater(np.max(np.abs(off_diagonal)), 1e-8)

    def test_flat_tangential_eigenmode_has_expected_decay(self):
        surface = heat_bem.periodic_cap_surface(
            n_x=4, n_y=4, cap_height=0.0, base_thickness=0.075
        )
        dt = 0.01
        film = surface_film.SurfaceModalFilmDtn(
            surface,
            surface.normals[:, 2],
            diffusion=1.0,
            dt=dt,
            n_modes=4,
        )
        eigenvalues, eigenvectors = np.linalg.eig(
            film.fem["laplace_beltrami"]
        )
        order = np.argsort(np.real(eigenvalues))
        index = next(
            item for item in order if np.real(eigenvalues[item]) > 1e-8
        )
        tangential_rate = float(np.real(eigenvalues[index]))
        vector = np.real(eigenvectors[:, index])
        normal_rate = float(film.mu[0, 0] ** 2)
        expected = np.exp(-(normal_rate + tangential_rate) * dt) * vector
        np.testing.assert_allclose(film.decay[0] @ vector, expected, atol=2e-12)

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
        surface_modal_result = curved.simulate_curved_pi_dtn_bie(
            time,
            config,
            parameters,
            n_film_modes=48,
            history_quadrature=5,
            film_model="surface_modal",
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
        np.testing.assert_allclose(
            surface_modal_result["mean_reaction_flux"],
            result["mean_reaction_flux"],
            atol=2e-10,
        )
        np.testing.assert_allclose(
            surface_modal_result["electrode_current_density"],
            result["electrode_current_density"],
            atol=2e-10,
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
            film_model="surface_modal",
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
