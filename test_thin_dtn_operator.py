import unittest

import numpy as np

import pinn_thin_layer_v9_6 as pinn
import prototype_coupled_productintegral_dtn as coupled


class CoupledThinDtnOperatorTests(unittest.TestCase):
    def tearDown(self):
        pinn.configure_physical_parameters(
            pinn.REFERENCE_GAMMA,
            pinn.REFERENCE_K_CAT_STAR,
        )

    def test_scalar_reaction_closure_root(self):
        value, iterations, residual = coupled.reaction_closure_root(
            b_intercept=0.8,
            b_slope=-0.04,
            c_intercept=3.0,
            c_slope=-0.08,
            k_cat=1.7,
            initial=1.0,
            max_iterations=12,
        )
        expected = 1.7 * (0.8 - 0.04 * value) * (3.0 - 0.08 * value)
        self.assertAlmostEqual(value, expected, places=11)
        self.assertLessEqual(residual, 1e-11)
        self.assertLessEqual(iterations, 12)

    def test_coupled_operator_is_finite_and_closed_across_parameters(self):
        time = np.linspace(0.0, float(pinn.T_sim), 64)
        for k_cat, gamma in (
            (0.1, 0.1),
            (0.01, 1.0),
            (1.0, 1.0),
            (1.0, 10.0),
            (1.0, 100.0),
            (100.0, 10.0),
        ):
            pinn.configure_physical_parameters(gamma, k_cat)
            history = coupled.solve_coupled_operator(
                time,
                gamma,
                k_cat,
                n_modes=24,
                newton_iterations=16,
            )
            for name in (
                "C_B_int",
                "C_C_int",
                "C_D_int",
                "J_rxn",
                "amplitudes",
            ):
                self.assertTrue(np.isfinite(history[name]).all(), name)
            closure = (
                history["J_rxn"]
                -k_cat * history["C_B_int"] * history["C_C_int"]
            )
            self.assertLessEqual(float(np.max(np.abs(closure))), 1e-10)
            self.assertGreaterEqual(float(np.min(history["C_B_int"])), -1e-10)
            self.assertGreaterEqual(float(np.min(history["C_C_int"])), -1e-10)
            self.assertLessEqual(
                float(np.max(history["C_C_int"])),
                gamma * (1.0 + 1e-10),
            )

    def test_initial_state_is_exact(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        time = np.linspace(0.0, float(pinn.T_sim), 32)
        history = coupled.solve_coupled_operator(
            time,
            gamma=10.0,
            k_cat=1.0,
            n_modes=16,
            newton_iterations=12,
        )
        state = coupled.reconstruct_state(
            history,
            np.linspace(0.0, float(pinn.delta), 17),
        )
        self.assertEqual(history["J_rxn"][0], 0.0)
        self.assertEqual(history["C_B_int"][0], 0.0)
        self.assertTrue(np.all(state["C_B"][0] == 0.0))
        self.assertEqual(state["J_surface"][0], 0.0)
        self.assertEqual(state["J_inventory_backward"][0], 0.0)

    def test_fdm_parameter_mismatch_is_rejected(self):
        fdm = {
            "params": {
                "gamma": 10.0,
                "k_cat": 1.0,
                "delta": float(pinn.delta),
                "D_A": 1.0,
                "D_B": 1.0,
                "D_C": 1.0,
                "D_D": 1.0,
            }
        }
        coupled.validate_fdm_parameters(fdm, gamma=10.0, k_cat=1.0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            coupled.validate_fdm_parameters(fdm, gamma=1.0, k_cat=1.0)
        with self.assertRaisesRegex(ValueError, "k_cat"):
            coupled.validate_fdm_parameters(fdm, gamma=10.0, k_cat=0.1)

    def test_tracegreen_preserves_interface_and_far_boundary(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        time = np.linspace(0.0, float(pinn.T_sim), 64)
        history = coupled.solve_coupled_operator(
            time,
            gamma=10.0,
            k_cat=1.0,
            n_modes=16,
            newton_iterations=12,
        )
        time_eval = np.array([0.0, 0.17, 0.51, 0.83, 1.0])
        x_eval = np.array([float(pinn.delta), float(pinn.X_ext_max)])
        _, c_d = coupled.tracegreen_external_field(
            history,
            time_eval,
            x_eval,
            kernel_points=32,
            batch_size=32,
        )
        expected_interface = np.interp(
            time_eval,
            history["time"],
            history["C_D_int"],
        )
        np.testing.assert_allclose(
            c_d[:, 0],
            expected_interface,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            c_d[:, -1],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
