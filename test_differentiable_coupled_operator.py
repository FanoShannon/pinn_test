import unittest

import numpy as np
import torch

import differentiable_coupled_operator as differentiable
import pinn_thin_layer_v9_6 as pinn
import prototype_coupled_productintegral_dtn as numpy_operator


class DifferentiableCoupledOperatorTests(unittest.TestCase):
    def test_torch_recurrence_matches_numpy_forward(self):
        time = np.linspace(0.0, float(pinn.T_sim), 65)
        for delta in (0.01, 0.035, 0.14):
            expected = numpy_operator.solve_coupled_operator(
                time,
                gamma=10.0,
                k_cat=1.0,
                n_modes=24,
                newton_iterations=16,
                delta=delta,
            )
            actual = differentiable.solve_coupled_operator(
                torch.from_numpy(time),
                gamma=10.0,
                k_cat=1.0,
                delta=torch.tensor(delta, dtype=torch.float64),
                n_modes=24,
                newton_iterations=10,
            )
            for name in ("C_B_int", "C_C_int", "C_D_int", "J_rxn"):
                np.testing.assert_allclose(
                    actual[name].detach().numpy(),
                    expected[name],
                    rtol=2e-11,
                    atol=2e-12,
                    err_msg=f"{name}, delta={delta}",
                )

    def test_log_delta_gradient_matches_centered_difference(self):
        time = torch.linspace(
            0.0,
            float(pinn.T_sim),
            129,
            dtype=torch.float64,
        )
        log_delta = torch.tensor(
            np.log(0.07),
            dtype=torch.float64,
            requires_grad=True,
        )

        def objective(value):
            state = differentiable.solve_coupled_operator(
                time,
                gamma=10.0,
                k_cat=1.0,
                delta=torch.exp(value),
                n_modes=48,
                newton_iterations=10,
            )
            return torch.mean(state["J_surface"][1:] ** 2)

        loss = objective(log_delta)
        gradient = torch.autograd.grad(loss, log_delta)[0]
        step = 1e-5
        with torch.no_grad():
            finite_difference = (
                objective(log_delta + step) - objective(log_delta - step)
            ) / (2.0 * step)
        relative_error = abs(
            float(gradient - finite_difference)
        ) / max(abs(float(finite_difference)), 1e-12)
        self.assertLess(relative_error, 2e-5)

    def test_delta_gradient_is_finite_across_calibration_range(self):
        time = torch.linspace(
            0.0,
            float(pinn.T_sim),
            65,
            dtype=torch.float64,
        )
        for delta_value in (0.01, 0.0175, 0.035, 0.07, 0.14):
            log_delta = torch.tensor(
                np.log(delta_value),
                dtype=torch.float64,
                requires_grad=True,
            )
            state = differentiable.solve_coupled_operator(
                time,
                gamma=10.0,
                k_cat=1.0,
                delta=torch.exp(log_delta),
                n_modes=24,
                newton_iterations=10,
            )
            objective = torch.mean(state["J_surface"][1:] ** 2)
            gradient = torch.autograd.grad(objective, log_delta)[0]
            self.assertTrue(torch.isfinite(objective))
            self.assertTrue(torch.isfinite(gradient))
            self.assertGreater(abs(float(gradient)), 1e-8)

    def test_runtime_scan_period_matches_numpy_operator(self):
        for simulation_time in (0.5, 2.0):
            time = np.linspace(0.0, simulation_time, 65)
            expected = numpy_operator.solve_coupled_operator(
                time,
                gamma=10.0,
                k_cat=1.0,
                n_modes=24,
                newton_iterations=16,
                delta=0.035,
            )
            actual = differentiable.solve_coupled_operator(
                torch.from_numpy(time),
                gamma=10.0,
                k_cat=1.0,
                delta=torch.tensor(0.035, dtype=torch.float64),
                n_modes=24,
                newton_iterations=10,
            )
            np.testing.assert_allclose(
                actual["J_surface"].detach().numpy(),
                numpy_operator.reconstruct_state(
                    expected,
                    np.array([0.0, 0.035]),
                )["J_surface"],
                rtol=2e-11,
                atol=2e-12,
            )

    def test_differentiable_soe_matches_direct_outputs_and_gradient(self):
        time = torch.linspace(
            0.0,
            float(pinn.T_sim),
            129,
            dtype=torch.float64,
        )

        def objective(log_delta, backend):
            state = differentiable.solve_coupled_operator(
                time,
                gamma=10.0,
                k_cat=1.0,
                delta=torch.exp(log_delta),
                n_modes=48,
                newton_iterations=10,
                history_backend=backend,
            )
            return torch.mean(state["J_surface"][1:] ** 2), state

        direct_delta = torch.tensor(
            np.log(0.035),
            dtype=torch.float64,
            requires_grad=True,
        )
        fast_delta = direct_delta.detach().clone().requires_grad_(True)
        direct_loss, direct_state = objective(direct_delta, "direct")
        fast_loss, fast_state = objective(fast_delta, "soe")
        direct_gradient = torch.autograd.grad(direct_loss, direct_delta)[0]
        fast_gradient = torch.autograd.grad(fast_loss, fast_delta)[0]
        np.testing.assert_allclose(
            fast_state["J_rxn"].detach().numpy(),
            direct_state["J_rxn"].detach().numpy(),
            rtol=2e-9,
            atol=2e-10,
        )
        self.assertLess(
            abs(float(fast_gradient - direct_gradient))
            /max(abs(float(direct_gradient)), 1e-15),
            2e-8,
        )

    def test_soe_preserves_joint_parameter_gradient(self):
        time = torch.linspace(
            0.0,
            float(pinn.T_sim),
            65,
            dtype=torch.float64,
        )

        def objective(values, backend):
            state = differentiable.solve_coupled_operator(
                time,
                gamma=torch.exp(values[1]),
                k_cat=torch.exp(values[0]),
                delta=torch.exp(values[2]),
                n_modes=24,
                newton_iterations=10,
                history_backend=backend,
            )
            return (
                torch.mean(state["J_surface"][1:] ** 2)
                +torch.mean(state["C_D_int"][1:] ** 2)
            )

        reference_values = torch.tensor(
            [np.log(1.0), np.log(10.0), np.log(0.035)],
            dtype=torch.float64,
        )
        direct_values = reference_values.clone().requires_grad_(True)
        fast_values = reference_values.clone().requires_grad_(True)
        direct_gradient = torch.autograd.grad(
            objective(direct_values, "direct"),
            direct_values,
        )[0]
        fast_gradient = torch.autograd.grad(
            objective(fast_values, "soe"),
            fast_values,
        )[0]
        np.testing.assert_allclose(
            fast_gradient.detach().numpy(),
            direct_gradient.detach().numpy(),
            rtol=2e-8,
            atol=2e-9,
        )


if __name__ == "__main__":
    unittest.main()
