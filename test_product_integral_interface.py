import unittest

import torch

import pinn_thin_layer_v9_6 as pinn


class ProductIntegralInterfaceTests(unittest.TestCase):
    def setUp(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        pinn.set_seed(1234)

    def _models(self, time_grid=64):
        return pinn.create_models_v96(
            "multiscale_film_tracegreen_productintegral",
            green_time_grid=time_grid,
            green_kernel_points=8,
            green_cache_history=False,
        )

    def test_clean_checkpoint_state_dict_is_strictly_compatible(self):
        clean_thin, clean_ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_clean",
            green_time_grid=32,
            green_kernel_points=8,
            green_cache_history=False,
        )
        product_thin, product_ext = self._models(time_grid=32)
        product_thin.load_state_dict(clean_thin.state_dict(), strict=True)
        product_ext.load_state_dict(clean_ext.state_dict(), strict=True)

    def test_trace_is_bounded_finite_and_exact_at_initial_time(self):
        _, model_ext = self._models()
        state = model_ext.interface_state
        with torch.no_grad():
            history = state._history_grid(torch.zeros(1, 1))

        c_d = history["C_D_int"]
        self.assertEqual(float(c_d[0]), 0.0)
        self.assertTrue(torch.isfinite(c_d).all())
        self.assertGreaterEqual(float(c_d.min()), 0.0)
        self.assertLessEqual(float(c_d.max()), pinn.gamma)
        self.assertFalse(any(parameter.requires_grad for parameter in state.parameters()))

    def test_two_fixed_point_updates_match_six(self):
        # The production configuration uses 256 history points; the fixed-point
        # contraction is grid-spacing dependent, so test the deployed grid.
        _, model_ext = self._models(time_grid=256)
        state = model_ext.interface_state
        with torch.no_grad():
            state.fixed_point_iterations = 2
            state.clear_step_cache()
            trace_two = state._history_grid(torch.zeros(1, 1))["C_D_int"].clone()
            state.fixed_point_iterations = 6
            state.clear_step_cache()
            trace_six = state._history_grid(torch.zeros(1, 1))["C_D_int"].clone()
        torch.testing.assert_close(trace_two, trace_six, rtol=0.0, atol=2e-6)

    def test_remaining_network_parameters_backpropagate(self):
        model_thin, model_ext = self._models(time_grid=16)
        t = torch.linspace(0.05, 0.95, 8).reshape(-1, 1)
        x_thin = torch.linspace(0.0, pinn.delta, 8).reshape(-1, 1)
        x_ext = torch.linspace(pinn.delta, pinn.X_ext_max, 8).reshape(-1, 1)
        c_a, c_b = model_thin(torch.cat([t, x_thin], dim=1))
        c_c, c_d = model_ext(torch.cat([t, x_ext], dim=1))
        loss = torch.mean(c_a**2 + c_b**2 + c_c**2 + c_d**2)
        loss.backward()

        active_parameters = [
            parameter
            for module in (model_thin, model_ext)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in active_parameters
        ))


if __name__ == "__main__":
    unittest.main()
