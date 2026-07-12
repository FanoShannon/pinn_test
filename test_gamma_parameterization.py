import unittest

try:
    import numpy as np
    import torch
    import pinn_thin_layer_v9_6 as pinn
except ImportError:
    np = None
    torch = None
    pinn = None


@unittest.skipIf(torch is None, "PyTorch is required")
class GammaParameterizationTest(unittest.TestCase):
    def setUp(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        self.thin, self.ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_gammaparam",
            green_time_grid=32,
            green_kernel_points=8,
        )
        self.thin.eval()
        self.ext.eval()

    def evaluate_interface(self, gamma_value):
        pinn.set_model_gamma(self.thin, self.ext, gamma_value)
        time = torch.linspace(0.0, float(pinn.T_sim), 32).reshape(-1, 1)
        state = self.ext.interface_state(time)
        return state, self.ext.interface_state._last_product_integral_diagnostics

    def test_gamma_switch_invalidates_history(self):
        first, _ = self.evaluate_interface(0.1)
        self.evaluate_interface(100.0)
        repeated, _ = self.evaluate_interface(0.1)
        self.assertTrue(torch.allclose(
            first["C_C_int"], repeated["C_C_int"], rtol=0.0, atol=1e-7
        ))

    def test_normalized_product_integral_is_finite_and_bounded(self):
        for gamma_value in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            state, diagnostics = self.evaluate_interface(gamma_value)
            fraction = state["C_C_int"] / gamma_value
            self.assertTrue(torch.isfinite(fraction).all())
            self.assertGreaterEqual(float(fraction.min()), -1e-6)
            self.assertLessEqual(float(fraction.max()), 1.0 + 1e-6)
            self.assertTrue(bool(diagnostics["all_finite"]))

    def test_reference_adapter_gate_is_zero(self):
        pinn.set_model_gamma(self.thin, self.ext, 10.0)
        time = torch.linspace(0.0, float(pinn.T_sim), 16).reshape(-1, 1)
        eta = self.thin.interface_state.log_gamma_condition(time)
        self.assertLessEqual(float(torch.abs(eta).max()), 1e-12)

    def test_gamma10_matches_fixed_product_integral(self):
        fixed_thin, fixed_ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_productintegral",
            green_time_grid=32,
            green_kernel_points=8,
        )
        pinn.load_compatible_state_dict(
            self.thin, fixed_thin.state_dict(), "gamma reference thin"
        )
        pinn.load_compatible_state_dict(
            self.ext, fixed_ext.state_dict(), "gamma reference external"
        )
        fixed_thin.eval()
        fixed_ext.eval()
        pinn.set_model_gamma(self.thin, self.ext, 10.0)
        time = torch.linspace(0.0, float(pinn.T_sim), 32).reshape(-1, 1)
        x_thin = torch.linspace(0.0, float(pinn.delta), 32).reshape(-1, 1)
        fixed = fixed_thin(torch.cat([time, x_thin], dim=1))[1]
        conditioned = self.thin(pinn.conditioned_model_inputs(
            self.thin, time, x_thin, 10.0
        ))[1]
        self.assertLessEqual(float(torch.max(torch.abs(fixed - conditioned))), 1e-6)

    def test_sampling_stays_in_calibration_range(self):
        values = np.asarray([pinn.sample_gammaparam_value(0.0) for _ in range(1000)])
        self.assertTrue(np.all(values >= pinn.GAMMAPARAM_CALIBRATION_RANGE[0]))
        self.assertTrue(np.all(values <= pinn.GAMMAPARAM_CALIBRATION_RANGE[1]))


if __name__ == "__main__":
    unittest.main()
