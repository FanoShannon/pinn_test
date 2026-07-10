import unittest

import numpy as np
import torch

import pinn_thin_layer_v9_6 as pinn


class ParameterScaleConsistencyTests(unittest.TestCase):
    def tearDown(self):
        pinn.configure_physical_parameters(
            pinn.REFERENCE_GAMMA,
            pinn.REFERENCE_K_CAT_STAR,
        )

    def test_reference_calibration_is_unchanged(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        self.assertAlmostEqual(pinn.external_residual_scale(), 1.0)
        self.assertAlmostEqual(pinn.flux_residual_scale(), 1.0)

    def test_smooth_bound_map_is_scale_equivariant_and_differentiable(self):
        normalized_outputs = []
        for gamma in (1.0, 10.0, 100.0):
            values = torch.tensor(
                [[-0.1 * gamma], [0.1 * gamma], [0.5 * gamma],
                 [0.9 * gamma], [1.1 * gamma]],
                dtype=torch.float64,
                requires_grad=True,
            )
            mapped = pinn.smooth_bounded_concentration(
                values,
                gamma,
                transition_fraction=0.02,
                transition_gate=torch.ones_like(values),
            )
            mapped.sum().backward()
            self.assertTrue(torch.all(mapped > 0.0))
            self.assertTrue(torch.all(mapped < gamma))
            self.assertTrue(torch.all(values.grad > 0.0))
            normalized_outputs.append((mapped / gamma).detach().numpy())

        np.testing.assert_allclose(normalized_outputs[0], normalized_outputs[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(normalized_outputs[0], normalized_outputs[2], rtol=1e-10, atol=1e-12)

    def test_stage1_interface_uses_fractional_c_c_state(self):
        normalized_states = []
        times = torch.linspace(0.0, 1.0, 17).reshape(-1, 1)
        for gamma in (1.0, 10.0, 100.0):
            pinn.configure_physical_parameters(gamma, 1.0)
            pinn.set_seed(123)
            model = pinn.InterfaceStateNet_v9_6(normalize_inputs=True)
            state = model(times)["C_C_int"]
            self.assertTrue(torch.all(state >= 0.0))
            self.assertTrue(torch.all(state <= gamma))
            normalized_states.append((state / gamma).detach().numpy())

        np.testing.assert_allclose(normalized_states[0], normalized_states[1], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(normalized_states[0], normalized_states[2], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
