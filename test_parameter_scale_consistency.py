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

    def test_characteristic_flux_matches_film_limits(self):
        expected = {
            0.1: 1.0 / 1.035,
            1.0: 10.0 / 1.35,
            10.0: 100.0 / 4.5,
        }
        for k_value, target in expected.items():
            self.assertAlmostEqual(
                pinn.characteristic_reaction_flux(10.0, k_value),
                target,
                places=10,
            )

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

    def test_k_condition_invalidates_history_cache(self):
        model = pinn.InterfaceStateNet_v9_6_FilmTraceKParam(
            time_grid_points=16,
            kernel_points=8,
        )
        model.train()
        times = torch.linspace(0.05, 0.95, 9).reshape(-1, 1)

        model.set_k_cat(0.1)
        low_first = model(times)["J_rxn"].detach()
        self.assertIsNotNone(model._film_abel_cache)
        self.assertIn(0.1, model._film_abel_cache[0])

        model.set_k_cat(10.0)
        self.assertIsNone(model._film_abel_cache)
        high = model(times)["J_rxn"].detach()
        self.assertIn(10.0, model._film_abel_cache[0])

        model.set_k_cat(0.1)
        low_second = model(times)["J_rxn"].detach()
        torch.testing.assert_close(low_first, low_second)
        self.assertFalse(torch.allclose(low_first, high))

    def test_k_reference_matches_fixed_product_integral_architecture(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        pinn.set_seed(321)
        fixed_thin, fixed_ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_productintegral",
            green_time_grid=16,
            green_kernel_points=8,
        )
        pinn.set_seed(654)
        param_thin, param_ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_kparam",
            green_time_grid=16,
            green_kernel_points=8,
        )
        pinn.load_compatible_state_dict(
            param_thin, fixed_thin.state_dict(), "test thin warm-start"
        )
        pinn.load_compatible_state_dict(
            param_ext, fixed_ext.state_dict(), "test external warm-start"
        )
        pinn.set_model_k_cat(param_thin, param_ext, 1.0)
        fixed_thin.eval()
        fixed_ext.eval()
        param_thin.eval()
        param_ext.eval()

        times = torch.linspace(0.05, 0.95, 9).reshape(-1, 1)
        x_thin = torch.linspace(0.0, pinn.delta, 9).reshape(-1, 1)
        x_ext = torch.linspace(pinn.delta, pinn.X_ext_max, 9).reshape(-1, 1)
        with torch.no_grad():
            fixed_ab = torch.cat(fixed_thin(torch.cat([times, x_thin], dim=1)), dim=1)
            param_ab = torch.cat(param_thin(torch.cat([
                times, x_thin, torch.ones_like(times)
            ], dim=1)), dim=1)
            fixed_cd = torch.cat(fixed_ext(torch.cat([times, x_ext], dim=1)), dim=1)
            param_cd = torch.cat(param_ext(torch.cat([
                times, x_ext, torch.ones_like(times)
            ], dim=1)), dim=1)
        torch.testing.assert_close(fixed_ab, param_ab, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(fixed_cd, param_cd, rtol=1e-6, atol=1e-6)

    def test_product_integral_is_stable_across_k_range_and_holdouts(self):
        model = pinn.InterfaceStateNet_v9_6_FilmTraceKParam(
            time_grid_points=256,
            kernel_points=8,
        )
        self.assertTrue(model.product_integral_interface)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

        for k_value in pinn.KPARAM_VALIDATION_VALUES:
            model.set_k_cat(k_value)
            with torch.no_grad():
                history = model._history_grid(torch.zeros(1, 1))
            diagnostics = model._last_product_integral_diagnostics
            self.assertTrue(bool(diagnostics["all_finite"]))
            self.assertLessEqual(float(diagnostics["bounds_max"]), 1e-6 * pinn.gamma)
            self.assertLessEqual(float(diagnostics["fixed_point_max"]), 1e-5 * pinn.gamma)
            self.assertGreaterEqual(float(history["C_D_int"].min()), -1e-6 * pinn.gamma)
            self.assertLessEqual(float(history["C_D_int"].max()), (1.0 + 1e-6) * pinn.gamma)

    def test_checkpoint_validation_distinguishes_resume_and_warm_start(self):
        _, model_ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_kparam",
            green_time_grid=16,
            green_kernel_points=8,
        )
        fixed = {"parameters": {"gamma": 10.0, "delta": pinn.delta, "k_cat_star": 1.0}}
        with self.assertRaises(ValueError):
            pinn.validate_checkpoint_physical_parameters(
                fixed, "fixed.pth", model_ext=model_ext
            )
        pinn.validate_checkpoint_physical_parameters(
            fixed,
            "fixed.pth",
            model_ext=model_ext,
            evaluation_k=1.0,
            allow_fixed_reference=True,
        )

        parameterized = {
            "parameters": {
                "gamma": 10.0,
                "delta": pinn.delta,
                "parameterization": pinn.KPARAM_PARAMETERIZATION,
                "k_support": pinn.KPARAM_SUPPORT,
                "k_reference": 1.0,
                "k_condition_transform": "tanh(log10(k/k_reference))",
                "k_interface_operator": pinn.KPARAM_INTERFACE_OPERATOR,
                "product_integral_fixed_point_iterations": 2,
            }
        }
        pinn.validate_checkpoint_physical_parameters(
            parameterized, "kparam.pth", model_ext=model_ext, evaluation_k=0.001
        )
        old_operator = {"parameters": dict(parameterized["parameters"])}
        old_operator["parameters"]["k_interface_operator"] = "film_abel_clean_v1"
        with self.assertRaises(ValueError):
            pinn.validate_checkpoint_physical_parameters(
                old_operator, "old_kparam.pth", model_ext=model_ext, evaluation_k=1.0
            )
        pinn.validate_checkpoint_physical_parameters(
            parameterized, "kparam.pth", model_ext=model_ext, evaluation_k=1000.0
        )
        with self.assertRaises(ValueError):
            pinn.validate_checkpoint_physical_parameters(
                parameterized, "kparam.pth", model_ext=model_ext, evaluation_k=0.0
            )

    def test_k_input_requires_one_positive_condition_per_batch(self):
        model = pinn.InterfaceStateNet_v9_6_FilmTraceKParam(
            time_grid_points=16,
            kernel_points=8,
        )
        inputs = torch.tensor([[0.1, 0.0, 0.01], [0.2, 0.01, 100.0]])
        with self.assertRaises(ValueError):
            pinn.activate_k_from_input(model, inputs)
        for invalid in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                model.set_k_cat(invalid)

    def test_kparam_film_reaction_has_correct_positive_k_limits(self):
        model = pinn.InterfaceStateNet_v9_6_FilmTraceKParam(
            time_grid_points=16,
            kernel_points=8,
        )
        c_b_surface = torch.tensor([[0.4]], dtype=torch.float64)
        c_c_int = torch.tensor([[10.0]], dtype=torch.float64)

        model.set_k_cat(1e-10)
        _, j_low, _ = model._film_reaction(c_b_surface, c_c_int)
        torch.testing.assert_close(
            j_low,
            1e-10 * c_b_surface * c_c_int,
            rtol=1e-8,
            atol=1e-18,
        )

        model.set_k_cat(1e12)
        c_b_high, j_high, _ = model._film_reaction(c_b_surface, c_c_int)
        transport_limit = (pinn.D_rel_B / pinn.delta) * c_b_surface
        torch.testing.assert_close(j_high, transport_limit, rtol=1e-10, atol=1e-12)
        self.assertLess(float(c_b_high), 1e-9)


if __name__ == "__main__":
    unittest.main()
