import unittest

import torch

import pinn_thin_layer_v9_6 as pinn


class JointParameterizationTests(unittest.TestCase):
    def setUp(self):
        pinn.configure_physical_parameters(10.0, 1.0)
        self.thin, self.ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_kgparam",
            green_time_grid=32,
            green_kernel_points=8,
        )
        self.thin.eval()
        self.ext.eval()

    def test_reference_matches_fixed_product_integral(self):
        fixed_thin, fixed_ext = pinn.create_models_v96(
            "multiscale_film_tracegreen_productintegral",
            green_time_grid=32,
            green_kernel_points=8,
        )
        pinn.load_compatible_state_dict(self.thin, fixed_thin.state_dict(), "joint thin")
        pinn.load_compatible_state_dict(self.ext, fixed_ext.state_dict(), "joint ext")
        fixed_thin.eval()
        fixed_ext.eval()
        pinn.set_model_k_cat(self.thin, self.ext, 1.0)
        pinn.set_model_gamma(self.thin, self.ext, 10.0)
        time = torch.linspace(0.0, float(pinn.T_sim), 24).reshape(-1, 1)
        x_thin = torch.linspace(0.0, float(pinn.delta), 24).reshape(-1, 1)
        x_ext = torch.linspace(float(pinn.delta), float(pinn.X_ext_max), 24).reshape(-1, 1)
        with torch.no_grad():
            fixed_ab = torch.cat(fixed_thin(torch.cat([time, x_thin], dim=1)), dim=1)
            joint_ab = torch.cat(self.thin(pinn.conditioned_model_inputs(
                self.thin, time, x_thin, (1.0, 10.0)
            )), dim=1)
            fixed_cd = torch.cat(fixed_ext(torch.cat([time, x_ext], dim=1)), dim=1)
            joint_cd = torch.cat(self.ext(pinn.conditioned_model_inputs(
                self.ext, time, x_ext, (1.0, 10.0)
            )), dim=1)
        torch.testing.assert_close(fixed_ab, joint_ab, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(fixed_cd, joint_cd, rtol=1e-6, atol=1e-6)

    def test_interaction_gate_vanishes_on_both_reference_axes(self):
        time = torch.ones(4, 1) * 0.2
        for k_value, gamma_value in ((1.0, 0.1), (1.0, 100.0), (0.1, 10.0), (10.0, 10.0)):
            self.ext.interface_state.set_conditions(k_value, gamma_value)
            eta_k = self.ext.interface_state.log_k_condition(time)
            eta_gamma = self.ext.interface_state.log_gamma_condition(time)
            self.assertLessEqual(float(torch.max(torch.abs(eta_k * eta_gamma))), 1e-12)

    def test_pair_switch_invalidates_history_and_is_repeatable(self):
        self.ext.train()
        time = torch.linspace(0.0, float(pinn.T_sim), 16).reshape(-1, 1)
        self.ext.interface_state.set_conditions(0.1, 0.1)
        first = self.ext.interface_state(time)["C_C_int"].detach().clone()
        self.assertIn(("kg", 0.1, 0.1), self.ext.interface_state._film_abel_cache[0])
        self.ext.interface_state.set_conditions(10.0, 100.0)
        self.assertIsNone(self.ext.interface_state._film_abel_cache)
        self.ext.interface_state(time)
        self.ext.interface_state.set_conditions(0.1, 0.1)
        repeated = self.ext.interface_state(time)["C_C_int"].detach()
        torch.testing.assert_close(first, repeated, rtol=0.0, atol=1e-7)

    def test_interaction_adapter_receives_finite_gradient_off_axes(self):
        self.thin.train()
        time = torch.linspace(0.05, 0.95, 12).reshape(-1, 1)
        x_thin = torch.linspace(0.0, float(pinn.delta), 12).reshape(-1, 1)
        inputs = pinn.conditioned_model_inputs(
            self.thin, time, x_thin, (0.1, 0.1)
        )
        _, c_b = self.thin(inputs)
        c_b.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in self.thin.kg_adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(float(torch.max(torch.abs(gradient))) > 0.0 for gradient in gradients))

    def test_joint_product_integral_is_finite_on_validation_grid(self):
        state = self.ext.interface_state
        for k_value, gamma_value in pinn.KGPARAM_VALIDATION_PAIRS:
            state.set_conditions(k_value, gamma_value)
            with torch.no_grad():
                history = state._history_grid(torch.zeros(1, 1))
            diagnostics = state._last_product_integral_diagnostics
            self.assertTrue(bool(diagnostics["all_finite"]))
            self.assertLessEqual(float(diagnostics["bounds_max"]), 1e-6)
            fraction = history["C_D_fraction"]
            self.assertGreaterEqual(float(fraction.min()), -1e-6)
            self.assertLessEqual(float(fraction.max()), 1.0 + 1e-6)

    def test_checkpoint_metadata_records_physics_only_training(self):
        metadata = pinn.checkpoint_parameters(self.ext)
        self.assertEqual(metadata["parameterization"], pinn.KGPARAM_PARAMETERIZATION)
        self.assertEqual(metadata["training_data"], "physics_only")
        self.assertFalse(metadata["fdm_used_for_training"])

    def test_checkpoint_validation_separates_warm_start_and_resume(self):
        fixed = {
            "parameters": {
                "gamma": 10.0,
                "k_cat_star": 1.0,
                "delta": pinn.delta,
            }
        }
        with self.assertRaises(ValueError):
            pinn.validate_checkpoint_physical_parameters(
                fixed, "fixed.pth", model_ext=self.ext
            )
        pinn.validate_checkpoint_physical_parameters(
            fixed,
            "fixed.pth",
            model_ext=self.ext,
            evaluation_k=0.1,
            evaluation_gamma=100.0,
            allow_fixed_reference=True,
        )

        joint = {"parameters": pinn.checkpoint_parameters(self.ext)}
        pinn.validate_checkpoint_physical_parameters(
            joint,
            "joint.pth",
            model_ext=self.ext,
            evaluation_k=100.0,
            evaluation_gamma=0.1,
        )
        leaked = {"parameters": dict(joint["parameters"])}
        leaked["parameters"]["fdm_used_for_training"] = True
        with self.assertRaises(ValueError):
            pinn.validate_checkpoint_physical_parameters(
                leaked, "leaked.pth", model_ext=self.ext
            )

    def test_joint_training_rejects_in_training_fdm(self):
        with self.assertRaisesRegex(ValueError, "physics-only"):
            pinn.train_model_v9_6(
                self.thin,
                self.ext,
                n_epochs=1,
                fdm_compare_pkl="forbidden.pkl",
                fdm_compare_every=1,
            )

    def test_joint_batch_requires_one_shared_pair(self):
        inputs = torch.tensor([
            [0.1, 0.0, 0.1, 1.0],
            [0.2, 0.01, 10.0, 1.0],
        ])
        with self.assertRaises(ValueError):
            pinn.activate_kg_from_input(self.ext.interface_state, inputs)


if __name__ == "__main__":
    unittest.main()
