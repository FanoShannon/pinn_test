import tempfile
import unittest
from pathlib import Path

import torch

from core import MinimalKGModel, load_base_checkpoint, save_base_checkpoint


class MinimalModelTest(unittest.TestCase):
    def test_switching_conditions_clears_and_rebuilds_physics(self):
        model = MinimalKGModel(1.0, 10.0, history_points=48, kernel_points=16, lift_points=256)
        t = torch.linspace(0, 1, 17).reshape(-1, 1)
        first = model.interface(t).flux.clone()
        model.set_conditions(0.1, 0.1)
        low = model.interface(t).flux.clone()
        model.set_conditions(1.0, 10.0)
        repeated = model.interface(t).flux.clone()
        self.assertTrue(torch.isfinite(low).all())
        self.assertFalse(torch.allclose(first, low))
        self.assertTrue(torch.allclose(first, repeated, atol=1e-7, rtol=1e-6))

    def test_interface_bounds_flux_sign_and_external_endpoints(self):
        model = MinimalKGModel(10.0, 100.0, history_points=48, kernel_points=16, lift_points=256)
        t = torch.linspace(0, 1, 17).reshape(-1, 1)
        state = model.interface(t)
        self.assertTrue((state.c_d_interface >= 0).all())
        self.assertTrue((state.c_d_interface <= 100.0).all())
        self.assertTrue((state.surface_slope <= 0).all())
        cc_i, cd_i = model.forward_external(t, torch.full_like(t, 0.035))
        cc_f, cd_f = model.forward_external(t, torch.full_like(t, 6.035))
        self.assertTrue(torch.allclose(cd_i, state.c_d_interface, atol=1e-6))
        self.assertTrue(torch.allclose(cc_i + cd_i, torch.full_like(cc_i, 100.0), atol=1e-5))
        self.assertTrue(torch.allclose(cd_f, torch.zeros_like(cd_f), atol=1e-6))

    def test_conservation_and_mandatory_lift(self):
        model = MinimalKGModel(1.0, 10.0, history_points=48, kernel_points=16, lift_points=256)
        t = torch.rand(32, 1)
        x = torch.rand(32, 1) * 0.035
        ca, cb = model.forward_thin(t, x)
        self.assertTrue(torch.allclose(ca + cb, torch.ones_like(ca), atol=1e-6))
        with self.assertRaises(ValueError):
            model.forward_thin(t, x, lifted=False)

    def test_checkpoint_certifies_physics_only(self):
        model = MinimalKGModel(0.3, 2.0, history_points=48, kernel_points=16, lift_points=256)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.pth"
            save_base_checkpoint(path, model, 4)
            loaded, payload = load_base_checkpoint(path)
            self.assertEqual(payload["parameters"]["training_data"], "physics_only")
            self.assertFalse(payload["parameters"]["fdm_used_for_training"])
            self.assertEqual(loaded.base_k, 0.3)

    def test_network_zero_mode_is_exact_and_clears_lift(self):
        model = MinimalKGModel(1.0, 10.0, history_points=48, kernel_points=16, lift_points=256)
        with torch.no_grad():
            model.thin.correction.output.bias.fill_(0.5)
        t = torch.tensor([[0.25], [0.75]])
        x = torch.tensor([[0.01], [0.02]])
        _, trained = model.thin(t, x)
        model.lift._cache[("sentinel",)] = ()
        model.set_network_enabled(False)
        _, physics = model.thin(t, x)
        _, _, raw = model.thin.forward_base(t, x)
        self.assertFalse(torch.allclose(trained, physics))
        self.assertTrue(torch.equal(raw, torch.zeros_like(raw)))
        self.assertEqual(model.lift._cache, {})


if __name__ == "__main__":
    unittest.main()
