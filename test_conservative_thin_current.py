import unittest

import torch

import pinn_thin_layer_v9_6 as pinn


class ConservativeThinCurrentTests(unittest.TestCase):
    def setUp(self):
        pinn.configure_physical_parameters(1.0, 1.0)
        pinn.set_seed(1234)

    def _models(self, arch):
        return pinn.create_models_v96(
            arch,
            green_time_grid=16,
            green_kernel_points=8,
            green_cache_history=False,
        )

    def test_conservative_warm_start_preserves_concentration(self):
        clean_thin, clean_ext = self._models("multiscale_film_tracegreen_clean")
        conservative_thin, conservative_ext = self._models(
            "multiscale_film_tracegreen_clean_conservative"
        )
        conservative_thin.load_state_dict(clean_thin.state_dict(), strict=True)
        conservative_ext.load_state_dict(clean_ext.state_dict(), strict=True)

        t = torch.linspace(0.05, 0.95, 9).reshape(-1, 1)
        x_thin = torch.linspace(0.0, pinn.delta, 9).reshape(-1, 1)
        x_ext = torch.linspace(pinn.delta, pinn.X_ext_max, 9).reshape(-1, 1)
        with torch.no_grad():
            clean_ab = torch.cat(clean_thin(torch.cat([t, x_thin], dim=1)), dim=1)
            conservative_ab = torch.cat(
                conservative_thin(torch.cat([t, x_thin], dim=1)), dim=1
            )
            clean_cd = torch.cat(clean_ext(torch.cat([t, x_ext], dim=1)), dim=1)
            conservative_cd = torch.cat(
                conservative_ext(torch.cat([t, x_ext], dim=1)), dim=1
            )
        torch.testing.assert_close(clean_ab, conservative_ab, rtol=0.0, atol=0.0)
        torch.testing.assert_close(clean_cd, conservative_cd, rtol=0.0, atol=0.0)

    def test_mixed_flux_warm_start_preserves_concentration(self):
        conservative_thin, _ = self._models(
            "multiscale_film_tracegreen_clean_conservative"
        )
        mixed_thin, _ = self._models("multiscale_film_tracegreen_clean_mixedflux")
        pinn.load_compatible_state_dict(
            mixed_thin, conservative_thin.state_dict(), "mixed test warm-start"
        )

        t = torch.linspace(0.05, 0.95, 9).reshape(-1, 1)
        x = torch.linspace(0.0, pinn.delta, 9).reshape(-1, 1)
        with torch.no_grad():
            expected = torch.cat(conservative_thin(torch.cat([t, x], dim=1)), dim=1)
            actual = torch.cat(mixed_thin(torch.cat([t, x], dim=1)), dim=1)
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

    def test_mixed_flux_has_exact_endpoint_states(self):
        model_thin, _ = self._models("multiscale_film_tracegreen_clean_mixedflux")
        t = torch.linspace(0.05, 0.95, 11).reshape(-1, 1)
        x_surface = torch.zeros_like(t)
        x_interface = torch.ones_like(t) * pinn.delta

        q_surface = model_thin.flux_b(torch.cat([t, x_surface], dim=1))
        q_interface = model_thin.flux_b(torch.cat([t, x_interface], dim=1))
        expected_surface = -(
            pinn.D_rel_B / pinn.D_rel_A
        ) * model_thin.surface_current_state(t)
        expected_interface = model_thin.interface_state(t)["J_rxn"]

        torch.testing.assert_close(q_surface, expected_surface, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(q_interface, expected_interface, rtol=1e-6, atol=1e-7)

    def test_current_balance_and_mixed_losses_backpropagate(self):
        conservative_thin, _ = self._models(
            "multiscale_film_tracegreen_clean_conservative"
        )
        t = torch.linspace(0.05, 0.95, 12).reshape(-1, 1).requires_grad_(True)
        current = pinn.thin_current_components(
            conservative_thin, t, quadrature_points=8, create_graph=True
        )
        balance_loss = torch.mean(current["balance_residual"] ** 2)
        self.assertTrue(torch.isfinite(balance_loss))
        balance_loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in conservative_thin.parameters()
        ))

        mixed_thin, _ = self._models("multiscale_film_tracegreen_clean_mixedflux")
        t = torch.linspace(0.05, 0.95, 12).reshape(-1, 1).requires_grad_(True)
        x = torch.linspace(0.001, pinn.delta - 0.001, 12).reshape(-1, 1)
        x.requires_grad_(True)
        inputs = torch.cat([t, x], dim=1)
        _, c_b = mixed_thin(inputs)
        q_b = mixed_thin.flux_b(inputs)
        c_b_t = torch.autograd.grad(c_b.sum(), t, create_graph=True)[0]
        c_b_x = torch.autograd.grad(c_b.sum(), x, create_graph=True)[0]
        q_b_x = torch.autograd.grad(q_b.sum(), x, create_graph=True)[0]
        mixed_loss = torch.mean((c_b_t + q_b_x) ** 2) + 2.0 * torch.mean(
            (q_b + pinn.D_rel_B * c_b_x) ** 2
        )
        self.assertTrue(torch.isfinite(mixed_loss))
        mixed_loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in mixed_thin.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
