import unittest

import numpy as np
import torch

import differentiable_productintegral_dtn as differentiable
from design_multiscan_rates import matrix_metrics, rank_combinations
from invert_multiscan_parameters import (
    PARAMETER_NAMES,
    ScanCurve,
    add_noise,
    decode_parameters,
    encode_start,
    multiscan_loss,
    scan_duration,
)


class MultiscanInversionTests(unittest.TestCase):
    def setUp(self):
        self.bounds = {
            "k_cat": (0.01, 100.0),
            "gamma": (0.1, 100.0),
            "delta": (0.005, 0.2),
        }
        self.parameters = {
            "k_cat": 0.7,
            "gamma": 8.0,
            "delta": 0.06,
        }

    def test_bounded_log_parameter_round_trip(self):
        raw = encode_start(
            self.parameters,
            PARAMETER_NAMES,
            self.bounds,
        )
        decoded = decode_parameters(
            raw,
            PARAMETER_NAMES,
            self.parameters,
            self.bounds,
        )
        for name in PARAMETER_NAMES:
            self.assertAlmostEqual(
                float(decoded[name].detach()),
                self.parameters[name],
                places=12,
            )

    def test_noise_is_scaled_per_curve_and_preserves_initial_point(self):
        clean = np.linspace(-2.0, 3.0, 33)
        curve = ScanCurve(
            sigma=40.0,
            time=np.linspace(0.0, 1.0, len(clean)),
            clean_current=clean,
            observed_current=clean.copy(),
            current_span=5.0,
        )
        first = add_noise([curve], 0.01, np.random.default_rng(7))[0]
        second = add_noise([curve], 0.01, np.random.default_rng(7))[0]
        np.testing.assert_array_equal(
            first.observed_current,
            second.observed_current,
        )
        self.assertEqual(first.observed_current[0], clean[0])
        self.assertGreater(
            np.linalg.norm(first.observed_current - clean),
            0.0,
        )

    def test_joint_multiscan_objective_has_finite_gradient(self):
        observations = []
        for sigma in (10.0, 160.0):
            time = torch.linspace(
                0.0,
                scan_duration(sigma),
                33,
                dtype=torch.float64,
            )
            target_state = differentiable.solve_coupled_operator(
                time,
                gamma=10.0,
                k_cat=1.0,
                delta=0.035,
                n_modes=16,
                history_backend="direct",
            )
            target = target_state["J_surface"].detach()
            observations.append({
                "sigma": sigma,
                "time": time,
                "observed": target,
                "span": max(float(torch.max(target) - torch.min(target)), 1e-12),
            })
        start = {"k_cat": 0.5, "gamma": 5.0, "delta": 0.07}
        raw = encode_start(start, PARAMETER_NAMES, self.bounds)
        loss, _ = multiscan_loss(
            raw,
            PARAMETER_NAMES,
            start,
            self.bounds,
            observations,
            16,
            10,
            "soe",
            8,
            64,
            1e-9,
        )
        gradient = torch.autograd.grad(loss, raw)[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertTrue(torch.all(torch.abs(gradient) > 1e-8))

    def test_rate_design_metrics_detect_collinearity(self):
        orthogonal = np.eye(3)
        metrics = matrix_metrics(orthogonal, (0, 1, 2))
        self.assertAlmostEqual(
            metrics["column_normalized_condition_number"],
            1.0,
            places=12,
        )
        nearly_collinear = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1e-6, 1.0],
            [0.0, 0.0, 1e-6],
        ])
        metrics = matrix_metrics(nearly_collinear, (0, 1, 2))
        self.assertGreater(
            metrics["column_normalized_condition_number"],
            1e5,
        )

    def test_single_parameter_design_maximizes_sensitivity(self):
        weak = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        strong = 5.0 * weak
        ranking = rank_combinations(
            {"case": {1.0: weak, 2.0: strong}},
            [1.0, 2.0],
            1,
            "delta-only",
            2,
        )
        self.assertEqual(ranking[0]["scan_rates"], [2.0])
        self.assertEqual(
            ranking[0]["ranking_metric"],
            "worst_inverse_relative_sensitivity",
        )


if __name__ == "__main__":
    unittest.main()
