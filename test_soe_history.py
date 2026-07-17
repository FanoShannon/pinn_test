import unittest

import numpy as np

import productintegral_dtn as coupled
import soe_abel_history


class FastAbelHistoryTests(unittest.TestCase):
    def test_soe_completed_history_matches_direct_product_integral(self):
        n_steps = 1025
        dt = 1.0 / (n_steps - 1)
        time = np.linspace(0.0, 1.0, n_steps)
        current = (
            np.sin(7.0 * time)
            +0.2 * np.sin(31.0 * time)
            +0.1 * time ** 2
        )
        current[0] = 0.0
        plan = soe_abel_history.build_soe_history_plan(
            n_steps,
            dt,
            diffusion=1.0,
            near_cells=16,
            n_terms=128,
            tolerance=1e-10,
        )
        history = soe_abel_history.NumpySoeHistory(plan)
        errors = []
        references = []
        for step in range(1, n_steps):
            fast_value = history.completed(current, step)
            direct_value = coupled.completed_product_integral(
                current,
                step,
                dt,
                diffusion=1.0,
            )
            errors.append(fast_value - direct_value)
            references.append(direct_value)
            history.advance(current, step)
        errors = np.asarray(errors)
        references = np.asarray(references)
        self.assertLess(float(np.max(np.abs(errors))), 1e-10)
        self.assertLess(
            float(np.sqrt(np.mean(errors ** 2)))
            /max(float(np.ptp(references)), 1e-15),
            1e-11,
        )

    def test_soe_plan_is_positive_and_cached(self):
        first = soe_abel_history.build_soe_history_plan(
            257,
            1.0 / 256.0,
            diffusion=1.0,
        )
        second = soe_abel_history.build_soe_history_plan(
            257,
            1.0 / 256.0,
            diffusion=1.0,
        )
        self.assertIs(first, second)
        self.assertTrue(np.all(first.near_weights > 0.0))
        self.assertTrue(np.all(first.decay > 0.0))
        self.assertTrue(np.all(first.decay <= 1.0))
        self.assertTrue(np.any(first.decay < 1.0))
        self.assertTrue(np.all(first.tail_amplitudes > 0.0))

    def test_invalid_soe_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "near_cells"):
            soe_abel_history.build_soe_history_plan(
                64,
                1.0 / 63.0,
                diffusion=1.0,
                near_cells=0,
            )
        with self.assertRaisesRegex(ValueError, "n_terms"):
            soe_abel_history.build_soe_history_plan(
                64,
                1.0 / 63.0,
                diffusion=1.0,
                n_terms=4,
            )


if __name__ == "__main__":
    unittest.main()
