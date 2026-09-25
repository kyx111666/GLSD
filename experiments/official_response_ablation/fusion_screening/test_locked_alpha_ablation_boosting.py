import unittest

import numpy as np

import run_locked_alpha_ablation_boosting as runner


class LockedBoostingAlphaTests(unittest.TestCase):
    def test_alpha_half_matches_equal_mean_and_endpoints_are_components(self):
        class BaseFeatures:
            def __init__(self, response, k):
                pass

            def evidence(self, reference, radius):
                return np.array([4, 8, 12]), np.array([
                    [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]
                ])

        features = runner.make_weighted_features(BaseFeatures)(np.zeros(5), 1)

        def selected(alpha):
            config = runner.WeightedConfig("WeightedMean", 0, 1.0, 1.0, 0.5, alpha)
            return features.selected_peaks(config)

        np.testing.assert_array_equal(selected(0.5), np.array([4, 8, 12]))
        np.testing.assert_array_equal(selected(1.0), np.array([8, 12]))
        np.testing.assert_array_equal(selected(0.0), np.array([4]))

    def test_alpha_grid_requires_unique_values_in_unit_interval(self):
        self.assertEqual(runner.parse_alphas("0,.5,1"), (0.0, 0.5, 1.0))
        with self.assertRaises(Exception):
            runner.parse_alphas("0,.5,.5")
        with self.assertRaises(Exception):
            runner.parse_alphas("-0.1,0.5")


if __name__ == "__main__":
    unittest.main()
