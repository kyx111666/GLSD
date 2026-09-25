"""Unit tests for the fixed-alpha sensitivity protocol."""
import unittest

import numpy as np

import run_alpha_hparam_ablation as alpha
import run_full_fusion_tuning as full


class AlphaHparamTests(unittest.TestCase):
    def test_default_grid_is_interior_and_main_alpha_is_present(self):
        self.assertEqual(alpha.DEFAULT_ALPHAS, (0.1, 0.3, 0.5, 0.7, 0.9))
        self.assertIn(0.5, alpha.DEFAULT_ALPHAS)
        self.assertEqual(len(alpha.alpha_grid(0.5)), 90)

    def test_each_alpha_uses_same_structure_grid(self):
        left = alpha.alpha_grid(0.1)
        right = alpha.alpha_grid(0.7)
        self.assertEqual(
            [(c.reference_scale, c.local_radius, c.threshold) for c in left],
            [(c.reference_scale, c.local_radius, c.threshold) for c in right],
        )
        self.assertTrue(all(c.alpha == 0.1 for c in left))
        self.assertTrue(all(c.alpha == 0.7 for c in right))

    def test_alpha_score_matches_formula(self):
        g = np.array([0.2, 0.8])
        l = np.array([0.9, 0.1])
        # The production runner replaces the scorer's declared alpha grid
        # before decoding; mirror that setup here.
        full.ALPHAS = (0.3,)
        config = full.Config("WeightedMean", 0, 1.0, 2.0, 0.5, alpha=0.3)
        np.testing.assert_allclose(full.scores(g, l, config), [0.69, 0.31])

    def test_alpha_parser_rejects_invalid_values(self):
        self.assertEqual(alpha.parse_alphas("0.1, 0.5"), (0.1, 0.5))
        with self.assertRaises(Exception):
            alpha.parse_alphas("-0.1,0.5")
        with self.assertRaises(Exception):
            alpha.parse_alphas("0.5,0.5")

    def test_bootstrap_comparison_is_zero_for_identical_counts(self):
        counts = np.array([[2, 1, 3], [1, 2, 4], [0, 1, 2]], dtype=np.int64)
        rows = alpha.paired_alpha_comparisons({0.3: counts, 0.5: counts}, ["a", "b", "c"], 0.5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta_F1"], 0.0)
        self.assertEqual(rows[0]["CI_low"], 0.0)
        self.assertEqual(rows[0]["CI_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
