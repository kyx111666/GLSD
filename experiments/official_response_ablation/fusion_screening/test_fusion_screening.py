"""Protocol tests only. Synthetic examples are never experiment results."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

import run_fusion_screening as screen


class FusionTests(unittest.TestCase):
    def test_grid_is_exactly_the_six_screening_methods(self):
        grids = screen.grids()
        self.assertEqual({m: len(g) for m, g in grids.items()},
                         dict(G=9, L=9, EqualMean=9, WeightedMean=45, DualOR=81, CrossSupported=243))
        for method, grid in grids.items():
            self.assertEqual([c.config_id for c in grid], list(range(len(grid))))
            self.assertTrue(all(c.reference_scale == 1.5 and c.local_radius == 2 for c in grid))
        self.assertEqual(sum(map(len, grids.values())), 396)

    def test_cross_support_boundaries_and_endpoints(self):
        g = np.array([.8, .4, .98, .7, .35, 0.])
        l = np.array([.25, .65, .18, .2, .4, 1.])
        c = screen.Config("CrossSupported", 0, tau_g=.7, tau_l=.4, eta=.5)
        np.testing.assert_array_equal(screen.accept(g, l, c), [True, True, False, True, True, False])
        for eta, expected in ((0, (g >= .7) | (l >= .4)), (1, (g >= .7) & (l >= .4))):
            c = screen.Config("CrossSupported", 0, tau_g=.7, tau_l=.4, eta=eta)
            np.testing.assert_array_equal(screen.accept(g, l, c), expected)

    def test_weight_endpoints_recover_single_modules(self):
        g, l = np.random.default_rng(1).uniform(size=(2, 60))
        for threshold in screen.THRESHOLDS:
            for alpha, method in ((0., "L"), (1., "G")):
                weighted = screen.Config("WeightedMean", 0, threshold=threshold, alpha=alpha)
                single = screen.Config(method, 0, threshold=threshold)
                np.testing.assert_array_equal(screen.accept(g, l, weighted), screen.accept(g, l, single))

    def test_held_out_counts_cannot_change_selection(self):
        table = np.array([[[1, 0, 0], [1, 9, 2], [1, 8, 3]],
                          [[0, 8, 4], [4, 0, 0], [3, 0, 0]]])
        original, pooled = screen.choose(table, 0)
        self.assertEqual(original, 1)
        poisoned = table.copy()
        poisoned[:, 0] = [[100000, 0, 0], [0, 100000, 100000]]
        got, got_pool = screen.choose(poisoned, 0)
        self.assertEqual(got, original)
        np.testing.assert_array_equal(pooled, got_pool)

    def test_selection_tie_break_is_deterministic(self):
        table = np.array([[[0, 0, 0], [2, 2, 0]],
                          [[0, 0, 0], [2, 0, 2]],
                          [[0, 0, 0], [2, 0, 2]]])
        winner, _ = screen.choose(table, 0)
        self.assertEqual(winner, 1)  # equal F1, higher precision, then original order

    def test_paired_bootstrap_identical_predictions_have_zero_delta(self):
        counts = np.array([[3, 2, 5], [2, 4, 3], [0, 2, 5]])
        comparisons = screen.paired_comparisons("synthetic", {m: counts for m in screen.METHODS}, ["a", "b", "c"])
        self.assertEqual(len(comparisons), 5)
        for row in comparisons:
            self.assertEqual((row["delta_F1"], row["CI_low"], row["CI_high"]), (0., 0., 0.))

    def test_decision_does_not_launch_or_claim_significance(self):
        rows = []
        for setting in ("sammlv", "casme3"):
            rows += [dict(setting=setting, method=m, F1=.3 if m == "CrossSupported" else .2) for m in screen.METHODS]
        report = screen.decision(rows, [])
        self.assertEqual(report["recommendation"], "PROMISING_REVIEW_BEFORE_FORMAL")
        self.assertIn("not significance", report["interpretation"])
        self.assertEqual(screen.decision(rows[:6], [])["recommendation"], "WAIT_FOR_OTHER_DATASET")

    def test_cached_features_preserve_real_locked_mean_and_invalidate_on_change(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "official_BoostingVRME/boosting_official_glds_full.py"
        spec = importlib.util.spec_from_file_location("screen_test_locked_core", path)
        base = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = base
        spec.loader.exec_module(base)
        core = screen.FusionCore(base)
        response = np.random.default_rng(3).uniform(size=200)
        original = base.GLSDFeatures(response, 5)
        view = core.GLSDFeatures(response, 5)
        self.assertIs(core.GLSDFeatures(response.copy(), 5), view)
        for config in screen.grids()["EqualMean"]:
            np.testing.assert_array_equal(view.selected_peaks(config), original.selected_peaks(config))
        self.assertIsNot(core.GLSDFeatures(response, 6), view)
        response[100] += .2
        self.assertIsNot(core.GLSDFeatures(response, 5), view)
        constant = core.GLSDFeatures(np.ones(10), 2)
        self.assertEqual(len(constant.selected_peaks(screen.grids()["G"][0])), 0)


if __name__ == "__main__":
    unittest.main()
