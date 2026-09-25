"""Unit tests for the phase-1 generalized-mean operator and selector."""

import unittest

import numpy as np

import run_generalized_mean_screening as screen


class GeneralizedMeanTests(unittest.TestCase):
    def test_p1_is_exact_arithmetic_mean(self):
        g = np.array([0.0, 0.2, 0.9, 1.0])
        l = np.array([1.0, 0.4, 0.1, 0.0])
        np.testing.assert_array_equal(screen.generalized_mean(g, l, 1), (g + l) / 2.0)

    def test_monotone_in_p_and_bounded_by_mean_and_max(self):
        g = np.array([0.1, 0.4, 0.9, 1.0])
        l = np.array([0.7, 0.2, 0.8, 0.0])
        values = [screen.generalized_mean(g, l, p) for p in (1, 2, 4, 8, np.inf)]
        for left, right in zip(values, values[1:]):
            self.assertTrue(np.all(right >= left - 1e-12))
        self.assertTrue(np.all(values[0] <= values[-1] + 1e-12))
        np.testing.assert_array_equal(values[-1], np.maximum(g, l))

    def test_infinity_equals_max(self):
        rng = np.random.default_rng(4)
        g, l = rng.uniform(size=(2, 100))
        np.testing.assert_array_equal(screen.generalized_mean(g, l, np.inf), np.maximum(g, l))

    def test_invalid_p_and_invalid_evidence_are_rejected(self):
        with self.assertRaises(ValueError):
            screen.generalized_mean([0.2], [0.3], 0.5)
        with self.assertRaises(ValueError):
            screen.generalized_mean([np.nan], [0.3], 2)
        with self.assertRaises(ValueError):
            screen.generalized_mean([1.2], [0.3], 2)

    def test_held_out_subject_cannot_change_selection(self):
        table = np.array([
            [[1, 0, 0], [1, 9, 2], [1, 8, 3]],
            [[0, 8, 4], [4, 0, 0], [3, 0, 0]],
        ])
        original, pooled = screen.choose(table, 0)
        poisoned = table.copy()
        poisoned[:, 0] = [[100000, 0, 0], [0, 100000, 100000]]
        changed, poisoned_pool = screen.choose(poisoned, 0)
        self.assertEqual(original, changed)
        np.testing.assert_array_equal(pooled, poisoned_pool)

    def test_tie_break_is_deterministic(self):
        table = np.array([
            [[0, 0, 0], [2, 2, 0]],
            [[0, 0, 0], [2, 0, 2]],
            [[0, 0, 0], [2, 0, 2]],
        ])
        winner, _ = screen.choose(table, 0)
        self.assertEqual(winner, 1)

    def test_grid_has_finite_selection_and_infinity_diagnostic(self):
        grid = screen.grids()["GM"]
        self.assertEqual(len(grid), 45)
        self.assertEqual([c.config_id for c in grid], list(range(45)))
        self.assertEqual({c.p for c in grid}, set(screen.P_ALL))
        self.assertEqual(len(screen.finite_gm_configs(grid)), 36)

    def test_mean_hook_replay_gate_uses_same_candidate_mask(self):
        class FakeFeatures:
            def __init__(self):
                self.peaks = np.arange(4)
                self._evidence = np.array([[.2, .4], [.8, .4], [.9, .1], [.1, .9]])

            def evidence(self, _a0, _rho):
                return self.peaks, self._evidence

            def selected_peaks(self, config):
                values = self._evidence.mean(axis=1)
                return self.peaks[values >= config.threshold]

        class FakeBase:
            def GLSDFeatures(self, _response, _k):
                return FakeFeatures()

        core = screen.FusionCore(FakeBase())
        view = core.GLSDFeatures(np.ones(8), 3)
        peaks, evidence = view.peaks, view.values
        g, l = evidence.T
        for tau in screen.THRESHOLDS:
            expected = view.base.selected_peaks(type("Config", (), {
                "reference_scale": screen.A0, "local_radius": screen.RHO, "threshold": tau,
            })())
            got = np.asarray(peaks)[screen.accept(g, l, screen.Config("GM", 0, tau, p=1.0))]
            np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main()
