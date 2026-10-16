"""Near-optimal selection and held-out subject isolation."""
import unittest
import numpy as np
import run_full_fusion_tuning as full


class SelectionTests(unittest.TestCase):
    def tables(self):
        mean = np.tile([10, 10, 10], (90, 3, 1))
        g = np.tile([9, 11, 11], (30, 3, 1))
        local = np.tile([9, 11, 11], (90, 3, 1))
        return mean, g, local

    def test_held_out_counts_do_not_affect_selection(self):
        mean, g, local = self.tables()
        before = full.choose_mean_synergy(mean, g, local, 0)
        mean[1, 0] = [100000, 0, 0]
        g[:, 0] = [0, 100000, 100000]
        local[:, 0] = [0, 100000, 100000]
        after = full.choose_mean_synergy(mean, g, local, 0)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])

    def test_synergy_selects_near_optimal_not_only_best_f1(self):
        mean, g, local = self.tables()
        mean[:] = [1000, 1000, 1000]
        mean[1] = [999, 1001, 1001]
        g[:] = [998, 1002, 1002]
        local[:] = [1000, 1000, 1000]
        local[1] = [990, 1010, 1010]
        winner, _, _ = full.choose_mean_synergy(mean, g, local, 0)
        self.assertEqual(winner, 1)
        winner, _, _ = full.choose_mean_synergy(mean, g, local, 0, epsilon=0)
        self.assertEqual(winner, 0)


if __name__ == '__main__':
    unittest.main()
