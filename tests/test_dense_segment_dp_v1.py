import unittest

import numpy as np

import run_dense_segment_dp_v1 as decoder


class DenseSegmentDPV1Tests(unittest.TestCase):
    def test_dp_selects_best_nonoverlapping_intervals(self):
        scores = {
            2: np.asarray([4.0, 1.0, 4.0]),
            3: np.asarray([6.9, 0.0]),
        }
        predictions = decoder.decode_dp(scores, length=4, base_duration=2,
                                        lam=1.0, eta=0.0)
        self.assertEqual([(row["onset"], row["offset"]) for row in predictions],
                         [(0, 1), (2, 3)])

    def test_large_event_cost_allows_empty_output(self):
        scores = {2: np.asarray([4.0, 1.0, 4.0])}
        self.assertEqual(decoder.decode_dp(scores, 4, 2, lam=10.0, eta=0.0), [])

    def test_duration_penalty_can_change_selected_interval(self):
        scores = {2: np.asarray([2.0, 0.0, 0.0]), 4: np.asarray([3.0])}
        without_penalty = decoder.decode_dp(scores, 4, 2, lam=0.5, eta=0.0)
        with_penalty = decoder.decode_dp(scores, 4, 2, lam=0.5, eta=2.0)
        self.assertEqual(without_penalty[0]["duration"], 4)
        self.assertEqual(with_penalty, [])

    def test_configuration_selection_uses_pooled_counts(self):
        candidates = (("fixed", 1.0, 0.0), ("fixed", 2.0, 0.0))
        counts = {candidates[0]: (1, 0, 9), candidates[1]: (2, 2, 8)}
        self.assertEqual(decoder.choose_config(candidates, counts), candidates[1])


if __name__ == "__main__":
    unittest.main()
