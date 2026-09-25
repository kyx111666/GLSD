import unittest

import numpy as np

import run_dense_segment_phase0_v2 as phase0


class DenseSegmentPhase0Tests(unittest.TestCase):
    def test_dense_candidates_do_not_need_gt_and_cover_all_starts(self):
        candidates, fallback = phase0.dense_candidates(np.arange(9.0), k=1)
        durations = phase0.duration_set(1, 9)
        self.assertEqual(len(candidates), sum(9 - duration + 1 for duration in durations))
        self.assertEqual({row["duration"] for row in candidates}, set(durations))
        self.assertIn(fallback, {"mad", "std"})

    def test_formal_iou_oracles_respect_nonoverlap(self):
        candidates = [
            {"onset": 0, "offset": 4},
            {"onset": 3, "offset": 7},
            {"onset": 8, "offset": 12},
        ]
        gt = [[0, 2, 4], [3, 5, 7], [8, 10, 12]]
        self.assertEqual(phase0.maximum_iou_matching(candidates, gt), 3)
        self.assertEqual(phase0.nonoverlap_oracle(candidates, gt), 2)

    def test_greedy_budget_is_deterministic_and_nonoverlapping(self):
        candidates = [
            {"onset": 0, "offset": 4, "duration": 5, "score": 3.0},
            {"onset": 3, "offset": 7, "duration": 5, "score": 2.0},
            {"onset": 8, "offset": 12, "duration": 5, "score": 1.0},
        ]
        selected = phase0.greedy_budget(candidates, "score", 2)
        self.assertEqual([(row["onset"], row["offset"]) for row in selected],
                         [(0, 4), (8, 12)])

    def test_boosting_response_uses_native_smoothing(self):
        curve = np.arange(8.0)
        expected = np.convolve(curve, np.ones(4) / 4, mode="same")
        np.testing.assert_allclose(
            phase0.native_smoothed_response("BoostingVRME", curve, k=2), expected)
        np.testing.assert_array_equal(
            phase0.native_smoothed_response("ME-TST", curve, k=2), curve)


if __name__ == "__main__":
    unittest.main()
