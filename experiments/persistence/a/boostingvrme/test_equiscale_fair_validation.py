"""Focused tests for the experiment's controls, matching and fold isolation."""

import copy
import unittest

import numpy as np

import equiscale_fair_validation as fair
from tune_equiscale import native_predictions, NativeConfig, count_video


class FairValidationTests(unittest.TestCase):
    def test_duration_uses_training_subjects_only(self):
        records = [{"subject": str(i), "gt": [[0, 1, length]]}
                   for i, length in enumerate((6, 8, 12, 100))]
        for backbone in ("metst", "boostingvrme"):
            expected = fair.duration_k(records, {"2", "3"}, backbone)
            changed = copy.deepcopy(records)
            changed[2]["gt"] = [[0, 10000, 100000]]
            changed[3]["gt"] = [[0, 10000, 200000]]
            self.assertEqual(expected, fair.duration_k(changed, {"2", "3"}, backbone))

    def test_grid_components_are_available_to_single_scale(self):
        configs, size, batches = fair.configuration_grid()
        single = {(round(c.smooth, 8), c.p, round(c.distance, 8)) for c in configs[:size]}
        for indexes in batches:
            c = configs[indexes[0]]
            for scale in fair.SCALE_SETS[c.scales]:
                self.assertIn((round(c.smooth*scale, 8), c.p, round(c.distance*scale, 8)), single)
        self.assertIn(fair.Config("single", 2.0, 0.55, 1.0), configs)

    def test_same_scale_nearby_peak_is_not_deleted(self):
        clusters = fair.associate([[(10, 1.0), (11, 0.8)], [(10, 0.9)]], 3)
        self.assertEqual([c[0] for c in clusters], [10, 11])
        self.assertEqual([c[1] for c in clusters], [2, 1])

    def test_association_does_not_chain(self):
        clusters = fair.associate([[(10, 1.0)], [(12, 0.9)], [(14, 0.8)]], 2)
        self.assertEqual([c[1] for c in clusters], [2, 1])

    def test_rounded_scales_are_not_multiple_votes(self):
        curve = np.array([0., 0., 0.1, 0.4, 0.8, 0.4, 0.1, 0., 0.])
        v = fair.VideoFeatures({"curve": curve, "gt": []}, 1, "metst")
        c = fair.Config("union", 1.0, 0.4, 1.0, "centred_3", 0.5)
        self.assertTrue(v.clusters(c))
        self.assertTrue(all(cluster[1] == 1 for cluster in v.clusters(c)))

    def test_batched_source_decoder_equals_reference(self):
        rng = np.random.default_rng(34)
        c = fair.Config("single", 2.0, 0.55, 1.0)
        for _ in range(30):
            curve = rng.uniform(0, 1, size=150)
            record = {"curve": curve, "gt": [[5, 10, 16], [62, 70, 78], [122, 128, 136]]}
            v = fair.VideoFeatures(record, 5, "boostingvrme")
            counts, output = v.prepare(v.clusters(c)).evaluate([c], details=True)
            expected = native_predictions(curve, 5, NativeConfig(2., 0.55, 1.))
            actual = [(p["onset"], p["offset"], p["peak"]) for p in output[0]]
            self.assertEqual(expected, actual)
            reference = count_video(expected, record)
            np.testing.assert_array_equal(counts[0], (reference.tp, reference.fp, reference.fn))

    def test_no_second_best_gt_rematching(self):
        clusters = [(10, 1, 1, 1, 1), (11, 1, 1, 1, 1)]
        v = fair.VideoFeatures({"curve": np.zeros(30), "gt": [[5, 10, 15], [5, 11, 15]]}, 5, "metst")
        result = v.prepare(clusters).evaluate([fair.Config("single", 2., 0.55, 1.)])
        np.testing.assert_array_equal(result[0], (1, 1, 1))

    def test_height_weight_one_has_no_support_effect(self):
        first = fair.Prepared([(10, 1, 1., 1., 1.), (20, 3, 0.5, 0.5, 0.5)],
                              np.array([[9, 11], [19, 21]]), np.zeros((2, 2), bool), np.array([-1, -1]), 0)
        second = fair.Prepared([(10, 3, 1., 1., 1.), (20, 1, 0.5, 0.5, 0.5)],
                               first.intervals, first.conflicts, first.best, 0)
        c = fair.Config("score", 2., 0.5, 1., "centred_3", 0.5, "max", 0., 1.)
        np.testing.assert_array_equal(first.masks([c]), second.masks([c]))

    def test_identical_paired_bootstrap_is_zero(self):
        values = np.array([[1, 2, 3], [0, 4, 5], [8, 2, 1]])
        result = fair.paired_bootstrap(values, values, repeats=200)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertEqual(result["delta_F1"], 0.0)

    def test_empty_candidates(self):
        v = fair.VideoFeatures({"curve": np.zeros(30), "gt": [[1, 3, 5]]}, 3, "metst")
        c = fair.Config("score", 2., 0.5, 1., "centred_3", 0.5, "max", 0., 0.5)
        counts, predictions = v.prepare([]).evaluate([c], details=True)
        np.testing.assert_array_equal(counts[0], (0, 0, 1))
        self.assertEqual(predictions, [[]])


if __name__ == "__main__":
    unittest.main()
