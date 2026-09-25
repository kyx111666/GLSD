"""Selection constraints and diagnostics for matched persistence validation."""

import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np

import pure_persistence_validation as pure
import unified_persistence as engine


class PurePersistenceTests(unittest.TestCase):
    def test_matched_selection_keeps_anchor_and_exact_controls(self):
        configs = engine.configuration_grid()
        lookup = {c: i for i, c in enumerate(configs)}
        train = np.tile([0, 100, 100], (len(configs), 1))
        desired = engine.Config("unified", 2, 0, 0.6, 3)
        anchor = engine.Config("global", 1.5, 0, 0.4)
        calibrated = engine.Config("unified", 1.5, 0, 0.3, 2)
        for config, counts in ((desired, [100, 2, 0]), (anchor, [90, 10, 10]),
                               (calibrated, [90, 12, 10])):
            train[lookup[config]] = counts
        # Height-using candidates must not re-enter the pure selection pool.
        train[lookup[engine.Config("unified", 1, 0.75, 0.05)]] = [1000, 0, 0]
        before = train.copy()
        selected = pure.selected_configs(train, configs)
        self.assertEqual(configs[selected["pure"]], desired)
        self.assertEqual(configs[selected["global_tuned"]], anchor)
        self.assertEqual(configs[selected["fusion_at_global_candidates"]], calibrated)
        for name in ("global_at_pure_settings", "local_at_pure_settings"):
            config = pure.resolved_config(name, selected, configs)
            self.assertEqual((config.reference, config.radius, config.threshold),
                             (desired.reference, desired.radius, desired.threshold))
        self.assertEqual(configs[selected["local_at_global_candidates"]].reference, anchor.reference)
        np.testing.assert_array_equal(train, before)

    def test_pure_grid_has_only_ninety_zero_height_configs(self):
        configs = engine.configuration_grid()
        indexes = pure.indexed_pools(configs)["pure"]
        self.assertEqual(len(indexes), 90)
        self.assertTrue(all(configs[i].family == "unified" and configs[i].height_weight == 0 for i in indexes))

    def test_threshold_perturbation_rounds_and_freezes_generation(self):
        original = engine.Config("unified", 1.5, 0, 0.6, 3)
        changed = pure.perturb(original, -0.05)
        self.assertEqual(changed.threshold, 0.55)
        self.assertEqual(original.threshold, 0.6)
        self.assertEqual((changed.family, changed.reference, changed.radius, changed.height_weight),
                         (original.family, original.reference, original.radius, 0))
        self.assertEqual(pure.perturb(original, 0), original)
        self.assertEqual(pure.perturb(original, -1).threshold, 0)

    def test_public_config_has_no_height_weight_and_is_immutable(self):
        config = pure.PersistenceConfig(1.5, 2, 0.4)
        self.assertEqual(config.as_engine_config().height_weight, 0)
        with self.assertRaises(FrozenInstanceError):
            config.radius = 1
        with self.assertRaises(TypeError):
            pure.PersistenceConfig(1.5, 2, 0.4, height_weight=0.5)
        self.assertEqual(pure.detect_curve(np.ones(30), 3, config), [])

    def test_overlap_handles_empty_candidates_and_gt(self):
        self.assertEqual(pure.overlap_matrix([], [[10, 15, 20]]).shape, (0, 1))
        self.assertEqual(pure.overlap_matrix([[10, 20]], []).shape, (1, 0))
        np.testing.assert_array_equal(pure.overlap_matrix([[10, 20]], [[10, 15, 20]]), [[1]])

    def test_error_audit_distinguishes_missing_candidates_from_rejected_scores(self):
        record = {"subject": "1", "video": "a", "gt": [[10, 15, 20], [50, 55, 60]]}
        prepared = SimpleNamespace(intervals=np.array([[10, 20], [30, 40]]),
                                   clusters=[(15,), (35,)], evidence=np.array([[0, 0.2, 0.2], [0, 0.8, 0.8]]))
        baseline = [{"onset": 10, "peak": 15, "offset": 20, "matched_gt": 0},
                    {"onset": 50, "peak": 55, "offset": 60, "matched_gt": 1}]
        predictions = [{"onset": 30, "peak": 35, "offset": 40, "matched_gt": -1}]
        rows = pure.error_audit(record, predictions, baseline, prepared, engine.Config("unified", 2, 0, 0.5))
        self.assertEqual([row["reason"] for row in rows],
                         ["score_rejected", "candidate_or_interval_miss", "low_overlap"])
        self.assertEqual(rows[0]["score"], 0.2)
        self.assertIsNone(rows[1]["score"])
        self.assertEqual(rows[2]["best_iou"], 0)

    def test_scalar_verification_catches_wrong_match_assignment(self):
        record = {"gt": [[10, 15, 20], [10, 15, 20]]}
        predictions = [{"onset": 10, "peak": 15, "offset": 20, "matched_gt": 0},
                       {"onset": 10, "peak": 16, "offset": 20, "matched_gt": -1}]
        pure.verify_predictions(predictions, record, [1, 1, 1])
        predictions[1]["matched_gt"] = 1
        with self.assertRaises(AssertionError):
            pure.verify_predictions(predictions, record, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
