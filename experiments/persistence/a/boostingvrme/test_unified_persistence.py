"""Behavioral checks for label isolation, evidence, and independent decoding."""

import unittest

import numpy as np

import unified_persistence as unified
from tune_equiscale import (
    Cluster,
    NativeConfig,
    apply_source_nms,
    count_video,
    native_predictions,
    source_interval,
)


def synthetic_curve(seed=7):
    rng = np.random.default_rng(seed)
    frames = np.arange(140)
    curve = rng.uniform(0, 0.04, len(frames))
    for peak, width, height in ((5, 2, 0.6), (38, 5, 0.95), (47, 3, 0.75),
                                (82, 2, 0.5), (103, 8, 0.8), (133, 3, 0.65)):
        curve += height * np.exp(-0.5 * ((frames - peak) / width) ** 2)
    return curve


def intervals_only(predictions):
    return [(row["onset"], row["offset"], row["peak"]) for row in predictions]


def reference_decode(curve, k, peaks, keep, backbone, mode):
    if backbone == "boostingvrme" and mode == "native":
        clusters = [Cluster(int(peak), source_interval(curve, int(peak), k), 1, 0, 0, 0)
                    for peak in peaks]
        return apply_source_nms(clusters, np.flatnonzero(keep), k)
    return [(max(0, int(peak) - k), min(len(curve) - 1, int(peak) + k), int(peak))
            for peak in peaks[keep]]


class UnifiedPersistenceTests(unittest.TestCase):
    def test_inner_counts_excludes_held_subject_and_uses_inner_k(self):
        stats = {k: np.arange(36).reshape(3, 4, 3) + 1000 * k for k in (2, 3)}
        inner_k = np.array([[0, 2, 3, 2], [2, 0, 2, 3],
                            [3, 2, 0, 3], [2, 3, 3, 0]])
        held = 2
        expected = stats[3][:, 0] + stats[2][:, 1] + stats[3][:, 3]
        np.testing.assert_array_equal(unified.inner_counts(stats, held, inner_k), expected)
        contaminated = {k: values.copy() for k, values in stats.items()}
        for values in contaminated.values():
            values[:, held] = 10 ** 8
        inner_k[held, held] = 999
        np.testing.assert_array_equal(unified.inner_counts(contaminated, held, inner_k), expected)
        contaminated[2][:, 1] += 7
        np.testing.assert_array_equal(unified.inner_counts(contaminated, held, inner_k), expected + 7)

    def test_ground_truth_cannot_change_evidence_or_detections(self):
        curve = synthetic_curve()
        features = unified.CurveFeatures(curve, 5)
        original = features.evidence(1.5, 2)[1].copy()
        records = [{"curve": curve, "gt": gt} for gt in (
            [], [[31, 38, 44], [77, 82, 87]],
            [[0, 50, 139], [1000, 1001, 1002], [40, 41, 42]],
        )]
        configs = [unified.Config("unified", 1.5, weight, threshold, 2)
                   for weight in (0, 0.5) for threshold in (0.05, 0.5)]
        for backbone, mode in (("metst", "fixed"), ("boostingvrme", "fixed"),
                               ("boostingvrme", "native")):
            expected = None
            for record in records:
                prepared = unified.prepare(record, features, 1.5, backbone, mode, 2)
                _, predictions = prepared.evaluate(configs, details=True)
                actual = [intervals_only(rows) for rows in predictions]
                if expected is None:
                    expected = actual
                    self.assertTrue(any(expected))
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(features.evidence(1.5, 2)[1], original)

    def test_unified_formula_keeps_low_height_candidates_eligible(self):
        evidence = np.array([[0, 0.8, 0.6], [1, 0.1, 0.1], [0.4, 0.2, 0.8]])
        config = unified.Config("unified", 1.5, 0.25, 0.5)
        scores = unified.evidence_scores(evidence, config)
        np.testing.assert_allclose(scores, (0.525, 0.325, 0.475))
        np.testing.assert_array_equal(scores >= config.threshold, (True, False, False))
        zero_weight = unified.Config("unified", 1.5, 0, 0.5)
        changed = evidence.copy()
        changed[:, 0] = (100, -100, 50)
        np.testing.assert_array_equal(unified.evidence_scores(evidence, zero_weight),
                                      unified.evidence_scores(changed, zero_weight))

    def test_evidence_and_detections_are_invariant_to_positive_rescaling(self):
        curve = synthetic_curve()
        first, second = unified.CurveFeatures(curve, 5), unified.CurveFeatures(curve * 7, 5)
        for reference in unified.REFERENCE_SCALES:
            peaks, evidence = first.evidence(reference, 2)
            scaled_peaks, scaled = second.evidence(reference, 2)
            np.testing.assert_array_equal(peaks, scaled_peaks)
            np.testing.assert_allclose(evidence, scaled, rtol=1e-10, atol=1e-12)
            config = unified.Config("unified", reference, 0.5, 0.4, 2)
            for backbone, mode in (("metst", "fixed"), ("boostingvrme", "native")):
                self.assertEqual(unified.detect_curve(curve, 5, config, backbone, mode),
                                 unified.detect_curve(curve * 7, 5, config, backbone, mode))

    def test_wider_local_radius_never_reduces_aligned_evidence(self):
        for seed, k in ((7, 1), (10, 3), (27, 5)):
            features = unified.CurveFeatures(synthetic_curve(seed), k)
            for reference in unified.REFERENCE_SCALES:
                peaks, previous = features.evidence(reference, 1)
                for radius in (2, 3):
                    current_peaks, current = features.evidence(reference, radius)
                    np.testing.assert_array_equal(current_peaks, peaks)
                    np.testing.assert_array_equal(current[:, :2], previous[:, :2])
                    self.assertTrue(np.all(current[:, 2] >= previous[:, 2] - 1e-12))
                    previous = current

    def test_rounded_scales_have_one_vote_per_physical_width(self):
        curve = np.array([0, 0, 0.1, 0.4, 0.8, 0.4, 0.1, 0, 0])
        features = unified.CurveFeatures(curve, 1)
        peaks, evidence = features.evidence(1)
        np.testing.assert_array_equal(peaks, [4])
        self.assertEqual(set(features.effective_scales), {1, 2})
        # The width-one and width-two votes are 0.5 and 0.0. A repeated
        # width-two vote would incorrectly change the median to zero.
        np.testing.assert_allclose(evidence, [[1, 1, 0.25]])

    def test_zero_curve_has_no_candidates_and_preserves_false_negatives(self):
        curve = np.zeros(30)
        config = unified.Config("unified", 1.5, 0.5, 0.05)
        features = unified.CurveFeatures(curve, 3)
        record = {"curve": curve, "gt": [[2, 5, 8]]}
        for backbone, mode in (("metst", "fixed"), ("boostingvrme", "native")):
            prepared = unified.prepare(record, features, 1.5, backbone, mode)
            counts, predictions = prepared.evaluate([config], details=True)
            np.testing.assert_array_equal(counts, [[0, 0, 1]])
            self.assertEqual(predictions, [[]])
            self.assertEqual(unified.detect_curve(curve, 3, config, backbone, mode), [])

    def test_positive_constant_curve_does_not_create_an_edge_padding_peak(self):
        config = unified.Config("unified", 2, 0.5, 0.3)
        self.assertEqual(unified.detect_curve(np.ones(20), 3, config, "metst", "fixed"), [])

    def test_empty_short_and_invalid_curves_are_rejected(self):
        for curve in ([], [0], [0, 0], [[0, 0, 0]], [0, np.nan, 0], [0, np.inf, 0]):
            with self.subTest(curve=curve), self.assertRaises(ValueError):
                unified.CurveFeatures(curve, 3)
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                unified.CurveFeatures(np.zeros(20), k)

    def test_batched_predictions_and_counts_match_original_decoder(self):
        for seed in (7, 11, 42):
            curve = synthetic_curve(seed)
            record = {"curve": curve, "gt": [[0, 5, 11], [31, 38, 45],
                                              [32, 39, 45], [76, 82, 88], [96, 103, 111]]}
            features = unified.CurveFeatures(curve, 5)
            for reference, radius in ((1, 1), (1.5, 2), (2, 3)):
                peaks, evidence = features.evidence(reference, radius)
                configs = [unified.Config(family, reference, weight, threshold, radius)
                           for family, weight in (("unified", 0), ("unified", 0.5),
                                                  ("height", 1), ("global", 0), ("local", 0))
                           for threshold in (0.05, 0.4, 0.75)]
                for backbone, mode in (("metst", "fixed"), ("boostingvrme", "fixed"),
                                       ("boostingvrme", "native")):
                    prepared = unified.prepare(record, features, reference, backbone, mode, radius)
                    counts, predictions = prepared.evaluate(configs, details=True)
                    for index, config in enumerate(configs):
                        with self.subTest(seed=seed, config=config, backbone=backbone, mode=mode):
                            keep = unified.evidence_scores(evidence, config) >= config.threshold
                            expected = reference_decode(curve, 5, peaks, keep, backbone, mode)
                            self.assertEqual(intervals_only(predictions[index]), expected)
                            reference_counts = count_video(expected, record)
                            np.testing.assert_array_equal(counts[index],
                                                          (reference_counts.tp, reference_counts.fp,
                                                           reference_counts.fn))

    def test_native_control_matches_original_detection_entrypoint(self):
        config = unified.Config("native", 2, 1, 0.55)
        for seed in (7, 11, 42):
            curve = synthetic_curve(seed)
            expected = native_predictions(curve, 5, NativeConfig(2, 0.55, 1))
            actual = unified.detect_curve(curve, 5, config, "boostingvrme", "native")
            self.assertEqual(intervals_only(actual), expected)


if __name__ == "__main__":
    unittest.main()
