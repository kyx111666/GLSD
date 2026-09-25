from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

from my_method.gl_saliency_skill.adapters import (
    BoostingNativeIntervalAdapter,
    FixedIntervalAdapter,
)
from my_method.gl_saliency_skill.evidence import (
    Config,
    CurveFeatures,
    configuration_grid,
    evidence_scores,
    pure_config_indexes,
)
from my_method.gl_saliency_skill.selection import choose
from my_method.gl_saliency_skill.skill import GLSaliencySkill


WORKSPACE = Path(__file__).resolve().parents[4]
HISTORICAL_SOURCE = (
    WORKSPACE
    / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
)
sys.path.insert(0, str(HISTORICAL_SOURCE))
import unified_persistence as historical  # noqa: E402


def synthetic_curve(seed=7):
    rng = np.random.default_rng(seed)
    frames = np.arange(140)
    curve = rng.uniform(0, 0.04, len(frames))
    for peak, width, height in (
        (5, 2, 0.6), (38, 5, 0.95), (47, 3, 0.75),
        (82, 2, 0.5), (103, 8, 0.8), (133, 3, 0.65),
    ):
        curve += height * np.exp(-0.5 * ((frames - peak) / width) ** 2)
    return curve


class FinalGLSkillTests(unittest.TestCase):
    def test_01_full_grid_and_pure_90_match_history(self):
        current = configuration_grid()
        old = historical.configuration_grid()
        self.assertEqual([config.identifier for config in current], [config.identifier for config in old])
        self.assertEqual(len(pure_config_indexes(current)), 90)

    def test_02_G_and_L_are_exact_historical_values(self):
        curve = synthetic_curve()
        for reference in (1.0, 1.5, 2.0):
            for radius in (1.0, 2.0, 3.0):
                peaks, evidence = CurveFeatures(curve, 5).evidence(reference, radius)
                old_peaks, old_evidence = historical.CurveFeatures(curve, 5).evidence(reference, radius)
                np.testing.assert_array_equal(peaks, old_peaks)
                np.testing.assert_array_equal(evidence, old_evidence)

    def test_03_fusion_is_exact_arithmetic_mean(self):
        evidence = np.array([[0.7, 0.2, 0.8], [0.1, 0.3, 0.5]])
        config = Config("unified", 1.5, 0, 0.4, 2)
        np.testing.assert_array_equal(evidence_scores(evidence, config), [0.5, 0.4])

    def test_04_fixed_decoder_matches_historical(self):
        curve = synthetic_curve()
        config = Config("unified", 1.5, 0, 0.4, 2)
        events = GLSaliencySkill(config).decode(curve, 5, {"interval_adapter": FixedIntervalAdapter()})
        old = historical.detect_curve(curve, 5, historical.Config("unified", 1.5, 0, 0.4, 2), "metst", "fixed")
        self.assertEqual([(e["onset"], e["offset"], e["peak"]) for e in events], [(e["onset"], e["offset"], e["peak"]) for e in old])

    def test_05_boosting_decoder_matches_historical(self):
        curve = synthetic_curve()
        config = Config("unified", 2, 0, 0.3, 3)
        events = GLSaliencySkill(config).decode(curve, 5, {"interval_adapter": BoostingNativeIntervalAdapter()})
        old = historical.detect_curve(curve, 5, historical.Config("unified", 2, 0, 0.3, 3), "boostingvrme", "native")
        self.assertEqual([(e["onset"], e["offset"], e["peak"]) for e in events], [(e["onset"], e["offset"], e["peak"]) for e in old])

    def test_06_historical_tie_break_is_exact(self):
        rng = np.random.default_rng(9)
        counts = rng.integers(0, 100, size=(511, 3))
        indexes = pure_config_indexes(configuration_grid())
        self.assertEqual(choose(counts, indexes), historical.choose(counts, indexes))

    def test_07_zero_curve_has_no_candidates(self):
        config = Config("unified", 1.5, 0, 0.05, 1)
        self.assertEqual(GLSaliencySkill(config).decode(np.zeros(30), 3, {"interval_adapter": FixedIntervalAdapter()}), [])

    def test_08_evidence_and_decode_apis_do_not_accept_GT(self):
        self.assertNotIn("gt", inspect.signature(CurveFeatures).parameters)
        self.assertNotIn("gt", inspect.signature(GLSaliencySkill.decode).parameters)
        self.assertNotIn("ground_truth", inspect.signature(GLSaliencySkill.decode).parameters)

    def test_09_trainable_parameters_are_zero(self):
        skill = GLSaliencySkill(Config("unified", 1, 0, 0.5, 1))
        self.assertEqual(skill.trainable_parameters, 0)

    def test_10_learned_or_weighted_fusion_is_rejected(self):
        with self.assertRaises(ValueError):
            GLSaliencySkill(Config("unified", 1, 0.25, 0.5, 1))


if __name__ == "__main__":
    unittest.main()
