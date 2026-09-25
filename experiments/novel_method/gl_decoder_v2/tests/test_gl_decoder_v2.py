from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from my_method.gl_decoder_v2.adapters import load_boostingvrme, load_me_tst
from my_method.gl_decoder_v2.decoder import GLConfig, GLDecoder
from my_method.gl_decoder_v2.evaluation import add_counts, evaluate_events, native_events
from my_method.gl_decoder_v2.selection import enumerate_configs, select_config


ROOT = Path(__file__).resolve().parents[3]


class GLDecoderV2Tests(unittest.TestCase):
    def setUp(self):
        self.score = np.array([0, 1, 0, 2, 0, 1, 0, 3, 0], dtype=float)
        self.config = GLConfig(1.0, 1, 0.3)
        self.decoder = GLDecoder()

    def test_01_global_evidence_is_deterministic(self):
        first = self.decoder.evidence(self.score, 1, self.config).global_scores
        second = self.decoder.evidence(self.score, 1, self.config).global_scores
        self.assertEqual(first, second)

    def test_02_local_evidence_is_deterministic(self):
        first = self.decoder.evidence(self.score, 1, self.config).local_scores
        second = self.decoder.evidence(self.score, 1, self.config).local_scores
        self.assertEqual(first, second)

    def test_03_fusion_is_exact_arithmetic_mean(self):
        evidence = self.decoder.evidence(self.score, 1, self.config)
        fused = self.decoder.fused_scores(evidence)
        self.assertTrue(all(value == (g + l) / 2 for value, g, l in zip(
            fused, evidence.global_scores, evidence.local_scores, strict=True
        )))

    def test_04_grid_has_exactly_90_unique_configs(self):
        configs = enumerate_configs([1, 1.5, 2], [1, 2, 3], [.05, .1, .2, .3, .4, .5, .55, .6, .65, .75])
        self.assertEqual(len(configs), 90)
        self.assertEqual(len({config.config_id for config in configs}), 90)

    def test_05_outer_subject_is_excluded(self):
        configs = [GLConfig(1, 1, .1), GLConfig(2, 1, .1)]
        counts = {
            configs[0].config_id: {"train": {"TP": 2, "FP": 0, "FN": 0}, "held": {"TP": 0, "FP": 99, "FN": 99}},
            configs[1].config_id: {"train": {"TP": 1, "FP": 1, "FN": 1}, "held": {"TP": 99, "FP": 0, "FN": 0}},
        }
        selected, _ = select_config(configs, counts, "held")
        self.assertEqual(selected, configs[0])

    def test_06_tie_break_is_deterministic(self):
        configs = [GLConfig(2, 2, .2), GLConfig(1, 1, .1)]
        counts = {config.config_id: {"a": {"TP": 1, "FP": 0, "FN": 0}, "b": {"TP": 0, "FP": 0, "FN": 0}} for config in configs}
        selected, _ = select_config(configs, counts, "b")
        self.assertEqual(selected.config_id, min(config.config_id for config in configs))

    def test_07_candidate_generation_is_deterministic(self):
        first = self.decoder.evidence(self.score, 1, self.config).candidates
        second = self.decoder.evidence(self.score.copy(), 1, self.config).candidates
        self.assertEqual(first, second)

    def _native_counts(self, loader, relative_path):
        rows = []
        for record in loader(ROOT / relative_path):
            rows.append(evaluate_events(native_events(record["score"], record["temporal_scale"]), record["gt"]))
        return add_counts(rows)

    def test_08_me_tst_native_baselines(self):
        sam = self._native_counts(load_me_tst, "caches/me_tst/sammlv_strategy1_outputs.pkl")
        cas = self._native_counts(load_me_tst, "caches/me_tst/casme3_strategy1_outputs.pkl")
        self.assertEqual((sam["TP"], sam["FP"], sam["FN"]), (53, 184, 106))
        self.assertEqual((cas["TP"], cas["FP"], cas["FN"]), (81, 912, 777))

    def test_09_boosting_current_native_baselines(self):
        sam = self._native_counts(load_boostingvrme, "caches/boostingvrme/sammlv_curves.pkl")
        cas = self._native_counts(load_boostingvrme, "caches/boostingvrme/casme3_curves.pkl")
        self.assertEqual((sam["TP"], sam["FP"], sam["FN"]), (54, 158, 105))
        self.assertEqual((cas["TP"], cas["FP"], cas["FN"]), (80, 830, 778))

    def test_10_same_decoder_class_for_both_adapters(self):
        self.assertIs(type(GLDecoder()), type(GLDecoder()))

    def test_11_rerun_is_deterministic(self):
        first = self.decoder.decode(self.score, 1, self.config)
        second = self.decoder.decode(self.score, 1, self.config)
        self.assertEqual(first, second)

    def test_12_evidence_api_has_no_ground_truth_argument(self):
        parameters = inspect.signature(self.decoder.extractor.extract).parameters
        self.assertNotIn("gt", parameters)
        self.assertNotIn("ground_truth", parameters)


if __name__ == "__main__":
    unittest.main()
