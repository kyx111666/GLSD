"""Regression checks using the local official STR evaluator, not real ME-TST data."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "official_BoostingVRME"))
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
import one_to_one_evaluator as adapter
import run_full_fusion_tuning as full
import run_one_to_one_full_tuning as runner

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, **kw: pd.concat([self, other], **kw)


def make_metric(predictions, targets):
    m = MeanAveragePrecision2d(num_classes=1)
    p = np.array([[a, 0, b, 0, 0, 1, 0, 0] for a, b in predictions], dtype=float).reshape(-1, 8)
    g = np.array([[a, 0, b, 0, 0, 0, 0, 0] for a, b in targets], dtype=float).reshape(-1, 8)
    m.add(p, g)
    return m


def counts(m):
    v = m.value(iou_thresholds=0.5)[0.5][0]
    tp, fp = int(sum(v["tp"])), int(sum(v["fp"]))
    return tp, fp, int(m.class_counter.sum())-tp


class EvaluatorTests(unittest.TestCase):
    def test_reproduces_and_fixes_negative_fn(self):
        m = make_metric([(0,9), (0,9), (20,29), (40,49)], [(0,9), (20,29), (40,49)])
        self.assertEqual(counts(m), (4,0,-1))
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(m), (3,1,0))
            self.assertEqual(m.value(0.5)[0.5][0]["pred_match_gt"][0], [[0], [-1], [1], [2]])
        self.assertEqual(counts(m), (4,0,-1))

    def test_one_prediction_cannot_match_two_gt(self):
        m = make_metric([(0,19)], [(0,9), (10,19)])
        self.assertEqual(counts(m), (2,0,0))
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(m), (1,0,1))

    def test_duplicate_and_new_gt_soft_matching(self):
        m = make_metric([(0,19), (0,19), (0,19)], [(0,9), (10,19)])
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(m), (2,1,0))

    def test_unambiguous_results_preserved(self):
        m = make_metric([(0,9), (20,29), (50,59)], [(0,9), (20,29), (80,89)])
        before = counts(m)
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(m), before)
            self.assertEqual(before, (2,1,1))

    def test_gt_indices_are_independent_between_videos(self):
        m = make_metric([(0,9)], [(0,9)])
        m.add(m.pred_all[0].copy(), m.gt_all[0].copy())
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(m), (2,0,0))

    def test_empty_predictions_or_gt(self):
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(make_metric([], [(0,9)])), (0,0,1))
            self.assertEqual(counts(make_metric([(0,9)], [])), (0,1,0))

    def test_patch_restores_even_after_exception(self):
        original = MeanAveragePrecision2d._evaluate_class
        check = original.__globals__["check_box"]
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with adapter.install(MeanAveragePrecision2d):
                raise RuntimeError("synthetic")
        self.assertIs(MeanAveragePrecision2d._evaluate_class, original)
        self.assertIs(original.__globals__["check_box"], check)

    def test_gt_conservation_and_inner_subject_isolation(self):
        runner.validate((3,1,0), (2,1,1), 3, "synthetic")
        with self.assertRaises(RuntimeError):
            runner.validate((3,1,1), None, 3, "synthetic")
        table = np.array([[[0,99,0],[2,0,1]], [[9,0,0],[1,2,2]]])
        winner, pooled = full.shared.choose(table, 0)
        table[:,0] = [[999,0,0],[0,999,999]]
        got, updated = full.shared.choose(table, 0)
        self.assertEqual(winner, got)
        np.testing.assert_array_equal(pooled, updated)

    def test_beta_endpoints_match_original_mean_and_max(self):
        g, l = np.random.default_rng(10).uniform(size=(2,1000))
        for beta, expected in ((.5,(g+l)/2), (1.,np.maximum(g,l))):
            c = full.Config("DominantEvidence",0,1.5,2,.5,beta=beta)
            np.testing.assert_array_equal(full.scores(g,l,c), expected)

    def test_full_structure_hook_preserves_sealed_mean_for_all_90_configs(self):
        path = ROOT / "official_BoostingVRME/boosting_official_glds_full.py"
        spec = importlib.util.spec_from_file_location("oto_test_sealed_core", path)
        base = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = base
        spec.loader.exec_module(base)
        response = np.random.default_rng(12).uniform(size=250)
        sealed = base.GLSDFeatures(response, 5)
        core = full.FusionCore(base)
        view = core.GLSDFeatures(response, 5)
        for config in full.grids()["EqualMean"]:
            np.testing.assert_array_equal(view.selected_peaks(config), sealed.selected_peaks(config))
        self.assertEqual(len(view.evidence_cache), 9)

    def test_refined_grid_is_shared_and_contains_every_legacy_config(self):
        runner.configure_threshold_grid("legacy")
        original = full.grids()
        try:
            runner.configure_threshold_grid("refined")
            refined = full.grids()
            self.assertEqual({m: len(g) for m,g in refined.items()},
                             dict(G=57, L=171, EqualMean=171, WeightedMean=1026, DominantEvidence=1026))
            self.assertIn(.70, full.THRESHOLDS)
            self.assertEqual(full.THRESHOLDS[-1], .95)
            def key(c):
                return (c.reference_scale, c.local_radius, c.threshold, c.alpha, c.beta)
            for method in original:
                self.assertTrue({key(c) for c in original[method]}.issubset({key(c) for c in refined[method]}))
            self.assertEqual(sum((c.reference_scale,c.local_radius,c.threshold)==(1.,2.,.05)
                                 for c in refined["L"]),1)
        finally:
            runner.configure_threshold_grid("legacy")


if __name__ == "__main__":
    unittest.main()
