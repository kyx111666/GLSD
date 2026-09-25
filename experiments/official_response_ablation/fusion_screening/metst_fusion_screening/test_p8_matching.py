"""Regression tests against the installed STR evaluator; synthetic evidence only."""
import sys
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import colab_p8_entry  # pandas compatibility used by the actual runner
import run_p8_one_to_one as run
import one_to_one_evaluator as adapter

# Local source tree fallback; Colab uses its actual configured ME-TST checkout.
here = Path(__file__).resolve()
for parent in here.parents:
    candidate = parent / 'official_BoostingVRME'
    if candidate.is_dir():
        sys.path.append(str(candidate))
        break
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d


def metric(predictions, targets):
    m = MeanAveragePrecision2d(num_classes=1)
    p = np.array([[a, 0, b, 0, 0, 1, 0, 0] for a, b in predictions], float).reshape(-1, 8)
    g = np.array([[a, 0, b, 0, 0, 0, 0, 0] for a, b in targets], float).reshape(-1, 8)
    m.add(p, g)
    return m


def counts(m):
    values = m.value(iou_thresholds=0.5)[0.5][0]
    tp, fp = int(sum(values['tp'])), int(sum(values['fp']))
    return (tp, fp, int(m.class_counter.sum()) - tp), values['pred_match_gt']


class Tests(unittest.TestCase):
    def test_duplicate_tp_repair_and_restoration(self):
        m = metric([(0,9), (0,9), (20,29), (40,49)], [(0,9), (20,29), (40,49)])
        before = counts(m)
        with adapter.install(MeanAveragePrecision2d):
            actual, matches = counts(m)
            self.assertEqual(actual, (3,1,0))
            self.assertEqual(matches[0], [[0], [-1], [1], [2]])
        self.assertEqual(counts(m), before)

    def test_one_prediction_one_tp(self):
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(metric([(0,19)], [(0,9),(10,19)]))[0], (1,0,1))

    def test_empty_and_cross_video_cases(self):
        with adapter.install(MeanAveragePrecision2d):
            self.assertEqual(counts(metric([], [(0,9)]))[0], (0,0,1))
            self.assertEqual(counts(metric([(0,9)], []))[0], (0,1,0))
            m = metric([(0,9)], [(0,9)])
            m.add(m.pred_all[0].copy(), m.gt_all[0].copy())
            self.assertEqual(counts(m)[0], (2,0,0))

    def test_guard_rejects_negative_or_inconsistent_gt(self):
        run.validate((3,1,0), (2,1,1), 3)
        for raw in ((4,15,-1), (3,1,1), (1.5,0,1.5)):
            with self.assertRaises(RuntimeError):
                run.validate(raw, None, 3)

    def test_prepared_context_keeps_same_metric_and_checks_gt(self):
        from types import SimpleNamespace
        core = SimpleNamespace(expected_gt=[3])
        with patch.object(run.shared, 'decode', return_value=((3,1,1),(3,1,1),[],[])):
            with self.assertRaisesRegex(RuntimeError, 'GT conservation'):
                run.phase.checked_decode(None, core, 0, None)

    def test_failing_case_config_is_preserved(self):
        c = run.phase.grids()['L'][38]
        self.assertEqual((c.reference_scale,c.local_radius,c.threshold),(1.,2.,.05))
        self.assertEqual(set(run.phase.grids()), {'G','L','Mean','GM_p8'})

    def test_full_search_uses_corrected_metric_and_prepared_context(self):
        from types import SimpleNamespace
        from test_p8_phase2 import BASE

        class Runner:
            def final_samples(self, records):
                return [[[(0,9),(20,29),(40,49)]]] * 2

            def decode_glsd_subject(self, records, si, c, core, metric_class, official, recognition):
                self.recognition = recognition
                selected = core.GLSDFeatures([1,2,3,4], 2).selected_peaks(c)
                predictions = [(0,9),(0,9),(20,29),(40,49)] if len(selected) else []
                raw, matches = counts(metric(predictions, [(0,9),(20,29),(40,49)]))
                return raw, predictions, [], [], None

            def full_counts_from_official_synergy(self, raw, preds, gt):
                return raw

        runner = Runner()
        context = (runner, BASE, MeanAveragePrecision2d, None, [], [], ['037','other'], None)
        ora = SimpleNamespace(SPECS={'synthetic': {}}, metst_context=lambda _: self.fail('must use prepared context'))
        tiny = {m: [run.phase.Config(m,i,1,1,t) for i,t in enumerate((.2,.95))] for m in run.phase.METHODS}
        with tempfile.TemporaryDirectory() as td, patch.object(run.phase, 'grids', return_value=tiny):
            with adapter.install(MeanAveragePrecision2d):
                rows, _ = run.phase.run_setting(ora, 'synthetic', Path(td),
                    prepared_context=context, evaluation_protocol=adapter.PROTOCOL)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all((r['TP'],r['FP'],r['FN']) == (6,2,0) for r in rows))
            self.assertTrue(runner.recognition)
            protocol = json.loads((Path(td)/'protocol.json').read_text())
            self.assertEqual(protocol['evaluation_protocol'], adapter.PROTOCOL)


if __name__ == '__main__':
    unittest.main()
