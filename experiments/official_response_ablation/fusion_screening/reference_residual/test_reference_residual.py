"""Critical-path checks; synthetic cases are not method-performance evidence."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE.parent/'metst_fusion_screening').is_dir():
    sys.path.insert(0, str(HERE.parent/'metst_fusion_screening'))
    sys.path.append(str(HERE.parent))
for parent in HERE.parents:
    if (parent/'official_BoostingVRME').is_dir():
        sys.path.append(str(parent/'official_BoostingVRME'))
        break
import colab_p8_entry  # Same pandas compatibility shim as the real entry point.
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d as Aggregate
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d as Video
import run_reference_residual as run
import residual_features as features
import typed_matching as matching


def load_core():
    paths = [p/'official_BoostingVRME/boosting_official_glds_full.py' for p in HERE.parents]
    data_root = Path(os.environ.get('GLSD_DATA_ROOT', '/content/drive/MyDrive'))
    paths.append(data_root/'GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/boosting_official_glds_full_LOCKED.py')
    source = next(p for p in paths if p.is_file())
    spec = importlib.util.spec_from_file_location('residual_test_core', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture():
    base = load_core()
    x = np.arange(100)
    response = sum(a*np.exp(-((x-c)/3.)**2) for a, c in ((1., 15), (.65, 45), (.85, 75)))
    peak = int(base.GLSDFeatures(response, 3).evidence(2., 2.)[0][0])
    scope = {'MeanAveragePrecision2d': Video}
    exec('def spotting():\n    pass\n', scope)
    official = SimpleNamespace(spotting=scope['spotting'])
    class Runner:
        def decode_glsd_subject(self, records, si, config, core, metric_class, official, recognition):
            record = records[si]
            peaks = core.GLSDFeatures(record['result_all'][0], 3).selected_peaks(config)
            preds = np.array([[p-2, 0, p+2, 0, 0, 1, p] for p in peaks], float).reshape(-1, 7)
            gt = np.array([[peak-2, 0, peak+2, 0, 0, 0, 0, peak]], float)
            aggregate, video = metric_class(num_classes=1), Video(num_classes=1)
            aggregate.add(np.column_stack((preds, np.zeros(len(preds)))), gt)
            video.add(preds, gt)
            values = aggregate.value(iou_thresholds=.5)[.5][0]
            raw = (int(sum(values['tp'])), int(sum(values['fp'])), 1-int(sum(values['tp'])))
            maps = video.value(iou_thresholds=.5)[.5][0]['pred_match_gt']
            ids = matching.normalize_map(maps).get(0, [])
            targets = [1 if i >= 0 else -1 for i in ids]
            labels = [1 if i >= 0 else 4 for i in ids]
            # Match real spotting(): nonempty predictions are Python lists.
            return raw, [preds.tolist() if len(preds) else np.empty((0, 7))], labels, targets, maps, video
        def full_counts_from_official_synergy(self, raw, labels, targets):
            tp, fp, fn = raw
            for label, target in zip(labels, targets):
                if label == 4:
                    if target == -1:
                        fp -= 1
                    else:
                        tp -= 1
                        fn += 1
            return tp, fp, fn
    records = [dict(result_all=[response], k_p=3, videos=['v'], samples=[[[peak-2, peak, peak+2]]]) for _ in range(3)]
    context = (Runner(), base, Aggregate, official, records, [], ['a', 'b', 'c'], None)
    core = features.FusionCore(base)
    core.expected_gt = [1, 1, 1]
    return context, core


class Tests(unittest.TestCase):
    def test_typed_matching_both_classes_and_restoration(self):
        originals = [cls._evaluate_class for cls in (Aggregate, Video)]
        with contextlib.ExitStack() as stack:
            contracts = [stack.enter_context(matching.install(cls))['return_contract'] for cls in (Aggregate, Video)]
            self.assertEqual(contracts, ['numeric_list', 'string_scalar'])
            self.assertEqual(len(matching.synthetic_probe([Aggregate, Video])), 28)
        self.assertEqual(originals, [cls._evaluate_class for cls in (Aggregate, Video)])

    def test_video_binding_and_distinct_caller_policies_are_preserved(self):
        original_check = Video._evaluate_class.__globals__['check_box']
        p = np.array([[0, 0, 19, 0, 0, 1, 9]]*2, float)
        g = np.array([[0, 0, 9, 0, 0, 0, 0, 4], [10, 0, 19, 0, 0, 0, 0, 14]], float)
        with matching.install(Aggregate), matching.install(Video):
            # The original function's module binding must remain untouched.
            self.assertIs(original_check, sys.modules[Video.__module__].check_box)
            a, v = Aggregate(num_classes=1), Video(num_classes=1)
            a.add(p, g)
            v.add(p, g)
            self.assertEqual(sum(a.value(iou_thresholds=.5)[.5][0]['tp']), 2)
            self.assertEqual(sum(v.value(iou_thresholds=.5)[.5][0]['tp']), 1)

    def test_real_decoder_and_remote_consumers_with_lists_and_empty_video(self):
        import sealed_decoder_fixture as runner
        import remote_official_fixture as official
        base = load_core()
        x = np.arange(100)
        response = sum(a*np.exp(-((x-c)/3.)**2) for a, c in ((1., 15), (.65, 45), (.85, 75)))
        peak = int(base.GLSDFeatures(response, 3).evidence(2., 2.)[0][0])
        sequence = np.ones(100, dtype=int)
        sequence[35:85] = 4
        record = dict(result_all=[response, np.zeros(100)], k_p=3, frame_skip=1,
            result1_all=[sequence, np.ones(100, dtype=int)], videos=['event', 'empty'],
            samples=[[[peak-3, peak, peak+3]], [[10, 13, 16]]],
            emotions=[['positive'], ['positive']])
        context = (runner, base, Aggregate, official, [record], [], ['test'], None)
        core = features.FusionCore(base)
        core.expected_gt = [2]
        with matching.install(Aggregate), matching.install(Video):
            for method in (*features.METHODS, *features.LOCKED):
                for tau in (.05, .95):
                    row = run.decode(context, core, 0, run.phase.Config(method, 0, 2., 2., tau), True)
                    self.assertEqual(row['full_counts'][0]+row['full_counts'][2], 2)
                    self.assertEqual(row['candidates'][1]['peaks'], [])
            row = run.decode(context, core, 0, run.phase.Config('G', 0, 2., 2., .05), True)
            self.assertEqual(row['raw_counts'], [1, 2, 1])
            self.assertEqual(row['full_counts'], [1, 0, 1])

    def test_empty_map_interface_bug_is_detected(self):
        scope = Video._evaluate_class.__globals__
        with matching.install(Video):
            scope['check_box'] = lambda *a, **k: (1, [0])
            metric = Video(num_classes=1)
            metric.add(np.array([[0, 0, 9, 0, 0, 1, 4.]]), np.array([[0, 0, 9, 0, 0, 0, 0, 4.]]))
            with self.assertRaisesRegex(RuntimeError, 'missing prediction match'):
                metric.value(iou_thresholds=.5)

    def test_signed_score_and_locked_module_removal(self):
        g, l0, l = np.array([.85, .6, .8]), np.array([.75, .2, .5]), np.array([.3, .5, .5])
        np.testing.assert_allclose(features.score(g, l, l0, 'RefResidual'), [.4, .9, .8])
        np.testing.assert_array_equal(features.score(g, l, l0, 'no_local'), g)
        np.testing.assert_allclose(features.score(g, l, l0, 'no_G'), [0, .3, 0])
        np.testing.assert_allclose(features.score(g, l, l0, 'no_negative'), [.85, .9, .8])

    def test_intermediates_against_actual_core_with_empty_and_duplicate_scales(self):
        base = load_core()
        rng = np.random.default_rng(17)
        for response, k in [(np.zeros(30), 3), (rng.random(100), 1), (rng.random(100), 3)]:
            feature = features.FusionCore(base).GLSDFeatures(response, k)
            for reference in (1., 1.5, 2.):
                data = feature.residual_evidence(reference, 2.)
                peaks, evidence = feature.base.evidence(reference, 2.)
                np.testing.assert_array_equal(data['G'], evidence[:, 0])
                np.testing.assert_array_equal(data['L'], evidence[:, 1])
                self.assertEqual(len(data['peaks']), len(peaks))
                self.assertEqual(len(data['physical_widths']), len(set(data['physical_widths'])))
        feature = features.FusionCore(base).GLSDFeatures(rng.random(100), 3)
        feature.evidence(2., 2.)
        width = next(iter(feature.base.effective_scales))
        feature.base.local_cache[width, 6][0] += .1
        with self.assertRaisesRegex(RuntimeError, 'sealed local-support'):
            feature.residual_evidence(2., 2.)

    def test_dual_evaluator_ledger_and_neutral_FP_removal(self):
        context, core = fixture()
        with matching.install(Aggregate), matching.install(Video):
            result = run.decode(context, core, 0, run.phase.Config('G', 0, 2., 2., .05), True)
        self.assertEqual(result['raw_counts'], [1, 2, 0])
        self.assertEqual(result['full_counts'], [1, 0, 0])
        self.assertEqual(sum(e['raw_FP']-e['full_FP'] for e in result['events']), 2)
        self.assertEqual(result['events'][0]['matched_gt_id'], 'a/video_0/gt_0')

    def test_small_complete_screen_and_checkpoints(self):
        context, core = fixture()
        with tempfile.TemporaryDirectory() as directory, matching.install(Aggregate), matching.install(Video):
            output = Path(directory)
            with contextlib.redirect_stdout(run.io.StringIO()):
                run.screen(context, core, output)
            for method in features.METHODS:
                path = output/('search_counts_%s.npz' % method)
                with np.load(path) as saved:
                    self.assertTrue(saved['done'].all())
                    self.assertEqual(saved['full'].shape, (19, 3, 3))
                before = core.calls
                run.search(context, core, run.grids()[method], path)
                self.assertEqual(before, core.calls)
            self.assertTrue((output/'event_records.jsonl.gz').is_file())
            self.assertEqual(len(json.loads((output/'decision.json').read_text())['event_deltas']), 7)


if __name__ == '__main__':
    unittest.main()
