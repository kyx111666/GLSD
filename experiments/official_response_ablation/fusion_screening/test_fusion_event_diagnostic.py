import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import run_fusion_event_diagnostic as run
from test_one_to_one_evaluator import MeanAveragePrecision2d, make_metric, counts


class DiagnosticTests(unittest.TestCase):
    def test_capture_with_real_one_to_one_metric_and_full_neutral_filter(self):
        class Runner:
            def decode_glsd_subject(self, records, si, config, core, metric, official, recognition):
                response = np.asarray(records[si]['result_all'][0], dtype=float)
                core.calls += 1
                core.trace.append(dict(response_sha256=run.hashlib.sha256(response.tobytes()).hexdigest(),
                    peaks=[4,24],G=[.9,.8],L=[.7,.6],score=[.9,.8],retained=[True,True]))
                m = make_metric([(0,9),(20,29)],[(0,9),(20,29)])
                match = m.value(.5)[.5][0]['pred_match_gt']
                return counts(m),m.pred_all,[1,4],[1,1],match,m
            def final_samples(self, records): return [r['samples'] for r in records]
            def full_counts_from_official_synergy(self, raw, labels, targets):
                tp,fp,fn = raw
                for label,target in zip(labels,targets):
                    if label == 4:
                        if target == -1: fp-=1
                        else: tp-=1;fn+=1
                return tp,fp,fn
        records=[dict(videos=['v'],samples=[[[0,4,9],[20,24,29]]],result_all=[[0.,1.,0.]])]
        context=(Runner(),None,MeanAveragePrecision2d,None,records,None,['037'],None)
        core=SimpleNamespace(calls=0,trace=[],capture=False)
        original=MeanAveragePrecision2d.add
        with run.evaluator.install(MeanAveragePrecision2d):
            result=run.detailed_decode(context,core,0,run.tuning.Config('G',0,2.,1.,.65))
        self.assertIs(MeanAveragePrecision2d.add, original)
        self.assertEqual(result['raw_counts'],[2,0,0])
        self.assertEqual(result['full_counts'],[1,0,1])
        self.assertEqual(result['events'][0]['matched_gt_id'],'037/video_0/gt_0')
        self.assertFalse(result['events'][1]['full_retained'])
        self.assertEqual(result['candidates'][0]['video_name'],'v')

    def test_capture_restored_after_exception(self):
        original=MeanAveragePrecision2d.add
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with run.capture_registered_inputs(MeanAveragePrecision2d):
                raise RuntimeError('deliberate')
        self.assertIs(original,MeanAveragePrecision2d.add)

    def test_same_tp_count_can_hide_different_detected_gt(self):
        def record(gt):
            return dict(subject='s',variant='expanded',method='G',ground_truth=['g0','g1'],
                raw_counts=[1,0,1],full_counts=[1,0,1],events=[dict(matched_gt_id=gt,raw_TP=1,full_TP=1)])
        result=run.compare_records(record('g1'),record('g0'))
        self.assertEqual(result['full_delta_TP'],0)
        self.assertEqual(result['full_added_gt_ids'],['g1'])
        self.assertEqual(result['full_lost_gt_ids'],['g0'])

    def test_summary_outputs_all_comparisons_and_score_rows(self):
        records=[]
        for variant in ('original','expanded'):
            for method in run.tuning.METHODS:
                records.append(dict(subject='s',variant=variant,method=method,
                    config=dict(reference_scale=2.,local_radius=2.,threshold=.5,beta=None),
                    ground_truth=['g0'],raw_counts=[1,0,0],full_counts=[1,0,0],
                    events=[dict(video_id='s/video_0',video_name='v',interval=[0,0,9,0,0,1,4],
                        matched_gt_id='g0',raw_TP=1,raw_FP=0,full_TP=1,full_FP=0,recognition_prediction=1)],
                    candidates=[dict(video_id='s/video_0',peaks=[4],G=[.9],L=[.7],score=[.8],retained=[True])]))
        with TemporaryDirectory() as directory:
            output=Path(directory)
            run.summarize(records,output)
            differences=json.loads((output/'event_differences.json').read_text())
            self.assertEqual(len(differences),15)
            self.assertTrue(all(d['full_common_gt']==1 and not d['full_added_gt_ids'] for d in differences))
            self.assertEqual(len((output/'selected_event_scores.csv').read_text().splitlines()),11)


if __name__=='__main__': unittest.main()
