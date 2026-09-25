"""Critical-path tests; synthetic fixtures are never experimental results."""
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import run_grayzone_rho3 as run
from test_p8_phase2 import BASE


class Tests(unittest.TestCase):
    def test_boundaries_nonmonotonicity_and_endpoints(self):
        c = run.Config('Gray', 0, 1., 1., branch='G', tau_low=.4, tau_high=.8, eta=.5)
        v = np.array([[.399,1], [.4,.2], [.6,.25], [.79,.25], [.8,0], [0,1]])
        self.assertEqual(run.masks(v,c)[0].tolist(), [False,True,False,False,True,False])
        # Same R: increasing B within the gray zone can turn acceptance off.
        self.assertEqual(run.masks(np.array([[.5,.25],[.6,.25],[.8,.25]]),c)[0].tolist(), [True,False,True])
        for branch in ('G','L'):
            cc = replace(c,branch=branch)
            b = v[:,0 if branch=='G' else 1]
            np.testing.assert_array_equal(run.masks(v,replace(cc,eta=0))[0], b>=.4)
            np.testing.assert_array_equal(run.masks(v,replace(cc,method='Gray_no_support'))[0], b>=.4)
            np.testing.assert_array_equal(run.masks(v,replace(cc,method='Gray_high_only'))[0], b>=.8)
        self.assertEqual(run.masks(np.empty((0,2)),c)[0].size,0)
        for bad in (replace(c,tau_low=.8),replace(c,eta=-1),replace(c,branch='bad')):
            with self.assertRaises(ValueError):run.masks(v,bad)

    def test_grid_coverage_unique_and_strong_single_controls(self):
        grids=run.grids()
        self.assertEqual(len(grids['G']),231)
        self.assertEqual(len(grids['L']),924)
        self.assertEqual(len(grids['Mean']),480)
        self.assertEqual(len(grids['P8_control']),480)
        self.assertEqual(len(grids['Gray']),5088)
        self.assertEqual(sum(map(len,grids.values())),7203)
        for m,cs in grids.items():
            self.assertEqual([c.config_id for c in cs],list(range(len(cs))))
            signatures=[tuple(v for k,v in run.asdict(c).items() if k!='config_id') for c in cs]
            self.assertEqual(len(signatures),len(set(signatures)))
        self.assertEqual({c.local_radius for c in grids['G']},{1.})
        for m in ('L','Mean','P8_control','Gray'):
            self.assertEqual({c.local_radius for c in grids[m]},{.5,1.,2.,3.})
        self.assertEqual({c.branch for c in grids['Gray']},{'G','L'})
        self.assertTrue(all(c.eta>0 and 0<c.tau_low<c.tau_high<=1 for c in grids['Gray']))
        singles={c.threshold for c in grids['G']}
        self.assertTrue({.35,.4,.7,.8}.issubset(singles))
        self.assertTrue({x for pair in run.PAIRS for x in pair}.issubset(singles))
        for t in run.TAUS:
            effective=2**(1/8)*t
            self.assertTrue(effective in singles or effective>1)
        self.assertEqual(sum(t>1 for t in singles),1)

    def test_outer_subject_cannot_affect_choice(self):
        counts=np.array([[[1,9,0],[3,0,0],[3,0,0]],[[1,0,0],[2,2,1],[2,2,1]]])
        w,_=run.shared.choose(counts,0)
        counts[:,0]=[[0,100000,1],[100000,0,0]]
        w2,_=run.shared.choose(counts,0)
        self.assertEqual(w,0);self.assertEqual(w,w2)

    def test_candidate_only_guard(self):
        class Good:
            def decode_glsd_subject(self,records,si,config,core,metric,official,rec):
                return core.selected_peaks(config)
        # bound methods exclude self from inspect.signature
        self.assertTrue(run.verify_candidate_only_config(Good())['config_only_in_selected_peaks'])
        class Bad:
            def decode_glsd_subject(self,records,si,config,core,metric,official,rec):
                return core.selected_peaks(config)+config.threshold
        with self.assertRaises(RuntimeError):run.verify_candidate_only_config(Bad())

    def test_event_rematching_delta_and_fp_identity(self):
        def rec(gt,fp):
            return dict(subject='s',method='m',full_counts=[1,1,1],candidates=[],
                events=[dict(video_id='s/v',interval=[0,9],matched_gt_id=gt,full_TP=1,full_FP=0),
                        dict(video_id='s/v',interval=fp,matched_gt_id=None,full_TP=0,full_FP=1)])
        d=run.event_delta(rec('g1',[20,29]),rec('g0',[30,39]))
        self.assertEqual(d['added_gt_ids'],['g1']);self.assertEqual(d['lost_gt_ids'],['g0'])
        self.assertEqual(d['delta_FP'],0)
        self.assertEqual(len(d['added_fp_events']),1);self.assertEqual(len(d['removed_fp_events']),1)

    def test_search_deduplicates_exact_masks_and_resumes(self):
        context=(None,BASE,None,None,[dict(result_all=[[1,2,3,4]],k_p=2)],[],['s'],None)
        core=run.FusionCore(BASE);core.expected_gt=[1]
        cs=[run.Config('G',0,1,1,.5),run.Config('Gray',0,1,1,branch='G',tau_low=.4,tau_high=.5,eta=1)]
        # Both keep peak1; Gray keeps peak3 too, so use high=.6/low=.5 to match.
        cs[1]=replace(cs[1],tau_low=.5,tau_high=.6)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with patch.object(run.ledger,'detailed_decode',return_value=dict(raw_counts=[1,0,0],full_counts=[1,0,0])) as decode:
                a=run.search(context,core,cs,out)
                self.assertEqual(decode.call_count,1)
                b=run.search(context,core,cs,out)
                self.assertEqual(decode.call_count,1)
            np.testing.assert_array_equal(a,b)
            with self.assertRaisesRegex(RuntimeError,'identity'):
                run.search(context,core,[replace(cs[0],threshold=.7),cs[1]],out)

    def test_full_report_selects_branch_and_locks_both_ablations(self):
        cs={m:[run.Config(m,0,1,1,.5)] for m in ('G','L','Mean','P8_control')}
        cs['Gray']=[run.Config('Gray',0,1,2,branch='L',tau_low=.3,tau_high=.5,eta=.75)]
        g=np.array([[[1,1,1],[1,1,1]]]);l=np.array([[[2,0,0],[2,0,0]]])
        tables={m:(l,l) if m in ('L','Gray') else (g,g) for m in cs}
        ctx=(None,BASE,None,None,[],[],['a','b'],None)
        seen=[]
        def detailed(context,core,si,c):
            seen.append(c);n=2 if c.method in ('L','Gray') else 1
            s=context[6][si]
            ev=[dict(video_id=s+'/v',interval=[10*i,10*i+9],matched_gt_id=s+'/v/g'+str(i),full_TP=1,full_FP=0) for i in range(n)]
            if n==1:ev.append(dict(video_id=s+'/v',interval=[50,59],matched_gt_id=None,full_TP=0,full_FP=1))
            return dict(subject=s,method=c.method,config=run.asdict(c),events=ev,candidates=[],
                        raw_counts=[n,2-n,2-n],full_counts=[n,2-n,2-n])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with patch.object(run.ledger,'detailed_decode',side_effect=detailed):
                run.report(ctx,run.FusionCore(BASE),cs,tables,(g[0],g[0]),out)
            self.assertEqual(len([c for c in seen if c.method in run.ABLATIONS]),4)
            self.assertTrue(all((c.branch,c.local_radius,c.tau_low,c.tau_high,c.eta)==('L',2,.3,.5,.75) for c in seen if c.method in run.ABLATIONS))
            decision=json.loads((out/'decision.json').read_text())
            self.assertFalse(decision['exceeds_tuned_G_and_L'])
            self.assertFalse(decision['exceeds_training_selected_single'])
            with gzip.open(out/'event_records.jsonl.gz','rt') as f:recs=[json.loads(x) for x in f]
            self.assertEqual([r['full_counts'] for r in recs if r['method']=='Single_selected'],[[2,0,0]]*2)

    def test_gray_detailed_decode_with_distinct_official_evaluators(self):
        from test_p8_matching import MeanAveragePrecision2d as Aggregate
        from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d as Video
        scope={'MeanAveragePrecision2d':Video,'np':np}
        exec('def spotting(aggregate,preds,gt):\n    video=MeanAveragePrecision2d(num_classes=1)\n    video.add(preds,gt)\n    aggregate.add(np.column_stack((preds,np.zeros(len(preds)))),gt)\n    return video\n',scope)
        official=SimpleNamespace(spotting=scope['spotting'])
        class Runner:
            def decode_glsd_subject(self,records,si,c,core,metric,official,rec):
                peaks=core.GLSDFeatures(records[si]['result_all'][0],2).selected_peaks(c)
                preds=np.array([[0,0,9,0,0,1,4] for _ in peaks],float).reshape(-1,7)
                gt=np.array([[0,0,9,0,0,0,0,4]],float)
                agg=metric(num_classes=1);video=official.spotting(agg,preds,gt)
                v=agg.value(iou_thresholds=.5)[.5][0]
                raw=(int(sum(v['tp'])),int(sum(v['fp'])),1-int(sum(v['tp'])))
                matches=video.value(iou_thresholds=.5)[.5][0]['pred_match_gt']
                targets=[1 if np.asarray(x).reshape(-1)[0]>=0 else -1 for x in matches.get(0,[])]
                return raw,[preds],[1]*len(preds),targets,matches,video
            def full_counts_from_official_synergy(self,raw,labels,targets):return raw
        record=dict(result_all=[[1,2,3,4]],k_p=2,videos=['v'],samples=[[[0,4,9]]])
        ctx=(Runner(),BASE,Aggregate,official,[record],[],['s'],None)
        core=run.FusionCore(BASE);core.expected_gt=[1]
        c=run.Config('Gray',0,1,1,branch='G',tau_low=.5,tau_high=.9,eta=.25)
        with run.evaluator.install(Aggregate):r=run.ledger.detailed_decode(ctx,core,0,c)
        self.assertEqual(r['full_counts'],[1,0,0])
        self.assertEqual(r['events'][0]['matched_gt_id'],'s/video_0/gt_0')
        self.assertEqual(r['candidates'][0]['gray_passed'],[True,False])


if __name__=='__main__':unittest.main()
