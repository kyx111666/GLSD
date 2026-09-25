"""Regression tests of incremental assembly, joint selection and event reuse."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import run_joint_gm as run
from test_p8_phase2 import BASE


def oracle(c, n=2):
    # Valid synthetic counts that distinguish the p/config axis order.
    p = '1' if c.method == 'Mean' else getattr(c, 'p', '8')
    raw = np.tile([1, int(float(c.threshold)*10) + (0 if p == 'inf' else int(p)), 1], (n, 1))
    return raw


class Tests(unittest.TestCase):
    def test_new_power_checkpoint_resume_and_p_identity(self):
        cs=[run.Config('GM_joint',i,1,1,t,'4') for i,t in enumerate((.3,.5))]
        ctx=(None,None,None,None,None,None,['a','b'],None)
        calls=[]
        def interrupted(context,core,si,c):
            calls.append((si,c.config_id))
            if si==1: raise RuntimeError('interrupt')
            return (1,0,1),(1,0,1),None,[]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'new_counts_p4.npz'
            with patch.object(run.phase,'checked_decode',side_effect=interrupted):
                with self.assertRaises(RuntimeError): run.phase.search_counts(ctx,None,cs,path)
            calls.clear()
            def resumed(context,core,si,c):
                calls.append((si,c.config_id))
                return (1,0,1),(1,0,1),None,[]
            with patch.object(run.phase,'checked_decode',side_effect=resumed):
                run.phase.search_counts(ctx,None,cs,path)
                self.assertEqual(calls,[(1,0),(1,1)])
                with self.assertRaisesRegex(RuntimeError,'configs/subjects'):
                    run.phase.search_counts(ctx,None,[run.replace(c,p='2') for c in cs],path)

    def test_grids_and_mapping_all_powers(self):
        g = run.grids()
        self.assertEqual(len(g['GM_joint']),855)
        self.assertEqual(len(g['G_union']),258)
        self.assertEqual(len({(c.reference_scale,c.threshold) for c in g['G_union']}),258)
        self.assertGreater(max(c.threshold for c in g['G_union']),1)
        self.assertTrue(all(c.local_radius==1 for c in g['G_union']))
        old={(c.reference_scale,c.threshold) for c in g['G']}
        old|={(c.reference_scale,run.factor('8')*c.threshold) for c in g['G']}
        self.assertEqual(sum((c.reference_scale,c.threshold) not in old for c in g['G_union']),144)
        for p in run.POWERS:
            for tau in run.phase.TAUS:
                values=np.array([[0,0],[.5,.3],[1,1],[run.factor(p)*tau, .1]])
                values=values[values[:,0]<=1]
                c=run.Config('G_rescaled_p'+p,0,1,1,tau,p)
                _,keep=run.score_keep(values,c)
                _,direct=run.score_keep(values,run.replace(c,method='G_union',threshold=run.factor(p)*tau))
                np.testing.assert_array_equal(keep,direct)
                _,abl=run.score_keep(values,run.replace(c,method='GM_no_L'))
                np.testing.assert_array_equal(keep,abl)
        values=np.array([[.8,.2],[0,0],[.3,.9]])
        for p in run.POWERS:
            c=run.Config('GM_joint',0,1,1,.5,p)
            scores,_=run.score_keep(values,c)
            expected=np.max(values,1) if p=='inf' else np.mean(values**float(p),1)**(1/float(p))
            np.testing.assert_allclose(scores,expected)

    def test_incremental_assembly_reuses_endpoints_and_maps(self):
        old={m:(np.stack([oracle(run.Config(**run.asdict(c),p='1' if m=='Mean' else '8')) for c in cs]),)*2
             for m,cs in run.phase.grids().items()}
        old['G_rescaled_p8']=(old['G'][0]+np.array([0,3,0]),)*2
        calls=[]
        def search(context,core,cs,path):
            calls.append((path.name,cs))
            a=np.stack([oracle(c) for c in cs]); return a,a.copy()
        ctx=(None,None,None,None,None,None,['a','b'],None)
        with tempfile.TemporaryDirectory() as td,patch.object(run.phase,'search_counts',side_effect=search):
            tables=run.assemble_counts(ctx,SimpleNamespace(expected_gt=[2,2]),old,run.grids(),Path(td))
            self.assertEqual([len(cs) for _,cs in calls],[171,171,171,144])
            self.assertEqual([name for name,_ in calls],['new_counts_p2.npz','new_counts_p4.npz','new_counts_pinf.npz','new_counts_G_union.npz'])
            np.testing.assert_array_equal(tables['GM_joint'][1][:171],old['Mean'][1])
            np.testing.assert_array_equal(tables['GM_joint'][1][513:684],old['GM_p8'][1])
            np.testing.assert_array_equal(tables['G_rescaled_pinf'][1],old['G'][1])
            union=run.grids()['G_union']
            for p in run.POWERS:
                for c in run.grids()['G_rescaled_p'+p]:
                    ix=next(i for i,u in enumerate(union) if u.reference_scale==c.reference_scale and u.threshold==run.factor(p)*c.threshold)
                    np.testing.assert_array_equal(tables['G_rescaled_p'+p][1][c.config_id],tables['G_union'][1][ix])

    def test_joint_selection_excludes_outer_and_ties_follow_grid(self):
        cs=run.grids()['GM_joint']
        a=np.tile([1,9,1],(len(cs),3,1))
        a[171,1:]=[2,0,0]  # p=2 inner winner for outer 0
        a[342,0]=[2,0,0]   # p=4 wins only on held-out data
        w,_=run.shared.choose(a,0)
        self.assertEqual(cs[w].p,'2')
        a[:,0]=[999,0,0]
        self.assertEqual(run.shared.choose(a,0)[0],w)
        self.assertEqual(run.shared.choose(np.ones_like(a),0)[0],0)

    def test_event_relabel_equivalence_and_rejection(self):
        c=run.Config('GM_joint',0,2,1,.4,'1')
        rec=dict(subject='s',method='Mean',config=run.asdict(run.phase.Config('Mean',0,2,1,.4)),
                 candidates=[dict(G=[.9,.1],L=[.5,.2],score=[.7,.15],retained=[True,False])],
                 raw_counts=[1,0,0],full_counts=[1,0,0],events=[],ground_truth=[])
        r=run.relabel_record(rec,c)
        self.assertEqual(r['config']['p'],'1')
        self.assertEqual(rec['method'],'Mean')
        self.assertEqual(run.event_signature(c),run.event_signature(run.Config(**rec['config'])))
        with self.assertRaisesRegex(RuntimeError,'selection mismatch'):
            run.relabel_record(rec,run.replace(c,threshold=.9))
        for p in run.POWERS:
            a=run.Config('GM_no_L',0,1,1,.4,p)
            b=run.Config('G_union',0,1,1,run.factor(p)*.4,'inf')
            self.assertEqual(run.event_signature(a),run.event_signature(b))

    def test_new_score_paths_with_distinct_real_evaluators(self):
        from test_p8_matching import MeanAveragePrecision2d as Aggregate
        from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d as Video
        scope={'MeanAveragePrecision2d':Video,'np':np}
        exec('def spotting(aggregate, preds, gt):\n'
             '    video=MeanAveragePrecision2d(num_classes=1)\n'
             '    video.add(preds,gt)\n'
             '    aggregate.add(np.column_stack((preds,np.zeros(len(preds)))),gt)\n'
             '    return video\n',scope)
        official=SimpleNamespace(spotting=scope['spotting'])
        class Runner:
            def decode_glsd_subject(self,records,si,c,core,metric_class,official,recognition):
                peaks=core.GLSDFeatures(records[si]['result_all'][0],2).selected_peaks(c)
                preds=np.array([[int(i),0,int(i)+4,0,0,1,int(i)+2] for i in peaks],float).reshape(-1,7)
                gt=np.array([[1,0,5,0,0,0,0,3]],float)
                aggregate=metric_class(num_classes=1)
                video=official.spotting(aggregate,preds,gt)
                v=aggregate.value(iou_thresholds=.5)[.5][0]
                raw=(int(sum(v['tp'])),int(sum(v['fp'])),1-int(sum(v['tp'])))
                matches=video.value(iou_thresholds=.5)[.5][0]['pred_match_gt']
                targets=[1 if np.asarray(x).reshape(-1)[0]>=0 else -1 for x in matches.get(0,[])]
                return raw,[preds],[1]*len(preds),targets,matches,video
            def full_counts_from_official_synergy(self,raw,labels,targets): return raw
        rec=dict(result_all=[[1,2,3,4]],k_p=2,videos=['v'],samples=[[[1,3,5]]])
        ctx=(Runner(),BASE,Aggregate,official,[rec],[],['s'],None)
        core=run.FusionCore(BASE);core.expected_gt=[1]
        original=(Aggregate.add,Video.add)
        with run.evaluator.install(Aggregate):
            for p in run.POWERS:
                for method in ('GM_joint','GM_no_L','GM_no_G','G_rescaled_p'+p):
                    r=run.prior.detailed_decode(ctx,core,0,run.Config(method,0,1,1,.4,p))
                    self.assertEqual(sum(r['full_counts'][::2]),1)
                    self.assertEqual(len(r['ground_truth']),1)
        self.assertEqual((Aggregate.add,Video.add),original)

    def test_full_orchestration_and_selected_event_resume(self):
        subjects=['a','b']
        all_grids=run.grids()
        tables={m:(np.tile([1,0,1],(len(cs),2,1)),)*2 for m,cs in all_grids.items()}
        old={m:tables['Mean' if m=='GM_p8' else m] for m in run.phase.METHODS}
        old['G_rescaled_p8']=tables['G_rescaled_p8']
        runner=SimpleNamespace(final_samples=lambda records:[[[(0,1),(2,3)]]]*2)
        ctx=(runner,BASE,None,None,[],[],subjects,None)
        def detail(ctx,core,si,c):
            return dict(subject=subjects[si],method=c.method,config=run.asdict(c),
                raw_counts=[1,0,1],full_counts=[1,0,1],candidates=[],
                events=[dict(matched_gt_id=subjects[si]+'/v/0',full_TP=1)],ground_truth=[])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'old/synthetic_audit').mkdir(parents=True)
            (root/'old/synthetic_audit/native_per_subject_counts.csv').write_text('subject,evaluation,TP,FP,FN\na,full,1,0,1\nb,full,1,0,1\n')
            output=root/'out'
            with patch.object(run,'load_reuse',return_value=old),patch.object(run,'assemble_counts',return_value=tables), \
                 patch.object(run,'load_event_cache',return_value={}),patch.object(run,'run_probe'), \
                 patch.object(run.prior,'detailed_decode',side_effect=detail) as replay:
                run.run_setting(ctx,root/'old',root/'latest','synthetic',output,'full')
                self.assertGreater(replay.call_count,0)
                replay.reset_mock()
                run.run_setting(ctx,root/'old',root/'latest','synthetic',output,'full')
                replay.assert_not_called()
            d=json.loads((output/'decision.json').read_text())
            self.assertFalse(d['exceeds_independent_controls'])
            self.assertFalse(d['exceeds_both_locked_ablations'])
            with gzip.open(output/'event_records.jsonl.gz','rt') as f:
                records=[json.loads(s) for s in f]
            for subject in subjects:
                full=next(r for r in records if r['subject']==subject and r['method']=='GM_joint')['config']
                for m in ('GM_no_L','GM_no_G'):
                    c=next(r for r in records if r['subject']==subject and r['method']==m)['config']
                    self.assertEqual({k:v for k,v in c.items() if k!='method'},{k:v for k,v in full.items() if k!='method'})

    def test_reused_tensor_identity_rejects_wrong_subject(self):
        cs=[run.phase.Config('G',0,1,1,.5)]
        a=np.array([[[1,0,1],[0,0,2]]])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'counts.npz'
            run.save_counts(path,cs,(a,a),['a','b'])
            np.testing.assert_array_equal(run.load_counts(path,cs,['a','b'],[2,2])[0],a)
            with self.assertRaisesRegex(RuntimeError,'subject'):
                run.load_counts(path,cs,['b','a'],[2,2])
            with self.assertRaisesRegex(RuntimeError,'configuration'):
                run.load_counts(path,[run.replace(cs[0],threshold=.6)],['a','b'],[2,2])

if __name__=='__main__': unittest.main()
