"""Behavior, leakage, reuse and checkpoint tests for the frozen union protocol."""
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import reliable_features as f
import run_reliable_local_union as run
from test_eta_gm import Fixture as BaseFixture

class Fixture(BaseFixture):
    def selected_peaks(self,c):
        p,v=self.evidence(c.reference_scale,c.local_radius)
        return p[v.mean(1)>=c.threshold]
BASE=SimpleNamespace(GLSDFeatures=Fixture)

class Tests(unittest.TestCase):
    def test_frozen_budget_and_effective_local_thresholds(self):
        gs=f.grids()
        self.assertEqual({m:len(c) for m,c in gs.items()},dict(Full=513,G=57,L=351,L_rel=351,NoR=513,Mean=171,L_old=171))
        self.assertEqual(len(set(f.U)),39)
        self.assertGreater(max(f.U),1)
        self.assertEqual({c.threshold for c in gs['L']},{c.threshold/c.beta for c in gs['Full']})
        self.assertTrue(all(c.local_radius==1 for c in gs['G']))

    def test_reference_exclusion_missing_empty_and_zero_local_support(self):
        r=f.reliability([[0,2,4],[0,-1,0],[0,0,0]],0,10)
        np.testing.assert_allclose(r,[np.exp(-.3),0,1],atol=1e-15)
        np.testing.assert_array_equal(f.reliability([[0],[0]],0,1),[0,0])
        np.testing.assert_allclose(f.reliability([[2,0,4]],1,10),[np.exp(-.3)])
        self.assertEqual(len(f.reliability(np.empty((0,3)),0,2)),0)

    def test_G_survives_missing_and_local_only_supplement_no_clipping(self):
        v=np.array([[.8,.9],[.1,.9],[.1,.9],[0.,0.]])
        r=np.array([0.,1.,0.,1.])
        c=f.Config('Full',0,2,2,.7,2)
        scores,keep=f.score_keep(v,r,c)
        np.testing.assert_array_equal(keep,[True,True,False,False])
        self.assertEqual(scores[1],1.8)
        for beta in f.BETAS:
            for tau in f.T:
                c=replace(c,beta=beta,threshold=float(tau))
                _,k=f.score_keep(v,r,c)
                np.testing.assert_array_equal(k,(v[:,0]>=c.threshold)|(r*v[:,1]>=c.threshold/beta))
                np.testing.assert_array_equal(f.score_keep(v,r,replace(c,method='Full_no_G'))[1],r*v[:,1]>=c.threshold/beta)
        with self.assertRaises(ValueError):f.score_keep(v,r,replace(c,beta=0))

    def test_feature_exact_and_missing_does_not_veto_G(self):
        core=f.FusionCore(BASE);view=core.GLSDFeatures(np.arange(16),5)
        for a in f.phase.SCALES:
            for rho in f.phase.RADII:
                d=view.reliable_evidence(a,rho);p,v=view.base.evidence(a,rho)
                np.testing.assert_array_equal(d['peaks'],p);np.testing.assert_array_equal(d['values'],v)
        d=view.reliable_evidence(1,2);self.assertEqual(d['R'][0],0)
        np.testing.assert_array_equal(view.selected_peaks(f.Config('Full',0,1,2,.7)),[4])

    def test_nearest_peak_tie_prefers_higher_local_and_no_extra_support_cutoff(self):
        core=f.FusionCore(BASE);v=core.GLSDFeatures(np.arange(16),5);b=v.base
        # Existing cache values are intentionally all below any selection threshold.
        b.scale_data[1.5]=(np.array([3,5]),np.zeros(16),1.,np.array([0.,0.]))
        b.effective_scales={0:b.scale_data[1.],1:b.scale_data[1.5],2:b.scale_data[2.]}
        b.local_cache={(0,10):np.array([.2]),(1,10):np.array([.01,.02]),(2,10):np.array([0.])}
        d=v.reliable_evidence(1,2)
        self.assertEqual(d['matched_peaks'][0,1],5)
        self.assertAlmostEqual(d['R'][0],np.exp(-.1))
        self.assertGreater(d['R'][0],0)  # zero local score is still a match

    def test_deduplicated_physical_scales_and_degenerate_ratio(self):
        core=f.FusionCore(BASE);v=core.GLSDFeatures(np.arange(16),5);b=v.base
        one=b.scale_data[1.]
        b.scale_data={a:one for a in f.phase.SCALES};b.effective_scales={5:one}
        d=v.reliable_evidence(2,2)
        self.assertTrue(d['B_empty']);self.assertEqual(d['widths'],[5]);self.assertEqual(d['R'][0],0)
        self.assertGreater(d['values'][0,1],0)

    def test_incremental_search_never_repeats_old_configs(self):
        old={m:(np.tile([1,0,1],(len(cs),2,1)),)*2 for m,cs in run.eta.grids().items() if m in ('G','L','Mean','GM_eta','G_union',*('G_rescaled_p'+p for p in run.joint.POWERS))}
        calls=[]
        def search(ctx,core,cs,path):
            calls.extend(cs);a=np.tile([1,3,1],(len(cs),2,1));return a,a.copy()
        ctx=(None,None,None,None,None,None,['a','b'],None)
        with tempfile.TemporaryDirectory() as td,patch.object(run.phase,'search_counts',side_effect=search):
            tables=run.assemble(ctx,SimpleNamespace(expected_gt=[2,2]),old,Path(td))
        self.assertEqual(len(calls),1386)
        self.assertEqual(set(c.method for c in calls),{'Full','L','L_rel','NoR'})
        self.assertTrue(all(c.beta!=1 for c in calls if c.method=='NoR'))
        self.assertTrue(all(c.threshold not in f.phase.TAUS for c in calls if c.method=='L'))
        np.testing.assert_array_equal(tables['G'][1],old['G'][1])

    def test_selection_excludes_outer_and_stable_ties(self):
        a=np.array([[[0,200,2],[1,0,1],[1,0,1]],[[2,0,0],[0,0,2],[0,0,2]]])
        self.assertEqual(run.shared.choose(a,0)[0],0)
        a[:,0]=[999,0,0];self.assertEqual(run.shared.choose(a,0)[0],0)
        self.assertEqual(run.shared.choose(np.tile([1,2,1],(2,3,1)),0)[0],0)

    def test_resume_rejects_each_identity_axis_and_changed_grid(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out';identity=dict(code='a',inputs='b',runtime='c',reuse='d',evaluation='e',mode='probe')
            self.assertFalse(run.check_resume(out,identity,False))
            self.assertFalse(run.check_resume(out,identity,True))
            for key in identity:
                with self.assertRaisesRegex(RuntimeError,'identical'):run.check_resume(out,dict(identity,**{key:'changed'}),True)
            c=f.Config('Full',0,1,1,.5)
            run.joint.save_counts(out/'counts.npz',[c],(np.tile([1,0,1],(1,2,1)),)*2,['a','b'])
            ctx=(None,None,None,None,None,None,['a','b'],None)
            with self.assertRaisesRegex(RuntimeError,'configs/subjects'):
                run.phase.search_counts(ctx,None,[replace(c,beta=2)],out/'counts.npz')

    def test_old_reuse_rejects_runtime_input_code_inventory_and_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            runtime={'python':'original','numpy':'original','pandas':'original'};inputs={'dump':'hash'}
            identity=dict(protocol=run.eta.PROTOCOL,evaluation=run.evaluator.PROTOCOL,setting='metst_sammlv',mode='full',
                          runtime=runtime,inputs=inputs,code=json.loads((run.ROOT/'inherited_manifest.json').read_text()))
            done=dict(completed=True,mode='full',setting='metst_sammlv',protocol=run.eta.PROTOCOL)
            run.atomic_json(root/'run_manifest.json',dict(identity=identity));run.atomic_json(root/'completion.json',done)
            run.verify_reuse(root,'metst_sammlv',runtime,inputs)
            with self.assertRaisesRegex(RuntimeError,'runtime changed'):run.verify_reuse(root,'metst_sammlv',dict(runtime,numpy='new'),inputs)
            with self.assertRaisesRegex(RuntimeError,'input inventory'):run.verify_reuse(root,'metst_sammlv',runtime,{'other':'hash'})
            identity['code'].pop(next(iter(identity['code'])))
            run.atomic_json(root/'run_manifest.json',dict(identity=identity))
            with self.assertRaisesRegex(RuntimeError,'inventory'):run.verify_reuse(root,'metst_sammlv',runtime,inputs)
            run.atomic_json(root/'completion.json',dict(done,completed=False))
            with self.assertRaisesRegex(RuntimeError,'completed ETA'):run.verify_reuse(root,'metst_sammlv',runtime,inputs)

    def test_completion_marker_and_float_threshold_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';ident=dict(protocol=run.PROTOCOL,mode='full',setting='metst_sammlv')
            run.check_resume(out,ident,False)
            run.atomic_json(out/'completion.json',dict(ident,completed=True))
            self.assertTrue(run.check_resume(out,ident,True))
            run.atomic_json(out/'completion.json',dict(ident,completed=True,mode='probe'))
            with self.assertRaisesRegex(RuntimeError,'completion'):run.check_resume(out,ident,True)
        for beta in f.BETAS:
            for tau in f.phase.TAUS:
                threshold=tau/beta
                local=np.array([np.nextafter(threshold,0),threshold,np.nextafter(threshold,np.inf)])
                local=local[local<=1]
                values=np.column_stack((np.zeros(len(local)),local))
                c=f.Config('Full',0,1,1,tau,beta)
                np.testing.assert_array_equal(f.score_keep(values,np.ones(len(local)),c)[1],local>=threshold)

    def test_ledger_rejects_duplicate_GT(self):
        r=dict(ground_truth=[dict(gt_id='g')],events=[dict(matched_gt_id='g',raw_TP=1,raw_FP=0,full_TP=1,full_FP=0)]*2,raw_counts=[2,0,-1],full_counts=[2,0,-1])
        with self.assertRaises(RuntimeError):run.eta.validate_record(r)

    def test_dual_observers_restore_even_on_error(self):
        class A:
            def add(self,p,g):return None
        class V:
            def add(self,p,g):return None
        ns={'MeanAveragePrecision2d':V};exec('def spotting(): pass',ns)
        official=SimpleNamespace(spotting=ns['spotting']);old=A.add,V.add
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with run.prior.capture_decoder_inputs(A,official) as cap:
                A().add([1],[2]);V().add([1],[2]);self.assertEqual([len(x) for x in cap],[1,1]);raise RuntimeError('injected')
        self.assertEqual((A.add,V.add),old)

    def test_full_exports_lock_params_resume_and_FP_identity(self):
        subjects=['a','b'];gs={m:[replace(cs[0],threshold=.55,beta=2 if m=='Full' else cs[0].beta)] for m,cs in f.grids().items()}
        tables={m:(np.tile([1,1,1],(1,2,1)),)*2 for m in gs}
        def detail(ctx,core,si,c):
            vid=subjects[si]+'/video_0'
            events=[dict(video_id=vid,interval=[1,0,3,0,0,1,2],peak=2,matched_gt_id=vid+'/gt_0',raw_TP=1,raw_FP=0,full_TP=1,full_FP=0),
                    dict(video_id=vid,interval=[8,0,10,0,0,1,9],peak=9,matched_gt_id=None,raw_TP=0,raw_FP=1,full_TP=0,full_FP=1)]
            return dict(subject=subjects[si],method=c.method,config=asdict(c),raw_counts=[1,1,1],full_counts=[1,1,1],events=events,ground_truth=[dict(gt_id=vid+'/gt_0'),dict(gt_id=vid+'/gt_1')],candidates=[])
        ctx=(None,None,None,None,None,None,subjects,None)
        with tempfile.TemporaryDirectory() as td,patch.object(run,'all_grids',return_value=gs),patch.object(run,'detailed_decode',side_effect=detail) as decode:
            out=Path(td);run.selected_results(ctx,None,tables,out)
            self.assertEqual(decode.call_count,20);decode.reset_mock()
            run.selected_results(ctx,None,tables,out);decode.assert_not_called()
            with gzip.open(out/'event_records.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
            for s in subjects:
                full=next(r['config'] for r in rows if r['subject']==s and r['method']=='Full')
                for m in f.LOCKED:
                    c=next(r['config'] for r in rows if r['subject']==s and r['method']==m)
                    self.assertEqual({k:v for k,v in c.items() if k!='method'},{k:v for k,v in full.items() if k!='method'})
            a=rows[0];b=json.loads(json.dumps(a));b['events'][1]['interval'][0]=7
            d=run.event_delta(a,b)
            self.assertEqual(len(d['added_FP_events']),1);self.assertEqual(len(d['removed_FP_events']),1);self.assertEqual(d['delta_FP'],0)

if __name__=='__main__':unittest.main()
