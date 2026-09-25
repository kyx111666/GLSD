"""Synthetic tests; fixture evidence is copied verbatim from the local sealed-style core."""
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import run_eta_gm as run

class Fixture:
    def __init__(self,response,k):
        self.k=int(k)
        smooth=np.asarray([0,0,.2,.5,1,.7,.3,0,0,0,0,0,0,0,0,0],float)
        self.scale_data={1.:(np.array([4]),smooth,1.,np.array([.8])),
                         1.5:(np.array([7]),np.roll(smooth,3),1.,np.array([.8])),
                         2.:(np.array([4]),smooth,1.,np.array([.8]))}
        self.effective_scales={i:v for i,v in enumerate(self.scale_data.values())}
        self.local_cache={}
    def evidence(self, reference: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
        peaks, _, _, global_values = self.scale_data[reference]
        window = max(1, round(radius * self.k))
        tolerance = max(1, round(0.5 * self.k))
        for width, (other_peaks, smooth, spread, _) in self.effective_scales.items():
            key = width, window
            if key not in self.local_cache:
                values = []
                for peak in other_peaks:
                    left = float(np.min(smooth[max(0, peak - window):peak + 1]))
                    right = float(np.min(smooth[peak:min(len(smooth), peak + window + 1)]))
                    values.append(max(0.0, float(smooth[peak]) - max(left, right)) / spread)
                self.local_cache[key] = np.asarray(values, dtype=float)
        local_values = []
        for peak in peaks:
            aligned = []
            for width, (other_peaks, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                candidates = np.flatnonzero(np.abs(other_peaks - peak) <= tolerance)
                if not len(candidates):
                    aligned.append(0.0)  # locked missing-scale policy
                    continue
                chosen = min(candidates, key=lambda index: (abs(int(other_peaks[index]) - int(peak)), -values[index]))
                aligned.append(float(values[chosen]))
            local_values.append(float(np.median(aligned)))  # locked median aggregation
        evidence = np.column_stack((global_values, np.asarray(local_values, dtype=float)))
        return np.asarray(peaks, dtype=int), evidence


BASE=SimpleNamespace(GLSDFeatures=Fixture)

class Tests(unittest.TestCase):
    def test_exact_baseline_and_only_tolerance_changes(self):
        core=run.FusionCore(BASE);f=core.GLSDFeatures(np.arange(16),5)
        for a in run.phase.SCALES:
            for rho in run.phase.RADII:
                oldp,oldv=f.base.evidence(a,rho)
                p,v=f.evidence(a,rho,.5)
                np.testing.assert_array_equal(p,oldp);np.testing.assert_array_equal(v,oldv)
                for eta in run.ETAS:
                    q,w=f.evidence(a,rho,eta)
                    np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(v[:,0],w[:,0])
        narrow=f.evidence(1.5,2.,.25)[1][0,1]
        wide=f.evidence(1.5,2.,.75)[1][0,1]
        self.assertGreater(wide,narrow)
        self.assertFalse(hasattr(f.base,'_experiment_eta'))

    def test_refuses_unrecognized_tolerance_source(self):
        class Wrong:
            def evidence(self,a,r):
                tolerance=3
                return tolerance
        with self.assertRaisesRegex(RuntimeError,'tolerance expression'):
            run.alignment_method(Wrong)

    def test_fair_grids_and_ablations(self):
        gs=run.grids()
        self.assertEqual(len(gs['GM_eta']),2565)
        self.assertEqual(len(gs['L_eta']),513)
        self.assertEqual(len(gs['Mean_eta']),513)
        self.assertTrue(all(c.eta==.5 for c in gs['GM_eta'][:855]))
        values=np.array([[.8,.3],[.4,.9],[0,0]])
        for eta in run.ETAS:
            for power in run.joint.POWERS:
                c=run.Config('GM_eta',0,1,2,.5,power,eta)
                _,keep=run.score_keep(values,c)
                for method,index in [('GM_eta_no_L',0),('GM_eta_no_G',1)]:
                    scores,k=run.score_keep(values,run.replace(c,method=method))
                    np.testing.assert_array_equal(k,values[:,index]>=run.joint.factor(power)*.5)
                np.testing.assert_allclose(run.score_keep(values,c)[0],run.shared.generalized_mean(values[:,0],values[:,1],float(power)))

    def test_incremental_assembly_reuses_half_and_mean(self):
        old={m:(np.tile([1,0,1],(len(cs),2,1)),)*2 for m,cs in run.joint.grids().items()}
        calls=[]
        def search(context,core,cs,path):
            calls.append((path.name,cs))
            a=np.tile([1,3,1],(len(cs),2,1));return a,a.copy()
        context=(None,None,None,None,None,None,['a','b'],None)
        with tempfile.TemporaryDirectory() as td,patch.object(run.phase,'search_counts',side_effect=search):
            tables=run.assemble(context,SimpleNamespace(expected_gt=np.array([2,2])),old.copy(),Path(td))
        self.assertEqual(len(calls),12)
        self.assertEqual(sum(len(cs) for _,cs in calls),2052)
        self.assertTrue(all(all(c.eta!=.5 for c in cs) for _,cs in calls))
        np.testing.assert_array_equal(tables['GM_eta'][1][:855],old['GM_joint'][1])
        np.testing.assert_array_equal(tables['Mean_eta'][1][171:342],tables['GM_eta'][1][855:1026])

    def test_selection_excludes_outer_and_eta_checkpoint_identity(self):
        cs=run.grids()['GM_eta'];a=np.tile([1,10,1],(len(cs),3,1))
        a[855,1:]=[2,0,0];w,_=run.shared.choose(a,0);self.assertEqual(cs[w].eta,.25)
        a[:,0]=[100,0,0];self.assertEqual(run.shared.choose(a,0)[0],w)
        ctx=(None,None,None,None,None,None,['a','b'],None)
        short=[run.replace(cs[855],config_id=0)]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'counts.npz'
            run.joint.save_counts(path,short,(np.array([[[1,0,1],[1,0,1]]]),)*2,['a','b'])
            with self.assertRaisesRegex(RuntimeError,'configs/subjects'):
                run.phase.search_counts(ctx,None,[run.replace(short[0],eta=.75)],path)

    def test_dual_evaluator_events_with_new_eta(self):
        from test_p8_matching import MeanAveragePrecision2d as Aggregate
        from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d as Video
        scope={'MeanAveragePrecision2d':Video,'np':np}
        exec('def spotting(agg,preds,gt):\n    v=MeanAveragePrecision2d(num_classes=1)\n    v.add(preds,gt)\n    agg.add(np.column_stack((preds,np.zeros(len(preds)))),gt)\n    return v\n',scope)
        official=SimpleNamespace(spotting=scope['spotting'])
        class Runner:
            def decode_glsd_subject(self,records,si,c,core,metric_class,official,recognition):
                peaks=core.GLSDFeatures(records[si]['result_all'][0],5).selected_peaks(c)
                preds=np.array([[int(i)-2,0,int(i)+2,0,0,1,int(i)] for i in peaks],float).reshape(-1,7)
                gt=np.array([[5,0,9,0,0,0,0,7]],float)
                agg=metric_class(num_classes=1);v=official.spotting(agg,preds,gt)
                result=agg.value(iou_thresholds=.5)[.5][0]
                raw=(int(sum(result['tp'])),int(sum(result['fp'])),1-int(sum(result['tp'])))
                matches=v.value(iou_thresholds=.5)[.5][0]['pred_match_gt']
                labels=[1 if np.asarray(x).reshape(-1)[0]>=0 else -1 for x in matches.get(0,[])]
                return raw,[preds],[1]*len(preds),labels,matches,v
            def full_counts_from_official_synergy(self,raw,labels,targets):return raw
        rec=dict(result_all=[np.arange(16)],k_p=5,videos=['v'],samples=[[[5,7,9]]])
        ctx=(Runner(),BASE,Aggregate,official,[rec],[],['s'],None)
        core=run.FusionCore(BASE);core.expected_gt=[1]
        original=(Aggregate.add,Video.add)
        with run.evaluator.install(Aggregate):
            for eta in run.ETAS:
                for method in ('GM_eta','L_eta','GM_eta_no_L','GM_eta_no_G'):
                    r=run.prior.detailed_decode(ctx,core,0,run.Config(method,0,1.5,2.,.3,'8',eta))
                    run.validate_record(r)
                    self.assertEqual(r['candidates'][0]['tolerance'],max(1,round(eta*5)))
        self.assertEqual((Aggregate.add,Video.add),original)

    def test_full_export_and_resume(self):
        gs=run.grids();subs=['a','b']
        tables={m:(np.tile([1,0,1],(len(cs),2,1)),)*2 for m,cs in gs.items()}
        runner=SimpleNamespace(final_samples=lambda records:[[[(0,1),(2,3)]]]*2)
        ctx=(runner,BASE,None,None,[],[],subs,None)
        def detail(ctx,core,si,c):
            return dict(subject=subs[si],method=c.method,config=run.asdict(c),raw_counts=[1,0,1],full_counts=[1,0,1],candidates=[],
                events=[dict(matched_gt_id='g0',raw_TP=1,raw_FP=0,full_TP=1,full_FP=0)],ground_truth=[dict(gt_id='g0'),dict(gt_id='g1')])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=root/'old'/'synthetic';old.mkdir(parents=True)
            run.shared.write_csv(old/'summary_full.csv',[dict(setting='synthetic',method='Native',selection='none',configs=0,**run.shared.metrics([2,0,2]))])
            with patch.object(run,'load_reuse',return_value=tables.copy()),patch.object(run,'assemble',return_value=tables), \
                 patch.object(run,'sensitivity'),patch.object(run,'probe'),patch.object(run.joint,'load_event_cache',return_value={}), \
                 patch.object(run.prior,'detailed_decode',side_effect=detail) as decode:
                run.run_setting(ctx,root/'old','synthetic',root/'out','full')
                self.assertGreater(decode.call_count,0);decode.reset_mock()
                run.run_setting(ctx,root/'old','synthetic',root/'out','full');decode.assert_not_called()
            d=json.loads((root/'out'/'decision.json').read_text());self.assertFalse(d['exceeds_independent_controls'])
            with gzip.open(root/'out'/'event_records.jsonl.gz','rt') as f:records=[json.loads(x) for x in f]
            for subject in subs:
                full=next(r for r in records if r['subject']==subject and r['method']=='GM_eta')['config']
                for m in ('GM_eta_no_L','GM_eta_no_G'):
                    c=next(r for r in records if r['subject']==subject and r['method']==m)['config']
                    self.assertEqual({k:v for k,v in full.items() if k!='method'},{k:v for k,v in c.items() if k!='method'})

if __name__=='__main__':unittest.main()
