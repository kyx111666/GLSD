"""Changed diagnostic paths only; no synthetic scientific claims."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import mechanism_features as mf
import analyze_export
import run_mechanism as run
import sealed_core_test_fixture as sealed


class Tests(unittest.TestCase):
    def test_real_sealed_core_default_reconstruction_and_physical_dedup(self):
        self.assertEqual(hashlib.sha256(Path(sealed.__file__).read_bytes()).hexdigest(),
                         '6f500e5af179a28e8a4f5e49a926004c1a60887a3cc27b1e9df2b9d61c94ca3d')
        for k in (1,5,17):
            for response in (np.zeros(100),np.random.default_rng(5).random(100)):
                f=sealed.GLSDFeatures(response,k)
                for a in (1.,1.5,2.):
                    for r in mf.RADII:
                        p,e=f.evidence(a,r);v=mf.candidate_rows(f,a,r)
                        self.assertEqual(v['peaks'],p.tolist())
                        np.testing.assert_array_equal(v['L'],e[:,1])
                        np.testing.assert_allclose(v['scales']['0.5']['median'],e[:,1])
                        self.assertEqual(len(v['scales']['0.5']['widths']),len(f.effective_scales))
                        for lo,hi in zip(mf.GAMMAS,mf.GAMMAS[1:]):
                            self.assertTrue(np.all(np.asarray(v['scales'][str(hi)]['median'])>=np.asarray(v['scales'][str(lo)]['median'])))

    def test_alignment_missing_and_tie_policy(self):
        # distance takes precedence; equal distance chooses larger local evidence.
        class F:
            k=4
            effective_scales={4:(np.array([8,12,15]),None,None,None)}
            local_cache={(4,4):np.array([.2,.7,.9])}
            def evidence(self,a,r):return np.array([10,20]),np.array([[.1,.7],[.2,0.]])
        f=F();d=mf.scale_details(f,1,1,.5)
        self.assertEqual(d['aligned'],[[.7],[0.]])
        self.assertEqual(d['distance'],[[2],[-1]])
        self.assertEqual(d['nearest_distance'],[[2],[5]])
        self.assertEqual(mf.scale_details(f,1,1,1.)['matched'],[[True],[False]])

    def test_conditional_model_training_only_and_increment(self):
        rng=np.random.default_rng(100);n=600
        g=rng.uniform(.05,.95,n);l=rng.uniform(.3,.7,n);y=(g>.55).astype(int)
        s=np.array(['s'+str(i%6) for i in range(n)])
        x=mf.logit_scores(np.column_stack((l,g)));tr=np.arange(n)<500
        base,_=mf.fit_predict(x[tr,:1],y[tr],s[tr],x[~tr,:1])
        full,info=mf.fit_predict(x[tr],y[tr],s[tr],x[~tr])
        self.assertLess(mf.binary_logloss(y[~tr],full).mean(),mf.binary_logloss(y[~tr],base).mean())
        # Test values cannot change trained normalization/coefficients.
        _,info2=mf.fit_predict(x[tr],y[tr],s[tr],np.ones_like(x[~tr])*1000)
        self.assertEqual(info,info2)
        pred,_=mf.fit_predict(x[tr],np.zeros(tr.sum()),s[tr],x[~tr])
        self.assertTrue(np.isfinite(pred).all())

    def test_raw_geometry_uses_official_iou_and_all_ordered_candidates(self):
        from test_p8_matching import MeanAveragePrecision2d as Aggregate
        from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d as Video
        from test_p8_phase2 import BASE
        scope={'MeanAveragePrecision2d':Video}
        exec('def spotting():\n    return MeanAveragePrecision2d(num_classes=1)\n',scope)
        official=SimpleNamespace(spotting=scope['spotting'])
        class Runner:
            def decode_glsd_subject(self,records,si,c,core,metric,official,recognition):
                assert recognition is False
                peaks=core.GLSDFeatures(records[si]['result_all'][0],2).selected_peaks(c)
                preds=np.array([[p-2,0,p+2,0,0,0,p] for p in peaks],float).reshape(-1,7)
                gt=np.array([[-1,0,3,0,0,0,0,1]],float)
                a=metric(num_classes=1);v=official.spotting()
                a.add(np.column_stack((preds,np.zeros(len(preds)))),gt);v.add(preds,gt)
                vv=a.value(iou_thresholds=.5)[.5][0];tp=int(sum(vv['tp']));fp=int(sum(vv['fp']))
                return (tp,fp,1-tp),[preds],None,None,{},v
        ctx=(Runner(),BASE,Aggregate,official,[dict(result_all=[[1,2,3,4]],k_p=2,videos=['v'])],[],['s'],None)
        core=run.prior.FusionCore(BASE);core.expected_gt=[1]
        with run.prior.evaluator.install(Aggregate):geo=run.geometry_decode(ctx,core,0,1.)
        self.assertEqual(geo[0]['peaks'],[1,3]);self.assertEqual(geo[0]['eligible_gt_ids'],[['s/video_0/gt_0'],[]])
        self.assertEqual(geo[0]['covered_GT'],1)

    def test_analysis_all_fixed_structures_and_output_contract(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'subjects').mkdir()
            for si in range(3):
                feats=[]
                for a in (1.,1.5,2.):
                    for r in mf.RADII:
                        row=dict(a0=a,rho=r,video_id=str(si)+'/v',peaks=list(range(8)),
                            geometry_eligible=[0,0,0,0,1,1,1,1],G=[.1,.2,.3,.4,.6,.7,.8,.9],L=[.3,.4,.5,.6]*2)
                        row['scales']={str(g):dict(median=row['L'],mean=row['L'],matched_only_median=row['L'],
                            matched=[[True]*3]*8,nearest_distance=[[0]*3]*8,tolerance=2) for g in mf.GAMMAS}
                        feats.append(row)
                with gzip.open(out/'subjects'/('%03d.json.gz'%si),'wt') as f:
                    json.dump(dict(subject=str(si),features=feats,coverage=[dict(subject=str(si),a0=1,GT=4,geometry_covered_GT=4,candidates=8)]),f)
            summaries=analyze_export.analyze(out)
            self.assertEqual(len(summaries),12)
            self.assertTrue(all(r['delta_LG_minus_L']<0 for r in summaries))
            self.assertTrue((out/'radius_effects.csv').is_file())
            self.assertIn('NOT TP',json.loads((out/'analysis_notes.json').read_text())['label'])


if __name__=='__main__':unittest.main()
