"""Algorithm edge cases and experiment-selection invariants, no efficacy claims."""
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_shift as run
import shift_features as sf


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = run.runtime.load_module(run.VENDOR/'sealed/boosting_official_glds_full_LOCKED.py', 'test_core')
        cls.base.SCALES = sf.SCALES

    def feature(self, n=300, k=5):
        t = np.arange(n)
        x = np.sin(t*.17)+.6*np.sin(t*.43)+.002*t
        return self.base.GLSDFeatures(x, k)

    def test_dense_distance_ties_missing_and_order(self):
        a = sf.dense_support(np.array([2,6]), np.array([.3,.8]), 10, 2)
        self.assertEqual(a[4], .8)
        self.assertEqual(a[3], .3)
        self.assertEqual(a[9], 0.)
        a = sf.dense_support(np.array([6,2]), np.array([.4,.4]), 10, 2)
        self.assertEqual(a[4], .4)

    def test_exact_controls_against_scalar_oracle(self):
        base = self.feature()
        data = sf.compute(base, chunk_size=7)
        p, gl = base.evidence(2,3)
        widths, q = data['physical_widths'], data['protection_radius']
        ref = widths.index(data['reference_width'])
        for i, peak in enumerate(p):
            vals = []
            for t in range(q,len(base.response)-q):
                if abs(t-peak) <= 2*q:
                    continue
                votes = [data['L0'][i]]
                for j,width in enumerate(widths):
                    if j==ref: continue
                    peaks = base.effective_scales[width][0]
                    local = base.local_cache[width,data['window']]
                    ids = [j for j,p in enumerate(peaks) if abs(p-t)<=data['tolerance']]
                    best = min(ids,key=lambda j:(abs(peaks[j]-t),-local[j])) if ids else None
                    votes.append(0. if best is None else local[best])
                vals.append(np.median(votes))
            self.assertEqual(len(vals),data['control_count'][i])
            self.assertAlmostEqual(data['B'][i], np.mean(vals) if vals else gl[i,1], places=14)
        np.testing.assert_array_equal(data['peaks'],p)
        np.testing.assert_array_equal(data['L'],gl[:,1])

    def test_chunk_invariance(self):
        base=self.feature()
        a,b=sf.compute(base,chunk_size=1),sf.compute(base,chunk_size=10000)
        np.testing.assert_allclose(a['B'],b['B'],rtol=0,atol=1e-14)

    def test_single_physical_scale_exact_G(self):
        with patch.object(self.base,'SCALES',(1.,1.01)):
            base=self.feature(k=1)
            d=sf.compute(base,reference=1.)
        self.assertEqual(len(d['physical_widths']),1)
        self.assertTrue(all(r=='single_physical_scale' for r in d['abstention']))
        np.testing.assert_array_equal(d['B'],d['L'])
        np.testing.assert_array_equal(d['S'],d['G'])
        self.assertTrue(all(x==0 for x in d['D']))

    def test_physical_deduplication(self):
        base=self.feature(k=1)
        d=sf.compute(base)
        self.assertEqual(d['physical_widths'],[1,2,3,4])

    def test_empty_controls_and_no_candidates(self):
        d=sf.compute(self.feature(n=40))
        self.assertTrue(all(x==0 for x in d['control_count']))
        np.testing.assert_array_equal(d['S'],d['G'])
        d=sf.compute(self.base.GLSDFeatures(np.ones(100),5))
        self.assertEqual(d['peaks'],[])
        for m in sf.METHODS:self.assertEqual(sf.score(d,m).shape,(0,))

    def test_invalid_response_stops(self):
        with self.assertRaises(ValueError):self.base.GLSDFeatures([1,np.nan,3],5)

    def test_scores_ablations_and_endpoints(self):
        d=dict(G=[0.,1.,.3],L=[0.,1.,.4],L0=[0.,0.,.5],B=[.5,0.,.9],S=[-.5,2.,-.2],video_mean_B=1.4/3)
        self.assertEqual(len(sf.TAUS),301)
        np.testing.assert_array_equal(sf.score(d,'G'),d['G'])
        np.testing.assert_allclose(sf.score(d,'D'),[-.5,1.,-.5])
        self.assertEqual((sf.score(d,'Full')>=2).tolist(),[False,True,False])
        self.assertEqual((sf.score(d,'Full')>=-.5).tolist(),[True,True,True])
        self.assertEqual((sf.score(d,'Full')>=1.01).tolist(),[False,True,False])
        np.testing.assert_allclose(sf.score(d,'RefResidualUnclipped'),[0.,2.,.2],rtol=0,atol=1e-15)
        self.assertTrue(np.all(sf.score(d,'G_plus_L')/2== (np.array(d['G'])+d['L'])/2))

    def test_candidate_order(self):
        core=sf.FusionCore(self.base)
        f=core.GLSDFeatures(self.feature().response,5)
        d=f.shift_evidence()
        for method in sf.METHODS:
            cfg=run.config(method,165)
            expected=np.asarray(d['peaks'])[sf.score(d,method)>=cfg.threshold]
            np.testing.assert_array_equal(f.selected_peaks(cfg),expected)


class ProtocolTests(unittest.TestCase):
    def test_outer_exclusion_and_ties(self):
        counts=np.array([[[5,0,0],[1,0,1],[1,0,1]],[[0,50,5],[2,3,0],[2,3,0]]])
        a,_=run.engine.shared.choose(counts,0)
        counts[:,0]=[[0,999,5],[5,0,0]]
        b,_=run.engine.shared.choose(counts,0)
        self.assertEqual(a,b)
        self.assertEqual(a,0)
        self.assertEqual(run.engine.shared.choose(np.tile(counts[:1],(2,1,1)),0)[0],0)
        precision_tie=np.array([[[0,0,10],[4,10,6]],[[0,0,10],[2,0,8]]])
        self.assertEqual(run.engine.shared.choose(precision_tie,0)[0],1)
        zero_tp=np.array([[[0,0,10],[0,2,10]],[[0,0,10],[0,1,10]]])
        self.assertEqual(run.engine.shared.choose(zero_tp,0)[0],1)

    def test_cas_exact_exception_only(self):
        r=dict(subject_id='216', videos=['e'],samples=[[[462,470,492]]],emotions=[['others','surprise']])
        self.assertIsNotNone(run.annotation_exception(r,0,'casme3'))
        with self.assertRaises(RuntimeError):run.annotation_exception(r,0,'sammlv')
        r['emotions']=[['others','anger']]
        with self.assertRaises(RuntimeError):run.annotation_exception(r,0,'casme3')

    def test_mask_cache_and_atomic_resume(self):
        class Core:
            expected_gt=[2,3]
            cache={}
        context=(None,None,None,None,None,None,['s0','s1'],None)
        calls=[]
        def decode(c,core,si,cfg):
            calls.append((si,cfg.threshold))
            tp=int(cfg.threshold<.5)
            return dict(raw_counts=[tp,1,core.expected_gt[si]-tp],full_counts=[tp,0,core.expected_gt[si]-tp])
        with tempfile.TemporaryDirectory() as td, patch.object(run,'mask_key',side_effect=lambda c,o,s,cfg:(cfg.threshold<.5,)), patch.object(run.engine,'decode',side_effect=decode):
            path=Path(td)
            run.search(context,Core(),path,indices=[0])
            self.assertEqual(len(calls),2)
            raw,full,done=run.search(context,Core(),path)
            self.assertEqual(len(calls),4)
            self.assertTrue(done.all())
            run.search(context,Core(),path)
            self.assertEqual(len(calls),4)
            with np.load(path/'search_counts.npz') as z:data={k:z[k].copy() for k in z.files}
            data['full'][0,0,0]=[0,0,2]
            np.savez_compressed(path/'search_counts.npz',**data)
            with self.assertRaisesRegex(RuntimeError,'digest'):run.search(context,Core(),path)


if __name__=='__main__':unittest.main(verbosity=2)
