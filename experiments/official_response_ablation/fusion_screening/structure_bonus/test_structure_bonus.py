import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run_structure_bonus as r

class Features:
    def __init__(self,response,k): pass
    def evidence(self,a0,rho):
        return np.array([1,4]),np.array([[.4+a0/10,.1+rho/10],[.7,.5]])
    def selected_peaks(self,c):
        peaks,values=self.evidence(c.reference_scale,c.local_radius)
        return peaks[values.mean(axis=1)>=c.threshold]

class Tests(unittest.TestCase):
    def test_grid_and_formula(self):
        grids=r.grids()
        self.assertEqual({k:len(v) for k,v in grids.items()},{'G':57,'L':171,'Mean':171,'SimpleBonus':171})
        g,l=np.meshgrid(np.linspace(0,1,31),np.linspace(0,1,31))
        g,l=g.ravel(),l.ravel()
        scores=r.scores(g,l,grids['SimpleBonus'][0])
        np.testing.assert_allclose(scores,g+l-g*l,atol=1e-15)
        self.assertTrue(np.all(scores>=np.maximum(g,l)))
        self.assertTrue(np.all(scores<=1))
    def test_structure_cache_and_mean_replay(self):
        core=r.FusionCore(SimpleNamespace(GLSDFeatures=Features))
        view=core.GLSDFeatures([1.,2.,3.],2)
        for method,grid in r.grids().items():
            for c in grid:
                peaks,values=view.evidence(c.reference_scale,c.local_radius)
                np.testing.assert_array_equal(view.selected_peaks(c),peaks[r.scores(*values.T,c)>=c.threshold])
        self.assertEqual(len(view.evidence_cache),9)
        self.assertIs(view,core.GLSDFeatures([1.,2.,3.],2))
    def test_outer_exclusion(self):
        table=np.array([[[0,0,0],[1,0,1]],[[0,0,0],[0,1,2]]])
        a,_=r.shared.choose(table,0)
        table[:,0,:]=[[0,10000,10000],[10000,0,0]]
        b,_=r.shared.choose(table,0)
        self.assertEqual(a,b)
    def test_synthetic_official_run(self):
        base=SimpleNamespace(GLSDFeatures=Features)
        def decode(records,i,c,core,metric,official,full):
            peaks=core.GLSDFeatures([.1,.2,.3],2).selected_peaks(c)
            tp=int(4 in peaks);fp=int(1 in peaks);counts=(tp,fp,1-tp)
            return counts,peaks.tolist(),[],[]
        runner=SimpleNamespace(decode_glsd_subject=decode,full_counts_from_official_synergy=lambda raw,p,g:raw)
        subjects=['s1','s2','s3']
        context=(runner,base,None,None,[],[],subjects,None)
        def locked(kind,context,core,chosen,subjects):
            return {s:(decode([],i,chosen[s],core,None,None,True)[0],)*2 for i,s in enumerate(subjects)}
        ora=SimpleNamespace(SPECS={'fake':{'videos':3,'gt':3,'source':Path(__file__),'core':Path(__file__)}},
            metst_context=lambda _:context,sha256=lambda _: 'test',
            context_video_and_gt_counts=lambda *args:(3,3),
            locked_grid=lambda _: [r.Config('Mean',i,a,rho,t) for i,(a,rho,t) in enumerate((a,rho,t) for a in r.SCALES for rho in r.RADII for t in r.LEGACY_THRESHOLDS)],
            locked_glsd_subject_counts=locked)
        with tempfile.TemporaryDirectory() as d,patch.object(r,'THRESHOLDS',(.4,.8)):
            out=Path(d)/'out'
            summary,paired=r.run_setting(ora,'fake',out)
            self.assertEqual(len(summary),4);self.assertEqual(len(paired),3)
            self.assertTrue((out/'completion.json').exists())

if __name__=='__main__':unittest.main()
