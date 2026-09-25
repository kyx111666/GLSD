import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run_agreement_bonus as r

class Tests(unittest.TestCase):
    def test_bounds_and_endpoints(self):
        g,l=np.meshgrid(np.linspace(0,1,101),np.linspace(0,1,101))
        for lam in r.LAMBDAS:
            for gamma in r.GAMMAS:
                s=r.scores(g,l,r.Config('Fusion',0,.5,lam,gamma))
                self.assertTrue(np.all(s>=np.maximum(g,l)))
                self.assertTrue(np.all(s<=1+1e-12))
                np.testing.assert_allclose(s,r.scores(l,g,r.Config('Fusion',0,.5,lam,gamma)))
        np.testing.assert_array_equal(r.scores(g,l,r.Config('Fusion',0,.5)),np.maximum(g,l))
        self.assertEqual(r.scores(np.array([1.]),np.array([0.]),r.Config('Fusion',0,.5,1.,0.))[0],1.)
        self.assertAlmostEqual(r.scores(np.array([.5]),np.array([.5]),r.Config('Fusion',0,.5,1.,2.))[0],.75)
    def test_held_subject_cannot_change_selection(self):
        table=np.array([[[0,0,0],[1,0,1],[1,0,1]],[[0,0,0],[0,1,2],[0,1,2]]])
        first,_=r.shared.choose(table,0)
        table[:,0,:]=[[0,100000,100000],[100000,0,0]]
        second,_=r.shared.choose(table,0)
        self.assertEqual(first,0); self.assertEqual(first,second)
    def test_grid_and_invalid_scores(self):
        configs,subsets=r.grids()
        self.assertEqual(len(configs),456)
        self.assertEqual(len(subsets['Full_joint']),399)
        self.assertEqual(len(subsets['Bonus_joint']),114)
        self.assertEqual(len(subsets['Max']),19)
        with self.assertRaises(ValueError): r.scores([1.1],[.2],configs[0])
        with self.assertRaises(ValueError): r.scores([np.nan],[.2],configs[0])
    def test_fake_official_adapter_end_to_end(self):
        class BaseFeatures:
            def evidence(self,a0,rho):
                return np.array([1,3]),np.array([[.8,.2],[.6,.6]])
            def selected_peaks(self,c):
                peaks,v=self.evidence(c.reference_scale,c.local_radius)
                return peaks[np.mean(v,axis=1)>=c.threshold]
        base=SimpleNamespace(GLSDFeatures=lambda response,k:BaseFeatures())
        context=(None,base,None,None,[],[],['s1','s2','s3'],None)
        ora=SimpleNamespace(SPECS={'fake':{}},metst_context=lambda _:context,sha256=lambda _: '')
        def decode(context,core,i,c,full,capture=False):
            core.trace=[];core.capture=capture
            peaks=core.GLSDFeatures([.1,.2,.3],2).selected_peaks(c)
            tp=int(3 in peaks);fp=int(1 in peaks); counts=(tp,fp,1-tp)
            return counts,counts if full else None,peaks.tolist(),list(core.trace)
        with tempfile.TemporaryDirectory() as d,patch.object(r.shared,'decode',decode),patch.object(r,'TAUS',(.4,.7)):
            out=Path(d)/'result'
            rows=r.run_setting(ora,'fake',out)
            self.assertTrue((out/'completion.json').exists())
            self.assertTrue((out/'fusion_heatmap.csv').exists())
            self.assertEqual(len([x for x in rows if x['variant']=='Full_joint']),1)

if __name__=='__main__': unittest.main()
