"""Unit checks independent of held-out experimental results."""
import unittest
import numpy as np
from run_first3_nested import (METHODS,grid,ranking,features,common_pool,union,
    label_training,model_fit,model_predict,robust_z,expit,peak_prominences,find_peaks)
from run_morphology_audit import extract_patch,representations


class First3Tests(unittest.TestCase):
    def test_fixed_grids(self):
        self.assertEqual([len(grid(m)) for m in METHODS],[4,60,100,288])
        for m in METHODS:
            self.assertEqual(len(grid(m)),len({str(c) for c in grid(m)}))

    def test_tie_order(self):
        counts={'TP':2,'FP':8,'FN':8,'rescued_candidate_count':4}
        a=dict(Delta_p=.05,gamma=.4,alpha=.25)
        b=dict(Delta_p=.1,gamma=.7,alpha=.25)
        self.assertLess(ranking(counts,a),ranking(counts,b))
        self.assertLess(ranking(counts,{**a,'gamma':.7}),ranking(counts,a))

    def test_patch_scaling(self):
        curve=np.array([.1,.3,.7,.9,.5,.4,.2])
        x=features(curve,3,.44,1.,'peak-relative')
        expected=representations(extract_patch(curve,3,5))['relative_l2']
        np.testing.assert_array_equal(x[1:],expected)
        self.assertEqual(x[0],.44)
        self.assertEqual(len(x),32)
        z=features(curve,0,.2,.5,'raw-z')
        self.assertAlmostEqual(z[1:].mean(),0)
        np.testing.assert_array_equal(features(np.ones(11),5,.1,2.,'raw-z')[1:],np.zeros(31))

    def test_shared_pool_and_high_preservation(self):
        t=np.arange(150)
        score=np.exp(-((t-30)/3)**2)+.6*np.exp(-((t-80)/3)**2)
        signal={'subject':'s','video':'v','score':score,'logits':np.zeros((150,5))}
        pool=common_pool(signal,dict(c_s=2.,p_s=.65,c_d=1.25,c_b=1.25))
        for delta,data in pool['pools'].items():
            decoded=union(pool,delta,np.ones(len(data['events']),dtype=bool))
            self.assertEqual(decoded['events'],data['low_events'])
            self.assertEqual([e for e in decoded['events'] if e['source']=='tuned_native'],pool['high'])
            np.testing.assert_allclose(data['R2'],.2)
        with self.assertRaises(AssertionError):common_pool({**signal,'samples':[]},dict(c_s=2.,p_s=.65,c_d=1.25,c_b=1.25))

    def test_label_iou_and_missed_semantics(self):
        high={'peak':5,'onset':0,'offset':10,'source':'tuned_native'}
        weak=[{'peak':5,'onset':0,'offset':10,'source':'rescue'},
              {'peak':22,'onset':18,'offset':26,'source':'rescue'},
              {'peak':41,'onset':0,'offset':80,'source':'rescue'}]
        record={'subject':'s','video':'v','samples':[[0,5,10],[18,22,26],[40,41,42]]}
        bank={('s','v'):{'high':[high],'pools':{.1:{'events':weak}}}}
        np.testing.assert_array_equal(label_training([record],bank,.1)[('s','v')],[0,1,0])

    def test_fitted_state_predict_replay(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(30,32));y=np.array([0,1]*15)
        scaler,model=model_fit(x,y,1.)
        test=rng.normal(size=(4,32))
        replay=expit(((test-scaler.mean_)/scaler.scale_)@model.coef_[0]+model.intercept_[0])
        np.testing.assert_allclose(model_predict(scaler,model,test),replay,rtol=1e-14)


if __name__=='__main__':unittest.main()
