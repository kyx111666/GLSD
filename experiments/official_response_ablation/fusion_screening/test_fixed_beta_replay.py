import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import run_fixed_beta_replay as run


class FixedBetaTests(unittest.TestCase):
    def test_conditional_selection_cannot_see_held_subject(self):
        grid=[dict(beta=b,config_id=2*j+i) for j,b in enumerate(run.BETAS) for i in (0,1)]
        table=np.array([[[0,99,2],[2,0,0]],[[2,0,0],[1,2,1]]]*6)
        a=run.conditional_choices(table,grid,['a','b'])
        table[:,0]=[999,0,0]
        b=run.conditional_choices(table,grid,['a','b'])
        for before,after in zip(a,b):
            if before['subject']=='a':
                self.assertEqual(before['config'],after['config'])
                self.assertEqual(before['inner_raw_counts'],after['inner_raw_counts'])

    def test_real_saved_tensor_matches_all_six_diagnostic_totals(self):
        with np.load(Path(run.__file__).with_name('fixed_beta_search_input.npz'),allow_pickle=False) as z:
            rows=run.conditional_choices(z['raw_counts'],json.loads(z['configs'].item()),z['subjects'].tolist())
        expected={.5:[42,124,117],.6:[45,119,114],.7:[41,119,118],.8:[46,132,113],.9:[45,112,114],1.:[45,109,114]}
        self.assertEqual(len(rows),174)
        for beta,counts in expected.items():
            chosen=[r for r in rows if r['label']==f'FixedBeta_{beta:.1f}']
            self.assertEqual(np.sum([r['expected_raw'] for r in chosen],axis=0).tolist(),counts)
            self.assertTrue(all(r['config']['beta']==beta and r['inner_subject_count']==28 for r in chosen))

    def test_raw_and_full_summaries_remain_separate(self):
        labels=[f'FixedBeta_{b:.1f}' for b in run.BETAS]+[f'Expanded_{m}' for m in run.tuning.METHODS]+['Original_EqualMean']
        records=[dict(label=l,subject=s,raw_counts=[2,2,1],full_counts=[1,1,2]) for l in labels for s in ['a','b']]
        with TemporaryDirectory() as directory:
            output=Path(directory)
            run.summary_and_pairs(records,['a','b'],output)
            with (output/'summary_raw_full.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),24)
            self.assertEqual(rows[0]['TP'],'4')
            self.assertEqual(rows[1]['TP'],'2')
            with (output/'paired_comparisons.csv').open() as f:
                pairs=list(csv.DictReader(f))
            self.assertEqual(len(pairs),36)
            self.assertTrue(all(float(r['delta_F1'])==0 and float(r['CI_low'])==0 and float(r['CI_high'])==0 for r in pairs))
            records[0]['full_counts']=[2,1,1]
            with self.assertRaisesRegex(RuntimeError,'Fixed beta=.5'):
                run.summary_and_pairs(records,['a','b'],output)


if __name__=='__main__':unittest.main()
