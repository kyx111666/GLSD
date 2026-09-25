import unittest
import numpy as np
from one_to_one_evaluator import check_box_one_to_one as match
class Tests(unittest.TestCase):
    def test_duplicate_gt_is_false_positive(self):
        args=(np.array([.8]),np.array([0]),np.array([0]),np.array([0]))
        self.assertEqual(match(*args,[],.5),(1,[0]))
        self.assertEqual(match(*args,[[0]],.5),(0,[]))
    def test_one_prediction_gets_one_tp(self):
        args=(np.array([.8,.7]),np.array([0,0]),np.array([0,0]),np.array([0,1]))
        self.assertEqual(match(*args,[],.5,'soft'),(1,[0]))
        self.assertEqual(match(*args,[[0]],.5,'soft'),(1,[1]))
        self.assertEqual(match(*args,[[0]],.5,'greedy'),(0,[]))
    def test_below_iou_and_no_gt(self):
        self.assertEqual(match(np.array([.4]),np.array([0]),np.array([0]),np.array([0]),[],.5),(0,[]))
        self.assertEqual(match(np.array([]),np.array([]),np.array([]),np.array([],int),[],.5),(0,[]))
if __name__=='__main__':unittest.main()
