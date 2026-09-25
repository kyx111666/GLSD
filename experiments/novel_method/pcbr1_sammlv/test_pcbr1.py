"""Small geometry tests independent of SAMMLV outcomes."""
import ast
import inspect
import unittest

import numpy as np

from run_pcbr1_nested_loso import ALPHAS, reconstruct


def candidate(p, k, length):
    return {"candidate_id": "synthetic/pred_0", "peak": p,
            "onset": max(0, p-k), "offset": min(length-1, p+k),
            "emotion_id": 2, "confidence": 0.8}


class ReconstructionTests(unittest.TestCase):
    def test_asymmetric_contiguous_and_equality(self):
        # theta=.5; equality counts, but a later high island is not traversed.
        curve = np.array([0., .8, 0., .6, .8, 1., .8, .7, .5, .2, 0.])
        original = candidate(5, 2, len(curve))
        result = reconstruct(curve, [original], 2, .5)[0]
        self.assertEqual((result["onset"], result["offset"]), (3, 8))
        self.assertEqual(original, candidate(5, 2, len(curve)))

    def test_flat_curve_cap_and_edges(self):
        for peak in (0, 1, 5, 9, 10):
            curve = np.ones(11)
            result = reconstruct(curve, [candidate(peak, 2, 11)], 2, .7)[0]
            self.assertEqual((result["onset"], result["offset"]),
                             (max(0, peak-4), min(10, peak+4)))

    def test_no_shrink_no_metadata_change(self):
        curve = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
        original = candidate(4, 2, len(curve))
        self.assertEqual(reconstruct(curve, [original], 2, .3), [original])
        self.assertEqual(reconstruct(curve, [], 2, .3), [])

    def test_alpha_monotonicity_and_score_immutability(self):
        rng = np.random.default_rng(23)
        for _ in range(40):
            curve = rng.random(31)
            curve[15] = 1.
            before = curve.copy()
            original = candidate(15, 5, len(curve))
            events = [reconstruct(curve, [original], 5, a)[0] for a in ALPHAS]
            self.assertTrue(all(a["onset"] <= b["onset"] and a["offset"] >= b["offset"]
                                for a, b in zip(events, events[1:])))
            np.testing.assert_array_equal(curve, before)

    def test_inference_access_surface(self):
        self.assertEqual(list(inspect.signature(reconstruct).parameters),
                         ["curve", "candidates", "k", "alpha"])
        tree = ast.parse(inspect.getsource(reconstruct))
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertEqual(calls, {"len", "np.isfinite(curve).all", "np.isfinite", "max", "min",
                                "float", "np.min", "candidate.items", "original.items", "output.append"})


if __name__ == "__main__":
    unittest.main()
