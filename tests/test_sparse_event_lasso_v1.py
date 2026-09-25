import unittest

import numpy as np

import run_sparse_event_lasso_v1 as lasso


class SparseEventLassoV1Tests(unittest.TestCase):
    def test_design_has_unit_norm_atoms_and_expected_count(self):
        matrix, metadata, durations = lasso.design_matrix(9, 1, (0.5, 1.0, 2.0))
        self.assertEqual(matrix.shape[1], sum(9 - d + 1 for d in durations))
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        np.testing.assert_allclose(norms, 1.0)
        self.assertEqual(len(metadata), matrix.shape[1])

    def test_alpha_path_includes_empty_solution(self):
        matrix, metadata, _ = lasso.design_matrix(9, 1, (1.0,))
        outputs, alpha_max = lasso.fit_path(np.zeros(9), matrix, metadata)
        self.assertEqual(outputs[max(lasso.RATIOS)]["selected"], ())
        self.assertEqual(alpha_max, 0.0)

    def test_ratio_one_is_zero_solution(self):
        matrix, metadata, _ = lasso.design_matrix(9, 1, (1.0,))
        signal = np.asarray([0, 3, 3, 3, 0, 2, 2, 2, 0], dtype=float)
        outputs, alpha_max = lasso.fit_path(signal, matrix, metadata)
        self.assertGreater(alpha_max, 0.0)
        self.assertEqual(outputs[1.0]["selected"], ())

    def test_matched_filter_uses_lasso_atom_count(self):
        matrix, metadata, _ = lasso.design_matrix(9, 1, (1.0,))
        signal = np.asarray([0, 3, 3, 3, 0, 2, 2, 2, 0], dtype=float)
        selected = lasso.matched_filter_equal_count(signal, matrix, metadata, 2)
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
