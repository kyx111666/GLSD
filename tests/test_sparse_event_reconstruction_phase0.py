import unittest

import numpy as np

import run_sparse_event_reconstruction_phase0 as sparse


class SparseEventReconstructionPhase0Tests(unittest.TestCase):
    def test_templates_have_unit_norm(self):
        for kind in sparse.TEMPLATE_KINDS:
            for duration in (3, 4, 9):
                self.assertAlmostEqual(np.linalg.norm(sparse.template(kind, duration)), 1.0)

    def test_allowed_starts_exclude_native_overlap(self):
        allowed = sparse.allowed_starts(10, (3,), [{"onset": 3, "offset": 5}])[3]
        self.assertEqual(np.flatnonzero(allowed).tolist(), [0, 6, 7])

    def test_residual_pursuit_recovers_separated_weaker_pulse(self):
        signal = np.asarray([0, 5, 5, 5, 0, 3, 3, 3, 0], dtype=float)
        templates = {3: sparse.template("box", 3)}
        allowed = sparse.allowed_starts(len(signal), (3,))
        matched = sparse.matched_filter(signal, templates, allowed, budget=2)
        residual, _, explained = sparse.residual_pursuit(signal, templates, allowed, budget=2)
        self.assertEqual(matched[0], (3, 1))
        self.assertIn((3, 5), residual)
        self.assertGreater(explained, 0.9)

    def test_native_anchor_changes_first_residual_candidate(self):
        signal = np.asarray([0, 5, 5, 5, 0, 3, 3, 3, 0], dtype=float)
        templates = {3: sparse.template("box", 3)}
        native_predictions = [{"onset": 1, "offset": 3}]
        allowed = sparse.allowed_starts(len(signal), (3,), native_predictions)
        anchors = sparse.anchor_columns(len(signal), native_predictions, "box")
        selected, initial, final = sparse.residual_pursuit(
            signal, templates, allowed, budget=1, fixed_columns=anchors)
        self.assertEqual(selected, [(3, 5)])
        self.assertGreater(final, initial)


if __name__ == "__main__":
    unittest.main()
