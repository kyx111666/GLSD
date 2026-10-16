import unittest
from types import SimpleNamespace

import run_locked_alpha_ablation as locked


class LockedAlphaTests(unittest.TestCase):
    def test_source_config_without_config_id_is_supported(self):
        grid = [
            SimpleNamespace(reference_scale=1.0, local_radius=1.0, threshold=0.05),
            SimpleNamespace(reference_scale=1.5, local_radius=2.0, threshold=0.4),
        ]
        self.assertEqual(locked.locked_config_index(grid[1], grid), 1)
        converted = locked.locked_weighted_config(grid[1], 0.5, 1)
        self.assertEqual(converted.config_id, 1)
        self.assertEqual(converted.alpha, 0.5)

    def test_counts_are_validated_against_subject_gt(self):
        self.assertEqual(locked.validate_counts((6, 10, 5), 11, "subject 006"), (6, 10, 5))
        with self.assertRaises(RuntimeError):
            locked.validate_counts((6, 10, 5), 159, "incorrect global GT")

    def test_subject_gt_uses_only_requested_subject(self):
        runner = SimpleNamespace(final_samples=lambda records: [
            [[1], [2, 3]],
            [[4, 5, 6]],
        ])
        context = (runner, None, None, None, object(), None, None, None)
        self.assertEqual(locked.subject_gt(context, 0), 3)
        self.assertEqual(locked.subject_gt(context, 1), 3)


if __name__ == "__main__":
    unittest.main()
