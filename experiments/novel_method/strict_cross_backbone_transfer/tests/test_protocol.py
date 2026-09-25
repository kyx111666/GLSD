from __future__ import annotations

import unittest

import numpy as np

from my_method.gl_saliency_skill.evidence import configuration_grid, pure_config_indexes
from my_method.strict_cross_backbone_transfer.run import (
    NEUTRAL_TOLERANCE,
    paired_bootstrap,
    source_manifest,
    transfer_label,
)


class StrictTransferProtocolTests(unittest.TestCase):
    def test_locked_grid(self):
        self.assertEqual(len(pure_config_indexes(configuration_grid())), 90)

    def test_predeclared_transfer_labels(self):
        self.assertEqual(transfer_label(2 * NEUTRAL_TOLERANCE), "POSITIVE_TRANSFER")
        self.assertEqual(transfer_label(-2 * NEUTRAL_TOLERANCE), "NEGATIVE_TRANSFER")
        self.assertEqual(transfer_label(0.5 * NEUTRAL_TOLERANCE), "NEUTRAL_TRANSFER")

    def test_paired_bootstrap_identity(self):
        counts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        result = paired_bootstrap(counts, counts)
        self.assertEqual(result["point_delta_F1"], 0.0)
        self.assertEqual(result["bootstrap_mean_delta_F1"], 0.0)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 0.0)

    def test_skill_manifest_is_nonempty(self):
        manifest = source_manifest()
        self.assertIn("skill.py", manifest)
        self.assertIn("evidence.py", manifest)


if __name__ == "__main__":
    unittest.main()
