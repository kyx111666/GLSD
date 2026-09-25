from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from extract_candidate_frames import sample_temporal_indices  # noqa: E402
from parse_output import parse_model_output  # noqa: E402
from protocol import (  # noqa: E402
    CANONICAL_PREDICTIONS,
    assert_no_label_leakage,
    replay_gate,
)


class ProtocolTests(unittest.TestCase):
    def test_locked_glsd_replay(self) -> None:
        if not CANONICAL_PREDICTIONS.is_file():
            self.skipTest("canonical predictions are an external replay input")
        gate = replay_gate()
        self.assertEqual(gate["counts"], {"TP": 48, "FP": 126, "FN": 111})
        self.assertEqual(gate["candidate_count"], 174)

    def test_leakage_gate_rejects_evaluation_label(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "LABEL_LEAKAGE_GATE_FAILED"):
            assert_no_label_leakage(
                {"candidates": [{"candidate_id": "x", "gt_match_label_for_evaluation_only": "TP"}]}
            )

    def test_leakage_gate_accepts_sanitized_manifest(self) -> None:
        assert_no_label_leakage(
            {
                "protocol": {"candidate_count": 1},
                "candidates": [
                    {
                        "candidate_id": "x",
                        "ordered_frame_paths": ["a.jpg"],
                        "ordered_frame_roles": ["B1"],
                    }
                ],
            }
        )

    def test_output_parser_strict_and_single_repair(self) -> None:
        payload = {
            "local_facial_change": True,
            "brief_transient_change": True,
            "return_toward_baseline": False,
            "global_motion_artifact": False,
            "verdict": "keep",
        }
        parsed, ok, repaired, error = parse_model_output(json.dumps(payload))
        self.assertTrue(ok)
        self.assertFalse(repaired)
        self.assertIsNone(error)
        self.assertEqual(parsed, payload)
        wrapped = f"```json\n{json.dumps(payload)[:-1]},}}\n```"
        parsed, ok, repaired, error = parse_model_output(wrapped)
        self.assertTrue(ok)
        self.assertTrue(repaired)
        self.assertIsNone(error)
        self.assertEqual(parsed, payload)

    def test_nine_frame_sampling_is_unique_and_monotonic(self) -> None:
        indices, warnings = sample_temporal_indices(100, 110, 500)
        self.assertEqual(len(indices), 9)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
