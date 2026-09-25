#!/usr/bin/env python3
"""Compare two real inference result files on five seed-fixed candidates."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import load_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/deterministic_replay.json")
    args = parser.parse_args()
    first, second = load_json(args.first), load_json(args.second)
    if first.get("MOCK_OUTPUT") or second.get("MOCK_OUTPUT"):
        raise RuntimeError("MOCK_RESULT_NOT_EVALUABLE")
    left = {item["candidate_id"]: item for item in first.get("results", [])}
    right = {item["candidate_id"]: item for item in second.get("results", [])}
    if set(left) != set(right) or len(left) != 40:
        raise RuntimeError("Replay inputs do not contain the same 40 candidates")
    selected = random.Random(100).sample(sorted(left), 5)
    comparisons = [
        {
            "candidate_id": candidate_id,
            "verdict_equal": left[candidate_id].get("verdict") == right[candidate_id].get("verdict"),
            "structured_json_equal": left[candidate_id].get("structured_output") == right[candidate_id].get("structured_output"),
        }
        for candidate_id in selected
    ]
    passed = all(item["verdict_equal"] and item["structured_json_equal"] for item in comparisons)
    write_json(args.output, {"seed": 100, "candidate_count": 5, "comparisons": comparisons, "status": "PASS" if passed else "FAIL"})
    print("DETERMINISTIC_REPLAY_PASS" if passed else "DETERMINISTIC_REPLAY_FAILED")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
