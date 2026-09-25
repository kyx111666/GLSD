"""Protocol constants and gates for the isolated MLLM verifier exploration."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


EXPLORATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPLORATION_ROOT.parent
CANONICAL_PREDICTIONS = (
    PROJECT_ROOT
    / "historical_gl_exact_fresh_reproduction/fresh_run/metst/results"
    / "pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json"
)
CANONICAL_PREDICTIONS_SHA256 = (
    "bbf0d559db15aaa5cd1ccc69e0dac3ef49cdd6c7d5294d07b8f10b7c38858643"
)
VIDEO_ROOT = Path(os.environ.get("SAMMLV_VIDEO_ROOT", "data/SAMM_longvideos"))
EXPECTED_COUNTS = {"TP": 48, "FP": 126, "FN": 111}
EXPECTED_F1 = 0.2882882882882883
FRAME_SKIP = 7
FPS = 200.0
SEED = 100
FORBIDDEN_INFERENCE_KEY_PARTS = (
    "tp",
    "fp",
    "gt",
    "match_label",
    "matched_gt",
    "ground_truth",
    "label_for_evaluation",
    "glsd_score",
    "threshold",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def interval_iou(event: dict[str, int], ground_truth: list[int]) -> float:
    left = max(int(event["onset"]), int(ground_truth[0]))
    right = min(int(event["offset"]), int(ground_truth[2]))
    intersection = max(0, right - left + 1)
    union = (
        int(event["offset"])
        - int(event["onset"])
        + 1
        + int(ground_truth[2])
        - int(ground_truth[0])
        + 1
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def match_events(
    events: Iterable[dict[str, int]], ground_truth: list[list[int]]
) -> tuple[dict[str, int], list[int]]:
    """Exact chronological, no-second-best-rematch GLSD evaluator semantics."""
    events = list(events)
    matched: set[int] = set()
    assignments: list[int] = []
    for event in events:
        overlaps = [interval_iou(event, item) for item in ground_truth]
        best = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1
        assigned = (
            best
            if best >= 0 and overlaps[best] >= 0.5 and best not in matched
            else -1
        )
        if assigned >= 0:
            matched.add(assigned)
        assignments.append(assigned)
    return {
        "TP": len(matched),
        "FP": len(events) - len(matched),
        "FN": len(ground_truth) - len(matched),
    }, assignments


def replay_gate(path: Path = CANONICAL_PREDICTIONS) -> dict[str, Any]:
    if path.resolve() == CANONICAL_PREDICTIONS.resolve():
        actual_hash = sha256_file(path)
        if actual_hash != CANONICAL_PREDICTIONS_SHA256:
            raise RuntimeError(
                "GLSD_REPLAY_GATE_FAILED: canonical prediction file SHA256 changed"
            )
    rows = load_json(path)
    totals = {"TP": 0, "FP": 0, "FN": 0}
    saved_assignment_mismatches = 0
    geometries: list[tuple[Any, ...]] = []
    for row in rows:
        events = row["predictions"]["pure"]
        counts, assignments = match_events(events, row["gt"])
        for key in totals:
            totals[key] += counts[key]
        saved = [int(event["matched_gt"]) for event in events]
        saved_assignment_mismatches += int(saved != assignments)
        geometries.extend(
            (
                str(row["subject"]),
                str(row["video"]),
                int(event["onset"]),
                int(event["peak"]),
                int(event["offset"]),
            )
            for event in events
        )
    denominator = 2 * totals["TP"] + totals["FP"] + totals["FN"]
    f1 = 2 * totals["TP"] / denominator if denominator else 0.0
    passed = (
        totals == EXPECTED_COUNTS
        and abs(f1 - EXPECTED_F1) <= 1e-15
        and saved_assignment_mismatches == 0
        and len(geometries) == len(set(geometries))
    )
    if not passed:
        raise RuntimeError(
            "GLSD_REPLAY_GATE_FAILED: "
            f"counts={totals}, F1={f1}, assignment_mismatches="
            f"{saved_assignment_mismatches}, candidates={len(geometries)}, "
            f"unique={len(set(geometries))}"
        )
    return {
        "status": "PASS",
        "counts": totals,
        "F1": f1,
        "videos": len(rows),
        "subjects": len({str(row["subject"]) for row in rows}),
        "GT": sum(len(row["gt"]) for row in rows),
        "candidate_count": len(geometries),
        "candidate_geometry_sha256": canonical_json_hash(geometries),
        "source_sha256": sha256_file(path),
    }


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def ordered_frame_paths(video_path: Path) -> list[Path]:
    frames = sorted(
        [*video_path.glob("*.jpg"), *video_path.glob("*.jpeg"), *video_path.glob("*.png")],
        key=natural_key,
    )
    if not frames:
        raise RuntimeError(f"No image frames found in {video_path}")
    return frames


def assert_no_label_leakage(value: Any, location: str = "root") -> None:
    """Reject evaluation or GLSD-evidence keys before model loading."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in FORBIDDEN_INFERENCE_KEY_PARTS):
                raise RuntimeError(
                    f"LABEL_LEAKAGE_GATE_FAILED: forbidden key {key!r} at {location}"
                )
            assert_no_label_leakage(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_label_leakage(child, f"{location}[{index}]")
