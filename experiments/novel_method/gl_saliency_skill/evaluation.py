"""Exact historical chronological, no-second-best-rematch evaluator."""

from __future__ import annotations

import numpy as np


def overlap_matrix(intervals, ground_truth) -> np.ndarray:
    events = np.asarray(ground_truth, dtype=int).reshape(-1, 3)
    intervals = np.asarray(intervals, dtype=int).reshape(-1, 2)
    left = np.maximum(intervals[:, 0, None], events[:, 0])
    right = np.minimum(intervals[:, 1, None], events[:, 2])
    intersection = np.maximum(0, right - left + 1)
    union = (
        (intervals[:, 1] - intervals[:, 0] + 1)[:, None]
        + events[:, 2]
        - events[:, 0]
        + 1
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )


def evaluate(events: list[dict], ground_truth: list) -> tuple[dict, list[dict]]:
    intervals = [(event["onset"], event["offset"]) for event in events]
    overlaps = overlap_matrix(intervals, ground_truth)
    matched: set[int] = set()
    output = []
    for index, event in enumerate(events):
        values = overlaps[index]
        gt_index = int(np.argmax(values)) if len(values) else -1
        assigned = (
            gt_index
            if gt_index >= 0 and values[gt_index] >= 0.5 and gt_index not in matched
            else -1
        )
        if assigned >= 0:
            matched.add(assigned)
        output.append({**event, "matched_gt": assigned})
    tp = len(matched)
    counts = {"TP": tp, "FP": len(events) - tp, "FN": len(ground_truth) - tp}
    return counts, output


def metrics(counts) -> dict:
    tp, fp, fn = (int(counts[key]) for key in ("TP", "FP", "FN"))
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
