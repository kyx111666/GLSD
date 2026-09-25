"""Native decoding and chronological greedy interval evaluation."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .evidence import boxcar


def interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return float(intersection / union) if union > 0 else 0.0


def native_events(score: np.ndarray, temporal_scale: int) -> list[dict]:
    curve = boxcar(np.asarray(score, dtype=float), 2 * temporal_scale)
    threshold = float(np.mean(curve) + 0.55 * (np.max(curve) - np.mean(curve)))
    peaks = find_peaks(curve, height=threshold, distance=temporal_scale)[0]
    return [
        {
            "peak": int(peak),
            "onset": int(peak - temporal_scale),
            "offset": int(peak + temporal_scale),
            "score": float(curve[peak]),
        }
        for peak in peaks.tolist()
    ]


def evaluate_events(events: list[dict], ground_truth: list) -> dict:
    predictions = sorted(events, key=lambda event: (event["peak"], event["onset"], event["offset"]))
    gt_intervals = [(int(item[0]), int(item[-1])) for item in ground_truth]
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[dict] = []
    for prediction_index, prediction in enumerate(predictions):
        prediction_interval = (int(prediction["onset"]), int(prediction["offset"]))
        choices = [
            (interval_iou(prediction_interval, gt_interval), gt_index)
            for gt_index, gt_interval in enumerate(gt_intervals)
            if gt_index not in matched_gt
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= 0.5:
            matched_gt.add(best_index)
            matched_predictions.add(prediction_index)
            matches.append(
                {"prediction_index": prediction_index, "gt_index": best_index, "iou": best_iou}
            )
    tp = len(matches)
    return {
        "TP": tp,
        "FP": len(predictions) - tp,
        "FN": len(gt_intervals) - tp,
        "matched_gt": sorted(matched_gt),
        "matched_predictions": sorted(matched_predictions),
        "matches": matches,
    }


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def add_counts(rows: list[dict]) -> dict:
    result = {name: sum(int(row[name]) for row in rows) for name in ("TP", "FP", "FN")}
    result["F1"] = f1(result["TP"], result["FP"], result["FN"])
    return result
