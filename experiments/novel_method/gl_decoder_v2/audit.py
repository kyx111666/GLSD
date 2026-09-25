"""Agreement and mechanism summaries for GL-Decoder-v2."""

from __future__ import annotations


def set_jaccard(first: set, second: set) -> float:
    union = first | second
    return float(len(first & second) / len(union)) if union else 1.0


def neighborhood_peak_jaccard(first: list[int], second: list[int], tolerance: int) -> float:
    available = set(range(len(second)))
    matches = 0
    for peak in sorted(first):
        eligible = [index for index in available if abs(peak - second[index]) <= tolerance]
        if eligible:
            chosen = min(eligible, key=lambda index: (abs(peak - second[index]), second[index]))
            available.remove(chosen)
            matches += 1
    denominator = len(first) + len(second) - matches
    return float(matches / denominator) if denominator else 1.0


def compare_predictions(current: list[dict], historical: list[dict], tolerance: int) -> dict:
    current_intervals = {(int(item["onset"]), int(item["offset"])) for item in current}
    historical_intervals = {(int(item["onset"]), int(item["offset"])) for item in historical}
    current_peaks = [int(item["peak"]) for item in current]
    historical_peaks = [int(item["peak"]) for item in historical]
    return {
        "exact_interval_jaccard": set_jaccard(current_intervals, historical_intervals),
        "exact_peak_jaccard": set_jaccard(set(current_peaks), set(historical_peaks)),
        "temporal_neighborhood_jaccard": neighborhood_peak_jaccard(
            current_peaks, historical_peaks, tolerance
        ),
        "v2_count": len(current),
        "historical_count": len(historical),
    }
