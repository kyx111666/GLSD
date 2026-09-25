"""Deterministic persistence evidence for a fixed temporal candidate pool."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.signal._peak_finding_utils import PeakPropertyWarning


OPERATORS = (
    "P0_no_persistence",
    "P1_peak_prominence",
    "P2_topological_0d",
    "P3_scale_space_survival",
)
EPS = 1e-12


def robust_z(values: np.ndarray) -> np.ndarray:
    """Candidate-wise robust z-score with a conservative deterministic fallback.

    Fewer than three candidates or zero MAD returns all zeros.  Thus tau=0 keeps
    every candidate, while positive thresholds reject an uninformative score set.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return np.zeros_like(values)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if not np.isfinite(mad) or mad <= EPS:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad + 1e-8)


def peak_prominence(curve: np.ndarray, peak: int) -> float:
    """Current project definition: scipy prominence on the supplied curve."""
    curve = np.asarray(curve, dtype=np.float64)
    peak = int(peak)
    if not 0 <= peak < len(curve):
        raise IndexError(f"peak {peak} outside curve of length {len(curve)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PeakPropertyWarning)
        return float(peak_prominences(curve, np.asarray([peak], dtype=int))[0][0])


def _plateau_runs(curve: np.ndarray) -> list[dict]:
    runs = []
    start = 0
    for index in range(1, len(curve) + 1):
        if index == len(curve) or curve[index] != curve[start]:
            end = index - 1
            runs.append({
                "start": start,
                "end": end,
                "representative": (start + end) // 2,
                "value": float(curve[start]),
            })
            start = index
    return runs


def topological_births(curve: np.ndarray) -> dict[int, float]:
    """Return birth-index -> 0D superlevel persistence for a 1D curve.

    Equal-valued contiguous samples are one plateau, represented by its left
    middle sample. Plateaus are activated by descending value then temporal
    position. At a merge, higher birth wins; tied births use the lower index.
    The final surviving component dies at the global curve minimum.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 1:
        raise ValueError("curve must be one-dimensional")
    if not len(curve):
        return {}
    if not np.all(np.isfinite(curve)):
        raise ValueError("curve must contain only finite values")

    runs = _plateau_runs(curve)
    count = len(runs)
    parent = np.arange(count, dtype=int)
    active = np.zeros(count, dtype=bool)
    birth_value = np.zeros(count, dtype=np.float64)
    birth_index = np.zeros(count, dtype=int)
    result: dict[int, float] = {}

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def older(left: int, right: int) -> tuple[int, int]:
        left, right = root(left), root(right)
        left_key = (-birth_value[left], birth_index[left])
        right_key = (-birth_value[right], birth_index[right])
        return (left, right) if left_key <= right_key else (right, left)

    order = sorted(range(count), key=lambda i: (-runs[i]["value"], runs[i]["start"]))
    for node in order:
        active[node] = True
        parent[node] = node
        birth_value[node] = runs[node]["value"]
        birth_index[node] = runs[node]["representative"]
        neighbors = [neighbor for neighbor in (node - 1, node + 1)
                     if 0 <= neighbor < count and active[neighbor]]
        for neighbor in neighbors:
            current, other = root(node), root(neighbor)
            if current == other:
                continue
            survivor, dying = older(current, other)
            result[int(birth_index[dying])] = float(
                max(0.0, birth_value[dying] - runs[node]["value"])
            )
            parent[dying] = survivor
            parent[current] = survivor
            parent[other] = survivor

    survivor = root(order[-1])
    result[int(birth_index[survivor])] = float(
        max(0.0, birth_value[survivor] - float(np.min(curve)))
    )
    return result


def topological_persistence(curve: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    births = topological_births(curve)
    if not len(peaks):
        return np.empty(0, dtype=np.float64)
    if not births:
        return np.zeros(len(peaks), dtype=np.float64)
    birth_indices = np.asarray(sorted(births), dtype=int)
    scores = []
    for peak in peaks:
        if not 0 <= peak < len(curve):
            raise IndexError(f"peak {peak} outside curve of length {len(curve)}")
        containing = [run for run in _plateau_runs(curve)
                      if run["start"] <= peak <= run["end"]
                      and run["representative"] in births]
        if containing:
            mapped = int(containing[0]["representative"])
        else:
            mapped = min(
                birth_indices.tolist(),
                key=lambda index: (abs(index - int(peak)), -curve[index], index),
            )
        scores.append(births[mapped])
    return np.asarray(scores, dtype=np.float64)


def moving_average(curve: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.asarray(curve, dtype=np.float64), kernel, mode="same")


def scale_space_matches(curve: np.ndarray, peaks: np.ndarray, k_p: int) -> list[list[int | None]]:
    """Matched local maximum at each locked scale for every base peak."""
    curve = np.asarray(curve, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    tolerance = 0.5 * int(k_p)
    all_matches: list[list[int | None]] = [[] for _ in peaks]
    for factor in (0.5, 1.0, 1.5, 2.0):
        # The project moving-average convention uses 2*k as the full window.
        smoothed = moving_average(curve, max(1, int(round(2.0 * factor * k_p))))
        scale_peaks = find_peaks(smoothed)[0].astype(int)
        for index, base_peak in enumerate(peaks):
            eligible = [int(p) for p in scale_peaks if abs(int(p) - int(base_peak)) <= tolerance]
            match = min(
                eligible,
                key=lambda p: (abs(p - int(base_peak)), -float(smoothed[p]), p),
                default=None,
            )
            all_matches[index].append(match)
    return all_matches


def scale_space_survival(curve: np.ndarray, peaks: np.ndarray, k_p: int) -> np.ndarray:
    matches = scale_space_matches(curve, peaks, k_p)
    return np.asarray([sum(match is not None for match in row) / 4.0 for row in matches],
                      dtype=np.float64)


def raw_scores(curve: np.ndarray, peaks: np.ndarray, k_p: int, operator: str) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator: {operator}")
    if operator == "P0_no_persistence":
        return np.ones(len(peaks), dtype=np.float64)
    if operator == "P1_peak_prominence":
        return np.asarray([peak_prominence(curve, peak) for peak in peaks], dtype=np.float64)
    if operator == "P2_topological_0d":
        return topological_persistence(curve, peaks)
    return scale_space_survival(curve, peaks, k_p)


def score_candidates(curve: np.ndarray, peaks: np.ndarray, k_p: int, operator: str) -> np.ndarray:
    """Return normalized verification evidence aligned one-to-one with peaks."""
    scores = raw_scores(curve, peaks, k_p, operator)
    return scores if operator == "P0_no_persistence" else robust_z(scores)
