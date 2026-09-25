"""Deterministic evidence construction. This module never receives ground truth."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import find_peaks


def round_half_up(value: float) -> int:
    return max(1, int(math.floor(float(value) + 0.5)))


def boxcar(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(np.asarray(values, dtype=float), np.ones(width) / width, mode="same")


def mean_to_max(values: np.ndarray) -> np.ndarray:
    """Normalize on the full video curve using the frozen v2 convention."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    maximum = float(np.max(values))
    denominator = maximum - mean
    if denominator <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / denominator


@dataclass(frozen=True)
class Evidence:
    candidates: tuple[int, ...]
    global_scores: tuple[float, ...]
    local_scores: tuple[float, ...]


class EvidenceExtractor:
    """Build G and L from one frozen score curve, without labels."""

    def __init__(self, local_scales: tuple[float, ...] = (1.0, 1.5, 2.0)) -> None:
        self.local_scales = tuple(float(value) for value in local_scales)

    @staticmethod
    def _curve_and_peaks(score: np.ndarray, scale: float, temporal_scale: int):
        scale_frames = round_half_up(scale * temporal_scale)
        curve = boxcar(score, 2 * scale_frames)
        normalized = mean_to_max(curve)
        peaks = find_peaks(curve, distance=scale_frames)[0].astype(int)
        return normalized, peaks

    def extract(
        self,
        score: np.ndarray,
        temporal_scale: int,
        reference_scale: float,
        radius: int,
    ) -> Evidence:
        score = np.asarray(score, dtype=float)
        global_curve, candidates_array = self._curve_and_peaks(
            score, reference_scale, temporal_scale
        )
        candidates = tuple(int(value) for value in candidates_array.tolist())
        global_scores = tuple(float(global_curve[position]) for position in candidates)
        tolerance = round_half_up(radius * temporal_scale)

        local_banks: list[tuple[np.ndarray, np.ndarray]] = []
        for scale in self.local_scales:
            normalized, peaks = self._curve_and_peaks(score, scale, temporal_scale)
            local_banks.append((normalized, peaks))

        local_scores: list[float] = []
        for candidate in candidates:
            aligned: list[float] = []
            for normalized, peaks in local_banks:
                eligible = [
                    int(peak) for peak in peaks.tolist() if abs(int(peak) - candidate) <= tolerance
                ]
                if not eligible:
                    aligned.append(0.0)
                    continue
                chosen = min(
                    eligible,
                    key=lambda peak: (
                        abs(peak - candidate),
                        -float(normalized[peak]),
                        peak,
                    ),
                )
                aligned.append(float(normalized[chosen]))
            local_scores.append(float(np.mean(aligned)))

        return Evidence(candidates, global_scores, tuple(local_scores))
