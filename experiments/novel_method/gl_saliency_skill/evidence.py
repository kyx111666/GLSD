"""Exact extraction of the signed historical G/L evidence implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences


SCALES = (1.0, 1.5, 2.0)
REFERENCE_SCALES = (1.0, 1.5, 2.0)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)
HEIGHT_WEIGHTS = (0.0, 0.25, 0.5, 0.75)
LOCAL_RADII = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class Config:
    family: str
    reference: float
    height_weight: float
    threshold: float
    radius: float = 1.0

    @property
    def identifier(self) -> str:
        return (
            f"{self.family}|scale={self.reference:g}"
            f"|height={self.height_weight:g}|tau={self.threshold:g}|radius={self.radius:g}"
        )


def configuration_grid() -> list[Config]:
    """Preserve the historical 511-entry order and therefore historical IDs."""
    configs = [Config("native", 2.0, 1.0, 0.55)]
    for reference in REFERENCE_SCALES:
        for radius in LOCAL_RADII:
            for weight in HEIGHT_WEIGHTS:
                configs.extend(
                    Config("unified", reference, weight, threshold, radius)
                    for threshold in THRESHOLDS
                )
            configs.extend(
                Config("local", reference, 0.0, threshold, radius)
                for threshold in THRESHOLDS
            )
        for family in ("height", "global"):
            configs.extend(
                Config(
                    family,
                    reference,
                    1.0 if family == "height" else 0.0,
                    threshold,
                )
                for threshold in THRESHOLDS
            )
    return configs


def pure_config_indexes(configs: list[Config]) -> list[int]:
    indexes = [
        index
        for index, config in enumerate(configs)
        if config.family == "unified" and config.height_weight == 0
    ]
    if len(indexes) != 90:
        raise AssertionError("Historical pure G+L grid must contain exactly 90 configs")
    return indexes


class CurveFeatures:
    """Label-free candidate generation and exact historical H/G/L evidence."""

    def __init__(self, curve, k):
        self.curve = np.asarray(curve, dtype=float)
        if self.curve.ndim != 1 or len(self.curve) < 3 or not np.isfinite(self.curve).all():
            raise ValueError("Expected a finite one-dimensional curve with at least three samples")
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.scales = {}
        self.local_cache = {}
        self.evidence_cache = {}
        physical = {}
        for scale in SCALES:
            width = min(len(self.curve), max(1, round(scale * k)))
            if width not in physical:
                smooth = np.convolve(self.curve, np.ones(width) / width, mode="same")
                peaks = (
                    find_peaks(smooth, distance=k)[0]
                    if np.ptp(self.curve) > 0
                    else np.empty(0, dtype=int)
                )
                spread = max(float(np.ptp(smooth)), 1e-12)
                mean = float(smooth.mean())
                height = np.maximum(
                    0.0,
                    (smooth[peaks] - mean) / max(float(smooth.max() - mean), 1e-12),
                )
                global_values = peak_prominences(smooth, peaks)[0] / spread
                physical[width] = (peaks, height, global_values, smooth, spread)
            self.scales[scale] = physical[width]
        self.effective_scales = physical

    def evidence(self, reference, radius=1.0):
        if (reference, radius) in self.evidence_cache:
            return self.evidence_cache[reference, radius]
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))
        for width, (other_peaks, _, _, smooth, spread) in self.effective_scales.items():
            if (width, window) not in self.local_cache:
                self.local_cache[width, window] = np.array(
                    [
                        max(
                            0.0,
                            float(smooth[peak])
                            - max(
                                float(np.min(smooth[max(0, peak - window) : peak + 1])),
                                float(
                                    np.min(
                                        smooth[
                                            peak : min(len(smooth), peak + window + 1)
                                        ]
                                    )
                                ),
                            ),
                        )
                        / spread
                        for peak in other_peaks
                    ]
                )
        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (
                        abs(int(other_peaks[index]) - point),
                        -values[index],
                    ),
                )
                aligned.append(float(values[chosen]))
            local.append(float(np.median(aligned)))
        output = peaks, np.column_stack((height, global_values, local))
        self.evidence_cache[reference, radius] = output
        return output


def evidence_scores(evidence: np.ndarray, config: Config) -> np.ndarray:
    height, global_values, local = evidence.T
    if config.family in ("native", "height"):
        return height
    if config.family == "global":
        return global_values
    if config.family == "local":
        return local
    if config.family != "unified":
        raise ValueError(f"Unknown family: {config.family}")
    if config.height_weight != 0:
        raise ValueError("FINAL-GL-SKILL permits only historical pure G+L with height_weight=0")
    return 0.5 * (global_values + local)
