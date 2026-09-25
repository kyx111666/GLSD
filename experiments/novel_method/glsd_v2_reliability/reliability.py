"""Alignment reliability derived from the unchanged GLSD-v1 alignment rule."""

from __future__ import annotations

import numpy as np

from my_method.gl_saliency_skill.evidence import CurveFeatures


def evidence_and_reliability(curve, temporal_scale: int, reference: float, radius: float):
    """Return v1 peaks/evidence and the prescribed label-free reliability A(c).

    The call to ``CurveFeatures.evidence`` is the sole source of G and L.  The
    aligned peak is recovered with the exact same tolerance, eligible set and
    deterministic nearest-then-higher-local-score tie-break used by GLSD-v1.
    """
    features = CurveFeatures(curve, temporal_scale)
    peaks, evidence = features.evidence(reference, radius)
    tolerance = max(1, round(0.5 * temporal_scale))
    window = max(1, round(radius * temporal_scale))
    reliability = []
    for candidate in peaks:
        per_scale = []
        for width, (other_peaks, _, _, _, _) in features.effective_scales.items():
            local_values = features.local_cache[width, window]
            eligible = np.flatnonzero(np.abs(other_peaks - candidate) <= tolerance)
            if not len(eligible):
                per_scale.append(0.0)
                continue
            chosen = min(
                eligible,
                key=lambda index: (
                    abs(int(other_peaks[index]) - int(candidate)),
                    -float(local_values[index]),
                ),
            )
            alignment = max(
                0.0,
                1.0
                - abs(int(other_peaks[chosen]) - int(candidate)) / float(tolerance),
            )
            per_scale.append(alignment)
        reliability.append(float(np.mean(per_scale)) if per_scale else 0.0)
    values = np.asarray(reliability, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise AssertionError("Alignment reliability A(c) is outside [0,1]")
    return peaks, evidence, values


def v2_score(evidence: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Fixed GLSD-v2 formula: G + A*L - G*A*L."""
    values = np.asarray(evidence, dtype=float)
    reliability = np.asarray(reliability, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(reliability):
        raise ValueError("Expected N x 3 v1 evidence and N reliability values")
    global_score = values[:, 1]
    local_score = values[:, 2]
    return 1.0 - (1.0 - global_score) * (1.0 - reliability * local_score)

