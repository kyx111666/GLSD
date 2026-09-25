"""Backbone-agnostic candidate scoring, suppression, and event decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .evidence import Config, CurveFeatures, evidence_scores


@dataclass(frozen=True)
class Geometry:
    order: np.ndarray
    intervals: np.ndarray
    conflicts: np.ndarray


class IntervalAdapter(Protocol):
    name: str

    def geometry(self, temporal_scores: np.ndarray, temporal_scale: int, peaks: np.ndarray) -> Geometry:
        ...


@dataclass(frozen=True)
class CandidateAudit:
    peak: int
    onset: int
    offset: int
    height: float
    global_score: float
    local_score: float
    fused_score: float
    decision_score: float
    selected: bool


def decode_candidates(
    temporal_scores: np.ndarray,
    temporal_scale: int,
    config: Config,
    interval_adapter: IntervalAdapter,
) -> tuple[list[dict], list[CandidateAudit]]:
    features = CurveFeatures(temporal_scores, temporal_scale)
    peaks, evidence = features.evidence(config.reference, config.radius)
    geometry = interval_adapter.geometry(temporal_scores, temporal_scale, peaks)
    peaks = peaks[geometry.order]
    evidence = evidence[geometry.order]
    scores = evidence_scores(evidence, config)
    keep = scores >= config.threshold
    for index in range(len(peaks)):
        if geometry.conflicts[index, index + 1 :].any():
            keep[index + 1 :] &= ~(
                keep[index] & geometry.conflicts[index, index + 1 :]
            )
    events = []
    audits = []
    for index, peak in enumerate(peaks):
        onset, offset = (int(value) for value in geometry.intervals[index])
        fused = float(0.5 * (evidence[index, 1] + evidence[index, 2]))
        audits.append(
            CandidateAudit(
                peak=int(peak),
                onset=onset,
                offset=offset,
                height=float(evidence[index, 0]),
                global_score=float(evidence[index, 1]),
                local_score=float(evidence[index, 2]),
                fused_score=fused,
                decision_score=float(scores[index]),
                selected=bool(keep[index]),
            )
        )
        if keep[index]:
            events.append(
                {
                    "onset": onset,
                    "peak": int(peak),
                    "offset": offset,
                    "confidence": float(scores[index]),
                }
            )
    return events, audits
