"""The one shared GL-Decoder-v2 implementation used by every backbone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evidence import Evidence, EvidenceExtractor


@dataclass(frozen=True, order=True)
class GLConfig:
    reference_scale: float
    radius: int
    threshold: float

    @property
    def config_id(self) -> str:
        return (
            f"scale={self.reference_scale:g}|radius={self.radius:d}|"
            f"threshold={self.threshold:g}"
        )

    def as_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "reference_scale": self.reference_scale,
            "radius": self.radius,
            "threshold": self.threshold,
        }


class GLDecoder:
    """Decode fixed candidates using G, L, or their exact arithmetic mean."""

    def __init__(self, local_scales: tuple[float, ...] = (1.0, 1.5, 2.0)) -> None:
        self.extractor = EvidenceExtractor(local_scales)

    def evidence(self, score: np.ndarray, temporal_scale: int, config: GLConfig) -> Evidence:
        return self.extractor.extract(
            score=score,
            temporal_scale=temporal_scale,
            reference_scale=config.reference_scale,
            radius=config.radius,
        )

    @staticmethod
    def fused_scores(evidence: Evidence) -> tuple[float, ...]:
        return tuple(
            (global_score + local_score) / 2.0
            for global_score, local_score in zip(
                evidence.global_scores, evidence.local_scores, strict=True
            )
        )

    @staticmethod
    def events_from_evidence(
        evidence: Evidence,
        temporal_scale: int,
        threshold: float,
        mode: str = "GL",
    ) -> list[dict]:
        if mode == "G":
            scores = evidence.global_scores
        elif mode == "L":
            scores = evidence.local_scores
        elif mode == "GL":
            scores = GLDecoder.fused_scores(evidence)
        else:
            raise ValueError(f"Unsupported evidence mode: {mode}")
        return [
            {
                "peak": int(peak),
                "onset": int(peak - temporal_scale),
                "offset": int(peak + temporal_scale),
                "score": float(value),
                "G": float(global_score),
                "L": float(local_score),
            }
            for peak, value, global_score, local_score in zip(
                evidence.candidates,
                scores,
                evidence.global_scores,
                evidence.local_scores,
                strict=True,
            )
            if value >= threshold
        ]

    def decode(
        self,
        score: np.ndarray,
        temporal_scale: int,
        config: GLConfig,
        mode: str = "GL",
    ) -> tuple[list[dict], Evidence]:
        evidence = self.evidence(score, temporal_scale, config)
        return (
            self.events_from_evidence(evidence, temporal_scale, config.threshold, mode),
            evidence,
        )
