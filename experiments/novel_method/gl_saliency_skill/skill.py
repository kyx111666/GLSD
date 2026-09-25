"""Public inference API for the exact historical Global-Local decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoder import CandidateAudit, IntervalAdapter, decode_candidates
from .evidence import Config


@dataclass(frozen=True)
class GLSaliencySkill:
    config: Config

    def __post_init__(self) -> None:
        allowed = self.config.family in ("native", "height", "global", "local")
        pure_gl = self.config.family == "unified" and self.config.height_weight == 0
        if not (allowed or pure_gl):
            raise ValueError("Only historical Native/G/L/pure-G+L configurations are supported")

    @property
    def trainable_parameters(self) -> int:
        return 0

    def decode(
        self,
        temporal_scores,
        temporal_scale: int,
        metadata: dict,
    ) -> list[dict]:
        events, _ = self.decode_with_audit(temporal_scores, temporal_scale, metadata)
        return events

    def decode_with_audit(
        self,
        temporal_scores,
        temporal_scale: int,
        metadata: dict,
    ) -> tuple[list[dict], list[CandidateAudit]]:
        adapter: IntervalAdapter = metadata["interval_adapter"]
        events, candidates = decode_candidates(
            np.asarray(temporal_scores, dtype=float),
            int(temporal_scale),
            self.config,
            adapter,
        )
        optional = {
            key: metadata[key]
            for key in ("emotion", "class")
            if key in metadata and metadata[key] is not None
        }
        if optional:
            events = [{**event, **optional} for event in events]
        return events, candidates
