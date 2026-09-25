"""Shared adapter record structures; no G/L logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..decoder import IntervalAdapter


@dataclass(frozen=True)
class Record:
    subject: str
    video: str
    score: np.ndarray
    ground_truth: list
    metadata: dict


@dataclass(frozen=True)
class DatasetBundle:
    backbone: str
    dataset: str
    records: list[Record]
    subjects: list[str]
    legacy_temporal_scale: int
    interval_adapter: IntervalAdapter
    cache_path: str
