"""ME-TST frozen-cache schema and historical fixed-interval adapter."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..decoder import Geometry
from .base import DatasetBundle, Record


class FixedIntervalAdapter:
    name = "metst_fixed"

    def geometry(self, temporal_scores: np.ndarray, temporal_scale: int, peaks: np.ndarray) -> Geometry:
        order = np.argsort(peaks, kind="stable")
        ordered = peaks[order]
        intervals = np.array(
            [
                (
                    max(0, int(peak) - temporal_scale),
                    min(len(temporal_scores) - 1, int(peak) + temporal_scale),
                )
                for peak in ordered
            ],
            dtype=int,
        ).reshape(-1, 2)
        return Geometry(
            order=order,
            intervals=intervals,
            conflicts=np.zeros((len(ordered), len(ordered)), dtype=bool),
        )


def load_metst(path: Path, dataset: str) -> DatasetBundle:
    with path.open("rb") as handle:
        package = pickle.load(handle)
    records = [
        Record(
            subject=str(source["subject"]),
            video=str(source["video"]),
            score=np.asarray(source["score"], dtype=float),
            ground_truth=[[int(value) for value in event] for event in source["samples"]],
            metadata={},
        )
        for source in package["records"]
    ]
    subjects = sorted({record.subject for record in records}, key=lambda value: (int(value), value))
    return DatasetBundle(
        backbone="metst",
        dataset=dataset,
        records=records,
        subjects=subjects,
        legacy_temporal_scale=int(package["k_p"]),
        interval_adapter=FixedIntervalAdapter(),
        cache_path=str(path.resolve()),
    )
