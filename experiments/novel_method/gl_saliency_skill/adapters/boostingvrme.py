"""BoostingVRME frozen-cache schema and historical native interval metadata."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..decoder import Geometry
from .base import DatasetBundle, Record


def source_interval(curve: np.ndarray, peak: int, temporal_scale: int) -> tuple[int, int]:
    length = len(curve)
    left = max(0, peak - temporal_scale)
    right = min(length - 1, peak + temporal_scale)
    mean = float(curve.mean())
    start = next(
        (
            index - 1
            for index in range(left + 1, peak)
            if curve[index] > curve[index - 1] and curve[index] > mean / 1.5
        ),
        left,
    )
    end = next(
        (
            index + 1
            for index in range(right - 1, peak, -1)
            if curve[index] > curve[index + 1] and curve[index] > mean / 1.5
        ),
        right,
    )
    for index in range(start - 1, max(-1, peak - temporal_scale * 5) - 1, -1):
        if index > 0 and curve[index] < mean and curve[index - 1] < mean:
            start = index + 1
            break
    for index in range(end + 1, min(length - 1, peak + temporal_scale * 5) + 1):
        if index < length - 1 and curve[index] < mean and curve[index + 1] < mean:
            end = index - 1
            break
    return max(0, start), min(length - 1, end)


class BoostingNativeIntervalAdapter:
    name = "boostingvrme_native"

    def geometry(self, temporal_scores: np.ndarray, temporal_scale: int, peaks: np.ndarray) -> Geometry:
        raw_intervals = np.array(
            [source_interval(temporal_scores, int(peak), temporal_scale) for peak in peaks],
            dtype=int,
        ).reshape(-1, 2)
        order = np.array(
            sorted(
                range(len(peaks)),
                key=lambda index: (raw_intervals[index, 0], int(peaks[index])),
            ),
            dtype=int,
        )
        ordered_peaks = peaks[order]
        intervals = raw_intervals[order]
        count = len(ordered_peaks)
        conflicts = np.zeros((count, count), dtype=bool)
        if count:
            starts, ends = intervals.T
            intersection = np.maximum(
                0,
                np.minimum(ends[:, None], ends) - np.maximum(starts[:, None], starts),
            )
            span = np.maximum(ends[:, None], ends) - np.minimum(starts[:, None], starts)
            iou = np.divide(
                intersection,
                span,
                out=np.zeros_like(intersection, dtype=float),
                where=span > 0,
            )
            conflicts = (iou >= 0.2) | (
                np.abs(ordered_peaks[:, None] - ordered_peaks) <= temporal_scale
            )
        return Geometry(order=order, intervals=intervals, conflicts=conflicts)


def _duration_k(records: list[Record]) -> int:
    durations = sorted(
        event[2] - event[0]
        for record in records
        for event in record.ground_truth
    )
    return max(1, int((durations[len(durations) // 2] + 1) / 2))


def load_boostingvrme(path: Path, dataset: str) -> DatasetBundle:
    with path.open("rb") as handle:
        package = pickle.load(handle)
    records = [
        Record(
            subject=str(source["subject"]),
            video=str(source["video"]),
            score=np.asarray(package["curves"][index], dtype=float),
            ground_truth=[[int(value) for value in event] for event in source["gt"]],
            metadata={"emotion": source.get("emotions")},
        )
        for index, source in enumerate(package["records"])
    ]
    subjects = sorted({record.subject for record in records}, key=lambda value: (int(value), value))
    return DatasetBundle(
        backbone="boostingvrme",
        dataset=dataset,
        records=records,
        subjects=subjects,
        legacy_temporal_scale=_duration_k(records),
        interval_adapter=BoostingNativeIntervalAdapter(),
        cache_path=str(path.resolve()),
    )
