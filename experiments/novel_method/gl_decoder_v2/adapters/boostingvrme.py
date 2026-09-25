"""BoostingVRME current-cache schema adapter."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_boostingvrme(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    temporal_scale = int(payload["k_p"])
    records = []
    for subject in payload["subject_curves"]:
        for index, score in enumerate(subject["score"]):
            records.append(
                {
                    "subject": str(subject["subject"]),
                    "video": str(subject["videos"][index]),
                    "score": np.asarray(score, dtype=float),
                    "gt": list(subject["samples"][index]),
                    "temporal_scale": temporal_scale,
                }
            )
    return records
