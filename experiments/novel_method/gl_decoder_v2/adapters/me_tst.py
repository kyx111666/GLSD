"""ME-TST cache schema adapter."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_me_tst(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    temporal_scale = int(payload["k_p"])
    records = []
    for source in payload["records"]:
        records.append(
            {
                "subject": str(source["subject"]),
                "video": str(source["video"]),
                "score": np.asarray(source["score"], dtype=float),
                "gt": list(source["samples"]),
                "temporal_scale": temporal_scale,
            }
        )
    return records
