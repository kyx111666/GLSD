"""Subject-paired bootstrap for GL minus Native raw F1."""

from __future__ import annotations

import numpy as np

from .evaluation import f1


def paired_bootstrap(subject_rows: list[dict], seed: int = 100, resamples: int = 10_000) -> dict:
    rng = np.random.default_rng(seed)
    count = len(subject_rows)
    deltas = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        sample = rng.integers(0, count, size=count)
        native = [0, 0, 0]
        gl = [0, 0, 0]
        for index in sample:
            row = subject_rows[int(index)]
            for column, key in enumerate(("TP", "FP", "FN")):
                native[column] += int(row[f"Native_{key}"])
                gl[column] += int(row[f"GL_{key}"])
        deltas[iteration] = f1(*gl) - f1(*native)
    point_native = [sum(int(row[f"Native_{key}"]) for row in subject_rows) for key in ("TP", "FP", "FN")]
    point_gl = [sum(int(row[f"GL_{key}"]) for row in subject_rows) for key in ("TP", "FP", "FN")]
    return {
        "delta_f1": f1(*point_gl) - f1(*point_native),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "seed": seed,
        "resamples": resamples,
        "status": "recomputed_v2",
    }
