"""Exact historical grid pools, nested count aggregation, and tie-break."""

from __future__ import annotations

import numpy as np

from .evidence import Config


def indexed_pools(configs: list[Config]) -> dict[str, list[int]]:
    return {
        "GL": [
            index
            for index, config in enumerate(configs)
            if config.family == "unified" and config.height_weight == 0
        ],
        "G": [index for index, config in enumerate(configs) if config.family == "global"],
        "L": [index for index, config in enumerate(configs) if config.family == "local"],
    }


def choose(counts: np.ndarray, indexes: list[int]) -> int:
    indexes_array = np.asarray(indexes, dtype=int)
    tp, fp, fn = counts[indexes_array].astype(float).T
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return int(indexes_array[np.lexsort((indexes_array, fp, -precision, -f1))[0]])


def inner_counts(stats: dict[int, np.ndarray], held: int, inner_k: np.ndarray) -> np.ndarray:
    sample = stats[next(iter(stats))]
    total = np.zeros((sample.shape[0], 3), dtype=np.int64)
    for validation in range(len(inner_k)):
        if validation != held:
            total += stats[int(inner_k[held, validation])][:, validation]
    return total


def select_all(
    configs: list[Config],
    stats: dict[int, np.ndarray],
    subjects: list[str],
    outer_k: np.ndarray,
    inner_k: np.ndarray,
) -> tuple[dict[str, dict[str, int]], list[dict]]:
    pools = indexed_pools(configs)
    selected_by_subject = {}
    rows = []
    for held, subject in enumerate(subjects):
        training = inner_counts(stats, held, inner_k)
        selected = {name: choose(training, indexes) for name, indexes in pools.items()}
        selected_by_subject[subject] = selected
        for name in ("GL", "G", "L"):
            index = selected[name]
            tp, fp, fn = (int(value) for value in training[index])
            denominator = 2 * tp + fp + fn
            rows.append(
                {
                    "family": name,
                    "subject": subject,
                    "config_id": index,
                    "config": configs[index].identifier,
                    "k": int(outer_k[held]),
                    "inner_F1": 2 * tp / denominator if denominator else 0.0,
                }
            )
    return selected_by_subject, rows
