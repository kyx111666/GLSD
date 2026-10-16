#!/usr/bin/env python3
"""Shared, dependency-light protocol utilities for GLSD configuration transfer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SEED = 100
BOOTSTRAP_ITERATIONS = 10_000
THETA_FIELDS = ("reference_scale", "local_radius", "threshold")
FORBIDDEN_SOURCE_FIELDS = {
    "k", "k_p", "target_k", "source_k", "event_interval", "interval_strategy",
    "frame_skip", "boundary", "boundary_rule", "candidate_order", "conflict_rule",
}


DATASETS = {
    "sammlv": {"subjects": 29, "videos": 79, "gt": 159},
    "casme3": {"subjects": 94, "videos": 462, "gt": 853},
}


LOCKED = {
    "metst_sammlv": {
        "pipeline": "ME-TST+", "dataset": "sammlv",
        "native": (52, 171, 107), "tuned": (47, 121, 112),
    },
    "metst_casme3": {
        "pipeline": "ME-TST+", "dataset": "casme3",
        "native": (76, 736, 777), "tuned": (90, 898, 763),
    },
    "boosting_sammlv": {
        "pipeline": "BoostingVRME", "dataset": "sammlv",
        "native": (51, 141, 108), "tuned": (44, 82, 115),
    },
    "boosting_casme3": {
        "pipeline": "BoostingVRME", "dataset": "casme3",
        "native": (84, 786, 769), "tuned": (120, 1110, 733),
    },
}


DIRECTIONS = {
    "metst_to_boosting_sammlv": {
        "dataset": "sammlv", "source": "metst_sammlv", "target": "boosting_sammlv",
        "relative_output": "sammlv/metst_to_boosting",
    },
    "boosting_to_metst_sammlv": {
        "dataset": "sammlv", "source": "boosting_sammlv", "target": "metst_sammlv",
        "relative_output": "sammlv/boosting_to_metst",
    },
    "metst_to_boosting_casme3": {
        "dataset": "casme3", "source": "metst_casme3", "target": "boosting_casme3",
        "relative_output": "casme3/metst_to_boosting",
    },
    "boosting_to_metst_casme3": {
        "dataset": "casme3", "source": "boosting_casme3", "target": "metst_casme3",
        "relative_output": "casme3/boosting_to_metst",
    },
}


def f1(counts) -> float:
    tp, fp, fn = (int(value) for value in counts)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def metrics(counts) -> dict:
    tp, fp, fn = (int(value) for value in counts)
    return {"TP": tp, "FP": fp, "FN": fn, "F1": f1((tp, fp, fn))}


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def canonical_subject(dataset: str, value) -> str:
    text = str(value).strip()
    if not text:
        raise RuntimeError("empty subject ID")
    try:
        number = int(text)
    except ValueError as exc:
        raise RuntimeError(f"non-numeric {dataset} subject ID: {text!r}") from exc
    return f"{number:03d}" if dataset == "sammlv" else str(number)


def field_name(row: dict, candidates: tuple[str, ...], label: str) -> str:
    lowered = {name.lower(): name for name in row}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise RuntimeError(f"cannot resolve {label}; fields={list(row)}")


def selected_config_rows(path: Path, dataset: str) -> dict[str, dict]:
    rows = load_csv(path)
    subject_field = field_name(rows[0], ("outer_subject", "subject", "subject_id"), "subject")
    a0_field = field_name(rows[0], ("reference_scale", "a0"), "a0")
    rho_field = field_name(rows[0], ("local_radius", "rho"), "rho")
    tau_field = field_name(rows[0], ("threshold", "tau"), "tau")
    forbidden = {name.lower() for name in rows[0]} & FORBIDDEN_SOURCE_FIELDS
    if forbidden:
        raise RuntimeError(f"source selected-config table contains forbidden transfer fields: {sorted(forbidden)}")
    result = {}
    for row in rows:
        subject = canonical_subject(dataset, row[subject_field])
        if subject in result:
            raise RuntimeError(f"duplicate selected config for subject {subject}")
        result[subject] = {
            "source_subject_raw": str(row[subject_field]),
            "source_outer_fold": str(row.get("outer_subject_index", row[subject_field])),
            "a0": float(row[a0_field]),
            "rho": float(row[rho_field]),
            "tau": float(row[tau_field]),
            "source_config_origin": str(path),
        }
    return result


def normalize_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    return value


def pooled(rows: list[dict], prefix: str) -> tuple[int, int, int]:
    return tuple(sum(int(row[f"{prefix}_{name}"]) for row in rows) for name in ("TP", "FP", "FN"))


def paired_bootstrap(
    native: np.ndarray, tuned: np.ndarray, transferred: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS, seed: int = SEED,
) -> dict:
    arrays = [np.asarray(value, dtype=np.int64) for value in (native, tuned, transferred)]
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2 or arrays[0].shape[1] != 3:
        raise RuntimeError("paired bootstrap expects three aligned N x 3 count arrays")
    n_subjects = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    delta_native = np.empty(iterations, dtype=float)
    delta_tuned = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draw = rng.integers(0, n_subjects, size=n_subjects)
        sampled = [array[draw].sum(axis=0) for array in arrays]
        delta_native[index] = f1(sampled[2]) - f1(sampled[0])
        delta_tuned[index] = f1(sampled[2]) - f1(sampled[1])
    point_native = f1(arrays[2].sum(axis=0)) - f1(arrays[0].sum(axis=0))
    point_tuned = f1(arrays[2].sum(axis=0)) - f1(arrays[1].sum(axis=0))
    return {
        "iterations": iterations,
        "seed": seed,
        "unit": "subject",
        "aggregation": "pooled TP/FP/FN before F1",
        "paired_resampling": True,
        "vs_native": {
            "point_estimate": point_native,
            "percentile_2_5": float(np.percentile(delta_native, 2.5)),
            "percentile_97_5": float(np.percentile(delta_native, 97.5)),
        },
        "vs_tuned": {
            "point_estimate": point_tuned,
            "percentile_2_5": float(np.percentile(delta_tuned, 2.5)),
            "percentile_97_5": float(np.percentile(delta_tuned, 97.5)),
        },
    }
