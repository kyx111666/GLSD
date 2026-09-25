"""Pure protocol helpers for BoostingVRME official-response Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODES = ("H", "G", "L", "G+L")
SCALES = (1.0, 1.5, 2.0)
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.75)
FIXED_RHO = 2.0
BOOTSTRAP_SEED = 100
BOOTSTRAP_ITERATIONS = 10_000


@dataclass(frozen=True)
class ComponentConfig:
    config_id: int
    reference_scale: float
    threshold: float
    local_radius: float = FIXED_RHO

    @property
    def identifier(self) -> str:
        return f"a0={self.reference_scale:g}|tau={self.threshold:g}"


def component_grid() -> list[ComponentConfig]:
    grid = [
        ComponentConfig(index, scale, threshold)
        for index, (scale, threshold) in enumerate(
            (scale, threshold) for scale in SCALES for threshold in THRESHOLDS
        )
    ]
    if len(grid) != 30:
        raise AssertionError("matched-budget component grid must contain 30 configs")
    return grid


def height_scores(curve, peaks) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if curve.ndim != 1 or np.any(peaks < 0) or np.any(peaks >= len(curve)):
        raise ValueError("invalid reference curve or candidate indexes")
    if not len(peaks):
        return np.empty(0, dtype=float)
    centre = float(np.mean(curve))
    denominator = max(float(np.max(curve) - centre), 1e-12)
    return np.maximum(0.0, (curve[peaks] - centre) / denominator)


def component_scores(mode: str, curve, peaks, global_values, local_values) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"unknown component mode: {mode}")
    peaks = np.asarray(peaks, dtype=int)
    global_values = np.asarray(global_values, dtype=float)
    local_values = np.asarray(local_values, dtype=float)
    if not (len(peaks) == len(global_values) == len(local_values)):
        raise ValueError("H/G/L evidence cardinality mismatch")
    if mode == "H":
        return height_scores(curve, peaks)
    if mode == "G":
        return global_values
    if mode == "L":
        return local_values
    return (global_values + local_values) / 2.0


def threshold_candidates(peaks, scores, threshold: float) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(peaks) != len(scores):
        raise ValueError("candidate/score cardinality mismatch")
    return peaks[scores >= float(threshold)]


def counts_tuple(counts) -> tuple[int, int, int]:
    result = tuple(int(value) for value in counts)
    if len(result) != 3 or any(value < 0 for value in result):
        raise ValueError(f"invalid TP/FP/FN counts: {result}")
    return result


def pool_counts(rows) -> tuple[int, int, int]:
    array = np.asarray(list(rows), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3 or np.any(array < 0):
        raise ValueError("counts must have shape (N, 3) and be nonnegative")
    return counts_tuple(array.sum(axis=0))


def metrics(counts) -> dict:
    tp, fp, fn = counts_tuple(counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def selection_order(pooled_raw_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(pooled_raw_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 3 or np.any(counts < 0):
        raise ValueError("pooled raw counts must have shape (configs, 3)")
    tp, fp, fn = counts.astype(float).T
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    grid_order = np.arange(len(counts))
    return np.lexsort((grid_order, fp, -precision, -f1))


def nested_loso_selection(raw_table: np.ndarray, subjects: list[str], grid):
    table = np.asarray(raw_table, dtype=np.int64)
    expected = (len(grid), len(subjects), 3)
    if table.shape != expected or np.any(table < 0):
        raise ValueError(f"raw count table shape={table.shape}, expected={expected}")
    if len(grid) != 30:
        raise AssertionError("component selection requires exactly 30 configs")

    winners = []
    winner_rows = []
    traces = []
    for outer, subject in enumerate(subjects):
        inner = [index for index in range(len(subjects)) if index != outer]
        if len(inner) != len(subjects) - 1 or outer in inner:
            raise AssertionError("invalid nested LOSO inner subject universe")
        pooled = table[:, inner, :].sum(axis=1)
        order = selection_order(pooled)
        winner = int(order[0])
        ranks = np.empty(len(grid), dtype=np.int64)
        ranks[order] = np.arange(1, len(grid) + 1)
        winner_metric = metrics(pooled[winner])
        winners.append(winner)
        winner_rows.append({
            "outer_subject_index": outer,
            "outer_subject": subject,
            "inner_subject_count": len(inner),
            "config_id": winner,
            "reference_scale": grid[winner].reference_scale,
            "threshold": grid[winner].threshold,
            "rho": FIXED_RHO,
            "pooled_inner_raw_TP": winner_metric["TP"],
            "pooled_inner_raw_FP": winner_metric["FP"],
            "pooled_inner_raw_FN": winner_metric["FN"],
            "pooled_inner_raw_precision": winner_metric["precision"],
            "pooled_inner_raw_recall": winner_metric["recall"],
            "pooled_inner_raw_F1": winner_metric["F1"],
            "winner_rank": int(ranks[winner]),
        })
        candidates = []
        for config_id, config in enumerate(grid):
            value = metrics(pooled[config_id])
            candidates.append({
                "config_id": config_id,
                "reference_scale": config.reference_scale,
                "threshold": config.threshold,
                "rho": FIXED_RHO,
                "pooled_inner_raw_TP": value["TP"],
                "pooled_inner_raw_FP": value["FP"],
                "pooled_inner_raw_FN": value["FN"],
                "precision": value["precision"],
                "recall": value["recall"],
                "F1": value["F1"],
                "rank": int(ranks[config_id]),
                "selected": config_id == winner,
            })
        traces.append({
            "outer_subject_index": outer,
            "outer_subject": subject,
            "inner_subject_indices": inner,
            "inner_subjects": [subjects[index] for index in inner],
            "tie_break": [
                "higher_raw_F1",
                "higher_raw_precision",
                "fewer_raw_FP",
                "fixed_grid_order",
            ],
            "winner_config_id": winner,
            "candidates": candidates,
        })
    return winners, winner_rows, traces


def component_summaries(raw_rows, full_rows) -> tuple[dict, dict]:
    raw_total = pool_counts(raw_rows)
    full_total = pool_counts(full_rows)
    raw_summary = {"metric_stage": "raw_audit", **metrics(raw_total)}
    full_summary = {"metric_stage": "full_official_recognition_synergy", **metrics(full_total)}
    if full_summary["TP"] != full_total[0] or full_summary["FP"] != full_total[1]:
        raise AssertionError("full summary was not computed from full counts")
    return raw_summary, full_summary


def paired_subject_bootstrap(
    native_full: np.ndarray,
    glsd_full: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    native = np.asarray(native_full, dtype=np.int64)
    glsd = np.asarray(glsd_full, dtype=np.int64)
    if native.shape != glsd.shape or native.ndim != 2 or native.shape[1] != 3:
        raise ValueError("paired full-count arrays must share shape (subjects, 3)")
    if len(native) < 2 or np.any(native < 0) or np.any(glsd < 0):
        raise ValueError("invalid subject-level full counts")
    if iterations != 10_000 or seed != 100:
        raise AssertionError("locked bootstrap requires iterations=10000 and seed=100")

    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(native), size=(iterations, len(native)))
    native_totals = native[indexes].sum(axis=1)
    glsd_totals = glsd[indexes].sum(axis=1)

    def vector_f1(totals):
        totals = totals.astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator,
            out=np.zeros(len(totals), dtype=float), where=denominator > 0,
        )

    native_samples = vector_f1(native_totals)
    glsd_samples = vector_f1(glsd_totals)
    delta_samples = glsd_samples - native_samples
    point_delta = metrics(glsd.sum(axis=0))["F1"] - metrics(native.sum(axis=0))["F1"]
    return {
        "indexes": indexes,
        "native_f1_samples": native_samples,
        "glsd_f1_samples": glsd_samples,
        "delta_f1_samples": delta_samples,
        "summary": {
            "statistic": "pooled_full_F1_GLSD_minus_Native",
            "point_delta_F1": point_delta,
            "percentile_2_5": float(np.quantile(delta_samples, 0.025)),
            "percentile_97_5": float(np.quantile(delta_samples, 0.975)),
            "seed": seed,
            "iterations": iterations,
            "unit": "subject",
            "paired": True,
            "aggregation": "sample subjects, pool full TP/FP/FN, then compute F1",
        },
    }


def run_protocol_smoke_assertions() -> None:
    grid = component_grid()
    assert len(grid) == 30
    assert [(item.reference_scale, item.threshold) for item in grid] == [
        (scale, threshold) for scale in SCALES for threshold in THRESHOLDS
    ]

    curve = np.asarray([0.0, 1.0, 3.0, 1.0, 0.0])
    peaks = np.asarray([2])
    h = height_scores(curve, peaks)
    assert np.array_equal(h, np.asarray([1.0]))
    g = np.asarray([0.7])
    local = np.asarray([0.3])
    assert np.array_equal(component_scores("H", curve, peaks, g, local), h)
    assert np.array_equal(component_scores("G", curve, peaks, g, local), g)
    assert np.array_equal(component_scores("L", curve, peaks, g, local), local)
    assert np.array_equal(component_scores("G+L", curve, peaks, g, local), np.asarray([0.5]))

    subjects = ["s0", "s1", "s2"]
    table = np.full((30, 3, 3), (1, 9, 9), dtype=np.int64)
    table[0] = np.asarray([(2, 8, 8), (2, 8, 8), (2, 8, 8)])
    table[1] = np.asarray([(2, 7, 9), (2, 7, 9), (2, 7, 9)])
    winners, rows, _ = nested_loso_selection(table, subjects, grid)
    assert winners == [1, 1, 1]
    assert all(row["inner_subject_count"] == 2 for row in rows)

    raw = np.asarray([(2, 8, 8), (1, 9, 9), (0, 10, 10)])
    full = np.asarray([(1, 6, 9), (1, 7, 9), (0, 8, 10)])
    raw_summary, full_summary = component_summaries(raw, full)
    assert (raw_summary["TP"], raw_summary["FP"], raw_summary["FN"]) == (3, 27, 27)
    assert (full_summary["TP"], full_summary["FP"], full_summary["FN"]) == (2, 21, 28)
    assert full_summary["F1"] == metrics((2, 21, 28))["F1"]

    boot = paired_subject_bootstrap(full, full + np.asarray((1, 0, 0)))
    indexes = boot["indexes"]
    first_native = full[indexes[0]].sum(axis=0)
    first_glsd = (full + np.asarray((1, 0, 0)))[indexes[0]].sum(axis=0)
    expected_delta = metrics(first_glsd)["F1"] - metrics(first_native)["F1"]
    assert boot["delta_f1_samples"][0] == expected_delta
    assert boot["indexes"].shape == (10_000, len(full))
    assert boot["summary"]["paired"] is True
    assert boot["summary"]["unit"] == "subject"
