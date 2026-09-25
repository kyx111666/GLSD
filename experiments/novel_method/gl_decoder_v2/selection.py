"""Leakage-free outer-subject LOSO selection."""

from __future__ import annotations

from fractions import Fraction

from .decoder import GLConfig


def enumerate_configs(scales: list[float], radii: list[int], thresholds: list[float]):
    return [
        GLConfig(float(scale), int(radius), float(threshold))
        for scale in scales
        for radius in radii
        for threshold in thresholds
    ]


def select_config(
    configs: list[GLConfig],
    per_config_subject_counts: dict[str, dict[str, dict]],
    outer_subject: str,
) -> tuple[GLConfig, dict]:
    ranked: list[tuple[tuple, GLConfig, dict]] = []
    for config in configs:
        training = [
            counts
            for subject, counts in per_config_subject_counts[config.config_id].items()
            if subject != outer_subject
        ]
        tp = sum(int(row["TP"]) for row in training)
        fp = sum(int(row["FP"]) for row in training)
        fn = sum(int(row["FN"]) for row in training)
        denominator = 2 * tp + fp + fn
        exact_f1 = Fraction(2 * tp, denominator) if denominator else Fraction(0, 1)
        prediction_count = tp + fp
        key = (-exact_f1, fp, prediction_count, config.config_id)
        ranked.append((key, config, {"TP": tp, "FP": fp, "FN": fn, "F1": float(exact_f1)}))
    _, selected, inner = min(ranked, key=lambda item: item[0])
    return selected, inner
