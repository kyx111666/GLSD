'Paired subject bootstrap for the locked fusion-weight ablation.'
from __future__ import annotations


import argparse


import contextlib


import csv


import gzip


import json


import time


from dataclasses import asdict


from datetime import datetime, timezone


from pathlib import Path


import numpy as np


import one_to_one_evaluator as evaluator


import run_full_fusion_tuning as full


shared = full.shared


DEFAULT_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def paired_alpha_comparisons(counts_by_alpha, subjects, baseline: float):
    """Paired subject bootstrap of baseline alpha minus each alternative."""
    draws = np.random.default_rng(shared.SEED).integers(
        0, len(subjects), size=(shared.RESAMPLES, len(subjects)))
    distributions = {}
    for alpha, counts in counts_by_alpha.items():
        totals = counts[draws].sum(axis=1).astype(float)
        denom = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        distributions[alpha] = np.divide(
            2 * totals[:, 0], denom, out=np.zeros(shared.RESAMPLES), where=denom > 0)
    if baseline not in distributions:
        raise RuntimeError(f"baseline alpha={baseline} is absent from the grid")
    base_counts = counts_by_alpha[baseline].sum(axis=0)
    rows = []
    for alpha in counts_by_alpha:
        if alpha == baseline:
            continue
        delta = distributions[baseline] - distributions[alpha]
        lo, hi = np.quantile(delta, [.025, .975])
        rows.append(dict(
            target_alpha=baseline,
            reference_alpha=alpha,
            metric_stage="full",
            delta_F1=shared.metrics(base_counts)["F1"] - shared.metrics(counts_by_alpha[alpha].sum(axis=0))["F1"],
            CI_low=float(lo), CI_high=float(hi),
            seed=shared.SEED, resamples=shared.RESAMPLES,
            subjects=len(subjects),
        ))
    return rows
