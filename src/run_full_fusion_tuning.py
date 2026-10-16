'Multi-scale feature views used by locked-alpha evaluation.'
from __future__ import annotations


import argparse


import csv


import contextlib


import gzip


import hashlib


import io


import json


import time


from collections import Counter


from dataclasses import asdict, dataclass, replace


from datetime import datetime, timezone


from pathlib import Path


from types import SimpleNamespace


import numpy as np


import run_fusion_screening as shared


SCALES = (1.0, 1.5, 2.0)


RADII = (1.0, 2.0, 3.0)


LEGACY_SCALES, LEGACY_RADII = SCALES, RADII


THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)


LEGACY_THRESHOLDS = THRESHOLDS


ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


BETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


METHODS = ("G", "L", "EqualMean", "WeightedMean", "DominantEvidence")


MEAN_SELECTION_EPSILON = 0.002


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    reference_scale: float
    local_radius: float
    threshold: float
    alpha: float | None = None
    beta: float | None = None

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def grids():
    result = {}
    for method in METHODS:
        grid = []
        weights = ALPHAS if method == "WeightedMean" else BETAS if method == "DominantEvidence" else (None,)
        for a0 in SCALES:
            # G is independent of rho. 1.0 preserves the first legacy grid tie.
            for rho in ((1.0,) if method == "G" else RADII):
                for weight in weights:
                    for tau in THRESHOLDS:
                        grid.append(Config(method, len(grid), a0, rho, tau,
                                           alpha=weight if method == "WeightedMean" else None,
                                           beta=weight if method == "DominantEvidence" else None))
        # Retain the old structure first for exact ties, including when a new
        # decimal scale maps to an already represented integer window.
        grid.sort(key=lambda c: not (c.reference_scale in LEGACY_SCALES and c.local_radius in LEGACY_RADII))
        result[method] = [replace(config, config_id=index) for index, config in enumerate(grid)]
    return result


def scores(g, l, config):
    g, l = np.asarray(g, dtype=float), np.asarray(l, dtype=float)
    if g.shape != l.shape or g.ndim != 1 or not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError("Expected aligned finite G/L vectors")
    if np.any(g < -1e-12) or np.any(l < -1e-12) or np.any(g > 1 + 1e-12) or np.any(l > 1 + 1e-12):
        raise ValueError("G/L outside [0,1]; no clipping is performed")
    if config.method == "G":
        return g
    if config.method == "L":
        return l
    if config.method == "EqualMean":
        return (g + l) / 2.0
    if config.method == "WeightedMean":
        if config.alpha not in ALPHAS:
            raise ValueError("Invalid alpha")
        return config.alpha * g + (1 - config.alpha) * l
    if config.method == "DominantEvidence":
        if config.beta not in BETAS:
            raise ValueError("Invalid beta")
        # Preserve the legacy expression exactly at the nested endpoint.
        if config.beta == 0.5:
            return (g + l) / 2.0
        if config.beta == 1.0:
            return np.maximum(g, l)
        return config.beta * np.maximum(g, l) + (1 - config.beta) * np.minimum(g, l)
    raise ValueError(config.method)


class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        self.evidence_cache = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evidence(self, a0, rho):
        if a0 not in SCALES or rho not in RADII:
            raise ValueError("Structure parameter outside declared grid")
        if (a0, rho) not in self.evidence_cache:
            peaks, evidence = self.base.evidence(a0, rho)
            peaks = np.array(peaks, dtype=int, copy=True)
            values = np.array(evidence, dtype=float, copy=True)
            if values.shape != (len(peaks), 2):
                raise RuntimeError("Sealed evidence must have columns G,L")
            for tau in THRESHOLDS:
                config = Config("EqualMean", 0, a0, rho, tau)
                got = peaks[scores(*values.T, config) >= tau]
                expected = self.base.selected_peaks(config)
                if not np.array_equal(got, expected):
                    raise RuntimeError(f"Mean hook mismatch: response={self.key}, a0={a0}, rho={rho}, tau={tau}")
            peaks.setflags(write=False)
            values.setflags(write=False)
            self.evidence_cache[a0, rho] = peaks, values
        return self.evidence_cache[a0, rho]

    def selected_peaks(self, config):
        peaks, evidence = self.evidence(config.reference_scale, config.local_radius)
        g, l = evidence.T
        score = scores(g, l, config)
        keep = score >= config.threshold
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(
                feature_call_index=len(self.owner.trace), response_sha256=self.key[2],
                temporal_scale=self.key[0], reference_scale=config.reference_scale,
                local_radius=config.local_radius, peaks=peaks.tolist(), G=g.tolist(),
                L=l.tolist(), score=score.tolist(), retained=keep.tolist(),
            ))
        return peaks[keep]


class FusionCore(shared.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def decode_with_context(context, core, subject_index, subject, config, full, setting, method):
    """Add fold/config provenance to failures from the sealed evaluator."""
    try:
        runner, _base, metric, official, records, _paths, _subjects, _native = context
        before = core.calls
        core.trace, core.capture = [], full
        with contextlib.redirect_stdout(io.StringIO()):
            raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
                records, subject_index, config, core, metric, official, full
            )
            result = runner.full_counts_from_official_synergy(raw, pred_list, gt_list) if full else None
        raw = tuple(map(int, raw))
        result = tuple(map(int, result)) if result is not None else None
        if core.calls == before:
            raise RuntimeError("sealed decoder bypassed fusion hook")
        if len(raw) != 3 or min(raw) < 0 or (result is not None and (len(result) != 3 or min(result) < 0)):
            raise RuntimeError(f"invalid official counts raw={raw!r} result={result!r}")
        return raw, result, predictions, core.trace
    except Exception as exc:
        details = dict(setting=setting, method=method, subject_index=subject_index,
                       subject=subject, config=asdict(config), full=full)
        raise RuntimeError(f"official decoder/count failure: {json.dumps(details, sort_keys=True)}") from exc




def choose_mean_synergy(mean_table, g_table, l_table, outer_index,
                        epsilon=MEAN_SELECTION_EPSILON):
    """Select Mean on inner subjects with a frozen near-optimal synergy rule.

    First retain Mean configurations within ``epsilon`` F1 of the best inner
    Mean configuration. Among those rows, maximize the smaller of the Mean
    F1 margins over the same-structure G and L rows. Final ties use the
    ordinary Mean precision, FP, and config-index order. The held-out subject
    is excluded before every calculation.
    """
    mean_pooled = np.delete(mean_table, outer_index, axis=1).sum(axis=1)
    g_pooled = np.delete(g_table, outer_index, axis=1).sum(axis=1)
    l_pooled = np.delete(l_table, outer_index, axis=1).sum(axis=1)
    if mean_pooled.shape != l_pooled.shape or mean_pooled.shape[0] != 90:
        raise ValueError("Mean/G/L synergy selection expects the legacy 90-row structure grid")
    mapping = np.asarray([(i // 30) * 10 + (i % 10) for i in range(90)], dtype=int)
    g_same = g_pooled[mapping]
    mean_metrics = [shared.metrics(row) for row in mean_pooled]
    g_metrics = [shared.metrics(row) for row in g_same]
    l_metrics = [shared.metrics(row) for row in l_pooled]
    best_f1 = max(m["F1"] for m in mean_metrics)
    eligible = [i for i, m in enumerate(mean_metrics) if m["F1"] >= best_f1 - epsilon]
    margins = {i: min(mean_metrics[i]["F1"] - g_metrics[i]["F1"],
                       mean_metrics[i]["F1"] - l_metrics[i]["F1"])
               for i in eligible}
    max_margin = max(margins.values())
    candidates = [i for i in eligible if abs(margins[i] - max_margin) <= 1e-15]
    winner = min(candidates, key=lambda i: (-mean_metrics[i]["F1"],
                                            -mean_metrics[i]["precision"],
                                            mean_metrics[i]["FP"], i))
    validation = dict(selection_rule="near_optimal_mean_synergy_v1",
                 epsilon=float(epsilon), best_inner_mean_F1=float(best_f1),
                 eligible_count=len(eligible), max_synergy_margin=float(max_margin),
                 selected_synergy_margin=float(margins[winner]),
                 selected_config_id=int(winner),
                 eligible_config_ids=[int(i) for i in eligible])
    return winner, mean_pooled, validation
