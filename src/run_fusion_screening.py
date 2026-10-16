'Shared feature adapters and serialization for locked-alpha evaluation.'
from __future__ import annotations


import argparse


import contextlib


import csv


from dataclasses import asdict, dataclass


from datetime import datetime, timezone


import gzip


import hashlib


import importlib.util


import io


import json


from pathlib import Path


import sys


import time


from types import SimpleNamespace


import numpy as np


A0, RHO, GAMMA = 1.5, 2.0, 0.5


THRESHOLDS = tuple(i / 10 for i in range(1, 10))


ETAS = (0.25, 0.5, 0.75)


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


METHODS = ("G", "L", "EqualMean", "WeightedMean", "DualOR", "CrossSupported")


SETTINGS = {"sammlv": "metst_sammlv", "casme3": "metst_casme3"}


SEED, RESAMPLES = 100, 10_000


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    threshold: float = 1.0
    tau_g: float | None = None
    tau_l: float | None = None
    eta: float | None = None
    alpha: float | None = None
    reference_scale: float = A0
    local_radius: float = RHO

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def accept(g, l, config):
    g, l = np.asarray(g, dtype=float), np.asarray(l, dtype=float)
    if g.shape != l.shape or g.ndim != 1 or not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError("G/L must be finite aligned vectors")
    if np.any(g < -1e-12) or np.any(l < -1e-12) or np.any(g > 1 + 1e-12) or np.any(l > 1 + 1e-12):
        raise ValueError("G/L out of expected [0,1] range; no clipping is applied")
    if config.method == "G":
        return g >= config.threshold
    if config.method == "L":
        return l >= config.threshold
    if config.method == "EqualMean":
        return (g + l) / 2 >= config.threshold
    if config.method == "WeightedMean":
        return config.alpha * g + (1 - config.alpha) * l >= config.threshold
    if config.method in ("DualOR", "CrossSupported"):
        return ((g >= config.tau_g) & (l >= config.eta * config.tau_l)) | (
            (l >= config.tau_l) & (g >= config.eta * config.tau_g)
        )
    raise ValueError(config.method)


def metrics(counts):
    tp, fp, fn = map(int, counts)
    return dict(TP=tp, FP=fp, FN=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def choose(table, outer_index):
    """Exclude the held subject before all selection metrics and tie-breaks."""
    pooled = np.delete(table, outer_index, axis=1).sum(axis=1)
    m = [metrics(row) for row in pooled]
    winner = min(range(len(m)), key=lambda i: (-m[i]["F1"], -m[i]["precision"], m[i]["FP"], i))
    return winner, pooled


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(serializable(obj), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        peaks, evidence = base.evidence(A0, RHO)
        self.peaks = np.array(peaks, dtype=int, copy=True)
        self.values = np.array(evidence, dtype=float, copy=True)
        if self.values.shape != (len(self.peaks), 2):
            raise RuntimeError("official evidence must have two columns G,L")
        self.peaks.setflags(write=False)
        self.values.setflags(write=False)
        for t in THRESHOLDS:
            c = SimpleNamespace(reference_scale=A0, local_radius=RHO, threshold=t)
            expected = base.selected_peaks(c)
            got = self.peaks[accept(*self.values.T, Config("EqualMean", 0, threshold=t))]
            if not np.array_equal(got, expected):
                raise RuntimeError("new EqualMean hook does not exactly reproduce the sealed scorer")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def selected_peaks(self, config):
        if config.reference_scale != A0 or config.local_radius != RHO:
            raise RuntimeError("screening structure parameters changed")
        self.owner.calls += 1
        g, l = self.values.T
        keep = accept(g, l, config)
        if self.owner.capture:
            row = dict(feature_call_index=len(self.owner.trace), response_sha256=self.key[2],
                       temporal_scale=self.key[0], peaks=self.peaks.tolist(),
                       G=g.tolist(), L=l.tolist(), retained=keep.tolist())
            if config.method in ("DualOR", "CrossSupported"):
                row["global_branch"] = ((g >= config.tau_g) & (l >= config.eta * config.tau_l)).tolist()
                row["local_branch"] = ((l >= config.tau_l) & (g >= config.eta * config.tau_g)).tolist()
            self.owner.trace.append(row)
        return self.peaks[keep]


class FusionCore:
    """Reuse label-free official features, keyed by response content and k."""
    def __init__(self, base):
        self.base, self.cache = base, {}
        self.calls, self.capture, self.trace = 0, False, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def load_helper():
    name = "official_response_component_ablation.py"
    directory = Path(__file__).resolve().parent
    path = next((p for p in (directory / name, directory.parent / name) if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Place the unchanged {name} beside this script")
    spec = importlib.util.spec_from_file_location("fusion_screening_original_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path
