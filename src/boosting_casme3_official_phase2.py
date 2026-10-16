'Native-count and saved-count adapters for CASME3 configuration transfer.'
from __future__ import annotations


import argparse


import contextlib


import csv


import hashlib


import importlib.util


import io


import json


import sys


from datetime import datetime, timezone


from pathlib import Path


from types import SimpleNamespace


import numpy as np


EXPECTED_SUBJECTS = 94


EXPECTED_VIDEOS = 462


EXPECTED_GT = 853


EXPECTED_NATIVE_FULL = (84, 786, 769)


EXPECTED_GLSD_RAW = (120, 1113, 733)


EXPECTED_GLSD_FULL = (120, 1110, 733)


from release_config import paths


_RELEASE_INPUTS = paths("boosting_casme3")


DEFAULT_RUNNER = _RELEASE_INPUTS["source"]


DEFAULT_REPO_ROOT = _RELEASE_INPUTS["upstream"]


DEFAULT_CACHE = _RELEASE_INPUTS["responses"]


DEFAULT_EVIDENCE = _RELEASE_INPUTS["evidence"]


DEFAULT_PREFLIGHT = Path(__file__).with_name(
    "boosting_casme3_official_response_preflight.py"
)


def full_counts_from_recognition(raw, pred_list, gt_list, cache_config):
    neutral = int(cache_config["emotion_type"]) - 1
    neutral_tp = 0
    neutral_fp = 0
    for prediction, target in zip(pred_list, gt_list):
        if int(prediction) == neutral:
            if int(target) == -1:
                neutral_fp += 1
            else:
                neutral_tp += 1
    return (
        int(raw[0]) - neutral_tp,
        int(raw[1]) - neutral_fp,
        int(raw[2]) + neutral_tp,
    )


def native_subject_counts(
    runner, cache_config, grouped, final_samples, final_emotions, subject_index,
):
    videos = grouped[subject_index]
    responses = [np.asarray(video["spot_response"], dtype=float) for video in videos]
    sequences = [np.asarray(video["recognition_sequence"]) for video in videos]
    metric_final = runner.tu.MeanAveragePrecision2d(num_classes=1)
    predictions, _, total_gt, metric_video, metric_final = runner.tu.spotting(
        final_samples, subject_index, responses, 0, metric_final,
        int(cache_config["k_p"]), int(cache_config["interval_strategy"]), runner.DATASET,
    )
    raw = tuple(int(value) for value in runner.official_metric_counts(metric_final, total_gt))
    with contextlib.redirect_stdout(io.StringIO()):
        pred_list, _, gt_list, _, _ = runner.tu.recognition(
            sequences, predictions, metric_video, final_emotions, subject_index,
            [], [], final_samples, [], [], int(cache_config["frame_skip"]),
            int(cache_config["penalty_strategy"]),
        )
    full = full_counts_from_recognition(raw, pred_list, gt_list, cache_config)
    return raw, full


def load_locked_glsd_counts(path: Path, subjects: list[str]):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(subjects):
        raise RuntimeError("CAS(ME)3 sealed per-subject row count mismatch")
    raw, full = [], []
    for index, row in enumerate(rows):
        if int(row["outer_subject_index"]) != index or str(row["outer_subject"]) != subjects[index]:
            raise RuntimeError(f"CAS(ME)3 sealed subject order mismatch at {index}")
        raw.append(tuple(int(row[f"raw_{name}"]) for name in ("tp", "fp", "fn")))
        full.append(tuple(int(row[f"full_{name}"]) for name in ("tp", "fp", "fn")))
    return np.asarray(raw, dtype=np.int64), np.asarray(full, dtype=np.int64)
