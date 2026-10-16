'Evaluation invariant helpers used by the regression suite.'
import argparse


import contextlib


from dataclasses import asdict


from datetime import datetime, timezone


import io


from pathlib import Path


import numpy as np


import run_full_fusion_tuning as full


import one_to_one_evaluator as evaluator


shared = full.shared


def configure_threshold_grid(mode):
    if mode == "legacy":
        full.THRESHOLDS = full.LEGACY_THRESHOLDS
    elif mode == "refined":
        full.THRESHOLDS = tuple(i / 100 for i in range(5, 100, 5))
    else:
        raise ValueError(mode)


def validate(raw, final, gt, label):
    for stage, values in (("raw", raw), ("full", final)):
        if values is None:
            continue
        values = tuple(values)
        if len(values) != 3 or any(not np.isfinite(v) or v < 0 or int(v) != v for v in values):
            raise RuntimeError(f"Invalid {label} {stage} counts: {values}")
        if values[0] + values[2] != gt or values[0] > gt:
            raise RuntimeError(f"GT conservation failed for {label} {stage}: {values}, GT={gt}")


def subject_gt(context, index):
    runner, _, _, _, records, *_ = context
    return sum(len(v) for v in runner.final_samples(records)[index])
