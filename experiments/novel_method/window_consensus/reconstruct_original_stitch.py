#!/usr/bin/env python3
"""Reconstruct original ME-TST+ stitching solely from a raw window dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_SCHEMA = "sammlv_raw_prestitch_window_dump_v1"
REQUIRED_KEYS = {
    "schema", "subject", "video", "batch_size", "window_length", "stride",
    "subject_flat_window_index", "batch_index", "index_within_batch", "video_window_id",
    "framecount_before_write", "global_start", "global_indices",
    "raw_window_score", "raw_window_logits", "original_fresh_result_video",
    "original_fresh_logits_video",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def scalar(data, key: str):
    return np.asarray(data[key]).item()


def validate_dump(data) -> None:
    missing = sorted(REQUIRED_KEYS.difference(data.keys()))
    if missing:
        raise KeyError(f"dump is missing keys: {missing}")
    if str(scalar(data, "schema")) != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected schema: {scalar(data, 'schema')!r}")

    batch_size = int(scalar(data, "batch_size"))
    window_length = int(scalar(data, "window_length"))
    stride = int(scalar(data, "stride"))
    raw_score = np.asarray(data["raw_window_score"])
    raw_logits = np.asarray(data["raw_window_logits"])
    n_windows = len(raw_score)
    if (batch_size, window_length, stride) != (32, 30, 15):
        raise ValueError(
            "frozen geometry changed: "
            f"batch_size={batch_size}, window_length={window_length}, stride={stride}"
        )
    if raw_score.shape != (n_windows, window_length):
        raise ValueError(f"invalid raw score shape: {raw_score.shape}")
    if raw_logits.shape[:2] != raw_score.shape or raw_logits.ndim != 3:
        raise ValueError(f"invalid raw logits shape: {raw_logits.shape}")

    for key in (
        "batch_index", "index_within_batch", "video_window_id",
        "framecount_before_write", "global_start",
    ):
        if np.asarray(data[key]).shape != (n_windows,):
            raise ValueError(f"invalid {key} shape: {np.asarray(data[key]).shape}")
    if np.asarray(data["global_indices"]).shape != (n_windows, window_length):
        raise ValueError(f"invalid global_indices shape: {np.asarray(data['global_indices']).shape}")

    expected_global = (
        np.asarray(data["video_window_id"], dtype=np.int64)[:, None] * stride
        + np.arange(window_length, dtype=np.int64)[None, :]
    )
    if not np.array_equal(np.asarray(data["global_indices"]), expected_global):
        raise ValueError("global_indices != video_window_id * stride + local_t")
    if not np.array_equal(np.asarray(data["global_start"]), expected_global[:, 0]):
        raise ValueError("global_start does not match global_indices[:, 0]")
    if not np.array_equal(
        np.asarray(data["framecount_before_write"]),
        np.asarray(data["video_window_id"]),
    ):
        raise ValueError("framecount_before_write differs from video_window_id")
    subject_flat_index = np.asarray(data["subject_flat_window_index"])
    expected_batch_index = subject_flat_index // batch_size
    expected_batch_local = subject_flat_index % batch_size
    if not np.array_equal(np.asarray(data["batch_index"]), expected_batch_index):
        raise ValueError("batch_index is inconsistent with subject-level batching")
    if not np.array_equal(
        np.asarray(data["index_within_batch"]), expected_batch_local
    ):
        raise ValueError("index_within_batch is inconsistent with subject-level batching")


def reconstruct(data) -> tuple[np.ndarray, np.ndarray]:
    """Apply the original batch-local branch without using fresh stitched data."""
    validate_dump(data)
    stride = int(scalar(data, "stride"))
    raw_score = np.asarray(data["raw_window_score"])
    raw_logits = np.asarray(data["raw_window_logits"])
    framecounts = np.asarray(data["framecount_before_write"], dtype=np.int64)
    batch_local = np.asarray(data["index_within_batch"], dtype=np.int64)
    original_score = np.asarray(data["original_fresh_result_video"])
    original_logits = np.asarray(data["original_fresh_logits_video"])
    reconstructed_score = np.zeros_like(original_score)
    reconstructed_logits = np.zeros_like(original_logits)

    for row in range(len(raw_score)):
        framecount = int(framecounts[row])
        if int(batch_local[row]) == 0:
            start = framecount * stride
            stop = (framecount + 2) * stride
            reconstructed_score[start:stop] = raw_score[row]
            reconstructed_logits[start:stop] = raw_logits[row]
        else:
            start = (framecount + 1) * stride
            stop = (framecount + 2) * stride
            reconstructed_score[start:stop] = raw_score[row, stride:]
            reconstructed_logits[start:stop] = raw_logits[row, stride:]
    return reconstructed_score, reconstructed_logits


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size == 0:
        return None
    if np.array_equal(left, right):
        return 1.0
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    shape_match = actual.shape == expected.shape
    if not shape_match:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "max_abs_error": None,
            "mean_abs_error": None,
            "np_allclose": False,
            "array_equal": False,
            "pearson": None,
        }
    difference = np.abs(
        np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    )
    return {
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "np_allclose": bool(np.allclose(actual, expected, rtol=0.0, atol=0.0)),
        "array_equal": bool(np.array_equal(actual, expected)),
        "pearson": pearson(actual, expected),
    }


def evaluate(data) -> dict:
    reconstructed_score, reconstructed_logits = reconstruct(data)
    score_metrics = comparison(
        reconstructed_score, np.asarray(data["original_fresh_result_video"])
    )
    logits_metrics = comparison(
        reconstructed_logits, np.asarray(data["original_fresh_logits_video"])
    )
    passed = bool(score_metrics["array_equal"] and logits_metrics["array_equal"])
    return {
        "status": (
            "RAW-WINDOW-INSTRUMENTATION-READY"
            if passed
            else "RAW-WINDOW-DUMP-EQUIVALENCE-FAIL"
        ),
        "subject": str(scalar(data, "subject")),
        "video": str(scalar(data, "video")),
        "num_windows": int(len(data["raw_window_score"])),
        "score_reconstruction": score_metrics,
        "logits_reconstruction": logits_metrics,
    }


def synthetic_self_test() -> dict:
    rng = np.random.default_rng(20260906)
    n_windows = 70
    raw_score = rng.normal(size=(n_windows, 30)).astype(np.float32)
    raw_logits = rng.normal(size=(n_windows, 30, 5)).astype(np.float32)
    framecount = np.arange(n_windows, dtype=np.int32)
    # Start mid-batch so the self-test exercises the fact that a video's first
    # window is not necessarily batch-local index zero.
    flat_index = np.arange(17, 17 + n_windows, dtype=np.int32)
    batch_local = (flat_index % 32).astype(np.int16)
    base = {
        "schema": np.asarray(EXPECTED_SCHEMA),
        "subject": np.asarray("006"),
        "video": np.asarray("006_1"),
        "batch_size": np.asarray(32, dtype=np.int32),
        "window_length": np.asarray(30, dtype=np.int32),
        "stride": np.asarray(15, dtype=np.int32),
        "batch_index": flat_index // 32,
        "index_within_batch": batch_local,
        "video_window_id": framecount.copy(),
        "framecount_before_write": framecount.copy(),
        "global_start": framecount * 15,
        "global_indices": framecount[:, None] * 15 + np.arange(30)[None, :],
        "subject_flat_window_index": flat_index,
        "raw_window_score": raw_score,
        "raw_window_logits": raw_logits,
        "original_fresh_result_video": np.zeros(((n_windows + 1) * 15,), dtype=np.float32),
        "original_fresh_logits_video": np.zeros(((n_windows + 1) * 15, 5), dtype=np.float32),
    }
    score, logits = reconstruct(base)
    base["original_fresh_result_video"] = score.copy()
    base["original_fresh_logits_video"] = logits.copy()
    return evaluate(base)


def main() -> int:
    args = parse_args()
    if args.self_test:
        report = synthetic_self_test()
        report["test_type"] = "synthetic_static_self_test"
    else:
        if args.input is None:
            raise SystemExit("--input is required unless --self-test is used")
        with np.load(args.input, allow_pickle=False) as data:
            report = evaluate(data)

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "RAW-WINDOW-INSTRUMENTATION-READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
