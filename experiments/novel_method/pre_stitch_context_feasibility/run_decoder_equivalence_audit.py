#!/usr/bin/env python3
"""Compare historical and fresh ME-TST+ scores at the native decoder level.

This is an engineering alignment audit.  It never changes either score and
does not run Context Agreement or any other scientific experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-cache", type=Path, required=True)
    parser.add_argument("--fresh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--subject", default="006")
    parser.add_argument("--video", default="006_1")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_smooth(score: np.ndarray, width: int) -> np.ndarray:
    box = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.asarray(score, dtype=np.float64), box, mode="same")


def native_decode(score: np.ndarray, emotion: np.ndarray, k_p: int, p: float) -> dict:
    smoothed = native_smooth(score, 2 * k_p)
    threshold = float(smoothed.mean() + p * (smoothed.max() - smoothed.mean()))
    peaks, _ = find_peaks(smoothed, height=threshold, distance=k_p)
    events = []
    for peak in peaks:
        onset = int(peak - k_p)
        offset = int(peak + k_p)
        values = list(emotion[max(0, onset + 1) : max(1, offset - 1)])
        emotion_id = int(Counter(int(x) for x in values).most_common(1)[0][0])
        events.append(
            {
                "onset": onset,
                "peak": int(peak),
                "offset": offset,
                "emotion_id": emotion_id,
                "score": float(score[peak]),
                "smoothed_score": float(smoothed[peak]),
                "margin": float(smoothed[peak] - threshold),
            }
        )
    return {"smoothed": smoothed, "threshold": threshold, "peaks": peaks, "events": events}


def interval_iou(first: dict, second: dict) -> float:
    left = max(first["onset"], second["onset"])
    right = min(first["offset"], second["offset"])
    intersection = max(0, right - left + 1)
    if not intersection:
        return 0.0
    first_len = first["offset"] - first["onset"] + 1
    second_len = second["offset"] - second["onset"] + 1
    return intersection / (first_len + second_len - intersection)


def greedy_pairs(first: list[dict], second: list[dict], minimum_iou: float = 0.0) -> list[dict]:
    candidates = []
    for first_index, first_event in enumerate(first):
        for second_index, second_event in enumerate(second):
            iou = interval_iou(first_event, second_event)
            if iou > minimum_iou:
                candidates.append((iou, first_index, second_index))
    candidates.sort(reverse=True)
    used_first, used_second, pairs = set(), set(), []
    for iou, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        pairs.append(
            {"old_index": first_index, "fresh_index": second_index, "iou": float(iou)}
        )
    return pairs


def tolerance_match(old: np.ndarray, fresh: np.ndarray, tolerance: int) -> dict:
    available = set(int(x) for x in fresh)
    pairs = []
    for old_peak in (int(x) for x in old):
        choices = sorted(
            (x for x in available if abs(x - old_peak) <= tolerance),
            key=lambda x: (abs(x - old_peak), x),
        )
        if choices:
            fresh_peak = choices[0]
            available.remove(fresh_peak)
            pairs.append([old_peak, fresh_peak])
    matched = len(pairs)
    union = len(old) + len(fresh) - matched
    return {
        "tolerance": tolerance,
        "matched_count": matched,
        "pairs": pairs,
        "jaccard": float(matched / union) if union else 1.0,
    }


def gt_metrics(events: list[dict], samples: list) -> dict:
    gt = [
        {"onset": int(sample[0]), "peak": int(sample[1]), "offset": int(sample[2])}
        for sample in samples
    ]
    raw_matches = greedy_pairs(events, gt, minimum_iou=0.5 - np.finfo(float).eps)
    matches = [
        {
            "pred_index": row["old_index"],
            "gt_index": row["fresh_index"],
            "iou": row["iou"],
        }
        for row in raw_matches
    ]
    tp = len(matches)
    fp = len(events) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Spotting_Precision": float(precision),
        "Spotting_Recall": float(recall),
        "Spotting_F1": float(f1),
        "matches": matches,
    }


def main() -> int:
    args = arguments()
    with args.historical_cache.open("rb") as handle:
        cache = pickle.load(handle)
    record = next(
        row
        for row in cache["records"]
        if str(row["subject"]) == args.subject and str(row["video"]) == args.video
    )
    with np.load(args.fresh_npz, allow_pickle=False) as saved:
        fresh = np.asarray(saved["score"], dtype=np.float64)
        fresh_keys = list(saved.files)
    old = np.asarray(record["score"], dtype=np.float64)
    if old.shape != fresh.shape:
        raise ValueError(f"score length mismatch: historical={old.shape}, fresh={fresh.shape}")

    # Fresh logits were not archived.  Reusing the same historical emotion
    # sequence isolates whether score-only differences change spotting events.
    emotion = np.asarray(record["emotion"])
    k_p = int(cache["k_p"])
    p = 0.55
    old_decoded = native_decode(old, emotion, k_p, p)
    fresh_decoded = native_decode(fresh, emotion, k_p, p)

    error = np.abs(old - fresh)
    smooth_error = np.abs(old_decoded["smoothed"] - fresh_decoded["smoothed"])
    old_peaks = old_decoded["peaks"]
    fresh_peaks = fresh_decoded["peaks"]
    exact_common = sorted(set(map(int, old_peaks)) & set(map(int, fresh_peaks)))
    exact_union = set(map(int, old_peaks)) | set(map(int, fresh_peaks))
    event_pairs = greedy_pairs(old_decoded["events"], fresh_decoded["events"])
    event_ious = [pair["iou"] for pair in event_pairs]
    old_triples = {(x["onset"], x["peak"], x["offset"]) for x in old_decoded["events"]}
    fresh_triples = {(x["onset"], x["peak"], x["offset"]) for x in fresh_decoded["events"]}
    exact_event_count = len(old_triples & fresh_triples)
    old_signatures = {
        (x["onset"], x["peak"], x["offset"], x["emotion_id"])
        for x in old_decoded["events"]
    }
    fresh_signatures = {
        (x["onset"], x["peak"], x["offset"], x["emotion_id"])
        for x in fresh_decoded["events"]
    }

    inconsistent_peaks = sorted(set(map(int, old_peaks)) ^ set(map(int, fresh_peaks)))
    threshold_disagreements = np.flatnonzero(
        (old_decoded["smoothed"] >= old_decoded["threshold"])
        != (fresh_decoded["smoothed"] >= fresh_decoded["threshold"])
    )
    margins = []
    for index in inconsistent_peaks:
        margins.append(
            {
                "temporal_index": index,
                "old": {
                    "score": float(old[index]),
                    "smoothed_score": float(old_decoded["smoothed"][index]),
                    "threshold": old_decoded["threshold"],
                    "score_minus_threshold": float(
                        old_decoded["smoothed"][index] - old_decoded["threshold"]
                    ),
                },
                "fresh": {
                    "score": float(fresh[index]),
                    "smoothed_score": float(fresh_decoded["smoothed"][index]),
                    "threshold": fresh_decoded["threshold"],
                    "score_minus_threshold": float(
                        fresh_decoded["smoothed"][index] - fresh_decoded["threshold"]
                    ),
                },
            }
        )

    old_gt = gt_metrics(old_decoded["events"], record["samples"])
    fresh_gt = gt_metrics(fresh_decoded["events"], record["samples"])
    same_events = old_triples == fresh_triples
    same_gt = all(old_gt[key] == fresh_gt[key] for key in ("TP", "FP", "FN"))
    pass_gate = np.array_equal(old_peaks, fresh_peaks) and same_events and same_gt

    result = {
        "status": "PASS-DECODER-EQUIVALENT" if pass_gate else "BLOCKED-DECODER-MISMATCH",
        "scope": {"subject": args.subject, "video_id": args.video, "T_old": len(old), "T_fresh": len(fresh)},
        "provenance": {
            "historical_cache": str(args.historical_cache),
            "historical_cache_sha256": sha256(args.historical_cache),
            "fresh_npz": str(args.fresh_npz),
            "fresh_npz_sha256": sha256(args.fresh_npz),
            "fresh_npz_keys": fresh_keys,
            "scores_modified": False,
            "calibration_or_scaling": False,
            "native_parameters": {"k_p": k_p, "smooth_width": 2 * k_p, "threshold_p": p, "peak_distance": k_p},
        },
        "score_numerical_comparison": {
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "RMSE": float(np.sqrt(np.mean((old - fresh) ** 2))),
            "Pearson": float(pearsonr(old, fresh).statistic),
            "Spearman": float(spearmanr(old, fresh).statistic),
            "error_quantiles": {
                "p50": float(np.quantile(error, 0.5)),
                "p90": float(np.quantile(error, 0.9)),
                "p95": float(np.quantile(error, 0.95)),
                "p99": float(np.quantile(error, 0.99)),
                "p99.9": float(np.quantile(error, 0.999)),
            },
            "np_allclose_default": bool(np.allclose(old, fresh)),
        },
        "native_smoothing_equivalence": {
            "smoothed_MAE": float(smooth_error.mean()),
            "smoothed_RMSE": float(np.sqrt(np.mean((old_decoded["smoothed"] - fresh_decoded["smoothed"]) ** 2))),
            "smoothed_max_error": float(smooth_error.max()),
            "smoothed_Pearson": float(pearsonr(old_decoded["smoothed"], fresh_decoded["smoothed"]).statistic),
        },
        "native_threshold_equivalence": {
            "threshold_old": old_decoded["threshold"],
            "threshold_fresh": fresh_decoded["threshold"],
            "absolute_delta": abs(old_decoded["threshold"] - fresh_decoded["threshold"]),
            "relative_delta": abs(old_decoded["threshold"] - fresh_decoded["threshold"]) / abs(old_decoded["threshold"]),
        },
        "peak_candidate_equivalence": {
            "peaks_old": old_peaks.astype(int).tolist(),
            "peaks_fresh": fresh_peaks.astype(int).tolist(),
            "old_peak_count": len(old_peaks),
            "fresh_peak_count": len(fresh_peaks),
            "exact_common_peaks": exact_common,
            "exact_common_count": len(exact_common),
            "exact_Jaccard": float(len(exact_common) / len(exact_union)) if exact_union else 1.0,
            "plus_or_minus_1": tolerance_match(old_peaks, fresh_peaks, 1),
            "plus_or_minus_2": tolerance_match(old_peaks, fresh_peaks, 2),
        },
        "native_event_decoding_equivalence": {
            "events_old": old_decoded["events"],
            "events_fresh": fresh_decoded["events"],
            "old_event_count": len(old_decoded["events"]),
            "fresh_event_count": len(fresh_decoded["events"]),
            "exact_event_match_count": exact_event_count,
            "all_event_boundaries_exact": same_events,
            "all_decoded_events_exact_including_controlled_emotion": old_signatures == fresh_signatures,
            "event_pairs": event_pairs,
            "mean_matched_IoU": float(np.mean(event_ious)) if event_ious else None,
            "minimum_matched_IoU": float(np.min(event_ious)) if event_ious else None,
            "fraction_IoU_ge_0_5": float(np.mean(np.asarray(event_ious) >= 0.5)) if event_ious else None,
            "fraction_IoU_ge_0_9": float(np.mean(np.asarray(event_ious) >= 0.9)) if event_ious else None,
            "fraction_IoU_eq_1": float(np.mean(np.asarray(event_ious) == 1.0)) if event_ious else None,
            "emotion_protocol": "same archived historical emotion sequence used for both score decodes",
            "fresh_emotion_independently_verified": False,
        },
        "GT_evaluation_equivalence": {"historical_compact": old_gt, "fresh_frozen_forward": fresh_gt, "TP_FP_FN_identical": same_gt},
        "margin_sensitivity": {
            "inconsistent_candidate_count": len(inconsistent_peaks),
            "inconsistent_candidates": margins,
            "threshold_classification_disagreement_count": int(len(threshold_disagreements)),
            "threshold_classification_disagreement_indices": threshold_disagreements.astype(int).tolist(),
            "minimum_selected_peak_margin_old": min(
                (row["margin"] for row in old_decoded["events"]), default=None
            ),
            "minimum_selected_peak_margin_fresh": min(
                (row["margin"] for row in fresh_decoded["events"]), default=None
            ),
            "interpretation": "No peak candidate differs; no threshold-crossing candidate requires margin diagnosis."
            if not margins
            else "At least one peak candidate differs; inspect the listed native margins.",
        },
        "gate_conditions": {
            "exact_peaks_identical": bool(np.array_equal(old_peaks, fresh_peaks)),
            "final_spotting_events_identical": same_events,
            "TP_FP_FN_identical": same_gt,
            "fresh_recognition_logits_archived": False,
            "recognition_required_for_this_gate": False,
        },
        "scientific_audits_run": [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
