#!/usr/bin/env python3
"""Falsification-first test for persistence-guided event discovery.

This script does not implement the final PSED decoder. It asks only whether
1D peak persistence (equivalent here to topographic prominence) separates
event-related peaks from background peaks better than peak height.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
CACHES = {
    ("me_tst_plus", "SAMMLV"): ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
    ("me_tst_plus", "CASME_3"): ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
    ("boostingvrme", "SAMMLV"): ROOT / "caches/boostingvrme/sammlv_curves.pkl",
    ("boostingvrme", "CASME_3"): ROOT / "caches/boostingvrme/casme3_curves.pkl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_videos(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    k_p = int(payload["k_p"])
    if "records" in payload:
        videos = [
            {
                "subject": str(row["subject"]),
                "video": str(row["video"]),
                "score": np.asarray(row["score"], dtype=float),
                "samples": row["samples"],
            }
            for row in payload["records"]
        ]
    else:
        videos = []
        for subject in payload["subject_curves"]:
            for index, score in enumerate(subject["score"]):
                videos.append({
                    "subject": str(subject["subject"]),
                    "video": str(subject["videos"][index]),
                    "score": np.asarray(score, dtype=float),
                    "samples": subject["samples"][index],
                })
    return payload, videos, k_p


def smooth(score: np.ndarray, width: int) -> np.ndarray:
    box = np.ones(max(1, width), dtype=float) / max(1, width)
    return np.convolve(score, box, mode="same")


def robust_z(score: np.ndarray) -> np.ndarray:
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    return (score - median) / (1.4826 * mad + 1e-8)


def interval_iou(left: int, right: int, gt) -> float:
    gt_left, gt_right = int(gt[0]), int(gt[2])
    inter = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return inter / union if union else 0.0


def native_peaks(curve: np.ndarray, k_p: int) -> np.ndarray:
    threshold = float(curve.mean() + 0.55 * (curve.max() - curve.mean()))
    return find_peaks(curve, height=threshold, distance=k_p)[0]


def analyze_cache(backbone: str, dataset: str, path: Path):
    payload, videos, k_p = load_videos(path)
    peak_rows, raw_peak_rows, native_rows, gt_rows = [], [], [], []
    native_totals = {"tp": 0, "fp": 0, "fn": 0}

    for video in videos:
        curve = smooth(video["score"], 2 * k_p)
        z = robust_z(curve)
        peaks = find_peaks(z)[0]
        prominences = peak_prominences(z, peaks)[0] if len(peaks) else np.array([])
        prominence_by_peak = dict(zip(peaks.tolist(), prominences.tolist()))
        native = native_peaks(curve, k_p)
        native_set = set(native.tolist())

        raw_z = robust_z(video["score"])
        raw_peaks = find_peaks(raw_z)[0]
        raw_prominences = (
            peak_prominences(raw_z, raw_peaks)[0] if len(raw_peaks) else np.array([])
        )
        for peak, prominence in zip(raw_peaks, raw_prominences):
            inside = any(int(gt[0]) <= peak <= int(gt[2]) for gt in video["samples"])
            best_iou = max(
                [interval_iou(int(peak - k_p), int(peak + k_p), gt) for gt in video["samples"]]
                or [0.0]
            )
            raw_peak_rows.append({
                "backbone": backbone,
                "dataset": dataset,
                "subject": video["subject"],
                "video": video["video"],
                "peak": int(peak),
                "height_z": float(raw_z[peak]),
                "prominence_z": float(prominence),
                "inside_gt": int(inside),
                "iou_positive": int(best_iou >= 0.5),
                "native_peak": 0,
            })

        for peak, prominence in zip(peaks, prominences):
            inside = any(int(gt[0]) <= peak <= int(gt[2]) for gt in video["samples"])
            best_iou = max(
                [interval_iou(int(peak - k_p), int(peak + k_p), gt) for gt in video["samples"]]
                or [0.0]
            )
            peak_rows.append({
                "backbone": backbone,
                "dataset": dataset,
                "subject": video["subject"],
                "video": video["video"],
                "peak": int(peak),
                "height_z": float(z[peak]),
                "prominence_z": float(prominence),
                "inside_gt": int(inside),
                "iou_positive": int(best_iou >= 0.5),
                "native_peak": int(peak in native_set),
            })

        unmatched_gt = set(range(len(video["samples"])))
        for peak in native:
            overlaps = [(interval_iou(int(peak - k_p), int(peak + k_p), gt), idx)
                        for idx, gt in enumerate(video["samples"]) if idx in unmatched_gt]
            best_iou, best_idx = max(overlaps, default=(0.0, -1))
            matched = best_iou >= 0.5
            if matched:
                unmatched_gt.remove(best_idx)
                native_totals["tp"] += 1
            else:
                native_totals["fp"] += 1
            native_rows.append({
                "subject": video["subject"],
                "height_z": float(z[peak]),
                "prominence_z": float(prominence_by_peak.get(int(peak), 0.0)),
                "is_tp": int(matched),
            })
        native_totals["fn"] += len(unmatched_gt)

        for gt_index, gt in enumerate(video["samples"]):
            candidates = [int(p) for p in peaks if int(gt[0]) <= p <= int(gt[2])]
            gt_rows.append({
                "subject": video["subject"],
                "matched_native": int(gt_index not in unmatched_gt),
                "has_local_peak": int(bool(candidates)),
                "max_height_z": max([float(z[p]) for p in candidates] or [float("nan")]),
                "max_prominence_z": max([float(prominence_by_peak[p]) for p in candidates] or [float("nan")]),
            })

    return {
        "metadata": {
            "backbone": backbone,
            "dataset": dataset,
            "cache": str(path),
            "cache_sha256": sha256(path),
            "videos": len(videos),
            "subjects": len({row["subject"] for row in videos}),
            "gt_events": sum(len(row["samples"]) for row in videos),
            "k_p": k_p,
            "frame_skip": int(payload["frame_skip"]),
        },
        "peak_rows": peak_rows,
        "raw_peak_rows": raw_peak_rows,
        "native_rows": native_rows,
        "gt_rows": gt_rows,
        "native_counts_check": native_totals,
    }


def metric_pair(rows, label: str):
    y = np.asarray([row[label] for row in rows], dtype=int)
    result = {"n": int(len(y)), "positive": int(y.sum())}
    for feature in ("height_z", "prominence_z"):
        x = np.asarray([row[feature] for row in rows], dtype=float)
        result[feature] = {
            "roc_auc": float(roc_auc_score(y, x)),
            "pr_auc": float(average_precision_score(y, x)),
            "positive_median": float(np.median(x[y == 1])),
            "negative_median": float(np.median(x[y == 0])),
        }
    result["roc_auc_delta_prominence_minus_height"] = (
        result["prominence_z"]["roc_auc"] - result["height_z"]["roc_auc"]
    )
    return result


def subject_bootstrap(rows, label: str, repeats: int = 1000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    subject_deltas = []
    for subject in sorted(grouped):
        subject_rows = grouped[subject]
        y = np.asarray([row[label] for row in subject_rows], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        height = np.asarray([row["height_z"] for row in subject_rows])
        prominence = np.asarray([row["prominence_z"] for row in subject_rows])
        subject_deltas.append(
            roc_auc_score(y, prominence) - roc_auc_score(y, height)
        )
    rng = np.random.default_rng(20260902)
    deltas = [
        float(np.mean(rng.choice(subject_deltas, size=len(subject_deltas), replace=True)))
        for _ in range(repeats)
    ]
    return {
        "subjects_with_both_classes": len(subject_deltas),
        "repeats": len(deltas),
        "delta_mean": float(np.mean(deltas)),
        "delta_ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
    }


def best_f1_threshold(rows, feature: str):
    y = np.asarray([row["inside_gt"] for row in rows], dtype=int)
    x = np.asarray([row[feature] for row in rows], dtype=float)
    thresholds = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 501)))
    candidates = [(f1_score(y, x >= threshold), float(threshold)) for threshold in thresholds]
    return max(candidates, key=lambda item: (item[0], item[1]))


def transfer_result(source_rows, target_rows, feature: str):
    source_f1, threshold = best_f1_threshold(source_rows, feature)
    y = np.asarray([row["inside_gt"] for row in target_rows], dtype=int)
    pred = np.asarray([row[feature] >= threshold for row in target_rows], dtype=int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "feature": feature,
        "source_threshold": threshold,
        "source_peak_f1": float(source_f1),
        "target_peak_f1": float(f1_score(y, pred)),
        "target_precision": precision,
        "target_recall": recall,
        "target_tp": tp,
        "target_fp": fp,
        "target_fn": fn,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    analyses = {}
    report = {"protocol": {
        "purpose": "feasibility only; not a final decoder evaluation",
        "smoothing": "native moving average width 2*k_p",
        "normalization": "per-video median/MAD robust z-score",
        "persistence_proxy": "scipy peak prominence on the 1D superlevel signal",
        "primary_peak_label": "local maximum lies inside a GT onset-offset interval",
        "secondary_peak_label": "fixed +/-k_p interval reaches IoU >= 0.5",
        "bootstrap_unit": "subject",
        "seed": 20260902,
    }, "datasets": {}, "locked_transfer": {}}

    for key, path in CACHES.items():
        analysis = analyze_cache(*key, path)
        analyses[key] = analysis
        peak_rows = analysis["peak_rows"]
        raw_peak_rows = analysis["raw_peak_rows"]
        native_rows = analysis["native_rows"]
        gt_rows = analysis["gt_rows"]
        missed = [row for row in gt_rows if not row["matched_native"]]
        summary = {
            "metadata": analysis["metadata"],
            "native_counts_check": analysis["native_counts_check"],
            "all_local_peaks_inside_gt": metric_pair(peak_rows, "inside_gt"),
            "all_local_peaks_iou_positive": metric_pair(peak_rows, "iou_positive"),
            "raw_local_peaks_inside_gt": metric_pair(raw_peak_rows, "inside_gt"),
            "raw_local_peaks_iou_positive": metric_pair(raw_peak_rows, "iou_positive"),
            "native_tp_vs_fp": metric_pair(native_rows, "is_tp"),
            "subject_bootstrap_inside_gt": subject_bootstrap(peak_rows, "inside_gt"),
            "native_missed_gt": {
                "n": len(missed),
                "with_any_local_peak_inside": sum(row["has_local_peak"] for row in missed),
                "coverage": sum(row["has_local_peak"] for row in missed) / len(missed) if missed else 0.0,
                "max_prominence_median_when_present": float(np.nanmedian(
                    [row["max_prominence_z"] for row in missed]
                )) if missed else None,
            },
        }
        name = f"{key[0]}__{key[1]}"
        report["datasets"][name] = summary
    for backbone in ("me_tst_plus", "boostingvrme"):
        source = analyses[(backbone, "SAMMLV")]["peak_rows"]
        target = analyses[(backbone, "CASME_3")]["peak_rows"]
        report["locked_transfer"][backbone] = {
            feature: transfer_result(source, target, feature)
            for feature in ("height_z", "prominence_z")
        }
        raw_source = analyses[(backbone, "SAMMLV")]["raw_peak_rows"]
        raw_target = analyses[(backbone, "CASME_3")]["raw_peak_rows"]
        report["locked_transfer"][backbone]["raw_curve"] = {
            feature: transfer_result(raw_source, raw_target, feature)
            for feature in ("height_z", "prominence_z")
        }

    with (OUT / "feasibility_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
