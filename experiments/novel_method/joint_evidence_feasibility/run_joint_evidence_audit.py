#!/usr/bin/env python3
"""Audit whether recognition adds information beyond frozen spotting scores.

This is a falsification test, not a structured decoder implementation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 20260902
CACHES = {
    ("me_tst_plus", "SAMMLV"): ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
    ("me_tst_plus", "CASME_3"): ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
    ("boostingvrme", "SAMMLV"): ROOT / "caches/boostingvrme/sammlv_curves.pkl",
    ("boostingvrme", "CASME_3"): ROOT / "caches/boostingvrme/casme3_curves.pkl",
}
PRIMARY = ("spotting", "recognition", "joint_mean")
ALL_EVIDENCE = PRIMARY + ("recognition_contrast", "joint_rank", "joint_a025", "joint_a075")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(values, np.ones(width) / width, mode="same")


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - median) / (1.4826 * mad)


def local_mean(values: np.ndarray, center: int, radius: int) -> float:
    left, right = max(0, center - radius), min(len(values), center + radius + 1)
    return float(np.mean(values[left:right]))


def local_curve(values: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray([local_mean(values, index, radius) for index in range(len(values))])


def contrast_curve(values: np.ndarray, radius: int) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    for center in range(len(values)):
        local = local_mean(values, center, radius)
        left = values[max(0, center - 3 * radius):max(0, center - radius)]
        right = values[min(len(values), center + radius + 1):min(len(values), center + 3 * radius + 1)]
        contexts = [part for part in (left, right) if len(part)]
        context = float(np.mean(np.concatenate(contexts))) if contexts else local
        output[center] = local - context
    return output


def interval_iou(left: int, right: int, gt) -> float:
    gt_left, gt_right = int(gt[0]), int(gt[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def match_native(curve: np.ndarray, samples, k_p: int):
    threshold = float(curve.mean() + 0.55 * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0]
    unmatched = set(range(len(samples)))
    tp = fp = 0
    for peak in peaks:
        overlaps = [(interval_iou(int(peak - k_p), int(peak + k_p), gt), index)
                    for index, gt in enumerate(samples) if index in unmatched]
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            tp += 1
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": len(unmatched), "missed_gt": unmatched}


def candidate_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order / max(1, len(values) - 1)


def load_cache(backbone: str, dataset: str, path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    videos = []
    structure = {
        "backbone": backbone,
        "dataset": dataset,
        "cache": str(path),
        "cache_sha256": sha256(path),
        "k_p": int(payload["k_p"]),
        "frame_skip": int(payload["frame_skip"]),
        "class_semantics": {"0": "negative", "1": "positive", "2": "surprise", "3": "others", "4": "neutral"},
        "neutral_class_id": 4,
    }
    if "records" in payload:
        aligned = True
        dimensions = set()
        for row in payload["records"]:
            score = np.asarray(row["score"], dtype=float)
            logits = np.asarray(row["logits"], dtype=float)
            emotion = np.asarray(row["emotion"], dtype=int)
            aligned &= len(score) == len(logits) == len(emotion)
            dimensions.add(int(logits.shape[1]))
            videos.append({
                "subject": str(row["subject"]), "video": str(row["video"]),
                "score": score, "logits": logits, "emotion": emotion,
                "samples": row["samples"],
            })
        structure.update({
            "continuous_logits": True,
            "softmax_probabilities_saved": False,
            "softmax_probabilities_derivable": True,
            "hard_argmax_saved": True,
            "class_dimensions": sorted(dimensions),
            "temporal_alignment_all_videos": bool(aligned),
            "mirror_logits_available": all(row.get("mirror_logits") is not None for row in payload["records"]),
            "mirror_used_in_audit": False,
            "strategy": int(payload["strategy"]),
            "strategy_name": payload["strategy_name"],
        })
    else:
        aligned = True
        unique_ids = set()
        for subject in payload["subject_curves"]:
            for index, score in enumerate(subject["score"]):
                score = np.asarray(score, dtype=float)
                emotion = np.asarray(subject["emotion_pred"][index], dtype=int)
                aligned &= len(score) == len(emotion)
                unique_ids.update(emotion.tolist())
                videos.append({
                    "subject": str(subject["subject"]), "video": str(subject["videos"][index]),
                    "score": score, "logits": None, "emotion": emotion,
                    "samples": subject["samples"][index],
                })
        structure.update({
            "continuous_logits": False,
            "softmax_probabilities_saved": False,
            "softmax_probabilities_derivable": False,
            "hard_argmax_saved": True,
            "observed_class_ids": sorted(int(value) for value in unique_ids),
            "temporal_alignment_all_videos": bool(aligned),
            "limitation": "Only hard frame-level class IDs are cached; entropy, margin and calibrated non-neutral probability cannot be audited.",
        })
    structure.update({
        "subjects": len({video["subject"] for video in videos}),
        "videos": len(videos),
        "gt_events": sum(len(video["samples"]) for video in videos),
    })
    return videos, structure


def analyze_cache(backbone: str, dataset: str, path: Path):
    videos, structure = load_cache(backbone, dataset, path)
    k_p = structure["k_p"]
    rows = []
    native_counts = {"tp": 0, "fp": 0, "fn": 0}
    missed_total = 0

    for video in videos:
        curve = smooth(video["score"], 2 * k_p)
        spotting_z = robust_z(curve)
        peaks = find_peaks(curve)[0]
        native = match_native(curve, video["samples"], k_p)
        for key in native_counts:
            native_counts[key] += native[key]
        missed_total += len(native["missed_gt"])

        if video["logits"] is not None:
            probabilities = softmax(video["logits"])
            frame_recognition = 1.0 - probabilities[:, 4]
            recognition_kind = "mean_1_minus_p_neutral"
        else:
            frame_recognition = (video["emotion"] != 4).astype(float)
            recognition_kind = "hard_non_neutral_fraction"
        recognition_curve = local_curve(frame_recognition, k_p)
        recognition_contrast = contrast_curve(frame_recognition, k_p)
        recognition_z = robust_z(recognition_curve)
        contrast_z = robust_z(recognition_contrast)

        peak_s = spotting_z[peaks]
        peak_r = recognition_z[peaks]
        rank_joint = (candidate_ranks(peak_s) + candidate_ranks(peak_r)) / 2.0
        gt_hit = set(range(len(video["samples"]))) - native["missed_gt"]

        for candidate_index, peak in enumerate(peaks):
            containing = {index for index, gt in enumerate(video["samples"])
                          if int(gt[0]) <= peak <= int(gt[2])}
            best_iou = max(
                [interval_iou(int(peak - k_p), int(peak + k_p), gt) for gt in video["samples"]]
                or [0.0]
            )
            row = {
                "subject": video["subject"], "video": video["video"], "peak": int(peak),
                "inside_gt": int(bool(containing)), "iou_positive": int(best_iou >= 0.5),
                "inside_missed_gt": int(bool(containing & native["missed_gt"])),
                "inside_native_hit_gt": int(bool(containing & gt_hit)),
                "missed_gt_ids": [f"{video['subject']}::{video['video']}::{index}"
                                  for index in sorted(containing & native["missed_gt"])],
                "spotting": float(peak_s[candidate_index]),
                "recognition": float(peak_r[candidate_index]),
                "joint_mean": float((peak_s[candidate_index] + peak_r[candidate_index]) / 2.0),
                "recognition_contrast": float(contrast_z[peak]),
                "joint_rank": float(rank_joint[candidate_index]),
                "joint_a025": float(0.25 * peak_s[candidate_index] + 0.75 * peak_r[candidate_index]),
                "joint_a075": float(0.75 * peak_s[candidate_index] + 0.25 * peak_r[candidate_index]),
            }
            rows.append(row)

    structure["recognition_primary_definition"] = recognition_kind
    return rows, structure, native_counts, missed_total


def metric_summary(rows, label: str):
    y = np.asarray([row[label] for row in rows], dtype=int)
    result = {"n": len(rows), "positive": int(y.sum())}
    for evidence in ALL_EVIDENCE:
        values = np.asarray([row[evidence] for row in rows], dtype=float)
        result[evidence] = {
            "roc_auc": float(roc_auc_score(y, values)),
            "pr_auc": float(average_precision_score(y, values)),
        }
    return result


def subject_bootstrap(rows, label: str, repeats: int = 2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    subject_deltas = {evidence: {"roc_auc": [], "pr_auc": []}
                      for evidence in ("recognition", "joint_mean")}
    for subject_rows in grouped.values():
        y = np.asarray([row[label] for row in subject_rows], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        spotting = np.asarray([row["spotting"] for row in subject_rows])
        baseline_auc = roc_auc_score(y, spotting)
        baseline_ap = average_precision_score(y, spotting)
        for evidence in subject_deltas:
            values = np.asarray([row[evidence] for row in subject_rows])
            subject_deltas[evidence]["roc_auc"].append(roc_auc_score(y, values) - baseline_auc)
            subject_deltas[evidence]["pr_auc"].append(average_precision_score(y, values) - baseline_ap)

    rng = np.random.default_rng(SEED)
    output = {}
    for evidence, metrics in subject_deltas.items():
        output[evidence] = {}
        for metric, values in metrics.items():
            values = np.asarray(values, dtype=float)
            samples = [float(np.mean(rng.choice(values, len(values), replace=True)))
                       for _ in range(repeats)]
            output[evidence][metric] = {
                "subjects": int(len(values)),
                "delta_mean": float(np.mean(samples)),
                "delta_ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
            }
    return output


def matched_fp(rows, fp_budget: int, missed_total: int):
    results = {}
    for evidence in PRIMARY:
        ordered = sorted(rows, key=lambda row: (-row[evidence], row["subject"], row["video"], row["peak"]))
        fp = 0
        recovered = set()
        selected = 0
        for row in ordered:
            if not row["inside_gt"]:
                fp += 1
            selected += 1
            recovered.update(row["missed_gt_ids"])
            if fp >= fp_budget:
                break
        count = len(recovered)
        results[evidence] = {
            "fp_budget": fp_budget, "fp": fp, "selected_candidates": selected,
            "recovered_missed_gt": count,
            "precision_recovered_vs_fp": count / (count + fp) if count + fp else 0.0,
            "recall_of_all_native_missed_gt": count / missed_total if missed_total else 0.0,
        }
    return results


def best_threshold(rows, evidence: str):
    y = np.asarray([row["inside_gt"] for row in rows], dtype=int)
    values = np.asarray([row[evidence] for row in rows], dtype=float)
    thresholds = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 501)))
    return max(((f1_score(y, values >= threshold), float(threshold)) for threshold in thresholds),
               key=lambda item: (item[0], item[1]))


def locked_transfer(source, target):
    output = {}
    y = np.asarray([row["inside_gt"] for row in target], dtype=int)
    for evidence in PRIMARY:
        source_f1, threshold = best_threshold(source, evidence)
        values = np.asarray([row[evidence] for row in target], dtype=float)
        pred = values >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        output[evidence] = {
            "source_threshold": threshold, "source_peak_f1": float(source_f1),
            "target_peak_f1": float(f1_score(y, pred)),
            "target_precision": tp / (tp + fp) if tp + fp else 0.0,
            "target_recall": tp / (tp + fn) if tp + fn else 0.0,
            "target_tp": tp, "target_fp": fp, "target_fn": fn,
        }
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    analyses = {}
    report = {
        "protocol": {
            "purpose": "joint-evidence feasibility audit; no decoder training or implementation",
            "candidate_pool": "all scipy local maxima of native-smoothed spotting curve",
            "smoothing": "moving average width 2*k_p",
            "spotting": "per-video robust z-score at candidate peak",
            "recognition_window_radius": "k_p",
            "joint_primary": "equal average of per-video robust-z spotting and recognition",
            "matched_fp_budget": "native decoder FP count for the same cache",
            "bootstrap_unit": "subject", "bootstrap_repeats": 2000, "seed": SEED,
        },
        "cache_structure": {}, "datasets": {}, "locked_transfer": {},
    }
    for key, path in CACHES.items():
        rows, structure, native_counts, missed_total = analyze_cache(*key, path)
        analyses[key] = rows
        name = f"{key[0]}__{key[1]}"
        report["cache_structure"][name] = structure
        missed_subset = [row for row in rows if row["inside_missed_gt"] or not row["inside_gt"]]
        report["datasets"][name] = {
            "native_counts_check": native_counts,
            "all_candidates_primary_label": metric_summary(rows, "inside_gt"),
            "all_candidates_secondary_label": metric_summary(rows, "iou_positive"),
            "native_missed_gt_vs_background": metric_summary(missed_subset, "inside_missed_gt"),
            "subject_bootstrap_primary": subject_bootstrap(rows, "inside_gt"),
            "subject_bootstrap_missed_subset": subject_bootstrap(missed_subset, "inside_missed_gt"),
            "matched_fp_native_budget": matched_fp(rows, native_counts["fp"], missed_total),
        }
    for backbone in ("me_tst_plus", "boostingvrme"):
        report["locked_transfer"][backbone] = locked_transfer(
            analyses[(backbone, "SAMMLV")], analyses[(backbone, "CASME_3")]
        )
    with (OUT / "joint_evidence_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
