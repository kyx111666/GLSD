#!/usr/bin/env python3
"""Cache-only ME-TST+ native inference failure-mode audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


P = 0.55
IOU_THRESHOLD = 0.5
EMOTION_ID = {"negative": 0, "positive": 1, "surprise": 2, "others": 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sammlv-cache", type=Path, required=True)
    parser.add_argument("--casme3-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {
        "dataset", "strategy", "strategy_name", "frame_skip", "k", "k_p",
        "num_subjects", "num_videos", "num_gt", "records",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path}: missing fields {missing}")
    record_required = {"subject", "video", "samples", "gt_emotions", "score", "emotion"}
    issues = []
    for index, record in enumerate(payload["records"]):
        missing_record = sorted(record_required - set(record))
        if missing_record:
            raise KeyError(f"{path}: record {index} missing {missing_record}")
        if len(record["samples"]) != len(record["gt_emotions"]):
            issues.append({
                "record_index": index, "subject": str(record["subject"]),
                "video": str(record["video"]), "issue": "GT/emotion length mismatch",
                "samples": len(record["samples"]), "gt_emotions": len(record["gt_emotions"]),
                "handling": "spotting retained; this video's matched events excluded from recognition-only summary",
            })
        if len(record["score"]) != len(record["emotion"]):
            raise ValueError(f"{path}: record {index} score/emotion length mismatch")
    if len(payload["records"]) != int(payload["num_videos"]):
        raise ValueError(f"{path}: num_videos mismatch")
    if sum(len(row["samples"]) for row in payload["records"]) != int(payload["num_gt"]):
        raise ValueError(f"{path}: num_gt mismatch")
    return payload, issues


def smooth(score, width):
    box = np.ones(int(width), dtype=np.float64) / int(width)
    return np.convolve(np.asarray(score, dtype=np.float64), box, mode="same")


def iou_interval(left, right, sample):
    gt_left, gt_right = int(sample[0]), int(sample[2])
    intersection = max(0, min(int(right), gt_right) - max(int(left), gt_left) + 1)
    union = max(int(right), gt_right) - min(int(left), gt_left) + 1
    return intersection / union if union else 0.0


def distribution(values):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "IQR": None}
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return {
        "n": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "q25": float(q25), "q75": float(q75), "IQR": float(q75 - q25),
    }


def interval_max(values, onset, offset):
    left = max(0, int(onset))
    right = min(len(values) - 1, int(offset))
    if right < left:
        return None
    return float(np.asarray(values)[left:right + 1].max())


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "Precision": float(precision), "Recall": float(recall), "Spotting_F1": float(f1),
    }


def native_decode(score, k_p):
    curve = smooth(score, 2 * k_p)
    threshold = float(curve.mean() + P * (curve.max() - curve.mean()))
    all_peaks = find_peaks(curve, distance=k_p)[0].astype(int)
    thresholded = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
    return curve, threshold, all_peaks, thresholded


def greedy_native_match(peaks, samples, k_p):
    unmatched = set(range(len(samples)))
    pred_to_gt = []
    for peak in peaks:
        choices = [
            (iou_interval(peak - k_p, peak + k_p, samples[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            pred_to_gt.append(int(best_index))
        else:
            pred_to_gt.append(-1)
    return pred_to_gt, unmatched


def maximum_coverage(peaks, samples, k_p):
    adjacency = {
        index: [
            gt_index for gt_index, sample in enumerate(samples)
            if iou_interval(int(peak) - k_p, int(peak) + k_p, sample) >= IOU_THRESHOLD
        ]
        for index, peak in enumerate(peaks)
    }
    gt_owner = {}

    def augment(pred_index, visited):
        for gt_index in adjacency[pred_index]:
            if gt_index in visited:
                continue
            visited.add(gt_index)
            if gt_index not in gt_owner or augment(gt_owner[gt_index], visited):
                gt_owner[gt_index] = pred_index
                return True
        return False

    for pred_index in adjacency:
        augment(pred_index, set())
    return len(gt_owner)


def voted_emotion(sequence, onset, offset, peak):
    start = max(0, int(onset) + 1)
    stop = max(1, int(offset) - 1)
    values = list(np.asarray(sequence)[start:stop])
    if not values:
        clipped_peak = min(max(0, int(peak)), len(sequence) - 1)
        values = [int(np.asarray(sequence)[clipped_peak])] if len(sequence) else [3]
    return int(Counter(int(value) for value in values).most_common(1)[0][0])


def recognition_summary(gt_ids, pred_ids):
    per_class = []
    for class_id in [0, 1, 2, 3]:
        tp = sum(g == class_id and p == class_id for g, p in zip(gt_ids, pred_ids))
        fp = sum(g != class_id and p == class_id for g, p in zip(gt_ids, pred_ids))
        fn = sum(g == class_id and p != class_id for g, p in zip(gt_ids, pred_ids))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class.append({
            "class_id": class_id, "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall,
        })
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if macro_precision + macro_recall else 0.0
    )
    return {
        "matched_events": len(gt_ids),
        "correct": sum(g == p for g, p in zip(gt_ids, pred_ids)),
        "wrong": sum(g != p for g, p in zip(gt_ids, pred_ids)),
        "Recognition_F1_repo_definition": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
    }


def rank_in_video(peak, peaks, curve):
    ordered = sorted((int(item) for item in peaks), key=lambda item: (-curve[item], item))
    return ordered.index(int(peak)) + 1


def analyze_dataset(payload, cache_path, data_quality_issues, candidate_rows, gt_rows, prediction_rows, summary_rows):
    dataset = str(payload["dataset"])
    k_p = int(payload["k_p"])
    gt_count = int(payload["num_gt"])
    raw_tp = raw_fp = final_tp = final_fp = 0
    fn_counts = Counter()
    fp_counts = Counter()
    neutral_tp_removed = neutral_fp_removed = 0
    stage_tp = Counter()
    boundary_linked_fp = set()
    peak_inside_boundary = 0
    gt_group_values = defaultdict(lambda: defaultdict(list))
    threshold_values = defaultdict(list)
    fn_b_threshold_only_recoverable = 0
    recognition_gt = []
    recognition_pred = []
    recognition_excluded_ambiguous = 0
    dataset_gt_rows = []
    dataset_prediction_rows = []

    for video_order, record in enumerate(payload["records"]):
        subject = str(record["subject"])
        video = str(record["video"])
        score = np.asarray(record["score"], dtype=np.float64)
        samples = list(record["samples"])
        curve, threshold, all_peaks, thresholded = native_decode(score, k_p)
        pred_to_gt, raw_unmatched = greedy_native_match(thresholded, samples, k_p)
        peak_to_pred = {int(peak): index for index, peak in enumerate(thresholded)}
        emotions = [
            voted_emotion(record["emotion"], peak - k_p, peak + k_p, peak)
            for peak in thresholded
        ]

        raw_tp += sum(index >= 0 for index in pred_to_gt)
        raw_fp += sum(index < 0 for index in pred_to_gt)
        final_matched_gt = set()
        for pred_index, (peak, matched_gt, emotion_id) in enumerate(zip(thresholded, pred_to_gt, emotions)):
            prediction_id = f"{dataset}:{subject}:{video}:{pred_index}"
            overlaps = [iou_interval(peak - k_p, peak + k_p, sample) for sample in samples]
            max_iou = max(overlaps, default=0.0)
            closest_gt = int(np.argmax(overlaps)) if overlaps else -1
            neutral = emotion_id == 4
            final_prediction = not neutral
            if neutral:
                if matched_gt >= 0:
                    neutral_tp_removed += 1
                else:
                    neutral_fp_removed += 1
            elif matched_gt >= 0:
                final_tp += 1
                final_matched_gt.add(int(matched_gt))
                if len(record["samples"]) == len(record["gt_emotions"]):
                    recognition_gt.append(EMOTION_ID.get(str(record["gt_emotions"][matched_gt]), 3))
                    recognition_pred.append(int(emotion_id))
                else:
                    recognition_excluded_ambiguous += 1
            else:
                final_fp += 1
                if max_iou == 0:
                    fp_category = "FP-A_BACKGROUND"
                elif max_iou < IOU_THRESHOLD:
                    fp_category = "FP-B_NEAR_GT_LOCALIZATION"
                else:
                    fp_category = "FP-C_DUPLICATE"
                fp_counts[fp_category] += 1
                if fp_category == "FP-A_BACKGROUND":
                    threshold_values["final_background_fp_margin"].append(float(curve[peak] - threshold))
                    threshold_values["final_background_fp_height"].append(float(curve[peak]))
                    threshold_values["final_background_fp_rank"].append(rank_in_video(peak, all_peaks, curve))

            if neutral:
                final_role = "removed_neutral"
                fp_category = ""
            elif matched_gt >= 0:
                final_role = "TP"
                fp_category = ""
            else:
                final_role = "FP"

            row = {
                "dataset": dataset, "subject": subject, "video": video,
                "prediction_id": prediction_id, "peak_time": int(peak),
                "raw_score": float(score[peak]), "smoothed_score": float(curve[peak]),
                "threshold": threshold, "margin_to_threshold": float(curve[peak] - threshold),
                "candidate_onset": int(peak - k_p), "candidate_offset": int(peak + k_p),
                "candidate_confidence": 0.0, "emotion_id": int(emotion_id),
                "max_iou_to_gt": float(max_iou), "closest_gt_id": closest_gt,
                "matched_gt_id": int(matched_gt), "final_prediction": final_prediction,
                "final_tp_or_fp": final_role, "fp_category": fp_category,
                "removal_stage": "S6_result_synergy" if neutral else "",
                "removal_reason": "neutral_prediction" if neutral else "",
            }
            prediction_rows.append(row)
            dataset_prediction_rows.append(row)

        # Candidate trace includes all local peaks so S2 threshold losses remain observable.
        thresholded_set = set(map(int, thresholded))
        for peak in all_peaks:
            peak = int(peak)
            generated = peak in thresholded_set
            pred_index = peak_to_pred.get(peak, -1)
            overlaps = [iou_interval(peak - k_p, peak + k_p, sample) for sample in samples]
            max_iou = max(overlaps, default=0.0)
            matched_gt = pred_to_gt[pred_index] if generated else -1
            emotion_id = emotions[pred_index] if generated else ""
            neutral = bool(generated and emotion_id == 4)
            candidate_rows.append({
                "dataset": dataset, "subject": subject, "video": video,
                "peak_time": peak, "raw_score": float(score[peak]),
                "smoothed_score": float(curve[peak]), "threshold": threshold,
                "margin_to_threshold": float(curve[peak] - threshold),
                "generated_as_peak": generated,
                "candidate_onset": int(peak - k_p), "candidate_offset": int(peak + k_p),
                "candidate_confidence": 0.0 if generated else "",
                "survived_filter": "not_applicable", "survived_nms": "not_applicable",
                "final_prediction": bool(generated and not neutral),
                "max_iou_to_gt": float(max_iou), "matched_gt_id": int(matched_gt),
                "final_tp_or_fp": (
                    "sub_threshold" if not generated else
                    "removed_neutral" if neutral else
                    "TP" if matched_gt >= 0 else "FP"
                ),
                "removal_stage": "S2_threshold" if not generated else "S6_result_synergy" if neutral else "",
                "removal_reason": "below_native_threshold" if not generated else "neutral_prediction" if neutral else "",
            })

        stage_tp["all_local_peaks"] += maximum_coverage(all_peaks, samples, k_p)
        stage_tp["thresholded"] += maximum_coverage(thresholded, samples, k_p)
        stage_tp["pre_final"] += maximum_coverage(thresholded, samples, k_p)

        for gt_index, sample in enumerate(samples):
            gt_id = f"{dataset}:{subject}:{video}:gt{gt_index}"
            onset, apex, offset = map(int, sample[:3])
            valid_geometry = offset >= onset
            in_score_range = onset >= 0 and offset < len(score)
            if not valid_geometry:
                data_quality_issues.append({
                    "record_index": video_order, "subject": subject, "video": video,
                    "gt_index": gt_index, "issue": "Malformed GT interval: offset < onset",
                    "GT": [onset, apex, offset], "score_length": len(score),
                    "handling": "native evaluator geometry retained; unmatched GT assigned FN-E_OTHER and excluded from duration/boundary summaries",
                })
            elif not in_score_range:
                data_quality_issues.append({
                    "record_index": video_order, "subject": subject, "video": video,
                    "gt_index": gt_index, "issue": "GT interval outside score range",
                    "GT": [onset, apex, offset], "score_length": len(score),
                    "handling": "native interval geometry retained; inside-GT score summary clipped, or null if disjoint",
                })
            duration = offset - onset + 1 if valid_geometry else None
            center = (onset + offset) / 2 if valid_geometry else None
            all_associated = [
                int(peak) for peak in all_peaks if onset - k_p <= int(peak) <= offset + k_p
            ]
            threshold_associated = [
                int(peak) for peak in thresholded if onset - k_p <= int(peak) <= offset + k_p
            ]
            correct_candidates = [
                int(peak) for peak in thresholded
                if iou_interval(int(peak) - k_p, int(peak) + k_p, sample) >= IOU_THRESHOLD
            ]
            correct_local_candidates = [
                int(peak) for peak in all_peaks
                if iou_interval(int(peak) - k_p, int(peak) + k_p, sample) >= IOU_THRESHOLD
            ]
            all_iou = [
                (iou_interval(int(peak) - k_p, int(peak) + k_p, sample), int(peak))
                for peak in thresholded
            ]
            best_iou, best_peak = max(all_iou, default=(0.0, -1))
            matched_pred_index = next(
                (index for index, matched_gt in enumerate(pred_to_gt) if matched_gt == gt_index), -1
            )
            final_is_tp = gt_index in final_matched_gt

            if not valid_geometry:
                category = "FN-E_OTHER"
                failure_reason = "malformed GT metadata (offset < onset); no scientific failure-stage attribution"
                chosen_peak = -1
            elif final_is_tp:
                category = "TP"
                failure_reason = ""
                chosen_peak = int(thresholded[matched_pred_index])
            elif not all_associated:
                category = "FN-A_NO_PEAK"
                failure_reason = "no local peak in [GT onset-k_p, GT offset+k_p]"
                chosen_peak = -1
            elif not threshold_associated:
                category = "FN-B_SUB_THRESHOLD"
                failure_reason = "associated local peak below native threshold"
                chosen_peak = max(all_associated, key=lambda peak: curve[peak])
                if correct_local_candidates:
                    fn_b_threshold_only_recoverable += 1
                threshold_values["fn_b_margin"].append(float(curve[chosen_peak] - threshold))
                threshold_values["fn_b_height"].append(float(curve[chosen_peak]))
                threshold_values["fn_b_rank"].append(rank_in_video(chosen_peak, all_peaks, curve))
            elif not correct_candidates:
                category = "FN-C_BOUNDARY"
                failure_reason = "associated thresholded peak but fixed interval IoU < 0.5"
                chosen_peak = max(
                    threshold_associated,
                    key=lambda peak: iou_interval(peak - k_p, peak + k_p, sample),
                )
                if onset <= chosen_peak <= offset:
                    peak_inside_boundary += 1
            elif matched_pred_index >= 0 and emotions[matched_pred_index] == 4:
                category = "FN-D_SUPPRESSED"
                failure_reason = "correct raw spotting match removed by neutral result_synergy"
                chosen_peak = int(thresholded[matched_pred_index])
            else:
                category = "FN-E_OTHER"
                failure_reason = "one-to-one evaluator assignment conflict"
                chosen_peak = max(correct_candidates, key=lambda peak: curve[peak])

            if category != "TP":
                fn_counts[category] += 1

            if category == "FN-C_BOUNDARY" and chosen_peak >= 0:
                linked = next(
                    (
                        row["prediction_id"] for row in dataset_prediction_rows
                        if row["subject"] == subject and row["video"] == video
                        and row["peak_time"] == chosen_peak
                        and row["final_tp_or_fp"] == "FP"
                        and row["fp_category"] == "FP-B_NEAR_GT_LOCALIZATION"
                    ),
                    None,
                )
                if linked:
                    boundary_linked_fp.add(linked)

            nearest_local_distance = (
                min(abs(int(peak) - apex) for peak in all_peaks) if len(all_peaks) else None
            )
            group = "matched_TP_GT" if final_is_tp else "FN_GT"
            diagnostic_peak = chosen_peak if chosen_peak >= 0 else best_peak
            center_error = (
                abs(diagnostic_peak - center)
                if valid_geometry and diagnostic_peak >= 0 else None
            )
            if valid_geometry:
                gt_group_values[group]["GT_duration"].append(duration)
                gt_group_values[group]["duration_ratio"].append(duration / (2 * k_p + 1))
                gt_group_values[group]["center_error"].append(center_error)
                gt_group_values[group]["best_native_candidate_IoU"].append(best_iou)
                if category == "FN-C_BOUNDARY":
                    gt_group_values["FN_C_boundary_only"]["GT_duration"].append(duration)
                    gt_group_values["FN_C_boundary_only"]["duration_ratio"].append(duration / (2 * k_p + 1))
                    gt_group_values["FN_C_boundary_only"]["center_error"].append(center_error)
                    gt_group_values["FN_C_boundary_only"]["best_native_candidate_IoU"].append(best_iou)

            gt_row = {
                "dataset": dataset, "subject": subject, "video": video,
                "gt_id": gt_id, "gt_index": gt_index, "gt_onset": onset,
                "gt_apex": apex, "gt_offset": offset, "gt_duration": duration,
                "valid_gt_geometry": valid_geometry, "in_score_range": in_score_range,
                "native_event_duration": 2 * k_p + 1,
                "duration_ratio": duration / (2 * k_p + 1) if valid_geometry else None,
                "final_status": "TP" if final_is_tp else "FN",
                "fn_category": "" if final_is_tp else category,
                "failure_reason": failure_reason,
                "max_raw_score_inside_gt": interval_max(score, onset, offset),
                "max_smoothed_score_inside_gt": interval_max(curve, onset, offset),
                "nearest_local_peak_distance_to_apex": nearest_local_distance,
                "diagnostic_peak": chosen_peak,
                "diagnostic_peak_score": float(curve[chosen_peak]) if chosen_peak >= 0 else None,
                "threshold": threshold,
                "peak_threshold_margin": float(curve[chosen_peak] - threshold) if chosen_peak >= 0 else None,
                "peak_rank_within_video": rank_in_video(chosen_peak, all_peaks, curve) if chosen_peak >= 0 else None,
                "candidate_onset": int(chosen_peak - k_p) if chosen_peak >= 0 else None,
                "candidate_offset": int(chosen_peak + k_p) if chosen_peak >= 0 else None,
                "center_error": center_error, "best_native_candidate_iou": float(best_iou),
                "peak_inside_gt_but_iou_lt_0_5": bool(
                    category == "FN-C_BOUNDARY" and onset <= chosen_peak <= offset and best_iou < 0.5
                ),
                "matched_prediction_index_pre_synergy": matched_pred_index,
                "threshold_only_recoverable_with_native_interval": bool(
                    category == "FN-B_SUB_THRESHOLD" and correct_local_candidates
                ),
            }
            gt_rows.append(gt_row)
            dataset_gt_rows.append(gt_row)

        # Supplementary comparator needed to assess what threshold lowering would admit.
        for peak in all_peaks:
            if curve[peak] >= threshold:
                continue
            max_iou = max(
                [iou_interval(int(peak) - k_p, int(peak) + k_p, sample) for sample in samples],
                default=0.0,
            )
            if max_iou == 0:
                threshold_values["subthreshold_background_margin"].append(float(curve[peak] - threshold))
                threshold_values["subthreshold_background_height"].append(float(curve[peak]))

    final_fn = gt_count - final_tp
    raw_fn = gt_count - raw_tp
    if sum(fn_counts.values()) != final_fn:
        raise AssertionError(f"{dataset}: FN decomposition does not conserve")
    if sum(fp_counts.values()) != final_fp:
        raise AssertionError(f"{dataset}: FP decomposition does not conserve")
    if stage_tp["pre_final"] != stage_tp["thresholded"]:
        raise AssertionError(f"{dataset}: nonexistent pre-final filtering changed ceiling")

    raw_metrics = metrics(raw_tp, raw_fp, raw_fn)
    final_metrics = metrics(final_tp, final_fp, final_fn)
    stage_ceiling = {}
    for stage in ["all_local_peaks", "thresholded", "pre_final"]:
        tp = int(stage_tp[stage])
        stage_ceiling[stage] = {
            "TP_ceiling": tp, "FN_remaining": gt_count - tp,
            "Recall_ceiling": tp / gt_count if gt_count else 0.0,
        }
    stage_ceiling["native_final"] = {
        "TP_ceiling": final_tp, "FN_remaining": final_fn,
        "Recall_ceiling": final_tp / gt_count if gt_count else 0.0,
    }

    def oracle(recover_fn=0, remove_fp=0):
        value = metrics(final_tp + recover_fn, final_fp - remove_fp, final_fn - recover_fn)
        value["delta_F1_vs_native"] = value["Spotting_F1"] - final_metrics["Spotting_F1"]
        return value

    boundary_count = int(fn_counts["FN-C_BOUNDARY"])
    suppression_count = int(fn_counts["FN-D_SUPPRESSED"])
    threshold_count = int(fn_counts["FN-B_SUB_THRESHOLD"])
    duplicate_count = int(fp_counts["FP-C_DUPLICATE"])
    oracles = {
        "Boundary_oracle": {
            **oracle(boundary_count, len(boundary_linked_fp)),
            "recovered_FN_C": boundary_count,
            "removed_linked_near_GT_FP": len(boundary_linked_fp),
            "label": "Oracle only — impossible upper bound, not experimental Method result.",
        },
        "Threshold_oracle": {
            **oracle(fn_b_threshold_only_recoverable, 0),
            "earliest_stage_FN_B": threshold_count,
            "recovered_FN_B_with_native_interval": fn_b_threshold_only_recoverable,
            "label": "Oracle only — impossible upper bound; removes threshold for recoverable FN-B while retaining native interval and assumes no new FP.",
        },
        "Suppression_oracle": {
            **oracle(suppression_count, 0), "recovered_FN_D": suppression_count,
            "label": "Oracle only — impossible upper bound.",
        },
        "Duplicate_oracle": {
            **oracle(0, duplicate_count), "removed_duplicate_FP": duplicate_count,
            "label": "Oracle only — impossible upper bound.",
        },
    }

    boundary_diagnostics = {
        "native_event_duration": 2 * k_p + 1,
        "peak_inside_GT_but_IoU_lt_0_5": peak_inside_boundary,
        "groups": {
            group: {name: distribution(values) for name, values in measures.items()}
            for group, measures in gt_group_values.items()
        },
    }
    threshold_diagnostics = {
        name: distribution(values) for name, values in threshold_values.items()
    }
    threshold_diagnostics["FN_B_threshold_only_recoverable_with_native_interval"] = (
        fn_b_threshold_only_recoverable
    )
    recognition = recognition_summary(recognition_gt, recognition_pred)
    recognition["neutral_TP_removed"] = neutral_tp_removed
    recognition["neutral_FP_removed"] = neutral_fp_removed
    recognition["excluded_matched_events_due_ambiguous_GT_emotion"] = recognition_excluded_ambiguous

    fn_table = {
        key: {"count": int(fn_counts[key]), "percent_all_FN": fn_counts[key] / final_fn if final_fn else 0.0}
        for key in ["FN-A_NO_PEAK", "FN-B_SUB_THRESHOLD", "FN-C_BOUNDARY", "FN-D_SUPPRESSED", "FN-E_OTHER"]
    }
    fp_table = {
        key: {"count": int(fp_counts[key]), "percent_all_FP": fp_counts[key] / final_fp if final_fp else 0.0}
        for key in ["FP-A_BACKGROUND", "FP-B_NEAR_GT_LOCALIZATION", "FP-C_DUPLICATE", "FP-D_OTHER"]
    }
    for category, value in fn_table.items():
        summary_rows.append({
            "dataset": dataset, "error_type": "FN", "category": category,
            "count": value["count"], "fraction": value["percent_all_FN"],
        })
    for category, value in fp_table.items():
        summary_rows.append({
            "dataset": dataset, "error_type": "FP", "category": category,
            "count": value["count"], "fraction": value["percent_all_FP"],
        })

    failure_scores = {
        "threshold": threshold_count,
        "boundary": boundary_count + int(fp_counts["FP-B_NEAR_GT_LOCALIZATION"]),
        "background_FP": int(fp_counts["FP-A_BACKGROUND"]),
        "suppression": suppression_count,
        "duplicate": duplicate_count,
        "evidence_absent": int(fn_counts["FN-A_NO_PEAK"]),
    }
    ranking = [
        {"rank": rank, "failure": name, "diagnostic_error_count": count}
        for rank, (name, count) in enumerate(
            sorted(failure_scores.items(), key=lambda item: (-item[1], item[0])), 1
        )
    ]

    return {
        "dataset": dataset,
        "cache": {
            "path": str(cache_path.resolve()), "sha256": sha256(cache_path),
            "subjects": len({str(row["subject"]) for row in payload["records"]}),
            "videos": len(payload["records"]), "GT_count": gt_count,
            "native_k_p": k_p,
            "available_payload_fields": sorted(payload.keys()),
            "available_record_fields": sorted(payload["records"][0].keys()),
            "data_quality_issues": data_quality_issues,
        },
        "native_raw_spotting": raw_metrics,
        "native_final_result_synergy": final_metrics,
        "FN_decomposition": fn_table,
        "FP_decomposition": fp_table,
        "stage_recall_ceiling": stage_ceiling,
        "boundary_diagnostics": boundary_diagnostics,
        "threshold_diagnostics": threshold_diagnostics,
        "duplicate_suppression": {
            "GT_with_more_than_one_thresholded_candidate_IoU_ge_0_5": sum(
                sum(
                    iou_interval(row["peak_time"] - k_p, row["peak_time"] + k_p, [gt["gt_onset"], gt["gt_apex"], gt["gt_offset"]]) >= 0.5
                    for row in dataset_prediction_rows
                    if row["subject"] == gt["subject"] and row["video"] == gt["video"]
                ) > 1
                for gt in dataset_gt_rows
            ),
            "duplicate_FP": duplicate_count,
            "correct_candidate_removed_by_result_synergy": suppression_count,
            "native_NMS_or_candidate_filter": "not present",
        },
        "recognition_secondary": recognition,
        "oracle_headroom": oracles,
        "failure_ranking": ranking,
    }


def choose_decision(results):
    boundary_checks = {}
    suppression_checks = {}
    threshold_checks = {}
    for dataset, result in results.items():
        fn = result["native_final_result_synergy"]["FN"]
        fp = result["native_final_result_synergy"]["FP"]
        fn_c = result["FN_decomposition"]["FN-C_BOUNDARY"]["count"]
        near_fp = result["FP_decomposition"]["FP-B_NEAR_GT_LOCALIZATION"]["count"]
        fn_d = result["FN_decomposition"]["FN-D_SUPPRESSED"]["count"]
        duplicate = result["FP_decomposition"]["FP-C_DUPLICATE"]["count"]
        fn_b = result["FN_decomposition"]["FN-B_SUB_THRESHOLD"]["count"]
        fn_b_margin = result["threshold_diagnostics"]["fn_b_margin"]["median"]
        boundary_delta = result["oracle_headroom"]["Boundary_oracle"]["delta_F1_vs_native"]
        suppression_delta = result["oracle_headroom"]["Suppression_oracle"]["delta_F1_vs_native"]
        threshold_delta = result["oracle_headroom"]["Threshold_oracle"]["delta_F1_vs_native"]
        bg_sub = result["threshold_diagnostics"]["subthreshold_background_margin"]
        fn_b_dist = result["threshold_diagnostics"]["fn_b_margin"]
        iqr_overlap = (
            max(0.0, min(fn_b_dist["q75"], bg_sub["q75"]) - max(fn_b_dist["q25"], bg_sub["q25"]))
            if fn_b_dist["n"] and bg_sub["n"] else None
        )
        threshold_checks[dataset] = {
            "FN_B_fraction_ge_0_20": fn_b / fn >= 0.20 if fn else False,
            "median_margin_within_0_05_below_threshold": fn_b_margin is not None and fn_b_margin >= -0.05,
            "FN_B_vs_subthreshold_background_IQR_nonoverlap": iqr_overlap == 0 if iqr_overlap is not None else False,
            "oracle_delta_F1_ge_0_03": threshold_delta >= 0.03,
            "observed_IQR_overlap": iqr_overlap,
        }
        boundary_checks[dataset] = {
            "FN_C_fraction_ge_0_10": fn_c / fn >= 0.10 if fn else False,
            "near_GT_FP_fraction_ge_0_05": near_fp / fp >= 0.05 if fp else False,
            "oracle_delta_F1_ge_0_03": boundary_delta >= 0.03,
        }
        suppression_checks[dataset] = {
            "FN_D_fraction_ge_0_05": fn_d / fn >= 0.05 if fn else False,
            "duplicate_FP_fraction_ge_0_10": duplicate / fp >= 0.10 if fp else False,
            "oracle_delta_F1_ge_0_02": suppression_delta >= 0.02,
        }

    boundary_go = all(all(row.values()) for row in boundary_checks.values())
    suppression_go = all(all(row.values()) for row in suppression_checks.values())
    threshold_go = all(
        all(value for key, value in row.items() if key != "observed_IQR_overlap")
        for row in threshold_checks.values()
    )
    eligible = [
        name for name, passed in [
            ("GO-BOUNDARY", boundary_go), ("GO-SUPPRESSION", suppression_go), ("GO-THRESHOLD", threshold_go)
        ] if passed
    ]
    if len(eligible) == 1:
        decision = eligible[0]
    elif len(eligible) > 1:
        # Deterministic tie-break by the minimum cross-dataset oracle F1 headroom.
        oracle_key = {
            "GO-BOUNDARY": "Boundary_oracle",
            "GO-SUPPRESSION": "Suppression_oracle",
            "GO-THRESHOLD": "Threshold_oracle",
        }
        decision = max(
            eligible,
            key=lambda name: min(
                result["oracle_headroom"][oracle_key[name]]["delta_F1_vs_native"]
                for result in results.values()
            ),
        )
    else:
        decision = "NO-INFERENCE-HEADROOM"
    return {
        "decision": decision,
        "eligible_directions": eligible,
        "fixed_operational_rules": {
            "boundary": "both datasets: FN-C/FN>=0.10, near-GT-FP/FP>=0.05, boundary oracle delta F1>=0.03",
            "suppression": "both datasets: FN-D/FN>=0.05, duplicate-FP/FP>=0.10, suppression oracle delta F1>=0.02",
            "threshold": "both datasets: FN-B/FN>=0.20, median FN-B margin>=-0.05, FN-B and subthreshold-background IQRs non-overlap, threshold oracle delta F1>=0.03",
        },
        "boundary_checks": boundary_checks,
        "suppression_checks": suppression_checks,
        "threshold_checks": threshold_checks,
        "tie_break": "largest minimum cross-dataset oracle delta F1 if multiple directions qualify",
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_pipeline_trace(path, inventories):
    lines = [
        "# ME-TST+ Native Inference Trace",
        "",
        "## 结论",
        "",
        "原生 spotting 路径没有 NMS、候选级 filter、冲突消解或 interval 合并。`result_synergy` 不是 spotting decoder；它在 IoU matching 与 recognition 之后，按 neutral 标签调整最终计数。",
        "",
        "## Exact pipeline",
        "",
        "| Stage | 输入 | 原生操作 | 输出 |",
        "|---|---|---|---|",
        "| S0 | frozen `score[T]` | 不修改 | raw score |",
        "| S1 | raw score | `np.convolve` moving average，宽度 `2*k_p`，`mode=same` | smoothed score |",
        "| S2 | smoothed score | `mean + 0.55*(max-mean)` | 每视频 threshold |",
        "| S3 | smoothed score | `scipy.signal.find_peaks(height=threshold, distance=k_p)` | thresholded peaks |",
        "| S4 | peak | `[peak-k_p, peak+k_p]`，不在生成阶段裁边；confidence 字段固定为 0 | spotting events |",
        "| S5 | event + emotion sequence | 区间内部多数票 | event emotion |",
        "| S6 | 已完成 IoU matching 的 event | strategy-1 `result_synergy` 移除 neutral TP/FP 并相应增加 FN | final reported counts |",
        "",
        "## 不存在的 stage",
        "",
        "- native NMS：不存在。",
        "- native downstream candidate filter：不存在。",
        "- native conflict resolution：不存在；只有 evaluator 的一对一 IoU matching。",
        "- mirror：cache 中虽有字段，但本 audit 完全不读取。",
        "",
        "## 数据集参数",
        "",
        "| Dataset | k_p | smoothing width | interval duration | strategy |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, inventory in inventories.items():
        kp = inventory["native_k_p"]
        lines.append(f"| {dataset} | {kp} | {2*kp} | {2*kp+1} | recognition_mainly_with_neutral_synergy |")
    lines.extend([
        "",
        "## 代码证据",
        "",
        "- `training_utils.py:9-13,15-41`：smoothing、Moilanen threshold、find_peaks、固定区间。",
        "- `paper_metrics.py:39-55`：paper-aligned 等价 candidate 实现。",
        "- `paper_metrics.py:79-87`：区间情绪多数票。",
        "- `compare_paper_aligned_strategies.py:243-284`：matching 后的 neutral result_synergy。",
        "",
        "本文件在任何 GT failure 分类计算之前由审计脚本写出。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path, results, decision):
    lines = [
        "# ME-TST+ Native Inference Failure-Mode Audit",
        "",
        "## 1. 审计性质",
        "",
        "本报告是 cache-only GT diagnostic/oracle analysis，不是 Method 结果。没有训练、re-forward、CUDA、新 threshold、新 NMS、新 decoder 或参数搜索。原始 cache 全程只读。",
        "",
        "## 2. Native pipeline",
        "",
        "`score → moving-average(2*k_p) → Moilanen p=0.55 → find_peaks(distance=k_p) → [peak-k_p,peak+k_p] → IoU matching → emotion vote → neutral result_synergy`。native candidate pipeline 不含 NMS/filter。完整证据见 `NATIVE_INFERENCE_TRACE_CN.md`。",
        "",
    ]
    for dataset, result in results.items():
        cache = result["cache"]
        lines.extend([
            f"## 3. {dataset}", "",
            f"- Cache：`{cache['path']}`", f"- SHA-256：`{cache['sha256']}`",
            f"- Subjects / videos / GT：{cache['subjects']} / {cache['videos']} / {cache['GT_count']}",
            f"- Native k_p：{cache['native_k_p']}",
            f"- Payload fields：`{', '.join(cache['available_payload_fields'])}`",
            f"- Record fields：`{', '.join(cache['available_record_fields'])}`",
            f"- Data-quality warnings：{len(cache['data_quality_issues'])} 条；原始 native geometry 保留，畸形区间不用于科学 boundary/duration 归因。",
            "",
            "### Native metrics", "",
            "| Endpoint | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in [("raw spotting", "native_raw_spotting"), ("final result_synergy", "native_final_result_synergy")]:
            value = result[key]
            lines.append(f"| {label} | {value['TP']} | {value['FP']} | {value['FN']} | {fmt(value['Precision'])} | {fmt(value['Recall'])} | {fmt(value['Spotting_F1'])} |")
        lines.extend(["", "### FN decomposition", "", "| Category | Count | % all FN |", "|---|---:|---:|"])
        for category, value in result["FN_decomposition"].items():
            lines.append(f"| {category} | {value['count']} | {100*value['percent_all_FN']:.2f}% |")
        lines.extend(["", "### FP decomposition", "", "| Category | Count | % all FP |", "|---|---:|---:|"])
        for category, value in result["FP_decomposition"].items():
            lines.append(f"| {category} | {value['count']} | {100*value['percent_all_FP']:.2f}% |")
        lines.extend(["", "### Stage-wise recall ceiling", "", "| Stage | TP ceiling | FN remaining | Recall ceiling |", "|---|---:|---:|---:|"])
        for stage, value in result["stage_recall_ceiling"].items():
            lines.append(f"| {stage} | {value['TP_ceiling']} | {value['FN_remaining']} | {fmt(value['Recall_ceiling'])} |")
        lines.extend(["", "### Boundary diagnostics", ""])
        lines.append(f"- `peak inside GT but IoU<0.5`：{result['boundary_diagnostics']['peak_inside_GT_but_IoU_lt_0_5']}")
        lines.append(f"- Native event duration：{result['boundary_diagnostics']['native_event_duration']}")
        lines.extend(["", "| Group | median GT duration | median ratio | median center error | median best IoU |", "|---|---:|---:|---:|---:|"])
        for group, values in result["boundary_diagnostics"]["groups"].items():
            lines.append(f"| {group} | {fmt(values['GT_duration']['median'])} | {fmt(values['duration_ratio']['median'])} | {fmt(values['center_error']['median'])} | {fmt(values['best_native_candidate_IoU']['median'])} |")
        td = result["threshold_diagnostics"]
        lines.extend(["", "### Threshold diagnostics", "", "| Group | n | median margin | IQR |", "|---|---:|---:|---:|"])
        for key in ["fn_b_margin", "final_background_fp_margin", "subthreshold_background_margin"]:
            value = td.get(key, distribution([]))
            lines.append(f"| {key} | {value['n']} | {fmt(value['median'])} | [{fmt(value['q25'])}, {fmt(value['q75'])}] |")
        lines.append(
            f"\n- FN-B 中仅移除 threshold、保留 native interval 即可恢复："
            f"{td['FN_B_threshold_only_recoverable_with_native_interval']}"
        )
        ds = result["duplicate_suppression"]
        lines.extend([
            "", "### Duplicate / suppression", "",
            f"- GT with >1 correct thresholded candidates：{ds['GT_with_more_than_one_thresholded_candidate_IoU_ge_0_5']}",
            f"- Duplicate FP：{ds['duplicate_FP']}",
            f"- Correct candidate removed by result_synergy：{ds['correct_candidate_removed_by_result_synergy']}",
            f"- Native NMS/filter：{ds['native_NMS_or_candidate_filter']}",
            "", "### Recognition secondary diagnosis", "",
            f"- Correct / wrong on final matched spotting：{result['recognition_secondary']['correct']} / {result['recognition_secondary']['wrong']}",
            f"- Recognition F1 (repo definition)：{fmt(result['recognition_secondary']['Recognition_F1_repo_definition'])}",
            f"- Neutral TP / FP removed：{result['recognition_secondary']['neutral_TP_removed']} / {result['recognition_secondary']['neutral_FP_removed']}",
            "", "### Oracle headroom", "",
            "| Oracle | TP | FP | FN | F1 | ΔF1 |", "|---|---:|---:|---:|---:|---:|",
        ])
        for oracle_name, value in result["oracle_headroom"].items():
            lines.append(f"| {oracle_name} | {value['TP']} | {value['FP']} | {value['FN']} | {fmt(value['Spotting_F1'])} | {fmt(value['delta_F1_vs_native'])} |")
        lines.extend(["", "### Failure ranking", ""])
        for row in result["failure_ranking"][:3]:
            lines.append(f"- #{row['rank']} {row['failure']}：{row['diagnostic_error_count']}")
        lines.append("")

    lines.extend([
        "## 4. Cross-Dataset Bottleneck", "",
        "为把任务中的定性 GO 条件落实为可复核的唯一输出，审计脚本使用固定的保守 operational rules；完整阈值和逐数据集布尔检查保存在 JSON。它们是审计决策规则，不是调参搜索，也不是性能结果。",
        "",
        f"- Boundary eligible：{all(all(row.values()) for row in decision['boundary_checks'].values())}",
        f"- Suppression eligible：{all(all(row.values()) for row in decision['suppression_checks'].values())}",
        f"- Threshold eligible：{'GO-THRESHOLD' in decision['eligible_directions']}",
        "", "## 5. 最终唯一 decision", "",
        f"**{decision['decision']}**", "",
    ])
    if decision["decision"] == "GO-BOUNDARY":
        lines.append("两个数据集都存在 thresholded peak 已形成、但固定 ±k_p interval 未达到 IoU=0.5 的同向 bottleneck，且 near-GT FP 与 boundary oracle headroom 同时达到预固定门槛。下一阶段只允许设计 training-free adaptive event-boundary inference。")
    elif decision["decision"] == "GO-SUPPRESSION":
        lines.append("两个数据集的 duplicate/suppression bottleneck 同时达到预固定门槛。下一阶段只允许研究 event-level conflict/suppression inference。")
    elif decision["decision"] == "GO-THRESHOLD":
        lines.append("两个数据集的 sub-threshold failure、margin separation 与 oracle headroom 同时达到严格门槛。下一阶段只允许研究 training-free threshold/recovery inference。")
    else:
        lines.append("当前 frozen ME-TST+ 输出没有显示足够统一、可利用的 inference-side bottleneck，不建议继续发明新的后处理 decoder。")
    lines.extend([
        "", "## 6. 限制", "",
        "- 所有 oracle 都使用 GT，是不可能上界，不是正式方法结果。",
        "- result_synergy 是 evaluator 的 matching 后计数调整，不是可独立观察的 native NMS。",
        "- all-local-peak ceiling 仍保留 native `distance=k_p`，只移除 height threshold。",
        "- FN-B 是 earliest-failure 分类；Threshold oracle 只计入放过 threshold 后以 native interval 已能 IoU≥0.5 的子集，避免把后续 boundary failure 也归功于 threshold。",
        "- threshold lowering 会引入多少额外 background peak 不能由 final background FP 单独回答，因此同时报告了 subthreshold background local-peak 分布。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = args.output_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    caches = {
        "SAMMLV": (args.sammlv_cache, *load_cache(args.sammlv_cache)),
        "CASME_3": (args.casme3_cache, *load_cache(args.casme3_cache)),
    }
    inventories = {
        name: {"native_k_p": int(payload["k_p"])} for name, (_, payload, _) in caches.items()
    }
    write_pipeline_trace(args.output_root / "NATIVE_INFERENCE_TRACE_CN.md", inventories)

    candidate_rows, gt_rows, prediction_rows, summary_rows = [], [], [], []
    results = {}
    for name, (path, payload, issues) in caches.items():
        results[name] = analyze_dataset(
            payload, path, issues, candidate_rows, gt_rows, prediction_rows, summary_rows
        )
    decision = choose_decision(results)
    artifact = {
        "status": "COMPLETE",
        "audit_type": "cache-only GT diagnostic/oracle; not a Method result",
        "constraints": {
            "training": False, "frozen_re_forward": False, "CUDA": False,
            "hidden_features": False, "mirror_used": False, "new_decoder": False,
            "parameter_search": False, "original_caches_modified": False,
        },
        "pipeline_trace": "NATIVE_INFERENCE_TRACE_CN.md",
        "datasets": results,
        "cross_dataset_decision": decision,
    }
    result_path = outputs / "native_failure_mode_results.json"
    result_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(outputs / "native_candidate_trace.csv", candidate_rows)
    write_csv(outputs / "gt_failure_trace.csv", gt_rows)
    write_csv(outputs / "prediction_failure_trace.csv", prediction_rows)
    write_csv(outputs / "failure_mode_summary.csv", summary_rows)
    write_report(args.output_root / "ME_TST_NATIVE_FAILURE_MODE_AUDIT_CN.md", results, decision)
    print(json.dumps({
        "status": "COMPLETE", "decision": decision["decision"],
        "results": str(result_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
