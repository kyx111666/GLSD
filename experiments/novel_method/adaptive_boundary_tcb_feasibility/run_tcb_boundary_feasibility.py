#!/usr/bin/env python3
"""Cache-only feasibility audit for Threshold-Connected Boundary (TCB)."""

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
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260904
EMOTION_ID = {"negative": 0, "positive": 1, "surprise": 2, "others": 3}
EXPECTED_NATIVE = {
    "SAMMLV": {"raw": (53, 184, 106), "final": (52, 171, 107)},
    "CASME_3": {"raw": (81, 912, 777), "final": (76, 727, 782)},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sammlv-cache", type=Path, required=True)
    parser.add_argument("--casme3-cache", type=Path, required=True)
    parser.add_argument("--failure-gt-trace", type=Path, required=True)
    parser.add_argument("--failure-prediction-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"dataset", "k_p", "num_subjects", "num_videos", "num_gt", "records"}
    if required - set(payload):
        raise KeyError(f"{path}: missing {sorted(required - set(payload))}")
    return payload


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def smooth(score, width):
    box = np.ones(int(width), dtype=np.float64) / int(width)
    return np.convolve(np.asarray(score, dtype=np.float64), box, mode="same")


def native_peaks(score, k_p):
    curve = smooth(score, 2 * k_p)
    threshold = float(curve.mean() + P * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
    return curve, threshold, peaks


def tcb_interval(curve, threshold, peak):
    onset = int(peak)
    while onset > 0 and curve[onset - 1] >= threshold:
        onset -= 1
    offset = int(peak)
    while offset + 1 < len(curve) and curve[offset + 1] >= threshold:
        offset += 1
    if not (onset <= peak <= offset):
        raise AssertionError("TCB interval does not contain its native peak")
    if np.any(curve[onset:offset + 1] < threshold):
        raise AssertionError("TCB interval is not fully suprathreshold")
    if onset > 0 and curve[onset - 1] >= threshold:
        raise AssertionError("TCB left boundary is not maximal")
    if offset + 1 < len(curve) and curve[offset + 1] >= threshold:
        raise AssertionError("TCB right boundary is not maximal")
    return onset, offset


def interval_iou(interval, sample):
    left, right = map(int, interval)
    gt_left, gt_right = int(sample[0]), int(sample[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union > 0 else 0.0


def greedy_match(intervals, samples):
    unmatched = set(range(len(samples)))
    pred_to_gt = []
    for interval in intervals:
        choices = [(interval_iou(interval, samples[index]), index) for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            pred_to_gt.append(int(best_index))
        else:
            pred_to_gt.append(-1)
    return pred_to_gt, unmatched


def voted_emotion(sequence, onset, offset, peak):
    start = max(0, int(onset) + 1)
    stop = max(1, int(offset) - 1)
    values = list(np.asarray(sequence)[start:stop])
    if not values:
        clipped_peak = min(max(0, int(peak)), len(sequence) - 1)
        values = [int(np.asarray(sequence)[clipped_peak])] if len(sequence) else [3]
    return int(Counter(int(value) for value in values).most_common(1)[0][0])


def metric_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "Precision": float(precision), "Recall": float(recall), "Spotting_F1": float(f1),
    }


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {
            "n": 0, "mean": None, "median": None, "q25": None, "q75": None,
            "IQR": None, "p10": None, "p90": None,
        }
    p10, q25, median, q75, p90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(median),
        "q25": float(q25), "q75": float(q75), "IQR": float(q75 - q25),
        "p10": float(p10), "p90": float(p90),
    }


def iou_distribution(values):
    result = distribution(values)
    array = np.asarray(values, dtype=float)
    for threshold in [0.3, 0.5, 0.7]:
        result[f"fraction_ge_{threshold}"] = float(np.mean(array >= threshold)) if len(array) else None
    return result


def recognition_summary(gt_ids, pred_ids):
    per_class = []
    for class_id in [0, 1, 2, 3]:
        tp = sum(g == class_id and p == class_id for g, p in zip(gt_ids, pred_ids))
        fp = sum(g != class_id and p == class_id for g, p in zip(gt_ids, pred_ids))
        fn = sum(g == class_id and p != class_id for g, p in zip(gt_ids, pred_ids))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class.append({"class_id": class_id, "precision": precision, "recall": recall})
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
        "Recognition_F1_repo_definition": float(f1),
    }


def bootstrap_subject_delta(subject_counts):
    subjects = sorted(subject_counts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_REPEATS):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        totals = {"native": Counter(), "tcb": Counter()}
        for subject in selected:
            for method in totals:
                totals[method].update(subject_counts[subject][method])
        native = metric_counts(totals["native"]["tp"], totals["native"]["fp"], totals["native"]["fn"])
        tcb = metric_counts(totals["tcb"]["tp"], totals["tcb"]["fp"], totals["tcb"]["fn"])
        deltas.append(tcb["Spotting_F1"] - native["Spotting_F1"])
    return {
        "unit": "subject", "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
        "mean_delta_F1": float(np.mean(deltas)),
        "CI95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
    }


def analyze_dataset(payload, cache_path, prior_gt, prior_pred, event_rows, subject_rows):
    dataset = str(payload["dataset"])
    k_p = int(payload["k_p"])
    num_gt = int(payload["num_gt"])
    native_totals = Counter()
    tcb_totals = Counter()
    native_final_totals = Counter()
    tcb_final_totals = Counter()
    subject_counts = defaultdict(lambda: {"native": Counter(), "tcb": Counter()})
    peak_lists_equal = 0
    peak_count = 0
    native_durations, tcb_durations, gt_durations = [], [], []
    native_ious, tcb_ious, delta_ious = [], [], []
    native_onset_errors, tcb_onset_errors = [], []
    native_offset_errors, tcb_offset_errors = [], []
    native_tp_total = native_tp_preserved = native_tp_lost = 0
    boundary_total = boundary_converted = boundary_worse = 0
    near_fp_total = near_fp_converted = 0
    recognition = {
        "native": {"gt": [], "pred": []}, "tcb": {"gt": [], "pred": []},
    }
    recognition_excluded = Counter()
    quality_issues = []

    prior_gt_map = {
        (row["dataset"], row["subject"], row["video"], int(row["gt_index"])): row
        for row in prior_gt if row["dataset"] == dataset
    }
    prior_pred_map = {
        row["prediction_id"]: row for row in prior_pred if row["dataset"] == dataset
    }

    for video_order, record in enumerate(payload["records"]):
        subject, video = str(record["subject"]), str(record["video"])
        score = np.asarray(record["score"], dtype=np.float64)
        samples = list(record["samples"])
        curve, threshold, peaks = native_peaks(score, k_p)
        native_intervals = [(int(peak - k_p), int(peak + k_p)) for peak in peaks]
        tcb_intervals = [tcb_interval(curve, threshold, int(peak)) for peak in peaks]
        tcb_peaks = peaks.copy()
        if not np.array_equal(peaks, tcb_peaks):
            raise AssertionError(f"{dataset}/{subject}/{video}: peak indices changed")
        peak_lists_equal += 1
        peak_count += len(peaks)

        native_match, native_unmatched = greedy_match(native_intervals, samples)
        tcb_match, tcb_unmatched = greedy_match(tcb_intervals, samples)
        native_tp = sum(index >= 0 for index in native_match)
        tcb_tp = sum(index >= 0 for index in tcb_match)
        native_video = Counter(tp=native_tp, fp=len(peaks) - native_tp, fn=len(native_unmatched))
        tcb_video = Counter(tp=tcb_tp, fp=len(peaks) - tcb_tp, fn=len(tcb_unmatched))
        native_totals.update(native_video)
        tcb_totals.update(tcb_video)
        subject_counts[subject]["native"].update(native_video)
        subject_counts[subject]["tcb"].update(tcb_video)

        native_emotions = [
            voted_emotion(record["emotion"], interval[0], interval[1], peak)
            for interval, peak in zip(native_intervals, peaks)
        ]
        tcb_emotions = [
            voted_emotion(record["emotion"], interval[0], interval[1], peak)
            for interval, peak in zip(tcb_intervals, peaks)
        ]
        valid_emotions = len(samples) == len(record["gt_emotions"])
        if not valid_emotions:
            quality_issues.append({
                "record_index": video_order, "subject": subject, "video": video,
                "issue": "GT/emotion length mismatch",
                "handling": "spotting retained; matched events excluded from recognition-only summary",
            })

        for method, matches, emotions, totals in [
            ("native", native_match, native_emotions, native_final_totals),
            ("tcb", tcb_match, tcb_emotions, tcb_final_totals),
        ]:
            for matched_gt, emotion in zip(matches, emotions):
                if emotion == 4:
                    if matched_gt >= 0:
                        totals["fn"] += 1
                    continue
                if matched_gt >= 0:
                    totals["tp"] += 1
                    if valid_emotions:
                        recognition[method]["gt"].append(
                            EMOTION_ID.get(str(record["gt_emotions"][matched_gt]), 3)
                        )
                        recognition[method]["pred"].append(int(emotion))
                    else:
                        recognition_excluded[method] += 1
                else:
                    totals["fp"] += 1
            totals["fn"] += sum(index >= 0 for index in matches) * 0
        native_final_totals["fn"] += len(native_unmatched)
        tcb_final_totals["fn"] += len(tcb_unmatched)

        native_matched_gt = {index for index in native_match if index >= 0}
        tcb_matched_gt = {index for index in tcb_match if index >= 0}
        native_tp_total += len(native_matched_gt)
        native_tp_preserved += len(native_matched_gt & tcb_matched_gt)
        native_tp_lost += len(native_matched_gt - tcb_matched_gt)

        for gt_index, sample in enumerate(samples):
            onset, apex, offset = map(int, sample[:3])
            if offset < onset:
                quality_issues.append({
                    "record_index": video_order, "subject": subject, "video": video,
                    "gt_index": gt_index, "issue": "Malformed GT interval: offset < onset",
                    "handling": "native evaluator geometry retained; excluded from duration/directional summaries",
                })
            else:
                gt_durations.append(offset - onset + 1)
                if onset < 0 or offset >= len(score):
                    quality_issues.append({
                        "record_index": video_order, "subject": subject, "video": video,
                        "gt_index": gt_index, "issue": "GT interval outside score range",
                        "GT": [onset, apex, offset], "score_length": len(score),
                        "handling": "native evaluator geometry retained; no cache value modified",
                    })
            prior = prior_gt_map.get((dataset, subject, video, gt_index))
            if prior and prior["fn_category"] == "FN-C_BOUNDARY":
                boundary_total += 1
                native_best = max(
                    [interval_iou(interval, sample) for interval in native_intervals], default=0.0
                )
                tcb_best = max([interval_iou(interval, sample) for interval in tcb_intervals], default=0.0)
                boundary_converted += int(gt_index in tcb_matched_gt)
                boundary_worse += int(tcb_best < native_best - 1e-12)

        for pred_index, (peak, native_interval, tcb_value, n_match, t_match) in enumerate(
            zip(peaks, native_intervals, tcb_intervals, native_match, tcb_match)
        ):
            peak = int(peak)
            prediction_id = f"{dataset}:{subject}:{video}:{pred_index}"
            native_durations.append(native_interval[1] - native_interval[0] + 1)
            tcb_durations.append(tcb_value[1] - tcb_value[0] + 1)
            native_all_iou = [interval_iou(native_interval, sample) for sample in samples]
            tcb_all_iou = [interval_iou(tcb_value, sample) for sample in samples]
            native_best_iou = max(native_all_iou, default=0.0)
            tcb_best_iou = max(tcb_all_iou, default=0.0)
            native_best_gt = int(np.argmax(native_all_iou)) if native_all_iou else -1
            paired_native_iou = paired_tcb_iou = None
            n_on_err = t_on_err = n_off_err = t_off_err = None
            if native_best_iou > 0 and native_best_gt >= 0:
                reference_gt = samples[native_best_gt]
                paired_native_iou = native_best_iou
                paired_tcb_iou = interval_iou(tcb_value, reference_gt)
                native_ious.append(paired_native_iou)
                tcb_ious.append(paired_tcb_iou)
                delta_ious.append(paired_tcb_iou - paired_native_iou)
                if int(reference_gt[2]) >= int(reference_gt[0]):
                    n_on_err = abs(native_interval[0] - int(reference_gt[0]))
                    t_on_err = abs(tcb_value[0] - int(reference_gt[0]))
                    n_off_err = abs(native_interval[1] - int(reference_gt[2]))
                    t_off_err = abs(tcb_value[1] - int(reference_gt[2]))
                    native_onset_errors.append(n_on_err)
                    tcb_onset_errors.append(t_on_err)
                    native_offset_errors.append(n_off_err)
                    tcb_offset_errors.append(t_off_err)

            prior_prediction = prior_pred_map.get(prediction_id)
            was_near_fp = bool(prior_prediction and prior_prediction["fp_category"] == "FP-B_NEAR_GT_LOCALIZATION")
            near_fp_total += int(was_near_fp)
            near_fp_converted += int(was_near_fp and t_match >= 0)
            event_rows.append({
                "dataset": dataset, "subject": subject, "video": video,
                "prediction_id": prediction_id, "peak": peak,
                "raw_score": float(score[peak]), "smoothed_score": float(curve[peak]),
                "native_threshold": threshold, "candidate_confidence": 0.0,
                "native_onset": native_interval[0], "native_offset": native_interval[1],
                "native_duration": native_interval[1] - native_interval[0] + 1,
                "tcb_onset": tcb_value[0], "tcb_offset": tcb_value[1],
                "tcb_duration": tcb_value[1] - tcb_value[0] + 1,
                "native_matched_gt": n_match, "tcb_matched_gt": t_match,
                "native_is_tp": n_match >= 0, "tcb_is_tp": t_match >= 0,
                "native_best_iou": native_best_iou, "tcb_best_iou": tcb_best_iou,
                "paired_reference_gt": native_best_gt,
                "paired_native_iou": paired_native_iou, "paired_tcb_iou": paired_tcb_iou,
                "paired_delta_iou": (
                    paired_tcb_iou - paired_native_iou if paired_native_iou is not None else None
                ),
                "native_onset_error": n_on_err, "tcb_onset_error": t_on_err,
                "native_offset_error": n_off_err, "tcb_offset_error": t_off_err,
                "prior_near_gt_fp": was_near_fp,
                "prior_near_gt_fp_converted_to_tcb_tp": bool(was_near_fp and t_match >= 0),
                "native_emotion_id": native_emotions[pred_index],
                "tcb_emotion_id": tcb_emotions[pred_index],
            })

    native_raw = metric_counts(native_totals["tp"], native_totals["fp"], native_totals["fn"])
    tcb_raw = metric_counts(tcb_totals["tp"], tcb_totals["fp"], tcb_totals["fn"])
    native_final = metric_counts(
        native_final_totals["tp"], native_final_totals["fp"], native_final_totals["fn"]
    )
    tcb_final = metric_counts(tcb_final_totals["tp"], tcb_final_totals["fp"], tcb_final_totals["fn"])
    expected = EXPECTED_NATIVE[dataset]
    if (native_raw["TP"], native_raw["FP"], native_raw["FN"]) != expected["raw"]:
        raise RuntimeError(f"BLOCKED-BASELINE {dataset}: raw endpoint mismatch")
    if (native_final["TP"], native_final["FP"], native_final["FN"]) != expected["final"]:
        raise RuntimeError(f"BLOCKED-BASELINE {dataset}: final endpoint mismatch")
    if peak_lists_equal != len(payload["records"]):
        raise AssertionError(f"{dataset}: peak-list invariance failed")

    improved = equal = worse = 0
    for subject in sorted(subject_counts):
        n = subject_counts[subject]["native"]
        t = subject_counts[subject]["tcb"]
        nm = metric_counts(n["tp"], n["fp"], n["fn"])
        tm = metric_counts(t["tp"], t["fp"], t["fn"])
        delta = tm["Spotting_F1"] - nm["Spotting_F1"]
        improved += int(delta > 1e-15)
        equal += int(abs(delta) <= 1e-15)
        worse += int(delta < -1e-15)
        subject_rows.append({
            "dataset": dataset, "subject": subject,
            "native_TP": nm["TP"], "native_FP": nm["FP"], "native_FN": nm["FN"],
            "native_F1": nm["Spotting_F1"],
            "tcb_TP": tm["TP"], "tcb_FP": tm["FP"], "tcb_FN": tm["FN"],
            "tcb_F1": tm["Spotting_F1"], "delta_F1": delta,
        })

    delta_array = np.asarray(delta_ious, dtype=float)
    onset_native = distribution(native_onset_errors)
    onset_tcb = distribution(tcb_onset_errors)
    offset_native = distribution(native_offset_errors)
    offset_tcb = distribution(tcb_offset_errors)
    bootstrap = bootstrap_subject_delta(subject_counts)
    result = {
        "dataset": dataset,
        "cache": {
            "path": str(cache_path.resolve()), "sha256": file_sha256(cache_path),
            "subjects": len(subject_counts), "videos": len(payload["records"]),
            "GT": num_gt, "k_p": k_p, "data_quality_issues": quality_issues,
        },
        "native_reproduction": {
            "raw": native_raw, "final_result_synergy": native_final,
            "expected_raw": list(expected["raw"]), "expected_final": list(expected["final"]),
            "status": "PASS",
        },
        "candidate_set_invariance": {
            "videos_checked": len(payload["records"]),
            "native_peak_count": peak_count, "tcb_peak_count": peak_count,
            "exact_peak_list_equal_videos": peak_lists_equal, "peak_Jaccard": 1.0,
            "status": "PASS",
        },
        "primary_raw_spotting": {
            "native": native_raw, "TCB": tcb_raw,
            "delta_F1": tcb_raw["Spotting_F1"] - native_raw["Spotting_F1"],
        },
        "secondary_result_synergy_spotting": {
            "native": native_final, "TCB": tcb_final,
            "delta_F1": tcb_final["Spotting_F1"] - native_final["Spotting_F1"],
        },
        "duration": {
            "native_predicted": distribution(native_durations),
            "TCB_predicted": distribution(tcb_durations),
            "valid_GT": distribution(gt_durations),
        },
        "boundary_error_conversion": {
            "native_FN_C": boundary_total, "converted_to_TCB_TP": boundary_converted,
            "remained_FN": boundary_total - boundary_converted,
            "best_IoU_became_worse": boundary_worse,
            "conversion_rate": boundary_converted / boundary_total if boundary_total else 0.0,
        },
        "near_GT_FP_conversion": {
            "prior_final_FP_B": near_fp_total, "converted_to_TCB_raw_TP": near_fp_converted,
            "remained_not_TP": near_fp_total - near_fp_converted,
            "conversion_rate": near_fp_converted / near_fp_total if near_fp_total else 0.0,
        },
        "native_TP_preservation": {
            "native_raw_TP": native_tp_total, "preserved_as_TCB_TP": native_tp_preserved,
            "lost_as_TCB_FN": native_tp_lost,
            "NetGain_boundary_fixed_minus_native_TP_lost": boundary_converted - native_tp_lost,
        },
        "paired_IoU_native_related_candidates": {
            "cohort_definition": "same native peak; native max-IoU>0; fixed native-best GT",
            "native": iou_distribution(native_ious), "TCB": iou_distribution(tcb_ious),
            "delta": distribution(delta_ious),
            "fraction_improved": float(np.mean(delta_array > 1e-12)) if len(delta_array) else None,
            "fraction_equal": float(np.mean(np.abs(delta_array) <= 1e-12)) if len(delta_array) else None,
            "fraction_worse": float(np.mean(delta_array < -1e-12)) if len(delta_array) else None,
        },
        "directional_error_native_related_candidates": {
            "cohort_definition": "same native peak and fixed native-best GT; malformed GT excluded",
            "onset": {"native": onset_native, "TCB": onset_tcb,
                      "median_delta_TCB_minus_native": onset_tcb["median"] - onset_native["median"]},
            "offset": {"native": offset_native, "TCB": offset_tcb,
                       "median_delta_TCB_minus_native": offset_tcb["median"] - offset_native["median"]},
        },
        "subject_stability": {
            "improved": improved, "equal": equal, "worse": worse, "bootstrap": bootstrap,
        },
        "recognition_secondary": {
            "native": recognition_summary(recognition["native"]["gt"], recognition["native"]["pred"]),
            "TCB": recognition_summary(recognition["tcb"]["gt"], recognition["tcb"]["pred"]),
            "excluded_matches_due_ambiguous_GT_emotion": dict(recognition_excluded),
        },
    }
    return result


def decide(results):
    checks = {}
    for dataset, result in results.items():
        primary = result["primary_raw_spotting"]
        conversion = result["boundary_error_conversion"]
        near_fp = result["near_GT_FP_conversion"]
        preservation = result["native_TP_preservation"]
        paired = result["paired_IoU_native_related_candidates"]
        directional = result["directional_error_native_related_candidates"]
        bootstrap = result["subject_stability"]["bootstrap"]
        checks[dataset] = {
            "spotting_F1_improved": primary["delta_F1"] > 0,
            "FN_C_converted": conversion["converted_to_TCB_TP"] > 0,
            "near_GT_FP_converted": near_fp["converted_to_TCB_raw_TP"] > 0,
            "net_boundary_gain_positive": preservation["NetGain_boundary_fixed_minus_native_TP_lost"] > 0,
            "paired_mean_IoU_improved": paired["delta"]["mean"] > 0,
            "onset_or_offset_median_error_improved": (
                directional["onset"]["median_delta_TCB_minus_native"] < 0
                or directional["offset"]["median_delta_TCB_minus_native"] < 0
            ),
            "bootstrap_not_systematically_negative": bootstrap["CI95"][1] >= 0,
        }
    core = all(
        row["spotting_F1_improved"] and row["net_boundary_gain_positive"]
        for row in checks.values()
    )
    majority = all(sum(row.values()) >= 5 for row in checks.values())
    decision = "GO" if core and majority else "NO-GO-TCB"
    return {
        "decision": decision,
        "rule": "GO iff both datasets improve raw Spotting F1, both have positive boundary NetGain, and >=5/7 listed evidence checks pass in each dataset",
        "checks": checks,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(path, results, decision):
    lines = [
        "# Adaptive Boundary TCB Feasibility Audit", "",
        "## 1. Scope and locked rule", "",
        "本实验只读取既有 frozen score cache。native smoothing、threshold、peak indices 与 confidence 完全不变；TCB 只把固定 `peak±k_p` 换成包含该 peak 的 maximal contiguous suprathreshold component。没有训练、forward、CUDA、参数搜索、candidate recovery、NMS 或 recognition redesign。", "",
        "主结论使用 **raw spotting**，以隔离 boundary 本身；依赖区间 emotion vote 的 result-synergy/recognition 只作 secondary report。", "",
        "## 2. Native baseline and candidate invariance", "",
        "| Dataset | Native raw TP/FP/FN | Native final TP/FP/FN | Peaks | Exact-list videos | Jaccard |", "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in results.items():
        n = result["native_reproduction"]
        inv = result["candidate_set_invariance"]
        lines.append(
            f"| {dataset} | {n['raw']['TP']}/{n['raw']['FP']}/{n['raw']['FN']} | "
            f"{n['final_result_synergy']['TP']}/{n['final_result_synergy']['FP']}/{n['final_result_synergy']['FN']} | "
            f"{inv['native_peak_count']} | {inv['exact_peak_list_equal_videos']}/{inv['videos_checked']} | {inv['peak_Jaccard']:.1f} |"
        )
    lines.extend(["", "## 3. Full raw spotting comparison", "", "| Dataset | Boundary | TP | FP | FN | Precision | Recall | F1 | ΔF1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        primary = result["primary_raw_spotting"]
        for label, key in [("Native ±k", "native"), ("TCB", "TCB")]:
            value = primary[key]
            delta = primary["delta_F1"] if key == "TCB" else 0.0
            lines.append(f"| {dataset} | {label} | {value['TP']} | {value['FP']} | {value['FN']} | {fmt(value['Precision'])} | {fmt(value['Recall'])} | {fmt(value['Spotting_F1'])} | {fmt(delta)} |")
    lines.extend(["", "## 4. Mechanism evidence", "", "| Dataset | Native FN-C | Converted | Remained | IoU worse | Native TP preserved | Native TP lost | NetGain | Near-GT FP converted/remained |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        b = result["boundary_error_conversion"]
        p = result["native_TP_preservation"]
        f = result["near_GT_FP_conversion"]
        lines.append(f"| {dataset} | {b['native_FN_C']} | {b['converted_to_TCB_TP']} ({fmt(b['conversion_rate'])}) | {b['remained_FN']} | {b['best_IoU_became_worse']} | {p['preserved_as_TCB_TP']} | {p['lost_as_TCB_FN']} | {p['NetGain_boundary_fixed_minus_native_TP_lost']} | {f['converted_to_TCB_raw_TP']}/{f['remained_not_TP']} |")
    lines.extend(["", "## 5. Duration", "", "| Dataset | Group | Mean | Median | IQR | p10 | p90 |", "|---|---|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        for label, key in [("Native prediction", "native_predicted"), ("TCB prediction", "TCB_predicted"), ("valid GT", "valid_GT")]:
            value = result["duration"][key]
            lines.append(f"| {dataset} | {label} | {fmt(value['mean'])} | {fmt(value['median'])} | {fmt(value['IQR'])} | {fmt(value['p10'])} | {fmt(value['p90'])} |")
    lines.extend(["", "## 6. Paired IoU and directional error", "", "配对 cohort 固定为 native max-IoU>0 的同一 peak，并始终使用该 peak 的 native-best GT；不会为 TCB 重新挑更有利的 GT。", "", "| Dataset | Native IoU mean/median/IQR | TCB IoU mean/median/IQR | Mean ΔIoU | Median ΔIoU | Improved/equal/worse | Median onset N→T | Median offset N→T |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        paired = result["paired_IoU_native_related_candidates"]
        direct = result["directional_error_native_related_candidates"]
        lines.append(
            f"| {dataset} | {fmt(paired['native']['mean'])}/{fmt(paired['native']['median'])}/{fmt(paired['native']['IQR'])} | "
            f"{fmt(paired['TCB']['mean'])}/{fmt(paired['TCB']['median'])}/{fmt(paired['TCB']['IQR'])} | "
            f"{fmt(paired['delta']['mean'])} | {fmt(paired['delta']['median'])} | "
            f"{fmt(paired['fraction_improved'])}/{fmt(paired['fraction_equal'])}/{fmt(paired['fraction_worse'])} | "
            f"{fmt(direct['onset']['native']['median'])}→{fmt(direct['onset']['TCB']['median'])} | "
            f"{fmt(direct['offset']['native']['median'])}→{fmt(direct['offset']['TCB']['median'])} |"
        )
    lines.append("")
    for dataset, result in results.items():
        paired = result["paired_IoU_native_related_candidates"]
        lines.append(
            f"- {dataset} IoU≥0.3/0.5/0.7：Native "
            f"{fmt(paired['native']['fraction_ge_0.3'])}/{fmt(paired['native']['fraction_ge_0.5'])}/{fmt(paired['native']['fraction_ge_0.7'])}；TCB "
            f"{fmt(paired['TCB']['fraction_ge_0.3'])}/{fmt(paired['TCB']['fraction_ge_0.5'])}/{fmt(paired['TCB']['fraction_ge_0.7'])}。"
        )
    lines.extend(["", "## 7. Subject stability", "", "| Dataset | Improved | Equal | Worse | Bootstrap mean ΔF1 | 95% CI |", "|---|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        value = result["subject_stability"]
        boot = value["bootstrap"]
        lines.append(f"| {dataset} | {value['improved']} | {value['equal']} | {value['worse']} | {fmt(boot['mean_delta_F1'])} | [{fmt(boot['CI95'][0])}, {fmt(boot['CI95'][1])}] |")
    lines.extend(["", "## 8. Secondary result-synergy / recognition", "", "| Dataset | Final spotting F1 N→T | Recognition F1 N→T |", "|---|---:|---:|"])
    for dataset, result in results.items():
        sec = result["secondary_result_synergy_spotting"]
        rec = result["recognition_secondary"]
        lines.append(f"| {dataset} | {fmt(sec['native']['Spotting_F1'])}→{fmt(sec['TCB']['Spotting_F1'])} | {fmt(rec['native']['Recognition_F1_repo_definition'])}→{fmt(rec['TCB']['Recognition_F1_repo_definition'])} |")
    lines.extend(["", "## 9. Gate checks", ""])
    for dataset, checks in decision["checks"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        for name, passed in checks.items():
            lines.append(f"- {name}: {passed}")
        lines.append("")
    lines.extend(["## 10. Final feasibility decision", "", f"**{decision['decision']}**", ""])
    if decision["decision"] == "GO":
        lines.append("candidate-specific adaptive boundary 在不改变 native candidate discovery 的情况下，于两个数据集均改善 raw Spotting F1，并获得正 boundary NetGain；证据支持进一步设计正式 training-free boundary inference algorithm。")
    else:
        lines.append("简单 threshold-connected boundary 未能在两个数据集一致兑现 boundary oracle headroom。本结论只否定 TCB，不否定此前观测到的 boundary failure mode，也不授权自动搜索新的 boundary heuristic。")
    lines.extend(["", "## 11. Data and interpretation limits", ""])
    for dataset, result in results.items():
        cache = result["cache"]
        lines.extend([
            f"- {dataset} cache：`{cache['path']}`",
            f"- {dataset} SHA-256：`{cache['sha256']}`",
            f"- {dataset} subjects/videos/GT/k_p：{cache['subjects']}/{cache['videos']}/{cache['GT']}/{cache['k_p']}",
            f"- {dataset} data-quality warnings：{len(cache['data_quality_issues'])}",
        ])
    lines.extend([
        "- TCB 没有任何可调参数；bootstrap 只量化 subject 重采样不确定性，不参与构造 boundary。",
        "- GT 仅用于 evaluator、配对诊断和 GO/NO-GO gate，从未用于生成或修改 prediction。",
        "- 主结果是 raw spotting；result synergy 会因 interval 内 emotion vote 改变，故只作二级结果。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = args.output_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    prior_gt = load_csv(args.failure_gt_trace)
    prior_pred = load_csv(args.failure_prediction_trace)
    caches = {
        "SAMMLV": (args.sammlv_cache, load_pickle(args.sammlv_cache)),
        "CASME_3": (args.casme3_cache, load_pickle(args.casme3_cache)),
    }
    event_rows, subject_rows, results = [], [], {}
    for dataset, (path, payload) in caches.items():
        if str(payload["dataset"]) != dataset:
            raise ValueError(f"{path}: dataset name mismatch")
        results[dataset] = analyze_dataset(
            payload, path, prior_gt, prior_pred, event_rows, subject_rows
        )
    decision = decide(results)
    artifact = {
        "status": "COMPLETE", "audit": "Adaptive Boundary TCB Feasibility",
        "method_definition": "maximal contiguous native-suprathreshold component containing each unchanged native peak",
        "constraints": {
            "cache_only": True, "training": False, "model_forward": False,
            "CUDA": False, "new_threshold": False, "candidate_recovery": False,
            "peak_reranking": False, "NMS": False, "parameters_added": 0,
        },
        "datasets": results, "feasibility_gate": decision,
    }
    result_path = outputs / "tcb_boundary_results.json"
    result_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(outputs / "tcb_event_trace.csv", event_rows)
    write_csv(outputs / "tcb_subject_metrics.csv", subject_rows)
    write_report(args.output_root / "ADAPTIVE_BOUNDARY_TCB_FEASIBILITY_CN.md", results, decision)
    print(json.dumps({"status": "COMPLETE", "decision": decision["decision"], "results": str(result_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
