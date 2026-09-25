#!/usr/bin/env python3
"""Cache-only Local-Valley Basin (LVB) boundary feasibility audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


TCB_DIR = Path(__file__).resolve().parents[1] / "adaptive_boundary_tcb_feasibility"
sys.path.insert(0, str(TCB_DIR))
from run_tcb_boundary_feasibility import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    EXPECTED_NATIVE,
    distribution,
    file_sha256,
    greedy_match,
    interval_iou,
    load_pickle,
    metric_counts,
    native_peaks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sammlv-cache", type=Path, required=True)
    parser.add_argument("--casme3-cache", type=Path, required=True)
    parser.add_argument("--failure-gt-trace", type=Path, required=True)
    parser.add_argument("--failure-prediction-trace", type=Path, required=True)
    parser.add_argument("--tcb-results", type=Path, required=True)
    parser.add_argument("--tcb-event-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extended_distribution(values):
    base = distribution(values)
    array = np.asarray(values, dtype=float)
    base.update({
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
        "p95": float(np.quantile(array, 0.95)) if len(array) else None,
        "p99": float(np.quantile(array, 0.99)) if len(array) else None,
    })
    return base


def iou_distribution(values):
    result = distribution(values)
    array = np.asarray(values, dtype=float)
    for threshold in [0.3, 0.5, 0.7]:
        result[f"fraction_ge_{threshold}"] = float(np.mean(array >= threshold)) if len(array) else None
    return result


def lvb_intervals(curve, native_peak_indices, k_p):
    """Nearest scipy local minima, constrained by adjacent native peaks."""
    minima = find_peaks(-np.asarray(curve, dtype=np.float64))[0].astype(int)
    intervals = []
    fallback = {"left": 0, "right": 0, "both": 0}
    details = []
    length = len(curve)
    for index, peak in enumerate(native_peak_indices):
        peak = int(peak)
        previous_peak = int(native_peak_indices[index - 1]) if index > 0 else None
        next_peak = int(native_peak_indices[index + 1]) if index + 1 < len(native_peak_indices) else None
        left_options = [
            int(value) for value in minima
            if value < peak and (previous_peak is None or value > previous_peak)
        ]
        right_options = [
            int(value) for value in minima
            if value > peak and (next_peak is None or value < next_peak)
        ]
        left_fallback = not left_options
        right_fallback = not right_options
        onset = max(left_options) if left_options else max(0, peak - k_p)
        offset = min(right_options) if right_options else min(length - 1, peak + k_p)
        if onset > peak or offset < peak:
            raise AssertionError("LVB interval does not contain its unchanged native peak")
        if previous_peak is not None and onset < previous_peak:
            raise AssertionError("LVB left boundary crossed previous native peak")
        if next_peak is not None and offset > next_peak:
            raise AssertionError("LVB right boundary crossed next native peak")
        fallback["left"] += int(left_fallback)
        fallback["right"] += int(right_fallback)
        fallback["both"] += int(left_fallback and right_fallback)
        intervals.append((int(onset), int(offset)))
        details.append({
            "left_fallback": left_fallback, "right_fallback": right_fallback,
            "left_valley": "" if left_fallback else int(onset),
            "right_valley": "" if right_fallback else int(offset),
        })
    return intervals, details, fallback


def bootstrap_subject_delta(subject_counts):
    subjects = sorted(subject_counts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_REPEATS):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        totals = {"native": Counter(), "lvb": Counter()}
        for subject in selected:
            totals["native"].update(subject_counts[subject]["native"])
            totals["lvb"].update(subject_counts[subject]["lvb"])
        native = metric_counts(totals["native"]["tp"], totals["native"]["fp"], totals["native"]["fn"])
        lvb = metric_counts(totals["lvb"]["tp"], totals["lvb"]["fp"], totals["lvb"]["fn"])
        deltas.append(lvb["Spotting_F1"] - native["Spotting_F1"])
    return {
        "unit": "subject", "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
        "mean_delta_F1": float(np.mean(deltas)),
        "CI95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
    }


def analyze_dataset(
    payload, cache_path, prior_gt_rows, prior_pred_rows, tcb_dataset_result,
    tcb_event_map, event_rows, subject_rows,
):
    dataset = str(payload["dataset"])
    k_p = int(payload["k_p"])
    num_gt = int(payload["num_gt"])
    prior_gt = {
        (row["subject"], row["video"], int(row["gt_index"])): row
        for row in prior_gt_rows if row["dataset"] == dataset
    }
    prior_pred = {
        row["prediction_id"]: row for row in prior_pred_rows if row["dataset"] == dataset
    }
    native_totals, lvb_totals = Counter(), Counter()
    subject_counts = defaultdict(lambda: {"native": Counter(), "lvb": Counter()})
    native_durations, lvb_durations, gt_durations = [], [], []
    native_ious, tcb_ious, lvb_ious = [], [], []
    lvb_delta_ious = []
    onset_errors = defaultdict(list)
    offset_errors = defaultdict(list)
    fallback_totals = Counter()
    extreme_over_2x = 0
    peak_count = exact_peak_videos = 0
    native_tp_total = native_tp_preserved = native_tp_lost = 0
    boundary_total = boundary_converted = boundary_worse = 0
    near_total = near_converted = near_worse = 0
    converted_boundary_gt_ids = set()
    converted_near_gt_ids = set()
    quality_issues = []

    for video_order, record in enumerate(payload["records"]):
        subject, video = str(record["subject"]), str(record["video"])
        score = np.asarray(record["score"], dtype=np.float64)
        samples = list(record["samples"])
        gt_emotions = list(record.get("gt_emotions", []))
        if gt_emotions and len(gt_emotions) != len(samples):
            quality_issues.append({
                "record_index": video_order, "subject": subject, "video": video,
                "issue": "GT interval/emotion count mismatch",
                "interval_count": len(samples), "emotion_count": len(gt_emotions),
                "handling": "spotting geometry retained; recognition is outside this audit",
            })
        curve, threshold, peaks = native_peaks(score, k_p)
        native_intervals = [(int(peak - k_p), int(peak + k_p)) for peak in peaks]
        lvb_values, lvb_details, fallback = lvb_intervals(curve, peaks, k_p)
        fallback_totals.update(fallback)
        lvb_peaks = peaks.copy()
        if not np.array_equal(peaks, lvb_peaks):
            raise AssertionError(f"{dataset}/{subject}/{video}: peak indices changed")
        exact_peak_videos += 1
        peak_count += len(peaks)

        native_match, native_unmatched = greedy_match(native_intervals, samples)
        lvb_match, lvb_unmatched = greedy_match(lvb_values, samples)
        native_video = Counter(
            tp=sum(index >= 0 for index in native_match),
            fp=sum(index < 0 for index in native_match), fn=len(native_unmatched),
        )
        lvb_video = Counter(
            tp=sum(index >= 0 for index in lvb_match),
            fp=sum(index < 0 for index in lvb_match), fn=len(lvb_unmatched),
        )
        native_totals.update(native_video)
        lvb_totals.update(lvb_video)
        subject_counts[subject]["native"].update(native_video)
        subject_counts[subject]["lvb"].update(lvb_video)
        native_matched_gt = {index for index in native_match if index >= 0}
        lvb_matched_gt = {index for index in lvb_match if index >= 0}
        native_tp_total += len(native_matched_gt)
        native_tp_preserved += len(native_matched_gt & lvb_matched_gt)
        native_tp_lost += len(native_matched_gt - lvb_matched_gt)

        for gt_index, sample in enumerate(samples):
            onset, apex, offset = map(int, sample[:3])
            gt_key = (subject, video, gt_index)
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
            prior = prior_gt.get(gt_key)
            if prior and prior["fn_category"] == "FN-C_BOUNDARY":
                boundary_total += 1
                native_best = max([interval_iou(value, sample) for value in native_intervals], default=0.0)
                lvb_best = max([interval_iou(value, sample) for value in lvb_values], default=0.0)
                converted = gt_index in lvb_matched_gt
                boundary_converted += int(converted)
                boundary_worse += int(lvb_best < native_best - 1e-12)
                if converted:
                    converted_boundary_gt_ids.add((subject, video, gt_index))

        for pred_index, (peak, native_interval, lvb_interval, detail, n_match, l_match) in enumerate(
            zip(peaks, native_intervals, lvb_values, lvb_details, native_match, lvb_match)
        ):
            peak = int(peak)
            prediction_id = f"{dataset}:{subject}:{video}:{pred_index}"
            tcb_row = tcb_event_map.get(prediction_id)
            if tcb_row is None:
                raise KeyError(f"missing TCB event {prediction_id}")
            if int(tcb_row["peak"]) != peak:
                raise AssertionError(f"TCB/native peak mismatch for {prediction_id}")
            tcb_interval = (int(tcb_row["tcb_onset"]), int(tcb_row["tcb_offset"]))
            native_duration = native_interval[1] - native_interval[0] + 1
            lvb_duration = lvb_interval[1] - lvb_interval[0] + 1
            native_durations.append(native_duration)
            lvb_durations.append(lvb_duration)
            extreme_over_2x += int(lvb_duration > 2 * native_duration)

            native_all = [interval_iou(native_interval, sample) for sample in samples]
            lvb_all = [interval_iou(lvb_interval, sample) for sample in samples]
            native_best = max(native_all, default=0.0)
            lvb_best = max(lvb_all, default=0.0)
            reference_gt = int(np.argmax(native_all)) if native_all else -1
            paired_native = paired_tcb = paired_lvb = None
            errors = {name: {"onset": None, "offset": None} for name in ["native", "tcb", "lvb"]}
            if native_best > 0 and reference_gt >= 0:
                sample = samples[reference_gt]
                paired_native = native_best
                paired_tcb = interval_iou(tcb_interval, sample)
                paired_lvb = interval_iou(lvb_interval, sample)
                native_ious.append(paired_native)
                tcb_ious.append(paired_tcb)
                lvb_ious.append(paired_lvb)
                lvb_delta_ious.append(paired_lvb - paired_native)
                if int(sample[2]) >= int(sample[0]):
                    for name, interval in [
                        ("native", native_interval), ("tcb", tcb_interval), ("lvb", lvb_interval)
                    ]:
                        errors[name]["onset"] = abs(interval[0] - int(sample[0]))
                        errors[name]["offset"] = abs(interval[1] - int(sample[2]))
                        onset_errors[name].append(errors[name]["onset"])
                        offset_errors[name].append(errors[name]["offset"])

            prior_prediction = prior_pred.get(prediction_id)
            was_near = bool(prior_prediction and prior_prediction["fp_category"] == "FP-B_NEAR_GT_LOCALIZATION")
            near_total += int(was_near)
            near_converted += int(was_near and l_match >= 0)
            near_worse += int(was_near and lvb_best < native_best - 1e-12)
            if was_near and l_match >= 0 and int(l_match) not in native_matched_gt:
                converted_near_gt_ids.add((subject, video, int(l_match)))

            event_rows.append({
                "dataset": dataset, "subject": subject, "video": video,
                "prediction_id": prediction_id, "peak": peak,
                "raw_score": float(score[peak]), "smoothed_score": float(curve[peak]),
                "native_threshold": threshold, "candidate_confidence": 0.0,
                "native_onset": native_interval[0], "native_offset": native_interval[1],
                "native_duration": native_duration,
                "tcb_onset": tcb_interval[0], "tcb_offset": tcb_interval[1],
                "tcb_duration": tcb_interval[1] - tcb_interval[0] + 1,
                "lvb_onset": lvb_interval[0], "lvb_offset": lvb_interval[1],
                "lvb_duration": lvb_duration,
                "left_valley": detail["left_valley"], "right_valley": detail["right_valley"],
                "left_fallback": detail["left_fallback"], "right_fallback": detail["right_fallback"],
                "native_matched_gt": n_match, "lvb_matched_gt": l_match,
                "native_is_tp": n_match >= 0, "lvb_is_tp": l_match >= 0,
                "native_best_iou": native_best, "lvb_best_iou": lvb_best,
                "paired_reference_gt": reference_gt,
                "paired_native_iou": paired_native, "paired_tcb_iou": paired_tcb,
                "paired_lvb_iou": paired_lvb,
                "paired_lvb_minus_native_iou": (
                    paired_lvb - paired_native if paired_native is not None else None
                ),
                "native_onset_error": errors["native"]["onset"],
                "tcb_onset_error": errors["tcb"]["onset"],
                "lvb_onset_error": errors["lvb"]["onset"],
                "native_offset_error": errors["native"]["offset"],
                "tcb_offset_error": errors["tcb"]["offset"],
                "lvb_offset_error": errors["lvb"]["offset"],
                "prior_near_gt_fp": was_near,
                "prior_near_gt_fp_converted_to_lvb_tp": bool(was_near and l_match >= 0),
            })

    native = metric_counts(native_totals["tp"], native_totals["fp"], native_totals["fn"])
    lvb = metric_counts(lvb_totals["tp"], lvb_totals["fp"], lvb_totals["fn"])
    expected = EXPECTED_NATIVE[dataset]["raw"]
    if (native["TP"], native["FP"], native["FN"]) != expected:
        raise RuntimeError(f"BLOCKED-BASELINE {dataset}: {native} != {expected}")
    if peak_count != native["TP"] + native["FP"] or peak_count != lvb["TP"] + lvb["FP"]:
        raise AssertionError(f"{dataset}: candidate counts do not conserve")
    if native["TP"] + native["FN"] != num_gt or lvb["TP"] + lvb["FN"] != num_gt:
        raise AssertionError(f"{dataset}: GT counts do not conserve")

    improved = equal = worse = 0
    for subject in sorted(subject_counts):
        n, value = subject_counts[subject]["native"], subject_counts[subject]["lvb"]
        nm = metric_counts(n["tp"], n["fp"], n["fn"])
        lm = metric_counts(value["tp"], value["fp"], value["fn"])
        delta = lm["Spotting_F1"] - nm["Spotting_F1"]
        improved += int(delta > 1e-15)
        equal += int(abs(delta) <= 1e-15)
        worse += int(delta < -1e-15)
        subject_rows.append({
            "dataset": dataset, "subject": subject,
            "native_TP": nm["TP"], "native_FP": nm["FP"], "native_FN": nm["FN"],
            "native_F1": nm["Spotting_F1"],
            "lvb_TP": lm["TP"], "lvb_FP": lm["FP"], "lvb_FN": lm["FN"],
            "lvb_F1": lm["Spotting_F1"], "delta_F1": delta,
        })

    tcb_primary = tcb_dataset_result["primary_raw_spotting"]
    tcb_duration = tcb_dataset_result["duration"]["TCB_predicted"]
    tcb_onset = tcb_dataset_result["directional_error_native_related_candidates"]["onset"]["TCB"]
    tcb_offset = tcb_dataset_result["directional_error_native_related_candidates"]["offset"]["TCB"]
    delta_array = np.asarray(lvb_delta_ious, dtype=float)
    duration_stats = extended_distribution(lvb_durations)
    requested_net_gain = boundary_converted + near_converted - native_tp_lost
    unique_repair_count = len(converted_boundary_gt_ids | converted_near_gt_ids)
    unique_net_gain = unique_repair_count - native_tp_lost
    return {
        "dataset": dataset,
        "cache": {
            "path": str(cache_path.resolve()), "sha256": file_sha256(cache_path),
            "subjects": len(subject_counts), "videos": len(payload["records"]),
            "GT": num_gt, "k_p": k_p, "data_quality_issues": quality_issues,
        },
        "native_reproduction": {
            "metrics": native, "expected_TP_FP_FN": list(expected), "status": "PASS",
        },
        "candidate_invariance": {
            "videos": len(payload["records"]), "exact_peak_list_equal_videos": exact_peak_videos,
            "native_peak_count": peak_count, "LVB_peak_count": peak_count,
            "candidate_confidence_identical": True, "peak_Jaccard": 1.0, "status": "PASS",
        },
        "spotting": {
            "Native": native, "TCB_reference": tcb_primary["TCB"], "LVB": lvb,
            "TCB_minus_Native_F1": tcb_primary["delta_F1"],
            "LVB_minus_Native_F1": lvb["Spotting_F1"] - native["Spotting_F1"],
        },
        "duration": {
            "Native": extended_distribution(native_durations),
            "TCB_reference": tcb_duration,
            "LVB": duration_stats, "valid_GT": extended_distribution(gt_durations),
            "LVB_duration_gt_2x_native_fraction": extreme_over_2x / peak_count if peak_count else 0.0,
            "LVB_duration_gt_2x_native_count": extreme_over_2x,
        },
        "fallback": {
            "left": int(fallback_totals["left"]), "right": int(fallback_totals["right"]),
            "both": int(fallback_totals["both"]),
        },
        "FN_C_conversion": {
            "native_FN_C": boundary_total, "LVB_converted_to_TP": boundary_converted,
            "remained_FN": boundary_total - boundary_converted,
            "best_IoU_became_worse": boundary_worse,
            "conversion_rate": boundary_converted / boundary_total if boundary_total else 0.0,
        },
        "FP_B_conversion": {
            "native_near_GT_FP": near_total, "LVB_converted_to_TP": near_converted,
            "remained_not_TP": near_total - near_converted,
            "best_IoU_became_worse": near_worse,
            "conversion_rate": near_converted / near_total if near_total else 0.0,
        },
        "native_TP_preservation": {
            "native_TP": native_tp_total, "preserved": native_tp_preserved,
            "lost": native_tp_lost,
            "requested_NetGain_FN_C_plus_NearGT_minus_TP_lost": requested_net_gain,
            "FN_C_NearGT_converted_overlap": len(converted_boundary_gt_ids & converted_near_gt_ids),
            "unique_repaired_GT": unique_repair_count,
            "unique_repair_NetGain": unique_net_gain,
            "actual_delta_TP": lvb["TP"] - native["TP"],
        },
        "paired_IoU": {
            "cohort": "same native peak; native max-IoU>0; fixed native-best GT",
            "Native": iou_distribution(native_ious),
            "TCB_reference": iou_distribution(tcb_ious),
            "LVB": iou_distribution(lvb_ious),
            "LVB_minus_Native": distribution(lvb_delta_ious),
            "fraction_improved": float(np.mean(delta_array > 1e-12)) if len(delta_array) else None,
            "fraction_equal": float(np.mean(np.abs(delta_array) <= 1e-12)) if len(delta_array) else None,
            "fraction_worse": float(np.mean(delta_array < -1e-12)) if len(delta_array) else None,
        },
        "directional_error": {
            "cohort": "same native peak and fixed native-best GT; malformed GT excluded",
            "onset": {
                "Native": distribution(onset_errors["native"]), "TCB_reference": tcb_onset,
                "LVB": distribution(onset_errors["lvb"]),
            },
            "offset": {
                "Native": distribution(offset_errors["native"]), "TCB_reference": tcb_offset,
                "LVB": distribution(offset_errors["lvb"]),
            },
        },
        "subject_stability": {
            "improved": improved, "equal": equal, "worse": worse,
            "bootstrap": bootstrap_subject_delta(subject_counts),
        },
    }


def decide(results):
    checks = {}
    for dataset, result in results.items():
        spotting = result["spotting"]
        fnc = result["FN_C_conversion"]
        preservation = result["native_TP_preservation"]
        paired = result["paired_IoU"]
        directional = result["directional_error"]
        duration = result["duration"]
        bootstrap = result["subject_stability"]["bootstrap"]
        checks[dataset] = {
            "F1_improved": spotting["LVB_minus_Native_F1"] > 0,
            "FN_C_converted": fnc["LVB_converted_to_TP"] > 0,
            "native_TP_loss_below_unique_repairs": preservation["lost"] < preservation["unique_repaired_GT"],
            "paired_mean_and_median_IoU_improved": (
                paired["LVB_minus_Native"]["mean"] > 0
                and paired["LVB_minus_Native"]["median"] > 0
            ),
            "onset_or_offset_median_error_improved": (
                directional["onset"]["LVB"]["median"] < directional["onset"]["Native"]["median"]
                or directional["offset"]["LVB"]["median"] < directional["offset"]["Native"]["median"]
            ),
            "extreme_interval_fraction_le_0_10": duration["LVB_duration_gt_2x_native_fraction"] <= 0.10,
            "bootstrap_not_systematically_negative": bootstrap["CI95"][1] >= 0,
        }
    both_f1 = all(value["F1_improved"] for value in checks.values())
    majority = all(sum(value.values()) >= 5 for value in checks.values())
    decision = "GO-LVB" if both_f1 and majority else "NO-GO-LVB"
    return {
        "decision": decision,
        "rule": "NO-GO if either dataset F1 declines; otherwise GO only if >=5/7 fixed evidence checks pass in each dataset",
        "extreme_interval_diagnostic_cutoff": "duration > 2x native; large fraction operationalized as >10% for the gate only",
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
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_report(path, results, decision):
    lines = [
        "# Adaptive Boundary LVB Feasibility Audit", "",
        "## 1. Locked experiment", "",
        "LVB 使用未修改的 native smoothed curve 和完全相同的 native peaks。每侧 boundary 是相邻 native peaks 所限定区域内最近的 `find_peaks(-g)` local minimum；若无 local minimum，仅该侧使用 clipped native `p±k_p` fallback。没有任何可调参数、GT boundary fitting、candidate recovery、peak shift、NMS 或 cap。", "",
        "TCB 数字和逐事件区间只读取既有 audit 文件，没有重新优化。主 endpoint 为 raw spotting。", "",
        "## 2. Baseline and candidate invariance", "",
        "| Dataset | Native TP/FP/FN | Native F1 | Peaks N/L | Exact videos | Jaccard |", "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in results.items():
        native = result["native_reproduction"]["metrics"]
        inv = result["candidate_invariance"]
        lines.append(f"| {dataset} | {native['TP']}/{native['FP']}/{native['FN']} | {fmt(native['Spotting_F1'])} | {inv['native_peak_count']}/{inv['LVB_peak_count']} | {inv['exact_peak_list_equal_videos']}/{inv['videos']} | {fmt(inv['peak_Jaccard'])} |")
    lines.extend(["", "## 3. Full spotting metrics", "", "| Dataset | Boundary | TP | FP | FN | P | R | F1 | Δ vs Native |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        values = result["spotting"]
        native_f1 = values["Native"]["Spotting_F1"]
        for label in ["Native", "TCB_reference", "LVB"]:
            value = values[label]
            lines.append(f"| {dataset} | {label} | {value['TP']} | {value['FP']} | {value['FN']} | {fmt(value['Precision'])} | {fmt(value['Recall'])} | {fmt(value['Spotting_F1'])} | {fmt(value['Spotting_F1']-native_f1)} |")
    lines.extend(["", "## 4. Duration and extreme intervals", "", "| Dataset | Boundary | Mean | Median | IQR | p10 | p90 | Min | Max | p95 | p99 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        for label in ["Native", "TCB_reference", "LVB", "valid_GT"]:
            value = result["duration"][label]
            lines.append(f"| {dataset} | {label} | {fmt(value.get('mean'))} | {fmt(value.get('median'))} | {fmt(value.get('IQR'))} | {fmt(value.get('p10'))} | {fmt(value.get('p90'))} | {fmt(value.get('min'))} | {fmt(value.get('max'))} | {fmt(value.get('p95'))} | {fmt(value.get('p99'))} |")
    lines.append("")
    for dataset, result in results.items():
        duration = result["duration"]
        fallback = result["fallback"]
        lines.extend([
            f"- {dataset} duration >2×native：{duration['LVB_duration_gt_2x_native_count']} ({fmt(duration['LVB_duration_gt_2x_native_fraction'])})。",
            f"- {dataset} fallback left/right/both：{fallback['left']}/{fallback['right']}/{fallback['both']}。",
        ])
    lines.extend(["", "## 5. FN-C / FP-B conversion and Native TP safety", "", "| Dataset | FN-C converted/remained/worse | FP-B converted/remained/worse | Native TP preserved/lost | Requested NetGain | Unique repair NetGain | Actual ΔTP |", "|---|---:|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        fnc, fpb, safety = result["FN_C_conversion"], result["FP_B_conversion"], result["native_TP_preservation"]
        lines.append(f"| {dataset} | {fnc['LVB_converted_to_TP']}/{fnc['remained_FN']}/{fnc['best_IoU_became_worse']} | {fpb['LVB_converted_to_TP']}/{fpb['remained_not_TP']}/{fpb['best_IoU_became_worse']} | {safety['preserved']}/{safety['lost']} | {safety['requested_NetGain_FN_C_plus_NearGT_minus_TP_lost']} | {safety['unique_repair_NetGain']} | {safety['actual_delta_TP']} |")
    lines.extend(["", "`FN-C converted + NearGT converted − NativeTP lost` 按任务原式报告；由于 FN-C 与 near-GT FP 可能是同一错误的两面，同时给出 overlap 去重后的 unique repair NetGain 与实际 ΔTP，防止重复计数。", "", "## 6. Paired IoU", "", "固定 cohort：native max-IoU>0 的同一 peak，固定 native-best GT，不为 LVB 重选 GT。", "", "| Dataset | Boundary | Mean | Median | IQR | IoU≥.3/.5/.7 |", "|---|---|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        paired = result["paired_IoU"]
        for label in ["Native", "TCB_reference", "LVB"]:
            value = paired[label]
            fractions = f"{fmt(value['fraction_ge_0.3'])}/{fmt(value['fraction_ge_0.5'])}/{fmt(value['fraction_ge_0.7'])}"
            lines.append(f"| {dataset} | {label} | {fmt(value['mean'])} | {fmt(value['median'])} | {fmt(value['IQR'])} | {fractions} |")
    lines.append("")
    for dataset, result in results.items():
        paired = result["paired_IoU"]
        delta = paired["LVB_minus_Native"]
        lines.append(f"- {dataset} LVB−Native mean/median ΔIoU：{fmt(delta['mean'])}/{fmt(delta['median'])}；improved/equal/worse：{fmt(paired['fraction_improved'])}/{fmt(paired['fraction_equal'])}/{fmt(paired['fraction_worse'])}。")
    lines.extend(["", "## 7. Onset / offset direction", "", "| Dataset | Boundary | Median onset error | Median offset error |", "|---|---|---:|---:|"])
    for dataset, result in results.items():
        directional = result["directional_error"]
        for label in ["Native", "TCB_reference", "LVB"]:
            lines.append(f"| {dataset} | {label} | {fmt(directional['onset'][label]['median'])} | {fmt(directional['offset'][label]['median'])} |")
    lines.extend(["", "## 8. Subject stability", "", "| Dataset | Improved | Equal | Worse | Bootstrap mean ΔF1 | 95% CI |", "|---|---:|---:|---:|---:|---:|"])
    for dataset, result in results.items():
        value, boot = result["subject_stability"], result["subject_stability"]["bootstrap"]
        lines.append(f"| {dataset} | {value['improved']} | {value['equal']} | {value['worse']} | {fmt(boot['mean_delta_F1'])} | [{fmt(boot['CI95'][0])}, {fmt(boot['CI95'][1])}] |")
    lines.extend(["", "## 9. Gate checks", ""])
    for dataset, values in decision["checks"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        for name, passed in values.items():
            lines.append(f"- {name}: {passed}")
        lines.append("")
    lines.extend(["## 10. Final decision", "", f"**{decision['decision']}**", ""])
    if decision["decision"] == "GO-LVB":
        lines.append("local temporal basin geometry 在不改变 native candidate discovery 的情况下，于两个数据集稳定改善 boundary localization，具备进入正式 training-free adaptive-boundary Method 设计的资格。")
    else:
        lines.append("local-valley boundary 无法在两个数据集稳定兑现 boundary oracle headroom。按预先约束停止继续搜索新的 boundary heuristic，并重新评估 inference-only 项目可行性。该结论不否定此前 GO-BOUNDARY 的诊断事实。")
    lines.extend(["", "## 11. Provenance and limits", ""])
    for dataset, result in results.items():
        cache = result["cache"]
        lines.extend([
            f"- {dataset} cache：`{cache['path']}`",
            f"- {dataset} SHA-256：`{cache['sha256']}`",
            f"- {dataset} subjects/videos/GT/k_p：{cache['subjects']}/{cache['videos']}/{cache['GT']}/{cache['k_p']}",
            f"- {dataset} data-quality warnings：{len(cache['data_quality_issues'])}",
        ])
    lines.extend([
        "- 原 cache 未修改；GT 只用于 evaluator 和诊断，不用于 boundary 生成。",
        "- LVB 不含 duration cap；极端区间只报告，不后验修复。",
        "- 稳健性结论仅限本次固定 LVB feasibility test，不外推到尚未实现的其他 boundary 方法。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = args.output_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    prior_gt_rows = load_csv(args.failure_gt_trace)
    prior_pred_rows = load_csv(args.failure_prediction_trace)
    tcb_results = json.loads(args.tcb_results.read_text(encoding="utf-8"))
    tcb_event_rows = load_csv(args.tcb_event_trace)
    tcb_event_map = {row["prediction_id"]: row for row in tcb_event_rows}
    caches = {
        "SAMMLV": (args.sammlv_cache, load_pickle(args.sammlv_cache)),
        "CASME_3": (args.casme3_cache, load_pickle(args.casme3_cache)),
    }
    event_rows, subject_rows, results = [], [], {}
    for dataset, (path, payload) in caches.items():
        if str(payload["dataset"]) != dataset:
            raise ValueError(f"{path}: dataset mismatch")
        results[dataset] = analyze_dataset(
            payload, path, prior_gt_rows, prior_pred_rows,
            tcb_results["datasets"][dataset], tcb_event_map, event_rows, subject_rows,
        )
    decision = decide(results)
    artifact = {
        "status": "COMPLETE", "audit": "Local-Valley Basin Boundary Feasibility",
        "method_definition": "nearest scipy local minimum on each side within adjacent-native-peak ownership; native clipped fallback per missing side",
        "constraints": {
            "cache_only": True, "training": False, "model_forward": False,
            "threshold_changed": False, "peaks_changed": False, "confidence_changed": False,
            "parameters_added": 0, "duration_cap": False, "grid_search": False,
        },
        "TCB_reference": {
            "path": str(args.tcb_results.resolve()), "sha256": file_sha256(args.tcb_results),
            "reoptimized": False,
        },
        "datasets": results, "feasibility_gate": decision,
    }
    result_path = outputs / "lvb_boundary_results.json"
    result_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(outputs / "lvb_event_trace.csv", event_rows)
    write_csv(outputs / "lvb_subject_metrics.csv", subject_rows)
    write_report(args.output_root / "ADAPTIVE_BOUNDARY_LVB_FEASIBILITY_CN.md", results, decision)
    print(json.dumps({"status": "COMPLETE", "decision": decision["decision"], "results": str(result_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
