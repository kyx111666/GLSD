#!/usr/bin/env python3
"""Cache-only Boundary Predictability Information Audit.

The probe is diagnostic only. It keeps native candidates fixed and uses strict
subject-LOSO Ridge(alpha=1.0) with either peak height or a normalized 41-point
smoothed-score patch.
"""

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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


P = 0.55
IOU_THRESHOLD = 0.5
RIDGE_ALPHA = 1.0
PATCH_POINTS = 41
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260904
EXPECTED_NATIVE = {
    "SAMMLV": (53, 184, 106, 237),
    "CASME_3": (81, 912, 777, 993),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sammlv-cache", type=Path, required=True)
    parser.add_argument("--casme3-cache", type=Path, required=True)
    parser.add_argument("--failure-gt-trace", type=Path, required=True)
    parser.add_argument("--failure-prediction-trace", type=Path, required=True)
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
    required = {"dataset", "k_p", "num_subjects", "num_videos", "num_gt", "records"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")
    return payload


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def smooth(score, width):
    kernel = np.ones(int(width), dtype=np.float64) / int(width)
    return np.convolve(np.asarray(score, dtype=np.float64), kernel, mode="same")


def native_decode(score, k_p):
    curve = smooth(score, 2 * k_p)
    threshold = float(curve.mean() + P * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
    return curve, threshold, peaks


def interval_iou(interval, sample):
    left, right = map(int, interval)
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union > 0 else 0.0


def greedy_match(peaks, samples, k_p):
    unmatched = set(range(len(samples)))
    matches = []
    for peak in peaks:
        choices = [
            (interval_iou((int(peak) - k_p, int(peak) + k_p), samples[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            matches.append(int(best_index))
        else:
            matches.append(-1)
    return matches, unmatched


def normalized_patch(curve, peak, k_p):
    # Sampling coordinates are locked to [p-2k_p, p+2k_p]. Values outside the
    # video are edge-replicated by clipping coordinates before interpolation.
    coordinates = np.linspace(peak - 2 * k_p, peak + 2 * k_p, PATCH_POINTS)
    coordinates = np.clip(coordinates, 0, len(curve) - 1)
    values = np.interp(coordinates, np.arange(len(curve)), curve)
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo + 1e-12)


def valid_gt(sample, length):
    onset, _, offset = map(int, sample[:3])
    return offset >= onset and onset >= 0 and offset < length


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "IQR": None}
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {
        "n": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "q25": float(q25),
        "q75": float(q75), "IQR": float(q75 - q25),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_associations(payload, cache_path, gt_trace, pred_trace):
    dataset = str(payload["dataset"])
    k_p = int(payload["k_p"])
    records = {(str(r["subject"]), str(r["video"])): r for r in payload["records"]}
    trace_predictions = {
        r["prediction_id"]: r for r in pred_trace if r["dataset"] == dataset
    }
    peak_to_prediction = {
        (r["subject"], r["video"], int(r["peak_time"])): r
        for r in pred_trace if r["dataset"] == dataset
    }
    association_sources = defaultdict(set)
    excluded = Counter()
    native_totals = Counter()
    peak_count = 0
    feature_lookup = {}

    # Rebuild native candidates from the untouched cache and confirm trace identity.
    for record in payload["records"]:
        subject, video = str(record["subject"]), str(record["video"])
        score = np.asarray(record["score"], dtype=np.float64)
        samples = list(record["samples"])
        curve, threshold, peaks = native_decode(score, k_p)
        matches, unmatched = greedy_match(peaks, samples, k_p)
        peak_count += len(peaks)
        native_totals.update(tp=sum(x >= 0 for x in matches), fp=sum(x < 0 for x in matches), fn=len(unmatched))
        for pred_index, (peak, match) in enumerate(zip(peaks, matches)):
            prediction_id = f"{dataset}:{subject}:{video}:{pred_index}"
            trace = trace_predictions.get(prediction_id)
            if trace is None or int(trace["peak_time"]) != int(peak):
                raise AssertionError(f"native candidate/trace mismatch: {prediction_id}")
            if int(trace["matched_gt_id"]) != int(match):
                raise AssertionError(f"native association/trace mismatch: {prediction_id}")
            feature_lookup[prediction_id] = {
                "dataset": dataset, "subject": subject, "video": video,
                "prediction_id": prediction_id, "peak": int(peak), "k_p": k_p,
                "video_length": len(score), "peak_height": float(curve[peak]),
                "threshold": threshold, "patch": normalized_patch(curve, int(peak), k_p),
            }
            if match >= 0:
                association_sources[(prediction_id, int(match))].add("A_NATIVE_TP")

    expected = EXPECTED_NATIVE[dataset]
    observed = (native_totals["tp"], native_totals["fp"], native_totals["fn"], peak_count)
    if observed != expected:
        raise RuntimeError(f"BLOCKED-BASELINE {dataset}: observed={observed}, expected={expected}")

    # FN-C uses its own deterministic diagnostic_peak -> GT association.
    for row in gt_trace:
        if row["dataset"] != dataset or row["fn_category"] != "FN-C_BOUNDARY":
            continue
        key = (row["subject"], row["video"], int(row["diagnostic_peak"]))
        pred = peak_to_prediction.get(key)
        if pred is None:
            raise AssertionError(f"FN-C diagnostic peak absent from native predictions: {key}")
        association_sources[(pred["prediction_id"], int(row["gt_index"]))].add("B_FNC")

    # FP-B uses the deterministic closest_gt_id recorded by Failure-Mode Audit.
    for row in pred_trace:
        if row["dataset"] == dataset and row["fp_category"] == "FP-B_NEAR_GT_LOCALIZATION":
            association_sources[(row["prediction_id"], int(row["closest_gt_id"]))].add("B_NEAR")

    rows = []
    for (prediction_id, gt_index), sources in sorted(association_sources.items()):
        feature = feature_lookup[prediction_id]
        record = records[(feature["subject"], feature["video"])]
        if not (0 <= gt_index < len(record["samples"])):
            excluded["missing_gt_index"] += 1
            continue
        sample = record["samples"][gt_index]
        if not valid_gt(sample, feature["video_length"]):
            excluded["invalid_or_out_of_range_gt"] += 1
            continue
        onset, apex, offset = map(int, sample[:3])
        source_set = set(sources)
        cohort = "A_NATIVE_TP" if "A_NATIVE_TP" in source_set else "B_LOCALIZATION"
        rows.append({
            **feature, "gt_index": gt_index, "gt_onset": onset, "gt_apex": apex,
            "gt_offset": offset, "gt_duration": offset - onset + 1,
            "d_left": feature["peak"] - onset, "d_right": offset - feature["peak"],
            "cohort": cohort, "is_native_tp": "A_NATIVE_TP" in source_set,
            "is_fnc": "B_FNC" in source_set, "is_near_fp": "B_NEAR" in source_set,
            "association_sources": "+".join(sorted(source_set)),
        })

    candidate_targets = defaultdict(set)
    for row in rows:
        candidate_targets[row["prediction_id"]].add(row["gt_index"])
    conflicts = sum(len(targets) > 1 for targets in candidate_targets.values())
    inventory = {
        "path": str(cache_path.resolve()), "sha256": sha256(cache_path),
        "subjects": int(payload["num_subjects"]), "videos": int(payload["num_videos"]),
        "GT": int(payload["num_gt"]), "k_p": k_p,
        "native_TP_FP_FN": list(observed[:3]), "native_peak_count": peak_count,
        "baseline_status": "PASS", "association_rows": len(rows),
        "unique_candidates": len(candidate_targets),
        "multi_target_candidate_count": conflicts, "excluded": dict(excluded),
    }
    return rows, inventory


def loso_predictions(rows, feature_name, normalized_target=False):
    subjects = sorted({row["subject"] for row in rows})
    predictions = np.full((len(rows), 2), np.nan, dtype=np.float64)
    if feature_name == "peak_only":
        matrix = np.asarray([[row["peak_height"]] for row in rows], dtype=np.float64)
    elif feature_name == "patch":
        matrix = np.asarray([row["patch"] for row in rows], dtype=np.float64)
    else:
        raise ValueError(feature_name)
    targets = np.asarray([[row["d_left"], row["d_right"]] for row in rows], dtype=np.float64)
    if normalized_target:
        targets = targets / np.asarray([[row["k_p"], row["k_p"]] for row in rows])
    for held_out in subjects:
        test = np.asarray([row["subject"] == held_out for row in rows])
        train = ~test
        scaler = StandardScaler().fit(matrix[train])
        x_train, x_test = scaler.transform(matrix[train]), scaler.transform(matrix[test])
        for side in range(2):
            model = Ridge(alpha=RIDGE_ALPHA).fit(x_train, targets[train, side])
            predictions[test, side] = model.predict(x_test)
    if not np.isfinite(predictions).all():
        raise AssertionError("LOSO did not create finite OOF predictions")
    if normalized_target:
        predictions *= np.asarray([[row["k_p"], row["k_p"]] for row in rows])
    return predictions


def legal_interval(row, prediction):
    # Apply the locked equations directly. Endpoints are rounded and clipped to
    # the video. If onset > offset, no post-hoc repair is allowed; IoU is zero.
    onset = int(np.clip(np.rint(row["peak"] - float(prediction[0])), 0, row["video_length"] - 1))
    offset = int(np.clip(np.rint(row["peak"] + float(prediction[1])), 0, row["video_length"] - 1))
    return onset, offset


def attach_predictions(rows):
    peak_predictions = loso_predictions(rows, "peak_only")
    patch_predictions = loso_predictions(rows, "patch")
    output = []
    for index, row in enumerate(rows):
        sample = (row["gt_onset"], row["gt_apex"], row["gt_offset"])
        native_interval = (row["peak"] - row["k_p"], row["peak"] + row["k_p"])
        peak_interval = legal_interval(row, peak_predictions[index])
        patch_interval = legal_interval(row, patch_predictions[index])
        clean = {key: value for key, value in row.items() if key != "patch"}
        clean["loso_held_out_subject"] = row["subject"]
        for label, prediction, interval in [
            ("peak_only", peak_predictions[index], peak_interval),
            ("patch", patch_predictions[index], patch_interval),
        ]:
            clean[f"{label}_pred_d_left"] = float(prediction[0])
            clean[f"{label}_pred_d_right"] = float(prediction[1])
            clean[f"{label}_onset"] = interval[0]
            clean[f"{label}_offset"] = interval[1]
            clean[f"{label}_interval_valid"] = interval[0] <= interval[1]
            clean[f"{label}_iou"] = interval_iou(interval, sample)
            clean[f"{label}_abs_error_left"] = abs(float(prediction[0]) - row["d_left"])
            clean[f"{label}_abs_error_right"] = abs(float(prediction[1]) - row["d_right"])
        clean["native_onset"], clean["native_offset"] = native_interval
        clean["native_iou"] = interval_iou(native_interval, sample)
        clean["native_abs_error_left"] = abs(row["k_p"] - row["d_left"])
        clean["native_abs_error_right"] = abs(row["k_p"] - row["d_right"])
        output.append(clean)
    return output


def predictor_metrics(rows, label):
    left = [row[f"{label}_abs_error_left"] for row in rows]
    right = [row[f"{label}_abs_error_right"] for row in rows]
    ious = np.asarray([row[f"{label}_iou"] for row in rows], dtype=np.float64)
    invalid = sum(row[f"{label}_onset"] > row[f"{label}_offset"] for row in rows)
    return {
        "n": len(rows), "left_absolute_error": distribution(left),
        "right_absolute_error": distribution(right), "IoU": distribution(ious),
        "fraction_IoU_ge_0.5": float(np.mean(ious >= 0.5)) if len(ious) else None,
        "fraction_IoU_ge_0.7": float(np.mean(ious >= 0.7)) if len(ious) else None,
        "invalid_interval_count": int(invalid),
    }


def paired_metrics(rows, label):
    native = np.asarray([row["native_iou"] for row in rows], dtype=np.float64)
    other = np.asarray([row[f"{label}_iou"] for row in rows], dtype=np.float64)
    delta = other - native
    return {
        "delta_IoU": distribution(delta),
        "fraction_improved": float(np.mean(delta > 1e-12)),
        "fraction_equal": float(np.mean(np.abs(delta) <= 1e-12)),
        "fraction_worse": float(np.mean(delta < -1e-12)),
    }


def subject_bootstrap(rows, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject"]].append(float(row["patch_iou"] - row["native_iou"]))
    subjects = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        draw = [value for subject in sampled for value in groups[subject]]
        values.append(float(np.mean(draw)))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "unit": "subject", "repeats": repeats, "seed": seed,
        "mean_delta_IoU": float(np.mean(values)), "CI95": [float(lo), float(hi)],
    }


def cohort_summary(rows):
    native = predictor_metrics(rows, "native")
    peak = predictor_metrics(rows, "peak_only")
    patch = predictor_metrics(rows, "patch")
    return {
        "targets": {
            "d_left": distribution([row["d_left"] for row in rows]),
            "d_right": distribution([row["d_right"] for row in rows]),
            "GT_duration": distribution([row["gt_duration"] for row in rows]),
        },
        "predictors": {"Native_fixed": native, "Peak_only_Ridge": peak, "Patch_Ridge": patch},
        "paired": {
            "Peak_only_minus_Native": paired_metrics(rows, "peak_only"),
            "Patch_minus_Native": paired_metrics(rows, "patch"),
        },
    }


def analyze_dataset(rows, inventory):
    cohorts = {
        "A_NATIVE_TP": [row for row in rows if row["cohort"] == "A_NATIVE_TP"],
        "B_LOCALIZATION": [row for row in rows if row["cohort"] == "B_LOCALIZATION"],
        "B_FNC": [row for row in rows if row["is_fnc"]],
        "B_NEAR_FP": [row for row in rows if row["is_near_fp"]],
        "COMBINED": rows,
    }
    summaries = {name: cohort_summary(values) for name, values in cohorts.items() if values}
    native_tp = cohorts["A_NATIVE_TP"]
    fnc = cohorts["B_FNC"]
    preserved = sum(row["patch_iou"] >= 0.5 for row in native_tp)
    rescued = sum(row["patch_iou"] >= 0.5 for row in fnc)
    safety = {
        "native_TP_total": len(native_tp), "native_TP_preserved": preserved,
        "native_TP_lost": len(native_tp) - preserved,
        "native_FN_C_total": len(fnc), "native_FN_C_rescued": rescued,
        "native_FN_C_remained": len(fnc) - rescued,
        "NetGain_FN_C_rescued_minus_NativeTP_lost": rescued - (len(native_tp) - preserved),
    }
    bootstrap = {
        name: subject_bootstrap(values)
        for name, values in cohorts.items() if values
    }
    combined = summaries["COMBINED"]
    native_metrics = combined["predictors"]["Native_fixed"]
    peak_metrics = combined["predictors"]["Peak_only_Ridge"]
    patch_metrics = combined["predictors"]["Patch_Ridge"]
    patch_delta = combined["paired"]["Patch_minus_Native"]["delta_IoU"]
    checks = {
        "patch_left_and_right_mean_MAE_below_native": (
            patch_metrics["left_absolute_error"]["mean"] < native_metrics["left_absolute_error"]["mean"]
            and patch_metrics["right_absolute_error"]["mean"] < native_metrics["right_absolute_error"]["mean"]
        ),
        "patch_left_and_right_mean_MAE_below_peak_only": (
            patch_metrics["left_absolute_error"]["mean"] < peak_metrics["left_absolute_error"]["mean"]
            and patch_metrics["right_absolute_error"]["mean"] < peak_metrics["right_absolute_error"]["mean"]
        ),
        "paired_mean_and_median_IoU_improved": patch_delta["mean"] > 0 and patch_delta["median"] > 0,
        "IoU_ge_0.5_fraction_improved": patch_metrics["fraction_IoU_ge_0.5"] > native_metrics["fraction_IoU_ge_0.5"],
        "FN_C_rescue_exceeds_native_TP_loss": safety["NetGain_FN_C_rescued_minus_NativeTP_lost"] > 0,
        "bootstrap_CI_strictly_positive": bootstrap["COMBINED"]["CI95"][0] > 0,
    }
    internal_go = sum(checks.values()) >= 5
    return {
        "inventory": inventory, "cohorts": summaries, "safety": safety,
        "bootstrap": bootstrap, "internal_checks": checks,
        "internal_positive": internal_go,
        "internal_rule": ">=5/6 locked checks",
    }


def cross_dataset_sam_to_cas(sam_rows, cas_rows):
    # Locked cross-dataset target representation: relative offsets d/k_p.
    x_train = np.asarray([row["patch"] for row in sam_rows], dtype=np.float64)
    x_test = np.asarray([row["patch"] for row in cas_rows], dtype=np.float64)
    y_train = np.asarray([[row["d_left"] / row["k_p"], row["d_right"] / row["k_p"]] for row in sam_rows])
    scaler = StandardScaler().fit(x_train)
    train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
    predictions = np.zeros((len(cas_rows), 2), dtype=np.float64)
    for side in range(2):
        model = Ridge(alpha=RIDGE_ALPHA).fit(train_scaled, y_train[:, side])
        predictions[:, side] = model.predict(test_scaled) * np.asarray([row["k_p"] for row in cas_rows])
    evaluated = []
    for row, prediction in zip(cas_rows, predictions):
        sample = (row["gt_onset"], row["gt_apex"], row["gt_offset"])
        interval = legal_interval(row, prediction)
        evaluated.append({
            **{key: value for key, value in row.items() if key != "patch"},
            "cross_iou": interval_iou(interval, sample),
            "cross_abs_error_left": abs(float(prediction[0]) - row["d_left"]),
            "cross_abs_error_right": abs(float(prediction[1]) - row["d_right"]),
        })
    delta = np.asarray([row["cross_iou"] - row["native_iou"] for row in evaluated])
    lost = sum(row["is_native_tp"] and row["cross_iou"] < 0.5 for row in evaluated)
    rescued = sum(row["is_fnc"] and row["cross_iou"] >= 0.5 for row in evaluated)
    return {
        "status": "RUN", "training_dataset": "SAMMLV", "test_dataset": "CASME_3",
        "target": "r_left=d_left/k_p, r_right=d_right/k_p",
        "delta_IoU": distribution(delta),
        "fraction_IoU_ge_0.5": float(np.mean([row["cross_iou"] >= 0.5 for row in evaluated])),
        "native_TP_lost": lost, "FN_C_rescued": rescued, "NetGain": rescued - lost,
    }


def fmt(value):
    return "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def write_report(path, results):
    lines = [
        "# Boundary Predictability Information Audit", "",
        "## 1. Locked diagnostic", "",
        "本实验不是正式 Method。它只使用现有 frozen score；native smoothing、threshold 和 peaks 完全不变。输入仅为 smoothed peak height 或固定 `[p-2k_p,p+2k_p]` 的 41 点 patch，Ridge `alpha=1.0`，严格 subject-LOSO。patch 逐样本 min-max normalization，随后 scaler 只在训练 subjects 拟合。", "",
        "OOF 连续 offset 用于 MAE；转换区间严格按 `onset=p-d_left`、`offset=p+d_right`，使用 `np.rint` 后 clip 到视频范围。若产生 `onset>offset`，不做后验修复且 IoU 记 0。没有 duration cap、GT correction 或超参数搜索。", "",
        "## 2. Cache and association audit", "",
        "| Dataset | SHA-256 | Subjects/Videos/GT | k_p | Native TP/FP/FN | Peaks | Assoc rows/unique candidates/multi-target |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in results["datasets"].items():
        inv = result["inventory"]
        lines.append(f"| {dataset} | `{inv['sha256']}` | {inv['subjects']}/{inv['videos']}/{inv['GT']} | {inv['k_p']} | {'/'.join(map(str, inv['native_TP_FP_FN']))} | {inv['native_peak_count']} | {inv['association_rows']}/{inv['unique_candidates']}/{inv['multi_target_candidate_count']} |")
    lines.append("")
    for dataset, result in results["datasets"].items():
        lines.append(f"- {dataset} excluded associations：`{result['inventory']['excluded']}`。")
    lines.extend(["", "Association 沿用 Failure-Mode Audit：A 为 native greedy raw TP；B 为 FN-C `diagnostic_peak→该GT` 与 FP-B `closest_gt_id`。同一 peak 若确定性地关联多个 GT，保留为不同 association pair，并在同一 subject fold 内，绝不跨 LOSO 泄漏。", "", "## 3. Target distributions", "", "| Dataset | Cohort | N | d_left mean/median | d_right mean/median | GT duration mean/median |", "|---|---|---:|---:|---:|---:|"])
    for dataset, result in results["datasets"].items():
        for cohort, summary in result["cohorts"].items():
            targets = summary["targets"]
            lines.append(f"| {dataset} | {cohort} | {targets['d_left']['n']} | {fmt(targets['d_left']['mean'])}/{fmt(targets['d_left']['median'])} | {fmt(targets['d_right']['mean'])}/{fmt(targets['d_right']['median'])} | {fmt(targets['GT_duration']['mean'])}/{fmt(targets['GT_duration']['median'])} |")
    lines.extend(["", "## 4. OOF offset error — combined cohort", "", "| Dataset | Predictor | Left mean/median AE | Right mean/median AE |", "|---|---|---:|---:|"])
    for dataset, result in results["datasets"].items():
        predictors = result["cohorts"]["COMBINED"]["predictors"]
        for label in ["Native_fixed", "Peak_only_Ridge", "Patch_Ridge"]:
            value = predictors[label]
            lines.append(f"| {dataset} | {label} | {fmt(value['left_absolute_error']['mean'])}/{fmt(value['left_absolute_error']['median'])} | {fmt(value['right_absolute_error']['mean'])}/{fmt(value['right_absolute_error']['median'])} |")
    lines.extend(["", "## 5. Paired IoU — combined cohort", "", "| Dataset | Predictor | Mean/median IoU | Mean/median ΔIoU | Improved/equal/worse | IoU≥.5/.7 | Invalid intervals |", "|---|---|---:|---:|---:|---:|---:|"])
    for dataset, result in results["datasets"].items():
        combined = result["cohorts"]["COMBINED"]
        native = combined["predictors"]["Native_fixed"]
        lines.append(f"| {dataset} | Native_fixed | {fmt(native['IoU']['mean'])}/{fmt(native['IoU']['median'])} | 0/0 | 0/1/0 | {fmt(native['fraction_IoU_ge_0.5'])}/{fmt(native['fraction_IoU_ge_0.7'])} | {native['invalid_interval_count']} |")
        for predictor, pair_key in [("Peak_only_Ridge", "Peak_only_minus_Native"), ("Patch_Ridge", "Patch_minus_Native")]:
            value, paired = combined["predictors"][predictor], combined["paired"][pair_key]
            delta = paired["delta_IoU"]
            lines.append(f"| {dataset} | {predictor} | {fmt(value['IoU']['mean'])}/{fmt(value['IoU']['median'])} | {fmt(delta['mean'])}/{fmt(delta['median'])} | {fmt(paired['fraction_improved'])}/{fmt(paired['fraction_equal'])}/{fmt(paired['fraction_worse'])} | {fmt(value['fraction_IoU_ge_0.5'])}/{fmt(value['fraction_IoU_ge_0.7'])} | {value['invalid_interval_count']} |")
    lines.extend(["", "### 5.1 Cohort-specific Patch evidence", "", "| Dataset | Cohort | Native/Patch mean IoU | Mean/median ΔIoU | Native/Patch IoU≥.5 | Bootstrap 95% CI |", "|---|---|---:|---:|---:|---:|"])
    for dataset, result in results["datasets"].items():
        for cohort in ["A_NATIVE_TP", "B_LOCALIZATION", "B_FNC", "B_NEAR_FP"]:
            if cohort not in result["cohorts"]:
                continue
            summary = result["cohorts"][cohort]
            native, patch = summary["predictors"]["Native_fixed"], summary["predictors"]["Patch_Ridge"]
            delta = summary["paired"]["Patch_minus_Native"]["delta_IoU"]
            boot = result["bootstrap"][cohort]
            lines.append(f"| {dataset} | {cohort} | {fmt(native['IoU']['mean'])}/{fmt(patch['IoU']['mean'])} | {fmt(delta['mean'])}/{fmt(delta['median'])} | {fmt(native['fraction_IoU_ge_0.5'])}/{fmt(patch['fraction_IoU_ge_0.5'])} | [{fmt(boot['CI95'][0])}, {fmt(boot['CI95'][1])}] |")
    lines.extend(["", "## 6. Native TP preservation and FN-C rescue", "", "| Dataset | TP preserved/lost | FN-C rescued/remained | NetGain |", "|---|---:|---:|---:|"])
    for dataset, result in results["datasets"].items():
        value = result["safety"]
        lines.append(f"| {dataset} | {value['native_TP_preserved']}/{value['native_TP_lost']} | {value['native_FN_C_rescued']}/{value['native_FN_C_remained']} | {value['NetGain_FN_C_rescued_minus_NativeTP_lost']} |")
    lines.extend(["", "## 7. Subject bootstrap", "", "| Dataset | Cohort | Mean ΔIoU | 95% CI |", "|---|---|---:|---:|"])
    for dataset, result in results["datasets"].items():
        for cohort in ["A_NATIVE_TP", "B_LOCALIZATION", "COMBINED"]:
            if cohort not in result["bootstrap"]:
                continue
            value = result["bootstrap"][cohort]
            lines.append(f"| {dataset} | {cohort} | {fmt(value['mean_delta_IoU'])} | [{fmt(value['CI95'][0])}, {fmt(value['CI95'][1])}] |")
    lines.extend(["", "## 8. Locked checks", ""])
    for dataset, result in results["datasets"].items():
        lines.extend([f"### {dataset}", ""])
        for name, passed in result["internal_checks"].items():
            lines.append(f"- {name}: {passed}")
        lines.append(f"- Internal status (`>=5/6`): {result['internal_positive']}")
        lines.append("")
    cross = results["cross_dataset"]
    lines.extend(["## 9. Locked SAMMLV→CASME3 diagnostic", "", f"Status: **{cross['status']}**.", ""])
    if cross["status"] == "RUN":
        lines.append(f"Normalized-target mean/median ΔIoU：{fmt(cross['delta_IoU']['mean'])}/{fmt(cross['delta_IoU']['median'])}；NetGain：{cross['NetGain']}。")
        lines.append("")
    else:
        lines.append("未运行：两个数据集内部 LOSO 没有同时通过预先锁定的 positive gate，因此按协议禁止启动跨数据集 probe。")
        lines.append("")
    lines.extend(["## 10. Final decision", "", f"**{results['feasibility_gate']['decision']}**", "", results["feasibility_gate"]["statement"], "", "原 cache 未修改；未运行 backbone forward、CUDA、hidden feature、recognition、candidate recovery 或任何新 boundary heuristic。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output = args.output_root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    gt_trace, pred_trace = load_csv(args.failure_gt_trace), load_csv(args.failure_prediction_trace)
    payloads = [load_cache(args.sammlv_cache), load_cache(args.casme3_cache)]
    cache_paths = [args.sammlv_cache, args.casme3_cache]
    raw_by_dataset, oof_rows, dataset_results = {}, [], {}
    for payload, cache_path in zip(payloads, cache_paths):
        raw_rows, inventory = extract_associations(payload, cache_path, gt_trace, pred_trace)
        raw_by_dataset[payload["dataset"]] = raw_rows
        evaluated = attach_predictions(raw_rows)
        oof_rows.extend(evaluated)
        dataset_results[payload["dataset"]] = analyze_dataset(evaluated, inventory)

    both_internal = all(result["internal_positive"] for result in dataset_results.values())
    if both_internal:
        cross = cross_dataset_sam_to_cas(raw_by_dataset["SAMMLV"], raw_by_dataset["CASME_3"])
        cross_pass = cross["delta_IoU"]["mean"] > 0 and cross["NetGain"] > 0
    else:
        cross = {"status": "NOT_RUN_INTERNAL_GATE_FAILED", "reason": "both datasets were not internally positive"}
        cross_pass = False
    decision = "GO-PREDICTABLE" if both_internal and cross_pass else "NO-GO-BOUNDARY-INFORMATION"
    statement = (
        "frozen score neighborhood 包含跨 subject 且可迁移的 boundary-location information，值得设计 lightweight post-hoc decoder。"
        if decision == "GO-PREDICTABLE" else
        "当前 frozen score 上的低容量监督 probe 没有提供足够稳定的 boundary predictability。按预设协议停止整个 score-level adaptive-boundary 方向，不再尝试 nonlinear probe 或 feature zoo。"
    )
    results = {
        "status": "COMPLETE", "audit": "Boundary Predictability Information Audit",
        "scope": "diagnostic probe only; not a Method",
        "locked_configuration": {
            "native_candidates_unchanged": True, "patch_support": "[p-2k_p,p+2k_p]",
            "patch_points": PATCH_POINTS, "patch_amplitude_normalization": "per-patch min-max",
            "edge_handling": "coordinate clipping / edge replication", "model": "Ridge",
            "alpha": RIDGE_ALPHA, "split": "strict subject-LOSO",
            "scaler": "StandardScaler fitted on training subjects only",
            "targets": ["d_left=p-GT_onset", "d_right=GT_offset-p"],
            "interval_conversion": "direct equations; np.rint; video-boundary clipping; invalid onset>offset receives IoU 0 without repair",
            "cross_dataset_target": "r_left=d_left/k_p, r_right=d_right/k_p",
            "bootstrap_repeats": BOOTSTRAP_REPEATS, "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "datasets": dataset_results, "cross_dataset": cross,
        "feasibility_gate": {
            "decision": decision,
            "internal_rule": "each dataset must pass >=5/6 locked checks before cross-dataset diagnostic",
            "final_rule": "both internal gates pass and locked SAMMLV->CASME3 mean delta IoU and NetGain are positive",
            "statement": statement,
        },
    }
    write_csv(output / "boundary_oof_predictions.csv", oof_rows)
    (output / "boundary_predictability_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_root / "BOUNDARY_PREDICTABILITY_AUDIT_CN.md", results)
    print(json.dumps({
        "status": "COMPLETE", "decision": decision,
        "results": str(output / "boundary_predictability_results.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
