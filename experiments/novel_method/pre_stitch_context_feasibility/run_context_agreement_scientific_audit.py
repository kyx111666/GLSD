#!/usr/bin/env python3
"""Fixed-rule source-side Context Agreement audit for ME-TST+ × SAMMLV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260903
BOOTSTRAP_REPEATS = 1000
THRESHOLD_P = 0.55


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-context-cache", type=Path, required=True)
    parser.add_argument("--gt-reference-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smooth(score: np.ndarray, width: int) -> np.ndarray:
    box = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.asarray(score, dtype=np.float64), box, mode="same")


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-8:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad)


def interval_iou(left: int, right: int, sample: list) -> float:
    gt_left, gt_right = int(sample[0]), int(sample[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def native_decode(score: np.ndarray, k_p: int) -> dict:
    curve = smooth(score, 2 * k_p)
    threshold = float(curve.mean() + THRESHOLD_P * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
    return {"curve": curve, "threshold": threshold, "peaks": peaks}


def stitch_subject(
    window_scores: np.ndarray,
    video_num: list[int],
    batch_size: int,
    window_length: int,
    stride: int,
) -> list[np.ndarray]:
    outputs = []
    video_index = 0
    frame_count = 0
    global_index = 0
    current = np.zeros((video_num[0] + 1) * stride, dtype=np.float32)
    for batch_start in range(0, len(window_scores), batch_size):
        batch_stop = min(batch_start + batch_size, len(window_scores))
        for batch_local_index in range(batch_stop - batch_start):
            if frame_count == video_num[video_index]:
                outputs.append(current)
                video_index += 1
                frame_count = 0
                current = np.zeros((video_num[video_index] + 1) * stride, dtype=np.float32)
            score = window_scores[global_index]
            if batch_local_index == 0:
                current[frame_count * stride : (frame_count + 2) * stride] = score
            else:
                current[(frame_count + 1) * stride : (frame_count + 2) * stride] = score[stride:window_length]
            frame_count += 1
            global_index += 1
    outputs.append(current)
    return outputs


def native_match(peaks: np.ndarray, samples: list, k_p: int) -> tuple[dict, set, dict]:
    unmatched = set(range(len(samples)))
    roles = {}
    for peak in peaks:
        overlaps = [
            (interval_iou(int(peak - k_p), int(peak + k_p), sample), index)
            for index, sample in enumerate(samples)
            if index in unmatched
        ]
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            roles[int(peak)] = ("native_tp", int(best_index))
        else:
            roles[int(peak)] = ("native_fp", None)
    counts = {
        "TP": sum(role == "native_tp" for role, _ in roles.values()),
        "FP": sum(role == "native_fp" for role, _ in roles.values()),
        "FN": len(unmatched),
    }
    return roles, unmatched, counts


def metric_summary(rows: list[dict], score_names: tuple[str, ...], label: str) -> dict:
    y = np.asarray([row[label] for row in rows], dtype=int)
    output = {}
    for name in score_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        if len(y) == 0 or len(np.unique(y)) < 2:
            output[name] = {
                "n": int(len(y)), "positive": int(y.sum()), "prevalence": None,
                "ROC_AUC": None, "PR_AUC": None,
            }
        else:
            output[name] = {
                "n": int(len(y)), "positive": int(y.sum()),
                "prevalence": float(y.mean()),
                "ROC_AUC": float(roc_auc_score(y, values)),
                "PR_AUC": float(average_precision_score(y, values)),
            }
    return output


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "std": None, "IQR": None}
    return {
        "n": len(array), "mean": float(array.mean()),
        "median": float(np.median(array)), "std": float(array.std()),
        "IQR": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    }


def spotting_summary(counts: dict) -> dict:
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "Precision": precision, "Recall": recall, "Spotting_F1": f1}


def height_match(rows: list[dict]) -> tuple[list[dict], dict]:
    matched = []
    differences = []
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    for subject in sorted(by_subject):
        positives = sorted(
            (row for row in by_subject[subject] if row["GT_related"]),
            key=lambda row: (row["height"], row["video_id"], row["peak"]),
        )
        negatives = [row for row in by_subject[subject] if not row["GT_related"]]
        for positive in positives:
            if not negatives:
                break
            best_index = min(
                range(len(negatives)),
                key=lambda index: (
                    abs(negatives[index]["height"] - positive["height"]),
                    negatives[index]["video_id"], negatives[index]["peak"],
                ),
            )
            negative = negatives.pop(best_index)
            matched.extend((positive, negative))
            differences.append(abs(negative["height"] - positive["height"]))
    metrics = metric_summary(matched, ("height", "agreement"), "GT_related")
    return matched, {
        "method": "within-subject greedy 1:1 Height nearest matching without replacement",
        "matched_pairs": len(differences),
        "mean_abs_delta_H": float(np.mean(differences)) if differences else None,
        "median_abs_delta_H": float(np.median(differences)) if differences else None,
        "Height_ROC_AUC": metrics["height"]["ROC_AUC"],
        "Agreement_ROC_AUC": metrics["agreement"]["ROC_AUC"],
    }


def subject_bootstrap(rows: list[dict], matched_rows: list[dict]) -> dict:
    by_subject = defaultdict(list)
    matched_by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    for row in matched_rows:
        matched_by_subject[row["subject"]].append(row)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(SEED)
    delta_ap = []
    agreement_auc_minus_half = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        current = [row for subject in sampled for row in by_subject[subject]]
        y = np.asarray([row["GT_related"] for row in current], dtype=int)
        if len(np.unique(y)) == 2:
            h = np.asarray([row["height"] for row in current])
            j = np.asarray([row["combined"] for row in current])
            delta_ap.append(float(average_precision_score(y, j) - average_precision_score(y, h)))
        current_matched = [row for subject in sampled for row in matched_by_subject[subject]]
        matched_y = np.asarray([row["GT_related"] for row in current_matched], dtype=int)
        if len(np.unique(matched_y)) == 2:
            agreement = np.asarray([row["agreement"] for row in current_matched])
            agreement_auc_minus_half.append(float(roc_auc_score(matched_y, agreement) - 0.5))
    return {
        "unit": "subject", "repeats": BOOTSTRAP_REPEATS, "seed": SEED,
        "delta_AP_combined_minus_height": {
            "valid_draws": len(delta_ap),
            "mean": float(np.mean(delta_ap)) if delta_ap else None,
            "CI95": np.quantile(delta_ap, [0.025, 0.975]).tolist() if delta_ap else [None, None],
        },
        "height_conditioned_Agreement_AUC_minus_0_5": {
            "valid_draws": len(agreement_auc_minus_half),
            "mean": float(np.mean(agreement_auc_minus_half)) if agreement_auc_minus_half else None,
            "CI95": np.quantile(agreement_auc_minus_half, [0.025, 0.975]).tolist()
            if agreement_auc_minus_half else [None, None],
        },
    }


def matched_fp(rows: list[dict], fp_budget: int) -> dict:
    result = {}
    for name in ("height", "combined"):
        ordered = sorted(
            rows,
            key=lambda row: (-row[name], row["subject"], row["video_id"], row["peak"]),
        )
        fp = 0
        selected = []
        if fp_budget > 0:
            for row in ordered:
                selected.append(row)
                if not row["GT_related"]:
                    fp += 1
                if fp >= fp_budget:
                    break
        recovered = {item for row in selected for item in row["missed_gt_ids"]}
        result[name] = {
            "GT_related": sum(row["GT_related"] for row in selected),
            "Native_missed_recovered": len(recovered),
            "FP": fp,
            "selected_candidates": len(selected),
        }
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if isinstance(output.get("missed_gt_ids"), list):
                output["missed_gt_ids"] = ";".join(output["missed_gt_ids"])
            writer.writerow({field: output.get(field) for field in fields})


def render_report(result: dict) -> str:
    if result["status"] == "BLOCKED-FULL-REPRODUCTION":
        return (
            "# Pre-Stitch Context Agreement Scientific Audit\n\n"
            "> **Engineering equivalence：PASS-DECODER-EQUIVALENT**\n\n"
            "> **Scientific audit：BLOCKED-FULL-REPRODUCTION**\n\n"
            "全量 fresh 与 historical reproduction reference 的 TP/FP/FN 不一致，"
            "因此未运行 Agreement 科学分析。详见 `context_agreement_results.json`。\n"
        )
    if result["status"] == "NO-GO-INSUFFICIENT-MULTIPLICITY":
        return (
            "# Pre-Stitch Context Agreement Scientific Audit\n\n"
            "> **Engineering equivalence：PASS-DECODER-EQUIVALENT**\n\n"
            "> **Scientific audit：NO-GO-INSUFFICIENT-MULTIPLICITY**\n\n"
            "超过一半 native candidates 的真实 context_count < 2，未运行 Agreement 主实验。\n"
        )
    full = result["full_reproduction_sanity"]
    metrics = result["candidate_discrimination"]["primary_GT_related"]
    matched = result["height_conditioned"]
    bootstrap = result["subject_bootstrap"]["delta_AP_combined_minus_height"]
    fp = result["matched_FP_recovery"]
    multiplicity = result["context_multiplicity"]
    conclusion = (
        "GO：pre-stitch overlapping-context agreement 提供 stitched peak height 之外的稳定 "
        "inference uncertainty information，可以进一步设计 reliability-aware training-free event decoder。"
        if result["status"] == "GO"
        else "NO-GO：pre-stitch context agreement 没有提供足够稳定的、超越 stitched peak height 的独立 inference information。"
    )
    lines = [
        "# Pre-Stitch Context Agreement Scientific Audit",
        "", "> **Engineering equivalence：PASS-DECODER-EQUIVALENT**",
        "", f"> **Context Agreement scientific verdict：{result['status']}**",
        "", "## 1. 结论", "", conclusion,
        "", "## 2. 数据协议", "",
        "仅使用同一次 fresh frozen inference 的 raw window scores、原 batch-sensitive stitching "
        "与 fresh stitched score。historical numeric score 只用于 full-reproduction sanity reference，"
        "没有进入 Agreement candidate features。训练、hidden、recognition、mirror、multi-scale 和 learned fusion 均未使用。",
        "", "## 3. Full-dataset reproduction sanity", "",
        "| Source | Subjects | Videos | GT | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Fresh | 29 | 79 | {full['GT']} | {full['fresh']['TP']} | {full['fresh']['FP']} | {full['fresh']['FN']} | {full['fresh']['Precision']:.6f} | {full['fresh']['Recall']:.6f} | {full['fresh']['Spotting_F1']:.6f} |",
        f"| Historical reference | 29 | 79 | {full['GT']} | {full['historical_reference']['TP']} | {full['historical_reference']['FP']} | {full['historical_reference']['FN']} | {full['historical_reference']['Precision']:.6f} | {full['historical_reference']['Recall']:.6f} | {full['historical_reference']['Spotting_F1']:.6f} |",
        "", f"Exact peak-list equivalent videos: {full['exact_peak_list_equivalent_videos']}/79。",
        "", "## 4. Context multiplicity", "",
        f"Candidate count={multiplicity['candidate_count']}；eligible n>=2={multiplicity['eligible_candidates']}；"
        f"ineligible={multiplicity['ineligible_candidates']} ({multiplicity['ineligible_fraction']:.4%})。",
        "", f"Position multiplicity distribution: `{json.dumps(multiplicity['all_positions'], ensure_ascii=False)}`。",
        "", "## 5. Candidate discrimination", "",
        "| Evidence | ROC-AUC | PR-AUC |",
        "|---|---:|---:|",
    ]
    for name, label in (("height", "Height"), ("agreement", "Agreement"), ("combined", "Height+Agreement")):
        row = metrics[name]
        lines.append(f"| {label} | {row['ROC_AUC']:.6f} | {row['PR_AUC']:.6f} |")
    delta = metrics["combined"]["PR_AUC"] - metrics["height"]["PR_AUC"]
    lines.extend([
        "", f"Primary ΔAP = AP(H+A)-AP(H) = **{delta:.8f}**。",
        "", "## 6. Basic direction", "",
        f"GT-related D: `{json.dumps(result['disagreement_direction']['GT_related'], ensure_ascii=False)}`。",
        "", f"Background D: `{json.dumps(result['disagreement_direction']['background'], ensure_ascii=False)}`。",
        "", "## 7. Height-conditioned Agreement", "",
        f"Matched pairs={matched['matched_pairs']}；mean |ΔH|={matched['mean_abs_delta_H']}；"
        f"median |ΔH|={matched['median_abs_delta_H']}；Height AUC={matched['Height_ROC_AUC']}；"
        f"Agreement AUC={matched['Agreement_ROC_AUC']}。",
        "", "## 8. Context-count and boundary controls", "",
        f"Count-controlled: `{json.dumps(result['context_count_control'], ensure_ascii=False)}`。",
        "", f"Boundary control: `{json.dumps(result['boundary_control'], ensure_ascii=False)}`。",
        "", "## 9. Native-missed subset", "",
        f"`{json.dumps(result['native_missed_subset'], ensure_ascii=False)}`",
        "", "## 10. Matched-FP recovery", "",
        "| Evidence | GT-related | Native-missed recovered | FP |",
        "|---|---:|---:|---:|",
        f"| Height | {fp['height']['GT_related']} | {fp['height']['Native_missed_recovered']} | {fp['height']['FP']} |",
        f"| Height+Agreement | {fp['combined']['GT_related']} | {fp['combined']['Native_missed_recovered']} | {fp['combined']['FP']} |",
        "", "## 11. Subject bootstrap", "",
        f"ΔAP mean={bootstrap['mean']}；95% CI={bootstrap['CI95']}。",
        "", "## 12. Pre-fixed GO checks", "",
        f"`{json.dumps(result['go_checks'], ensure_ascii=False)}`",
        "", "## 13. Protocol exclusions", "",
        "未搜索 std window、MAD/std、context subset、context weighting、fusion weight、threshold、"
        "candidate generation 或 smoothing。MAD 只作为预先允许的 secondary robustness 描述，"
        "不参与 verdict。未运行 CASME3、BoostingVRME、Skill 或新 decoder。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = arguments()
    with args.fresh_context_cache.open("rb") as handle:
        fresh = pickle.load(handle)
    with args.gt_reference_cache.open("rb") as handle:
        reference = pickle.load(handle)
    if fresh.get("status") != "COMPLETE" or fresh.get("schema") != "sammlv_pre_stitch_context_v1":
        raise RuntimeError("fresh pre-stitch cache is not COMPLETE v1")
    if fresh.get("num_subjects") != 29 or fresh.get("num_videos") != 79:
        raise RuntimeError("fresh cache does not cover 29 subjects / 79 videos")
    if fresh.get("historical_prediction_fields_used") != []:
        raise RuntimeError("fresh extraction provenance mixed historical predictions")
    reference_rows = {
        (str(row["subject"]), str(row["video"])): row for row in reference["records"]
    }
    k_p = int(reference["k_p"])
    stride = int(fresh["stride"])
    window_length = int(fresh["window_length"])
    batch_size = int(fresh["batch_size"])

    records_by_subject = defaultdict(list)
    for record in fresh["records"]:
        records_by_subject[str(record["subject"])].append(record)
    for subject, records in records_by_subject.items():
        records.sort(key=lambda row: int(row["subject_video_index"]))
        video_num = [int(row["num_windows"]) for row in records]
        raw_windows = []
        for record in records:
            n_windows = int(record["num_windows"])
            expected_window_id = np.repeat(np.arange(n_windows, dtype=np.int32), window_length)
            expected_local_t = np.tile(np.arange(window_length, dtype=np.int16), n_windows)
            expected_global_t = expected_window_id * stride + expected_local_t
            if not np.array_equal(np.asarray(record["window_id"]), expected_window_id):
                raise RuntimeError(f"{subject}/{record['video_id']}: window_id mapping mismatch")
            if not np.array_equal(np.asarray(record["local_t"]), expected_local_t):
                raise RuntimeError(f"{subject}/{record['video_id']}: local_t mapping mismatch")
            if not np.array_equal(np.asarray(record["global_t"]), expected_global_t):
                raise RuntimeError(f"{subject}/{record['video_id']}: global_t mapping mismatch")
            raw_windows.append(
                np.asarray(record["raw_score"], dtype=np.float32).reshape(n_windows, window_length)
            )
        rebuilt = stitch_subject(
            np.concatenate(raw_windows), video_num, batch_size, window_length, stride
        )
        for record, rebuilt_score in zip(records, rebuilt):
            if not np.array_equal(
                np.asarray(record["fresh_stitched_score"], dtype=np.float32), rebuilt_score
            ):
                raise RuntimeError(f"{subject}/{record['video_id']}: raw-to-stitch self-check failed")

    fresh_counts = {"TP": 0, "FP": 0, "FN": 0}
    old_counts = {"TP": 0, "FP": 0, "FN": 0}
    exact_peak_videos = 0
    gt_total = 0
    video_data = []
    all_context_counts = []
    for record in fresh["records"]:
        key = (str(record["subject"]), str(record["video_id"]))
        ref = reference_rows[key]
        score = np.asarray(record["fresh_stitched_score"], dtype=float)
        old_score = np.asarray(ref["score"], dtype=float)
        decoded = native_decode(score, k_p)
        old_decoded = native_decode(old_score, k_p)
        roles, missed, counts = native_match(decoded["peaks"], ref["samples"], k_p)
        _, _, old_video_counts = native_match(old_decoded["peaks"], ref["samples"], k_p)
        for name in fresh_counts:
            fresh_counts[name] += counts[name]
            old_counts[name] += old_video_counts[name]
        exact_peak_videos += int(np.array_equal(decoded["peaks"], old_decoded["peaks"]))
        gt_total += len(ref["samples"])
        context_count = np.asarray(record["context_count"], dtype=int)
        expected_count = np.bincount(
            np.asarray(record["global_t"], dtype=int), minlength=len(score)
        )
        if not np.array_equal(context_count, expected_count):
            raise RuntimeError(f"{key}: context_count is not derived from raw observations")
        all_context_counts.extend(context_count.tolist())
        video_data.append((record, ref["samples"], decoded, roles, missed))

    fresh_summary = spotting_summary(fresh_counts)
    old_summary = spotting_summary(old_counts)
    reproduction = {
        "subjects": 29, "videos": 79, "GT": gt_total,
        "fresh": fresh_summary, "historical_reference": old_summary,
        "exact_peak_list_equivalent_videos": exact_peak_videos,
        "gate_rule": "continue only when aggregate TP/FP/FN are exactly identical",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "context_agreement_results.json"
    report_path = args.output_dir / "PRE_STITCH_CONTEXT_AGREEMENT_AUDIT_CN.md"
    base = {
        "engineering_equivalence": "PASS-DECODER-EQUIVALENT",
        "protocol": {
            "dataset": "SAMMLV", "backbone": "ME-TST+", "source_side_only": True,
            "fresh_only_scientific_features": True,
            "historical_numeric_use": "full-reproduction sanity reference only",
            "candidate_generation": "fresh smooth(2*k_p) -> native threshold p=0.55 -> find_peaks(distance=k_p)",
            "disagreement_primary": "population std of all true overlapping raw window scores at candidate global_t",
            "disagreement_secondary": "median absolute deviation; descriptive robustness only",
            "combined": "per-video robust Height minus per-video robust Disagreement; no searched weight",
            "boundary_radius": stride,
            "bootstrap_unit": "subject", "bootstrap_repeats": BOOTSTRAP_REPEATS, "seed": SEED,
            "forbidden_methods_run": [],
        },
        "provenance": {
            "fresh_context_cache": str(args.fresh_context_cache),
            "fresh_context_cache_sha256": sha256(args.fresh_context_cache),
            "gt_reference_cache": str(args.gt_reference_cache),
            "gt_reference_cache_sha256": sha256(args.gt_reference_cache),
            "fresh_extraction": {key: fresh.get(key) for key in (
                "source_commit", "input_feature_cache", "input_feature_cache_sha256",
                "weights_dir", "batch_size", "window_length", "stride", "stitching",
                "environment", "training_calls",
            )},
            "analysis_environment": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            },
        },
        "full_reproduction_sanity": reproduction,
    }
    if any(fresh_counts[key] != old_counts[key] for key in fresh_counts):
        result = {**base, "status": "BLOCKED-FULL-REPRODUCTION", "scientific_audit_executed": False}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(render_report(result), encoding="utf-8")
        return 2

    candidates = []
    native_missed_ids = set()
    for record, samples, decoded, roles, missed in video_data:
        subject, video_id = str(record["subject"]), str(record["video_id"])
        raw_global = np.asarray(record["global_t"], dtype=int)
        raw_score = np.asarray(record["raw_score"], dtype=float)
        context_count = np.asarray(record["context_count"], dtype=int)
        height_curve = robust_z(decoded["curve"])
        missed_ids = {f"{subject}::{video_id}::{index}" for index in missed}
        native_missed_ids.update(missed_ids)
        for peak in decoded["peaks"]:
            observations = raw_score[raw_global == peak]
            if len(observations) != context_count[peak]:
                raise RuntimeError(f"{subject}/{video_id}/{peak}: raw context mismatch")
            containing = {
                index for index, sample in enumerate(samples)
                if int(sample[0]) <= int(peak) <= int(sample[2])
            }
            missed_containing = containing & missed
            best_iou = max(
                [interval_iou(int(peak - k_p), int(peak + k_p), sample) for sample in samples]
                or [0.0]
            )
            native_role, native_gt_index = roles[int(peak)]
            if native_role == "native_tp":
                category = "native_tp_related"
            elif missed_containing:
                category = "native_missed_gt_local_candidate"
            elif containing:
                category = "GT_related_native_FP"
            else:
                category = "native_FP_background"
            median = float(np.median(observations))
            disagreement_mad = float(np.median(np.abs(observations - median)))
            candidates.append({
                "subject": subject, "video_id": video_id, "peak": int(peak),
                "GT_related": int(bool(containing)), "IoU_positive": int(best_iou >= 0.5),
                "native_TP_related": int(native_role == "native_tp"),
                "native_FP": int(native_role == "native_fp"),
                "native_missed_GT_local_candidate": int(bool(missed_containing)),
                "native_gt_index": native_gt_index, "category": category,
                "missed_gt_ids": [f"{subject}::{video_id}::{index}" for index in sorted(missed_containing)],
                "context_count": int(len(observations)),
                "height_raw": float(decoded["curve"][peak]),
                "height": float(height_curve[peak]),
                "disagreement": float(np.std(observations, ddof=0)),
                "agreement": float(-np.std(observations, ddof=0)),
                "disagreement_MAD": disagreement_mad, "agreement_MAD": -disagreement_mad,
                "distance_video_start": int(peak),
                "distance_video_end": int(len(decoded["curve"]) - 1 - peak),
                "threshold": decoded["threshold"],
            })

    count_values, count_frequencies = np.unique(all_context_counts, return_counts=True)
    position_distribution = {
        str(int(count)): {
            "positions": int(frequency),
            "fraction": float(frequency / len(all_context_counts)),
        }
        for count, frequency in zip(count_values, count_frequencies)
    }
    candidate_count_values, candidate_count_frequencies = np.unique(
        [row["context_count"] for row in candidates], return_counts=True
    )
    candidate_distribution = {
        str(int(count)): {
            "candidates": int(frequency), "fraction": float(frequency / len(candidates))
        }
        for count, frequency in zip(candidate_count_values, candidate_count_frequencies)
    } if candidates else {}
    eligible = [row for row in candidates if row["context_count"] >= 2]
    multiplicity = {
        "all_positions": position_distribution,
        "position_min": int(min(all_context_counts)),
        "position_median": float(np.median(all_context_counts)),
        "position_mean": float(np.mean(all_context_counts)),
        "position_max": int(max(all_context_counts)),
        "candidate_context_counts": candidate_distribution,
        "candidate_count": len(candidates), "eligible_candidates": len(eligible),
        "ineligible_candidates": len(candidates) - len(eligible),
        "ineligible_fraction": (len(candidates) - len(eligible)) / len(candidates)
        if candidates else 1.0,
    }
    if multiplicity["ineligible_fraction"] > 0.5:
        result = {**base, "status": "NO-GO-INSUFFICIENT-MULTIPLICITY",
                  "scientific_audit_executed": False, "context_multiplicity": multiplicity}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(render_report(result), encoding="utf-8")
        return 3

    by_video = defaultdict(list)
    for index, row in enumerate(eligible):
        by_video[(row["subject"], row["video_id"])].append(index)
    for indices in by_video.values():
        z_disagreement = robust_z(np.asarray([eligible[index]["disagreement"] for index in indices]))
        for index, value in zip(indices, z_disagreement):
            eligible[index]["z_disagreement"] = float(value)
            eligible[index]["combined"] = float(eligible[index]["height"] - value)

    score_names = ("height", "agreement", "combined")
    primary = metric_summary(eligible, score_names, "GT_related")
    secondary = metric_summary(eligible, score_names, "IoU_positive")
    mad_secondary = metric_summary(eligible, ("agreement_MAD",), "GT_related")
    positive_d = [row["disagreement"] for row in eligible if row["GT_related"]]
    negative_d = [row["disagreement"] for row in eligible if not row["GT_related"]]
    matched_rows, matched_summary = height_match(eligible)

    subject_rows = []
    subjects_lower_median_d = 0
    subjects_with_both_groups = 0
    for subject in sorted({row["subject"] for row in eligible}):
        current = [row for row in eligible if row["subject"] == subject]
        metrics = metric_summary(current, score_names, "GT_related")
        positive_distribution = distribution([
            row["disagreement"] for row in current if row["GT_related"]
        ])
        negative_distribution = distribution([
            row["disagreement"] for row in current if not row["GT_related"]
        ])
        if positive_distribution["n"] and negative_distribution["n"]:
            subjects_with_both_groups += 1
            subjects_lower_median_d += int(
                positive_distribution["median"] < negative_distribution["median"]
            )
        subject_rows.append({
            "subject": subject, "candidates": len(current),
            "positive": sum(row["GT_related"] for row in current),
            "mean_D_positive": positive_distribution["mean"],
            "median_D_positive": positive_distribution["median"],
            "std_D_positive": positive_distribution["std"],
            "IQR_D_positive": positive_distribution["IQR"],
            "mean_D_negative": negative_distribution["mean"],
            "median_D_negative": negative_distribution["median"],
            "std_D_negative": negative_distribution["std"],
            "IQR_D_negative": negative_distribution["IQR"],
            "Height_ROC_AUC": metrics["height"]["ROC_AUC"],
            "Agreement_ROC_AUC": metrics["agreement"]["ROC_AUC"],
            "Combined_ROC_AUC": metrics["combined"]["ROC_AUC"],
            "Height_PR_AUC": metrics["height"]["PR_AUC"],
            "Agreement_PR_AUC": metrics["agreement"]["PR_AUC"],
            "Combined_PR_AUC": metrics["combined"]["PR_AUC"],
        })

    count_control = {}
    for name, subset in (
        ("count_2", [row for row in eligible if row["context_count"] == 2]),
        ("count_3", [row for row in eligible if row["context_count"] == 3]),
        ("count_ge_4", [row for row in eligible if row["context_count"] >= 4]),
    ):
        _, match_summary = height_match(subset)
        count_control[name] = {
            "candidates": len(subset),
            "D_GT_related": distribution([row["disagreement"] for row in subset if row["GT_related"]]),
            "D_background": distribution([row["disagreement"] for row in subset if not row["GT_related"]]),
            "metrics": metric_summary(subset, score_names, "GT_related"),
            "height_conditioned": match_summary,
        }

    boundary_kept = [
        row for row in eligible
        if min(row["distance_video_start"], row["distance_video_end"]) >= stride
    ]
    positive_boundary_distance = [
        min(row["distance_video_start"], row["distance_video_end"])
        for row in eligible if row["GT_related"]
    ]
    negative_boundary_distance = [
        min(row["distance_video_start"], row["distance_video_end"])
        for row in eligible if not row["GT_related"]
    ]
    boundary_ks = (
        ks_2samp(positive_boundary_distance, negative_boundary_distance)
        if positive_boundary_distance and negative_boundary_distance else None
    )
    boundary = {
        "boundary_radius": stride,
        "GT_related_distance_to_nearest_boundary": distribution(positive_boundary_distance),
        "background_distance_to_nearest_boundary": distribution(negative_boundary_distance),
        "two_sample_KS": {
            "statistic": float(boundary_ks.statistic) if boundary_ks is not None else None,
            "pvalue": float(boundary_ks.pvalue) if boundary_ks is not None else None,
        },
        "boundary_candidates_removed": len(eligible) - len(boundary_kept),
        "after_boundary_exclusion": metric_summary(boundary_kept, score_names, "GT_related"),
    }

    missed_subset_rows = [
        row for row in eligible
        if row["native_missed_GT_local_candidate"] or not row["GT_related"]
    ]
    missed_subset = {
        "native_missed_GT_total": len(native_missed_ids),
        "native_missed_GT_with_eligible_local_candidate": len({
            item for row in eligible for item in row["missed_gt_ids"]
        }),
        "metrics": metric_summary(
            missed_subset_rows, score_names, "native_missed_GT_local_candidate"
        ),
    }
    fp_recovery = matched_fp(eligible, fresh_counts["FP"])
    bootstrap = subject_bootstrap(eligible, matched_rows)

    combined_delta = primary["combined"]["PR_AUC"] - primary["height"]["PR_AUC"]
    agreement_visible = (
        primary["agreement"]["ROC_AUC"] is not None
        and primary["agreement"]["ROC_AUC"] > 0.55
        and primary["agreement"]["PR_AUC"] > primary["agreement"]["prevalence"]
    )
    count2_metrics = count_control["count_2"]["metrics"]
    boundary_metrics = boundary["after_boundary_exclusion"]
    missed_metrics = missed_subset["metrics"]
    checks = {
        "lower_D_for_GT_related": distribution(positive_d)["median"] < distribution(negative_d)["median"],
        "Agreement_visible_discrimination": agreement_visible,
        "height_conditioned_Agreement_AUC_gt_0_55": (
            matched_summary["Agreement_ROC_AUC"] is not None
            and matched_summary["Agreement_ROC_AUC"] > 0.55
        ),
        "combined_AP_gt_Height": combined_delta > 0,
        "bootstrap_delta_AP_CI_strictly_positive": bootstrap["delta_AP_combined_minus_height"]["CI95"][0] is not None
        and bootstrap["delta_AP_combined_minus_height"]["CI95"][0] > 0,
        "missed_subset_combined_AP_gt_Height": missed_metrics["combined"]["PR_AUC"] is not None
        and missed_metrics["combined"]["PR_AUC"] > missed_metrics["height"]["PR_AUC"],
        "matched_FP_recovers_more_missed": fp_recovery["combined"]["Native_missed_recovered"]
        > fp_recovery["height"]["Native_missed_recovered"],
        "count_2_signal_persists": count2_metrics["agreement"]["ROC_AUC"] is not None
        and count2_metrics["agreement"]["ROC_AUC"] > 0.5
        and count2_metrics["combined"]["PR_AUC"] > count2_metrics["height"]["PR_AUC"],
        "boundary_excluded_signal_persists": boundary_metrics["agreement"]["ROC_AUC"] is not None
        and boundary_metrics["agreement"]["ROC_AUC"] > 0.5
        and boundary_metrics["combined"]["PR_AUC"] > boundary_metrics["height"]["PR_AUC"],
    }
    core = (
        "height_conditioned_Agreement_AUC_gt_0_55", "combined_AP_gt_Height",
        "bootstrap_delta_AP_CI_strictly_positive", "matched_FP_recovers_more_missed",
    )
    go = all(checks[name] for name in core) and sum(checks.values()) >= 7
    result = {
        **base, "status": "GO" if go else "NO-GO", "scientific_audit_executed": True,
        "context_multiplicity": multiplicity,
        "disagreement_direction": {
            "GT_related": distribution(positive_d), "background": distribution(negative_d),
            "subjects_with_both_groups": subjects_with_both_groups,
            "subjects_with_lower_median_D_for_GT_related": subjects_lower_median_d,
            "subject_level_csv": "sammlv_context_subject_metrics.csv",
        },
        "candidate_discrimination": {
            "primary_GT_related": primary, "secondary_IoU_positive": secondary,
            "MAD_secondary_only": mad_secondary,
            "delta_AP_combined_minus_height": combined_delta,
        },
        "height_conditioned": matched_summary,
        "context_count_control": count_control,
        "boundary_control": boundary,
        "native_missed_subset": missed_subset,
        "matched_FP_recovery": fp_recovery,
        "subject_bootstrap": bootstrap,
        "go_rule": "all four core checks and at least 7 of 9 total checks must pass",
        "go_checks": checks,
        "candidate_csv": "sammlv_context_candidates.csv",
        "subject_metrics_csv": "sammlv_context_subject_metrics.csv",
    }
    candidate_fields = [key for key in eligible[0] if key != "missed_gt_ids"] + ["missed_gt_ids"]
    write_csv(args.output_dir / "sammlv_context_candidates.csv", eligible, candidate_fields)
    write_csv(args.output_dir / "sammlv_context_subject_metrics.csv", subject_rows, list(subject_rows[0]))
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "full_reproduction": reproduction,
        "eligible_candidates": len(eligible), "delta_AP": combined_delta,
        "height_conditioned_AUC": matched_summary["Agreement_ROC_AUC"],
        "bootstrap_delta_AP": bootstrap["delta_AP_combined_minus_height"],
        "outputs": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
