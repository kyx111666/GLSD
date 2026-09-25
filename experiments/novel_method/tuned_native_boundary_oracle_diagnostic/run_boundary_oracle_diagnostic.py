#!/usr/bin/env python3
"""Retrospective fixed-peak boundary oracle for the locked SAMMLV Tuned Native decoder.

This script replays already-selected outer-fold configurations.  Ground truth is
used only after native decoding and formal matching, to measure an oracle upper
bound.  It does not alter scores, peaks, candidate membership, or model outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
LEGACY_DIR = REPRO_ROOT / "my_method" / "multi_scale_candidate_rescue"
sys.path.insert(0, str(LEGACY_DIR))

from run_mscr_nested_loso import (  # noqa: E402
    K_P,
    interval_iou,
    match_events,
    tuned_native_decode,
)


EXPERIMENT = "SAMMLV-Tuned-Native-Boundary-Error-Oracle-Diagnostic"
EXPECTED = {"TP": 49, "FP": 143, "FN": 110, "event_count": 192}
CLASS_ORDER = ("negative", "positive", "surprise", "others", "neutral")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def native_config(config):
    return {"c_s": config["c_s"], "p": config["p_s"],
            "c_d": config["c_d"], "c_b": config["c_b"]}


def majority_emotion(emotion, onset, offset, peak):
    start = max(0, int(onset) + 1)
    stop = max(1, int(offset) - 1)
    values = list(np.asarray(emotion)[start:stop])
    if not values:
        clipped_peak = min(max(0, int(peak)), len(emotion) - 1)
        values = [int(np.asarray(emotion)[clipped_peak])] if len(emotion) else [3]
    return int(Counter(int(value) for value in values).most_common(1)[0][0])


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def percentile_stats(values, requested=(10, 25, 50, 75, 90)):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None,
                **{f"p{p}": None for p in requested}}
    result = {"count": int(len(array)), "mean": float(array.mean()),
              "median": float(np.median(array))}
    result.update({f"p{p}": float(np.percentile(array, p)) for p in requested})
    return result


def gt_fields(sample):
    return int(sample[0]), int(sample[1]), int(sample[2])


def pair_metrics(candidate, sample):
    gt_onset, gt_peak, gt_offset = gt_fields(sample)
    peak = int(candidate["peak"])
    pred_onset, pred_offset = int(candidate["onset"]), int(candidate["offset"])
    peak_in = gt_onset <= peak <= gt_offset
    peak_distance = 0 if peak_in else min(abs(peak - gt_onset), abs(peak - gt_offset))
    center_distance = abs((pred_onset + pred_offset) / 2 - (gt_onset + gt_offset) / 2)
    gt_length = gt_offset - gt_onset + 1
    pred_length = pred_offset - pred_onset + 1
    return {
        "iou": float(interval_iou(candidate, sample)),
        "peak_in_gt": bool(peak_in),
        "peak_distance_to_gt": int(peak_distance),
        "center_distance": float(center_distance),
        "onset_error": int(pred_onset - gt_onset),
        "offset_error": int(pred_offset - gt_offset),
        "pred_length": int(pred_length),
        "gt_length": int(gt_length),
        "length_ratio": float(pred_length / gt_length),
        "length_error": int(pred_length - gt_length),
        "gt_onset": gt_onset,
        "gt_peak": gt_peak,
        "gt_offset": gt_offset,
    }


def nearest_gt(candidate, samples, gt_indices):
    rows = []
    for index in sorted(gt_indices):
        metrics = pair_metrics(candidate, samples[index])
        rows.append(((-metrics["iou"], metrics["peak_distance_to_gt"],
                      metrics["center_distance"], index), index, metrics))
    if not rows:
        return None, None
    _, index, metrics = min(rows, key=lambda item: item[0])
    return index, metrics


def classify_cohort(metrics, k_native):
    if metrics is None:
        return "B4_PURE_FP"
    if metrics["peak_in_gt"] and metrics["iou"] < 0.5:
        return "B1_PEAK_IN_GT"
    if metrics["peak_distance_to_gt"] <= k_native:
        return "B2_PEAK_NEAR_GT"
    if 0.0 < metrics["iou"] < 0.5:
        return "B3_OVERLAP_NEAR_MISS"
    return "B4_PURE_FP"


def exhaustive_fixed_peak_oracle(candidate, sample, video_length):
    """Exhaustively search every legal integer [onset, offset] containing peak."""
    peak = int(candidate["peak"])
    old_onset, old_offset = int(candidate["onset"]), int(candidate["offset"])
    gt_onset, _, gt_offset = gt_fields(sample)
    best_feasible = None
    max_iou = -1.0
    max_iou_interval = None
    offsets = np.arange(peak, video_length, dtype=np.int64)
    for onset in range(0, peak + 1):
        intersections = np.maximum(
            0,
            np.minimum(offsets, gt_offset) - max(onset, gt_onset) + 1,
        )
        unions = np.maximum(offsets, gt_offset) - min(onset, gt_onset) + 1
        ious = intersections / unions
        local_max = float(ious.max())
        if local_max > max_iou:
            max_iou = local_max
            max_offset = int(offsets[int(np.argmax(ious))])
            max_iou_interval = (onset, max_offset)
        feasible_positions = np.flatnonzero(ious >= 0.5)
        if not len(feasible_positions):
            continue
        feasible_offsets = offsets[feasible_positions]
        changes = abs(onset - old_onset) + np.abs(feasible_offsets - old_offset)
        min_change = int(changes.min())
        for position in feasible_positions[changes == min_change]:
            offset = int(offsets[position])
            iou = float(ious[position])
            rank = (min_change, -iou, onset, offset)
            if best_feasible is None or rank < best_feasible[0]:
                best_feasible = (rank, onset, offset, iou)
    if best_feasible is None:
        return {"recoverable": False, "max_achievable_iou": float(max_iou),
                "max_iou_onset": int(max_iou_interval[0]),
                "max_iou_offset": int(max_iou_interval[1])}
    _, onset, offset, oracle_iou = best_feasible
    return {
        "recoverable": True,
        "min_boundary_change": int(abs(onset - old_onset) + abs(offset - old_offset)),
        "oracle_onset": int(onset),
        "oracle_offset": int(offset),
        "oracle_iou": float(oracle_iou),
        "left_change": int(onset - old_onset),
        "right_change": int(offset - old_offset),
        "max_achievable_iou": float(max_iou),
        "max_iou_onset": int(max_iou_interval[0]),
        "max_iou_offset": int(max_iou_interval[1]),
    }


def oracle_assignment(candidates, unmatched_gts, edge_results):
    candidate_ids = sorted(candidates)
    gt_ids = sorted(unmatched_gts)
    if not candidate_ids or not gt_ids:
        return []
    gt_position = {gt_id: index for index, gt_id in enumerate(gt_ids)}
    dummy_cost = 1e12
    costs = np.full((len(candidate_ids), len(gt_ids) + len(candidate_ids)),
                    dummy_cost, dtype=np.float64)
    for row_index, candidate_id in enumerate(candidate_ids):
        candidate = candidates[candidate_id]
        for gt_id in gt_ids:
            result = edge_results.get((candidate_id, gt_id))
            if not result or not result["recoverable"]:
                costs[row_index, gt_position[gt_id]] = dummy_cost * 2
                continue
            # Cardinality is protected by the dummy cost.  Within maximum
            # cardinality, minimize correction, then prefer achievable IoU and
            # original confidence. Tiny ID terms make ties deterministic.
            confidence = float(candidate["confidence"])
            costs[row_index, gt_position[gt_id]] = (
                result["min_boundary_change"] * 1e6
                + (1.0 - result["max_achievable_iou"]) * 1e3
                + (1.0 - confidence)
                + row_index * 1e-6
                + gt_position[gt_id] * 1e-9
            )
    rows, columns = linear_sum_assignment(costs)
    assigned = []
    for row, column in zip(rows, columns):
        if column >= len(gt_ids) or costs[row, column] >= dummy_cost:
            continue
        candidate_id = candidate_ids[row]
        gt_id = gt_ids[column]
        assigned.append((candidate_id, gt_id, edge_results[(candidate_id, gt_id)]))
    return assigned


def error_type(candidate, metrics, oracle):
    if not metrics["peak_in_gt"]:
        return "PEAK_OUTSIDE_GT"
    pred_inside_gt = (candidate["onset"] >= metrics["gt_onset"]
                      and candidate["offset"] <= metrics["gt_offset"])
    gt_inside_pred = (candidate["onset"] <= metrics["gt_onset"]
                      and candidate["offset"] >= metrics["gt_offset"])
    if pred_inside_gt and metrics["pred_length"] < metrics["gt_length"]:
        return "TOO_SHORT"
    if gt_inside_pred and metrics["pred_length"] > metrics["gt_length"]:
        return "TOO_LONG"
    left = abs(oracle["left_change"])
    right = abs(oracle["right_change"])
    if left > 2 * right:
        return "LEFT_BOUNDARY_ERROR"
    if right > 2 * left:
        return "RIGHT_BOUNDARY_ERROR"
    return "BOTH_BOUNDARIES"


def asymmetry_row(label, peak, metrics):
    left = int(peak - metrics["gt_onset"])
    right = int(metrics["gt_offset"] - peak)
    denominator = max(1, left + right)
    return {
        "group": label,
        "L_gt": left,
        "R_gt": right,
        "asymmetry": float(abs(left - right) / denominator),
        "signed_asymmetry": float((right - left) / denominator),
        "L_not_equal_R": bool(left != right),
        "length_ratio": metrics["length_ratio"],
        "length_error": metrics["length_error"],
    }


def summarize_asymmetry(rows, group):
    subset = [row for row in rows if row["group"] == group]
    result = {"group": group, "count": len(subset)}
    for field in ("L_gt", "R_gt", "asymmetry", "signed_asymmetry",
                  "length_ratio", "length_error"):
        stats = percentile_stats([row[field] for row in subset], requested=(25, 50, 75))
        for key, value in stats.items():
            if key != "count":
                result[f"{field}_{key}"] = value
    result["proportion_L_not_equal_R"] = (
        float(np.mean([row["L_not_equal_R"] for row in subset])) if subset else None
    )
    result["proportion_asymmetry_ge_0_2"] = (
        float(np.mean([row["asymmetry"] >= 0.2 for row in subset])) if subset else None
    )
    result["proportion_asymmetry_ge_0_4"] = (
        float(np.mean([row["asymmetry"] >= 0.4 for row in subset])) if subset else None
    )
    return result


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main():
    args = parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {args.output_root}")
    args.output_root.mkdir(parents=True)
    errors = []

    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    cache_digest = sha256(args.cache)
    if payload.get("dataset") != "SAMMLV" or int(payload.get("k_p", -1)) != K_P:
        raise RuntimeError("expected the locked SAMMLV cache with k_p=5")
    source_tuned = source_report.get("aggregate_metrics", {}).get("Tuned Native", {})
    if source_report.get("incomplete"):
        raise RuntimeError("source Tuned Native report is incomplete")
    if source_report.get("cache_sha256") != cache_digest:
        raise RuntimeError("source report and supplied frozen cache hashes differ")
    if any(int(source_tuned.get(key, -1)) != value for key, value in EXPECTED.items()):
        raise RuntimeError("source report does not contain the locked Tuned Native anchor")
    selections = {
        str(fold["subject"]): fold["selected_strong_config"]
        for fold in source_report["outer_folds"]
    }
    records = list(payload["records"])
    subjects = sorted({str(record["subject"]) for record in records})
    if set(selections) != set(subjects):
        raise RuntimeError("source-report outer folds do not exactly cover cache subjects")

    candidate_rows = []
    tp_rows = []
    fp_candidates = {}
    unmatched_gt_global = {}
    record_lookup = {}
    curve_lookup = {}
    native_totals = Counter(TP=0, FP=0, FN=0, event_count=0)

    for record in records:
        subject, video = str(record["subject"]), str(record["video"])
        record_key = f"{subject}/{video}"
        record_lookup[record_key] = record
        selected = selections[subject]
        decoded = tuned_native_decode(record, native_config(selected))
        curve_lookup[record_key] = decoded["curve"]
        events = decoded["events"]
        matches, unmatched = match_events(events, record["samples"])
        native_totals["TP"] += sum(index >= 0 for index in matches)
        native_totals["FP"] += sum(index < 0 for index in matches)
        native_totals["FN"] += len(unmatched)
        native_totals["event_count"] += len(events)
        boundary = max(1, int(round(float(selected["c_b"]) * K_P)))
        unmatched_gt_global[record_key] = {
            f"{record_key}/gt_{index}": int(index) for index in sorted(unmatched)
        }
        all_gt_indices = set(range(len(record["samples"])))
        for video_candidate_index, (candidate, match) in enumerate(zip(events, matches)):
            candidate_id = f"{record_key}/pred_{video_candidate_index}"
            emotion_id = majority_emotion(record["emotion"], candidate["onset"],
                                           candidate["offset"], candidate["peak"])
            base = {
                "candidate_id": candidate_id,
                "subject": subject,
                "video": video,
                "onset_pred": int(candidate["onset"]),
                "peak_pred": int(candidate["peak"]),
                "offset_pred": int(candidate["offset"]),
                "interval_length": int(candidate["offset"] - candidate["onset"] + 1),
                "confidence": float(decoded["curve"][candidate["peak"]]),
                "confidence_definition": "Tuned Native smoothed spotting score at fixed peak",
                "emotion_id": emotion_id,
                "emotion": CLASS_ORDER[emotion_id] if 0 <= emotion_id < len(CLASS_ORDER) else str(emotion_id),
                "outer_fold": subject,
                "tuned_native_config": json.dumps(selected, sort_keys=True),
                "k_native": boundary,
                "formal_match_status": "TP" if match >= 0 else "FP",
                "formal_matched_gt_index": int(match),
            }
            nearest_all_index, _ = nearest_gt(candidate, record["samples"], all_gt_indices)
            nearest_unmatched_index, nearest_unmatched_metrics = nearest_gt(
                candidate, record["samples"], unmatched
            )
            cohort = ("A_NATIVE_TP" if match >= 0
                      else classify_cohort(nearest_unmatched_metrics, boundary))
            # Preserve one row for every candidate x same-video GT pair. This
            # makes the prescribed nearest-GT ranking independently auditable.
            if record["samples"]:
                for gt_index, sample in enumerate(record["samples"]):
                    metrics_for_pair = pair_metrics(candidate, sample)
                    candidate_rows.append({
                        **base,
                        **{f"pair_{key}": value for key, value in metrics_for_pair.items()},
                        "pair_gt_id": f"{record_key}/gt_{gt_index}",
                        "pair_gt_index": gt_index,
                        "pair_is_formal_match": bool(gt_index == match),
                        "pair_is_currently_unmatched_gt": bool(gt_index in unmatched),
                        "pair_is_nearest_all_gt": bool(gt_index == nearest_all_index),
                        "pair_is_nearest_unmatched_gt": bool(gt_index == nearest_unmatched_index),
                        "cohort": cohort,
                    })
            else:
                candidate_rows.append({**base, "pair_gt_id": "", "cohort": cohort})
            if match >= 0:
                metrics_row = pair_metrics(candidate, record["samples"][match])
                tp_rows.append({
                    "candidate_id": candidate_id, "subject": subject, "video": video,
                    "gt_id": f"{record_key}/gt_{match}", "iou": metrics_row["iou"],
                    "peak_in_gt": metrics_row["peak_in_gt"],
                    "pred_onset": candidate["onset"], "pred_peak": candidate["peak"],
                    "pred_offset": candidate["offset"],
                    "gt_onset": metrics_row["gt_onset"], "gt_peak": metrics_row["gt_peak"],
                    "gt_offset": metrics_row["gt_offset"],
                })
                base["pair_metrics"] = metrics_row
                base["candidate"] = candidate
                base["record_key"] = record_key
                base["matched_gt_id"] = f"{record_key}/gt_{match}"
                fp_candidates.setdefault("__native_tp__", {})[candidate_id] = base
            else:
                fp_candidates[candidate_id] = {
                    **base, "candidate": candidate, "record_key": record_key,
                    "nearest_gt_index": nearest_unmatched_index,
                    "nearest_metrics": nearest_unmatched_metrics, "cohort": cohort,
                }

    observed = dict(native_totals)
    observed["F1"] = f1(observed["TP"], observed["FP"], observed["FN"])
    anchor_pass = all(observed[key] == value for key, value in EXPECTED.items())
    if not anchor_pass:
        errors.append(f"Tuned Native anchor mismatch: {observed} != {EXPECTED}")
        report = {
            "experiment_name": EXPERIMENT, "dataset": "SAMMLV", "backbone": "ME-TST+ frozen",
            "TUNED_NATIVE_ANCHOR": "FAIL", "source_result_anchor": EXPECTED,
            "observed_native": observed, "cache_path": str(args.cache.resolve()),
            "cache_sha256": cache_digest, "no_training": True,
            "no_new_decoder": True, "no_candidate_addition": True,
            "fixed_peak_oracle": True, "errors": errors, "incomplete": True,
        }
        (args.output_root / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    # Oracle search is reached only after the exact anchor gate passes.
    edge_results = {}
    eligible_candidates = {
        candidate_id: candidate for candidate_id, candidate in fp_candidates.items()
        if candidate_id != "__native_tp__" and candidate["cohort"] != "B4_PURE_FP"
    }
    all_unmatched_gt_ids = {}
    for record_key, mapping in unmatched_gt_global.items():
        all_unmatched_gt_ids.update({gt_id: (record_key, index) for gt_id, index in mapping.items()})
    for candidate_id, candidate in eligible_candidates.items():
        record = record_lookup[candidate["record_key"]]
        for gt_id, gt_index in unmatched_gt_global[candidate["record_key"]].items():
            edge_results[(candidate_id, gt_id)] = exhaustive_fixed_peak_oracle(
                candidate["candidate"], record["samples"][gt_index], len(record["score"])
            )

    assignments = oracle_assignment(eligible_candidates, all_unmatched_gt_ids, edge_results)
    oracle_rows = []
    asymmetry_cases = []
    trajectory_rows = []
    for candidate_id, gt_id, oracle in assignments:
        candidate = eligible_candidates[candidate_id]
        record_key, gt_index = all_unmatched_gt_ids[gt_id]
        record = record_lookup[record_key]
        metrics_row = pair_metrics(candidate["candidate"], record["samples"][gt_index])
        category = error_type(candidate["candidate"], metrics_row, oracle)
        row = {
            "candidate_id": candidate_id, "subject": candidate["subject"], "video": candidate["video"],
            "pred_onset": candidate["onset_pred"], "pred_peak": candidate["peak_pred"],
            "pred_offset": candidate["offset_pred"], "nearest_gt_id": gt_id,
            "nearest_gt_onset": metrics_row["gt_onset"], "nearest_gt_peak": metrics_row["gt_peak"],
            "nearest_gt_offset": metrics_row["gt_offset"], "original_iou": metrics_row["iou"],
            "peak_in_gt": metrics_row["peak_in_gt"],
            "peak_distance": metrics_row["peak_distance_to_gt"],
            "oracle_onset": oracle["oracle_onset"], "oracle_offset": oracle["oracle_offset"],
            "oracle_iou": oracle["oracle_iou"],
            "max_achievable_iou": oracle["max_achievable_iou"],
            "min_boundary_change": oracle["min_boundary_change"],
            "left_change": oracle["left_change"], "right_change": oracle["right_change"],
            "error_type": category, "candidate_confidence": candidate["confidence"],
            "cohort": candidate["cohort"], "k_native": candidate["k_native"],
            "outer_fold": candidate["outer_fold"],
            "tuned_native_config": candidate["tuned_native_config"],
        }
        oracle_rows.append(row)
        asymmetry_cases.append(asymmetry_row("ORACLE_RECOVERABLE_NEAR_MISS",
                                              candidate["peak_pred"], metrics_row))
        curve = curve_lookup[record_key]
        radius = 2 * candidate["k_native"]
        for index in range(max(0, candidate["peak_pred"] - radius),
                           min(len(curve), candidate["peak_pred"] + radius + 1)):
            trajectory_rows.append({
                "candidate_id": candidate_id, "subject": candidate["subject"],
                "video": candidate["video"], "absolute_index": index,
                "relative_index": index - candidate["peak_pred"],
                "smoothed_spotting_score": float(curve[index]),
                "k_native": candidate["k_native"],
            })

    for candidate in fp_candidates.get("__native_tp__", {}).values():
        asymmetry_cases.append(asymmetry_row("NATIVE_TP", candidate["peak_pred"],
                                              candidate["pair_metrics"]))

    cohort_order = ["B1_PEAK_IN_GT", "B2_PEAK_NEAR_GT",
                    "B3_OVERLAP_NEAR_MISS", "B4_PURE_FP"]
    cohort_counts = Counter(candidate["cohort"] for key, candidate in fp_candidates.items()
                            if key != "__native_tp__")
    cohort_rows = [{"cohort": cohort, "count": cohort_counts[cohort],
                    "percent_of_native_fp": 100 * cohort_counts[cohort] / observed["FP"]}
                   for cohort in cohort_order]
    error_counts = Counter(row["error_type"] for row in oracle_rows)
    error_order = ["TOO_SHORT", "TOO_LONG", "LEFT_BOUNDARY_ERROR",
                   "RIGHT_BOUNDARY_ERROR", "BOTH_BOUNDARIES", "PEAK_OUTSIDE_GT"]
    error_rows = [{"error_type": name, "count": error_counts[name],
                   "percent_of_recoverable": (100 * error_counts[name] / len(oracle_rows)
                                                if oracle_rows else 0.0)}
                  for name in error_order]

    asymmetry_summary = [summarize_asymmetry(asymmetry_cases, "NATIVE_TP"),
                         summarize_asymmetry(asymmetry_cases, "ORACLE_RECOVERABLE_NEAR_MISS")]
    tp_ious = [row["iou"] for row in tp_rows]
    tp_iou_stats = percentile_stats(tp_ious, requested=(10, 25, 50, 75, 90))
    tp_iou_stats.update({
        "proportion_iou_ge_0_7": float(np.mean(np.asarray(tp_ious) >= 0.7)),
        "proportion_iou_ge_0_8": float(np.mean(np.asarray(tp_ious) >= 0.8)),
        "proportion_iou_ge_0_9": float(np.mean(np.asarray(tp_ious) >= 0.9)),
        "proportion_peak_in_gt": float(np.mean([row["peak_in_gt"] for row in tp_rows])),
    })
    correction_stats = percentile_stats(
        [row["min_boundary_change"] for row in oracle_rows], requested=(25, 50, 75, 100)
    )
    recoverable = len(oracle_rows)
    oracle_metrics = {
        "TP": observed["TP"] + recoverable,
        "FP": observed["FP"] - recoverable,
        "FN": observed["FN"] - recoverable,
    }
    oracle_metrics["F1"] = f1(oracle_metrics["TP"], oracle_metrics["FP"], oracle_metrics["FN"])
    recoverable_subjects = sorted({row["subject"] for row in oracle_rows})
    recoverable_gt_ids = {row["nearest_gt_id"] for row in oracle_rows}
    peak_in_gt_oracle_level1 = sum(
        candidate["nearest_metrics"] is not None and candidate["nearest_metrics"]["peak_in_gt"]
        for key, candidate in fp_candidates.items() if key != "__native_tp__"
    )

    recoverable_asymmetry = asymmetry_summary[1]
    top_error_fraction = max(error_counts.values(), default=0) / recoverable if recoverable else 0.0
    systematic_pattern = bool(
        top_error_fraction >= 0.35
        or (recoverable_asymmetry.get("asymmetry_median") is not None
            and recoverable_asymmetry["asymmetry_median"] >= 0.2)
    )
    oracle_delta_f1 = oracle_metrics["F1"] - observed["F1"]
    if ((recoverable >= 10 or recoverable >= 0.10 * observed["FN"])
            and len(recoverable_subjects) >= 2 and systematic_pattern
            and oracle_delta_f1 >= 0.01):
        grade = "BOUNDARY-EVIDENCE-STRONG"
    elif recoverable >= 4:
        grade = "BOUNDARY-EVIDENCE-WEAK"
    else:
        grade = "BOUNDARY-EVIDENCE-NONE"

    complete_cohort_counts = {name: int(cohort_counts[name]) for name in cohort_order}
    complete_error_counts = {name: int(error_counts[name]) for name in error_order}
    report = {
        "experiment_name": EXPERIMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "SAMMLV",
        "backbone": "ME-TST+ frozen",
        "source_decoder": "existing Tuned Native decoder; locked outer-fold configs replayed",
        "source_result_anchor": {**EXPECTED, "Spotting_F1": 0.2792022792022792,
                                 "Recognition_F1": 0.6990, "STRS": 0.19516239316239314},
        "TUNED_NATIVE_ANCHOR": "PASS",
        "cache_path": str(args.cache.resolve()),
        "cache_sha256": cache_digest,
        "source_report_path": str(args.source_report.resolve()),
        "no_training": True,
        "no_model_forward": True,
        "no_new_decoder": True,
        "no_candidate_addition": True,
        "no_candidate_deletion": True,
        "no_peak_movement": True,
        "candidate_membership_changed": False,
        "fixed_peak_oracle": True,
        "oracle_warning": "THIS IS AN ORACLE UPPER BOUND, NOT A MODEL RESULT",
        "subject_count": len(subjects),
        "video_count": len(records),
        "gt_count": int(payload["num_gt"]),
        "Native": observed,
        "cohort_counts": complete_cohort_counts,
        "oracle_level_1": {"oracle_recoverable_peak_in_gt": int(peak_in_gt_oracle_level1)},
        "oracle_recoverable_count": recoverable,
        "oracle_unique_recoverable_gt_count": len(recoverable_gt_ids),
        "oracle_unrecoverable_native_fn_count": observed["FN"] - recoverable,
        "oracle_recoverable_fraction_of_native_fn": recoverable / observed["FN"],
        "oracle_recoverable_fraction_of_native_fp": recoverable / observed["FP"],
        "oracle_upper_bound": oracle_metrics,
        "oracle_spotting_f1_gain": oracle_delta_f1,
        "boundary_correction_magnitude": correction_stats,
        "asymmetry_statistics": {row["group"]: row for row in asymmetry_summary},
        "native_tp_iou_statistics": tp_iou_stats,
        "boundary_error_type_statistics": complete_error_counts,
        "recoverable_subject_count": len(recoverable_subjects),
        "recoverable_subjects": recoverable_subjects,
        "evidence_grade_inputs": {
            "systematic_pattern": systematic_pattern,
            "largest_error_type_fraction": top_error_fraction,
            "oracle_delta_f1": oracle_delta_f1,
        },
        "final_evidence_grade": grade,
        "errors": errors,
        "incomplete": False,
    }

    write_csv(args.output_root / "boundary_cohort_summary.csv", cohort_rows,
              ["cohort", "count", "percent_of_native_fp"])
    write_csv(args.output_root / "candidate_gt_pairs.csv", candidate_rows)
    write_csv(args.output_root / "oracle_recoverable_cases.csv", oracle_rows)
    write_csv(args.output_root / "native_tp_iou_distribution.csv", tp_rows)
    write_csv(args.output_root / "boundary_error_type_summary.csv", error_rows,
              ["error_type", "count", "percent_of_recoverable"])
    write_csv(args.output_root / "gt_asymmetry_summary.csv", asymmetry_summary)
    write_csv(args.output_root / "trajectory_data.csv", trajectory_rows)
    (args.output_root / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    native_asymmetry = asymmetry_summary[0]
    recover_asymmetry = asymmetry_summary[1]
    report_md = f"""# Tuned Native Boundary Oracle Diagnostic（SAMMLV）

## A. Integrity

- Tuned Native anchor: **PASS**
- TP / FP / FN / F1: **{observed['TP']} / {observed['FP']} / {observed['FN']} / {observed['F1']:.6f}**
- same frozen cache: YES（SHA-256 `{report['cache_sha256']}`）
- new decoder executed: NO
- training executed: NO
- candidate membership changed: NO

## B. Error cohorts

| Cohort | Count | % of Native FP |
|---|---:|---:|
""" + "\n".join(
        f"| {row['cohort']} | {row['count']} | {row['percent_of_native_fp']:.2f}% |"
        for row in cohort_rows
    ) + f"""

四组互斥且总计 {sum(row['count'] for row in cohort_rows)} 个 Native FP。

## C. Oracle recoverability

- recoverable predictions: **{recoverable}**
- unique recoverable GT: **{len(recoverable_gt_ids)}**
- 占 current FN: **{100 * recoverable / observed['FN']:.2f}%**
- 占 current FP: **{100 * recoverable / observed['FP']:.2f}%**
- Oracle Level 1（peak 位于 unmatched GT）: **{peak_in_gt_oracle_level1}**

| Method | TP | FP | FN | Spot F1 |
|---|---:|---:|---:|---:|
| Tuned Native | {observed['TP']} | {observed['FP']} | {observed['FN']} | {observed['F1']:.6f} |
| Fixed-Peak Boundary Oracle | {oracle_metrics['TP']} | {oracle_metrics['FP']} | {oracle_metrics['FN']} | {oracle_metrics['F1']:.6f} |

**THIS IS AN ORACLE UPPER BOUND, NOT A MODEL RESULT.** GT 只用于事后诊断；候选数量和 peak 均未改变。

## D. Boundary correction magnitude

- min boundary change：median {correction_stats['median']}, p25 {correction_stats['p25']}, p75 {correction_stats['p75']}, max {correction_stats['p100']}
- left-only dominated: {error_counts['LEFT_BOUNDARY_ERROR']}
- right-only dominated: {error_counts['RIGHT_BOUNDARY_ERROR']}
- both: {error_counts['BOTH_BOUNDARIES']}
- too-short: {error_counts['TOO_SHORT']}
- too-long: {error_counts['TOO_LONG']}
- peak-outside-GT: {error_counts['PEAK_OUTSIDE_GT']}

## E. GT asymmetry

Oracle-recoverable GT（相对于 fixed prediction peak）：

- median L_gt: {recover_asymmetry.get('L_gt_median')}
- median R_gt: {recover_asymmetry.get('R_gt_median')}
- median asymmetry: {recover_asymmetry.get('asymmetry_median')}
- proportion asymmetry >= 0.2: {recover_asymmetry.get('proportion_asymmetry_ge_0_2')}
- proportion asymmetry >= 0.4: {recover_asymmetry.get('proportion_asymmetry_ge_0_4')}

作为参照，Native TP 的 median L_gt / R_gt / asymmetry 为 {native_asymmetry.get('L_gt_median')} / {native_asymmetry.get('R_gt_median')} / {native_asymmetry.get('asymmetry_median')}。

## F. Native TP stability

- current TP count: {len(tp_rows)}
- median IoU: {tp_iou_stats['median']:.6f}
- p10 IoU: {tp_iou_stats['p10']:.6f}
- proportion IoU >= 0.7: {tp_iou_stats['proportion_iou_ge_0_7']:.4f}
- proportion peak inside GT: {tp_iou_stats['proportion_peak_in_gt']:.4f}

## G. Subject distribution

可恢复案例来自 **{len(recoverable_subjects)}** 个 subject：{', '.join(recoverable_subjects) if recoverable_subjects else '无'}。

## H. Final evidence grade

**{grade}**

事实依据：

1. fixed-peak boundary-only oracle 可恢复 {recoverable} 个 current FN。
2. recoverable 占 Native FN {100 * recoverable / observed['FN']:.2f}%，占 Native FP {100 * recoverable / observed['FP']:.2f}%。
3. 案例分布于 {len(recoverable_subjects)} 个 subject。
4. oracle Spotting F1 upper bound 为 {oracle_metrics['F1']:.6f}，绝对增量 {oracle_delta_f1:.6f}。
5. 最大 error type 占比为 {100 * top_error_fraction:.2f}%；其中 peak-outside-GT 为 {error_counts['PEAK_OUTSIDE_GT']}/{recoverable}，因此该增益是边界包络的理论上界，不等同于 peak 已精确定位。
6. recoverable GT 的 median asymmetry 为 {recover_asymmetry.get('asymmetry_median')}。
7. Native TP 的 p10 IoU 为 {tp_iou_stats['p10']:.6f}，用于描述已有 TP 的边界敏感性。

此等级只评价 boundary route 的证据强弱，不构成算法或正式模型结果。

## I. Files

- `report.json`
- `boundary_cohort_summary.csv`
- `candidate_gt_pairs.csv`
- `oracle_recoverable_cases.csv`
- `native_tp_iou_distribution.csv`
- `boundary_error_type_summary.csv`
- `gt_asymmetry_summary.csv`
- `trajectory_data.csv`
- `BOUNDARY_ORACLE_DIAGNOSTIC_REPORT_CN.md`
"""
    (args.output_root / "BOUNDARY_ORACLE_DIAGNOSTIC_REPORT_CN.md").write_text(
        report_md, encoding="utf-8"
    )
    print(json.dumps({
        "TUNED_NATIVE_ANCHOR": "PASS", "Native": observed,
        "cohorts": complete_cohort_counts, "oracle_recoverable": recoverable,
        "oracle_upper_bound": oracle_metrics, "final_evidence_grade": grade,
        "output_root": str(args.output_root.resolve()),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
