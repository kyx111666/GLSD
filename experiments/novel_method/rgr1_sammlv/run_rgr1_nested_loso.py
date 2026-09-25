#!/usr/bin/env python3
"""Strict RGR-1 cache-only SAMMLV nested-LOSO experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import subprocess
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
LEGACY_DIR = REPRO_ROOT / "my_method" / "multi_scale_candidate_rescue"
sys.path.insert(0, str(LEGACY_DIR))

from run_mscr_nested_loso import (  # noqa: E402
    EXPECTED_NATIVE, K_P, TUNED_CB, TUNED_CD, TUNED_CS, TUNED_P,
    add_counts, author_native, empty_counts, evaluate_decoding, event, f1,
    metrics, sha256, threshold, tuned_native_decode,
)


EXPERIMENT = "RGR-1-SAMMLV-Nested-LOSO"
VERSION = "RGR1-v1-locked-2026-09-05"
DELTAS = (0.05, 0.10, 0.15, 0.20)
# Reused exactly from the already-existing previous RGR experiment.
GAMMAS = (0.25, 0.35, 0.45, 0.55, 0.65)
CLASS_ORDER = ("negative", "positive", "surprise", "others", "neutral")
NEUTRAL_ID = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--held-subjects", nargs="*", default=None)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_verified_recognition_helpers():
    """Import the existing paper-aligned recognition implementation unchanged.

    Its module imports the unavailable mAP package even though the two helpers
    used here do not need it. A guard satisfies import only and raises if any
    matching code accidentally tries to instantiate it.
    """
    helper_dir = REPRO_ROOT / "third_party" / "metst_plus"
    dependency_dir = helper_dir
    if not (helper_dir / "paper_metrics.py").exists():
        raise RuntimeError("verified paper_metrics.py is missing")
    guard_module = types.ModuleType("Utils.mean_average_precision.mean_average_precision")

    class MatchingGuard:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("RGR-1 must use the already-validated spotting evaluator")

    guard_module.MeanAveragePrecision2d = MatchingGuard
    sys.modules.setdefault("Utils.mean_average_precision.mean_average_precision", guard_module)
    sys.path[:0] = [str(helper_dir), str(dependency_dir)]
    from paper_metrics import _safe_majority_emotion, paper_recognition_summary
    return _safe_majority_emotion, paper_recognition_summary, helper_dir / "paper_metrics.py"


MAJORITY_EMOTION, RECOGNITION_SUMMARY, PAPER_METRICS_PATH = load_verified_recognition_helpers()


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prepare_records(payload):
    prepared, lengths = [], []
    for source in payload["records"]:
        record = dict(source)
        score = np.asarray(record["score"])
        logits = np.asarray(record["logits"])
        emotion = np.asarray(record["emotion"])
        if logits.ndim != 2 or logits.shape[1] != 5:
            raise RuntimeError(f"invalid logits shape: {record['subject']}/{record['video']} {logits.shape}")
        if not (len(score) == len(logits) == len(emotion)):
            raise RuntimeError(f"temporal mismatch: {record['subject']}/{record['video']}")
        if not (np.isfinite(score).all() and np.isfinite(logits).all()):
            raise RuntimeError(f"nonfinite frozen output: {record['subject']}/{record['video']}")
        probabilities = softmax(logits)
        record["R2"] = np.max(probabilities[:, :NEUTRAL_ID], axis=1)
        record["score"] = score
        record["emotion"] = emotion
        prepared.append(record)
        lengths.append(len(score))
    return prepared, {
        "logits_shape": "[T,5]", "softmax_axis": 1,
        "neutral_class_id": NEUTRAL_ID, "class_order": list(CLASS_ORDER),
        "records_checked": len(prepared), "temporal_T_min": min(lengths),
        "temporal_T_max": max(lengths), "all_aligned": True, "all_finite": True,
    }


def strong_grid():
    grid = [dict(zip(("c_s", "p_s", "c_d", "c_b"), values))
            for values in itertools.product(TUNED_CS, TUNED_P, TUNED_CD, TUNED_CB)]
    assert len(grid) == 81
    return grid


def native_config(config):
    return {"c_s": config["c_s"], "p": config["p_s"],
            "c_d": config["c_d"], "c_b": config["c_b"]}


def candidate_r2(record, candidate):
    left = max(0, int(candidate["onset"]))
    right = min(len(record["R2"]), int(candidate["offset"]) + 1)
    if right <= left:
        raise RuntimeError("empty candidate interval")
    return float(np.mean(record["R2"][left:right]))


def decode_rgr(record, strong_config, delta_p, gamma_r, trace=False):
    strong = tuned_native_decode(record, native_config(strong_config))
    distance = max(1, int(round(float(strong_config["c_d"]) * K_P)))
    boundary = max(1, int(round(float(strong_config["c_b"]) * K_P)))
    p_w = float(strong_config["p_s"]) - float(delta_p)
    tau_w = threshold(strong["curve"], p_w)
    low_peaks = find_peaks(strong["curve"], height=tau_w, distance=distance)[0].astype(int)
    strong_peaks = set(map(int, strong["peaks"]))
    low_peak_set = set(map(int, low_peaks))
    if not strong_peaks.issubset(low_peak_set):
        raise RuntimeError("low-threshold peaks do not contain all strong peaks")
    weak_peaks = sorted(low_peak_set - strong_peaks)
    weak_events = [event(peak, boundary, "weak") for peak in weak_peaks]
    rescued, diagnostics = [], []
    for candidate in weak_events:
        recognition = candidate_r2(record, candidate)
        selected = recognition >= float(gamma_r)
        if selected:
            rescued.append({**candidate, "source": "rescue"})
        if trace:
            diagnostics.append({**candidate, "R2_mean": recognition, "selected": selected})
    union = sorted(strong["events"] + rescued,
                   key=lambda item: (item["peak"], 0 if item["source"] == "tuned_native" else 1))
    strong_geometry = {(x["onset"], x["peak"], x["offset"]) for x in strong["events"]}
    union_geometry = {(x["onset"], x["peak"], x["offset"]) for x in union}
    assert strong_geometry.issubset(union_geometry)
    assert all(x["peak"] in low_peak_set for x in rescued)
    assert all(x["peak"] not in strong_peaks for x in rescued)
    return {"native": strong, "weak": weak_peaks, "rescued": rescued,
            "events": union, "diagnostics": diagnostics, "p_w": p_w,
            "native_preserved": len(strong_geometry), "native_total": len(strong_geometry)}


def decode_low(record, strong_config, delta_p, trace=False):
    # Same weak-threshold source as RGR; all W are accepted.
    return decode_rgr(record, strong_config, delta_p, gamma_r=-np.inf, trace=trace)


def evaluate_spotting(records, subjects, decoder):
    total = empty_counts()
    for record in records:
        if str(record["subject"]) in subjects:
            add_counts(total, evaluate_decoding(record, decoder(record)))
    return total


def select_strong(records, train_subjects, grid, bank):
    ranked = []
    for index, config in enumerate(grid):
        total = empty_counts()
        for record_index, record in enumerate(records):
            if str(record["subject"]) in train_subjects:
                add_counts(total, evaluate_decoding(record, bank[index][record_index]))
        ranked.append(((-f1(total), total["FP"], total["rescued_candidate_count"], index),
                       index, config, metrics(total)))
    _, index, config, inner = min(ranked, key=lambda row: row[0])
    return index, config, inner


def select_rgr(records, train_subjects, strong_config):
    ranked = []
    configs = [{"delta_p": delta, "gamma_R": gamma}
               for delta, gamma in itertools.product(DELTAS, GAMMAS)]
    assert len(configs) == 20
    for index, config in enumerate(configs):
        total = evaluate_spotting(
            records, train_subjects,
            lambda record, c=config: decode_rgr(record, strong_config, c["delta_p"], c["gamma_R"]),
        )
        # Exact task tie-break: higher F1, smaller Delta-p, higher gamma-R.
        rank = (-f1(total), config["delta_p"], -config["gamma_R"])
        ranked.append((rank, index, config, metrics(total)))
    _, index, config, inner = min(ranked, key=lambda row: row[0])
    return index, config, inner


def aggregate_evaluation(records, subjects, decoder):
    total = empty_counts()
    gt_ids, pred_ids = [], []
    video_details = []
    for record in records:
        if str(record["subject"]) not in subjects:
            continue
        decoded = decoder(record)
        result = evaluate_decoding(record, decoded)
        add_counts(total, result)
        for event_row, match in zip(decoded["events"], result["matches"]):
            pred_emotion = int(MAJORITY_EMOTION(
                record["emotion"], event_row["onset"], event_row["offset"], event_row["peak"]
            ))
            video_details.append({"subject": str(record["subject"]), "video": str(record["video"]),
                                  **event_row, "matched_gt": int(match),
                                  "emotion_id": pred_emotion,
                                  "emotion": CLASS_ORDER[pred_emotion] if 0 <= pred_emotion < 5 else str(pred_emotion)})
            if match >= 0:
                gt_name = str(record["gt_emotions"][match])
                gt_id = {"negative": 0, "positive": 1, "surprise": 2, "others": 3}.get(gt_name, 3)
                gt_ids.append(gt_id)
                pred_ids.append(pred_emotion)
    recognition_4 = RECOGNITION_SUMMARY(gt_ids, pred_ids, [0, 1, 2, 3])
    filtered = [(gt, pred) for gt, pred in zip(gt_ids, pred_ids) if gt != 3]
    recognition_3 = RECOGNITION_SUMMARY([x[0] for x in filtered], [x[1] for x in filtered], [0, 1, 2])
    spot = metrics(total)
    rec_f1 = float(recognition_3["f1_score"])
    return total, {
        **spot, "Recognition_F1": rec_f1, "STRS": float(spot["F1"] * rec_f1),
        "recognition_f1_score_3emo_wo_others": rec_f1,
        "recognition_f1_score_4emo": float(recognition_4["f1_score"]),
        "recognition_num_matched": len(gt_ids),
    }, video_details


def candidate_diagnostics(record, decoded, fold, config):
    evaluated = evaluate_decoding(record, decoded)
    event_match = {(x["peak"], x["source"]): match
                   for x, match in zip(decoded["events"], evaluated["matches"])}
    rows = []
    for candidate in decoded["diagnostics"]:
        match = event_match.get((candidate["peak"], "rescue"), -1) if candidate["selected"] else -1
        emotion_id = int(MAJORITY_EMOTION(
            record["emotion"], candidate["onset"], candidate["offset"], candidate["peak"]
        ))
        rows.append({
            "outer_subject": fold, "subject": str(record["subject"]), "video": str(record["video"]),
            "onset": candidate["onset"], "peak": candidate["peak"], "offset": candidate["offset"],
            "emotion_id": emotion_id, "emotion": CLASS_ORDER[emotion_id],
            "confidence": candidate["R2_mean"], "recognition_feature": "R2_mean_max_non_neutral",
            "selected": candidate["selected"], "matched_gt": match,
            "rescued_TP": bool(candidate["selected"] and match >= 0),
            "rescued_FP": bool(candidate["selected"] and match < 0),
            "delta_p": config["delta_p"], "gamma_R": config["gamma_R"],
        })
    return rows


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_commit(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    args = parse_args()
    if args.output_root.exists() and (args.output_root / "report.json").exists() and not args.overwrite:
        raise RuntimeError("output already exists; use a new directory or --overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("dataset") != "SAMMLV" or int(payload.get("k_p", -1)) != K_P:
        raise RuntimeError("expected SAMMLV cache with k_p=5")
    records, recognition_validation = prepare_records(payload)
    all_subjects = sorted({str(record["subject"]) for record in records})
    held_subjects = list(args.held_subjects) if args.held_subjects else list(all_subjects)
    if any(subject not in all_subjects for subject in held_subjects):
        raise RuntimeError("unknown held subject")
    if args.mode == "smoke" and len(held_subjects) != 1:
        raise RuntimeError("smoke mode requires exactly one --held-subjects value")
    if args.mode == "full" and held_subjects != all_subjects:
        raise RuntimeError("full mode requires all subjects in canonical sorted order")

    grid = strong_grid()
    strong_bank = [[{"events": tuned_native_decode(record, native_config(config))["events"]}
                    for record in records] for config in grid]
    fold_rows, frequency_rows, rescued_rows, diagnostic_rows = [], [], [], []
    failed_folds, fold_payloads = [], []
    for held in held_subjects:
        try:
            train_subjects = set(all_subjects) - {held}
            assert held not in train_subjects
            strong_index, strong_config, strong_inner = select_strong(records, train_subjects, grid, strong_bank)
            rgr_index, rgr_config, rgr_inner = select_rgr(records, train_subjects, strong_config)

            decoders = {
                "Author Native": lambda record: {"events": author_native(record)["events"]},
                "Tuned Native": lambda record: {"events": tuned_native_decode(record, native_config(strong_config))["events"]},
                "Low-threshold Native": lambda record: decode_low(record, strong_config, rgr_config["delta_p"]),
                "RGR-1": lambda record: decode_rgr(record, strong_config, rgr_config["delta_p"], rgr_config["gamma_R"]),
            }
            fold_metrics = {}
            for method, decoder in decoders.items():
                _, method_metrics, _ = aggregate_evaluation(records, {held}, decoder)
                fold_metrics[method] = method_metrics

            # Trace only the selected primary RGR weak pool.
            preserved = total_strong = 0
            fold_diagnostics = []
            for record in records:
                if str(record["subject"]) != held:
                    continue
                decoded = decode_rgr(record, strong_config, rgr_config["delta_p"], rgr_config["gamma_R"], trace=True)
                preserved += decoded["native_preserved"]
                total_strong += decoded["native_total"]
                fold_diagnostics.extend(candidate_diagnostics(record, decoded, held, rgr_config))
            assert preserved == total_strong
            diagnostic_rows.extend(fold_diagnostics)
            rescued_rows.extend([row for row in fold_diagnostics if row["selected"]])

            row = {"subject": held, "native_config": json.dumps(strong_config, sort_keys=True),
                   "selected_delta_p": rgr_config["delta_p"], "selected_gamma_R": rgr_config["gamma_R"],
                   "inner_score": rgr_inner["F1"], "strong_inner_score": strong_inner["F1"],
                   "weak_count": fold_metrics["RGR-1"]["weak_candidate_count"],
                   "rescued_count": fold_metrics["RGR-1"]["rescued_candidate_count"],
                   "rescued_tp": fold_metrics["RGR-1"]["rescue_TP"],
                   "rescued_fp": fold_metrics["RGR-1"]["rescue_FP"]}
            for method, prefix in (("Author Native", "author"), ("Tuned Native", "tuned"),
                                   ("Low-threshold Native", "low"), ("RGR-1", "rgr")):
                value = fold_metrics[method]
                for key in ("TP", "FP", "FN", "F1"):
                    row[f"{prefix}_{key}"] = value[key]
            fold_rows.append(row)
            fold_payloads.append({"subject": held, "train_subjects": sorted(train_subjects),
                                  "selected_strong_config_index": strong_index,
                                  "selected_strong_config": strong_config,
                                  "selected_rgr_config_index": rgr_index,
                                  "selected_rgr_config": rgr_config,
                                  "strong_inner_metrics": strong_inner,
                                  "rgr_inner_metrics": rgr_inner,
                                  "outer_metrics": fold_metrics,
                                  "invariants": {"held_out_excluded": held not in train_subjects,
                                                 "strong_preserved": preserved == total_strong,
                                                 "strong_count": total_strong}})
        except Exception as exc:
            failed_folds.append({"subject": held, "error": repr(exc)})

    completed_subjects = {row["subject"] for row in fold_rows}
    aggregate = {}
    method_details = {}
    for method in ("Author Native", "Tuned Native", "Low-threshold Native", "RGR-1"):
        if not completed_subjects:
            continue
        if method == "Author Native":
            decoder_by_fold = lambda held: (lambda record: {"events": author_native(record)["events"]})
        else:
            selections = {item["subject"]: item for item in fold_payloads}
            def decoder_by_fold(held, method=method):
                item = selections[held]
                strong_config = item["selected_strong_config"]
                rgr_config = item["selected_rgr_config"]
                if method == "Tuned Native":
                    return lambda record: {"events": tuned_native_decode(record, native_config(strong_config))["events"]}
                if method == "Low-threshold Native":
                    return lambda record: decode_low(record, strong_config, rgr_config["delta_p"])
                return lambda record: decode_rgr(record, strong_config, rgr_config["delta_p"], rgr_config["gamma_R"])
        total = empty_counts()
        gt_ids, pred_ids = [], []
        for held in sorted(completed_subjects):
            counts, fold_metric, details = aggregate_evaluation(records, {held}, decoder_by_fold(held))
            add_counts(total, counts)
            # Recreate recognition lists from details and cache GT for exact pooled evaluation.
            record_lookup = {(str(r["subject"]), str(r["video"])): r for r in records}
            for detail in details:
                if detail["matched_gt"] >= 0:
                    record = record_lookup[(detail["subject"], detail["video"])]
                    gt_name = str(record["gt_emotions"][detail["matched_gt"]])
                    gt_ids.append({"negative": 0, "positive": 1, "surprise": 2, "others": 3}.get(gt_name, 3))
                    pred_ids.append(detail["emotion_id"])
        rec4 = RECOGNITION_SUMMARY(gt_ids, pred_ids, [0, 1, 2, 3])
        filtered = [(gt, pred) for gt, pred in zip(gt_ids, pred_ids) if gt != 3]
        rec3 = RECOGNITION_SUMMARY([x[0] for x in filtered], [x[1] for x in filtered], [0, 1, 2])
        spot = metrics(total)
        aggregate[method] = {**spot, "Recognition_F1": float(rec3["f1_score"]),
                             "STRS": float(spot["F1"] * rec3["f1_score"]),
                             "Recognition_F1_4emo": float(rec4["f1_score"]),
                             "recognition_num_matched": len(gt_ids)}
        method_details[method] = {"recognition_3emo_wo_others": rec3, "recognition_4emo": rec4}

    if args.mode == "full" and "Author Native" in aggregate:
        observed = (aggregate["Author Native"]["TP"], aggregate["Author Native"]["FP"], aggregate["Author Native"]["FN"])
        if observed != EXPECTED_NATIVE:
            failed_folds.append({"subject": "AGGREGATE", "error": f"Author baseline {observed} != {EXPECTED_NATIVE}"})

    for parameter, values in (("delta_p", DELTAS), ("gamma_R", GAMMAS)):
        counts = Counter(str(item["selected_rgr_config"][parameter]) for item in fold_payloads)
        for value in values:
            frequency_rows.append({"frequency_type": parameter, "delta_p": value if parameter == "delta_p" else "",
                                   "gamma_R": value if parameter == "gamma_R" else "", "count": counts[str(value)]})
    joint = Counter((str(item["selected_rgr_config"]["delta_p"]),
                     str(item["selected_rgr_config"]["gamma_R"])) for item in fold_payloads)
    for delta in DELTAS:
        for gamma in GAMMAS:
            frequency_rows.append({"frequency_type": "joint", "delta_p": delta,
                                   "gamma_R": gamma, "count": joint[(str(delta), str(gamma))]})

    incomplete = bool(failed_folds) or (args.mode == "full" and len(fold_rows) != len(all_subjects))
    low = aggregate.get("Low-threshold Native", {})
    tuned = aggregate.get("Tuned Native", {})
    rgr = aggregate.get("RGR-1", {})
    comparisons = {}
    if rgr and tuned and low:
        comparisons = {
            "RGR_Spot_F1_minus_Tuned": rgr["F1"] - tuned["F1"],
            "RGR_TP_minus_Tuned": rgr["TP"] - tuned["TP"],
            "RGR_FP_minus_Tuned": rgr["FP"] - tuned["FP"],
            "Low_TP_minus_Tuned": low["TP"] - tuned["TP"],
            "Low_FP_minus_Tuned": low["FP"] - tuned["FP"],
            "RGR_controls_added_FP_better_than_Low": (rgr["FP"] - tuned["FP"]) < (low["FP"] - tuned["FP"]),
        }
    report = {
        "experiment_name": EXPERIMENT, "algorithm_version": VERSION,
        "dataset": "SAMMLV", "backbone": "ME-TST+ frozen",
        "cache_input_paths": [str(args.cache.resolve())], "cache_sha256": sha256(args.cache),
        "git_commit": git_commit(REPRO_ROOT), "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "outer_protocol": "subject LOSO; held-out subject excluded",
        "inner_protocol": "pooled train-subject Spotting F1; RGR tie: smaller Delta_p then higher gamma_R",
        "delta_p_grid": list(DELTAS), "gamma_R_grid": list(GAMMAS),
        "gamma_R_grid_source": "reused_from_existing_previous_RGR_experiment",
        "recognition_feature": "R2_mean_max_non_neutral",
        "recognition_metric_primary": "3emo_wo_others paper-aligned recognition F1",
        "strong_decoder_definition": "existing fold-selected Tuned Native",
        "weak_decoder_definition": "same native decoder with p_w=p_s-Delta_p; peak identity difference",
        "evaluation_protocol": "existing validated evaluate_decoding IoU=0.5 plus existing paper_metrics recognition helpers",
        "evaluation_source": str(PAPER_METRICS_PATH.resolve()),
        "subject_count": len(all_subjects), "video_count": len(records), "GT_count": int(payload["num_gt"]),
        "recognition_validation": recognition_validation,
        "requested_folds": held_subjects, "completed_fold_count": len(fold_rows),
        "errors": failed_folds, "incomplete": incomplete,
        "aggregate_metrics": aggregate, "recognition_details": method_details,
        "RGR_diagnostics": {"total_weak_candidates": len(diagnostic_rows),
                            "rescued_candidate_count": len(rescued_rows),
                            "rescued_TP": sum(bool(row["rescued_TP"]) for row in rescued_rows),
                            "rescued_FP": sum(bool(row["rescued_FP"]) for row in rescued_rows),
                            "rescue_precision": (sum(bool(row["rescued_TP"]) for row in rescued_rows) / len(rescued_rows)
                                                 if rescued_rows else 0.0)},
        "Low_threshold_diagnostics": ({"additional_candidates_vs_tuned": low.get("event_count", 0)-tuned.get("event_count", 0),
                                       "additional_TP_vs_tuned": low.get("TP", 0)-tuned.get("TP", 0),
                                       "additional_FP_vs_tuned": low.get("FP", 0)-tuned.get("FP", 0)}
                                      if low and tuned else {}),
        "comparisons": comparisons, "outer_folds": fold_payloads,
        "invariants": {"backbone_frozen": True, "same_frozen_cache_for_all_methods": True,
                       "held_out_excluded_all_completed_folds": all(x["invariants"]["held_out_excluded"] for x in fold_payloads),
                       "strong_candidate_preservation_all_completed_folds": all(x["invariants"]["strong_preserved"] for x in fold_payloads),
                       "all_folds_completed": args.mode == "full" and len(fold_rows) == len(all_subjects) and not failed_folds,
                       "same_predefined_grid_all_folds": True,
                       "all_metrics_finite": all(np.isfinite(value) for method in aggregate.values()
                                                 for value in method.values() if isinstance(value, (int, float)))},
    }
    summary_rows = [{"Method": method, "TP": value["TP"], "FP": value["FP"], "FN": value["FN"],
                     "Precision": value["Precision"], "Recall": value["Recall"],
                     "Spot_F1": value["F1"], "Rec_F1": value["Recognition_F1"], "STRS": value["STRS"]}
                    for method, value in aggregate.items()]
    write_csv(args.output_root / "summary.csv", summary_rows,
              ["Method", "TP", "FP", "FN", "Precision", "Recall", "Spot_F1", "Rec_F1", "STRS"])
    write_csv(args.output_root / "outer_fold_metrics.csv", fold_rows)
    write_csv(args.output_root / "selected_config_frequency.csv", frequency_rows,
              ["frequency_type", "delta_p", "gamma_R", "count"])
    write_csv(args.output_root / "rescued_candidates.csv", rescued_rows)
    write_csv(args.output_root / "weak_candidate_diagnostics.csv", diagnostic_rows)
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "incomplete": incomplete,
                      "completed_folds": len(fold_rows), "errors": failed_folds,
                      "aggregate_metrics": aggregate, "comparisons": comparisons,
                      "RGR_diagnostics": report["RGR_diagnostics"],
                      "invariants": report["invariants"]}, indent=2, ensure_ascii=False))
    return 1 if incomplete and args.mode == "full" else 0


if __name__ == "__main__":
    raise SystemExit(main())
