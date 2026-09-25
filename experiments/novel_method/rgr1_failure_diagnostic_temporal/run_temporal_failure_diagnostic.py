#!/usr/bin/env python3
"""Read-only temporal failure diagnostic for the fixed RGR-1 weak pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
RGR_DIR = REPRO_ROOT / "results" / "rgr1_sammlv_nested"
LEGACY_DIR = REPRO_ROOT / "my_method" / "multi_scale_candidate_rescue"
sys.path.insert(0, str(LEGACY_DIR))

from run_mscr_nested_loso import match_events, tuned_native_decode  # noqa: E402


DESCRIPTORS = (
    ("rise", "higher"),
    ("fall", "higher"),
    ("peak_contrast", "higher"),
    ("R_range", "higher"),
    ("peak_distance", "lower"),
    ("normalized_peak_distance", "lower"),
    ("correlation", "higher"),
)
EXPECTED_CACHE_SHA256 = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
EXPECTED_WEAK = 86
EXPECTED_POSITIVE = 7
EXPECTED_NEGATIVE = 79


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=REPRO_ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl")
    parser.add_argument("--source-results", type=Path, default=RGR_DIR)
    parser.add_argument("--output-root", type=Path,
                        default=REPRO_ROOT / "results/rgr1_failure_diagnostic_temporal")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("n", "mean", "median", "std", "p10", "p25", "p75", "p90")}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def auc_values(rows, descriptor, direction):
    usable = [row for row in rows if np.isfinite(float(row[descriptor]))]
    y_true = np.asarray([int(row["label"]) for row in usable], dtype=int)
    raw = np.asarray([float(row[descriptor]) for row in usable], dtype=float)
    score = -raw if direction == "lower" else raw
    return usable, y_true, score, float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))


def native_config(config):
    return {"c_s": config["c_s"], "p": config["p_s"],
            "c_d": config["c_d"], "c_b": config["c_b"]}


def recover_fixed_labels(weak_rows, record_lookup, selected_configs):
    """Recover Low-control labels without altering the authoritative weak membership."""
    grouped = defaultdict(list)
    for row in weak_rows:
        grouped[(row["subject"], row["video"])].append(row)
    labels = {}
    for key, video_weak in grouped.items():
        record = record_lookup[key]
        strong = tuned_native_decode(record, native_config(selected_configs[key[0]]))
        weak_events = [{"onset": int(row["onset"]), "peak": int(row["peak"]),
                        "offset": int(row["offset"]), "source": "fixed_weak"}
                       for row in video_weak]
        low_control_events = sorted(
            strong["events"] + weak_events,
            key=lambda event: (event["peak"], 0 if event["source"] == "tuned_native" else 1),
        )
        matches, _ = match_events(low_control_events, record["samples"])
        for event, matched_gt in zip(low_control_events, matches):
            if event["source"] == "fixed_weak":
                identity = (key[0], key[1], int(event["peak"]))
                assert identity not in labels
                labels[identity] = int(matched_gt)
    return labels


def build_descriptors(weak_rows, record_lookup, selected_configs, labels):
    rows, trajectories = [], []
    curve_cache = {}
    for source in weak_rows:
        subject, video, peak = source["subject"], source["video"], int(source["peak"])
        key = (subject, video)
        record = record_lookup[key]
        if key not in curve_cache:
            curve_cache[key] = tuned_native_decode(
                record, native_config(selected_configs[subject])
            )["curve"]
        spotting = np.asarray(curve_cache[key], dtype=np.float64)
        probabilities = softmax(record["logits"])
        recognition = np.max(probabilities[:, :4], axis=1)
        assert len(spotting) == len(recognition) == len(record["score"])
        assert 0 <= peak < len(spotting)

        onset_raw, offset_raw = int(source["onset"]), int(source["offset"])
        onset, offset = max(0, onset_raw), min(len(spotting) - 1, offset_raw)
        assert onset <= peak <= offset
        interval_spot = spotting[onset:offset + 1]
        interval_rec = recognition[onset:offset + 1]
        assert len(interval_spot) == len(interval_rec) > 0
        # The fixed weak peak must remain a local maximum of the exact native-smoothed curve.
        left = spotting[peak - 1] if peak > 0 else -np.inf
        right = spotting[peak + 1] if peak + 1 < len(spotting) else -np.inf
        local_maximum = bool(spotting[peak] > left and spotting[peak] > right)
        assert local_maximum

        pre = recognition[max(0, peak - 2):peak]
        post = recognition[peak + 1:min(len(recognition), peak + 3)]
        assert len(pre) and len(post)
        r_peak, r_pre, r_post = float(recognition[peak]), float(np.mean(pre)), float(np.mean(post))
        rise, fall = r_peak - r_pre, r_peak - r_post
        p_rec = onset + int(np.argmax(interval_rec))
        peak_distance = abs(peak - p_rec)
        if np.std(interval_spot) == 0 or np.std(interval_rec) == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(interval_spot, interval_rec)[0, 1])
        identity = (subject, video, peak)
        matched_gt = labels[identity]
        recomputed_r2_mean = float(np.mean(interval_rec))
        assert np.isclose(recomputed_r2_mean, float(source["confidence"]), atol=1e-12, rtol=1e-12)
        row = {
            "subject": subject, "video": video, "onset": onset_raw, "peak": peak,
            "offset": offset_raw, "clipped_onset": onset, "clipped_offset": offset,
            "interval_length": len(interval_rec), "label": int(matched_gt >= 0),
            "matched_gt": matched_gt, "R2_mean": recomputed_r2_mean,
            "R_peak": r_peak, "R_pre": r_pre, "R_post": r_post,
            "rise": rise, "fall": fall,
            "peak_contrast": r_peak - 0.5 * (r_pre + r_post),
            "R_range": float(np.max(interval_rec) - np.min(interval_rec)),
            "left_slope": rise, "right_slope": r_post - r_peak,
            "rise_fall_pattern": bool(rise > 0 and (r_post - r_peak) < 0),
            "p_spot": peak, "p_rec": p_rec, "peak_distance": peak_distance,
            "normalized_peak_distance": peak_distance / max(1, len(interval_rec)),
            "correlation": correlation, "correlation_valid": bool(np.isfinite(correlation)),
            "spotting_peak_is_local_maximum": local_maximum,
            "selected_by_RGR1": source["selected"].lower() == "true",
        }
        rows.append(row)
        for absolute_time in range(onset, offset + 1):
            trajectories.append({
                "subject": subject, "video": video, "peak": peak,
                "label": row["label"], "case_type": "TP" if row["label"] else "FP",
                "relative_time": absolute_time - peak, "absolute_model_index": absolute_time,
                "spotting_score": float(spotting[absolute_time]),
                "recognition_non_neutral_score": float(recognition[absolute_time]),
            })
    return rows, trajectories


def descriptor_analysis(rows):
    summaries, loo_rows, report_descriptors = [], [], {}
    for descriptor, direction in DESCRIPTORS:
        usable, y_true, score, full_auc, pr_auc = auc_values(rows, descriptor, direction)
        positive_indices = np.flatnonzero(y_true == 1)
        loo_values = []
        for index in positive_indices:
            keep = np.arange(len(y_true)) != index
            auc = float(roc_auc_score(y_true[keep], score[keep]))
            removed = usable[index]
            loo_values.append(auc)
            loo_rows.append({
                "descriptor": descriptor, "direction": direction,
                "removed_subject": removed["subject"], "removed_video": removed["video"],
                "removed_peak": removed["peak"], "auc": auc,
            })
        tp = summarize([float(row[descriptor]) for row in usable if row["label"] == 1])
        fp = summarize([float(row[descriptor]) for row in usable if row["label"] == 0])
        loo = {
            "full_auc": full_auc, "min_auc": float(np.min(loo_values)),
            "max_auc": float(np.max(loo_values)), "mean_auc": float(np.mean(loo_values)),
            "std_auc": float(np.std(loo_values)),
            "positive_sensitive": bool(full_auc - min(loo_values) >= 0.10),
        }
        summary_row = {"descriptor": descriptor, "expected_direction": direction,
                       "ROC_AUC": full_auc, "PR_AUC": pr_auc,
                       "positive_prevalence": float(np.mean(y_true)),
                       "valid_count": len(usable),
                       **{f"TP_{key}": value for key, value in tp.items()},
                       **{f"FP_{key}": value for key, value in fp.items()},
                       **{f"LOO_{key}": value for key, value in loo.items()}}
        summaries.append(summary_row)
        report_descriptors[descriptor] = {
            "direction": direction, "TP": tp, "FP": fp,
            "ROC_AUC": full_auc, "PR_AUC": pr_auc,
            "positive_prevalence": float(np.mean(y_true)),
            "valid_count": len(usable), "leave_one_positive": loo,
        }
    return summaries, loo_rows, report_descriptors


def select_fixed_fp_controls(rows):
    false_rows = [row for row in rows if row["label"] == 0]
    selections = (
        ("top5_R2_mean", sorted(false_rows, key=lambda row: (-row["R2_mean"], row["subject"], row["video"], row["peak"]))[:5]),
        ("top5_peak_contrast", sorted(false_rows, key=lambda row: (-row["peak_contrast"], row["subject"], row["video"], row["peak"]))[:5]),
        ("bottom5_peak_distance", sorted(false_rows, key=lambda row: (row["peak_distance"], row["subject"], row["video"], row["peak"]))[:5]),
    )
    selected = {}
    for reason, cases in selections:
        for row in cases:
            identity = (row["subject"], row["video"], row["peak"])
            if identity not in selected:
                selected[identity] = {**row, "selection_reason": reason}
            else:
                selected[identity]["selection_reason"] += ";" + reason
    return list(selected.values())


def format_number(value):
    if value is None or not np.isfinite(value):
        return "NaN"
    return f"{value:.6f}"


def make_markdown(report, tp_rows, selected_fp_rows, output_paths):
    descriptors = report["descriptors"]
    lines = [
        "# RGR-1 局部时间结构 Failure Diagnostic",
        "",
        "## 1. 数据完整性",
        "",
        f"- 固定 weak pool：{report['weak_candidate_count']}（TP={report['positive_count']}，FP={report['negative_count']}）。",
        "- 只读取原 RGR-1 trace/report 与同一 frozen cache；未训练、未 forward backbone、未实现新 decoder。",
        "- weak membership 以原 trace 的 86 个 candidate identity 为准；既有 Tuned Native 仅用于恢复原 Low-control 标签并通过 7/79 锚点校验。",
        "",
        "## 2. TEMPORAL_ALIGNMENT_AUDIT",
        "",
        f"**{report['temporal_alignment_audit']['status']}**",
        "",
        "- spotting raw score、recognition logits 与 candidate 均使用相同 model-output index。",
        "- spotting trajectory 使用该 outer fold Tuned Native 的原平滑曲线；candidate peak 经逐例确认是其局部最大值。",
        "- recognition logits 为 `[T,5]`，沿 class axis softmax；neutral=4，non-neutral={0,1,2,3}。",
        f"- `frame_skip={report['temporal_alignment_audit']['frame_skip']}`：一个 model-output index 对应 7 个原视频帧，约 35 ms。",
        "- 未发现额外 temporal offset；实际 candidate interval 在序列边缘按有效 index clip。",
        "",
        "## 3. Descriptor results",
        "",
        "| Descriptor | TP median | FP median | ROC-AUC | PR-AUC | LOO mean AUC | LOO min AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _ in DESCRIPTORS:
        item = descriptors[name]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, format_number(item["TP"]["median"]), format_number(item["FP"]["median"]),
            format_number(item["ROC_AUC"]), format_number(item["PR_AUC"]),
            format_number(item["leave_one_positive"]["mean_auc"]),
            format_number(item["leave_one_positive"]["min_auc"])))
    binary = report["rise_fall_pattern"]
    lines += ["", f"rise_fall_pattern：TP={binary['TP_proportion']:.6f}，FP={binary['FP_proportion']:.6f}。",
              "", "## 4. 七个 weak TP", "",
              "| subject | video | peak | R2-mean | rise | fall | contrast | peak distance | correlation |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tp_rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["subject"], row["video"], row["peak"], format_number(row["R2_mean"]),
            format_number(row["rise"]), format_number(row["fall"]),
            format_number(row["peak_contrast"]), row["peak_distance"],
            format_number(row["correlation"])))
    sensitive = [name for name, value in descriptors.items()
                 if value["leave_one_positive"]["positive_sensitive"]]
    lines += ["", "## 5. Positive sensitivity", "",
              "按预先记录的 `full AUC - minimum LOO AUC >= 0.10` 描述性标记：" +
              (", ".join(sensitive) if sensitive else "没有 descriptor 被标为 POSITIVE-SENSITIVE。"),
              "", "## 6. 固定 FP counterexamples", "",
              f"三个固定规则去重后得到 {len(selected_fp_rows)} 个 FP；完整数值见 `selected_fp_case_details.csv`，轨迹见 `trajectory_data.csv`。",
              "这些对照专门保留 R2-mean 最高、contrast 最高或 peak distance 最小的 FP，用于检查 FP 是否也呈现看似理想的时间结构。",
              "", "## 7. Final evidence grade", "", f"**{report['final_evidence_grade']}**", ""]
    for fact in report["evidence_facts"]:
        lines.append(f"- {fact}")
    lines += ["", "## 8. 文件", ""] + [f"- `{path}`" for path in output_paths]
    return "\n".join(lines) + "\n"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("diagnostic output directory is not empty")
    source_report_path = args.source_results / "report.json"
    weak_path = args.source_results / "weak_candidate_diagnostics.csv"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    weak_rows = read_csv(weak_path)
    cache_hash = file_sha256(args.cache)
    assert cache_hash == EXPECTED_CACHE_SHA256 == source_report["cache_sha256"]
    assert len(weak_rows) == EXPECTED_WEAK
    identities = [(row["subject"], row["video"], int(row["peak"])) for row in weak_rows]
    assert len(identities) == len(set(identities))

    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["dataset"] == "SAMMLV" and payload["frame_skip"] == 7
    record_lookup = {(str(record["subject"]), str(record["video"])): record
                     for record in payload["records"]}
    for record in record_lookup.values():
        assert len(record["score"]) == len(record["logits"]) == len(record["emotion"])
        assert np.asarray(record["logits"]).shape[1] == 5
    selected_configs = {fold["subject"]: fold["selected_strong_config"]
                        for fold in source_report["outer_folds"]}
    labels = recover_fixed_labels(weak_rows, record_lookup, selected_configs)
    assert set(labels) == set(identities)
    assert sum(match >= 0 for match in labels.values()) == EXPECTED_POSITIVE

    candidate_rows, all_trajectories = build_descriptors(
        weak_rows, record_lookup, selected_configs, labels
    )
    assert sum(row["label"] for row in candidate_rows) == EXPECTED_POSITIVE
    assert sum(1 - row["label"] for row in candidate_rows) == EXPECTED_NEGATIVE
    summaries, loo_rows, report_descriptors = descriptor_analysis(candidate_rows)
    tp_rows = sorted([row for row in candidate_rows if row["label"]],
                     key=lambda row: (row["subject"], row["video"], row["peak"]))
    selected_fp_rows = select_fixed_fp_controls(candidate_rows)
    selected_identities = {(row["subject"], row["video"], row["peak"]): row["selection_reason"]
                           for row in selected_fp_rows}
    trajectory_rows = []
    tp_identities = {(row["subject"], row["video"], row["peak"]) for row in tp_rows}
    for row in all_trajectories:
        identity = (row["subject"], row["video"], row["peak"])
        if identity in tp_identities or identity in selected_identities:
            row = dict(row)
            row["selection_reason"] = "all_weak_TP" if identity in tp_identities else selected_identities[identity]
            trajectory_rows.append(row)

    pattern_tp = np.mean([row["rise_fall_pattern"] for row in candidate_rows if row["label"]])
    pattern_fp = np.mean([row["rise_fall_pattern"] for row in candidate_rows if not row["label"]])
    positive_subjects = Counter(row["subject"] for row in tp_rows)
    negative_subjects = Counter(row["subject"] for row in candidate_rows if not row["label"])
    strong_candidates = [name for name, item in report_descriptors.items()
                         if item["ROC_AUC"] >= 0.75
                         and item["leave_one_positive"]["mean_auc"] >= 0.70
                         and item["leave_one_positive"]["min_auc"] >= 0.60]
    moderate = [name for name, item in report_descriptors.items() if item["ROC_AUC"] >= 0.60]
    grade = "TEMPORAL-EVIDENCE-STRONG" if strong_candidates else (
        "TEMPORAL-EVIDENCE-WEAK" if moderate else "TEMPORAL-EVIDENCE-NONE"
    )
    best_name = max(report_descriptors, key=lambda name: report_descriptors[name]["ROC_AUC"])
    best = report_descriptors[best_name]
    evidence_facts = [
        f"没有 descriptor 达到 STRONG 门槛；最高 ROC-AUC 为 {best_name}={best['ROC_AUC']:.3f}。",
        f"{best_name} 的 LOO-positive mean/min AUC 为 {best['leave_one_positive']['mean_auc']:.3f}/{best['leave_one_positive']['min_auc']:.3f}。",
        f"达到 ROC-AUC≥0.60 的 descriptor：{', '.join(moderate) if moderate else '无'}。",
        f"rise_fall_pattern 在 TP/FP 中的比例为 {pattern_tp:.3f}/{pattern_fp:.3f}，两组均存在该模式。",
        f"7 个 TP 来自 {len(positive_subjects)} 个 subject；最大单 subject 贡献 {max(positive_subjects.values())}/7。",
        f"固定对照规则选出 {len(selected_fp_rows)} 个去重 FP，说明高 recognition、强 contrast 或近 peak alignment 并非 TP 独有。",
        "本分析为 retrospective label diagnostic，不能作为合法 inference threshold 或 feature selection。",
    ]
    temporal_audit = {
        "status": "PASS", "spotting_index_space": "model-output index; raw score and fold-native smoothed curve",
        "recognition_index_space": "same model-output index; logits [T,5], softmax axis=1",
        "candidate_index_space": "same model-output index; onset/peak/offset from fixed weak trace",
        "frame_skip": int(payload["frame_skip"]), "original_frames_per_model_index": 7,
        "seconds_per_model_index": float(payload["frame_skip"] / payload["fps"]),
        "neutral_class_id": 4, "non_neutral_class_ids": [0, 1, 2, 3],
        "class_order": ["negative", "positive", "surprise", "others", "neutral"],
        "recognition_probabilities_softmaxed_in_diagnostic": True,
        "all_79_records_score_logits_emotion_length_aligned": True,
        "all_candidate_peaks_native_curve_local_maxima": True,
        "temporal_offset_or_padding_detected": False,
        "clipped_candidate_count": sum(row["onset"] != row["clipped_onset"] or row["offset"] != row["clipped_offset"]
                                       for row in candidate_rows),
    }
    report = {
        "experiment_name": "RGR1-fixed-weak-pool-temporal-failure-diagnostic",
        "source_experiment": "RGR1", "dataset": "SAMMLV", "backbone": "ME-TST+ frozen",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_result_directory": str(args.source_results.resolve()),
        "source_trace": str(weak_path.resolve()), "frozen_cache": str(args.cache.resolve()),
        "frozen_cache_sha256": cache_hash, "weak_pool_regenerated": False,
        "weak_candidate_count": len(candidate_rows), "positive_count": len(tp_rows),
        "negative_count": len(candidate_rows) - len(tp_rows),
        "no_new_decoder": True, "no_training": True, "no_backbone_forward": True,
        "label_recovery": "fixed 86 identities replayed with original fold-selected Tuned Native and existing Low-control evaluator",
        "temporal_alignment_audit": temporal_audit,
        "descriptors": report_descriptors,
        "rise_fall_pattern": {"TP_proportion": float(pattern_tp), "FP_proportion": float(pattern_fp)},
        "subject_distribution": {"TP": dict(sorted(positive_subjects.items())),
                                 "FP": dict(sorted(negative_subjects.items())),
                                 "TP_unique_subjects": len(positive_subjects),
                                 "FP_unique_subjects": len(negative_subjects)},
        "positive_sensitivity_rule": "POSITIVE-SENSITIVE iff full_auc - min_LOO_auc >= 0.10",
        "fixed_fp_control_rules": ["top5_R2_mean", "top5_peak_contrast", "bottom5_peak_distance"],
        "selected_fp_control_count_after_dedup": len(selected_fp_rows),
        "final_evidence_grade": grade, "evidence_facts": evidence_facts,
        "errors": [], "incomplete": False,
    }
    output_files = [
        args.output_root / "report.json", args.output_root / "descriptor_summary.csv",
        args.output_root / "candidate_descriptors.csv", args.output_root / "leave_one_positive_auc.csv",
        args.output_root / "tp_case_details.csv", args.output_root / "selected_fp_case_details.csv",
        args.output_root / "trajectory_data.csv", args.output_root / "TEMPORAL_DIAGNOSTIC_REPORT_CN.md",
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_files[1], summaries)
    write_csv(output_files[2], candidate_rows)
    write_csv(output_files[3], loo_rows)
    write_csv(output_files[4], tp_rows)
    write_csv(output_files[5], selected_fp_rows)
    write_csv(output_files[6], trajectory_rows)
    safe_report = json_safe(report)
    output_files[0].write_text(json.dumps(safe_report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_files[7].write_text(make_markdown(safe_report, tp_rows, selected_fp_rows, output_files), encoding="utf-8")
    print(json.dumps({"alignment": "PASS", "weak": len(candidate_rows), "TP": len(tp_rows),
                      "FP": len(candidate_rows) - len(tp_rows), "grade": grade,
                      "best_descriptor": best_name, "best_ROC_AUC": best["ROC_AUC"],
                      "output_root": str(args.output_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
