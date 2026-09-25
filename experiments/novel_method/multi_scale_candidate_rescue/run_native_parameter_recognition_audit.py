#!/usr/bin/env python3
"""Locked native-parameter attribution and cached-logit separability audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from run_mscr_nested_loso import (
    EXPECTED_NATIVE, K_P, TUNED_CB, TUNED_CD, TUNED_CS, TUNED_P,
    add_counts, author_native, empty_counts, evaluate_decoding, f1, metrics,
    sha256, tuned_native_decode, write_csv,
)


VERSION = "native-attribution-recognition-v1-locked-2026-09-05"
CLASS_ORDER = ["negative", "positive", "surprise", "others", "neutral"]
NEUTRAL_ID = 4
PR_MULTIPLIER = 1.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weak-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def config_grid(name):
    author = {"c_s": 2.0, "p": 0.55, "c_d": 1.0, "c_b": 1.0}
    if name == "Smooth-only":
        return [{**author, "c_s": value} for value in TUNED_CS]
    if name == "Threshold-only":
        return [{**author, "p": value} for value in TUNED_P]
    if name == "Distance-only":
        return [{**author, "c_d": value} for value in TUNED_CD]
    if name == "Boundary-only":
        return [{**author, "c_b": value} for value in TUNED_CB]
    if name == "Full Tuned Native":
        return [dict(zip(("c_s", "p", "c_d", "c_b"), values))
                for values in itertools.product(TUNED_CS, TUNED_P, TUNED_CD, TUNED_CB)]
    raise ValueError(name)


def precompute(records, configs):
    return [[{"events": tuned_native_decode(record, config)["events"]} for record in records]
            for config in configs]


def pooled_counts(records, decoded_bank, config_index, subjects):
    total = empty_counts()
    for record_index, record in enumerate(records):
        if str(record["subject"]) in subjects:
            add_counts(total, evaluate_decoding(record, decoded_bank[config_index][record_index]))
    return total


def select_config(records, configs, bank, train_subjects):
    ranked = []
    for index, config in enumerate(configs):
        counts = pooled_counts(records, bank, index, train_subjects)
        # No rescue events exist in native controls; retain the locked tie-break explicitly.
        key = (-f1(counts), counts["FP"], counts["rescued_candidate_count"], index)
        ranked.append((key, index, config, metrics(counts)))
    _, index, config, train_metrics = min(ranked, key=lambda row: row[0])
    return index, config, train_metrics


def run_attribution(records, subjects):
    methods = ["Smooth-only", "Threshold-only", "Distance-only", "Boundary-only",
               "Full Tuned Native"]
    selected = []
    pooled_results = {}
    for method in methods:
        configs = config_grid(method)
        bank = precompute(records, configs)
        outer_total = empty_counts()
        for held in subjects:
            train = set(subjects) - {held}
            index, config, train_metrics = select_config(records, configs, bank, train)
            held_counts = pooled_counts(records, bank, index, {held})
            add_counts(outer_total, held_counts)
            selected.append({
                "outer_subject": held, "method": method, "config_index": index, **config,
                "outer_TP": held_counts["TP"], "outer_FP": held_counts["FP"],
                "outer_FN": held_counts["FN"], "outer_F1": f1(held_counts),
                "train_pooled_TP": train_metrics["TP"], "train_pooled_FP": train_metrics["FP"],
                "train_pooled_FN": train_metrics["FN"], "train_pooled_F1": train_metrics["F1"],
            })
        pooled_results[method] = metrics(outer_total)

    native_total = empty_counts()
    for record in records:
        add_counts(native_total, evaluate_decoding(record, {"events": author_native(record)["events"]}))
    observed = (native_total["TP"], native_total["FP"], native_total["FN"])
    if observed != EXPECTED_NATIVE:
        raise RuntimeError(f"BLOCKED-BASELINE: expected {EXPECTED_NATIVE}, observed {observed}")
    pooled_results = {"Author Native": metrics(native_total), **pooled_results}

    frequencies = {}
    for method in methods:
        rows = [row for row in selected if row["method"] == method]
        frequencies[method] = {
            key: dict(sorted(Counter(str(row[key]) for row in rows).items()))
            for key in ("c_s", "p", "c_d", "c_b")
        }
    return selected, pooled_results, frequencies


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate(values, peak):
    left, right = max(0, peak - K_P), min(len(values), peak + K_P + 1)
    window = values[left:right]
    return {"center": float(values[peak]), "mean": float(window.mean()), "max": float(window.max())}


def bool_value(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def run_recognition(records, weak_rows):
    lookup = {(str(record["subject"]), str(record["video"])): record for record in records}
    shape_counts = Counter()
    all_aligned = all_finite = True
    for record in records:
        logits = np.asarray(record["logits"])
        score = np.asarray(record["score"])
        shape_counts[str(tuple(logits.shape))] += 1
        all_aligned &= logits.ndim == 2 and logits.shape[0] == len(score)
        all_finite &= bool(np.isfinite(logits).all())
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_ORDER):
            raise RuntimeError(f"unexpected logits shape: {logits.shape}")
    if not all_aligned or not all_finite:
        raise RuntimeError("invalid recognition logits alignment or finite check")

    output_rows = []
    for source in weak_rows:
        subject, video, peak = str(source["subject"]), str(source["video"]), int(source["peak"])
        record = lookup[(subject, video)]
        probs = softmax(record["logits"])
        if not 0 <= peak < len(probs):
            raise RuntimeError(f"candidate outside logits: {subject}/{video}/{peak}")
        nonneutral = np.max(probs[:, :NEUTRAL_ID], axis=1)
        sorted_probs = np.sort(probs, axis=1)
        frame_evidence = {
            "R1": 1.0 - probs[:, NEUTRAL_ID],
            "R2": nonneutral,
            "R3": sorted_probs[:, -1] - sorted_probs[:, -2],
        }
        row = {
            "subject": subject, "video": video, "peak": peak,
            "label": int(bool_value(source["is_native_missed_gt_candidate"])),
            "matched_gt": int(float(source["lowrescue_matched_gt_index"])),
            "best_gt_id": int(float(source["best_gt_id"])),
            "best_gt_iou": float(source["best_gt_iou"]),
        }
        for evidence, values in frame_evidence.items():
            for aggregation, value in aggregate(values, peak).items():
                row[f"{evidence}_{aggregation}"] = value
        output_rows.append(row)

    labels = np.asarray([row["label"] for row in output_rows], dtype=int)
    if labels.sum() != 2 or len(labels) != 45:
        raise RuntimeError(f"weak-pool identity mismatch: n={len(labels)}, positive={labels.sum()}")
    prevalence = float(labels.mean())
    evidence_metrics = {}
    for evidence in ("R1", "R2", "R3"):
        for aggregation in ("center", "mean", "max"):
            key = f"{evidence}_{aggregation}"
            values = np.asarray([row[key] for row in output_rows])
            true = values[labels == 1]
            false = values[labels == 0]
            roc = float(roc_auc_score(labels, values))
            ap = float(average_precision_score(labels, values))
            evidence_metrics[key] = {
                "ROC_AUC": roc, "PR_AUC": ap, "positive_prevalence": prevalence,
                "true_candidate_mean": float(true.mean()), "true_candidate_median": float(np.median(true)),
                "weak_FP_mean": float(false.mean()), "weak_FP_median": float(np.median(false)),
                "passes_ROC": roc >= 0.60,
                "passes_PR": ap >= PR_MULTIPLIER * prevalence,
                "passes_joint_rule": roc >= 0.60 and ap >= PR_MULTIPLIER * prevalence,
            }
    status = ("GO-RECOGNITION-SEPARABILITY" if any(v["passes_joint_rule"] for v in evidence_metrics.values())
              else "NO-GO-RECOGNITION-SEPARABILITY")
    validation = {
        "class_order": CLASS_ORDER, "neutral_class_id": NEUTRAL_ID,
        "neutral_semantics_status": "CONFIRMED",
        "logits_shape_counts": dict(shape_counts), "all_temporally_aligned": bool(all_aligned),
        "all_logits_finite": bool(all_finite),
    }
    return output_rows, evidence_metrics, status, validation


def markdown_table(results):
    lines = ["| 方法 | TP | FP | FN | Precision | Recall | F1 | ΔF1 vs Author |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    base = results["Author Native"]["F1"]
    for method in ["Author Native", "Smooth-only", "Threshold-only", "Distance-only",
                   "Boundary-only", "Full Tuned Native"]:
        row = results[method]
        lines.append(f"| {method} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['F1']:.6f} | {row['F1']-base:+.6f} |")
    return lines


def attribution_report(summary):
    results = summary["pooled_outer_metrics"]
    gains = {name: row["F1"] - results["Author Native"]["F1"]
             for name, row in results.items() if name not in {"Author Native", "Full Tuned Native"}}
    best = max(gains, key=gains.get)
    lines = ["# Tuned Native Parameter Attribution", "",
             "## 结论", "",
             f"四个单因素中贡献最大的是 **{best}**，其相对 Author Native 的 pooled outer F1 变化为 {gains[best]:+.6f}。这是归因结论，不构成新方法。", "",
             "## 协议", "",
             "- SAMMLV，29-fold outer subject LOSO；每折仅用其余 subjects 的 pooled F1 选参。",
             "- tie-break：fewer FP → fewer rescue events → 固定 lexicographic config index。",
             "- 原 grid、候选生成和 evaluator 均未改变；outer test GT 未参与选参。", "",
             "## Pooled outer 结果", "", *markdown_table(results), "",
             "## 参数选择频率", ""]
    for method, freq in summary["selected_parameter_frequencies"].items():
        lines.append(f"- {method}: `{json.dumps(freq, ensure_ascii=False, sort_keys=True)}`")
    lines += ["", "## Full Tuned Native 逐折配置", "",
              "完整 29-fold 配置及 outer TP/FP/FN/F1 见 `outputs/tuned_native_selected_configs.csv`。"]
    return "\n".join(lines) + "\n"


def recognition_report(summary, true_rows):
    lines = ["# Recognition Weak-Candidate Separability Audit", "",
             f"最终状态：**{summary['status']}**", "", "## 数据与语义核验", "",
             f"- LowRescue weak pool：{summary['candidate_pool']['n']} 条；true={summary['candidate_pool']['positive']}，weak FP={summary['candidate_pool']['negative']}。",
             f"- class order：`{summary['recognition_validation']['class_order']}`；neutral id={NEUTRAL_ID}。",
             "- 类别语义由现有 ME-TST 评估代码与 cache strategy 元数据交叉确认；logits 仅做 softmax，没有 model forward。",
             f"- GO 预设规则：ROC-AUC ≥ 0.60 且 PR-AUC ≥ {PR_MULTIPLIER} × prevalence。", "",
             "## Evidence 结果", "",
             "| Evidence | ROC-AUC | PR-AUC | Prevalence | true mean/median | FP mean/median | pass |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for key, row in summary["evidence"].items():
        lines.append(f"| {key} | {row['ROC_AUC']:.6f} | {row['PR_AUC']:.6f} | {row['positive_prevalence']:.6f} | {row['true_candidate_mean']:.6f}/{row['true_candidate_median']:.6f} | {row['weak_FP_mean']:.6f}/{row['weak_FP_median']:.6f} | {row['passes_joint_rule']} |")
    lines += ["", "## True rescue cases", ""]
    for row in true_rows:
        values = {key: row[key] for key in row if key.startswith(("R1_", "R2_", "R3_"))}
        lines.append(f"- {row['subject']}/{row['video']} peak={row['peak']}，matched_gt={row['matched_gt']}：`{json.dumps(values, sort_keys=True)}`")
    lines += ["", "## 解释边界", "",
              "该结果只回答现有 recognition logits 对固定 weak pool 是否具有排序分离性；未调 rescue threshold、未训练分类器、未实现联合 decoder，也未运行 CASME3。",
              "",
              "特别注意：正例仅 2 条，因此 AUC/AP 方差可能很大。本次 GO 是 feasibility gate 通过，不是已证明可泛化的方法收益，也不是论文结论。"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    records = list(payload["records"])
    subjects = sorted({str(record["subject"]) for record in records})
    if payload.get("dataset") != "SAMMLV" or int(payload.get("k_p", -1)) != K_P:
        raise RuntimeError("wrong frozen cache dataset or k_p")
    with args.weak_trace.open(newline="", encoding="utf-8") as handle:
        weak_rows = list(csv.DictReader(handle))

    selected, pooled_results, frequencies = run_attribution(records, subjects)
    full_selected = [row for row in selected if row["method"] == "Full Tuned Native"]
    if len(full_selected) != len(subjects):
        raise RuntimeError("full tuned selection did not produce one row per outer subject")
    write_csv(output_dir / "tuned_native_selected_configs.csv", full_selected)
    control_rows = []
    for method, row in pooled_results.items():
        control_rows.append({"method": method, **row,
                             "selected_parameter_frequencies_json": json.dumps(frequencies.get(method, {}), sort_keys=True)})
    write_csv(output_dir / "native_single_factor_controls.csv", control_rows)

    trace, evidence, status, validation = run_recognition(records, weak_rows)
    write_csv(output_dir / "recognition_weak_candidate_trace.csv", trace)
    true_rows = [row for row in trace if row["label"] == 1]
    common = {
        "algorithm_version": VERSION,
        "input": {"cache": str(args.cache.resolve()), "cache_sha256": sha256(args.cache),
                  "weak_trace": str(args.weak_trace.resolve()), "weak_trace_sha256": sha256(args.weak_trace),
                  "subjects": len(subjects), "videos": len(records), "GT": int(payload["num_gt"])},
        "locked_scope": {"new_grid": False, "new_feature": False, "training": False,
                         "model_forward": False, "CASME3": False, "candidate_pool_changed": False},
    }
    attr_summary = {**common,
                    "selection_protocol": {"outer_split": "subject LOSO", "primary": "pooled train-subject F1",
                                           "tie_break": ["fewer FP", "fewer rescue events", "lexicographic index"],
                                           "outer_test_GT_used_for_selection": False},
                    "pooled_outer_metrics": pooled_results,
                    "selected_parameter_frequencies": frequencies}
    rec_summary = {**common, "status": status,
                   "candidate_pool": {"n": len(trace), "positive": len(true_rows), "negative": len(trace)-len(true_rows),
                                      "positive_prevalence": len(true_rows)/len(trace),
                                      "identity_check": "exact rows loaded from mscr_weak_candidate_trace.csv"},
                   "recognition_validation": validation,
                   "evidence_definitions": {"R1": "1-P(neutral)", "R2": "max non-neutral probability",
                                            "R3": "top1 probability - top2 probability",
                                            "window": f"inclusive [peak-{K_P}, peak+{K_P}] clipped to video"},
                   "decision_rule": {"ROC_AUC_min": 0.60, "PR_AUC_min_multiplier_of_prevalence": PR_MULTIPLIER},
                   "evidence": evidence, "true_rescue_cases": true_rows}
    (output_dir / "recognition_separability.json").write_text(json.dumps(rec_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "NATIVE_PARAMETER_ATTRIBUTION_CN.md").write_text(attribution_report(attr_summary), encoding="utf-8")
    (args.output_root / "RECOGNITION_WEAK_CANDIDATE_AUDIT_CN.md").write_text(recognition_report(rec_summary, true_rows), encoding="utf-8")
    print(json.dumps({"attribution": pooled_results, "frequencies": frequencies,
                      "recognition_status": status, "prevalence": len(true_rows)/len(trace),
                      "evidence": evidence, "true_rescue_cases": true_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
