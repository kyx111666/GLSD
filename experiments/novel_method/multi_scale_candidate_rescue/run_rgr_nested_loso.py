#!/usr/bin/env python3
"""Recognition-Guided Rescue: locked SAMMLV nested subject-LOSO experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from run_mscr_nested_loso import (
    EXPECTED_NATIVE, K_P, TUNED_CB, TUNED_CD, TUNED_CS, TUNED_P,
    add_counts, author_native, empty_counts, evaluate_decoding, event, f1,
    match_events, metrics, sha256, threshold, tuned_native_decode, write_csv,
)


VERSION = "RGR-SAMMLV-v1-locked-2026-09-05"
DELTAS = (0.05, 0.10, 0.15, 0.20)
GAMMAS = (0.25, 0.35, 0.45, 0.55, 0.65)
CLASS_ORDER = ("negative", "positive", "surprise", "others", "neutral")
NEUTRAL_ID = 4
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260905
METHODS = ("Tuned Native", "LowRescue", "R1-mean Rescue", "R2-mean RGR")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prepare_records(payload):
    records = []
    shape_counts = Counter()
    temporal_lengths = []
    for source in payload["records"]:
        record = dict(source)
        logits = np.asarray(record["logits"])
        score = np.asarray(record["score"])
        temporal_lengths.append(len(score))
        shape_counts[str(tuple(logits.shape))] += 1
        if logits.ndim != 2 or logits.shape[1] != 5 or logits.shape[0] != len(score):
            raise RuntimeError(f"BLOCKED-LOGITS-ALIGNMENT: {record['subject']}/{record['video']} {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError(f"BLOCKED-NONFINITE-LOGITS: {record['subject']}/{record['video']}")
        probabilities = softmax(logits)
        record["R1"] = 1.0 - probabilities[:, NEUTRAL_ID]
        record["R2"] = np.max(probabilities[:, :NEUTRAL_ID], axis=1)
        records.append(record)
    validation = {
        "logits_shape_counts": dict(shape_counts), "softmax_axis": 1,
        "logits_shape_schema": "[T, 5]", "record_count": len(records),
        "temporal_length_min": min(temporal_lengths), "temporal_length_max": max(temporal_lengths),
        "class_order": list(CLASS_ORDER), "neutral_class_id": NEUTRAL_ID,
        "temporal_alignment_all_records": True, "all_logits_finite": True,
    }
    return records, validation


def tuned_configs():
    return [dict(zip(("c_s", "p_H", "c_d", "c_b"), values))
            for values in itertools.product(TUNED_CS, TUNED_P, TUNED_CD, TUNED_CB)]


def to_native_config(config):
    return {"c_s": config["c_s"], "p": config["p_H"],
            "c_d": config["c_d"], "c_b": config["c_b"]}


def mean_evidence(values, peak):
    left = max(0, int(peak) - K_P)
    right = min(len(values), int(peak) + K_P + 1)
    return float(np.mean(values[left:right]))


def rescue_decode(record, high_config, delta_p, gamma=None, evidence="R2", trace=False):
    high = tuned_native_decode(record, to_native_config(high_config))
    distance = max(1, int(round(float(high_config["c_d"]) * K_P)))
    boundary = max(1, int(round(float(high_config["c_b"]) * K_P)))
    p_low = float(high_config["p_H"]) - float(delta_p)
    tau_low = threshold(high["curve"], p_low)
    low_peaks = find_peaks(high["curve"], height=tau_low, distance=distance)[0].astype(int)
    high_set = set(map(int, high["peaks"]))
    weak = np.asarray(sorted(set(map(int, low_peaks)) - high_set), dtype=int)
    rescued, weak_rows = [], []
    for peak in weak:
        recognition = mean_evidence(record[evidence], int(peak))
        keep = gamma is None or recognition >= float(gamma)
        if keep:
            rescued.append(event(int(peak), boundary, "rescue"))
        if trace:
            weak_rows.append({"peak": int(peak), "R_rec": recognition, "selected": bool(keep)})
    final_events = sorted(high["events"] + rescued,
                          key=lambda item: (item["peak"], 0 if item["source"] == "tuned_native" else 1))
    high_geometry = [(x["onset"], x["peak"], x["offset"]) for x in high["events"]]
    final_geometry = {(x["onset"], x["peak"], x["offset"]) for x in final_events}
    if not all(x in final_geometry for x in high_geometry):
        raise RuntimeError("native preservation failure")
    return {"native": high, "weak": weak, "rescued": rescued, "events": final_events,
            "weak_rows": weak_rows, "p_low": p_low, "tau_low": tau_low,
            "native_preserved": len(high_geometry), "native_total": len(high_geometry)}


def evaluate_records(records, subjects, decoder):
    total = empty_counts()
    for record in records:
        if str(record["subject"]) in subjects:
            add_counts(total, evaluate_decoding(record, decoder(record)))
    return total


def select_stage_a(records, train_subjects, configs, bank):
    ranked = []
    for index, config in enumerate(configs):
        counts = empty_counts()
        for record_index, record in enumerate(records):
            if str(record["subject"]) in train_subjects:
                add_counts(counts, evaluate_decoding(record, bank[index][record_index]))
        ranked.append(((-f1(counts), counts["FP"], counts["rescued_candidate_count"], index),
                       index, config, metrics(counts)))
    _, index, config, train_metrics = min(ranked, key=lambda row: row[0])
    return index, config, train_metrics


def stage_b_configs(method):
    if method == "LowRescue":
        return [{"delta_p": delta, "gamma_R": None} for delta in DELTAS]
    return [{"delta_p": delta, "gamma_R": gamma}
            for delta, gamma in itertools.product(DELTAS, GAMMAS)]


def select_stage_b(records, train_subjects, high_config, method):
    evidence = "R1" if method == "R1-mean Rescue" else "R2"
    ranked = []
    configs = stage_b_configs(method)
    for index, config in enumerate(configs):
        decoder = lambda record, c=config: rescue_decode(
            record, high_config, c["delta_p"], c["gamma_R"], evidence=evidence
        )
        counts = evaluate_records(records, train_subjects, decoder)
        ranked.append(((-f1(counts), counts["FP"], counts["rescued_candidate_count"], index),
                       index, config, metrics(counts)))
    _, index, config, train_metrics = min(ranked, key=lambda row: row[0])
    return index, config, train_metrics


def trace_selected(records, held, method, high_config, rescue_config):
    evidence = "R1" if method == "R1-mean Rescue" else "R2"
    rows = []
    preserved = total_native = 0
    for record in records:
        if str(record["subject"]) != held:
            continue
        decoded = rescue_decode(record, high_config, rescue_config["delta_p"],
                                rescue_config["gamma_R"], evidence=evidence, trace=True)
        result = evaluate_decoding(record, decoded)
        native_matches, native_unmatched = match_events(decoded["native"]["events"], record["samples"])
        del native_matches
        event_matches = {(item["peak"], item["source"]): match
                         for item, match in zip(decoded["events"], result["matches"])}
        preserved += decoded["native_preserved"]
        total_native += decoded["native_total"]
        for item in decoded["weak_rows"]:
            match = event_matches.get((item["peak"], "rescue"), -1) if item["selected"] else -1
            rows.append({
                "outer_subject": held, "method": method, "video": str(record["video"]),
                "peak": item["peak"], "evidence": evidence + "-mean", "R_rec": item["R_rec"],
                "selected": item["selected"], "matched_gt": match,
                "rescue_TP": bool(item["selected"] and match >= 0),
                "rescue_FP": bool(item["selected"] and match < 0),
                "native_missed_GT_recovered": bool(item["selected"] and match in native_unmatched),
                "c_s": high_config["c_s"], "p_H": high_config["p_H"],
                "c_d": high_config["c_d"], "c_b": high_config["c_b"],
                "delta_p": rescue_config["delta_p"], "p_L": high_config["p_H"]-rescue_config["delta_p"],
                "gamma_R": rescue_config["gamma_R"],
            })
    return rows, preserved, total_native


def bootstrap(subject_rows):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(0, len(subject_rows), size=len(subject_rows))
        tuned, rgr = empty_counts(), empty_counts()
        for index in indices:
            row = subject_rows[int(index)]
            for prefix, target in (("Tuned", tuned), ("RGR", rgr)):
                add_counts(target, {key: row[f"{prefix}_{key}"] for key in ("TP", "FP", "FN")})
        draws.append(f1(rgr) - f1(tuned))
    lo, hi = np.quantile(draws, (0.025, 0.975))
    return {"unit": "subject", "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
            "mean_delta_F1": float(np.mean(draws)), "CI95": [float(lo), float(hi)]}


def frequency(selected_rows, method, key):
    return dict(sorted(Counter(str(row[key]) for row in selected_rows
                               if row["method"] == method).items()))


def report(summary):
    metrics_by_method = summary["pooled_outer_metrics"]
    lines = ["# Recognition-Guided Rescue (RGR) SAMMLV Nested LOSO Report", "",
             f"最终状态：**{summary['status']}**", "", "## 1. 锁定协议", "",
             "- R2-mean 是上一阶段 recognition separability audit 后锁定的 primary evidence，不是根据本次 outer test result 选择。",
             "- 29-fold outer subject LOSO；Stage A 与 Stage B 均只用 train subjects 的 pooled Spotting F1 选参。",
             "- Stage A 的 conservative events 在 Stage B 中只允许原样保留；RGR 只可加入 `P_low - P_high`。",
             "- R1-mean 是独立 secondary control，未与 R2 在 outer result 后择优。",
             "- 未使用 MSCR/multi-scale、未训练、未 model forward、未运行 CASME3。", "",
             "## 2. Recognition 输入核验", "",
             f"- logits shape=`{summary['recognition_validation']['logits_shape_schema']}`；79/79 records 对齐，T 范围为 {summary['recognition_validation']['temporal_length_min']}–{summary['recognition_validation']['temporal_length_max']}。",
             f"- softmax axis={summary['recognition_validation']['softmax_axis']}；class order={summary['recognition_validation']['class_order']}；neutral id={summary['recognition_validation']['neutral_class_id']}。", "",
             "## 3. Outer pooled metrics", "",
             "| Baseline | TP | FP | FN | Precision | Recall | F1 | weak | rescued | rescue TP/FP | rescue precision | recovered missed GT |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    display = [("A0 Author Native", "Author Native"), ("A1 Tuned Native", "Tuned Native"),
               ("A2 LowRescue", "LowRescue"), ("A3 R1-mean Rescue", "R1-mean Rescue"),
               ("A4 R2-mean RGR", "R2-mean RGR")]
    for label, method in display:
        row = metrics_by_method[method]
        lines.append(f"| {label} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['F1']:.6f} | {row.get('weak_candidate_count',0)} | {row.get('rescued_candidate_count',0)} | {row.get('rescue_TP',0)}/{row.get('rescue_FP',0)} | {row.get('rescue_precision',0):.6f} | {row.get('native_missed_GT_recovered',0)} |")
    lines += ["", "## 4. LowRescue vs RGR", "",
              f"- LowRescue rescue precision={metrics_by_method['LowRescue']['rescue_precision']:.6f}；RGR={metrics_by_method['R2-mean RGR']['rescue_precision']:.6f}。",
              f"- LowRescue rescue TP/FP={metrics_by_method['LowRescue']['rescue_TP']}/{metrics_by_method['LowRescue']['rescue_FP']}；RGR={metrics_by_method['R2-mean RGR']['rescue_TP']}/{metrics_by_method['R2-mean RGR']['rescue_FP']}。", "",
              "## 5. Subject stability 与 bootstrap", "",
              f"- improved/equal/worse={summary['subject_stability']['improved']}/{summary['subject_stability']['equal']}/{summary['subject_stability']['worse']}。",
              f"- leave-best-contributor-out pooled ΔF1={summary['subject_stability']['leave_best_contributor_out_delta_F1']:.6f}。",
              f"- bootstrap mean ΔF1={summary['bootstrap']['mean_delta_F1']:.6f}，95% CI=[{summary['bootstrap']['CI95'][0]:.6f}, {summary['bootstrap']['CI95'][1]:.6f}]。", "",
              "## 6. Selected parameter frequencies", ""]
    for method, values in summary["selected_parameter_frequencies"].items():
        lines.append(f"- {method}: `{json.dumps(values, sort_keys=True)}`")
    lines += ["", "## 7. GO checks", ""]
    for key, value in summary["go_checks"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## 8. 结论", "", summary["conclusion"]]
    lines += ["", "## 9. 解释边界", "",
              "- 上一阶段 R2-mean separability 的固定 45 条 pool 来自 Author Native/MSCR diagnostic；本轮 W 来自每折 Conservative Tuned Native（主要为 p_H=0.65），因此是不同的候选分布。前一阶段的排序 GO 不保证在本轮 W 上成功。",
              "- R2-mean 虽在本轮 outer test 前锁定，但它是在同一 SAMMLV 数据集的前置 audit 中从九种 evidence 里识别出来的；所以本实验是第一阶段 feasibility test，不是完全独立的 confirmatory generalization test。"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("dataset") != "SAMMLV" or int(payload.get("k_p", -1)) != K_P:
        raise RuntimeError("BLOCKED-BASELINE: expected SAMMLV k_p=5")
    records, recognition_validation = prepare_records(payload)
    subjects = sorted({str(record["subject"]) for record in records})

    native_total = empty_counts()
    native_by_subject = {}
    for subject in subjects:
        counts = evaluate_records(records, {subject}, lambda record: {"events": author_native(record)["events"]})
        native_by_subject[subject] = counts
        add_counts(native_total, counts)
    if (native_total["TP"], native_total["FP"], native_total["FN"]) != EXPECTED_NATIVE:
        raise RuntimeError(f"BLOCKED-BASELINE: observed {metrics(native_total)}")

    stage_a_configs = tuned_configs()
    if len(stage_a_configs) != 81:
        raise RuntimeError("Stage A grid is not 81 configs")
    stage_a_bank = [[{"events": tuned_native_decode(record, to_native_config(config))["events"]}
                     for record in records] for config in stage_a_configs]

    selected_rows, subject_rows, rescue_trace = [], [], []
    outer = defaultdict(dict)
    preservation = Counter()
    for held in subjects:
        train = set(subjects) - {held}
        stage_a_index, high_config, stage_a_train = select_stage_a(
            records, train, stage_a_configs, stage_a_bank
        )
        tuned_counts = evaluate_records(
            records, {held}, lambda record: {"events": tuned_native_decode(record, to_native_config(high_config))["events"]}
        )
        outer["Tuned Native"][held] = tuned_counts
        selected_rows.append({"outer_subject": held, "method": "Tuned Native",
                              "config_index": stage_a_index, **high_config,
                              "delta_p": "", "gamma_R": "", "train_pooled_F1": stage_a_train["F1"],
                              "outer_F1": f1(tuned_counts)})

        fold_configs = {}
        for method in ("LowRescue", "R1-mean Rescue", "R2-mean RGR"):
            index, rescue_config, train_metrics = select_stage_b(records, train, high_config, method)
            fold_configs[method] = rescue_config
            evidence = "R1" if method == "R1-mean Rescue" else "R2"
            counts = evaluate_records(
                records, {held},
                lambda record, c=rescue_config, e=evidence: rescue_decode(
                    record, high_config, c["delta_p"], c["gamma_R"], evidence=e
                ),
            )
            outer[method][held] = counts
            selected_rows.append({"outer_subject": held, "method": method,
                                  "config_index": index, **high_config, **rescue_config,
                                  "train_pooled_F1": train_metrics["F1"], "outer_F1": f1(counts)})
            traces, preserved, total = trace_selected(records, held, method, high_config, rescue_config)
            rescue_trace.extend(traces)
            if method == "R2-mean RGR":
                preservation.update(preserved=preserved, total=total)

        row = {"outer_subject": held,
               "Author_F1": f1(native_by_subject[held]),
               "Tuned_Native_F1": f1(tuned_counts),
               "RGR_F1": f1(outer["R2-mean RGR"][held]),
               "Delta_RGR_minus_TunedNative": f1(outer["R2-mean RGR"][held]) - f1(tuned_counts),
               "rescue_TP": outer["R2-mean RGR"][held]["rescue_TP"],
               "rescue_FP": outer["R2-mean RGR"][held]["rescue_FP"],
               "selected_Delta_p": fold_configs["R2-mean RGR"]["delta_p"],
               "selected_gamma_R": fold_configs["R2-mean RGR"]["gamma_R"]}
        for method, prefix in (("Tuned Native", "Tuned"), ("R2-mean RGR", "RGR")):
            for key in ("TP", "FP", "FN"):
                row[f"{prefix}_{key}"] = outer[method][held][key]
        subject_rows.append(row)

    pooled = {"Author Native": metrics(native_total)}
    for method in METHODS:
        total = empty_counts()
        for subject in subjects:
            add_counts(total, outer[method][subject])
        pooled[method] = metrics(total)

    deltas = np.asarray([row["Delta_RGR_minus_TunedNative"] for row in subject_rows])
    stability = {"improved": int((deltas > 1e-12).sum()),
                 "equal": int((np.abs(deltas) <= 1e-12).sum()),
                 "worse": int((deltas < -1e-12).sum())}
    best_subject = subject_rows[int(np.argmax(deltas))]["outer_subject"]
    tuned_without, rgr_without = empty_counts(), empty_counts()
    for row in subject_rows:
        if row["outer_subject"] == best_subject:
            continue
        for prefix, target in (("Tuned", tuned_without), ("RGR", rgr_without)):
            add_counts(target, {key: row[f"{prefix}_{key}"] for key in ("TP", "FP", "FN")})
    stability["largest_subject_delta"] = float(deltas.max())
    stability["largest_contributor_subject"] = best_subject
    stability["leave_best_contributor_out_delta_F1"] = f1(rgr_without) - f1(tuned_without)
    stability["not_single_subject"] = stability["improved"] >= 2

    frequencies = {
        "Tuned Native": {key: frequency(selected_rows, "Tuned Native", key)
                         for key in ("c_s", "p_H", "c_d", "c_b")},
        "LowRescue": {"delta_p": frequency(selected_rows, "LowRescue", "delta_p")},
        "R1-mean Rescue": {key: frequency(selected_rows, "R1-mean Rescue", key)
                           for key in ("delta_p", "gamma_R")},
        "R2-mean RGR": {key: frequency(selected_rows, "R2-mean RGR", key)
                        for key in ("delta_p", "gamma_R")},
    }
    tuned, low, rgr = pooled["Tuned Native"], pooled["LowRescue"], pooled["R2-mean RGR"]
    gain = rgr["F1"] - tuned["F1"]
    checks = {"RGR_F1_ge_Tuned_plus_0.005": gain >= 0.005 - 1e-15,
              "RGR_rescue_TP_gt_0": rgr["rescue_TP"] > 0,
              "RGR_rescue_precision_gt_LowRescue": rgr["rescue_precision"] > low["rescue_precision"],
              "gain_not_single_subject": stability["not_single_subject"]}
    if all(checks.values()):
        status = "STRONG-GO-RGR"
        conclusion = "**STRONG-GO-RGR：R2-mean RGR 达到预设 F1、rescue TP、相对 LowRescue precision 与多 subject 贡献四项条件。按 stop rule 在 SAMMLV 第一阶段停止。**"
    elif rgr["F1"] > tuned["F1"] and gain < 0.005:
        status = "WEAK-GO-RGR"
        conclusion = "**WEAK-GO-RGR：R2-mean RGR 高于 Tuned Native，但绝对 F1 增益小于 0.005。按 stop rule 停止。**"
    else:
        status = "NO-GO-RGR"
        reason = "RGR 未超过 Tuned Native" if rgr["F1"] <= tuned["F1"] else "RGR 虽超过 Tuned Native，但未满足预设 STRONG/WEAK 完整条件"
        conclusion = f"**NO-GO-RGR：{reason}。按 stop rule 停止，不扩展后续方法。**"

    boot = bootstrap(subject_rows)
    summary = {
        "status": status, "algorithm_version": VERSION,
        "input": {"cache": str(args.cache.resolve()), "sha256": sha256(args.cache),
                  "dataset": payload["dataset"], "subjects": len(subjects),
                  "videos": len(records), "GT": int(payload["num_gt"]), "k_p": K_P},
        "recognition_validation": recognition_validation,
        "locked_primary_evidence": "R2-mean selected before this experiment from the prior recognition separability audit",
        "grids": {"Stage_A": {"c_s": TUNED_CS, "p_H": TUNED_P, "c_d": TUNED_CD, "c_b": TUNED_CB,
                              "count": len(stage_a_configs)},
                  "Stage_B": {"delta_p": DELTAS, "gamma_R": GAMMAS, "RGR_count": len(DELTAS)*len(GAMMAS)}},
        "selection_protocol": {"outer": "subject LOSO", "inner_objective": "pooled train-subject Spotting F1",
                               "tie_break": ["fewer FP", "fewer rescue events", "lexicographic config index"],
                               "test_subject_GT_used_for_selection": False},
        "locked_scope": {"MSCR": False, "model_forward": False, "training": False, "CASME3": False,
                         "new_recognition_features": False},
        "pooled_outer_metrics": pooled,
        "native_preservation": {"preserved": int(preservation["preserved"]), "total": int(preservation["total"]),
                                "rate": preservation["preserved"] / preservation["total"]},
        "selected_parameter_frequencies": frequencies,
        "subject_stability": stability, "bootstrap": boot,
        "delta_F1_RGR_minus_TunedNative": gain, "go_checks": checks,
        "conclusion": conclusion,
    }
    write_csv(output_dir / "rgr_outer_metrics.csv", subject_rows)
    write_csv(output_dir / "rgr_selected_configs.csv", selected_rows)
    write_csv(output_dir / "rgr_rescue_trace.csv", rescue_trace)
    (output_dir / "rgr_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "RGR_NESTED_LOSO_SAMMLV_REPORT_CN.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps({"status": status, "pooled": pooled, "delta_F1": gain,
                      "subject_stability": stability, "bootstrap": boot,
                      "native_preservation": summary["native_preservation"],
                      "frequencies": frequencies, "go_checks": checks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
