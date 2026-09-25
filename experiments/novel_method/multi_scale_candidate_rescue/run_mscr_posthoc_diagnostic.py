#!/usr/bin/env python3
"""Locked post-hoc diagnostics for the completed SAMMLV MSCR audit."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, roc_auc_score

from run_mscr_nested_loso import (
    EPS, FAMILIES, K_P, P_HIGH, add_counts, author_native, decode,
    empty_counts, evaluate_decoding, evaluate_subject, f1, make_configs,
    match_events, metrics, moving_average, mscr_decode, precompute_score_only,
    scale_alignment, sha256, threshold, write_csv,
)


ALGORITHM_VERSION = "MSCR-posthoc-diagnostic-v1-locked-2026-09-04"
PROTOCOL_ISSUE_MIN_F1_GAIN = 0.005
SEPARABLE_FP_QUANTILE = 0.75
SEPARABLE_MAX_JOINT_DOMINANCE = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--original-selected-configs", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def load_original_configs(path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    lookup = {}
    for row in rows:
        if row["method"] in {"LowRescue", "MSCR"}:
            lookup[(row["subject"], row["method"])] = json.loads(row["selected_config_json"])
    return lookup


def select_pooled_config(method, configs, score_only_bank, records, train_subjects):
    ranked = []
    for index, config in enumerate(configs[method]):
        pooled = empty_counts()
        for record_index, record in enumerate(records):
            if str(record["subject"]) not in train_subjects:
                continue
            result = evaluate_decoding(record, score_only_bank[method][index][record_index])
            add_counts(pooled, result)
        ranked.append(((-f1(pooled), pooled["FP"], pooled["rescued_candidate_count"], index),
                       index, config, metrics(pooled)))
    _, index, config, train_metrics = min(ranked, key=lambda row: row[0])
    return index, config, train_metrics


def run_pooled_objective(records, subjects):
    configs = make_configs()
    score_only_bank = precompute_score_only(records, configs)
    native_by_subject = {s: empty_counts() for s in subjects}
    native_total = empty_counts()
    for record in records:
        native = author_native(record)
        result = evaluate_decoding(record, {"events": native["events"]})
        add_counts(native_by_subject[str(record["subject"])], result)
        add_counts(native_total, result)

    outer = defaultdict(dict)
    selected = []
    preservation = Counter()
    for held in subjects:
        train = {s for s in subjects if s != held}
        for method in ["TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
            index, config, train_metrics = select_pooled_config(
                method, configs, score_only_bank, records, train
            )
            counts, _, preserved, total = evaluate_subject(records, held, method, config)
            outer[method][held] = counts
            selected.append({"outer_subject": held, "method": method, "config_index": index,
                             "selected_config": config, "train_pooled_metrics": train_metrics,
                             "test_metrics": metrics(counts)})
            if method == "MSCR":
                preservation.update(preserved=preserved, total=total)

    pooled = {"AuthorNative": metrics(native_total)}
    for method in ["TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
        total = empty_counts()
        for subject in subjects:
            add_counts(total, outer[method][subject])
        pooled[method] = metrics(total)

    frequencies = {}
    for method in ["TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
        frequencies[method] = dict(Counter(
            json.dumps(row["selected_config"], sort_keys=True)
            for row in selected if row["method"] == method
        ))
    return {
        "selection_primary": "pooled train-subject F1",
        "tie_break": ["fewer FP", "fewer rescue events", "lexicographic config order"],
        "outer_test_GT_used_for_selection": False,
        "pooled_metrics": pooled,
        "selected_configs": selected,
        "selected_config_frequencies": frequencies,
        "native_preservation": {"preserved": int(preservation["preserved"]),
                                "total": int(preservation["total"]),
                                "rate": preservation["preserved"] / preservation["total"]},
    }


def interval_iou_at_peak(peak, sample):
    event = {"onset": int(peak - K_P), "offset": int(peak + K_P)}
    left, right = event["onset"], event["offset"]
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    inter = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return inter / union if union else 0.0


def evidence_for_peak(record, peak, config):
    native = author_native(record)
    curve, tau_high = native["curve"], native["threshold"]
    p_low = P_HIGH - float(config["delta_p"])
    tau_low = threshold(curve, p_low)
    low_peaks = find_peaks(curve, height=tau_low, distance=K_P)[0].astype(int)
    selected_weak_pool = set(map(int, low_peaks)) | set(map(int, native["peaks"]))
    selected_weak_pool -= set(map(int, native["peaks"]))
    height = float(np.clip((curve[peak] - tau_low) / (tau_high - tau_low + EPS), 0.0, 1.0))
    radius = float(config["c_r"]) * K_P
    details = []
    alignments = []
    for scale in FAMILIES[config["scale_family"]]:
        k_a = max(1, int(round(scale * K_P)))
        g_a = moving_average(record["score"], 2 * k_a)
        tau_a = threshold(g_a, p_low)
        peaks = find_peaks(g_a, height=tau_a, distance=k_a)[0].astype(int)
        if len(peaks):
            nearest = int(peaks[np.argmin(np.abs(peaks - peak))])
            distance = abs(nearest - int(peak))
        else:
            nearest, distance = None, None
        alignment = scale_alignment(peaks, int(peak), radius)
        alignments.append(alignment)
        details.append({"scale": scale, "k_a": k_a, "nearest_peak": nearest,
                        "distance": distance, "alignment": alignment})
    multi = float(np.mean(alignments))
    lam = float(config["lambda"])
    rescue_score = lam * height + (1.0 - lam) * multi
    in_pool = int(peak) in selected_weak_pool
    selected = in_pool and rescue_score >= float(config["gamma"])
    if not in_pool:
        reason = "NOT_IN_SELECTED_MSCR_WEAK_POOL"
    elif rescue_score < float(config["gamma"]):
        reason = "R_BELOW_GAMMA"
    else:
        reason = "SELECTED"
    return {"g_peak": float(curve[peak]), "tau_high": tau_high, "tau_low": tau_low,
            "H": height, "M": multi, "R": rescue_score,
            "nearest_peak_each_scale": json.dumps({str(d["scale"]): d["nearest_peak"] for d in details}),
            "distance_each_scale": json.dumps({str(d["scale"]): d["distance"] for d in details}),
            "alignment_each_scale": json.dumps({str(d["scale"]): d["alignment"] for d in details}),
            "scale_details_json": json.dumps(details), "in_selected_mscr_weak_pool": in_pool,
            "selected_by_mscr": selected, "mscr_rejection_reason": reason}


def weak_candidate_trace(records, subjects, original_configs):
    trace = []
    for subject in subjects:
        low_config = original_configs[(subject, "LowRescue")]
        mscr_config = original_configs[(subject, "MSCR")]
        for record in records:
            if str(record["subject"]) != subject:
                continue
            low = mscr_decode(record, low_config, mode="low", trace=True)
            low_result = evaluate_decoding(record, low)
            _, native_unmatched = match_events(low["native"]["events"], record["samples"])
            low_match = {(event["peak"], event["source"]): match
                         for event, match in zip(low["events"], low_result["matches"])}
            selected_mscr = mscr_decode(record, mscr_config, mode="full")
            selected_mscr_result = evaluate_decoding(record, selected_mscr)
            mscr_match = {(event["peak"], event["source"]): match
                          for event, match in zip(selected_mscr["events"], selected_mscr_result["matches"])}
            for weak_row in low["weak_rows"]:
                peak = int(weak_row["peak"])
                match = low_match.get((peak, "rescue"), -1)
                ious = [interval_iou_at_peak(peak, sample) for sample in record["samples"]]
                best_gt = int(np.argmax(ious)) if ious else -1
                best_iou = float(ious[best_gt]) if ious else 0.0
                is_true = match >= 0 and match in native_unmatched
                evidence = evidence_for_peak(record, peak, mscr_config)
                mscr_outcome = "NOT_SELECTED"
                if evidence["selected_by_mscr"]:
                    mscr_outcome = "TP" if mscr_match.get((peak, "rescue"), -1) >= 0 else "FP"
                trace.append({
                    "subject": subject, "video": str(record["video"]), "peak": peak,
                    "is_native_missed_gt_candidate": is_true, "best_gt_id": best_gt,
                    "best_gt_iou": best_iou, **evidence,
                    "scale_family": mscr_config["scale_family"], "delta_p": mscr_config["delta_p"],
                    "c_r": mscr_config["c_r"], "lambda": mscr_config["lambda"],
                    "gamma": mscr_config["gamma"],
                    "selected_mscr_config_json": json.dumps(mscr_config, sort_keys=True),
                    "selected_lowrescue_config_json": json.dumps(low_config, sort_keys=True),
                    "rescue_tp_or_fp": "TP" if match >= 0 else "FP",
                    "lowrescue_matched_gt_index": match,
                    "mscr_rescue_tp_or_fp": mscr_outcome,
                })
    return trace


def describe(values):
    a = np.asarray(values, dtype=float)
    q10, q25, q75, q90 = np.quantile(a, [.10, .25, .75, .90])
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(q10), "p25": float(q25), "p75": float(q75), "p90": float(q90),
            "min": float(a.min()), "max": float(a.max())}


def separability(trace):
    true_rows = [row for row in trace if row["is_native_missed_gt_candidate"]]
    weak_fp = [row for row in trace if row["rescue_tp_or_fp"] == "FP"]
    y = np.asarray([int(row["is_native_missed_gt_candidate"]) for row in true_rows + weak_fp])
    prevalence = float(y.mean())
    univariate = {}
    for key in ["H", "M"]:
        scores = np.asarray([row[key] for row in true_rows + weak_fp])
        univariate[key] = {"ROC_AUC": float(roc_auc_score(y, scores)),
                           "PR_AUC": float(average_precision_score(y, scores)),
                           "positive_prevalence": prevalence}
    fp_h = describe([row["H"] for row in weak_fp])
    fp_m = describe([row["M"] for row in weak_fp])
    joint = []
    for row in true_rows:
        dominance = float(np.mean([fp["H"] >= row["H"] and fp["M"] >= row["M"] for fp in weak_fp]))
        distances = [float(np.hypot(fp["H"] - row["H"], fp["M"] - row["M"])) for fp in weak_fp]
        joint.append({"subject": row["subject"], "video": row["video"], "peak": row["peak"],
                      "H": row["H"], "M": row["M"], "R": row["R"],
                      "joint_FP_dominance_fraction": dominance,
                      "nearest_FP_distance_in_HM": min(distances) if distances else None,
                      "H_ge_FP_p75": row["H"] >= fp_h["p75"],
                      "M_ge_FP_p75": row["M"] >= fp_m["p75"]})
    separable = len(true_rows) == 2 and all(
        row["H_ge_FP_p75"] and row["M_ge_FP_p75"] and
        row["joint_FP_dominance_fraction"] <= SEPARABLE_MAX_JOINT_DOMINANCE
        for row in joint
    )
    return {"true_rescue_count": len(true_rows), "weak_FP_count": len(weak_fp),
            "weak_FP_distribution": {"H": fp_h, "M": fp_m},
            "univariate": univariate, "joint_HM": joint,
            "separable_rule": "both true cases H/M >= weak-FP p75 and joint dominance <=10%",
            "evidence_separable": separable}


def make_report(result, true_rows):
    a = result["diagnostic_A_pooled_objective"]
    b = result["diagnostic_B_separability"]
    lines = ["# MSCR Post-hoc Diagnostic Audit", "", f"最终状态：**{result['status']}**", "",
             "## 1. 审计边界", "",
             "- 仅执行 pooled-F1 selection objective 与 weak-candidate separability trace。",
             "- MSCR formula、candidate generation、全部网格、outer folds、evaluator 与 Native preservation 未改变。",
             "- 未运行 CASME3、recognition、hidden、训练模型或新方法。", "",
             "## 2. Diagnostic A：Pooled-F1 Selection", "",
             "| Method | TP | FP | FN | Precision | Recall | F1 | Rescue TP/FP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ["AuthorNative", "TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
        m = a["pooled_metrics"][method]
        lines.append(f"| {method} | {m['TP']} | {m['FP']} | {m['FN']} | {m['Precision']:.6f} | {m['Recall']:.6f} | {m['F1']:.6f} | {m.get('rescue_TP',0)}/{m.get('rescue_FP',0)} |")
    lines += ["", f"Native event exact preservation：{a['native_preservation']['preserved']}/{a['native_preservation']['total']}。",
              f"MSCR-AuthorNative F1 delta：{result['protocol_issue_test']['F1_delta']:.6f}；预固定明显改善阈值：{PROTOCOL_ISSUE_MIN_F1_GAIN:.3f}。", "",
              "## 3. Diagnostic B：True LowRescue cases", ""]
    for row in true_rows:
        lines += [f"### {row['subject']}/{row['video']} peak={row['peak']}", "",
                  f"- H={row['H']:.6f}，M={row['M']:.6f}，R={row['R']:.6f}，gamma={float(row['gamma']):.2f}。",
                  f"- per-scale nearest peaks：`{row['nearest_peak_each_scale']}`。",
                  f"- per-scale distances：`{row['distance_each_scale']}`。",
                  f"- selected config：`{row['selected_mscr_config_json']}`。",
                  f"- MSCR direct rejection reason：`{row['mscr_rejection_reason']}`。", ""]
    lines += ["## 4. Weak-FP evidence distribution", "", "| Evidence | mean | median | p10 | p25 | p75 | p90 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key in ["H", "M"]:
        d = b["weak_FP_distribution"][key]
        lines.append(f"| {key} | {d['mean']:.6f} | {d['median']:.6f} | {d['p10']:.6f} | {d['p25']:.6f} | {d['p75']:.6f} | {d['p90']:.6f} |")
    lines += ["", "## 5. Separability", "",
              f"- H：ROC-AUC={b['univariate']['H']['ROC_AUC']:.6f}，PR-AUC={b['univariate']['H']['PR_AUC']:.6f}。",
              f"- M：ROC-AUC={b['univariate']['M']['ROC_AUC']:.6f}，PR-AUC={b['univariate']['M']['PR_AUC']:.6f}。",
              f"- positive prevalence={b['univariate']['H']['positive_prevalence']:.6f}。",
              f"- locked [H,M] separability rule passed={b['evidence_separable']}。", ""]
    for row in b["joint_HM"]:
        lines.append(f"- {row['subject']}/{row['video']} peak={row['peak']}：joint FP dominance={row['joint_FP_dominance_fraction']:.6f}，nearest FP distance={row['nearest_FP_distance_in_HM']:.6f}，H/M≥FP-p75={row['H_ge_FP_p75']}/{row['M_ge_FP_p75']}。")
    lines += ["", "## 6. Final interpretation", "", result["conclusion"], "",
              "本报告只解释现有失败，不提出或运行新的 fusion、threshold、feature 或 classifier。"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as f:
        payload = pickle.load(f)
    records = list(payload["records"])
    subjects = sorted({str(record["subject"]) for record in records})
    if payload["dataset"] != "SAMMLV" or len(subjects) != 29 or len(records) != 79:
        raise RuntimeError("unexpected diagnostic input")

    diagnostic_a = run_pooled_objective(records, subjects)
    original_configs = load_original_configs(args.original_selected_configs)
    trace = weak_candidate_trace(records, subjects, original_configs)
    true_rows = [row for row in trace if row["is_native_missed_gt_candidate"]]
    if len(true_rows) != 2:
        raise RuntimeError(f"expected exactly 2 historical LowRescue recoveries, observed {len(true_rows)}")
    diagnostic_b = separability(trace)
    author_f1 = diagnostic_a["pooled_metrics"]["AuthorNative"]["F1"]
    mscr_f1 = diagnostic_a["pooled_metrics"]["MSCR"]["F1"]
    delta = mscr_f1 - author_f1
    protocol_issue = delta >= PROTOCOL_ISSUE_MIN_F1_GAIN
    if protocol_issue:
        status = "MSCR-PROTOCOL-ISSUE"
        conclusion = "**MSCR-PROTOCOL-ISSUE：将 inner objective 改为 pooled train-subject F1 后，MSCR 明显超过 Author Native，原 NO-GO 对选择目标敏感。**"
    elif diagnostic_b["evidence_separable"]:
        status = "MSCR-EVIDENCE-SEPARABLE"
        conclusion = "**MSCR-EVIDENCE-SEPARABLE：两个 true LowRescue candidates 的 H/M 均位于预固定高可靠区域，但当前 fusion/threshold 未选中。**"
    else:
        status = "MSCR-EVIDENCE-NONSEPARABLE"
        conclusion = "**MSCR-EVIDENCE-NONSEPARABLE：pooled-F1 selection 未逆转 source-side NO-GO，且两个 true rescue candidates 与 weak FP 在 H/M 上未满足预固定 separability 条件。**"

    result = {
        "status": status, "algorithm_version": ALGORITHM_VERSION,
        "input": {"cache_path": str(args.cache.resolve()), "cache_sha256": sha256(args.cache),
                  "original_selected_configs_path": str(args.original_selected_configs.resolve()),
                  "original_selected_configs_sha256": sha256(args.original_selected_configs),
                  "subjects": len(subjects), "videos": len(records), "GT": int(payload["num_gt"])},
        "locked_scope": {"new_parameters": False, "new_method": False, "CASME3": False,
                         "candidate_generation_changed": False, "evaluator_changed": False},
        "diagnostic_A_pooled_objective": diagnostic_a,
        "protocol_issue_test": {"F1_delta": delta, "minimum_clear_gain": PROTOCOL_ISSUE_MIN_F1_GAIN,
                                "passed": protocol_issue},
        "diagnostic_B_separability": diagnostic_b,
        "true_rescue_cases": true_rows,
        "conclusion": conclusion,
    }
    pooled_rows = []
    for method in ["AuthorNative", "TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
        pooled_rows.append({"method": method, **diagnostic_a["pooled_metrics"][method]})
    write_csv(out / "mscr_pooled_objective_results.csv", pooled_rows)
    write_csv(out / "mscr_weak_candidate_trace.csv", trace)
    write_csv(out / "mscr_true_rescue_cases.csv", true_rows)
    (out / "mscr_diagnostic.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "MSCR_POSTHOC_DIAGNOSTIC_CN.md").write_text(make_report(result, true_rows), encoding="utf-8")
    print(json.dumps({"status": status, "pooled_metrics": diagnostic_a["pooled_metrics"],
                      "protocol_issue_test": result["protocol_issue_test"],
                      "separability": diagnostic_b}, indent=2))


if __name__ == "__main__":
    main()
