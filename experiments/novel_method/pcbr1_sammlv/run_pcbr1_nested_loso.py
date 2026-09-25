#!/usr/bin/env python3
"""Locked, cache-only PCBR-1. No GT or recognition enters reconstruct()."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "my_method/multi_scale_candidate_rescue"))
from run_mscr_nested_loso import (
    K_P, add_counts, empty_counts, evaluate_decoding, f1, interval_iou,
    metrics, sha256, tuned_native_decode,
)
# Only the previously verified recognition helper loader is reused. No RGR
# decoder, recognition feature, selection or candidate generation is invoked.
sys.path.insert(0, str(ROOT / "my_method/rgr1_sammlv"))
from run_rgr1_nested_loso import MAJORITY_EMOTION, RECOGNITION_SUMMARY, PAPER_METRICS_PATH

ALPHAS = (0.30, 0.40, 0.50, 0.60, 0.70)
CACHE_HASH = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
METRIC_KEYS = ("TP", "FP", "FN", "Precision", "Recall", "F1", "Recognition_F1", "STRS")
EXPECTED = dict(TP=49, FP=143, FN=110, event_count=192,
                F1=0.2792022792022792, Recognition_F1=0.699,
                STRS=0.19516239316239314)


def reconstruct(curve, candidates, k, alpha):
    """Pure score-only inference; metadata is copied verbatim, never used."""
    assert alpha in ALPHAS and k >= 1
    assert len(curve) and np.isfinite(curve).all()
    output = []
    for original in candidates:
        p = original["peak"]
        lo, hi = max(0, p - 2*k), min(len(curve)-1, p + 2*k)
        background = float(np.min(curve[lo:hi+1]))
        theta = background + alpha * (float(curve[p]) - background)
        left = right = p
        while left > lo and curve[left-1] >= theta:
            left -= 1
        while right < hi and curve[right+1] >= theta:
            right += 1
        candidate = {**original, "onset": min(original["onset"], left),
                     "offset": max(original["offset"], right)}
        assert lo <= candidate["onset"] <= original["onset"] <= p
        assert p <= original["offset"] <= candidate["offset"] <= hi
        assert 0 <= original["onset"] - candidate["onset"] <= k
        assert 0 <= candidate["offset"] - original["offset"] <= k
        assert {a: b for a, b in candidate.items() if a not in ("onset", "offset")} == {
            a: b for a, b in original.items() if a not in ("onset", "offset")}
        output.append(candidate)
    assert len(output) == len(candidates)
    assert [e["candidate_id"] for e in output] == [e["candidate_id"] for e in candidates]
    return output


def prepare_native(record, config):
    # The existing native decoder receives score only, never the GT record.
    native_config = {"c_s": config["c_s"], "p": config["p_s"],
                     "c_d": config["c_d"], "c_b": config["c_b"]}
    decoded = tuned_native_decode({"score": record["score"]}, native_config)
    k = max(1, int(round(config["c_b"] * K_P)))
    events = []
    for i, event in enumerate(decoded["events"]):
        events.append({**event,
            "candidate_id": f"{record['subject']}/{record['video']}/pred_{i}",
            "raw_native_onset": event["onset"], "raw_native_offset": event["offset"],
            "onset": max(0, event["onset"]),
            "offset": min(len(decoded["curve"])-1, event["offset"]),
            "confidence": float(decoded["curve"][event["peak"]]),
            "emotion_id": int(MAJORITY_EMOTION(record["emotion"], event["onset"],
                                               event["offset"], event["peak"]))})
    return {"curve": decoded["curve"], "events": events, "k": k}


def evaluate(records, bank):
    counts = empty_counts()
    true_labels, labels, details = [], [], []
    for record in records:
        events = bank[(str(record["subject"]), str(record["video"]))]
        result = evaluate_decoding(record, {"events": events})
        add_counts(counts, result)
        for event, match in zip(events, result["matches"]):
            details.append({"subject": str(record["subject"]), "video": str(record["video"]),
                            **event, "matched_gt": int(match)})
            if match >= 0:
                gt = {"negative": 0, "positive": 1, "surprise": 2, "others": 3}.get(
                    str(record["gt_emotions"][match]), 3)
                if gt != 3:
                    true_labels.append(gt)
                    labels.append(event["emotion_id"])
    rec = float(RECOGNITION_SUMMARY(true_labels, labels, [0, 1, 2])["f1_score"])
    return {**metrics(counts), "Recognition_F1": rec, "STRS": f1(counts)*rec}, details


def select_alpha(training_records, held, all_subjects, native_bank):
    """Pool the 28 validation subjects; no fitted component in this algorithm."""
    training_subjects = sorted({str(r["subject"]) for r in training_records})
    assert set(training_subjects) == set(all_subjects) - {held}
    assert held not in training_subjects
    scores = []
    validation_rows = []
    for alpha in ALPHAS:
        pooled = empty_counts()
        for validation_subject in training_subjects:
            inner_train = set(training_subjects) - {validation_subject}
            assert held not in inner_train and len(inner_train) == 27
            counts = empty_counts()
            for record in training_records:
                if str(record["subject"]) != validation_subject:
                    continue
                native = native_bank[(str(record["subject"]), str(record["video"]))]
                events = reconstruct(native["curve"], native["events"], native["k"], alpha)
                add_counts(counts, evaluate_decoding(record, {"events": events}))
            add_counts(pooled, counts)
            validation_rows.append({"outer_subject": held, "inner_validation_subject": validation_subject,
                "inner_training_subjects": ";".join(sorted(inner_train)), "alpha": alpha,
                **{key: metrics(counts)[key] for key in ("TP", "FP", "FN", "F1")}})
        denominator = 2*pooled["TP"] + pooled["FP"] + pooled["FN"]
        exact_f1 = Fraction(2*pooled["TP"], denominator) if denominator else Fraction(0)
        scores.append((exact_f1, alpha, metrics(pooled)))
    _, alpha, inner = max(scores, key=lambda row: (row[0], row[1]))
    return alpha, inner, validation_rows


def write_csv(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def posthoc(records, native_details, pcbr_details, selected, oracle_dir):
    with (oracle_dir / "oracle_recoverable_cases.csv").open() as handle:
        oracle = list(csv.DictReader(handle))
    with (oracle_dir / "candidate_gt_pairs.csv").open() as handle:
        cohort_rows = list(csv.DictReader(handle))
    cohorts = {row["candidate_id"]: row["cohort"] for row in cohort_rows}
    assert len(oracle) == 11 and len({r["candidate_id"] for r in oracle}) == 11
    assert sum(v == "B4_PURE_FP" for v in cohorts.values()) == 131
    natives = {r["candidate_id"]: r for r in native_details}
    pcbrs = {r["candidate_id"]: r for r in pcbr_details}
    lookup = {(str(r["subject"]), str(r["video"])): r for r in records}
    assert set(natives) == set(pcbrs) == set(cohorts)
    changes, preservation, pure_conversions = [], [], []
    for identity, native in natives.items():
        pcbr = pcbrs[identity]
        record = lookup[(native["subject"], native["video"])]
        ni, pi = native["matched_gt"], pcbr["matched_gt"]
        # One common reference GT for both IoUs. Native TP uses its original
        # matched GT; FP uses Native best-IoU GT (diagnostic only).
        ref = ni if ni >= 0 else max(range(len(record["samples"])), default=-1,
            key=lambda j: (interval_iou(native, record["samples"][j]), j))
        old_iou = interval_iou(native, record["samples"][ref]) if ref >= 0 else None
        new_iou = interval_iou(pcbr, record["samples"][ref]) if ref >= 0 else None
        row = {"subject": native["subject"], "video": native["video"], "candidate_id": identity,
            "raw_native_onset": native["raw_native_onset"], "raw_native_offset": native["raw_native_offset"],
            "native_onset": native["onset"], "peak": native["peak"], "native_offset": native["offset"],
            "pcbr_onset": pcbr["onset"], "pcbr_offset": pcbr["offset"],
            "left_extension": native["onset"]-pcbr["onset"],
            "right_extension": pcbr["offset"]-native["offset"],
            "alpha": selected[native["subject"]], "confidence": native["confidence"],
            "native_emotion_id": native["emotion_id"], "pcbr_emotion_id": pcbr["emotion_id"],
            "native_iou_if_available_for_posthoc": old_iou,
            "pcbr_iou_if_available_for_posthoc": new_iou, "posthoc_reference_gt_index": ref,
            "native_matched_gt": ni, "pcbr_matched_gt": pi,
            "native_match_status": "TP" if ni >= 0 else "FP",
            "pcbr_match_status": "TP" if pi >= 0 else "FP", "native_cohort": cohorts[identity]}
        changes.append(row)
        if ni >= 0:
            preservation.append({**row, "native_tp_preserved": pi >= 0, "native_tp_lost": pi < 0,
                "native_tp_iou_improved": new_iou > old_iou,
                "native_tp_iou_reduced_but_still_tp": new_iou < old_iou and pi >= 0,
                "native_tp_iou_unchanged": new_iou == old_iou,
                "same_gt_match_preserved": pi == ni,
                "pcbr_iou_to_native_gt_ge_0_5": new_iou >= 0.5})
        if cohorts[identity] == "B4_PURE_FP" and pi >= 0:
            pure_conversions.append(row)
    tracking = []
    for case in oracle:
        identity = case["candidate_id"]
        n, p = natives[identity], pcbrs[identity]
        assert n["matched_gt"] < 0
        assert (n["raw_native_onset"], n["peak"], n["raw_native_offset"]) == (
            int(case["pred_onset"]), int(case["pred_peak"]), int(case["pred_offset"]))
        gt_index = int(case["nearest_gt_id"].rsplit("_", 1)[1])
        tracking.append({"candidate_id": identity, "subject": n["subject"], "video": n["video"],
            "oracle_gt_id": case["nearest_gt_id"], "actual_recovered": p["matched_gt"] == gt_index,
            "became_tp": p["matched_gt"] >= 0, "still_fp": p["matched_gt"] < 0,
            "pcbr_matched_gt": p["matched_gt"], "pcbr_iou_to_oracle_gt": interval_iou(p,
                lookup[(n["subject"], n["video"])]["samples"][gt_index])})
    facts = {
        "original_native_tp": len(preservation),
        "preserved_native_TP_count": sum(r["native_tp_preserved"] for r in preservation),
        "lost_native_TP_count": sum(r["native_tp_lost"] for r in preservation),
        "native_tp_iou_improved": sum(r["native_tp_iou_improved"] for r in preservation),
        "native_tp_iou_reduced_but_still_tp": sum(r["native_tp_iou_reduced_but_still_tp"] for r in preservation),
        "native_tp_iou_unchanged": sum(r["native_tp_iou_unchanged"] for r in preservation),
        "native_tp_matched_gt_changed": sum(r["native_tp_preserved"] and not r["same_gt_match_preserved"] for r in preservation),
        "oracle_recoverable_11_recovered_by_pcbr": sum(r["actual_recovered"] for r in tracking),
        "oracle_11_still_fp": sum(r["still_fp"] for r in tracking),
        "oracle_11_became_tp": sum(r["became_tp"] for r in tracking),
        "oracle_recovered_subjects": sorted({r["subject"] for r in tracking if r["actual_recovered"]}),
        "pure_fp_to_tp_by_expansion": len(pure_conversions),
        "FP_to_TP_total": sum(r["native_match_status"] == "FP" and r["pcbr_match_status"] == "TP" for r in changes),
        "changed_boundary_candidates": sum(r["left_extension"] + r["right_extension"] > 0 for r in changes),
    }
    assert facts["original_native_tp"] == 49
    return changes, preservation, tracking, pure_conversions, facts


def markdown_report(report, summary, fixed, deltas, tracking, output):
    facts = report["posthoc"]
    lines = ["# PCBR-1 SAMMLV Nested LOSO 实验报告", "", "## A. Integrity", "",
        f"- Anchor: PASS；{report['anchor']}",
        f"- Cache SHA-256: `{report['cache_sha256']}`",
        "- 29/29 outer folds 完成；Native 与 PCBR 均 192 candidates；identity、peak、confidence、emotion 固定。",
        "- 每折 alpha 选择仅访问该折 28 个训练 subject；逐 inner validation 计数见 inner_validation_metrics.csv。",
        "- k 来自各折 Native c_b；复用原 smoothing；无 GT 输入 inference。",
        "- 一个 Native onset=-1 按请求裁剪为 0；已验证 anchor 及匹配均不变。",
        "- emotion 按原始 Native 区间固定；Rec F1 的变化只来自匹配集合变化。",
        "", "## B. Main results", "",
        "| Method | TP | FP | FN | Precision | Recall | Spot F1 | Rec F1 | STRS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['Method']} | " + " | ".join(str(row[k]) if k in ("TP", "FP", "FN")
            else f"{row[k]:.6f}" for k in METRIC_KEYS) + " |")
    lines += ["", f"Delta (PCBR − Native): `{report['delta']}`", "", "## C. Native TP preservation", "",
        f"- Original: 49；preserved: {facts['preserved_native_TP_count']}；lost: {facts['lost_native_TP_count']}。",
        f"- 同一原 matched GT：IoU improved {facts['native_tp_iou_improved']}；reduced but still TP {facts['native_tp_iou_reduced_but_still_tp']}；unchanged {facts['native_tp_iou_unchanged']}。",
        f"- 保留 TP 中 matched GT 变化：{facts['native_tp_matched_gt_changed']}。",
        "", "## D. Boundary recovery", "",
        f"- Oracle 11：实际匹配原 oracle GT {facts['oracle_recoverable_11_recovered_by_pcbr']}；still FP {facts['oracle_11_still_fp']}；became TP {facts['oracle_11_became_tp']}。",
        f"- Recovered subjects: {facts['oracle_recovered_subjects']}。",
        "- Oracle 是允许更广泛边界修改的 GT 上界；PCBR outward-only 与 k cap 更严格，11 不是本方法必能达到的数量。",
        "", "| Candidate identity | Recovered oracle GT | Became TP |", "|---|---|---|"]
    lines += [f"| {r['candidate_id']} | {r['actual_recovered']} | {r['became_tp']} |" for r in tracking]
    lines += ["", "## E. Other conversions", "",
        f"- B4 pure FP → TP: {facts['pure_fp_to_tp_by_expansion']}；TP → FP: {facts['lost_native_TP_count']}；FP → TP total: {facts['FP_to_TP_total']}。",
        f"- Boundary changed: {facts['changed_boundary_candidates']} / 192。",
        "", "## F. Alpha selections", "", "| Alpha | Folds |", "|---|---:|"]
    lines += [f"| {a:.2f} | {report['selected_alpha_frequency'][str(a)]} |" for a in ALPHAS]
    lines += ["", "## G. Fixed-alpha sensitivity", "", "这些完整缓存结果仅用于诊断，未用于修改选择或规则。", "",
        "| Alpha | TP | FP | FN | Spot F1 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['alpha']:.2f} | {r['TP']} | {r['FP']} | {r['FN']} | {r['F1']:.6f} |" for r in fixed]
    lines += ["", "## H. Subject-level effects", ""]
    for name, sign in (("Positive", 1), ("Negative", -1), ("Zero", 0)):
        ids = [r["subject"] for r in deltas if (r["delta_F1"] > 0)-(r["delta_F1"] < 0) == sign]
        lines.append(f"- {name}: {', '.join(ids) or '无'}。")
    lines += ["", "只报告本次预定义规则的观察结果；GO / NO-GO 留待人工判断。未执行 bootstrap 或后续实验。",
              "", "## I. Files", ""]
    files = sorted(output.iterdir())
    lines += [f"- `{p.resolve()}`" for p in files]
    lines += [f"- `{(output / 'PCBR1_EXPERIMENT_REPORT_CN.md').resolve()}`",
              f"- `{Path(__file__).resolve()}`", f"- `{(HERE / 'PROTOCOL_CN.md').resolve()}`"]
    (output / "PCBR1_EXPERIMENT_REPORT_CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    output = ROOT / "results" / ("pcbr1_sammlv_nested" if args.mode == "full" else "pcbr1_sammlv_smoke_006")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    cache = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
    source = ROOT / "results/rgr1_sammlv_nested/report.json"
    oracle_dir = ROOT / "results/tuned_native_boundary_oracle_diagnostic"
    provenance = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), HERE / "PROTOCOL_CN.md", source, PAPER_METRICS_PATH,
        ROOT / "my_method/multi_scale_candidate_rescue/run_mscr_nested_loso.py",
        ROOT / "my_method/rgr1_sammlv/run_rgr1_nested_loso.py")}
    if args.mode == "full":
        smoke = json.loads((ROOT / "results/pcbr1_sammlv_smoke_006/report.json").read_text())
        assert smoke["smoke_pass"] and smoke["provenance"] == provenance
        assert smoke["cache_sha256"] == CACHE_HASH
    assert sha256(cache) == CACHE_HASH
    with cache.open("rb") as handle:
        payload = pickle.load(handle)
    records = list(payload["records"])
    assert payload["dataset"] == "SAMMLV" and payload["k_p"] == K_P
    for record in records:
        record["score"] = np.asarray(record["score"])
        record["score"].flags.writeable = False
    subjects = sorted({str(r["subject"]) for r in records})
    assert len(subjects) == 29 and len(records) == 79
    assert sum(len(r["samples"]) for r in records) == 159
    saved = json.loads(source.read_text())
    assert not saved["incomplete"] and saved["cache_sha256"] == CACHE_HASH
    folds = {f["subject"]: f for f in saved["outer_folds"]}
    assert set(folds) == set(subjects)
    for subject in subjects:
        assert set(folds[subject]["train_subjects"]) == set(subjects)-{subject}
    native_by_config = {}
    for fold in folds.values():
        config = fold["selected_strong_config"]
        key = json.dumps(config, sort_keys=True)
        if key not in native_by_config:
            native_by_config[key] = {(str(r["subject"]), str(r["video"])): prepare_native(r, config) for r in records}
    native_bank = {}
    for record in records:
        subject, video = str(record["subject"]), str(record["video"])
        key = json.dumps(folds[subject]["selected_strong_config"], sort_keys=True)
        native_bank[(subject, video)] = native_by_config[key][(subject, video)]["events"]
    anchor, native_details = evaluate(records, native_bank)
    assert all(anchor[k] == value for k, value in EXPECTED.items()), anchor
    # Explicit raw-vs-clipped anchor, matching, and per-fold source checks.
    raw_bank = {key: [{**e, "onset": e["raw_native_onset"], "offset": e["raw_native_offset"]}
                     for e in events] for key, events in native_bank.items()}
    raw_anchor, raw_details = evaluate(records, raw_bank)
    assert all(raw_anchor[k] == anchor[k] for k in METRIC_KEYS)
    assert [r["matched_gt"] for r in raw_details] == [r["matched_gt"] for r in native_details]
    for subject in subjects:
        actual, _ = evaluate([r for r in records if str(r["subject"]) == subject], native_bank)
        expected = folds[subject]["outer_metrics"]["Tuned Native"]
        assert all(actual[k] == expected[k] for k in METRIC_KEYS)
    print(f"ANCHOR PASS {anchor}", flush=True)
    requested = ["006"] if args.mode == "smoke" else subjects
    selected, outer_rows, inner_rows, nested_bank, fold_reports = {}, [], [], {}, []
    for held in requested:
        config = folds[held]["selected_strong_config"]
        bank = native_by_config[json.dumps(config, sort_keys=True)]
        train = [r for r in records if str(r["subject"]) != held]
        alpha, inner, validation = select_alpha(train, held, subjects, bank)
        selected[held] = alpha
        inner_rows.extend(validation)
        test = [r for r in records if str(r["subject"]) == held]
        for record in test:
            key = (held, str(record["video"]))
            native = bank[key]
            nested_bank[key] = reconstruct(native["curve"], native["events"], native["k"], alpha)
        native_metric, _ = evaluate(test, native_bank)
        pcbr_metric, _ = evaluate(test, nested_bank)
        assert native_metric["event_count"] == pcbr_metric["event_count"]
        for name, value in (("Tuned Native", native_metric), ("PCBR-1 Nested", pcbr_metric)):
            outer_rows.append({"outer_subject": held, "selected_alpha": alpha,
                "inner_F1": inner["F1"], "Method": name, **{k: value[k] for k in METRIC_KEYS}})
        fold_reports.append({"outer_subject": held, "train_subjects": sorted(set(subjects)-{held}),
            "native_config": config, "selected_alpha": alpha, "inner_metrics": inner,
            "native_metrics": native_metric, "pcbr_metrics": pcbr_metric})
        print(f"FOLD {held} alpha={alpha} inner={inner['F1']:.6f} complete", flush=True)
    assert len(fold_reports) == len(requested)
    output.mkdir(parents=True)
    common = {"experiment": "PCBR-1", "mode": args.mode, "cache_path": str(cache),
        "cache_sha256": CACHE_HASH, "provenance": provenance, "anchor": anchor,
        "alpha_grid": ALPHAS, "alpha_grid_source": "User PCBR-1 request, 2026-09-05",
        "inner_protocol": "pooled F1 on 28 inner validation subjects, fixed outer Native config, no fitting",
        "tie_break": "higher exact rational pooled F1, then higher alpha",
        "completed_fold_count": len(fold_reports), "outer_folds": fold_reports,
        "invariants": {"A_count": True, "B_peak": True, "C_identity": True,
            "D_outward_onset": True, "E_outward_offset": True, "F_cap_k": True,
            "G_outer_test_excluded": True, "H_frozen_cache": True,
            "I_all_29_folds": len(fold_reports) == 29,
            "J_GT_free_inference": True, "confidence_and_emotion_unchanged": True},
        "no_backbone_forward": True, "no_training": True, "no_bootstrap": True}
    write_csv(output / "outer_fold_metrics.csv", outer_rows)
    write_csv(output / "inner_validation_metrics.csv", inner_rows)
    if args.mode == "smoke":
        smoke_records = [r for r in records if str(r["subject"]) == "006"]
        _, smoke_predictions = evaluate(smoke_records, nested_bank)
        write_csv(output / "predictions.csv", smoke_predictions)
        dump(output / "report.json", {**common, "smoke_pass": True})
        print("SMOKE PASS", flush=True)
        return
    # All outer selections are now frozen. Full-cache fixed-alpha controls and
    # oracle GT diagnostics occur strictly after this point.
    pcbr_metric, pcbr_details = evaluate(records, nested_bank)
    assert pcbr_metric["event_count"] == anchor["event_count"] == 192
    fixed_rows, fixed_predictions = [], []
    for alpha in ALPHAS:
        fixed_bank = {}
        for record in records:
            key = (str(record["subject"]), str(record["video"]))
            config_key = json.dumps(folds[key[0]]["selected_strong_config"], sort_keys=True)
            native = native_by_config[config_key][key]
            fixed_bank[key] = reconstruct(native["curve"], native["events"], native["k"], alpha)
        result, details = evaluate(records, fixed_bank)
        assert result["event_count"] == 192
        fixed_rows.append({"alpha": alpha, **{k: result[k] for k in METRIC_KEYS}})
        fixed_predictions.extend({"alpha": alpha, **r} for r in details)
    changes, preservation, tracking, pure, facts = posthoc(records, native_details, pcbr_details, selected, oracle_dir)
    assert facts["FP_to_TP_total"]-facts["lost_native_TP_count"] == pcbr_metric["TP"]-anchor["TP"]
    frequency = Counter(selected.values())
    deltas = [{"subject": f["outer_subject"], **{
        f"delta_{k}": f["pcbr_metrics"][k]-f["native_metrics"][k] for k in METRIC_KEYS}} for f in fold_reports]
    summary = [{"Method": name, **{k: value[k] for k in METRIC_KEYS}}
               for name, value in (("Tuned Native", anchor), ("PCBR-1 Nested", pcbr_metric))]
    report = {**common, "aggregate_metrics": {"Tuned Native": anchor, "PCBR-1 Nested": pcbr_metric},
        "delta": {k: pcbr_metric[k]-anchor[k] for k in METRIC_KEYS}, "posthoc": facts,
        "selected_alpha_frequency": {str(a): frequency[a] for a in ALPHAS},
        "fixed_alpha_summary": fixed_rows,
        "oracle_posthoc_file_hashes": {str((oracle_dir / n).resolve()): sha256(oracle_dir / n)
            for n in ("oracle_recoverable_cases.csv", "candidate_gt_pairs.csv")},
        "decision": "MANUAL_REVIEW_REQUIRED", "incomplete": False}
    assert sha256(cache) == CACHE_HASH
    dump(output / "report.json", report)
    for filename, rows in (("summary.csv", summary), ("fixed_alpha_summary.csv", fixed_rows),
        ("candidate_boundary_changes.csv", changes), ("native_tp_preservation.csv", preservation),
        ("oracle_recoverable_tracking.csv", tracking), ("subject_level_delta.csv", deltas),
        ("nested_predictions.csv", pcbr_details), ("native_predictions.csv", native_details),
        ("fixed_alpha_predictions.csv", fixed_predictions),
        ("selected_alpha_frequency.csv", [{"alpha": a, "fold_count": frequency[a]} for a in ALPHAS])):
        write_csv(output / filename, rows)
    write_csv(output / "pure_fp_to_tp.csv", pure, fields=list(changes[0]))
    markdown_report(report, summary, fixed_rows, deltas, tracking, output)
    print(json.dumps({"main": summary, "posthoc": facts, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
