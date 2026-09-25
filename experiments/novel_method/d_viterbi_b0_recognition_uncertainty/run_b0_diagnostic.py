#!/usr/bin/env python3
"""B0 frozen recognition-uncertainty separability diagnostic on SAMMLV."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/d_viterbi_b0_recognition_uncertainty"
CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
SOURCE_REPORT = ROOT / "results/rgr1_sammlv_nested/report.json"
COHORT_REPORT = ROOT / "results/tuned_native_boundary_oracle_diagnostic/report.json"
COHORT_PAIRS = COHORT_REPORT.with_name("candidate_gt_pairs.csv")
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
EVALUATOR_SOURCE = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"

spec = importlib.util.spec_from_file_location("native", NATIVE_SOURCE)
native = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(native)

EPS = 1e-8
EXPECTED_SHA = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
DESCRIPTORS = {
    "E1_-MeanEntropy": "E1",
    "E2_MeanNonNeutral": "E2",
    "E3_-MeanNeutral": "E3",
    "E4_MeanMargin": "E4",
}
QUANTITIES = ("entropy", "p_neutral", "p_nonneutral_max", "active_margin")
COMPARISONS = ("TP_vs_All_FP", "TP_vs_Pure_FP")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                     default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n",
                          encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configs_from_report(report):
    configs = {}
    for fold in report["outer_folds"]:
        cfg = fold["selected_strong_config"]
        configs[str(fold["subject"])] = {"c_s": float(cfg["c_s"]), "p": float(cfg["p_s"]),
                                        "c_d": float(cfg["c_d"]), "c_b": float(cfg["c_b"])}
    return configs


def cohort_manifest():
    manifest = {}
    with COHORT_PAIRS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identity = row["candidate_id"]
            stable = {key: row[key] for key in ("candidate_id", "cohort", "formal_match_status", "subject", "video",
                                                  "onset_pred", "peak_pred", "offset_pred", "outer_fold")}
            if identity in manifest and manifest[identity] != stable:
                raise RuntimeError(f"inconsistent cohort rows for {identity}")
            manifest[identity] = stable
    return manifest


def metric_rows(rows, comparison):
    negatives = [row for row in rows if row["label"] == "FP" and
                 (comparison == "TP_vs_All_FP" or row["cohort"] == "B4_PURE_FP")]
    positives = [row for row in rows if row["label"] == "TP"]
    output = []
    for descriptor in DESCRIPTORS:
        tp = np.asarray([row[descriptor] for row in positives], dtype=float)
        fp = np.asarray([row[descriptor] for row in negatives], dtype=float)
        labels = np.r_[np.ones(len(tp), dtype=int), np.zeros(len(fp), dtype=int)]
        scores = np.r_[tp, fp]
        output.append({"comparison": comparison, "descriptor": descriptor, "TP_count": len(tp), "FP_count": len(fp),
                       "positive_prevalence": len(tp) / len(scores), "TP_median": float(np.median(tp)),
                       "FP_median": float(np.median(fp)), "TP_mean": float(np.mean(tp)), "FP_mean": float(np.mean(fp)),
                       "median_difference_TP_minus_FP": float(np.median(tp) - np.median(fp)),
                       "mean_difference_TP_minus_FP": float(np.mean(tp) - np.mean(fp)),
                       "effect_direction": "TP_HIGHER" if np.median(tp) > np.median(fp) else "TP_LOWER" if np.median(tp) < np.median(fp) else "EQUAL",
                       "ROC_AUC": float(roc_auc_score(labels, scores)),
                       "PR_AUC": float(average_precision_score(labels, scores))})
    return output


def loo_robustness(rows):
    subjects = sorted({row["subject"] for row in rows})
    fold_rows, summary = [], {}
    for comparison in COMPARISONS:
        summary[comparison] = {}
        for descriptor in DESCRIPTORS:
            roc_values, pr_values = [], []
            for held in subjects:
                retained = [row for row in rows if row["subject"] != held and
                            (row["label"] == "TP" or (row["label"] == "FP" and
                             (comparison == "TP_vs_All_FP" or row["cohort"] == "B4_PURE_FP")))]
                labels = np.asarray([row["label"] == "TP" for row in retained], dtype=int)
                scores = np.asarray([row[descriptor] for row in retained], dtype=float)
                roc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
                pr = float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None
                fold_rows.append({"comparison": comparison, "descriptor": descriptor, "held_out_subject": held,
                                  "candidate_count": len(retained), "TP_count": int(labels.sum()),
                                  "FP_count": int(len(labels) - labels.sum()), "ROC_AUC": roc, "PR_AUC": pr})
                if roc is not None:
                    roc_values.append(roc)
                    pr_values.append(pr)
            roc_array, pr_array = np.asarray(roc_values), np.asarray(pr_values)
            summary[comparison][descriptor] = {"valid_folds": len(roc_array), "LOO_ROC_mean": float(roc_array.mean()),
                "LOO_ROC_median": float(np.median(roc_array)), "LOO_ROC_min": float(roc_array.min()),
                "LOO_ROC_max": float(roc_array.max()), "LOO_PR_mean": float(pr_array.mean()),
                "LOO_PR_median": float(np.median(pr_array)),
                "proportion_LOO_ROC_ge_0.65": float(np.mean(roc_array >= .65))}
    return fold_rows, summary


def subject_direction(rows):
    output, summary = [], {}
    subjects = sorted({row["subject"] for row in rows})
    for comparison in COMPARISONS:
        summary[comparison] = {}
        for descriptor in DESCRIPTORS:
            positive, negative, equal, evaluable, positive_ids, negative_ids = 0, 0, 0, 0, [], []
            for subject in subjects:
                tp = [row[descriptor] for row in rows if row["subject"] == subject and row["label"] == "TP"]
                fp = [row[descriptor] for row in rows if row["subject"] == subject and row["label"] == "FP" and
                      (comparison == "TP_vs_All_FP" or row["cohort"] == "B4_PURE_FP")]
                if not tp or not fp:
                    continue
                evaluable += 1
                difference = float(np.median(tp) - np.median(fp))
                if difference > 0:
                    positive += 1
                    positive_ids.append(subject)
                elif difference < 0:
                    negative += 1
                    negative_ids.append(subject)
                else:
                    equal += 1
            item = {"comparison": comparison, "descriptor": descriptor, "evaluable_subjects": evaluable,
                    "positive_direction_subjects": positive, "negative_direction_subjects": negative,
                    "equal_direction_subjects": equal, "positive_subject_ids": ";".join(positive_ids),
                    "negative_subject_ids": ";".join(negative_ids)}
            output.append(item)
            summary[comparison][descriptor] = item
    return output, summary


def temporal_profiles(rows, sequences):
    output, diagnostic = [], {}
    grid = np.linspace(0.0, 1.0, 21)
    for quantity in QUANTITIES:
        groups = {}
        for group, selected in (("TP", [r for r in rows if r["label"] == "TP"]),
                                ("Pure_FP", [r for r in rows if r["cohort"] == "B4_PURE_FP"])):
            profiles = []
            for row in selected:
                values = sequences[row["candidate_id"]][quantity]
                profiles.append(np.interp(grid, np.linspace(0.0, 1.0, len(values)), values))
            groups[group] = np.asarray(profiles)
        tp_mean, fp_mean = groups["TP"].mean(axis=0), groups["Pure_FP"].mean(axis=0)
        for position in range(21):
            output.append({"quantity": quantity, "normalized_position": position,
                           "normalized_coordinate": float(grid[position]), "TP_count": len(groups["TP"]),
                           "Pure_FP_count": len(groups["Pure_FP"]), "TP_mean": float(tp_mean[position]),
                           "Pure_FP_mean": float(fp_mean[position]),
                           "difference_TP_minus_Pure_FP": float(tp_mean[position] - fp_mean[position])})
        expected = (tp_mean < fp_mean) if quantity in ("entropy", "p_neutral") else (tp_mean > fp_mean)
        diagnostic[quantity] = {"expected_direction_positions": int(expected.sum()), "total_positions": 21,
                                "sustained_expected_direction_ge_16_of_21": bool(expected.sum() >= 16),
                                "mean_difference_TP_minus_Pure_FP": float(np.mean(tp_mean - fp_mean)),
                                "max_absolute_position_difference": float(np.max(np.abs(tp_mean - fp_mean)))}
    return output, diagnostic


def render_table(summary_rows, best=None):
    lines = ["| Descriptor | TP median | FP median | TP mean | FP mean | ROC-AUC | PR-AUC |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        name = f"**{row['descriptor']}**" if row["descriptor"] == best else row["descriptor"]
        lines.append(f"| {name} | {row['TP_median']:.6f} | {row['FP_median']:.6f} | {row['TP_mean']:.6f} | {row['FP_mean']:.6f} | {row['ROC_AUC']:.6f} | {row['PR_AUC']:.6f} |")
    return lines


def render_report(report, all_summary, pure_summary, near_summary):
    best = report["evidence_grade"]["best_descriptor"]
    robust = report["loo_summary"]["TP_vs_Pure_FP"]
    direction = report["subject_direction_summary"]["TP_vs_Pure_FP"]
    lines = ["# B0 — Recognition-Uncertainty Separability Diagnostic", "",
             f"最终等级：`{report['final_grade']}`。", "", "## A. Integrity", ""]
    for key, item in report["integrity_assertions"].items():
        lines.append(f"- {key}: PASS — {item['detail']}")
    lines += ["", "## B. TP vs All FP", "", *render_table(all_summary), "", "## C. TP vs Pure FP", "", *render_table(pure_summary, best),
              "", "## D. Robustness", "", "| Descriptor | LOO ROC mean | median | min | max | LOO PR mean | LOO PR median | % ROC≥.65 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for descriptor in DESCRIPTORS:
        row = robust[descriptor]
        lines.append(f"| {descriptor} | {row['LOO_ROC_mean']:.6f} | {row['LOO_ROC_median']:.6f} | {row['LOO_ROC_min']:.6f} | {row['LOO_ROC_max']:.6f} | {row['LOO_PR_mean']:.6f} | {row['LOO_PR_median']:.6f} | {row['proportion_LOO_ROC_ge_0.65']:.6f} |")
    lines += ["", "## E. Subject Direction", ""]
    for descriptor in DESCRIPTORS:
        row = direction[descriptor]
        lines.append(f"- {descriptor}: positive/negative/equal/evaluable = {row['positive_direction_subjects']}/{row['negative_direction_subjects']}/{row['equal_direction_subjects']}/{row['evaluable_subjects']}。")
    overlap = report["counterexample_overlap"]
    lines += ["", "## F. Counterexamples", "",
              f"- Best descriptor：{best}。Pure FP 中 score≥TP median：{overlap['high_score_Pure_FP_count']}/{overlap['Pure_FP_count']} ({overlap['high_score_Pure_FP_fraction']:.2%})。",
              f"- TP 中 score≤Pure-FP median：{overlap['low_score_TP_count']}/{overlap['TP_count']} ({overlap['low_score_TP_fraction']:.2%})。",
              f"- 按预先写入的 20% descriptive overlap 标尺，high-certainty Pure FP / low-certainty TP 大量存在：{overlap['substantial_overlap_ge_20_percent']}。这只用于解释重叠，不参与 grade。",
              "", "## Near-GT FP", "", "| Descriptor | Count | Mean | Median | P25 | P75 |", "|---|---:|---:|---:|---:|---:|"]
    for row in near_summary:
        lines.append(f"| {row['descriptor']} | {row['count']} | {row['mean']:.6f} | {row['median']:.6f} | {row['p25']:.6f} | {row['p75']:.6f} |")
    lines += ["", "## G. Temporal profile", ""]
    for quantity, item in report["temporal_profile_diagnostic"].items():
        lines.append(f"- {quantity}: expected-direction positions={item['expected_direction_positions']}/21，sustained={item['sustained_expected_direction_ge_16_of_21']}，mean TP−PureFP={item['mean_difference_TP_minus_Pure_FP']:+.6f}。")
    lines += ["", "该 profile 仅作连续状态假设的描述，没有据此构造新 scalar feature。", "", "## H. Final Grade", "",
              f"`{report['final_grade']}`", "", f"- Best descriptor conditions：{report['evidence_grade']['conditions']}。",
              f"- Grade rationale：{report['evidence_grade']['rationale']}。", "",
              "已停止；未训练、未实现 Viterbi、未调 threshold/lambda/transition，未运行 CASME3。", "", "## Outputs", ""]
    for path in [OUT / "report.json", OUT / "candidate_recognition_descriptors.csv", OUT / "tp_vs_fp_summary.csv",
                 OUT / "tp_vs_pure_fp_summary.csv", OUT / "near_gt_fp_summary.csv", OUT / "loo_subject_robustness.csv",
                 OUT / "subject_direction_summary.csv", OUT / "normalized_temporal_profiles.csv",
                 OUT / "counterexample_high_score_fp.csv", OUT / "counterexample_low_score_tp.csv",
                 OUT / "D_VITERBI_B0_DIAGNOSTIC_CN.md", HERE / "run_b0_diagnostic.py"]:
        lines.append(f"- `{path.resolve()}`")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    cohort_report = json.loads(COHORT_REPORT.read_text(encoding="utf-8"))
    manifest = cohort_manifest()
    if len(manifest) != 192 or Counter(row["cohort"] for row in manifest.values()) != Counter({"A_NATIVE_TP": 49, "B4_PURE_FP": 131, "B2_PEAK_NEAR_GT": 11, "B1_PEAK_IN_GT": 1}):
        raise RuntimeError("boundary cohort identity gate failed")
    if sha256(CACHE) != EXPECTED_SHA:
        raise RuntimeError("frozen cache SHA mismatch")
    payload, records, subjects, observed = native.load_payload(CACHE)
    if observed != {"subjects": 29, "videos": 79, "gt": 159, "k_p": 5}:
        raise RuntimeError(f"cache metadata mismatch: {observed}")
    recognition = source_report["recognition_validation"]
    expected_recognition = {"logits_shape": "[T,5]", "softmax_axis": 1, "neutral_class_id": 4,
                            "class_order": ["negative", "positive", "surprise", "others", "neutral"]}
    if any(recognition[key] != value for key, value in expected_recognition.items()):
        raise RuntimeError("recognition metadata validation failed")
    configs = configs_from_report(source_report)
    rows, sequences, total = [], {}, native.counts_empty()
    clipped_intervals = 0
    generated_ids = set()
    for record in records:
        subject, video = str(record["subject"]), str(record["video"])
        score, logits = np.asarray(record["score"], dtype=float), np.asarray(record["logits"], dtype=float)
        if logits.shape != (len(score), 5) or not np.isfinite(logits).all() or not np.isfinite(score).all():
            raise RuntimeError(f"temporal alignment/finiteness failed: {subject}/{video}")
        probabilities = softmax(logits, axis=1)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError(f"softmax axis check failed: {subject}/{video}")
        events = native.tuned_decode(record, configs[subject], observed["k_p"])
        matches, _ = native.match_events(events, record["samples"])
        native.add_counts(total, native.evaluate(record, events))
        for candidate_index, (event, match) in enumerate(zip(events, matches)):
            identity = f"{subject}/{video}/pred_{candidate_index}"
            generated_ids.add(identity)
            source = manifest.get(identity)
            if source is None:
                raise RuntimeError(f"candidate identity missing from frozen cohort: {identity}")
            if (int(source["onset_pred"]), int(source["peak_pred"]), int(source["offset_pred"])) != (int(event["onset"]), int(event["peak"]), int(event["offset"])):
                raise RuntimeError(f"Native interval identity drift: {identity}")
            label = "TP" if match >= 0 else "FP"
            if source["formal_match_status"] != label:
                raise RuntimeError(f"official match status drift: {identity}")
            left, right = max(0, int(event["onset"])), min(len(logits) - 1, int(event["offset"]))
            clipped_intervals += int(left != int(event["onset"]) or right != int(event["offset"]))
            interval = probabilities[left:right + 1]
            entropy = -np.sum(interval * np.log(interval + EPS), axis=1)
            neutral = interval[:, 4]
            nonneutral = np.max(interval[:, :4], axis=1)
            margin = nonneutral - neutral
            descriptor = {"E1_-MeanEntropy": float(-np.mean(entropy)),
                          "E2_MeanNonNeutral": float(np.mean(nonneutral)),
                          "E3_-MeanNeutral": float(-np.mean(neutral)),
                          "E4_MeanMargin": float(np.mean(margin))}
            rows.append({"candidate_id": identity, "subject": subject, "video": video, "candidate_index": candidate_index,
                         "onset_native": int(event["onset"]), "peak_native": int(event["peak"]), "offset_native": int(event["offset"]),
                         "descriptor_read_onset_clipped": left, "descriptor_read_offset_clipped": right,
                         "interval_clipped_for_array_access": left != int(event["onset"]) or right != int(event["offset"]),
                         "label": label, "cohort": source["cohort"], **descriptor})
            sequences[identity] = {"entropy": entropy, "p_neutral": neutral,
                                   "p_nonneutral_max": nonneutral, "active_margin": margin}
    aggregate = native.metrics(total)
    anchor = source_report["aggregate_metrics"]["Tuned Native"]
    if (aggregate["TP"], aggregate["FP"], aggregate["FN"], aggregate["event_count"]) != (49, 143, 110, 192):
        raise RuntimeError(f"Tuned Native anchor failed: {aggregate}")
    if abs(anchor["Recognition_F1"] - .6990) > 1e-12 or abs(anchor["STRS"] - .19516239316239314) > 1e-14:
        raise RuntimeError("Recognition F1 / STRS source anchor failed")
    if generated_ids != set(manifest):
        raise RuntimeError("candidate membership changed")
    all_summary = metric_rows(rows, "TP_vs_All_FP")
    pure_summary = metric_rows(rows, "TP_vs_Pure_FP")
    near_rows = [row for row in rows if row["label"] == "FP" and row["cohort"] != "B4_PURE_FP"]
    near_summary = []
    for descriptor in DESCRIPTORS:
        values = np.asarray([row[descriptor] for row in near_rows])
        near_summary.append({"descriptor": descriptor, "count": len(values), "mean": float(values.mean()),
                             "median": float(np.median(values)), "p25": float(np.quantile(values, .25)),
                             "p75": float(np.quantile(values, .75)), "min": float(values.min()), "max": float(values.max())})
    loo_rows, loo_summary = loo_robustness(rows)
    direction_rows, direction_summary = subject_direction(rows)
    profiles, profile_diagnostic = temporal_profiles(rows, sequences)
    best_row = max(pure_summary, key=lambda row: (row["ROC_AUC"], row["PR_AUC"], row["descriptor"]))
    best = best_row["descriptor"]
    robustness = loo_summary["TP_vs_Pure_FP"][best]
    direction = direction_summary["TP_vs_Pure_FP"][best]
    conditions = {"ROC_AUC_ge_0.70": best_row["ROC_AUC"] >= .70, "PR_AUC_ge_0.40": best_row["PR_AUC"] >= .40,
                  "LOO_ROC_median_ge_0.68": robustness["LOO_ROC_median"] >= .68,
                  "proportion_LOO_ROC_ge_0.65_ge_0.75": robustness["proportion_LOO_ROC_ge_0.65"] >= .75,
                  "subject_direction_majority_correct": direction["positive_direction_subjects"] > direction["negative_direction_subjects"]}
    if all(conditions.values()):
        grade, rationale = "D-VITERBI-EVIDENCE-STRONG", "best descriptor satisfies all five pre-registered strong conditions"
    elif best_row["ROC_AUC"] >= .62 and conditions["subject_direction_majority_correct"]:
        grade, rationale = "D-VITERBI-EVIDENCE-WEAK", "best descriptor reaches the weak pooled-AUC range but not every strong condition"
    else:
        grade, rationale = "D-VITERBI-EVIDENCE-NO-GO", "all descriptors are below ROC-AUC 0.62 or best-descriptor subject direction collapses/reverses"
    pure_fp = [row for row in rows if row["cohort"] == "B4_PURE_FP"]
    tp = [row for row in rows if row["label"] == "TP"]
    tp_median, fp_median = np.median([row[best] for row in tp]), np.median([row[best] for row in pure_fp])
    high_fp_count = sum(row[best] >= tp_median for row in pure_fp)
    low_tp_count = sum(row[best] <= fp_median for row in tp)
    overlap = {"best_descriptor": best, "TP_count": len(tp), "Pure_FP_count": len(pure_fp),
               "high_score_Pure_FP_count": high_fp_count, "high_score_Pure_FP_fraction": high_fp_count / len(pure_fp),
               "low_score_TP_count": low_tp_count, "low_score_TP_fraction": low_tp_count / len(tp),
               "substantial_overlap_threshold": .20,
               "substantial_overlap_ge_20_percent": bool(high_fp_count / len(pure_fp) >= .20 or low_tp_count / len(tp) >= .20)}
    counterexample_high = sorted(pure_fp, key=lambda row: row[best], reverse=True)[:20]
    counterexample_low = sorted(tp, key=lambda row: row[best])[:20]
    identity_material = "\n".join(f"{row['candidate_id']}|{row['onset_native']}|{row['peak_native']}|{row['offset_native']}" for row in rows)
    integrity = {
        "A_Tuned_Native_anchor": {"status": "PASS", "detail": "49/143/110; 192 predictions; F1=0.279202; Recognition F1=0.6990; STRS=0.195162"},
        "B_candidate_count": {"status": "PASS", "detail": "192"},
        "C_candidate_identity_unchanged": {"status": "PASS", "detail": "generated identity set exactly equals frozen boundary cohort identity set"},
        "D_Native_interval_unchanged": {"status": "PASS", "detail": f"all onset/peak/offset exact; {clipped_intervals} intervals clipped only for legal array reads"},
        "E_recognition_output_frozen": {"status": "PASS", "detail": "cache logits only"},
        "F_spotting_output_frozen": {"status": "PASS", "detail": "cache score only"},
        "G_no_GT_in_descriptors": {"status": "PASS", "detail": "E1-E4 read probabilities only inside Native intervals"},
        "H_GT_only_retrospective_labels": {"status": "PASS", "detail": "official match/cohort labels only"},
        "I_no_new_prediction": {"status": "PASS", "detail": "descriptor rows map one-to-one to 192 Native predictions"},
        "J_no_training": {"status": "PASS", "detail": "no model fitting or classifier"},
        "K_no_Viterbi": {"status": "PASS", "detail": "no state decoder or transition computation"},
        "L_cache_SHA256": {"status": "PASS", "detail": EXPECTED_SHA},
    }
    report = {"status": "COMPLETE", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "final_grade": grade,
              "provenance": {"cache_path": str(CACHE.resolve()), "cache_sha256": EXPECTED_SHA,
                             "source_report": str(SOURCE_REPORT.resolve()), "source_report_sha256": sha256(SOURCE_REPORT),
                             "cohort_source": str(COHORT_PAIRS.resolve()), "cohort_source_sha256": sha256(COHORT_PAIRS),
                             "evaluator_path": str(EVALUATOR_SOURCE.resolve()), "evaluator_sha256": sha256(EVALUATOR_SOURCE),
                             "GT_source": "verified cache records[*].samples"},
              "anchor": {**aggregate, "Recognition_F1": anchor["Recognition_F1"], "STRS": anchor["STRS"]},
              "recognition_validation": {**expected_recognition, "non_neutral_class_ids": [0, 1, 2, 3],
                                         "temporal_alignment_pass": True, "records_checked": len(records)},
              "cohort_counts": dict(Counter(row["cohort"] for row in rows)),
              "candidate_identity_interval_sha256": hashlib.sha256(identity_material.encode()).hexdigest(),
              "TP_vs_All_FP": {row["descriptor"]: row for row in all_summary},
              "TP_vs_Pure_FP": {row["descriptor"]: row for row in pure_summary},
              "near_GT_FP": {row["descriptor"]: row for row in near_summary},
              "loo_summary": loo_summary, "subject_direction_summary": direction_summary,
              "temporal_profile_diagnostic": profile_diagnostic, "counterexample_overlap": overlap,
              "evidence_grade": {"best_descriptor": best, "conditions": conditions, "rationale": rationale},
              "integrity_assertions": integrity,
              "forbidden_actions": {"training": False, "classifier": False, "Viterbi": False, "threshold_sweep": False,
                                    "candidate_change": False, "boundary_or_peak_change": False, "Nested_LOSO": False,
                                    "CASME3": False, "bootstrap": False, "new_features": False}}
    write_csv(OUT / "candidate_recognition_descriptors.csv", rows)
    write_csv(OUT / "tp_vs_fp_summary.csv", all_summary)
    write_csv(OUT / "tp_vs_pure_fp_summary.csv", pure_summary)
    write_csv(OUT / "near_gt_fp_summary.csv", near_summary)
    write_csv(OUT / "loo_subject_robustness.csv", loo_rows)
    write_csv(OUT / "subject_direction_summary.csv", direction_rows)
    write_csv(OUT / "normalized_temporal_profiles.csv", profiles)
    write_csv(OUT / "counterexample_high_score_fp.csv", counterexample_high)
    write_csv(OUT / "counterexample_low_score_tp.csv", counterexample_low)
    dump(OUT / "report.json", report)
    (OUT / "D_VITERBI_B0_DIAGNOSTIC_CN.md").write_text(render_report(report, all_summary, pure_summary, near_summary), encoding="utf-8")
    print(json.dumps({"final_grade": grade, "best_descriptor": best, "best_metrics": best_row,
                      "best_robustness": robustness, "best_subject_direction": direction,
                      "counterexample_overlap": overlap, "temporal_profile": profile_diagnostic}, ensure_ascii=False, indent=2,
                     default=lambda item: item.item() if isinstance(item, np.generic) else str(item)))


if __name__ == "__main__":
    main()
