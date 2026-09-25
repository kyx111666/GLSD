#!/usr/bin/env python3
"""Expanded Native baseline fairness audit (CASME3 + optional SAMMLV).

Cache-only.  This script never imports or runs Morphology Rescue fitting.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import pickle
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/expanded_native_fairness_audit"
CASME3_CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
SAMMLV_CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
MORPH_SUMMARY = ROOT / "results/morphology_casme3_locked_validation/outputs/casme3_morph_summary.json"
SAMMLV_TUNED_REPORT = ROOT / "results/rgr1_sammlv_nested/report.json"
EXPECTED_CASME3_SHA = "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda"
EXPECTED_SAMMLV_SHA = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
EXPECTED_AUTHOR_CASME3 = (81, 912, 777)
EXPECTED_OLD_TUNED_CASME3 = (111, 1079, 747)
EXPECTED_AUTHOR_SAMMLV = (53, 184, 106)
EXPECTED_OLD_TUNED_SAMMLV = (49, 143, 110)
OLD_TUNED = {"c_s": 1.5, "p": 0.45, "c_d": 0.75, "c_b": 0.75}
EXPANDED_CS = (1.0, 1.5, 2.0, 2.5)
EXPANDED_P = (0.35, 0.45, 0.55, 0.65)
EXPANDED_CD = (0.5, 0.75, 1.0, 1.25)
EXPANDED_CB = (0.5, 0.75, 1.0, 1.25)
SEED = 20260905


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def moving_average(score, width):
    width = max(1, int(width))
    return np.convolve(np.asarray(score, dtype=np.float64), np.ones(width) / width, mode="same")


def threshold(curve, p):
    curve = np.asarray(curve, dtype=np.float64)
    return float(curve.mean() + float(p) * (float(curve.max()) - float(curve.mean())))


def event(peak, boundary, source):
    return {"peak": int(peak), "onset": int(peak - boundary), "offset": int(peak + boundary), "source": source}


def interval_iou(pred, sample):
    left, right = int(pred["onset"]), int(pred["offset"])
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def match_events(events, samples):
    unmatched = set(range(len(samples)))
    matches = []
    for pred in events:
        choices = [(interval_iou(pred, samples[index]), index) for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            matches.append(int(best_index))
        else:
            matches.append(-1)
    return matches, unmatched


def counts_empty():
    return {key: 0 for key in ("TP", "FP", "FN", "event_count")}


def add_counts(target, source):
    for key in target:
        target[key] += int(source[key])


def f1(counts):
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return 2 * counts["TP"] / denominator if denominator else 0.0


def metrics(counts):
    tp, fp, fn = (int(counts["TP"]), int(counts["FP"]), int(counts["FN"]))
    return {"TP": tp, "FP": fp, "FN": fn, "event_count": int(counts["event_count"]),
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn) if tp + fn else 0.0,
            "F1": f1(counts)}


def author_decode(record, k_p):
    curve = moving_average(record["score"], 2 * k_p)
    tau = threshold(curve, 0.55)
    peaks = find_peaks(curve, height=tau, distance=k_p)[0].astype(int)
    return [event(peak, k_p, "native") for peak in peaks]


def tuned_decode(record, config, k_p):
    width = max(1, int(round(float(config["c_s"]) * k_p)))
    distance = max(1, int(round(float(config["c_d"]) * k_p)))
    boundary = max(1, int(round(float(config["c_b"]) * k_p)))
    curve = moving_average(record["score"], width)
    tau = threshold(curve, float(config["p"]))
    peaks = find_peaks(curve, height=tau, distance=distance)[0].astype(int)
    return [event(peak, boundary, "tuned_native") for peak in peaks]


def evaluate(record, events):
    matches, unmatched = match_events(events, record["samples"])
    counts = counts_empty()
    counts["TP"] = sum(match >= 0 for match in matches)
    counts["FP"] = sum(match < 0 for match in matches)
    counts["FN"] = len(unmatched)
    counts["event_count"] = len(events)
    return counts


def grid():
    return [dict(zip(("c_s", "p", "c_d", "c_b"), values))
            for values in itertools.product(EXPANDED_CS, EXPANDED_P, EXPANDED_CD, EXPANDED_CB)]


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def load_payload(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    records = [dict(record) for record in payload["records"]]
    subjects = sorted({str(record["subject"]) for record in records})
    observed = {"subjects": len(subjects), "videos": len(records),
                "gt": sum(len(record["samples"]) for record in records),
                "k_p": int(payload["k_p"])}
    return payload, records, subjects, observed


def dataset_audit(path, expected_sha, expected_meta, expected_author, expected_old_tuned,
                   old_config_by_subject=None):
    payload, records, subjects, observed = load_payload(path)
    if sha256(path) != expected_sha:
        raise RuntimeError(f"cache hash mismatch for {path}")
    if observed != expected_meta:
        raise RuntimeError(f"cache metadata mismatch for {path}: {observed} != {expected_meta}")
    author_subject = {}
    author_total = counts_empty()
    old_subject = {}
    old_total = counts_empty()
    for subject in subjects:
        a = counts_empty()
        o = counts_empty()
        for record in records:
            if str(record["subject"]) != subject:
                continue
            add_counts(a, evaluate(record, author_decode(record, observed["k_p"])))
            old_config = (old_config_by_subject or {}).get(subject, OLD_TUNED)
            add_counts(o, evaluate(record, tuned_decode(record, old_config, observed["k_p"])))
        author_subject[subject] = metrics(a)
        old_subject[subject] = metrics(o)
        add_counts(author_total, a)
        add_counts(old_total, o)
    author_metrics = metrics(author_total)
    old_metrics = metrics(old_total)
    if tuple(author_total[key] for key in ("TP", "FP", "FN")) != expected_author:
        raise RuntimeError(f"Author anchor mismatch: {author_metrics}")
    if tuple(old_total[key] for key in ("TP", "FP", "FN")) != expected_old_tuned:
        raise RuntimeError(f"Old Tuned anchor mismatch: {old_metrics}")
    return {"payload": payload, "records": records, "subjects": subjects,
            "observed": observed, "author_subject": author_subject,
            "old_subject": old_subject, "author": author_metrics, "old_tuned": old_metrics}


def expanded_dataset_audit(data):
    records, subjects, k_p = data["records"], data["subjects"], data["observed"]["k_p"]
    configs = grid()
    # Precompute each config's per-subject pooled counts. Config selection is
    # then exactly train-subject pooled F1 for each outer held subject.
    by_config_subject = []
    for index, config in enumerate(configs):
        per_subject = {}
        for subject in subjects:
            total = counts_empty()
            for record in records:
                if str(record["subject"]) == subject:
                    add_counts(total, evaluate(record, tuned_decode(record, config, k_p)))
            per_subject[subject] = total
        by_config_subject.append(per_subject)
    selected = {}
    outer_rows = []
    inner_rows = []
    for held in subjects:
        ranked = []
        for index, config in enumerate(configs):
            train_total = counts_empty()
            for subject in subjects:
                if subject != held:
                    add_counts(train_total, by_config_subject[index][subject])
            rank = (-Fraction(2 * train_total["TP"],
                              2 * train_total["TP"] + train_total["FP"] + train_total["FN"])
                    if 2 * train_total["TP"] + train_total["FP"] + train_total["FN"] else Fraction(0),
                    train_total["FP"], 0, index)
            ranked.append((rank, index, config, metrics(train_total)))
        _, index, config, inner = min(ranked, key=lambda row: row[0])
        selected[held] = config
        outer = by_config_subject[index][held]
        outer_rows.append({"outer_subject": held, "selected_config_index": index,
                           **config, **metrics(outer), "inner_F1": inner["F1"],
                           "inner_TP": inner["TP"], "inner_FP": inner["FP"], "inner_FN": inner["FN"]})
        inner_rows.append({"outer_subject": held, "selected_config_index": index,
                           "config_json": config_key(config), **inner, "selected": True})
    total = counts_empty()
    for row in outer_rows:
        add_counts(total, row)
    aggregate = metrics(total)
    return {"configs": configs, "selected": selected, "outer_rows": outer_rows,
            "inner_rows": inner_rows, "aggregate": aggregate}


def frequencies(selected, subjects):
    out = {}
    for field in ("c_s", "p", "c_d", "c_b"):
        values = [selected[s][field] for s in subjects]
        possible = {"c_s": EXPANDED_CS, "p": EXPANDED_P,
                    "c_d": EXPANDED_CD, "c_b": EXPANDED_CB}[field]
        counts = Counter(str(value) for value in values)
        endpoint = max(counts.get(str(possible[0]), 0), counts.get(str(possible[-1]), 0))
        out[field] = {"frequency": dict(sorted(counts.items())),
                      "maximum_single_endpoint_frequency": int(endpoint),
                      "status": "EXPANDED-NATIVE-BOUNDARY-REMAINS" if endpoint > len(subjects) / 2 else None}
    return out


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "CASME3": {"cache_path": str(CASME3_CACHE.resolve()), "cache_sha256": sha256(CASME3_CACHE),
                   "gt_source": "verified cache records[*].samples (onset, apex, offset)",
                   "subjects": 94, "videos": 462, "gt": 858, "k_p": 17},
        "SAMMLV": {"cache_path": str(SAMMLV_CACHE.resolve()), "cache_sha256": sha256(SAMMLV_CACHE),
                   "gt_source": "verified cache records[*].samples (onset, apex, offset)",
                   "subjects": 29, "videos": 79, "gt": 159, "k_p": 5},
        "evaluator": {"native_audit_path": str(NATIVE_AUDIT.resolve()), "native_audit_sha256": sha256(NATIVE_AUDIT),
                      "paper_metrics_reference_path": str(PAPER_METRICS.resolve()),
                      "paper_metrics_reference_sha256": sha256(PAPER_METRICS)},
        "SAMMLV_old_tuned_selection_source": {"path": str(SAMMLV_TUNED_REPORT.resolve()),
                                               "sha256": sha256(SAMMLV_TUNED_REPORT)},
    }
    try:
        casme3 = dataset_audit(CASME3_CACHE, EXPECTED_CASME3_SHA,
                               {"subjects": 94, "videos": 462, "gt": 858, "k_p": 17},
                               EXPECTED_AUTHOR_CASME3, EXPECTED_OLD_TUNED_CASME3)
        sammlv_report = json.loads(SAMMLV_TUNED_REPORT.read_text())
        sammlv_old_configs = {
            str(row["subject"]): {"c_s": float(row["selected_strong_config"]["c_s"]),
                                   "p": float(row["selected_strong_config"]["p_s"]),
                                   "c_d": float(row["selected_strong_config"]["c_d"]),
                                   "c_b": float(row["selected_strong_config"]["c_b"])}
            for row in sammlv_report["outer_folds"]
        }
        sammlv = dataset_audit(SAMMLV_CACHE, EXPECTED_SAMMLV_SHA,
                               {"subjects": 29, "videos": 79, "gt": 159, "k_p": 5},
                               EXPECTED_AUTHOR_SAMMLV, EXPECTED_OLD_TUNED_SAMMLV,
                               sammlv_old_configs)
    except Exception as exc:
        summary = {"status": "BLOCKED-CASME3-NATIVE-MISMATCH", "provenance": provenance,
                   "reason": repr(exc)}
        (OUT / "expanded_native_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (OUT / "EXPANDED_NATIVE_FAIRNESS_AUDIT_CN.md").write_text(
            "# Expanded Native Baseline Fairness Audit\n\n**BLOCKED-CASME3-NATIVE-MISMATCH**\n\n" + repr(exc) + "\n", encoding="utf-8")
        raise

    casme3_expanded = expanded_dataset_audit(casme3)
    sammlv_expanded = expanded_dataset_audit(sammlv)
    casme3_frequency = frequencies(casme3_expanded["selected"], casme3["subjects"])
    sammlv_frequency = frequencies(sammlv_expanded["selected"], sammlv["subjects"])
    morph_f1 = 0.11054342777519739
    expanded_f1 = casme3_expanded["aggregate"]["F1"]
    decision = "MORPHOLOGY-SURVIVES-EXPANDED-NATIVE" if expanded_f1 < morph_f1 else "EXPANDED-NATIVE-CHALLENGES-MORPHOLOGY"
    summary = {
        "status": "EXPANDED-NATIVE-FAIRNESS-AUDIT-COMPLETE", "decision": decision,
        "provenance": provenance,
        "anchors": {"CASME3_Author_Native": casme3["author"], "CASME3_Old_Tuned_Native": casme3["old_tuned"],
                    "SAMMLV_Author_Native": sammlv["author"], "SAMMLV_Old_Tuned_Native": sammlv["old_tuned"]},
        "expanded_grid": {"count": 256, "c_s": EXPANDED_CS, "p": EXPANDED_P,
                          "c_d": EXPANDED_CD, "c_b": EXPANDED_CB,
                          "selection": "outer subject LOSO; pooled outer-train subject Spotting F1",
                          "rounding": "integer round(config multiplier * dataset k_p)"},
        "CASME3": {"expanded_native": casme3_expanded["aggregate"],
                   "selected_parameter_frequency": casme3_frequency},
        "SAMMLV_optional_consistency": {"expanded_native": sammlv_expanded["aggregate"],
                                         "selected_parameter_frequency": sammlv_frequency},
        "comparison": {"old_tuned_native_F1": casme3["old_tuned"]["F1"],
                       "expanded_tuned_native_F1": expanded_f1,
                       "frozen_morphology_F1": morph_f1,
                       "delta_expanded_minus_old": expanded_f1 - casme3["old_tuned"]["F1"],
                       "delta_expanded_minus_morphology": expanded_f1 - morph_f1},
        "integrity": {"Author_anchor_exact": True, "Old_Tuned_anchor_exact": True,
                      "outer_test_excluded_from_selection": True, "morphology_modified": False,
                      "morphology_tuning_run": False, "recognition_STRS_run": False,
                      "new_decoder": False, "new_features": False, "grid_expanded_again": False},
        "method_development_status": "SAMMLV-METHOD-DEVELOPMENT-FROZEN",
    }
    write_csv(OUT / "expanded_native_outer_metrics.csv", casme3_expanded["outer_rows"] +
              [{"outer_subject": "aggregate", **casme3_expanded["aggregate"]}])
    write_csv(OUT / "expanded_native_selected_configs.csv",
              [{"outer_subject": subject, **casme3_expanded["selected"][subject],
                "config_index": casme3_expanded["outer_rows"][index]["selected_config_index"]}
               for index, subject in enumerate(casme3["subjects"])])
    (OUT / "expanded_native_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Expanded Native Baseline Fairness Audit", "",
        "本任务只审计 Native decoder 的有限 grid 边界；Morphology、feature、Morphology tuning 与其它 decoder 全程未运行。", "",
        "## Provenance and anchor gates", "",
        f"- CASME3 cache：`{CASME3_CACHE.resolve()}`；SHA-256=`{provenance['CASME3']['cache_sha256']}`。",
        "- CASME3：94 subjects / 462 videos / 858 GT / k_p=17；GT=`records[*].samples`。",
        f"- evaluator：`{NATIVE_AUDIT.resolve()}`；hash=`{provenance['evaluator']['native_audit_sha256']}`；paper reference hash=`{provenance['evaluator']['paper_metrics_reference_sha256']}`。",
        f"- Author Native exact：{casme3['author']['TP']}/{casme3['author']['FP']}/{casme3['author']['FN']}, F1={casme3['author']['F1']:.6f}。",
        f"- Old Tuned Native exact：{casme3['old_tuned']['TP']}/{casme3['old_tuned']['FP']}/{casme3['old_tuned']['FN']}, F1={casme3['old_tuned']['F1']:.6f}。", "",
        "## Expanded Native grid", "",
        "`c_s={1.0,1.5,2.0,2.5}`, `p={0.35,0.45,0.55,0.65}`, `c_d={0.5,0.75,1.0,1.25}`, `c_b={0.5,0.75,1.0,1.25}`; total 256 configs。",
        "每个 outer fold 仅用 outer-train subjects 的 pooled Spotting F1 选择配置；没有使用 outer-test GT。", "",
        "## CASME3 comparison", "",
        "| Method | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("Author Native", casme3["author"]), ("Old Tuned Native", casme3["old_tuned"]),
                      ("Expanded Tuned Native", casme3_expanded["aggregate"]),
                      ("Frozen Morphology Rescue", {"TP": 119, "FP": 1176, "FN": 739,
                                                       "Precision": 0.0918918918918919,
                                                       "Recall": 0.1386946386946387, "F1": morph_f1})):
        report.append(f"| {name} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['F1']:.6f} |")
    report += ["", "### Selected parameter frequency", "", f"`{json.dumps(casme3_frequency, ensure_ascii=False)}`", ""]
    for field, row in casme3_frequency.items():
        if row["status"]:
            report.append(f"- {field}: **{row['status']}**")
    report += ["", "## Optional SAMMLV consistency", "",
               f"Expanded Tuned Native：{sammlv_expanded['aggregate']['TP']}/{sammlv_expanded['aggregate']['FP']}/{sammlv_expanded['aggregate']['FN']}, F1={sammlv_expanded['aggregate']['F1']:.6f}。",
               f"Author anchor：{sammlv['author']['TP']}/{sammlv['author']['FP']}/{sammlv['author']['FN']}；old Tuned anchor：{sammlv['old_tuned']['TP']}/{sammlv['old_tuned']['FP']}/{sammlv['old_tuned']['FN']}。", "",
               "## Decision", "", f"- Expanded Tuned Native − Old Tuned Native ΔF1 = {expanded_f1 - casme3['old_tuned']['F1']:+.6f}。",
               f"- Expanded Tuned Native − Frozen Morphology ΔF1 = {expanded_f1 - morph_f1:+.6f}。",
               f"- **{decision}**", "- **EXPANDED-NATIVE-FAIRNESS-AUDIT-COMPLETE**", "- **SAMMLV-METHOD-DEVELOPMENT-FROZEN**", "",
               "未修改 Morphology；未重新运行 morphology tuning；未运行 recognition/STRS。", "",
               "## Outputs", ""]
    report.extend(f"- `{path.resolve()}`" for path in sorted(OUT.iterdir()))
    (OUT / "EXPANDED_NATIVE_FAIRNESS_AUDIT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "CASME3": casme3_expanded["aggregate"],
                      "SAMMLV_optional": sammlv_expanded["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
