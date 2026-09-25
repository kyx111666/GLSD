#!/usr/bin/env python3
"""Frozen SAMMLV C2 recognition-only pruning validation on CASME3."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/frozen_recognition_pruning_casme3"
OUTPUTS = OUT / "outputs"
PRUNING_SOURCE = ROOT / "my_method/recognition_conservative_pruning_sammlv/run_pruning_gate.py"
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
CONFIG_SOURCE = ROOT / "results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv"
OUTER_SOURCE = CONFIG_SOURCE.with_name("final_strong_native_outer_metrics.csv")

spec = importlib.util.spec_from_file_location("frozen_pruning", PRUNING_SOURCE)
pruning = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pruning)
native = pruning.native

SEED = 20260905
REPS = 1000
DELTA = 0.10
GAMMA = 0.35
EXPECTED_SHA = "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda"
EXPECTED = {"subjects": 94, "videos": 462, "gt": 858, "k_p": 17}
ANCHOR = {"TP": 124, "FP": 1148, "FN": 734, "F1": 0.11643192488262911}
BASE = "Final_Strong_Native"
METHOD = "Frozen_Recognition_Only_Conservative_Pruning"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_sources():
    if sha256(CACHE) != EXPECTED_SHA:
        raise RuntimeError("CASME3 cache SHA mismatch")
    payload, records, subjects, observed = native.load_payload(CACHE)
    if observed != EXPECTED:
        raise RuntimeError(f"CASME3 cache metadata mismatch: {observed}")
    configs = {}
    with CONFIG_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            configs[str(row["outer_subject"])] = {key: float(row[key]) for key in ("c_s", "p", "c_d", "c_b")}
    references = {}
    with OUTER_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["outer_subject"] != "aggregate":
                references[str(row["outer_subject"])] = {key: int(row[key]) for key in ("TP", "FP", "FN", "event_count")}
    if set(configs) != set(subjects) or set(references) != set(subjects):
        raise RuntimeError("CASME3 fold source subjects mismatch")
    by_subject = {subject: [] for subject in subjects}
    for index, record in enumerate(records):
        score, logits = np.asarray(record["score"]), np.asarray(record["logits"])
        if score.ndim != 1 or logits.shape != (len(score), 5) or not np.isfinite(score).all() or not np.isfinite(logits).all():
            raise RuntimeError(f"invalid frozen arrays: record {index}")
        by_subject[str(record["subject"])].append(index)
    total = pruning.empty_counts()
    for subject in subjects:
        counts = pruning.empty_counts()
        for index in by_subject[subject]:
            pruning.add_counts(counts, native.evaluate(records[index], native.tuned_decode(records[index], configs[subject], EXPECTED["k_p"])))
        if counts != references[subject]:
            raise RuntimeError(f"CASME3 fold anchor mismatch: {subject}")
        pruning.add_counts(total, counts)
    aggregate = pruning.metrics(total)
    if any(aggregate[key] != ANCHOR[key] for key in ("TP", "FP", "FN")) or abs(aggregate["F1"] - ANCHOR["F1"]) > 1e-14:
        raise RuntimeError(f"CASME3 aggregate anchor mismatch: {aggregate}")
    return records, list(subjects), by_subject, configs, aggregate


def run_fold(held, records, subjects, by_subject, config):
    bank = pruning.build_bank(records, config, EXPECTED["k_p"])
    train_indices = [index for subject in subjects if subject != held for index in by_subject[subject]]
    test_indices = by_subject[held]
    train_items = pruning.candidate_items(bank, records, train_indices)
    test_items = pruning.candidate_items(bank, records, test_indices)
    probabilities, handling = pruning.fit_probabilities(train_items, test_items, "R")
    pmap = pruning.probability_map(test_items, probabilities)
    base, base_diag, base_trace = pruning.apply_method(
        bank, records, test_indices, "C0_Final_Strong_Native", traces=True, outer_fold=held)
    result, diagnostic, method_trace = pruning.apply_method(
        bank, records, test_indices, "C2_Recognition_Only", DELTA, GAMMA, pmap, True, held)
    for row in base_trace:
        row.update({"method": BASE, "selected_delta": None, "selected_gamma": None})
    for row in method_trace:
        row.update({"method": METHOD, "selected_delta": DELTA, "selected_gamma": GAMMA})
    return base, result, dict(diagnostic), base_trace + method_trace, handling


def bootstrap(subject_rows, subjects):
    lookup = {(row["subject"], row["method"]): row for row in subject_rows}
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(subjects), size=(REPS, len(subjects)))
    values = []
    for sample in samples:
        counts = {BASE: pruning.empty_counts(), METHOD: pruning.empty_counts()}
        for position in sample:
            subject = subjects[int(position)]
            pruning.add_counts(counts[BASE], lookup[(subject, BASE)])
            pruning.add_counts(counts[METHOD], lookup[(subject, METHOD)])
        values.append(pruning.metrics(counts[METHOD])["F1"] - pruning.metrics(counts[BASE])["F1"])
    array = np.asarray(values)
    return {"seed": SEED, "subjects": subjects, "sampled_subject_indices": samples.tolist(),
            "observed_comparison": f"{METHOD}_minus_{BASE}", "mean_delta_F1": float(array.mean()),
            "ci95": [float(x) for x in np.quantile(array, [.025, .975])], "replicates": values}


def render_report(summary):
    base, method = summary["aggregate"][BASE], summary["aggregate"][METHOD]
    diag, stability, boot = summary["pruning_diagnostics"], summary["subject_stability"], summary["bootstrap"]
    utility = "INF" if diag["FP_removed_per_TP_lost"] == "INF" else f"{diag['FP_removed_per_TP_lost']:.6f}"
    return f"""# Frozen Recognition-Only Conservative Pruning — CASME3 Cross-Dataset Validation

最终状态：`{summary['decision']}`。

本次只验证从 SAMMLV 冻结的 C2：R2-mean、delta=0.10、gamma=0.35、固定 LR；CASME3 未进行任何 tuning。

## Provenance / anchor gate

- 94 subjects / 462 videos / 858 GT / k_p=17。
- Final Strong Native PASS：124/1148/734，F1={base['F1']:.6f}。
- Cache：`{summary['provenance']['cache_path']}`；SHA-256 `{summary['provenance']['cache_sha256']}`。
- Evaluator SHA-256：`{summary['provenance']['evaluator_sha256']}`。
- Recognition：frozen logits；softmax axis=1；neutral class id=4；clipped p±k_p R2-mean。

## Aggregate comparison

| Method | TP/FP/FN | Precision | Recall | F1 | Events/video |
|---|---:|---:|---:|---:|---:|
| Final Strong Native | {base['TP']}/{base['FP']}/{base['FN']} | {base['Precision']:.6f} | {base['Recall']:.6f} | {base['F1']:.6f} | {base['events_per_video']:.6f} |
| Frozen Recognition-only Pruning | {method['TP']}/{method['FP']}/{method['FN']} | {method['Precision']:.6f} | {method['Recall']:.6f} | {method['F1']:.6f} | {method['events_per_video']:.6f} |

Delta TP/FP/FN/F1：{method['delta_TP']:+d} / {method['delta_FP']:+d} / {method['delta_FN']:+d} / {method['delta_F1']:+.6f}。

## Pruning diagnostics

- Total Native events：{diag['total_native_events']}；gray-zone eligible：{diag['eligible_gray_zone_events']}；pruned：{diag['pruned_events']}；rate={diag['pruning_rate']:.6f}。
- Pruned TP/FP：{diag['pruned_true_positives']}/{diag['pruned_false_positives']}；prune precision={diag['prune_precision']:.6f}。
- FP removed / TP lost：{utility}。
- Eligible-event P_keep min/median/max：{diag['eligible_P_keep_min']:.6f} / {diag['eligible_P_keep_median']:.6f} / {diag['eligible_P_keep_max']:.6f}；低于冻结 gamma=0.35 的事件数为 {diag['eligible_below_gamma_count']}。

## Subject stability

- Improved/equal/worse：{stability['improved']}/{stability['equal']}/{stability['worse']}。
- Leave-one-subject aggregate Delta F1 min/median/max：{stability['leave_one_subject']['min']:+.6f} / {stability['leave_one_subject']['median']:+.6f} / {stability['leave_one_subject']['max']:+.6f}。

## Bootstrap

- Observed Delta F1：{method['delta_F1']:+.6f}。
- Bootstrap mean：{boot['mean_delta_F1']:+.6f}。
- 95% CI：[{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}]。

## Decision

`{summary['decision']}`

- Checks：{summary['decision_checks']}。
- 已停止；未修改参数、Recognition feature 或 decoder，未启动其他实验。

## Outputs

- `{(OUT / 'FROZEN_RECOGNITION_PRUNING_CASME3_CN.md').resolve()}`
- `{(OUTPUTS / 'casme3_pruning_outer_metrics.csv').resolve()}`
- `{(OUTPUTS / 'casme3_pruning_subject_metrics.csv').resolve()}`
- `{(OUTPUTS / 'casme3_pruning_event_trace.csv').resolve()}`
- `{(OUTPUTS / 'casme3_pruning_bootstrap.json').resolve()}`
- `{(OUTPUTS / 'casme3_pruning_summary.json').resolve()}`
- `{(HERE / 'run_frozen_validation.py').resolve()}`
"""


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    input_paths = (CACHE, CONFIG_SOURCE, OUTER_SOURCE, PRUNING_SOURCE, NATIVE_SOURCE, NATIVE_AUDIT, PAPER_METRICS)
    hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    try:
        records, subjects, by_subject, configs, anchor = load_sources()
    except Exception as exc:
        blocked = {"status": "BLOCKED-FROZEN-RECOGNITION-PRUNING", "reason": repr(exc), "input_sha256": hashes}
        dump(OUTPUTS / "casme3_pruning_summary.json", blocked)
        (OUT / "FROZEN_RECOGNITION_PRUNING_CASME3_CN.md").write_text(
            f"# Frozen Recognition-Only Conservative Pruning — CASME3\n\n`BLOCKED-FROZEN-RECOGNITION-PRUNING`\n\n{exc!r}\n", encoding="utf-8")
        raise
    rows, traces, handling = [], [], Counter()
    for held in subjects:
        base, result, diagnostic, fold_trace, status = run_fold(held, records, subjects, by_subject, configs[held])
        handling[status] += 1
        traces.extend(fold_trace)
        rows.append({"subject": held, "outer_subject": held, "method": BASE, **base})
        rows.append({"subject": held, "outer_subject": held, "method": METHOD, **result,
                     **{f"diagnostic_{key}": value for key, value in diagnostic.items()}})
    aggregate = {}
    for method_name in (BASE, METHOD):
        counts = pruning.empty_counts()
        for row in rows:
            if row["method"] == method_name:
                pruning.add_counts(counts, row)
        aggregate[method_name] = pruning.metrics(counts)
        aggregate[method_name]["events_per_video"] = aggregate[method_name]["event_count"] / EXPECTED["videos"]
    base, method = aggregate[BASE], aggregate[METHOD]
    if any(base[key] != anchor[key] for key in ("TP", "FP", "FN")) or abs(base["F1"] - anchor["F1"]) > 1e-14:
        raise RuntimeError("post-OOF Native anchor drift")
    method.update({"delta_TP": method["TP"] - base["TP"], "delta_FP": method["FP"] - base["FP"],
                   "delta_FN": method["FN"] - base["FN"], "delta_F1": method["F1"] - base["F1"]})
    base.update({"delta_TP": 0, "delta_FP": 0, "delta_FN": 0, "delta_F1": 0.0})
    method_rows = [row for row in rows if row["method"] == METHOD]
    total = sum(row.get("diagnostic_total_native_events", 0) for row in method_rows)
    eligible = sum(row.get("diagnostic_eligible_gray_zone_events", 0) for row in method_rows)
    pruned = sum(row.get("diagnostic_pruned_events", 0) for row in method_rows)
    pruned_tp = sum(row.get("diagnostic_pruned_KEEP", 0) for row in method_rows)
    pruned_fp = sum(row.get("diagnostic_pruned_PRUNE", 0) for row in method_rows)
    tp_lost, fp_removed = base["TP"] - method["TP"], base["FP"] - method["FP"]
    ratio = "INF" if tp_lost == 0 else fp_removed / tp_lost
    eligible_probabilities = np.asarray([float(row["P_keep"]) for row in traces
                                         if row["method"] == METHOD and row["eligible_gray_zone"]], dtype=float)
    diagnostics = {"total_native_events": total, "eligible_gray_zone_events": eligible,
                   "pruned_events": pruned, "pruning_rate": pruned / total if total else 0.0,
                   "pruned_true_positives": pruned_tp, "pruned_false_positives": pruned_fp,
                   "ambiguous_pruned": pruned - pruned_tp - pruned_fp,
                   "prune_precision": pruned_fp / pruned if pruned else 0.0,
                   "FP_removed": fp_removed, "TP_lost": tp_lost, "FP_removed_per_TP_lost": ratio}
    diagnostics.update({"eligible_P_keep_min": float(eligible_probabilities.min()),
                        "eligible_P_keep_median": float(np.median(eligible_probabilities)),
                        "eligible_P_keep_max": float(eligible_probabilities.max()),
                        "eligible_below_gamma_count": int(np.sum(eligible_probabilities < GAMMA))})
    lookup = {(row["subject"], row["method"]): row for row in rows}
    relation, comparison_rows, loo = Counter(), [], []
    for subject in subjects:
        c0, c1 = lookup[(subject, BASE)], lookup[(subject, METHOD)]
        delta = c1["F1"] - c0["F1"]
        status = "improved" if delta > 1e-15 else "worse" if delta < -1e-15 else "equal"
        relation[status] += 1
        comparison_rows.append({"subject": subject, **{f"Native_{key}": c0[key] for key in ("TP", "FP", "FN", "Precision", "Recall", "F1")},
                                **{f"Pruning_{key}": c1[key] for key in ("TP", "FP", "FN", "Precision", "Recall", "F1")},
                                "delta_F1": delta, "Pruning_vs_Native": status})
        leave_base, leave_method = pruning.empty_counts(), pruning.empty_counts()
        for other in subjects:
            if other != subject:
                pruning.add_counts(leave_base, lookup[(other, BASE)])
                pruning.add_counts(leave_method, lookup[(other, METHOD)])
        loo.append(pruning.metrics(leave_method)["F1"] - pruning.metrics(leave_base)["F1"])
    boot = bootstrap(rows, subjects)
    severe_tp_destruction = bool(tp_lost > 0 and fp_removed / tp_lost < 2.0)
    core = {"F1_gt_anchor": method["F1"] > ANCHOR["F1"], "FP_lt_anchor": method["FP"] < ANCHOR["FP"],
            "bootstrap_mean_positive": boot["mean_delta_F1"] > 0, "not_severe_TP_destruction": not severe_tp_destruction}
    stable = {"LOSO_min_positive": min(loo) > 0, "bootstrap_CI_lower_positive": boot["ci95"][0] > 0}
    if all(core.values()) and all(stable.values()):
        decision = "FROZEN-RECOGNITION-PRUNING-CROSS-DATASET-GO"
    elif method["F1"] > ANCHOR["F1"] and method["FP"] < ANCHOR["FP"] and not severe_tp_destruction:
        decision = "FROZEN-RECOGNITION-PRUNING-WEAK-GO"
    else:
        decision = "FROZEN-RECOGNITION-PRUNING-NO-GO"
    stability = {key: int(relation[key]) for key in ("improved", "equal", "worse")}
    stability["leave_one_subject"] = {"values": loo, "min": float(min(loo)),
                  "median": float(np.median(loo)), "max": float(max(loo)), "min_below_zero": bool(min(loo) < 0)}
    provenance = {"cache_path": str(CACHE.resolve()), "cache_sha256": EXPECTED_SHA,
                  "evaluator_path": str(NATIVE_AUDIT.resolve()), "evaluator_sha256": sha256(NATIVE_AUDIT),
                  "GT_source": "verified cache records[*].samples", "fold_config_source": str(CONFIG_SOURCE.resolve()),
                  "recognition_source": "verified cache records[*].logits", "subjects": 94, "videos": 462,
                  "GT": 858, "k_p": 17, "anchor_exact": True}
    summary = {"status": "COMPLETE", "decision": decision, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "provenance": provenance, "input_sha256": hashes, "frozen_method": {"feature": "R2-mean only",
                   "delta": DELTA, "gamma": GAMMA, "classifier": "StandardScaler + LogisticRegression(liblinear, balanced, C=1, max_iter=1000, seed=20260905)"},
               "aggregate": aggregate, "pruning_diagnostics": diagnostics, "subject_stability": stability,
               "bootstrap": boot, "decision_checks": {"core": core, "stability": stable,
                   "severe_TP_destruction_predeclared_as_FP_removed_per_TP_lost_lt_2": severe_tp_destruction},
               "fit_handling": dict(handling), "integrity": {"no_hyperparameter_tuning": True, "fixed_delta": DELTA,
                   "fixed_gamma": GAMMA, "R_only_classifier": True, "outer_test_GT_used_for_fit": False,
                   "deletion_only": True, "new_predictions_added": 0, "backbone_forward": False,
                   "morphology_used": False, "rescue_used": False}}
    outer_rows = rows + [{"subject": "aggregate", "outer_subject": "aggregate", "method": name, **value}
                         for name, value in aggregate.items()]
    write_csv(OUTPUTS / "casme3_pruning_outer_metrics.csv", outer_rows)
    write_csv(OUTPUTS / "casme3_pruning_subject_metrics.csv", comparison_rows)
    write_csv(OUTPUTS / "casme3_pruning_event_trace.csv", traces)
    dump(OUTPUTS / "casme3_pruning_bootstrap.json", boot)
    dump(OUTPUTS / "casme3_pruning_summary.json", summary)
    (OUT / "FROZEN_RECOGNITION_PRUNING_CASME3_CN.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"decision": decision, "aggregate": aggregate, "diagnostics": diagnostics,
                      "stability": stability, "bootstrap": {"mean": boot["mean_delta_F1"], "ci95": boot["ci95"]}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
