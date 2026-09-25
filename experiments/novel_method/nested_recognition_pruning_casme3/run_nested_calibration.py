#!/usr/bin/env python3
"""Nested-calibrated recognition-only conservative pruning on CASME3."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/nested_recognition_pruning_casme3"
OUTPUTS = OUT / "outputs"
FROZEN_SOURCE = ROOT / "my_method/frozen_recognition_pruning_casme3/run_frozen_validation.py"
PRUNING_SOURCE = ROOT / "my_method/recognition_conservative_pruning_sammlv/run_pruning_gate.py"

spec = importlib.util.spec_from_file_location("frozen_validation", FROZEN_SOURCE)
frozen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(frozen)
pruning = frozen.pruning

SEED = 20260905
REPS = 1000
DELTAS = (0.05, 0.10, 0.15, 0.20)
GAMMAS = (0.20, 0.35, 0.50)
GRID = tuple(itertools.product(DELTAS, GAMMAS))
BASE = "Final_Strong_Native"
METHOD = "Nested_Calibrated_Recognition_Only_Pruning"
SAMMLV_REFERENCE = {"Native_F1": 0.2792022792022792, "Recognition_Only_F1": 0.2823529411764706}


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


def config_id(config):
    return tuple(float(config[key]) for key in ("c_s", "p", "c_d", "c_b"))


def select_key(row, native_metrics):
    diagnostic = row["diagnostic"]
    rate = diagnostic["pruned_events"] / diagnostic["total_native_events"] if diagnostic["total_native_events"] else 0.0
    return (row["metrics"]["F1"], -row["metrics"]["FP"],
            -(native_metrics["TP"] - row["metrics"]["TP"]), -rate, -row["delta"], -row["gamma"])


def nested_fold(held, records, subjects, by_subject, bank):
    outer_train = [subject for subject in subjects if subject != held]
    inner_probabilities, handling = {}, Counter()
    for validation in outer_train:
        train_indices = [index for subject in outer_train if subject != validation for index in by_subject[subject]]
        validation_indices = by_subject[validation]
        train_items = pruning.candidate_items(bank, records, train_indices)
        validation_items = pruning.candidate_items(bank, records, validation_indices)
        probabilities, status = pruning.fit_probabilities(train_items, validation_items, "R")
        handling[f"inner_{status}"] += 1
        inner_probabilities.update(pruning.probability_map(validation_items, probabilities))
    inner_indices = [index for subject in outer_train for index in by_subject[subject]]
    inner_native, _, _ = pruning.apply_method(bank, records, inner_indices, "C0_Final_Strong_Native")
    scored = []
    for delta, gamma in GRID:
        result, diagnostic, _ = pruning.apply_method(
            bank, records, inner_indices, "C2_Recognition_Only", delta, gamma, inner_probabilities)
        scored.append({"delta": delta, "gamma": gamma, "metrics": result, "diagnostic": diagnostic})
    selected = max(scored, key=lambda row: select_key(row, inner_native))
    train_items = pruning.candidate_items(bank, records, inner_indices)
    test_indices = by_subject[held]
    test_items = pruning.candidate_items(bank, records, test_indices)
    probabilities, status = pruning.fit_probabilities(train_items, test_items, "R")
    handling[f"outer_{status}"] += 1
    test_probability_map = pruning.probability_map(test_items, probabilities)
    base, _, base_trace = pruning.apply_method(
        bank, records, test_indices, "C0_Final_Strong_Native", traces=True, outer_fold=held)
    result, diagnostic, method_trace = pruning.apply_method(
        bank, records, test_indices, "C2_Recognition_Only", selected["delta"], selected["gamma"],
        test_probability_map, True, held)
    for row in base_trace:
        row.update({"method": BASE, "selected_delta": None, "selected_gamma": None})
    for row in method_trace:
        row.update({"method": METHOD, "selected_delta": selected["delta"], "selected_gamma": selected["gamma"]})
    selection = {"outer_subject": held, "selected_delta": selected["delta"],
                 "selected_gamma": selected["gamma"], **{f"inner_{key}": value for key, value in selected["metrics"].items()},
                 "inner_native_TP": inner_native["TP"], "inner_native_FP": inner_native["FP"],
                 "inner_native_FN": inner_native["FN"], "inner_native_F1": inner_native["F1"],
                 "inner_pruned_events": selected["diagnostic"]["pruned_events"],
                 "inner_total_native_events": selected["diagnostic"]["total_native_events"]}
    return base, result, dict(diagnostic), base_trace + method_trace, selection, handling


def subject_bootstrap(rows, subjects):
    lookup = {(row["subject"], row["method"]): row for row in rows}
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
            "mean_delta_F1": float(array.mean()), "ci95": [float(x) for x in np.quantile(array, [.025, .975])],
            "replicates": values}


def render_report(summary):
    base, method = summary["aggregate"][BASE], summary["aggregate"][METHOD]
    diagnostic, stability, bootstrap = summary["pruning_diagnostics"], summary["subject_stability"], summary["bootstrap"]
    utility = "INF" if diagnostic["FP_removed_per_TP_lost"] == "INF" else f"{diagnostic['FP_removed_per_TP_lost']:.6f}"
    return f"""# Nested-Calibrated Recognition-Only Conservative Pruning — CASME3

最终状态：`{summary['decision']}`。

仅使用 R2-mean；H_s 只作为 eligibility gate。固定 LR、C=1；nested selection 严格限制在原 12 个 delta/gamma configs。

## Anchor / provenance

- Final Strong Native PASS：124/1148/734，F1={base['F1']:.6f}。
- 94 subjects / 462 videos / 858 GT / k_p=17。
- Cache SHA-256：`{summary['provenance']['cache_sha256']}`；evaluator SHA-256：`{summary['provenance']['evaluator_sha256']}`。

## Aggregate results

| Method | TP/FP/FN | Precision | Recall | F1 | Delta TP/FP/FN | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| Final Strong Native | {base['TP']}/{base['FP']}/{base['FN']} | {base['Precision']:.6f} | {base['Recall']:.6f} | {base['F1']:.6f} | 0/0/0 | 0 |
| Nested Recognition-only | {method['TP']}/{method['FP']}/{method['FN']} | {method['Precision']:.6f} | {method['Recall']:.6f} | {method['F1']:.6f} | {method['delta_TP']:+d}/{method['delta_FP']:+d}/{method['delta_FN']:+d} | {method['delta_F1']:+.6f} |

## Pruning diagnostics

- Eligible/pruned：{diagnostic['eligible_gray_zone_events']}/{diagnostic['pruned_events']}；pruned TP/FP={diagnostic['pruned_true_positives']}/{diagnostic['pruned_false_positives']}。
- Prune precision={diagnostic['prune_precision']:.6f}；FP removed / TP lost={utility}。

## Stability

- Subject improved/equal/worse：{stability['improved']}/{stability['equal']}/{stability['worse']}。
- Leave-one-subject Delta F1 min/median/max：{stability['leave_one_subject']['min']:+.6f} / {stability['leave_one_subject']['median']:+.6f} / {stability['leave_one_subject']['max']:+.6f}。
- Bootstrap observed/mean：{method['delta_F1']:+.6f} / {bootstrap['mean_delta_F1']:+.6f}；95% CI=[{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}]。

## Parameter stability

- Selected delta：{summary['parameter_stability']['selected_delta_frequency']}。
- Selected gamma：{summary['parameter_stability']['selected_gamma_frequency']}。
- `NESTED-PRUNING-BOUNDARY-SELECTION`={summary['parameter_stability']['NESTED-PRUNING-BOUNDARY-SELECTION']}；禁止扩 grid。

## Decision

`{summary['decision']}`

- Checks：{summary['decision_checks']}。
- SAMMLV reference：0.282353 > 0.279202；CASME3 结论仅来自合法 nested calibration，不是 training-free zero-shot transfer。
- 已停止，未修改 Recognition、Native、classifier 或 grid。

## Outputs

- `{(OUT / 'NESTED_RECOGNITION_PRUNING_CASME3_CN.md').resolve()}`
- `{(OUTPUTS / 'nested_pruning_outer_metrics.csv').resolve()}`
- `{(OUTPUTS / 'nested_pruning_subject_metrics.csv').resolve()}`
- `{(OUTPUTS / 'nested_pruning_selected_configs.csv').resolve()}`
- `{(OUTPUTS / 'nested_pruning_event_trace.csv').resolve()}`
- `{(OUTPUTS / 'nested_pruning_bootstrap.json').resolve()}`
- `{(OUTPUTS / 'nested_pruning_parameter_stability.json').resolve()}`
- `{(OUTPUTS / 'nested_pruning_summary.json').resolve()}`
- `{(HERE / 'run_nested_calibration.py').resolve()}`
"""


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    inputs = (frozen.CACHE, frozen.CONFIG_SOURCE, frozen.OUTER_SOURCE, FROZEN_SOURCE, PRUNING_SOURCE,
              frozen.NATIVE_SOURCE, frozen.NATIVE_AUDIT, frozen.PAPER_METRICS)
    hashes = {str(path.resolve()): sha256(path) for path in inputs}
    try:
        records, subjects, by_subject, configs, anchor = frozen.load_sources()
    except Exception as exc:
        blocked = {"status": "BLOCKED-NESTED-RECOGNITION-PRUNING-ANCHOR", "reason": repr(exc), "input_sha256": hashes}
        dump(OUTPUTS / "nested_pruning_summary.json", blocked)
        (OUT / "NESTED_RECOGNITION_PRUNING_CASME3_CN.md").write_text(
            f"# Nested Recognition-Only Pruning — CASME3\n\n`BLOCKED-NESTED-RECOGNITION-PRUNING-ANCHOR`\n\n{exc!r}\n", encoding="utf-8")
        raise
    unique_configs = {config_id(config): config for config in configs.values()}
    banks = {key: pruning.build_bank(records, config, frozen.EXPECTED["k_p"]) for key, config in unique_configs.items()}
    rows, traces, selections, handling = [], [], [], Counter()
    for fold_index, held in enumerate(subjects, 1):
        base, result, diagnostic, fold_trace, selection, fold_handling = nested_fold(
            held, records, subjects, by_subject, banks[config_id(configs[held])])
        handling.update(fold_handling)
        traces.extend(fold_trace)
        selections.append(selection)
        rows.append({"subject": held, "outer_subject": held, "method": BASE, **base})
        rows.append({"subject": held, "outer_subject": held, "method": METHOD, **result,
                     **{f"diagnostic_{key}": value for key, value in diagnostic.items()}})
        if fold_index % 10 == 0:
            print(f"completed outer folds: {fold_index}/94", flush=True)
    aggregate = {}
    for method_name in (BASE, METHOD):
        counts = pruning.empty_counts()
        for row in rows:
            if row["method"] == method_name:
                pruning.add_counts(counts, row)
        aggregate[method_name] = pruning.metrics(counts)
    base, method = aggregate[BASE], aggregate[METHOD]
    if any(base[key] != anchor[key] for key in ("TP", "FP", "FN")) or abs(base["F1"] - anchor["F1"]) > 1e-14:
        raise RuntimeError("post-nested Native anchor drift")
    base.update({"delta_TP": 0, "delta_FP": 0, "delta_FN": 0, "delta_F1": 0.0})
    method.update({"delta_TP": method["TP"] - base["TP"], "delta_FP": method["FP"] - base["FP"],
                   "delta_FN": method["FN"] - base["FN"], "delta_F1": method["F1"] - base["F1"]})
    method_rows = [row for row in rows if row["method"] == METHOD]
    total = sum(row.get("diagnostic_total_native_events", 0) for row in method_rows)
    eligible = sum(row.get("diagnostic_eligible_gray_zone_events", 0) for row in method_rows)
    pruned = sum(row.get("diagnostic_pruned_events", 0) for row in method_rows)
    pruned_tp = sum(row.get("diagnostic_pruned_KEEP", 0) for row in method_rows)
    pruned_fp = sum(row.get("diagnostic_pruned_PRUNE", 0) for row in method_rows)
    tp_lost, fp_removed = base["TP"] - method["TP"], base["FP"] - method["FP"]
    diagnostic = {"total_native_events": total, "eligible_gray_zone_events": eligible,
                  "pruned_events": pruned, "pruning_rate": pruned / total if total else 0.0,
                  "pruned_true_positives": pruned_tp, "pruned_false_positives": pruned_fp,
                  "ambiguous_pruned": pruned - pruned_tp - pruned_fp,
                  "prune_precision": pruned_fp / pruned if pruned else 0.0,
                  "FP_removed": fp_removed, "TP_lost": tp_lost,
                  "FP_removed_per_TP_lost": "INF" if tp_lost == 0 else fp_removed / tp_lost}
    lookup = {(row["subject"], row["method"]): row for row in rows}
    relations, comparison_rows, loo = Counter(), [], []
    for subject in subjects:
        c0, c1 = lookup[(subject, BASE)], lookup[(subject, METHOD)]
        delta_f1 = c1["F1"] - c0["F1"]
        relation = "improved" if delta_f1 > 1e-15 else "worse" if delta_f1 < -1e-15 else "equal"
        relations[relation] += 1
        comparison_rows.append({"subject": subject, **{f"Native_{key}": c0[key] for key in ("TP", "FP", "FN", "Precision", "Recall", "F1")},
                                **{f"Pruning_{key}": c1[key] for key in ("TP", "FP", "FN", "Precision", "Recall", "F1")},
                                "delta_F1": delta_f1, "Pruning_vs_Native": relation})
        leave = {BASE: pruning.empty_counts(), METHOD: pruning.empty_counts()}
        for other in subjects:
            if other != subject:
                pruning.add_counts(leave[BASE], lookup[(other, BASE)])
                pruning.add_counts(leave[METHOD], lookup[(other, METHOD)])
        loo.append(pruning.metrics(leave[METHOD])["F1"] - pruning.metrics(leave[BASE])["F1"])
    bootstrap = subject_bootstrap(rows, subjects)
    delta_frequency = dict(Counter(str(row["selected_delta"]) for row in selections))
    gamma_frequency = dict(Counter(str(row["selected_gamma"]) for row in selections))
    if any((row["selected_delta"], row["selected_gamma"]) not in GRID for row in selections):
        raise RuntimeError("selected config escaped fixed grid")
    boundary = delta_frequency.get("0.2", 0) > len(subjects) / 2 or gamma_frequency.get("0.5", 0) > len(subjects) / 2
    parameter_stability = {"selected_delta_frequency": delta_frequency, "selected_gamma_frequency": gamma_frequency,
                           "NESTED-PRUNING-BOUNDARY-SELECTION": boundary, "single_class_handling": dict(handling)}
    severe = bool(tp_lost > 0 and fp_removed / tp_lost < 2.0)
    checks = {"CASME3_F1_gt_anchor": method["F1"] > anchor["F1"], "CASME3_FP_lt_anchor": method["FP"] < anchor["FP"],
              "bootstrap_mean_positive": bootstrap["mean_delta_F1"] > 0, "not_severe_TP_destruction": not severe,
              "SAMMLV_direction_positive": SAMMLV_REFERENCE["Recognition_Only_F1"] > SAMMLV_REFERENCE["Native_F1"]}
    decision = ("NESTED-RECOGNITION-PRUNING-BOTH-DATASETS-GO" if all(checks.values())
                else "NESTED-RECOGNITION-PRUNING-CASME3-NO-GO")
    stability = {key: int(relations[key]) for key in ("improved", "equal", "worse")}
    stability["leave_one_subject"] = {"values": loo, "min": float(min(loo)), "median": float(np.median(loo)),
                                      "max": float(max(loo)), "min_below_zero": bool(min(loo) < 0)}
    provenance = {"cache_path": str(frozen.CACHE.resolve()), "cache_sha256": frozen.EXPECTED_SHA,
                  "evaluator_path": str(frozen.NATIVE_AUDIT.resolve()), "evaluator_sha256": sha256(frozen.NATIVE_AUDIT),
                  "GT_source": "verified cache records[*].samples", "fold_config_source": str(frozen.CONFIG_SOURCE.resolve()),
                  "recognition_source": "verified cache records[*].logits", "subjects": 94, "videos": 462, "GT": 858,
                  "k_p": 17, "anchor_exact": True}
    summary = {"status": "COMPLETE", "decision": decision, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "provenance": provenance, "input_sha256": hashes, "SAMMLV_reference": SAMMLV_REFERENCE,
               "method": {"feature": "R2-mean only", "H_role": "eligibility gate only", "delta_grid": DELTAS,
                          "gamma_grid": GAMMAS, "classifier": "StandardScaler + LogisticRegression(liblinear, balanced, C=1, max_iter=1000, seed=20260905)",
                          "tie_break": ["higher F1", "fewer FP", "fewer TP lost", "smaller pruning rate", "smaller delta", "smaller gamma"]},
               "aggregate": aggregate, "pruning_diagnostics": diagnostic, "subject_stability": stability,
               "bootstrap": bootstrap, "parameter_stability": parameter_stability, "decision_checks": checks,
               "severe_TP_destruction_predeclared_as_FP_removed_per_TP_lost_lt_2": severe,
               "integrity": {"nested_subject_disjoint": True, "outer_test_GT_used_for_selection": False,
                             "grid_size": 12, "grid_expanded": False, "C_tuned": False, "H_in_classifier": False,
                             "deletion_only": True, "new_predictions_added": 0, "backbone_forward": False,
                             "morphology_used": False, "rescue_used": False}}
    outer_rows = rows + [{"subject": "aggregate", "outer_subject": "aggregate", "method": name, **value}
                         for name, value in aggregate.items()]
    write_csv(OUTPUTS / "nested_pruning_outer_metrics.csv", outer_rows)
    write_csv(OUTPUTS / "nested_pruning_subject_metrics.csv", comparison_rows)
    write_csv(OUTPUTS / "nested_pruning_selected_configs.csv", selections)
    write_csv(OUTPUTS / "nested_pruning_event_trace.csv", traces)
    dump(OUTPUTS / "nested_pruning_bootstrap.json", bootstrap)
    dump(OUTPUTS / "nested_pruning_parameter_stability.json", parameter_stability)
    dump(OUTPUTS / "nested_pruning_summary.json", summary)
    (OUT / "NESTED_RECOGNITION_PRUNING_CASME3_CN.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"decision": decision, "aggregate": aggregate, "diagnostics": diagnostic,
                      "stability": stability, "bootstrap": {"mean": bootstrap["mean_delta_F1"], "ci95": bootstrap["ci95"]},
                      "parameter_stability": parameter_stability}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
