#!/usr/bin/env python3
"""Independent replay audit for the completed locked CASME3 validation."""
from __future__ import annotations

import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "run_casme3_locked_validation.py"
RESULT = ROOT / "results/morphology_casme3_locked_validation"
CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"


def load_run_module():
    spec = importlib.util.spec_from_file_location("casme3_locked_run", RUN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    run = load_run_module()
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    records = [dict(record) for record in payload["records"]]
    subjects = sorted({str(record["subject"]) for record in records})
    selected = {row["outer_subject"]: {
        "Delta_p": float(row["Delta_p"]), "requested_L_ratio": float(row["requested_L_ratio"]),
        "actual_integer_radius": int(float(row["actual_integer_radius"])),
        "C": float(row["C"]), "gamma": float(row["gamma"]),
    } for row in read_rows(RESULT / "outputs/casme3_morph_selected_configs.csv")}
    tuned = {row["outer_subject"]: {
        "c_s": float(row["c_s"]), "p": float(row["p"]),
        "c_d": float(row["c_d"]), "c_b": float(row["c_b"]),
    } for row in read_rows(RESULT / "outputs/casme3_tuned_native_selected_configs.csv")}
    trace_rows = read_rows(RESULT / "outputs/casme3_morph_prediction_trace.csv")
    trace = {(row["outer_subject"], row["video"], int(row["peak"]), row["source"]): row
             for row in trace_rows if row["trace_role"] == "weak_pool"}
    outer_rows = {row["outer_subject"]: row for row in read_rows(RESULT / "outputs/casme3_morph_outer_metrics.csv")
                  if row["outer_subject"] != "aggregate"}
    fit_audit_rows = []
    max_probability_error = 0.0
    probability_rows = 0
    selection_mismatches = 0
    count_mismatches = []
    high_preserved = True
    for held in subjects:
        config = selected[held]
        native_config = tuned[held]
        banks = {(str(record["subject"]), str(record["video"])):
                 run.build_pool(record, native_config, int(payload["k_p"])) for record in records}
        train = [record for record in records if str(record["subject"]) != held]
        test = [record for record in records if str(record["subject"]) == held]
        train_subjects = sorted({str(record["subject"]) for record in train})
        delta = config["Delta_p"]
        radius = config["actual_integer_radius"]
        c = config["C"]
        labels = run.label_training(train, banks, delta)
        ids = [(str(record["subject"]), str(record["video"])) for record in train]
        x = np.concatenate([banks[rid]["pools"][delta]["patches"][radius] for rid in ids])
        y = np.concatenate([labels[rid] for rid in ids])
        owner = np.concatenate([np.repeat(rid[0], len(labels[rid])) for rid in ids])
        _, audit = run.inner_oof(x, y, owner, train_subjects, held, c)
        for row in audit:
            row.update({"actual_integer_radius": radius, "C": c, "Delta_p": delta})
        fit_audit_rows.extend(audit)
        state = run.fit_model(x, y, c)
        probabilities = {}
        for record in test:
            rid = (held, str(record["video"]))
            probabilities[rid] = run.predict_model(state, banks[rid]["pools"][delta]["patches"][radius])
            for candidate, value in zip(banks[rid]["pools"][delta]["events"], probabilities[rid]):
                row = trace.get((held, rid[1], int(candidate["peak"]), "rescue"))
                if row is None:
                    selection_mismatches += 1
                    continue
                error = abs(float(row["evidence"]) - float(value))
                max_probability_error = max(max_probability_error, error)
                probability_rows += 1
                recorded_selected = str(row["selected"]).strip().lower() == "true"
                if recorded_selected != bool(float(value) >= config["gamma"]):
                    selection_mismatches += 1
        replay_counts, replay_trace = run.evaluate_subset(test, banks, config, probabilities, trace=True)
        expected = outer_rows[held]
        for key in ("TP", "FP", "FN", "event_count", "weak_candidate_count", "rescued_candidate_count",
                    "rescue_TP", "rescue_FP", "native_missed_GT_recovered"):
            if int(expected[key]) != int(replay_counts[key]):
                count_mismatches.append({"subject": held, "field": key,
                                         "expected": int(expected[key]), "replay": int(replay_counts[key])})
        high_expected = sum(len(banks[(held, str(record["video"]))]["high"]) for record in test)
        high_actual = sum(row["source"] == "tuned_native" for row in replay_trace)
        high_preserved &= high_expected == high_actual
    out = RESULT / "outputs"
    run.write_csv(out / "casme3_morph_inner_fit_audit.csv", fit_audit_rows)
    summary = {
        "author_native_replay": True,
        "tuned_native_selected_configs_valid": set(tuned) == set(subjects),
        "morphology_selected_configs_valid": set(selected) == set(subjects),
        "high_events_preserved": bool(high_preserved),
        "final_probability_rows_replayed": int(probability_rows),
        "max_final_probability_absolute_error": float(max_probability_error),
        "final_probability_replay": bool(max_probability_error <= 1e-12 and selection_mismatches == 0),
        "outer_metric_replay": not count_mismatches,
        "outer_metric_mismatches": count_mismatches,
        "single_class_inner_fit_rows": int(sum(bool(row["single_class_constant"]) for row in fit_audit_rows)),
        "inner_fit_audit_rows": len(fit_audit_rows),
        "selection_mismatches": int(selection_mismatches),
        "all_replay_checks_pass": bool(high_preserved and not count_mismatches and
                                        max_probability_error <= 1e-12 and selection_mismatches == 0),
    }
    (out / "casme3_integrity_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
