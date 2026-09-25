#!/usr/bin/env python3
"""Independent replay audit for saved roughness-gate results."""
from __future__ import annotations

import csv
import json
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks

import run_roughness_gate as stage


def read_csv(name):
    with (stage.OUTPUTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = json.loads((stage.OUTPUTS / "roughness_summary.json").read_text(encoding="utf-8"))
    for path, digest in summary["input_sha256"].items():
        assert stage.sha256(stage.Path(path)) == digest
    datasets, gates = stage.gate()
    assert all(gates[name]["anchor_exact"] for name in gates)
    selected = read_csv("roughness_selected_rules.csv")
    fixed = read_csv("roughness_fixed_cs_control.csv")
    assignments = read_csv("roughness_video_assignments.csv")
    assert len(selected) == 123 and len(fixed) == 123 and len(assignments) == 3 * (79 + 462)
    selected_by_fold = {(r["dataset"], r["outer_subject"]): r for r in selected}
    fixed_by_fold = {(r["dataset"], r["outer_subject"]): r for r in fixed}

    for name, data in datasets.items():
        records, subjects, kp = data["records"], data["subjects"], data["kp"]
        index_by_video = {(str(r["subject"]), str(r["video"])): i for i, r in enumerate(records)}
        roughness = [stage.nr1(r["score"]) for r in records]

        @lru_cache(None)
        def independent(base, cs, index):
            p, cd, cb = base
            width = max(1, round(cs * kp))
            distance = max(1, round(cd * kp))
            radius = max(1, round(cb * kp))
            curve = np.convolve(np.asarray(records[index]["score"], dtype=float), np.ones(width) / width, mode="same")
            tau = float(curve.mean() + p * (curve.max() - curve.mean()))
            legal = find_peaks(curve, distance=distance)[0]
            kept = legal[curve[legal] >= tau]
            direct = find_peaks(curve, height=tau, distance=distance)[0]
            assert np.array_equal(kept, direct)
            events = [stage.native.event(peak, radius, "audit") for peak in kept]
            return stage.native.evaluate(records[index], events), len(legal), tau, width, distance, radius

        for held in subjects:
            config = data["configs"][held]
            base = stage.base_id(config)
            row = selected_by_fold[name, held]
            train_indices = [i for subject in subjects if subject != held for i in data["by_subject"][subject]]
            train_nr1 = np.asarray([roughness[i] for i in train_indices])
            candidates = []
            for rule_index, quantiles in enumerate(stage.RULES):
                thresholds = tuple(float(np.quantile(train_nr1, q, method="linear")) for q in quantiles)
                if not thresholds[0] < thresholds[1] < thresholds[2]:
                    continue
                counts = stage.empty_counts()
                for index in train_indices:
                    cs = stage.assign_cs(roughness[index], thresholds)
                    stage.add(counts, independent(base, cs, index)[0])
                candidates.append((stage.rank_counts(counts, rule_index), rule_index, quantiles, thresholds))
            chosen = min(candidates, key=lambda item: item[0])
            assert int(row["rule_index"]) == chosen[1]
            assert tuple(float(row[key]) for key in ("q1", "q2", "q3")) == chosen[2]
            np.testing.assert_allclose([float(row[key]) for key in ("r1", "r2", "r3")], chosen[3], rtol=0, atol=1e-14)

            fixed_candidates = []
            for cs_index, cs in enumerate(stage.CS_VALUES):
                counts = stage.empty_counts()
                for index in train_indices:
                    stage.add(counts, independent(base, cs, index)[0])
                fixed_candidates.append((stage.rank_counts(counts, cs_index), cs))
            assert float(fixed_by_fold[name, held]["selected_fixed_c_s"]) == min(fixed_candidates, key=lambda item: item[0])[1]

        for row in (r for r in assignments if r["dataset"] == name):
            held, method = row["outer_subject"], row["method"]
            index = index_by_video[held, row["video"]]
            config = data["configs"][held]
            base = stage.base_id(config)
            rule = selected_by_fold[name, held]
            thresholds = tuple(float(rule[key]) for key in ("r1", "r2", "r3"))
            if method == stage.METHODS[0]:
                cs = config["c_s"]
            elif method == stage.METHODS[1]:
                cs = float(fixed_by_fold[name, held]["selected_fixed_c_s"])
            else:
                cs = stage.assign_cs(roughness[index], thresholds)
            assert cs == float(row["assigned_c_s"])
            counts, legal, tau, width, distance, radius = independent(base, cs, index)
            assert all(counts[key] == int(row[key]) for key in ("TP", "FP", "FN"))
            assert counts["event_count"] == int(row["kept_peak_count"])
            assert legal == int(row["legal_peak_count"])
            assert abs(tau - float(row["threshold"])) < 1e-14
            assert (width, distance, radius) == tuple(int(row[key]) for key in ("smoothing_width", "peak_distance", "boundary_radius"))

    boots = json.loads((stage.OUTPUTS / "roughness_bootstrap.json").read_text(encoding="utf-8"))
    for name in ("SAMMLV", "CASME3"):
        rows = [r for r in read_csv(f"roughness_outer_metrics_{name.lower()}.csv") if r["row_type"] == "outer_subject"]
        subjects = datasets[name]["subjects"]
        by_method = {
            method: np.asarray([[int(next(r for r in rows if r["outer_subject"] == subject and r["method"] == method)[key]) for key in ("TP", "FP", "FN")] for subject in subjects])
            for method in stage.METHODS
        }
        indices = np.asarray(boots[name]["sampled_subject_indices"])
        def f1(values):
            totals = values[indices].sum(axis=1)
            den = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
            return np.divide(2 * totals[:, 0], den, out=np.zeros(len(totals)), where=den != 0)
        adaptive = f1(by_method[stage.METHODS[2]])
        for reference in stage.METHODS[:2]:
            delta = adaptive - f1(by_method[reference])
            saved = boots[name]["comparisons"][reference]
            np.testing.assert_allclose(delta, saved["replicate_delta_F1"], rtol=0, atol=1e-14)
            np.testing.assert_allclose(np.quantile(delta, [.025, .975]), saved["ci95"], rtol=0, atol=1e-14)
    assert stage.decision({name: {"comparisons": summary["datasets"][name]["comparisons"]} for name in ("SAMMLV", "CASME3")}, boots) == summary["decision"]
    audit = {"status": "PASS", "checks": {
        "input_hashes_unchanged": True, "both_final_strong_anchors_exact": True,
        "all_123_train_only_threshold_triplets_replayed": True,
        "all_123_selector_and_fixed_control_choices_replayed": True,
        "all_1623_test_video_decodings_replayed": True,
        "peak_distance_and_boundary_unchanged": True,
        "both_1000_subject_bootstraps_replayed": True, "decision_matches_evidence": True,
    }}
    stage.dump(stage.OUT / "SAVED_RESULTS_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
