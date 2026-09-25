#!/usr/bin/env python3
"""Read-only independent audit for the completed refinement artifact."""
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/morphology_refinement_sammlv/outputs"
CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
SOURCE = ROOT / "results/rgr1_sammlv_nested/report.json"
V1 = ROOT / "results/first3_candidate_recovery_nested/outputs/morphology_outer_metrics.csv"
sys.path.insert(0, str(ROOT / "my_method/multi_scale_candidate_rescue"))
from run_mscr_nested_loso import match_events, tuned_native_decode  # noqa: E402


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def f1(counts):
    counts = np.asarray(counts, dtype=float)
    denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(2 * counts[..., 0], denominator,
                     out=np.zeros_like(denominator), where=denominator != 0)


def main():
    report = json.loads((OUT / "morph_refined_summary.json").read_text())
    assert report["completed_outer_folds"] == 29 and not report["incomplete"]
    assert report["cache_sha256"] == "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
    for path, expected in report["provenance"].items():
        assert digest(path) == expected, path
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    records = {(str(r["subject"]), str(r["video"])): r for r in payload["records"]}
    source = json.loads(SOURCE.read_text())
    folds = {f["subject"]: f for f in source["outer_folds"]}
    subjects = sorted(folds)
    assert len(records) == 79 and len(subjects) == 29
    final_trace = read(OUT / "morph_refined_prediction_trace.csv")
    final = [r for r in final_trace if r["trace_role"] == "final_prediction"]
    weak = [r for r in final_trace if r["trace_role"] == "weak_pool"]
    assert len(final) == 203
    totals = Counter(TP=0, FP=0, FN=0, event_count=0, rescue_TP=0, rescue_FP=0)
    high_count = 0
    for (subject, video), record in records.items():
        fold = folds[subject]["selected_strong_config"]
        config = {"c_s": fold["c_s"], "p": fold["p_s"],
                  "c_d": fold["c_d"], "c_b": fold["c_b"]}
        expected_high = tuned_native_decode({"score": record["score"]}, config)["events"]
        rows = [r for r in final if r["subject"] == subject and r["video"] == video]
        high = [r for r in rows if r["source"] == "tuned_native"]
        actual_high = [{"onset": int(r["onset"]), "peak": int(r["peak"]),
                        "offset": int(r["offset"]), "source": r["source"]} for r in high]
        assert actual_high == expected_high
        high_count += len(high)
        matches, unmatched = match_events(rows, record["samples"])
        assert matches == [int(r["matched_gt"]) for r in rows]
        totals.update(TP=sum(m >= 0 for m in matches), FP=sum(m < 0 for m in matches),
                      FN=len(unmatched), event_count=len(rows),
                      rescue_TP=sum(m >= 0 and r["source"] == "rescue"
                                    for m, r in zip(matches, rows)),
                      rescue_FP=sum(m < 0 and r["source"] == "rescue"
                                    for m, r in zip(matches, rows)))
    assert high_count == 192
    assert dict(totals) == {"TP": 54, "FP": 149, "FN": 105, "event_count": 203,
                            "rescue_TP": 5, "rescue_FP": 6}
    assert len(read(OUT / "morph_refined_inner_scores.csv")) == 29 * 192
    assert len(read(OUT / "morph_refined_inner_fit_audit.csv")) == 29 * 28 * 4 * 4 * 3
    selected = read(OUT / "morph_refined_selected_configs.csv")
    scores = read(OUT / "morph_refined_inner_scores.csv")
    assert len(selected) == 29
    for row in selected:
        candidates = [x for x in scores if x["outer_subject"] == row["outer_subject"]]

        def rank(score_row):
            config = json.loads(score_row["config_json"])
            tp, fp, fn = (int(score_row[key]) for key in ("TP", "FP", "FN"))
            denominator = 2 * tp + fp + fn
            return (-Fraction(2 * tp, denominator), int(score_row["FP"]),
                    int(score_row["rescued_candidate_count"]), config["Delta_p"],
                    -config["gamma"], config["requested_L_ratio"],
                    config["actual_integer_radius"], config["C"])

        best = json.loads(min(candidates, key=rank)["config_json"])
        assert all(row[key] == str(value) for key, value in best.items())
    fit_rows = read(OUT / "morph_refined_inner_fit_audit.csv")
    assert all(
        len(row["inner_training_subjects"].split(";")) == 27
        and row["outer_subject"] not in row["inner_training_subjects"].split(";")
        and row["inner_validation_subject"] not in row["inner_training_subjects"].split(";")
        for row in fit_rows
    )
    assert len(json.loads((OUT / "morph_refined_fitted_models.json").read_text())) == 29
    bootstrap = json.loads((OUT / "morph_refined_bootstrap.json").read_text())
    indices = np.asarray(bootstrap["indices"])
    np.testing.assert_array_equal(indices, np.random.default_rng(20260902).integers(
        0, 29, size=(1000, 29)))
    outer = {r["outer_subject"]: r for r in read(OUT / "morph_refined_outer_metrics.csv")}
    v1 = {r["outer_subject"]: r for r in read(V1)}
    tuned = {s: folds[s]["outer_metrics"]["Tuned Native"] for s in subjects}
    refined_array = np.asarray([[int(outer[s][key]) for key in ("TP", "FP", "FN")] for s in subjects])
    tuned_array = np.asarray([[int(tuned[s][key]) for key in ("TP", "FP", "FN")] for s in subjects])
    v1_array = np.asarray([[int(v1[s][key]) for key in ("TP", "FP", "FN")] for s in subjects])
    for name, values, expected in (
        ("Refined_minus_TunedNative", f1(refined_array[indices].sum(1)) - f1(tuned_array[indices].sum(1)), bootstrap["Refined_minus_TunedNative"]),
        ("Refined_minus_Morphology_v1", f1(refined_array[indices].sum(1)) - f1(v1_array[indices].sum(1)), bootstrap["Refined_minus_Morphology_v1"]),
    ):
        assert abs(float(values.mean()) - expected["mean_delta"]) < 1e-15, name
        np.testing.assert_allclose(np.percentile(values, [2.5, 97.5]), expected["ci95"], rtol=0, atol=1e-15)
    assert report["decision"] == "MORPHOLOGY-REFINED-WINNER"
    assert report["method_development_status"] == "SAMMLV-METHOD-DEVELOPMENT-FROZEN"
    summary = {
        "status": "PASS", "checks": [
            "anchor and v1 metrics",
            "203 final candidates and official matching",
            "192 high events exactly preserved",
            "5568 exact inner config replays",
            "38976 subject-disjoint fit audits",
            "29 final model states",
            "shared 1000-repetition bootstrap replay",
            "all provenance hashes",
        ],
        "single_class_inner_rows": sum(row["single_class_constant"] == "True" for row in fit_rows),
        "no_validation_candidate_rows": sum(row["no_validation_candidates"] == "True" for row in fit_rows),
    }
    (OUT / "INDEPENDENT_REFINEMENT_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT / "INDEPENDENT_REFINEMENT_AUDIT.md").write_text(
        "# Independent refinement artifact audit\n\n"
        + "\n".join(f"- PASS: {item}" for item in summary["checks"])
        + f"\n\n- Single-class inner rows: {summary['single_class_inner_rows']}"
        + f"\n- Empty validation candidate rows: {summary['no_validation_candidate_rows']}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
