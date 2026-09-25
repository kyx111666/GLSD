#!/usr/bin/env python3
"""Final Native closure and frozen Morphology strong-anchor stress test.

Native closure uses one fixed 75-config CASME3 grid.  The subsequent
Morphology run reuses the already-frozen implementation unchanged; only its
per-outer-fold high-confidence Native anchor is replaced by Final Strong
Native.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/final_native_closure_strong_anchor_morphology"
CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
MORPH_SUMMARY = ROOT / "results/morphology_casme3_locked_validation/outputs/casme3_morph_summary.json"
EXPANDED_SUMMARY = ROOT / "results/expanded_native_fairness_audit/expanded_native_summary.json"
EXPECTED_SHA = "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda"
EXPECTED_AUTHOR = (81, 912, 777)
EXPECTED_OLD = (111, 1079, 747)
EXPECTED_EXPANDED = (115, 1105, 743)
OLD_CONFIG = {"c_s": 1.5, "p": 0.45, "c_d": 0.75, "c_b": 0.75}
EXPANDED_CS = (1.0, 1.5, 2.0, 2.5)
EXPANDED_P = (0.35, 0.45, 0.55, 0.65)
EXPANDED_CD = (0.5, 0.75, 1.0, 1.25)
EXPANDED_CB = (0.5, 0.75, 1.0, 1.25)
FINAL_CS = (0.50, 0.75, 1.00, 1.25, 1.50)
FINAL_CD = (0.75, 1.00, 1.25, 1.50, 1.75)
FINAL_P = (0.35, 0.45, 0.55)
FINAL_CB = (0.75,)
SEED = 20260905

MORPH_DIR = ROOT / "my_method/morphology_casme3_locked_validation"
sys.path.insert(0, str(MORPH_DIR))
import run_casme3_locked_validation as frozen_morph  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def grid(cs, ps, cds, cbs):
    return [dict(zip(("c_s", "p", "c_d", "c_b"), values))
            for values in itertools.product(cs, ps, cds, cbs)]


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def empty_counts():
    return {"TP": 0, "FP": 0, "FN": 0, "event_count": 0,
            "weak_candidate_count": 0, "rescued_candidate_count": 0,
            "rescue_TP": 0, "rescue_FP": 0,
            "native_missed_GT_recovered": 0}


def add_counts(target, source):
    for key in target:
        target[key] += int(source.get(key, 0))


def f1(counts):
    den = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return 2 * counts["TP"] / den if den else 0.0


def metrics(counts):
    tp, fp, fn = int(counts["TP"]), int(counts["FP"]), int(counts["FN"])
    return {"TP": tp, "FP": fp, "FN": fn, "event_count": int(counts["event_count"]),
            "weak_candidate_count": int(counts.get("weak_candidate_count", 0)),
            "rescued_candidate_count": int(counts.get("rescued_candidate_count", 0)),
            "rescue_TP": int(counts.get("rescue_TP", 0)),
            "rescue_FP": int(counts.get("rescue_FP", 0)),
            "native_missed_GT_recovered": int(counts.get("native_missed_GT_recovered", 0)),
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn) if tp + fn else 0.0,
            "F1": f1(counts),
            "rescue_precision": counts.get("rescue_TP", 0) / (counts.get("rescue_TP", 0) + counts.get("rescue_FP", 0)) if counts.get("rescue_TP", 0) + counts.get("rescue_FP", 0) else 0.0}


def author_decode(record, k_p):
    curve = frozen_morph.moving_average(record["score"], 2 * k_p)
    tau = frozen_morph.threshold(curve, 0.55)
    peaks = find_peaks(curve, height=tau, distance=k_p)[0].astype(int)
    return [frozen_morph.event(peak, k_p, "native") for peak in peaks]


def tuned_decode(record, config, k_p):
    width = max(1, int(round(float(config["c_s"]) * k_p)))
    distance = max(1, int(round(float(config["c_d"]) * k_p)))
    boundary = max(1, int(round(float(config["c_b"]) * k_p)))
    curve = frozen_morph.moving_average(record["score"], width)
    tau = frozen_morph.threshold(curve, float(config["p"]))
    peaks = find_peaks(curve, height=tau, distance=distance)[0].astype(int)
    return [frozen_morph.event(peak, boundary, "tuned_native") for peak in peaks]


def evaluate(record, events):
    matches, unmatched = frozen_morph.match_events(events, record["samples"])
    return {"TP": sum(match >= 0 for match in matches),
            "FP": sum(match < 0 for match in matches), "FN": len(unmatched),
            "event_count": len(events)}


def precompute(records, subjects, configs, k_p):
    by_config_subject = []
    for config in configs:
        per_subject = {}
        for subject in subjects:
            total = empty_counts()
            for record in records:
                if str(record["subject"]) == subject:
                    add_counts(total, evaluate(record, tuned_decode(record, config, k_p)))
            per_subject[subject] = total
        by_config_subject.append(per_subject)
    return by_config_subject


def select_from_precompute(by_config_subject, configs, subjects, held):
    ranked = []
    for index, config in enumerate(configs):
        train_total = empty_counts()
        for subject in subjects:
            if subject != held:
                add_counts(train_total, by_config_subject[index][subject])
        den = 2 * train_total["TP"] + train_total["FP"] + train_total["FN"]
        rank = (-Fraction(2 * train_total["TP"], den) if den else Fraction(0),
                train_total["FP"], 0, index)
        ranked.append((rank, index, config, metrics(train_total)))
    _, index, config, inner = min(ranked, key=lambda row: row[0])
    return index, config, inner


def outer_metrics_from_precompute(by_config_subject, index, subjects, held):
    return metrics(by_config_subject[index][held])


def parameter_frequency(selected, subjects):
    output = {}
    spaces = {"c_s": FINAL_CS, "c_d": FINAL_CD, "p": FINAL_P, "c_b": FINAL_CB}
    for field, values in spaces.items():
        frequency = Counter(str(selected[subject][field]) for subject in subjects)
        endpoints = max(frequency.get(str(values[0]), 0), frequency.get(str(values[-1]), 0))
        output[field] = {"frequency": dict(sorted(frequency.items())),
                         "maximum_single_endpoint_frequency": int(endpoints),
                         "status": ("EXPANDED-NATIVE-BOUNDARY-REMAINS"
                                    if len(values) > 1 and endpoints > len(subjects) / 2 else None)}
    return output


def morphology_parameter_frequency(morph_results, subjects, k_p):
    spaces = {"Delta_p": tuple(frozen_morph.DELTAS),
              "requested_L_ratio": tuple(frozen_morph.L_RATIOS),
              "actual_integer_radius": tuple(sorted(set(frozen_morph.radius_map(k_p).values()))),
              "C": tuple(frozen_morph.CS), "gamma": tuple(frozen_morph.GAMMAS)}
    output = {"requested_L_ratio_to_actual_integer_radius": frozen_morph.radius_map(k_p)}
    for field, values in spaces.items():
        frequency = Counter(str(morph_results[subject]["selected_config"][field]) for subject in subjects)
        endpoints = max(frequency.get(str(values[0]), 0), frequency.get(str(values[-1]), 0))
        output[field] = {"frequency": dict(sorted(frequency.items())),
                         "maximum_single_endpoint_frequency": int(endpoints),
                         "status": "CASME3-BOUNDARY-SELECTION" if endpoints > len(subjects) / 2 else None}
    return output


def bootstrap(refined_rows, native_rows, subjects):
    refined = np.asarray([[refined_rows[s][key] for key in ("TP", "FP", "FN")] for s in subjects])
    native = np.asarray([[native_rows[s][key] for key in ("TP", "FP", "FN")] for s in subjects])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(subjects), size=(1000, len(subjects)))
    def values(array):
        totals = array[indices].sum(axis=1)
        den = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(2 * totals[:, 0], den, out=np.zeros(1000), where=den != 0)
    delta = values(refined) - values(native)
    return {"repetitions": 1000, "seed": SEED, "mean_delta": float(delta.mean()),
            "ci95": list(map(float, np.percentile(delta, [2.5, 97.5]))),
            "indices": indices.tolist()}


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    cache_hash = sha256(CACHE)
    if cache_hash != EXPECTED_SHA:
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / "final_native_closure_summary.json", {"status": "BLOCKED-CASME3-PROVENANCE",
              "observed_cache_sha256": cache_hash, "expected_cache_sha256": EXPECTED_SHA})
        raise RuntimeError("BLOCKED-CASME3-PROVENANCE")
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    records = [dict(record) for record in payload["records"]]
    k_p = int(payload["k_p"])
    subjects = sorted({str(record["subject"]) for record in records})
    observed = {"subjects": len(subjects), "videos": len(records),
                "gt": sum(len(record["samples"]) for record in records), "k_p": k_p}
    if observed != {"subjects": 94, "videos": 462, "gt": 858, "k_p": 17}:
        raise RuntimeError(f"BLOCKED-CASME3-PROVENANCE: {observed}")

    # Anchor gates: Author, old Tuned, and prior Expanded Tuned are all
    # recomputed with the verified score/matching implementation.
    author_subject = {}
    author_total = empty_counts()
    old_subject = {}
    old_total = empty_counts()
    for subject in subjects:
        a, o = empty_counts(), empty_counts()
        for record in records:
            if str(record["subject"]) == subject:
                add_counts(a, evaluate(record, author_decode(record, k_p)))
                add_counts(o, evaluate(record, tuned_decode(record, OLD_CONFIG, k_p)))
        author_subject[subject], old_subject[subject] = metrics(a), metrics(o)
        add_counts(author_total, a)
        add_counts(old_total, o)
    author_metrics, old_metrics = metrics(author_total), metrics(old_total)
    if tuple(author_total[key] for key in ("TP", "FP", "FN")) != EXPECTED_AUTHOR:
        raise RuntimeError(f"BLOCKED-CASME3-NATIVE-MISMATCH Author: {author_metrics}")
    if tuple(old_total[key] for key in ("TP", "FP", "FN")) != EXPECTED_OLD:
        raise RuntimeError(f"BLOCKED-CASME3-NATIVE-MISMATCH Old Tuned: {old_metrics}")

    expanded_configs = grid(EXPANDED_CS, EXPANDED_P, EXPANDED_CD, EXPANDED_CB)
    expanded_bank = precompute(records, subjects, expanded_configs, k_p)
    expanded_selected = {}
    expanded_subject = {}
    for held in subjects:
        index, config, inner = select_from_precompute(expanded_bank, expanded_configs, subjects, held)
        expanded_selected[held] = config
        expanded_subject[held] = outer_metrics_from_precompute(expanded_bank, index, subjects, held)
        del inner
    expanded_total = empty_counts()
    for row in expanded_subject.values():
        add_counts(expanded_total, row)
    expanded_metrics = metrics(expanded_total)
    if tuple(expanded_total[key] for key in ("TP", "FP", "FN")) != EXPECTED_EXPANDED:
        raise RuntimeError(f"BLOCKED-CASME3-NATIVE-MISMATCH Expanded Tuned: {expanded_metrics}")

    # Final Native closure, exactly one 75-config grid.
    final_configs = grid(FINAL_CS, FINAL_P, FINAL_CD, FINAL_CB)
    assert len(final_configs) == 75
    final_bank = precompute(records, subjects, final_configs, k_p)
    final_selected, final_inner, final_subject = {}, {}, {}
    for held in subjects:
        index, config, inner = select_from_precompute(final_bank, final_configs, subjects, held)
        final_selected[held], final_inner[held] = config, inner
        final_subject[held] = outer_metrics_from_precompute(final_bank, index, subjects, held)
    final_total = empty_counts()
    for row in final_subject.values():
        add_counts(final_total, row)
    final_metrics = metrics(final_total)
    final_frequency = parameter_frequency(final_selected, subjects)
    print(json.dumps({"anchors": {"author": author_metrics, "old": old_metrics, "expanded": expanded_metrics},
                      "final_native": final_metrics, "final_frequency": final_frequency}, indent=2), flush=True)

    # Frozen Morphology worker: its native_config map is the Final Strong
    # Native selection for each outer fold. No Morphology definition changes.
    context = {"records": records, "subjects": subjects, "k_p": k_p, "tuned_configs": final_selected}
    morph_results = {}
    with ProcessPoolExecutor(max_workers=4, initializer=frozen_morph.init_worker,
                             initargs=(context,)) as executor:
        futures = {executor.submit(frozen_morph.worker, subject): subject for subject in subjects}
        for future in as_completed(futures):
            result = future.result()
            morph_results[result["outer_subject"]] = result
            print(f"STRESS FOLD {result['outer_subject']} ({len(morph_results)}/{len(subjects)})", flush=True)
    assert set(morph_results) == set(subjects)
    morph_subject = {subject: morph_results[subject]["outer_metrics"] for subject in subjects}
    morph_total = empty_counts()
    for row in morph_subject.values():
        add_counts(morph_total, row)
    morph_metrics = metrics(morph_total)
    morph_frequency = morphology_parameter_frequency(morph_results, subjects, k_p)

    # Subject sensitivity and bootstrap are computed against Final Strong Native.
    per_subject_delta = {s: float(frozen_morph.f1(morph_subject[s]) - f1(final_subject[s])) for s in subjects}
    loo_delta = {}
    for subject in subjects:
        m = {key: sum(morph_subject[s][key] for s in subjects if s != subject) for key in ("TP", "FP", "FN")}
        n = {key: sum(final_subject[s][key] for s in subjects if s != subject) for key in ("TP", "FP", "FN")}
        loo_delta[subject] = float(frozen_morph.f1(m) - f1(n))
    stability = {"improved": int(sum(v > 0 for v in per_subject_delta.values())),
                 "equal": int(sum(v == 0 for v in per_subject_delta.values())),
                 "worse": int(sum(v < 0 for v in per_subject_delta.values())),
                 "per_subject_delta_F1": per_subject_delta,
                 "leave_one_subject_out_aggregate_delta": loo_delta,
                 "minimum_leave_one_subject_out_delta": float(min(loo_delta.values()))}
    boot = bootstrap(morph_subject, final_subject, subjects)
    final_native_f1 = final_metrics["F1"]
    morph_f1 = morph_metrics["F1"]
    if morph_f1 > final_native_f1 and morph_total["rescue_TP"] > 0:
        decision = "MORPHOLOGY-INCREMENTAL-VALUE-SURVIVES"
    else:
        decision = "MORPHOLOGY-CASME3-INCREMENTAL-VALUE-NOT-CONFIRMED"

    provenance = {
        "cache_path": str(CACHE.resolve()), "cache_sha256": cache_hash,
        "gt_source": "verified CASME3 cache records[*].samples (onset, apex, offset)",
        "subjects": 94, "videos": 462, "gt": 858, "k_p": 17,
        "native_audit_path": str(NATIVE_AUDIT.resolve()), "native_audit_sha256": sha256(NATIVE_AUDIT),
        "paper_metrics_reference_path": str(PAPER_METRICS.resolve()), "paper_metrics_reference_sha256": sha256(PAPER_METRICS),
        "frozen_morphology_summary_path": str(MORPH_SUMMARY.resolve()), "frozen_morphology_summary_sha256": sha256(MORPH_SUMMARY),
        "expanded_native_audit_summary_path": str(EXPANDED_SUMMARY.resolve()), "expanded_native_audit_summary_sha256": sha256(EXPANDED_SUMMARY),
    }
    summary = {
        "status": "CASME3-NATIVE-BASELINE-PERMANENTLY-FROZEN",
        "validation_status": "FINAL-NATIVE-CLOSURE-AND-STRONG-ANCHOR-STRESS-TEST-COMPLETE",
        "decision": decision, "provenance": provenance,
        "anchors": {"Author Native": author_metrics, "Old Tuned Native": old_metrics,
                    "Expanded Tuned Native": expanded_metrics},
        "final_strong_native": {"metrics": final_metrics, "selected_parameter_frequency": final_frequency,
                                "grid_count": len(final_configs), "grid": {"c_s": FINAL_CS, "c_d": FINAL_CD,
                                "p": FINAL_P, "c_b": FINAL_CB},
                                "selection": "outer subject LOSO; pooled outer-train subject Spotting F1"},
        "frozen_morphology_stress": {"metrics": morph_metrics,
                                      "selected_parameter_frequency": morph_frequency,
                                      "high_anchor": "per outer-fold Final Strong Native",
                                      "feature": "peak-relative [H(p), x1,...,x31]",
                                      "classifier": "LogisticRegression(liblinear, balanced)",
                                      "Delta_p": [0.15, 0.20, 0.25, 0.30],
                                      "L_over_k_p": [0.25, 0.50, 0.75, 1.00],
                                      "C": [0.1, 1.0, 10.0], "gamma": [0.60, 0.70, 0.80, 0.90]},
        "subject_stability": stability, "bootstrap": boot,
        "integrity": {"author_anchor_exact": True, "old_tuned_anchor_exact": True,
                      "expanded_tuned_anchor_exact": True, "morphology_modified": False,
                      "morphology_grid_modified": False, "morphology_classifier_modified": False,
                      "morphology_candidate_rule_modified": False, "outer_test_gt_in_selection": False,
                      "high_event_preservation_all_folds": bool(all(morph_results[s]["high_preserved"] for s in subjects)),
                      "recognition_STRS_run": False, "new_native_grid_after_closure": False},
        "permanently_frozen": True,
    }
    OUT.mkdir(parents=True)
    dump(OUT / "final_native_closure_summary.json", summary)
    dump(OUT / "strong_anchor_morph_bootstrap.json", boot)
    dump(OUT / "strong_anchor_morph_parameter_stability.json", morph_frequency)
    write_csv(OUT / "final_strong_native_outer_metrics.csv",
              [{"outer_subject": s, **final_subject[s], "inner_F1": final_inner[s]["F1"]} for s in subjects] +
              [{"outer_subject": "aggregate", **final_metrics}])
    write_csv(OUT / "final_strong_native_selected_configs.csv",
              [{"outer_subject": s, **final_selected[s], "inner_F1": final_inner[s]["F1"]} for s in subjects])
    write_csv(OUT / "strong_anchor_morph_outer_metrics.csv",
              [{"outer_subject": s, **morph_subject[s],
                **{f"selected_{key}": value for key, value in morph_results[s]["selected_config"].items()},
                "native_config": config_key(final_selected[s]),
                "inner_F1": morph_results[s]["inner_metrics"]["F1"]} for s in subjects] +
              [{"outer_subject": "aggregate", **morph_metrics}])
    write_csv(OUT / "strong_anchor_morph_selected_configs.csv",
              [{"outer_subject": s, **morph_results[s]["selected_config"],
                "native_config": config_key(final_selected[s]),
                "inner_F1": morph_results[s]["inner_metrics"]["F1"]} for s in subjects])
    write_csv(OUT / "strong_anchor_morph_inner_scores.csv",
              [row for s in subjects for row in morph_results[s]["inner_rows"]])
    pred_rows = []
    for s in subjects:
        pred_rows.extend([{**row, "trace_role": "final_prediction", "native_config": config_key(final_selected[s])}
                          for row in morph_results[s]["predictions"]])
        pred_rows.extend([{**row, "trace_role": "weak_pool", "native_config": config_key(final_selected[s])}
                          for row in morph_results[s]["weak_trace"]])
    write_csv(OUT / "strong_anchor_morph_prediction_trace.csv", pred_rows)
    write_csv(OUT / "strong_anchor_morph_subject_deltas.csv",
              [{"outer_subject": s, "delta_F1": per_subject_delta[s], "loo_delta_F1": loo_delta[s]} for s in subjects])
    write_csv(OUT / "strong_anchor_morph_inner_fit_audit.csv",
              [row for s in subjects for row in morph_results[s]["fit_audit"]])

    lines = [
        "# Final Native Closure + Strong-Anchor Morphology Stress Test", "",
        "本报告执行一次且仅一次 CASME3 Final Native closure（75 configs），随后以每个 outer fold 的 Final Strong Native 作为 Morphology high-confidence anchor。Morphology 的 feature、classifier、grid、normalization、candidate rule 与 evaluator 均未修改。", "",
        "## Anchor gates", "",
        f"- Author Native exact：{author_metrics['TP']}/{author_metrics['FP']}/{author_metrics['FN']}, F1={author_metrics['F1']:.6f}。",
        f"- Old Tuned Native exact：{old_metrics['TP']}/{old_metrics['FP']}/{old_metrics['FN']}, F1={old_metrics['F1']:.6f}。",
        f"- Expanded Tuned Native exact：{expanded_metrics['TP']}/{expanded_metrics['FP']}/{expanded_metrics['FN']}, F1={expanded_metrics['F1']:.6f}。", "",
        "## Provenance", "",
        f"- Cache：`{CACHE.resolve()}`；SHA-256=`{cache_hash}`；94 subjects / 462 videos / 858 GT / k_p=17。",
        "- GT source：verified cache `records[*].samples`（onset, apex, offset）。",
        f"- Evaluator：`{NATIVE_AUDIT.resolve()}`；hash=`{provenance['native_audit_sha256']}`；paper reference hash=`{provenance['paper_metrics_reference_sha256']}`。", "",
        "## Final Strong Native closure", "",
        "- Grid：c_s={0.50,0.75,1.00,1.25,1.50}；c_d={0.75,1.00,1.25,1.50,1.75}；p={0.35,0.45,0.55}；c_b=0.75 fixed；75 configs。",
        f"- 结果：{final_metrics['TP']}/{final_metrics['FP']}/{final_metrics['FN']}，Precision={final_metrics['Precision']:.6f}，Recall={final_metrics['Recall']:.6f}，F1={final_metrics['F1']:.6f}。",
        f"- 参数频率：`{json.dumps(final_frequency, ensure_ascii=False)}`。", "",
        "## Strong-Anchor Morphology", "",
        "- Per-fold high anchor：Final Strong Native；其余 Morphology 定义完全冻结。",
        f"- 结果：{morph_metrics['TP']}/{morph_metrics['FP']}/{morph_metrics['FN']}，Precision={morph_metrics['Precision']:.6f}，Recall={morph_metrics['Recall']:.6f}，F1={morph_metrics['F1']:.6f}。",
        f"- Rescue TP/FP={morph_metrics['rescue_TP']}/{morph_metrics['rescue_FP']}，rescue precision={morph_metrics['rescue_precision']:.6f}。",
        f"- Morphology selected parameter frequency：`{json.dumps(morph_frequency, ensure_ascii=False)}`。",
        f"- improved/equal/worse={stability['improved']}/{stability['equal']}/{stability['worse']}；最小 leave-one-subject-out ΔF1={stability['minimum_leave_one_subject_out_delta']:+.6f}。",
        f"- 1000 subject bootstrap：mean ΔF1={boot['mean_delta']:+.6f}，95% CI=[{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}]。", "",
        "## Decision", "",
        f"- Strong-Anchor Morphology − Final Strong Native ΔF1={morph_f1 - final_native_f1:+.6f}。",
        f"- **{decision}**", "- **CASME3-NATIVE-BASELINE-PERMANENTLY-FROZEN**", "- **FINAL-NATIVE-CLOSURE-AND-STRESS-TEST-COMPLETE**", "",
        "无论结果如何，不修改 Morphology、不扩大 Native grid、不进入新实验。", "",
        "## Outputs", ""]
    lines.extend(f"- `{path.resolve()}`" for path in sorted(OUT.iterdir()))
    (OUT / "FINAL_NATIVE_CLOSURE_STRONG_ANCHOR_MORPHOLOGY_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "final_native": final_metrics,
                      "strong_anchor_morphology": morph_metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
