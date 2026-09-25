#!/usr/bin/env python3
"""Roughness-calibrated smoothing feasibility gate, cache-only."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/roughness_calibrated_smoothing"
OUTPUTS = OUT / "outputs"
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
SAMM_SELECTED = ROOT / "results/rgr1_sammlv_nested/report.json"
CAS_SELECTED = ROOT / "results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv"
CAS_OUTER = CAS_SELECTED.with_name("final_strong_native_outer_metrics.csv")
SAMM_CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
CAS_CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"

spec = importlib.util.spec_from_file_location("native_only", NATIVE_SOURCE)
native = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(native)

CS_VALUES = (0.75, 1.00, 1.50, 2.00)
Q1_VALUES = (0.20, 0.30)
Q2_VALUES = (0.45, 0.55)
Q3_VALUES = (0.70, 0.80)
RULES = tuple(itertools.product(Q1_VALUES, Q2_VALUES, Q3_VALUES))
METHODS = ("Final Strong Native", "Best Fixed-c_s Control", "Roughness-Calibrated Smoothing")
COUNT_KEYS = ("TP", "FP", "FN", "event_count")
SEED = 20260905
BOOTSTRAP_REPS = 1000
TIE_TOLERANCE = 0.001
EXPECTED = {
    "SAMMLV": {
        "cache": SAMM_CACHE,
        "sha": "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569",
        "subjects": 29, "videos": 79, "gt": 159, "k_p": 5,
        "strong": (49, 143, 110), "strong_f1": 0.2792022792022792,
    },
    "CASME3": {
        "cache": CAS_CACHE,
        "sha": "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda",
        "subjects": 94, "videos": 462, "gt": 858, "k_p": 17,
        "strong": (124, 1148, 734), "strong_f1": 0.11643192488262911,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def empty_counts():
    return {key: 0 for key in COUNT_KEYS}


def add(target, source):
    for key in COUNT_KEYS:
        target[key] += int(source[key])


def subtract(a, b):
    return {key: int(a[key]) - int(b[key]) for key in COUNT_KEYS}


def metric(counts):
    tp, fp, fn = (int(counts[k]) for k in ("TP", "FP", "FN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "event_count": int(counts["event_count"]),
            "Precision": precision, "Recall": recall, "F1": f1}


def nr1(score):
    score = np.asarray(score, dtype=np.float64)
    return float(np.mean(np.abs(np.diff(score))) / (np.std(score) + 1e-8)) if len(score) > 1 else 0.0


def nr2(score):
    score = np.asarray(score, dtype=np.float64)
    if len(score) < 3:
        return 0.0
    return float(np.mean(np.abs(score[2:] - 2 * score[1:-1] + score[:-2])) / (np.std(score) + 1e-8))


def config_id(config):
    return tuple(float(config[k]) for k in ("c_s", "p", "c_d", "c_b"))


def base_id(config):
    return tuple(float(config[k]) for k in ("p", "c_d", "c_b"))


def integer_settings(config, kp):
    return {
        "smoothing_width": max(1, int(round(float(config["c_s"]) * kp))),
        "peak_distance": max(1, int(round(float(config["c_d"]) * kp))),
        "boundary_radius": max(1, int(round(float(config["c_b"]) * kp))),
    }


def decode_record(record, base, cs, kp):
    config = {"c_s": cs, "p": base[0], "c_d": base[1], "c_b": base[2]}
    ints = integer_settings(config, kp)
    curve = native.moving_average(record["score"], ints["smoothing_width"])
    tau = native.threshold(curve, config["p"])
    legal = find_peaks(curve, distance=ints["peak_distance"])[0].astype(int)
    kept = find_peaks(curve, height=tau, distance=ints["peak_distance"])[0].astype(int)
    events = [native.event(peak, ints["boundary_radius"], "roughness_gate") for peak in kept]
    counts = native.evaluate(record, events)
    return {
        "counts": counts, "threshold": float(tau), "legal_peak_count": int(len(legal)),
        "kept_peak_count": int(len(kept)), "peak_times": kept.tolist(),
        "smoothing_width": ints["smoothing_width"], "peak_distance": ints["peak_distance"],
        "boundary_radius": ints["boundary_radius"],
    }


def load_selected():
    configs = {"SAMMLV": {}, "CASME3": {}}
    references = {"SAMMLV": {}, "CASME3": {}}
    samm = json.loads(SAMM_SELECTED.read_text(encoding="utf-8"))
    for fold in samm["outer_folds"]:
        cfg = fold["selected_strong_config"]
        subject = str(fold["subject"])
        configs["SAMMLV"][subject] = {"c_s": float(cfg["c_s"]), "p": float(cfg["p_s"]),
                                              "c_d": float(cfg["c_d"]), "c_b": float(cfg["c_b"])}
        references["SAMMLV"][subject] = {key: int(fold["outer_metrics"]["Tuned Native"][key]) for key in COUNT_KEYS}
    with CAS_SELECTED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            configs["CASME3"][str(row["outer_subject"])] = {key: float(row[key]) for key in ("c_s", "p", "c_d", "c_b")}
    with CAS_OUTER.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["outer_subject"] != "aggregate":
                references["CASME3"][str(row["outer_subject"])] = {key: int(row[key]) for key in COUNT_KEYS}
    return configs, references


def gate():
    configs, references = load_selected()
    datasets, provenance = {}, {}
    for name, expected in EXPECTED.items():
        if sha256(expected["cache"]) != expected["sha"]:
            raise RuntimeError(f"{name} cache SHA mismatch")
        payload, records, subjects, observed = native.load_payload(expected["cache"])
        wanted = {key: expected[key] for key in ("subjects", "videos", "gt", "k_p")}
        if observed != wanted:
            raise RuntimeError(f"{name} metadata mismatch: {observed}")
        if set(configs[name]) != set(subjects):
            raise RuntimeError(f"{name} frozen-config subject mismatch")
        by_subject = {subject: [] for subject in subjects}
        roughness = []
        for index, record in enumerate(records):
            subject = str(record["subject"])
            by_subject[subject].append(index)
            roughness.append((nr1(record["score"]), nr2(record["score"])))
        total = empty_counts()
        per_subject = {}
        for subject in subjects:
            counts = empty_counts()
            for index in by_subject[subject]:
                events = native.tuned_decode(records[index], configs[name][subject], expected["k_p"])
                add(counts, native.evaluate(records[index], events))
            if counts != references[name][subject]:
                raise RuntimeError(f"{name} subject {subject} anchor mismatch")
            per_subject[subject] = counts
            add(total, counts)
        values = metric(total)
        if tuple(values[key] for key in ("TP", "FP", "FN")) != expected["strong"] or abs(values["F1"] - expected["strong_f1"]) > 1e-14:
            raise RuntimeError(f"{name} aggregate anchor mismatch: {values}")
        datasets[name] = {"records": records, "subjects": subjects, "by_subject": by_subject,
                          "configs": configs[name], "baseline": per_subject, "roughness": roughness,
                          "kp": expected["k_p"]}
        provenance[name] = {"cache_path": str(expected["cache"].resolve()), "cache_sha256": expected["sha"],
                            **observed, "Final Strong Native": values,
                            "frozen_config_frequency": dict(Counter(str(config_id(c)) for c in configs[name].values())),
                            "anchor_exact": True}
    return datasets, provenance


def precompute(data):
    records, kp = data["records"], data["kp"]
    bases = sorted({base_id(config) for config in data["configs"].values()})
    cache = {}
    for base, cs, index in itertools.product(bases, CS_VALUES, range(len(records))):
        cache[(base, cs, index)] = decode_record(records[index], base, cs, kp)
    return cache


def assign_cs(value, thresholds):
    r1, r2, r3 = thresholds
    if value < r1:
        return 0.75
    if value < r2:
        return 1.00
    if value < r3:
        return 1.50
    return 2.00


def rank_counts(counts, index):
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    score = 2 * counts["TP"] / denominator if denominator else 0.0
    return (-score, counts["FP"], index)


def run_dataset(name, data):
    records, subjects, roughness = data["records"], data["subjects"], data["roughness"]
    cached = precompute(data)
    outer_rows, rule_rows, fixed_rows, video_rows = [], [], [], []
    subject_counts = {method: {} for method in METHODS}
    rule_frequency = Counter()
    fixed_frequency = Counter()
    assignment_frequency = Counter()

    for held in subjects:
        config = data["configs"][held]
        base = base_id(config)
        train_indices = [index for subject in subjects if subject != held for index in data["by_subject"][subject]]
        test_indices = list(data["by_subject"][held])
        train_nr1 = np.asarray([roughness[index][0] for index in train_indices], dtype=float)
        candidates = []
        for rule_index, quantiles in enumerate(RULES):
            thresholds = tuple(float(np.quantile(train_nr1, q, method="linear")) for q in quantiles)
            if not thresholds[0] < thresholds[1] < thresholds[2]:
                continue
            counts = empty_counts()
            assigned = Counter()
            for index in train_indices:
                cs = assign_cs(roughness[index][0], thresholds)
                assigned[str(cs)] += 1
                add(counts, cached[(base, cs, index)]["counts"])
            candidates.append((rank_counts(counts, rule_index), rule_index, quantiles, thresholds, counts, assigned))
        if not candidates:
            raise RuntimeError(f"{name}/{held}: no strictly increasing selector thresholds")
        _, selected_rule_index, selected_quantiles, thresholds, selected_train, train_assigned = min(candidates, key=lambda row: row[0])
        rule_frequency[str(selected_quantiles)] += 1

        fixed_candidates = []
        for fixed_index, cs in enumerate(CS_VALUES):
            counts = empty_counts()
            for index in train_indices:
                add(counts, cached[(base, cs, index)]["counts"])
            fixed_candidates.append((rank_counts(counts, fixed_index), fixed_index, cs, counts))
        _, _, selected_fixed_cs, fixed_train = min(fixed_candidates, key=lambda row: row[0])
        fixed_frequency[str(selected_fixed_cs)] += 1

        rule_rows.append({
            "dataset": name, "outer_subject": held, "rule_index": selected_rule_index,
            "q1": selected_quantiles[0], "q2": selected_quantiles[1], "q3": selected_quantiles[2],
            "r1": thresholds[0], "r2": thresholds[1], "r3": thresholds[2],
            "train_subject_count": len(subjects) - 1, "train_video_count": len(train_indices),
            "train_subjects": json.dumps([s for s in subjects if s != held]),
            "train_TP": selected_train["TP"], "train_FP": selected_train["FP"], "train_FN": selected_train["FN"],
            "train_F1": metric(selected_train)["F1"], "train_assignment_frequency": json.dumps(dict(train_assigned), sort_keys=True),
            "frozen_p": config["p"], "frozen_c_d": config["c_d"], "frozen_c_b": config["c_b"],
        })
        fixed_rows.append({
            "dataset": name, "outer_subject": held, "selected_fixed_c_s": selected_fixed_cs,
            "train_TP": fixed_train["TP"], "train_FP": fixed_train["FP"], "train_FN": fixed_train["FN"],
            "train_F1": metric(fixed_train)["F1"], "frozen_p": config["p"],
            "frozen_c_d": config["c_d"], "frozen_c_b": config["c_b"],
        })

        totals = {method: empty_counts() for method in METHODS}
        for index in test_indices:
            record = records[index]
            choices = {
                METHODS[0]: config["c_s"],
                METHODS[1]: selected_fixed_cs,
                METHODS[2]: assign_cs(roughness[index][0], thresholds),
            }
            for method, cs in choices.items():
                decoded = cached[(base, cs, index)]
                add(totals[method], decoded["counts"])
                if method == METHODS[2]:
                    assignment_frequency[str(cs)] += 1
                video_rows.append({
                    "dataset": name, "outer_subject": held, "subject": str(record["subject"]), "video": str(record["video"]),
                    "method": method, "NR1": roughness[index][0], "NR2": roughness[index][1], "assigned_c_s": cs,
                    "smoothing_width": decoded["smoothing_width"], "peak_distance": decoded["peak_distance"],
                    "boundary_radius": decoded["boundary_radius"], "threshold": decoded["threshold"],
                    "legal_peak_count": decoded["legal_peak_count"], "kept_peak_count": decoded["kept_peak_count"],
                    "peak_density_after_smoothing_per_1000": 1000 * decoded["legal_peak_count"] / len(record["score"]),
                    "events_per_video": decoded["counts"]["event_count"], "TP": decoded["counts"]["TP"],
                    "FP": decoded["counts"]["FP"], "FN": decoded["counts"]["FN"],
                })
        for method in METHODS:
            subject_counts[method][held] = totals[method]
            outer_rows.append({"dataset": name, "row_type": "outer_subject", "outer_subject": held,
                               "method": method, **metric(totals[method]),
                               "events_per_video": totals[method]["event_count"] / len(test_indices)})

    aggregate = {}
    for method in METHODS:
        total = empty_counts()
        for subject in subjects:
            add(total, subject_counts[method][subject])
        aggregate[method] = {**metric(total), "events_per_video": total["event_count"] / len(records)}
        outer_rows.append({"dataset": name, "row_type": "aggregate", "outer_subject": "aggregate",
                           "method": method, **aggregate[method]})

    comparisons = {}
    for reference in METHODS[:2]:
        improved = equal = worse = 0
        loo = []
        adaptive_total = aggregate[METHODS[2]]
        reference_total = aggregate[reference]
        for subject in subjects:
            a = metric(subject_counts[METHODS[2]][subject])["F1"]
            b = metric(subject_counts[reference][subject])["F1"]
            if a > b + 1e-15:
                improved += 1
            elif a < b - 1e-15:
                worse += 1
            else:
                equal += 1
            a_loo = metric(subtract(adaptive_total, subject_counts[METHODS[2]][subject]))["F1"]
            b_loo = metric(subtract(reference_total, subject_counts[reference][subject]))["F1"]
            loo.append({"omitted_subject": subject, "adaptive_F1": a_loo, "reference_F1": b_loo, "delta_F1": a_loo - b_loo})
        comparisons[reference] = {
            "observed_delta_F1": aggregate[METHODS[2]]["F1"] - aggregate[reference]["F1"],
            "improved": improved, "equal": equal, "worse": worse,
            "leave_one_subject_out": loo,
            "minimum_leave_one_subject_out_delta_F1": min(row["delta_F1"] for row in loo),
            "maximum_leave_one_subject_out_delta_F1": max(row["delta_F1"] for row in loo),
        }

    assigned_nr1 = {}
    adaptive_rows = [row for row in video_rows if row["method"] == METHODS[2]]
    for cs in CS_VALUES:
        values = np.asarray([row["NR1"] for row in adaptive_rows if row["assigned_c_s"] == cs], dtype=float)
        assigned_nr1[str(cs)] = {
            "n": int(len(values)), "fraction": float(len(values) / len(adaptive_rows)),
            "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p25": float(np.quantile(values, .25)) if len(values) else None,
            "p75": float(np.quantile(values, .75)) if len(values) else None,
        }
    peak_density = {}
    for method in METHODS:
        values = [row["peak_density_after_smoothing_per_1000"] for row in video_rows if row["method"] == method]
        peak_density[method] = {"mean": float(np.mean(values)), "median": float(np.median(values))}
    return {
        "outer_rows": outer_rows, "rule_rows": rule_rows, "fixed_rows": fixed_rows, "video_rows": video_rows,
        "subject_counts": subject_counts, "aggregate": aggregate, "comparisons": comparisons,
        "selected_rule_frequency": dict(rule_frequency), "selected_fixed_c_s_frequency": dict(fixed_frequency),
        "adaptive_assignment_frequency": {str(cs): int(assignment_frequency[str(cs)]) for cs in CS_VALUES},
        "NR1_by_assigned_c_s": assigned_nr1, "peak_density_after_smoothing": peak_density,
        "invariants": {"outer_test_subject_excluded_from_threshold_quantiles": True,
                       "strict_threshold_order_all_folds": all(row["r1"] < row["r2"] < row["r3"] for row in rule_rows),
                       "same_rule_grid": list(RULES), "test_video_count": len(adaptive_rows)},
    }


def bootstrap(name, result):
    subjects = list(result["subject_counts"][METHODS[2]])
    rng = np.random.default_rng(SEED + (0 if name == "SAMMLV" else 1))
    indices = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPS, len(subjects)))
    arrays = {
        method: np.asarray([[result["subject_counts"][method][subject][key] for key in ("TP", "FP", "FN")] for subject in subjects], dtype=np.int64)
        for method in METHODS
    }
    def f1(matrix):
        total = matrix[indices].sum(axis=1)
        denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
        return np.divide(2 * total[:, 0], denominator, out=np.zeros(len(total), dtype=float), where=denominator != 0)
    adaptive = f1(arrays[METHODS[2]])
    out = {"dataset": name, "repetitions": BOOTSTRAP_REPS, "seed": SEED + (0 if name == "SAMMLV" else 1),
           "sampled_subject_indices": indices.tolist(), "comparisons": {}}
    for reference in METHODS[:2]:
        delta = adaptive - f1(arrays[reference])
        out["comparisons"][reference] = {
            "mean_delta_F1": float(np.mean(delta)), "ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
            "fraction_positive": float(np.mean(delta > 0)), "replicate_delta_F1": delta.tolist(),
        }
    return out


def decision(results, boots):
    def clearly_degraded(name, reference):
        delta = results[name]["comparisons"][reference]["observed_delta_F1"]
        upper = boots[name]["comparisons"][reference]["ci95"][1]
        return delta < -TIE_TOLERANCE and upper < 0
    if any(clearly_degraded(name, reference) for name in results for reference in METHODS[:2]):
        return "ROUGHNESS-SMOOTHING-NO-GO"
    strong = all(
        results[name]["comparisons"][reference]["observed_delta_F1"] > 0
        and results[name]["comparisons"][reference]["minimum_leave_one_subject_out_delta_F1"] > 0
        for name in results for reference in METHODS[:2]
    )
    if strong:
        return "ROUGHNESS-SMOOTHING-STRONG-GO"
    improved_datasets = sum(all(results[name]["comparisons"][reference]["observed_delta_F1"] > TIE_TOLERANCE for reference in METHODS[:2]) for name in results)
    tied_datasets = sum(all(abs(results[name]["comparisons"][reference]["observed_delta_F1"]) <= TIE_TOLERANCE and boots[name]["comparisons"][reference]["ci95"][0] <= 0 <= boots[name]["comparisons"][reference]["ci95"][1] for reference in METHODS[:2]) for name in results)
    if improved_datasets == 1 and tied_datasets == 1:
        return "ROUGHNESS-SMOOTHING-WEAK-GO"
    return "ROUGHNESS-SMOOTHING-NO-GO"


def report(summary):
    lines = [
        "# ME-TST+ Roughness-Calibrated Smoothing — Feasibility Gate", "",
        f"判定：`{summary['decision']}`。", "", "## Provenance / anchor gate", "",
    ]
    for name in ("SAMMLV", "CASME3"):
        p = summary["provenance_anchor_gate"][name]
        m = p["Final Strong Native"]
        lines += [f"- {name}：{p['subjects']} subjects / {p['videos']} videos / {p['gt']} GT / k_p={p['k_p']}；cache SHA-256 `{p['cache_sha256']}`。",
                  f"  - Final Strong Native PASS：{m['TP']}/{m['FP']}/{m['FN']}，F1={m['F1']:.6f}。"]
    lines += ["", "## Aggregate results", "", "| Dataset | Method | TP | FP | FN | Precision | Recall | F1 | events/video |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("SAMMLV", "CASME3"):
        for method in METHODS:
            m = summary["datasets"][name]["aggregate_metrics"][method]
            lines.append(f"| {name} | {method} | {m['TP']} | {m['FP']} | {m['FN']} | {m['Precision']:.6f} | {m['Recall']:.6f} | {m['F1']:.6f} | {m['events_per_video']:.4f} |")
    lines += ["", "## Stability and bootstrap", "", "| Dataset | Comparison | observed ΔF1 | improved/equal/worse | min leave-one-subject-out ΔF1 | bootstrap mean | 95% CI |", "|---|---|---:|---|---:|---:|---|"]
    for name in ("SAMMLV", "CASME3"):
        for reference in METHODS[:2]:
            c = summary["datasets"][name]["comparisons"][reference]
            b = summary["bootstrap"][name]["comparisons"][reference]
            lines.append(f"| {name} | Adaptive − {reference} | {c['observed_delta_F1']:+.6f} | {c['improved']}/{c['equal']}/{c['worse']} | {c['minimum_leave_one_subject_out_delta_F1']:+.6f} | {b['mean_delta_F1']:+.6f} | [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] |")
    lines += ["", "## Selector and assignment stability", ""]
    for name in ("SAMMLV", "CASME3"):
        d = summary["datasets"][name]
        lines += [f"### {name}", "", f"- selected selector frequency：`{json.dumps(d['selected_rule_frequency'], ensure_ascii=False)}`。",
                  f"- selected fixed-c_s frequency：`{json.dumps(d['selected_fixed_c_s_frequency'], ensure_ascii=False)}`。",
                  f"- adaptive video assignment frequency：`{json.dumps(d['adaptive_assignment_frequency'], ensure_ascii=False)}`。",
                  f"- NR1 by assigned c_s：`{json.dumps(d['NR1_by_assigned_c_s'], ensure_ascii=False)}`。",
                  f"- peak density after smoothing：`{json.dumps(d['peak_density_after_smoothing'], ensure_ascii=False)}`。", ""]
    lines += ["## Decision", "", f"`{summary['decision']}`", ""]
    for name in ("SAMMLV", "CASME3"):
        fixed = summary["datasets"][name]["comparisons"][METHODS[1]]["observed_delta_F1"]
        strong = summary["datasets"][name]["comparisons"][METHODS[0]]["observed_delta_F1"]
        lines.append(f"- {name}：Adaptive − Best Fixed={fixed:+.6f}；Adaptive − Final Strong={strong:+.6f}。")
    lines += ["", "选择规则在两个数据集均使用同一单调结构和同一 quantile grid；数值阈值只由各 outer-train videos 的 NR1 得到。Stage B 未启动。", "", "## Outputs", ""]
    for path in [OUT / "ROUGHNESS_CALIBRATED_SMOOTHING_CN.md", *sorted(OUTPUTS.glob("*")), HERE / "run_roughness_gate.py"]:
        lines.append(f"- `{path.resolve()}`")
    return "\n".join(lines) + "\n"


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    input_paths = [NATIVE_SOURCE, SAMM_SELECTED, CAS_SELECTED, CAS_OUTER, SAMM_CACHE, CAS_CACHE]
    input_hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    try:
        datasets, provenance = gate()
    except Exception as exc:
        summary = {"status": "BLOCKED-ROUGHNESS-ANCHOR", "reason": repr(exc), "input_sha256": input_hashes}
        dump(OUTPUTS / "roughness_summary.json", summary)
        (OUT / "ROUGHNESS_CALIBRATED_SMOOTHING_CN.md").write_text(f"# Roughness-Calibrated Smoothing\n\n`BLOCKED-ROUGHNESS-ANCHOR`\n\n{exc!r}\n", encoding="utf-8")
        raise
    results = {name: run_dataset(name, data) for name, data in datasets.items()}
    boots = {name: bootstrap(name, result) for name, result in results.items()}
    result_decision = decision(results, boots)
    summary = {
        "status": "COMPLETE", "decision": result_decision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_anchor_gate": provenance, "input_sha256": input_hashes,
        "selector": {"primary_statistic": "NR1", "candidate_c_s": CS_VALUES,
                     "q1_grid": Q1_VALUES, "q2_grid": Q2_VALUES, "q3_grid": Q3_VALUES,
                     "rule_count": len(RULES), "mapping": "higher NR1 never selects weaker smoothing"},
        "datasets": {}, "bootstrap": boots,
        "decision_policy": {"effective_tie_tolerance": TIE_TOLERANCE,
                            "clearly_degraded": "observed delta < -0.001 and bootstrap CI upper < 0"},
        "integrity": {"cache_only": True, "only_smoothing_selection_changed": True,
                      "threshold_unchanged": True, "peak_distance_unchanged": True,
                      "boundary_unchanged": True, "test_GT_excluded_from_selection": True,
                      "no_quantile_threshold": True, "no_backbone_forward": True,
                      "no_recognition_or_morphology": True, "no_stage_B": True},
    }
    for name, result in results.items():
        summary["datasets"][name] = {
            "aggregate_metrics": result["aggregate"], "comparisons": result["comparisons"],
            "selected_rule_frequency": result["selected_rule_frequency"],
            "selected_fixed_c_s_frequency": result["selected_fixed_c_s_frequency"],
            "adaptive_assignment_frequency": result["adaptive_assignment_frequency"],
            "NR1_by_assigned_c_s": result["NR1_by_assigned_c_s"],
            "peak_density_after_smoothing": result["peak_density_after_smoothing"],
            "invariants": result["invariants"],
        }
    write_csv(OUTPUTS / "roughness_outer_metrics_sammlv.csv", results["SAMMLV"]["outer_rows"])
    write_csv(OUTPUTS / "roughness_outer_metrics_casme3.csv", results["CASME3"]["outer_rows"])
    write_csv(OUTPUTS / "roughness_selected_rules.csv", results["SAMMLV"]["rule_rows"] + results["CASME3"]["rule_rows"])
    write_csv(OUTPUTS / "roughness_video_assignments.csv", results["SAMMLV"]["video_rows"] + results["CASME3"]["video_rows"])
    write_csv(OUTPUTS / "roughness_fixed_cs_control.csv", results["SAMMLV"]["fixed_rows"] + results["CASME3"]["fixed_rows"])
    dump(OUTPUTS / "roughness_bootstrap.json", boots)
    dump(OUTPUTS / "roughness_summary.json", summary)
    (OUT / "ROUGHNESS_CALIBRATED_SMOOTHING_CN.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps({"decision": result_decision,
                      "SAMMLV": summary["datasets"]["SAMMLV"]["aggregate_metrics"],
                      "CASME3": summary["datasets"]["CASME3"]["aggregate_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
