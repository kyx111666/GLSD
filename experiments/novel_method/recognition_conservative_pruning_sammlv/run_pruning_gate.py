#!/usr/bin/env python3
"""SAMMLV-only nested recognition-assisted conservative pruning gate."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/recognition_conservative_pruning_sammlv"
OUTPUTS = OUT / "outputs"
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
CONFIG_SOURCE = ROOT / "results/rgr1_sammlv_nested/report.json"

spec = importlib.util.spec_from_file_location("native_only", NATIVE_SOURCE)
native = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(native)

SEED = 20260905
REPS = 1000
EPS = 1e-8
DELTAS = (0.05, 0.10, 0.15, 0.20)
GAMMAS = (0.20, 0.35, 0.50)
METHODS = ("C0_Final_Strong_Native", "C1_Margin_Only", "C2_Recognition_Only", "C3_Proposed", "C4_Unconstrained_F2")
EXPECTED_SHA = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
EXPECTED = {"subjects": 29, "videos": 79, "gt": 159, "k_p": 5}
ANCHOR = {"TP": 49, "FP": 143, "FN": 110, "F1": 0.2792022792022792}
GRAYZONE_F2_REFERENCE = {"ROC_AUC": 0.640645, "PR_AUC": 0.389504}


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


def empty_counts():
    return {key: 0 for key in ("TP", "FP", "FN", "event_count")}


def add_counts(target, source):
    for key in target:
        target[key] += int(source[key])


def metrics(counts):
    tp, fp, fn = (int(counts[key]) for key in ("TP", "FP", "FN"))
    return {"TP": tp, "FP": fp, "FN": fn, "event_count": int(counts["event_count"]),
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn) if tp + fn else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def config_id(config):
    return tuple(float(config[key]) for key in ("c_s", "p", "c_d", "c_b"))


def causal_label(base, changed):
    delta = {key: int(changed[key]) - int(base[key]) for key in ("TP", "FP", "FN")}
    if delta == {"TP": -1, "FP": 0, "FN": 1}:
        return 1, "KEEP"
    if delta == {"TP": 0, "FP": -1, "FN": 0}:
        return 0, "PRUNE"
    return None, "AMBIGUOUS_MATCH_INTERACTION"


def load_gate():
    if sha256(CACHE) != EXPECTED_SHA:
        raise RuntimeError("cache SHA mismatch")
    payload, records, subjects, observed = native.load_payload(CACHE)
    if observed != EXPECTED:
        raise RuntimeError(f"cache metadata mismatch: {observed}")
    frozen = json.loads(CONFIG_SOURCE.read_text(encoding="utf-8"))
    configs, references = {}, {}
    for fold in frozen["outer_folds"]:
        subject = str(fold["subject"])
        cfg = fold["selected_strong_config"]
        configs[subject] = {"c_s": float(cfg["c_s"]), "p": float(cfg["p_s"]),
                            "c_d": float(cfg["c_d"]), "c_b": float(cfg["c_b"])}
        references[subject] = {key: int(fold["outer_metrics"]["Tuned Native"][key])
                               for key in ("TP", "FP", "FN", "event_count")}
    if set(configs) != set(subjects):
        raise RuntimeError("fold config subjects mismatch")
    by_subject = {subject: [] for subject in subjects}
    for index, record in enumerate(records):
        score, logits = np.asarray(record["score"]), np.asarray(record["logits"])
        if score.ndim != 1 or logits.shape != (len(score), 5) or not np.isfinite(score).all() or not np.isfinite(logits).all():
            raise RuntimeError(f"invalid frozen arrays: record {index}")
        probabilities = softmax(logits.astype(float), axis=1)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError(f"recognition softmax axis validation failed: record {index}")
        by_subject[str(record["subject"])].append(index)
    total = empty_counts()
    for subject in subjects:
        fold_counts = empty_counts()
        for index in by_subject[subject]:
            add_counts(fold_counts, native.evaluate(records[index], native.tuned_decode(records[index], configs[subject], EXPECTED["k_p"])))
        if fold_counts != references[subject]:
            raise RuntimeError(f"fold anchor mismatch: {subject}")
        add_counts(total, fold_counts)
    aggregate = metrics(total)
    if any(aggregate[key] != ANCHOR[key] for key in ("TP", "FP", "FN")) or abs(aggregate["F1"] - ANCHOR["F1"]) > 1e-14:
        raise RuntimeError(f"aggregate anchor mismatch: {aggregate}")
    return records, list(subjects), by_subject, configs, references, aggregate


def build_bank(records, config, kp):
    width = max(1, int(round(config["c_s"] * kp)))
    distance = max(1, int(round(config["c_d"] * kp)))
    boundary = max(1, int(round(config["c_b"] * kp)))
    bank = {}
    for index, record in enumerate(records):
        curve = native.moving_average(record["score"], width)
        mean, maximum = float(curve.mean()), float(curve.max())
        tau = native.threshold(curve, config["p"])
        peaks = find_peaks(curve, height=tau, distance=distance)[0].astype(int)
        events = [native.event(peak, boundary, "tuned_native") for peak in peaks]
        base = native.evaluate(record, events)
        probabilities = softmax(np.asarray(record["logits"], dtype=float), axis=1)
        r2 = np.max(probabilities[:, :4], axis=1)  # verified class order: neutral id=4
        candidates = []
        for candidate_index, event in enumerate(events):
            changed = native.evaluate(record, events[:candidate_index] + events[candidate_index + 1:])
            label, label_name = causal_label(base, changed)
            peak = int(event["peak"])
            left, right = max(0, peak - kp), min(len(r2), peak + kp + 1)
            candidates.append({"candidate_index": candidate_index, "event": event, "peak": peak,
                               "peak_score": float(curve[peak]), "native_threshold": float(tau),
                               "H": float((curve[peak] - mean) / (maximum - mean + EPS) - config["p"]),
                               "R": float(np.mean(r2[left:right])), "label": label, "label_name": label_name})
        bank[index] = {"candidates": candidates, "base": base}
    return bank


def feature_matrix(items, feature):
    if feature == "R":
        return np.asarray([[item["candidate"]["R"]] for item in items], dtype=float).reshape(len(items), 1)
    return np.asarray([[item["candidate"]["H"], item["candidate"]["R"]] for item in items], dtype=float).reshape(len(items), 2)


def fit_probabilities(train, target, feature):
    train = [item for item in train if item["candidate"]["label"] is not None]
    if not train:
        raise RuntimeError("empty causal-label training bank")
    labels = np.asarray([item["candidate"]["label"] for item in train], dtype=int)
    classes = np.unique(labels)
    if len(classes) == 1:
        return np.full(len(target), float(classes[0])), f"single_class_constant_{int(classes[0])}"
    scaler = StandardScaler().fit(feature_matrix(train, feature))
    model = LogisticRegression(solver="liblinear", class_weight="balanced", C=1.0,
                               max_iter=1000, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(feature_matrix(train, feature)), labels)
    if int(max(model.n_iter_)) >= 1000:
        raise RuntimeError("LR did not converge")
    probabilities = model.predict_proba(scaler.transform(feature_matrix(target, feature)))[:, 1] if target else np.empty(0)
    return probabilities, "fitted"


def candidate_items(bank, records, indices, include_ambiguous=True):
    items = []
    for index in indices:
        for candidate in bank[index]["candidates"]:
            if include_ambiguous or candidate["label"] is not None:
                items.append({"record_index": index, "candidate": candidate,
                              "subject": str(records[index]["subject"])})
    return items


def probability_map(items, probabilities):
    return {(item["record_index"], item["candidate"]["candidate_index"]): float(probability)
            for item, probability in zip(items, probabilities)}


def method_grid(method):
    if method == "C1_Margin_Only":
        return [(delta, None) for delta in DELTAS]
    if method in ("C2_Recognition_Only", "C3_Proposed"):
        return list(itertools.product(DELTAS, GAMMAS))
    if method == "C4_Unconstrained_F2":
        return [(None, gamma) for gamma in GAMMAS]
    return [(None, None)]


def apply_method(bank, records, indices, method, delta=None, gamma=None, probabilities=None, traces=False, outer_fold=None):
    counts, diagnostic, trace_rows = empty_counts(), Counter(), []
    for index in indices:
        kept = []
        for candidate in bank[index]["candidates"]:
            key = (index, candidate["candidate_index"])
            p_keep = probabilities.get(key) if probabilities is not None else None
            eligible = (candidate["H"] <= delta) if delta is not None else method == "C4_Unconstrained_F2"
            if method == "C0_Final_Strong_Native":
                eligible, pruned = False, False
            elif method == "C1_Margin_Only":
                pruned = eligible
            else:
                pruned = eligible and p_keep < gamma
            diagnostic["total_native_events"] += 1
            diagnostic["eligible_gray_zone_events"] += int(eligible)
            diagnostic["pruned_events"] += int(pruned)
            if pruned:
                diagnostic[f"pruned_{candidate['label_name']}"] += 1
            else:
                kept.append(candidate["event"])
            if traces:
                event = candidate["event"]
                trace_rows.append({"subject": str(records[index]["subject"]), "video": str(records[index]["video"]),
                    "onset": event["onset"], "apex": event["peak"], "offset": event["offset"], "peak": event["peak"],
                    "peak_score": candidate["peak_score"], "native_threshold": candidate["native_threshold"],
                    "H_s": candidate["H"], "R": candidate["R"], "P_keep": p_keep,
                    "eligible_gray_zone": eligible, "pruned": pruned, "causal_native_label": candidate["label_name"],
                    "outer_fold": outer_fold, "selected_delta": delta, "selected_gamma": gamma, "method": method})
        add_counts(counts, native.evaluate(records[index], kept))
    return metrics(counts), diagnostic, trace_rows


def selection_key(result, native_result, delta, gamma):
    pruning_rate = result["diagnostic"]["pruned_events"] / result["diagnostic"]["total_native_events"] if result["diagnostic"]["total_native_events"] else 0.0
    return (result["metrics"]["F1"], -result["metrics"]["FP"],
            -(native_result["TP"] - result["metrics"]["TP"]), -pruning_rate,
            -(delta if delta is not None else 0.0), -(gamma if gamma is not None else 0.0))


def nested_fold(held, records, subjects, by_subject, config, kp):
    bank = build_bank(records, config, kp)
    outer_train = [subject for subject in subjects if subject != held]
    inner_maps = {"R": {}, "HR": {}}
    handling = Counter()
    for validation in outer_train:
        train_indices = [index for subject in outer_train if subject != validation for index in by_subject[subject]]
        validation_indices = by_subject[validation]
        train_items = candidate_items(bank, records, train_indices)
        validation_items = candidate_items(bank, records, validation_indices)
        for feature in ("R", "HR"):
            probabilities, status = fit_probabilities(train_items, validation_items, feature)
            handling[f"inner_{feature}_{status}"] += 1
            inner_maps[feature].update(probability_map(validation_items, probabilities))
    inner_indices = [index for subject in outer_train for index in by_subject[subject]]
    inner_native, _, _ = apply_method(bank, records, inner_indices, "C0_Final_Strong_Native")
    selections = {}
    for method in METHODS[1:]:
        pmap = None if method == "C1_Margin_Only" else inner_maps["R" if method == "C2_Recognition_Only" else "HR"]
        scored = []
        for delta, gamma in method_grid(method):
            result, diagnostic, _ = apply_method(bank, records, inner_indices, method, delta, gamma, pmap)
            scored.append({"delta": delta, "gamma": gamma, "metrics": result, "diagnostic": diagnostic})
        selections[method] = max(scored, key=lambda row: selection_key(row, inner_native, row["delta"], row["gamma"]))
    train_indices = inner_indices
    test_indices = by_subject[held]
    train_items = candidate_items(bank, records, train_indices)
    test_items = candidate_items(bank, records, test_indices)
    outer_maps = {}
    for feature in ("R", "HR"):
        probabilities, status = fit_probabilities(train_items, test_items, feature)
        handling[f"outer_{feature}_{status}"] += 1
        outer_maps[feature] = probability_map(test_items, probabilities)
    fold_results, traces, selected_rows = {}, [], []
    for method in METHODS:
        selected = selections.get(method, {"delta": None, "gamma": None, "metrics": inner_native,
                                           "diagnostic": Counter()})
        pmap = None if method in ("C0_Final_Strong_Native", "C1_Margin_Only") else outer_maps["R" if method == "C2_Recognition_Only" else "HR"]
        result, diagnostic, rows = apply_method(bank, records, test_indices, method, selected["delta"], selected["gamma"], pmap, True, held)
        fold_results[method] = {"metrics": result, "diagnostic": dict(diagnostic)}
        traces.extend(rows)
        selected_rows.append({"outer_subject": held, "method": method, "selected_delta": selected["delta"],
                              "selected_gamma": selected["gamma"],
                              **{f"inner_{key}": value for key, value in selected["metrics"].items()},
                              "inner_pruned_events": selected["diagnostic"].get("pruned_events", 0),
                              "inner_total_native_events": selected["diagnostic"].get("total_native_events", 0)})
    return fold_results, traces, selected_rows, handling


def aggregate_results(subject_rows):
    aggregate = {}
    for method in METHODS:
        counts = empty_counts()
        for row in subject_rows:
            if row["method"] == method:
                add_counts(counts, row)
        aggregate[method] = metrics(counts)
    return aggregate


def bootstrap(subject_rows, subjects):
    lookup = {(row["subject"], row["method"]): row for row in subject_rows}
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(subjects), size=(REPS, len(subjects)))
    output = {"seed": SEED, "subjects": subjects, "sampled_subject_indices": indices.tolist(), "comparisons": {}}
    for method in ("C1_Margin_Only", "C2_Recognition_Only", "C3_Proposed"):
        values = []
        for sample in indices:
            counts = {key: empty_counts() for key in ("base", "method")}
            for position in sample:
                subject = subjects[int(position)]
                add_counts(counts["base"], lookup[(subject, "C0_Final_Strong_Native")])
                add_counts(counts["method"], lookup[(subject, method)])
            values.append(metrics(counts["method"])["F1"] - metrics(counts["base"])["F1"])
        array = np.asarray(values)
        output["comparisons"][f"{method}_minus_C0"] = {"mean": float(array.mean()),
            "ci95": [float(x) for x in np.quantile(array, [.025, .975])], "replicates": values}
    return output


def render_report(summary):
    fmt = lambda value: "NA" if value is None else f"{value:.6f}"
    lines = ["# Recognition-Assisted Conservative Pruning — SAMMLV Development Gate", "",
             f"最终状态：`{summary['decision']}`。", "",
             "本实验仅使用 SAMMLV verified frozen cache；没有读取或评估 CASME3 event-level 结果。", "",
             "## Provenance / anchor gate", "",
             f"- 29 subjects / 79 videos / 159 GT / k_p=5。",
             f"- Final Strong Native PASS：49/143/110，F1={summary['anchor']['F1']:.6f}。",
             f"- Cache：`{summary['provenance']['cache_path']}`，SHA-256 `{summary['provenance']['cache_sha256']}`。",
             f"- Fold configs：`{summary['provenance']['fold_config_source']}`。",
             f"- Recognition：frozen cache logits，softmax axis=1，class order=5 classes，neutral id=4，R2-mean over clipped p±k_p。", "",
             "## Aggregate results", "",
             "| Method | TP/FP/FN | Precision | Recall | F1 | ΔTP | ΔFP | ΔFN | ΔF1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row = summary["aggregate"][method]
        lines.append(f"| {method} | {row['TP']}/{row['FP']}/{row['FN']} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['F1']:.6f} | {row['delta_TP']:+d} | {row['delta_FP']:+d} | {row['delta_FN']:+d} | {row['delta_F1']:+.6f} |")
    lines += ["", "## Pruning diagnostics", "",
              "| Method | Eligible | Pruned | Rate | Pruned TP | Pruned FP | Prune precision | FP removed / TP lost |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS[1:]:
        row = summary["pruning_diagnostics"][method]
        utility = "INF" if row["FP_removed_per_TP_lost"] == "INF" else fmt(row["FP_removed_per_TP_lost"])
        lines.append(f"| {method} | {row['eligible_gray_zone_events']} | {row['pruned_events']} | {row['pruning_rate']:.6f} | {row['pruned_true_positives']} | {row['pruned_false_positives']} | {row['prune_precision']:.6f} | {utility} |")
    sensitivity = summary["C3_subject_stability"]
    boot = summary["bootstrap"]["comparisons"]["C3_Proposed_minus_C0"]
    lines += ["", "## C3 stability", "",
              f"- Subject improved/equal/worse：{sensitivity['improved']}/{sensitivity['equal']}/{sensitivity['worse']}。",
              f"- Leave-one-subject aggregate ΔF1 min/median/max：{sensitivity['leave_one_subject']['min']:+.6f} / {sensitivity['leave_one_subject']['median']:+.6f} / {sensitivity['leave_one_subject']['max']:+.6f}；min<0={sensitivity['leave_one_subject']['min_below_zero']}。",
              f"- Shared-subject bootstrap：observed={sensitivity['observed_delta_F1']:+.6f}，mean={boot['mean']:+.6f}，95% CI=[{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}]。", "",
              "## Secondary bootstrap controls", ""]
    for comparison, result in summary["bootstrap"]["comparisons"].items():
        lines.append(f"- {comparison}：mean={result['mean']:+.6f}，95% CI=[{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}]。")
    lines += ["",
              "## Parameter stability", "",
              f"- {summary['parameter_stability']['methods']}",
              f"- Single-class handling counts：{summary['parameter_stability']['single_class_handling']}。",
              f"- Boundary flag：{summary['parameter_stability']['CONSERVATIVE-PRUNING-BOUNDARY-SELECTION']}（不扩 grid）。", "",
              "## Decision", "", f"`{summary['decision']}`", "",
              f"- Strong criteria：{summary['decision_checks']['strong']}。",
              f"- C3 与 C1 F1 差：{summary['decision_checks']['C3_minus_C1_F1']:+.6f}。",
              "- 该结论仅是 SAMMLV DEVELOPMENT RESULT；未运行 CASME3 locked validation。", "",
              "## Outputs", ""]
    for path in [OUT / "RECOGNITION_CONSERVATIVE_PRUNING_SAMMLV_CN.md", *sorted(OUTPUTS.glob("*")), HERE / "run_pruning_gate.py"]:
        lines.append(f"- `{path.resolve()}`")
    return "\n".join(lines) + "\n"


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    input_paths = (CACHE, CONFIG_SOURCE, NATIVE_SOURCE, NATIVE_AUDIT, PAPER_METRICS)
    hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    try:
        records, subjects, by_subject, configs, references, anchor = load_gate()
    except Exception as exc:
        blocked = {"status": "BLOCKED-CONSERVATIVE-PRUNING-ANCHOR", "reason": repr(exc), "input_sha256": hashes}
        dump(OUTPUTS / "pruning_summary.json", blocked)
        (OUT / "RECOGNITION_CONSERVATIVE_PRUNING_SAMMLV_CN.md").write_text(f"# Recognition-Assisted Conservative Pruning\n\n`BLOCKED-CONSERVATIVE-PRUNING-ANCHOR`\n\n{exc!r}\n", encoding="utf-8")
        raise
    all_subject_rows, selected_rows, traces, handling = [], [], [], Counter()
    for held in subjects:
        fold_results, fold_traces, fold_selected, fold_handling = nested_fold(held, records, subjects, by_subject, configs[held], EXPECTED["k_p"])
        handling.update(fold_handling)
        traces.extend(fold_traces)
        selected_rows.extend(fold_selected)
        for method in METHODS:
            result = fold_results[method]
            all_subject_rows.append({"subject": held, "outer_subject": held, "method": method,
                                     **result["metrics"], **{f"diagnostic_{key}": value for key, value in result["diagnostic"].items()}})
    aggregate = aggregate_results(all_subject_rows)
    base = aggregate["C0_Final_Strong_Native"]
    if any(base[key] != anchor[key] for key in ("TP", "FP", "FN")) or abs(base["F1"] - anchor["F1"]) > 1e-14:
        raise RuntimeError("post-nested C0 anchor drift")
    for method, row in aggregate.items():
        row.update({"delta_TP": row["TP"] - base["TP"], "delta_FP": row["FP"] - base["FP"],
                    "delta_FN": row["FN"] - base["FN"], "delta_F1": row["F1"] - base["F1"]})
    diagnostics = {}
    for method in METHODS[1:]:
        rows = [row for row in all_subject_rows if row["method"] == method]
        total = sum(row.get("diagnostic_total_native_events", 0) for row in rows)
        eligible = sum(row.get("diagnostic_eligible_gray_zone_events", 0) for row in rows)
        pruned = sum(row.get("diagnostic_pruned_events", 0) for row in rows)
        pruned_tp = sum(row.get("diagnostic_pruned_KEEP", 0) for row in rows)
        pruned_fp = sum(row.get("diagnostic_pruned_PRUNE", 0) for row in rows)
        tp_lost, fp_removed = base["TP"] - aggregate[method]["TP"], base["FP"] - aggregate[method]["FP"]
        diagnostics[method] = {"total_native_events": total, "eligible_gray_zone_events": eligible,
            "pruned_events": pruned, "pruning_rate": pruned / total if total else 0.0,
            "pruned_true_positives": pruned_tp, "pruned_false_positives": pruned_fp,
            "ambiguous_pruned": pruned - pruned_tp - pruned_fp,
            "prune_precision": pruned_fp / pruned if pruned else 0.0,
            "FP_removed": fp_removed, "TP_lost": tp_lost,
            "FP_removed_per_TP_lost": "INF" if tp_lost == 0 else fp_removed / tp_lost}
    lookup = {(row["subject"], row["method"]): row for row in all_subject_rows}
    relation = Counter()
    loo = []
    for subject in subjects:
        c0, c3 = lookup[(subject, METHODS[0])], lookup[(subject, "C3_Proposed")]
        delta = c3["F1"] - c0["F1"]
        relation["improved" if delta > 1e-15 else "worse" if delta < -1e-15 else "equal"] += 1
        c0_counts, c3_counts = empty_counts(), empty_counts()
        for other in subjects:
            if other != subject:
                add_counts(c0_counts, lookup[(other, METHODS[0])])
                add_counts(c3_counts, lookup[(other, "C3_Proposed")])
        loo.append(metrics(c3_counts)["F1"] - metrics(c0_counts)["F1"])
    bootstrap_result = bootstrap(all_subject_rows, subjects)
    c3 = aggregate["C3_Proposed"]
    strong = {"F1_gt_C0": c3["F1"] > base["F1"], "FP_lt_143": c3["FP"] < 143,
              "TP_ge_47": c3["TP"] >= 47, "LOSO_min_gt_0": min(loo) > 0,
              "bootstrap_mean_gt_0": bootstrap_result["comparisons"]["C3_Proposed_minus_C0"]["mean"] > 0,
              "F1_gt_C1": c3["F1"] > aggregate["C1_Margin_Only"]["F1"]}
    if all(strong.values()):
        decision = "RECOGNITION-PRUNING-STRONG-GO"
    elif c3["F1"] > base["F1"] and c3["FP"] <= base["FP"] and c3["TP"] >= 47:
        decision = "RECOGNITION-PRUNING-WEAK-GO"
    else:
        decision = "RECOGNITION-PRUNING-NO-GO"
    stability = {"methods": {}, "single_class_handling": dict(handling)}
    boundary = False
    for method in METHODS[1:]:
        rows = [row for row in selected_rows if row["method"] == method]
        delta_frequency = dict(Counter(str(row["selected_delta"]) for row in rows if row["selected_delta"] is not None))
        gamma_frequency = dict(Counter(str(row["selected_gamma"]) for row in rows if row["selected_gamma"] is not None))
        flags = {"delta_0.20_over_half": delta_frequency.get("0.2", 0) > len(subjects) / 2,
                 "gamma_0.50_over_half": gamma_frequency.get("0.5", 0) > len(subjects) / 2}
        boundary = boundary or any(flags.values())
        stability["methods"][method] = {"selected_delta_frequency": delta_frequency,
            "selected_gamma_frequency": gamma_frequency, "endpoint_flags": flags}
    stability["CONSERVATIVE-PRUNING-BOUNDARY-SELECTION"] = boundary
    c3_stability = {**relation, "observed_delta_F1": c3["F1"] - base["F1"],
                    "leave_one_subject": {"values": loo, "min": float(min(loo)), "median": float(np.median(loo)),
                                          "max": float(max(loo)), "min_below_zero": bool(min(loo) < 0)}}
    provenance = {"cache_path": str(CACHE.resolve()), "cache_sha256": EXPECTED_SHA,
                  "evaluator_path": str(NATIVE_AUDIT.resolve()), "evaluator_sha256": sha256(NATIVE_AUDIT),
                  "GT_source": "verified cache records[*].samples", "fold_config_source": str(CONFIG_SOURCE.resolve()),
                  "recognition_output_source": "verified cache records[*].logits", "softmax_axis": 1,
                  "class_count": 5, "neutral_class_id": 4, "anchor_exact": True}
    summary = {"status": "COMPLETE", "decision": decision, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "provenance": provenance, "input_sha256": hashes, "anchor": anchor,
               "grayzone_F2_protocol_reference": GRAYZONE_F2_REFERENCE, "aggregate": aggregate, "pruning_diagnostics": diagnostics,
               "C3_subject_stability": c3_stability, "bootstrap": bootstrap_result,
               "parameter_stability": stability, "decision_checks": {"strong": strong,
                   "C3_minus_C1_F1": c3["F1"] - aggregate["C1_Margin_Only"]["F1"]},
               "protocol": {"features": {"C2": ["R"], "C3_C4": ["H_s", "R"]}, "delta_grid": DELTAS,
                            "gamma_grid": GAMMAS, "classifier": "StandardScaler + LogisticRegression(liblinear, balanced, C=1, max_iter=1000, seed=20260905)",
                            "tie_break": ["higher F1", "fewer FP", "fewer TP lost", "smaller pruning rate", "smaller delta", "smaller gamma"]},
               "integrity": {"SAMMLV_only": True, "CASME3_event_level_read": False, "deletion_only": True,
                             "new_predictions_added": 0, "backbone_forward": False, "morphology_used": False,
                             "outer_test_GT_used_for_selection": False}}
    outer_rows = list(all_subject_rows) + [{"subject": "aggregate", "outer_subject": "aggregate", "method": method, **row}
                                            for method, row in aggregate.items()]
    comparison_rows = []
    for subject in subjects:
        c0, c3 = lookup[(subject, METHODS[0])], lookup[(subject, "C3_Proposed")]
        delta = c3["F1"] - c0["F1"]
        comparison_rows.append({"subject": subject, **{f"C0_{key}": c0[key] for key in ("TP", "FP", "FN", "Precision", "Recall", "F1")},
                                **{f"C3_{key}": c3[key] for key in ("TP", "FP", "FN", "Precision", "Recall", "F1")},
                                "delta_F1": delta,
                                "C3_vs_C0": "improved" if delta > 1e-15 else "worse" if delta < -1e-15 else "equal"})
    write_csv(OUTPUTS / "pruning_outer_metrics.csv", outer_rows)
    write_csv(OUTPUTS / "pruning_subject_metrics.csv", comparison_rows)
    write_csv(OUTPUTS / "pruning_selected_configs.csv", selected_rows)
    write_csv(OUTPUTS / "pruning_event_trace.csv", traces)
    dump(OUTPUTS / "pruning_bootstrap.json", bootstrap_result)
    dump(OUTPUTS / "pruning_parameter_stability.json", stability)
    write_csv(OUTPUTS / "pruning_ablation_controls.csv", [{"method": method, **aggregate[method], **diagnostics.get(method, {})} for method in METHODS])
    dump(OUTPUTS / "pruning_summary.json", summary)
    (OUT / "RECOGNITION_CONSERVATIVE_PRUNING_SAMMLV_CN.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"decision": decision, "aggregate": aggregate, "C3_stability": c3_stability,
                      "parameter_boundary": boundary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
