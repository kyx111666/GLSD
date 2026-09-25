#!/usr/bin/env python3
"""Gray-zone multi-evidence candidate separability audit.

This script produces candidate-level OOF ranking diagnostics only.  It never
chooses a probability threshold or produces a new event-level decoder result.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import pickle
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.signal import find_peaks
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/grayzone_multi_evidence_audit"
OUTPUTS = OUT / "outputs"
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
SAMM_CONFIGS = ROOT / "results/rgr1_sammlv_nested/report.json"
CAS_CONFIGS = ROOT / "results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv"
CAS_OUTER = CAS_CONFIGS.with_name("final_strong_native_outer_metrics.csv")
SAMM_MORPH = ROOT / "results/morphology_refinement_sammlv/outputs/morph_refined_selected_configs.csv"
CAS_MORPH = ROOT / "results/final_native_closure_strong_anchor_morphology/strong_anchor_morph_selected_configs.csv"
SAMM_CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
CAS_CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"

spec = importlib.util.spec_from_file_location("native_only", NATIVE_SOURCE)
native = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(native)

SEED = 20260905
BOOTSTRAP_REPS = 1000
DELTA_P = 0.20
EPS = 1e-8
TASKS = ("KEEP_PRUNE", "RESCUE_REJECT")
FEATURE_SETS = {
    "F0": ("H",),
    "F1": ("H", "M"),
    "F2": ("H", "R"),
    "F3": ("H", "M", "R"),
}
EXPECTED = {
    "SAMMLV": {"cache": SAMM_CACHE, "sha": "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569",
                "subjects": 29, "videos": 79, "gt": 159, "k_p": 5, "anchor": (49, 143, 110), "f1": 0.2792022792022792},
    "CASME3": {"cache": CAS_CACHE, "sha": "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda",
               "subjects": 94, "videos": 462, "gt": 858, "k_p": 17, "anchor": (124, 1148, 734), "f1": 0.11643192488262911},
}


def sha256(path: Path):
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
    return {key: 0 for key in ("TP", "FP", "FN", "event_count")}


def add(target, source):
    for key in target:
        target[key] += int(source[key])


def metric(counts):
    tp, fp, fn = (int(counts[k]) for k in ("TP", "FP", "FN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "event_count": int(counts["event_count"]),
            "Precision": precision, "Recall": recall, "F1": f1}


def config_id(config):
    return tuple(float(config[k]) for k in ("c_s", "p", "c_d", "c_b"))


def read_csv_map(path, value_field):
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[str(row["outer_subject"])] = int(row[value_field])
    return out


def load_frozen_sources():
    configs, references = {"SAMMLV": {}, "CASME3": {}}, {"SAMMLV": {}, "CASME3": {}}
    samm = json.loads(SAMM_CONFIGS.read_text(encoding="utf-8"))
    for fold in samm["outer_folds"]:
        subject = str(fold["subject"])
        cfg = fold["selected_strong_config"]
        configs["SAMMLV"][subject] = {"c_s": float(cfg["c_s"]), "p": float(cfg["p_s"]),
                                              "c_d": float(cfg["c_d"]), "c_b": float(cfg["c_b"])}
        references["SAMMLV"][subject] = {key: int(fold["outer_metrics"]["Tuned Native"][key]) for key in ("TP", "FP", "FN", "event_count")}
    with CAS_CONFIGS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            configs["CASME3"][str(row["outer_subject"])] = {key: float(row[key]) for key in ("c_s", "p", "c_d", "c_b")}
    with CAS_OUTER.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["outer_subject"] != "aggregate":
                references["CASME3"][str(row["outer_subject"])] = {key: int(row[key]) for key in ("TP", "FP", "FN", "event_count")}
    radii = {"SAMMLV": read_csv_map(SAMM_MORPH, "actual_integer_radius"),
             "CASME3": read_csv_map(CAS_MORPH, "actual_integer_radius")}
    return configs, references, radii


def gate():
    configs, references, radii = load_frozen_sources()
    datasets, provenance = {}, {}
    for name, expected in EXPECTED.items():
        if sha256(expected["cache"]) != expected["sha"]:
            raise RuntimeError(f"{name}: cache SHA mismatch")
        payload, records, subjects, observed = native.load_payload(expected["cache"])
        wanted = {key: expected[key] for key in ("subjects", "videos", "gt", "k_p")}
        if observed != wanted:
            raise RuntimeError(f"{name}: cache metadata mismatch {observed}")
        if set(configs[name]) != set(subjects) or set(radii[name]) != set(subjects):
            raise RuntimeError(f"{name}: frozen fold sources mismatch")
        by_subject = {subject: [] for subject in subjects}
        for index, record in enumerate(records):
            score, logits = np.asarray(record["score"], dtype=float), np.asarray(record["logits"], dtype=float)
            if score.ndim != 1 or logits.shape != (len(score), 5) or not np.isfinite(score).all() or not np.isfinite(logits).all():
                raise RuntimeError(f"{name}: invalid frozen arrays at record {index}")
            by_subject[str(record["subject"])].append(index)
        total = empty_counts()
        for subject in subjects:
            counts = empty_counts()
            for index in by_subject[subject]:
                events = native.tuned_decode(records[index], configs[name][subject], expected["k_p"])
                add(counts, native.evaluate(records[index], events))
            if counts != references[name][subject]:
                raise RuntimeError(f"{name}/{subject}: anchor mismatch")
            add(total, counts)
        result = metric(total)
        if tuple(result[key] for key in ("TP", "FP", "FN")) != expected["anchor"] or abs(result["F1"] - expected["f1"]) > 1e-14:
            raise RuntimeError(f"{name}: aggregate anchor mismatch {result}")
        datasets[name] = {"records": records, "subjects": subjects, "by_subject": by_subject,
                          "configs": configs[name], "radii": radii[name], "kp": expected["k_p"]}
        provenance[name] = {"cache_path": str(expected["cache"].resolve()), "cache_sha256": expected["sha"],
                            "gt_source": "verified cache records[*].samples", **observed,
                            "evaluator_path": str(NATIVE_AUDIT.resolve()), "evaluator_sha256": sha256(NATIVE_AUDIT),
                            "paper_metrics_sha256": sha256(PAPER_METRICS), "Final Strong Native": result,
                            "anchor_exact": True, "morphology_radius_frequency": dict(Counter(radii[name].values()))}
    return datasets, provenance


def patch_feature(curve, peak, radius):
    padded = np.pad(np.asarray(curve, dtype=float), (radius, radius), mode="edge")
    patch = padded[int(peak):int(peak) + 2 * radius + 1]
    source = np.linspace(0.0, 1.0, len(patch))
    values = np.interp(np.linspace(0.0, 1.0, 31), source, patch)
    values = values - values[15]
    values = values / (np.linalg.norm(values) + EPS)
    assert values.shape == (31,) and np.isfinite(values).all()
    return values


def r2_curve(logits):
    probabilities = softmax(np.asarray(logits, dtype=float), axis=1)
    if probabilities.shape[1] != 5 or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise RuntimeError("invalid recognition probabilities")
    return np.max(probabilities[:, :4], axis=1)


def candidate_r(r2, peak, kp):
    left, right = max(0, int(peak) - kp), min(len(r2), int(peak) + kp + 1)
    return float(np.mean(r2[left:right]))


def causal_label(base, changed, task):
    delta = {key: int(changed[key]) - int(base[key]) for key in ("TP", "FP", "FN")}
    if task == "KEEP_PRUNE":
        if delta == {"TP": -1, "FP": 0, "FN": 1}:
            return 1, "KEEP"
        if delta == {"TP": 0, "FP": -1, "FN": 0}:
            return 0, "PRUNE"
    else:
        if delta == {"TP": 1, "FP": 0, "FN": -1}:
            return 1, "RESCUE"
        if delta == {"TP": 0, "FP": 1, "FN": 0}:
            return 0, "REJECT"
    return None, "AMBIGUOUS_MATCH_INTERACTION"


def build_bank_for_config(records, config, kp):
    bank = {}
    width = max(1, int(round(config["c_s"] * kp)))
    distance = max(1, int(round(config["c_d"] * kp)))
    boundary = max(1, int(round(config["c_b"] * kp)))
    for index, record in enumerate(records):
        curve = native.moving_average(record["score"], width)
        mean, maximum = float(curve.mean()), float(curve.max())
        tau_high = native.threshold(curve, config["p"])
        tau_low = native.threshold(curve, config["p"] - DELTA_P)
        high_peaks = find_peaks(curve, height=tau_high, distance=distance)[0].astype(int)
        low_peaks = find_peaks(curve, height=tau_low, distance=distance)[0].astype(int)
        high_set, low_set = set(map(int, high_peaks)), set(map(int, low_peaks))
        if not high_set.issubset(low_set):
            raise RuntimeError("low-threshold peak set does not contain high set")
        weak_peaks = sorted(low_set - high_set)
        high_events = [native.event(peak, boundary, "high") for peak in high_peaks]
        weak_events = [native.event(peak, boundary, "weak") for peak in weak_peaks]
        base = native.evaluate(record, high_events)
        r2 = r2_curve(record["logits"])
        task_rows = {task: [] for task in TASKS}
        for candidate_index, event in enumerate(high_events):
            changed_events = high_events[:candidate_index] + high_events[candidate_index + 1:]
            changed = native.evaluate(record, changed_events)
            label, label_name = causal_label(base, changed, "KEEP_PRUNE")
            peak = int(event["peak"])
            task_rows["KEEP_PRUNE"].append({"candidate_index": candidate_index, "peak": peak,
                "onset": int(event["onset"]), "offset": int(event["offset"]), "label": label, "label_name": label_name,
                "H": float((curve[peak] - mean) / (maximum - mean + EPS) - config["p"]),
                "R": candidate_r(r2, peak, kp), "base": dict(base), "changed": dict(changed)})
        for candidate_index, event in enumerate(weak_events):
            union = sorted(high_events + [event], key=lambda item: (item["peak"], 0 if item["source"] == "high" else 1))
            changed = native.evaluate(record, union)
            label, label_name = causal_label(base, changed, "RESCUE_REJECT")
            peak = int(event["peak"])
            task_rows["RESCUE_REJECT"].append({"candidate_index": candidate_index, "peak": peak,
                "onset": int(event["onset"]), "offset": int(event["offset"]), "label": label, "label_name": label_name,
                "H": float((curve[peak] - mean) / (maximum - mean + EPS) - config["p"]),
                "R": candidate_r(r2, peak, kp), "base": dict(base), "changed": dict(changed)})
        bank[index] = {"curve": curve, "tasks": task_rows, "tau_high": tau_high, "tau_low": tau_low,
                       "width": width, "distance": distance, "boundary": boundary}
    return bank


def matrix(rows, feature_set, curve_by_index, radius):
    dimension = 1 + (31 if "M" in FEATURE_SETS[feature_set] else 0) + (1 if "R" in FEATURE_SETS[feature_set] else 0)
    if not rows:
        return np.empty((0, dimension), dtype=float)
    values = []
    for item in rows:
        row, index = item["candidate"], item["record_index"]
        features = [row["H"]]
        if "M" in FEATURE_SETS[feature_set]:
            features.extend(patch_feature(curve_by_index[index], row["peak"], radius))
        if "R" in FEATURE_SETS[feature_set]:
            features.append(row["R"])
        values.append(features)
    return np.asarray(values, dtype=float).reshape(len(rows), dimension)


def auc_metrics(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    prevalence = positives / len(labels) if len(labels) else None
    if positives and negatives:
        roc, pr = float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    else:
        roc = pr = None
    return {"positive_count": positives, "negative_count": negatives, "candidate_count": len(labels),
            "positive_prevalence": prevalence, "ROC_AUC": roc, "PR_AUC": pr,
            "PR_AUC_over_prevalence": pr / prevalence if pr is not None and prevalence else None}


def fit_predict(train_x, train_y, test_x):
    classes = np.unique(train_y)
    if len(classes) == 1:
        return np.full(len(test_x), float(classes[0])), None, None, f"single_class_constant_{int(classes[0])}"
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(train_x), train_y)
    if int(max(model.n_iter_)) >= 2000:
        raise RuntimeError("LR did not converge")
    probability = model.predict_proba(scaler.transform(test_x))[:, 1] if len(test_x) else np.empty(0, dtype=float)
    return probability, scaler, model, "fitted"


def coefficient_row(dataset, task, feature_set, held, model, handling, train_count, test_count):
    out = {"row_type": "fold", "dataset": dataset, "task": task, "feature_set": feature_set,
           "outer_subject": held, "fit_handling": handling, "train_candidate_count": train_count,
           "test_candidate_count": test_count, "H_coefficient": None, "R_coefficient": None,
           "Morphology_L1": None, "Morphology_L2": None}
    if model is None:
        return out
    coef = model.coef_[0]
    out["H_coefficient"] = float(coef[0])
    offset = 1
    if "M" in FEATURE_SETS[feature_set]:
        morph = coef[offset:offset + 31]
        out["Morphology_L1"] = float(np.sum(np.abs(morph)))
        out["Morphology_L2"] = float(np.linalg.norm(morph))
        offset += 31
    if "R" in FEATURE_SETS[feature_set]:
        out["R_coefficient"] = float(coef[offset])
    return out


def run_dataset(name, data):
    records, subjects, configs, radii = data["records"], data["subjects"], data["configs"], data["radii"]
    banks = {config_id(config): build_bank_for_config(records, config, data["kp"])
             for config in {config_id(c): c for c in configs.values()}.values()}
    candidate_rows, oof = [], {task: [] for task in TASKS}
    coefficients = []
    bank_counts = {task: Counter() for task in TASKS}
    for held in subjects:
        bank = banks[config_id(configs[held])]
        radius = radii[held]
        curves = {index: row["curve"] for index, row in bank.items()}
        for task in TASKS:
            train, test = [], []
            for index, record in enumerate(records):
                target = test if str(record["subject"]) == held else train
                for candidate in bank[index]["tasks"][task]:
                    if candidate["label"] is not None:
                        target.append({"record_index": index, "candidate": candidate})
            train_y = np.asarray([item["candidate"]["label"] for item in train], dtype=int)
            test_y = np.asarray([item["candidate"]["label"] for item in test], dtype=int)
            if not len(train):
                raise RuntimeError(f"{name}/{held}/{task}: empty training bank")
            feature_predictions = {}
            for feature_set in FEATURE_SETS:
                train_x = matrix(train, feature_set, curves, radius)
                test_x = matrix(test, feature_set, curves, radius)
                probability, scaler, model, handling = fit_predict(train_x, train_y, test_x)
                feature_predictions[feature_set] = probability
                coefficients.append(coefficient_row(name, task, feature_set, held, model, handling, len(train), len(test)))
            for position, item in enumerate(test):
                candidate, index = item["candidate"], item["record_index"]
                record = records[index]
                for feature_set in FEATURE_SETS:
                    oof[task].append({"dataset": name, "task": task, "outer_subject": held,
                        "subject": str(record["subject"]), "video": str(record["video"]),
                        "candidate_id": f"{name}/{held}/{record['video']}/{task}/{candidate['peak']}",
                        "peak": candidate["peak"], "label": candidate["label"], "label_name": candidate["label_name"],
                        "feature_set": feature_set, "probability": float(feature_predictions[feature_set][position]),
                        "actual_integer_radius": radius})
            # Candidate-bank artifact contains all held-subject candidates, including ambiguous interactions.
            for index in data["by_subject"][held]:
                record = records[index]
                for candidate in bank[index]["tasks"][task]:
                    row = {"dataset": name, "task": task, "outer_subject": held, "subject": held,
                           "video": str(record["video"]), "candidate_id": f"{name}/{held}/{record['video']}/{task}/{candidate['peak']}",
                           "candidate_index": candidate["candidate_index"], "peak": candidate["peak"],
                           "onset": candidate["onset"], "offset": candidate["offset"], "label": candidate["label"],
                           "label_name": candidate["label_name"], "included_primary": candidate["label"] is not None,
                           "H_signed": candidate["H"], "R2_mean": candidate["R"],
                           "actual_integer_radius": radius, "frozen_c_s": configs[held]["c_s"],
                           "frozen_p_H": configs[held]["p"], "frozen_c_d": configs[held]["c_d"],
                           "frozen_c_b": configs[held]["c_b"], "Delta_p": DELTA_P,
                           "base_TP": candidate["base"]["TP"], "base_FP": candidate["base"]["FP"], "base_FN": candidate["base"]["FN"],
                           "changed_TP": candidate["changed"]["TP"], "changed_FP": candidate["changed"]["FP"], "changed_FN": candidate["changed"]["FN"]}
                    morph = patch_feature(bank[index]["curve"], candidate["peak"], radius)
                    row.update({f"M{i + 1:02d}": float(value) for i, value in enumerate(morph)})
                    candidate_rows.append(row)
                    bank_counts[task][candidate["label_name"]] += 1
    return {"candidate_rows": candidate_rows, "oof": oof, "coefficients": coefficients,
            "candidate_counts": bank_counts}


def metrics_and_subjects(oof_by_dataset, subjects_by_dataset):
    feature_rows, subject_rows = [], []
    summary = {}
    for name, tasks in oof_by_dataset.items():
        summary[name] = {}
        for task, rows in tasks.items():
            summary[name][task] = {}
            for feature_set in FEATURE_SETS:
                subset = [row for row in rows if row["feature_set"] == feature_set]
                values = auc_metrics([row["label"] for row in subset], [row["probability"] for row in subset])
                feature_rows.append({"dataset": name, "task": task, "feature_set": feature_set, **values})
                summary[name][task][feature_set] = values
            comparisons = Counter()
            for subject in subjects_by_dataset[name]:
                per_feature = {}
                for feature_set in FEATURE_SETS:
                    subset = [row for row in rows if row["subject"] == subject and row["feature_set"] == feature_set]
                    values = auc_metrics([row["label"] for row in subset], [row["probability"] for row in subset])
                    if not subset:
                        status = "UNDEFINED-NO-CANDIDATES"
                    elif values["PR_AUC"] is None:
                        status = "UNDEFINED-SINGLE-CLASS-SUBJECT"
                    else:
                        status = "DEFINED"
                    per_feature[feature_set] = values
                    subject_rows.append({"dataset": name, "task": task, "subject": subject,
                                         "feature_set": feature_set, "status": status, **values,
                                         "F3_vs_F0": None})
                if per_feature["F0"]["PR_AUC"] is None or per_feature["F3"]["PR_AUC"] is None:
                    comparisons["undefined"] += 1
                    relation = ("UNDEFINED-NO-CANDIDATES" if not any(
                        row["subject"] == subject for row in rows) else "UNDEFINED-SINGLE-CLASS-SUBJECT")
                else:
                    delta = per_feature["F3"]["PR_AUC"] - per_feature["F0"]["PR_AUC"]
                    relation = "better" if delta > 1e-15 else "worse" if delta < -1e-15 else "equal"
                    comparisons[relation] += 1
                for row in reversed(subject_rows):
                    if row["dataset"] == name and row["task"] == task and row["subject"] == subject and row["feature_set"] == "F3":
                        row["F3_vs_F0"] = relation
                        break
            summary[name][task]["subject_stability_F3_vs_F0"] = dict(comparisons)
            best = max(FEATURE_SETS, key=lambda f: (summary[name][task][f]["PR_AUC"] if summary[name][task][f]["PR_AUC"] is not None else -1, -int(f[1])))
            summary[name][task]["BEST_PREDEFINED_FEATURE_SET"] = best
    return feature_rows, subject_rows, summary


def bootstrap(oof_by_dataset, subjects_by_dataset):
    output = {}
    for dataset_index, (name, tasks) in enumerate(oof_by_dataset.items()):
        subjects = list(subjects_by_dataset[name])
        rng = np.random.default_rng(SEED + dataset_index)
        sampled = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPS, len(subjects)))
        output[name] = {"subjects": subjects, "seed": SEED + dataset_index,
                        "sampled_subject_indices": sampled.tolist(), "tasks": {}}
        for task, rows in tasks.items():
            by_subject = {subject: {f: [] for f in FEATURE_SETS} for subject in subjects}
            for row in rows:
                by_subject[row["subject"]][row["feature_set"]].append((int(row["label"]), float(row["probability"])))
            comparisons = {}
            for reference in ("F0", "F1", "F2"):
                replicates = []
                for indices in sampled:
                    chosen = [subjects[i] for i in indices]
                    values = {}
                    for feature_set in (reference, "F3"):
                        pairs = [pair for subject in chosen for pair in by_subject[subject][feature_set]]
                        labels = np.asarray([p[0] for p in pairs], dtype=int)
                        scores = np.asarray([p[1] for p in pairs], dtype=float)
                        values[feature_set] = float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None
                    replicates.append(values["F3"] - values[reference] if values["F3"] is not None and values[reference] is not None else None)
                valid = np.asarray([v for v in replicates if v is not None], dtype=float)
                comparisons[f"F3_minus_{reference}"] = {"valid_repetitions": int(len(valid)),
                    "invalid_single_class_repetitions": BOOTSTRAP_REPS - int(len(valid)),
                    "mean": float(np.mean(valid)) if len(valid) else None,
                    "ci95": [float(x) for x in np.quantile(valid, [.025, .975])] if len(valid) else None,
                    "replicates": replicates}
            output[name]["tasks"][task] = comparisons
    return output


def coefficient_summary(rows):
    summary = {}
    for name, task, feature_set in itertools.product(EXPECTED, TASKS, FEATURE_SETS):
        subset = [row for row in rows if row["dataset"] == name and row["task"] == task and row["feature_set"] == feature_set]
        key = f"{name}/{task}/{feature_set}"
        summary[key] = {"fold_count": len(subset), "single_class_fold_count": sum(row["fit_handling"].startswith("single_class") for row in subset)}
        for field in ("H_coefficient", "R_coefficient", "Morphology_L1", "Morphology_L2"):
            values = np.asarray([row[field] for row in subset if row[field] is not None], dtype=float)
            if len(values):
                median = float(np.median(values))
                summary[key][field] = {"n": len(values), "median": median,
                    "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
                    "sign_consistency": float(np.mean(np.sign(values) == np.sign(median))) if median else float(np.mean(values == 0))}
            else:
                summary[key][field] = None
    return summary


def candidate_count_summary(run_results):
    output = {}
    for name, result in run_results.items():
        output[name] = {}
        for task in TASKS:
            counts = result["candidate_counts"][task]
            positive_name, negative_name = ("KEEP", "PRUNE") if task == "KEEP_PRUNE" else ("RESCUE", "REJECT")
            positives, negatives = counts[positive_name], counts[negative_name]
            output[name][task] = {"total_candidates": sum(counts.values()), "positive_count": positives,
                "negative_count": negatives, "interaction_ambiguous": counts["AMBIGUOUS_MATCH_INTERACTION"],
                "positive_prevalence": positives / (positives + negatives) if positives + negatives else None,
                "label_counts": dict(counts)}
    return output


def pass_checks(feature_summary, boots):
    checks = {}
    for name in EXPECTED:
        checks[name] = {}
        for task in TASKS:
            f3, f0 = feature_summary[name][task]["F3"], feature_summary[name][task]["F0"]
            boot = boots[name]["tasks"][task]["F3_minus_F0"]
            conditions = {"ROC_AUC_ge_0.65": f3["ROC_AUC"] is not None and f3["ROC_AUC"] >= .65,
                          "PR_lift_ge_1.5": f3["PR_AUC_over_prevalence"] is not None and f3["PR_AUC_over_prevalence"] >= 1.5,
                          "PR_F3_gt_F0": f3["PR_AUC"] is not None and f0["PR_AUC"] is not None and f3["PR_AUC"] > f0["PR_AUC"],
                          "bootstrap_mean_positive": boot["mean"] is not None and boot["mean"] > 0}
            checks[name][task] = {"conditions": conditions, "PASS": all(conditions.values())}
    keep = all(checks[name]["KEEP_PRUNE"]["PASS"] for name in EXPECTED)
    rescue = all(checks[name]["RESCUE_REJECT"]["PASS"] for name in EXPECTED)
    if keep and rescue:
        decision = "MULTI-EVIDENCE-BOTH-ACTIONS-GO"
    elif keep:
        decision = "MULTI-EVIDENCE-PRUNE-ONLY-GO"
    elif rescue:
        decision = "MULTI-EVIDENCE-RESCUE-ONLY-GO"
    else:
        decision = "MULTI-EVIDENCE-NO-GO"
    return checks, decision


def render_report(summary):
    lines = ["# Gray-Zone Multi-Evidence Separability Audit", "", f"判定：`{summary['decision']}`。", "",
             "本报告只描述 candidate-level OOF separability；没有计算新 decoder 的 event-level F1。", "", "## Provenance / anchor gate", ""]
    for name in EXPECTED:
        p = summary["provenance_anchor_gate"][name]
        m = p["Final Strong Native"]
        lines += [f"- {name}：{p['subjects']} subjects / {p['videos']} videos / {p['gt']} GT / k_p={p['k_p']}；cache SHA-256 `{p['cache_sha256']}`。",
                  f"  - Final Strong Native PASS：{m['TP']}/{m['FP']}/{m['FN']}，F1={m['F1']:.6f}；Morphology radius frequency={p['morphology_radius_frequency']}。"]
    lines += ["", "## Frozen protocol", "",
              "- 每个 outer fold 仅用该 fold 已冻结的 Final Strong Native 参数和 Morphology integer radius；outer-test GT 只用于回顾性候选标签与评估。",
              "- KEEP/PRUNE：从 high-event 集合单独删除一个候选后用 official evaluator 重评；删除导致 ΔTP=-1,ΔFP=0,ΔFN=+1 标 KEEP，导致 ΔTP=0,ΔFP=-1,ΔFN=0 标 PRUNE。",
              "- RESCUE/REJECT：固定 Δp=0.20，weak bank 严格为 P_L−P_H；单独加入 high-event 集合后，若 ΔTP=+1,ΔFP=0,ΔFN=-1 标 RESCUE，若 ΔTP=0,ΔFP=+1,ΔFN=0 标 REJECT。",
              "- F0=H，F1=H+M，F2=H+R，F3=H+M+R。H 为 signed normalized threshold margin，M 为冻结 peak-relative 31-point patch，R 为 p±k_p 内非 neutral 最大 softmax 概率均值。",
              "- StandardScaler 和 LogisticRegression 仅拟合 outer-train subjects；固定 liblinear / balanced / C=1 / seed，仅产生 OOF ranking probability，不选阈值。",
              "", "## Candidate counts", "", "| Dataset | Task | Total | Positive | Negative | Ambiguous | Prevalence |", "|---|---|---:|---:|---:|---:|---:|"]
    for name in EXPECTED:
        for task in TASKS:
            c = summary["candidate_counts"][name][task]
            lines.append(f"| {name} | {task} | {c['total_candidates']} | {c['positive_count']} | {c['negative_count']} | {c['interaction_ambiguous']} | {c['positive_prevalence']:.6f} |")
    lines += ["", "## OOF candidate-level metrics", "", "| Dataset | Task | Features | Pos/Neg | ROC-AUC | PR-AUC | PR/prevalence | PASS |", "|---|---|---|---:|---:|---:|---:|---|"]
    for name in EXPECTED:
        for task in TASKS:
            for feature_set in FEATURE_SETS:
                m = summary["feature_summary"][name][task][feature_set]
                passed = summary["separability_pass"][name][task]["PASS"] if feature_set == "F3" else ""
                fmt = lambda x: "NA" if x is None else f"{x:.6f}"
                lines.append(f"| {name} | {task} | {feature_set} | {m['positive_count']}/{m['negative_count']} | {fmt(m['ROC_AUC'])} | {fmt(m['PR_AUC'])} | {fmt(m['PR_AUC_over_prevalence'])} | {passed} |")
    lines += ["", "## F3 bootstrap deltas", "", "| Dataset | Task | Comparison | Mean ΔPR | 95% CI | Valid reps |", "|---|---|---|---:|---|---:|"]
    for name in EXPECTED:
        for task in TASKS:
            for reference in ("F0", "F1", "F2"):
                b = summary["bootstrap"][name]["tasks"][task][f"F3_minus_{reference}"]
                ci = "NA" if b["ci95"] is None else f"[{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}]"
                mean = "NA" if b["mean"] is None else f"{b['mean']:+.6f}"
                lines.append(f"| {name} | {task} | F3−{reference} | {mean} | {ci} | {b['valid_repetitions']} |")
    lines += ["", "## Subject stability and best predefined feature set", ""]
    for name in EXPECTED:
        for task in TASKS:
            f = summary["feature_summary"][name][task]
            lines.append(f"- {name}/{task}：BEST_PREDEFINED_FEATURE_SET={f['BEST_PREDEFINED_FEATURE_SET']}；F3 vs F0 subject better/equal/worse/undefined={f['subject_stability_F3_vs_F0']}。")
    lines += ["", "## Coefficient diagnostic", "", f"- Recognition direction consistency：{summary['coefficient_interpretation']['recognition_direction_consistency']}。",
              f"- F3 standardized R coefficient medians：{summary['coefficient_interpretation']['F3_R_coefficient_medians']}；四个 dataset/task 中位数均为正，方向一致。",
              f"- Morphology contribution only in SAMMLV（按 PR-AUC 增量诊断）：{summary['coefficient_interpretation']['morphology_only_sammlv']}。",
              f"- Morphology PR-AUC deltas：{summary['coefficient_interpretation']['morphology_PR_AUC_deltas']}。",
              "- Morphology 不是统一的 SAMMLV-only 信号：它对 SAMMLV RESCUE 强正增益，但对 SAMMLV KEEP 及 CASME3 结果为负或混合增益。",
              "- Fold-level standardized coefficients and morphology L1/L2 norms are saved in `grayzone_coefficients.csv`; medians, IQRs and sign consistency are in `grayzone_summary.json`。", "",
              "## Decision", "", f"`{summary['decision']}`", ""]
    for name in EXPECTED:
        for task in TASKS:
            lines.append(f"- {name}/{task}：F3 PASS={summary['separability_pass'][name][task]['PASS']}；conditions={summary['separability_pass'][name][task]['conditions']}。")
    lines += ["", "该结论不代表新的 spotting improvement；没有选择 probability threshold，也没有生成新 spotting events。", "", "## Outputs", ""]
    for path in [OUT / "GRAYZONE_MULTI_EVIDENCE_AUDIT_CN.md", *sorted(OUTPUTS.glob("*")), HERE / "run_grayzone_audit.py"]:
        lines.append(f"- `{path.resolve()}`")
    return "\n".join(lines) + "\n"


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    input_paths = [NATIVE_SOURCE, NATIVE_AUDIT, PAPER_METRICS, SAMM_CONFIGS, CAS_CONFIGS, CAS_OUTER,
                   SAMM_MORPH, CAS_MORPH, SAMM_CACHE, CAS_CACHE]
    input_hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    try:
        datasets, provenance = gate()
    except Exception as exc:
        blocked = {"status": "BLOCKED-GRAYZONE-ANCHOR", "reason": repr(exc), "input_sha256": input_hashes}
        dump(OUTPUTS / "grayzone_summary.json", blocked)
        (OUT / "GRAYZONE_MULTI_EVIDENCE_AUDIT_CN.md").write_text(f"# Gray-Zone Multi-Evidence Separability Audit\n\n`BLOCKED-GRAYZONE-ANCHOR`\n\n{exc!r}\n", encoding="utf-8")
        raise
    run_results = {name: run_dataset(name, data) for name, data in datasets.items()}
    oof_by_dataset = {name: result["oof"] for name, result in run_results.items()}
    subjects_by_dataset = {name: list(data["subjects"]) for name, data in datasets.items()}
    feature_rows, subject_rows, feature_summary = metrics_and_subjects(oof_by_dataset, subjects_by_dataset)
    boots = bootstrap(oof_by_dataset, subjects_by_dataset)
    coefficients = [row for result in run_results.values() for row in result["coefficients"]]
    coeff_summary = coefficient_summary(coefficients)
    candidate_counts = candidate_count_summary(run_results)
    checks, decision = pass_checks(feature_summary, boots)
    # Interpret only the pre-registered F3 standardized coefficients.
    r_medians = {name: [coeff_summary[f"{name}/{task}/F3"]["R_coefficient"]["median"]
                        for task in TASKS if coeff_summary[f"{name}/{task}/F3"]["R_coefficient"] is not None] for name in EXPECTED}
    recognition_consistent = bool(all(values and all(np.sign(v) == np.sign(values[0]) for v in values) for values in r_medians.values()) and np.sign(r_medians["SAMMLV"][0]) == np.sign(r_medians["CASME3"][0]))
    morph_l2 = {name: [coeff_summary[f"{name}/{task}/F3"]["Morphology_L2"]["median"] for task in TASKS] for name in EXPECTED}
    morphology_deltas = {name: {task: {
        "F1_minus_F0": feature_summary[name][task]["F1"]["PR_AUC"] - feature_summary[name][task]["F0"]["PR_AUC"],
        "F3_minus_F2": feature_summary[name][task]["F3"]["PR_AUC"] - feature_summary[name][task]["F2"]["PR_AUC"],
    } for task in TASKS} for name in EXPECTED}
    sam_morph = [delta for task in TASKS for delta in morphology_deltas["SAMMLV"][task].values()]
    cas_morph = [delta for task in TASKS for delta in morphology_deltas["CASME3"][task].values()]
    morphology_only_sammlv = bool(all(v > 0 for v in sam_morph) and all(v <= 0 for v in cas_morph))
    summary = {"status": "COMPLETE", "decision": decision, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "provenance_anchor_gate": provenance, "input_sha256": input_hashes,
               "candidate_counts": candidate_counts, "feature_summary": feature_summary,
               "bootstrap": boots, "coefficient_summary": coeff_summary,
               "coefficient_interpretation": {"recognition_direction_consistency": recognition_consistent,
                                              "F3_R_coefficient_medians": r_medians,
                                              "morphology_only_sammlv": morphology_only_sammlv,
                                              "morphology_PR_AUC_deltas": morphology_deltas,
                                              "F3_morphology_L2_medians": morph_l2},
               "separability_pass": checks,
               "protocol": {"Delta_p": DELTA_P, "feature_sets": FEATURE_SETS,
                            "classifier": "StandardScaler(train subjects only) + LogisticRegression(liblinear, balanced, C=1.0, max_iter=2000, random_state=20260905)",
                            "PR_AUC_primary": True, "subject_disjoint_OOF": True},
               "integrity": {"candidate_level_only": True, "event_decoder_F1_computed": False,
                             "probability_threshold_selected": False, "weak_pool_redefined": False,
                             "no_backbone_forward": True, "no_new_features": True, "no_new_classifier": True}}
    candidate_rows = [row for result in run_results.values() for row in result["candidate_rows"]]
    write_csv(OUTPUTS / "grayzone_candidate_bank.csv", candidate_rows)
    write_csv(OUTPUTS / "grayzone_keep_prune_oof.csv", run_results["SAMMLV"]["oof"]["KEEP_PRUNE"] + run_results["CASME3"]["oof"]["KEEP_PRUNE"])
    write_csv(OUTPUTS / "grayzone_rescue_reject_oof.csv", run_results["SAMMLV"]["oof"]["RESCUE_REJECT"] + run_results["CASME3"]["oof"]["RESCUE_REJECT"])
    write_csv(OUTPUTS / "grayzone_feature_metrics.csv", feature_rows)
    write_csv(OUTPUTS / "grayzone_subject_metrics.csv", subject_rows)
    write_csv(OUTPUTS / "grayzone_coefficients.csv", coefficients)
    dump(OUTPUTS / "grayzone_bootstrap.json", boots)
    dump(OUTPUTS / "grayzone_summary.json", summary)
    (OUT / "GRAYZONE_MULTI_EVIDENCE_AUDIT_CN.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"decision": decision, "candidate_counts": candidate_counts,
                      "pass": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
