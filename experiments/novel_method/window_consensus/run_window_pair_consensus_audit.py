#!/usr/bin/env python3
"""Diagnostic-only SAMMLV raw window-pair consensus separability audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_SOURCE = HERE / "run_overlap_aggregation_feasibility.py"
PRUNING_SOURCE = ROOT / "my_method/recognition_conservative_pruning_sammlv/run_pruning_gate.py"
DEFAULT_RAW = Path(
    os.environ.get("SAMMLV_RAW_ROOT", "data/raw_prestitch_sammlv_full")
)
DEFAULT_OUT = ROOT / "results/window_pair_consensus_separability_sammlv"
EXPECTED_ANCHOR = (49, 143, 110)
EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79
K_P = 5
EPS = 1e-8
SEED = 20260906
BOOTSTRAP_REPEATS = 1000
FEATURES = ("C1", "C2", "C3", "C4", "C5")
FEATURE_SETS = {
    "W0": ("C1",),
    "W1": ("C1", "C2", "C3"),
    "W2": ("C1", "C2", "C3", "C4", "C5"),
}
# Semantic direction for interpretation only; raw AUC/AP are always retained.
ORIENTATION = {"C1": -1.0, "C2": 1.0, "C3": 1.0, "C4": -1.0, "C5": 1.0}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_module("window_pair_base", BASE_SOURCE)
native = base.native
pruning = import_module("window_pair_causal_labels", PRUNING_SOURCE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def nearest_raw_peak(raw_score: np.ndarray, global_index: np.ndarray, peak: int):
    local_peaks = find_peaks(np.asarray(raw_score, dtype=np.float64))[0].astype(int)
    candidates = [
        (abs(int(global_index[local]) - peak), int(global_index[local]), int(local))
        for local in local_peaks
        if abs(int(global_index[local]) - peak) <= K_P
    ]
    if not candidates:
        return None
    _distance, global_peak, local_peak = min(candidates)
    return {"global_peak": global_peak, "local_peak": local_peak}


def pair_features(raw: np.ndarray, global_indices: np.ndarray, peak: int) -> dict:
    occurrences = np.argwhere(global_indices == int(peak))
    rows = sorted({int(row) for row, _local in occurrences})
    if len(rows) == 1:
        return {"coverage_count": 1, "coverage_status": "SINGLE-VIEW-CANDIDATE", **{key: None for key in FEATURES}}
    if len(rows) != 2:
        return {"coverage_count": len(rows), "coverage_status": "NON-PAIR-COVERAGE", **{key: None for key in FEATURES}}

    row_a, row_b = rows
    local_a = int(np.flatnonzero(global_indices[row_a] == peak)[0])
    local_b = int(np.flatnonzero(global_indices[row_b] == peak)[0])
    score_a = float(raw[row_a, local_a])
    score_b = float(raw[row_b, local_b])
    c1 = abs(score_a - score_b)
    c2 = min(score_a, score_b) / (max(score_a, score_b) + EPS)

    peak_a = nearest_raw_peak(raw[row_a], global_indices[row_a], peak)
    peak_b = nearest_raw_peak(raw[row_b], global_indices[row_b], peak)
    support_count = int(peak_a is not None) + int(peak_b is not None)
    c3 = support_count / 2.0
    c4 = None
    if peak_a is not None and peak_b is not None:
        c4 = abs(peak_a["global_peak"] - peak_b["global_peak"]) / K_P

    map_a = {int(t): float(s) for t, s in zip(global_indices[row_a], raw[row_a])}
    map_b = {int(t): float(s) for t, s in zip(global_indices[row_b], raw[row_b])}
    common = sorted(set(map_a) & set(map_b) & set(range(peak - K_P, peak + K_P + 1)))
    c5 = None
    if len(common) >= 5:
        patch_a = np.asarray([map_a[t] for t in common], dtype=np.float64)
        patch_b = np.asarray([map_b[t] for t in common], dtype=np.float64)
        if np.std(patch_a) > 0 and np.std(patch_b) > 0:
            c5 = float(np.corrcoef(patch_a, patch_b)[0, 1])
    return {
        "coverage_count": 2,
        "coverage_status": "PAIR",
        "view_A_row": row_a,
        "view_B_row": row_b,
        "view_A_local_peak_index": local_a,
        "view_B_local_peak_index": local_b,
        "view_A_score_at_candidate": score_a,
        "view_B_score_at_candidate": score_b,
        "view_A_nearest_peak_global": None if peak_a is None else peak_a["global_peak"],
        "view_B_nearest_peak_global": None if peak_b is None else peak_b["global_peak"],
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": finite_or_none(c5),
        "C5_common_frame_count": len(common),
    }


def build_candidate_bank(gated: dict):
    candidates = []
    subject_counts = {subject: native.counts_empty() for subject in gated["subjects"]}
    for (subject, video), item in sorted(gated["loaded"].items()):
        record = dict(item["record"])
        record["score"] = item["original"]
        config = gated["configs"][subject]
        events = native.tuned_decode(record, config, K_P)
        base_counts = native.evaluate(record, events)
        native.add_counts(subject_counts[subject], base_counts)
        for candidate_index, event in enumerate(events):
            changed_events = events[:candidate_index] + events[candidate_index + 1 :]
            changed_counts = native.evaluate(record, changed_events)
            label, label_name = pruning.causal_label(base_counts, changed_counts)
            features = pair_features(item["raw"], item["global_indices"], int(event["peak"]))
            candidates.append(
                {
                    "subject": subject,
                    "video": video,
                    "candidate_index": candidate_index,
                    "peak": int(event["peak"]),
                    "onset": int(event["onset"]),
                    "offset": int(event["offset"]),
                    "label": label,
                    "label_name": label_name,
                    "included_primary": label is not None and features["coverage_status"] == "PAIR",
                    "base_TP": base_counts["TP"],
                    "base_FP": base_counts["FP"],
                    "base_FN": base_counts["FN"],
                    "changed_TP": changed_counts["TP"],
                    "changed_FP": changed_counts["FP"],
                    "changed_FN": changed_counts["FN"],
                    "frozen_c_s": config["c_s"],
                    "frozen_p": config["p"],
                    "frozen_c_d": config["c_d"],
                    "frozen_c_b": config["c_b"],
                    **features,
                }
            )
    aggregate = native.metrics(base.sum_counts(subject_counts.values()))
    anchor = tuple(aggregate[key] for key in ("TP", "FP", "FN"))
    if anchor != EXPECTED_ANCHOR:
        raise RuntimeError(f"window-pair candidate anchor={anchor}, expected={EXPECTED_ANCHOR}")
    return candidates, aggregate, subject_counts


def auc_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"ROC_AUC": None, "PR_AUC": None, "positive_prevalence": None, "PR_AUC_over_prevalence": None}
    prevalence = float(labels.mean())
    pr_auc = float(average_precision_score(labels, scores))
    return {
        "ROC_AUC": float(roc_auc_score(labels, scores)),
        "PR_AUC": pr_auc,
        "positive_prevalence": prevalence,
        "PR_AUC_over_prevalence": pr_auc / prevalence if prevalence else None,
    }


def scalar_diagnostics(primary: list[dict]):
    rows = []
    summary = {}
    for feature in FEATURES:
        usable = [row for row in primary if row[feature] is not None and np.isfinite(float(row[feature]))]
        labels = np.asarray([row["label"] for row in usable], dtype=int)
        scores = np.asarray([row[feature] for row in usable], dtype=float)
        raw_metrics = auc_metrics(labels, scores)
        oriented_scores = ORIENTATION[feature] * scores
        oriented_roc = (
            float(roc_auc_score(labels, oriented_scores)) if len(np.unique(labels)) == 2 else None
        )
        summary[feature] = {
            **raw_metrics,
            "oriented_ROC_AUC": oriented_roc,
            "orientation_for_interpretation": "lower_is_KEEP" if ORIENTATION[feature] < 0 else "higher_is_KEEP",
            "usable_count": len(usable),
            "missing_count": len(primary) - len(usable),
        }
        for label_value, label_name in ((1, "KEEP"), (0, "PRUNE")):
            values = scores[labels == label_value]
            q25, median, q75 = (np.quantile(values, (0.25, 0.5, 0.75)) if len(values) else (np.nan, np.nan, np.nan))
            rows.append(
                {
                    "feature": feature,
                    "group": label_name,
                    "count": len(values),
                    "median": finite_or_none(median),
                    "q25": finite_or_none(q25),
                    "q75": finite_or_none(q75),
                    "IQR": finite_or_none(q75 - q25),
                    **raw_metrics,
                    "oriented_ROC_AUC": oriented_roc,
                    "orientation_for_interpretation": summary[feature]["orientation_for_interpretation"],
                    "feature_missing_count": summary[feature]["missing_count"],
                }
            )
    return rows, summary


def feature_matrix(rows: list[dict], feature_set: str) -> np.ndarray:
    names = FEATURE_SETS[feature_set]
    return np.asarray([[row[name] if row[name] is not None else np.nan for name in names] for row in rows], dtype=float)


def median_impute_train_only(train: np.ndarray, test: np.ndarray):
    medians = np.empty(train.shape[1], dtype=float)
    for column in range(train.shape[1]):
        values = train[np.isfinite(train[:, column]), column]
        if len(values) == 0:
            raise RuntimeError(f"W2 train column {column} entirely missing")
        medians[column] = float(np.median(values))
    train_filled = np.where(np.isfinite(train), train, medians)
    test_filled = np.where(np.isfinite(test), test, medians)
    return train_filled, test_filled, medians


def loso_oof(primary: list[dict], subjects: list[str]):
    output = []
    fold_rows = []
    for held in subjects:
        train_rows = [row for row in primary if row["subject"] != held]
        test_rows = [row for row in primary if row["subject"] == held]
        if not test_rows:
            continue
        y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
        if len(np.unique(y_train)) != 2:
            raise RuntimeError(f"single-class LOSO training fold: {held}")
        for feature_set in FEATURE_SETS:
            train_x = feature_matrix(train_rows, feature_set)
            test_x = feature_matrix(test_rows, feature_set)
            medians = None
            if feature_set == "W2":
                train_x, test_x, medians = median_impute_train_only(train_x, test_x)
            elif not (np.isfinite(train_x).all() and np.isfinite(test_x).all()):
                raise RuntimeError(f"unexpected missing value in {feature_set}/{held}")
            scaler = StandardScaler().fit(train_x)
            model = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                C=1.0,
                max_iter=1000,
                random_state=SEED,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(scaler.transform(train_x), y_train)
            if int(np.max(model.n_iter_)) >= 1000:
                raise RuntimeError(f"LR did not converge: {feature_set}/{held}")
            probabilities = model.predict_proba(scaler.transform(test_x))[:, list(model.classes_).index(1)]
            for row, probability in zip(test_rows, probabilities):
                output.append(
                    {
                        "subject": row["subject"],
                        "video": row["video"],
                        "candidate_index": row["candidate_index"],
                        "peak": row["peak"],
                        "label": row["label"],
                        "label_name": row["label_name"],
                        "feature_set": feature_set,
                        "P_KEEP": float(probability),
                    }
                )
            fold_rows.append(
                {
                    "outer_subject": held,
                    "feature_set": feature_set,
                    "train_candidates": len(train_rows),
                    "test_candidates": len(test_rows),
                    "train_KEEP": int(y_train.sum()),
                    "train_PRUNE": int(len(y_train) - y_train.sum()),
                    "W2_train_medians_json": None if medians is None else json.dumps(medians.tolist()),
                    "n_iter": int(np.max(model.n_iter_)),
                }
            )
    expected = len(primary) * len(FEATURE_SETS)
    if len(output) != expected:
        raise RuntimeError(f"OOF prediction cardinality={len(output)}, expected={expected}")
    return output, fold_rows


def multivariate_metrics(oof: list[dict]):
    rows = []
    summary = {}
    for feature_set in FEATURE_SETS:
        subset = [row for row in oof if row["feature_set"] == feature_set]
        metrics = auc_metrics(
            np.asarray([row["label"] for row in subset], dtype=int),
            np.asarray([row["P_KEEP"] for row in subset], dtype=float),
        )
        summary[feature_set] = metrics
        rows.append({"feature_set": feature_set, "features": json.dumps(FEATURE_SETS[feature_set]), **metrics})
    return rows, summary


def subject_metrics(oof: list[dict], subjects: list[str]):
    rows = []
    comparisons = Counter()
    for subject in subjects:
        values = {}
        for feature_set in FEATURE_SETS:
            subset = [row for row in oof if row["subject"] == subject and row["feature_set"] == feature_set]
            metrics = auc_metrics(
                np.asarray([row["label"] for row in subset], dtype=int),
                np.asarray([row["P_KEEP"] for row in subset], dtype=float),
            )
            values[feature_set] = metrics
            rows.append({"subject": subject, "feature_set": feature_set, "candidate_count": len(subset), **metrics})
        w0, w2 = values["W0"]["PR_AUC"], values["W2"]["PR_AUC"]
        if w0 is None or w2 is None:
            relation = "undefined"
        elif w2 > w0 + 1e-15:
            relation = "improved"
        elif w2 < w0 - 1e-15:
            relation = "worse"
        else:
            relation = "equal"
        comparisons[relation] += 1
        for row in rows:
            if row["subject"] == subject:
                row["W2_vs_W0_subject_PR_AUC"] = relation if row["feature_set"] == "W2" else None
    return rows, {key: int(comparisons.get(key, 0)) for key in ("improved", "equal", "worse", "undefined")}


def bootstrap(oof: list[dict], subjects: list[str]):
    by_subject = {
        subject: {
            feature_set: [row for row in oof if row["subject"] == subject and row["feature_set"] == feature_set]
            for feature_set in FEATURE_SETS
        }
        for subject in subjects
    }
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPEATS, len(subjects)))
    deltas = []
    invalid = 0
    for indices in sampled:
        scores = {}
        for feature_set in FEATURE_SETS:
            batch = [row for index in indices for row in by_subject[subjects[int(index)]][feature_set]]
            metrics = auc_metrics(
                np.asarray([row["label"] for row in batch], dtype=int),
                np.asarray([row["P_KEEP"] for row in batch], dtype=float),
            )
            scores[feature_set] = metrics["PR_AUC"]
        if scores["W0"] is None or scores["W2"] is None:
            invalid += 1
        else:
            deltas.append(scores["W2"] - scores["W0"])
    values = np.asarray(deltas, dtype=float)
    return {
        "unit": "subject",
        "paired": True,
        "same_sampled_subject_indices_for_W0_W1_W2": True,
        "repetitions": BOOTSTRAP_REPEATS,
        "valid_repetitions": len(deltas),
        "invalid_single_class_repetitions": invalid,
        "seed": SEED,
        "sampled_subject_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
        "observed_delta_PR_W2_minus_W0": multivariate_metrics(oof)[1]["W2"]["PR_AUC"] - multivariate_metrics(oof)[1]["W0"]["PR_AUC"],
        "mean_delta_PR": float(values.mean()) if len(values) else None,
        "CI95": [float(x) for x in np.quantile(values, (0.025, 0.975))] if len(values) else None,
    }


def decide(multivariate: dict, scalar: dict, boot: dict):
    w0, w2 = multivariate["W0"], multivariate["W2"]
    individual = {
        feature: bool(scalar[feature]["oriented_ROC_AUC"] is not None and scalar[feature]["oriented_ROC_AUC"] >= 0.60)
        for feature in ("C3", "C4", "C5")
    }
    gates = {
        "W2_ROC_AUC_ge_0.70": w2["ROC_AUC"] >= 0.70,
        "W2_PR_ratio_ge_1.5": w2["PR_AUC_over_prevalence"] >= 1.5,
        "W2_PR_AUC_gt_W0": w2["PR_AUC"] > w0["PR_AUC"],
        "bootstrap_mean_delta_PR_gt_0": boot["mean_delta_PR"] > 0,
    }
    useful = any(individual.values())
    if all(gates.values()) and useful:
        decision = "WINDOW-PAIR-CONSENSUS-STRONG-GO"
    elif sum(gates.values()) == len(gates) - 1 and useful:
        decision = "WINDOW-PAIR-CONSENSUS-WEAK-GO"
    else:
        decision = "WINDOW-PAIR-CONSENSUS-NO-GO"
    return decision, gates, individual


def fmt(value) -> str:
    return "NA" if value is None else f"{value:.6f}"


def report_markdown(summary: dict) -> str:
    anchor = summary["anchor"]
    bank = summary["candidate_bank"]
    lines = [
        "# Window-Pair Consensus Separability Audit — SAMMLV",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "本实验只诊断 B0 Native OOF candidates 的 raw two-window relationship；没有创建、删除或重新解码事件，因此不报告新的 Spotting F1。",
        "",
        "## Provenance / anchor gate",
        "",
        f"- Raw manifest SHA-256：`{summary['provenance']['raw_manifest_sha256']}`。",
        f"- Causal-label implementation SHA-256：`{summary['provenance']['causal_label_source_sha256']}`。",
        f"- B0：{anchor['TP']}/{anchor['FP']}/{anchor['FN']}，F1={anchor['F1']:.6f}，PASS。",
        f"- Candidate bank：total={bank['total']}，KEEP={bank['KEEP']}，PRUNE={bank['PRUNE']}，ambiguous={bank['AMBIGUOUS_MATCH_INTERACTION']}。",
        f"- Coverage：pair primary={bank['primary_pair_labeled']}，single-view={bank['SINGLE-VIEW-CANDIDATE']}，other={bank['NON-PAIR-COVERAGE']}。",
        "",
        "## Scalar diagnostics（raw direction retained）",
        "",
        "| Feature | usable/missing | KEEP median [Q1,Q3] | PRUNE median [Q1,Q3] | Raw ROC | Oriented ROC | Raw PR-AUC | Prevalence | PR/prevalence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    scalar_rows = summary["scalar_group_rows"]
    for feature in FEATURES:
        value = summary["scalar"][feature]
        keep = next(row for row in scalar_rows if row["feature"] == feature and row["group"] == "KEEP")
        prune = next(row for row in scalar_rows if row["feature"] == feature and row["group"] == "PRUNE")
        lines.append(
            f"| {feature} | {value['usable_count']}/{value['missing_count']} | "
            f"{fmt(keep['median'])} [{fmt(keep['q25'])},{fmt(keep['q75'])}] | "
            f"{fmt(prune['median'])} [{fmt(prune['q25'])},{fmt(prune['q75'])}] | "
            f"{fmt(value['ROC_AUC'])} | {fmt(value['oriented_ROC_AUC'])} | {fmt(value['PR_AUC'])} | "
            f"{fmt(value['positive_prevalence'])} | {fmt(value['PR_AUC_over_prevalence'])} |"
        )
    lines += [
        "",
        "C1/C4 的预定义解释方向为 lower-is-KEEP；C2/C3/C5 为 higher-is-KEEP。Oriented ROC 只用于解释和 auxiliary useful-separation gate，raw ROC 与 raw PR-AUC 均保留。",
        "",
        "## Fixed multivariate subject-disjoint OOF",
        "",
        "| Set | Features | ROC-AUC | PR-AUC | Prevalence | PR/prevalence |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for feature_set, feature_names in FEATURE_SETS.items():
        value = summary["multivariate"][feature_set]
        lines.append(
            f"| {feature_set} | {', '.join(feature_names)} | {fmt(value['ROC_AUC'])} | {fmt(value['PR_AUC'])} | "
            f"{fmt(value['positive_prevalence'])} | {fmt(value['PR_AUC_over_prevalence'])} |"
        )
    stability = summary["subject_stability_W2_vs_W0"]
    boot = summary["bootstrap"]
    lines += [
        "",
        "## Subject stability and bootstrap",
        "",
        f"- W2 vs W0 subject PR-AUC improved/equal/worse/undefined：{stability['improved']}/{stability['equal']}/{stability['worse']}/{stability['undefined']}。",
        f"- 1000 paired subject bootstrap ΔPR(W2−W0)：observed={boot['observed_delta_PR_W2_minus_W0']:+.6f}，mean={boot['mean_delta_PR']:+.6f}，95% CI=[{boot['CI95'][0]:+.6f}, {boot['CI95'][1]:+.6f}]；valid={boot['valid_repetitions']}。",
        "",
        "## Decision gates",
        "",
    ]
    for name, passed in summary["decision_gates"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.append(f"- individual useful separation (oriented ROC≥0.60)：`{json.dumps(summary['individual_useful_separation'], sort_keys=True)}`")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Causal KEEP/PRUNE labels and all reported AUC/AP values are retrospective diagnostics. GT never enters C1–C5 or model inputs. W2 missing C4/C5 values use only each LOSO training fold's medians before scaling。",
        "",
        "本结果不构成新 event decoder，也不支持任何 Spotting F1 增益声明。实验到此停止；未执行 pruning、probability-threshold tuning、rescue、recognition、hidden features 或新 aggregation。",
    ]
    return "\n".join(lines) + "\n"


def blocked_report(output_root: Path, reason: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "WINDOW_PAIR_CONSENSUS_SEPARABILITY_AUDIT_SAMMLV_CN.md").write_text(
        "# Window-Pair Consensus Separability Audit — SAMMLV\n\n"
        "`BLOCKED-WINDOW-PAIR-ANCHOR`\n\n"
        f"{reason}\n\nFeature extraction and classifiers were not run.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        gated = base.provenance_gate(argparse.Namespace(raw_dir=args.raw_dir, output_root=args.output_root))
        candidates, anchor, subject_anchor = build_candidate_bank(gated)
        if len(gated["subjects"]) != EXPECTED_SUBJECTS or len(gated["loaded"]) != EXPECTED_VIDEOS:
            raise RuntimeError("raw source cardinality mismatch")
    except Exception as error:
        blocked_report(args.output_root, f"{type(error).__name__}: {error}")
        print("BLOCKED-WINDOW-PAIR-ANCHOR")
        print(f"{type(error).__name__}: {error}")
        return 2

    primary = [row for row in candidates if row["included_primary"]]
    scalar_rows, scalar_summary = scalar_diagnostics(primary)
    oof, fold_rows = loso_oof(primary, list(gated["subjects"]))
    multivariate_rows, multivariate_summary = multivariate_metrics(oof)
    subject_rows, stability = subject_metrics(oof, list(gated["subjects"]))
    bootstrap_result = bootstrap(oof, list(gated["subjects"]))
    decision, gates, individual = decide(multivariate_summary, scalar_summary, bootstrap_result)
    label_counts = Counter(row["label_name"] for row in candidates)
    coverage_counts = Counter(row["coverage_status"] for row in candidates)
    candidate_summary = {
        "total": len(candidates),
        "KEEP": int(label_counts.get("KEEP", 0)),
        "PRUNE": int(label_counts.get("PRUNE", 0)),
        "AMBIGUOUS_MATCH_INTERACTION": int(label_counts.get("AMBIGUOUS_MATCH_INTERACTION", 0)),
        "primary_pair_labeled": len(primary),
        "SINGLE-VIEW-CANDIDATE": int(coverage_counts.get("SINGLE-VIEW-CANDIDATE", 0)),
        "NON-PAIR-COVERAGE": int(coverage_counts.get("NON-PAIR-COVERAGE", 0)),
        "excluded_single_view_labeled": sum(row["label"] is not None and row["coverage_status"] == "SINGLE-VIEW-CANDIDATE" for row in candidates),
        "excluded_ambiguous_pair": sum(row["label"] is None and row["coverage_status"] == "PAIR" for row in candidates),
    }
    summary = {
        "experiment": "window_pair_consensus_separability_audit_sammlv",
        "decision": decision,
        "provenance": {
            "raw_manifest": str(gated["manifest_path"]),
            "raw_manifest_sha256": gated["manifest_sha"],
            "compact_cache_sha256": base.EXPECTED_CACHE_SHA,
            "frozen_config_report_sha256": gated["config_sha"],
            "causal_label_source": str(PRUNING_SOURCE),
            "causal_label_source_sha256": sha256(PRUNING_SOURCE),
            "subjects": EXPECTED_SUBJECTS,
            "videos": EXPECTED_VIDEOS,
            "k_p": K_P,
        },
        "anchor": anchor,
        "subject_anchor": {subject: native.metrics(counts) for subject, counts in subject_anchor.items()},
        "candidate_bank": candidate_summary,
        "scalar": scalar_summary,
        "scalar_group_rows": scalar_rows,
        "multivariate": multivariate_summary,
        "subject_stability_W2_vs_W0": stability,
        "bootstrap": bootstrap_result,
        "decision_gates": gates,
        "individual_useful_separation": individual,
        "protocol": {
            "positive_label": "KEEP",
            "causal_label": "delete one B0 event; KEEP iff delta=(-1,0,+1), PRUNE iff delta=(0,-1,0)",
            "primary_subset": "labeled candidates with peak covered by exactly two raw windows",
            "scalar_missing_not_imputed": True,
            "W2_train_only_median_imputation": True,
            "LOSO_subject_disjoint": True,
            "classifier": "StandardScaler + LogisticRegression(liblinear, balanced, C=1.0, max_iter=1000, random_state=20260906)",
            "bootstrap_repetitions": BOOTSTRAP_REPEATS,
            "bootstrap_seed": SEED,
            "individual_useful_definition": "C3/C4/C5 semantic-oriented ROC-AUC >= 0.60",
            "weak_go_definition": "individual useful and exactly three of four STRONG numeric gates pass",
        },
        "integrity": {
            "GT_in_features": False,
            "new_event_decoder": False,
            "spotting_F1_reported_for_new_method": False,
            "recognition_used": False,
            "stitched_height_used": False,
            "morphology_used": False,
            "window_center_weighting_used": False,
            "aggregation_search_used": False,
            "classifier_tuned": False,
            "probability_threshold_tuned": False,
            "backbone_forward": False,
        },
    }
    output_dir = args.output_root / "outputs"
    write_csv(output_dir / "window_pair_candidate_features.csv", candidates)
    write_csv(output_dir / "window_pair_scalar_diagnostics.csv", scalar_rows)
    write_csv(output_dir / "window_pair_multivariate_metrics.csv", multivariate_rows)
    write_csv(output_dir / "window_pair_oof_predictions.csv", oof)
    write_csv(output_dir / "window_pair_subject_metrics.csv", subject_rows)
    write_csv(output_dir / "window_pair_fold_diagnostics.csv", fold_rows)
    (output_dir / "window_pair_bootstrap.json").write_text(
        json.dumps(bootstrap_result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "window_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "WINDOW_PAIR_CONSENSUS_SEPARABILITY_AUDIT_SAMMLV_CN.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "anchor": anchor, "candidate_bank": candidate_summary, "scalar": scalar_summary, "multivariate": multivariate_summary, "stability": stability, "bootstrap": bootstrap_result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
