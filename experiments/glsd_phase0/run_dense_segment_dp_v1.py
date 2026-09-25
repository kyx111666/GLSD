#!/usr/bin/env python3
"""Nested-LOSO duration-aware dense segment decoder V1.

The backbone and curves are frozen. Candidate intervals and their scores never
read GT. Outer-test GT is used only after the configuration has been selected
on the remaining subjects.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_boundary_phase0_audit as native
import run_dense_segment_phase0_v2 as phase0


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/dense_segment_dp_v1"
# The last two values are the two predeclared upper-bound expansions after the
# initial grid ended at 32 with every fold selecting that boundary.
LAMBDAS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0)
ETAS = (0.0, 1.0, 2.0)
FAMILIES = {
    "fixed": (1.0,),
    "compact": (0.5, 1.0, 2.0),
    "wide": (0.25, 0.5, 1.0, 2.0, 4.0),
}
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260908


def write_csv(path, rows):
    if not rows:
        rows = [{"note": "no rows"}]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add(first, second):
    return tuple(first[index] + second[index] for index in range(3))


def subtract(first, second):
    return tuple(first[index] - second[index] for index in range(3))


def f1(counts):
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def durations_for(factors, k, length):
    base = 2 * int(k) + 1
    return tuple(sorted({min(length, max(1, int(round(base * factor))))
                         for factor in factors}))


def interval_scores(curve, durations):
    z, fallback = phase0.robust_z(curve)
    prefix = np.r_[0.0, np.cumsum(z)]
    return {
        duration: (prefix[duration:] - prefix[:-duration]) / np.sqrt(duration)
        for duration in durations
    }, fallback


def decode_dp(score_by_duration, length, base_duration, lam, eta):
    """Exact weighted non-overlap interval DP with deterministic ties."""
    best = np.zeros(length + 1, dtype=float)
    chosen_duration = np.zeros(length + 1, dtype=np.int32)
    durations = tuple(sorted(score_by_duration))
    for end in range(1, length + 1):
        value = best[end - 1]
        selected = 0
        for duration in durations:
            if duration > end:
                break
            candidate = (best[end - duration]
                         + score_by_duration[duration][end - duration]
                         - eta * duration / base_duration - lam)
            if candidate > value + 1e-12:
                value = candidate
                selected = duration
        best[end] = value
        chosen_duration[end] = selected

    predictions = []
    end = length
    while end:
        duration = int(chosen_duration[end])
        if duration == 0:
            end -= 1
            continue
        predictions.append({"onset": end - duration, "offset": end - 1,
                            "duration": duration})
        end -= duration
    return list(reversed(predictions))


def configs(method):
    families = ("fixed",) if method == "FixedDuration" else ("compact", "wide")
    rows = []
    for family in families:
        etas = (0.0,) if family == "fixed" else ETAS
        for lam in LAMBDAS:
            for eta in etas:
                rows.append((family, lam, eta))
    return tuple(rows)


def choose_config(method_configs, training_counts):
    best_config = method_configs[0]
    best_key = None
    for order, config in enumerate(method_configs):
        value = training_counts[config]
        key = (f1(value), -value[1], -order)
        if best_key is None or key > best_key:
            best_key = key
            best_config = config
    return best_config


def bootstrap_delta(subject_rows, proposed, repeats=BOOTSTRAP_REPEATS):
    subjects = sorted(subject_rows)
    native_values = np.asarray([subject_rows[s]["Native"] for s in subjects], dtype=int)
    proposed_values = np.asarray([subject_rows[s][proposed] for s in subjects], dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(subjects), len(subjects))
        deltas[index] = f1(proposed_values[sample].sum(axis=0)) - f1(native_values[sample].sum(axis=0))
    return {
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "repeats": repeats,
        "seed": BOOTSTRAP_SEED,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pmf = native.load_pmf()
    summary_rows = []
    selection_rows = []
    training_score_rows = []
    prediction_rows = []
    reports = []

    for backbone, dataset, cache_path, target_path in native.CASES:
        k, videos = native.group_audit(backbone, dataset, cache_path, target_path, pmf)
        methods = ("FixedDuration", "VariableDuration")
        method_configs = {method: configs(method) for method in methods}
        all_configs = tuple(dict.fromkeys(config for method in methods
                                          for config in method_configs[method]))
        subject_counts = defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))
        native_subject_counts = defaultdict(lambda: (0, 0, 0))
        prepared_videos = []

        for video_index, (subject, video, gt, native_predictions, _, _, source_curve, _, _) in enumerate(videos):
            curve = phase0.native_smoothed_response(backbone, source_curve, k)
            native_value, _ = phase0.counts(native_predictions, gt)
            native_subject_counts[str(subject)] = add(native_subject_counts[str(subject)], native_value)
            family_scores = {}
            fallbacks = {}
            for family, factors in FAMILIES.items():
                durations = durations_for(factors, k, len(curve))
                family_scores[family], fallbacks[family] = interval_scores(curve, durations)
            prepared_videos.append((str(subject), str(video), gt, native_predictions,
                                    len(curve), family_scores, fallbacks))
            for family, lam, eta in all_configs:
                predictions = decode_dp(family_scores[family], len(curve), 2 * k + 1, lam, eta)
                value, _ = phase0.counts(predictions, gt)
                subject_counts[str(subject)][family, lam, eta] = add(
                    subject_counts[str(subject)][family, lam, eta], value)

        subjects = sorted(native_subject_counts)
        total_by_config = {
            config: tuple(sum(subject_counts[s][config][index] for s in subjects)
                          for index in range(3))
            for config in all_configs
        }
        selected_by_method = {method: {} for method in methods}
        for outer in subjects:
            for method in methods:
                training = {config: subtract(total_by_config[config], subject_counts[outer][config])
                            for config in method_configs[method]}
                for config_order, config in enumerate(method_configs[method]):
                    family_i, lam_i, eta_i = config
                    value_i = training[config]
                    training_score_rows.append({
                        "backbone": backbone, "dataset": dataset,
                        "outer_subject": outer, "method": method,
                        "config_order": config_order, "family": family_i,
                        "lambda": lam_i, "eta": eta_i,
                        "training_TP": value_i[0], "training_FP": value_i[1],
                        "training_FN": value_i[2], "training_F1": f1(value_i),
                    })
                selected = choose_config(method_configs[method], training)
                selected_by_method[method][outer] = selected
                family, lam, eta = selected
                best_counts = training[selected]
                lambda_index = LAMBDAS.index(lam)
                selection_rows.append({
                    "backbone": backbone, "dataset": dataset, "outer_subject": outer,
                    "method": method, "family": family, "lambda": lam, "eta": eta,
                    "training_TP": best_counts[0], "training_FP": best_counts[1],
                    "training_FN": best_counts[2], "training_F1": f1(best_counts),
                    "lambda_at_lower_boundary": int(lambda_index == 0),
                    "lambda_at_upper_boundary": int(lambda_index == len(LAMBDAS) - 1),
                })

        aggregate = {"Native": (0, 0, 0), **{method: (0, 0, 0) for method in methods}}
        per_subject = {}
        for subject in subjects:
            per_subject[subject] = {"Native": native_subject_counts[subject]}
            aggregate["Native"] = add(aggregate["Native"], native_subject_counts[subject])
            for method in methods:
                selected = selected_by_method[method][subject]
                value = subject_counts[subject][selected]
                per_subject[subject][method] = value
                aggregate[method] = add(aggregate[method], value)

        for subject, video, gt, native_predictions, length, family_scores, fallbacks in prepared_videos:
            for row in native_predictions:
                prediction_rows.append({"backbone": backbone, "dataset": dataset,
                                        "subject": subject, "video": video, "method": "Native",
                                        "onset": row["onset"], "offset": row["offset"]})
            for method in methods:
                family, lam, eta = selected_by_method[method][subject]
                predictions = decode_dp(family_scores[family], length, 2 * k + 1, lam, eta)
                for row in predictions:
                    prediction_rows.append({"backbone": backbone, "dataset": dataset,
                                            "subject": subject, "video": video, "method": method,
                                            "onset": row["onset"], "offset": row["offset"]})

        native_value = aggregate["Native"]
        for method, value in aggregate.items():
            delta = f1(value) - f1(native_value)
            if method == "Native":
                boot = {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                        "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED}
            else:
                boot = bootstrap_delta(per_subject, method)
            directions = Counter()
            for subject in subjects:
                difference = f1(per_subject[subject][method]) - f1(per_subject[subject]["Native"])
                directions["improved" if difference > 0 else "worse" if difference < 0 else "equal"] += 1
            summary_rows.append({
                "backbone": backbone, "dataset": dataset, "method": method,
                "TP": value[0], "FP": value[1], "FN": value[2], "F1": f1(value),
                "delta_vs_native": delta, "bootstrap_mean_delta": boot["mean_delta"],
                "bootstrap_ci_low": boot["ci_low"], "bootstrap_ci_high": boot["ci_high"],
                "subjects_improved": directions["improved"], "subjects_equal": directions["equal"],
                "subjects_worse": directions["worse"],
            })

        variable_selections = [row for row in selection_rows
                               if row["backbone"] == backbone and row["dataset"] == dataset
                               and row["method"] == "VariableDuration"]
        lambda_frequency = Counter(str(row["lambda"]) for row in variable_selections)
        upper_count = sum(row["lambda_at_upper_boundary"] for row in variable_selections)
        report = {
            "backbone": backbone, "dataset": dataset,
            "native_exact_replay": native_value == phase0.EXPECTED[backbone, dataset],
            "aggregate": {method: {"TP": value[0], "FP": value[1], "FN": value[2],
                                    "F1": f1(value),
                                    "delta_vs_native": f1(value) - f1(native_value)}
                          for method, value in aggregate.items()},
            "variable_lambda_frequency": dict(lambda_frequency),
            "variable_upper_boundary_count": upper_count,
            "grid_status": "UNRESOLVED_UPPER_LAMBDA" if upper_count > len(subjects) / 2
                           else "NO_MAJORITY_UPPER_LAMBDA",
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    write_csv(OUT / "summary.csv", summary_rows)
    write_csv(OUT / "outer_loso_selections.csv", selection_rows)
    write_csv(OUT / "outer_train_config_scores.csv", training_score_rows)
    write_csv(OUT / "selected_predictions.csv", prediction_rows)
    protocol = {
        "experiment": "duration-aware dense segment DP V1",
        "trainable_model_parameters": 0,
        "candidate_generation_reads_GT": False,
        "selection": "outer-train subjects pooled F1; tie fewer FP then fixed config order",
        "score": "sum(robust_z)/sqrt(duration) - eta*duration/(2*k+1) - lambda",
        "families": FAMILIES, "lambdas": LAMBDAS, "etas": ETAS,
        "DP": "exact non-overlapping weighted interval selection; skip wins ties",
        "test_GT_use": "evaluation only after training-only config selection",
        "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
                      "unit": "subject", "metric": "pooled F1 delta"},
        "phase0_report": str(phase0.OUT / "combined_report.json"),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    exact = all(report["native_exact_replay"] for report in reports)
    variable_rows = [row for row in summary_rows if row["method"] == "VariableDuration"]
    fixed_rows = [row for row in summary_rows if row["method"] == "FixedDuration"]
    positive_native = sum(row["delta_vs_native"] > 0 for row in variable_rows)
    positive_fixed = sum(row["F1"] > fixed["F1"]
                         for row, fixed in zip(variable_rows, fixed_rows))
    unresolved = any(report["grid_status"].startswith("UNRESOLVED") for report in reports)
    if not exact:
        decision = "REPLAY_MISMATCH"
    elif unresolved:
        decision = "BORDERLINE_GRID_UNRESOLVED"
    elif positive_native >= 3 and positive_fixed >= 3:
        decision = "GO"
    else:
        decision = "STOP_V1"
    combined = {
        "status": "DENSE_SEGMENT_DP_V1_COMPLETE" if exact else "NATIVE_REPLAY_MISMATCH",
        "decision": decision,
        "positive_vs_native": positive_native,
        "positive_vs_fixed_duration": positive_fixed,
        "reports": reports,
        "summary": summary_rows,
        "files": {path.name: str(path) for path in OUT.iterdir() if path.is_file()},
    }
    (OUT / "combined_report.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
