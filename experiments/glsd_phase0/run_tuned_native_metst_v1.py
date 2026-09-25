#!/usr/bin/env python3
"""Training-only nested-LOSO sensitivity audit for the ME-TST native decoder."""
import csv
import json
import pickle
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

import run_boundary_phase0_audit as native
import run_dense_segment_phase0_v2 as dense


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/tuned_native_metst_v1"
SMOOTH = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
THRESHOLD = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
DISTANCE = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
BOUNDARY = (0.5, 0.75, 1.0, 1.25, 1.5)
GRID = tuple(product(SMOOTH, THRESHOLD, DISTANCE, BOUNDARY))
NATIVE_CONFIG = (2.0, 0.55, 1.0, 1.0)
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260908
EXPANSION_ROUND = 2
CASES = native.CASES[:2]


def add(first, second):
    return tuple(first[i] + second[i] for i in range(3))


def subtract(first, second):
    return tuple(first[i] - second[i] for i in range(3))


def f1(value):
    tp, fp, fn = value
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def config_distance(config):
    scales = (0.5, 0.1, 0.25, 0.25)
    return sum(abs(value - reference) / scale
               for value, reference, scale in zip(config, NATIVE_CONFIG, scales))


def choose(training_counts):
    return max(GRID, key=lambda config: (
        f1(training_counts[config]), -training_counts[config][1],
        -config_distance(config), -GRID.index(config)
    ))


def decode(record, k, pmf, config):
    smooth, threshold_factor, distance, boundary = config
    curve = pmf.moving_average(np.asarray(record["score"], dtype=float),
                               max(1, round(smooth * k)))
    threshold = float(curve.mean() + threshold_factor * (curve.max() - curve.mean()))
    peaks = pmf.local_maxima(curve, threshold, max(1, round(distance * k)))
    half_width = max(1, round(boundary * k))
    length = len(record["emotion"])
    return [{"onset": max(0, int(peak) - half_width), "peak": int(peak),
             "offset": min(length - 1, int(peak) + half_width)} for peak in peaks]


def decode_grid(record, k, pmf):
    """Decode the full grid while reusing each candidate-generation result."""
    values = np.asarray(record["score"], dtype=float)
    length = len(record["emotion"])
    decoded = {}
    for smooth in SMOOTH:
        curve = pmf.moving_average(values, max(1, round(smooth * k)))
        mean = float(curve.mean())
        spread = float(curve.max() - mean)
        for threshold_factor in THRESHOLD:
            threshold = mean + threshold_factor * spread
            for distance in DISTANCE:
                peaks = pmf.local_maxima(curve, threshold, max(1, round(distance * k)))
                for boundary in BOUNDARY:
                    half_width = max(1, round(boundary * k))
                    decoded[(smooth, threshold_factor, distance, boundary)] = [
                        {"onset": max(0, int(peak) - half_width), "peak": int(peak),
                         "offset": min(length - 1, int(peak) + half_width)}
                        for peak in peaks
                    ]
    return decoded


def bootstrap(subject_counts):
    subjects = sorted(subject_counts)
    baseline = np.asarray([subject_counts[s]["Native"] for s in subjects], dtype=int)
    proposed = np.asarray([subject_counts[s]["TunedNative"] for s in subjects], dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPEATS)
    for repeat in range(BOOTSTRAP_REPEATS):
        sample = rng.integers(0, len(subjects), len(subjects))
        values[repeat] = f1(proposed[sample].sum(axis=0)) - f1(baseline[sample].sum(axis=0))
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pmf = native.load_pmf()
    summaries, selections, train_scores, predictions, reports = [], [], [], [], []
    for backbone, dataset, cache_path, _, in CASES:
        payload = pickle.load(cache_path.open("rb"))
        k = int(payload["k_p"])
        by_subject = defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))
        stored = []
        native_prediction_match = True
        for record in payload["records"]:
            subject, video = str(record["subject"]), str(record["video"])
            gt = record["samples"]
            official = pmf.native_detections(record, payload)[0]
            replay = decode(record, k, pmf, NATIVE_CONFIG)
            native_prediction_match &= [
                (x["onset"], x["peak"], x["offset"]) for x in replay
            ] == [(x["onset"], x["peak"], x["offset"]) for x in official]
            decoded = decode_grid(record, k, pmf)
            for config, rows in decoded.items():
                value, _ = dense.counts(rows, gt)
                by_subject[subject][config] = add(by_subject[subject][config], value)
            stored.append((subject, video, gt, decoded))

        subjects = sorted(by_subject)
        totals = {config: tuple(sum(by_subject[s][config][i] for s in subjects)
                                for i in range(3)) for config in GRID}
        selected = {}
        for outer in subjects:
            training = {config: subtract(totals[config], by_subject[outer][config])
                        for config in GRID}
            chosen = choose(training)
            selected[outer] = chosen
            for rank, config in enumerate(GRID):
                value = training[config]
                train_scores.append({
                    "backbone": backbone, "dataset": dataset, "outer_subject": outer,
                    "config_order": rank, "smooth": config[0], "threshold": config[1],
                    "distance": config[2], "boundary": config[3], "TP": value[0],
                    "FP": value[1], "FN": value[2], "F1": f1(value),
                })
            value = training[chosen]
            selections.append({
                "backbone": backbone, "dataset": dataset, "outer_subject": outer,
                "smooth": chosen[0], "threshold": chosen[1], "distance": chosen[2],
                "boundary": chosen[3], "training_TP": value[0], "training_FP": value[1],
                "training_FN": value[2], "training_F1": f1(value),
                "smooth_boundary": int(chosen[0] in (min(SMOOTH), max(SMOOTH))),
                "threshold_boundary": int(chosen[1] in (min(THRESHOLD), max(THRESHOLD))),
                "distance_boundary": int(chosen[2] in (min(DISTANCE), max(DISTANCE))),
                "interval_boundary": int(chosen[3] in (min(BOUNDARY), max(BOUNDARY))),
            })

        aggregate = {"Native": (0, 0, 0), "TunedNative": (0, 0, 0)}
        subject_values = {s: {"Native": (0, 0, 0), "TunedNative": (0, 0, 0)} for s in subjects}
        for subject, video, gt, decoded in stored:
            for method, config in (("Native", NATIVE_CONFIG), ("TunedNative", selected[subject])):
                rows = decoded[config]
                value, _ = dense.counts(rows, gt)
                aggregate[method] = add(aggregate[method], value)
                subject_values[subject][method] = add(subject_values[subject][method], value)
                for row in rows:
                    predictions.append({"backbone": backbone, "dataset": dataset,
                                        "subject": subject, "video": video, "method": method,
                                        "onset": row["onset"], "peak": row["peak"],
                                        "offset": row["offset"]})
        boot = bootstrap(subject_values)
        native_value, tuned_value = aggregate["Native"], aggregate["TunedNative"]
        frequency = Counter(selected.values())
        group_selections = [row for row in selections
                            if row["backbone"] == backbone and row["dataset"] == dataset]
        report = {
            "backbone": backbone, "dataset": dataset,
            "native_prediction_exact_replay": native_prediction_match,
            "native_counts_exact_replay": native_value == dense.EXPECTED[(backbone, dataset)],
            "Native": {"TP": native_value[0], "FP": native_value[1], "FN": native_value[2],
                       "F1": f1(native_value)},
            "TunedNative": {"TP": tuned_value[0], "FP": tuned_value[1], "FN": tuned_value[2],
                            "F1": f1(tuned_value), "delta": f1(tuned_value) - f1(native_value),
                            "bootstrap_mean_delta": boot[0], "bootstrap_ci": [boot[1], boot[2]]},
            "selection_frequency": [
                {"config": config, "count": count} for config, count in frequency.most_common()
            ],
            "boundary_fold_counts": {
                name: sum(row[name] for row in group_selections)
                for name in ("smooth_boundary", "threshold_boundary", "distance_boundary",
                             "interval_boundary")
            },
        }
        reports.append(report)
        for method, value in aggregate.items():
            summaries.append({"backbone": backbone, "dataset": dataset, "method": method,
                              "TP": value[0], "FP": value[1], "FN": value[2], "F1": f1(value),
                              "delta_vs_native": f1(value) - f1(native_value)})
        print(json.dumps(report, ensure_ascii=False), flush=True)

    write_csv(OUT / "summary.csv", summaries)
    write_csv(OUT / "outer_loso_selections.csv", selections)
    write_csv(OUT / "outer_train_config_scores.csv", train_scores)
    write_csv(OUT / "selected_predictions.csv", predictions)
    protocol = {
        "purpose": "ME-TST native decoder parameter sensitivity; not a proposed method",
        "grid": {"smooth": SMOOTH, "threshold": THRESHOLD,
                 "distance": DISTANCE, "boundary": BOUNDARY},
        "native_config": NATIVE_CONFIG,
        "selection": "outer-train pooled F1; ties fewer FP then closest to Native",
        "test_subject_used_for_selection": False,
        "candidate_generator_reads_GT": False,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "expansion_round": EXPANSION_ROUND,
        "maximum_expansion_rounds": 2,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    boundary = any(any(value > 0 for value in report["boundary_fold_counts"].values())
                   for report in reports)
    exact = all(report["native_prediction_exact_replay"] and report["native_counts_exact_replay"]
                for report in reports)
    combined = {
        "status": "COMPLETE" if exact else "REPLAY_MISMATCH",
        "decision": "FINAL_GRID_WITH_SPARSE_BOUNDARY_FOLDS" if exact and boundary else
                    "TUNED_NATIVE_LOCKABLE" if exact else "REPLAY_MISMATCH",
        "reports": reports,
    }
    (OUT / "combined_report.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
