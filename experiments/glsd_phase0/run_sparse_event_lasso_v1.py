#!/usr/bin/env python3
"""Nested-LOSO positive-Lasso sparse event decoder V1."""
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import run_boundary_phase0_audit as native
import run_dense_segment_phase0_v2 as dense
import run_sparse_event_reconstruction_phase0 as phase0


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/sparse_event_lasso_v1"
RATIOS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
FAMILIES = {"fixed": (1.0,), "compact": (0.5, 1.0, 2.0)}
COEF_TOL = 1e-8
MAX_ITER = 5000
SOLVER_TOL = 1e-5
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


def design_matrix(length, k, factors):
    durations = tuple(sorted({min(length, max(1, int(round((2 * k + 1) * factor))))
                              for factor in factors}))
    row_parts, col_parts, data_parts, metadata = [], [], [], []
    column_offset = 0
    for duration in durations:
        count = length - duration + 1
        starts = np.arange(count, dtype=np.int32)
        rows = (starts[:, None] + np.arange(duration, dtype=np.int32)).ravel()
        columns = np.repeat(np.arange(column_offset, column_offset + count,
                                      dtype=np.int32), duration)
        values = np.tile(phase0.template("box", duration), count)
        row_parts.append(rows)
        col_parts.append(columns)
        data_parts.append(values)
        metadata.extend((duration, int(start)) for start in starts)
        column_offset += count
    matrix = sparse.csc_matrix((np.concatenate(data_parts),
                                (np.concatenate(row_parts), np.concatenate(col_parts))),
                               shape=(length, column_offset))
    return matrix, tuple(metadata), durations


def fit_path(signal, matrix, metadata):
    alpha_max = float(np.max(matrix.T @ signal) / len(signal)) if matrix.shape[1] else 0.0
    outputs = {}
    if alpha_max <= 0:
        empty = {"selected": (), "n_iter": 0, "dual_gap": 0.0,
                 "convergence_warning": False,
                 "objective": 0.5 * float(np.sum(signal ** 2)), "alpha": 0.0}
        return {ratio: dict(empty) for ratio in RATIOS}, alpha_max
    model = Lasso(alpha=alpha_max, positive=True, fit_intercept=False,
                  max_iter=MAX_ITER, tol=SOLVER_TOL, warm_start=True,
                  selection="cyclic")
    for ratio in sorted(RATIOS, reverse=True):
        alpha = ratio * alpha_max
        model.alpha = alpha
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(matrix, signal)
        active = np.flatnonzero(model.coef_ > COEF_TOL)
        selected = tuple(metadata[index] for index in active)
        outputs[ratio] = {
            "selected": selected,
            "n_iter": int(model.n_iter_),
            "dual_gap": float(model.dual_gap_),
            "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
            "objective": (0.5 * float(np.sum((signal - model.predict(matrix)) ** 2))
                          + len(signal) * alpha * float(model.coef_.sum())),
            "alpha": alpha,
        }
    return outputs, alpha_max


def matched_filter_equal_count(signal, matrix, metadata, count):
    if count == 0:
        return ()
    values = np.asarray(matrix.T @ signal).ravel()
    order = np.argsort(-values, kind="stable")[:count]
    return tuple(metadata[index] for index in order if values[index] > 0)


def to_predictions(selected):
    return sorted(({"onset": start, "offset": start + duration - 1,
                    "duration": duration} for duration, start in selected),
                  key=lambda row: (row["onset"], row["offset"]))


def choose_ratio(training_counts):
    best = RATIOS[0]
    best_key = None
    for order, ratio in enumerate(RATIOS):
        counts = training_counts[ratio]
        key = (f1(counts), -counts[1], -order)
        if best_key is None or key > best_key:
            best = ratio
            best_key = key
    return best


def bootstrap(subject_rows, method):
    subjects = sorted(subject_rows)
    native_values = np.asarray([subject_rows[s]["Native"] for s in subjects], dtype=int)
    method_values = np.asarray([subject_rows[s][method] for s in subjects], dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = np.empty(BOOTSTRAP_REPEATS)
    for repeat in range(BOOTSTRAP_REPEATS):
        sample = rng.integers(0, len(subjects), len(subjects))
        delta[repeat] = f1(method_values[sample].sum(axis=0)) - f1(native_values[sample].sum(axis=0))
    return float(delta.mean()), float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pmf = native.load_pmf()
    summary_rows, selection_rows, score_rows = [], [], []
    solver_rows, prediction_rows, reports = [], [], []

    for backbone, dataset, cache_path, target_path in native.CASES:
        k, videos = native.group_audit(backbone, dataset, cache_path, target_path, pmf)
        native_by_subject = defaultdict(lambda: (0, 0, 0))
        counts_by_subject = {
            family: defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))
            for family in FAMILIES
        }
        stored = []

        for subject, video, gt, native_predictions, _, _, source_curve, _, _ in videos:
            subject = str(subject)
            curve = dense.native_smoothed_response(backbone, source_curve, k)
            signal, _ = dense.robust_z(curve)
            native_counts, _ = dense.counts(native_predictions, gt)
            native_by_subject[subject] = add(native_by_subject[subject], native_counts)
            paths = {}
            matrices = {}
            metadata_by_family = {}
            for family, factors in FAMILIES.items():
                matrix, metadata, durations = design_matrix(len(signal), k, factors)
                path, alpha_max = fit_path(signal, matrix, metadata)
                paths[family] = path
                matrices[family] = matrix
                metadata_by_family[family] = metadata
                for ratio in RATIOS:
                    selected = path[ratio]["selected"]
                    value, _ = dense.counts(to_predictions(selected), gt)
                    counts_by_subject[family][subject][ratio] = add(
                        counts_by_subject[family][subject][ratio], value)
                    solver_rows.append({
                        "backbone": backbone, "dataset": dataset, "subject": subject,
                        "video": video, "family": family, "ratio": ratio,
                        "alpha": path[ratio]["alpha"],
                        "alpha_max": alpha_max, "active_atoms": len(selected),
                        "n_iter": path[ratio]["n_iter"], "dual_gap": path[ratio]["dual_gap"],
                        "objective": path[ratio]["objective"],
                        "convergence_warning": int(path[ratio]["convergence_warning"]),
                    })
            stored.append((subject, str(video), gt, native_predictions, signal,
                           paths, matrices, metadata_by_family))

        subjects = sorted(native_by_subject)
        selected_ratio = {family: {} for family in FAMILIES}
        for family in FAMILIES:
            total = {ratio: tuple(sum(counts_by_subject[family][s][ratio][index]
                                      for s in subjects) for index in range(3))
                     for ratio in RATIOS}
            for outer in subjects:
                training = {ratio: subtract(total[ratio],
                                            counts_by_subject[family][outer][ratio])
                            for ratio in RATIOS}
                for order, ratio in enumerate(RATIOS):
                    value = training[ratio]
                    score_rows.append({
                        "backbone": backbone, "dataset": dataset,
                        "outer_subject": outer, "family": family,
                        "ratio_order": order, "ratio": ratio,
                        "training_TP": value[0], "training_FP": value[1],
                        "training_FN": value[2], "training_F1": f1(value),
                    })
                chosen = choose_ratio(training)
                selected_ratio[family][outer] = chosen
                value = training[chosen]
                selection_rows.append({
                    "backbone": backbone, "dataset": dataset,
                    "outer_subject": outer, "family": family, "ratio": chosen,
                    "training_TP": value[0], "training_FP": value[1],
                    "training_FN": value[2], "training_F1": f1(value),
                    "ratio_at_lower_boundary": int(chosen == min(RATIOS)),
                    "ratio_at_upper_boundary": int(chosen == max(RATIOS)),
                })

        methods = ("Native", "LassoFixed", "LassoCompact", "MatchedEqualCount")
        aggregate = {method: (0, 0, 0) for method in methods}
        subject_values = {s: {method: (0, 0, 0) for method in methods} for s in subjects}
        for subject, video, gt, native_predictions, signal, paths, matrices, metadata in stored:
            native_counts, _ = dense.counts(native_predictions, gt)
            aggregate["Native"] = add(aggregate["Native"], native_counts)
            subject_values[subject]["Native"] = add(subject_values[subject]["Native"], native_counts)
            for row in native_predictions:
                prediction_rows.append({"backbone": backbone, "dataset": dataset,
                                        "subject": subject, "video": video, "method": "Native",
                                        "onset": row["onset"], "offset": row["offset"]})

            fixed_ratio = selected_ratio["fixed"][subject]
            compact_ratio = selected_ratio["compact"][subject]
            selections = {
                "LassoFixed": paths["fixed"][fixed_ratio]["selected"],
                "LassoCompact": paths["compact"][compact_ratio]["selected"],
            }
            selections["MatchedEqualCount"] = matched_filter_equal_count(
                signal, matrices["compact"], metadata["compact"],
                len(selections["LassoCompact"]))
            for method in methods[1:]:
                rows = to_predictions(selections[method])
                value, _ = dense.counts(rows, gt)
                aggregate[method] = add(aggregate[method], value)
                subject_values[subject][method] = add(subject_values[subject][method], value)
                for row in rows:
                    prediction_rows.append({"backbone": backbone, "dataset": dataset,
                                            "subject": subject, "video": video, "method": method,
                                            "onset": row["onset"], "offset": row["offset"]})

        native_value = aggregate["Native"]
        for method in methods:
            value = aggregate[method]
            if method == "Native":
                boot = (0.0, 0.0, 0.0)
            else:
                boot = bootstrap(subject_values, method)
            direction = Counter()
            for subject in subjects:
                delta = f1(subject_values[subject][method]) - f1(subject_values[subject]["Native"])
                direction["improved" if delta > 0 else "worse" if delta < 0 else "equal"] += 1
            summary_rows.append({
                "backbone": backbone, "dataset": dataset, "method": method,
                "TP": value[0], "FP": value[1], "FN": value[2], "F1": f1(value),
                "delta_vs_native": f1(value) - f1(native_value),
                "bootstrap_mean_delta": boot[0], "bootstrap_ci_low": boot[1],
                "bootstrap_ci_high": boot[2], "subjects_improved": direction["improved"],
                "subjects_equal": direction["equal"], "subjects_worse": direction["worse"],
            })

        selected_compact = [row for row in selection_rows
                            if row["backbone"] == backbone and row["dataset"] == dataset
                            and row["family"] == "compact"]
        ratio_frequency = Counter(str(row["ratio"]) for row in selected_compact)
        convergence = [row for row in solver_rows
                       if row["backbone"] == backbone and row["dataset"] == dataset]
        report = {
            "backbone": backbone, "dataset": dataset,
            "native_exact_replay": native_value == dense.EXPECTED[backbone, dataset],
            "aggregate": {method: {"TP": value[0], "FP": value[1], "FN": value[2],
                                    "F1": f1(value),
                                    "delta_vs_native": f1(value) - f1(native_value)}
                          for method, value in aggregate.items()},
            "compact_ratio_frequency": dict(ratio_frequency),
            "compact_lower_boundary_count": sum(row["ratio_at_lower_boundary"] for row in selected_compact),
            "compact_upper_boundary_count": sum(row["ratio_at_upper_boundary"] for row in selected_compact),
            "convergence_warning_count_all_grid": sum(row["convergence_warning"] for row in convergence),
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    write_csv(OUT / "summary.csv", summary_rows)
    write_csv(OUT / "outer_loso_selections.csv", selection_rows)
    write_csv(OUT / "outer_train_config_scores.csv", score_rows)
    write_csv(OUT / "solver_audit.csv", solver_rows)
    write_csv(OUT / "selected_predictions.csv", prediction_rows)
    protocol = {
        "method": "positive Lasso fixed-dictionary event decoding",
        "objective": "(1/(2T))*||z-Ka||^2 + alpha*||a||_1, a>=0",
        "template": "box, locked after Phase 0", "families": FAMILIES,
        "regularization_ratios": RATIOS,
        "alpha_definition": "alpha = ratio * max(K^T z) / T per video",
        "coefficient_tolerance": COEF_TOL,
        "max_iter": MAX_ITER, "solver_tolerance": SOLVER_TOL,
        "candidate_generator_reads_GT": False, "trainable_model_parameters": 0,
        "selection": "outer-train pooled F1; ties fewer FP then lower alpha order",
        "matched_control": "top K^Tz atoms with the same per-video atom count as LassoCompact",
        "development_limitation": "box template selected using exploratory four-group Phase 0",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    exact = all(report["native_exact_replay"] for report in reports)
    compact_rows = [row for row in summary_rows if row["method"] == "LassoCompact"]
    fixed_rows = [row for row in summary_rows if row["method"] == "LassoFixed"]
    matched_rows = [row for row in summary_rows if row["method"] == "MatchedEqualCount"]
    positive_native = sum(row["delta_vs_native"] > 0 for row in compact_rows)
    positive_fixed = sum(row["F1"] > control["F1"] for row, control in zip(compact_rows, fixed_rows))
    positive_matched = sum(row["F1"] > control["F1"] for row, control in zip(compact_rows, matched_rows))
    boundary = any(report["compact_lower_boundary_count"] > 0
                   or report["compact_upper_boundary_count"] > 0 for report in reports)
    convergence = sum(report["convergence_warning_count_all_grid"] for report in reports)
    if not exact:
        decision = "REPLAY_MISMATCH"
    elif boundary or convergence:
        decision = "BORDERLINE_GRID_OR_SOLVER_UNRESOLVED"
    elif positive_native == 4 and positive_fixed >= 3 and positive_matched >= 3:
        decision = "STRONG_GO_DEVELOPMENT"
    elif positive_native >= 3 and positive_matched >= 3:
        decision = "GO_DEVELOPMENT"
    else:
        decision = "STOP_V1"
    combined = {
        "status": "SPARSE_EVENT_LASSO_V1_COMPLETE" if exact else "NATIVE_REPLAY_MISMATCH",
        "decision": decision, "positive_vs_native": positive_native,
        "positive_vs_fixed_duration": positive_fixed,
        "positive_vs_matched_equal_count": positive_matched,
        "solver_warning_count_all_grid": convergence,
        "reports": reports, "summary": summary_rows,
        "files": {path.name: str(path) for path in OUT.iterdir() if path.is_file()},
    }
    (OUT / "combined_report.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
