#!/usr/bin/env python3
"""Fixed-dictionary sparse event reconstruction Phase 0.

Compares independent matched-filter ranking with nonnegative orthogonal
matching pursuit on the same GT-independent interval dictionary. This is a
mechanism audit, not the final L1 decoder.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import average_precision_score, roc_auc_score

import run_boundary_phase0_audit as native
import run_dense_segment_phase0_v2 as dense


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/sparse_event_reconstruction_phase0"
TEMPLATE_KINDS = ("box", "triangle")
DURATION_FACTORS = (0.5, 1.0, 2.0)
REPLACEMENT_BUDGETS = ("native_count", "native_or_one")
RESCUE_BUDGETS = ("one_per_video", "three_per_video", "native_or_one")


def write_csv(path, rows):
    if not rows:
        rows = [{"note": "no rows"}]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add(first, second):
    return tuple(first[index] + second[index] for index in range(3))


def f1(counts):
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def template(kind, duration):
    if kind == "box":
        values = np.ones(duration, dtype=float)
    elif kind == "triangle":
        values = np.bartlett(duration + 2)[1:-1]
    else:
        raise ValueError(f"unknown template kind: {kind}")
    norm = np.linalg.norm(values)
    return values / norm


def dictionary(curve, k, kind):
    durations = dense.duration_set(k, len(curve))
    templates = {duration: template(kind, duration) for duration in durations}
    return durations, templates


def correlations(signal, templates):
    return {duration: np.correlate(signal, atom, mode="valid")
            for duration, atom in templates.items()}


def allowed_starts(length, durations, blocked_intervals=()):
    allowed = {}
    for duration in durations:
        starts = np.arange(length - duration + 1)
        ends = starts + duration - 1
        mask = np.ones(len(starts), dtype=bool)
        for blocked in blocked_intervals:
            mask &= (ends < int(blocked["onset"])) | (starts > int(blocked["offset"]))
        allowed[duration] = mask
    return allowed


def best_atom(correlation, allowed, selected):
    best = None
    for duration in sorted(correlation):
        values = correlation[duration]
        valid = np.flatnonzero(allowed[duration])
        if not len(valid):
            continue
        order = valid[np.argsort(-values[valid], kind="stable")]
        for start in order:
            key = (duration, int(start))
            if key in selected:
                continue
            candidate = (float(values[start]), -duration, -int(start), duration, int(start))
            if best is None or candidate > best:
                best = candidate
            break
    if best is None or best[0] <= 0:
        return None
    return best[3], best[4], best[0]


def atom_column(length, duration, start, values):
    column = np.zeros(length, dtype=float)
    column[start:start + duration] = values
    return column


def anchor_columns(length, native_predictions, kind):
    columns = []
    for prediction in native_predictions:
        start = max(0, int(prediction["onset"]))
        end = min(length - 1, int(prediction["offset"]))
        if end < start:
            continue
        values = template(kind, end - start + 1)
        columns.append(atom_column(length, len(values), start, values))
    return columns


def refit_residual(signal, fixed_columns, selected, templates):
    columns = list(fixed_columns)
    columns.extend(atom_column(len(signal), duration, start, templates[duration])
                   for duration, start in selected)
    if not columns:
        return signal.copy(), 0.0
    design = np.column_stack(columns)
    coefficients, _ = nnls(design, signal, maxiter=max(3 * design.shape[1], 30))
    residual = signal - design @ coefficients
    explained = 1.0 - float(residual @ residual) / max(float(signal @ signal), 1e-12)
    return residual, explained


def matched_filter(signal, templates, allowed, budget):
    corr = correlations(signal, templates)
    ranked = []
    for duration in sorted(corr):
        for start in np.flatnonzero(allowed[duration]):
            score = float(corr[duration][start])
            if score > 0:
                ranked.append((score, -duration, -int(start), duration, int(start)))
    ranked.sort(reverse=True)
    selected = [(row[3], row[4]) for row in ranked[:budget]]
    return selected


def residual_pursuit(signal, templates, allowed, budget, fixed_columns=()):
    selected = []
    selected_set = set()
    residual, initial_explained = refit_residual(signal, fixed_columns, selected, templates)
    for _ in range(budget):
        candidate = best_atom(correlations(residual, templates), allowed, selected_set)
        if candidate is None:
            break
        duration, start, _ = candidate
        selected.append((duration, start))
        selected_set.add((duration, start))
        residual, _ = refit_residual(signal, fixed_columns, selected, templates)
    _, final_explained = refit_residual(signal, fixed_columns, selected, templates)
    return selected, initial_explained, final_explained


def predictions(selected):
    return sorted(({"onset": start, "offset": start + duration - 1,
                    "duration": duration} for duration, start in selected),
                  key=lambda row: (row["onset"], row["offset"]))


def duplicate_pairs(rows):
    return sum(native.iou((rows[i]["onset"], rows[i]["offset"]),
                          (rows[j]["onset"], rows[j]["offset"])) >= 0.5
               for i in range(len(rows)) for j in range(i + 1, len(rows)))


def budget_value(rule, native_count):
    if rule == "one_per_video":
        return 1
    if rule == "three_per_video":
        return 3
    if rule == "native_count":
        return native_count
    return max(native_count, 1)


def generation_miss_ids(gt, native_predictions, native_candidates):
    matched = {gt_index for _, gt_index in native.match(native_predictions, gt)}
    selected = [row for row in native_candidates if row["selected"]]
    ids = set()
    for gt_index, event in enumerate(gt):
        if gt_index in matched:
            continue
        if not any(event[0] <= row["peak"] <= event[2] for row in selected):
            ids.add(gt_index)
    return ids


def rank_metrics(labels, values):
    labels = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(values, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None
    return float(roc_auc_score(labels, values)), float(average_precision_score(labels, values))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pmf = native.load_pmf()
    full_pool_rows = []
    result_rows = []
    mechanism_rows = []
    group_rows = []

    for backbone, dataset, cache_path, target_path in native.CASES:
        k, videos = native.group_audit(backbone, dataset, cache_path, target_path, pmf)
        native_total = (0, 0, 0)
        totals = defaultdict(lambda: (0, 0, 0))
        selected_counts = defaultdict(int)
        duplicates = defaultdict(int)
        rescued_gt = defaultdict(int)
        rescued_generation = defaultdict(int)
        explained_gain = defaultdict(list)
        pool_labels = {kind: [] for kind in TEMPLATE_KINDS}
        pool_scores = {kind: [] for kind in TEMPLATE_KINDS}

        for subject, video, gt, native_predictions, native_candidates, _, source_curve, _, _ in videos:
            curve = dense.native_smoothed_response(backbone, source_curve, k)
            signal, _ = dense.robust_z(curve)
            native_value, native_pairs = dense.counts(native_predictions, gt)
            native_total = add(native_total, native_value)
            native_gt = {gt_index for _, gt_index in native_pairs}
            generation_ids = generation_miss_ids(gt, native_predictions, native_candidates)

            for kind in TEMPLATE_KINDS:
                durations, templates = dictionary(signal, k, kind)
                all_allowed = allowed_starts(len(signal), durations)
                blocked_allowed = allowed_starts(len(signal), durations, native_predictions)
                corr = correlations(signal, templates)

                candidates = []
                values = []
                for duration in durations:
                    for start, score in enumerate(corr[duration]):
                        candidates.append({"onset": start, "offset": start + duration - 1})
                        values.append(float(score))
                pool_labels[kind].append(dense.candidate_labels(candidates, gt))
                pool_scores[kind].append(np.asarray(values))

                for rule in REPLACEMENT_BUDGETS:
                    budget = budget_value(rule, len(native_predictions))
                    for method in ("MatchedFilter", "ResidualPursuit"):
                        if method == "MatchedFilter":
                            chosen = matched_filter(signal, templates, all_allowed, budget)
                            initial_fit = final_fit = 0.0
                        else:
                            chosen, initial_fit, final_fit = residual_pursuit(
                                signal, templates, all_allowed, budget)
                        rows = predictions(chosen)
                        key = ("replacement", kind, rule, method)
                        value, pairs = dense.counts(rows, gt)
                        totals[key] = add(totals[key], value)
                        selected_counts[key] += len(rows)
                        duplicates[key] += duplicate_pairs(rows)
                        explained_gain[key].append(final_fit - initial_fit)
                        chosen_gt = {gt_index for _, gt_index in pairs}
                        rescued_gt[key] += len(chosen_gt - native_gt)
                        rescued_generation[key] += len(chosen_gt & generation_ids)

                anchors = anchor_columns(len(signal), native_predictions, kind)
                for rule in RESCUE_BUDGETS:
                    budget = budget_value(rule, len(native_predictions))
                    for method in ("MatchedFilter", "ResidualPursuit"):
                        if method == "MatchedFilter":
                            chosen = matched_filter(signal, templates, blocked_allowed, budget)
                            initial_fit = final_fit = 0.0
                        else:
                            chosen, initial_fit, final_fit = residual_pursuit(
                                signal, templates, blocked_allowed, budget, anchors)
                        additions = predictions(chosen)
                        rows = sorted([*native_predictions, *additions],
                                      key=lambda row: (row["onset"], row["offset"]))
                        key = ("rescue", kind, rule, method)
                        value, pairs = dense.counts(rows, gt)
                        totals[key] = add(totals[key], value)
                        selected_counts[key] += len(additions)
                        duplicates[key] += duplicate_pairs(additions)
                        explained_gain[key].append(final_fit - initial_fit)
                        chosen_gt = {gt_index for _, gt_index in pairs}
                        rescued_gt[key] += len(chosen_gt - native_gt)
                        rescued_generation[key] += len(chosen_gt & generation_ids)

        exact = native_total == dense.EXPECTED[backbone, dataset]
        group_rows.append({"backbone": backbone, "dataset": dataset,
                           "native_exact_replay": exact,
                           "native_TP": native_total[0], "native_FP": native_total[1],
                           "native_FN": native_total[2], "native_F1": f1(native_total)})
        for kind in TEMPLATE_KINDS:
            labels = np.concatenate(pool_labels[kind])
            values = np.concatenate(pool_scores[kind])
            roc, ap = rank_metrics(labels, values)
            full_pool_rows.append({
                "backbone": backbone, "dataset": dataset, "template": kind,
                "candidate_count": len(labels), "positive_count": int(labels.sum()),
                "positive_rate": float(labels.mean()), "ROC_AUC": roc, "PR_AUC": ap,
                "label": "interval IoU>=0.5; post-hoc only",
            })

        for key in sorted(totals):
            mode, kind, rule, method = key
            value = totals[key]
            result_rows.append({
                "backbone": backbone, "dataset": dataset, "mode": mode,
                "template": kind, "budget_rule": rule, "method": method,
                "TP": value[0], "FP": value[1], "FN": value[2], "F1": f1(value),
                "delta_vs_native": f1(value) - f1(native_total),
                "selected_or_added_atoms": selected_counts[key],
                "duplicate_atom_pairs_iou_ge_0.5": duplicates[key],
                "newly_matched_GT": rescued_gt[key],
                "generation_miss_GT_recovered": rescued_generation[key],
                "mean_incremental_explained_fraction": float(np.mean(explained_gain[key])),
            })

        for mode in ("replacement", "rescue"):
            rules = REPLACEMENT_BUDGETS if mode == "replacement" else RESCUE_BUDGETS
            for kind in TEMPLATE_KINDS:
                for rule in rules:
                    mf = totals[mode, kind, rule, "MatchedFilter"]
                    rp = totals[mode, kind, rule, "ResidualPursuit"]
                    mechanism_rows.append({
                        "backbone": backbone, "dataset": dataset, "mode": mode,
                        "template": kind, "budget_rule": rule,
                        "MatchedFilter_F1": f1(mf), "ResidualPursuit_F1": f1(rp),
                        "Residual_minus_Matched": f1(rp) - f1(mf),
                        "Residual_better": int(f1(rp) > f1(mf)),
                    })
        print(json.dumps(group_rows[-1], ensure_ascii=False), flush=True)

    write_csv(OUT / "group_summary.csv", group_rows)
    write_csv(OUT / "full_pool_matched_filter.csv", full_pool_rows)
    write_csv(OUT / "fixed_budget_results.csv", result_rows)
    write_csv(OUT / "residual_vs_matched_filter.csv", mechanism_rows)

    shared_configs = []
    for mode in ("replacement", "rescue"):
        rules = REPLACEMENT_BUDGETS if mode == "replacement" else RESCUE_BUDGETS
        for kind in TEMPLATE_KINDS:
            for rule in rules:
                rows = [row for row in mechanism_rows
                        if row["mode"] == mode and row["template"] == kind
                        and row["budget_rule"] == rule]
                deltas = [row["Residual_minus_Matched"] for row in rows]
                shared_configs.append({
                    "mode": mode, "template": kind, "budget_rule": rule,
                    "positive_groups": sum(delta > 0 for delta in deltas),
                    "negative_groups": sum(delta < 0 for delta in deltas),
                    "worst_delta": min(deltas), "mean_delta": float(np.mean(deltas)),
                })
    write_csv(OUT / "shared_config_screen.csv", shared_configs)
    best = max(shared_configs, key=lambda row: (row["positive_groups"],
                                                row["worst_delta"], row["mean_delta"]))
    exact = all(row["native_exact_replay"] for row in group_rows)
    go = best["positive_groups"] >= 3 and best["worst_delta"] >= -0.003
    decision = "REPLAY_MISMATCH" if not exact else "GO_TO_CONVEX_V1" if go else "STOP_PHASE0"
    protocol = {
        "candidate_generator_reads_GT": False,
        "templates": TEMPLATE_KINDS, "duration_factors": DURATION_FACTORS,
        "template_normalization": "unit L2",
        "matched_filter": "top positive K^T z atoms; no inter-atom suppression",
        "residual_method": "nonnegative orthogonal matching pursuit with NNLS refit",
        "rescue": "Native predictions retained; candidate atoms overlapping Native are ineligible",
        "training": "none", "trainable_model_parameters": 0,
        "GT_use": "labels, matching, generation-miss audit, and decision only",
        "decision_rule": "a shared config must improve Residual over Matched in >=3/4 groups and worst delta >=-0.003",
        "warning": "Phase-0 pursuit proxy is not the final convex L1 method",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    report = {
        "status": "SPARSE_EVENT_RECONSTRUCTION_PHASE0_COMPLETE" if exact else "NATIVE_REPLAY_MISMATCH",
        "decision": decision, "best_shared_config": best,
        "groups": group_rows, "full_pool": full_pool_rows,
        "shared_config_screen": shared_configs,
        "files": {path.name: str(path) for path in OUT.iterdir() if path.is_file()},
    }
    (OUT / "combined_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
