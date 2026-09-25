#!/usr/bin/env python3
"""Dense segment Phase 0 on frozen temporal curves.

This is an opportunity audit, not a tuned decoder. Candidate generation is
GT-independent. Ground truth is read only by the oracle/evaluation functions.
"""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import run_boundary_phase0_audit as native


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/dense_segment_phase0_v2"
DURATION_FACTORS = (0.5, 1.0, 2.0)
BUDGET_RULES = ("native_count", "native_or_one", "double_plus_one")
RESCUE_BUDGETS = ("one_per_video", "native_or_one")
EXPECTED = {
    ("ME-TST", "SAMMLV"): (53, 184, 106),
    ("ME-TST", "CASME3"): (81, 912, 777),
    ("BoostingVRME", "SAMMLV"): (49, 145, 110),
    ("BoostingVRME", "CASME3"): (93, 818, 765),
}


def write_csv(path, rows):
    if not rows:
        rows = [{"note": "no rows"}]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_z(curve):
    curve = np.asarray(curve, dtype=float)
    center = float(np.median(curve))
    scale = 1.4826 * float(np.median(np.abs(curve - center)))
    fallback = "mad"
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(curve))
        fallback = "std"
    if not np.isfinite(scale) or scale < 1e-8:
        return np.zeros_like(curve), "constant"
    return (curve - center) / scale, fallback


def native_smoothed_response(backbone, source_curve, k):
    curve = np.asarray(source_curve, dtype=float)
    if backbone == "BoostingVRME":
        width = 2 * int(k)
        return np.convolve(curve, np.ones(width, dtype=float) / width, mode="same")
    return curve


def duration_set(k, length):
    base = 2 * int(k) + 1
    return tuple(sorted({min(length, max(1, int(round(base * factor))))
                         for factor in DURATION_FACTORS}))


def dense_candidates(curve, k):
    """Generate every legal start for three fixed duration scales."""
    z, fallback = robust_z(curve)
    prefix = np.r_[0.0, np.cumsum(z)]
    candidates = []
    for duration in duration_set(k, len(z)):
        starts = np.arange(len(z) - duration + 1, dtype=int)
        means = (prefix[starts + duration] - prefix[starts]) / duration
        sums = means * np.sqrt(duration)
        maxima = np.lib.stride_tricks.sliding_window_view(z, duration).max(axis=1)
        for start, mean, total, maximum in zip(starts, means, sums, maxima):
            candidates.append({
                "onset": int(start),
                "offset": int(start + duration - 1),
                "duration": int(duration),
                "mean_z": float(mean),
                "sum_z_sqrt_d": float(total),
                "max_z": float(maximum),
            })
    return candidates, fallback


def overlap(first, second):
    return not (first["offset"] < second["onset"] or second["offset"] < first["onset"])


def greedy_budget(candidates, score_name, budget):
    ranked = sorted(candidates,
                    key=lambda row: (-row[score_name], row["duration"], row["onset"]))
    selected = []
    for row in ranked:
        if all(not overlap(row, old) for old in selected):
            selected.append(row)
            if len(selected) == budget:
                break
    return sorted(selected, key=lambda row: (row["onset"], row["offset"]))


def budget_value(rule, native_count):
    if rule == "one_per_video":
        return 1
    if rule == "native_count":
        return native_count
    if rule == "native_or_one":
        return max(native_count, 1)
    return 2 * native_count + 1


def rescue_budget(candidates, native_predictions, score_name, budget):
    eligible = [row for row in candidates
                if all(not overlap(row, old) for old in native_predictions)]
    additions = greedy_budget(eligible, score_name, budget) if budget else []
    return sorted([*native_predictions, *additions],
                  key=lambda row: (row["onset"], row["offset"]))


def counts(predictions, gt):
    pairs = native.match(predictions, gt)
    tp = len(pairs)
    return (tp, len(predictions) - tp, len(gt) - tp), pairs


def f1(value):
    tp, fp, fn = value
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def add(first, second):
    return tuple(first[i] + second[i] for i in range(3))


def matching_adjacency(candidates, gt):
    return [[index for index, event in enumerate(gt)
             if native.iou((row["onset"], row["offset"]),
                           (event[0], event[2])) >= 0.5]
            for row in candidates]


def maximum_iou_matching(candidates, gt):
    adjacency = matching_adjacency(candidates, gt)
    owner = {}

    def augment(candidate_index, seen):
        for gt_index in adjacency[candidate_index]:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            if gt_index not in owner or augment(owner[gt_index], seen):
                owner[gt_index] = candidate_index
                return True
        return False

    for candidate_index, edges in enumerate(adjacency):
        if edges:
            augment(candidate_index, set())
    return len(owner)


def nonoverlap_oracle(candidates, gt):
    """Max matched ordered GTs using non-overlapping dense intervals."""
    if not gt:
        return 0
    ordered_gt = sorted(gt, key=lambda event: (event[0], event[2]))
    options = []
    for event in ordered_gt:
        valid = [(row["onset"], row["offset"]) for row in candidates
                 if native.iou((row["onset"], row["offset"]),
                               (event[0], event[2])) >= 0.5]
        options.append(tuple(sorted(set(valid))))

    @lru_cache(None)
    def solve(gt_index, previous_end):
        if gt_index == len(options):
            return 0
        best = solve(gt_index + 1, previous_end)
        for start, end in options[gt_index]:
            if start > previous_end:
                best = max(best, 1 + solve(gt_index + 1, end))
        return best

    return solve(0, -1)


def candidate_labels(candidates, gt):
    return np.asarray([
        any(native.iou((row["onset"], row["offset"]),
                       (event[0], event[2])) >= 0.5 for event in gt)
        for row in candidates
    ], dtype=np.uint8)


def safe_rank_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pmf = native.load_pmf()
    group_rows = []
    separation_rows = []
    budget_rows = []
    rescue_rows = []
    oracle_rows = []
    invalid_rows = []
    fallback_rows = []
    group_reports = []

    for backbone, dataset, cache_path, target_path in native.CASES:
        k, videos = native.group_audit(backbone, dataset, cache_path, target_path, pmf)
        native_total = (0, 0, 0)
        oracle_bipartite = 0
        oracle_nonoverlap = 0
        total_gt = 0
        score_labels = []
        score_values = {name: [] for name in ("max_z", "mean_z", "sum_z_sqrt_d")}
        budget_totals = {(score, rule): (0, 0, 0)
                         for score in score_values for rule in BUDGET_RULES}
        rescued_totals = {(score, rule): 0 for score in score_values for rule in BUDGET_RULES}
        prediction_totals = {(score, rule): 0 for score in score_values for rule in BUDGET_RULES}
        rescue_counts = {(score, rule): (0, 0, 0)
                         for score in score_values for rule in RESCUE_BUDGETS}
        rescue_added = {(score, rule): 0 for score in score_values for rule in RESCUE_BUDGETS}
        rescue_new_gt = {(score, rule): 0 for score in score_values for rule in RESCUE_BUDGETS}

        for subject, video, gt, native_predictions, _, _, source_curve, _, _ in videos:
            curve = native_smoothed_response(backbone, source_curve, k)
            native_counts, native_pairs = counts(native_predictions, gt)
            native_total = add(native_total, native_counts)
            native_matched = {gt_index for _, gt_index in native_pairs}
            total_gt += len(gt)
            candidates, fallback = dense_candidates(curve, k)
            fallback_rows.append({"backbone": backbone, "dataset": dataset,
                                  "subject": subject, "video": video,
                                  "normalization": fallback})

            for gt_index, event in enumerate(gt):
                if event[0] > event[2] or event[0] < 0 or event[2] >= len(curve):
                    invalid_rows.append({
                        "backbone": backbone, "dataset": dataset,
                        "subject": subject, "video": video, "gt_id": gt_index,
                        "onset": event[0], "apex": event[1], "offset": event[2],
                        "curve_length": len(curve),
                        "reason": "reversed" if event[0] > event[2] else "out_of_curve",
                    })

            labels = candidate_labels(candidates, gt)
            score_labels.append(labels)
            for score in score_values:
                score_values[score].append(np.asarray([row[score] for row in candidates]))

            oracle_bipartite += maximum_iou_matching(candidates, gt)
            oracle_nonoverlap += nonoverlap_oracle(candidates, gt)

            for score in score_values:
                for rule in BUDGET_RULES:
                    budget = budget_value(rule, len(native_predictions))
                    predictions = greedy_budget(candidates, score, budget) if budget else []
                    current_counts, current_pairs = counts(predictions, gt)
                    budget_totals[score, rule] = add(budget_totals[score, rule], current_counts)
                    prediction_totals[score, rule] += len(predictions)
                    selected_gt = {gt_index for _, gt_index in current_pairs}
                    rescued_totals[score, rule] += len(selected_gt - native_matched)
                for rule in RESCUE_BUDGETS:
                    budget = budget_value(rule, len(native_predictions))
                    predictions = rescue_budget(candidates, native_predictions, score, budget)
                    current_counts, current_pairs = counts(predictions, gt)
                    rescue_counts[score, rule] = add(rescue_counts[score, rule], current_counts)
                    rescue_added[score, rule] += len(predictions) - len(native_predictions)
                    selected_gt = {gt_index for _, gt_index in current_pairs}
                    rescue_new_gt[score, rule] += len(selected_gt - native_matched)

        exact = native_total == EXPECTED[backbone, dataset]
        labels = np.concatenate(score_labels)
        for score, parts in score_values.items():
            values = np.concatenate(parts)
            roc, ap = safe_rank_metrics(labels, values)
            separation_rows.append({
                "backbone": backbone, "dataset": dataset, "score": score,
                "candidate_count": len(labels), "positive_interval_count": int(labels.sum()),
                "positive_rate": float(labels.mean()), "ROC_AUC": roc, "PR_AUC": ap,
                "label": "interval IoU>=0.5 with any GT; post-hoc evaluation only",
            })
            for rule in BUDGET_RULES:
                value = budget_totals[score, rule]
                budget_rows.append({
                    "backbone": backbone, "dataset": dataset, "score": score,
                    "budget_rule": rule, "TP": value[0], "FP": value[1], "FN": value[2],
                    "F1": f1(value), "delta_vs_native": f1(value) - f1(native_total),
                    "prediction_count": prediction_totals[score, rule],
                    "native_generation_miss_or_other_GT_recovered": rescued_totals[score, rule],
                    "interpretation": "rank-budget diagnostic; not deployable threshold selection",
                })
            for rule in RESCUE_BUDGETS:
                value = rescue_counts[score, rule]
                added = rescue_added[score, rule]
                recovered = rescue_new_gt[score, rule]
                rescue_rows.append({
                    "backbone": backbone, "dataset": dataset, "score": score,
                    "rescue_budget_rule": rule, "TP": value[0], "FP": value[1],
                    "FN": value[2], "F1": f1(value),
                    "delta_vs_native": f1(value) - f1(native_total),
                    "added_predictions": added, "newly_matched_GT": recovered,
                    "added_FP_per_new_GT": ((added - recovered) / recovered
                                             if recovered else None),
                    "interpretation": "Native-preserving rank-budget diagnostic; no tuned threshold",
                })

        oracle_rows.append({
            "backbone": backbone, "dataset": dataset, "k": k,
            "durations": ";".join(map(str, duration_set(k, max(len(v[6]) for v in videos)))),
            "GT": total_gt, "bipartite_oracle_TP": oracle_bipartite,
            "bipartite_recall_ceiling": oracle_bipartite / total_gt,
            "nonoverlap_oracle_TP": oracle_nonoverlap,
            "nonoverlap_recall_ceiling": oracle_nonoverlap / total_gt,
            "oracle_uses_GT": True,
        })
        group_rows.append({
            "backbone": backbone, "dataset": dataset, "k": k,
            "native_TP": native_total[0], "native_FP": native_total[1],
            "native_FN": native_total[2], "native_F1": f1(native_total),
            "expected_TP_FP_FN": "/".join(map(str, EXPECTED[backbone, dataset])),
            "native_exact_replay": exact, "video_count": len(videos),
        })
        group_reports.append({"backbone": backbone, "dataset": dataset,
                              "native_exact_replay": exact})

    write_csv(OUT / "group_summary.csv", group_rows)
    write_csv(OUT / "full_pool_separation.csv", separation_rows)
    write_csv(OUT / "rank_budget_diagnostic.csv", budget_rows)
    write_csv(OUT / "native_preserving_rescue_diagnostic.csv", rescue_rows)
    write_csv(OUT / "formal_iou_geometry_oracle.csv", oracle_rows)
    write_csv(OUT / "invalid_gt_audit.csv", invalid_rows)
    write_csv(OUT / "normalization_fallbacks.csv", fallback_rows)

    provenance = {
        "candidate_generator_reads_GT": False,
        "candidate_space": "all legal starts at three fixed duration scales",
        "duration_factors": DURATION_FACTORS,
        "base_duration": "2*k+1 from frozen/native metadata",
        "scores": {
            "max_z": "single-height control: max robust z within interval",
            "mean_z": "mean robust z within interval",
            "sum_z_sqrt_d": "signed accumulated robust z divided by sqrt(duration)",
        },
        "normalization": "per-video median/MAD; std then zeros only for degenerate scale",
        "response": "historical Native 2*k smoothed temporal response for both backbones",
        "GT_use": "candidate labels, formal IoU oracle, and post-hoc metrics only",
        "rank_budget_warning": "diagnostic anchored to Native prediction count; not a deployable decoder",
        "rescue_warning": "Native-preserving diagnostic with a fixed per-video count budget; not a deployable threshold",
        "input_hashes": {str(path): file_sha256(path)
                         for _, _, cache, target in native.CASES
                         for path in (cache, target)},
    }
    (OUT / "protocol_and_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2))

    exact = all(row["native_exact_replay"] for row in group_reports)
    report = {
        "status": "DENSE_SEGMENT_PHASE0_V2_NATIVE_REPLAY_EXACT" if exact
                  else "DENSE_SEGMENT_PHASE0_V2_NATIVE_REPLAY_MISMATCH",
        "decision": "NOT_PRECOMMITTED_UNTIL_RESULTS_ARE_INTERPRETED",
        "phase": "opportunity_audit_only",
        "trainable_model_parameters": 0,
        "semi_markov_decoder_run": False,
        "nested_LOSO_run": False,
        "groups": group_rows,
        "geometry_oracle": oracle_rows,
        "full_pool_separation": separation_rows,
        "rank_budget_diagnostic": budget_rows,
        "native_preserving_rescue_diagnostic": rescue_rows,
        "invalid_gt_records": len(invalid_rows),
        "files": {path.name: str(path) for path in OUT.iterdir() if path.is_file()},
    }
    (OUT / "combined_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
