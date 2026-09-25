#!/usr/bin/env python3
"""Independent replay and structural audit for sparse-event Lasso V1 outputs."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import run_boundary_phase0_audit as native
import run_dense_segment_phase0_v2 as dense


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/sparse_event_lasso_v1"


def interval_iou(first, second):
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = first[1] - first[0] + second[1] - second[0] + 2 - intersection
    return intersection / union if union else 0.0


def main():
    predictions = defaultdict(list)
    with (RESULTS / "selected_predictions.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["backbone"], row["dataset"], row["subject"], row["video"], row["method"])
            predictions[key].append({"onset": int(row["onset"]), "offset": int(row["offset"])})

    expected = {}
    structural = defaultdict(lambda: {"events": 0, "overlap_pairs_iou_ge_0_5": 0})
    pmf = native.load_pmf()
    for backbone, dataset, cache_path, target_path in native.CASES:
        _, videos = native.group_audit(backbone, dataset, cache_path, target_path, pmf)
        totals = defaultdict(lambda: (0, 0, 0))
        for subject, video, gt, _, _, _, _, _, _ in videos:
            for method in ("Native", "LassoFixed", "LassoCompact", "MatchedEqualCount"):
                rows = predictions[(backbone, dataset, str(subject), str(video), method)]
                value, _ = dense.counts(rows, gt)
                totals[method] = tuple(totals[method][i] + value[i] for i in range(3))
                intervals = [(row["onset"], row["offset"]) for row in rows]
                item = structural[(backbone, dataset, method)]
                item["events"] += len(intervals)
                item["overlap_pairs_iou_ge_0_5"] += sum(
                    interval_iou(intervals[i], intervals[j]) >= 0.5
                    for i in range(len(intervals)) for j in range(i + 1, len(intervals))
                )
        expected[(backbone, dataset)] = dict(totals)

    with (RESULTS / "summary.csv").open(encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    summary_match = all(
        expected[(row["backbone"], row["dataset"])][row["method"]]
        == (int(row["TP"]), int(row["FP"]), int(row["FN"]))
        for row in summary
    )

    with (RESULTS / "outer_loso_selections.csv").open(encoding="utf-8-sig") as handle:
        selections = list(csv.DictReader(handle))
    with (RESULTS / "solver_audit.csv").open(encoding="utf-8-sig") as handle:
        solver = list(csv.DictReader(handle))
    ratio_one_nonempty = sum(int(row["active_atoms"]) > 0 for row in solver if float(row["ratio"]) == 1.0)
    warnings = sum(int(row["convergence_warning"]) for row in solver)
    boundary = sum(int(row["ratio_at_lower_boundary"]) + int(row["ratio_at_upper_boundary"])
                   for row in selections)
    report = {
        "summary_counts_recomputed_exactly": summary_match,
        "native_expected_exactly": all(
            expected[key]["Native"] == dense.EXPECTED[key] for key in expected
        ),
        "selection_rows": len(selections),
        "selected_ratio_frequency": dict(Counter(row["ratio"] for row in selections)),
        "boundary_selection_count": boundary,
        "solver_warning_count": warnings,
        "ratio_one_nonempty_solution_count": ratio_one_nonempty,
        "structural_counts": [
            {"backbone": key[0], "dataset": key[1], "method": key[2], **value}
            for key, value in sorted(structural.items())
        ],
    }
    (RESULTS / "independent_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
