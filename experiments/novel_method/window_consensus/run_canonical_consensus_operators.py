#!/usr/bin/env python3
"""Paired nested SAMMLV audit of fixed canonical two-view operators."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRED_SOURCE = HERE / "run_paired_nested_uniform_mean.py"
DEFAULT_RAW = Path(
    os.environ.get("SAMMLV_RAW_ROOT", "data/raw_prestitch_sammlv_full")
)
DEFAULT_OUT = ROOT / "results/canonical_consensus_operators_sammlv"
EXPECTED_B0 = (49, 143, 110)
EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79
EPS = 1e-8
METHODS = (
    "O0 Original",
    "O1 Arithmetic Mean",
    "O3 Geometric Mean",
    "O4 Harmonic Mean",
    "O5 Minimum Consensus",
)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = import_module("canonical_paired_native", PAIRED_SOURCE)
base = paired.base
native = paired.native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_operators(raw: np.ndarray, global_indices: np.ndarray, length: int):
    raw = np.asarray(raw, dtype=np.float64)
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if raw.shape != global_indices.shape or raw.ndim != 2 or raw.shape[1] != 30:
        raise RuntimeError(f"raw/global shape mismatch: {raw.shape}/{global_indices.shape}")
    if not np.isfinite(raw).all() or float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
        raise RuntimeError("raw spotting scores are not finite sigmoid outputs in [0,1]")
    flat_index = global_indices.reshape(-1)
    flat_score = raw.reshape(-1)
    if flat_index.min(initial=0) < 0 or flat_index.max(initial=-1) >= length:
        raise RuntimeError("global index outside video")
    count = np.bincount(flat_index, minlength=length).astype(np.int64)
    if np.any(count < 1) or np.any(count > 2):
        raise RuntimeError(f"unsupported overlap counts: {np.unique(count).tolist()}")
    score_sum = np.bincount(flat_index, weights=flat_score, minlength=length)
    score_product = np.ones(length, dtype=np.float64)
    np.multiply.at(score_product, flat_index, flat_score)
    score_minimum = np.full(length, np.inf, dtype=np.float64)
    np.minimum.at(score_minimum, flat_index, flat_score)
    pair = count == 2

    arithmetic = score_sum / count
    geometric = score_sum.copy()
    geometric[pair] = np.sqrt(score_product[pair])
    harmonic = score_sum.copy()
    harmonic[pair] = 2.0 * score_product[pair] / (score_sum[pair] + EPS)
    minimum = score_minimum
    if not all(np.isfinite(curve).all() and len(curve) == length for curve in (arithmetic, geometric, harmonic, minimum)):
        raise RuntimeError("canonical operator produced invalid curve")
    # On single-view frames every operator must equal the sole raw score exactly.
    single = count == 1
    for curve in (arithmetic, geometric, harmonic, minimum):
        if not np.array_equal(curve[single], score_sum[single]):
            raise RuntimeError("single-view fallback mismatch")
    return {
        METHODS[1]: arithmetic,
        METHODS[2]: geometric,
        METHODS[3]: harmonic,
        METHODS[4]: minimum,
    }, count


def make_records(gated: dict):
    by_method = {method: [] for method in METHODS}
    diagnostics = []
    raw_minimum = np.inf
    raw_maximum = -np.inf
    for identity, item in sorted(gated["loaded"].items()):
        subject, video = identity
        original = dict(item["record"])
        original["score"] = item["original"]
        by_method[METHODS[0]].append(original)
        curves, count = aggregate_operators(item["raw"], item["global_indices"], len(item["original"]))
        arithmetic_reference, _center, reference_count = base.aggregate_raw(
            {
                "raw_window_score": item["raw"],
                "global_indices": item["global_indices"],
                "original_fresh_result_video": item["original"],
            }
        )
        if not np.array_equal(curves[METHODS[1]], arithmetic_reference) or not np.array_equal(count, reference_count):
            raise RuntimeError(f"O1 reference mismatch: {subject}/{video}")
        for method, curve in curves.items():
            record = dict(item["record"])
            record["score"] = curve
            by_method[method].append(record)
        raw_minimum = min(raw_minimum, float(np.min(item["raw"])))
        raw_maximum = max(raw_maximum, float(np.max(item["raw"])))
        diagnostics.append(
            {
                "subject": subject,
                "video": video,
                "frames": len(item["original"]),
                "single_view_frames": int(np.sum(count == 1)),
                "two_view_frames": int(np.sum(count == 2)),
                "maximum_overlap": int(np.max(count)),
                "raw_score_min": float(np.min(item["raw"])),
                "raw_score_max": float(np.max(item["raw"])),
                "O1_reference_array_equal": True,
            }
        )
    identity = lambda row: (str(row["subject"]), str(row["video"]))
    reference = [identity(row) for row in by_method[METHODS[0]]]
    if len(reference) != EXPECTED_VIDEOS or len(set(reference)) != EXPECTED_VIDEOS:
        raise RuntimeError("canonical record identity/cardinality mismatch")
    for method in METHODS[1:]:
        if [identity(row) for row in by_method[method]] != reference:
            raise RuntimeError(f"paired record order mismatch: {method}")
    return by_method, diagnostics, {"minimum": raw_minimum, "maximum": raw_maximum}


def parameter_frequency(selected_rows: list[dict]) -> tuple[list[dict], dict]:
    possible = {"c_s": paired.TUNED_CS, "p": paired.TUNED_P, "c_d": paired.TUNED_CD, "c_b": paired.TUNED_CB}
    rows = []
    summary = {}
    for method in METHODS:
        subset = [row for row in selected_rows if row["method"] == method]
        summary[method] = {}
        for parameter, values in possible.items():
            counts = Counter(str(row[parameter]) for row in subset)
            summary[method][parameter] = dict(sorted(counts.items()))
            for value in values:
                rows.append(
                    {
                        "method": method,
                        "parameter": parameter,
                        "value": value,
                        "count": int(counts.get(str(value), 0)),
                        "proportion": counts.get(str(value), 0) / EXPECTED_SUBJECTS,
                    }
                )
    return rows, summary


def report_markdown(summary: dict) -> str:
    baseline = summary["aggregate"][METHODS[0]]
    best = summary["best_non_O0_operator"]
    bootstrap = summary["best_vs_O0_bootstrap"]
    lines = [
        "# Canonical Window-Consensus Operators — Paired Nested SAMMLV",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "本实验只比较预定义 O0/O1/O3/O4/O5 score aggregation；每个分支独立获得完全相同的 81-config Native nested search budget。未使用 classifier、recognition、W2 verifier、weighting、rescue 或任何 aggregation 参数。",
        "",
        "## Provenance and gates",
        "",
        f"- Raw manifest SHA-256：`{summary['provenance']['raw_manifest_sha256']}`。",
        f"- Raw score range：[{summary['raw_score_range']['minimum']:.9f}, {summary['raw_score_range']['maximum']:.9f}]，确认全部为 `[0,1]`，O3 使用 `sqrt(s1*s2)`。",
        "- 79/79 O1 curves 与上一轮 verified Arithmetic Mean array-equal。",
        f"- O0：{baseline['TP']}/{baseline['FP']}/{baseline['FN']}，F1={baseline['F1']:.6f}；29/29 历史 config index/value exact PASS。",
        "- Single-view frames 对 O1/O3/O4/O5 均 exact fallback 到唯一 raw score。",
        "",
        "## Paired nested outer results",
        "",
        "| Operator | TP | FP | FN | Precision | Recall | F1 | ΔF1 vs O0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["aggregate"][method]
        lines.append(
            f"| {method} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Precision']:.6f} | "
            f"{row['Recall']:.6f} | {row['F1']:.6f} | {row['F1']-baseline['F1']:+.6f} |"
        )
    lines += ["", "## Selected parameter frequencies", ""]
    for method in METHODS:
        lines.append(f"### {method}")
        lines.append("")
        for parameter, frequency in summary["selected_parameter_frequencies"][method].items():
            lines.append(f"- `{parameter}`: `{json.dumps(frequency, sort_keys=True)}`")
        lines.append("")
    lines += [
        "## Subject stability and leave-one-subject sensitivity",
        "",
        "| Operator vs O0 | Improved | Equal | Worse | Leave-one ΔF1 min | median | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS[1:]:
        stable = summary["subject_stability"][method]
        sensitivity = summary["leave_one_subject_delta_F1"][method]
        lines.append(
            f"| {method} | {stable['improved']} | {stable['equal']} | {stable['worse']} | "
            f"{sensitivity['min']:+.6f} | {sensitivity['median']:+.6f} | {sensitivity['max']:+.6f} |"
        )
    lines += [
        "",
        "## Best non-O0 bootstrap",
        "",
        f"Best non-O0 operator：**{best}**。",
        f"1000 paired subject bootstrap：observed ΔF1={bootstrap['observed_delta_F1']:+.6f}，mean={bootstrap['bootstrap_mean_delta_F1']:+.6f}，95% CI=[{bootstrap['CI95'][0]:+.6f}, {bootstrap['CI95'][1]:+.6f}]。",
        "",
        "## Decision gates",
        "",
    ]
    for name, passed in summary["decision_gates"].items():
        lines.append(f"- {name}: `{passed}`")
    lines += [
        "",
        "“Not from a single subject” 在运行前固定为 leave-one-subject minimum ΔF1 > 0。GO 需要最佳 operator 同时通过全部 gate；否则在本任务允许的两个状态中判为 NO-GO。",
        "",
        "实验到此停止；未调 power mean、alpha、lambda 或任何其它 aggregation 参数。",
    ]
    return "\n".join(lines) + "\n"


def blocked_report(output_root: Path, reason: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "CANONICAL_CONSENSUS_OPERATORS_PAIRED_NESTED_SAMMLV_CN.md").write_text(
        "# Canonical Window-Consensus Operators — Paired Nested SAMMLV\n\n"
        "`BLOCKED-CANONICAL-CONSENSUS-O0-ANCHOR`\n\n"
        f"{reason}\n\nO1/O3/O4/O5 were not run.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    configs = paired.grid()
    try:
        gated = base.provenance_gate(argparse.Namespace(raw_dir=args.raw_dir, output_root=args.output_root))
        if len(gated["subjects"]) != EXPECTED_SUBJECTS or len(gated["loaded"]) != EXPECTED_VIDEOS:
            raise RuntimeError("raw source cardinality mismatch")
        records, curve_diagnostics, raw_range = make_records(gated)
        b0_bank = paired.precompute(records[METHODS[0]], configs, list(gated["subjects"]))
        results = {METHODS[0]: paired.nested_select(METHODS[0], list(gated["subjects"]), configs, b0_bank)}
        paired.historical_b0_gate(results[METHODS[0]], configs)
        b0_counts = tuple(results[METHODS[0]]["aggregate"][key] for key in ("TP", "FP", "FN"))
        if b0_counts != EXPECTED_B0:
            raise RuntimeError(f"O0 anchor={b0_counts}, expected={EXPECTED_B0}")
    except Exception as error:
        blocked_report(args.output_root, f"{type(error).__name__}: {error}")
        print("BLOCKED-CANONICAL-CONSENSUS-O0-ANCHOR")
        print(f"{type(error).__name__}: {error}")
        return 2

    # Non-O0 operators are evaluated only after the full O0 provenance/fold gate.
    for method in METHODS[1:]:
        bank = paired.precompute(records[method], configs, list(gated["subjects"]))
        results[method] = paired.nested_select(method, list(gated["subjects"]), configs, bank)

    baseline = results[METHODS[0]]["aggregate"]
    aggregate = {method: results[method]["aggregate"] for method in METHODS}
    stability = {}
    sensitivity = {}
    for method in METHODS[1:]:
        stability[method], sensitivity[method] = paired.stability_and_sensitivity(
            results[METHODS[0]]["subject_counts"], results[method]["subject_counts"], list(gated["subjects"])
        )
    best = max(
        METHODS[1:],
        key=lambda method: (
            aggregate[method]["F1"],
            -aggregate[method]["FP"],
            -METHODS.index(method),
        ),
    )
    bootstrap = paired.paired_bootstrap(
        results[METHODS[0]]["subject_counts"],
        results[best]["subject_counts"],
        list(gated["subjects"]),
        f"{best} vs {METHODS[0]}",
    )
    selected_rows = [row for method in METHODS for row in results[method]["selected_rows"]]
    frequency_rows, frequency_summary = parameter_frequency(selected_rows)
    best_sensitivity = sensitivity[best]
    gates = {
        "best_nested_F1_gt_O0": aggregate[best]["F1"] > baseline["F1"],
        "bootstrap_mean_delta_gt_0": bootstrap["bootstrap_mean_delta_F1"] > 0,
        "leave_one_subject_median_delta_gt_0": best_sensitivity["median"] > 0,
        "improvement_not_from_single_subject_leave_one_min_gt_0": best_sensitivity["min"] > 0,
    }
    decision = (
        "CANONICAL-CONSENSUS-OPERATOR-GO"
        if all(gates.values())
        else "CANONICAL-CONSENSUS-OPERATOR-NO-GO"
    )
    summary_rows = []
    for method in METHODS:
        current = aggregate[method]
        summary_rows.append(
            {
                "method": method,
                **current,
                "Delta_TP_vs_O0": current["TP"] - baseline["TP"],
                "Delta_FP_vs_O0": current["FP"] - baseline["FP"],
                "Delta_FN_vs_O0": current["FN"] - baseline["FN"],
                "Delta_F1_vs_O0": current["F1"] - baseline["F1"],
                "is_best_non_O0": method == best,
            }
        )
    outer_rows = [row for method in METHODS for row in results[method]["outer_rows"]]
    lookup = {(row["method"], row["outer_subject"]): row for row in outer_rows}
    for subject in gated["subjects"]:
        b0 = lookup[(METHODS[0], subject)]
        for method in METHODS:
            row = lookup[(method, subject)]
            row["Delta_F1_vs_O0"] = row["F1"] - b0["F1"]
            row["comparison_vs_O0"] = (
                "baseline"
                if method == METHODS[0]
                else paired.comparison_label(results[method]["subject_counts"][subject], results[METHODS[0]]["subject_counts"][subject])
            )
    summary = {
        "experiment": "canonical_consensus_operators_paired_nested_sammlv",
        "decision": decision,
        "provenance": {
            "raw_manifest": str(gated["manifest_path"]),
            "raw_manifest_sha256": gated["manifest_sha"],
            "compact_cache_sha256": base.EXPECTED_CACHE_SHA,
            "frozen_config_report_sha256": gated["config_sha"],
            "subjects": EXPECTED_SUBJECTS,
            "videos": EXPECTED_VIDEOS,
            "k_p": base.K_P,
        },
        "raw_score_range": raw_range,
        "gates": {
            "raw_scores_all_in_0_1": True,
            "O1_matches_verified_uniform_mean_79_of_79": True,
            "single_view_exact_fallback_all_operators": True,
            "O0_anchor_exact": True,
            "O0_historical_config_indices_and_values_29_of_29": True,
        },
        "protocol": {
            "grid_count_per_operator": len(configs),
            "same_grid_and_search_budget": True,
            "selection": "outer-train 28 subjects pooled Spotting F1",
            "tie_break": ["higher exact Fraction F1", "fewer FP", "fixed Cartesian config index"],
            "outer_test_GT_used_for_selection": False,
            "operator_selection_for_bootstrap": "highest nested pooled outer F1; then fewer FP; then fixed method order",
            "not_single_subject_definition": "leave-one-subject minimum Delta F1 > 0",
            "bootstrap_repetitions": paired.BOOTSTRAP_REPEATS,
            "bootstrap_seed": paired.BOOTSTRAP_SEED,
        },
        "operator_definitions": {
            METHODS[0]: "saved original_fresh_result_video",
            METHODS[1]: "(s1+s2)/2",
            METHODS[2]: "sqrt(s1*s2)",
            METHODS[3]: "2*s1*s2/(s1+s2+1e-8)",
            METHODS[4]: "min(s1,s2)",
            "single_view": "sole raw score for O1/O3/O4/O5",
        },
        "aggregate": aggregate,
        "selected_parameter_frequencies": frequency_summary,
        "subject_stability": stability,
        "leave_one_subject_delta_F1": sensitivity,
        "best_non_O0_operator": best,
        "best_vs_O0_bootstrap": bootstrap,
        "decision_gates": gates,
        "integrity": {
            "weighted_mean_used": False,
            "aggregation_parameter_tuned": False,
            "classifier_used": False,
            "recognition_used": False,
            "morphology_used": False,
            "W2_verifier_used": False,
            "pruning_used": False,
            "rescue_used": False,
            "native_grid_expanded": False,
            "backbone_forward": False,
        },
    }
    output_dir = args.output_root / "outputs"
    write_csv(output_dir / "canonical_operator_summary.csv", summary_rows)
    write_csv(output_dir / "canonical_operator_outer_metrics.csv", outer_rows)
    write_csv(output_dir / "canonical_operator_selected_configs.csv", selected_rows)
    write_csv(output_dir / "canonical_operator_parameter_frequency.csv", frequency_rows)
    write_csv(output_dir / "canonical_operator_inner_scores.csv", [row for method in METHODS for row in results[method]["inner_rows"]])
    write_csv(output_dir / "canonical_operator_curve_diagnostics.csv", curve_diagnostics)
    (output_dir / "canonical_operator_bootstrap.json").write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "canonical_operator_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "CANONICAL_CONSENSUS_OPERATORS_PAIRED_NESTED_SAMMLV_CN.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "aggregate": aggregate, "best": best, "bootstrap": bootstrap, "decision_gates": gates}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
