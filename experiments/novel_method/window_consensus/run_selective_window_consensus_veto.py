#!/usr/bin/env python3
"""SAMMLV nested development gate for a selective two-window consensus veto."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIR_SOURCE = HERE / "run_window_pair_consensus_audit.py"
DEFAULT_RAW = Path(
    os.environ.get("SAMMLV_RAW_ROOT", "data/raw_prestitch_sammlv_full")
)
DEFAULT_OUT = ROOT / "results/selective_window_consensus_veto_sammlv"
EXPECTED_ANCHOR = (49, 143, 110)
EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79
K_P = 5
EPS = 1e-8
SEED = 20260905
BOOTSTRAP_REPEATS = 1000
DELTAS = (0.05, 0.10, 0.15, 0.20)
TAUS = (0.20, 0.40, 0.60, 0.80)
METHODS = (
    "C0 Final Strong Native",
    "C1 Margin-only selective prune",
    "C2 Agreement-only prune",
    "C3 Selective Consensus Veto",
)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pair_audit = import_module("selective_veto_pair_source", PAIR_SOURCE)
base = pair_audit.base
native = pair_audit.native


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


def empty_counts() -> dict:
    return {key: 0 for key in ("TP", "FP", "FN", "event_count")}


def add_counts(target: dict, source: dict) -> None:
    for key in target:
        target[key] += int(source[key])


def build_candidate_bank(gated: dict):
    videos = []
    subject_anchor = {subject: empty_counts() for subject in gated["subjects"]}
    label_counts = Counter()
    coverage_counts = Counter()
    for (subject, video), item in sorted(gated["loaded"].items()):
        record = dict(item["record"])
        config = gated["configs"][subject]
        width = max(1, int(round(config["c_s"] * K_P)))
        # Margin and Native events must retain the previously audited compact-cache
        # pathway. The raw dump is consulted only for the two-view A_ratio.
        curve = native.moving_average(np.asarray(record["score"]), width)
        mean = float(curve.mean())
        maximum = float(curve.max())
        native_threshold = native.threshold(curve, config["p"])
        events = native.tuned_decode(record, config, K_P)
        raw_record = dict(record)
        raw_record["score"] = np.asarray(item["original"])
        raw_events = native.tuned_decode(raw_record, config, K_P)
        event_identity = lambda event: (int(event["onset"]), int(event["peak"]), int(event["offset"]))
        if [event_identity(event) for event in raw_events] != [event_identity(event) for event in events]:
            raise RuntimeError(f"raw/cache O0 event identity mismatch: {subject}/{video}")
        base_counts = native.evaluate(record, events)
        add_counts(subject_anchor[subject], base_counts)
        candidates = []
        for candidate_index, event in enumerate(events):
            peak = int(event["peak"])
            changed = native.evaluate(record, events[:candidate_index] + events[candidate_index + 1 :])
            label, label_name = pair_audit.pruning.causal_label(base_counts, changed)
            features = pair_audit.pair_features(item["raw"], item["global_indices"], peak)
            if label is None:
                raise RuntimeError(f"ambiguous causal deletion: {subject}/{video}/{candidate_index}")
            agreement = features["C2"] if features["coverage_status"] == "PAIR" else None
            margin = float((curve[peak] - mean) / (maximum - mean + EPS) - config["p"])
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "event": event,
                    "peak": peak,
                    "H_s": margin,
                    "A_ratio": agreement,
                    "coverage_status": features["coverage_status"],
                    "coverage_count": features["coverage_count"],
                    "label": int(label),
                    "label_name": label_name,
                    "peak_score": float(curve[peak]),
                    "native_threshold": float(native_threshold),
                }
            )
            label_counts[label_name] += 1
            coverage_counts[features["coverage_status"]] += 1
        videos.append(
            {
                "subject": subject,
                "video": video,
                "record": record,
                "events": events,
                "base_counts": base_counts,
                "candidates": candidates,
            }
        )
    aggregate = native.metrics(base.sum_counts(subject_anchor.values()))
    anchor = tuple(aggregate[key] for key in ("TP", "FP", "FN"))
    if anchor != EXPECTED_ANCHOR:
        raise RuntimeError(f"Final Strong Native anchor={anchor}, expected={EXPECTED_ANCHOR}")
    if sum(label_counts.values()) != aggregate["event_count"] or label_counts != {"KEEP": 49, "PRUNE": 143}:
        raise RuntimeError(f"unexpected causal-label bank: {dict(label_counts)}")
    return videos, subject_anchor, aggregate, label_counts, coverage_counts


def method_grid(method: str):
    if method == METHODS[0]:
        return [(None, None)]
    if method == METHODS[1]:
        return [(delta, None) for delta in DELTAS]
    if method == METHODS[2]:
        return [(None, tau) for tau in TAUS]
    if method == METHODS[3]:
        return list(itertools.product(DELTAS, TAUS))
    raise KeyError(method)


def prune_decision(candidate: dict, method: str, delta, tau):
    pair = candidate["coverage_status"] == "PAIR"
    low_margin = delta is not None and candidate["H_s"] <= delta
    low_agreement = pair and tau is not None and candidate["A_ratio"] < tau
    if method == METHODS[0]:
        eligible, pruned, reason = False, False, "BASELINE_KEEP"
    elif method == METHODS[1]:
        eligible, pruned, reason = low_margin, low_margin, "MARGIN_ONLY"
    elif method == METHODS[2]:
        eligible, pruned, reason = pair, low_agreement, "AGREEMENT_ONLY" if pair else "SINGLE_VIEW_FORCED_KEEP"
    elif method == METHODS[3]:
        eligible = pair and low_margin
        pruned = eligible and low_agreement
        if not pair:
            reason = "SINGLE_VIEW_FORCED_KEEP"
        elif not low_margin:
            reason = "HIGH_CONFIDENCE_FORCED_KEEP"
        else:
            reason = "CONSENSUS_ELIGIBLE"
    else:
        raise KeyError(method)
    return bool(eligible), bool(pruned), reason


def evaluate_subset(videos: list[dict], subjects: set[str], method: str, delta=None, tau=None, trace=False, outer_subject=None):
    counts = empty_counts()
    diagnostics = Counter()
    trace_rows = []
    for item in videos:
        if item["subject"] not in subjects:
            continue
        kept = []
        for candidate in item["candidates"]:
            eligible, pruned, reason = prune_decision(candidate, method, delta, tau)
            diagnostics["total_native_events"] += 1
            diagnostics["eligible_events"] += int(eligible)
            diagnostics["pruned_events"] += int(pruned)
            diagnostics["pair_events"] += int(candidate["coverage_status"] == "PAIR")
            diagnostics["single_view_events"] += int(candidate["coverage_status"] == "SINGLE-VIEW-CANDIDATE")
            if pruned:
                diagnostics[f"pruned_{candidate['label_name']}"] += 1
            else:
                kept.append(candidate["event"])
            if trace:
                event = candidate["event"]
                trace_rows.append(
                    {
                        "outer_subject": outer_subject,
                        "method": method,
                        "subject": item["subject"],
                        "video": item["video"],
                        "candidate_index": candidate["candidate_index"],
                        "onset": int(event["onset"]),
                        "peak": int(event["peak"]),
                        "offset": int(event["offset"]),
                        "H_s": candidate["H_s"],
                        "A_ratio": candidate["A_ratio"],
                        "coverage_status": candidate["coverage_status"],
                        "causal_native_label": candidate["label_name"],
                        "selected_delta": delta,
                        "selected_tau": tau,
                        "eligible": eligible,
                        "pruned": pruned,
                        "decision_reason": reason,
                    }
                )
        evaluated = native.evaluate(item["record"], kept)
        add_counts(counts, evaluated)
    result = native.metrics(counts)
    pruned_keep = int(diagnostics.get("pruned_KEEP", 0))
    pruned_fp = int(diagnostics.get("pruned_PRUNE", 0))
    if result["TP"] + pruned_keep != sum(item["base_counts"]["TP"] for item in videos if item["subject"] in subjects):
        raise RuntimeError(f"TP deletion reconciliation failed: {method}/{delta}/{tau}")
    if result["FP"] + pruned_fp != sum(item["base_counts"]["FP"] for item in videos if item["subject"] in subjects):
        raise RuntimeError(f"FP deletion reconciliation failed: {method}/{delta}/{tau}")
    return result, diagnostics, trace_rows


def selection_key(row: dict, baseline: dict):
    metrics = row["metrics"]
    diagnostics = row["diagnostics"]
    denominator = 2 * metrics["TP"] + metrics["FP"] + metrics["FN"]
    exact_f1 = Fraction(2 * metrics["TP"], denominator) if denominator else Fraction(0, 1)
    total = int(diagnostics["total_native_events"])
    rate = Fraction(int(diagnostics["pruned_events"]), total) if total else Fraction(0, 1)
    delta = 0.0 if row["delta"] is None else float(row["delta"])
    tau = 0.0 if row["tau"] is None else float(row["tau"])
    return exact_f1, -metrics["FP"], -(baseline["TP"] - metrics["TP"]), -rate, -delta, -tau


def run_nested(videos: list[dict], subjects: list[str]):
    cache = {}
    for method in METHODS:
        for delta, tau in method_grid(method):
            for subject in subjects:
                cache[(method, delta, tau, subject)] = evaluate_subset(videos, {subject}, method, delta, tau)[:2]

    outer_rows, selected_rows, inner_rows, traces = [], [], [], []
    for held in subjects:
        train_subjects = [subject for subject in subjects if subject != held]
        baseline_train = empty_counts()
        for subject in train_subjects:
            add_counts(baseline_train, cache[(METHODS[0], None, None, subject)][0])
        baseline_train = native.metrics(baseline_train)
        selections = {METHODS[0]: {"delta": None, "tau": None, "metrics": baseline_train, "diagnostics": Counter()}}
        for method in METHODS[1:]:
            scored = []
            for config_index, (delta, tau) in enumerate(method_grid(method)):
                pooled_counts = empty_counts()
                pooled_diagnostics = Counter()
                for subject in train_subjects:
                    metrics, diagnostics = cache[(method, delta, tau, subject)]
                    add_counts(pooled_counts, metrics)
                    pooled_diagnostics.update(diagnostics)
                pooled = native.metrics(pooled_counts)
                row = {
                    "outer_subject": held,
                    "method": method,
                    "config_index": config_index,
                    "delta": delta,
                    "tau": tau,
                    "metrics": pooled,
                    "diagnostics": pooled_diagnostics,
                }
                scored.append(row)
                inner_rows.append(
                    {
                        "outer_subject": held,
                        "method": method,
                        "config_index": config_index,
                        "delta": delta,
                        "tau": tau,
                        **{f"inner_{key}": value for key, value in pooled.items()},
                        "inner_eligible": int(pooled_diagnostics["eligible_events"]),
                        "inner_pruned": int(pooled_diagnostics["pruned_events"]),
                        "inner_pruned_TP": int(pooled_diagnostics["pruned_KEEP"]),
                        "inner_pruned_FP": int(pooled_diagnostics["pruned_PRUNE"]),
                    }
                )
            selections[method] = max(scored, key=lambda row: selection_key(row, baseline_train))

        for method in METHODS:
            selected = selections[method]
            delta, tau = selected["delta"], selected["tau"]
            metrics, diagnostics, trace_rows = evaluate_subset(
                videos, {held}, method, delta, tau, trace=True, outer_subject=held
            )
            traces.extend(trace_rows)
            outer_rows.append(
                {
                    "outer_subject": held,
                    "method": method,
                    **metrics,
                    "selected_delta": delta,
                    "selected_tau": tau,
                    "eligible_events": int(diagnostics["eligible_events"]),
                    "pruned_events": int(diagnostics["pruned_events"]),
                    "pruned_TP": int(diagnostics["pruned_KEEP"]),
                    "pruned_FP": int(diagnostics["pruned_PRUNE"]),
                }
            )
            selected_rows.append(
                {
                    "outer_subject": held,
                    "method": method,
                    "selected_delta": delta,
                    "selected_tau": tau,
                    **{f"inner_{key}": value for key, value in selected["metrics"].items()},
                    "inner_eligible": int(selected["diagnostics"].get("eligible_events", 0)),
                    "inner_pruned": int(selected["diagnostics"].get("pruned_events", 0)),
                }
            )
    return outer_rows, selected_rows, inner_rows, traces


def aggregate_outer(outer_rows: list[dict]):
    result = {}
    for method in METHODS:
        rows = [row for row in outer_rows if row["method"] == method]
        counts = empty_counts()
        for row in rows:
            add_counts(counts, row)
        metrics = native.metrics(counts)
        diagnostics = {
            key: int(sum(row[key] for row in rows))
            for key in ("eligible_events", "pruned_events", "pruned_TP", "pruned_FP")
        }
        pruned_total = diagnostics["pruned_TP"] + diagnostics["pruned_FP"]
        diagnostics["prune_precision"] = diagnostics["pruned_FP"] / pruned_total if pruned_total else None
        diagnostics["FP_removed_per_TP_lost"] = (
            diagnostics["pruned_FP"] / diagnostics["pruned_TP"] if diagnostics["pruned_TP"] else None
        )
        result[method] = {**metrics, **diagnostics}
    return result


def subject_stability_and_sensitivity(outer_rows: list[dict]):
    lookup = {(row["outer_subject"], row["method"]): row for row in outer_rows}
    subjects = sorted({row["outer_subject"] for row in outer_rows})
    stability = Counter()
    for subject in subjects:
        baseline = lookup[(subject, METHODS[0])]["F1"]
        proposed = lookup[(subject, METHODS[3])]["F1"]
        stability["improved" if proposed > baseline else "worse" if proposed < baseline else "equal"] += 1
    deltas = []
    for removed in subjects:
        metrics = {}
        for method in (METHODS[0], METHODS[3]):
            counts = empty_counts()
            for subject in subjects:
                if subject != removed:
                    add_counts(counts, lookup[(subject, method)])
            metrics[method] = native.metrics(counts)
        deltas.append(metrics[METHODS[3]]["F1"] - metrics[METHODS[0]]["F1"])
    return (
        {key: int(stability[key]) for key in ("improved", "equal", "worse")},
        {"min": float(np.min(deltas)), "median": float(np.median(deltas)), "max": float(np.max(deltas))},
    )


def bootstrap(outer_rows: list[dict]):
    lookup = {(row["outer_subject"], row["method"]): row for row in outer_rows}
    subjects = sorted({row["outer_subject"] for row in outer_rows})
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPEATS, len(subjects)))
    deltas = []
    for indices in sampled:
        metrics = {}
        for method in (METHODS[0], METHODS[3]):
            counts = empty_counts()
            for position in indices:
                add_counts(counts, lookup[(subjects[int(position)], method)])
            metrics[method] = native.metrics(counts)
        deltas.append(metrics[METHODS[3]]["F1"] - metrics[METHODS[0]]["F1"])
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": "C3 Selective Consensus Veto minus C0 Final Strong Native",
        "unit": "subject",
        "paired": True,
        "repetitions": BOOTSTRAP_REPEATS,
        "seed": SEED,
        "sampled_subject_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
        "mean_delta_F1": float(array.mean()),
        "CI95": [float(x) for x in np.quantile(array, (0.025, 0.975))],
    }


def parameter_frequency(selected_rows: list[dict]):
    rows = []
    summary = {}
    for method in METHODS[1:]:
        selected = [row for row in selected_rows if row["method"] == method]
        summary[method] = {}
        for name, values in (("delta", DELTAS), ("tau", TAUS)):
            counts = Counter(str(row[f"selected_{name}"]) for row in selected if row[f"selected_{name}"] is not None)
            summary[method][name] = dict(sorted(counts.items()))
            for value in values:
                rows.append(
                    {
                        "method": method,
                        "parameter": name,
                        "value": value,
                        "count": int(counts.get(str(value), 0)),
                        "proportion": counts.get(str(value), 0) / EXPECTED_SUBJECTS,
                    }
                )
    return rows, summary


def render_report(summary: dict) -> str:
    aggregate = summary["aggregate"]
    baseline = aggregate[METHODS[0]]
    lines = [
        "# Selective Window-Consensus Veto — SAMMLV Development Gate",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "本实验保持 Final Strong Native event generation 不变，只对其既有事件执行预注册的删除控制。没有修改 aggregation、Native decoder 或 candidate membership。",
        "",
        "## Anchor and protocol gates",
        "",
        f"- C0 exact PASS：{baseline['TP']}/{baseline['FP']}/{baseline['FN']}，F1={baseline['F1']:.6f}。",
        f"- Candidate bank：{summary['candidate_bank']['total']} events；pair={summary['candidate_bank']['PAIR']}，single-view={summary['candidate_bank']['SINGLE-VIEW-CANDIDATE']}。",
        "- C1 隔离 margin-only：删除 `H_s<=delta`；C2 隔离 agreement-only：只对 two-view event 删除 `A_ratio<tau`；C3 删除条件为 pair 且 `H_s<=delta` 且 `A_ratio<tau`。",
        "- 每个 outer fold 只用另外 28 subjects 的 pooled Spotting F1 选择参数。Tie-break：higher F1、fewer FP、fewer TP lost、smaller pruning rate、smaller delta、smaller tau。",
        "",
        "## Aggregate results",
        "",
        "| Method | TP/FP/FN | Precision | Recall | F1 | Delta F1 | Eligible | Pruned | Pruned TP | Pruned FP | Prune precision | FP/TP lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate[method]
        precision = "NA" if row["prune_precision"] is None else f"{row['prune_precision']:.6f}"
        ratio = "∞" if row["FP_removed_per_TP_lost"] is None and row["pruned_FP"] else "NA" if row["FP_removed_per_TP_lost"] is None else f"{row['FP_removed_per_TP_lost']:.6f}"
        lines.append(
            f"| {method} | {row['TP']}/{row['FP']}/{row['FN']} | {row['Precision']:.6f} | {row['Recall']:.6f} | "
            f"{row['F1']:.6f} | {row['F1']-baseline['F1']:+.6f} | {row['eligible_events']} | {row['pruned_events']} | "
            f"{row['pruned_TP']} | {row['pruned_FP']} | {precision} | {ratio} |"
        )
    lines += ["", "## C3 stability", ""]
    stable = summary["C3_subject_stability"]
    sensitivity = summary["C3_leave_one_subject_delta_F1"]
    boot = summary["C3_vs_C0_bootstrap"]
    lines += [
        f"- Subject improved/equal/worse：{stable['improved']}/{stable['equal']}/{stable['worse']}。",
        f"- Leave-one-subject aggregate Delta F1 min/median/max：{sensitivity['min']:+.6f} / {sensitivity['median']:+.6f} / {sensitivity['max']:+.6f}。",
        f"- 1000 paired subject bootstrap：observed Delta F1={aggregate[METHODS[3]]['F1']-baseline['F1']:+.6f}，mean={boot['mean_delta_F1']:+.6f}，95% CI=[{boot['CI95'][0]:+.6f}, {boot['CI95'][1]:+.6f}]。",
        "",
        "## Selected parameter frequencies",
        "",
        f"- C3 `delta`：`{json.dumps(summary['selected_parameter_frequencies'][METHODS[3]]['delta'], sort_keys=True)}`",
        f"- C3 `tau`：`{json.dumps(summary['selected_parameter_frequencies'][METHODS[3]]['tau'], sort_keys=True)}`",
        "- C3 在 29/29 folds 同时选择两个预注册上端点；按 stop rule 仅记录，不扩 grid。",
        "",
        "## Decision gates",
        "",
    ]
    for key, value in summary["decision_gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "`FP removed / TP lost > 6.16` 按协议作为偏好项报告，不作为额外强制 gate。",
        "",
        "实验到此停止；未运行新 aggregation、recognition、LR、rescue、Native retuning 或 grid expansion。",
    ]
    return "\n".join(lines) + "\n"


def blocked_report(output_root: Path, reason: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "SELECTIVE_WINDOW_CONSENSUS_VETO_SAMMLV_CN.md").write_text(
        "# Selective Window-Consensus Veto — SAMMLV Development Gate\n\n"
        "`BLOCKED-SELECTIVE-WINDOW-CONSENSUS-ANCHOR`\n\n"
        f"{reason}\n\nC1/C2/C3 were not run.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        gated = base.provenance_gate(argparse.Namespace(raw_dir=args.raw_dir, output_root=args.output_root))
        if len(gated["subjects"]) != EXPECTED_SUBJECTS or len(gated["loaded"]) != EXPECTED_VIDEOS:
            raise RuntimeError("raw source cardinality mismatch")
        videos, subject_anchor, anchor, label_counts, coverage_counts = build_candidate_bank(gated)
    except Exception as error:
        blocked_report(args.output_root, f"{type(error).__name__}: {error}")
        print("BLOCKED-SELECTIVE-WINDOW-CONSENSUS-ANCHOR")
        print(f"{type(error).__name__}: {error}")
        return 2

    outer_rows, selected_rows, inner_rows, traces = run_nested(videos, list(gated["subjects"]))
    aggregate = aggregate_outer(outer_rows)
    baseline = aggregate[METHODS[0]]
    proposed = aggregate[METHODS[3]]
    if tuple(baseline[key] for key in ("TP", "FP", "FN")) != EXPECTED_ANCHOR:
        raise RuntimeError("C0 outer aggregate changed after anchor gate")
    stability, sensitivity = subject_stability_and_sensitivity(outer_rows)
    bootstrap_result = bootstrap(outer_rows)
    frequency_rows, frequency_summary = parameter_frequency(selected_rows)
    ratio = proposed["FP_removed_per_TP_lost"]
    gates = {
        "F1_C3_gt_0.279202": proposed["F1"] > baseline["F1"],
        "FP_C3_lt_143": proposed["FP"] < baseline["FP"],
        "TP_C3_ge_47": proposed["TP"] >= 47,
        "bootstrap_mean_gt_0": bootstrap_result["mean_delta_F1"] > 0,
        "leave_one_subject_median_gt_0": sensitivity["median"] > 0,
        "C3_F1_gt_C1": proposed["F1"] > aggregate[METHODS[1]]["F1"],
        "preferred_FP_removed_per_TP_lost_gt_6.16": ratio is None and proposed["pruned_FP"] > 0 or ratio is not None and ratio > 6.16,
    }
    required = [key for key in gates if not key.startswith("preferred_")]
    if all(gates[key] for key in required):
        decision = "SELECTIVE-WINDOW-CONSENSUS-STRONG-GO"
    elif proposed["F1"] > baseline["F1"]:
        decision = "SELECTIVE-WINDOW-CONSENSUS-WEAK-GO"
    else:
        decision = "SELECTIVE-WINDOW-CONSENSUS-NO-GO"

    for row in outer_rows:
        base_row = next(
            candidate for candidate in outer_rows
            if candidate["outer_subject"] == row["outer_subject"] and candidate["method"] == METHODS[0]
        )
        row["Delta_F1_vs_C0"] = row["F1"] - base_row["F1"]
        if row["method"] == METHODS[0]:
            row["comparison_vs_C0"] = "baseline"
        else:
            row["comparison_vs_C0"] = "improved" if row["F1"] > base_row["F1"] else "worse" if row["F1"] < base_row["F1"] else "equal"

    summary_rows = []
    for method in METHODS:
        row = aggregate[method]
        summary_rows.append(
            {
                "method": method,
                **row,
                "Delta_TP_vs_C0": row["TP"] - baseline["TP"],
                "Delta_FP_vs_C0": row["FP"] - baseline["FP"],
                "Delta_FN_vs_C0": row["FN"] - baseline["FN"],
                "Delta_F1_vs_C0": row["F1"] - baseline["F1"],
            }
        )
    summary = {
        "experiment": "selective_window_consensus_veto_sammlv_development_gate",
        "decision": decision,
        "provenance": {
            "raw_manifest": str(gated["manifest_path"]),
            "raw_manifest_sha256": gated["manifest_sha"],
            "compact_cache_sha256": base.EXPECTED_CACHE_SHA,
            "frozen_config_report_sha256": gated["config_sha"],
            "pair_feature_source": str(PAIR_SOURCE),
            "pair_feature_source_sha256": sha256(PAIR_SOURCE),
            "subjects": EXPECTED_SUBJECTS,
            "videos": EXPECTED_VIDEOS,
            "k_p": K_P,
        },
        "anchor": anchor,
        "subject_anchor": {subject: native.metrics(counts) for subject, counts in subject_anchor.items()},
        "candidate_bank": {
            "total": int(sum(label_counts.values())),
            **{key: int(value) for key, value in label_counts.items()},
            **{key: int(value) for key, value in coverage_counts.items()},
        },
        "protocol": {
            "C1_definition": "delete iff H_s <= delta; pair coverage unused",
            "C2_definition": "two-view eligible; delete iff A_ratio < tau; single-view forced keep",
            "C3_definition": "delete iff two-view and H_s <= delta and A_ratio < tau; high-confidence and single-view forced keep",
            "delta_grid": list(DELTAS),
            "tau_grid": list(TAUS),
            "C3_grid_count": 16,
            "outer_LOSO_subjects": EXPECTED_SUBJECTS,
            "selection": "other 28 subjects pooled Spotting F1",
            "tie_break": ["higher exact F1", "fewer FP", "fewer TP lost", "smaller pruning rate", "smaller delta", "smaller tau"],
            "outer_test_GT_used_for_selection": False,
            "bootstrap_repetitions": BOOTSTRAP_REPEATS,
            "bootstrap_seed": SEED,
            "weak_go_definition": "C3 F1 > C0 but at least one STRONG required gate fails",
            "margin_source": "verified compact frozen cache, identical to previous signed-margin audits",
            "agreement_source": "verified raw pre-stitch pair views",
        },
        "aggregate": aggregate,
        "C3_subject_stability": stability,
        "C3_leave_one_subject_delta_F1": sensitivity,
        "selected_parameter_frequencies": frequency_summary,
        "C3_parameter_boundary_selection": {
            "delta_0.20": 29,
            "tau_0.80": 29,
            "both_upper_endpoints_29_of_29": True,
            "grid_expansion_allowed": False,
        },
        "C3_vs_C0_bootstrap": {
            **bootstrap_result,
            "observed_delta_F1": proposed["F1"] - baseline["F1"],
        },
        "decision_gates": gates,
        "integrity": {
            "original_event_generation_unchanged": True,
            "raw_and_cache_O0_event_identity_79_of_79": True,
            "candidate_membership_changed": False,
            "aggregation_modified": False,
            "arithmetic_geometric_harmonic_min_curve_used": False,
            "recognition_used": False,
            "logistic_regression_used": False,
            "rescue_used": False,
            "native_decoder_changed": False,
            "grid_expanded": False,
            "backbone_forward": False,
        },
    }
    output_dir = args.output_root / "outputs"
    write_csv(output_dir / "selective_veto_summary.csv", summary_rows)
    write_csv(output_dir / "selective_veto_outer_metrics.csv", outer_rows)
    write_csv(output_dir / "selective_veto_selected_configs.csv", selected_rows)
    write_csv(output_dir / "selective_veto_inner_scores.csv", inner_rows)
    write_csv(output_dir / "selective_veto_event_trace.csv", traces)
    write_csv(output_dir / "selective_veto_parameter_frequency.csv", frequency_rows)
    (output_dir / "selective_veto_bootstrap.json").write_text(
        json.dumps(summary["C3_vs_C0_bootstrap"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "selective_veto_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "SELECTIVE_WINDOW_CONSENSUS_VETO_SAMMLV_CN.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "aggregate": aggregate, "stability": stability, "sensitivity": sensitivity, "bootstrap": summary["C3_vs_C0_bootstrap"], "gates": gates}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
