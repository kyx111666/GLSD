#!/usr/bin/env python3
"""E3 — Controlled Weak-Candidate Recovery, cache-only and additive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECOVERY_DIR = ROOT / "my_method/recovery_pool_feasibility"
SCALE_DIR = ROOT / "my_method/scale_consistency_feasibility"
sys.path.insert(0, str(RECOVERY_DIR))
sys.path.insert(0, str(SCALE_DIR))

import run_recovery_pool_audit as recovery  # noqa: E402
from run_scale_consistency_audit import fixed_scales, generate_scale_peaks  # noqa: E402


OUTPUT = ROOT / "results/e3_controlled_weak_candidate_recovery_v1"
METHODS = ("V0_native", "V1_global", "V2_local", "V3_mean", "V4_consensus")
THRESHOLDS = (0.0, 0.25, 0.5, 0.75)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 100
NATIVE_FACTOR = 0.55
LOW_FACTOR = 0.25
RECOVERY_RESULTS = ROOT / "results/recovery_pool_feasibility_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scale_peak_bank(score: np.ndarray, k_p: int) -> tuple[list[int], list[list[int]]]:
    scales = fixed_scales(k_p)
    positions = [
        [int(item["position"]) for item in generate_scale_peaks(score, scale)]
        for scale in scales
    ]
    return scales, positions


def local_evidence(score: np.ndarray, peak: int, k_p: int,
                   precomputed: tuple[list[int], list[list[int]]] | None = None
                   ) -> tuple[int, float]:
    """Reuse locked scale-consistency support at an unchanged weak peak."""
    scales, positions_by_scale = precomputed or scale_peak_bank(score, k_p)
    radius = k_p // 2
    support = sum(
        int(any(abs(position - int(peak)) <= radius for position in positions))
        for positions in positions_by_scale
    )
    return support, support / len(scales)


def g_margin(curve: np.ndarray, peak: int, low_gate: float, strict_gate: float) -> float:
    denominator = strict_gate - low_gate
    if denominator <= 0:
        return 0.0
    return float(np.clip((float(curve[peak]) - low_gate) / denominator, 0.0, 1.0))


def signature(events: list[dict]) -> tuple:
    return tuple((int(event["peak"]), int(event["onset"]), int(event["offset"]))
                 for event in events)


def make_bank(records: list[dict], k_p: int):
    bank = []
    for record in records:
        curve, strict_raw, strict_gate = recovery.proposal(record, k_p, NATIVE_FACTOR)
        _curve, low_raw, low_gate = recovery.proposal(record, k_p, LOW_FACTOR)
        strict = [{key: value for key, value in event.items() if key != "sources"}
                  for event in strict_raw]
        low = [{key: value for key, value in event.items() if key != "sources"}
               for event in low_raw]
        weak = [event for event in low
                if recovery.nearest_event(strict, event["peak"], k_p) is None]
        strict_signature = signature(strict)
        weak_signature = signature(weak)
        if any(signature(strict) != strict_signature or signature(weak) != weak_signature
               for _method in METHODS[1:]):
            raise RuntimeError("same-candidate invariant failed")
        enriched = []
        precomputed = scale_peak_bank(record["score"], k_p)
        for event in weak:
            l_raw, l_norm = local_evidence(
                record["score"], event["peak"], k_p, precomputed=precomputed)
            g_norm = g_margin(curve, event["peak"], low_gate, strict_gate)
            enriched.append({
                **event, "G_raw": float(curve[event["peak"]]), "G_norm": g_norm,
                "L_raw": int(l_raw), "L_norm": float(l_norm),
                "mean_score": float((g_norm + l_norm) / 2.0),
                "consensus_score": float(min(g_norm, l_norm)),
            })
        bank.append({"subject": record["subject"], "video": record["video"],
                     "samples": record["samples"], "strict": strict, "weak": enriched,
                     "strict_gate": strict_gate, "low_gate": low_gate})
    return bank


def score_name(method: str) -> str:
    return {"V1_global": "G_norm", "V2_local": "L_norm", "V3_mean": "mean_score",
            "V4_consensus": "consensus_score"}[method]


def accepted_weak(record: dict, method: str, tau: float | None) -> list[dict]:
    if method == "V0_native":
        return []
    field = score_name(method)
    return [event for event in record["weak"] if event[field] >= float(tau)]


def decode(record: dict, method: str, tau: float | None) -> tuple[list[dict], list[dict]]:
    accepted = accepted_weak(record, method, tau)
    events = sorted(
        [{**event, "source": "native"} for event in record["strict"]]
        + [{**event, "source": "recovery"} for event in accepted],
        key=lambda event: (event["peak"], 0 if event["source"] == "native" else 1),
    )
    if signature([event for event in events if event["source"] == "native"]) != signature(record["strict"]):
        raise RuntimeError("native candidate changed")
    return events, accepted


def evaluate_record(record: dict, method: str, tau: float | None) -> dict:
    native_events, _ = decode(record, "V0_native", None)
    native_assignment, _ = greedy_assignment(native_events, record["samples"])
    events, accepted = decode(record, method, tau)
    assignment, unmatched = greedy_assignment(events, record["samples"])
    base_gt = {index for index in native_assignment if index >= 0}
    final_gt = {index for index in assignment if index >= 0}
    return {
        "TP": sum(index >= 0 for index in assignment),
        "FP": sum(index < 0 for index in assignment),
        "FN": len(unmatched), "event_count": len(events),
        "accepted_recovery_count": len(accepted),
        "recovered_GT": len(final_gt - base_gt),
        "lost_native_GT": len(base_gt - final_gt),
        "assignments": assignment, "events": events,
    }


def greedy_assignment(events: list[dict], samples) -> tuple[list[int], set[int]]:
    unmatched = set(range(len(samples)))
    assignment = []
    for event in events:
        choices = [(recovery.interval_iou(event, samples[index]), index) for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            assignment.append(int(best_index))
        else:
            assignment.append(-1)
    return assignment, unmatched


def empty_counts() -> Counter:
    return Counter({key: 0 for key in ("TP", "FP", "FN", "event_count",
                                       "accepted_recovery_count", "recovered_GT",
                                       "lost_native_GT")})


def add_counts(target: Counter, source: dict):
    for key in target:
        target[key] += int(source.get(key, 0))


def f1(values: dict) -> float:
    denominator = 2 * values["TP"] + values["FP"] + values["FN"]
    return 2 * values["TP"] / denominator if denominator else 0.0


def metrics(values: dict) -> dict:
    tp, fp, fn = (int(values[key]) for key in ("TP", "FP", "FN"))
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn) if tp + fn else 0.0,
            "Raw_Spotting_F1": f1(values),
            "event_count": int(values["event_count"]),
            "accepted_recovery_count": int(values["accepted_recovery_count"]),
            "recovered_GT": int(values["recovered_GT"]),
            "lost_native_GT": int(values["lost_native_GT"])}


def select_threshold(bank: list[dict], method: str, train_subjects: list[str]) -> float:
    train_subjects = set(map(str, train_subjects))
    ranked = []
    for index, tau in enumerate(THRESHOLDS):
        total = empty_counts()
        for record in bank:
            if record["subject"] in train_subjects:
                add_counts(total, evaluate_record(record, method, tau))
        ranked.append(((-f1(total), total["FP"], total["accepted_recovery_count"], index), tau))
    return float(min(ranked, key=lambda item: item[0])[1])


def evaluate_outer(bank: list[dict], method: str, subjects: list[str]):
    subject_metrics, selections, predictions = {}, [], []
    for held in subjects:
        tau = None if method == "V0_native" else select_threshold(
            bank, method, [subject for subject in subjects if subject != held])
        total = empty_counts()
        for record in bank:
            if record["subject"] != held:
                continue
            result = evaluate_record(record, method, tau)
            add_counts(total, result)
            predictions.append({"subject": held, "video": record["video"], "method": method,
                "selected_threshold": tau,
                "events": [{**event, "matched_gt_index": match}
                           for event, match in zip(result["events"], result["assignments"])]})
        subject_metrics[held] = metrics(total)
        if tau is not None:
            selections.append({"subject": held, "method": method, "selected_threshold": tau,
                               "training_subject_count": len(subjects) - 1,
                               "test_labels_used_for_selection": False})
    pooled = empty_counts()
    for value in subject_metrics.values():
        add_counts(pooled, value)
    return metrics(pooled), subject_metrics, selections, predictions


def paired_bootstrap(per_subject: dict, left: str, right: str) -> dict:
    subjects = sorted(per_subject[left])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        lhs, rhs = empty_counts(), empty_counts()
        for subject in sampled:
            add_counts(lhs, per_subject[left][subject])
            add_counts(rhs, per_subject[right][subject])
        draws[repeat] = f1(lhs) - f1(rhs)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"comparison": f"{left}_minus_{right}", "unit": "subject",
            "resamples": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
            "mean_delta": float(np.mean(draws)), "CI95": [float(low), float(high)]}


def run_spec(spec: dict, output_root: Path):
    cache_sha = sha256(spec["cache"])
    if cache_sha != spec["sha"]:
        raise RuntimeError(f"cache SHA mismatch: {spec['cache']}")
    _payload, records, k_p = recovery.load_records(spec["cache"])
    bank = make_bank(records, k_p)
    recovery_csv = (RECOVERY_RESULTS / spec["backbone"].lower().replace("-", "_")
                    / spec["dataset"].lower() / "pool_coverage.csv")
    with recovery_csv.open(newline="", encoding="utf-8") as handle:
        recovery_rows = list(csv.DictReader(handle))
    expected_weak = int(next(row for row in recovery_rows
                             if row["pool"] == "R1_0.25")["additional_candidates"])
    observed_weak = sum(len(record["weak"]) for record in bank)
    if observed_weak != expected_weak:
        raise RuntimeError(f"weak-pool mismatch: {observed_weak} != {expected_weak}")
    subjects = sorted({record["subject"] for record in bank})
    observed = Counter()
    for record in bank:
        observed.update(dict(zip(("TP", "FP", "FN"),
                                 recovery.greedy_counts(record["strict"], record["samples"]))))
    anchor = tuple(observed[key] for key in ("TP", "FP", "FN"))
    if anchor != spec["expected"]:
        raise RuntimeError(f"BLOCKED-BASELINE {spec['backbone']}/{spec['dataset']}: "
                           f"{anchor} != {spec['expected']}")

    pooled, per_subject, selected_rows, predictions = {}, {}, [], []
    for method in METHODS:
        pooled[method], per_subject[method], selected, method_predictions = evaluate_outer(
            bank, method, subjects)
        selected_rows.extend(selected)
        predictions.extend(method_predictions)
    v0 = pooled["V0_native"]
    break_even = (v0["TP"] + v0["FP"] + v0["FN"]) / v0["TP"]
    bootstraps = {f"{method}_vs_V0": paired_bootstrap(per_subject, method, "V0_native")
                  for method in METHODS[1:]}
    for right in ("V1_global", "V2_local", "V3_mean"):
        bootstraps[f"V4_vs_{right[:2]}"] = paired_bootstrap(
            per_subject, "V4_consensus", right)

    comparison = []
    for method in METHODS:
        value = pooled[method]
        additional_tp = value["TP"] - v0["TP"]
        additional_fp = value["FP"] - v0["FP"]
        recovered = value["recovered_GT"]
        fp_per_tp = additional_fp / recovered if recovered else "inf"
        accepted_per_tp = value["accepted_recovery_count"] / recovered if recovered else "inf"
        ci = [0.0, 0.0] if method == "V0_native" else bootstraps[f"{method}_vs_V0"]["CI95"]
        comparison.append({"backbone": spec["backbone"], "dataset": spec["dataset"],
            "method": method, **value, "delta_F1_vs_V0": value["Raw_Spotting_F1"] - v0["Raw_Spotting_F1"],
            "additional_TP_vs_V0": additional_tp, "additional_FP_vs_V0": additional_fp,
            "FP_per_recovered_TP": fp_per_tp,
            "accepted_candidates_per_recovered_TP": accepted_per_tp,
            "break_even_FP_per_TP": break_even,
            "FP_per_TP_below_break_even": isinstance(fp_per_tp, float) and fp_per_tp < break_even,
            "CI95_low": ci[0], "CI95_high": ci[1]})

    subject_rows = [{"backbone": spec["backbone"], "dataset": spec["dataset"],
                     "subject": subject, "method": method, **per_subject[method][subject]}
                    for method in METHODS for subject in subjects]
    threshold_lookup = {(row["subject"], row["method"]): row["selected_threshold"]
                        for row in selected_rows}
    mechanism_rows = []
    for record in bank:
        for event in record["weak"]:
            row = {"backbone": spec["backbone"], "dataset": spec["dataset"],
                   "subject": record["subject"], "video": record["video"],
                   "peak": event["peak"], "interval": f"[{event['onset']},{event['offset']}]",
                   **{key: event[key] for key in ("G_raw", "G_norm", "L_raw", "L_norm",
                                                   "mean_score", "consensus_score")}}
            for method in METHODS[1:]:
                tau = threshold_lookup[(record["subject"], method)]
                row[f"accepted_{method[:2]}"] = event[score_name(method)] >= tau
                row[f"selected_tau_{method[:2]}"] = tau
            row["best_GT_IoU"] = max(
                [recovery.interval_iou(event, sample) for sample in record["samples"]] or [0.0])
            mechanism_rows.append(row)

    native_signatures = {(record["subject"], record["video"]): signature(record["strict"])
                         for record in bank}
    for prediction in predictions:
        current = signature([event for event in prediction["events"] if event["source"] == "native"])
        if current != native_signatures[(prediction["subject"], prediction["video"])]:
            raise RuntimeError("native untouched invariant failed")

    destination = output_root / spec["backbone"].lower().replace("-", "_") / spec["dataset"].lower()
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "verifier_comparison.csv", comparison)
    write_csv(destination / "per_subject_metrics.csv", subject_rows)
    write_csv(destination / "selected_thresholds.csv", selected_rows)
    write_csv(destination / "weak_candidates.csv", mechanism_rows)
    (destination / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    (destination / "bootstrap.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    report = {"study": "E3 Controlled Weak-Candidate Recovery", "backbone": spec["backbone"],
        "dataset": spec["dataset"], "cache": {"path": str(spec["cache"].resolve()),
        "sha256": cache_sha, "expected_TP_FP_FN": list(spec["expected"]),
        "observed_TP_FP_FN": list(anchor), "status": "PASS"},
        "proposal": {"strict_factor": NATIVE_FACTOR, "low_factor": LOW_FACTOR,
                     "peak_distance": k_p, "interval": "[peak-k_p,peak+k_p] inclusive",
                     "weak_definition": "R1_0.25 candidate with no R0 peak within <=k_p",
                     "weak_candidate_count": observed_weak,
                     "expected_from_recovery_audit": expected_weak,
                     "recovery_audit_gate": "PASS"},
        "evidence": {"G_main": "clip((G[p]-low_gate)/(strict_gate-low_gate),0,1)",
                     "L_source": str((SCALE_DIR / "run_scale_consistency_audit.py").resolve()),
                     "L_source_sha256": sha256(SCALE_DIR / "run_scale_consistency_audit.py"),
                     "L_formula": "count of locked-scale native peaks within floor(k_p/2) of fixed weak peak / number of scales",
                     "fixed_scale_ratios": ["2/3", "4/5", "1", "5/4", "3/2"],
                     "normalization": "G and L both deterministic per-video [0,1]; no dataset statistics"},
        "selection": {"grid": list(THRESHOLDS), "outer_split": "subject LOSO",
                      "target": "pooled training-subject Raw Spotting F1",
                      "tie_break": ["fewer FP", "fewer recovery predictions", "lower threshold index"],
                      "test_subject_used_for_selection": False},
        "same_candidate_verified": True, "native_untouched_verified": True,
        "matching": "chronological greedy one-to-one IoU>=0.5",
        "break_even_FP_per_TP": break_even, "pooled_metrics": pooled,
        "comparison": comparison, "bootstrap": bootstraps,
        "outputs": {name: str((destination / name).resolve()) for name in
                    ("report.json", "verifier_comparison.csv", "per_subject_metrics.csv",
                     "selected_thresholds.csv", "weak_candidates.csv", "predictions.json", "bootstrap.json")}}
    (destination / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    return report


def print_tables(reports):
    print("Backbone | Dataset | V0 Native | V1 G | V2 L | V3 Mean | V4 Consensus")
    print("---|---|---:|---:|---:|---:|---:")
    for report in reports:
        values = [report["pooled_metrics"][method]["Raw_Spotting_F1"] for method in METHODS]
        print(f"{report['backbone']} | {report['dataset']} | " + " | ".join(f"{x:.6f}" for x in values))
    print("\nBackbone | Dataset | Method | +TP | +FP | FP/TP | Break-even | Delta F1 | CI95")
    print("---|---|---|---:|---:|---:|---:|---:|---")
    for report in reports:
        for row in report["comparison"][1:]:
            fp_tp = row["FP_per_recovered_TP"]
            fp_tp = f"{fp_tp:.3f}" if isinstance(fp_tp, float) else fp_tp
            print(f"{report['backbone']} | {report['dataset']} | {row['method']} | "
                  f"{row['additional_TP_vs_V0']} | {row['additional_FP_vs_V0']} | {fp_tp} | "
                  f"{row['break_even_FP_per_TP']:.3f} | {row['delta_F1_vs_V0']:+.6f} | "
                  f"[{row['CI95_low']:+.6f},{row['CI95_high']:+.6f}]")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [run_spec(spec, args.output_root) for spec in recovery.SPECS]
    rows = [{"backbone": report["backbone"], "dataset": report["dataset"], **item}
            for report in reports for item in report["comparison"]]
    write_csv(args.output_root / "combined_verifier_summary.csv", rows)
    v4_deltas = [report["pooled_metrics"]["V4_consensus"]["Raw_Spotting_F1"]
                 - report["pooled_metrics"]["V0_native"]["Raw_Spotting_F1"] for report in reports]
    combined = {"study": "E3 Controlled Weak-Candidate Recovery", "status": "COMPLETE",
                "same_candidate_verified": True, "native_untouched_verified": True,
                "V4_stability": {"improved": sum(value > 0 for value in v4_deltas),
                                 "equal": sum(abs(value) <= 1e-15 for value in v4_deltas),
                                 "worse": sum(value < 0 for value in v4_deltas),
                                 "worst_case_delta_F1": min(v4_deltas)},
                "reports": reports, "result_root": str(args.output_root.resolve())}
    (args.output_root / "combined_report.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print_tables(reports)
    print(f"\nResults: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
