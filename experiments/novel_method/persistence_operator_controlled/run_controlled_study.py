#!/usr/bin/env python3
"""Cache-only Persistence Operator Controlled Study for four frozen payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from persistence_operators import OPERATORS, moving_average, raw_scores, robust_z


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = ROOT / "results/persistence_operator_controlled_v1"
THRESHOLDS = (0.0, 0.5, 1.0)
IOU_THRESHOLD = 0.5
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 100
P_HIGH = 0.55
SPECS = (
    {
        "backbone": "ME-TST", "dataset": "SAMMLV",
        "path": ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
        "sha256": "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569",
        "expected": (53, 184, 106),
    },
    {
        "backbone": "ME-TST", "dataset": "CASME_3",
        "path": ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
        "sha256": "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda",
        "expected": (81, 912, 777),
    },
    {
        "backbone": "BoostingVRME", "dataset": "SAMMLV",
        "path": ROOT / "caches/boostingvrme/sammlv_curves.pkl",
        "sha256": "abcda6477dcbf89a2083f6fdfe673b1686ca3cc100e62147b81a6806c67b05bc",
        "expected": (54, 158, 105),
    },
    {
        "backbone": "BoostingVRME", "dataset": "CASME_3",
        "path": ROOT / "caches/boostingvrme/casme3_curves.pkl",
        "sha256": "9775aa254149717b27e1ea8449a5fdf4e16a26b0a407ce501d6532dcef00a14a",
        "expected": (80, 830, 778),
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_records(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    k_p = int(payload["k_p"])
    if "records" in payload:
        records = [{
            "subject": str(row["subject"]), "video": str(row["video"]),
            "score": np.asarray(row["score"], dtype=np.float64),
            "samples": row["samples"],
        } for row in payload["records"]]
    else:
        records = []
        for subject in payload["subject_curves"]:
            for index, score in enumerate(subject["score"]):
                records.append({
                    "subject": str(subject["subject"]),
                    "video": str(subject["videos"][index]),
                    "score": np.asarray(score, dtype=np.float64),
                    "samples": subject["samples"][index],
                })
    return payload, records, k_p


def interval_iou(event: dict, sample) -> float:
    left, right = int(event["onset"]), int(event["offset"])
    gt_left, gt_right = int(sample[0]), int(sample[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union > 0 else 0.0


def match_events(events: list[dict], samples) -> tuple[list[int], set[int]]:
    unmatched = set(range(len(samples)))
    matches = []
    for event in events:
        choices = [(interval_iou(event, samples[index]), index) for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            matches.append(int(best_index))
        else:
            matches.append(-1)
    return matches, unmatched


def counts(events: list[dict], samples) -> tuple[dict, list[int]]:
    matches, unmatched = match_events(events, samples)
    return {
        "TP": int(sum(index >= 0 for index in matches)),
        "FP": int(sum(index < 0 for index in matches)),
        "FN": int(len(unmatched)),
        "event_count": int(len(events)),
    }, matches


def empty_counts() -> Counter:
    return Counter({"TP": 0, "FP": 0, "FN": 0, "event_count": 0})


def add_counts(target: Counter, source: dict):
    for key in target:
        target[key] += int(source.get(key, 0))


def f1(metric: dict) -> float:
    denominator = 2 * metric["TP"] + metric["FP"] + metric["FN"]
    return 2 * metric["TP"] / denominator if denominator else 0.0


def full_metrics(metric: dict) -> dict:
    tp, fp, fn = (int(metric[key]) for key in ("TP", "FP", "FN"))
    return {
        "TP": tp, "FP": fp, "FN": fn, "event_count": int(metric["event_count"]),
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
        "Raw_Spotting_F1": f1(metric),
    }


def base_decode(record: dict, k_p: int) -> tuple[np.ndarray, list[dict], float]:
    curve = moving_average(record["score"], 2 * k_p)
    threshold = float(curve.mean() + P_HIGH * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
    events = [{"peak": int(peak), "onset": int(peak - k_p),
               "offset": int(peak + k_p)} for peak in peaks]
    return curve, events, threshold


def candidate_signature(events: list[dict]) -> tuple:
    return tuple((int(event["peak"]), int(event["onset"]), int(event["offset"]))
                 for event in events)


def prepare_bank(records: list[dict], k_p: int, backbone: str, dataset: str):
    bank, candidate_rows = [], []
    for record in records:
        curve, base_events, proposal_threshold = base_decode(record, k_p)
        peaks = np.asarray([event["peak"] for event in base_events], dtype=int)
        signatures = {operator: candidate_signature(base_events) for operator in OPERATORS}
        if len(set(signatures.values())) != 1:
            raise RuntimeError(f"same-candidate invariant failed: {record['subject']}/{record['video']}")
        operator_raw = {
            operator: raw_scores(
                record["score"] if operator == "P3_scale_space_survival" else curve,
                peaks, k_p, operator,
            )
            for operator in OPERATORS
        }
        operator_norm = {
            operator: (operator_raw[operator] if operator == "P0_no_persistence"
                       else robust_z(operator_raw[operator]))
            for operator in OPERATORS
        }
        if any(len(operator_raw[operator]) != len(base_events) for operator in OPERATORS):
            raise RuntimeError(f"score alignment failed: {record['subject']}/{record['video']}")
        prepared = {
            "subject": record["subject"], "video": record["video"],
            "samples": record["samples"], "curve": curve,
            "proposal_threshold": proposal_threshold, "base_events": base_events,
            "by_operator": {},
        }
        for operator in OPERATORS:
            prepared["by_operator"][operator] = [
                {**event, "raw": float(operator_raw[operator][index]),
                 "norm": float(operator_norm[operator][index])}
                for index, event in enumerate(base_events)
            ]
        bank.append(prepared)
        for index, event in enumerate(base_events):
            best_iou = max([interval_iou(event, sample) for sample in record["samples"]] or [0.0])
            candidate_rows.append({
                "backbone": backbone, "dataset": dataset,
                "subject": record["subject"], "video": record["video"],
                "candidate_peak": event["peak"], "candidate_onset": event["onset"],
                "candidate_offset": event["offset"], "base_saliency": float(curve[event["peak"]]),
                "proposal_threshold": proposal_threshold,
                "P1_raw": float(operator_raw["P1_peak_prominence"][index]),
                "P1_norm": float(operator_norm["P1_peak_prominence"][index]),
                "P2_raw": float(operator_raw["P2_topological_0d"][index]),
                "P2_norm": float(operator_norm["P2_topological_0d"][index]),
                "P3_raw": float(operator_raw["P3_scale_space_survival"][index]),
                "P3_norm": float(operator_norm["P3_scale_space_survival"][index]),
                "GT_best_iou": float(best_iou),
            })
    return bank, candidate_rows


def operator_records(bank: list[dict], operator: str) -> list[dict]:
    return [{"subject": row["subject"], "video": row["video"], "samples": row["samples"],
             "events": row["by_operator"][operator]} for row in bank]


def select_threshold(records: list[dict], train_subjects: list[str], grid=THRESHOLDS) -> float:
    """Select solely from training-subject labels using pooled raw Spotting F1."""
    train_subjects = set(map(str, train_subjects))
    ranked = []
    for grid_index, tau in enumerate(grid):
        total = empty_counts()
        for record in records:
            if str(record["subject"]) not in train_subjects:
                continue
            kept = [event for event in record["events"] if float(event["norm"]) >= float(tau)]
            result, _ = counts(kept, record["samples"])
            add_counts(total, result)
        ranked.append(((-f1(total), total["FP"], total["event_count"], grid_index), float(tau)))
    return min(ranked, key=lambda item: item[0])[1]


def evaluate_outer(bank: list[dict], operator: str, subjects: list[str]):
    records = operator_records(bank, operator)
    selected, subject_metrics, predictions = [], {}, []
    for held in subjects:
        tau = None if operator == "P0_no_persistence" else select_threshold(
            records, [subject for subject in subjects if subject != held]
        )
        total = empty_counts()
        for record in records:
            if record["subject"] != held:
                continue
            kept = record["events"] if tau is None else [
                event for event in record["events"] if event["norm"] >= tau]
            result, matches = counts(kept, record["samples"])
            add_counts(total, result)
            predictions.append({
                "subject": held, "video": record["video"], "operator": operator,
                "selected_threshold": tau,
                "events": [{**event, "matched_gt_index": match,
                            "best_iou": max([interval_iou(event, sample) for sample in record["samples"]]
                                            or [0.0])}
                           for event, match in zip(kept, matches)],
            })
        subject_metrics[held] = full_metrics(total)
        if tau is not None:
            selected.append({"subject": held, "operator": operator, "selected_threshold": tau,
                             "training_subject_count": len(subjects) - 1,
                             "test_labels_used_for_selection": False})
    pooled = empty_counts()
    for metric in subject_metrics.values():
        add_counts(pooled, metric)
    return full_metrics(pooled), subject_metrics, selected, predictions


def paired_bootstrap(per_subject: dict, lhs: str, rhs: str) -> dict:
    subjects = sorted(per_subject[lhs])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        left, right = empty_counts(), empty_counts()
        for subject in sampled:
            add_counts(left, per_subject[lhs][subject])
            add_counts(right, per_subject[rhs][subject])
        draws[repeat] = f1(left) - f1(right)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "comparison": f"{lhs}_minus_{rhs}", "unit": "subject",
        "resamples": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
        "mean_delta": float(np.mean(draws)), "CI95": [float(low), float(high)],
    }


def prediction_lookup(predictions: list[dict], operator: str) -> dict:
    return {(row["subject"], row["video"]): row for row in predictions if row["operator"] == operator}


def iou_bin(value: float) -> str:
    if value < 0.1:
        return "IoU_lt_0.1"
    if value < 0.5:
        return "IoU_0.1_to_0.5"
    return "IoU_ge_0.5"


def error_attribution(all_predictions: list[dict], operator: str) -> list[dict]:
    baseline = prediction_lookup(all_predictions, "P0_no_persistence")
    current = prediction_lookup(all_predictions, operator)
    rows = []
    for key in sorted(baseline):
        base_events = baseline[key]["events"]
        current_events = current[key]["events"]
        base_by_peak = {event["peak"]: event for event in base_events}
        current_by_peak = {event["peak"]: event for event in current_events}
        base_gt = {event["matched_gt_index"] for event in base_events if event["matched_gt_index"] >= 0}
        current_gt = {event["matched_gt_index"] for event in current_events if event["matched_gt_index"] >= 0}
        deleted = [event for peak, event in base_by_peak.items() if peak not in current_by_peak]
        added = [event for peak, event in current_by_peak.items() if peak not in base_by_peak]
        deleted_bins, added_bins = Counter(), Counter()
        for event in deleted:
            deleted_bins[iou_bin(event["best_iou"])] += 1
        for event in added:
            added_bins[iou_bin(event["best_iou"])] += 1
        base_fp = {peak for peak, event in base_by_peak.items() if event["matched_gt_index"] < 0}
        current_fp = {peak for peak, event in current_by_peak.items() if event["matched_gt_index"] < 0}
        rows.append({
            "operator": operator, "subject": key[0], "video": key[1],
            "rescued_GT": len(current_gt - base_gt), "lost_GT": len(base_gt - current_gt),
            "removed_FP": len(base_fp - current_fp), "new_FP": len(current_fp - base_fp),
            "deleted_predictions": len(deleted), "added_predictions": len(added),
            **{f"deleted_{name}": deleted_bins[name] for name in
               ("IoU_lt_0.1", "IoU_0.1_to_0.5", "IoU_ge_0.5")},
            **{f"added_{name}": added_bins[name] for name in
               ("IoU_lt_0.1", "IoU_0.1_to_0.5", "IoU_ge_0.5")},
        })
    return rows


def run_spec(spec: dict, output_root: Path) -> dict:
    observed_sha = sha256(spec["path"])
    if observed_sha != spec["sha256"]:
        raise RuntimeError(f"cache hash mismatch: {spec['path']}")
    payload, records, k_p = load_records(spec["path"])
    bank, candidate_rows = prepare_bank(records, k_p, spec["backbone"], spec["dataset"])
    subjects = sorted({record["subject"] for record in bank})

    baseline = empty_counts()
    for record in bank:
        result, _ = counts(record["base_events"], record["samples"])
        add_counts(baseline, result)
    observed = tuple(baseline[key] for key in ("TP", "FP", "FN"))
    if observed != spec["expected"]:
        raise RuntimeError(f"BLOCKED-BASELINE {spec['backbone']}/{spec['dataset']}: "
                           f"observed={observed}, expected={spec['expected']}")

    pooled, per_subject, selected_rows, all_predictions = {}, {}, [], []
    for operator in OPERATORS:
        pooled[operator], per_subject[operator], selected, predictions = evaluate_outer(
            bank, operator, subjects)
        selected_rows.extend(selected)
        all_predictions.extend(predictions)

    bootstraps = {}
    for operator in OPERATORS:
        bootstraps[f"{operator}_vs_P0"] = paired_bootstrap(
            per_subject, operator, "P0_no_persistence")
        bootstraps[f"{operator}_vs_P1"] = paired_bootstrap(
            per_subject, operator, "P1_peak_prominence")

    comparison_rows = []
    for operator in OPERATORS:
        metric = pooled[operator]
        comparison_rows.append({
            "backbone": spec["backbone"], "dataset": spec["dataset"], "operator": operator,
            **metric,
            "delta_vs_P0": metric["Raw_Spotting_F1"] - pooled["P0_no_persistence"]["Raw_Spotting_F1"],
            "delta_vs_P1": metric["Raw_Spotting_F1"] - pooled["P1_peak_prominence"]["Raw_Spotting_F1"],
            "CI95_vs_P0_low": bootstraps[f"{operator}_vs_P0"]["CI95"][0],
            "CI95_vs_P0_high": bootstraps[f"{operator}_vs_P0"]["CI95"][1],
            "CI95_vs_P1_low": bootstraps[f"{operator}_vs_P1"]["CI95"][0],
            "CI95_vs_P1_high": bootstraps[f"{operator}_vs_P1"]["CI95"][1],
        })
    subject_rows = [{"backbone": spec["backbone"], "dataset": spec["dataset"],
                     "subject": subject, "operator": operator, **per_subject[operator][subject]}
                    for operator in OPERATORS for subject in subjects]
    attribution_rows = []
    for operator in OPERATORS[1:]:
        attribution_rows.extend(error_attribution(all_predictions, operator))

    destination = output_root / spec["backbone"].lower().replace("-", "_") / spec["dataset"].lower()
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "operator_comparison.csv", comparison_rows)
    write_csv(destination / "per_subject_metrics.csv", subject_rows)
    write_csv(destination / "candidate_scores.csv", candidate_rows)
    write_csv(destination / "selected_thresholds.csv", selected_rows)
    write_csv(destination / "error_attribution.csv", attribution_rows)
    (destination / "predictions.json").write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "study": "Persistence Operator Controlled Study",
        "backbone": spec["backbone"], "dataset": spec["dataset"],
        "input": {"cache": str(spec["path"].resolve()), "sha256": observed_sha,
                  "subjects": len(subjects), "videos": len(records),
                  "GT": sum(len(record["samples"]) for record in records), "k_p": k_p,
                  "payload_dataset": payload.get("dataset", payload.get("dataset_name"))},
        "baseline_verification": {"expected_TP_FP_FN": list(spec["expected"]),
                                  "observed_TP_FP_FN": list(observed), "status": "PASS"},
        "same_candidate_verified": True,
        "candidate_control": {
            "proposal": "G=boxcar_MA(score,2*k_p); tau=mean(G)+0.55*(max(G)-mean(G)); scipy.find_peaks(height=tau,distance=k_p)",
            "interval_decoder": "[peak-k_p, peak+k_p], inclusive, no clipping",
            "NMS": "the single shared scipy.find_peaks distance=k_p proposal; no post-verification NMS",
            "verification_position": "after the fixed proposal/interval/NMS geometry",
            "matching": "existing audit-side chronological greedy one-to-one IoU>=0.5",
        },
        "operator_details": {
            "P0_no_persistence": "all fixed candidates kept",
            "P1_peak_prominence": "scipy.signal.peak_prominences on G, matching current project feasibility code",
            "P2_topological_0d": {
                "filtration": "1D superlevel set, descending plateau values",
                "plateau": "contiguous equal values are one node; representative=floor((start+end)/2)",
                "elder_rule": "higher birth survives; tied birth uses smaller representative index",
                "global_max_death": "minimum(G)",
                "candidate_mapping": "containing birth plateau, otherwise nearest birth; ties higher birth then lower index",
            },
            "P3_scale_space_survival": {
                "scales": ["0.5*k_p", "1.0*k_p", "1.5*k_p", "2.0*k_p"],
                "input": "frozen raw temporal score; base reference peaks remain those proposed from G",
                "smoothing": "project boxcar convention with full width=round(2*scale), same-mode convolution",
                "peak_detection": "all scipy local maxima; never replaces base candidate",
                "tolerance": "+/-0.5*k_p",
                "tie_break": "nearest time, then higher smoothed height, then lower index",
            },
            "normalization": "per-video candidate robust_z=(x-median)/(1.4826*MAD+1e-8); n<3 or MAD<=1e-12 -> all zeros",
        },
        "threshold_protocol": {
            "grid": list(THRESHOLDS), "outer_split": "subject LOSO",
            "selection_target": "pooled training-subject raw Spotting F1",
            "tie_break": ["fewer FP", "fewer predictions", "lower grid index"],
            "held_out_subject_evaluations": 1, "test_labels_used_for_selection": False,
        },
        "pooled_metrics": pooled, "bootstrap": bootstraps,
        "error_attribution_totals": {
            operator: {
                key: int(sum(row[key] for row in attribution_rows
                             if row["operator"] == operator))
                for key in ("rescued_GT", "lost_GT", "removed_FP", "new_FP")
            }
            for operator in OPERATORS[1:]
        },
        "outputs": {name: str((destination / name).resolve()) for name in (
            "report.json", "operator_comparison.csv", "per_subject_metrics.csv",
            "candidate_scores.csv", "selected_thresholds.csv", "predictions.json",
            "error_attribution.csv")},
    }
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_summary(reports: list[dict]):
    labels = {
        "P0_no_persistence": "P0", "P1_peak_prominence": "Prominence",
        "P2_topological_0d": "Topological", "P3_scale_space_survival": "Scale-space",
    }
    print("\nBackbone | Dataset | P0 | Prominence | Topological | Scale-space")
    print("---|---|---:|---:|---:|---:")
    for report in reports:
        values = [report["pooled_metrics"][operator]["Raw_Spotting_F1"] for operator in OPERATORS]
        print(f"{report['backbone']} | {report['dataset']} | " + " | ".join(f"{value:.6f}" for value in values))
    print("\nTP/FP/FN, delta vs P0, delta vs current Prominence, paired 95% CI")
    for report in reports:
        print(f"\n{report['backbone']} / {report['dataset']}")
        p0 = report["pooled_metrics"]["P0_no_persistence"]["Raw_Spotting_F1"]
        p1 = report["pooled_metrics"]["P1_peak_prominence"]["Raw_Spotting_F1"]
        for operator in OPERATORS:
            metric = report["pooled_metrics"][operator]
            ci0 = report["bootstrap"][f"{operator}_vs_P0"]["CI95"]
            ci1 = report["bootstrap"][f"{operator}_vs_P1"]["CI95"]
            print(f"{labels[operator]}: {metric['TP']}/{metric['FP']}/{metric['FN']}; "
                  f"dP0={metric['Raw_Spotting_F1']-p0:+.6f} [{ci0[0]:+.6f},{ci0[1]:+.6f}]; "
                  f"dP1={metric['Raw_Spotting_F1']-p1:+.6f} [{ci1[0]:+.6f},{ci1[1]:+.6f}]")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [run_spec(spec, args.output_root) for spec in SPECS]
    deltas = {operator: [
        report["pooled_metrics"][operator]["Raw_Spotting_F1"]
        - report["pooled_metrics"]["P0_no_persistence"]["Raw_Spotting_F1"]
        for report in reports] for operator in OPERATORS[1:]}
    combined = {
        "study": "Persistence Operator Controlled Study", "same_candidate_verified": True,
        "reports": reports,
        "cross_group_stability": {operator: {
            "mean_delta_vs_P0": float(np.mean(values)),
            "worst_case_delta_vs_P0": float(np.min(values)),
            "improved_groups": int(sum(value > 0 for value in values)),
            "equal_groups": int(sum(abs(value) <= 1e-15 for value in values)),
            "worse_groups": int(sum(value < 0 for value in values)),
        } for operator, values in deltas.items()},
        "result_root": str(args.output_root.resolve()),
    }
    (args.output_root / "combined_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(reports)
    print(f"\nResults: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
