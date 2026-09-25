#!/usr/bin/env python3
"""Cache-only Recovery-Pool Feasibility Audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.signal._peak_finding_utils import PeakPropertyWarning


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/recovery_pool_feasibility_v1"
RAW_SAMMLV = Path(
    os.environ.get("SAMMLV_RAW_ROOT", "data/raw_prestitch_sammlv_full")
)
P_FACTORS = (0.25, 0.35, 0.45)
NATIVE_P = 0.55
IOU_THRESHOLD = 0.5
SPECS = (
    {"backbone": "ME-TST", "dataset": "SAMMLV",
     "cache": ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
     "sha": "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569",
     "expected": (53, 184, 106), "raw_windows": RAW_SAMMLV},
    {"backbone": "ME-TST", "dataset": "CASME_3",
     "cache": ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
     "sha": "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda",
     "expected": (81, 912, 777), "raw_windows": None},
    {"backbone": "BoostingVRME", "dataset": "SAMMLV",
     "cache": ROOT / "caches/boostingvrme/sammlv_curves.pkl",
     "sha": "abcda6477dcbf89a2083f6fdfe673b1686ca3cc100e62147b81a6806c67b05bc",
     "expected": (54, 158, 105), "raw_windows": None},
    {"backbone": "BoostingVRME", "dataset": "CASME_3",
     "cache": ROOT / "caches/boostingvrme/casme3_curves.pkl",
     "sha": "9775aa254149717b27e1ea8449a5fdf4e16a26b0a407ce501d6532dcef00a14a",
     "expected": (80, 830, 778), "raw_windows": None},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(np.asarray(values, dtype=np.float64),
                       np.ones(width, dtype=np.float64) / width, mode="same")


def robust_z(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return np.zeros_like(values)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad + 1e-8)


def load_records(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    k_p = int(payload["k_p"])
    if "records" in payload:
        records = [{"subject": str(row["subject"]), "video": str(row["video"]),
                    "score": np.asarray(row["score"], dtype=np.float64),
                    "samples": list(row["samples"])} for row in payload["records"]]
    else:
        records = []
        for subject in payload["subject_curves"]:
            for index, score in enumerate(subject["score"]):
                records.append({"subject": str(subject["subject"]),
                                "video": str(subject["videos"][index]),
                                "score": np.asarray(score, dtype=np.float64),
                                "samples": list(subject["samples"][index])})
    return payload, records, k_p


def proposal(record: dict, k_p: int, factor: float) -> tuple[np.ndarray, list[dict], float]:
    curve = moving_average(record["score"], 2 * k_p)
    threshold = float(curve.mean() + factor * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
    events = [{"peak": int(peak), "onset": int(peak - k_p),
               "offset": int(peak + k_p), "sources": set()} for peak in peaks]
    return curve, events, threshold


def interval_iou(event: dict, sample) -> float:
    left, right = int(event["onset"]), int(event["offset"])
    gt_left, gt_right = int(sample[0]), int(sample[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union > 0 else 0.0


def greedy_counts(events: list[dict], samples) -> tuple[int, int, int]:
    unmatched = set(range(len(samples)))
    tp = fp = 0
    for event in events:
        choices = [(interval_iou(event, samples[index]), index) for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(unmatched)


def attach_prominence(curve: np.ndarray, events: list[dict]):
    peaks = np.asarray([event["peak"] for event in events], dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PeakPropertyWarning)
        raw = peak_prominences(curve, peaks)[0] if len(peaks) else np.empty(0)
    normalized = robust_z(raw)
    for index, event in enumerate(events):
        event["prominence_raw"] = float(raw[index])
        event["prominence_norm"] = float(normalized[index])


def cluster_window_candidates(raw_candidates: list[dict], global_indices: np.ndarray,
                              k_p: int) -> list[dict]:
    """Support-first deterministic temporal NMS for window-local candidates."""
    by_peak = {}
    for candidate in raw_candidates:
        peak = int(candidate["peak"])
        item = by_peak.setdefault(peak, {"peak": peak, "windows": set(), "height": -np.inf})
        item["windows"].add(int(candidate["window_id"]))
        item["height"] = max(item["height"], float(candidate["height"]))
    remaining = list(by_peak.values())
    output = []
    while remaining:
        representative = min(
            remaining,
            key=lambda item: (-len(item["windows"]), -item["height"], item["peak"]),
        )
        cluster = [item for item in remaining
                   if abs(item["peak"] - representative["peak"]) <= k_p]
        support_windows = set().union(*(item["windows"] for item in cluster))
        peak = int(representative["peak"])
        eligible = sum(np.any((row >= peak - k_p) & (row <= peak + k_p))
                       for row in global_indices)
        output.append({
            "peak": peak, "onset": peak - k_p, "offset": peak + k_p,
            "sources": {"window"}, "support_window_count": len(support_windows),
            "eligible_window_count": int(eligible),
            "support_ratio": len(support_windows) / eligible if eligible else 0.0,
            "window_peak_height": float(representative["height"]),
        })
        clustered_ids = {id(item) for item in cluster}
        remaining = [item for item in remaining if id(item) not in clustered_ids]
    return sorted(output, key=lambda event: event["peak"])


def load_window_pool(raw_dir: Path, record: dict, k_p: int, manifest_hash: str | None):
    path = raw_dir / f"{record['subject']}_{record['video']}_raw_prestitch.npz"
    if not path.exists():
        raise RuntimeError(f"missing raw window file: {path}")
    if manifest_hash and sha256(path) != manifest_hash:
        raise RuntimeError(f"raw window hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_window_score"], dtype=np.float64)
        indices = np.asarray(data["global_indices"], dtype=np.int64)
        subject = str(np.asarray(data["subject"]).item())
        video = str(np.asarray(data["video"]).item())
    if subject != record["subject"] or video != record["video"]:
        raise RuntimeError(f"raw window identity mismatch: {path}")
    if raw.shape != indices.shape or raw.ndim != 2:
        raise RuntimeError(f"invalid raw window geometry: {path}")
    local_candidates = []
    for window_id, window in enumerate(raw):
        curve = moving_average(window, 2 * k_p)
        threshold = float(curve.mean() + NATIVE_P * (curve.max() - curve.mean()))
        peaks = find_peaks(curve, height=threshold, distance=k_p)[0].astype(int)
        for peak in peaks:
            local_candidates.append({"peak": int(indices[window_id, peak]),
                                     "window_id": window_id, "height": float(curve[peak])})
    unique = cluster_window_candidates(local_candidates, indices, k_p)
    return unique, len(local_candidates), str(path.resolve())


def nearest_event(events: list[dict], peak: int, tolerance: int):
    matches = [event for event in events if abs(int(event["peak"]) - int(peak)) <= tolerance]
    return min(matches, key=lambda event: (abs(event["peak"] - peak), event["peak"]), default=None)


def union_pool(native_events: list[dict], low_events: list[dict],
               window_events: list[dict], k_p: int) -> list[dict]:
    """Preserve native peaks, then merge recovery candidates without GT."""
    output = [{**event, "sources": {"native", "low_gate"}} for event in native_events]
    extras = []
    for event in low_events:
        if nearest_event(output, event["peak"], k_p) is None:
            extras.append({**event, "sources": {"low_gate"}})
    for event in window_events:
        native = nearest_event(output, event["peak"], k_p)
        if native is not None:
            native["sources"].add("window")
            native["support_window_count"] = max(
                native.get("support_window_count", 0), event["support_window_count"])
            native["eligible_window_count"] = event["eligible_window_count"]
            native["support_ratio"] = max(native.get("support_ratio", 0.0), event["support_ratio"])
        else:
            extras.append({**event, "sources": set(event["sources"])})
    while extras:
        representative = min(
            extras,
            key=lambda event: (-(event.get("support_window_count", 0)),
                               -event.get("window_peak_height", -np.inf), event["peak"]),
        )
        cluster = [event for event in extras
                   if abs(event["peak"] - representative["peak"]) <= k_p]
        merged = {**representative, "sources": set().union(*(event["sources"] for event in cluster))}
        window_members = [event for event in cluster if "window" in event["sources"]]
        if window_members:
            merged["support_window_count"] = max(event["support_window_count"] for event in window_members)
            merged["eligible_window_count"] = representative.get("eligible_window_count", 0)
            merged["support_ratio"] = max(event["support_ratio"] for event in window_members)
        output.append(merged)
        clustered_ids = {id(event) for event in cluster}
        extras = [event for event in extras if id(event) not in clustered_ids]
    return sorted(output, key=lambda event: event["peak"])


def coverage(events_by_video: dict, records: list[dict]):
    covered = set()
    for record in records:
        identity = (record["subject"], record["video"])
        for gt_index, sample in enumerate(record["samples"]):
            if any(interval_iou(event, sample) >= IOU_THRESHOLD for event in events_by_video[identity]):
                covered.add((*identity, gt_index))
    return covered


def best_iou(event: dict, samples) -> float:
    return max([interval_iou(event, sample) for sample in samples] or [0.0])


def iou_group(value: float) -> str:
    if value < 0.1:
        return "IoU_lt_0.1"
    if value < 0.5:
        return "IoU_0.1_to_0.5"
    return "IoU_ge_0.5"


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {"count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "q25": float(q25), "q75": float(q75)}


def pool_row(name: str, candidate_count: int, events_by_video: dict, covered: set,
             gt_count: int, native_by_video: dict, native_covered: set,
             tolerance: int, available=True):
    if not available:
        return {"pool": name, "status": "UNAVAILABLE_SOURCE", "candidate_count": None,
                "unique_candidate_count": None, "GT_count": gt_count, "coverable_GT": None,
                "oracle_recall": None, "additional_candidates": None,
                "additional_coverable_GT": None, "still_uncoverable_GT": None,
                "candidates_per_additional_GT": None}
    unique_count = sum(len(events) for events in events_by_video.values())
    additional = sum(
        nearest_event(native_by_video[identity], event["peak"], tolerance) is None
        for identity, events in events_by_video.items() for event in events
    )
    additional_gt = len(covered - native_covered)
    return {"pool": name, "status": "COMPLETE", "candidate_count": candidate_count,
            "unique_candidate_count": unique_count, "GT_count": gt_count,
            "coverable_GT": len(covered), "oracle_recall": len(covered) / gt_count,
            "additional_candidates": additional, "additional_coverable_GT": additional_gt,
            "still_uncoverable_GT": gt_count - len(covered),
            "candidates_per_additional_GT": additional / additional_gt if additional_gt else "inf"}


def clean_event(event: dict) -> dict:
    return {key: ("+".join(sorted(value)) if isinstance(value, set) else value)
            for key, value in event.items()}


def run_spec(spec: dict, output_root: Path):
    if sha256(spec["cache"]) != spec["sha"]:
        raise RuntimeError(f"cache identity mismatch: {spec['cache']}")
    payload, records, k_p = load_records(spec["cache"])
    identities = [(record["subject"], record["video"]) for record in records]
    native, low = {}, {factor: {} for factor in P_FACTORS}
    curves, thresholds = {}, {}
    baseline = Counter()
    for record in records:
        identity = (record["subject"], record["video"])
        curve, events, threshold = proposal(record, k_p, NATIVE_P)
        for event in events:
            event["sources"] = {"native"}
        attach_prominence(curve, events)
        native[identity], curves[identity], thresholds[identity] = events, curve, threshold
        baseline.update(dict(zip(("TP", "FP", "FN"), greedy_counts(events, record["samples"]))))
        for factor in P_FACTORS:
            _curve, low_events, _threshold = proposal(record, k_p, factor)
            for event in low_events:
                event["sources"] = {"low_gate"}
            attach_prominence(curve, low_events)
            low[factor][identity] = low_events
    observed = tuple(baseline[key] for key in ("TP", "FP", "FN"))
    if observed != spec["expected"]:
        raise RuntimeError(f"BLOCKED-BASELINE {spec['backbone']}/{spec['dataset']}: "
                           f"{observed} != {spec['expected']}")

    raw_available = spec["raw_windows"] is not None
    window, window_raw_count, raw_paths = {}, 0, []
    raw_provenance = {"status": "UNAVAILABLE_SOURCE"}
    if raw_available:
        manifest_path = spec["raw_windows"] / "full_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"status": "SAMMLV-RAW-WINDOW-SOURCE-READY", "complete": True,
                    "subjects": 29, "videos": 79, "videos_passed": 79, "videos_failed": 0}
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise RuntimeError(f"raw manifest gate failed: {key}")
        hashes = {(str(row["subject"]), str(row["video"])): row.get("sha256")
                  for row in manifest["video_reports"]}
        if set(hashes) != set(identities):
            raise RuntimeError("raw/compact video identity mismatch")
        for record in records:
            identity = (record["subject"], record["video"])
            events, raw_count, raw_path = load_window_pool(
                spec["raw_windows"], record, k_p, hashes[identity])
            attach_prominence(curves[identity], events)
            for event in events:
                event["absent_from_native"] = nearest_event(native[identity], event["peak"], k_p) is None
            window[identity] = events
            window_raw_count += raw_count
            raw_paths.append(raw_path)
        raw_provenance = {"status": "VERIFIED", "directory": str(spec["raw_windows"].resolve()),
                          "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
                          "videos": len(raw_paths)}

    union = {}
    if raw_available:
        for identity in identities:
            union[identity] = union_pool(native[identity], low[0.25][identity], window[identity], k_p)
            attach_prominence(curves[identity], union[identity])

    gt_count = sum(len(record["samples"]) for record in records)
    native_covered = coverage(native, records)
    low_covered = {factor: coverage(low[factor], records) for factor in P_FACTORS}
    window_covered = coverage(window, records) if raw_available else set()
    union_covered = coverage(union, records) if raw_available else set()
    rows = [pool_row("R0_native", sum(len(x) for x in native.values()), native,
                     native_covered, gt_count, native, native_covered, k_p)]
    for factor in P_FACTORS:
        rows.append(pool_row(f"R1_{factor:.2f}", sum(len(x) for x in low[factor].values()),
                             low[factor], low_covered[factor], gt_count, native, native_covered, k_p))
    rows.append(pool_row("R2_window_pre_stitch", window_raw_count, window, window_covered,
                         gt_count, native, native_covered, k_p, raw_available))
    rows.append(pool_row("R3_union_R1_0.25_R2",
                         sum(len(x) for x in low[0.25].values()) +
                         (sum(len(x) for x in window.values()) if raw_available else 0),
                         union, union_covered, gt_count, native, native_covered, k_p, raw_available))

    record_lookup = {(record["subject"], record["video"]): record for record in records}
    recovered_rows = []
    all_gt = {(*identity, index) for identity in identities
              for index in range(len(record_lookup[identity]["samples"]))}
    for gt_id in sorted(all_gt - native_covered):
        identity, gt_index = gt_id[:2], gt_id[2]
        sample = record_lookup[identity]["samples"][gt_index]
        by_low = gt_id in low_covered[0.25]
        by_window = gt_id in window_covered if raw_available else None
        candidates = ([event for event in low[0.25][identity]
                       if interval_iou(event, sample) >= IOU_THRESHOLD] +
                      ([event for event in window.get(identity, [])
                        if interval_iou(event, sample) >= IOU_THRESHOLD] if raw_available else []))
        if not candidates:
            continue
        best = min(candidates, key=lambda event: (-interval_iou(event, sample),
                                                  -event.get("support_window_count", 0), event["peak"]))
        recovered_rows.append({
            "backbone": spec["backbone"], "dataset": spec["dataset"],
            "subject": identity[0], "video": identity[1], "GT_onset": int(sample[0]),
            "GT_offset": int(sample[2]), "recovered_by_low_gate": by_low,
            "recovered_by_window": by_window,
            "recovered_by_both": (by_low and by_window) if raw_available else None,
            "best_candidate_peak": best["peak"],
            "best_candidate_interval": f"[{best['onset']},{best['offset']}]",
            "best_iou": interval_iou(best, sample),
            "window_support_count": best.get("support_window_count"),
            "window_support_ratio": best.get("support_ratio"),
        })

    candidate_rows = []
    pools = {"R0_native": native, **{f"R1_{factor:.2f}": low[factor] for factor in P_FACTORS}}
    if raw_available:
        pools.update({"R2_window_pre_stitch": window, "R3_union_R1_0.25_R2": union})
    for pool_name, bank in pools.items():
        for identity, events in bank.items():
            samples = record_lookup[identity]["samples"]
            for event in events:
                candidate_rows.append({"backbone": spec["backbone"], "dataset": spec["dataset"],
                    "subject": identity[0], "video": identity[1], "pool": pool_name,
                    **clean_event(event), "best_GT_IoU": best_iou(event, samples),
                    "best_GT_IoU_group": iou_group(best_iou(event, samples)),
                    "boundary_evidence_raw": None, "boundary_evidence_norm": None})

    source_rows = []
    if raw_available:
        for identity, events in union.items():
            for event in events:
                if "native" in event["sources"]:
                    continue
                source = ("both" if {"low_gate", "window"} <= event["sources"] else
                          "window_only" if "window" in event["sources"] else "low_gate_only")
                value = best_iou(event, record_lookup[identity]["samples"])
                source_rows.append({"backbone": spec["backbone"], "dataset": spec["dataset"],
                    "subject": identity[0], "video": identity[1], "peak": event["peak"],
                    "source": source, "best_GT_IoU": value, "IoU_group": iou_group(value),
                    "support_window_count": event.get("support_window_count"),
                    "eligible_window_count": event.get("eligible_window_count"),
                    "support_ratio": event.get("support_ratio")})
    else:
        for identity, events in low[0.25].items():
            for event in events:
                if nearest_event(native[identity], event["peak"], k_p) is not None:
                    continue
                value = best_iou(event, record_lookup[identity]["samples"])
                source_rows.append({"backbone": spec["backbone"], "dataset": spec["dataset"],
                    "subject": identity[0], "video": identity[1], "peak": event["peak"],
                    "source": "low_gate_only", "best_GT_IoU": value,
                    "IoU_group": iou_group(value), "support_window_count": None,
                    "eligible_window_count": None, "support_ratio": None})

    support_rows = []
    if raw_available:
        for group in ("IoU_lt_0.1", "IoU_0.1_to_0.5", "IoU_ge_0.5"):
            values = [event["support_ratio"] for identity, events in window.items() for event in events
                      if iou_group(best_iou(event, record_lookup[identity]["samples"])) == group]
            support_rows.append({"IoU_group": group, **distribution(values)})
    else:
        support_rows = [{"status": "UNAVAILABLE_SOURCE"}]
    boundary_rows = [{"status": "UNAVAILABLE_EXISTING_SCALAR_EVIDENCE",
                      "reason": "Existing TCB/LVB diagnostics change interval geometry; the RGR temporal descriptors are recognition-based and only cover a prior ME-TST/SAMMLV weak pool. No reusable B_raw/B_norm exists for arbitrary recovery candidates."}]

    destination = output_root / spec["backbone"].lower().replace("-", "_") / spec["dataset"].lower()
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "pool_coverage.csv", rows)
    write_csv(destination / "recovered_gt.csv", recovered_rows)
    write_csv(destination / "candidate_source_attribution.csv", source_rows)
    write_csv(destination / "window_support_distribution.csv", support_rows)
    write_csv(destination / "boundary_evidence_distribution.csv", boundary_rows)
    write_csv(destination / "candidates.csv", candidate_rows)
    best_r1 = min((row for row in rows if row["pool"].startswith("R1_")),
                  key=lambda row: (-row["coverable_GT"], row["unique_candidate_count"], row["pool"]))
    report = {
        "study": "Recovery-Pool Feasibility Audit", "backbone": spec["backbone"],
        "dataset": spec["dataset"], "status": "COMPLETE" if raw_available else "PARTIAL-R2-SOURCE-UNAVAILABLE",
        "cache": {"path": str(spec["cache"].resolve()), "sha256": spec["sha"],
                  "identity_verified": True, "baseline_expected": list(spec["expected"]),
                  "baseline_observed": list(observed), "baseline_status": "PASS"},
        "protocol": {"R0": "MA(score,2*k_p), p=0.55, find_peaks distance=k_p, interval p+-k_p",
                     "R1_factors": list(P_FACTORS), "R3_low_source": "fixed R1_0.25 (not GT-selected)",
                     "merge_tolerance": k_p, "GT_used_after_pool_generation_only": True,
                     "candidate_verification": False, "final_F1_optimization": False},
        "raw_window_provenance": raw_provenance,
        "pool_coverage": rows, "best_R1_coverage_diagnostic": best_r1,
        "recovered_GT_attribution": {
            "low_gate": sum(row["recovered_by_low_gate"] is True for row in recovered_rows),
            "window": sum(row["recovered_by_window"] is True for row in recovered_rows),
            "both": sum(row["recovered_by_both"] is True for row in recovered_rows),
        },
        "boundary_evidence_status": boundary_rows[0],
        "outputs": {name: str((destination / name).resolve()) for name in
                    ("report.json", "pool_coverage.csv", "recovered_gt.csv",
                     "candidate_source_attribution.csv", "window_support_distribution.csv",
                     "boundary_evidence_distribution.csv", "candidates.csv")},
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    return report


def print_summary(reports):
    print("Backbone | Dataset | Pool | Candidates | Oracle GT | Oracle Recall | +Candidates | +GT | Candidates/+GT")
    print("---|---|---|---:|---:|---:|---:|---:|---:")
    for report in reports:
        for row in report["pool_coverage"]:
            values = [row[key] for key in ("unique_candidate_count", "coverable_GT", "oracle_recall",
                                             "additional_candidates", "additional_coverable_GT",
                                             "candidates_per_additional_GT")]
            formatted = ["NA" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
                         for value in values]
            print(f"{report['backbone']} | {report['dataset']} | {row['pool']} | " + " | ".join(formatted))
    print("\nR0 Native | R1 best coverage diagnostic | R2 Window | R3 Union")
    for report in reports:
        lookup = {row["pool"]: row for row in report["pool_coverage"]}
        compact = [lookup["R0_native"], report["best_R1_coverage_diagnostic"],
                   lookup["R2_window_pre_stitch"], lookup["R3_union_R1_0.25_R2"]]
        text = [f"{row['pool']}={row['coverable_GT']}/{row['unique_candidate_count']}"
                if row["status"] == "COMPLETE" else f"{row['pool']}=UNAVAILABLE" for row in compact]
        print(f"{report['backbone']}/{report['dataset']}: " + " | ".join(text))


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [run_spec(spec, args.output_root) for spec in SPECS]
    coverage_rows = [{"backbone": report["backbone"], "dataset": report["dataset"], **row}
                     for report in reports for row in report["pool_coverage"]]
    write_csv(args.output_root / "coverage_summary.csv", coverage_rows)
    def row(report, pool):
        return next(item for item in report["pool_coverage"] if item["pool"] == pool)

    available = next(report for report in reports if report["raw_window_provenance"]["status"] == "VERIFIED")
    available_dir = (args.output_root / available["backbone"].lower().replace("-", "_") /
                     available["dataset"].lower())
    support = list(csv.DictReader((available_dir / "window_support_distribution.csv").open()))
    support_by_group = {item["IoU_group"]: item for item in support}
    attribution = available["recovered_GT_attribution"]
    recovery_union = attribution["low_gate"] + attribution["window"] - attribution["both"]
    window_only_recovery = attribution["window"] - attribution["both"]
    r0_values = {f"{report['backbone']}/{report['dataset']}": row(report, "R0_native")["oracle_recall"]
                 for report in reports}
    low_values = {f"{report['backbone']}/{report['dataset']}": {
        "additional_coverable_GT": report["best_R1_coverage_diagnostic"]["additional_coverable_GT"],
        "additional_candidates": report["best_R1_coverage_diagnostic"]["additional_candidates"],
        "candidates_per_additional_GT": report["best_R1_coverage_diagnostic"]["candidates_per_additional_GT"],
        "pool": report["best_R1_coverage_diagnostic"]["pool"],
    } for report in reports}
    combined = {"study": "Recovery-Pool Feasibility Audit",
                "status": "PARTIAL-R2-SOURCE-UNAVAILABLE-3-OF-4",
                "completed_R0_R1_groups": 4, "completed_R2_R3_groups": 1,
                "source_availability": {
                    "ME-TST/SAMMLV": "VERIFIED_RAW_PRE_STITCH_79_OF_79",
                    "ME-TST/CASME_3": "UNAVAILABLE_SOURCE",
                    "BoostingVRME/SAMMLV": "UNAVAILABLE_SOURCE",
                    "BoostingVRME/CASME_3": "UNAVAILABLE_SOURCE",
                    "missing_sources_are_not_interpreted_as_zero_candidates_or_zero_recovery": True,
                },
                "answers": {
                    "R0_native_oracle_recall": r0_values,
                    "maximum_low_gate_recovery": low_values,
                    "window_recovery_available_group": {
                        "group": f"{available['backbone']}/{available['dataset']}",
                        "additional_coverable_GT": row(available, "R2_window_pre_stitch")["additional_coverable_GT"],
                        "additional_candidates": row(available, "R2_window_pre_stitch")["additional_candidates"],
                        "candidates_per_additional_GT": row(available, "R2_window_pre_stitch")["candidates_per_additional_GT"],
                    },
                    "low_window_recovery_overlap_available_group": {
                        **attribution,
                        "jaccard": attribution["both"] / recovery_union if recovery_union else None,
                        "interpretation": "low overlap",
                    },
                    "R3_oracle_ceiling_available_group": {
                        "R0_recall": row(available, "R0_native")["oracle_recall"],
                        "R3_recall": row(available, "R3_union_R1_0.25_R2")["oracle_recall"],
                        "absolute_recall_gain": row(available, "R3_union_R1_0.25_R2")["oracle_recall"] - row(available, "R0_native")["oracle_recall"],
                        "additional_coverable_GT": row(available, "R3_union_R1_0.25_R2")["additional_coverable_GT"],
                    },
                    "window_support_separation": {
                        "useful_IoU_ge_0.5": support_by_group["IoU_ge_0.5"],
                        "pure_FP_IoU_lt_0.1": support_by_group["IoU_lt_0.1"],
                        "interpretation": "weak shift but heavy IQR overlap; not clear separation",
                    },
                    "boundary_evidence": "NOT AUDITABLE: no existing reusable scalar B_raw/B_norm for arbitrary recovery candidates",
                    "cross_group_consistency": "R1_0.25 raises oracle coverage in all four groups; magnitude and candidate cost are dataset/backbone specific. R2 cross-group consistency is not testable from available artifacts.",
                    "source_value": f"Neither observed source is wholly worthless: low-gate recovers GT in all four groups; window recovers {window_only_recovery} GT not recovered by low-gate in the one auditable group, but at high candidate cost.",
                },
                "reports": reports, "result_root": str(args.output_root.resolve()),
                "outputs": {"combined_report": str((args.output_root / "combined_report.json").resolve()),
                            "coverage_summary": str((args.output_root / "coverage_summary.csv").resolve())}}
    (args.output_root / "combined_report.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(reports)
    print(f"\nResults: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
