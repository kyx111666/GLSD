#!/usr/bin/env python3
"""Label-free dual-reference consistency probe on a frozen ME-TST+ cache.

This is a development probe, not the formal nested outer experiment. It uses
the sealed GLSDFeatures implementation to obtain the two reference views and
only uses labels after candidate generation for diagnostics. The probe keeps
the first reference's candidate order as the anchor and requires a matching
peak from the second reference within the locked alignment tolerance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
WORKSPACE = HERE.parents[3]
SEALED = WORKSPACE / "official_response_ablation/fusion_screening/reference_residual/server_standalone/sealed"
sys.path.insert(0, str(SEALED))
from boosting_official_glds_full_LOCKED import GLSDFeatures  # noqa: E402


PAIRS = ((1.0, 1.5), (1.5, 2.0))
RHO = 3.0
TAUS = tuple(round(i / 100.0, 2) for i in range(1, 100))
IOU_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path) -> tuple[dict, list[dict]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("dataset") != "SAMMLV" or int(payload.get("k_p", -1)) != 5:
        raise RuntimeError("expected the frozen SAMMLV cache with k_p=5")
    records = []
    for source in payload["records"]:
        records.append({
            "subject": str(source["subject"]),
            "video": str(source["video"]),
            "score": np.asarray(source["score"], dtype=float),
            "samples": [tuple(map(int, sample[:3])) for sample in source["samples"]],
        })
    return payload, records


def interval_iou(left: int, right: int, sample: tuple[int, int, int]) -> float:
    gt_left, gt_right = sample[0], sample[2]
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def match_events(peaks: list[int], samples: list[tuple[int, int, int]], k: int) -> tuple[int, int, int, set[int]]:
    unmatched = set(range(len(samples)))
    tp = fp = 0
    matched = set()
    for peak in sorted(peaks):
        choices = [
            (interval_iou(int(peak) - k, int(peak) + k, samples[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(unmatched), matched


def reference_rows(record: dict, reference: float) -> list[dict]:
    features = GLSDFeatures(record["score"], 5)
    peaks, evidence = features.evidence(reference, RHO)
    rows = []
    for peak, (global_score, local_score) in zip(peaks, evidence):
        rows.append({
            "peak": int(peak),
            "global": float(global_score),
            "local": float(local_score),
            "score": float((global_score + local_score) / 2.0),
        })
    return rows


def align_rows(anchor: list[dict], other: list[dict], tolerance: int) -> list[dict]:
    """Align each anchor peak to one nearest other-view peak without reuse."""
    available = set(range(len(other)))
    aligned = []
    for row in anchor:
        choices = [
            (abs(int(other[index]["peak"]) - int(row["peak"])), -other[index]["local"], index)
            for index in available
            if abs(int(other[index]["peak"]) - int(row["peak"])) <= tolerance
        ]
        if not choices:
            continue
        _, _, index = min(choices)
        available.remove(index)
        aligned.append({
            "peak": int(row["peak"]),
            "anchor_score": float(row["score"]),
            "other_score": float(other[index]["score"]),
            "consensus_score": float(min(row["score"], other[index]["score"])),
            "distance": abs(int(other[index]["peak"]) - int(row["peak"])),
        })
    return aligned


def score_counts(rows: list[dict], score_name: str, tau: float, samples, k: int) -> dict:
    peaks = [row["peak"] for row in rows if row[score_name] >= tau]
    tp, fp, fn, matched = match_events(peaks, samples, k)
    return {"TP": tp, "FP": fp, "FN": fn, "event_count": len(peaks), "matched_gt": sorted(matched)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dual_reference_consistency_probe_sammlv")
    parser.add_argument("--subjects", nargs="+", default=["006", "007"])
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output}")

    payload, records = load_records(args.cache)
    selected = [record for record in records if record["subject"] in set(args.subjects)]
    if {record["subject"] for record in selected} != set(args.subjects):
        raise RuntimeError("one or more requested subjects are absent from the cache")
    tolerance = max(1, round(0.5 * int(payload["k_p"])))
    summary_rows = []
    feature_rows = []

    for pair in PAIRS:
        pair_name = f"{pair[0]:g}_{pair[1]:g}"
        for record in selected:
            first = reference_rows(record, pair[0])
            second = reference_rows(record, pair[1])
            aligned = align_rows(first, second, tolerance)
            for row in aligned:
                feature_rows.append({
                    "subject": record["subject"], "video": record["video"],
                    "pair": pair_name, **row,
                })
            base = {
                "subject": record["subject"], "video": record["video"], "pair": pair_name,
                "anchor_candidates": len(first), "other_candidates": len(second),
                "aligned_candidates": len(aligned),
                "alignment_rate_anchor": len(aligned) / len(first) if first else 0.0,
                "alignment_rate_other": len(aligned) / len(second) if second else 0.0,
            }
            for tau in TAUS:
                controls = {
                    "single_ref_anchor": score_counts(first, "score", tau, record["samples"], int(payload["k_p"])),
                    "single_ref_other": score_counts(second, "score", tau, record["samples"], int(payload["k_p"])),
                    "dual_reference_consensus": score_counts(aligned, "consensus_score", tau, record["samples"], int(payload["k_p"])),
                }
                for method, counts in controls.items():
                    summary_rows.append({
                        **base, "tau": tau, "method": method,
                        **{key: value for key, value in counts.items() if key != "matched_gt"},
                        "matched_gt": json.dumps(counts["matched_gt"]),
                    })

    args.output.mkdir(parents=True)
    write_csv(args.output / "probe_summary.csv", summary_rows)
    write_csv(args.output / "aligned_features.csv", feature_rows)
    metadata = {
        "status": "PROBE_ONLY",
        "dataset": payload["dataset"], "subjects": sorted(set(args.subjects)),
        "videos": len(selected), "gt": sum(len(r["samples"]) for r in selected),
        "cache": str(args.cache.resolve()), "cache_sha256": sha256(args.cache),
        "k_p": int(payload["k_p"]), "rho": RHO, "alignment_tolerance": tolerance,
        "pairs": [list(pair) for pair in PAIRS], "thresholds": list(TAUS),
        "anchor_policy": "first reference view; nearest unused second-view peak within tolerance",
        "label_use": "diagnostic after candidate generation only; no parameter selection",
        "formal_experiment_started": False,
    }
    (args.output / "probe_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Aggregate all videos before ranking diagnostic thresholds. This remains
    # descriptive only and is not an outer-fold selection result.
    aggregate = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "event_count": 0})
    for row in summary_rows:
        key = (row["pair"], row["method"], float(row["tau"]))
        for field in aggregate[key]:
            aggregate[key][field] += int(row[field])
    diagnostics = []
    for pair in sorted({row["pair"] for row in summary_rows}):
        for method in ("single_ref_anchor", "single_ref_other", "dual_reference_consensus"):
            rows = [
                (tau, counts) for (row_pair, row_method, tau), counts in aggregate.items()
                if row_pair == pair and row_method == method
            ]
            best_tau, best = max(
                rows,
                key=lambda item: (
                    2 * item[1]["TP"] / (2 * item[1]["TP"] + item[1]["FP"] + item[1]["FN"]),
                    -item[1]["FP"], -item[0],
                ),
            )
            diagnostics.append({
                "pair": pair, "method": method, "descriptive_best_tau": best_tau,
                **best,
                "F1": 2 * best["TP"] / (2 * best["TP"] + best["FP"] + best["FN"]),
            })
    write_csv(args.output / "descriptive_diagnostics.csv", diagnostics)
    print(json.dumps(metadata, indent=2))
    print("OUTPUT =", args.output)


if __name__ == "__main__":
    main()
