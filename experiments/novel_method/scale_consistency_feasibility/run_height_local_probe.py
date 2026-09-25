#!/usr/bin/env python3
"""Fast label-free H/L score probe on the frozen SAMM-LV cache.

H is the existing normalized reference-curve height component. This probe only
tests whether replacing G by H changes the candidate ranking enough to merit a
formal official-response run; it is not a nested result.
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
    records = [{
        "subject": str(source["subject"]), "video": str(source["video"]),
        "score": np.asarray(source["score"], dtype=float),
        "samples": [tuple(map(int, sample[:3])) for sample in source["samples"]],
    } for source in payload["records"]]
    return payload, records


def interval_iou(left: int, right: int, sample: tuple[int, int, int]) -> float:
    gl, gr = sample[0], sample[2]
    inter = max(0, min(right, gr) - max(left, gl) + 1)
    union = max(right, gr) - min(left, gl) + 1
    return inter / union if union else 0.0


def match_events(peaks: list[int], samples: list[tuple[int, int, int]], k: int) -> tuple[int, int, int]:
    unmatched = set(range(len(samples)))
    tp = fp = 0
    for peak in sorted(peaks):
        choices = [(interval_iou(int(peak) - k, int(peak) + k, samples[index]), index)
                   for index in unmatched]
        best_iou, best_index = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(unmatched)


def rows_for_reference(record: dict, reference: float) -> list[dict]:
    features = GLSDFeatures(record["score"], 5)
    peaks, evidence = features.evidence(reference, RHO)
    _, curve, _, _ = features.scale_data[reference]
    width = min(len(record["score"]), max(1, round(reference * 5)))
    window = max(1, round(RHO * 5))
    own_local = features.local_cache[width, window]
    own_peaks = features.scale_data[reference][0]
    own_index = {int(peak): index for index, peak in enumerate(own_peaks)}
    centre = float(np.mean(curve))
    denominator = max(float(np.max(curve) - centre), 1e-12)
    rows = []
    for peak, (_, local) in zip(peaks, evidence):
        index = own_index[int(peak)]
        height = max(0.0, (float(curve[int(peak)]) - centre) / denominator)
        rows.append({
            "peak": int(peak), "G": float(evidence[index, 0]),
            "L": float(local), "H": float(height),
            "HL": float((height + local) / 2.0),
        })
    return rows


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
    parser.add_argument("--output", type=Path, default=ROOT / "results/height_local_probe_sammlv")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output}")
    payload, records = load_records(args.cache)
    summary = []
    for record in records:
        for reference in (1.0, 1.5, 2.0):
            rows = rows_for_reference(record, reference)
            for tau in TAUS:
                for method in ("G", "L", "H", "HL"):
                    counts = match_events([row["peak"] for row in rows if row[method] >= tau], record["samples"], 5)
                    summary.append({"subject": record["subject"], "video": record["video"], "reference": reference,
                                    "tau": tau, "method": method, "TP": counts[0], "FP": counts[1], "FN": counts[2],
                                    "event_count": sum(row[method] >= tau for row in rows)})
    args.output.mkdir(parents=True)
    write_csv(args.output / "probe_summary.csv", summary)
    aggregate = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "event_count": 0})
    for row in summary:
        key = (float(row["reference"]), row["method"], float(row["tau"]))
        for field in aggregate[key]:
            aggregate[key][field] += int(row[field])
    diagnostics = []
    for reference in (1.0, 1.5, 2.0):
        for method in ("G", "L", "H", "HL"):
            candidates = [(tau, values) for (ref, meth, tau), values in aggregate.items()
                          if ref == reference and meth == method]
            tau, values = max(candidates, key=lambda item: (
                2 * item[1]["TP"] / (2 * item[1]["TP"] + item[1]["FP"] + item[1]["FN"]),
                -item[1]["FP"], -item[0]))
            diagnostics.append({"reference": reference, "method": method, "descriptive_best_tau": tau,
                                **values, "F1": 2 * values["TP"] / (2 * values["TP"] + values["FP"] + values["FN"])})
    write_csv(args.output / "descriptive_diagnostics.csv", diagnostics)
    metadata = {"status": "PROBE_ONLY", "dataset": payload["dataset"], "subjects": 29, "videos": 79,
                "gt": 159, "k_p": 5, "rho": RHO, "cache": str(args.cache.resolve()),
                "cache_sha256": sha256(args.cache), "methods": ["G", "L", "H", "HL"],
                "label_use": "diagnostic after candidate generation only; no parameter selection",
                "formal_experiment_started": False}
    (args.output / "probe_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("OUTPUT =", args.output)


if __name__ == "__main__":
    main()
