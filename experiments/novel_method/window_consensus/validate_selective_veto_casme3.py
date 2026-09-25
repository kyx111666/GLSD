#!/usr/bin/env python3
"""Validate frozen and nested selective window-consensus vetoes on CASME3.

This CPU validation is deliberately gated on a complete 462/462 GPU raw dump.
It must not be run during source preparation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import pickle
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPORTER_SOURCE = HERE / "export_raw_prestitch_casme3_full.py"
PRUNING_SOURCE = ROOT / "my_method/recognition_conservative_pruning_sammlv/run_pruning_gate.py"
CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
CONFIG_SOURCE = ROOT / "results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv"
OUTER_SOURCE = CONFIG_SOURCE.with_name("final_strong_native_outer_metrics.csv")
DEFAULT_OUT = ROOT / "results/selective_window_consensus_veto_casme3"
EXPECTED_CACHE_SHA256 = "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda"
EXPECTED_ANCHOR = (124, 1148, 734)
EXPECTED_SUBJECTS = 94
EXPECTED_VIDEOS = 462
K_P = 17
EPS = 1e-8
FROZEN_DELTA = 0.20
FROZEN_TAU = 0.80
DELTAS = (0.05, 0.10, 0.15, 0.20)
TAUS = (0.20, 0.40, 0.60, 0.80)
BASELINE = "Final Strong Native"
FROZEN = "Frozen Cross-Dataset"
NESTED = "Nested Same-Grid Calibration"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = import_module("casme3_raw_exporter_gate", EXPORTER_SOURCE)
pruning = import_module("casme3_veto_verified_native", PRUNING_SOURCE)
native = pruning.native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
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


def load_sources():
    if sha256(CACHE) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("CASME3 compact cache SHA-256 mismatch")
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    records = [dict(record) for record in payload["records"]]
    subjects = []
    for record in records:
        subject = str(record["subject"])
        if subject not in subjects:
            subjects.append(subject)
    observed = (len(subjects), len(records), int(payload["k_p"]))
    if observed != (EXPECTED_SUBJECTS, EXPECTED_VIDEOS, K_P):
        raise RuntimeError(f"CASME3 compact metadata mismatch: {observed}")
    configs = {}
    with CONFIG_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            configs[str(row["outer_subject"])] = {
                key: float(row[key]) for key in ("c_s", "p", "c_d", "c_b")
            }
    references = {}
    with OUTER_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["outer_subject"] != "aggregate":
                references[str(row["outer_subject"])] = {
                    key: int(row[key]) for key in ("TP", "FP", "FN", "event_count")
                }
    if set(configs) != set(subjects) or set(references) != set(subjects):
        raise RuntimeError("Final Strong Native subject mapping mismatch")
    return records, subjects, configs, references


def precheck_raw_manifest(raw_dir: Path):
    """Hard stop before compact-cache loading or any validation computation."""
    manifest_path = raw_dir / "full_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema": exporter.FULL_SCHEMA,
        "status": "CASME3-RAW-WINDOW-SOURCE-READY",
        "complete": True,
        "subjects": EXPECTED_SUBJECTS,
        "videos": EXPECTED_VIDEOS,
        "subjects_executed": EXPECTED_SUBJECTS,
        "videos_checked": EXPECTED_VIDEOS,
        "videos_passed": EXPECTED_VIDEOS,
        "videos_failed": 0,
        "model_eval": True,
        "torch_no_grad": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"raw manifest {key}={manifest.get(key)!r}, expected={expected!r}")
    geometry = manifest.get("geometry", {})
    expected_geometry = {
        "k": exporter.WINDOW_LENGTH,
        "stride_expression": "k//2",
        "stride": exporter.STRIDE,
        "batch_size": exporter.BATCH_SIZE,
        "frame_skip": exporter.FRAME_SKIP,
    }
    if geometry != expected_geometry:
        raise RuntimeError(f"raw manifest geometry mismatch: {geometry}")
    subject_reports = manifest.get("subject_reports", [])
    if len(subject_reports) != EXPECTED_SUBJECTS or any(
        row.get("checkpoint_loads") != 1 or row.get("subject_forward_passes") != 1
        for row in subject_reports
    ):
        raise RuntimeError("raw manifest lacks 94 one-load/one-traversal subject reports")
    reports = manifest.get("video_reports", [])
    if len(reports) != EXPECTED_VIDEOS or not all(row.get("passed") for row in reports):
        raise RuntimeError("raw manifest lacks 462 exact video reconstruction PASS records")
    return manifest_path, manifest


def load_raw_source(raw_dir: Path, records: list[dict]):
    manifest_path, manifest = precheck_raw_manifest(raw_dir)
    report_lookup = {
        (str(row["subject"]), str(row["video"])): row for row in manifest["video_reports"]
    }
    raw = {}
    for record in records:
        identity = (str(record["subject"]), str(record["video"]))
        report = report_lookup.get(identity)
        if report is None:
            raise RuntimeError(f"raw manifest is missing {identity}")
        path = raw_dir / report.get("relative_path", Path(report["path"]).name)
        if not path.is_file() and Path(report["path"]).is_file():
            path = Path(report["path"])
        if sha256(path) != report["sha256"]:
            raise RuntimeError(f"raw NPZ SHA mismatch: {identity}")
        reconstruction = exporter.reconstruct_and_compare(path)
        if not reconstruction["passed"]:
            raise RuntimeError(f"raw reconstruction failed: {identity}")
        with np.load(path, allow_pickle=False) as data:
            subject = str(np.asarray(data["subject"]).item())
            video = str(np.asarray(data["video"]).item())
            if (subject, video) != identity:
                raise RuntimeError(f"raw NPZ identity mismatch: {identity}")
            raw[identity] = {
                "score": np.asarray(data["raw_window_score"], dtype=np.float64),
                "global_indices": np.asarray(data["global_indices"], dtype=np.int64),
                "original": np.asarray(data["original_fresh_result_video"]),
            }
    if len(raw) != EXPECTED_VIDEOS:
        raise RuntimeError("raw source identity cardinality mismatch")
    return raw, manifest_path, manifest


def relative_agreement(raw: dict, peak: int):
    occurrences = np.argwhere(raw["global_indices"] == int(peak))
    rows = sorted({int(row) for row, _local in occurrences})
    if len(rows) == 1:
        return None, "SINGLE-VIEW-CANDIDATE"
    if len(rows) != 2:
        raise RuntimeError(f"unsupported peak coverage count={len(rows)}")
    values = []
    for row in rows:
        local = int(np.flatnonzero(raw["global_indices"][row] == peak)[0])
        values.append(float(raw["score"][row, local]))
    return min(values) / (max(values) + EPS), "PAIR"


def empty_counts():
    return {key: 0 for key in ("TP", "FP", "FN", "event_count")}


def add_counts(target: dict, source: dict) -> None:
    for key in target:
        target[key] += int(source[key])


def build_bank(records: list[dict], raw: dict, configs: dict):
    bank = []
    subject_baseline = {subject: empty_counts() for subject in configs}
    for record in records:
        subject, video = str(record["subject"]), str(record["video"])
        config = configs[subject]
        curve = native.moving_average(
            np.asarray(record["score"]), max(1, int(round(config["c_s"] * K_P)))
        )
        mean, maximum = float(curve.mean()), float(curve.max())
        events = native.tuned_decode(record, config, K_P)
        raw_record = dict(record)
        raw_record["score"] = raw[(subject, video)]["original"]
        raw_events = native.tuned_decode(raw_record, config, K_P)
        identity = lambda event: (int(event["onset"]), int(event["peak"]), int(event["offset"]))
        if [identity(event) for event in raw_events] != [identity(event) for event in events]:
            raise RuntimeError(f"raw/cache Final Strong Native event mismatch: {subject}/{video}")
        base_counts = native.evaluate(record, events)
        add_counts(subject_baseline[subject], base_counts)
        candidates = []
        for index, event in enumerate(events):
            changed = native.evaluate(record, events[:index] + events[index + 1 :])
            label, label_name = pruning.causal_label(base_counts, changed)
            peak = int(event["peak"])
            agreement, coverage = relative_agreement(raw[(subject, video)], peak)
            candidates.append(
                {
                    "index": index,
                    "event": event,
                    "peak": peak,
                    "H_s": float((curve[peak] - mean) / (maximum - mean + EPS) - config["p"]),
                    "A_ratio": agreement,
                    "coverage": coverage,
                    "label": label,
                    "label_name": label_name,
                }
            )
        bank.append(
            {
                "subject": subject,
                "video": video,
                "record": record,
                "events": events,
                "base_counts": base_counts,
                "candidates": candidates,
            }
        )
    total = empty_counts()
    for counts in subject_baseline.values():
        add_counts(total, counts)
    aggregate = native.metrics(total)
    if tuple(aggregate[key] for key in ("TP", "FP", "FN")) != EXPECTED_ANCHOR:
        raise RuntimeError(f"CASME3 Final Strong Native anchor mismatch: {aggregate}")
    return bank, subject_baseline, aggregate


def evaluate(bank: list[dict], subjects: set[str], delta, tau, method: str, trace=False):
    counts = empty_counts()
    diagnostics = Counter()
    traces = []
    for item in bank:
        if item["subject"] not in subjects:
            continue
        kept = []
        for candidate in item["candidates"]:
            pair = candidate["coverage"] == "PAIR"
            if method == BASELINE:
                eligible = pruned = False
            else:
                eligible = pair and candidate["H_s"] <= delta
                pruned = eligible and candidate["A_ratio"] < tau
            diagnostics["eligible"] += int(eligible)
            diagnostics["pruned"] += int(pruned)
            if pruned:
                diagnostics[f"pruned_{candidate['label_name']}"] += 1
            else:
                kept.append(candidate["event"])
            if trace:
                traces.append(
                    {
                        "method": method,
                        "subject": item["subject"],
                        "video": item["video"],
                        "candidate_index": candidate["index"],
                        "peak": candidate["peak"],
                        "H_s": candidate["H_s"],
                        "A_ratio": candidate["A_ratio"],
                        "coverage": candidate["coverage"],
                        "delta": delta,
                        "tau": tau,
                        "eligible": eligible,
                        "pruned": pruned,
                        "causal_native_label": candidate["label_name"],
                    }
                )
        add_counts(counts, native.evaluate(item["record"], kept))
    return native.metrics(counts), diagnostics, traces


def selection_key(row: dict, baseline: dict):
    result = row["metrics"]
    denominator = 2 * result["TP"] + result["FP"] + result["FN"]
    exact_f1 = Fraction(2 * result["TP"], denominator) if denominator else Fraction(0, 1)
    return (
        exact_f1,
        -result["FP"],
        -(baseline["TP"] - result["TP"]),
        -row["diagnostics"]["pruned"],
        -row["delta"],
        -row["tau"],
    )


def main() -> int:
    args = parse_args()
    # Deliberately gate before loading the compact cache or computing candidates.
    precheck_raw_manifest(args.raw_dir)
    records, subjects, configs, references = load_sources()
    raw, raw_manifest_path, raw_manifest = load_raw_source(args.raw_dir, records)
    bank, subject_baseline, anchor = build_bank(records, raw, configs)
    for subject in subjects:
        if subject_baseline[subject] != references[subject]:
            raise RuntimeError(f"CASME3 subject anchor mismatch: {subject}")

    frozen_rows = []
    nested_rows = []
    selected_rows = []
    inner_rows = []
    traces = []
    configs_grid = list(itertools.product(DELTAS, TAUS))
    per_subject = {
        (subject, delta, tau): evaluate(bank, {subject}, delta, tau, NESTED)[:2]
        for subject in subjects for delta, tau in configs_grid
    }
    for held in subjects:
        frozen_metrics, frozen_diag, frozen_trace = evaluate(
            bank, {held}, FROZEN_DELTA, FROZEN_TAU, FROZEN, trace=True
        )
        traces.extend(frozen_trace)
        frozen_rows.append({"outer_subject": held, "method": FROZEN, **frozen_metrics, "delta": FROZEN_DELTA, "tau": FROZEN_TAU, "eligible": frozen_diag["eligible"], "pruned": frozen_diag["pruned"]})
        train = [subject for subject in subjects if subject != held]
        baseline_train = empty_counts()
        for subject in train:
            add_counts(baseline_train, subject_baseline[subject])
        baseline_train = native.metrics(baseline_train)
        scored = []
        for config_index, (delta, tau) in enumerate(configs_grid):
            counts = empty_counts()
            diagnostics = Counter()
            for subject in train:
                metrics, diagnostic = per_subject[(subject, delta, tau)]
                add_counts(counts, metrics)
                diagnostics.update(diagnostic)
            metrics = native.metrics(counts)
            row = {"outer_subject": held, "config_index": config_index, "delta": delta, "tau": tau, "metrics": metrics, "diagnostics": diagnostics}
            scored.append(row)
            inner_rows.append({"outer_subject": held, "config_index": config_index, "delta": delta, "tau": tau, **{f"inner_{key}": value for key, value in metrics.items()}, "inner_eligible": diagnostics["eligible"], "inner_pruned": diagnostics["pruned"]})
        selected = max(scored, key=lambda row: selection_key(row, baseline_train))
        metrics, diagnostics, fold_trace = evaluate(
            bank, {held}, selected["delta"], selected["tau"], NESTED, trace=True
        )
        traces.extend(fold_trace)
        nested_rows.append({"outer_subject": held, "method": NESTED, **metrics, "delta": selected["delta"], "tau": selected["tau"], "eligible": diagnostics["eligible"], "pruned": diagnostics["pruned"]})
        selected_rows.append({"outer_subject": held, "selected_delta": selected["delta"], "selected_tau": selected["tau"], **{f"inner_{key}": value for key, value in selected["metrics"].items()}})

    baseline_rows = [{"outer_subject": subject, "method": BASELINE, **native.metrics(subject_baseline[subject]), "delta": None, "tau": None, "eligible": 0, "pruned": 0} for subject in subjects]
    rows = baseline_rows + frozen_rows + nested_rows
    aggregate = {}
    for method in (BASELINE, FROZEN, NESTED):
        counts = empty_counts()
        for row in rows:
            if row["method"] == method:
                add_counts(counts, row)
        aggregate[method] = native.metrics(counts)
    summary = {
        "status": "CASME3-SELECTIVE-VETO-VALIDATION-COMPLETE",
        "anchor": anchor,
        "aggregate": aggregate,
        "frozen_parameters": {"delta": FROZEN_DELTA, "tau": FROZEN_TAU},
        "nested_grid": {"delta": list(DELTAS), "tau": list(TAUS), "configs": 16},
        "selected_frequency": {
            "delta": dict(Counter(str(row["selected_delta"]) for row in selected_rows)),
            "tau": dict(Counter(str(row["selected_tau"]) for row in selected_rows)),
        },
        "provenance": {
            "compact_cache": str(CACHE),
            "compact_cache_sha256": sha256(CACHE),
            "raw_manifest": str(raw_manifest_path),
            "raw_manifest_sha256": sha256(raw_manifest_path),
            "raw_alignment_cache_sha256": raw_manifest["alignment_cache_sha256"],
            "Final_Strong_config_source": str(CONFIG_SOURCE),
        },
        "integrity": {
            "raw_source_462_of_462_exact": True,
            "baseline_anchor_exact": True,
            "frozen_cross_dataset_tuning": False,
            "nested_grid_expanded": False,
            "native_decoder_changed": False,
            "recognition_used": False,
            "logistic_regression_used": False,
            "backbone_forward": False,
        },
    }
    output = args.output_root / "outputs"
    write_csv(output / "casme3_selective_veto_outer_metrics.csv", rows)
    write_csv(output / "casme3_selective_veto_selected_configs.csv", selected_rows)
    write_csv(output / "casme3_selective_veto_inner_scores.csv", inner_rows)
    write_csv(output / "casme3_selective_veto_event_trace.csv", traces)
    (output / "casme3_selective_veto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "SELECTIVE_WINDOW_CONSENSUS_VETO_CASME3_CN.md").write_text(
        "# Selective Window-Consensus Veto — CASME3\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"Final Strong Native: {aggregate[BASELINE]}\n\n"
        f"Frozen Cross-Dataset (delta=0.20, tau=0.80): {aggregate[FROZEN]}\n\n"
        f"Nested Same-Grid Calibration: {aggregate[NESTED]}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
