#!/usr/bin/env python3
"""Fail-closed functional preflight for BoostingVRME / CAS(ME)3 GLSD-90.

This phase-one adapter uses only APIs present in
``boosting_official_glds_full_casme3.py``. It does not run component
ablation or bootstrap. It writes recovery evidence only after every locked
functional replay and sealed-evidence comparison passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


DEFAULT_RUNNER = Path("/content/BoostingVRME/boosting_official_glds_full_casme3.py")
DEFAULT_OFFICIAL_REPO_ROOT = Path("/content/BoostingVRME")
DEFAULT_CACHE = Path(
    "/content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/"
    "official_response_cache/casme3_official_full_responses.pkl"
)
DEFAULT_EVIDENCE_DIR = Path(
    "/content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1"
)

EXPECTED_SUBJECTS = 94
EXPECTED_VIDEOS = 462
EXPECTED_GT = 853
FORBIDDEN_CONTROLLED_GT = 858
EXPECTED_CONFIGS = 90
EXPECTED_OUTER_FOLDS = 94
EXPECTED_INNER_SUBJECTS = 93
EXPECTED_INNER_ROWS = 94 * 90 * 93
EXPECTED_SEARCH_ROWS = 94 * 90
EXPECTED_NATIVE_RAW = (84, 793, 769)
EXPECTED_NATIVE_FULL = (84, 786, 769)
EXPECTED_GLSD_RAW = (120, 1113, 733)
EXPECTED_GLSD_FULL = (120, 1110, 733)

INNER_FIELDS = [
    "outer_subject_index", "outer_subject", "config_id", "reference_scale",
    "local_radius", "threshold", "inner_subject_index", "inner_subject",
    "tp", "fp", "fn",
]
SEARCH_FIELDS = [
    "outer_subject_index", "outer_subject", "config_id", "reference_scale",
    "local_radius", "threshold", "grid_order", "pooled_inner_tp",
    "pooled_inner_fp", "pooled_inner_fn", "precision", "recall", "f1",
    "rank", "final_rank", "selected",
]
SELECTED_FIELDS = [
    "outer_subject_index", "outer_subject", "inner_subject_count",
    "inner_subject_indices", "inner_subjects", "config_id", "reference_scale",
    "local_radius", "threshold", "grid_order", "pooled_inner_tp",
    "pooled_inner_fp", "pooled_inner_fn", "precision", "recall", "f1", "rank",
]
SUBJECT_FIELDS = [
    "outer_subject_index", "outer_subject", "config_id", "reference_scale",
    "local_radius", "threshold", "raw_tp", "raw_fp", "raw_fn",
    "raw_precision", "raw_recall", "raw_f1", "full_tp", "full_fp",
    "full_fn", "full_precision", "full_recall", "full_f1",
]
VIDEO_FIELDS = [
    "subject_index", "subject", "video_index_within_subject",
    "video_index_global", "video", "config_id", "gt_count", "prediction_count",
    "raw_tp", "raw_fp", "raw_fn", "neutral_tp", "neutral_fp", "full_tp",
    "full_fp", "full_fn",
]
PREDICTION_FIELDS = [
    "subject_index", "subject", "video_index_within_subject",
    "video_index_global", "video", "config_id", "prediction_index", "onset",
    "offset", "peak", "matched_gt_index", "matched_gt", "best_iou_inclusive",
    "matched_iou_inclusive", "raw_status", "recognition_prediction",
    "recognition_target", "neutral", "synergy_action",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing locked {label}: {path}")


def as_counts(value, label: str) -> tuple[int, int, int]:
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        try:
            value = (lowered["tp"], lowered["fp"], lowered["fn"])
        except KeyError as exc:
            raise RuntimeError(f"{label} is not a TP/FP/FN mapping") from exc
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise RuntimeError(f"{label} must contain TP/FP/FN, got {result}")
    return result


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: actual={actual!r}, expected={expected!r}")


def require_signature(function, parameters: tuple[str, ...], label: str) -> None:
    actual = tuple(inspect.signature(function).parameters)
    require_equal(actual, parameters, f"runner API signature {label}")


def import_runner(path: Path, official_repo_root: Path):
    spec = importlib.util.spec_from_file_location("locked_boosting_casme3_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path[:0] = [str(official_repo_root), str(path.parent)]
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        sys.path[:] = old_path
    return module


def validate_real_runner_api(runner) -> None:
    required = {
        "load_cache": ("path",),
        "group_videos": ("videos",),
        "native_exact_replay": ("cache", "grouped"),
        "build_grid": (),
        "decode_subject": (
            "subject_index", "config_item", "grouped", "feature_bank",
            "final_samples", "final_emotions", "cache_config", "need_recognition",
        ),
        "subject_prediction_evidence": (
            "subject_index", "subject_name", "config_item", "decoded", "grouped",
            "final_samples", "cache_config",
        ),
        "metrics": ("counts",),
        "add_counts": ("items",),
        "official_metric_counts": ("metric", "total_gt"),
        "normalize": ("obj",),
        "patched_candidate_source": ("selected_per_video",),
    }
    for name, parameters in required.items():
        function = getattr(runner, name, None)
        if not callable(function):
            raise RuntimeError(f"real runner API is missing callable {name}")
        require_signature(function, parameters, name)

    feature_class = getattr(runner, "GLSDFeatures", None)
    if not inspect.isclass(feature_class):
        raise RuntimeError("real runner API is missing GLSDFeatures")
    require_signature(feature_class, ("response", "k"), "GLSDFeatures")
    require_signature(
        feature_class.local_values, ("self", "width", "rho"),
        "GLSDFeatures.local_values",
    )
    require_signature(
        feature_class.scores, ("self", "reference_scale", "rho"),
        "GLSDFeatures.scores",
    )
    require_signature(
        feature_class.selected_peaks,
        ("self", "reference_scale", "rho", "tau"),
        "GLSDFeatures.selected_peaks",
    )

def validate_manifest(manifest_path: Path, cache_sha: str, runner_sha: str) -> dict:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("run_manifest.json must contain a JSON object")

    exact = {
        "dataset": "CASME_3",
        "subject_count": EXPECTED_SUBJECTS,
        "video_count": EXPECTED_VIDEOS,
        "gt_event_count": EXPECTED_GT,
        "glsd_grid_size": EXPECTED_CONFIGS,
        "inner_evidence_rows": EXPECTED_INNER_ROWS,
        "glsd_search_rows": EXPECTED_SEARCH_ROWS,
    }
    for key, expected in exact.items():
        if key not in manifest:
            raise RuntimeError(f"run_manifest.json lacks required field {key}")
        require_equal(manifest[key], expected, f"manifest {key}")

    require_equal(
        manifest.get("selection_tie_break"),
        ["higher_F1", "higher_precision", "fewer_FP", "fixed_grid_order"],
        "manifest selection_tie_break",
    )
    require_equal(
        manifest.get("glsd_grid_constants"),
        {
            "reference_scales": [1.0, 1.5, 2.0],
            "local_radii": [1.0, 2.0, 3.0],
            "thresholds": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75],
        },
        "manifest glsd_grid_constants",
    )
    require_equal(
        as_counts(manifest.get("native_expected_counts", {}).get("raw", ()),
                  "manifest native raw"),
        EXPECTED_NATIVE_RAW,
        "manifest Native raw counts",
    )
    require_equal(
        as_counts(manifest.get("native_expected_counts", {}).get("full", ()),
                  "manifest native full"),
        EXPECTED_NATIVE_FULL,
        "manifest Native full counts",
    )
    require_equal(
        as_counts(manifest.get("glsd_final_raw_counts", ()), "manifest GLSD raw"),
        EXPECTED_GLSD_RAW,
        "manifest GLSD raw counts",
    )
    require_equal(
        as_counts(manifest.get("glsd_final_full_counts", ()), "manifest GLSD full"),
        EXPECTED_GLSD_FULL,
        "manifest GLSD full counts",
    )

    historical_cache_sha = str(manifest.get("cache_sha256", "")).lower()
    if not historical_cache_sha:
        raise RuntimeError("run_manifest.json lacks cache_sha256")
    if cache_sha.lower() != historical_cache_sha:
        raise RuntimeError(
            "HARD FAIL: cache_sha256 mismatch: "
            f"current={cache_sha}, historical={historical_cache_sha}"
        )

    historical_core_sha = str(manifest.get("core_script_sha256", "")).lower()
    if not historical_core_sha:
        raise RuntimeError("run_manifest.json lacks core_script_sha256")
    historical_wrapper_sha = str(manifest.get("script_sha256", "")).lower()
    if not historical_wrapper_sha:
        raise RuntimeError("run_manifest.json lacks historical wrapper script_sha256")

    return {
        "manifest": manifest,
        "historical_cache_sha256": historical_cache_sha,
        "historical_core_runner_sha256": historical_core_sha,
        "historical_execution_wrapper_sha256": historical_wrapper_sha,
        "current_runner_sha256": runner_sha.lower(),
        "runner_sha_match": runner_sha.lower() == historical_core_sha,
    }


def validate_universe(cache: dict, videos: list[dict], grouped: list[list[dict]]):
    require_equal(cache.get("dataset"), "CASME_3", "cache dataset")
    require_equal(int(cache.get("subject_count", -1)), EXPECTED_SUBJECTS,
                  "cache subject_count")
    require_equal(int(cache.get("video_count", -1)), EXPECTED_VIDEOS,
                  "cache video_count")
    require_equal(len(videos), EXPECTED_VIDEOS, "loaded video count")
    require_equal(len(grouped), EXPECTED_SUBJECTS, "grouped subject count")
    require_equal(sum(len(subject) for subject in grouped), EXPECTED_VIDEOS,
                  "grouped video count")

    global_indices = [int(video["video_index_global"]) for video in videos]
    require_equal(global_indices, list(range(EXPECTED_VIDEOS)),
                  "global video ordering")
    subject_names = []
    for subject_index, subject in enumerate(grouped):
        if not subject:
            raise RuntimeError(f"empty official subject bucket {subject_index}")
        indices = {int(video["subject_index"]) for video in subject}
        require_equal(indices, {subject_index}, f"subject bucket {subject_index}")
        names = {str(video["subject"]) for video in subject}
        if len(names) != 1:
            raise RuntimeError(f"subject bucket {subject_index} has inconsistent names")
        subject_names.append(str(subject[0]["subject"]))
    require_equal(len(set(subject_names)), EXPECTED_SUBJECTS,
                  "unique official subject names")

    gt_count = sum(len(video["gt_intervals"]) for video in videos)
    if gt_count == FORBIDDEN_CONTROLLED_GT:
        raise RuntimeError("HARD FAIL: forbidden 858-event controlled cache detected")
    require_equal(gt_count, EXPECTED_GT, "official GT count")
    return subject_names, gt_count


def validate_native(runner, cache: dict, grouped: list[list[dict]]) -> dict:
    native = runner.native_exact_replay(cache, grouped)
    if not isinstance(native, dict):
        raise RuntimeError("native_exact_replay did not return its audit dictionary")
    required = {
        "passed", "raw_counts", "full_counts", "prediction_exact",
        "recognition_exact", "matched_gt_exact",
    }
    missing = sorted(required.difference(native))
    if missing:
        raise RuntimeError(f"native_exact_replay result lacks fields {missing}")
    for key in ("passed", "prediction_exact", "recognition_exact", "matched_gt_exact"):
        if native[key] is not True:
            raise RuntimeError(f"Native exact replay gate {key}=False")
    require_equal(as_counts(native["raw_counts"], "Native raw"), EXPECTED_NATIVE_RAW,
                  "Native raw counts")
    require_equal(as_counts(native["full_counts"], "Native full"), EXPECTED_NATIVE_FULL,
                  "Native full counts")
    return native


def build_main_state(runner, cache: dict, grouped: list[list[dict]]):
    cache_config = cache["config"]
    k = int(cache_config["k_p"])
    if k < 1:
        raise RuntimeError(f"invalid official cache k_p={k}")
    final_samples = [
        [video["gt_intervals"] for video in subject] for subject in grouped
    ]
    final_emotions = [
        [video["emotion_labels"] for video in subject] for subject in grouped
    ]
    feature_bank = [
        [runner.GLSDFeatures(video["spot_response"], k) for video in subject]
        for subject in grouped
    ]
    return cache_config, final_samples, final_emotions, feature_bank


def replay_subject_config_table(
    runner, grid: list[dict], grouped: list[list[dict]], feature_bank,
    final_samples, final_emotions, cache_config,
) -> np.ndarray:
    table = np.empty((EXPECTED_CONFIGS, EXPECTED_SUBJECTS, 3), dtype=np.int64)
    for config_item in grid:
        config_id = int(config_item["config_id"])
        for subject_index in range(EXPECTED_SUBJECTS):
            decoded = runner.decode_subject(
                subject_index, config_item, grouped, feature_bank, final_samples,
                final_emotions, cache_config, need_recognition=False,
            )
            table[config_id, subject_index] = as_counts(
                decoded["raw_counts"],
                f"raw config={config_id} subject={subject_index}",
            )
        print(f"functional raw replay config {config_id + 1:02d}/90")
    require_equal(tuple(table.shape), (90, 94, 3), "subject/config table shape")
    return table


def inner_rows(
    subject_names: list[str], grid: list[dict], table: np.ndarray,
) -> Iterator[dict]:
    for outer in range(EXPECTED_OUTER_FOLDS):
        for config_item in grid:
            config_id = int(config_item["config_id"])
            for inner in range(EXPECTED_SUBJECTS):
                if inner == outer:
                    continue
                tp, fp, fn = (int(value) for value in table[config_id, inner])
                yield {
                    "outer_subject_index": outer,
                    "outer_subject": subject_names[outer],
                    "config_id": config_id,
                    "reference_scale": config_item["reference_scale"],
                    "local_radius": config_item["local_radius"],
                    "threshold": config_item["threshold"],
                    "inner_subject_index": inner,
                    "inner_subject": subject_names[inner],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }


def rebuild_outer_selection(runner, subject_names, grid, table):
    search_rows = []
    selected_rows = []
    selection_trace = []
    selected_by_outer = {}

    for outer in range(EXPECTED_OUTER_FOLDS):
        inner_subjects = [index for index in range(EXPECTED_SUBJECTS) if index != outer]
        require_equal(len(inner_subjects), EXPECTED_INNER_SUBJECTS,
                      f"outer {outer} inner subject count")
        if outer in inner_subjects:
            raise RuntimeError(f"outer subject {outer} leaked into inner LOSO")

        candidates = []
        for config_item in grid:
            config_id = int(config_item["config_id"])
            pooled = runner.add_counts(
                [tuple(table[config_id, inner]) for inner in inner_subjects]
            )
            metric = runner.metrics(pooled)
            candidates.append({
                **config_item,
                "pooled_inner_tp": metric["tp"],
                "pooled_inner_fp": metric["fp"],
                "pooled_inner_fn": metric["fn"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
            })

        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(row["f1"]), -float(row["precision"]),
                int(row["pooled_inner_fp"]), int(row["grid_order"]),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        winner = ranked[0]
        selected_by_outer[outer] = dict(winner)
        rank_lookup = {
            int(row["config_id"]): int(row["rank"]) for row in ranked
        }

        for row in candidates:
            config_id = int(row["config_id"])
            search_rows.append({
                "outer_subject_index": outer,
                "outer_subject": subject_names[outer],
                **row,
                "final_rank": rank_lookup[config_id],
                "selected": int(config_id == int(winner["config_id"])),
            })

        selected_rows.append({
            "outer_subject_index": outer,
            "outer_subject": subject_names[outer],
            "inner_subject_count": len(inner_subjects),
            "inner_subject_indices": json.dumps(inner_subjects),
            "inner_subjects": json.dumps([subject_names[i] for i in inner_subjects]),
            **winner,
        })
        selection_trace.append({
            "outer_subject_index": outer,
            "outer_subject": subject_names[outer],
            "inner_subject_indices": inner_subjects,
            "inner_subjects": [subject_names[i] for i in inner_subjects],
            "tie_break": [
                "higher_F1", "higher_precision", "fewer_FP", "fixed_grid_order",
            ],
            "winner_config_id": int(winner["config_id"]),
            "winner_rank": int(winner["rank"]),
            "candidates": ranked,
        })

    require_equal(len(search_rows), EXPECTED_SEARCH_ROWS, "GLSD search rows")
    require_equal(len(selected_rows), EXPECTED_OUTER_FOLDS, "selected outer rows")
    require_equal(len(selection_trace), EXPECTED_OUTER_FOLDS, "selection trace folds")
    return search_rows, selected_rows, selection_trace, selected_by_outer


def csv_cell(value) -> str:
    if value is None:
        return ""
    return str(value)


def compare_csv_exact(
    path: Path, expected_rows: Iterable[dict], fields: list[str], label: str,
) -> dict:
    require_file(path, label)
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_equal(reader.fieldnames, fields, f"{label} header")
        for count, expected in enumerate(expected_rows, start=1):
            try:
                actual = next(reader)
            except StopIteration as exc:
                raise RuntimeError(f"{label} ended before row {count}") from exc
            expected_text = {field: csv_cell(expected[field]) for field in fields}
            if actual != expected_text:
                differences = {
                    field: (actual.get(field), expected_text[field])
                    for field in fields if actual.get(field) != expected_text[field]
                }
                raise RuntimeError(
                    f"{label} row {count} mismatch: {differences}"
                )
        try:
            extra = next(reader)
        except StopIteration:
            extra = None
        if extra is not None:
            raise RuntimeError(f"{label} has extra row after {count}: {extra}")
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "rows": count, "fields": fields, "exact_match": True,
    }


def compare_json_exact(path: Path, expected, label: str, runner) -> dict:
    require_file(path, label)
    actual = load_json(path)
    normalized_expected = runner.normalize(expected)
    if actual != normalized_expected:
        raise RuntimeError(f"{label} differs from functional replay")
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "exact_match": True,
    }


def replay_selected_outer(
    runner, subject_names, grouped, feature_bank, final_samples, final_emotions,
    cache_config, selected_by_outer, table,
):
    per_subject_rows = []
    per_video_rows = []
    per_prediction_rows = []
    outer_raw_counts = []
    outer_full_counts = []

    for outer in range(EXPECTED_OUTER_FOLDS):
        config_item = selected_by_outer[outer]
        decoded = runner.decode_subject(
            outer, config_item, grouped, feature_bank, final_samples,
            final_emotions, cache_config, need_recognition=True,
        )
        raw = as_counts(decoded["raw_counts"], f"outer {outer} raw")
        full = as_counts(decoded["full_counts"], f"outer {outer} full")
        config_id = int(config_item["config_id"])
        require_equal(raw, tuple(int(x) for x in table[config_id, outer]),
                      f"outer {outer} selected raw replay")

        prediction_rows, video_rows, _video_records = runner.subject_prediction_evidence(
            outer, subject_names[outer], config_item, decoded, grouped,
            final_samples, cache_config,
        )
        video_raw = (
            sum(int(row["raw_tp"]) for row in video_rows),
            sum(int(row["raw_fp"]) for row in video_rows),
            sum(int(row["raw_fn"]) for row in video_rows),
        )
        video_full = (
            sum(int(row["full_tp"]) for row in video_rows),
            sum(int(row["full_fp"]) for row in video_rows),
            sum(int(row["full_fn"]) for row in video_rows),
        )
        require_equal(video_raw, raw, f"outer {outer} independent raw recount")
        require_equal(video_full, full, f"outer {outer} independent full recount")

        raw_metric = runner.metrics(raw)
        full_metric = runner.metrics(full)
        per_subject_rows.append({
            "outer_subject_index": outer,
            "outer_subject": subject_names[outer],
            "config_id": config_id,
            "reference_scale": config_item["reference_scale"],
            "local_radius": config_item["local_radius"],
            "threshold": config_item["threshold"],
            "raw_tp": raw_metric["tp"],
            "raw_fp": raw_metric["fp"],
            "raw_fn": raw_metric["fn"],
            "raw_precision": raw_metric["precision"],
            "raw_recall": raw_metric["recall"],
            "raw_f1": raw_metric["f1"],
            "full_tp": full_metric["tp"],
            "full_fp": full_metric["fp"],
            "full_fn": full_metric["fn"],
            "full_precision": full_metric["precision"],
            "full_recall": full_metric["recall"],
            "full_f1": full_metric["f1"],
        })
        per_video_rows.extend(video_rows)
        per_prediction_rows.extend(prediction_rows)
        outer_raw_counts.append(raw)
        outer_full_counts.append(full)

    require_equal(len(per_subject_rows), EXPECTED_SUBJECTS, "per-subject replay rows")
    require_equal(len(per_video_rows), EXPECTED_VIDEOS, "per-video replay rows")
    final_raw = as_counts(runner.add_counts(outer_raw_counts), "GLSD aggregate raw")
    final_full = as_counts(runner.add_counts(outer_full_counts), "GLSD aggregate full")
    require_equal(final_raw, EXPECTED_GLSD_RAW, "GLSD-90 raw counts")
    require_equal(final_full, EXPECTED_GLSD_FULL, "GLSD-90 full counts")
    return (
        per_subject_rows, per_video_rows, per_prediction_rows, final_raw, final_full,
    )


def validate_grid(runner) -> list[dict]:
    grid = runner.build_grid()
    if not isinstance(grid, list):
        raise RuntimeError("build_grid() did not return a list")
    require_equal(len(grid), EXPECTED_CONFIGS, "configuration count")
    fields = {"config_id", "reference_scale", "local_radius", "threshold", "grid_order"}
    for index, config_item in enumerate(grid):
        if not isinstance(config_item, dict):
            raise RuntimeError(f"grid config {index} is not a dict")
        require_equal(set(config_item), fields, f"grid config {index} fields")
        require_equal(int(config_item["config_id"]), index, f"grid config {index} id")
        require_equal(int(config_item["grid_order"]), index,
                      f"grid config {index} order")
    return grid


def write_recovery_outputs(
    output: Path, recovery: dict, table: np.ndarray, selected_rows: list[dict],
    per_subject_rows: list[dict], runner,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    table_path = output / "functional_glsd90_subject_config_raw_counts.npy"
    selected_path = output / "functional_outer_selected_configs.json"
    subject_path = output / "functional_per_subject_counts.csv"
    manifest_path = output / "recovery_manifest.json"

    np.save(table_path, table)
    selected_path.write_text(
        json.dumps(runner.normalize(selected_rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with subject_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBJECT_FIELDS)
        writer.writeheader()
        writer.writerows(per_subject_rows)

    recovery["recovery_artifacts"] = {
        table_path.name: {
            "sha256": sha256_file(table_path), "shape": list(table.shape),
            "dtype": str(table.dtype),
        },
        selected_path.name: {
            "sha256": sha256_file(selected_path), "rows": len(selected_rows),
        },
        subject_path.name: {
            "sha256": sha256_file(subject_path), "rows": len(per_subject_rows),
        },
    }
    manifest_path.write_text(
        json.dumps(recovery, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_preflight(args) -> None:
    runner_path = args.runner.resolve()
    repo_root = args.official_repo_root.resolve()
    cache_path = args.cache.resolve()
    evidence_dir = args.evidence_dir.resolve()

    required = {
        "runner": runner_path,
        "cache": cache_path,
        "manifest": evidence_dir / "run_manifest.json",
        "per_subject": evidence_dir / "per_subject_counts.csv",
        "selected_json": evidence_dir / "outer_selected_configs.json",
        "trace": evidence_dir / "outer_selection_trace.json",
        "inner": evidence_dir / "glsd_search_inner_fold_counts.csv",
    }
    for label in ("runner", "cache", "manifest"):
        require_file(required[label], label)
    if not repo_root.is_dir():
        raise RuntimeError(f"official repository root is not a directory: {repo_root}")

    runner_sha = sha256_file(runner_path)
    cache_sha = sha256_file(cache_path)
    provenance = validate_manifest(required["manifest"], cache_sha, runner_sha)
    print(f"cache path = {cache_path}")
    print(f"cache SHA256 = {cache_sha}")
    print(
        "historical core runner SHA256 = "
        f"{provenance['historical_core_runner_sha256']}"
    )
    print(f"current runner SHA256 = {runner_sha}")
    if not provenance["runner_sha_match"]:
        print("runner SHA mismatch: continuing complete functional exact replay")

    for label in ("per_subject", "selected_json", "trace", "inner"):
        require_file(required[label], label)

    # A runner SHA mismatch is evidence, not a bypass: the complete replay below still runs.
    runner = import_runner(runner_path, repo_root)
    validate_real_runner_api(runner)
    cache, videos = runner.load_cache(cache_path)
    grouped = runner.group_videos(videos)
    subject_names, gt_count = validate_universe(cache, videos, grouped)
    native = validate_native(runner, cache, grouped)
    grid = validate_grid(runner)
    cache_config, final_samples, final_emotions, feature_bank = build_main_state(
        runner, cache, grouped
    )
    table = replay_subject_config_table(
        runner, grid, grouped, feature_bank, final_samples, final_emotions,
        cache_config,
    )

    evidence_checks = {}
    evidence_checks["inner_fold_counts"] = compare_csv_exact(
        required["inner"], inner_rows(subject_names, grid, table), INNER_FIELDS,
        "glsd_search_inner_fold_counts.csv",
    )
    require_equal(evidence_checks["inner_fold_counts"]["rows"], EXPECTED_INNER_ROWS,
                  "inner evidence row count")

    search_rows, selected_rows, trace, selected_by_outer = rebuild_outer_selection(
        runner, subject_names, grid, table
    )
    evidence_checks["outer_selected_configs_json"] = compare_json_exact(
        required["selected_json"], selected_rows, "outer_selected_configs.json", runner
    )
    evidence_checks["outer_selection_trace"] = compare_json_exact(
        required["trace"], trace, "outer_selection_trace.json", runner
    )

    selected_csv = evidence_dir / "outer_selected_configs.csv"
    if selected_csv.is_file():
        evidence_checks["outer_selected_configs_csv"] = compare_csv_exact(
            selected_csv, selected_rows, SELECTED_FIELDS,
            "outer_selected_configs.csv",
        )
    search_csv = evidence_dir / "glsd_search_all_90x94.csv"
    if search_csv.is_file():
        evidence_checks["glsd_search_all_90x94"] = compare_csv_exact(
            search_csv, search_rows, SEARCH_FIELDS, "glsd_search_all_90x94.csv"
        )

    (
        per_subject_rows, per_video_rows, per_prediction_rows, glsd_raw, glsd_full,
    ) = replay_selected_outer(
        runner, subject_names, grouped, feature_bank, final_samples, final_emotions,
        cache_config, selected_by_outer, table,
    )
    evidence_checks["per_subject_counts"] = compare_csv_exact(
        required["per_subject"], per_subject_rows, SUBJECT_FIELDS,
        "per_subject_counts.csv",
    )
    per_video_path = evidence_dir / "per_video_evaluation.csv"
    if per_video_path.is_file():
        evidence_checks["per_video_evaluation"] = compare_csv_exact(
            per_video_path, per_video_rows, VIDEO_FIELDS, "per_video_evaluation.csv"
        )
    per_prediction_path = evidence_dir / "per_prediction_evaluation.csv"
    if per_prediction_path.is_file():
        evidence_checks["per_prediction_evaluation"] = compare_csv_exact(
            per_prediction_path, per_prediction_rows, PREDICTION_FIELDS,
            "per_prediction_evaluation.csv",
        )

    manifest = provenance["manifest"]
    manifest_native = manifest["native_exact_replay"]
    require_equal(runner.normalize(native), manifest_native,
                  "manifest Native exact replay evidence")
    require_equal(list(glsd_raw), manifest["glsd_final_raw_counts"],
                  "manifest GLSD raw replay")
    require_equal(list(glsd_full), manifest["glsd_final_full_counts"],
                  "manifest GLSD full replay")

    native_full = as_counts(native["full_counts"], "Native full")
    native_metric = runner.metrics(native_full)
    glsd_metric = runner.metrics(glsd_full)
    require_equal(native_metric["f1"], runner.metrics(EXPECTED_NATIVE_FULL)["f1"],
                  "Native full-count F1")
    require_equal(glsd_metric["f1"], manifest["glsd_final_full_metrics"]["f1"],
                  "GLSD full-count F1 vs manifest")

    runner_status = (
        "EXACT_MATCH" if provenance["runner_sha_match"]
        else "SHA_MISMATCH_FUNCTIONALLY_REPLAYED"
    )
    input_hashes = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in required.items()
    }
    recovery = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "preflight_only": True,
        "ablation_started": False,
        "bootstrap_started": False,
        "historical_manifest_modified": False,
        "input_hashes": input_hashes,
        "provenance": {
            "historical_cache_sha256": provenance["historical_cache_sha256"],
            "current_cache_sha256": cache_sha,
            "cache": "EXACT_MATCH",
            "historical_execution_wrapper_sha256":
                provenance["historical_execution_wrapper_sha256"],
            "historical_core_runner_sha256":
                provenance["historical_core_runner_sha256"],
            "current_runner_sha256": runner_sha,
            "runner_sha_match": provenance["runner_sha_match"],
            "runner": runner_status,
        },
        "universe": {
            "subjects": len(subject_names), "videos": len(videos), "gt": gt_count,
            "forbidden_controlled_cache_gt": FORBIDDEN_CONTROLLED_GT,
        },
        "protocol": {
            "configs": len(grid), "outer_folds": EXPECTED_OUTER_FOLDS,
            "inner_subjects_per_fold": EXPECTED_INNER_SUBJECTS,
            "inner_evidence_rows": EXPECTED_INNER_ROWS,
            "selection_metric": "pooled inner raw Spotting F1",
            "paper_metric": "full-count Spotting F1 after official recognition synergy",
            "tie_break": [
                "higher_F1", "higher_precision", "fewer_FP", "fixed_grid_order",
            ],
        },
        "native": {
            "raw_counts": list(as_counts(native["raw_counts"], "Native raw")),
            "full_counts": list(native_full), "full_f1": native_metric["f1"],
            "locked_delta_counts": [0, 0, 0], "locked_delta_f1": 0.0,
            "exact_replay": runner.normalize(native),
        },
        "glsd_90": {
            "raw_counts": list(glsd_raw), "full_counts": list(glsd_full),
            "full_f1": glsd_metric["f1"], "locked_delta_counts": [0, 0, 0],
            "locked_delta_f1": 0.0,
        },
        "evidence_exact_comparisons": evidence_checks,
        "all_functional_checks_passed": True,
    }
    write_recovery_outputs(
        args.output.resolve(), recovery, table, selected_rows, per_subject_rows, runner
    )

    print(f"cache path = {cache_path}")
    print(f"cache SHA256 = {cache_sha}")
    print(
        "historical execution wrapper SHA256 = "
        f"{provenance['historical_execution_wrapper_sha256']}"
    )
    print(
        "historical core runner SHA256 = "
        f"{provenance['historical_core_runner_sha256']}"
    )
    print(f"current runner SHA256 = {runner_sha}")
    print(f"subjects = {len(subject_names)}")
    print(f"videos = {len(videos)}")
    print(f"GT = {gt_count}")
    print(
        "Native full-count TP/FP/FN/F1 = "
        f"{native_full[0]}/{native_full[1]}/{native_full[2]}/{native_metric['f1']:.10f}"
    )
    print(
        "GLSD-90 full-count TP/FP/FN/F1 = "
        f"{glsd_full[0]}/{glsd_full[1]}/{glsd_full[2]}/{glsd_metric['f1']:.10f}"
    )
    print("Native locked delta TP/FP/FN/F1 = (0, 0, 0)/+0.000000000000e+00")
    print("GLSD-90 locked delta TP/FP/FN/F1 = (0, 0, 0)/+0.000000000000e+00")
    print("BOOSTING_CASME3_FUNCTIONAL_PREFLIGHT = PASS")
    print("CACHE_PROVENANCE = EXACT_MATCH")
    print(f"RUNNER_PROVENANCE = {runner_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--official-repo-root", type=Path, default=DEFAULT_OFFICIAL_REPO_ROOT,
        help="directory containing the official training_utils.py",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only:
        parser.error(
            "phase one exposes --preflight-only only; ablation/bootstrap remain locked"
        )
    if args.output.exists():
        parser.error(f"refusing to overwrite output path: {args.output}")
    try:
        run_preflight(args)
    except Exception as exc:
        print(f"BOOSTING_CASME3_FUNCTIONAL_PREFLIGHT = HARD_FAIL: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
