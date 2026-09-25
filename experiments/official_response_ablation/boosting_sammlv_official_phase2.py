#!/usr/bin/env python3
"""Official-response Phase 2 for BoostingVRME / SAMMLV."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phase2_common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    FIXED_RHO,
    MODES,
    component_grid,
    component_scores,
    component_summaries,
    metrics,
    nested_loso_selection,
    paired_subject_bootstrap,
    run_protocol_smoke_assertions,
    threshold_candidates,
)


EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79
EXPECTED_GT = 159
EXPECTED_NATIVE_RAW = (51, 144, 108)
EXPECTED_NATIVE_FULL = (51, 141, 108)
EXPECTED_GLSD_RAW = (44, 84, 115)
EXPECTED_GLSD_FULL = (44, 82, 115)

DEFAULT_RUNNER = Path("/content/BoostingVRME/boosting_official_glds_full.py")
DEFAULT_REPO_ROOT = Path("/content/BoostingVRME")
DEFAULT_CACHE = Path(
    "/content/drive/MyDrive/GLSD_BoostingVRME_Full/"
    "sammlv_official_full_responses.pkl"
)
DEFAULT_EVIDENCE = Path(
    "/content/drive/MyDrive/GLSD_BoostingVRME_Full/results_evidence_v2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"sealed CSV is empty: {path}")
    return rows


def load_runner(path: Path, repo_root: Path):
    spec = importlib.util.spec_from_file_location("boosting_sammlv_locked_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import SAMMLV runner: {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path[:0] = [str(repo_root), str(path.parent)]
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        sys.path[:] = old_path
    return module


def require_api(runner) -> None:
    expected = {
        "load_cache": ("path",),
        "group_videos": ("videos",),
        "official_functions": (),
        "samples_and_emotions": ("grouped",),
        "native_exact_replay": ("cache", "grouped"),
        "configuration_grid": (),
        "decode_glsd_subject": (
            "grouped", "final_samples", "subject_index", "glsd_config",
            "official", "MeanAveragePrecision2d", "cache_config", "with_recognition",
        ),
        "selection_order": ("inner_counts",),
        "full_counts_from_official_synergy": (
            "raw_counts", "pred_list", "gt_tp_list", "config",
        ),
        "auditable_prediction_evidence": (
            "videos", "predictions", "prediction_matches", "metric_video",
            "pred_list", "gt_tp_list", "official", "cache_config",
        ),
    }
    for name, parameters in expected.items():
        function = getattr(runner, name, None)
        if not callable(function):
            raise RuntimeError(f"SAMMLV runner lacks {name}")
        actual = tuple(inspect.signature(function).parameters)
        if actual != parameters:
            raise RuntimeError(f"SAMMLV runner {name}{actual} != {parameters}")
    if not inspect.isclass(getattr(runner, "GLSDFeatures", None)):
        raise RuntimeError("SAMMLV runner lacks GLSDFeatures")


def sam_component_candidates(feature, config, mode: str) -> np.ndarray:
    peaks, evidence = feature.evidence(config.reference_scale, FIXED_RHO)
    peaks = np.asarray(peaks, dtype=int)
    if not len(peaks):
        return peaks
    global_values, local_values = np.asarray(evidence, dtype=float).T
    reference_peaks, curve, _spread, _prominence = feature.scale_data[
        config.reference_scale
    ]
    if not np.array_equal(peaks, np.asarray(reference_peaks, dtype=int)):
        raise AssertionError("SAMMLV component changed the runner candidate universe")
    scores = component_scores(
        mode, curve, peaks, global_values, local_values
    )
    return threshold_candidates(peaks, scores, config.threshold)


def component_feature_class(base_class, mode: str):
    class ComponentFeatures(base_class):
        def selected_peaks(self, config):
            return sam_component_candidates(self, config, mode)

    return ComponentFeatures


def native_subject_counts(runner, cache, grouped, subject_index, metric_class, official):
    samples, emotions = runner.samples_and_emotions(grouped)
    videos = grouped[subject_index]
    responses = [np.asarray(video["spot_response"]) for video in videos]
    sequences = [np.asarray(video["recognition_sequence"]) for video in videos]
    metric_final = metric_class(num_classes=1)
    predictions, _, total_gt, metric_video, metric_final = official.spotting(
        samples, subject_index, responses, 0, metric_final,
        cache["config"]["k_p"], cache["config"]["interval_strategy"], "SAMMLV",
    )
    raw = tuple(int(value) for value in official.sequence_evaluation(total_gt, metric_final))
    with contextlib.redirect_stdout(io.StringIO()):
        pred_list, _, gt_list, _, _ = official.recognition(
            sequences, predictions, metric_video, emotions, subject_index,
            [], [], samples, [], [], cache["config"]["frame_skip"],
            cache["config"]["penalty_strategy"],
        )
    full = runner.full_counts_from_official_synergy(
        raw, pred_list, gt_list, cache["config"]
    )
    return tuple(raw), tuple(int(value) for value in full)


def decode_component(
    runner, cache, grouped, samples, subject_index, config, feature_class,
    metric_class, official, with_recognition,
):
    original = runner.GLSDFeatures
    runner.GLSDFeatures = feature_class
    try:
        decoded = runner.decode_glsd_subject(
            grouped, samples, subject_index, config, official, metric_class,
            cache["config"], with_recognition,
        )
    finally:
        runner.GLSDFeatures = original
    raw, predictions, pred_list, gt_list, prediction_matches, metric_video = decoded
    raw = tuple(int(value) for value in raw)
    if not with_recognition:
        return raw, None
    if pred_list is None or gt_list is None:
        raise RuntimeError("official SAMMLV recognition did not return labels")
    full = tuple(int(value) for value in runner.full_counts_from_official_synergy(
        raw, pred_list, gt_list, cache["config"]
    ))
    evidence_raw, evidence_full, _, _ = runner.auditable_prediction_evidence(
        grouped[subject_index], predictions, prediction_matches, metric_video,
        pred_list, gt_list, official, cache["config"],
    )
    if tuple(evidence_raw) != raw or tuple(evidence_full) != full:
        raise RuntimeError("SAMMLV official evidence recount mismatch")
    return raw, full


def validate_universe(cache, subjects, grouped) -> None:
    if cache.get("dataset") != "SAMMLV":
        raise RuntimeError("wrong SAMMLV cache dataset")
    if len(subjects) != EXPECTED_SUBJECTS or len(grouped) != EXPECTED_SUBJECTS:
        raise RuntimeError("SAMMLV subject universe mismatch")
    if len(cache["videos"]) != EXPECTED_VIDEOS or sum(map(len, grouped)) != EXPECTED_VIDEOS:
        raise RuntimeError("SAMMLV video universe mismatch")
    gt = sum(len(video["gt_intervals"]) for video in cache["videos"])
    if gt != EXPECTED_GT:
        raise RuntimeError(f"SAMMLV GT={gt}, expected {EXPECTED_GT}")
    if len(set(subjects)) != EXPECTED_SUBJECTS:
        raise RuntimeError("duplicate SAMMLV subject IDs")


def compare_sealed_subject_rows(path, subjects, raw_rows, full_rows) -> None:
    rows = load_csv(path)
    if len(rows) != len(subjects):
        raise RuntimeError("sealed SAMMLV per-subject row count mismatch")
    fields = {name.lower(): name for name in rows[0]}
    subject_field = fields.get("outer_subject", fields.get("subject"))
    if subject_field is None:
        raise RuntimeError("sealed SAMMLV per-subject table lacks subject")
    for index, subject in enumerate(subjects):
        row = rows[index]
        if str(row[subject_field]) != subject:
            raise RuntimeError(f"sealed subject ordering mismatch at {index}")
        for stage, expected in (("raw", raw_rows[index]), ("full", full_rows[index])):
            actual = tuple(int(row[fields[f"{stage}_{name}".lower()]]) for name in ("TP", "FP", "FN"))
            if actual != tuple(expected):
                raise RuntimeError(f"sealed {stage} counts mismatch for {subject}")


def locked_preflight(args):
    required = {
        "runner": args.runner.resolve(),
        "cache": args.cache.resolve(),
        "manifest": args.evidence_dir.resolve() / "run_manifest.json",
        "selected": args.evidence_dir.resolve() / "outer_selected_configs.csv",
        "per_subject": args.evidence_dir.resolve() / "per_subject_counts.csv",
        "inner_counts": args.evidence_dir.resolve() / "glsd_search_inner_fold_counts.csv",
    }
    for label, path in required.items():
        if not path.is_file():
            raise RuntimeError(f"missing SAMMLV locked {label}: {path}")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    cache_sha = sha256_file(required["cache"])
    runner_sha = sha256_file(required["runner"])
    if manifest.get("cache_sha256") != cache_sha:
        raise RuntimeError("SAMMLV cache SHA mismatch")
    if manifest.get("script_sha256") != runner_sha:
        raise RuntimeError("SAMMLV runner SHA mismatch")
    for key, expected in (
        ("dataset", "SAMMLV"), ("subject_count", EXPECTED_SUBJECTS),
        ("video_count", EXPECTED_VIDEOS), ("glsd_grid_size", 90),
    ):
        if manifest.get(key) != expected:
            raise RuntimeError(f"SAMMLV manifest {key} mismatch")

    runner = load_runner(required["runner"], args.official_repo_root.resolve())
    require_api(runner)
    cache = runner.load_cache(required["cache"])
    subjects, grouped = runner.group_videos(cache["videos"])
    subjects = [str(subject) for subject in subjects]
    validate_universe(cache, subjects, grouped)
    native = runner.native_exact_replay(cache, grouped)
    if tuple(native["raw_counts"]) != EXPECTED_NATIVE_RAW:
        raise RuntimeError("SAMMLV locked Native raw mismatch")
    if tuple(native["full_counts"]) != EXPECTED_NATIVE_FULL:
        raise RuntimeError("SAMMLV locked Native full mismatch")
    for key in (
        "decoded_predictions_match", "recognition_predictions_match",
        "matched_gt_sequence_match",
    ):
        if native.get(key) is not True:
            raise RuntimeError(f"SAMMLV Native exact gate failed: {key}")

    metric_class, official = runner.official_functions()
    samples, _ = runner.samples_and_emotions(grouped)
    locked_grid = runner.configuration_grid()
    if len(locked_grid) != 90:
        raise RuntimeError("SAMMLV locked grid is not 90 configs")
    raw_table = np.zeros((90, EXPECTED_SUBJECTS, 3), dtype=np.int64)
    for subject_index in range(EXPECTED_SUBJECTS):
        for config_id, config in enumerate(locked_grid):
            decoded = runner.decode_glsd_subject(
                grouped, samples, subject_index, config, official, metric_class,
                cache["config"], False,
            )
            raw_table[config_id, subject_index] = decoded[0]

    with required["inner_counts"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        inner_row_count = 0
        for outer, outer_subject in enumerate(subjects):
            for config_id, config in enumerate(locked_grid):
                for inner, inner_subject in enumerate(subjects):
                    if inner == outer:
                        continue
                    try:
                        row = next(reader)
                    except StopIteration as exc:
                        raise RuntimeError("SAMMLV sealed inner evidence ended early") from exc
                    expected_counts = tuple(int(value) for value in raw_table[config_id, inner])
                    actual_counts = tuple(int(row[name]) for name in ("TP", "FP", "FN"))
                    identity = (
                        str(row["outer_subject"]), str(row["inner_validation_subject"]),
                        int(row["config_id"]), float(row["reference_scale"]),
                        float(row["local_radius"]), float(row["threshold"]),
                    )
                    expected_identity = (
                        outer_subject, inner_subject, config_id,
                        float(config.reference_scale), float(config.local_radius),
                        float(config.threshold),
                    )
                    if identity != expected_identity or actual_counts != expected_counts:
                        raise RuntimeError(
                            f"SAMMLV sealed inner evidence mismatch at row {inner_row_count + 1}"
                        )
                    inner_row_count += 1
        try:
            extra = next(reader)
        except StopIteration:
            extra = None
        if extra is not None or inner_row_count != 29 * 90 * 28:
            raise RuntimeError("SAMMLV sealed inner evidence row count mismatch")

    selected_rows = load_csv(required["selected"])
    if len(selected_rows) != EXPECTED_SUBJECTS:
        raise RuntimeError("SAMMLV sealed selection fold count mismatch")
    selected_ids = []
    for outer, subject in enumerate(subjects):
        inner = [index for index in range(EXPECTED_SUBJECTS) if index != outer]
        if len(inner) != EXPECTED_SUBJECTS - 1:
            raise AssertionError("SAMMLV inner LOSO count mismatch")
        pooled = raw_table[:, inner, :].sum(axis=1)
        winner = int(runner.selection_order(pooled)[0])
        row = selected_rows[outer]
        if str(row["outer_subject"]) != subject or int(row["config_id"]) != winner:
            raise RuntimeError(f"SAMMLV locked selection mismatch at {subject}")
        if tuple(int(row[f"inner_{name}"]) for name in ("TP", "FP", "FN")) != tuple(pooled[winner]):
            raise RuntimeError(f"SAMMLV locked inner counts mismatch at {subject}")
        selected_ids.append(winner)

    native_raw_rows, native_full_rows = [], []
    glsd_raw_rows, glsd_full_rows = [], []
    for subject_index in range(EXPECTED_SUBJECTS):
        native_raw, native_full = native_subject_counts(
            runner, cache, grouped, subject_index, metric_class, official
        )
        native_raw_rows.append(native_raw)
        native_full_rows.append(native_full)
        config_id = selected_ids[subject_index]
        decoded = runner.decode_glsd_subject(
            grouped, samples, subject_index, locked_grid[config_id], official,
            metric_class, cache["config"], True,
        )
        raw, predictions, pred_list, gt_list, matches, metric_video = decoded
        raw = tuple(int(value) for value in raw)
        if raw != tuple(raw_table[config_id, subject_index]):
            raise RuntimeError("SAMMLV selected outer raw replay mismatch")
        full = tuple(int(value) for value in runner.full_counts_from_official_synergy(
            raw, pred_list, gt_list, cache["config"]
        ))
        evidence_raw, evidence_full, _, _ = runner.auditable_prediction_evidence(
            grouped[subject_index], predictions, matches, metric_video, pred_list,
            gt_list, official, cache["config"],
        )
        if tuple(evidence_raw) != raw or tuple(evidence_full) != full:
            raise RuntimeError("SAMMLV locked GLSD evidence recount mismatch")
        glsd_raw_rows.append(raw)
        glsd_full_rows.append(full)

    if tuple(np.sum(native_raw_rows, axis=0)) != EXPECTED_NATIVE_RAW:
        raise RuntimeError("SAMMLV Native per-subject raw aggregate mismatch")
    if tuple(np.sum(native_full_rows, axis=0)) != EXPECTED_NATIVE_FULL:
        raise RuntimeError("SAMMLV Native per-subject full aggregate mismatch")
    if tuple(np.sum(glsd_raw_rows, axis=0)) != EXPECTED_GLSD_RAW:
        raise RuntimeError("SAMMLV GLSD-90 raw aggregate mismatch")
    if tuple(np.sum(glsd_full_rows, axis=0)) != EXPECTED_GLSD_FULL:
        raise RuntimeError("SAMMLV GLSD-90 full aggregate mismatch")
    compare_sealed_subject_rows(
        required["per_subject"], subjects, glsd_raw_rows, glsd_full_rows
    )
    return {
        "runner": runner, "cache": cache, "subjects": subjects, "grouped": grouped,
        "metric_class": metric_class, "official": official, "samples": samples,
        "native_raw": np.asarray(native_raw_rows, dtype=np.int64),
        "native_full": np.asarray(native_full_rows, dtype=np.int64),
        "glsd_raw": np.asarray(glsd_raw_rows, dtype=np.int64),
        "glsd_full": np.asarray(glsd_full_rows, dtype=np.int64),
        "locked_raw_table": raw_table, "selected_ids": selected_ids,
        "input_paths": required, "manifest": manifest,
    }


def raw_table_rows(mode, subjects, grid, table):
    rows = []
    for config_id, config in enumerate(grid):
        for subject_index, subject in enumerate(subjects):
            tp, fp, fn = (int(value) for value in table[config_id, subject_index])
            rows.append({
                "component": mode, "config_id": config_id,
                "reference_scale": config.reference_scale,
                "rho": FIXED_RHO if mode in ("L", "G+L") else "not_searched",
                "threshold": config.threshold, "subject_index": subject_index,
                "subject": subject, "raw_TP": tp, "raw_FP": fp, "raw_FN": fn,
            })
    return rows


def run_phase2(args) -> None:
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    context = locked_preflight(args)
    print("BOOSTING_SAMMLV_LOCKED_PREFLIGHT = PASS")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preflight_dir = output / "locked_preflight"
    preflight_dir.mkdir()
    np.save(preflight_dir / "locked_glsd90_raw_counts_90x29x3.npy", context["locked_raw_table"])

    runner = context["runner"]
    cache = context["cache"]
    subjects = context["subjects"]
    grouped = context["grouped"]
    samples = context["samples"]
    metric_class = context["metric_class"]
    official = context["official"]
    grid = component_grid()
    if len(grid) != 30:
        raise AssertionError("SAMMLV component grid is not 30 configs")

    pooled_raw_rows = []
    pooled_full_rows = []
    feature_classes = {
        mode: component_feature_class(runner.GLSDFeatures, mode) for mode in MODES
    }
    for mode in MODES:
        slug = mode.replace("+", "_plus_")
        mode_dir = output / f"component_{slug}"
        mode_dir.mkdir()
        raw_table = np.zeros((30, EXPECTED_SUBJECTS, 3), dtype=np.int64)
        for subject_index in range(EXPECTED_SUBJECTS):
            for config_id, config in enumerate(grid):
                raw, _ = decode_component(
                    runner, cache, grouped, samples, subject_index, config,
                    feature_classes[mode],
                    metric_class, official, False,
                )
                raw_table[config_id, subject_index] = raw
        winners, winner_rows, traces = nested_loso_selection(raw_table, subjects, grid)

        subject_rows = []
        raw_subject_counts = []
        full_subject_counts = []
        for subject_index, subject in enumerate(subjects):
            config_id = winners[subject_index]
            raw, full = decode_component(
                runner, cache, grouped, samples, subject_index, grid[config_id],
                feature_classes[mode],
                metric_class, official, True,
            )
            if raw != tuple(int(value) for value in raw_table[config_id, subject_index]):
                raise RuntimeError(f"{mode} selected outer raw cell mismatch at {subject}")
            if full is None:
                raise RuntimeError("full counts did not come from official recognition/synergy")
            raw_subject_counts.append(raw)
            full_subject_counts.append(full)
            subject_rows.append({
                "component": mode,
                "component_label": "matched-budget-G+L-30" if mode == "G+L" else mode,
                "subject_index": subject_index, "subject": subject,
                "config_id": config_id,
                "reference_scale": grid[config_id].reference_scale,
                "rho": FIXED_RHO if mode in ("L", "G+L") else "not_searched",
                "threshold": grid[config_id].threshold,
                "raw_TP": raw[0], "raw_FP": raw[1], "raw_FN": raw[2],
                "full_TP": full[0], "full_FP": full[1], "full_FN": full[2],
            })
        raw_summary, full_summary = component_summaries(
            raw_subject_counts, full_subject_counts
        )
        if (full_summary["TP"], full_summary["FP"], full_summary["FN"]) != tuple(
            np.sum(full_subject_counts, axis=0)
        ):
            raise AssertionError("SAMMLV full summary reused raw aggregate")
        label = "matched-budget-G+L-30" if mode == "G+L" else mode
        pooled_raw_rows.append({"component": mode, "component_label": label, **raw_summary})
        pooled_full_rows.append({"component": mode, "component_label": label, **full_summary})
        np.save(mode_dir / "raw_counts_30x29x3.npy", raw_table)
        write_csv(mode_dir / "raw_counts_30x29.csv", raw_table_rows(mode, subjects, grid, raw_table))
        write_csv(mode_dir / "outer_winner_configs.csv", winner_rows)
        write_json(mode_dir / "selection_trace.json", traces)
        write_csv(mode_dir / "per_subject_raw_full_counts.csv", subject_rows)

    write_csv(output / "component_pooled_raw_summary.csv", pooled_raw_rows)
    write_csv(output / "component_pooled_full_summary.csv", pooled_full_rows)

    locked_rows = []
    for index, subject in enumerate(subjects):
        native = context["native_full"][index]
        glsd = context["glsd_full"][index]
        locked_rows.append({
            "subject_index": index, "subject": subject,
            "Native_full_TP": int(native[0]), "Native_full_FP": int(native[1]),
            "Native_full_FN": int(native[2]), "GLSD90_full_TP": int(glsd[0]),
            "GLSD90_full_FP": int(glsd[1]), "GLSD90_full_FN": int(glsd[2]),
        })
    write_csv(output / "native_glsd90_per_subject_full_counts.csv", locked_rows)
    bootstrap = paired_subject_bootstrap(context["native_full"], context["glsd_full"])
    np.save(output / "paired_bootstrap_subject_indexes.npy", bootstrap["indexes"])
    np.save(output / "paired_bootstrap_native_f1_samples.npy", bootstrap["native_f1_samples"])
    np.save(output / "paired_bootstrap_glsd_f1_samples.npy", bootstrap["glsd_f1_samples"])
    np.save(output / "paired_bootstrap_delta_f1_samples.npy", bootstrap["delta_f1_samples"])
    write_csv(output / "paired_bootstrap_summary.csv", [bootstrap["summary"]])

    script_path = Path(__file__).resolve()
    common_path = script_path.with_name("phase2_common.py")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SAMMLV", "status": "PASS", "locked_preflight": "PASS",
        "universe": {"subjects": 29, "videos": 79, "gt_events": 159},
        "sha256": {
            "script": sha256_file(script_path), "common": sha256_file(common_path),
            "runner": sha256_file(args.runner.resolve()),
            "cache": sha256_file(args.cache.resolve()),
            "sealed_manifest": sha256_file(context["input_paths"]["manifest"]),
            "sealed_per_subject": sha256_file(context["input_paths"]["per_subject"]),
            "sealed_selected_configs": sha256_file(context["input_paths"]["selected"]),
            "sealed_inner_counts": sha256_file(context["input_paths"]["inner_counts"]),
        },
        "locked_results": {
            "Native_raw": list(EXPECTED_NATIVE_RAW),
            "Native_full": list(EXPECTED_NATIVE_FULL),
            "GLSD90_raw": list(EXPECTED_GLSD_RAW),
            "GLSD90_full": list(EXPECTED_GLSD_FULL),
        },
        "component_protocol": {
            "modes": list(MODES), "configs_per_mode": 30,
            "reference_scales": [1.0, 1.5, 2.0],
            "thresholds": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.75],
            "rho": {"H": "not_searched", "G": "not_searched", "L": 2.0, "G+L": 2.0},
            "grid_order": "a0-major,tau-minor",
            "selection": "pooled inner raw F1, precision, fewer FP, grid order",
            "paper_metric": "pooled full-count Spotting F1",
            "G+L_label": "matched-budget-G+L-30; not locked GLSD-90",
        },
        "bootstrap": bootstrap["summary"],
    }
    write_json(output / "run_manifest.json", manifest)
    print("BOOSTING_SAMMLV_OFFICIAL_PHASE2 = PASS")
    print("OUTPUT =", output)


def run_smoke_test() -> None:
    run_protocol_smoke_assertions()

    class FakeSAMFeature:
        def __init__(self):
            self._peaks = np.asarray([1, 3])
            self._curve = np.asarray([0.0, 2.0, 0.0, 1.0, 0.0])
            self.scale_data = {
                scale: (self._peaks, self._curve, 2.0, np.asarray([0.8, 0.4]))
                for scale in (1.0, 1.5, 2.0)
            }

        def evidence(self, reference, radius):
            assert radius == FIXED_RHO
            return self._peaks, np.asarray([[0.8, 0.2], [0.4, 0.6]])

    feature = FakeSAMFeature()
    config = component_grid()[5]  # tau=0.5 separates the four score paths.
    outputs = {
        mode: sam_component_candidates(feature, config, mode).tolist() for mode in MODES
    }
    expected = {"H": [1], "G": [1], "L": [3], "G+L": [1, 3]}
    if outputs != expected:
        raise AssertionError(f"SAMMLV component score path mismatch: {outputs}")
    print("BOOSTING_SAMMLV_PHASE2_SMOKE = PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--official-repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
        return
    if args.output is None:
        parser.error("--output is required for a formal Phase-2 run")
    run_phase2(args)


if __name__ == "__main__":
    main()
