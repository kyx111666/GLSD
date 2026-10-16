#!/usr/bin/env python3
"""Run one target-side transfer direction with no target-side selection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from transfer_common import (
    DATASETS, DIRECTIONS, LOCKED, canonical_subject, f1, load_csv, metrics,
    normalize_json, paired_bootstrap, selected_config_rows, write_csv, write_json,
)
from transfer_paths import EVIDENCE, RESPONSES, RUNNERS
from release_config import paths


def load_module(path: Path, name: str, repo_root: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    additions = [str(path.parent)]
    if repo_root is not None:
        additions.insert(0, str(repo_root))
    sys.path[:0] = additions
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def theta_of(config) -> tuple[float, float, float]:
    if isinstance(config, dict):
        return tuple(float(config[name]) for name in ("reference_scale", "local_radius", "threshold"))
    return tuple(float(getattr(config, name)) for name in ("reference_scale", "local_radius", "threshold"))


def target_config(grid, theta: tuple[float, float, float]):
    matches = [config for config in grid if theta_of(config) == theta]
    if len(matches) != 1:
        raise RuntimeError(f"source theta {theta} does not map uniquely into target locked GLSD-90 grid")
    return matches[0]


def assert_source_k_never_passed(source: dict) -> None:
    assert "source_k" not in source and "k" not in source and "k_p" not in source, (
        "source_k is never passed into target decoder execution"
    )


def sealed_tuned_counts(path: Path, subjects: list[str], dataset: str) -> dict[str, tuple[int, int, int]]:
    rows = load_csv(path)
    first = rows[0]
    subject_name = next((name for name in first if name.lower() in {"outer_subject", "subject", "subject_id"}), None)
    if subject_name is None:
        raise RuntimeError(f"sealed per-subject table lacks subject field: {path}")
    lowered = {name.lower(): name for name in first}
    result = {}
    for row in rows:
        subject = canonical_subject(dataset, row[subject_name])
        counts = []
        for metric_name in ("tp", "fp", "fn"):
            candidates = (f"full_{metric_name}", f"full_{metric_name.upper()}")
            key = next((lowered[name.lower()] for name in candidates if name.lower() in lowered), None)
            if key is None:
                raise RuntimeError(f"sealed table lacks full {metric_name}: {path}")
            counts.append(int(row[key]))
        result[subject] = tuple(counts)
    if set(result) != set(subjects):
        raise RuntimeError("target sealed tuned counts do not match target subject IDs")
    return result


def generic_target(setting: str, source_configs: dict[str, dict]):
    import official_response_component_ablation as ora

    spec = ora.SPECS[setting]
    ora.verify_sealed_inputs(spec)
    context = ora.metst_context(spec) if spec["kind"] == "metst" else ora.boosting_context(spec)
    if spec["kind"] == "metst":
        runner, base, metric_class, official, records, _paths, raw_subjects, _native = context
        grid = ora.locked_grid(base)
    else:
        runner, cache, grouped, raw_subjects, metric_class, official, _native = context
        base = runner
        grid = ora.locked_grid(base)
    dataset = LOCKED[setting]["dataset"]
    subjects = [canonical_subject(dataset, value) for value in raw_subjects]
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("duplicate canonical target subject IDs")
    if set(subjects) != set(source_configs):
        raise RuntimeError("source/target canonical subject mapping is incomplete")
    tuned = sealed_tuned_counts(EVIDENCE[setting] / "per_subject_counts.csv", subjects, dataset)

    rows, configs, prediction_rows = [], [], []
    for index, subject in enumerate(subjects):
        source = source_configs[subject]
        assert_source_k_never_passed(source)
        theta = (source["a0"], source["rho"], source["tau"])
        config = target_config(grid, theta)
        native_raw, native_full = ora.native_subject_counts(spec["kind"], context, index)
        if spec["kind"] == "metst":
            target_k = int(records[index]["k_p"])
            raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
                records, index, config, base, metric_class, official, True,
            )
            full = runner.full_counts_from_official_synergy(raw, pred_list, gt_list)
        else:
            target_k = int(cache["config"]["k_p"])
            samples, _ = runner.samples_and_emotions(grouped)
            raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
                grouped, samples, index, config, official, metric_class, cache["config"], True,
            )
            full = runner.full_counts_from_official_synergy(raw, pred_list, gt_list, cache["config"])
        raw = tuple(int(value) for value in raw)
        full = tuple(int(value) for value in full)
        tuned_counts = tuned[subject]
        row = {
            "dataset": dataset, "subject": subject,
            "source_pipeline": LOCKED[DIRECTION["source"]]["pipeline"],
            "target_pipeline": LOCKED[setting]["pipeline"],
            "source_a0": source["a0"], "source_rho": source["rho"], "source_tau": source["tau"],
            "source_config_origin": source["source_config_origin"],
            "source_outer_fold": source["source_outer_fold"], "target_k": target_k,
            "target_native_TP": native_full[0], "target_native_FP": native_full[1], "target_native_FN": native_full[2],
            "target_tuned_TP": tuned_counts[0], "target_tuned_FP": tuned_counts[1], "target_tuned_FN": tuned_counts[2],
            "transfer_TP": full[0], "transfer_FP": full[1], "transfer_FN": full[2],
            "target_native_F1_subject": f1(native_full), "target_tuned_F1_subject": f1(tuned_counts),
            "transfer_F1_subject": f1(full),
        }
        rows.append(row)
        configs.append({key: row[key] for key in (
            "dataset", "subject", "source_pipeline", "target_pipeline", "source_a0", "source_rho",
            "source_tau", "source_config_origin", "source_outer_fold", "target_k",
        )})
        prediction_rows.append({
            "dataset": dataset, "subject": subject, "target_pipeline": LOCKED[setting]["pipeline"],
            "theta_source": {"a0": source["a0"], "rho": source["rho"], "tau": source["tau"]},
            "target_k": target_k, "raw_counts": list(raw), "full_counts": list(full),
            "predictions": normalize_json(predictions),
        })
    return rows, configs, prediction_rows


def casme3_boosting_target(source_configs: dict[str, dict]):
    import boosting_casme3_official_phase2 as helper

    setting = "boosting_casme3"
    dataset = "casme3"
    runner = load_module(RUNNERS[setting], "transfer_boosting_casme3_runner", paths(setting)["upstream"])
    cache, videos = runner.load_cache(RESPONSES[setting])
    grouped = runner.group_videos(videos)
    raw_subjects = [str(subject[0]["subject"]) for subject in grouped]
    subjects = [canonical_subject(dataset, value) for value in raw_subjects]
    expected = DATASETS[dataset]
    gt = sum(len(video["gt_intervals"]) for video in videos)
    if (len(subjects), len(videos), gt) != (expected["subjects"], expected["videos"], expected["gt"]):
        raise RuntimeError("BoostingVRME/CAS(ME)3 target universe mismatch")
    if set(subjects) != set(source_configs):
        raise RuntimeError("source/target canonical subject mapping is incomplete")
    cache_config = cache["config"]
    target_k = int(cache_config["k_p"])
    final_samples = [[video["gt_intervals"] for video in subject] for subject in grouped]
    final_emotions = [[video["emotion_labels"] for video in subject] for subject in grouped]
    feature_bank = [
        [runner.GLSDFeatures(video["spot_response"], target_k) for video in subject]
        for subject in grouped
    ]
    grid = list(runner.build_grid())
    if len(grid) != 90:
        raise RuntimeError("BoostingVRME/CAS(ME)3 target grid is not locked GLSD-90")
    tuned_raw, tuned_full = helper.load_locked_glsd_counts(
        EVIDENCE[setting] / "per_subject_counts.csv", raw_subjects,
    )
    rows, configs, prediction_rows = [], [], []
    for index, subject in enumerate(subjects):
        source = source_configs[subject]
        assert_source_k_never_passed(source)
        theta = (source["a0"], source["rho"], source["tau"])
        config = target_config(grid, theta)
        native_raw, native_full = helper.native_subject_counts(
            runner, cache_config, grouped, final_samples, final_emotions, index,
        )
        decoded = runner.decode_subject(
            index, config, grouped, feature_bank, final_samples, final_emotions,
            cache_config, need_recognition=True,
        )
        raw = tuple(int(value) for value in decoded["raw_counts"])
        full = tuple(int(value) for value in decoded["full_counts"])
        tuned_counts = tuple(int(value) for value in tuned_full[index])
        row = {
            "dataset": dataset, "subject": subject,
            "source_pipeline": LOCKED[DIRECTION["source"]]["pipeline"],
            "target_pipeline": LOCKED[setting]["pipeline"],
            "source_a0": source["a0"], "source_rho": source["rho"], "source_tau": source["tau"],
            "source_config_origin": source["source_config_origin"],
            "source_outer_fold": source["source_outer_fold"], "target_k": target_k,
            "target_native_TP": int(native_full[0]), "target_native_FP": int(native_full[1]), "target_native_FN": int(native_full[2]),
            "target_tuned_TP": tuned_counts[0], "target_tuned_FP": tuned_counts[1], "target_tuned_FN": tuned_counts[2],
            "transfer_TP": full[0], "transfer_FP": full[1], "transfer_FN": full[2],
            "target_native_F1_subject": f1(native_full), "target_tuned_F1_subject": f1(tuned_counts),
            "transfer_F1_subject": f1(full),
        }
        rows.append(row)
        configs.append({key: row[key] for key in (
            "dataset", "subject", "source_pipeline", "target_pipeline", "source_a0", "source_rho",
            "source_tau", "source_config_origin", "source_outer_fold", "target_k",
        )})
        prediction_rows.append({
            "dataset": dataset, "subject": subject, "target_pipeline": LOCKED[setting]["pipeline"],
            "theta_source": {"a0": source["a0"], "rho": source["rho"], "tau": source["tau"]},
            "target_k": target_k, "raw_counts": list(raw), "full_counts": list(full),
            "predictions": normalize_json(decoded["preds"]),
        })
    return rows, configs, prediction_rows


def validate_and_write(output: Path, rows: list[dict], configs: list[dict], predictions: list[dict]) -> None:
    target_setting = DIRECTION["target"]
    expected = LOCKED[target_setting]
    native = np.asarray([[row[f"target_native_{name}"] for name in ("TP", "FP", "FN")] for row in rows])
    tuned = np.asarray([[row[f"target_tuned_{name}"] for name in ("TP", "FP", "FN")] for row in rows])
    transferred = np.asarray([[row[f"transfer_{name}"] for name in ("TP", "FP", "FN")] for row in rows])
    if tuple(native.sum(axis=0)) != expected["native"]:
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: target Native per-subject replay mismatch")
    if tuple(tuned.sum(axis=0)) != expected["tuned"]:
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: target tuned GLSD-90 evidence mismatch")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite transfer output: {output}")
    output.mkdir(parents=True)
    write_csv(output / "transferred_configs.csv", configs)
    write_csv(output / "per_subject_counts.csv", rows)
    write_json(output / "predictions.json", predictions)
    bootstrap = paired_bootstrap(native, tuned, transferred)
    write_json(output / "bootstrap.json", bootstrap)
    native_counts = tuple(int(x) for x in native.sum(axis=0))
    tuned_counts = tuple(int(x) for x in tuned.sum(axis=0))
    transfer_counts = tuple(int(x) for x in transferred.sum(axis=0))
    result = {
        "direction": DIRECTION_NAME, "dataset": DIRECTION["dataset"],
        "source_pipeline": LOCKED[DIRECTION["source"]]["pipeline"],
        "target_pipeline": LOCKED[DIRECTION["target"]]["pipeline"],
        "target_native": metrics(native_counts), "target_tuned_glsd90": metrics(tuned_counts),
        "transferred": metrics(transfer_counts),
        "delta_vs_native": f1(transfer_counts) - f1(native_counts),
        "delta_vs_tuned": f1(transfer_counts) - f1(tuned_counts),
        "bootstrap_95_ci_vs_native": [bootstrap["vs_native"]["percentile_2_5"], bootstrap["vs_native"]["percentile_97_5"]],
        "bootstrap_95_ci_vs_tuned": [bootstrap["vs_tuned"]["percentile_2_5"], bootstrap["vs_tuned"]["percentile_97_5"]],
    }
    write_json(output / "metrics.json", result)
    write_json(output / "protocol_checks.json", {
        "status": "PASS", "source_config_fields": ["a0", "rho", "tau"],
        "source_k_passed_to_target": False, "target_k_origin": "target official-response context",
        "target_side_configuration_search": False,
        "target_event_geometry_preserved": True, "target_boundary_candidate_conflict_rules_preserved": True,
        "target_official_evaluator_preserved": True, "target_recognition_result_synergy_preserved": True,
    })


def main() -> None:
    global DIRECTION_NAME, DIRECTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    DIRECTION_NAME = args.direction
    DIRECTION = DIRECTIONS[DIRECTION_NAME]
    gate = json.loads((args.output_root / "validation/preflight_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not gate.get("transfer_allowed"):
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: four-setting exact replay gate is not PASS")
    source_path = EVIDENCE[DIRECTION["source"]] / "outer_selected_configs.csv"
    source_configs = selected_config_rows(source_path, DIRECTION["dataset"])
    if DIRECTION["target"] == "boosting_casme3":
        rows, configs, predictions = casme3_boosting_target(source_configs)
    else:
        rows, configs, predictions = generic_target(DIRECTION["target"], source_configs)
    expected_subjects = DATASETS[DIRECTION["dataset"]]["subjects"]
    if len(rows) != expected_subjects:
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: transfer subject count mismatch")
    output = args.output_root / DIRECTION["relative_output"]
    validate_and_write(output, rows, configs, predictions)
    print(f"TRANSFER_DIRECTION_PASS = {DIRECTION_NAME}")


if __name__ == "__main__":
    main()
