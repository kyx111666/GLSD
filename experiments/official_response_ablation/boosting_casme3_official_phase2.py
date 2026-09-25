#!/usr/bin/env python3
"""Official-response Phase 2 for BoostingVRME / CAS(ME)3."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from phase2_common import (
    FIXED_RHO,
    MODES,
    component_grid,
    component_scores,
    component_summaries,
    nested_loso_selection,
    paired_subject_bootstrap,
    run_protocol_smoke_assertions,
    threshold_candidates,
)


EXPECTED_SUBJECTS = 94
EXPECTED_VIDEOS = 462
EXPECTED_GT = 853
EXPECTED_NATIVE_FULL = (84, 786, 769)
EXPECTED_GLSD_RAW = (120, 1113, 733)
EXPECTED_GLSD_FULL = (120, 1110, 733)

DEFAULT_RUNNER = Path("/content/BoostingVRME/boosting_official_glds_full_casme3.py")
DEFAULT_REPO_ROOT = Path("/content/BoostingVRME")
DEFAULT_CACHE = Path(
    "/content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/"
    "official_response_cache/casme3_official_full_responses.pkl"
)
DEFAULT_EVIDENCE = Path(
    "/content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1"
)
DEFAULT_PREFLIGHT = Path(__file__).with_name(
    "boosting_casme3_official_response_preflight.py"
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
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path[:] = old_path
    return module


def cas_component_candidates(feature, config, mode: str) -> list[int]:
    evidence = feature.scores(config.reference_scale, FIXED_RHO)
    peaks = np.asarray(evidence["peaks"], dtype=int)
    if not len(peaks):
        return []
    width = feature.scale_to_width[float(config.reference_scale)]
    reference_peaks = np.asarray(feature.peaks[width], dtype=int)
    curve = np.asarray(feature.curves[width], dtype=float)
    if not np.array_equal(peaks, reference_peaks):
        raise AssertionError("CAS(ME)3 component changed the runner candidate universe")
    scores = component_scores(
        mode, curve, peaks, evidence["G"], evidence["L"]
    )
    return [
        int(value) for value in threshold_candidates(peaks, scores, config.threshold)
    ]


class CASComponentFeature:
    def __init__(self, base, mode: str):
        self.base = base
        self.mode = mode

    def selected_peaks(self, reference_scale, _rho, threshold):
        config = SimpleNamespace(
            reference_scale=float(reference_scale), threshold=float(threshold)
        )
        return cas_component_candidates(self.base, config, self.mode)


def component_config_dict(config) -> dict:
    return {
        "config_id": int(config.config_id),
        "reference_scale": float(config.reference_scale),
        "local_radius": FIXED_RHO,
        "threshold": float(config.threshold),
        "grid_order": int(config.config_id),
    }


def full_counts_from_recognition(raw, pred_list, gt_list, cache_config):
    neutral = int(cache_config["emotion_type"]) - 1
    neutral_tp = 0
    neutral_fp = 0
    for prediction, target in zip(pred_list, gt_list):
        if int(prediction) == neutral:
            if int(target) == -1:
                neutral_fp += 1
            else:
                neutral_tp += 1
    return (
        int(raw[0]) - neutral_tp,
        int(raw[1]) - neutral_fp,
        int(raw[2]) + neutral_tp,
    )


def native_subject_counts(
    runner, cache_config, grouped, final_samples, final_emotions, subject_index,
):
    videos = grouped[subject_index]
    responses = [np.asarray(video["spot_response"], dtype=float) for video in videos]
    sequences = [np.asarray(video["recognition_sequence"]) for video in videos]
    metric_final = runner.tu.MeanAveragePrecision2d(num_classes=1)
    predictions, _, total_gt, metric_video, metric_final = runner.tu.spotting(
        final_samples, subject_index, responses, 0, metric_final,
        int(cache_config["k_p"]), int(cache_config["interval_strategy"]), runner.DATASET,
    )
    raw = tuple(int(value) for value in runner.official_metric_counts(metric_final, total_gt))
    with contextlib.redirect_stdout(io.StringIO()):
        pred_list, _, gt_list, _, _ = runner.tu.recognition(
            sequences, predictions, metric_video, final_emotions, subject_index,
            [], [], final_samples, [], [], int(cache_config["frame_skip"]),
            int(cache_config["penalty_strategy"]),
        )
    full = full_counts_from_recognition(raw, pred_list, gt_list, cache_config)
    return raw, full


def decode_component(
    runner, grouped, feature_bank, final_samples, final_emotions,
    cache_config, subject_index, config, with_recognition,
):
    config_item = component_config_dict(config)
    decoded = runner.decode_subject(
        subject_index, config_item, grouped, feature_bank, final_samples,
        final_emotions, cache_config, need_recognition=with_recognition,
    )
    raw = tuple(int(value) for value in decoded["raw_counts"])
    if not with_recognition:
        return raw, None
    if "full_counts" not in decoded:
        raise RuntimeError("CAS(ME)3 full counts did not come from official recognition")
    full = tuple(int(value) for value in decoded["full_counts"])
    _, video_rows, _ = runner.subject_prediction_evidence(
        subject_index, str(grouped[subject_index][0]["subject"]), config_item,
        decoded, grouped, final_samples, cache_config,
    )
    video_raw = tuple(sum(int(row[f"raw_{name}"]) for row in video_rows) for name in ("tp", "fp", "fn"))
    video_full = tuple(sum(int(row[f"full_{name}"]) for row in video_rows) for name in ("tp", "fp", "fn"))
    if video_raw != raw or video_full != full:
        raise RuntimeError("CAS(ME)3 official evidence recount mismatch")
    return raw, full


def load_locked_glsd_counts(path: Path, subjects: list[str]):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(subjects):
        raise RuntimeError("CAS(ME)3 sealed per-subject row count mismatch")
    raw, full = [], []
    for index, row in enumerate(rows):
        if int(row["outer_subject_index"]) != index or str(row["outer_subject"]) != subjects[index]:
            raise RuntimeError(f"CAS(ME)3 sealed subject order mismatch at {index}")
        raw.append(tuple(int(row[f"raw_{name}"]) for name in ("tp", "fp", "fn")))
        full.append(tuple(int(row[f"full_{name}"]) for name in ("tp", "fp", "fn")))
    return np.asarray(raw, dtype=np.int64), np.asarray(full, dtype=np.int64)


def run_locked_preflight(args, output: Path) -> Path:
    preflight_path = args.preflight_script.resolve()
    if not preflight_path.is_file():
        raise RuntimeError(f"missing successful CAS(ME)3 preflight implementation: {preflight_path}")
    preflight = load_module(preflight_path, "boosting_casme3_phase2_preflight")
    preflight_output = output / "locked_preflight"
    preflight.run_preflight(SimpleNamespace(
        runner=args.runner.resolve(),
        official_repo_root=args.official_repo_root.resolve(),
        cache=args.cache.resolve(),
        evidence_dir=args.evidence_dir.resolve(),
        output=preflight_output,
    ))
    recovery_path = preflight_output / "recovery_manifest.json"
    if not recovery_path.is_file():
        raise RuntimeError("CAS(ME)3 preflight did not produce recovery_manifest.json")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if recovery.get("status") != "PASS" or recovery.get("all_functional_checks_passed") is not True:
        raise RuntimeError("CAS(ME)3 functional preflight did not pass")
    return recovery_path


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
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    recovery_path = run_locked_preflight(args, output)
    print("BOOSTING_CASME3_LOCKED_PREFLIGHT_GATE = PASS")

    runner = load_module(
        args.runner.resolve(), "boosting_casme3_phase2_runner",
        args.official_repo_root.resolve(),
    )
    cache, videos = runner.load_cache(args.cache.resolve())
    grouped = runner.group_videos(videos)
    subjects = [str(subject[0]["subject"]) for subject in grouped]
    gt = sum(len(video["gt_intervals"]) for video in videos)
    if len(subjects) != EXPECTED_SUBJECTS or len(videos) != EXPECTED_VIDEOS or gt != EXPECTED_GT:
        raise RuntimeError("CAS(ME)3 official universe mismatch after preflight")
    if len(set(subjects)) != EXPECTED_SUBJECTS:
        raise RuntimeError("duplicate CAS(ME)3 subject IDs")

    cache_config = cache["config"]
    final_samples = [[video["gt_intervals"] for video in subject] for subject in grouped]
    final_emotions = [[video["emotion_labels"] for video in subject] for subject in grouped]
    k = int(cache_config["k_p"])
    base_feature_bank = [
        [runner.GLSDFeatures(video["spot_response"], k) for video in subject]
        for subject in grouped
    ]

    native_raw_rows, native_full_rows = [], []
    for subject_index in range(EXPECTED_SUBJECTS):
        raw, full = native_subject_counts(
            runner, cache_config, grouped, final_samples, final_emotions, subject_index
        )
        native_raw_rows.append(raw)
        native_full_rows.append(full)
    native_raw = np.asarray(native_raw_rows, dtype=np.int64)
    native_full = np.asarray(native_full_rows, dtype=np.int64)
    if tuple(native_full.sum(axis=0)) != EXPECTED_NATIVE_FULL:
        raise RuntimeError("CAS(ME)3 Native per-subject full aggregate mismatch")

    sealed_subject_path = args.evidence_dir.resolve() / "per_subject_counts.csv"
    glsd_raw, glsd_full = load_locked_glsd_counts(sealed_subject_path, subjects)
    if tuple(glsd_raw.sum(axis=0)) != EXPECTED_GLSD_RAW:
        raise RuntimeError("CAS(ME)3 locked GLSD-90 raw aggregate mismatch")
    if tuple(glsd_full.sum(axis=0)) != EXPECTED_GLSD_FULL:
        raise RuntimeError("CAS(ME)3 locked GLSD-90 full aggregate mismatch")

    grid = component_grid()
    if len(grid) != 30:
        raise AssertionError("CAS(ME)3 component grid is not 30 configs")
    pooled_raw_rows = []
    pooled_full_rows = []
    for mode in MODES:
        slug = mode.replace("+", "_plus_")
        mode_dir = output / f"component_{slug}"
        mode_dir.mkdir()
        feature_bank = [
            [CASComponentFeature(feature, mode) for feature in subject]
            for subject in base_feature_bank
        ]
        raw_table = np.zeros((30, EXPECTED_SUBJECTS, 3), dtype=np.int64)
        for subject_index in range(EXPECTED_SUBJECTS):
            for config_id, config in enumerate(grid):
                raw, _ = decode_component(
                    runner, grouped, feature_bank, final_samples, final_emotions,
                    cache_config, subject_index, config, False,
                )
                raw_table[config_id, subject_index] = raw
        winners, winner_rows, traces = nested_loso_selection(raw_table, subjects, grid)

        subject_rows = []
        raw_subject_counts = []
        full_subject_counts = []
        for subject_index, subject in enumerate(subjects):
            config_id = winners[subject_index]
            raw, full = decode_component(
                runner, grouped, feature_bank, final_samples, final_emotions,
                cache_config, subject_index, grid[config_id], True,
            )
            expected_raw = tuple(int(value) for value in raw_table[config_id, subject_index])
            if raw != expected_raw:
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
        expected_full = tuple(int(value) for value in np.sum(full_subject_counts, axis=0))
        if (full_summary["TP"], full_summary["FP"], full_summary["FN"]) != expected_full:
            raise AssertionError("CAS(ME)3 full summary reused raw aggregate")
        label = "matched-budget-G+L-30" if mode == "G+L" else mode
        pooled_raw_rows.append({"component": mode, "component_label": label, **raw_summary})
        pooled_full_rows.append({"component": mode, "component_label": label, **full_summary})
        np.save(mode_dir / "raw_counts_30x94x3.npy", raw_table)
        write_csv(mode_dir / "raw_counts_30x94.csv", raw_table_rows(mode, subjects, grid, raw_table))
        write_csv(mode_dir / "outer_winner_configs.csv", winner_rows)
        write_json(mode_dir / "selection_trace.json", traces)
        write_csv(mode_dir / "per_subject_raw_full_counts.csv", subject_rows)

    write_csv(output / "component_pooled_raw_summary.csv", pooled_raw_rows)
    write_csv(output / "component_pooled_full_summary.csv", pooled_full_rows)

    locked_rows = []
    for index, subject in enumerate(subjects):
        native = native_full[index]
        glsd = glsd_full[index]
        locked_rows.append({
            "subject_index": index, "subject": subject,
            "Native_full_TP": int(native[0]), "Native_full_FP": int(native[1]),
            "Native_full_FN": int(native[2]), "GLSD90_full_TP": int(glsd[0]),
            "GLSD90_full_FP": int(glsd[1]), "GLSD90_full_FN": int(glsd[2]),
        })
    write_csv(output / "native_glsd90_per_subject_full_counts.csv", locked_rows)
    bootstrap = paired_subject_bootstrap(native_full, glsd_full)
    np.save(output / "paired_bootstrap_subject_indexes.npy", bootstrap["indexes"])
    np.save(output / "paired_bootstrap_native_f1_samples.npy", bootstrap["native_f1_samples"])
    np.save(output / "paired_bootstrap_glsd_f1_samples.npy", bootstrap["glsd_f1_samples"])
    np.save(output / "paired_bootstrap_delta_f1_samples.npy", bootstrap["delta_f1_samples"])
    write_csv(output / "paired_bootstrap_summary.csv", [bootstrap["summary"]])

    script_path = Path(__file__).resolve()
    common_path = script_path.with_name("phase2_common.py")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CASME_3", "status": "PASS", "locked_preflight": "PASS",
        "universe": {"subjects": 94, "videos": 462, "gt_events": 853},
        "sha256": {
            "script": sha256_file(script_path), "common": sha256_file(common_path),
            "runner": sha256_file(args.runner.resolve()),
            "cache": sha256_file(args.cache.resolve()),
            "preflight_script": sha256_file(args.preflight_script.resolve()),
            "preflight_recovery_manifest": sha256_file(recovery_path),
            "sealed_per_subject": sha256_file(sealed_subject_path),
        },
        "locked_results": {
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
    print("BOOSTING_CASME3_OFFICIAL_PHASE2 = PASS")
    print("OUTPUT =", output)


def run_smoke_test() -> None:
    run_protocol_smoke_assertions()

    class FakeCASFeature:
        def __init__(self):
            self._peaks = np.asarray([1, 3])
            self._curve = np.asarray([0.0, 2.0, 0.0, 1.0, 0.0])
            self.scale_to_width = {1.0: 2, 1.5: 2, 2.0: 2}
            self.peaks = {2: self._peaks}
            self.curves = {2: self._curve}

        def scores(self, reference, rho):
            assert rho == FIXED_RHO
            return {
                "peaks": self._peaks,
                "G": np.asarray([0.8, 0.4]),
                "L": np.asarray([0.2, 0.6]),
                "S": np.asarray([0.5, 0.5]),
            }

    feature = FakeCASFeature()
    config = component_grid()[5]  # tau=0.5 separates the four score paths.
    outputs = {
        mode: cas_component_candidates(feature, config, mode) for mode in MODES
    }
    expected = {"H": [1], "G": [1], "L": [3], "G+L": [1, 3]}
    if outputs != expected:
        raise AssertionError(f"CAS(ME)3 component score path mismatch: {outputs}")
    print("BOOSTING_CASME3_PHASE2_SMOKE = PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--official-repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--preflight-script", type=Path, default=DEFAULT_PREFLIGHT)
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
