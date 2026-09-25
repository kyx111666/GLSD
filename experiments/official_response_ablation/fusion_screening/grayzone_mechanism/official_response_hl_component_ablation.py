"""Official-response H+L component ablation for sealed official responses.

Run after Drive is mounted.  This program never imports a backbone, reads a
video, generates an official response, or writes inside a locked evidence
directory.  It imports the exact decoder/evaluator source recorded by each
notebook and changes only candidate scores in memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Colab keeps the historical Drive location. Server runs can point this at a
# local mirror while preserving the sealed relative directory layout.
DRIVE = Path(os.environ.get("GLSD_DATA_ROOT", "/content/drive/MyDrive"))
MODES = ("H", "G", "L", "GL", "HL", "HL75")
MODE_LABELS = {
    "H": "H", "G": "G", "L": "L", "GL": "matched-budget-GL-30",
    "HL": "matched-budget-HL-30", "HL75": "matched-budget-HL75-30",
}
SCALES = (1.0, 1.5, 2.0)
RHO = 2.0
SEED = 100
REPEATS = 10_000


@dataclass(frozen=True)
class ComponentConfig:
    reference_scale: float
    local_radius: float
    threshold: float

    @property
    def identifier(self) -> str:
        return (
            f"reference_scale={self.reference_scale:g}|"
            f"local_radius={self.local_radius:g}|threshold={self.threshold:g}"
        )


# These locations are transcribed from the three supplied final Colab notebooks.
# The Boosting source programs live in the Colab checkout used by those notebooks;
# their SHA-256 is checked against the sealed result manifest after Drive mount.
SPECS = {
    "metst_sammlv": {
        "label": "ME-TST+ / SAMMLV",
        "kind": "metst",
        "dump": DRIVE / "ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1",
        "evidence": DRIVE / "GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE",
        "run_evidence": DRIVE / "GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/results_v1",
        "results": DRIVE / "GLSD_METST_OFFICIAL/results_v1",
        "locked_results": DRIVE / "GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/results_v1",
        "source": DRIVE / "GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/metst_official_glds_full.py",
        "core": DRIVE / "GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE/boosting_official_glds_full_LOCKED.py",
        "subjects": 29,
        "videos": 79,
        "gt": 159,
        "native_full": (52, 171, 107),
        "glsd_full": (47, 121, 112),
        "native_reported_f1": "0.2723",
        "glsd_reported_f1": "0.2875",
    },
    "metst_casme3": {
        "label": "ME-TST+ / CAS(ME)3",
        "kind": "metst",
        "dump": DRIVE / "ME-TST_OFFICIAL_DUMP/CASME_3_method1_strategy1",
        "evidence": DRIVE / "GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE",
        "run_evidence": DRIVE / "GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/results_v1",
        "results": DRIVE / "GLSD_METST_OFFICIAL/CASME3/results_v1",
        "locked_results": DRIVE / "GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/results_v1",
        "source": DRIVE / "GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/source/metst_casme3_official_glds_full.py",
        "core": DRIVE / "GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE/source/boosting_official_glds_full_LOCKED.py",
        "subjects": 94,
        "videos": 462,
        "gt": 853,
        "native_full": (76, 736, 777),
        "glsd_full": (90, 898, 763),
        "native_reported_f1": "0.0912912913",
        "glsd_reported_f1": "0.0977729495",
    },
    "boosting_sammlv": {
        "label": "BoostingVRME / SAMMLV",
        "kind": "boosting",
        "cache": DRIVE / "GLSD_BoostingVRME_Full/sammlv_official_full_responses.pkl",
        "evidence": DRIVE / "GLSD_BoostingVRME_Full/results_evidence_v2",
        "run_evidence": DRIVE / "GLSD_BoostingVRME_Full/results_evidence_v2",
        "locked_results": DRIVE / "GLSD_BoostingVRME_Full/results_evidence_v2",
        "source": Path("/content/BoostingVRME/boosting_official_glds_full.py"),
        "subjects": 29,
        "videos": 79,
        "gt": 159,
        "native_full": (51, 141, 108),
        "glsd_full": (44, 82, 115),
        "native_reported_f1": "0.2906",
        "glsd_reported_f1": "0.3088",
    },
    "boosting_casme3": {
        "label": "BoostingVRME / CAS(ME)3",
        "kind": "boosting",
        "cache": DRIVE / "GLSD_BoostingVRME_Full/CASME3/official_response_cache/casme3_official_full_responses.pkl",
        "evidence": DRIVE / "GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1",
        "run_evidence": DRIVE / "GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1",
        "locked_results": DRIVE / "GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1",
        "source": Path("/content/BoostingVRME/boosting_official_glds_full_casme3.py"),
        "subjects": 94,
        "videos": 462,
        "gt": 853,
        "native_full": (84, 786, 769),
        "glsd_full": (120, 1110, 733),
        "native_reported_f1": "0.0975043529",
        "glsd_reported_f1": "0.1152184349",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(counts) -> float:
    tp, fp, fn = (int(value) for value in counts)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def metrics(counts) -> dict:
    tp, fp, fn = (int(value) for value in counts)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "F1": f1((tp, fp, fn)),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sealed source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path.pop(0)
    return module


def required_paths(spec: dict) -> list[Path]:
    evidence = spec["evidence"]
    run_evidence = spec["run_evidence"]
    common = [
        run_evidence / "per_subject_counts.csv",
        run_evidence / "run_manifest.json",
        evidence / "evidence_bundle_sha256.txt",
        run_evidence / "outer_selected_configs.csv",
        spec["source"],
    ]
    if spec["kind"] == "metst":
        common.extend([spec["dump"], spec["core"]])
    else:
        common.extend([spec["cache"], evidence / "glsd_search_inner_fold_counts.csv"])
    return common


def verify_sealed_inputs(spec: dict) -> dict:
    missing = [str(path) for path in required_paths(spec) if not path.exists()]
    if missing:
        raise RuntimeError("Drive preflight failed; missing sealed input(s):\n" + "\n".join(missing))
    manifest = json.loads((spec["run_evidence"] / "run_manifest.json").read_text(encoding="utf-8"))
    text = (spec["evidence"] / "evidence_bundle_sha256.txt").read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError("sealed bundle checksum list is empty")
    for key, expected in (("subjects", spec["subjects"]), ("videos", spec["videos"])):
        if key in manifest and int(manifest[key]) != expected:
            raise RuntimeError(f"{spec['label']}: manifest {key} mismatch")
    for key in ("glsd_full_counts", "native_full_counts"):
        if key in manifest:
            got = tuple(int(x) for x in manifest[key])
            expected = spec["glsd_full"] if key.startswith("glsd") else spec["native_full"]
            if got != expected:
                raise RuntimeError(f"{spec['label']}: {key} does not match its locked result")
    return manifest


def component_grid(thresholds) -> list[ComponentConfig]:
    values = tuple(float(t) for t in thresholds)
    if len(values) != 10 or len(set(values)) != 10:
        raise RuntimeError("locked tau grid is not exactly ten unique values")
    return [ComponentConfig(scale, RHO, tau) for scale in SCALES for tau in values]


def make_component_features(base_features, mode: str):
    """Return a class that preserves the locked candidates and G/L construction.

    The only replacement is the scalar score used to threshold those candidates.
    H follows the locked protocol's normalized reference-curve amplitude:
    max(0, (peak - mean) / (max - mean)).
    """
    class ComponentFeatures(base_features):
        def selected_peaks(self, config):
            peaks, evidence = self.evidence(config.reference_scale, RHO)
            peaks = np.asarray(peaks, dtype=int)
            if not len(peaks):
                return peaks
            global_values, local_values = np.asarray(evidence, dtype=float).T
            if mode == "G":
                score = global_values
            elif mode == "L":
                score = local_values
            elif mode == "GL":
                score = (global_values + local_values) / 2.0
            elif mode == "H":
                # scale_data is the sealed core's reference-curve cache.  It is
                # inspected, never modified, and is the same curve that produced G.
                _peaks, curve, _spread, _prominence = self.scale_data[config.reference_scale]
                centre = float(np.mean(curve))
                # Exact inherited H definition from
                # my_method/gl_saliency_skill/evidence.py:99:
                # maximum(0, (smooth[peaks]-mean)/maximum(smooth.max()-mean, 1e-12)).
                score = np.maximum(
                    0.0,
                    (np.asarray(curve)[peaks] - centre)
                    / max(float(np.max(curve) - centre), 1e-12),
                )
            elif mode in ("HL", "HL75"):
                _peaks, curve, _spread, _prominence = self.scale_data[config.reference_scale]
                centre = float(np.mean(curve))
                height_values = np.maximum(
                    0.0,
                    (np.asarray(curve)[peaks] - centre)
                    / max(float(np.max(curve) - centre), 1e-12),
                )
                score = ((height_values + local_values) / 2.0
                         if mode == "HL" else 0.25 * height_values + 0.75 * local_values)
            else:
                raise ValueError(f"unknown component mode: {mode}")
            return peaks[np.asarray(score) >= float(config.threshold)]
    return ComponentFeatures


class ComponentCore:
    def __init__(self, base, mode: str, grid: list[ComponentConfig]):
        if not hasattr(base.GLSDFeatures, "evidence"):
            raise RuntimeError("locked GLSD interface has no evidence() method; abort rather than reimplement it")
        self.GLSDFeatures = make_component_features(base.GLSDFeatures, mode)
        self._grid = grid
        self.selection_order = base.selection_order

    def configuration_grid(self):
        return list(self._grid)


def outer_selection(raw_table: np.ndarray, subjects: list[str], selection_order):
    selected, traces, outer_rows = {}, [], []
    for outer, subject in enumerate(subjects):
        inner = [index for index in range(len(subjects)) if index != outer]
        pooled = raw_table[:, inner, :].sum(axis=1)
        order = np.asarray(selection_order(pooled), dtype=int)
        winner = int(order[0])
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        selected[subject] = winner
        candidates = []
        for config_id, values in enumerate(pooled):
            candidates.append({
                "config_id": config_id,
                "TP": int(values[0]), "FP": int(values[1]), "FN": int(values[2]),
                **metrics(values), "grid_order": config_id,
                "rank": int(ranks[config_id]), "selected": config_id == winner,
            })
        traces.append({
            "outer_subject": subject,
            "inner_subjects": [subjects[i] for i in inner],
            "winner_config_id": winner,
            "tie_break": ["higher F1", "higher precision", "fewer FP", "fixed grid order"],
            "candidates": candidates,
        })
        outer_rows.append((outer, subject, inner, winner, pooled[winner]))
    return selected, traces, outer_rows


def metst_context(spec: dict):
    runner = load_module(spec["source"], "sealed_metst_runner")
    base_core = load_module(spec["core"], "sealed_metst_core")
    metric, official = runner.load_official_metst()
    dump_paths, records = runner.load_records()
    native = runner.native_gate(records, metric, official)
    subjects = [str(x) for x in runner.subject_names(records)]
    if len(subjects) != spec["subjects"]:
        raise RuntimeError("ME-TST subject cardinality mismatch")
    return runner, base_core, metric, official, records, dump_paths, subjects, native


def boosting_context(spec: dict):
    runner = load_module(spec["source"], "sealed_boosting_runner")
    cache = runner.load_cache(spec["cache"])
    subjects, grouped = runner.group_videos(cache["videos"])
    metric, official = runner.official_functions()
    native = runner.native_exact_replay(cache, grouped)
    if len(subjects) != spec["subjects"]:
        raise RuntimeError("BoostingVRME subject cardinality mismatch")
    return runner, cache, grouped, [str(x) for x in subjects], metric, official, native


def native_subject_counts(kind: str, context, subject_index: int):
    if kind == "metst":
        runner, _core, metric, official, records, _paths, _subjects, _native = context
        raw, full, *_ = runner.decode_native_subject(records, subject_index, metric, official)
        return tuple(raw), tuple(full)
    runner, cache, grouped, _subjects, metric, official, _native = context
    videos = grouped[subject_index]
    samples, emotions = runner.samples_and_emotions(grouped)
    result_all = [np.asarray(video["spot_response"]) for video in videos]
    sequences = [np.asarray(video["recognition_sequence"]) for video in videos]
    map_cls, official_code = metric, official
    metric_final = map_cls(num_classes=1)
    predictions, _, total_gt, metric_video, metric_final = official_code.spotting(
        samples, subject_index, result_all, 0, metric_final,
        cache["config"]["k_p"], cache["config"]["interval_strategy"], cache["dataset"],
    )
    raw = tuple(int(x) for x in official_code.sequence_evaluation(total_gt, metric_final))
    pred_list, _, gt_tp_list, _, _ = official_code.recognition(
        sequences, predictions, metric_video, emotions, subject_index, [], [], samples, [], [],
        cache["config"]["frame_skip"], cache["config"]["penalty_strategy"],
    )
    return raw, tuple(runner.full_counts_from_official_synergy(raw, pred_list, gt_tp_list, cache["config"]))


def decode_component(kind: str, context, core, subject_index: int, config, with_recognition: bool):
    if kind == "metst":
        runner, _base, metric, official, records, _paths, _subjects, _native = context
        raw, _pred, pred_list, gt_tp_list, *_ = runner.decode_glsd_subject(
            records, subject_index, config, core, metric, official, with_recognition
        )
        full = None if not with_recognition else runner.full_counts_from_official_synergy(raw, pred_list, gt_tp_list)
        return tuple(raw), None if full is None else tuple(full)
    runner, cache, grouped, _subjects, metric, official, _native = context
    original = runner.GLSDFeatures
    runner.GLSDFeatures = core.GLSDFeatures
    try:
        samples, _emotions = runner.samples_and_emotions(grouped)
        raw, _pred, pred_list, gt_tp_list, *_ = runner.decode_glsd_subject(
            grouped, samples, subject_index, config, official, metric, cache["config"], with_recognition
        )
    finally:
        runner.GLSDFeatures = original
    full = None if not with_recognition else runner.full_counts_from_official_synergy(raw, pred_list, gt_tp_list, cache["config"])
    return tuple(raw), None if full is None else tuple(full)


def existing_glsd_subject_counts(path: Path, expected_subjects: int) -> dict[str, tuple[int, int, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_subjects:
        raise RuntimeError(f"sealed GLSD per-subject table has {len(rows)}, expected {expected_subjects}")
    field = {name.lower(): name for name in rows[0]}
    def get(row, name):
        key = field.get(name.lower())
        if key is None:
            raise RuntimeError(f"sealed per_subject_counts.csv lacks {name}")
        return int(row[key])
    result = {}
    for row in rows:
        subject = str(row[field.get("outer_subject", field.get("subject", ""))])
        if not subject:
            raise RuntimeError("sealed per-subject table lacks a subject identifier")
        result[subject] = (get(row, "full_TP"), get(row, "full_FP"), get(row, "full_FN"))
    return result


def bootstrap(setting: str, native: dict, glsd: dict, output: Path) -> None:
    subjects = sorted(native)
    if subjects != sorted(glsd):
        raise RuntimeError("Native/GLSD subject IDs differ; paired bootstrap is invalid")
    n = np.asarray([native[s] for s in subjects], dtype=np.int64)
    g = np.asarray([glsd[s] for s in subjects], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subjects), size=(REPEATS, len(subjects)))
    def sampled(values):
        totals = values[draws].sum(axis=1).astype(float)
        return np.divide(2 * totals[:, 0], 2 * totals[:, 0] + totals[:, 1] + totals[:, 2], out=np.zeros(REPEATS), where=(2 * totals[:, 0] + totals[:, 1] + totals[:, 2]) > 0)
    samples = sampled(g) - sampled(n)
    np.save(output / "native_vs_glsd_bootstrap_samples.npy", samples)
    write_csv(output / "native_vs_glsd_bootstrap_summary.csv", [{
        "setting": setting,
        "statistic": "F1_GLSD_minus_F1_Native",
        "point_estimate": f1(g.sum(axis=0)) - f1(n.sum(axis=0)),
        "percentile_2_5": float(np.quantile(samples, 0.025)),
        "percentile_97_5": float(np.quantile(samples, 0.975)),
        "resamples": REPEATS, "seed": SEED, "unit": "subject", "metric": "Full Spotting F1",
    }])
    write_json(output / "bootstrap_manifest.json", {
        "setting": setting, "seed": SEED, "resamples": REPEATS,
        "unit": "subject", "same_resampled_subject_ids_for_native_and_glsd": True,
        "aggregation": "sum TP/FP/FN then compute F1", "statistic": "F1_GLSD - F1_Native",
        "subject_ids": subjects,
    })


def locked_grid(base):
    if hasattr(base, "configuration_grid"):
        return list(base.configuration_grid())
    if hasattr(base, "build_grid"):
        return list(base.build_grid())
    raise RuntimeError("sealed GLSD source exposes neither configuration_grid() nor build_grid()")


def selected_locked_configs(path: Path, grid, subjects: list[str]) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(subjects):
        raise RuntimeError("locked outer_selected_configs.csv has the wrong fold count")
    fields = {name.lower(): name for name in rows[0]}
    subject_field = fields.get("outer_subject", fields.get("subject"))
    config_field = fields.get("config_id")
    if subject_field is None or config_field is None:
        raise RuntimeError("locked selection table lacks outer subject or config id")
    selected = {}
    for row in rows:
        subject = str(row[subject_field])
        config_id = int(row[config_field])
        if not 0 <= config_id < len(grid):
            raise RuntimeError("locked selected config is outside its GLSD-90 grid")
        selected[subject] = grid[config_id]
    if set(selected) != set(subjects):
        raise RuntimeError("locked selection subject IDs differ from official response subject IDs")
    return selected


def locked_glsd_subject_counts(kind: str, context, base, selected: dict[str, object], subjects: list[str]):
    result = {}
    for subject_index, subject in enumerate(subjects):
        config = selected[subject]
        if kind == "metst":
            runner, _core, metric, official, records, _paths, _subjects, _native = context
            raw, _pred, pred_list, gt_tp_list, *_ = runner.decode_glsd_subject(
                records, subject_index, config, base, metric, official, True
            )
            full = runner.full_counts_from_official_synergy(raw, pred_list, gt_tp_list)
        else:
            runner, cache, grouped, _subjects, metric, official, _native = context
            samples, _emotions = runner.samples_and_emotions(grouped)
            raw, _pred, pred_list, gt_tp_list, *_ = runner.decode_glsd_subject(
                grouped, samples, subject_index, config, official, metric, cache["config"], True
            )
            full = runner.full_counts_from_official_synergy(raw, pred_list, gt_tp_list, cache["config"])
        result[subject] = (tuple(int(x) for x in raw), tuple(int(x) for x in full))
    return result


def context_video_and_gt_counts(kind: str, context) -> tuple[int, int]:
    if kind == "metst":
        runner, _core, _metric, _official, records, _paths, _subjects, _native = context
        final_samples = runner.final_samples(records)
        return (
            sum(len(subject_videos) for subject_videos in final_samples),
            sum(len(video_gt) for subject_videos in final_samples for video_gt in subject_videos),
        )
    _runner, cache, _grouped, _subjects, _metric, _official, _native = context
    return (
        len(cache["videos"]),
        sum(len(video["gt_intervals"]) for video in cache["videos"]),
    )


def stage_description(kind: str) -> dict:
    if kind == "metst":
        return {
            "locked_table_stage": "official recognition/result-synergy final predictions (full counts)",
            "raw_call_chain": "official.spotting -> sequence_counts_quiet / official sequence evaluator",
            "full_call_chain": "raw counts -> official.recognition(result1_all) -> full_counts_from_official_synergy",
        }
    return {
        "locked_table_stage": "official recognition/result-synergy final predictions (full counts)",
        "raw_call_chain": "official.spotting -> official.sequence_evaluation",
        "full_call_chain": "raw counts -> official.recognition(recognition_sequence) -> full_counts_from_official_synergy",
    }


def run_preflight_setting(name: str, output: Path) -> dict:
    spec = SPECS[name]
    sealed_manifest = verify_sealed_inputs(spec)
    if spec["kind"] == "metst":
        context = metst_context(spec)
        runner, base, _metric, _official, _records, _paths, subjects, native_gate = context
    else:
        context = boosting_context(spec)
        runner, _cache, _grouped, subjects, _metric, _official, native_gate = context
        base = runner
    video_count, gt_count = context_video_and_gt_counts(spec["kind"], context)
    if video_count != spec["videos"]:
        raise RuntimeError(f"{spec['label']}: videos={video_count}, expected {spec['videos']}")
    if gt_count != spec["gt"]:
        raise RuntimeError(f"{spec['label']}: GT={gt_count}, expected {spec['gt']}")
    native_rows, native_raw, native_full = [], np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    for index, subject in enumerate(subjects):
        raw, full = native_subject_counts(spec["kind"], context, index)
        native_raw += raw
        native_full += full
        native_rows.append({"outer_subject": subject, "method": "Native", "raw_TP": raw[0], "raw_FP": raw[1], "raw_FN": raw[2], "full_TP": full[0], "full_FP": full[1], "full_FN": full[2]})
    if tuple(native_raw) != tuple(native_gate["raw_counts"]) or tuple(native_full) != spec["native_full"]:
        raise RuntimeError("Native direct per-subject replay fails its locked aggregate")
    grid = locked_grid(base)
    if len(grid) != 90:
        raise RuntimeError("locked main GLSD grid is not 90 configurations")
    selected = selected_locked_configs(spec["locked_results"] / "outer_selected_configs.csv", grid, subjects)
    glsd = locked_glsd_subject_counts(spec["kind"], context, base, selected, subjects)
    glsd_raw = np.asarray([value[0] for value in glsd.values()]).sum(axis=0)
    glsd_full = np.asarray([value[1] for value in glsd.values()]).sum(axis=0)
    if tuple(glsd_full) != spec["glsd_full"]:
        raise RuntimeError("locked GLSD direct per-subject replay fails its locked aggregate")
    sealed_subject = existing_glsd_subject_counts(spec["run_evidence"] / "per_subject_counts.csv", spec["subjects"])
    if {subject: values[1] for subject, values in glsd.items()} != sealed_subject:
        raise RuntimeError("locked GLSD replay differs from sealed per-subject full counts")
    glsd_rows = [{"outer_subject": subject, "method": "GLSD-90", "raw_TP": raw[0], "raw_FP": raw[1], "raw_FN": raw[2], "full_TP": full[0], "full_FP": full[1], "full_FN": full[2]} for subject, (raw, full) in glsd.items()]
    stage = stage_description(spec["kind"])
    row = {
        "setting": spec["label"], "evaluation_stage_for_locked_table_f1": stage["locked_table_stage"],
        "subjects": len(subjects), "videos": spec["videos"], "GT_events": gt_count,
        "Native_raw_TP": int(native_raw[0]), "Native_raw_FP": int(native_raw[1]), "Native_raw_FN": int(native_raw[2]), "Native_raw_F1": f1(native_raw),
        "Native_TP": int(native_full[0]), "Native_FP": int(native_full[1]), "Native_FN": int(native_full[2]), "Native_replay_F1": f1(native_full), "Native_locked_F1": f1(spec["native_full"]), "Native_locked_reported_F1": spec["native_reported_f1"], "Native_absolute_difference": abs(f1(native_full) - f1(spec["native_full"])),
        "GLSD_raw_TP": int(glsd_raw[0]), "GLSD_raw_FP": int(glsd_raw[1]), "GLSD_raw_FN": int(glsd_raw[2]), "GLSD_raw_F1": f1(glsd_raw),
        "GLSD_TP": int(glsd_full[0]), "GLSD_FP": int(glsd_full[1]), "GLSD_FN": int(glsd_full[2]), "GLSD_replay_F1": f1(glsd_full), "GLSD_locked_F1": f1(spec["glsd_full"]), "GLSD_locked_reported_F1": spec["glsd_reported_f1"], "GLSD_absolute_difference": abs(f1(glsd_full) - f1(spec["glsd_full"])),
    }
    write_csv(output / f"{name}_replay_per_subject.csv", native_rows + glsd_rows)
    write_json(output / f"{name}_preflight_detail.json", {
        "setting": spec["label"], "sealed_manifest": sealed_manifest, "stage": stage,
        "raw_counts_mean": "raw spotting evaluator counts before official recognition/result synergy",
        "full_counts_mean": "same official spot predictions after official recognition/result synergy conversion",
        "table_2_bootstrap_stage": "full counts; the locked Table 2 F1 values equal full-count F1",
        "H_lineage": "Exact matched-budget H formula inherited verbatim from gl_saliency_skill/evidence.py:99; no new normalization choice",
        "score_formulas": {"H": "max(0, (smooth[peak]-mean)/(smooth.max()-mean))", "G": "global prominence / reference smoothed range", "L": "median aligned local saliency / per-scale range, missing scale=0", "GL": "(G+L)/2"},
        "matched_budget": {"modes": list(MODES), "configs_per_mode": 30, "a0": list(SCALES), "tau_source": "locked GLSD ten-value grid", "rho": {"L": RHO, "GL": RHO, "HL": RHO, "HL75": RHO, "H": "not searched", "G": "not searched"}},
        "summary": row,
    })
    return row


def run_preflight(settings: list[str], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    rows = [run_preflight_setting(name, root) for name in settings]
    if any(row["Native_absolute_difference"] > 1e-12 or row["GLSD_absolute_difference"] > 1e-12 for row in rows):
        raise RuntimeError("locked-result replay tolerance failed")
    write_csv(root / "locked_replay_summary.csv", rows)
    write_json(root / "preflight_manifest.json", {"preflight_only": True, "ablation_search_started": False, "bootstrap_started": False, "settings": settings, "stage": "full counts / official recognition-result-synergy final predictions"})
    audit_lines = [
        "# Official-response locked-result preflight",
        "",
        "This run replayed sealed Native and locked GLSD-90 selections only. It did not construct a component-score grid, run H/G/L/GL selection, or run bootstrap.",
        "",
        "## Evaluation stage",
        "",
        "All four Table-2 locked F1 values are **full-count F1**: official spotting is evaluated first to form raw spotting counts, then the same spotting predictions pass through the locked official recognition/result-synergy call and `full_counts_from_official_synergy`. Bootstrap and any subsequent H/G/L/GL comparison must use these full counts.",
        "",
        "- ME-TST+: `official.spotting -> sequence_counts_quiet` (raw), then `official.recognition(result1_all) -> full_counts_from_official_synergy` (full).",
        "- BoostingVRME: `official.spotting -> official.sequence_evaluation` (raw), then `official.recognition(recognition_sequence) -> full_counts_from_official_synergy` (full).",
        "",
        "## Locked replay totals",
        "",
        "| Setting | Subjects | Videos | GT | Native full TP/FP/FN | Native replay / locked F1 | GLSD-90 full TP/FP/FN | GLSD-90 replay / locked F1 | Absolute differences |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    for row in rows:
        audit_lines.append(
            f"| {row['setting']} | {row['subjects']} | {row['videos']} | {row['GT_events']} | "
            f"{row['Native_TP']}/{row['Native_FP']}/{row['Native_FN']} | {row['Native_replay_F1']:.12f} / {row['Native_locked_reported_F1']} | "
            f"{row['GLSD_TP']}/{row['GLSD_FP']}/{row['GLSD_FN']} | {row['GLSD_replay_F1']:.12f} / {row['GLSD_locked_reported_F1']} | "
            f"Native={row['Native_absolute_difference']:.1e}; GLSD={row['GLSD_absolute_difference']:.1e} |"
        )
    audit_lines.extend([
        "",
        "## Future matched-budget component scores (not executed in this preflight)",
        "",
        "`H` directly inherits the exact prior matched-budget normalized reference-peak formula in `make_component_features()` (source lineage: `gl_saliency_skill/evidence.py:99`): `max(0, (smooth[peak] - mean) / max(smooth.max() - mean, 1e-12))`. `G` is the locked global normalized prominence, `L` is aligned multi-scale local saliency, and `GL` is the fixed `(G+L)/2`.",
        "",
        "Each future mode has exactly 30 configurations: 3 `a0` values × the locked 10-value `tau` grid. `rho=2` is fixed for L/GL/HL/HL75; H/G have no rho parameter. The output labels are `GL`, `HL`, and `HL75`; none is `GLSD-90`.",
        "",
    ])
    (root / "preflight_stage_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    print("OFFICIAL_RESPONSE_PREFLIGHT = PASS")
    print("PREFLIGHT_OUTPUT =", root)


def run_setting(name: str, root: Path) -> None:
    spec = SPECS[name]
    sealed_manifest = verify_sealed_inputs(spec)
    output = root / name
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    print(f"=== {spec['label']} ===")
    print("sealed input preflight = PASS")
    if spec["kind"] == "metst":
        context = metst_context(spec)
        runner, base, _metric, _official, _records, _paths, subjects, native_gate = context
        thresholds = base.configuration_grid()[0:10]
        thresholds = [cfg.threshold for cfg in thresholds]
    else:
        context = boosting_context(spec)
        runner, _cache, _grouped, subjects, _metric, _official, native_gate = context
        base = runner
        thresholds = [cfg.threshold for cfg in runner.configuration_grid()[0:10]] if hasattr(runner, "configuration_grid") else list(runner.THRESHOLDS)
    if tuple(native_gate["full_counts"]) != spec["native_full"]:
        raise RuntimeError("native exact replay does not equal sealed final result")
    grid = component_grid(thresholds)
    if len(grid) != 30:
        raise RuntimeError("matched budget is not 30 configurations")
    native_subject = {}
    for index, subject in enumerate(subjects):
        raw, full = native_subject_counts(spec["kind"], context, index)
        native_subject[subject] = full
    if tuple(np.asarray(list(native_subject.values())).sum(axis=0)) != spec["native_full"]:
        raise RuntimeError("per-subject Native recount fails sealed aggregate")
    locked_glsd_subject = existing_glsd_subject_counts(spec["run_evidence"] / "per_subject_counts.csv", spec["subjects"])
    if tuple(np.asarray(list(locked_glsd_subject.values())).sum(axis=0)) != spec["glsd_full"]:
        raise RuntimeError("sealed GLSD per-subject table fails locked aggregate")
    bootstrap(spec["label"], native_subject, locked_glsd_subject, output)
    summary, outer_rows, counts_rows, selections = [], [], [], {}
    search_rows, inner_rows = [], []
    for mode in MODES:
        print(f"mode={MODE_LABELS[mode]}: search {len(grid)} configs x {len(subjects)} outer subjects")
        core = ComponentCore(base, mode, grid)
        raw_table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
        for subject_index in range(len(subjects)):
            for config_id, config in enumerate(grid):
                raw, _ = decode_component(spec["kind"], context, core, subject_index, config, False)
                raw_table[config_id, subject_index] = raw
        selected, traces, folds = outer_selection(raw_table, subjects, core.selection_order)
        selections[mode] = {subject: {"config_id": cid, **asdict(grid[cid])} for subject, cid in selected.items()}
        for outer, subject in enumerate(subjects):
            inner = [index for index in range(len(subjects)) if index != outer]
            pooled = raw_table[:, inner, :].sum(axis=1)
            winner = selected[subject]
            for config_id, config in enumerate(grid):
                row_metrics = metrics(pooled[config_id])
                search_rows.append({
                    "setting": spec["label"], "score_mode": mode, "score_mode_label": MODE_LABELS[mode], "outer_subject": subject,
                    "config_id": config_id, "a0": config.reference_scale, "rho": RHO if mode in ("L", "GL", "HL", "HL75") else "not searched",
                    "tau": config.threshold, "pooled_inner_TP": row_metrics["TP"], "pooled_inner_FP": row_metrics["FP"],
                    "pooled_inner_FN": row_metrics["FN"], "precision": row_metrics["precision"], "recall": row_metrics["recall"],
                    "F1": row_metrics["F1"], "grid_order": config_id, "selected": int(config_id == winner),
                })
                for inner_index in inner:
                    values = raw_table[config_id, inner_index]
                    inner_rows.append({
                        "setting": spec["label"], "score_mode": mode, "score_mode_label": MODE_LABELS[mode], "outer_subject": subject,
                        "inner_validation_subject": subjects[inner_index], "config_id": config_id,
                        "a0": config.reference_scale, "rho": RHO if mode in ("L", "GL", "HL", "HL75") else "not searched", "tau": config.threshold,
                        "TP": int(values[0]), "FP": int(values[1]), "FN": int(values[2]),
                    })
        total = np.zeros(3, dtype=np.int64)
        for subject_index, subject, inner, config_id, inner_counts in folds:
            raw, full = decode_component(spec["kind"], context, core, subject_index, grid[config_id], True)
            if raw != tuple(raw_table[config_id, subject_index]):
                raise RuntimeError("selected outer subject does not replay its pre-selection count")
            total += raw
            outer_rows.append({
                "setting": spec["label"], "score_mode": mode, "score_mode_label": MODE_LABELS[mode], "outer_subject": subject,
                "config_id": config_id, "selected_a0": grid[config_id].reference_scale,
                "fixed_rho": RHO if mode in ("L", "GL", "HL", "HL75") else "not searched",
                "selected_tau": grid[config_id].threshold,
                "inner_subject_count": len(inner), "inner_TP": int(inner_counts[0]),
                "inner_FP": int(inner_counts[1]), "inner_FN": int(inner_counts[2]),
                "raw_TP": raw[0], "raw_FP": raw[1], "raw_FN": raw[2],
                "full_TP": full[0], "full_FP": full[1], "full_FN": full[2],
            })
            counts_rows.append({"setting": spec["label"], "score_mode": mode, "score_mode_label": MODE_LABELS[mode], "outer_subject": subject, "TP": raw[0], "FP": raw[1], "FN": raw[2], "full_TP": full[0], "full_FP": full[1], "full_FN": full[2]})
        m = metrics(total)
        summary.append({"setting": spec["label"], "score_mode": mode, "score_mode_label": MODE_LABELS[mode], **m})
        trace_name = "selection_trace_matched_budget_GL_30.json" if mode == "GL" else f"selection_trace_{mode}.json"
        write_json(output / trace_name, traces)
    write_csv(output / "ablation_summary.csv", summary)
    write_csv(output / "ablation_outer_folds.csv", outer_rows)
    write_json(output / "ablation_selected_configs.json", selections)
    write_csv(output / "ablation_per_subject_counts.csv", counts_rows)
    write_csv(output / "ablation_search_all.csv", search_rows)
    write_csv(output / "ablation_inner_fold_counts.csv", inner_rows)
    write_json(output / "ablation_manifest.json", {
        "setting": spec["label"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": "sealed official-response evidence only", "historical_controlled_cache_used": False,
        "source": str(spec["source"]), "source_sha256": sha256(spec["source"]),
        "sealed_evidence": str(spec["evidence"]), "sealed_manifest": sealed_manifest,
        "score_modes": {"H": "normalized reference peak-height amplitude", "G": "global normalized prominence", "L": "aligned multi-scale local saliency", "GL": "matched-budget-GL-30: fixed (G+L)/2; not locked GLSD-90", "HL": "matched-budget-HL-30: fixed (H+L)/2", "HL75": "matched-budget-HL75-30: fixed 0.25H+0.75L"},
        "grid": {"a0": list(SCALES), "tau": list(map(float, thresholds)), "rho": {"L": RHO, "GL": RHO, "HL": RHO, "HL75": RHO, "H": "not searched", "G": "not searched"}},
        "configurations_per_score_mode": 30, "selection": "outer LOSO; F1 -> precision -> fewer FP -> fixed grid order",
        "invariants": ["official response read only", "official candidate geometry and post-processing reused", "official evaluator reused", "no backbone", "no response generation", "no locked-evidence writes"],
    })
    print(f"{spec['label']}: ablation + paired bootstrap = PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DRIVE / "official_response_ablation")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="replay locked Native/GLSD-90 only; do not start ablation or bootstrap",
    )
    args = parser.parse_args()
    prefix = "preflight" if args.preflight_only else "ablation"
    run_id = datetime.now(timezone.utc).strftime(prefix + "_%Y%m%dT%H%M%SZ")
    root = args.output_root / run_id
    if root.exists():
        raise RuntimeError(f"refusing to overwrite: {root}")
    settings = SPECS if args.setting == "all" else {args.setting: SPECS[args.setting]}
    if args.preflight_only:
        run_preflight(list(settings), root)
        return
    root.mkdir(parents=True)
    for setting in settings:
        run_setting(setting, root)
    print("OFFICIAL_RESPONSE_COMPONENT_ABLATION = PASS")
    print("OUTPUT_ROOT =", root)


if __name__ == "__main__":
    main()
