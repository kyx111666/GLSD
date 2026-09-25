"""Nested Native--GLSD output fusion for the sealed ME-TST+ responses.

This runner is intentionally separate from the older G/L score-fusion runner.
It reads the official frozen response dumps, decodes Native and GLSD candidate
peaks independently, fuses the candidate peaks before official spotting, and
then runs the unchanged official recognition/result-synergy path.

The joint search is outer-LOSO: each configuration is (GLSD-90 config,
output-fusion mode, peak-distance tolerance).  The held-out subject is never
used for selection.  The script supports ME-TST+ SAMMLV and CAS(ME)3 only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import official_response_component_ablation as ora  # noqa: E402


FUSION_MODES = ("Agreement", "Union")
K_VALUES = (1, 2, 3)
SEED = 100


def f1(counts):
    tp, fp, fn = (int(x) for x in counts)
    return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def metrics(counts):
    tp, fp, fn = (int(x) for x in counts)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "F1": f1(counts),
    }


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class FusionConfig:
    glsd_config_id: int
    mode: str
    k: int

    @property
    def identifier(self):
        return f"glsd={self.glsd_config_id}|mode={self.mode}|k={self.k}"


def selection_winner(table: np.ndarray, outer_index: int):
    pooled = np.delete(table, outer_index, axis=1).sum(axis=1)
    scores = [metrics(row) for row in pooled]
    winner = min(
        range(len(scores)),
        key=lambda i: (-scores[i]["F1"], -scores[i]["precision"], scores[i]["FP"], i),
    )
    return int(winner), pooled


@contextlib.contextmanager
def capture_native_peaks(official):
    original = official.find_peaks
    captured = []

    def wrapped(signal, *args, **kwargs):
        peaks, props = original(signal, *args, **kwargs)
        captured.append(np.asarray(peaks, dtype=int).copy())
        return peaks, props

    official.find_peaks = wrapped
    try:
        yield captured
    finally:
        official.find_peaks = original


def native_decode(context, subject_index):
    runner, _base, metric, official, records, _paths, _subjects, _native = context
    with capture_native_peaks(official) as captured, contextlib.redirect_stdout(io.StringIO()):
        raw, full, predictions, pred_list, gt_tp_list, metric_video = runner.decode_native_subject(
            records, subject_index, metric, official
        )
    expected = len(records[subject_index]["result_all"])
    if len(captured) != expected:
        raise RuntimeError(f"Native candidate capture mismatch: {len(captured)} != {expected}")
    return {
        "peaks": [p.tolist() for p in captured],
        "raw": tuple(map(int, raw)),
        "full": tuple(map(int, full)),
        "predictions": predictions,
        "pred_list": pred_list,
        "gt_tp_list": gt_tp_list,
        "metric_video": metric_video,
    }


def glsd_peaks(context, subject_index, glsd_config):
    runner, base, _metric, _official, records, _paths, _subjects, _native = context
    record = records[subject_index]
    k_p = int(record["k_p"])
    return [
        np.asarray(base.GLSDFeatures(np.asarray(response, dtype=float), k_p).selected_peaks(glsd_config), dtype=int).tolist()
        for response in record["result_all"]
    ]


def dedup(peaks, k):
    values = sorted({int(x) for x in peaks})
    kept = []
    for peak in values:
        if not kept or peak - kept[-1] > k:
            kept.append(peak)
    return kept


def fused_peaks(native, glsd, mode, k):
    native = [int(x) for x in native]
    glsd = [int(x) for x in glsd]
    if mode == "GLSD":
        return dedup(glsd, 0)
    if mode == "Union":
        return dedup(native + glsd, k)
    if mode == "Agreement":
        return dedup([p for p in native if any(abs(p - q) <= k for q in glsd)], k)
    raise ValueError(mode)


def evaluate_candidate_lists(context, subject_index, candidate_lists, with_recognition):
    runner, _base, metric, official, records, _paths, _subjects, _native = context
    record = records[subject_index]
    samples = runner.final_samples(records)
    emotions = runner.final_emotions(records)
    responses = [np.asarray(x, dtype=float) for x in record["result_all"]]
    k_p = int(record["k_p"])
    metric_final = metric(num_classes=1)
    total_gt = 0
    with runner.metst_glsd_candidate_source(official, candidate_lists), contextlib.redirect_stdout(io.StringIO()):
        predictions, _, total_gt, metric_video, metric_final = official.spotting(
            samples, subject_index, responses, total_gt, runner.NATIVE_P, metric_final, k_p
        )
    raw = runner.sequence_counts_quiet(official, total_gt, metric_final)
    if not with_recognition:
        return tuple(map(int, raw)), None, predictions
    sequences = [np.asarray(x) for x in record["result1_all"]]
    with contextlib.redirect_stdout(io.StringIO()):
        pred_list, _, gt_tp_list, _, _ = official.recognition(
            sequences, predictions, metric_video, emotions, subject_index,
            [], [], samples, [], [], int(record["frame_skip"])
        )
    full = runner.full_counts_from_official_synergy(raw, pred_list, gt_tp_list)
    return tuple(map(int, raw)), tuple(map(int, full)), predictions


def subject_candidate_cache(context, subject_index, glsd_grid):
    native = native_decode(context, subject_index)
    glsd = [glsd_peaks(context, subject_index, config) for config in glsd_grid]
    return native, glsd


def evaluate_config(context, subject_index, native, glsd_lists, config, with_recognition):
    candidate_lists = [
        fused_peaks(n_peaks, g_peaks, config.mode, config.k)
        for n_peaks, g_peaks in zip(native["peaks"], glsd_lists[config.glsd_config_id])
    ]
    raw, full, predictions = evaluate_candidate_lists(context, subject_index, candidate_lists, with_recognition)
    return raw, full, predictions, candidate_lists


def run_setting(name: str, output: Path, requested_mode: str):
    spec = ora.SPECS[name]
    ora.verify_sealed_inputs(spec)
    if spec["kind"] != "metst":
        raise RuntimeError("This first runner supports ME-TST+ only")
    context = ora.metst_context(spec)
    runner, base, _metric, _official, records, _paths, subjects, native_gate = context
    if tuple(native_gate["full_counts"]) != tuple(spec["native_full"]):
        raise RuntimeError("Native exact replay failed before fusion search")
    if ora.context_video_and_gt_counts("metst", context) != (spec["videos"], spec["gt"]):
        raise RuntimeError("Official video/GT inventory mismatch")
    glsd_grid = list(base.configuration_grid())
    if len(glsd_grid) != 90:
        raise RuntimeError(f"Expected locked GLSD-90 grid, got {len(glsd_grid)}")
    if requested_mode not in FUSION_MODES:
        raise RuntimeError(f"requested_mode must be one of {FUSION_MODES}")
    configs = [FusionConfig(gid, requested_mode, k) for gid in range(len(glsd_grid)) for k in K_VALUES]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "status": "COMPLETED",
        "setting": spec["label"],
        "subjects": subjects,
        "videos": spec["videos"],
        "gt_events": spec["gt"],
        "glsd_grid_size": len(glsd_grid),
        "fusion_modes": [requested_mode],
        "k_values": list(K_VALUES),
        "selection": "outer LOSO; pooled inner raw counts; F1, precision, fewer FP, fixed order",
        "fusion_stage": "candidate peaks before official spotting",
        "recognition_stage": "unchanged official recognition/result synergy",
        "gpu_required": False,
    })

    caches = []
    for si, subject in enumerate(subjects):
        print(f"cache candidates {si + 1}/{len(subjects)} subject={subject}", flush=True)
        caches.append(subject_candidate_cache(context, si, glsd_grid))

    # The sealed GLSD result is kept as an independent comparison.  Its
    # subject-specific configurations come from the official nested-selection
    # artifact, whereas the Fusion configuration is selected below jointly.
    selected_glsd = ora.selected_locked_configs(
        spec["run_evidence"] / "outer_selected_configs.csv", glsd_grid, subjects
    )

    raw_table = np.zeros((len(configs), len(subjects), 3), dtype=np.int64)
    for ci, config in enumerate(configs):
        print(f"raw search {ci + 1}/{len(configs)} {config.identifier}", flush=True)
        for si in range(len(subjects)):
            raw, _full, _pred, _lists = evaluate_config(context, si, caches[si][0], caches[si][1], config, False)
            raw_table[ci, si] = raw

    selections = {}
    search_rows = []
    outer_rows = []
    full_counts_rows = []
    per_subject_method = []
    final_total = {requested_mode: np.zeros(3, dtype=np.int64)}
    final_total["Native"] = np.zeros(3, dtype=np.int64)

    native_full_by_subject = {}
    glsd_independent_total = np.zeros(3, dtype=np.int64)
    for si, subject in enumerate(subjects):
        native_full_by_subject[subject] = caches[si][0]["full"]
        final_total["Native"] += np.asarray(caches[si][0]["full"], dtype=np.int64)
        glsd_config = selected_glsd[subject]
        glsd_raw, glsd_full, _ = evaluate_candidate_lists(
            context, si, caches[si][1][glsd_grid.index(glsd_config)], True
        )
        glsd_independent_total += np.asarray(glsd_full, dtype=np.int64)
        per_subject_method.append({
            "outer_subject": subject, "method": "GLSD_independent",
            **{f"full_{key}": value for key, value in zip(("TP", "FP", "FN"), glsd_full)},
        })

    for outer, subject in enumerate(subjects):
        winner, pooled = selection_winner(raw_table, outer)
        selected = configs[winner]
        selections[subject] = {"config_id": winner, **asdict(selected), "identifier": selected.identifier}
        inner = [i for i in range(len(subjects)) if i != outer]
        for ci, config in enumerate(configs):
            row = metrics(pooled[ci])
            search_rows.append({
                "outer_subject": subject, "config_id": ci, "glsd_config_id": config.glsd_config_id,
                "mode": config.mode, "k": config.k, **{f"inner_{key}": value for key, value in row.items()}
            })
        raw, full, predictions, candidate_lists = evaluate_config(
            context, outer, caches[outer][0], caches[outer][1], selected, True
        )
        if raw != tuple(raw_table[winner, outer]):
            raise RuntimeError("Selected raw configuration failed replay")
        final_total[selected.mode] += np.asarray(full, dtype=np.int64)
        full_counts_rows.append({
            "outer_subject": subject, "method": "Fusion", "config_id": winner,
            "glsd_config_id": selected.glsd_config_id, "mode": selected.mode, "k": selected.k,
            "raw_TP": raw[0], "raw_FP": raw[1], "raw_FN": raw[2],
            "full_TP": full[0], "full_FP": full[1], "full_FN": full[2],
        })
        # Locked ablations use the selected full configuration but retain the
        # normal single-decoder output, so they are meaningful rather than zero.
        glsd_only_lists = caches[outer][1][selected.glsd_config_id]
        glsd_raw, glsd_full, _ = evaluate_candidate_lists(context, outer, glsd_only_lists, True)
        final_total["GLSD"] = final_total.get("GLSD", np.zeros(3, dtype=np.int64)) + np.asarray(glsd_full, dtype=np.int64)
        per_subject_method.extend([
            {"outer_subject": subject, "method": "Native", **{f"full_{k}": v for k, v in zip(("TP", "FP", "FN"), caches[outer][0]["full"])}},
            {"outer_subject": subject, "method": "GLSD_locked_full_config", **{f"full_{k}": v for k, v in zip(("TP", "FP", "FN"), glsd_full)}},
            {"outer_subject": subject, "method": "Fusion", **{f"full_{k}": v for k, v in zip(("TP", "FP", "FN"), full)}},
        ])
        outer_rows.append({
            "outer_subject": subject, "selected_config_id": winner,
            "selected_glsd_config_id": selected.glsd_config_id, "selected_mode": selected.mode,
            "selected_k": selected.k, "inner_subject_count": len(inner),
            "inner_TP": int(pooled[winner, 0]), "inner_FP": int(pooled[winner, 1]), "inner_FN": int(pooled[winner, 2]),
        })
        if (outer + 1) % 5 == 0 or outer + 1 == len(subjects):
            print(f"full replay {outer + 1}/{len(subjects)}", flush=True)

    # Aggregate the selected per-subject GLSD locked-config ablation from rows.
    glsd_total = np.zeros(3, dtype=np.int64)
    fusion_total = np.zeros(3, dtype=np.int64)
    for row in per_subject_method:
        if row["method"] == "GLSD_locked_full_config":
            glsd_total += np.asarray([row["full_TP"], row["full_FP"], row["full_FN"]])
        elif row["method"] == "Fusion":
            fusion_total += np.asarray([row["full_TP"], row["full_FP"], row["full_FN"]])
    summary = []
    for method, counts in (("Native", final_total["Native"]), ("GLSD_independent", glsd_independent_total), ("GLSD_locked_full_config", glsd_total), ("Fusion", fusion_total)):
        summary.append({"method": method, **metrics(counts)})

    write_json(output / "outer_selected_configs.json", selections)
    write_csv(output / "inner_search_all.csv", search_rows)
    write_csv(output / "outer_folds.csv", outer_rows)
    write_csv(output / "selected_fusion_per_subject_counts.csv", full_counts_rows)
    write_csv(output / "ablation_per_subject_counts.csv", per_subject_method)
    write_csv(output / "summary.csv", summary)
    write_json(output / "completion.json", {
        "completed": True, "setting": spec["label"], "subjects": len(subjects),
        "videos": spec["videos"], "gt_events": spec["gt"], "configurations": len(configs),
        "summary": summary,
    })
    print("NATIVE_GLSD_OUTPUT_FUSION = PASS", flush=True)
    print("OUTPUT =", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("metst_sammlv", "metst_casme3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=FUSION_MODES, default="Agreement")
    args = parser.parse_args()
    run_setting(args.setting, args.output, args.mode)


if __name__ == "__main__":
    main()
