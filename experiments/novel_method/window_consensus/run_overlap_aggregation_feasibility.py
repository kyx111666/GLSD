#!/usr/bin/env python3
"""SAMMLV O0/O1/O2 overlap aggregation feasibility on frozen raw dumps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RAW = Path(
    os.environ.get("SAMMLV_RAW_ROOT", "data/raw_prestitch_sammlv_full")
)
DEFAULT_OUT = ROOT / "results/overlap_aggregation_feasibility_sammlv"
COMPACT_CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
CONFIG_REPORT = ROOT / "results/rgr1_sammlv_nested/report.json"
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
RECON_SOURCE = HERE / "reconstruct_original_stitch.py"
EXPECTED_CACHE_SHA = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
EXPECTED_ANCHOR = (49, 143, 110)
EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79
K_P = 5
METHODS = ("O0 Original", "O1 Uniform Overlap Mean", "O2 Fixed Center-Weighted")
DECISION_TOLERANCE = 0.001


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = import_module("overlap_native", NATIVE_SOURCE)
reconstruction = import_module("overlap_reconstruction", RECON_SOURCE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f1_from_counts(counts: dict) -> float:
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return 2 * counts["TP"] / denominator if denominator else 0.0


def add_counts(target: dict, source: dict) -> None:
    for key in ("TP", "FP", "FN", "event_count"):
        target[key] += int(source[key])


def sum_counts(rows) -> dict:
    total = native.counts_empty()
    for row in rows:
        add_counts(total, row)
    return total


def load_configs() -> tuple[dict[str, dict], str]:
    report = json.loads(CONFIG_REPORT.read_text(encoding="utf-8"))
    if report.get("subject_count") != EXPECTED_SUBJECTS or report.get("video_count") != EXPECTED_VIDEOS:
        raise RuntimeError("frozen Final Strong Native config report cardinality mismatch")
    configs = {}
    for fold in report["outer_folds"]:
        source = fold["selected_strong_config"]
        configs[str(fold["subject"])] = {
            "c_s": float(source["c_s"]),
            "p": float(source["p_s"]),
            "c_d": float(source["c_d"]),
            "c_b": float(source["c_b"]),
        }
    if len(configs) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"frozen configs={len(configs)} expected={EXPECTED_SUBJECTS}")
    return configs, sha256(CONFIG_REPORT)


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not left.size:
        return None
    if np.array_equal(left, right):
        return 1.0
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def curve_metrics(curve: np.ndarray, original: np.ndarray) -> dict:
    difference = np.abs(
        np.asarray(curve, dtype=np.float64) - np.asarray(original, dtype=np.float64)
    )
    return {
        "pearson_vs_original": pearson(curve, original),
        "mae_vs_original": float(difference.mean()) if difference.size else 0.0,
        "max_abs_difference_vs_original": float(difference.max(initial=0.0)),
    }


def aggregate_raw(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(data["raw_window_score"], dtype=np.float64)
    global_indices = np.asarray(data["global_indices"], dtype=np.int64)
    original = np.asarray(data["original_fresh_result_video"], dtype=np.float64)
    if raw.shape != global_indices.shape or raw.ndim != 2 or raw.shape[1] != 30:
        raise RuntimeError(f"invalid raw/global shapes: {raw.shape}, {global_indices.shape}")
    if global_indices.min(initial=0) < 0 or global_indices.max(initial=-1) >= len(original):
        raise RuntimeError("global index outside original video curve")

    flat_index = global_indices.reshape(-1)
    flat_score = raw.reshape(-1)
    overlap = np.bincount(flat_index, minlength=len(original)).astype(np.int32)
    if np.any(overlap <= 0):
        raise RuntimeError("raw windows leave uncovered global frames")

    uniform_sum = np.bincount(flat_index, weights=flat_score, minlength=len(original))
    uniform = uniform_sum / overlap

    local_index = np.arange(30, dtype=np.float64)
    center_weight = 1.0 - np.abs(local_index - 14.5) / 15.0
    tiled_weight = np.broadcast_to(center_weight, raw.shape).reshape(-1)
    weighted_sum = np.bincount(
        flat_index, weights=flat_score * tiled_weight, minlength=len(original)
    )
    weight_denominator = np.bincount(
        flat_index, weights=tiled_weight, minlength=len(original)
    )
    if np.any(weight_denominator <= 0):
        raise RuntimeError("center weighting produced a zero denominator")
    center = weighted_sum / weight_denominator
    return uniform, center, overlap


def blocked_report(output_root: Path, details: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "OVERLAP_AGGREGATION_FEASIBILITY_SAMMLV_CN.md"
    path.write_text(
        "# Overlap-Aware Aggregation Feasibility — SAMMLV\n\n"
        "`BLOCKED-OVERLAP-AGGREGATION-ANCHOR`\n\n"
        f"{details}\n\nO1/O2 未执行。\n",
        encoding="utf-8",
    )


def provenance_gate(args: argparse.Namespace):
    manifest_path = args.raw_dir / "full_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing raw full manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest = {
        "status": "SAMMLV-RAW-WINDOW-SOURCE-READY",
        "complete": True,
        "subjects": EXPECTED_SUBJECTS,
        "videos": EXPECTED_VIDEOS,
        "videos_passed": EXPECTED_VIDEOS,
        "videos_failed": 0,
        "worst_score_max_abs_error": 0.0,
        "worst_logits_max_abs_error": 0.0,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"raw manifest {key}={manifest.get(key)!r}, expected={expected!r}")

    if sha256(COMPACT_CACHE) != EXPECTED_CACHE_SHA:
        raise RuntimeError("verified SAMMLV compact cache SHA mismatch")
    _payload, records, subjects, observed = native.load_payload(COMPACT_CACHE)
    if observed != {"subjects": 29, "videos": 79, "gt": 159, "k_p": K_P}:
        raise RuntimeError(f"compact cache metadata mismatch: {observed}")
    configs, config_sha = load_configs()
    if set(configs) != set(subjects):
        raise RuntimeError("fold-wise Final Strong Native subject set mismatch")

    record_lookup = {(str(row["subject"]), str(row["video"])): row for row in records}
    manifest_hashes = {
        (str(row["subject"]), str(row["video"])): row.get("sha256")
        for row in manifest.get("video_reports", [])
    }
    if set(record_lookup) != set(manifest_hashes) or len(record_lookup) != EXPECTED_VIDEOS:
        raise RuntimeError("raw/compact video identity mismatch")

    loaded = {}
    o0_subject_counts = {subject: native.counts_empty() for subject in subjects}
    for identity, record in record_lookup.items():
        subject, video = identity
        path = args.raw_dir / f"{subject}_{video}_raw_prestitch.npz"
        if not path.exists():
            raise RuntimeError(f"missing raw NPZ: {path}")
        expected_hash = manifest_hashes[identity]
        if expected_hash and sha256(path) != expected_hash:
            raise RuntimeError(f"raw NPZ SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["subject"]).item()) != subject:
                raise RuntimeError(f"NPZ subject mismatch: {path}")
            if str(np.asarray(data["video"]).item()) != video:
                raise RuntimeError(f"NPZ video mismatch: {path}")
            equivalence = reconstruction.evaluate(data)
            if equivalence["status"] != "RAW-WINDOW-INSTRUMENTATION-READY":
                raise RuntimeError(f"O0 reconstruction mismatch: {subject}/{video}")
            original = np.asarray(data["original_fresh_result_video"], dtype=np.float64).copy()
            raw = np.asarray(data["raw_window_score"], dtype=np.float64).copy()
            global_indices = np.asarray(data["global_indices"], dtype=np.int64).copy()
        if len(original) != len(record["score"]):
            raise RuntimeError(f"curve length mismatch: {subject}/{video}")
        raw_record = dict(record)
        raw_record["score"] = original
        events = native.tuned_decode(raw_record, configs[subject], K_P)
        add_counts(o0_subject_counts[subject], native.evaluate(raw_record, events))
        loaded[identity] = {
            "path": path,
            "record": record,
            "original": original,
            "raw": raw,
            "global_indices": global_indices,
        }

    o0_total = sum_counts(o0_subject_counts.values())
    anchor = tuple(o0_total[key] for key in ("TP", "FP", "FN"))
    if anchor != EXPECTED_ANCHOR:
        raise RuntimeError(f"raw-window O0 anchor={anchor}, expected={EXPECTED_ANCHOR}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha": sha256(manifest_path),
        "config_sha": config_sha,
        "subjects": subjects,
        "configs": configs,
        "loaded": loaded,
        "o0_subject_counts": o0_subject_counts,
    }


def comparison_label(candidate_f1: float, baseline_f1: float) -> str:
    if candidate_f1 > baseline_f1 + 1e-15:
        return "improved"
    if candidate_f1 < baseline_f1 - 1e-15:
        return "worse"
    return "equal"


def decision(aggregate: dict[str, dict]) -> str:
    deltas = [aggregate[method]["F1"] - aggregate[METHODS[0]]["F1"] for method in METHODS[1:]]
    if max(deltas) > DECISION_TOLERANCE:
        return "OVERLAP-AGGREGATION-SIGNAL-POSITIVE"
    if all(delta < -DECISION_TOLERANCE for delta in deltas):
        return "SIMPLE-AGGREGATION-NO-GO"
    return "SIMPLE-AGGREGATION-INCONCLUSIVE"


def run(args: argparse.Namespace, gated: dict) -> dict:
    subjects = gated["subjects"]
    subject_counts = {
        method: {subject: native.counts_empty() for subject in subjects} for method in METHODS
    }
    for subject in subjects:
        subject_counts[METHODS[0]][subject] = gated["o0_subject_counts"][subject]

    curve_rows = []
    pooled_curves = {method: [] for method in METHODS}
    pooled_original = []
    total_coverage = Counter()
    for (subject, video), item in gated["loaded"].items():
        original = item["original"]
        data = {
            "raw_window_score": item["raw"],
            "global_indices": item["global_indices"],
            "original_fresh_result_video": original,
        }
        uniform, center, overlap = aggregate_raw(data)
        curves = {
            METHODS[0]: original,
            METHODS[1]: uniform,
            METHODS[2]: center,
        }
        pooled_original.append(original)
        coverage = Counter(map(int, overlap))
        total_coverage.update(coverage)
        for method, curve in curves.items():
            pooled_curves[method].append(curve)
            record = dict(item["record"])
            record["score"] = curve
            events = native.tuned_decode(record, gated["configs"][subject], K_P)
            counts = native.evaluate(record, events)
            if method != METHODS[0]:
                add_counts(subject_counts[method][subject], counts)
            row = {
                "subject": subject,
                "video": video,
                "method": method,
                "score_length": len(curve),
                "predicted_events": len(events),
                "frames_covered_by_1_window": coverage.get(1, 0),
                "frames_covered_by_2_windows": coverage.get(2, 0),
                "maximum_overlap_count": int(overlap.max(initial=0)),
                **curve_metrics(curve, original),
            }
            curve_rows.append(row)

    aggregate = {
        method: native.metrics(sum_counts(subject_counts[method].values())) for method in METHODS
    }
    concatenated_original = np.concatenate(pooled_original)
    aggregate_curve = {
        method: curve_metrics(np.concatenate(pooled_curves[method]), concatenated_original)
        for method in METHODS
    }
    baseline = aggregate[METHODS[0]]
    if tuple(baseline[key] for key in ("TP", "FP", "FN")) != EXPECTED_ANCHOR:
        raise RuntimeError("internal O0 aggregate changed after anchor gate")

    video_count_by_subject = Counter(subject for subject, _video in gated["loaded"])
    subject_rows = []
    stability = {}
    sensitivity = {}
    for method in METHODS:
        labels = Counter()
        for subject in subjects:
            current = native.metrics(subject_counts[method][subject])
            base = native.metrics(subject_counts[METHODS[0]][subject])
            label = "baseline" if method == METHODS[0] else comparison_label(current["F1"], base["F1"])
            if method != METHODS[0]:
                labels[label] += 1
            subject_rows.append(
                {
                    "subject": subject,
                    "method": method,
                    "video_count": video_count_by_subject[subject],
                    **current,
                    "Delta_TP_vs_O0": current["TP"] - base["TP"],
                    "Delta_FP_vs_O0": current["FP"] - base["FP"],
                    "Delta_FN_vs_O0": current["FN"] - base["FN"],
                    "Delta_F1_vs_O0": current["F1"] - base["F1"],
                    "comparison_vs_O0": label,
                }
            )
        if method != METHODS[0]:
            stability[method] = {key: labels.get(key, 0) for key in ("improved", "equal", "worse")}
            deltas = []
            for held in subjects:
                candidate_leave_one = sum_counts(
                    subject_counts[method][subject] for subject in subjects if subject != held
                )
                baseline_leave_one = sum_counts(
                    subject_counts[METHODS[0]][subject] for subject in subjects if subject != held
                )
                deltas.append(f1_from_counts(candidate_leave_one) - f1_from_counts(baseline_leave_one))
            sensitivity[method] = {
                "min": float(np.min(deltas)),
                "median": float(np.median(deltas)),
                "max": float(np.max(deltas)),
            }

    summary_rows = []
    for method in METHODS:
        current = aggregate[method]
        summary_rows.append(
            {
                "method": method,
                **current,
                "predicted_events": current["event_count"],
                "events_per_video": current["event_count"] / EXPECTED_VIDEOS,
                "Delta_TP_vs_O0": current["TP"] - baseline["TP"],
                "Delta_FP_vs_O0": current["FP"] - baseline["FP"],
                "Delta_FN_vs_O0": current["FN"] - baseline["FN"],
                "Delta_F1_vs_O0": current["F1"] - baseline["F1"],
                **aggregate_curve[method],
            }
        )
    return {
        "aggregate": aggregate,
        "summary_rows": summary_rows,
        "subject_rows": subject_rows,
        "curve_rows": curve_rows,
        "aggregate_curve": aggregate_curve,
        "stability": stability,
        "sensitivity": sensitivity,
        "coverage": {
            "frames_covered_by_1_window": total_coverage.get(1, 0),
            "frames_covered_by_2_windows": total_coverage.get(2, 0),
            "maximum_overlap_count": max(total_coverage, default=0),
            "other_overlap_counts": {
                str(key): value for key, value in sorted(total_coverage.items()) if key not in (1, 2)
            },
        },
        "decision": decision(aggregate),
    }


def report_markdown(gated: dict, result: dict) -> str:
    aggregate = result["aggregate"]
    base = aggregate[METHODS[0]]
    lines = [
        "# Overlap-Aware Aggregation Feasibility — SAMMLV",
        "",
        "## 结论",
        "",
        f"`{result['decision']}`",
        "",
        "本实验只改变 raw overlapping-window spotting score 的聚合方式；三个分支使用完全相同的 fold-wise Final Strong Native decoder。未使用 recognition logits，也未执行 backbone forward、调参、训练、rescue、pruning 或其它 decoder。",
        "",
        "## Provenance 与 O0 anchor gate",
        "",
        f"- Raw source：`{gated['manifest_path']}`",
        f"- Raw manifest SHA-256：`{gated['manifest_sha']}`",
        f"- Compact cache SHA-256：`{EXPECTED_CACHE_SHA}`",
        f"- Frozen config report SHA-256：`{gated['config_sha']}`",
        "- Raw source：29 subjects / 79 videos，79/79 reconstruction exact PASS。",
        f"- O0 event anchor：{base['TP']}/{base['FP']}/{base['FN']}，F1={base['F1']:.6f}，PASS。",
        "",
        "## Aggregate results",
        "",
        "| Method | TP | FP | FN | Precision | Recall | F1 | Events | Events/video | ΔTP | ΔFP | ΔFN | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate[method]
        lines.append(
            f"| {method} | {row['TP']} | {row['FP']} | {row['FN']} | "
            f"{row['Precision']:.6f} | {row['Recall']:.6f} | {row['F1']:.6f} | "
            f"{row['event_count']} | {row['event_count']/EXPECTED_VIDEOS:.6f} | "
            f"{row['TP']-base['TP']:+d} | {row['FP']-base['FP']:+d} | "
            f"{row['FN']-base['FN']:+d} | {row['F1']-base['F1']:+.6f} |"
        )

    coverage = result["coverage"]
    lines += [
        "",
        "## Aggregation coverage",
        "",
        f"- 1-window frames：{coverage['frames_covered_by_1_window']}",
        f"- 2-window frames：{coverage['frames_covered_by_2_windows']}",
        f"- Maximum overlap count：{coverage['maximum_overlap_count']}",
        f"- Other overlap counts：`{json.dumps(coverage['other_overlap_counts'], ensure_ascii=False)}`",
        "",
        "## Curve-level diagnostics（all 44,295 frames pooled）",
        "",
        "| Method | Pearson vs Original | MAE vs Original | Max abs difference |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        curve = result["aggregate_curve"][method]
        pearson_text = "NA" if curve["pearson_vs_original"] is None else f"{curve['pearson_vs_original']:.9f}"
        lines.append(
            f"| {method} | {pearson_text} | {curve['mae_vs_original']:.9f} | "
            f"{curve['max_abs_difference_vs_original']:.9f} |"
        )
    lines += [
        "",
        "## Subject stability",
        "",
        "| Method | Improved | Equal | Worse | Leave-one ΔF1 min | median | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS[1:]:
        stable = result["stability"][method]
        sensitivity = result["sensitivity"][method]
        lines.append(
            f"| {method} | {stable['improved']} | {stable['equal']} | {stable['worse']} | "
            f"{sensitivity['min']:+.6f} | {sensitivity['median']:+.6f} | {sensitivity['max']:+.6f} |"
        )

    lines += [
        "",
        "## Curve diagnostics",
        "",
        "逐视频 Pearson、MAE、最大绝对差及 overlap coverage 已写入 `outputs/overlap_curve_diagnostics.csv`。O0 必须逐视频与保存的 original curve 完全相同。",
        "",
        "## Decision rule",
        "",
        f"本次预注册式解释使用绝对 ΔF1={DECISION_TOLERANCE:.3f} 作为“明显变化”容差：任一 O1/O2 超过该值为 signal positive；两者均低于负容差为 simple aggregation no-go；其余为 inconclusive。该容差只用于结果标签，不参与 decoder 或 aggregation。",
        "",
        "## Interpretation boundary",
        "",
        "`SIMPLE-AGGREGATION-NO-GO` 只否定这里固定的 uniform mean 与固定 center weighting，不等价于 `WINDOW-CONSENSUS-NO-GO`。本实验没有测试 cross-window support、trajectory 或 location stability。",
        "",
        "实验到此停止；未自动进入 Window-Consensus feature development。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        gated = provenance_gate(args)
    except Exception as error:
        blocked_report(args.output_root, f"Anchor/provenance gate 失败：`{type(error).__name__}: {error}`")
        print("BLOCKED-OVERLAP-AGGREGATION-ANCHOR")
        print(f"{type(error).__name__}: {error}")
        return 2

    result = run(args, gated)
    output_dir = args.output_root / "outputs"
    write_csv(output_dir / "overlap_aggregation_summary.csv", result["summary_rows"])
    write_csv(output_dir / "overlap_aggregation_subject_metrics.csv", result["subject_rows"])
    write_csv(output_dir / "overlap_curve_diagnostics.csv", result["curve_rows"])
    report_path = args.output_root / "OVERLAP_AGGREGATION_FEASIBILITY_SAMMLV_CN.md"
    report_path.write_text(report_markdown(gated, result), encoding="utf-8")
    print(result["decision"])
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
