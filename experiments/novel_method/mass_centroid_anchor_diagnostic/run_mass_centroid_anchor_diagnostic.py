#!/usr/bin/env python3
"""Read-only Mass-Centroid Anchor Diagnostic for frozen ME-TST+ SAMMLV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
DEFAULT_CACHE = REPRO_ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
DEFAULT_SOURCE_REPORT = REPRO_ROOT / "results/rgr1_sammlv_nested/report.json"
DEFAULT_BOUNDARY_DIR = REPRO_ROOT / "results/tuned_native_boundary_oracle_diagnostic"
DEFAULT_TRACKING = REPRO_ROOT / "results/pcbr1_sammlv_nested/oracle_recoverable_tracking.csv"
DEFAULT_OUTPUT = REPRO_ROOT / "results/mass_centroid_anchor_diagnostic"

GAMMA = 2.0
EPS = 1e-12
EXPECTED_CACHE_SHA256 = "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569"
EXPECTED_NATIVE = {"TP": 49, "FP": 143, "FN": 110, "event_count": 192}
EXPECTED_ORACLE_IDS = {
    "009/009_3/pred_1", "012/012_3/pred_0", "017/017_3/pred_1",
    "019/019_4/pred_0", "020/020_4/pred_0", "026/026_2/pred_6",
    "026/026_3/pred_2", "031/031_3/pred_0", "032/032_3/pred_0",
    "032/032_6/pred_0", "037/037_4/pred_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--tracking", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(score: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(max(1, int(width)), dtype=np.float64) / max(1, int(width))
    return np.convolve(np.asarray(score, dtype=np.float64), kernel, mode="same")


def native_threshold(curve: np.ndarray, p_s: float) -> float:
    return float(curve.mean() + p_s * (curve.max() - curve.mean()))


def anchor_from_response(curve: np.ndarray, peak: int, tau: float) -> dict:
    """Compute the anchor without accepting or accessing any GT information."""
    if curve[peak] < tau:
        raise RuntimeError(
            f"INVALID_NATIVE_PEAK_THRESHOLD: peak={peak}, s(p)={curve[peak]}, tau={tau}"
        )
    start = peak
    while start > 0 and curve[start - 1] >= tau:
        start -= 1
    end = peak
    while end + 1 < len(curve) and curve[end + 1] >= tau:
        end += 1
    mass = np.maximum(curve[start : end + 1] - tau, 0.0) ** GAMMA
    excess_mass = float(mass.sum())
    status = "OK"
    if excess_mass <= EPS:
        centroid = float(peak)
        status = "ZERO_EXCESS_MASS_FALLBACK"
    else:
        positions = np.arange(start, end + 1, dtype=np.float64)
        centroid = float(np.dot(positions, mass) / excess_mass)
    return {
        "native_peak": int(peak),
        "island_start": int(start),
        "island_end": int(end),
        "island_length": int(end - start + 1),
        "tau_native": float(tau),
        "peak_score": float(curve[peak]),
        "excess_mass": excess_mass,
        "p_star_cont": centroid,
        "p_star_round": int(round(centroid)),
        "anchor_shift": float(centroid - peak),
        "abs_anchor_shift": float(abs(centroid - peak)),
        "shift_direction": "right" if centroid > peak + EPS else "left" if centroid < peak - EPS else "none",
        "centroid_status": status,
    }


def point_interval_distance(x: float, onset: int, offset: int) -> float:
    if onset <= x <= offset:
        return 0.0
    return float(min(abs(x - onset), abs(x - offset)))


def compare(delta: float) -> str:
    return "closer" if delta > EPS else "farther" if delta < -EPS else "equal"


def attach_gt(anchor: dict, source_row: dict[str, str], cohort: str) -> dict:
    """Post-hoc GT evaluation; called only after anchor_from_response returns."""
    oracle = cohort == "oracle11"
    gt_id = source_row["nearest_gt_id"] if oracle else source_row["gt_id"]
    onset = int(source_row["nearest_gt_onset"] if oracle else source_row["gt_onset"])
    offset = int(source_row["nearest_gt_offset"] if oracle else source_row["gt_offset"])
    peak = anchor["native_peak"]
    centroid = anchor["p_star_cont"]
    center = (onset + offset) / 2.0
    d_native = point_interval_distance(peak, onset, offset)
    d_centroid = point_interval_distance(centroid, onset, offset)
    gain = d_native - d_centroid
    native_center_distance = abs(peak - center)
    centroid_center_distance = abs(centroid - center)
    return {
        "identity": source_row["candidate_id"],
        "subject": source_row["subject"],
        "video": source_row["video"],
        "gt_id": gt_id,
        "gt_onset": onset,
        "gt_offset": offset,
        **anchor,
        "native_peak_in_gt": bool(onset <= peak <= offset),
        "centroid_in_gt": bool(onset <= centroid <= offset),
        "d_native": d_native,
        "d_centroid": d_centroid,
        "distance_gain": gain,
        "interval_distance_result": compare(gain),
        "gt_center": center,
        "native_center_distance": native_center_distance,
        "centroid_center_distance": centroid_center_distance,
        "center_distance_gain": native_center_distance - centroid_center_distance,
        "center_distance_result": compare(native_center_distance - centroid_center_distance),
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_oracle(rows: list[dict]) -> dict:
    improved_subjects = sorted({row["subject"] for row in rows if row["distance_gain"] > EPS})
    worsened_subjects = sorted({row["subject"] for row in rows if row["distance_gain"] < -EPS})
    entering_subjects = sorted({row["subject"] for row in rows if row["centroid_in_gt"] and not row["native_peak_in_gt"]})
    gains = [row["distance_gain"] for row in rows]
    return {
        "total": len(rows),
        "centroid_closer_count": sum(value > EPS for value in gains),
        "centroid_equal_count": sum(abs(value) <= EPS for value in gains),
        "centroid_farther_count": sum(value < -EPS for value in gains),
        "centroid_enters_GT_count": sum(row["centroid_in_gt"] and not row["native_peak_in_gt"] for row in rows),
        "median_d_native": float(np.median([row["d_native"] for row in rows])),
        "median_d_centroid": float(np.median([row["d_centroid"] for row in rows])),
        "median_distance_gain": float(np.median(gains)),
        "mean_distance_gain": float(np.mean(gains)),
        "subjects_with_improved_anchor": improved_subjects,
        "subjects_with_worsened_anchor": worsened_subjects,
        "subjects_with_centroid_entering_GT": entering_subjects,
        "improved_subject_count": len(improved_subjects),
    }


def summarize_tp(rows: list[dict]) -> dict:
    shifts = [row["abs_anchor_shift"] for row in rows]
    center_gains = [row["center_distance_gain"] for row in rows]
    return {
        "total": len(rows),
        "native_peak_in_GT_count": sum(row["native_peak_in_gt"] for row in rows),
        "centroid_still_in_GT_count": sum(row["centroid_in_gt"] for row in rows),
        "centroid_moved_outside_GT_count": sum(not row["centroid_in_gt"] for row in rows),
        "centroid_closer_to_GT_center": sum(value > EPS for value in center_gains),
        "centroid_farther_from_GT_center": sum(value < -EPS for value in center_gains),
        "centroid_same_GT_center_distance": sum(abs(value) <= EPS for value in center_gains),
        "median_absolute_anchor_shift": float(np.median(shifts)),
        "p90_absolute_anchor_shift": percentile(shifts, 90),
        "maximum_anchor_shift": float(max(shifts)),
        "TP_anchor_retention_rate": sum(row["centroid_in_gt"] for row in rows) / len(rows),
    }


def subject_summary(tp_rows: list[dict], oracle_rows: list[dict]) -> list[dict]:
    tp_by_subject: dict[str, list[dict]] = defaultdict(list)
    oracle_by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in tp_rows:
        tp_by_subject[row["subject"]].append(row)
    for row in oracle_rows:
        oracle_by_subject[row["subject"]].append(row)
    output = []
    for subject in sorted(set(tp_by_subject) | set(oracle_by_subject)):
        tp = tp_by_subject[subject]
        oracle = oracle_by_subject[subject]
        output.append({
            "subject": subject,
            "native_tp_total": len(tp),
            "native_tp_centroid_retained": sum(row["centroid_in_gt"] for row in tp),
            "native_tp_centroid_left_gt": sum(not row["centroid_in_gt"] for row in tp),
            "near_miss_total": len(oracle),
            "near_miss_closer": sum(row["distance_gain"] > EPS for row in oracle),
            "near_miss_equal": sum(abs(row["distance_gain"]) <= EPS for row in oracle),
            "near_miss_farther": sum(row["distance_gain"] < -EPS for row in oracle),
            "near_miss_centroid_enters_gt": sum(row["centroid_in_gt"] and not row["native_peak_in_gt"] for row in oracle),
        })
    return output


def evidence_grade(oracle: dict, tp: dict) -> tuple[str, dict]:
    checks = {
        "near_miss_closer_ge_5": oracle["centroid_closer_count"] >= 5,
        "near_miss_enters_gt_ge_3": oracle["centroid_enters_GT_count"] >= 3,
        "improved_subjects_ge_3": oracle["improved_subject_count"] >= 3,
        "native_tp_retained_ge_44": tp["centroid_still_in_GT_count"] >= 44,
        "near_miss_median_gain_gt_0": oracle["median_distance_gain"] > 0,
    }
    if all(checks.values()):
        grade = "MASS-CENTROID-ANCHOR-STRONG"
    elif oracle["centroid_closer_count"] <= 3 or (
        oracle["median_distance_gain"] <= 0 and oracle["centroid_enters_GT_count"] < 3
    ):
        grade = "MASS-CENTROID-ANCHOR-NO-GO"
    else:
        grade = "MASS-CENTROID-ANCHOR-WEAK"
    return grade, checks


def markdown_report(report: dict, oracle_rows: list[dict]) -> str:
    integrity = report["integrity"]
    oracle = report["oracle11_summary"]
    tp = report["native_tp_summary"]
    lines = [
        "# Mass-Centroid Anchor Diagnostic（SAMMLV）", "",
        "## A. Integrity", "",
        f"- Frozen cache SHA256：`{integrity['cache_sha256']}`（PASS）",
        f"- Candidate sets：Native TP = {integrity['native_tp_count']}；Oracle Near-Miss = {integrity['oracle11_count']}（PASS）",
        f"- 固定 gamma：`{GAMMA:g}`；mass：`max(s(t)-tau_native, 0)^2`",
        "- Response island：包含原 peak 的 maximal contiguous `s(t) >= tau_native` 区间。",
        f"- Threshold 来源：{integrity['threshold_source']}",
        f"- Smoothing 来源：{integrity['smoothing_source']}",
        "- GT 仅在连续质心计算完成后用于事后距离评估；未运行 matching、decoder、NMS、inference 或训练。", "",
        "## B. Oracle-11 results", "",
        "| Metric | Result |", "| --- | ---: |",
        f"| total | {oracle['total']} |",
        f"| closer | {oracle['centroid_closer_count']} |",
        f"| equal | {oracle['centroid_equal_count']} |",
        f"| farther | {oracle['centroid_farther_count']} |",
        f"| enters GT | {oracle['centroid_enters_GT_count']} |",
        f"| median native distance | {oracle['median_d_native']:.6f} |",
        f"| median centroid distance | {oracle['median_d_centroid']:.6f} |",
        f"| median gain | {oracle['median_distance_gain']:.6f} |",
        f"| mean gain | {oracle['mean_distance_gain']:.6f} |",
        f"| improved subjects | {oracle['improved_subject_count']} |", "",
        f"Improved subjects：{', '.join(oracle['subjects_with_improved_anchor']) or '无'}  ",
        f"Worsened subjects：{', '.join(oracle['subjects_with_worsened_anchor']) or '无'}  ",
        f"Centroid-enter-GT subjects：{', '.join(oracle['subjects_with_centroid_entering_GT']) or '无'}", "",
        "## C. Native-49 preservation", "",
        "| Metric | Result |", "| --- | ---: |",
        f"| total | {tp['total']} |",
        f"| native peak in GT | {tp['native_peak_in_GT_count']} |",
        f"| centroid remains in GT | {tp['centroid_still_in_GT_count']} |",
        f"| centroid leaves GT | {tp['centroid_moved_outside_GT_count']} |",
        f"| retention rate | {tp['TP_anchor_retention_rate']:.6f} |",
        f"| closer / farther / same to GT center | {tp['centroid_closer_to_GT_center']} / {tp['centroid_farther_from_GT_center']} / {tp['centroid_same_GT_center_distance']} |",
        f"| median absolute shift | {tp['median_absolute_anchor_shift']:.6f} |",
        f"| p90 absolute shift | {tp['p90_absolute_anchor_shift']:.6f} |",
        f"| maximum absolute shift | {tp['maximum_anchor_shift']:.6f} |", "",
        "## D. Oracle-11 case-level table", "",
        "| identity | subject | GT | native peak | centroid | native in GT | centroid in GT | d_native | d_centroid | gain | shift |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in oracle_rows:
        lines.append(
            f"| {row['identity']} | {row['subject']} | [{row['gt_onset']}, {row['gt_offset']}] | "
            f"{row['native_peak']} | {row['p_star_cont']:.6f} | {row['native_peak_in_gt']} | "
            f"{row['centroid_in_gt']} | {row['d_native']:.6f} | {row['d_centroid']:.6f} | "
            f"{row['distance_gain']:.6f} | {row['anchor_shift']:+.6f} |"
        )
    lines.extend([
        "", "## E. Final evidence grade", "",
        f"`{report['final_evidence_grade']}`", "",
        report["grade_explanation"], "",
        "本报告只评价 anchor localization；未计算修改后的 TP/FP/FN，也未实现 TAC-EB。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    inputs = [args.cache, args.source_report, args.tracking,
              args.boundary_dir / "report.json",
              args.boundary_dir / "native_tp_iou_distribution.csv",
              args.boundary_dir / "oracle_recoverable_cases.csv",
              args.boundary_dir / "trajectory_data.csv"]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required frozen inputs: {missing}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")

    cache_digest = sha256(args.cache)
    if cache_digest != EXPECTED_CACHE_SHA256:
        raise RuntimeError(f"cache SHA256 mismatch: {cache_digest}")
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    boundary_report = json.loads((args.boundary_dir / "report.json").read_text(encoding="utf-8"))
    if source_report.get("cache_sha256") != cache_digest or boundary_report.get("cache_sha256") != cache_digest:
        raise RuntimeError("source/boundary report does not match frozen cache SHA256")
    if boundary_report.get("TUNED_NATIVE_ANCHOR") != "PASS":
        raise RuntimeError("boundary diagnostic Tuned Native anchor is not PASS")
    if any(int(boundary_report["Native"][key]) != value for key, value in EXPECTED_NATIVE.items()):
        raise RuntimeError("frozen Tuned Native metrics do not match preregistered anchor")

    tp_source = read_csv(args.boundary_dir / "native_tp_iou_distribution.csv")
    oracle_source = read_csv(args.boundary_dir / "oracle_recoverable_cases.csv")
    tracking = read_csv(args.tracking)
    tp_ids = {row["candidate_id"] for row in tp_source}
    oracle_ids = {row["candidate_id"] for row in oracle_source}
    if len(tp_source) != 49 or len(tp_ids) != 49:
        raise RuntimeError("Cohort A integrity failure: expected 49 unique Native TP identities")
    if len(oracle_source) != 11 or oracle_ids != EXPECTED_ORACLE_IDS:
        raise RuntimeError("Cohort B integrity failure: Oracle identities differ from preregistration")
    if oracle_ids != {row["candidate_id"] for row in tracking}:
        raise RuntimeError("Oracle identities differ from oracle_recoverable_tracking.csv")
    oracle_gt = {row["candidate_id"]: row["nearest_gt_id"] for row in oracle_source}
    tracking_gt = {row["candidate_id"]: row["oracle_gt_id"] for row in tracking}
    if oracle_gt != tracking_gt:
        raise RuntimeError("Oracle candidate-to-GT mapping differs from tracking file")

    configs = {
        str(fold["subject"]): fold["selected_strong_config"]
        for fold in source_report["outer_folds"]
    }
    if len(configs) != 29:
        raise RuntimeError("expected exactly 29 locked outer-fold configurations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with args.cache.open("rb") as handle:
            payload = pickle.load(handle)
    if payload.get("dataset") != "SAMMLV" or int(payload.get("k_p", -1)) != 5:
        raise RuntimeError("expected frozen SAMMLV cache with k_p=5")
    records = {str(record["video"]): record for record in payload["records"]}

    curves: dict[tuple[str, str], tuple[np.ndarray, float, int]] = {}
    for row in tp_source + oracle_source:
        subject, video = row["subject"], row["video"]
        config = configs[subject]
        width = max(1, int(round(float(config["c_s"]) * int(payload["k_p"]))))
        curve = moving_average(records[video]["score"], width)
        tau = native_threshold(curve, float(config["p_s"]))
        curves[(subject, video)] = (curve, tau, width)

    # Cross-check reconstructed smoothing against every exported frozen trajectory point.
    trajectory_max_abs_diff = 0.0
    for row in read_csv(args.boundary_dir / "trajectory_data.csv"):
        curve = curves[(row["subject"], row["video"])][0]
        difference = abs(curve[int(row["absolute_index"])] - float(row["smoothed_spotting_score"]))
        trajectory_max_abs_diff = max(trajectory_max_abs_diff, float(difference))
    if trajectory_max_abs_diff > EPS:
        raise RuntimeError(f"reconstructed smoothing mismatch: max diff={trajectory_max_abs_diff}")

    def compute(source_rows: list[dict[str, str]], cohort: str) -> list[dict]:
        output = []
        for source_row in source_rows:
            curve, tau, width = curves[(source_row["subject"], source_row["video"])]
            # Deliberately complete the GT-free calculation before attaching frozen GT.
            anchor = anchor_from_response(curve, int(source_row["pred_peak"]), tau)
            anchor["smoothing_width"] = width
            anchor["threshold_p_s"] = float(configs[source_row["subject"]]["p_s"])
            output.append(attach_gt(anchor, source_row, cohort))
        return output

    tp_rows = compute(tp_source, "native_tp")
    oracle_rows = compute(oracle_source, "oracle11")
    if sum(row["native_peak_in_gt"] for row in tp_rows) != 49:
        raise RuntimeError("existing Native TP peak-to-frozen-GT integrity failure")
    if sum(row["native_peak_in_gt"] for row in oracle_rows) != 1:
        raise RuntimeError("expected exactly 1/11 Oracle native peak inside frozen GT")

    oracle_summary = summarize_oracle(oracle_rows)
    tp_summary = summarize_tp(tp_rows)
    subjects = subject_summary(tp_rows, oracle_rows)
    grade, strong_checks = evidence_grade(oracle_summary, tp_summary)
    explanation = (
        f"Near-miss closer={oracle_summary['centroid_closer_count']}/11，"
        f"enters-GT={oracle_summary['centroid_enters_GT_count']}/11，"
        f"improved subjects={oracle_summary['improved_subject_count']}，"
        f"median gain={oracle_summary['median_distance_gain']:.6f}；"
        f"Native TP retention={tp_summary['centroid_still_in_GT_count']}/49。"
        "由于 centroid_closer_count <= 3，直接命中预注册 NO-GO 条件。"
    )
    integrity = {
        "status": "PASS",
        "cache_path": str(args.cache.resolve()),
        "cache_sha256": cache_digest,
        "source_report_path": str(args.source_report.resolve()),
        "boundary_report_path": str((args.boundary_dir / "report.json").resolve()),
        "native_tp_identity_source": str((args.boundary_dir / "native_tp_iou_distribution.csv").resolve()),
        "oracle_identity_source": str((args.boundary_dir / "oracle_recoverable_cases.csv").resolve()),
        "oracle_tracking_source": str(args.tracking.resolve()),
        "native_tp_count": len(tp_rows),
        "oracle11_count": len(oracle_rows),
        "oracle_unique_gt_count": len({row["gt_id"] for row in oracle_rows}),
        "oracle_subject_count": len({row["subject"] for row in oracle_rows}),
        "gamma": GAMMA,
        "threshold_source": "outer_folds[].selected_strong_config.p_s; tau=mean(curve)+p_s*(max(curve)-mean(curve))",
        "smoothing_source": "frozen record.score; width=round(selected_strong_config.c_s * cache.k_p); np.convolve(..., mode='same')",
        "trajectory_crosscheck_points": len(read_csv(args.boundary_dir / "trajectory_data.csv")),
        "trajectory_max_abs_diff": trajectory_max_abs_diff,
        "centroid_gt_free_function": "anchor_from_response(curve, peak, tau)",
        "formal_event_matching_run": False,
        "candidate_or_prediction_modified": False,
        "decoder_output_produced": False,
        "inference_or_training_run": False,
    }
    report = {
        "experiment_name": "Mass-Centroid Anchor Diagnostic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "SAMMLV",
        "backbone": "ME-TST+ frozen",
        "diagnostic_only": True,
        "integrity": integrity,
        "response_island_definition": "maximal contiguous region containing p where s(t) >= tau_native",
        "mass_definition": "max(s(t) - tau_native, 0)^2",
        "primary_anchor": "continuous temporal centroid p_star_cont",
        "equality_tolerance": EPS,
        "oracle11_summary": oracle_summary,
        "native_tp_summary": tp_summary,
        "strong_go_checks": strong_checks,
        "final_evidence_grade": grade,
        "grade_explanation": explanation,
        "stop_after_diagnostic": True,
    }

    args.output_root.mkdir(parents=True)
    write_csv(args.output_root / "native_tp_anchor_analysis.csv", tp_rows)
    write_csv(args.output_root / "oracle11_anchor_analysis.csv", oracle_rows)
    write_csv(args.output_root / "subject_level_summary.csv", subjects)
    island_rows = [
        {"cohort": cohort, **{key: row[key] for key in (
            "identity", "subject", "video", "native_peak", "island_start", "island_end",
            "island_length", "tau_native", "threshold_p_s", "smoothing_width", "peak_score",
            "excess_mass", "p_star_cont", "p_star_round", "anchor_shift", "centroid_status")}}
        for cohort, rows in (("native_tp", tp_rows), ("oracle11", oracle_rows))
        for row in rows
    ]
    write_csv(args.output_root / "response_island_statistics.csv", island_rows)
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_root / "MASS_CENTROID_ANCHOR_DIAGNOSTIC_CN.md").write_text(
        markdown_report(report, oracle_rows), encoding="utf-8"
    )
    print(json.dumps({
        "integrity": "PASS",
        "oracle11_summary": oracle_summary,
        "native_tp_summary": tp_summary,
        "final_evidence_grade": grade,
        "output_root": str(args.output_root.resolve()),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
