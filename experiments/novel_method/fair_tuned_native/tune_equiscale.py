"""Nested-LOSO EquiScale tuning for the verified BoostingVRME curve caches.

This post-processing experiment deliberately consumes only ``curve_cache``.
It neither modifies the published BoostingVRME repository nor references the
separate B2/experiments work.  Candidate intervals and final NMS retain the
original BoostingVRME implementation so that tuned Native and EquiScale differ
only in candidate generation and cross-scale evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent
CURVE_CACHE = ROOT / "curve_cache"
RESULT_ROOT = ROOT / "results" / "equiscale_tuning"

# Coarse candidate-generation search.  Scales are relative to the dataset's
# k_p, and the five sets cover narrow, centred, audit-like, and wide ranges.
SCALE_SETS: Mapping[str, Tuple[float, ...]] = {
    "low_compact": (0.50, 0.80, 1.25),
    "balanced_wide": (0.67, 1.00, 1.50),
    "centred_3": (0.80, 1.00, 1.25),
    "centred_4": (0.80, 1.00, 1.25, 1.50),
    "audit_5": (0.67, 0.80, 1.00, 1.25, 1.50),
    "high_wide": (1.00, 1.50, 2.00),
}
ALL_SCALES = tuple(sorted({scale for scales in SCALE_SETS.values() for scale in scales}))
SMOOTH_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5)
THRESHOLD_FACTORS = (0.40, 0.50, 0.60)
DISTANCE_MULTIPLIERS = (0.75, 1.0, 1.25)
ASSOCIATION_MULTIPLIERS = (0.25, 0.50, 0.75)
SUPPORT_MINIMUMS = (1, 2, 3)
FUSION_WEIGHTS = (0.25, 0.50, 0.75)
FUSION_THRESHOLDS = (0.0, 0.5)
AGGREGATORS = ("max", "mean", "top2_mean")


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def as_dict(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn) if 2 * self.tp + self.fp + self.fn else 0.0
        return {"TP": self.tp, "FP": self.fp, "FN": self.fn, "precision": precision, "recall": recall, "F1": f1}


@dataclass(frozen=True)
class NativeConfig:
    smooth: float
    threshold: float
    distance: float

    @property
    def identifier(self) -> str:
        return f"native|cs={self.smooth:g}|p={self.threshold:g}|d={self.distance:g}"


@dataclass(frozen=True)
class EquiConfig:
    scale_set: str
    smooth: float
    threshold: float
    distance: float
    association: float
    mode: str
    support_minimum: int | None = None
    aggregator: str | None = None
    fusion_weight: float | None = None
    fusion_threshold: float | None = None

    @property
    def identifier(self) -> str:
        head = (
            f"equi|scales={self.scale_set}|cs={self.smooth:g}|p={self.threshold:g}"
            f"|d={self.distance:g}|assoc={self.association:g}"
        )
        if self.mode == "support":
            return f"{head}|mode=support|min={self.support_minimum}"
        return (
            f"{head}|mode=fusion|agg={self.aggregator}|lambda={self.fusion_weight:g}"
            f"|tau={self.fusion_threshold:g}"
        )


@dataclass(frozen=True)
class Cluster:
    peak: int
    interval: Tuple[int, int]
    support: int
    maximum: float
    mean: float
    top2_mean: float

    def confidence(self, aggregator: str) -> float:
        return {"max": self.maximum, "mean": self.mean, "top2_mean": self.top2_mean}[aggregator]


Detection = Tuple[int, int, int]


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(np.asarray(values, dtype=float), np.ones(width, dtype=float) / width, mode="same")


def k_prime(records: Sequence[dict]) -> int:
    durations = sorted(event[2] - event[0] for record in records for event in record["gt"])
    return int((durations[len(durations) // 2] + 1) / 2)


def source_interval(curve: np.ndarray, peak: int, k_p: int) -> Tuple[int, int]:
    """The original BoostingVRME interval expansion, kept unchanged."""
    length = len(curve)
    left = max(0, peak - k_p)
    right = min(length - 1, peak + k_p)
    mean = float(curve.mean())
    start = next((index - 1 for index in range(left + 1, peak) if curve[index] > curve[index - 1] and curve[index] > mean / 1.5), left)
    end = next((index + 1 for index in range(right - 1, peak, -1) if curve[index] > curve[index + 1] and curve[index] > mean / 1.5), right)
    for index in range(start - 1, max(-1, peak - k_p * 5) - 1, -1):
        if index > 0 and curve[index] < mean and curve[index - 1] < mean:
            start = index + 1
            break
    for index in range(end + 1, min(length - 1, peak + k_p * 5) + 1):
        if index < length - 1 and curve[index] < mean and curve[index + 1] < mean:
            end = index - 1
            break
    return max(0, start), min(length - 1, end)


def source_nms_iou(first: Sequence[int], second: Sequence[int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def apply_source_nms(clusters: Sequence[Cluster], selected: Sequence[int], k_p: int) -> List[Detection]:
    kept: List[Detection] = []
    for index in sorted(selected, key=lambda value: (clusters[value].interval[0], clusters[value].peak)):
        cluster = clusters[index]
        candidate = (cluster.interval[0], cluster.interval[1], cluster.peak)
        if all(source_nms_iou(candidate[:2], old[:2]) < 0.2 and abs(candidate[2] - old[2]) > k_p for old in kept):
            kept.append(candidate)
    return kept


def native_predictions(curve: np.ndarray, k_p: int, config: NativeConfig) -> List[Detection]:
    smoothed = moving_average(curve, round(config.smooth * k_p))
    threshold = float(smoothed.mean() + config.threshold * (smoothed.max() - smoothed.mean()))
    peaks, _ = find_peaks(smoothed, height=threshold, distance=max(1, round(config.distance * k_p)))
    candidates = [(*source_interval(curve, int(peak), k_p), int(peak)) for peak in peaks]
    kept: List[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if all(source_nms_iou(candidate[:2], previous[:2]) < 0.2 and abs(candidate[2] - previous[2]) > k_p for previous in kept):
            kept.append(candidate)
    return kept


def interval_iou(first: Sequence[int], second: Sequence[int]) -> float:
    left = max(int(first[0]), int(second[0]))
    right = min(int(first[1]), int(second[2]))
    intersection = max(0, right - left + 1)
    if not intersection:
        return 0.0
    union = (int(first[1]) - int(first[0]) + 1) + (int(second[2]) - int(second[0]) + 1) - intersection
    return intersection / union


def count_video(detections: Sequence[Detection], record: dict) -> Counts:
    matched: set[int] = set()
    tp = fp = 0
    for detection in detections:
        overlaps = [interval_iou(detection[:2], event) for event in record["gt"]]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] >= 0.5 and best not in matched:
            matched.add(best)
            tp += 1
        else:
            fp += 1
    return Counts(tp=tp, fp=fp, fn=len(record["gt"]) - len(matched))


def robust_z(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return array
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    if scale <= 1e-12:
        scale = float(np.std(array))
    return (array - median) / scale if scale > 1e-12 else np.zeros_like(array)


def scale_candidates(
    curves: Sequence[np.ndarray], k_p: int, smooth: float, threshold: float, distance: float
) -> Dict[float, List[List[Tuple[int, float]]]]:
    """Generate each scale once for one (cs, p, d) tuple."""
    output: Dict[float, List[List[Tuple[int, float]]]] = {}
    for scale in ALL_SCALES:
        by_video: List[List[Tuple[int, float]]] = []
        width = max(1, round(smooth * scale * k_p))
        peak_distance = max(1, round(distance * scale * k_p))
        for curve in curves:
            smoothed = moving_average(curve, width)
            low = float(smoothed.mean())
            spread = max(float(smoothed.max() - low), 1e-12)
            peaks, _ = find_peaks(smoothed, height=low + threshold * spread, distance=peak_distance)
            by_video.append([(int(peak), float((smoothed[peak] - low) / spread)) for peak in peaks])
        output[scale] = by_video
    return output


def cluster_candidates(
    curve: np.ndarray,
    k_p: int,
    scale_set: Sequence[float],
    by_scale: Mapping[float, List[List[Tuple[int, float]]]],
    video_index: int,
    association: float,
) -> List[Cluster]:
    """Associate nearest peaks within r_assoc*k_p, keeping one peak per scale."""
    points = sorted(
        (peak, scale_index, height)
        for scale_index, scale in enumerate(scale_set)
        for peak, height in by_scale[scale][video_index]
    )
    tolerance = max(1, round(association * k_p))
    groups: List[Dict[int, Tuple[int, float]]] = []
    for peak, scale_index, height in points:
        eligible = []
        for group_index, group in enumerate(groups):
            center = float(np.mean([member[0] for member in group.values()]))
            if abs(peak - center) <= tolerance:
                eligible.append((abs(peak - center), group_index))
        if not eligible:
            groups.append({scale_index: (peak, height)})
            continue
        _, group_index = min(eligible)
        old = groups[group_index].get(scale_index)
        if old is None or height > old[1]:
            groups[group_index][scale_index] = (peak, height)

    output: List[Cluster] = []
    for group in groups:
        members = list(group.values())
        representative_peak, _ = max(members, key=lambda item: (item[1], -item[0]))
        heights = sorted((height for _, height in members), reverse=True)
        output.append(
            Cluster(
                peak=representative_peak,
                interval=source_interval(curve, representative_peak, k_p),
                support=len(group),
                maximum=heights[0],
                mean=float(np.mean(heights)),
                top2_mean=float(np.mean(heights[:2])),
            )
        )
    return sorted(output, key=lambda cluster: cluster.peak)


def equi_predictions(clusters: Sequence[Cluster], config: EquiConfig, k_p: int) -> List[Detection]:
    if config.mode == "support":
        selected = [index for index, cluster in enumerate(clusters) if cluster.support >= int(config.support_minimum)]
    else:
        heights = robust_z([cluster.confidence(str(config.aggregator)) for cluster in clusters])
        supports = robust_z([cluster.support for cluster in clusters])
        scores = float(config.fusion_weight) * heights + (1.0 - float(config.fusion_weight)) * supports
        selected = [index for index, score in enumerate(scores) if score >= float(config.fusion_threshold)]
    return apply_source_nms(clusters, selected, k_p)


def counts_matrix(
    predictions: Sequence[Sequence[Detection]], records: Sequence[dict], subject_index: Mapping[str, int]
) -> np.ndarray:
    output = np.zeros((len(subject_index), 3), dtype=np.int32)
    for detections, record in zip(predictions, records):
        count = count_video(detections, record)
        output[subject_index[str(record["subject"])] ] += (count.tp, count.fp, count.fn)
    return output


def native_configs() -> List[NativeConfig]:
    return [
        NativeConfig(smooth, threshold, distance)
        for smooth in SMOOTH_MULTIPLIERS
        for threshold in (0.35, 0.45, 0.55, 0.65)
        for distance in (0.50, 0.75, 1.0, 1.25)
    ]


def equi_configs() -> List[EquiConfig]:
    output: List[EquiConfig] = []
    for scale_set in SCALE_SETS:
        for smooth in SMOOTH_MULTIPLIERS:
            for threshold in THRESHOLD_FACTORS:
                for distance in DISTANCE_MULTIPLIERS:
                    for association in ASSOCIATION_MULTIPLIERS:
                        output.extend(
                            EquiConfig(scale_set, smooth, threshold, distance, association, "support", support_minimum=minimum)
                            for minimum in SUPPORT_MINIMUMS
                        )
                        output.extend(
                            EquiConfig(
                                scale_set,
                                smooth,
                                threshold,
                                distance,
                                association,
                                "fusion",
                                aggregator=aggregator,
                                fusion_weight=weight,
                                fusion_threshold=fusion_threshold,
                            )
                            for aggregator in AGGREGATORS
                            for weight in FUSION_WEIGHTS
                            for fusion_threshold in FUSION_THRESHOLDS
                        )
    return output


def score_key(values: np.ndarray, identifier: str) -> Tuple[float, float, int, str]:
    tp, fp, fn = (int(value) for value in values)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return f1, precision, -fp, identifier


def as_counts(values: np.ndarray) -> Counts:
    return Counts(*(int(value) for value in values))


def nested_select(
    configs: Sequence[str], stats: Mapping[str, np.ndarray], subjects: Sequence[str]
) -> Tuple[Counts, List[dict], Counter]:
    totals = {identifier: values.sum(axis=0) for identifier, values in stats.items()}
    final = np.zeros(3, dtype=np.int64)
    rows: List[dict] = []
    frequency: Counter = Counter()
    for held_index, subject in enumerate(subjects):
        options = ((score_key(totals[identifier] - stats[identifier][held_index], identifier), identifier) for identifier in configs)
        inner_score, selected = max(options, key=lambda item: item[0])
        test = stats[selected][held_index]
        final += test
        frequency[selected] += 1
        rows.append(
            {
                "subject": subject,
                "selected_config": selected,
                "inner_raw_f1": f"{inner_score[0]:.6f}",
                "test_raw_f1": f"{score_key(test, selected)[0]:.6f}",
            }
        )
    return as_counts(final), rows, frequency


def top_fixed_configs(configs: Sequence[str], stats: Mapping[str, np.ndarray], limit: int = 10) -> List[dict]:
    rows = []
    for identifier in configs:
        counts = as_counts(stats[identifier].sum(axis=0)).as_dict()
        rows.append({"config": identifier, **counts})
    return sorted(rows, key=lambda row: (row["F1"], row["precision"], -row["FP"], row["config"]), reverse=True)[:limit]


def write_selection_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "selected_config", "inner_raw_f1", "test_raw_f1"])
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(name: str) -> dict:
    cache_stem = {"sammlv": "sammlv", "casme3": "casme_3"}[name]
    with (CURVE_CACHE / f"{cache_stem}_curves.pkl").open("rb") as handle:
        package = pickle.load(handle)
    records = package["records"]
    curves = [np.asarray(curve, dtype=float) for curve in package["curves"]]
    if len(records) != len(curves) or not all(np.isfinite(curve).all() for curve in curves):
        raise RuntimeError(f"{name}: invalid curve cache")

    subjects = sorted({str(record["subject"]) for record in records}, key=lambda value: int(value) if value.isdigit() else value)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    k_p = k_prime(records)
    print(f"[{package['dataset']}] videos={len(records)}, subjects={len(subjects)}, k_p={k_p}", flush=True)

    native_stats: Dict[str, np.ndarray] = {}
    all_native = native_configs()
    for position, config in enumerate(all_native, start=1):
        predictions = [native_predictions(curve, k_p, config) for curve in curves]
        native_stats[config.identifier] = counts_matrix(predictions, records, subject_index)
        if position % 16 == 0:
            print(f"[{package['dataset']}] native grid {position}/{len(all_native)}", flush=True)

    all_equi = equi_configs()
    support_ids = [config.identifier for config in all_equi if config.mode == "support"]
    fusion_ids = [config.identifier for config in all_equi if config.mode == "fusion"]
    equi_stats: Dict[str, np.ndarray] = {}
    grouped_configs: Dict[Tuple[str, float, float, float, float], List[EquiConfig]] = defaultdict(list)
    for config in all_equi:
        grouped_configs[(config.scale_set, config.smooth, config.threshold, config.distance, config.association)].append(config)

    group_total = len(grouped_configs)
    for group_position, ((scale_name, smooth, threshold, distance, association), configs) in enumerate(grouped_configs.items(), start=1):
        by_scale = scale_candidates(curves, k_p, smooth, threshold, distance)
        by_video = [
            cluster_candidates(curve, k_p, SCALE_SETS[scale_name], by_scale, video_index, association)
            for video_index, curve in enumerate(curves)
        ]
        for config in configs:
            predictions = [equi_predictions(clusters, config, k_p) for clusters in by_video]
            equi_stats[config.identifier] = counts_matrix(predictions, records, subject_index)
        if group_position % 18 == 0 or group_position == group_total:
            print(f"[{package['dataset']}] EquiScale groups {group_position}/{group_total}; configs={len(equi_stats)}", flush=True)

    fixed_native = NativeConfig(2.0, 0.55, 1.0).identifier
    native_selected, native_rows, native_frequency = nested_select(list(native_stats), native_stats, subjects)
    support_selected, support_rows, support_frequency = nested_select(support_ids, equi_stats, subjects)
    fusion_selected, fusion_rows, fusion_frequency = nested_select(fusion_ids, equi_stats, subjects)
    equi_selected, equi_rows, equi_frequency = nested_select(list(equi_stats), equi_stats, subjects)
    combined_stats = {**native_stats, **equi_stats}
    combined_ids = [*native_stats, *equi_stats]
    combined_selected, combined_rows, combined_frequency = nested_select(combined_ids, combined_stats, subjects)

    target = RESULT_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    selection_sets = {
        "tuned_native": (native_rows, native_frequency),
        "equiscale_support": (support_rows, support_frequency),
        "equiscale_fusion": (fusion_rows, fusion_frequency),
        "equiscale_full": (equi_rows, equi_frequency),
        "combined": (combined_rows, combined_frequency),
    }
    for label, (rows, _frequency) in selection_sets.items():
        write_selection_csv(target / f"outer_loso_{label}.csv", rows)

    report = {
        "dataset": package["dataset"],
        "curve_cache": str(CURVE_CACHE / f"{cache_stem}_curves.pkl"),
        "videos": len(records),
        "subjects": len(subjects),
        "gt_events": sum(len(record["gt"]) for record in records),
        "k_p": k_p,
        "protocol": {
            "selection": "outer LOSO; each held subject is excluded from all configuration selection",
            "matching": "BoostingVRME source interval expansion/NMS and chronological IoU=0.5 greedy matching",
            "scope": "raw spotting only; no B2, experiments directory, recognition, or retraining",
        },
        "grid": {
            "tuned_native": {"count": len(all_native), "smooth": SMOOTH_MULTIPLIERS, "threshold": (0.35, 0.45, 0.55, 0.65), "distance": (0.50, 0.75, 1.0, 1.25)},
            "equiscale": {
                "count": len(all_equi),
                "scale_sets": SCALE_SETS,
                "smooth": SMOOTH_MULTIPLIERS,
                "threshold": THRESHOLD_FACTORS,
                "distance": DISTANCE_MULTIPLIERS,
                "association": ASSOCIATION_MULTIPLIERS,
                "support_minimum": SUPPORT_MINIMUMS,
                "aggregators": AGGREGATORS,
                "fusion_weight": FUSION_WEIGHTS,
                "fusion_threshold": FUSION_THRESHOLDS,
            },
        },
        "official_native_fixed": as_counts(native_stats[fixed_native].sum(axis=0)).as_dict(),
        "tuned_native_nested_loso": native_selected.as_dict(),
        "equiscale_support_nested_loso": support_selected.as_dict(),
        "equiscale_fusion_nested_loso": fusion_selected.as_dict(),
        "equiscale_full_nested_loso": equi_selected.as_dict(),
        "combined_native_equiscale_nested_loso": combined_selected.as_dict(),
        "selection_frequency": {label: dict(frequency.most_common()) for label, (_rows, frequency) in selection_sets.items()},
        "top_fixed_native_configs": top_fixed_configs(list(native_stats), native_stats),
        "top_fixed_equiscale_configs": top_fixed_configs(list(equi_stats), equi_stats),
    }
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_summary(reports: Sequence[dict]) -> None:
    lines = [
        "# BoostingVRME 方案 5 EquiScale 调参测试",
        "",
        "## 协议",
        "",
        "- 仅复用已验证的 BoostingVRME 曲线缓存、原始 Excel GT 和官方后处理的区间延展/NMS；未改动原项目，未使用 B2 或 `experiments/`。",
        "- 每个外层留出 subject 的配置只按其他 subjects 聚合 Raw Spotting F1 选择；留出 subject 的 GT 不参与该配置选择。",
        "- `tuned Native` 与 EquiScale 都可调平滑宽度、候选高度阈值和峰距。EquiScale 额外搜索尺度集合、跨尺度候选关联半径、最小支持尺度数，以及 Height-Support 融合。",
        "- 每个尺度先独立产生候选；时间相距不超过 `assoc*k_p` 的峰被聚成 cluster，每个尺度最多为 cluster 贡献一个峰。cluster 最终仍走 BoostingVRME 原始区间延展和 NMS。",
        "",
        "## 结果",
        "",
        "| 数据集 | 官方 Native | tuned Native | EquiScale support | EquiScale fusion | EquiScale 完整池 | Native+EquiScale 选择器 |",
        "| - | -: | -: | -: | -: | -: | -: |",
    ]
    keys = ("official_native_fixed", "tuned_native_nested_loso", "equiscale_support_nested_loso", "equiscale_fusion_nested_loso", "equiscale_full_nested_loso", "combined_native_equiscale_nested_loso")
    for report in reports:
        values = [report[key]["F1"] for key in keys]
        lines.append(f"| {report['dataset']} | " + " | ".join(f"{value:.4f}" for value in values) + " |")

    lines.extend(["", "## 解读", ""])
    for report in reports:
        fixed = report["official_native_fixed"]["F1"]
        tuned = report["tuned_native_nested_loso"]["F1"]
        fusion = report["equiscale_fusion_nested_loso"]["F1"]
        full = report["equiscale_full_nested_loso"]["F1"]
        frequency = report["selection_frequency"]["equiscale_fusion"]
        dominant, dominant_count = max(frequency.items(), key=lambda item: (item[1], item[0]))
        lines.append(
            f"- `{report['dataset']}`：受限 Height-Support 融合的 outer-LOSO F1 为 `{fusion:.4f}`，"
            f"相对官方 Native `{fusion - fixed:+.4f}`，相对 tuned Native `{fusion - tuned:+.4f}`。"
        )
        lines.append(
            f"  融合子池中最常被选择的配置覆盖 `{dominant_count}/{report['subjects']}` 个 outer folds：`{dominant}`。"
        )
        lines.append(
            f"  将硬支持度规则与融合规则混合为完整 EquiScale 池后为 `{full:.4f}`；"
            "因此完整池结果用于检验选择稳定性，不应替代受限融合子池作为方案 5 的主结论。"
        )

    lines.extend(["", "## 外层选择频次", ""])
    for report in reports:
        lines.append(f"### {report['dataset']}")
        for name, frequency in report["selection_frequency"].items():
            lines.append(f"- `{name}`: `{frequency}`")
        lines.append("")

    lines.extend(["## 固定配置全数据排名前十", ""])
    for report in reports:
        lines.append(f"### {report['dataset']}")
        lines.append("| 类型 | 配置 | TP | FP | FN | F1 |")
        lines.append("| - | - | -: | -: | -: | -: |")
        for kind, rows in (("Native", report["top_fixed_native_configs"]), ("EquiScale", report["top_fixed_equiscale_configs"])):
            for row in rows:
                lines.append(f"| {kind} | `{row['config']}` | {row['TP']} | {row['FP']} | {row['FN']} | {row['F1']:.4f} |")
    (RESULT_ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sammlv", "casme3", "both"), default="both")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = ("sammlv", "casme3") if args.dataset == "both" else (args.dataset,)
    reports = [run_dataset(dataset) for dataset in datasets]
    summary_reports = []
    for dataset in ("sammlv", "casme3"):
        report_path = RESULT_ROOT / dataset / "report.json"
        if report_path.exists():
            summary_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    write_summary(summary_reports)
    for report in reports:
        print(
            f"[{report['dataset']}] Native={report['tuned_native_nested_loso']['F1']:.4f}, "
            f"EquiScale={report['equiscale_full_nested_loso']['F1']:.4f}, "
            f"Combined={report['combined_native_equiscale_nested_loso']['F1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
