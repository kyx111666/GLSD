"""Nested-LOSO test for persistence + local-morphology candidate fusion.

This is an independent post-processing experiment.  It reads the strict
ME-TST score/logit caches but never writes to the ME-TST project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(r"D:\workspace\ME-TST-main\ME-TST-main")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "paper_aligned_strategy_cache"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# These are the strict-cache baseline counts documented by the reference skill.
EXPECTED_NATIVE = {
    "SAMMLV": {
        "raw": (53, 184, 106),
        "result_synergy": (52, 171, 107),
    },
    "CASME_3": {
        "raw": (81, 912, 777),
        "result_synergy": (76, 727, 782),
    },
}


@dataclass(frozen=True)
class FeatureKey:
    smooth_multiplier: float
    candidate_threshold_factor: float
    morphology_radius_multiplier: float


@dataclass(frozen=True)
class FusionConfig:
    family: str
    feature_key: FeatureKey
    height_weight: float
    persistence_weight: float
    morphology_weight: float
    final_score_threshold: float

    @property
    def identifier(self) -> str:
        key = self.feature_key
        return (
            f"{self.family}|cs={key.smooth_multiplier:g}|p={key.candidate_threshold_factor:g}"
            f"|r={key.morphology_radius_multiplier:g}|w={self.height_weight:g},"
            f"{self.persistence_weight:g},{self.morphology_weight:g}"
            f"|tau={self.final_score_threshold:g}"
        )


@dataclass
class CandidateFeatures:
    peaks: np.ndarray
    height_z: np.ndarray
    persistence_z: np.ndarray
    morphology_z: np.ndarray
    curve: np.ndarray
    candidate_threshold: float


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    num_gt: int = 0
    num_pred: int = 0
    neutral_pred: int = 0
    neutral_tp_removed: int = 0
    neutral_fp_removed: int = 0

    def copy(self) -> "Counts":
        return Counts(**self.__dict__)

    def add(self, other: "Counts") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def subtract(self, other: "Counts") -> "Counts":
        return Counts(**{
            field: getattr(self, field) - getattr(other, field)
            for field in self.__dataclass_fields__
        })

    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / denominator) if denominator else 0.0

    def precision(self) -> float:
        denominator = self.tp + self.fp
        return (self.tp / denominator) if denominator else 0.0

    def recall(self) -> float:
        denominator = self.tp + self.fn
        return (self.tp / denominator) if denominator else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "num_gt": self.num_gt,
            "num_pred": self.num_pred,
            "neutral_pred": self.neutral_pred,
            "neutral_tp_removed": self.neutral_tp_removed,
            "neutral_fp_removed": self.neutral_fp_removed,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
        }


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def local_maxima(values: np.ndarray, minimum_height: float, distance: int) -> np.ndarray:
    """A compact SciPy-free equivalent for the non-plateau score curves in cache."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.empty(0, dtype=int)

    peaks: List[int] = []
    index = 1
    while index < values.size - 1:
        if values[index] > values[index - 1]:
            left = index
            while index + 1 < values.size and values[index] == values[index + 1]:
                index += 1
            if index < values.size - 1 and values[index] > values[index + 1]:
                peak = (left + index) // 2
                if values[peak] >= minimum_height:
                    peaks.append(peak)
        index += 1

    if not peaks:
        return np.empty(0, dtype=int)

    # SciPy keeps the highest member of a distance-conflicted group first.
    distance = max(1, int(distance))
    priority = sorted(peaks, key=lambda point: (-values[point], point))
    kept: List[int] = []
    for point in priority:
        if all(abs(point - chosen) >= distance for chosen in kept):
            kept.append(point)
    return np.asarray(sorted(kept), dtype=int)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - median) / scale


def peak_prominence(values: np.ndarray, peak: int) -> float:
    """One-dimensional prominence using the first higher peak on either side."""
    height = float(values[peak])
    left_min = height
    right_min = height

    index = peak
    while index > 0:
        index -= 1
        current = float(values[index])
        if current > height:
            break
        left_min = min(left_min, current)

    index = peak
    while index < len(values) - 1:
        index += 1
        current = float(values[index])
        if current > height:
            break
        right_min = min(right_min, current)

    return max(0.0, height - max(left_min, right_min))


def morphology_values(values: np.ndarray, peaks: np.ndarray, k_p: int, radius: int) -> np.ndarray:
    """Return width-prior, symmetry, local area, and local stability per peak."""
    if peaks.size == 0:
        return np.empty((0, 4), dtype=float)

    rows: List[Tuple[float, float, float, float]] = []
    epsilon = 1e-8
    for peak in peaks:
        prominence = peak_prominence(values, int(peak))
        half_level = float(values[peak]) - 0.5 * prominence

        left = int(peak)
        while left > 0 and values[left] > half_level:
            left -= 1
        right = int(peak)
        while right < len(values) - 1 and values[right] > half_level:
            right += 1
        half_width = max(1, right - left)
        expected_width = max(1, 2 * int(k_p))
        width_prior = -abs(math.log((half_width + 1.0) / (expected_width + 1.0)))

        start = max(0, int(peak) - radius)
        stop = min(len(values), int(peak) + radius + 1)
        patch = values[start:stop]
        left_drop = float(values[peak] - values[start])
        right_drop = float(values[peak] - values[stop - 1])
        symmetry = -abs(left_drop - right_drop) / (abs(left_drop) + abs(right_drop) + epsilon)
        local_baseline = min(float(values[start]), float(values[stop - 1]))
        area = float(np.mean(np.maximum(patch - local_baseline, 0.0)))
        stability = -float(np.std(patch)) / (prominence + epsilon)
        rows.append((width_prior, symmetry, area, stability))
    return np.asarray(rows, dtype=float)


def build_candidate_features(record: dict, payload: dict, key: FeatureKey) -> CandidateFeatures:
    k_p = int(payload["k_p"])
    width = max(1, round(float(key.smooth_multiplier) * k_p))
    curve = moving_average(np.asarray(record["score"], dtype=float), width)
    all_peaks = local_maxima(curve, minimum_height=-np.inf, distance=k_p)
    threshold = float(curve.mean() + key.candidate_threshold_factor * (curve.max() - curve.mean()))
    if all_peaks.size == 0:
        return CandidateFeatures(
            peaks=np.empty(0, dtype=int),
            height_z=np.empty(0),
            persistence_z=np.empty(0),
            morphology_z=np.empty(0),
            curve=curve,
            candidate_threshold=threshold,
        )

    height = curve[all_peaks]
    persistence = np.asarray([peak_prominence(curve, int(peak)) for peak in all_peaks])
    radius = max(1, round(float(key.morphology_radius_multiplier) * k_p))
    morph_raw = morphology_values(curve, all_peaks, k_p=k_p, radius=radius)
    morphology = np.mean(np.column_stack([robust_z(morph_raw[:, column]) for column in range(4)]), axis=1)

    candidate_mask = height >= threshold
    return CandidateFeatures(
        peaks=all_peaks[candidate_mask],
        height_z=robust_z(height)[candidate_mask],
        persistence_z=robust_z(persistence)[candidate_mask],
        morphology_z=robust_z(morphology)[candidate_mask],
        curve=curve,
        candidate_threshold=threshold,
    )


def majority_emotion(emotion: np.ndarray, onset: int, offset: int, peak: int) -> int:
    start = max(0, onset + 1)
    stop = max(1, offset - 1)
    window = list(np.asarray(emotion)[start:stop])
    if not window:
        bounded_peak = min(max(0, peak), len(emotion) - 1)
        window = [int(np.asarray(emotion)[bounded_peak])] if len(emotion) else [3]
    return int(Counter(int(value) for value in window).most_common(1)[0][0])


def detections_from_features(
    features: CandidateFeatures,
    record: dict,
    payload: dict,
    config: FusionConfig,
) -> List[dict]:
    if features.peaks.size == 0:
        return []
    evidence = (
        config.height_weight * features.height_z
        + config.persistence_weight * features.persistence_z
        + config.morphology_weight * features.morphology_z
    )
    selected = np.flatnonzero(evidence >= config.final_score_threshold)
    k_p = int(payload["k_p"])
    emotion = np.asarray(record["emotion"], dtype=int)
    detections = []
    for index in selected:
        peak = int(features.peaks[index])
        onset = max(0, peak - k_p)
        offset = min(len(emotion) - 1, peak + k_p)
        detections.append({
            "onset": onset,
            "peak": peak,
            "offset": offset,
            "emotion_id": majority_emotion(emotion, onset, offset, peak),
            "evidence": float(evidence[index]),
        })
    return detections


def native_detections(record: dict, payload: dict) -> Tuple[List[dict], np.ndarray, float]:
    k_p = int(payload["k_p"])
    curve = moving_average(np.asarray(record["score"], dtype=float), 2 * k_p)
    threshold = float(curve.mean() + 0.55 * (curve.max() - curve.mean()))
    peaks = local_maxima(curve, threshold, k_p)
    emotion = np.asarray(record["emotion"], dtype=int)
    detections = []
    for peak in peaks:
        point = int(peak)
        onset = max(0, point - k_p)
        offset = min(len(emotion) - 1, point + k_p)
        detections.append({
            "onset": onset,
            "peak": point,
            "offset": offset,
            "emotion_id": majority_emotion(emotion, onset, offset, point),
        })
    return detections, curve, threshold


def interval_iou(first: Tuple[int, int], second: Tuple[int, int]) -> float:
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0, right - left + 1)
    if intersection == 0:
        return 0.0
    union = (first[1] - first[0] + 1) + (second[1] - second[0] + 1) - intersection
    return intersection / union


def match_detections(detections: Sequence[dict], samples: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
    """Mirror the repository's chronological, top-IoU greedy paper matching."""
    matched_gt: List[int] = []
    pred_to_gt: List[int] = []
    for detection in detections:
        if not samples:
            pred_to_gt.append(-1)
            continue
        ious = [interval_iou((detection["onset"], detection["offset"]), (int(sample[0]), int(sample[2]))) for sample in samples]
        best = int(np.argmax(ious))
        if ious[best] >= 0.5 and best not in matched_gt:
            matched_gt.append(best)
            pred_to_gt.append(best)
        else:
            pred_to_gt.append(-1)
    return pred_to_gt, matched_gt


def counts_for_video(detections: Sequence[dict], record: dict) -> Dict[str, Counts]:
    samples = record["samples"]
    pred_to_gt, _matched_gt = match_detections(detections, samples)
    raw = Counts(num_gt=len(samples), num_pred=len(detections))
    synergy = Counts(num_gt=len(samples), num_pred=len(detections))
    for detection, matched_index in zip(detections, pred_to_gt):
        if matched_index == -1:
            raw.fp += 1
        else:
            raw.tp += 1
        is_neutral = int(detection["emotion_id"]) == 4
        if is_neutral:
            synergy.neutral_pred += 1
            synergy.num_pred -= 1
            if matched_index == -1:
                synergy.neutral_fp_removed += 1
            else:
                synergy.neutral_tp_removed += 1
                synergy.fn += 1
            continue
        if matched_index == -1:
            synergy.fp += 1
        else:
            synergy.tp += 1
    raw.fn = raw.num_gt - raw.tp
    synergy.fn += synergy.num_gt - raw.tp
    return {"raw": raw, "result_synergy": synergy}


def add_pair(total: Dict[str, Counts], increment: Dict[str, Counts]) -> None:
    for name in total:
        total[name].add(increment[name])


def empty_pair() -> Dict[str, Counts]:
    return {"raw": Counts(), "result_synergy": Counts()}


def score_key(counts: Counts, identifier: str) -> Tuple[float, float, int, str]:
    return (counts.f1(), counts.precision(), -counts.fp, identifier)


def feature_keys_for_family(family: str) -> Iterable[FeatureKey]:
    smooth_values = (1.5, 2.0, 2.5)
    threshold_values = (0.35, 0.45, 0.55)
    radius_values = (1.0,) if family in {"height_only", "height_persistence"} else (1.0, 2.0)
    for smooth_multiplier in smooth_values:
        for threshold_factor in threshold_values:
            for radius_multiplier in radius_values:
                yield FeatureKey(smooth_multiplier, threshold_factor, radius_multiplier)


def fusion_configs() -> Dict[str, List[FusionConfig]]:
    thresholds = (-0.5, 0.0, 0.5, 1.0)
    weight_sets = {
        "height_only": ((1.0, 0.0, 0.0),),
        "morphology_only": ((0.0, 0.0, 1.0),),
        "height_persistence": ((0.75, 0.25, 0.0), (0.50, 0.50, 0.0)),
        "height_morphology": ((0.75, 0.0, 0.25), (0.50, 0.0, 0.50)),
        "height_persistence_morphology": (
            (0.60, 0.20, 0.20),
            (0.50, 0.25, 0.25),
            (0.40, 0.30, 0.30),
        ),
    }
    output: Dict[str, List[FusionConfig]] = {}
    for family, weights in weight_sets.items():
        configs = []
        for key in feature_keys_for_family(family):
            for height_weight, persistence_weight, morphology_weight in weights:
                for final_threshold in thresholds:
                    configs.append(FusionConfig(
                        family=family,
                        feature_key=key,
                        height_weight=height_weight,
                        persistence_weight=persistence_weight,
                        morphology_weight=morphology_weight,
                        final_score_threshold=final_threshold,
                    ))
        output[family] = configs
    return output


def record_groups(payload: dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(payload["records"]):
        groups[str(record["subject"])].append(index)
    return dict(groups)


def baseline_by_subject(payload: dict, groups: Dict[str, List[int]]) -> Tuple[Dict[str, Dict[str, Counts]], Dict[int, Tuple[List[dict], np.ndarray, float]]]:
    subject_stats = {subject: empty_pair() for subject in groups}
    details = {}
    for subject, indexes in groups.items():
        for index in indexes:
            detections, curve, threshold = native_detections(payload["records"][index], payload)
            details[index] = (detections, curve, threshold)
            add_pair(subject_stats[subject], counts_for_video(detections, payload["records"][index]))
    return subject_stats, details


def aggregate_pairs(pairs: Iterable[Dict[str, Counts]]) -> Dict[str, Counts]:
    total = empty_pair()
    for pair in pairs:
        add_pair(total, pair)
    return total


def precompute_features(payload: dict, configs: Dict[str, List[FusionConfig]]) -> Dict[FeatureKey, Dict[int, CandidateFeatures]]:
    keys = sorted({config.feature_key for family in configs.values() for config in family}, key=lambda item: (item.smooth_multiplier, item.candidate_threshold_factor, item.morphology_radius_multiplier))
    cache: Dict[FeatureKey, Dict[int, CandidateFeatures]] = {}
    for position, key in enumerate(keys, start=1):
        print(f"feature cache {position}/{len(keys)}: cs={key.smooth_multiplier:g}, p={key.candidate_threshold_factor:g}, r={key.morphology_radius_multiplier:g}")
        cache[key] = {
            index: build_candidate_features(record, payload, key)
            for index, record in enumerate(payload["records"])
        }
    return cache


def evaluate_config_by_subject(
    payload: dict,
    groups: Dict[str, List[int]],
    features: Dict[FeatureKey, Dict[int, CandidateFeatures]],
    config: FusionConfig,
) -> Dict[str, Dict[str, Counts]]:
    by_subject = {subject: empty_pair() for subject in groups}
    for subject, indexes in groups.items():
        for index in indexes:
            record = payload["records"][index]
            detections = detections_from_features(features[config.feature_key][index], record, payload, config)
            add_pair(by_subject[subject], counts_for_video(detections, record))
    return by_subject


def nested_select(
    family: str,
    configs: Sequence[FusionConfig],
    config_stats: Dict[str, Dict[str, Dict[str, Counts]]],
    features: Dict[FeatureKey, Dict[int, CandidateFeatures]],
    payload: dict,
    groups: Dict[str, List[int]],
) -> Tuple[Dict[str, Dict[str, Counts]], List[dict], Dict[int, List[dict]]]:
    all_stats = {
        config.identifier: aggregate_pairs(config_stats[config.identifier].values())
        for config in configs
    }
    final_stats = empty_pair()
    rows = []
    final_detections: Dict[int, List[dict]] = {}

    for subject, indexes in groups.items():
        best = None
        for config in configs:
            train_counts = {
                name: all_stats[config.identifier][name].subtract(config_stats[config.identifier][subject][name])
                for name in ("raw", "result_synergy")
            }
            candidate = (score_key(train_counts["raw"], config.identifier), config, train_counts)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _key, selected, inner_counts = best
        test_counts = config_stats[selected.identifier][subject]
        add_pair(final_stats, test_counts)
        rows.append({
            "subject": subject,
            "family": family,
            "selected_config": selected.identifier,
            "inner_raw_f1": round(inner_counts["raw"].f1(), 6),
            "test_raw_f1": round(test_counts["raw"].f1(), 6),
            "test_synergy_f1": round(test_counts["result_synergy"].f1(), 6),
        })
        for index in indexes:
            final_detections[index] = detections_from_features(
                features[selected.feature_key][index],
                payload["records"][index],
                payload,
                selected,
            )
    return final_stats, rows, final_detections


def subthreshold_proxy(
    payload: dict,
    baseline_details: Dict[int, Tuple[List[dict], np.ndarray, float]],
    final_detections: Dict[int, List[dict]],
) -> dict:
    total_proxy = 0
    rescued = 0
    total_native_missed = 0
    any_native_missed_rescued = 0
    k_p = int(payload["k_p"])
    for index, record in enumerate(payload["records"]):
        baseline, curve, threshold = baseline_details[index]
        _baseline_pred_to_gt, baseline_matched = match_detections(baseline, record["samples"])
        _final_pred_to_gt, final_matched = match_detections(final_detections[index], record["samples"])
        baseline_set = set(baseline_matched)
        final_set = set(final_matched)
        all_peaks = local_maxima(curve, -np.inf, k_p)
        for gt_index, sample in enumerate(record["samples"]):
            if gt_index in baseline_set:
                continue
            total_native_missed += 1
            if gt_index in final_set:
                any_native_missed_rescued += 1
            onset, offset = int(sample[0]), int(sample[2])
            local = all_peaks[(all_peaks >= onset) & (all_peaks <= offset)]
            no_native_height_peak = local.size > 0 and bool(np.all(curve[local] < threshold))
            if no_native_height_peak:
                total_proxy += 1
                if gt_index in final_set:
                    rescued += 1
    return {
        "definition": "Baseline-unmatched GT containing one or more local maxima inside the GT interval, all below the native p=0.55 height threshold.",
        "native_missed_gt": total_native_missed,
        "native_missed_gt_rescued": any_native_missed_rescued,
        "subthreshold_proxy_fn": total_proxy,
        "subthreshold_proxy_fn_rescued": rescued,
        "subthreshold_proxy_recall": round(rescued / total_proxy, 4) if total_proxy else 0.0,
    }


def pair_to_dict(pair: Dict[str, Counts]) -> dict:
    return {name: counts.as_dict() for name, counts in pair.items()}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(payload: dict, output_dir: Path) -> dict:
    dataset = str(payload["dataset"])
    groups = record_groups(payload)
    print(f"{dataset}: {len(groups)} subjects, {len(payload['records'])} videos, {payload['num_gt']} GT events")

    baseline_stats, baseline_details = baseline_by_subject(payload, groups)
    baseline_total = aggregate_pairs(baseline_stats.values())
    expected = EXPECTED_NATIVE.get(dataset)
    expected_match = None
    if expected:
        expected_match = all(
            tuple(baseline_total[name].__dict__[field] for field in ("tp", "fp", "fn")) == expected[name]
            for name in expected
        )
    print(f"{dataset}: native strict-cache reproduction={expected_match}, raw F1={baseline_total['raw'].f1():.4f}")

    configs_by_family = fusion_configs()
    feature_cache = precompute_features(payload, configs_by_family)
    report = {
        "dataset": dataset,
        "cache_structure": {
            "num_subjects": len(groups),
            "num_videos": len(payload["records"]),
            "num_gt": int(payload["num_gt"]),
            "k_p": int(payload["k_p"]),
            "frame_skip": int(payload["frame_skip"]),
        },
        "native_fixed": pair_to_dict(baseline_total),
        "native_reproduction_matches_reference": expected_match,
        "methods": {},
    }

    all_rows: List[dict] = []
    for family, configs in configs_by_family.items():
        print(f"{dataset}: evaluating {family} ({len(configs)} nested-selection candidates)")
        config_stats = {}
        for count, config in enumerate(configs, start=1):
            if count % 50 == 0 or count == len(configs):
                print(f"  {family}: {count}/{len(configs)}")
            config_stats[config.identifier] = evaluate_config_by_subject(
                payload, groups, feature_cache, config
            )

        final_stats, selections, selected_detections = nested_select(
            family, configs, config_stats, feature_cache, payload, groups
        )
        frequencies = Counter(row["selected_config"] for row in selections)
        method_report = {
            "nested_outer_loso": pair_to_dict(final_stats),
            "selected_config_frequency": dict(frequencies.most_common()),
            "subthreshold_proxy": subthreshold_proxy(payload, baseline_details, selected_detections),
        }
        report["methods"][family] = method_report
        all_rows.extend(selections)
        print(
            f"  {family}: raw F1={final_stats['raw'].f1():.4f}, "
            f"synergy F1={final_stats['result_synergy'].f1():.4f}"
        )

    dataset_dir = output_dir / dataset.lower().replace("_", "")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(dataset_dir / "outer_loso_selections.csv", all_rows)
    return report


def load_payload(cache_path: Path) -> dict:
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-cache persistence + morphology fusion experiment")
    parser.add_argument("--datasets", default="SAMMLV,CASME_3", help="Comma-separated: SAMMLV,CASME_3")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    name_to_cache = {
        "SAMMLV": "sammlv_strategy1_outputs.pkl",
        "CASME_3": "casme3_strategy1_outputs.pkl",
    }
    unknown = [name for name in requested if name not in name_to_cache]
    if unknown:
        raise SystemExit(f"Unsupported dataset(s): {', '.join(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / "combined_report.json"
    if combined_path.exists():
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
        if existing.get("experiment") == "persistence_morphology_fusion":
            combined = existing
        else:
            combined = {"experiment": "persistence_morphology_fusion", "datasets": {}}
    else:
        combined = {"experiment": "persistence_morphology_fusion", "datasets": {}}
    for name in requested:
        cache_path = args.cache_dir / name_to_cache[name]
        if not cache_path.exists():
            raise FileNotFoundError(f"Strict cache not found: {cache_path}")
        combined["datasets"][name] = run_dataset(load_payload(cache_path), args.out_dir)

    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
