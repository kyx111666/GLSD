"""Expanded, training-free nested-LOSO parameter sweep for ME-TST scheme 3."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import persistence_morphology_fusion as core


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "morphology_tuning"

PROFILE_WEIGHTS = {
    "balanced": (0.25, 0.25, 0.25, 0.25),
    "edge_shape": (0.50, 0.50, 0.00, 0.00),
    "area_stability": (0.00, 0.00, 0.50, 0.50),
    "width_area": (0.50, 0.00, 0.50, 0.00),
    "area_only": (0.00, 0.00, 1.00, 0.00),
}


@dataclass(frozen=True)
class MorphFeatureKey:
    smooth_multiplier: float
    candidate_threshold_factor: float
    radius_multiplier: float
    profile: str


@dataclass(frozen=True)
class MorphConfig:
    key: MorphFeatureKey
    final_score_threshold: float

    @property
    def identifier(self) -> str:
        return (
            f"morphology_only|cs={self.key.smooth_multiplier:g}|p={self.key.candidate_threshold_factor:g}"
            f"|r={self.key.radius_multiplier:g}|profile={self.key.profile}"
            f"|tau={self.final_score_threshold:g}"
        )

    def as_fusion_config(self) -> core.FusionConfig:
        return core.FusionConfig(
            family="morphology_only_tuned",
            feature_key=core.FeatureKey(
                self.key.smooth_multiplier,
                self.key.candidate_threshold_factor,
                self.key.radius_multiplier,
            ),
            height_weight=0.0,
            persistence_weight=0.0,
            morphology_weight=1.0,
            final_score_threshold=self.final_score_threshold,
        )


@dataclass
class BaseMorphFeatures:
    peaks: np.ndarray
    standardized_components: np.ndarray
    curve: np.ndarray


def build_base_features(
    record: dict,
    payload: dict,
    smooth_multiplier: float,
    radius_multiplier: float,
) -> BaseMorphFeatures:
    k_p = int(payload["k_p"])
    width = max(1, round(smooth_multiplier * k_p))
    curve = core.moving_average(np.asarray(record["score"], dtype=float), width)
    all_peaks = core.local_maxima(curve, minimum_height=-np.inf, distance=k_p)
    if all_peaks.size == 0:
        return BaseMorphFeatures(
            peaks=np.empty(0, dtype=int),
            standardized_components=np.empty((0, 4), dtype=float),
            curve=curve,
        )

    radius = max(1, round(radius_multiplier * k_p))
    components = core.morphology_values(curve, all_peaks, k_p=k_p, radius=radius)
    standardized = np.column_stack([
        core.robust_z(components[:, index]) for index in range(components.shape[1])
    ])
    return BaseMorphFeatures(all_peaks, standardized, curve)


def profile_features(base: BaseMorphFeatures, key: MorphFeatureKey) -> core.CandidateFeatures:
    threshold = float(
        base.curve.mean()
        + key.candidate_threshold_factor * (base.curve.max() - base.curve.mean())
    )
    if base.peaks.size == 0:
        return core.CandidateFeatures(
            peaks=np.empty(0, dtype=int),
            height_z=np.empty(0),
            persistence_z=np.empty(0),
            morphology_z=np.empty(0),
            curve=base.curve,
            candidate_threshold=threshold,
        )

    weights = np.asarray(PROFILE_WEIGHTS[key.profile], dtype=float)
    morphology = base.standardized_components @ weights
    candidate_mask = base.curve[base.peaks] >= threshold
    return core.CandidateFeatures(
        peaks=base.peaks[candidate_mask],
        height_z=np.zeros(int(np.sum(candidate_mask)), dtype=float),
        persistence_z=np.zeros(int(np.sum(candidate_mask)), dtype=float),
        morphology_z=core.robust_z(morphology)[candidate_mask],
        curve=base.curve,
        candidate_threshold=threshold,
    )


def all_configs() -> List[MorphConfig]:
    configs = []
    for smooth_multiplier in (1.0, 1.5, 2.0, 2.5):
        for threshold_factor in (0.25, 0.35, 0.45, 0.55, 0.65):
            for radius_multiplier in (0.5, 1.0, 1.5, 2.0, 3.0):
                for profile in PROFILE_WEIGHTS:
                    key = MorphFeatureKey(smooth_multiplier, threshold_factor, radius_multiplier, profile)
                    for final_score_threshold in (-1.0, -0.5, 0.0, 0.5, 1.0):
                        configs.append(MorphConfig(key, final_score_threshold))
    return configs


def groups_for_payload(payload: dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(payload["records"]):
        groups[str(record["subject"])].append(index)
    return dict(groups)


def configs_by_base_key(configs: Sequence[MorphConfig]) -> Dict[tuple[float, float], List[MorphConfig]]:
    grouped: Dict[tuple[float, float], List[MorphConfig]] = defaultdict(list)
    for config in configs:
        grouped[(config.key.smooth_multiplier, config.key.radius_multiplier)].append(config)
    return dict(grouped)


def build_base_cache(payload: dict, smooth_multiplier: float, radius_multiplier: float) -> Dict[int, BaseMorphFeatures]:
    return {
        index: build_base_features(record, payload, smooth_multiplier, radius_multiplier)
        for index, record in enumerate(payload["records"])
    }


def evaluate_config(
    payload: dict,
    groups: Dict[str, List[int]],
    base_cache: Dict[int, BaseMorphFeatures],
    config: MorphConfig,
) -> Dict[str, Dict[str, core.Counts]]:
    fusion = config.as_fusion_config()
    output = {subject: core.empty_pair() for subject in groups}
    for subject, record_indexes in groups.items():
        for index in record_indexes:
            record = payload["records"][index]
            detections = core.detections_from_features(profile_features(base_cache[index], config.key), record, payload, fusion)
            core.add_pair(output[subject], core.counts_for_video(detections, record))
    return output


def select_outer_loso(
    payload: dict,
    groups: Dict[str, List[int]],
    configs: Sequence[MorphConfig],
    config_stats: Dict[str, Dict[str, Dict[str, core.Counts]]],
) -> tuple[Dict[str, core.Counts], List[dict], Dict[str, MorphConfig]]:
    all_stats = {
        config.identifier: core.aggregate_pairs(config_stats[config.identifier].values())
        for config in configs
    }
    final_stats = core.empty_pair()
    selections = []
    selected_by_subject = {}

    for subject, record_indexes in groups.items():
        best = None
        for config in configs:
            train_counts = {
                name: all_stats[config.identifier][name].subtract(config_stats[config.identifier][subject][name])
                for name in ("raw", "result_synergy")
            }
            candidate = (core.score_key(train_counts["raw"], config.identifier), config, train_counts)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _score, selected, inner_counts = best
        test_counts = config_stats[selected.identifier][subject]
        core.add_pair(final_stats, test_counts)
        selections.append({
            "subject": subject,
            "selected_config": selected.identifier,
            "inner_raw_f1": round(inner_counts["raw"].f1(), 6),
            "test_raw_f1": round(test_counts["raw"].f1(), 6),
            "test_synergy_f1": round(test_counts["result_synergy"].f1(), 6),
        })
        selected_by_subject[subject] = selected
    return final_stats, selections, selected_by_subject


def selected_detections(
    payload: dict,
    groups: Dict[str, List[int]],
    selected_by_subject: Dict[str, MorphConfig],
) -> Dict[int, List[dict]]:
    by_base: Dict[tuple[float, float], List[tuple[str, int]]] = defaultdict(list)
    for subject, record_indexes in groups.items():
        config = selected_by_subject[subject]
        base_key = (config.key.smooth_multiplier, config.key.radius_multiplier)
        by_base[base_key].extend((subject, index) for index in record_indexes)

    output = {}
    for (smooth_multiplier, radius_multiplier), subject_records in by_base.items():
        indexes = {index for _subject, index in subject_records}
        base_cache = {
            index: build_base_features(payload["records"][index], payload, smooth_multiplier, radius_multiplier)
            for index in indexes
        }
        for subject, index in subject_records:
            config = selected_by_subject[subject]
            output[index] = core.detections_from_features(
                profile_features(base_cache[index], config.key),
                payload["records"][index],
                payload,
                config.as_fusion_config(),
            )
    return output


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(payload: dict, output_dir: Path) -> dict:
    groups = groups_for_payload(payload)
    configs = all_configs()
    print(f"{payload['dataset']}: {len(configs)} configurations, {len(groups)} subjects")
    baseline_by_subject, baseline_details = core.baseline_by_subject(payload, groups)
    baseline = core.aggregate_pairs(baseline_by_subject.values())
    expected = core.EXPECTED_NATIVE[payload["dataset"]]
    verified = all(
        tuple(baseline[name].__dict__[field] for field in ("tp", "fp", "fn")) == expected[name]
        for name in expected
    )
    if not verified:
        raise RuntimeError("Native baseline did not reproduce the strict-cache reference counts")

    config_stats = {}
    grouped_configs = configs_by_base_key(configs)
    for position, ((smooth_multiplier, radius_multiplier), group) in enumerate(grouped_configs.items(), start=1):
        print(
            f"base feature group {position}/{len(grouped_configs)}: "
            f"cs={smooth_multiplier:g}, r={radius_multiplier:g}, {len(group)} configurations"
        )
        base_cache = build_base_cache(payload, smooth_multiplier, radius_multiplier)
        for config in group:
            config_stats[config.identifier] = evaluate_config(payload, groups, base_cache, config)

    final, selections, selected_by_subject = select_outer_loso(
        payload, groups, configs, config_stats
    )
    final_detections = selected_detections(payload, groups, selected_by_subject)
    frequency = Counter(row["selected_config"] for row in selections)
    report = {
        "dataset": payload["dataset"],
        "baseline_reproduced": verified,
        "protocol": {
            "selection": "outer LOSO with deterministic configuration selected from aggregate inner held-out-subject raw Spotting F1",
            "matching": "paper-style chronological greedy IoU=0.5",
            "training_free": True,
            "grid": {
                "smooth_multiplier": [1.0, 1.5, 2.0, 2.5],
                "candidate_threshold_factor": [0.25, 0.35, 0.45, 0.55, 0.65],
                "morphology_radius_multiplier": [0.5, 1.0, 1.5, 2.0, 3.0],
                "profiles": PROFILE_WEIGHTS,
                "final_score_threshold": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
        },
        "native_fixed": core.pair_to_dict(baseline),
        "morphology_only_tuned": core.pair_to_dict(final),
        "subthreshold_proxy": core.subthreshold_proxy(payload, baseline_details, final_detections),
        "selected_config_frequency": dict(frequency.most_common()),
    }
    target = output_dir / payload["dataset"].lower().replace("_", "")
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(target / "outer_loso_selections.csv", selections)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded scheme-3 parameter test")
    parser.add_argument("--datasets", default="SAMMLV,CASME_3")
    parser.add_argument("--cache-dir", type=Path, default=core.DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    cache_names = {
        "SAMMLV": "sammlv_strategy1_outputs.pkl",
        "CASME_3": "casme3_strategy1_outputs.pkl",
    }
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = [item for item in requested if item not in cache_names]
    if unknown:
        raise SystemExit(f"Unsupported dataset(s): {', '.join(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / "combined_report.json"
    if combined_path.exists():
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
        reports = existing.get("datasets", {}) if existing.get("experiment") == "scheme3_expanded_morphology_sweep" else {}
    else:
        reports = {}
    for name in requested:
        cache_path = args.cache_dir / cache_names[name]
        with cache_path.open("rb") as handle:
            reports[name] = run_one(pickle.load(handle), args.out_dir)
    combined_path.write_text(
        json.dumps({"experiment": "scheme3_expanded_morphology_sweep", "datasets": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
