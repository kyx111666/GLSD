"""Nested-LOSO test of local, multi-scale persistence for ME-TST scheme 1.

The experiment is intentionally independent from the original project.  It
uses the strict score/logit caches and writes only beside this script.
"""

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


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "persistence_multiscale_tuning"
DEFAULT_CANDIDATE_THRESHOLD = 0.55
SCALE_SETS = {
    "low": (1.0, 1.5, 2.0),
    "native_centered": (1.5, 2.0, 2.5),
    "smooth": (2.0, 2.5, 3.0),
}


@dataclass(frozen=True)
class PersistenceFeatureKey:
    scale_set: str
    window_multiplier: float
    alignment_multiplier: float
    aggregation: str
    candidate_threshold_factor: float


@dataclass(frozen=True)
class PersistenceConfig:
    key: PersistenceFeatureKey
    final_score_threshold: float

    @property
    def identifier(self) -> str:
        return (
            f"local_multiscale_persistence|scales={self.key.scale_set}"
            f"|r={self.key.window_multiplier:g}|a={self.key.alignment_multiplier:g}"
            f"|agg={self.key.aggregation}|p={self.key.candidate_threshold_factor:g}"
            f"|tau={self.final_score_threshold:g}"
        )

    def as_fusion_config(self) -> core.FusionConfig:
        return core.FusionConfig(
            family="local_multiscale_persistence",
            feature_key=core.FeatureKey(2.0, self.key.candidate_threshold_factor, 1.0),
            height_weight=0.0,
            persistence_weight=1.0,
            morphology_weight=0.0,
            final_score_threshold=self.final_score_threshold,
        )


@dataclass
class ScaleFeatures:
    curve: np.ndarray
    peaks: np.ndarray
    persistence_z: np.ndarray


@dataclass
class BasePersistenceFeatures:
    scales: tuple[float, ...]
    scale_features: Dict[float, ScaleFeatures]

    @property
    def reference_scale(self) -> float:
        return self.scales[len(self.scales) // 2]


def local_prominence(values: np.ndarray, peak: int, radius: int) -> float:
    """Peak contrast relative to minima in a bounded temporal neighborhood."""
    peak = int(peak)
    radius = max(1, int(radius))
    left = values[max(0, peak - radius):peak + 1]
    right = values[peak:min(len(values), peak + radius + 1)]
    local_floor = max(float(np.min(left)), float(np.min(right)))
    return max(0.0, float(values[peak]) - local_floor)


def build_base_features(
    record: dict,
    payload: dict,
    scale_set: str,
    window_multiplier: float,
) -> BasePersistenceFeatures:
    k_p = int(payload["k_p"])
    radius = max(1, round(window_multiplier * k_p))
    output = {}
    for scale in SCALE_SETS[scale_set]:
        width = max(1, round(scale * k_p))
        curve = core.moving_average(np.asarray(record["score"], dtype=float), width)
        peaks = core.local_maxima(curve, minimum_height=-np.inf, distance=k_p)
        local_values = np.asarray(
            [local_prominence(curve, int(peak), radius) for peak in peaks], dtype=float
        )
        output[scale] = ScaleFeatures(
            curve=curve,
            peaks=peaks,
            persistence_z=core.robust_z(local_values),
        )
    return BasePersistenceFeatures(SCALE_SETS[scale_set], output)


def closest_peak_z(features: ScaleFeatures, point: int, tolerance: int) -> float | None:
    matches = np.flatnonzero(np.abs(features.peaks - point) <= tolerance)
    if matches.size == 0:
        return None
    # Prefer temporal alignment; use prominence to resolve an exact-distance tie.
    best = min(
        matches.tolist(),
        key=lambda index: (abs(int(features.peaks[index]) - point), -features.persistence_z[index]),
    )
    return float(features.persistence_z[best])


def candidate_features_for_payload(
    base: BasePersistenceFeatures,
    payload: dict,
    key: PersistenceFeatureKey,
) -> core.CandidateFeatures:
    reference = base.scale_features[base.reference_scale]
    curve = reference.curve
    threshold = float(
        curve.mean()
        + key.candidate_threshold_factor * (curve.max() - curve.mean())
    )
    tolerance = max(1, round(key.alignment_multiplier * int(payload["k_p"])))
    peaks = []
    scores = []
    for peak in reference.peaks:
        if curve[peak] < threshold:
            continue
        scale_scores = []
        for scale in base.scales:
            score = closest_peak_z(base.scale_features[scale], int(peak), tolerance)
            if score is None:
                break
            scale_scores.append(score)
        if len(scale_scores) != len(base.scales):
            continue
        peaks.append(int(peak))
        scores.append(float(np.mean(scale_scores)) if key.aggregation == "mean" else float(np.min(scale_scores)))
    return core.CandidateFeatures(
        peaks=np.asarray(peaks, dtype=int),
        height_z=np.zeros(len(peaks), dtype=float),
        persistence_z=np.asarray(scores, dtype=float),
        morphology_z=np.zeros(len(peaks), dtype=float),
        curve=curve,
        candidate_threshold=threshold,
    )


def all_configs(candidate_thresholds: Sequence[float]) -> List[PersistenceConfig]:
    configs = []
    for scale_set in SCALE_SETS:
        for window_multiplier in (0.5, 1.0, 1.5, 2.0, 3.0):
            for alignment_multiplier in (0.25, 0.5, 1.0):
                for aggregation in ("mean", "min"):
                    for candidate_threshold_factor in candidate_thresholds:
                        key = PersistenceFeatureKey(
                            scale_set,
                            window_multiplier,
                            alignment_multiplier,
                            aggregation,
                            candidate_threshold_factor,
                        )
                        for final_score_threshold in (-1.0, -0.5, 0.0, 0.5, 1.0):
                            configs.append(PersistenceConfig(key, final_score_threshold))
    return configs


def groups_for_payload(payload: dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(payload["records"]):
        groups[str(record["subject"])].append(index)
    return dict(groups)


def configs_by_base_key(
    configs: Sequence[PersistenceConfig],
) -> Dict[tuple[str, float], List[PersistenceConfig]]:
    grouped: Dict[tuple[str, float], List[PersistenceConfig]] = defaultdict(list)
    for config in configs:
        grouped[(config.key.scale_set, config.key.window_multiplier)].append(config)
    return dict(grouped)


def build_base_cache(
    payload: dict, scale_set: str, window_multiplier: float
) -> Dict[int, BasePersistenceFeatures]:
    return {
        index: build_base_features(record, payload, scale_set, window_multiplier)
        for index, record in enumerate(payload["records"])
    }


def evaluate_config(
    payload: dict,
    groups: Dict[str, List[int]],
    base_cache: Dict[int, BasePersistenceFeatures],
    config: PersistenceConfig,
) -> Dict[str, Dict[str, core.Counts]]:
    fusion = config.as_fusion_config()
    output = {subject: core.empty_pair() for subject in groups}
    for subject, record_indexes in groups.items():
        for index in record_indexes:
            record = payload["records"][index]
            detections = core.detections_from_features(
                candidate_features_for_payload(base_cache[index], payload, config.key),
                record,
                payload,
                fusion,
            )
            core.add_pair(output[subject], core.counts_for_video(detections, record))
    return output


def select_outer_loso(
    groups: Dict[str, List[int]],
    configs: Sequence[PersistenceConfig],
    config_stats: Dict[str, Dict[str, Dict[str, core.Counts]]],
    baseline_by_subject: Dict[str, Dict[str, core.Counts]] | None = None,
) -> tuple[Dict[str, core.Counts], List[dict], Dict[str, PersistenceConfig | None]]:
    all_stats = {
        config.identifier: core.aggregate_pairs(config_stats[config.identifier].values())
        for config in configs
    }
    baseline_total = (
        core.aggregate_pairs(baseline_by_subject.values()) if baseline_by_subject is not None else None
    )
    final_stats = core.empty_pair()
    selections = []
    selected_by_subject: Dict[str, PersistenceConfig | None] = {}

    for subject in groups:
        best = None
        if baseline_total is not None:
            native_train = {
                name: baseline_total[name].subtract(baseline_by_subject[subject][name])
                for name in ("raw", "result_synergy")
            }
            best = (
                core.score_key(native_train["raw"], "zz_native_fixed"),
                None,
                native_train,
            )
        for config in configs:
            train_counts = {
                name: all_stats[config.identifier][name].subtract(
                    config_stats[config.identifier][subject][name]
                )
                for name in ("raw", "result_synergy")
            }
            candidate = (
                core.score_key(train_counts["raw"], config.identifier),
                config,
                train_counts,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _score, selected, inner_counts = best
        test_counts = (
            baseline_by_subject[subject] if selected is None else config_stats[selected.identifier][subject]
        )
        core.add_pair(final_stats, test_counts)
        selections.append({
            "subject": subject,
            "selected_config": "native_fixed" if selected is None else selected.identifier,
            "inner_raw_f1": round(inner_counts["raw"].f1(), 6),
            "test_raw_f1": round(test_counts["raw"].f1(), 6),
            "test_synergy_f1": round(test_counts["result_synergy"].f1(), 6),
        })
        selected_by_subject[subject] = selected
    return final_stats, selections, selected_by_subject


def selected_detections(
    payload: dict,
    groups: Dict[str, List[int]],
    selected_by_subject: Dict[str, PersistenceConfig | None],
    baseline_details: Dict[int, tuple[List[dict], np.ndarray, float]],
) -> Dict[int, List[dict]]:
    output = {}
    by_base: Dict[tuple[str, float], List[tuple[str, int]]] = defaultdict(list)
    for subject, record_indexes in groups.items():
        config = selected_by_subject[subject]
        if config is None:
            for index in record_indexes:
                output[index] = baseline_details[index][0]
            continue
        base_key = (config.key.scale_set, config.key.window_multiplier)
        by_base[base_key].extend((subject, index) for index in record_indexes)

    for (scale_set, window_multiplier), subject_records in by_base.items():
        indexes = {index for _subject, index in subject_records}
        base_cache = {
            index: build_base_features(payload["records"][index], payload, scale_set, window_multiplier)
            for index in indexes
        }
        for subject, index in subject_records:
            config = selected_by_subject[subject]
            assert config is not None
            output[index] = core.detections_from_features(
                candidate_features_for_payload(base_cache[index], payload, config.key),
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


def run_one(
    payload: dict, output_dir: Path, candidate_thresholds: Sequence[float]
) -> dict:
    groups = groups_for_payload(payload)
    configs = all_configs(candidate_thresholds)
    print(f"{payload['dataset']}: {len(configs)} configurations, {len(groups)} subjects")
    baseline_by_subject, baseline_details = core.baseline_by_subject(payload, groups)
    baseline = core.aggregate_pairs(baseline_by_subject.values())
    expected = core.EXPECTED_NATIVE[payload["dataset"]]
    verified = all(
        tuple(baseline[name].__dict__[field] for field in ("tp", "fp", "fn"))
        == expected[name]
        for name in expected
    )
    if not verified:
        raise RuntimeError("Native baseline did not reproduce the strict-cache reference counts")

    config_stats = {}
    grouped_configs = configs_by_base_key(configs)
    for position, ((scale_set, window_multiplier), group) in enumerate(
        grouped_configs.items(), start=1
    ):
        print(
            f"base feature group {position}/{len(grouped_configs)}: "
            f"scales={scale_set}, r={window_multiplier:g}, {len(group)} configurations"
        )
        base_cache = build_base_cache(payload, scale_set, window_multiplier)
        for config in group:
            config_stats[config.identifier] = evaluate_config(
                payload, groups, base_cache, config
            )

    method_final, method_rows, method_selected = select_outer_loso(
        groups, configs, config_stats
    )
    guarded_final, guarded_rows, guarded_selected = select_outer_loso(
        groups, configs, config_stats, baseline_by_subject
    )
    method_detections = selected_detections(
        payload, groups, method_selected, baseline_details
    )
    guarded_detections = selected_detections(
        payload, groups, guarded_selected, baseline_details
    )
    report = {
        "dataset": payload["dataset"],
        "baseline_reproduced": verified,
        "protocol": {
            "selection": "outer LOSO with deterministic configuration selected from aggregate inner held-out-subject raw Spotting F1",
            "native_guard": "the guarded result also admits the fixed native baseline to the inner selection pool",
            "matching": "paper-style chronological greedy IoU=0.5",
            "training_free": True,
            "local_persistence_definition": "peak height minus the higher of the left and right local minima within r*k_p frames",
            "multiscale_definition": "a candidate-gated reference peak must align with a local maximum at each of three smoothing scales; per-scale robust-z local persistence is aggregated",
            "candidate_gate": {
                "smooth_scale": "center scale of each set",
                "p_values": list(candidate_thresholds),
                "peak_distance": "k_p",
            },
            "grid": {
                "scale_sets": SCALE_SETS,
                "local_window_multiplier": [0.5, 1.0, 1.5, 2.0, 3.0],
                "alignment_multiplier": [0.25, 0.5, 1.0],
                "aggregation": ["mean", "min"],
                "final_persistence_score_threshold": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
        },
        "native_fixed": core.pair_to_dict(baseline),
        "local_multiscale_persistence_only": {
            "nested_outer_loso": core.pair_to_dict(method_final),
            "subthreshold_proxy": core.subthreshold_proxy(
                payload, baseline_details, method_detections
            ),
            "selected_config_frequency": dict(
                Counter(row["selected_config"] for row in method_rows).most_common()
            ),
        },
        "native_guarded_selection": {
            "nested_outer_loso": core.pair_to_dict(guarded_final),
            "subthreshold_proxy": core.subthreshold_proxy(
                payload, baseline_details, guarded_detections
            ),
            "selected_config_frequency": dict(
                Counter(row["selected_config"] for row in guarded_rows).most_common()
            ),
        },
    }
    target = output_dir / payload["dataset"].lower().replace("_", "")
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(target / "outer_loso_method_only.csv", method_rows)
    write_csv(target / "outer_loso_native_guarded.csv", guarded_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local multi-scale persistence parameter test for scheme 1"
    )
    parser.add_argument("--datasets", default="SAMMLV,CASME_3")
    parser.add_argument("--cache-dir", type=Path, default=core.DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-threshold-factors",
        default=str(DEFAULT_CANDIDATE_THRESHOLD),
        help="Comma-separated candidate-height factors p in mean + p*(max-mean)",
    )
    args = parser.parse_args()
    cache_names = {
        "SAMMLV": "sammlv_strategy1_outputs.pkl",
        "CASME_3": "casme3_strategy1_outputs.pkl",
    }
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    candidate_thresholds = [
        float(item.strip())
        for item in args.candidate_threshold_factors.split(",")
        if item.strip()
    ]
    unknown = [item for item in requested if item not in cache_names]
    if unknown:
        raise SystemExit(f"Unsupported dataset(s): {', '.join(unknown)}")
    if not candidate_thresholds:
        raise SystemExit("At least one candidate threshold factor is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / "combined_report.json"
    if combined_path.exists():
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
        reports = (
            existing.get("datasets", {})
            if existing.get("experiment") == "scheme1_local_multiscale_persistence"
            else {}
        )
    else:
        reports = {}
    for name in requested:
        cache_path = args.cache_dir / cache_names[name]
        with cache_path.open("rb") as handle:
            reports[name] = run_one(
                pickle.load(handle), args.out_dir, candidate_thresholds
            )
    combined_path.write_text(
        json.dumps(
            {
                "experiment": "scheme1_local_multiscale_persistence",
                "candidate_threshold_factors": candidate_thresholds,
                "datasets": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
