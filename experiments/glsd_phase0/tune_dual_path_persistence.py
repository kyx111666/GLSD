"""Nested-LOSO test for a strict-global plus low-gate recovery persistence cascade."""

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
import tune_local_multiscale_persistence as local_ms


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "dual_path_persistence_tuning"
STRICT_SMOOTH_MULTIPLIER = 2.0
STRICT_CANDIDATE_FACTOR = 0.65
STRICT_DISTANCE_MULTIPLIER = 1.5
STRICT_PERSISTENCE_THRESHOLD = 1.0
RECOVERY_SCALE_SET = "low"
RECOVERY_HIGH_FACTOR = 0.65


@dataclass(frozen=True)
class CascadeConfig:
    candidate_threshold_factor: float
    local_window_multiplier: float
    alignment_multiplier: float
    local_persistence_threshold: float
    global_persistence_threshold: float
    recovery_mode: str

    @property
    def identifier(self) -> str:
        return (
            f"dual_path|p_low={self.candidate_threshold_factor:g}"
            f"|r={self.local_window_multiplier:g}|a={self.alignment_multiplier:g}"
            f"|tau_local={self.local_persistence_threshold:g}"
            f"|tau_global={self.global_persistence_threshold:g}"
            f"|mode={self.recovery_mode}"
        )


@dataclass(frozen=True)
class LocalOnlyConfig:
    candidate_threshold_factor: float
    local_window_multiplier: float
    alignment_multiplier: float
    local_persistence_threshold: float

    @property
    def identifier(self) -> str:
        return (
            f"local_multiscale_only|p={self.candidate_threshold_factor:g}"
            f"|r={self.local_window_multiplier:g}|a={self.alignment_multiplier:g}"
            f"|tau={self.local_persistence_threshold:g}"
        )


@dataclass
class GlobalFeatures:
    curve: np.ndarray
    peaks: np.ndarray
    persistence_z: np.ndarray


@dataclass
class SharedFeatures:
    strict: GlobalFeatures
    recovery_global: GlobalFeatures


def all_configs() -> List[CascadeConfig]:
    configs = []
    for candidate_threshold_factor in (0.25, 0.35, 0.45):
        for local_window_multiplier in (0.5, 1.0, 1.5):
            for alignment_multiplier in (0.25, 0.5, 1.0):
                for local_persistence_threshold in (0.5, 1.0):
                    for global_persistence_threshold in (-0.5, 0.0, 0.5, 1.0):
                        for recovery_mode in ("low_band", "strict_missed"):
                            configs.append(CascadeConfig(
                                candidate_threshold_factor,
                                local_window_multiplier,
                                alignment_multiplier,
                                local_persistence_threshold,
                                global_persistence_threshold,
                                recovery_mode,
                            ))
    return configs


def all_local_only_configs() -> List[LocalOnlyConfig]:
    configs = []
    for candidate_threshold_factor in (0.25, 0.35, 0.45):
        for local_window_multiplier in (0.5, 1.0, 1.5):
            for alignment_multiplier in (0.25, 0.5, 1.0):
                for local_persistence_threshold in (0.5, 1.0):
                    configs.append(LocalOnlyConfig(
                        candidate_threshold_factor,
                        local_window_multiplier,
                        alignment_multiplier,
                        local_persistence_threshold,
                    ))
    return configs


def groups_for_payload(payload: dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(payload["records"]):
        groups[str(record["subject"])].append(index)
    return dict(groups)


def build_global_features(record: dict, payload: dict, distance_multiplier: float) -> GlobalFeatures:
    k_p = int(payload["k_p"])
    width = max(1, round(STRICT_SMOOTH_MULTIPLIER * k_p))
    curve = core.moving_average(np.asarray(record["score"], dtype=float), width)
    distance = max(1, round(distance_multiplier * k_p))
    peaks = core.local_maxima(curve, minimum_height=-np.inf, distance=distance)
    persistence = np.asarray(
        [core.peak_prominence(curve, int(peak)) for peak in peaks], dtype=float
    )
    return GlobalFeatures(curve, peaks, core.robust_z(persistence))


def build_shared_cache(payload: dict) -> Dict[int, SharedFeatures]:
    return {
        index: SharedFeatures(
            strict=build_global_features(record, payload, STRICT_DISTANCE_MULTIPLIER),
            recovery_global=build_global_features(record, payload, 1.0),
        )
        for index, record in enumerate(payload["records"])
    }


def local_cache_for_window(
    payload: dict, window_multiplier: float
) -> Dict[int, local_ms.BasePersistenceFeatures]:
    return {
        index: local_ms.build_base_features(
            record, payload, RECOVERY_SCALE_SET, window_multiplier
        )
        for index, record in enumerate(payload["records"])
    }


def make_detection(point: int, evidence: float, record: dict, payload: dict) -> dict:
    k_p = int(payload["k_p"])
    emotion = np.asarray(record["emotion"], dtype=int)
    onset = max(0, point - k_p)
    offset = min(len(emotion) - 1, point + k_p)
    return {
        "onset": onset,
        "peak": int(point),
        "offset": offset,
        "emotion_id": core.majority_emotion(emotion, onset, offset, int(point)),
        "evidence": float(evidence),
    }


def strict_detections(
    shared: SharedFeatures, record: dict, payload: dict
) -> List[dict]:
    features = shared.strict
    threshold = float(
        features.curve.mean()
        + STRICT_CANDIDATE_FACTOR * (features.curve.max() - features.curve.mean())
    )
    output = []
    for peak, persistence_z in zip(features.peaks, features.persistence_z):
        if (
            features.curve[peak] >= threshold
            and persistence_z >= STRICT_PERSISTENCE_THRESHOLD
        ):
            output.append(make_detection(int(peak), float(persistence_z), record, payload))
    return output


def closest_global_evidence(
    features: GlobalFeatures, point: int, tolerance: int
) -> tuple[int, float] | None:
    matches = np.flatnonzero(np.abs(features.peaks - point) <= tolerance)
    if matches.size == 0:
        return None
    best = min(
        matches.tolist(),
        key=lambda index: (
            abs(int(features.peaks[index]) - point),
            -float(features.persistence_z[index]),
        ),
    )
    return int(features.peaks[best]), float(features.persistence_z[best])


def recovery_detections(
    shared: SharedFeatures,
    local_features: local_ms.BasePersistenceFeatures,
    config: CascadeConfig,
    strict_points: Sequence[int],
    record: dict,
    payload: dict,
) -> List[dict]:
    local_key = local_ms.PersistenceFeatureKey(
        RECOVERY_SCALE_SET,
        config.local_window_multiplier,
        config.alignment_multiplier,
        "min",
        config.candidate_threshold_factor,
    )
    candidates = local_ms.candidate_features_for_payload(local_features, payload, local_key)
    reference_curve = local_features.scale_features[local_features.reference_scale].curve
    recovery_upper = float(
        reference_curve.mean()
        + RECOVERY_HIGH_FACTOR * (reference_curve.max() - reference_curve.mean())
    )
    tolerance = max(1, int(payload["k_p"]))
    output = []
    for point, local_score in zip(candidates.peaks, candidates.persistence_z):
        if local_score < config.local_persistence_threshold:
            continue
        if config.recovery_mode == "low_band" and reference_curve[point] >= recovery_upper:
            continue
        global_match = closest_global_evidence(
            shared.recovery_global, int(point), tolerance
        )
        if global_match is None:
            continue
        global_point, global_score = global_match
        if global_score < config.global_persistence_threshold:
            continue
        if any(abs(global_point - strict_point) <= int(payload["k_p"]) for strict_point in strict_points):
            continue
        output.append(make_detection(
            global_point,
            min(float(local_score), float(global_score)),
            record,
            payload,
        ))
    return output


def cascade_detections(
    shared: SharedFeatures,
    local_features: local_ms.BasePersistenceFeatures,
    config: CascadeConfig,
    record: dict,
    payload: dict,
) -> List[dict]:
    strict = strict_detections(shared, record, payload)
    recovery = recovery_detections(
        shared,
        local_features,
        config,
        [item["peak"] for item in strict],
        record,
        payload,
    )
    return sorted(strict + recovery, key=lambda item: item["peak"])


def local_only_detections(
    local_features: local_ms.BasePersistenceFeatures,
    config: LocalOnlyConfig,
    record: dict,
    payload: dict,
) -> List[dict]:
    key = local_ms.PersistenceFeatureKey(
        RECOVERY_SCALE_SET,
        config.local_window_multiplier,
        config.alignment_multiplier,
        "min",
        config.candidate_threshold_factor,
    )
    features = local_ms.candidate_features_for_payload(local_features, payload, key)
    fusion = core.FusionConfig(
        family="local_multiscale_only",
        feature_key=core.FeatureKey(1.5, config.candidate_threshold_factor, 1.0),
        height_weight=0.0,
        persistence_weight=1.0,
        morphology_weight=0.0,
        final_score_threshold=config.local_persistence_threshold,
    )
    return core.detections_from_features(features, record, payload, fusion)


def evaluate_detections_by_subject(
    payload: dict,
    groups: Dict[str, List[int]],
    detections_by_index: Dict[int, List[dict]],
) -> Dict[str, Dict[str, core.Counts]]:
    output = {subject: core.empty_pair() for subject in groups}
    for subject, record_indexes in groups.items():
        for index in record_indexes:
            core.add_pair(
                output[subject],
                core.counts_for_video(detections_by_index[index], payload["records"][index]),
            )
    return output


def strict_stats(
    payload: dict,
    groups: Dict[str, List[int]],
    shared_cache: Dict[int, SharedFeatures],
) -> tuple[Dict[str, Dict[str, core.Counts]], Dict[int, List[dict]]]:
    detections = {
        index: strict_detections(shared_cache[index], record, payload)
        for index, record in enumerate(payload["records"])
    }
    return evaluate_detections_by_subject(payload, groups, detections), detections


def evaluate_config(
    payload: dict,
    groups: Dict[str, List[int]],
    shared_cache: Dict[int, SharedFeatures],
    local_cache: Dict[int, local_ms.BasePersistenceFeatures],
    config: CascadeConfig,
) -> tuple[Dict[str, Dict[str, core.Counts]], Dict[int, List[dict]]]:
    detections = {
        index: cascade_detections(
            shared_cache[index], local_cache[index], config, record, payload
        )
        for index, record in enumerate(payload["records"])
    }
    return evaluate_detections_by_subject(payload, groups, detections), detections


def evaluate_local_only_config(
    payload: dict,
    groups: Dict[str, List[int]],
    local_cache: Dict[int, local_ms.BasePersistenceFeatures],
    config: LocalOnlyConfig,
) -> tuple[Dict[str, Dict[str, core.Counts]], Dict[int, List[dict]]]:
    detections = {
        index: local_only_detections(local_cache[index], config, record, payload)
        for index, record in enumerate(payload["records"])
    }
    return evaluate_detections_by_subject(payload, groups, detections), detections


def select_outer_loso(
    groups: Dict[str, List[int]],
    configs: Sequence[CascadeConfig],
    config_stats: Dict[str, Dict[str, Dict[str, core.Counts]]],
    baseline_by_subject: Dict[str, Dict[str, core.Counts]] | None = None,
    strict_by_subject: Dict[str, Dict[str, core.Counts]] | None = None,
    extra_option_stats: Dict[str, Dict[str, Dict[str, core.Counts]]] | None = None,
) -> tuple[Dict[str, core.Counts], List[dict], Dict[str, CascadeConfig | str]]:
    all_config_stats = {
        config.identifier: core.aggregate_pairs(config_stats[config.identifier].values())
        for config in configs
    }
    baseline_total = (
        core.aggregate_pairs(baseline_by_subject.values()) if baseline_by_subject else None
    )
    strict_total = (
        core.aggregate_pairs(strict_by_subject.values()) if strict_by_subject else None
    )
    extra_totals = {
        identifier: core.aggregate_pairs(stats.values())
        for identifier, stats in (extra_option_stats or {}).items()
    }
    final_stats = core.empty_pair()
    rows = []
    selected_by_subject: Dict[str, CascadeConfig | str] = {}

    for subject in groups:
        best = None
        if baseline_total is not None:
            counts = {
                name: baseline_total[name].subtract(baseline_by_subject[subject][name])
                for name in ("raw", "result_synergy")
            }
            best = (core.score_key(counts["raw"], "zz_native_fixed"), "native_fixed", counts)
        if strict_total is not None:
            counts = {
                name: strict_total[name].subtract(strict_by_subject[subject][name])
                for name in ("raw", "result_synergy")
            }
            candidate = (core.score_key(counts["raw"], "strict_global"), "strict_global", counts)
            if best is None or candidate[0] > best[0]:
                best = candidate
        for identifier, total in extra_totals.items():
            counts = {
                name: total[name].subtract(extra_option_stats[identifier][subject][name])
                for name in ("raw", "result_synergy")
            }
            candidate = (core.score_key(counts["raw"], identifier), identifier, counts)
            if best is None or candidate[0] > best[0]:
                best = candidate
        for config in configs:
            counts = {
                name: all_config_stats[config.identifier][name].subtract(
                    config_stats[config.identifier][subject][name]
                )
                for name in ("raw", "result_synergy")
            }
            candidate = (core.score_key(counts["raw"], config.identifier), config, counts)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _score, selected, inner_counts = best
        if selected == "native_fixed":
            assert baseline_by_subject is not None
            test_counts = baseline_by_subject[subject]
        elif selected == "strict_global":
            assert strict_by_subject is not None
            test_counts = strict_by_subject[subject]
        elif isinstance(selected, str):
            assert extra_option_stats is not None
            test_counts = extra_option_stats[selected][subject]
        else:
            test_counts = config_stats[selected.identifier][subject]
        core.add_pair(final_stats, test_counts)
        rows.append({
            "subject": subject,
            "selected_config": selected if isinstance(selected, str) else selected.identifier,
            "inner_raw_f1": round(inner_counts["raw"].f1(), 6),
            "test_raw_f1": round(test_counts["raw"].f1(), 6),
            "test_synergy_f1": round(test_counts["result_synergy"].f1(), 6),
        })
        selected_by_subject[subject] = selected
    return final_stats, rows, selected_by_subject


def final_detections(
    groups: Dict[str, List[int]],
    selected_by_subject: Dict[str, CascadeConfig | str],
    baseline_details: Dict[int, tuple[List[dict], np.ndarray, float]],
    strict_detections_by_index: Dict[int, List[dict]],
    config_detections: Dict[str, Dict[int, List[dict]]],
    extra_option_detections: Dict[str, Dict[int, List[dict]]] | None = None,
) -> Dict[int, List[dict]]:
    output = {}
    for subject, record_indexes in groups.items():
        selected = selected_by_subject[subject]
        for index in record_indexes:
            if selected == "native_fixed":
                output[index] = baseline_details[index][0]
            elif selected == "strict_global":
                output[index] = strict_detections_by_index[index]
            elif isinstance(selected, str):
                assert extra_option_detections is not None
                output[index] = extra_option_detections[selected][index]
            else:
                output[index] = config_detections[selected.identifier][index]
    return output


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(payload: dict, output_dir: Path) -> dict:
    groups = groups_for_payload(payload)
    configs = all_configs()
    local_only_configs = all_local_only_configs()
    print(
        f"{payload['dataset']}: {len(configs)} cascade configurations, "
        f"{len(local_only_configs)} local-only configurations, {len(groups)} subjects"
    )
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

    shared_cache = build_shared_cache(payload)
    strict_by_subject, strict_by_index = strict_stats(payload, groups, shared_cache)
    config_stats = {}
    config_detections = {}
    local_only_stats = {}
    local_only_detections_by_config = {}
    configs_by_window: Dict[float, List[CascadeConfig]] = defaultdict(list)
    for config in configs:
        configs_by_window[config.local_window_multiplier].append(config)
    for position, (window, window_configs) in enumerate(configs_by_window.items(), start=1):
        print(
            f"recovery feature group {position}/{len(configs_by_window)}: "
            f"r={window:g}, {len(window_configs)} configurations"
        )
        local_cache = local_cache_for_window(payload, window)
        for config in window_configs:
            stats, detections = evaluate_config(
                payload, groups, shared_cache, local_cache, config
            )
            config_stats[config.identifier] = stats
            config_detections[config.identifier] = detections
        for config in local_only_configs:
            if config.local_window_multiplier != window:
                continue
            stats, detections = evaluate_local_only_config(
                payload, groups, local_cache, config
            )
            local_only_stats[config.identifier] = stats
            local_only_detections_by_config[config.identifier] = detections

    cascade_only, cascade_rows, cascade_selected = select_outer_loso(
        groups, configs, config_stats
    )
    policy_final, policy_rows, policy_selected = select_outer_loso(
        groups,
        configs,
        config_stats,
        baseline_by_subject=baseline_by_subject,
        strict_by_subject=strict_by_subject,
        extra_option_stats=local_only_stats,
    )
    strict_total = core.aggregate_pairs(strict_by_subject.values())
    cascade_final_detections = final_detections(
        groups,
        cascade_selected,
        baseline_details,
        strict_by_index,
        config_detections,
        extra_option_detections=None,
    )
    policy_final_detections = final_detections(
        groups,
        policy_selected,
        baseline_details,
        strict_by_index,
        config_detections,
        extra_option_detections=local_only_detections_by_config,
    )
    report = {
        "dataset": payload["dataset"],
        "baseline_reproduced": verified,
        "protocol": {
            "selection": "outer LOSO with deterministic configuration selected from aggregate inner held-out-subject raw Spotting F1",
            "policy_selection": "native fixed, strict global path, every dual-path configuration, and every local-only configuration compete in the inner selection pool",
            "matching": "paper-style chronological greedy IoU=0.5",
            "training_free": True,
            "strict_path": {
                "smooth_multiplier": STRICT_SMOOTH_MULTIPLIER,
                "candidate_threshold_factor": STRICT_CANDIDATE_FACTOR,
                "peak_distance_multiplier": STRICT_DISTANCE_MULTIPLIER,
                "global_persistence_threshold": STRICT_PERSISTENCE_THRESHOLD,
            },
            "recovery_path": {
                "scale_set": list(local_ms.SCALE_SETS[RECOVERY_SCALE_SET]),
                "requires": "low-height candidate, three-scale alignment, local-persistence minimum, and nearest global-peak persistence evidence",
                "deduplication": "recovery detections within k_p frames of a strict detection are removed",
                "modes": {
                    "low_band": "only candidates below the strict p=0.65 height gate can enter recovery",
                    "strict_missed": "any low-gate candidate not covered by a strict detection can enter recovery",
                },
            },
            "grid": {
                "candidate_threshold_factor": [0.25, 0.35, 0.45],
                "local_window_multiplier": [0.5, 1.0, 1.5],
                "alignment_multiplier": [0.25, 0.5, 1.0],
                "local_persistence_threshold": [0.5, 1.0],
                "global_persistence_threshold": [-0.5, 0.0, 0.5, 1.0],
                "recovery_mode": ["low_band", "strict_missed"],
                "local_only": {
                    "candidate_threshold_factor": [0.25, 0.35, 0.45],
                    "local_window_multiplier": [0.5, 1.0, 1.5],
                    "alignment_multiplier": [0.25, 0.5, 1.0],
                    "local_persistence_threshold": [0.5, 1.0],
                },
            },
        },
        "native_fixed": core.pair_to_dict(baseline),
        "strict_global_path": core.pair_to_dict(strict_total),
        "dual_path_only": {
            "nested_outer_loso": core.pair_to_dict(cascade_only),
            "subthreshold_proxy": core.subthreshold_proxy(
                payload, baseline_details, cascade_final_detections
            ),
            "selected_config_frequency": dict(
                Counter(row["selected_config"] for row in cascade_rows).most_common()
            ),
        },
        "policy_selection": {
            "nested_outer_loso": core.pair_to_dict(policy_final),
            "subthreshold_proxy": core.subthreshold_proxy(
                payload, baseline_details, policy_final_detections
            ),
            "selected_config_frequency": dict(
                Counter(row["selected_config"] for row in policy_rows).most_common()
            ),
        },
    }
    target = output_dir / payload["dataset"].lower().replace("_", "")
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(target / "outer_loso_dual_path_only.csv", cascade_rows)
    write_csv(target / "outer_loso_policy_selection.csv", policy_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-path persistence cascade test")
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
        reports = (
            existing.get("datasets", {})
            if existing.get("experiment") == "scheme11_dual_path_persistence"
            else {}
        )
    else:
        reports = {}
    for name in requested:
        cache_path = args.cache_dir / cache_names[name]
        with cache_path.open("rb") as handle:
            reports[name] = run_one(pickle.load(handle), args.out_dir)
    combined_path.write_text(
        json.dumps(
            {"experiment": "scheme11_dual_path_persistence", "datasets": reports},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
