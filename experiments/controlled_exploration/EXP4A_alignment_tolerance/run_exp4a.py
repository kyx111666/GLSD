"""EXP-4A: nested search over the canonical GLSD alignment tolerance."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
RETHINK = ROOT / "RethinkFuse_reproduction"
BASELINE = ROOT / "controlled_exploration/baseline_snapshot"

sys.path.insert(0, str(SIGNED))
import equiscale_fair_validation as fair  # noqa: E402
import unified_persistence as engine  # noqa: E402


GROUPS = (
    ("metst", "sammlv", "ME-TST/SAMMLV"),
    ("metst", "casme3", "ME-TST/CAS(ME)3"),
    ("boostingvrme", "sammlv", "BoostingVRME/SAMMLV"),
    ("boostingvrme", "casme3", "BoostingVRME/CAS(ME)3"),
)
GAMMAS = (0.25, 0.50, 0.75)
BASELINE_GAMMA = 0.50
BOOTSTRAP_RESAMPLES = 10_000
SEED = 100
FP_EXPLOSION_RELATIVE = 0.10
FP_EXPLOSION_ABSOLUTE = 10


@dataclass(frozen=True)
class GammaConfig:
    gamma: float
    reference: float
    radius: float
    threshold: float
    family: str = "unified"
    height_weight: float = 0.0

    @property
    def identifier(self) -> str:
        return (
            f"gamma={self.gamma:g}|unified|scale={self.reference:g}"
            f"|height=0|tau={self.threshold:g}|radius={self.radius:g}"
        )

    @property
    def canonical_identifier(self) -> str:
        return (
            f"unified|scale={self.reference:g}|height=0"
            f"|tau={self.threshold:g}|radius={self.radius:g}"
        )


class GammaCurveFeatures(engine.CurveFeatures):
    """Canonical features with only the alignment coefficient exposed."""

    def evidence(self, reference, radius=1.0, gamma=BASELINE_GAMMA):
        key = reference, radius, gamma
        if key in self.evidence_cache:
            return self.evidence_cache[key]
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(gamma * self.k))
        window = max(1, round(radius * self.k))
        for width, (other_peaks, _, _, smooth, spread) in self.effective_scales.items():
            if (width, window) not in self.local_cache:
                self.local_cache[width, window] = np.array([
                    max(
                        0.0,
                        float(smooth[peak])
                        - max(
                            float(np.min(smooth[max(0, peak - window):peak + 1])),
                            float(np.min(smooth[peak:min(len(smooth), peak + window + 1)])),
                        ),
                    ) / spread
                    for peak in other_peaks
                ])
        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (
                        abs(int(other_peaks[index]) - point), -values[index]
                    ),
                )
                aligned.append(float(values[chosen]))
            local.append(float(np.median(aligned)))
        output = peaks, np.column_stack((height, global_values, local))
        self.evidence_cache[key] = output
        return output


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def configs() -> list[GammaConfig]:
    values = [
        GammaConfig(gamma, reference, radius, threshold)
        for gamma in GAMMAS
        for reference in engine.REFERENCE_SCALES
        for radius in engine.LOCAL_RADII
        for threshold in engine.THRESHOLDS
    ]
    if len(values) != 270:
        raise AssertionError("EXP-4A grid must contain exactly 270 configurations")
    return values


def archived_paths(backbone: str, dataset: str) -> tuple[Path, Path]:
    mode = "native" if backbone == "boostingvrme" else "fixed"
    root = (
        ROOT / "historical_gl_exact_fresh_reproduction/fresh_run" / backbone
        / "results/pure_persistence_matched_v1" / dataset / mode
    )
    return root / "outer_loso_selections.csv", root / "subject_counts.csv"


def archived_baseline(backbone: str, dataset: str) -> tuple[list[dict], np.ndarray]:
    selections_path, counts_path = archived_paths(backbone, dataset)
    selections = [row for row in load_csv(selections_path) if row["family"] == "pure"]
    count_rows = [row for row in load_csv(counts_path) if row["family"] == "pure"]
    counts = np.array(
        [[int(row[key]) for key in ("TP", "FP", "FN")] for row in count_rows],
        dtype=np.int64,
    )
    return selections, counts


def prepared(record, features, config: GammaConfig, backbone: str, mode: str):
    peaks, evidence = features.evidence(config.reference, config.radius, config.gamma)
    decoder = fair.VideoFeatures(
        record, features.k, backbone if mode == "native" else "metst"
    )
    return engine.Prepared(decoder, peaks, evidence)


def evaluate_k(records, subjects, backbone, k, needed, all_configs, mode):
    stats = np.zeros((len(all_configs), len(subjects), 3), dtype=np.int32)
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    batches = {}
    for index, config in enumerate(all_configs):
        batches.setdefault((config.gamma, config.reference, config.radius), []).append(index)
    for video_index, record in enumerate(records):
        sid = subject_ids[record["subject"]]
        if not needed[sid]:
            continue
        features = GammaCurveFeatures(record["curve"], k)
        for indexes in batches.values():
            selected = [all_configs[index] for index in indexes]
            item = prepared(record, features, selected[0], backbone, mode)
            stats[indexes, sid] += item.evaluate(selected)
        if (video_index + 1) % 100 == 0:
            fair.log(
                f"[{backbone}] EXP-4A k={k}: {video_index + 1}/{len(records)} videos"
            )
    return stats


def select(stats, outer, inner, subjects, all_configs, indexes):
    choices, rows = [], []
    counts = np.zeros((len(subjects), 3), dtype=np.int64)
    for held, subject in enumerate(subjects):
        training = engine.inner_counts(stats, held, inner)
        choice = engine.choose(training, indexes)
        config = all_configs[choice]
        choices.append(choice)
        counts[held] = stats[int(outer[held])][choice, held]
        metrics = fair.metrics(training[choice])
        rows.append({
            "subject": subject,
            "selected_gamma": config.gamma,
            "selected_a0": config.reference,
            "selected_rho": config.radius,
            "selected_tau": config.threshold,
            "inner_f1": metrics["F1"],
            "inner_precision": metrics["precision"],
            "inner_fp": metrics["FP"],
            "outer_tp": int(counts[held, 0]),
            "outer_fp": int(counts[held, 1]),
            "outer_fn": int(counts[held, 2]),
        })
    return choices, counts, rows


def alignment_audit(features: GammaCurveFeatures, config: GammaConfig) -> dict:
    peaks = features.scales[config.reference][0]
    reference_width = min(
        len(features.curve), max(1, round(config.reference * features.k))
    )
    tolerance = max(1, round(config.gamma * features.k))
    nonreference = {
        width: values[0]
        for width, values in features.effective_scales.items()
        if width != reference_width
    }
    matched = missing = reuse = collisions = 0
    for other_peaks in nonreference.values():
        assignments = []
        for point in peaks:
            indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
            if not len(indexes):
                missing += 1
                continue
            chosen = min(indexes, key=lambda i: (abs(int(other_peaks[i]) - point), int(i)))
            assignments.append(int(chosen))
            matched += 1
        used = Counter(assignments)
        collisions += sum(count > 1 for count in used.values())
        reuse += sum(max(0, count - 1) for count in used.values())
    return {
        "total_reference_candidates": len(peaks),
        "total_nonreference_peaks": sum(len(value) for value in nonreference.values()),
        "matched_alignment_count": matched,
        "missing_alignment_count": missing,
        "reuse_count": reuse,
        "collision_count": collisions,
    }


def add_counts(target: dict, values: dict) -> None:
    for key, value in values.items():
        target[key] += value


def replay_and_audit(records, subjects, outer, backbone, mode, all_configs,
                     canonical_choices, selected_choices):
    replay = {
        "canonical": np.zeros((len(subjects), 3), dtype=np.int64),
        "selected": np.zeros((len(subjects), 3), dtype=np.int64),
    }
    alignment = {
        name: {key: 0 for key in (
            "total_reference_candidates", "total_nonreference_peaks",
            "matched_alignment_count", "missing_alignment_count",
            "reuse_count", "collision_count",
        )}
        for name in replay
    }
    alignment_strata = {
        gamma: {
            name: {key: 0 for key in alignment[name]}
            for name in replay
        }
        for gamma in (0.25, 0.75)
    }
    predictions = {name: [] for name in replay}
    l_scores = {name: {} for name in replay}
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for video_index, record in enumerate(records):
        held = subject_index[record["subject"]]
        features = GammaCurveFeatures(record["curve"], int(outer[held]))
        selected_gamma = all_configs[selected_choices[held]].gamma
        for name, choices in (
            ("canonical", canonical_choices), ("selected", selected_choices)
        ):
            config = all_configs[choices[held]]
            item = prepared(record, features, config, backbone, mode)
            values, details = item.evaluate([config], details=True)
            replay[name][held] += values[0]
            predictions[name].append(details[0])
            current_alignment = alignment_audit(features, config)
            add_counts(alignment[name], current_alignment)
            if selected_gamma in alignment_strata:
                add_counts(alignment_strata[selected_gamma][name], current_alignment)
            peaks, evidence = features.evidence(
                config.reference, config.radius, config.gamma
            )
            for peak, local in zip(peaks, evidence[:, 2]):
                l_scores[name][video_index, int(peak)] = float(local)
    changes = fair.event_changes(predictions["selected"], predictions["canonical"])
    common = sorted(set(l_scores["canonical"]) & set(l_scores["selected"]))
    deltas = np.array([
        l_scores["selected"][key] - l_scores["canonical"][key] for key in common
    ])
    changes.update({
        "shared_candidate_count_for_L": len(common),
        "mean_delta_L": float(np.mean(deltas)) if len(deltas) else "unavailable",
        "median_delta_L": float(np.median(deltas)) if len(deltas) else "unavailable",
    })
    return replay, alignment, alignment_strata, changes


def baseline_recovery_group(backbone: str, dataset: str, setting: str,
                            all_configs: list[GammaConfig]) -> dict:
    """Run the locked gamma=.50 grid before any EXP-4A search."""
    records, subjects, _, input_path = fair.load_data(backbone, dataset)
    outer, inner = fair.fold_priors(records, subjects, backbone)
    mode = "native" if backbone == "boostingvrme" else "fixed"
    canonical_configs = [
        config for config in all_configs if config.gamma == BASELINE_GAMMA
    ]
    by_k = {}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist())):
        needed = fair.required_subjects(k, outer, inner)
        by_k[k] = evaluate_k(
            records, subjects, backbone, k, needed, canonical_configs, mode
        )
    choices, counts, rows = select(
        by_k, outer, inner, subjects, canonical_configs,
        list(range(len(canonical_configs))),
    )
    archived_rows, archived_counts = archived_baseline(backbone, dataset)
    if [row["subject"] for row in archived_rows] != subjects:
        raise AssertionError(f"Baseline subject order mismatch for {setting}")
    for current, archived, choice in zip(rows, archived_rows, choices):
        config = canonical_configs[choice]
        if config.canonical_identifier != archived["config"]:
            raise AssertionError(
                f"Baseline config replay mismatch for {setting}/{current['subject']}"
            )
        if float(current["inner_f1"]) != float(archived["inner_F1"]):
            raise AssertionError(
                f"Baseline inner F1 replay mismatch for {setting}/{current['subject']}"
            )
    np.testing.assert_array_equal(counts, archived_counts)

    replay = np.zeros_like(counts)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for record in records:
        held = subject_index[record["subject"]]
        config = canonical_configs[choices[held]]
        features = GammaCurveFeatures(record["curve"], int(outer[held]))
        replay[held] += prepared(record, features, config, backbone, mode).evaluate([config])[0]
    np.testing.assert_array_equal(replay, counts)
    return {
        "input": str(input_path), "input_sha256": digest(Path(input_path)),
        "subjects": len(subjects), "videos": len(records),
        "configuration_budget": len(canonical_configs),
        "selected_configuration_replay": "PASS",
        "inner_f1_replay": "PASS", "outer_count_replay": "PASS",
        "independent_prediction_replay": "PASS",
    }


def paired_bootstrap(candidate: np.ndarray, baseline: np.ndarray) -> dict:
    if candidate.shape != baseline.shape:
        raise AssertionError("Paired subject-count shapes differ")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(candidate), size=(BOOTSTRAP_RESAMPLES, len(candidate)))

    def f1(values):
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator, out=np.zeros(BOOTSTRAP_RESAMPLES),
            where=denominator > 0,
        )

    deltas = f1(candidate) - f1(baseline)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "ci_low": float(low), "ci_high": float(high),
        "positive_resample_fraction": float(np.mean(deltas > 0)),
    }


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_group(backbone: str, dataset: str, setting: str, all_configs):
    records, subjects, _, input_path = fair.load_data(backbone, dataset)
    outer, inner = fair.fold_priors(records, subjects, backbone)
    mode = "native" if backbone == "boostingvrme" else "fixed"
    by_k = {}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist())):
        needed = fair.required_subjects(k, outer, inner)
        by_k[k] = evaluate_k(
            records, subjects, backbone, k, needed, all_configs, mode
        )

    gamma_indexes = {
        gamma: [i for i, config in enumerate(all_configs) if config.gamma == gamma]
        for gamma in GAMMAS
    }
    canonical_choices, canonical_counts, canonical_rows = select(
        by_k, outer, inner, subjects, all_configs, gamma_indexes[BASELINE_GAMMA]
    )
    archived_rows, archived_counts = archived_baseline(backbone, dataset)
    if [row["subject"] for row in archived_rows] != subjects:
        raise AssertionError(f"Baseline subject order mismatch for {setting}")
    for current, archived, choice in zip(canonical_rows, archived_rows, canonical_choices):
        config = all_configs[choice]
        if config.canonical_identifier != archived["config"]:
            raise AssertionError(
                f"Baseline config replay mismatch for {setting}/{current['subject']}"
            )
        if float(current["inner_f1"]) != float(archived["inner_F1"]):
            raise AssertionError(
                f"Baseline inner F1 replay mismatch for {setting}/{current['subject']}"
            )
    np.testing.assert_array_equal(canonical_counts, archived_counts)

    selected_choices, selected_counts, selected_rows = select(
        by_k, outer, inner, subjects, all_configs, list(range(len(all_configs)))
    )
    fixed = {BASELINE_GAMMA: canonical_counts}
    for gamma in (0.25, 0.75):
        _, fixed[gamma], _ = select(
            by_k, outer, inner, subjects, all_configs, gamma_indexes[gamma]
        )

    replay, alignment, alignment_strata, transitions = replay_and_audit(
        records, subjects, outer, backbone, mode, all_configs,
        canonical_choices, selected_choices,
    )
    np.testing.assert_array_equal(replay["canonical"], canonical_counts)
    np.testing.assert_array_equal(replay["selected"], selected_counts)
    delta = selected_counts.sum(axis=0) - canonical_counts.sum(axis=0)
    if transitions["net_TP"] != delta[0] or transitions["net_FP"] != delta[1]:
        raise AssertionError(f"Candidate transition accounting mismatch for {setting}")
    for row in selected_rows:
        row["setting"] = setting
    return {
        "subjects": subjects,
        "canonical_counts": canonical_counts,
        "selected_counts": selected_counts,
        "selected_rows": selected_rows,
        "fixed": fixed,
        "alignment": alignment,
        "alignment_strata": alignment_strata,
        "transitions": transitions,
        "audit": {
            "input": str(input_path), "input_sha256": digest(Path(input_path)),
            "subjects": len(subjects), "videos": len(records),
            "baseline_fold_replay": "PASS", "outer_count_replay": "PASS",
            "outer_k_frequency": dict(Counter(str(value) for value in outer)),
            "inner_k_frequency": dict(Counter(str(value) for value in inner[inner > 0])),
        },
    }


def gamma_behavior(distributions: list[dict]) -> str:
    modes = {}
    for row in distributions:
        counts = [row[f"gamma_{suffix}_count"] for suffix in ("025", "050", "075")]
        modes[row["setting"]] = GAMMAS[int(np.argmax(counts))]
    if len(set(modes.values())) == 1:
        return "GLOBAL"
    dataset_consistent = all(
        modes[f"ME-TST/{name}"] == modes[f"BoostingVRME/{name}"]
        for name in ("SAMMLV", "CAS(ME)3")
    )
    backbone_consistent = all(
        modes[f"{name}/SAMMLV"] == modes[f"{name}/CAS(ME)3"]
        for name in ("ME-TST", "BoostingVRME")
    )
    if dataset_consistent and not backbone_consistent:
        return "DATASET_DEPENDENT"
    if backbone_consistent and not dataset_consistent:
        return "BACKBONE_DEPENDENT"
    return "MIXED"


def decide_gate(results: list[dict]) -> tuple[str, str, dict]:
    canonical_mean = float(np.mean([row["canonical_f1"] for row in results]))
    selected_mean = float(np.mean([row["exp4a_f1"] for row in results]))
    no_significant_regression = all(row["ci_high"] >= 0 for row in results)
    noninferior_count = sum(row["exp4a_f1"] >= row["canonical_f1"] for row in results)
    significant_improvement = any(row["ci_low"] > 0 for row in results)
    clear_mechanism = any(
        (row["delta_tp"] > 0 and row["delta_fp"] <= 0)
        or (row["delta_fp"] < 0 and row["delta_tp"] >= 0)
        for row in results
    )
    fp_explosion = any(
        row["delta_fp"] > FP_EXPLOSION_ABSOLUTE
        and row["exp4a_fp"] > row["canonical_fp"] * (1 + FP_EXPLOSION_RELATIVE)
        for row in results
    )
    strong = all((
        no_significant_regression, noninferior_count >= 3,
        selected_mean > canonical_mean,
        significant_improvement or clear_mechanism,
        not fp_explosion,
    ))
    failed = (not no_significant_regression) or selected_mean <= canonical_mean or fp_explosion
    if strong:
        gate = "EXP4A_STRONG_SETTING_CANDIDATE"
        recommendation = "A. Proceed to fairness validation"
    elif failed:
        gate = "EXP4A_FAILED"
        recommendation = "B. Retain canonical GLSD-v1"
    else:
        gate = "EXP4A_PARTIAL_SUPPORT"
        recommendation = "B. Retain canonical GLSD-v1"
    return gate, recommendation, {
        "canonical_mean": canonical_mean,
        "selected_mean": selected_mean,
        "no_significant_regression": no_significant_regression,
        "significant_improvement": significant_improvement,
        "fp_explosion": fp_explosion,
        "noninferior_count": noninferior_count,
        "clear_mechanism_improvement": clear_mechanism,
    }


def make_analysis(results, fixed_rows, distributions, mechanisms, transitions,
                  gate, recommendation, checks) -> str:
    lines = [
        "# EXP-4A Alignment Tolerance Search", "", "## 1. Hypothesis", "",
        "EXP-4A tests whether the fixed cross-scale peak-alignment tolerance is limiting canonical GLSD-v1. It is a setting-only experiment.",
        "", "## 2. Protocol", "",
        "The sole added parameter is `delta = max(1, round(gamma*k))`, with `gamma in {0.25, 0.50, 0.75}`. The canonical reference scales, radii, ten thresholds, median aggregation, missing-scale zero, equal G/L fusion, event geometry, outer/inner LOSO, and tie-breaks are unchanged. The main search budget is 270 configurations.",
        "", "## 3. Baseline Replay", "",
        "`gamma=0.50` recovered every canonical outer-fold selected configuration, inner F1, and TP/FP/FN exactly for all four settings: **PASS**.",
        "", "## 4. Main Results", "",
        "| Setting | Canonical | EXP-4A | Delta | CI |", "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['setting']} | {row['canonical_f1']:.6f} | {row['exp4a_f1']:.6f} | "
            f"{row['delta_f1']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |"
        )
    lines.extend(["", "## 5. Fixed-Gamma Diagnostic", "",
                  "| Setting | gamma=.25 | gamma=.50 | gamma=.75 | Selected |",
                  "|---|---:|---:|---:|---:|"])
    for row in fixed_rows:
        lines.append(
            f"| {row['setting']} | {row['gamma_025_f1']:.6f} | {row['gamma_050_f1']:.6f} | "
            f"{row['gamma_075_f1']:.6f} | {row['gamma_selected_f1']:.6f} |"
        )
    lines.extend(["", "## 6. Selected Gamma Distribution", ""])
    for row in distributions:
        lines.append(
            f"- {row['setting']}: .25={row['gamma_025_count']} ({row['gamma_025_proportion']:.3f}), "
            f".50={row['gamma_050_count']} ({row['gamma_050_proportion']:.3f}), "
            f".75={row['gamma_075_count']} ({row['gamma_075_proportion']:.3f})."
        )
    lines.extend(["", "## 7. TP/FP/FN", ""])
    for row in results:
        lines.append(
            f"- {row['setting']}: canonical {row['canonical_tp']}/{row['canonical_fp']}/{row['canonical_fn']} -> "
            f"EXP-4A {row['exp4a_tp']}/{row['exp4a_fp']}/{row['exp4a_fn']} "
            f"(delta {row['delta_tp']:+d}/{row['delta_fp']:+d}/{row['delta_fn']:+d})."
        )
    lines.extend(["", "## 8. Alignment Mechanism", ""])
    for row in mechanisms:
        lines.append(
            f"- {row['setting']}: matched rate {row['canonical_alignment_rate']:.6f} -> "
            f"{row['exp4a_alignment_rate']:.6f}; missing rate {row['canonical_missing_rate']:.6f} -> "
            f"{row['exp4a_missing_rate']:.6f}; reuse {row['canonical_reuse_count']} -> "
            f"{row['exp4a_reuse_count']}; collisions {row['canonical_collision_count']} -> "
            f"{row['exp4a_collision_count']}."
        )
        for gamma, label in ((0.25, "stricter"), (0.75, "wider")):
            suffix = "025" if gamma == 0.25 else "075"
            folds = row[f"gamma_{suffix}_fold_count"]
            if not folds:
                lines.append(
                    f"  - No outer fold selected the {label} gamma={gamma:.2f} setting."
                )
                continue
            old_rate = row[f"gamma_{suffix}_canonical_alignment_rate"]
            new_rate = row[f"gamma_{suffix}_selected_alignment_rate"]
            lines.append(
                f"  - On {folds} fold(s) selecting the {label} gamma={gamma:.2f}, matched rate "
                f"changed {old_rate:.6f} -> {new_rate:.6f}; fold-pooled changes were "
                f"TP {row[f'gamma_{suffix}_delta_tp']:+d}, FP {row[f'gamma_{suffix}_delta_fp']:+d}, "
                f"FN {row[f'gamma_{suffix}_delta_fn']:+d}."
            )
    lines.extend(["", "## 9. Candidate Transition", ""])
    for row in transitions:
        lines.append(
            f"- {row['setting']}: rescued GT={row['rescued_gt']}, removed FP={row['removed_fp']}, "
            f"lost GT={row['lost_gt']}, new FP={row['new_fp']}; mean delta L={row['mean_delta_L']}, "
            f"median delta L={row['median_delta_L']} over {row['shared_candidate_count_for_L']} shared candidates."
        )
    lines.extend([
        "", "Observed matched/missing changes above—not the theoretical direction alone—are the basis for interpretation. Narrower selected tolerances are credited only where matched rate actually falls; wider tolerances only where it rises.",
        "", "## 10. Gate", "", f"**{gate}**", "",
        "Pre-registered checks: " + "; ".join(
            f"{key}={value}" for key, value in checks.items()
        ) + ".",
        "", "## 11. Recommendation", "", f"**{recommendation}**", "",
        "Canonical GLSD-v1 was not modified. No finer gamma grid, EXP-4B, or automatic canonical upgrade was run.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = RETHINK / "caches/me_tst"
    all_configs = configs()
    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }
    baseline_preflight = {}
    try:
        for backbone, dataset, setting in GROUPS:
            fair.log(f"[{setting}] locked gamma=.50 baseline recovery")
            baseline_preflight[setting] = baseline_recovery_group(
                backbone, dataset, setting, all_configs
            )
    except (AssertionError, ValueError) as error:
        failure = {"status": "BASELINE_REPLAY_FAILED", "error": str(error)}
        (OUT / "protocol.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print("BASELINE_REPLAY_FAILED")
        raise SystemExit(2) from error

    runs = []
    for backbone, dataset, setting in GROUPS:
        fair.log(f"[{setting}] formal 270-config EXP-4A search")
        runs.append((backbone, dataset, setting, run_group(
            backbone, dataset, setting, all_configs
        )))

    result_rows, fixed_rows, selection_rows = [], [], []
    distribution_rows, mechanism_rows, transition_rows, bootstrap_rows = [], [], [], []
    audits = {}
    for backbone, dataset, setting, run in runs:
        canonical = run["canonical_counts"]
        selected = run["selected_counts"]
        canonical_metrics = fair.metrics(canonical.sum(axis=0))
        selected_metrics = fair.metrics(selected.sum(axis=0))
        snapshot = baseline_main[backbone, dataset]
        expected = np.array([
            int(snapshot["GL_Skill_TP"]), int(snapshot["GL_Skill_FP"]),
            int(snapshot["GL_Skill_FN"]),
        ])
        np.testing.assert_array_equal(canonical.sum(axis=0), expected)
        interval = paired_bootstrap(selected, canonical)
        result_rows.append({
            "setting": setting,
            "canonical_f1": canonical_metrics["F1"],
            "exp4a_f1": selected_metrics["F1"],
            "delta_f1": selected_metrics["F1"] - canonical_metrics["F1"],
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "canonical_tp": canonical_metrics["TP"], "canonical_fp": canonical_metrics["FP"],
            "canonical_fn": canonical_metrics["FN"], "exp4a_tp": selected_metrics["TP"],
            "exp4a_fp": selected_metrics["FP"], "exp4a_fn": selected_metrics["FN"],
            "delta_tp": selected_metrics["TP"] - canonical_metrics["TP"],
            "delta_fp": selected_metrics["FP"] - canonical_metrics["FP"],
            "delta_fn": selected_metrics["FN"] - canonical_metrics["FN"],
        })
        bootstrap_rows.append({"setting": setting, "delta_f1": result_rows[-1]["delta_f1"], **interval,
                               "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED})
        fixed_metrics = {gamma: fair.metrics(value.sum(axis=0)) for gamma, value in run["fixed"].items()}
        fixed_rows.append({
            "setting": setting, "gamma_025_f1": fixed_metrics[0.25]["F1"],
            "gamma_050_f1": fixed_metrics[0.50]["F1"],
            "gamma_075_f1": fixed_metrics[0.75]["F1"],
            "gamma_selected_f1": selected_metrics["F1"],
        })
        selection_rows.extend(run["selected_rows"])
        gamma_counts = Counter(float(row["selected_gamma"]) for row in run["selected_rows"])
        total = len(run["selected_rows"])
        distribution_rows.append({
            "setting": setting,
            "gamma_025_count": gamma_counts[0.25], "gamma_050_count": gamma_counts[0.50],
            "gamma_075_count": gamma_counts[0.75],
            "gamma_025_proportion": gamma_counts[0.25] / total,
            "gamma_050_proportion": gamma_counts[0.50] / total,
            "gamma_075_proportion": gamma_counts[0.75] / total,
        })
        mechanism = {"setting": setting}
        for name, prefix in (("canonical", "canonical"), ("selected", "exp4a")):
            values = run["alignment"][name]
            opportunities = values["matched_alignment_count"] + values["missing_alignment_count"]
            for key, value in values.items():
                mechanism[f"{prefix}_{key}"] = value
            mechanism[f"{prefix}_alignment_rate"] = rate(values["matched_alignment_count"], opportunities)
            mechanism[f"{prefix}_missing_rate"] = rate(values["missing_alignment_count"], opportunities)
        selected_gammas = np.array([
            float(row["selected_gamma"]) for row in run["selected_rows"]
        ])
        for gamma, suffix in ((0.25, "025"), (0.75, "075")):
            mask = selected_gammas == gamma
            fold_delta = selected[mask].sum(axis=0) - canonical[mask].sum(axis=0)
            mechanism[f"gamma_{suffix}_fold_count"] = int(mask.sum())
            mechanism[f"gamma_{suffix}_delta_tp"] = int(fold_delta[0])
            mechanism[f"gamma_{suffix}_delta_fp"] = int(fold_delta[1])
            mechanism[f"gamma_{suffix}_delta_fn"] = int(fold_delta[2])
            for name in ("canonical", "selected"):
                values = run["alignment_strata"][gamma][name]
                opportunities = values["matched_alignment_count"] + values["missing_alignment_count"]
                mechanism[f"gamma_{suffix}_{name}_alignment_rate"] = rate(
                    values["matched_alignment_count"], opportunities
                )
                mechanism[f"gamma_{suffix}_{name}_missing_rate"] = rate(
                    values["missing_alignment_count"], opportunities
                )
        mechanism_rows.append(mechanism)
        changes = run["transitions"]
        transition_rows.append({
            "setting": setting, "rescued_gt": changes["GT_rescued"],
            "removed_fp": changes["FP_removed_exact_interval"], "lost_gt": changes["GT_lost"],
            "new_fp": changes["FP_added_exact_interval"],
            "shared_candidate_count_for_L": changes["shared_candidate_count_for_L"],
            "mean_delta_L": changes["mean_delta_L"], "median_delta_L": changes["median_delta_L"],
        })
        audits[setting] = run["audit"]

    write_csv(OUT / "results.csv", result_rows, list(result_rows[0]))
    write_csv(OUT / "fixed_gamma_results.csv", fixed_rows, list(fixed_rows[0]))
    write_csv(OUT / "selected_gamma_distribution.csv", distribution_rows, list(distribution_rows[0]))
    write_csv(OUT / "selected_config_per_subject.csv", selection_rows, [
        "setting", "subject", "selected_gamma", "selected_a0", "selected_rho", "selected_tau",
        "inner_f1", "inner_precision", "inner_fp", "outer_tp", "outer_fp", "outer_fn",
    ])
    write_csv(OUT / "alignment_mechanism.csv", mechanism_rows, list(mechanism_rows[0]))
    write_csv(OUT / "candidate_transition.csv", transition_rows, list(transition_rows[0]))
    write_csv(OUT / "bootstrap.csv", bootstrap_rows, list(bootstrap_rows[0]))

    gate, recommendation, checks = decide_gate(result_rows)
    behavior = gamma_behavior(distribution_rows)
    overall = Counter(float(row["selected_gamma"]) for row in selection_rows)
    maximum = max(overall.values())
    most_selected = ",".join(f"{gamma:.2f}" for gamma in GAMMAS if overall[gamma] == maximum)
    protocol = {
        "experiment": "GLSD controlled exploration EXP-4A",
        "name": "Cross-Scale Alignment Tolerance Search",
        "parent": "canonical GLSD-v1", "only_change": "alignment tolerance coefficient gamma",
        "formula": "delta=max(1, round(gamma*k))", "gamma_candidates": list(GAMMAS),
        "baseline_gamma": BASELINE_GAMMA, "reference_scales": list(engine.REFERENCE_SCALES),
        "local_radii": list(engine.LOCAL_RADII), "thresholds": list(engine.THRESHOLDS),
        "configuration_budget": 270, "fixed_gamma_budget": 90,
        "aggregation": "median", "missing_scale": 0, "fusion": "S=(G+L)/2",
        "outer": "LOSO", "inner": "LOSO excluding outer and inner validation subject",
        "selection": "pooled inner F1; ties higher precision, fewer FP, fixed grid order",
        "bootstrap": {"unit": "subject", "paired": True, "resamples": 10000, "seed": 100},
        "fp_explosion_rule": "delta FP > 10 and selected FP > 1.10 * canonical FP",
        "alignment_audit": {
            "nonreference": "distinct physical smoothing widths excluding the reference width",
            "matched_missing_unit": "reference candidate x nonreference physical width",
            "collision_count": "number of nonreference peaks assigned to more than one reference candidate",
            "reuse_count": "assignments beyond the first to a reused nonreference peak",
        },
        "baseline_replay": "PASS", "baseline_preflight": baseline_preflight,
        "canonical_glsd_modified": False,
        "continue_finer_gamma_search": False, "gate": gate,
        "recommendation": recommendation, "gamma_behavior": behavior,
        "source_sha256": {
            "run_exp4a.py": digest(Path(__file__)),
            "unified_persistence.py": digest(SIGNED / "unified_persistence.py"),
            "equiscale_fair_validation.py": digest(SIGNED / "equiscale_fair_validation.py"),
        },
        "audits": audits,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "EXP4A_ANALYSIS.md").write_text(
        make_analysis(result_rows, fixed_rows, distribution_rows, mechanism_rows,
                      transition_rows, gate, recommendation, checks), encoding="utf-8"
    )

    print("================================")
    print("EXP-4A FINAL VERDICT")
    print("================================")
    print("Baseline replay: PASS")
    print(f"Gate: {gate}")
    print(f"Canonical mean F1: {checks['canonical_mean']:.6f}")
    print(f"EXP-4A mean F1: {checks['selected_mean']:.6f}")
    for row in result_rows:
        print(f"{row['setting']}: {row['canonical_f1']:.6f} -> {row['exp4a_f1']:.6f} ({row['delta_f1']:+.6f})")
    print(f"Any significant regression: {'YES' if not checks['no_significant_regression'] else 'NO'}")
    print(f"Any significant improvement: {'YES' if checks['significant_improvement'] else 'NO'}")
    print(f"FP explosion: {'YES' if checks['fp_explosion'] else 'NO'}")
    print(f"Most selected gamma: {most_selected}")
    print(f"Gamma behavior: {behavior}")
    print("Canonical GLSD modified: NO")
    print("Continue finer gamma search: NO")
    print(f"Recommended next step: {recommendation}")


if __name__ == "__main__":
    main()
