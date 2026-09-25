"""EXP-3A: per-video mid-rank calibration of EXP-2C structural evidence.

This script deliberately imports the EXP-2C feature implementation and changes
only the two score columns used by the fixed fusion: G and L.  Candidate
generation, event decoding, nested LOSO, thresholds, and tie-breaking are
replayed through the archived parent engine.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / "controlled_exploration/EXP2C_reference_evidence_decoupling"
RETHINK = ROOT / "RethinkFuse_reproduction"
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
BASELINE = ROOT / "controlled_exploration/baseline_snapshot"

sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(SIGNED))
import equiscale_fair_validation as fair  # noqa: E402
import unified_persistence as engine  # noqa: E402
import run_exp2c as parent  # noqa: E402


GROUPS = parent.GROUPS
REFERENCE_SCALES = parent.REFERENCE_SCALES
SEED = 100
BOOTSTRAP_RESAMPLES = 10_000
TOP_FRACTION_IQR_LIMIT = 0.05
TOP_FRACTION_CENTRAL_MASS = 0.90
TOP_FRACTION_MEDIAN_BAND = 0.10
FP_INFLATION_RELATIVE_LIMIT = 0.05


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


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    """Deterministic average-rank percentile, (R - 0.5) / n."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("midrank_percentile expects a one-dimensional vector")
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(n, dtype=float)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranked[order[start:stop]] = (average_rank - 0.5) / n
        start = stop
    return ranked


def rank_evidence(evidence: np.ndarray) -> np.ndarray:
    ranked = np.asarray(evidence, dtype=float).copy()
    if len(ranked):
        ranked[:, 1] = midrank_percentile(ranked[:, 1])
        ranked[:, 2] = midrank_percentile(ranked[:, 2])
    return ranked


def ordering_preserved(raw: np.ndarray, ranked: np.ndarray) -> bool:
    if len(raw) < 2:
        return True
    order = np.argsort(raw, kind="mergesort")
    raw_sorted = raw[order]
    ranked_sorted = ranked[order]
    distinct = np.diff(raw_sorted) > 0
    return bool(np.all(np.diff(ranked_sorted)[distinct] > 0))


def spearman_for_transform(raw: np.ndarray, ranked: np.ndarray) -> float | None:
    if len(raw) < 2 or np.ptp(raw) == 0:
        return None
    raw_ranks = midrank_percentile(raw)
    ranked_ranks = midrank_percentile(ranked)
    return float(np.corrcoef(raw_ranks, ranked_ranks)[0, 1])


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {key: math.nan for key in ("mean", "median", "q1", "q3", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def calibration_sanity_audit() -> tuple[list[dict], list[dict], bool]:
    """Audit every outer-fold video/reference/radius context without using GT."""
    sanity_rows = []
    distribution_rows = []
    all_preserved = True
    for backbone, dataset, setting in GROUPS:
        records, subjects, _, _ = fair.load_data(backbone, dataset)
        outer, _ = fair.fold_priors(records, subjects, backbone)
        subject_index = {subject: index for index, subject in enumerate(subjects)}
        score_buckets = {
            "raw_g": [], "raw_l": [], "raw_s": [],
            "rank_g": [], "rank_l": [], "rank_s": [],
        }
        for reference in REFERENCE_SCALES:
            for radius in engine.LOCAL_RADII:
                counts = []
                g_ratios, l_ratios, g_ties, l_ties = [], [], [], []
                g_spearman, l_spearman = [], []
                inversions = 0
                for record in records:
                    held = subject_index[record["subject"]]
                    features = parent.DecoupledCurveFeatures(record["curve"], int(outer[held]))
                    _, evidence = features.evidence(reference, radius)
                    ranked = rank_evidence(evidence)
                    n = len(evidence)
                    counts.append(n)
                    if n:
                        for raw, calibrated, prefix in (
                            (evidence[:, 1], ranked[:, 1], "g"),
                            (evidence[:, 2], ranked[:, 2], "l"),
                        ):
                            unique_ratio = len(np.unique(raw)) / n
                            tie_rate = 1.0 - unique_ratio
                            (g_ratios if prefix == "g" else l_ratios).append(unique_ratio)
                            (g_ties if prefix == "g" else l_ties).append(tie_rate)
                            rho = spearman_for_transform(raw, calibrated)
                            if rho is not None:
                                (g_spearman if prefix == "g" else l_spearman).append(rho)
                            if not ordering_preserved(raw, calibrated):
                                inversions += 1
                    score_buckets["raw_g"].extend(evidence[:, 1].tolist())
                    score_buckets["raw_l"].extend(evidence[:, 2].tolist())
                    score_buckets["raw_s"].extend(((evidence[:, 1] + evidence[:, 2]) / 2).tolist())
                    score_buckets["rank_g"].extend(ranked[:, 1].tolist())
                    score_buckets["rank_l"].extend(ranked[:, 2].tolist())
                    score_buckets["rank_s"].extend(((ranked[:, 1] + ranked[:, 2]) / 2).tolist())
                summary = quantiles(counts)
                row = {
                    "setting": setting,
                    "reference_scale": reference,
                    "radius": radius,
                    "num_videos": len(records),
                    "videos_with_0_candidates": int(sum(value == 0 for value in counts)),
                    "videos_with_1_candidate": int(sum(value == 1 for value in counts)),
                    "videos_with_2_candidates": int(sum(value == 2 for value in counts)),
                    "videos_with_3plus_candidates": int(sum(value >= 3 for value in counts)),
                    "candidate_count_mean": summary["mean"],
                    "candidate_count_median": summary["median"],
                    "candidate_count_q1": summary["q1"],
                    "candidate_count_q3": summary["q3"],
                    "candidate_count_max": summary["max"],
                    "G_unique_value_ratio": float(np.mean(g_ratios)) if g_ratios else math.nan,
                    "L_unique_value_ratio": float(np.mean(l_ratios)) if l_ratios else math.nan,
                    "G_tie_rate": float(np.mean(g_ties)) if g_ties else math.nan,
                    "L_tie_rate": float(np.mean(l_ties)) if l_ties else math.nan,
                    "Spearman(G,G_rank)": float(np.mean(g_spearman)) if g_spearman else math.nan,
                    "Spearman(L,L_rank)": float(np.mean(l_spearman)) if l_spearman else math.nan,
                    "G_spearman_defined_sets": len(g_spearman),
                    "L_spearman_defined_sets": len(l_spearman),
                    "ordering_inversions": inversions,
                }
                sanity_rows.append(row)
                all_preserved = all_preserved and inversions == 0
        distribution = {"setting": setting}
        for name, values in score_buckets.items():
            summary = quantiles(values)
            for statistic in ("mean", "median", "q1", "q3"):
                distribution[f"{name}_{statistic}"] = summary[statistic]
        distribution_rows.append(distribution)
    return sanity_rows, distribution_rows, all_preserved


def exp3a_configs() -> list[engine.Config]:
    configs = parent.exp2c_configs()
    if tuple(engine.THRESHOLDS) != (
        0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75
    ):
        raise AssertionError("Threshold grid differs from the locked EXP-2C grid")
    return configs


def prepared_pair(record, features, reference, backbone, mode, radius):
    peaks, evidence = features.evidence(reference, radius)
    decoder = fair.VideoFeatures(record, features.k, backbone if mode == "native" else "metst")
    return (
        engine.Prepared(decoder, peaks, evidence),
        engine.Prepared(decoder, peaks, rank_evidence(evidence)),
    )


def evaluate_k_both(records, subjects, backbone, k, needed, configs, mode):
    stats = {
        method: np.zeros((len(configs), len(subjects), 3), dtype=np.int32)
        for method in ("exp2c", "exp3a")
    }
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    batches = {
        (reference, radius): [
            index for index, config in enumerate(configs)
            if (config.reference, config.radius) == (reference, radius)
        ]
        for reference in REFERENCE_SCALES
        for radius in engine.LOCAL_RADII
    }
    for video_index, record in enumerate(records):
        subject_id = subject_ids[record["subject"]]
        if not needed[subject_id]:
            continue
        features = parent.DecoupledCurveFeatures(record["curve"], k)
        for (reference, radius), indexes in batches.items():
            selected = [configs[index] for index in indexes]
            raw, ranked = prepared_pair(
                record, features, reference, backbone, mode, radius
            )
            stats["exp2c"][indexes, subject_id] += raw.evaluate(selected)
            stats["exp3a"][indexes, subject_id] += ranked.evaluate(selected)
        if (video_index + 1) % 100 == 0:
            fair.log(f"[{backbone}] EXP-3A k={k}: {video_index + 1}/{len(records)} videos")
    return stats


def metric_f1(counts: np.ndarray) -> float:
    return float(fair.metrics(counts.sum(axis=0))["F1"])


def paired_bootstrap(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.shape != reference.shape:
        raise AssertionError("Paired subject-count shapes differ")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(candidate), size=(BOOTSTRAP_RESAMPLES, len(candidate)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(
            2 * totals[:, 0], denominator,
            out=np.zeros(BOOTSTRAP_RESAMPLES), where=denominator > 0,
        )

    deltas = f1(candidate) - f1(reference)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "ci95_low": float(low),
        "ci95_high": float(high),
        "positive_resample_fraction": float(np.mean(deltas > 0)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
    }


def parent_selection_map(setting: str) -> dict[str, dict]:
    rows = load_csv(PARENT / "selected_reference_scale_distribution.csv")
    return {row["subject"]: row for row in rows if row["setting"] == setting}


def run_group(backbone: str, dataset: str, setting: str, configs: list[engine.Config]):
    records, subjects, _, input_path = fair.load_data(backbone, dataset)
    outer, inner = fair.fold_priors(records, subjects, backbone)
    mode = "native" if backbone == "boostingvrme" else "fixed"
    stats = {method: {} for method in ("exp2c", "exp3a")}
    needed_k = sorted(set(outer.tolist()) | set(inner[inner > 0].tolist()))
    for k in needed_k:
        needed = fair.required_subjects(k, outer, inner)
        evaluated = evaluate_k_both(records, subjects, backbone, k, needed, configs, mode)
        for method in stats:
            stats[method][k] = evaluated[method]

    choices = {method: [] for method in stats}
    counts = {
        method: np.zeros((len(subjects), 3), dtype=np.int64) for method in stats
    }
    rows = []
    indexes = list(range(len(configs)))
    for held, subject in enumerate(subjects):
        row = {"setting": setting, "subject": subject}
        for method in ("exp2c", "exp3a"):
            training = engine.inner_counts(stats[method], held, inner)
            choice = engine.choose(training, indexes)
            config = configs[choice]
            choices[method].append(choice)
            counts[method][held] = stats[method][int(outer[held])][choice, held]
            row.update({
                f"{method}_a0": config.reference,
                f"{method}_r": config.radius,
                f"{method}_tau": config.threshold,
                f"{method}_inner_f1": fair.metrics(training[choice])["F1"],
                f"{method}_outer_tp": int(counts[method][held, 0]),
                f"{method}_outer_fp": int(counts[method][held, 1]),
                f"{method}_outer_fn": int(counts[method][held, 2]),
            })
        rows.append(row)

    archived = parent_selection_map(setting)
    if list(archived) != subjects:
        raise AssertionError(f"EXP-2C archived subject order mismatch for {setting}")
    for row in rows:
        source = archived[row["subject"]]
        expected = (
            float(source["selected_a0"]), float(source["selected_radius"]),
            float(source["selected_tau"]), int(source["outer_tp"]),
            int(source["outer_fp"]), int(source["outer_fn"]),
        )
        actual = (
            float(row["exp2c_a0"]), float(row["exp2c_r"]),
            float(row["exp2c_tau"]), int(row["exp2c_outer_tp"]),
            int(row["exp2c_outer_fp"]), int(row["exp2c_outer_fn"]),
        )
        if actual != expected:
            raise AssertionError(f"EXP-2C fold replay mismatch for {setting}/{row['subject']}")

    replay = {method: np.zeros_like(counts[method]) for method in counts}
    fractions = {method: [] for method in counts}
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for record in records:
        held = subject_index[record["subject"]]
        features = parent.DecoupledCurveFeatures(record["curve"], int(outer[held]))
        for method_index, method in enumerate(("exp2c", "exp3a")):
            config = configs[choices[method][held]]
            prepared = prepared_pair(
                record, features, config.reference, backbone, mode, config.radius
            )[method_index]
            replay[method][held] += prepared.evaluate([config])[0]
            candidate_count = len(prepared.evidence)
            if candidate_count:
                selected_count = int(prepared.masks([config])[0].sum())
                fractions[method].append(selected_count / candidate_count)
    for method in counts:
        np.testing.assert_array_equal(replay[method], counts[method])

    return {
        "subjects": subjects,
        "counts": counts,
        "rows": rows,
        "choices": choices,
        "fractions": fractions,
        "input": str(input_path),
        "input_sha256": digest(Path(input_path)),
        "videos": len(records),
        "outer_k_frequency": dict(Counter(str(value) for value in outer)),
        "inner_k_frequency": dict(Counter(str(value) for value in inner[inner > 0])),
        "outer_count_replay": "PASS",
        "parent_fold_replay": "PASS",
    }


def fraction_summary(setting: str, fractions: dict[str, list[float]]) -> dict:
    row = {"setting": setting}
    for method in ("exp2c", "exp3a"):
        values = np.asarray(fractions[method], dtype=float)
        summary = quantiles(values.tolist())
        row.update({
            f"{method}_nonempty_videos": len(values),
            f"{method}_mean_selected_candidate_fraction": summary["mean"],
            f"{method}_median_selected_candidate_fraction": summary["median"],
            f"{method}_q1_selected_candidate_fraction": summary["q1"],
            f"{method}_q3_selected_candidate_fraction": summary["q3"],
            f"{method}_std_selected_candidate_fraction": float(np.std(values)) if len(values) else math.nan,
        })
    rank = np.asarray(fractions["exp3a"], dtype=float)
    if len(rank):
        median = float(np.median(rank))
        iqr = float(np.quantile(rank, 0.75) - np.quantile(rank, 0.25))
        central_mass = float(np.mean(np.abs(rank - median) <= TOP_FRACTION_MEDIAN_BAND))
        potential = iqr <= TOP_FRACTION_IQR_LIMIT and central_mass >= TOP_FRACTION_CENTRAL_MASS
    else:
        central_mass = math.nan
        potential = False
    row["exp3a_fraction_within_0.10_of_median"] = central_mass
    row["potential_top_fraction_behavior"] = "YES" if potential else "NO"
    return row


def threshold_distribution(rows: list[dict]) -> list[dict]:
    output = []
    for setting in [group[2] for group in GROUPS]:
        subset = [row for row in rows if row["setting"] == setting]
        for method in ("exp2c", "exp3a"):
            values = [float(row[f"{method}_tau"]) for row in subset]
            counts = Counter(values)
            for threshold in engine.THRESHOLDS:
                output.append({
                    "setting": setting,
                    "method": method.upper(),
                    "threshold": threshold,
                    "fold_count": counts.get(threshold, 0),
                    "fold_fraction": counts.get(threshold, 0) / len(values),
                })
    return output


def threshold_summary(rows: list[dict], setting: str, method: str) -> str:
    values = [float(row[f"{method}_tau"]) for row in rows if row["setting"] == setting]
    counts = Counter(values)
    return ", ".join(f"{value:g}:{counts[value]}" for value in sorted(counts))


def mechanism_label(exp2c: np.ndarray, exp3a: np.ndarray) -> str:
    delta = exp3a - exp2c
    if delta[0] > 0 and delta[2] < 0 and delta[1] <= 0:
        return "A. restored TP/recall without FP inflation"
    if delta[1] < 0 and delta[0] >= 0:
        return "B. suppressed FP without losing TP"
    if delta[1] > 0:
        return "D. increased FP (despite any recall gain)"
    return "C. shifted the precision-recall operating point"


def decide_gate(results: list[dict], mechanisms: list[dict]) -> tuple[str, str, dict]:
    no_significant_regression = all(row["ci_high_3a_vs_2c"] >= 0 for row in results)
    at_least_three_noninferior = sum(row["exp3a_f1"] >= row["exp2c_f1"] for row in results) >= 3
    cas_preserved = all(
        row["exp3a_f1"] > row["baseline_f1"]
        for row in results if "CAS(ME)3" in row["setting"]
    )
    mean_noninferior = np.mean([row["exp3a_f1"] for row in results]) >= np.mean(
        [row["exp2c_f1"] for row in results]
    )
    bs_result = next(row for row in results if row["setting"] == "BoostingVRME/SAMMLV")
    bs_counts = next(row for row in mechanisms if row["setting"] == "BoostingVRME/SAMMLV")
    fp_limit = bs_counts["exp2c_fp"] * (1 + FP_INFLATION_RELATIVE_LIMIT)
    bs_repaired = (
        bs_result["exp3a_f1"] >= bs_result["baseline_f1"]
        or (
            bs_counts["delta_tp_3a_vs_2c"] > 0
            and bs_counts["delta_fn_3a_vs_2c"] < 0
            and bs_counts["exp3a_fp"] <= fp_limit
        )
    )
    strong = all((
        no_significant_regression,
        at_least_three_noninferior,
        cas_preserved,
        bs_repaired,
        mean_noninferior,
    ))
    hard_fail = (
        not no_significant_regression
        or not cas_preserved
        or not mean_noninferior
    )
    if strong:
        gate = "EXP3A_STRONG_CANDIDATE"
        recommendation = "A. Proceed to prospective validation"
    elif hard_fail:
        gate = "EXP3A_FAILED"
        recommendation = "B. Retain EXP-2C"
    else:
        gate = "EXP3A_PARTIAL_SUPPORT"
        recommendation = "B. Retain EXP-2C"
    checks = {
        "no_significant_regression": no_significant_regression,
        "at_least_three_noninferior": at_least_three_noninferior,
        "cas_gains_preserved": cas_preserved,
        "boostingvrme_sammlv_repaired": bs_repaired,
        "mean_f1_noninferior": bool(mean_noninferior),
        "global_parameter_free_formula": True,
    }
    return gate, recommendation, checks


def fmt(value: float) -> str:
    return f"{value:.6f}"


def make_analysis(
    results: list[dict], mechanisms: list[dict], sanity: list[dict],
    fractions: list[dict], selections: list[dict], gate: str,
    recommendation: str, checks: dict, ranking_preserved: bool,
) -> str:
    lines = [
        "# EXP-3A Rank-Calibrated Structural Evidence",
        "",
        "## 1. Hypothesis",
        "",
        "EXP-3A tests whether cross-response-source differences are caused by score-calibration mismatch. It does not introduce new structural evidence: candidate generation, G, L, nearest-3 evidence, median aggregation, and event geometry are inherited unchanged from EXP-2C.",
        "",
        "## 2. Protocol",
        "",
        "Within each video candidate set, ties receive average rank `R`. The fixed transforms are `G_rank=(R(G)-0.5)/n` and `L_rank=(R(L)-0.5)/n`, followed by the unchanged equal fusion `S_rank=(G_rank+L_rank)/2`. For `n=1`, every calibrated score is 0.5; for `n=0`, the prediction remains empty. No GT or cross-video statistic enters calibration.",
        "",
        "The search remains 7 reference scales × 3 radii × 10 locked EXP-2C thresholds = 210 configurations. Outer LOSO, inner LOSO, tie-breaks, subject splits, seed 100, and 10,000 subject-level paired bootstrap resamples are unchanged.",
        "",
        "## 3. Calibration Sanity",
        "",
        f"The no-GT preflight audited {len(sanity)} setting/reference/radius rows. Candidate ranking preservation: **{'PASS' if ranking_preserved else 'FAIL'}**. Constant or singleton score vectors have undefined Spearman correlation and are counted separately; every defined per-video Spearman value is summarized in `calibration_sanity.csv`.",
        "",
        "## 4. Main Results",
        "",
        "| Setting | Baseline | 2C | 3A | 3A-2C | Paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['setting']} | {fmt(row['baseline_f1'])} | {fmt(row['exp2c_f1'])} | "
            f"{fmt(row['exp3a_f1'])} | {fmt(row['delta_3a_vs_2c'])} | "
            f"[{fmt(row['ci_low_3a_vs_2c'])}, {fmt(row['ci_high_3a_vs_2c'])}] |"
        )
    lines.extend(["", "## 5. TP/FP/FN", ""])
    for row in mechanisms:
        lines.append(
            f"- {row['setting']}: EXP-2C {row['exp2c_tp']}/{row['exp2c_fp']}/{row['exp2c_fn']} → "
            f"EXP-3A {row['exp3a_tp']}/{row['exp3a_fp']}/{row['exp3a_fn']} "
            f"(ΔTP {row['delta_tp_3a_vs_2c']:+d}, ΔFP {row['delta_fp_3a_vs_2c']:+d}, "
            f"ΔFN {row['delta_fn_3a_vs_2c']:+d})."
        )
    bs2 = next(row for row in mechanisms if row["setting"] == "BoostingVRME/SAMMLV")
    lines.append("")
    lines.append(f"For BoostingVRME/SAMMLV, the mechanism classification is **{mechanism_label(np.array([bs2['exp2c_tp'], bs2['exp2c_fp'], bs2['exp2c_fn']]), np.array([bs2['exp3a_tp'], bs2['exp3a_fp'], bs2['exp3a_fn']]))}**.")
    lines.extend(["", "## 6. Threshold Behavior", ""])
    for setting in [group[2] for group in GROUPS]:
        lines.append(
            f"- {setting}: EXP-2C `{threshold_summary(selections, setting, 'exp2c')}`; "
            f"EXP-3A `{threshold_summary(selections, setting, 'exp3a')}`."
        )
    lines.append("")
    lines.append("Threshold concentration is diagnostic only; it is not used as evidence of superiority and the threshold grid was not adapted to ranked scores.")
    lines.extend(["", "## 7. Score Distribution", ""])
    lines.append("The transform replaces within-video magnitudes with average-rank percentiles separately for G and L. `score_distribution_before_after.csv` reports pooled no-GT preflight summaries, while `selected_candidate_fraction.csv` checks whether the locked thresholds behave like a nearly fixed top-fraction selector.")
    for row in fractions:
        lines.append(
            f"- {row['setting']}: mean selected fraction {fmt(row['exp2c_mean_selected_candidate_fraction'])} → "
            f"{fmt(row['exp3a_mean_selected_candidate_fraction'])}; potential top-fraction behavior: "
            f"{row['potential_top_fraction_behavior']}."
        )
    lines.extend(["", "## 8. Failure Analysis", ""])
    declines = [row for row in results if row["exp3a_f1"] < row["exp2c_f1"]]
    if declines:
        mechanism_map = {row["setting"]: row for row in mechanisms}
        for row in declines:
            mech = mechanism_map[row["setting"]]
            lines.append(
                f"- {row['setting']} declined by {fmt(row['delta_3a_vs_2c'])}. The count change was "
                f"ΔTP {mech['delta_tp_3a_vs_2c']:+d}, ΔFP {mech['delta_fp_3a_vs_2c']:+d}, "
                f"ΔFN {mech['delta_fn_3a_vs_2c']:+d}; the paired CI was "
                f"[{fmt(row['ci_low_3a_vs_2c'])}, {fmt(row['ci_high_3a_vs_2c'])}]."
            )
    else:
        lines.append("No setting had lower aggregate F1 than EXP-2C.")
    lines.extend([
        "",
        "## 9. Gate",
        "",
        f"**{gate}**",
        "",
        "Pre-registered checks: " + "; ".join(
            f"{key}={'YES' if value else 'NO'}" for key, value in checks.items()
        ) + ".",
        "",
        "## 10. Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "Canonical GLSD-v1 remains unchanged. No additional calibration variant or threshold expansion was run.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = RETHINK / "caches/me_tst"
    engine.SCALES = REFERENCE_SCALES
    engine.REFERENCE_SCALES = REFERENCE_SCALES
    configs = exp3a_configs()

    sanity_rows, distribution_rows, ranking_preserved = calibration_sanity_audit()
    sanity_fields = [
        "setting", "reference_scale", "radius", "num_videos",
        "videos_with_0_candidates", "videos_with_1_candidate",
        "videos_with_2_candidates", "videos_with_3plus_candidates",
        "candidate_count_mean", "candidate_count_median", "candidate_count_q1",
        "candidate_count_q3", "candidate_count_max", "G_unique_value_ratio",
        "L_unique_value_ratio", "G_tie_rate", "L_tie_rate",
        "Spearman(G,G_rank)", "Spearman(L,L_rank)",
        "G_spearman_defined_sets", "L_spearman_defined_sets",
        "ordering_inversions",
    ]
    write_csv(OUT / "calibration_sanity.csv", sanity_rows, sanity_fields)
    write_csv(
        OUT / "score_distribution_before_after.csv", distribution_rows,
        list(distribution_rows[0]),
    )
    if not ranking_preserved:
        print("RANK_IMPLEMENTATION_ERROR")
        raise SystemExit(2)

    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }
    parent_main = {
        row["setting"]: row for row in load_csv(PARENT / "results.csv")
    }
    result_rows = []
    mechanism_rows = []
    selection_rows = []
    fraction_rows = []
    bootstrap_rows = []
    audits = {}
    for backbone, dataset, setting in GROUPS:
        run = run_group(backbone, dataset, setting, configs)
        exp2c_counts = run["counts"]["exp2c"]
        exp3a_counts = run["counts"]["exp3a"]
        baseline_counts = parent.baseline_subject_counts(backbone, dataset)
        baseline_f1 = float(baseline_main[backbone, dataset]["GL_Skill_F1"])
        exp2c_f1 = metric_f1(exp2c_counts)
        exp3a_f1 = metric_f1(exp3a_counts)
        archived_exp2c_f1 = float(parent_main[setting]["exp2c_f1"])
        if not np.isclose(exp2c_f1, archived_exp2c_f1, atol=0, rtol=0):
            raise AssertionError(f"EXP-2C aggregate replay mismatch for {setting}")
        vs_baseline = paired_bootstrap(exp3a_counts, baseline_counts)
        vs_exp2c = paired_bootstrap(exp3a_counts, exp2c_counts)
        result_rows.append({
            "setting": setting,
            "baseline_f1": baseline_f1,
            "exp2c_f1": exp2c_f1,
            "exp3a_f1": exp3a_f1,
            "delta_3a_vs_baseline": exp3a_f1 - baseline_f1,
            "ci_low_3a_vs_baseline": vs_baseline["ci95_low"],
            "ci_high_3a_vs_baseline": vs_baseline["ci95_high"],
            "delta_3a_vs_2c": exp3a_f1 - exp2c_f1,
            "ci_low_3a_vs_2c": vs_exp2c["ci95_low"],
            "ci_high_3a_vs_2c": vs_exp2c["ci95_high"],
        })
        baseline_total = baseline_counts.sum(axis=0)
        exp2c_total = exp2c_counts.sum(axis=0)
        exp3a_total = exp3a_counts.sum(axis=0)
        mechanism_rows.append({
            "setting": setting,
            "baseline_tp": int(baseline_total[0]),
            "baseline_fp": int(baseline_total[1]),
            "baseline_fn": int(baseline_total[2]),
            "exp2c_tp": int(exp2c_total[0]),
            "exp2c_fp": int(exp2c_total[1]),
            "exp2c_fn": int(exp2c_total[2]),
            "exp3a_tp": int(exp3a_total[0]),
            "exp3a_fp": int(exp3a_total[1]),
            "exp3a_fn": int(exp3a_total[2]),
            "delta_tp_3a_vs_2c": int(exp3a_total[0] - exp2c_total[0]),
            "delta_fp_3a_vs_2c": int(exp3a_total[1] - exp2c_total[1]),
            "delta_fn_3a_vs_2c": int(exp3a_total[2] - exp2c_total[2]),
        })
        selection_rows.extend(run["rows"])
        fraction_rows.append(fraction_summary(setting, run["fractions"]))
        for comparison, interval, delta in (
            ("EXP3A-minus-baseline", vs_baseline, exp3a_f1 - baseline_f1),
            ("EXP3A-minus-EXP2C", vs_exp2c, exp3a_f1 - exp2c_f1),
        ):
            bootstrap_rows.append({
                "setting": setting, "comparison": comparison, "delta_f1": delta,
                **interval,
            })
        audits[setting] = {
            key: value for key, value in run.items()
            if key not in ("subjects", "counts", "rows", "choices", "fractions")
        }

    result_fields = [
        "setting", "baseline_f1", "exp2c_f1", "exp3a_f1",
        "delta_3a_vs_baseline", "ci_low_3a_vs_baseline", "ci_high_3a_vs_baseline",
        "delta_3a_vs_2c", "ci_low_3a_vs_2c", "ci_high_3a_vs_2c",
    ]
    write_csv(OUT / "results.csv", result_rows, result_fields)
    write_csv(OUT / "mechanism_counts.csv", mechanism_rows, list(mechanism_rows[0]))
    selection_fields = [
        "setting", "subject", "exp2c_a0", "exp2c_r", "exp2c_tau",
        "exp3a_a0", "exp3a_r", "exp3a_tau", "exp2c_inner_f1", "exp3a_inner_f1",
        "exp2c_outer_tp", "exp2c_outer_fp", "exp2c_outer_fn",
        "exp3a_outer_tp", "exp3a_outer_fp", "exp3a_outer_fn",
    ]
    write_csv(OUT / "selected_config_per_subject.csv", selection_rows, selection_fields)
    threshold_rows = threshold_distribution(selection_rows)
    write_csv(
        OUT / "selected_threshold_distribution.csv", threshold_rows,
        ["setting", "method", "threshold", "fold_count", "fold_fraction"],
    )
    write_csv(
        OUT / "selected_candidate_fraction.csv", fraction_rows,
        list(fraction_rows[0]),
    )
    write_csv(
        OUT / "bootstrap.csv", bootstrap_rows,
        ["setting", "comparison", "delta_f1", "ci95_low", "ci95_high",
         "positive_resample_fraction", "resamples", "seed"],
    )

    gate, recommendation, checks = decide_gate(result_rows, mechanism_rows)
    potential_top_fraction = any(
        row["potential_top_fraction_behavior"] == "YES" for row in fraction_rows
    )
    protocol = {
        "experiment": "GLSD controlled exploration EXP-3A",
        "name": "Rank-Calibrated Structural Evidence",
        "direct_parent": "EXP-2C Reference-Scale / Evidence-Scale Decoupling",
        "only_change": "per-video separate average mid-rank percentile calibration of G and L",
        "formula": {
            "rank": "R(x_i) = average 1-based rank with exact-score ties",
            "percentile": "P(x_i) = (R(x_i)-0.5)/n",
            "G_rank": "P(G)", "L_rank": "P(L)",
            "fusion": "S_rank=(G_rank+L_rank)/2",
            "n_equals_1": "G_rank=L_rank=S_rank=0.5",
            "n_equals_0": "empty candidate set; original empty-prediction behavior",
        },
        "leakage_control": "calibration uses only candidate scores from the current video/configuration context; no GT or cross-video statistic",
        "reference_scale_candidates": list(REFERENCE_SCALES),
        "evidence_scale_map": {
            f"{reference:g}": list(parent.EVIDENCE_SCALE_MAP[reference])
            for reference in REFERENCE_SCALES
        },
        "local_aggregation": "median; missing scale is zero",
        "local_radii": list(engine.LOCAL_RADII),
        "thresholds": list(engine.THRESHOLDS),
        "configuration_budget": len(configs),
        "configuration_budget_formula": "7 reference scales x 3 radii x 10 thresholds = 210",
        "outer": "LOSO", "inner": "LOSO excluding outer test and inner validation subjects",
        "selection_tiebreak": ["higher F1", "higher precision", "fewer FP", "lower config order"],
        "bootstrap": {"unit": "subject", "paired": True, "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED},
        "sanity_audit": {
            "timing": "completed before nested LOSO evaluation",
            "unit": "setting x reference scale x radius; each row contains all videos at their outer-fold k",
            "unique_value_ratio": "mean per nonempty video of unique(score)/candidate_count",
            "tie_rate": "mean per nonempty video of 1-unique(score)/candidate_count",
            "spearman": "mean across nonconstant video score vectors; singleton/constant vectors excluded and counted",
        },
        "top_fraction_diagnostic_rule": {
            "flag_if": "IQR <= 0.05 and at least 90% of nonempty-video fractions lie within 0.10 of the median",
            "selection_effect": "none; diagnostic only",
        },
        "boostingvrme_sammlv_fp_not_obviously_inflated": "EXP-3A FP <= 1.05 * EXP-2C FP",
        "gate": gate,
        "gate_checks": checks,
        "canonical_glsd_modified": False,
        "automatic_followup": False,
        "parent_exact_replay": "PASS",
        "candidate_ranking_preserved": ranking_preserved,
        "potential_top_fraction_behavior": potential_top_fraction,
        "source_sha256": {
            "run_exp3a.py": digest(Path(__file__)),
            "run_exp2c.py": digest(PARENT / "run_exp2c.py"),
            "unified_persistence.py": digest(SIGNED / "unified_persistence.py"),
            "equiscale_fair_validation.py": digest(SIGNED / "equiscale_fair_validation.py"),
        },
        "audits": audits,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "EXP3A_ANALYSIS.md").write_text(
        make_analysis(
            result_rows, mechanism_rows, sanity_rows, fraction_rows, selection_rows,
            gate, recommendation, checks, ranking_preserved,
        ),
        encoding="utf-8",
    )

    means = {
        key: float(np.mean([row[key] for row in result_rows]))
        for key in ("baseline_f1", "exp2c_f1", "exp3a_f1")
    }
    significant_regression = any(row["ci_high_3a_vs_2c"] < 0 for row in result_rows)
    display_gate = (
        "EXP3A_STRONG_CANDIDATE_FOR_VALIDATION"
        if gate == "EXP3A_STRONG_CANDIDATE" else gate
    )
    print("================================")
    print("EXP-3A FINAL VERDICT")
    print("================================")
    print(f"Gate:\n{display_gate}")
    print(f"Mean F1 baseline:\n{fmt(means['baseline_f1'])}")
    print(f"Mean F1 EXP-2C:\n{fmt(means['exp2c_f1'])}")
    print(f"Mean F1 EXP-3A:\n{fmt(means['exp3a_f1'])}")
    for row in result_rows:
        print(
            f"{row['setting']}:\nF1={fmt(row['exp3a_f1'])}; "
            f"delta_vs_2C={fmt(row['delta_3a_vs_2c'])}; "
            f"CI=[{fmt(row['ci_low_3a_vs_2c'])}, {fmt(row['ci_high_3a_vs_2c'])}]"
        )
    print(f"CAS(ME)3 gains preserved:\n{'YES' if checks['cas_gains_preserved'] else 'NO'}")
    print(f"BoostingVRME/SAMMLV repaired:\n{'YES' if checks['boostingvrme_sammlv_repaired'] else 'NO'}")
    print(f"Any significant regression vs EXP-2C:\n{'YES' if significant_regression else 'NO'}")
    print(f"Candidate ranking preserved:\n{'YES' if ranking_preserved else 'NO'}")
    print(f"Potential top-fraction behavior:\n{'YES' if potential_top_fraction else 'NO'}")
    print("Canonical GLSD modified:\nNO")
    print("Continue calibration search automatically:\nNO")
    print(f"Recommended next step:\n{recommendation}")


if __name__ == "__main__":
    main()
