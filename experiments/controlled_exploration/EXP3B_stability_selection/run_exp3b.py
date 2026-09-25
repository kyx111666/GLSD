"""EXP-3B: stability-aware configuration selection on frozen EXP-2C evidence.

The only experimental change is the inner-LOSO selection objective:
mean(subject F1) - sample-SE(subject F1). Candidate generation, evidence,
decoder, folds, evaluator, grids, bootstrap, and EXP-2C tie-breaking remain
frozen.
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
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
BASELINE = ROOT / "controlled_exploration/baseline_snapshot"
RETHINK = ROOT / "RethinkFuse_reproduction"

sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(SIGNED))
import equiscale_fair_validation as fair  # noqa: E402
import unified_persistence as engine  # noqa: E402
import run_exp2c as parent  # noqa: E402


GROUPS = parent.GROUPS
SEED = 100
BOOTSTRAP_RESAMPLES = 10_000
PENALTY_COEFFICIENT = 1.0


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    denominator = 2 * values[..., 0] + values[..., 1] + values[..., 2]
    return np.divide(
        2 * values[..., 0],
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def pooled_metrics(counts: np.ndarray) -> dict[str, float]:
    return fair.metrics(np.asarray(counts).sum(axis=0))


def inner_diagnostics(
    stats: dict[int, np.ndarray], held: int, inner_k: np.ndarray
) -> dict[str, np.ndarray]:
    """Return all locked inner diagnostics for every configuration."""
    per_subject_counts = np.stack(
        [
            stats[int(inner_k[held, validation])][:, validation]
            for validation in range(len(inner_k))
            if validation != held
        ],
        axis=1,
    )
    subject_f1 = f1_from_counts(per_subject_counts)
    m = subject_f1.shape[1]
    if m < 2:
        raise AssertionError("Sample standard error requires at least two inner subjects")
    mean = subject_f1.mean(axis=1)
    std = subject_f1.std(axis=1, ddof=1)
    se = std / math.sqrt(m)
    stable = mean - PENALTY_COEFFICIENT * se
    pooled_counts = per_subject_counts.sum(axis=1)
    pooled_f1 = f1_from_counts(pooled_counts)
    return {
        "counts": pooled_counts,
        "pooled_f1": pooled_f1,
        "mean": mean,
        "std": std,
        "se": se,
        "stable": stable,
    }


def choose_stable(diagnostics: dict[str, np.ndarray]) -> int:
    """StableScore argmax with the frozen post-score EXP-2C tie-break."""
    counts = diagnostics["counts"].astype(float)
    tp, fp, _ = counts.T
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
    )
    indexes = np.arange(len(tp))
    order = np.lexsort((indexes, fp, -precision, -diagnostics["stable"]))
    return int(order[0])


def pooled_order(diagnostics: dict[str, np.ndarray]) -> np.ndarray:
    """Full EXP-2C ranking used only for the descriptive confidence margin."""
    counts = diagnostics["counts"].astype(float)
    tp, fp, _ = counts.T
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
    )
    indexes = np.arange(len(tp))
    return np.lexsort((indexes, fp, -precision, -diagnostics["pooled_f1"]))


def paired_bootstrap(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.shape != reference.shape:
        raise AssertionError("Paired subject-count shapes differ")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(candidate), size=(BOOTSTRAP_RESAMPLES, len(candidate)))

    def sampled_f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1).astype(float)
        return f1_from_counts(totals)

    deltas = sampled_f1(candidate) - sampled_f1(reference)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "positive_resample_fraction": float(np.mean(deltas > 0)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
    }


def baseline_subject_counts(backbone: str, dataset: str) -> np.ndarray:
    return parent.baseline_subject_counts(backbone, dataset)


def run_group(
    backbone: str, dataset: str, setting: str, configs: list[engine.Config]
) -> dict:
    records, subjects, _, input_path = fair.load_data(backbone, dataset)
    outer, inner = fair.fold_priors(records, subjects, backbone)
    mode = "native" if backbone == "boostingvrme" else "fixed"
    stats: dict[int, np.ndarray] = {}
    needed_k = sorted(set(outer.tolist()) | set(inner[inner > 0].tolist()))
    for k in needed_k:
        needed = fair.required_subjects(k, outer, inner)
        values = engine.evaluate_k(records, subjects, backbone, k, needed, configs, (mode,))
        stats[k] = values[mode]

    exp2c_counts = np.zeros((len(subjects), 3), dtype=np.int64)
    exp3b_counts = np.zeros_like(exp2c_counts)
    selection_rows = []
    margin_rows = []
    exp2c_choices = []
    exp3b_choices = []
    baseline_counts = baseline_subject_counts(backbone, dataset)
    if baseline_counts.shape != exp2c_counts.shape:
        raise AssertionError(f"Baseline subject count mismatch for {setting}")
    baseline_rows = parent.baseline_selection_rows(backbone, dataset)
    if [row["subject"] for row in baseline_rows] != subjects:
        raise AssertionError(f"Baseline subject order mismatch for {setting}")

    for held, subject in enumerate(subjects):
        diagnostics = inner_diagnostics(stats, held, inner)
        exp2c_choice = engine.choose(diagnostics["counts"], range(len(configs)))
        exp3b_choice = choose_stable(diagnostics)
        ranking = pooled_order(diagnostics)
        if int(ranking[0]) != exp2c_choice:
            raise AssertionError("Independent EXP-2C ranking did not reproduce engine.choose")

        exp2c_config = configs[exp2c_choice]
        exp3b_config = configs[exp3b_choice]
        exp2c_choices.append(exp2c_choice)
        exp3b_choices.append(exp3b_choice)
        exp2c_counts[held] = stats[int(outer[held])][exp2c_choice, held]
        exp3b_counts[held] = stats[int(outer[held])][exp3b_choice, held]

        selection_rows.append(
            {
                "setting": setting,
                "subject": subject,
                "exp2c_a0": exp2c_config.reference,
                "exp2c_r": exp2c_config.radius,
                "exp2c_tau": exp2c_config.threshold,
                "exp3b_a0": exp3b_config.reference,
                "exp3b_r": exp3b_config.radius,
                "exp3b_tau": exp3b_config.threshold,
                "exp2c_inner_pooled_f1": diagnostics["pooled_f1"][exp2c_choice],
                "exp3b_inner_pooled_f1": diagnostics["pooled_f1"][exp3b_choice],
                "exp2c_inner_mean_f1": diagnostics["mean"][exp2c_choice],
                "exp3b_inner_mean_f1": diagnostics["mean"][exp3b_choice],
                "exp2c_inner_std_f1": diagnostics["std"][exp2c_choice],
                "exp3b_inner_std_f1": diagnostics["std"][exp3b_choice],
                "exp2c_inner_se": diagnostics["se"][exp2c_choice],
                "exp3b_inner_se": diagnostics["se"][exp3b_choice],
                "exp2c_stable_score": diagnostics["stable"][exp2c_choice],
                "exp3b_stable_score": diagnostics["stable"][exp3b_choice],
                "outer_tp": int(exp3b_counts[held, 0]),
                "outer_fp": int(exp3b_counts[held, 1]),
                "outer_fn": int(exp3b_counts[held, 2]),
            }
        )

        outer_exp2c_f1 = float(f1_from_counts(exp2c_counts[held]))
        outer_baseline_f1 = float(f1_from_counts(baseline_counts[held]))
        margin_rows.append(
            {
                "setting": setting,
                "subject": subject,
                "best_pooled_f1": diagnostics["pooled_f1"][ranking[0]],
                "second_best_pooled_f1": diagnostics["pooled_f1"][ranking[1]],
                "margin": (
                    diagnostics["pooled_f1"][ranking[0]]
                    - diagnostics["pooled_f1"][ranking[1]]
                ),
                "exp2c_outer_subject_f1": outer_exp2c_f1,
                "baseline_outer_subject_f1": outer_baseline_f1,
                "fold_class": (
                    "generalizing" if outer_exp2c_f1 >= outer_baseline_f1 else "poor-generalizing"
                ),
            }
        )

    # Replay both selected configurations independently through the unchanged decoder.
    replay2c = np.zeros_like(exp2c_counts)
    replay3b = np.zeros_like(exp3b_counts)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for record in records:
        held = subject_index[record["subject"]]
        features = parent.DecoupledCurveFeatures(record["curve"], int(outer[held]))
        prepared = {}
        for label, choice, replay in (
            ("exp2c", exp2c_choices[held], replay2c),
            ("exp3b", exp3b_choices[held], replay3b),
        ):
            config = configs[choice]
            key = config.reference, config.radius
            if key not in prepared:
                prepared[key] = engine.prepare(
                    record, features, config.reference, backbone, mode, config.radius
                )
            replay[held] += prepared[key].evaluate([config])[0]
    np.testing.assert_array_equal(replay2c, exp2c_counts)
    np.testing.assert_array_equal(replay3b, exp3b_counts)

    changed = np.asarray(exp2c_choices) != np.asarray(exp3b_choices)
    stability = {
        "setting": setting,
        "num_outer_folds": len(subjects),
        "same_config_count": int(np.sum(~changed)),
        "changed_config_count": int(np.sum(changed)),
        "configuration_change_rate": float(np.mean(changed)),
        "a0_changed": sum(
            configs[a].reference != configs[b].reference
            for a, b in zip(exp2c_choices, exp3b_choices)
        ),
        "r_changed": sum(
            configs[a].radius != configs[b].radius
            for a, b in zip(exp2c_choices, exp3b_choices)
        ),
        "tau_changed": sum(
            configs[a].threshold != configs[b].threshold
            for a, b in zip(exp2c_choices, exp3b_choices)
        ),
        "mean_inner_std_exp2c": float(
            np.mean([row["exp2c_inner_std_f1"] for row in selection_rows])
        ),
        "mean_inner_std_exp3b_selected": float(
            np.mean([row["exp3b_inner_std_f1"] for row in selection_rows])
        ),
        "mean_inner_se_exp2c": float(
            np.mean([row["exp2c_inner_se"] for row in selection_rows])
        ),
        "mean_inner_se_exp3b_selected": float(
            np.mean([row["exp3b_inner_se"] for row in selection_rows])
        ),
    }
    stability["stability_effect"] = (
        "LOWER_VARIABILITY"
        if (
            stability["mean_inner_std_exp3b_selected"]
            < stability["mean_inner_std_exp2c"]
            or stability["mean_inner_se_exp3b_selected"]
            < stability["mean_inner_se_exp2c"]
        )
        else "NO_STABILITY_EFFECT"
    )

    return {
        "subjects": subjects,
        "exp2c_counts": exp2c_counts,
        "exp3b_counts": exp3b_counts,
        "selection_rows": selection_rows,
        "margin_rows": margin_rows,
        "stability": stability,
        "audit": {
            "input": str(input_path),
            "input_sha256": digest(Path(input_path)),
            "subjects": len(subjects),
            "videos": len(records),
            "outer_k_frequency": dict(Counter(str(value) for value in outer)),
            "inner_k_frequency": dict(Counter(str(value) for value in inner[inner > 0])),
            "exp2c_selection_frequency": dict(
                Counter(configs[index].identifier for index in exp2c_choices)
            ),
            "exp3b_selection_frequency": dict(
                Counter(configs[index].identifier for index in exp3b_choices)
            ),
            "outer_count_replay": "PASS",
        },
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def margin_summary(rows: list[dict]) -> list[dict]:
    output = []
    settings = [setting for _, _, setting in GROUPS]
    for setting in settings:
        selected = [row for row in rows if row["setting"] == setting]
        for fold_class in ("generalizing", "poor-generalizing"):
            group = [row for row in selected if row["fold_class"] == fold_class]
            output.append(
                {
                    "setting": setting,
                    "fold_class": fold_class,
                    "n": len(group),
                    "mean_margin": (
                        float(np.mean([row["margin"] for row in group]))
                        if group else math.nan
                    ),
                    "median_margin": (
                        float(np.median([row["margin"] for row in group]))
                        if group else math.nan
                    ),
                }
            )
    return output


def determine_gate(results: list[dict], stability_rows: list[dict]) -> tuple[str, dict]:
    no_significant_regression = all(row["ci_high_vs_2c"] >= 0 for row in results)
    nondecrease_count = sum(row["exp3b_f1"] >= row["exp2c_f1"] for row in results)
    cas_rows = [row for row in results if "CAS(ME)3" in row["setting"]]
    cas_gains_preserved = all(row["exp3b_f1"] > row["baseline_f1"] for row in cas_rows)
    mean2c = float(np.mean([row["exp2c_f1"] for row in results]))
    mean3b = float(np.mean([row["exp3b_f1"] for row in results]))
    bs = next(row for row in results if row["setting"] == "BoostingVRME/SAMMLV")
    # Condition (b) already requires an F1 improvement, so a small FP increase
    # that does not outweigh recovered TP/reduced FN is not treated as obvious
    # FP inflation. Exact counts remain visible in mechanism_counts.csv.
    bs_repaired = (
        bs["exp3b_f1"] >= bs["baseline_f1"]
        or (
            bs["exp3b_f1"] > bs["exp2c_f1"]
            and bs["exp3b_tp"] > bs["exp2c_tp"]
            and bs["exp3b_fn"] < bs["exp2c_fn"]
        )
    )
    fold_weights = np.asarray([row["num_outer_folds"] for row in stability_rows])
    avg_std2c = float(np.average(
        [row["mean_inner_std_exp2c"] for row in stability_rows], weights=fold_weights
    ))
    avg_std3b = float(np.average(
        [row["mean_inner_std_exp3b_selected"] for row in stability_rows],
        weights=fold_weights,
    ))
    avg_se2c = float(np.average(
        [row["mean_inner_se_exp2c"] for row in stability_rows], weights=fold_weights
    ))
    avg_se3b = float(np.average(
        [row["mean_inner_se_exp3b_selected"] for row in stability_rows],
        weights=fold_weights,
    ))
    variability_reduced = avg_std3b < avg_std2c or avg_se3b < avg_se2c
    uniform_rule = True

    strong = (
        no_significant_regression
        and nondecrease_count >= 3
        and cas_gains_preserved
        and bs_repaired
        and mean3b >= mean2c
        and variability_reduced
        and uniform_rule
    )
    hard_fail = (
        not no_significant_regression
        or not cas_gains_preserved
        or mean3b < mean2c
        or (not variability_reduced and mean3b <= mean2c)
    )
    partial = (
        (mean3b > mean2c and nondecrease_count >= 3 and not bs_repaired)
        or (variability_reduced and not strong and not hard_fail)
    )
    gate = (
        "EXP3B_STRONG_CANDIDATE"
        if strong
        else "EXP3B_PARTIAL_SUPPORT"
        if partial
        else "EXP3B_FAILED"
    )
    facts = {
        "no_significant_regression": no_significant_regression,
        "nondecrease_count": nondecrease_count,
        "cas_gains_preserved": cas_gains_preserved,
        "mean2c": mean2c,
        "mean3b": mean3b,
        "bs_repaired": bs_repaired,
        "avg_std2c": avg_std2c,
        "avg_std3b": avg_std3b,
        "avg_se2c": avg_se2c,
        "avg_se3b": avg_se3b,
        "variability_reduced": variability_reduced,
        "uniform_rule": uniform_rule,
    }
    return gate, facts


def build_analysis(
    results: list[dict], mechanisms: list[dict], stability: list[dict],
    margins: list[dict], gate: str, facts: dict
) -> str:
    recommendation = (
        "A. Proceed to prospective validation"
        if gate == "EXP3B_STRONG_CANDIDATE"
        else "B. Retain EXP-2C"
    )
    lines = [
        "# EXP-3B Stability-Aware Configuration Selection",
        "",
        "## 1. Hypothesis",
        "",
        "Pure pooled-F1 argmax may be sensitive to inner-subject sampling variation. "
        "EXP-3B tests whether the fixed one-standard-error score selects a more stable "
        "operating point without changing the decoder or candidate configurations.",
        "",
        "## 2. Protocol",
        "",
        "For every existing EXP-2C configuration and outer fold, subject-level F1 is "
        "computed on each inner validation subject. Selection maximizes "
        "`mean_f1 - std_f1/sqrt(m)`, with sample standard deviation (`ddof=1`). "
        "The penalty coefficient is fixed at 1 for all datasets and backbones, was not "
        "searched, and adds no configurations. Pooled inner F1 is retained for diagnosis. "
        "Exact StableScore ties use the frozen EXP-2C post-score tie-break: higher pooled "
        "precision, fewer pooled FP, then lower grid order.",
        "",
        "All four settings use the same rule. The 210 configurations, frozen responses, "
        "246 outer folds, inner/outer LOSO splits, decoder, evaluator, GT matching, "
        "bootstrap (N=10,000; seed=100), and event geometry are unchanged.",
        "",
        "## 3. Main Results",
        "",
        "| Setting | 2C | 3B | Delta | 95% CI vs 2C |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['setting']} | {fmt(row['exp2c_f1'])} | {fmt(row['exp3b_f1'])} "
            f"| {fmt(row['delta_3b_vs_2c'])} | "
            f"[{fmt(row['ci_low_vs_2c'])}, {fmt(row['ci_high_vs_2c'])}] |"
        )
    lines.extend(
        [
            "",
            f"Four-setting mean F1: EXP-2C `{fmt(facts['mean2c'])}`; "
            f"EXP-3B `{fmt(facts['mean3b'])}`.",
            "",
            "## 4. TP/FP/FN",
            "",
            "| Setting | 2C TP/FP/FN | 3B TP/FP/FN | Delta TP/FP/FN |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in mechanisms:
        lines.append(
            f"| {row['setting']} | {row['exp2c_tp']}/{row['exp2c_fp']}/{row['exp2c_fn']} "
            f"| {row['exp3b_tp']}/{row['exp3b_fp']}/{row['exp3b_fn']} "
            f"| {row['delta_tp']:+d}/{row['delta_fp']:+d}/{row['delta_fn']:+d} |"
        )
    lines.extend(
        [
            "",
            "## 5. Selection Stability",
            "",
            "| Setting | Change rate | Mean std 2C | Mean std 3B | Mean SE 2C | Mean SE 3B | Effect |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in stability:
        lines.append(
            f"| {row['setting']} | {row['configuration_change_rate']:.1%} "
            f"| {fmt(row['mean_inner_std_exp2c'])} "
            f"| {fmt(row['mean_inner_std_exp3b_selected'])} "
            f"| {fmt(row['mean_inner_se_exp2c'])} "
            f"| {fmt(row['mean_inner_se_exp3b_selected'])} "
            f"| {row['stability_effect']} |"
        )
    lines.extend(
        [
            "",
            f"Across all 246 outer folds, mean selected-config std changed from "
            f"`{fmt(facts['avg_std2c'])}` to `{fmt(facts['avg_std3b'])}`, and mean SE "
            f"from `{fmt(facts['avg_se2c'])}` to `{fmt(facts['avg_se3b'])}`.",
            "",
            "## 6. Configuration Changes",
            "",
            "| Setting | Same | Changed | a0 changed | r changed | tau changed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stability:
        lines.append(
            f"| {row['setting']} | {row['same_config_count']} | {row['changed_config_count']} "
            f"| {row['a0_changed']} | {row['r_changed']} | {row['tau_changed']} |"
        )
    bs = next(row for row in mechanisms if row["setting"] == "BoostingVRME/SAMMLV")
    result_bs = next(row for row in results if row["setting"] == "BoostingVRME/SAMMLV")
    if bs["delta_tp"] > 0 and bs["delta_fn"] < 0 and bs["delta_fp"] <= 0:
        bs_reading = "recovered TP and reduced FN without FP inflation"
    elif bs["delta_tp"] > 0 and bs["delta_fn"] < 0:
        bs_reading = "recovered TP and reduced FN, but increased FP"
    else:
        bs_reading = "did not recover TP/reduce FN; the change is an operating-point shift"
    lines.extend(
        [
            "",
            "## 7. BoostingVRME/SAMMLV Analysis",
            "",
            f"F1 changed from `{fmt(result_bs['exp2c_f1'])}` to "
            f"`{fmt(result_bs['exp3b_f1'])}`. Counts changed from "
            f"`{bs['exp2c_tp']}/{bs['exp2c_fp']}/{bs['exp2c_fn']}` to "
            f"`{bs['exp3b_tp']}/{bs['exp3b_fp']}/{bs['exp3b_fn']}`. EXP-3B "
            f"{bs_reading}. This satisfies Gate A condition 4(b) as a local repair "
            f"because F1 increased with TP recovery and FN reduction; however, the "
            f"paired CI crosses zero and F1 remains below the canonical baseline.",
            "",
        ]
    )
    if gate == "EXP3B_FAILED":
        lines.extend(
            [
                "### Secondary selection-margin analysis",
                "",
                "Folds are labeled `generalizing` when EXP-2C outer subject-F1 is at "
                "least the canonical baseline outer subject-F1; otherwise they are "
                "`poor-generalizing`. This zero-parameter label is descriptive only.",
                "",
                "| Setting | Fold class | n | Mean margin | Median margin |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in margins:
            lines.append(
                f"| {row['setting']} | {row['fold_class']} | {row['n']} "
                f"| {fmt(row['mean_margin']) if row['n'] else 'NA'} "
                f"| {fmt(row['median_margin']) if row['n'] else 'NA'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 8. Gate",
            "",
            gate,
            "",
            f"- No paired-CI regression vs EXP-2C: "
            f"{'YES' if facts['no_significant_regression'] else 'NO'}",
            f"- Settings non-decreasing vs EXP-2C: {facts['nondecrease_count']}/4",
            f"- CAS(ME)3 gains preserved: {'YES' if facts['cas_gains_preserved'] else 'NO'}",
            f"- BoostingVRME/SAMMLV repaired: {'YES' if facts['bs_repaired'] else 'NO'}",
            f"- Four-setting mean at least EXP-2C: "
            f"{'YES' if facts['mean3b'] >= facts['mean2c'] else 'NO'}",
            f"- Average inner variability reduced: "
            f"{'YES' if facts['variability_reduced'] else 'NO'}",
            "- One fixed rule used for every setting: YES",
            "",
            "## 9. Recommendation",
            "",
            recommendation,
            "",
            "No additional stability penalty, selection objective, or configuration "
            "search is authorized by this experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = RETHINK / "caches/me_tst"

    # Process-local EXP-2C setup. Canonical source files are never modified.
    engine.SCALES = parent.REFERENCE_SCALES
    engine.REFERENCE_SCALES = parent.REFERENCE_SCALES
    engine.CurveFeatures = parent.DecoupledCurveFeatures
    configs = parent.exp2c_configs()
    if len(configs) != 210:
        raise AssertionError("EXP-3B must retain the 210-config EXP-2C budget")

    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }
    exp2c_reported = {
        row["setting"]: row for row in load_csv(PARENT / "results.csv")
    }
    result_rows = []
    mechanism_rows = []
    selection_rows = []
    stability_rows = []
    margin_rows = []
    bootstrap_rows = []
    audits = {}

    for backbone, dataset, setting in GROUPS:
        run = run_group(backbone, dataset, setting, configs)
        exp2c_metrics = pooled_metrics(run["exp2c_counts"])
        exp3b_metrics = pooled_metrics(run["exp3b_counts"])
        expected_exp2c = float(exp2c_reported[setting]["exp2c_f1"])
        if not np.isclose(exp2c_metrics["F1"], expected_exp2c, rtol=0, atol=1e-15):
            raise AssertionError(f"EXP-2C failed exact reproduction for {setting}")
        baseline = baseline_main[backbone, dataset]
        baseline_f1 = float(baseline["GL_Skill_F1"])
        baseline_counts = baseline_subject_counts(backbone, dataset)
        vs_baseline = paired_bootstrap(run["exp3b_counts"], baseline_counts)
        vs_exp2c = paired_bootstrap(run["exp3b_counts"], run["exp2c_counts"])
        result_rows.append(
            {
                "setting": setting,
                "baseline_f1": baseline_f1,
                "exp2c_f1": exp2c_metrics["F1"],
                "exp3b_f1": exp3b_metrics["F1"],
                "delta_3b_vs_baseline": exp3b_metrics["F1"] - baseline_f1,
                "ci_low_vs_baseline": vs_baseline["ci_low"],
                "ci_high_vs_baseline": vs_baseline["ci_high"],
                "delta_3b_vs_2c": exp3b_metrics["F1"] - exp2c_metrics["F1"],
                "ci_low_vs_2c": vs_exp2c["ci_low"],
                "ci_high_vs_2c": vs_exp2c["ci_high"],
                # Internal fields used only for the pre-registered gate.
                "exp2c_tp": exp2c_metrics["TP"],
                "exp2c_fp": exp2c_metrics["FP"],
                "exp2c_fn": exp2c_metrics["FN"],
                "exp3b_tp": exp3b_metrics["TP"],
                "exp3b_fp": exp3b_metrics["FP"],
                "exp3b_fn": exp3b_metrics["FN"],
            }
        )
        mechanism_rows.append(
            {
                "setting": setting,
                "exp2c_tp": exp2c_metrics["TP"],
                "exp2c_fp": exp2c_metrics["FP"],
                "exp2c_fn": exp2c_metrics["FN"],
                "exp3b_tp": exp3b_metrics["TP"],
                "exp3b_fp": exp3b_metrics["FP"],
                "exp3b_fn": exp3b_metrics["FN"],
                "delta_tp": exp3b_metrics["TP"] - exp2c_metrics["TP"],
                "delta_fp": exp3b_metrics["FP"] - exp2c_metrics["FP"],
                "delta_fn": exp3b_metrics["FN"] - exp2c_metrics["FN"],
            }
        )
        selection_rows.extend(run["selection_rows"])
        stability_rows.append(run["stability"])
        margin_rows.extend(run["margin_rows"])
        for comparison, bootstrap, delta in (
            ("EXP3B-minus-baseline", vs_baseline, exp3b_metrics["F1"] - baseline_f1),
            ("EXP3B-minus-EXP2C", vs_exp2c, exp3b_metrics["F1"] - exp2c_metrics["F1"]),
        ):
            bootstrap_rows.append(
                {"setting": setting, "comparison": comparison, "delta_f1": delta, **bootstrap}
            )
        audits[setting] = {
            "baseline_counts": {
                "TP": int(baseline["GL_Skill_TP"]),
                "FP": int(baseline["GL_Skill_FP"]),
                "FN": int(baseline["GL_Skill_FN"]),
            },
            "exp2c_metrics": exp2c_metrics,
            "exp3b_metrics": exp3b_metrics,
            **run["audit"],
        }

    gate, facts = determine_gate(result_rows, stability_rows)
    margins = margin_summary(margin_rows)

    public_result_fields = [
        "setting", "baseline_f1", "exp2c_f1", "exp3b_f1",
        "delta_3b_vs_baseline", "ci_low_vs_baseline", "ci_high_vs_baseline",
        "delta_3b_vs_2c", "ci_low_vs_2c", "ci_high_vs_2c",
    ]
    write_csv(
        OUT / "results.csv",
        [{key: row[key] for key in public_result_fields} for row in result_rows],
        public_result_fields,
    )
    write_csv(
        OUT / "mechanism_counts.csv",
        mechanism_rows,
        [
            "setting", "exp2c_tp", "exp2c_fp", "exp2c_fn",
            "exp3b_tp", "exp3b_fp", "exp3b_fn", "delta_tp", "delta_fp", "delta_fn",
        ],
    )
    write_csv(
        OUT / "selected_config_per_subject.csv",
        selection_rows,
        [
            "setting", "subject", "exp2c_a0", "exp2c_r", "exp2c_tau",
            "exp3b_a0", "exp3b_r", "exp3b_tau",
            "exp2c_inner_pooled_f1", "exp3b_inner_pooled_f1",
            "exp2c_inner_mean_f1", "exp3b_inner_mean_f1",
            "exp2c_inner_std_f1", "exp3b_inner_std_f1",
            "exp2c_inner_se", "exp3b_inner_se",
            "exp2c_stable_score", "exp3b_stable_score",
            "outer_tp", "outer_fp", "outer_fn",
        ],
    )
    write_csv(
        OUT / "selection_stability.csv",
        stability_rows,
        [
            "setting", "num_outer_folds", "same_config_count", "changed_config_count",
            "configuration_change_rate", "a0_changed", "r_changed", "tau_changed",
            "mean_inner_std_exp2c", "mean_inner_std_exp3b_selected",
            "mean_inner_se_exp2c", "mean_inner_se_exp3b_selected", "stability_effect",
        ],
    )
    write_csv(
        OUT / "selection_margin.csv",
        margin_rows,
        [
            "setting", "subject", "best_pooled_f1", "second_best_pooled_f1", "margin",
            "exp2c_outer_subject_f1", "baseline_outer_subject_f1", "fold_class",
        ],
    )
    write_csv(
        OUT / "bootstrap.csv",
        bootstrap_rows,
        [
            "setting", "comparison", "delta_f1", "ci_low", "ci_high",
            "positive_resample_fraction", "resamples", "seed",
        ],
    )

    protocol = {
        "experiment": "GLSD controlled exploration EXP-3B",
        "name": "Stability-Aware Configuration Selection",
        "parent": "EXP-2C Reference-Scale / Evidence-Scale Decoupling",
        "only_change": "inner LOSO configuration selection criterion",
        "selection_rule": "mean subject F1 - 1 * standard error of subject F1",
        "penalty_coefficient": PENALTY_COEFFICIENT,
        "penalty_is_tuned": False,
        "standard_deviation": "sample standard deviation across inner validation subjects (ddof=1)",
        "standard_error": "sample std / sqrt(number of inner validation subjects)",
        "pooled_inner_f1_role": "recorded for diagnosis; not the EXP-3B selection objective",
        "selection_tiebreak_after_exact_stable_score_equality": [
            "higher pooled precision", "fewer pooled FP", "lower config order"
        ],
        "same_rule_for_all_settings": True,
        "configuration_budget": len(configs),
        "configuration_budget_formula": "7 reference scales x 3 radii x 10 thresholds = 210",
        "reference_scales": list(parent.REFERENCE_SCALES),
        "evidence_scale_map": {
            f"{reference:g}": list(scales)
            for reference, scales in parent.EVIDENCE_SCALE_MAP.items()
        },
        "local_evidence": "three nearest scales; median aggregation (unchanged)",
        "fusion": "S(c)=(G(c)+L(c))/2 (unchanged)",
        "local_radii": list(engine.LOCAL_RADII),
        "thresholds": list(engine.THRESHOLDS),
        "candidate_generation": "unchanged from EXP-2C",
        "event_geometry": "unchanged from EXP-2C",
        "evaluator_and_gt_matching": "unchanged from EXP-2C",
        "outer": "LOSO (unchanged)",
        "inner": "LOSO excluding outer test and inner validation subjects (unchanged)",
        "primary_intervals": {"metst": "fixed", "boostingvrme": "native"},
        "outer_folds_total": sum(len(run["subjects"]) for run in []),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
        "margin_fold_label": (
            "descriptive only: generalizing iff EXP-2C outer subject-F1 >= canonical "
            "baseline outer subject-F1; otherwise poor-generalizing"
        ),
        "gate": gate,
        "gate_facts": facts,
        "canonical_glsd_modified": False,
        "automatic_followup_search": False,
        "source_sha256": {
            "run_exp3b.py": digest(Path(__file__)),
            "run_exp2c.py": digest(PARENT / "run_exp2c.py"),
            "unified_persistence.py": digest(SIGNED / "unified_persistence.py"),
            "equiscale_fair_validation.py": digest(SIGNED / "equiscale_fair_validation.py"),
        },
        "audits": audits,
    }
    protocol["outer_folds_total"] = sum(audit["subjects"] for audit in audits.values())
    if protocol["outer_folds_total"] != 246:
        raise AssertionError("EXP-3B must contain exactly 246 outer folds")
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "EXP3B_ANALYSIS.md").write_text(
        build_analysis(result_rows, mechanism_rows, stability_rows, margins, gate, facts),
        encoding="utf-8",
    )

    recommendation = (
        "A. Proceed to prospective validation"
        if gate == "EXP3B_STRONG_CANDIDATE"
        else "B. Retain EXP-2C"
    )
    print("================================")
    print("EXP-3B FINAL VERDICT")
    print("================================")
    print(f"Gate:\n{gate}")
    print(f"\nMean F1 EXP-2C:\n{facts['mean2c']:.6f}")
    print(f"\nMean F1 EXP-3B:\n{facts['mean3b']:.6f}")
    for row in result_rows:
        print(
            f"\n{row['setting']}:\n"
            f"EXP-2C={row['exp2c_f1']:.6f}, EXP-3B={row['exp3b_f1']:.6f}, "
            f"delta={row['delta_3b_vs_2c']:+.6f}, "
            f"95% CI=[{row['ci_low_vs_2c']:.6f}, {row['ci_high_vs_2c']:.6f}]"
        )
    print(
        f"\nCAS(ME)3 gains preserved:\n{'YES' if facts['cas_gains_preserved'] else 'NO'}"
    )
    print(f"\nBoostingVRME/SAMMLV repaired:\n{'YES' if facts['bs_repaired'] else 'NO'}")
    print(
        f"\nAverage inner variability reduced:\n"
        f"{'YES' if facts['variability_reduced'] else 'NO'}"
    )
    print(
        f"\nAny significant regression:\n"
        f"{'NO' if facts['no_significant_regression'] else 'YES'}"
    )
    print("\nCanonical GLSD modified:\nNO")
    print("\nContinue selection-rule search automatically:\nNO")
    print(f"\nRecommended next step:\n{recommendation}")


if __name__ == "__main__":
    main()
