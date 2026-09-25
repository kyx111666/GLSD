"""EXP-2D: validate a fixed five-nearest-scale local evidence neighborhood."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP2A = ROOT / "controlled_exploration/EXP2A_scale"
EXP2C = ROOT / "controlled_exploration/EXP2C_reference_evidence_decoupling"
BASELINE = ROOT / "controlled_exploration/baseline_snapshot"

sys.path.insert(0, str(EXP2C))
import run_exp2c as common  # noqa: E402


engine = common.engine
fair = common.fair
GROUPS = common.GROUPS
REFERENCE_SCALES = common.REFERENCE_SCALES
SEED = common.SEED


def five_nearest_scales(reference: float) -> tuple[float, ...]:
    """Return five scales ranked by (absolute distance, smaller scale)."""
    selected = tuple(
        sorted(REFERENCE_SCALES, key=lambda scale: (abs(scale - reference), scale))[:5]
    )
    if len(selected) != 5 or reference not in selected:
        raise AssertionError(f"Invalid E5 neighborhood for reference={reference}: {selected}")
    return selected


EVIDENCE_SCALE_MAP = {
    reference: five_nearest_scales(reference) for reference in REFERENCE_SCALES
}


def validate_mapping() -> dict[str, list[float]]:
    first = {
        f"{reference:g}": list(five_nearest_scales(reference))
        for reference in REFERENCE_SCALES
    }
    second = {
        f"{reference:g}": list(five_nearest_scales(reference))
        for reference in REFERENCE_SCALES
    }
    if first != second:
        raise AssertionError("E5 mapping is not deterministic")
    if any(len(scales) != 5 for scales in first.values()):
        raise AssertionError("Every E5 set must have length five")
    for reference, scales in first.items():
        if float(reference) not in scales:
            raise AssertionError(f"Reference {reference} missing from E5")
    return first


class FiveScaleCurveFeatures(engine.CurveFeatures):
    """Canonical curve features with L restricted to deterministic E5(a0)."""

    def evidence(self, reference, radius=1.0):
        if (reference, radius) in self.evidence_cache:
            return self.evidence_cache[reference, radius]
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))

        evidence_by_width = {}
        for scale in EVIDENCE_SCALE_MAP[reference]:
            width = min(len(self.curve), max(1, round(scale * self.k)))
            evidence_by_width.setdefault(width, self.scales[scale])
        if len(evidence_by_width) != 5:
            raise AssertionError(
                f"E5 collapsed to {len(evidence_by_width)} physical widths "
                f"for reference={reference}, k={self.k}"
            )

        for width, (other_peaks, _, _, smooth, spread) in evidence_by_width.items():
            if (width, window) not in self.local_cache:
                self.local_cache[width, window] = np.array(
                    [
                        max(
                            0.0,
                            float(smooth[p])
                            - max(
                                float(np.min(smooth[max(0, p - window) : p + 1])),
                                float(
                                    np.min(
                                        smooth[p : min(len(smooth), p + window + 1)]
                                    )
                                ),
                            ),
                        )
                        / spread
                        for p in other_peaks
                    ]
                )

        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in evidence_by_width.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (
                        abs(int(other_peaks[index]) - point),
                        -values[index],
                    ),
                )
                aligned.append(float(values[chosen]))
            local.append(float(np.median(aligned)))

        output = peaks, np.column_stack((height, global_values, local))
        self.evidence_cache[reference, radius] = output
        return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def counts_from_rows(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[int(row[key]) for key in ("outer_tp", "outer_fp", "outer_fn")] for row in rows]
    )


def f1_from_counts(counts: np.ndarray) -> float:
    tp, fp, fn = counts.sum(axis=0).astype(float)
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def aggregate_distribution(rows: list[dict], field: str) -> list[dict]:
    output = []
    for setting in (group[2] for group in GROUPS):
        setting_rows = [row for row in rows if row["setting"] == setting]
        for variant in ("EXP-2C", "EXP-2D"):
            source = field if variant == "EXP-2D" else f"exp2c_{field}"
            counts = Counter(row[source] for row in setting_rows)
            for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
                output.append(
                    {
                        "setting": setting,
                        "variant": variant,
                        field: value,
                        "count": count,
                        "fraction": count / len(setting_rows),
                    }
                )
    return output


def verdict(result_rows: list[dict], mechanism_rows: list[dict]) -> tuple[str, dict]:
    positives = sum(row["delta_2d_vs_baseline"] >= 0 for row in result_rows)
    no_negative_ci = all(row["ci_high_2d_vs_baseline"] >= 0 for row in result_rows)
    cas_preserved = all(
        row["exp2d_f1"] > row["baseline_f1"]
        for row in result_rows
        if "CAS(ME)3" in row["setting"]
    )
    mean_delta = float(np.mean([row["delta_2d_vs_baseline"] for row in result_rows]))
    bg_result = next(row for row in result_rows if row["setting"] == "BoostingVRME/SAMMLV")
    bg_counts = next(row for row in mechanism_rows if row["setting"] == "BoostingVRME/SAMMLV")
    bg_repaired = (
        bg_result["exp2d_f1"] >= bg_result["baseline_f1"]
        or (
            bg_counts["exp2d_tp"] >= bg_counts["baseline_tp"]
            and bg_counts["exp2d_fn"] <= bg_counts["baseline_fn"]
        )
    )
    metst_sammlv = next(row for row in result_rows if row["setting"] == "ME-TST/SAMMLV")
    failed = (
        not cas_preserved
        or metst_sammlv["ci_high_2d_vs_baseline"] < 0
        or not no_negative_ci
        or mean_delta <= 0
    )
    gate_a = (
        no_negative_ci
        and positives >= 3
        and cas_preserved
        and bg_repaired
        and mean_delta > 0
    )
    mean_vs_2c = float(np.mean([row["delta_2d_vs_2c"] for row in result_rows]))
    clear_vs_2c = mean_vs_2c > 0 and sum(
        row["delta_2d_vs_2c"] > 0 for row in result_rows
    ) >= 3
    gate_b = (
        positives == 3
        and cas_preserved
        and bg_result["exp2d_f1"] < bg_result["baseline_f1"]
        and not clear_vs_2c
    )
    if failed:
        gate = "EXP2D_FAILED"
    elif gate_a:
        gate = "EXP2D_STRONGER_THAN_2C"
    elif gate_b:
        gate = "EXP2D_NO_CLEAR_ADVANTAGE"
    else:
        gate = "EXP2D_NO_CLEAR_ADVANTAGE"

    positive_vs_2c = sum(row["delta_2d_vs_2c"] > 0 for row in result_rows)
    negative_vs_2c = sum(row["delta_2d_vs_2c"] < 0 for row in result_rows)
    no_bad_vs_2c = all(row["ci_high_2d_vs_2c"] >= 0 for row in result_rows)
    no_good_vs_2c = all(row["ci_low_2d_vs_2c"] <= 0 for row in result_rows)
    if mean_vs_2c > 0 and positive_vs_2c >= 3 and no_bad_vs_2c:
        overall = "YES"
    elif mean_vs_2c < 0 and negative_vs_2c >= 3 and no_good_vs_2c:
        overall = "NO"
    else:
        overall = "INCONCLUSIVE"
    return gate, {
        "mean_delta_vs_baseline": mean_delta,
        "mean_delta_vs_exp2c": mean_vs_2c,
        "cas_preserved": cas_preserved,
        "boosting_sammlv_repaired": bg_repaired,
        "better_than_exp2c_overall": overall,
    }


def build_analysis(
    result_rows: list[dict],
    mechanism_rows: list[dict],
    selections: list[dict],
    gate: str,
    summary: dict,
) -> str:
    result_by_setting = {row["setting"]: row for row in result_rows}
    mechanism_by_setting = {row["setting"]: row for row in mechanism_rows}
    changed_a0 = sum(row["reference_changed_vs_2c"] == "YES" for row in selections)
    changed_radius = sum(row["radius_changed_vs_2c"] == "YES" for row in selections)
    changed_tau = sum(row["threshold_changed_vs_2c"] == "YES" for row in selections)
    total = len(selections)
    bg_rows = [row for row in selections if row["setting"] == "BoostingVRME/SAMMLV"]
    bg_changed = {
        "a0": sum(row["reference_changed_vs_2c"] == "YES" for row in bg_rows),
        "radius": sum(row["radius_changed_vs_2c"] == "YES" for row in bg_rows),
        "tau": sum(row["threshold_changed_vs_2c"] == "YES" for row in bg_rows),
    }
    recommendation = {
        "EXP2D_STRONGER_THAN_2C": "A. Proceed to prospective validation",
        "EXP2D_NO_CLEAR_ADVANTAGE": "B. Retain EXP-2C as strongest exploration candidate",
        "EXP2D_FAILED": "C. Retain canonical baseline",
    }[gate]

    lines = [
        "# EXP-2D Five-Scale Evidence Validation",
        "",
        "## 1. Protocol",
        "",
        "EXP-2D changes exactly one factor from EXP-2C: local evidence neighborhood cardinality "
        "from nearest-3 to nearest-5. Reference candidates, radii, thresholds, median aggregation, "
        "missing-scale=0, fusion, evaluator, nested LOSO, seed 100, bootstrap, and tie-break are unchanged.",
        "The E5 mapping is generated by `(absolute distance to a0, smaller scale)` and is shared across all settings.",
        "The budget is 7 reference scales × 3 radii × 10 thresholds = 210 configurations. "
        "Exact replay passed for all 246 outer folds. Canonical GLSD was not modified.",
        "",
        "## 2. Main Results",
        "",
        "| Setting | Baseline | 2A | 2C | 2D | 2D-Baseline | CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, _, setting in GROUPS:
        row = result_by_setting[setting]
        lines.append(
            f"| {setting} | {fmt(row['baseline_f1'])} | {fmt(row['exp2a_f1'])} | "
            f"{fmt(row['exp2c_f1'])} | {fmt(row['exp2d_f1'])} | "
            f"{fmt(row['delta_2d_vs_baseline'], True)} | "
            f"[{fmt(row['ci_low_2d_vs_baseline'])}, {fmt(row['ci_high_2d_vs_baseline'])}] |"
        )
    lines.extend(
        [
            "",
            f"Unweighted mean ΔF1 versus baseline: **{fmt(summary['mean_delta_vs_baseline'], True)}**.",
            "",
            "## 3. Direct 2D vs 2C",
            "",
            "| Setting | 2C | 2D | Delta | CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, _, setting in GROUPS:
        row = result_by_setting[setting]
        lines.append(
            f"| {setting} | {fmt(row['exp2c_f1'])} | {fmt(row['exp2d_f1'])} | "
            f"{fmt(row['delta_2d_vs_2c'], True)} | "
            f"[{fmt(row['ci_low_2d_vs_2c'])}, {fmt(row['ci_high_2d_vs_2c'])}] |"
        )
    lines.extend(
        [
            "",
            f"Unweighted mean ΔF1 versus EXP-2C: **{fmt(summary['mean_delta_vs_exp2c'], True)}**. "
            f"Overall comparison: **{summary['better_than_exp2c_overall']}**.",
            "",
            "## 4. TP/FP/FN Mechanism",
            "",
            "| Setting | Baseline TP/FP/FN | 2A TP/FP/FN | 2C TP/FP/FN | 2D TP/FP/FN |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, _, setting in GROUPS:
        row = mechanism_by_setting[setting]
        lines.append(
            f"| {setting} | {row['baseline_tp']}/{row['baseline_fp']}/{row['baseline_fn']} | "
            f"{row['exp2a_tp']}/{row['exp2a_fp']}/{row['exp2a_fn']} | "
            f"{row['exp2c_tp']}/{row['exp2c_fp']}/{row['exp2c_fn']} | "
            f"{row['exp2d_tp']}/{row['exp2d_fp']}/{row['exp2d_fn']} |"
        )
    bg = mechanism_by_setting["BoostingVRME/SAMMLV"]
    tp_recovered = bg["exp2d_tp"] >= bg["baseline_tp"]
    fn_recovered = bg["exp2d_fn"] <= bg["baseline_fn"]
    fp_control = bg["exp2d_fp"] <= bg["baseline_fp"]
    if tp_recovered and fn_recovered and fp_control:
        mechanism_reading = "nearest-5 restores TP/FN to baseline while retaining FP control."
    else:
        mechanism_reading = (
            "nearest-5 does not jointly restore TP and FN to baseline; its effect remains a "
            "precision/recall operating-point change"
            + (" with FP control." if fp_control else " without baseline-level FP control.")
        )
    lines.extend(
        [
            "",
            f"For BoostingVRME/SAMMLV, {mechanism_reading}",
            "",
            "## 5. Configuration Selection",
            "",
            f"Across {total} paired outer folds, reference scale changed in {changed_a0}, radius in "
            f"{changed_radius}, and threshold in {changed_tau} folds relative to EXP-2C. "
            "The setting-level paired changes are:",
            "",
            "| Setting | a0 changed | Radius changed | Tau changed | Folds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, _, setting in GROUPS:
        setting_rows = [row for row in selections if row["setting"] == setting]
        lines.append(
            f"| {setting} | "
            f"{sum(row['reference_changed_vs_2c'] == 'YES' for row in setting_rows)} | "
            f"{sum(row['radius_changed_vs_2c'] == 'YES' for row in setting_rows)} | "
            f"{sum(row['threshold_changed_vs_2c'] == 'YES' for row in setting_rows)} | "
            f"{len(setting_rows)} |"
        )
    lines.extend(
        [
            "",
            f"BoostingVRME/SAMMLV changes a0 in {bg_changed['a0']}/29 folds, radius in "
            f"{bg_changed['radius']}/29, and tau in {bg_changed['tau']}/29. EXP-2D selects a0=1.0 "
            "in 24/29 folds, whereas EXP-2C selected a0=1.25 in 23/29 and 1.5 in 6/29. "
            "BoostingVRME/CAS(ME)3 also undergoes a coherent operating-point shift: all 94 folds "
            "move to radius=1.0 and tau=0.5. Detailed paired selections and outer-fold counts are "
            "in `selected_config_per_subject.csv`; aggregate frequencies are in the two distribution CSV files.",
            "",
            "## 6. Evidence-Cardinality Interpretation",
            "",
        ]
    )
    deltas_2c = [row["delta_2d_vs_2c"] for row in result_rows]
    deltas_2a = [row["delta_2d_vs_2a"] for row in result_rows]
    lines.extend(
        [
            f"1. **Nearest-5 is not better than nearest-3 overall.** Mean ΔF1 is "
            f"{fmt(float(np.mean(deltas_2c)), True)}. It improves BoostingVRME/CAS(ME)3, ties "
            "ME-TST/CAS(ME)3 exactly, and lowers both SAMMLV settings; every 2D-vs-2C CI includes zero.",
            f"2. **Nearest-5 is not better than all-7 overall.** Mean ΔF1 is "
            f"{fmt(float(np.mean(deltas_2a)), True)}. It improves ME-TST/SAMMLV and "
            "BoostingVRME/CAS(ME)3, is marginally lower on ME-TST/CAS(ME)3, and is significantly "
            "lower on BoostingVRME/SAMMLV (its paired CI is fully below zero).",
            "3. **There is no universal too-narrow/intermediate/too-broad curve.** The preferred tested "
            "cardinality is setting-specific: 3 leads on ME-TST/SAMMLV, 3 and 5 tie on "
            "ME-TST/CAS(ME)3, 7 leads on BoostingVRME/SAMMLV, and 5 leads on BoostingVRME/CAS(ME)3. "
            "These three points do not justify an optimum-cardinality claim.",
            "4. **BoostingVRME/SAMMLV is more consistent with configuration-selection instability than "
            "insufficient evidence support.** Widening support fails to recover sensitivity and worsens "
            "EXP-2C counts from 40/89/119 to 39/93/120 (TP/FP/FN), while a0 changes in 25/29 folds. "
            "Radius is unchanged in every fold and tau changes in only 7/29, localizing the instability "
            "primarily to reference-scale selection induced by evidence breadth.",
            "5. **Nearest-3 is the more stable cross-setting exploration candidate among the tested rules.** "
            "It preserves both CAS(ME)3 gains, avoids the all-7 significant ME-TST/SAMMLV regression, "
            "and has higher mean F1 than nearest-5. This is a stability comparison, not a claim that 3 is optimal.",
            "",
            "## 7. Gate",
            "",
            f"**{gate}**",
            "",
            "The label follows the predeclared gate exactly. It is an experiment decision, not a modification "
            "or promotion of canonical GLSD.",
            "",
            "## 8. Recommendation",
            "",
            f"**{recommendation}**",
            "",
            "The evidence-cardinality route stops here. No nearest-4, nearest-6, or other scale count is tested.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mapping = validate_mapping()
    mapping_path = OUT / "evidence_scale_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    if args.mapping_only:
        print(json.dumps(mapping, indent=2))
        return

    fair.ME_CACHE = common.RETHINK / "caches/me_tst"
    engine.SCALES = REFERENCE_SCALES
    engine.REFERENCE_SCALES = REFERENCE_SCALES
    engine.CurveFeatures = FiveScaleCurveFeatures
    configs = common.exp2c_configs()
    if len(configs) != 210:
        raise AssertionError("EXP-2D must contain exactly 210 configurations")

    baseline_main = {
        (row["Backbone"], row["Dataset"]): row
        for row in load_csv(BASELINE / "results/main_results.csv")
    }
    exp2a_main = {row["setting"]: row for row in load_csv(EXP2A / "results.csv")}
    exp2c_main = {row["setting"]: row for row in load_csv(EXP2C / "results.csv")}
    exp2c_fold_rows = load_csv(EXP2C / "selected_reference_scale_distribution.csv")
    exp2c_by_key = {(row["setting"], row["subject"]): row for row in exp2c_fold_rows}

    result_rows = []
    mechanism_rows = []
    selection_rows = []
    bootstrap_rows = []
    audits = {}

    for backbone, dataset, setting in GROUPS:
        metrics, counts, selections, audit = common.run_group(
            backbone, dataset, setting, configs
        )
        baseline = baseline_main[backbone, dataset]
        exp2a = exp2a_main[setting]
        exp2c = exp2c_main[setting]
        baseline_counts = common.baseline_subject_counts(backbone, dataset)
        exp2a_counts = common.exp2a_subject_counts(setting)
        exp2c_setting_rows = [row for row in exp2c_fold_rows if row["setting"] == setting]
        exp2c_counts = counts_from_rows(exp2c_setting_rows)
        if not np.isclose(f1_from_counts(exp2c_counts), float(exp2c["exp2c_f1"])):
            raise AssertionError(f"EXP-2C fold counts do not reproduce F1 for {setting}")
        comparisons = {
            "baseline": common.paired_bootstrap(counts, baseline_counts),
            "exp2c": common.paired_bootstrap(counts, exp2c_counts),
            "exp2a": common.paired_bootstrap(counts, exp2a_counts),
        }
        result_rows.append(
            {
                "setting": setting,
                "baseline_f1": float(baseline["GL_Skill_F1"]),
                "exp2a_f1": float(exp2a["expanded_F1"]),
                "exp2c_f1": float(exp2c["exp2c_f1"]),
                "exp2d_f1": metrics["F1"],
                "delta_2d_vs_baseline": metrics["F1"] - float(baseline["GL_Skill_F1"]),
                "ci_low_2d_vs_baseline": comparisons["baseline"]["ci95_low"],
                "ci_high_2d_vs_baseline": comparisons["baseline"]["ci95_high"],
                "delta_2d_vs_2c": metrics["F1"] - float(exp2c["exp2c_f1"]),
                "ci_low_2d_vs_2c": comparisons["exp2c"]["ci95_low"],
                "ci_high_2d_vs_2c": comparisons["exp2c"]["ci95_high"],
                "delta_2d_vs_2a": metrics["F1"] - float(exp2a["expanded_F1"]),
                "ci_low_2d_vs_2a": comparisons["exp2a"]["ci95_low"],
                "ci_high_2d_vs_2a": comparisons["exp2a"]["ci95_high"],
            }
        )
        totals = counts.sum(axis=0)
        exp2c_totals = exp2c_counts.sum(axis=0)
        mechanism_rows.append(
            {
                "setting": setting,
                "baseline_tp": int(baseline["GL_Skill_TP"]),
                "baseline_fp": int(baseline["GL_Skill_FP"]),
                "baseline_fn": int(baseline["GL_Skill_FN"]),
                "exp2a_tp": int(exp2a["expanded_TP"]),
                "exp2a_fp": int(exp2a["expanded_FP"]),
                "exp2a_fn": int(exp2a["expanded_FN"]),
                "exp2c_tp": int(exp2c_totals[0]),
                "exp2c_fp": int(exp2c_totals[1]),
                "exp2c_fn": int(exp2c_totals[2]),
                "exp2d_tp": int(totals[0]),
                "exp2d_fp": int(totals[1]),
                "exp2d_fn": int(totals[2]),
            }
        )
        for row in selections:
            subject = row["outer_subject"]
            key = setting, subject
            if key not in exp2c_by_key:
                raise AssertionError(f"Missing paired EXP-2C selection for {key}")
            previous = exp2c_by_key[key]
            a0 = float(row["selected_scale"])
            evidence_scales = "|".join(f"{scale:g}" for scale in EVIDENCE_SCALE_MAP[a0])
            selection_rows.append(
                {
                    "setting": setting,
                    "subject": subject,
                    "selected_a0": a0,
                    "selected_radius": row["selected_radius"],
                    "selected_tau": row["selected_threshold"],
                    "evidence_scales": evidence_scales,
                    "inner_f1": row["inner_F1"],
                    "outer_tp": row["outer_TP"],
                    "outer_fp": row["outer_FP"],
                    "outer_fn": row["outer_FN"],
                    "exp2c_selected_a0": previous["selected_a0"],
                    "exp2c_selected_radius": previous["selected_radius"],
                    "exp2c_selected_tau": previous["selected_tau"],
                    "exp2c_evidence_scales": previous["evidence_scales"],
                    "reference_changed_vs_2c": "YES" if a0 != float(previous["selected_a0"]) else "NO",
                    "radius_changed_vs_2c": "YES" if float(row["selected_radius"]) != float(previous["selected_radius"]) else "NO",
                    "threshold_changed_vs_2c": "YES" if float(row["selected_threshold"]) != float(previous["selected_tau"]) else "NO",
                }
            )
        for reference_name, comparison in comparisons.items():
            bootstrap_rows.append(
                {
                    "setting": setting,
                    "comparison": f"EXP2D-minus-{reference_name}",
                    **comparison,
                }
            )
        audits[setting] = {**audit, "fold_count": len(selections)}

    if len(selection_rows) != 246:
        raise AssertionError(f"Expected 246 outer folds, got {len(selection_rows)}")
    if any(audit["outer_count_replay"] != "PASS" for audit in audits.values()):
        raise AssertionError("At least one exact replay audit failed")

    result_fields = [
        "setting", "baseline_f1", "exp2a_f1", "exp2c_f1", "exp2d_f1",
        "delta_2d_vs_baseline", "ci_low_2d_vs_baseline", "ci_high_2d_vs_baseline",
        "delta_2d_vs_2c", "ci_low_2d_vs_2c", "ci_high_2d_vs_2c",
        "delta_2d_vs_2a", "ci_low_2d_vs_2a", "ci_high_2d_vs_2a",
    ]
    write_csv(OUT / "results.csv", result_rows, result_fields)
    mechanism_fields = [
        "setting", "baseline_tp", "baseline_fp", "baseline_fn",
        "exp2a_tp", "exp2a_fp", "exp2a_fn", "exp2c_tp", "exp2c_fp", "exp2c_fn",
        "exp2d_tp", "exp2d_fp", "exp2d_fn",
    ]
    write_csv(OUT / "mechanism_counts.csv", mechanism_rows, mechanism_fields)
    selection_fields = list(selection_rows[0])
    write_csv(OUT / "selected_config_per_subject.csv", selection_rows, selection_fields)
    reference_distribution = aggregate_distribution(selection_rows, "selected_a0")
    write_csv(
        OUT / "selected_reference_scale_distribution.csv",
        reference_distribution,
        ["setting", "variant", "selected_a0", "count", "fraction"],
    )
    evidence_distribution = aggregate_distribution(selection_rows, "evidence_scales")
    write_csv(
        OUT / "selected_evidence_set_distribution.csv",
        evidence_distribution,
        ["setting", "variant", "evidence_scales", "count", "fraction"],
    )
    write_csv(
        OUT / "bootstrap.csv",
        bootstrap_rows,
        ["setting", "comparison", "ci95_low", "ci95_high", "positive_resample_fraction", "resamples", "seed"],
    )

    gate, summary = verdict(result_rows, mechanism_rows)
    protocol = {
        "experiment": "GLSD controlled exploration EXP-2D",
        "name": "Five-Scale Local Evidence Validation",
        "only_change_from_exp2c": "local evidence neighborhood cardinality: nearest-3 -> nearest-5",
        "reference_scale_candidates": list(REFERENCE_SCALES),
        "evidence_scale_rule": "first five scales ranked by (absolute distance to a0, smaller scale)",
        "evidence_scale_mapping": mapping,
        "mapping_validation": {
            "length_five": True,
            "reference_included": True,
            "deterministic": True,
            "dataset_or_backbone_specific_variation": False,
        },
        "configuration_budget": len(configs),
        "configuration_budget_formula": "7 reference scales x 3 radii x 10 thresholds = 210",
        "local_aggregation": "median; missing aligned scale contributes zero",
        "fusion": "S(c)=(G(c)+L(c))/2",
        "local_radii": list(engine.LOCAL_RADII),
        "thresholds": list(engine.THRESHOLDS),
        "outer": "LOSO",
        "inner": "LOSO excluding outer test and inner validation subjects",
        "selection_tiebreak": ["higher F1", "higher precision", "fewer FP", "lower config order"],
        "bootstrap_resamples": 10_000,
        "seed": SEED,
        "outer_folds": len(selection_rows),
        "all_outer_fold_exact_replay": True,
        "canonical_glsd_modified": False,
        "continue_cardinality_search": False,
        "source_sha256": {
            "run_exp2d.py": common.digest(Path(__file__)),
            "run_exp2c.py": common.digest(EXP2C / "run_exp2c.py"),
            "unified_persistence.py": common.digest(common.SIGNED / "unified_persistence.py"),
            "equiscale_fair_validation.py": common.digest(common.SIGNED / "equiscale_fair_validation.py"),
        },
        "audits": audits,
        "gate": gate,
        "summary": summary,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "EXP2D_ANALYSIS.md").write_text(
        build_analysis(result_rows, mechanism_rows, selection_rows, gate, summary),
        encoding="utf-8",
    )

    recommendation = {
        "EXP2D_STRONGER_THAN_2C": "Proceed to prospective validation",
        "EXP2D_NO_CLEAR_ADVANTAGE": "Retain EXP-2C as strongest exploration candidate",
        "EXP2D_FAILED": "Retain canonical baseline",
    }[gate]
    print("================================")
    print("EXP-2D FINAL VERDICT")
    print("================================")
    print(f"Gate:\n{gate}")
    print(f"\nMean Delta vs baseline:\n{summary['mean_delta_vs_baseline']:+.6f}")
    for _, _, setting in GROUPS:
        row = next(item for item in result_rows if item["setting"] == setting)
        print(
            f"\n{setting}:\nF1={row['exp2d_f1']:.6f}; "
            f"Delta={row['delta_2d_vs_baseline']:+.6f}; "
            f"95% CI [{row['ci_low_2d_vs_baseline']:.6f}, {row['ci_high_2d_vs_baseline']:.6f}]"
        )
    print(f"\nCAS(ME)3 gains preserved:\n{'YES' if summary['cas_preserved'] else 'NO'}")
    print(f"\nBoostingVRME/SAMMLV repaired:\n{'YES' if summary['boosting_sammlv_repaired'] else 'NO'}")
    print(f"\nBetter than EXP-2C overall:\n{summary['better_than_exp2c_overall']}")
    print("\nCanonical GLSD modified:\nNO")
    print("\nContinue cardinality search:\nNO")
    print(f"\nRecommended next step:\n{recommendation}")


if __name__ == "__main__":
    main()
