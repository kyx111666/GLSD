"""EXP-5B: bilateral-baseline asymmetry audit and gated mean-baseline study."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
RETHINK = ROOT / "RethinkFuse_reproduction"
EXP5A_SCRIPT = ROOT / "controlled_exploration/EXP5A_scale_support/run_exp5a.py"

sys.path.insert(0, str(SIGNED))
spec = importlib.util.spec_from_file_location("exp5a_base", EXP5A_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)
engine = base.engine
fair = base.fair

GROUPS = base.GROUPS
BOOTSTRAP_RESAMPLES = 10_000
SEED = 100
NEAR_MARGIN = 0.10
STRONG_CUTOFF = 0.05
EPS = 1e-12


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else "NA"


def numeric_summary(values: list[float]) -> tuple[float | str, float | str]:
    return (
        (float(np.mean(values)), float(np.median(values)))
        if values else ("NA", "NA")
    )


def fmt(value, digits: int = 6) -> str:
    return "NA" if value == "NA" else f"{float(value):.{digits}f}"


class BilateralCurveFeatures(engine.CurveFeatures):
    """Expose both baseline operators while preserving every other operation."""

    def __init__(self, curve, k):
        super().__init__(curve, k)
        self.bilateral_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def _scale_values(self, width: int, window: int) -> dict[str, np.ndarray]:
        key = width, window
        if key in self.bilateral_cache:
            return self.bilateral_cache[key]
        other_peaks, _, _, smooth, spread = self.effective_scales[width]
        centers, lefts, rights, canonical, mean_values = [], [], [], [], []
        for peak in other_peaks:
            center = float(smooth[peak])
            left = float(np.min(smooth[max(0, peak - window):peak + 1]))
            right = float(np.min(smooth[peak:min(len(smooth), peak + window + 1)]))
            conservative = max(left, right)
            arithmetic = (left + right) / 2.0
            centers.append(center)
            lefts.append(left)
            rights.append(right)
            canonical.append(max(0.0, center - conservative) / spread)
            mean_values.append(max(0.0, center - arithmetic) / spread)
        result = {
            "center": np.asarray(centers, dtype=float),
            "left": np.asarray(lefts, dtype=float),
            "right": np.asarray(rights, dtype=float),
            "canonical": np.asarray(canonical, dtype=float),
            "mean": np.asarray(mean_values, dtype=float),
        }
        self.bilateral_cache[key] = result
        return result

    def detailed_evidence(self, reference: float, radius: float, alignment: str):
        if alignment not in {"canonical", "mean"}:
            raise ValueError(alignment)
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))
        widths = list(self.effective_scales)
        reference_width = min(len(self.curve), max(1, round(reference * self.k)))
        if reference_width not in self.effective_scales:
            raise AssertionError("Reference physical width absent")
        reference_position = widths.index(reference_width)
        details = []
        local = []
        for point in peaks:
            canonical_vector, mean_vector, matched_vector, pair_rows = [], [], [], []
            for width, (other_peaks, _, _, _, spread) in self.effective_scales.items():
                arrays = self._scale_values(width, window)
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    canonical_vector.append(0.0)
                    mean_vector.append(0.0)
                    matched_vector.append(False)
                    pair_rows.append(None)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (
                        abs(int(other_peaks[index]) - int(point)),
                        -float(arrays[alignment][index]),
                    ),
                )
                left = float(arrays["left"][chosen])
                right = float(arrays["right"][chosen])
                raw = abs(left - right)
                denom = abs(left) + abs(right)
                relative = raw / denom if denom > 0 else 0.0
                canonical_vector.append(float(arrays["canonical"][chosen]))
                mean_vector.append(float(arrays["mean"][chosen]))
                matched_vector.append(True)
                pair_rows.append({
                    "width": int(width),
                    "matched_peak": int(other_peaks[chosen]),
                    "center": float(arrays["center"][chosen]),
                    "B_L": left,
                    "B_R": right,
                    "spread": float(spread),
                    "raw_asymmetry": raw,
                    "relative_asymmetry": relative,
                    "l_canonical": float(arrays["canonical"][chosen]),
                    "l_mean": float(arrays["mean"][chosen]),
                    "delta_l_scale": float(arrays["mean"][chosen] - arrays["canonical"][chosen]),
                })
            if not matched_vector[reference_position]:
                raise AssertionError("Reference candidate failed to self-match")
            canonical_l = float(np.median(canonical_vector))
            mean_l = float(np.median(mean_vector))
            matched_pairs = [row for row in pair_rows if row is not None]
            candidate_raw = float(np.median([row["raw_asymmetry"] for row in matched_pairs]))
            candidate_relative = float(np.median([row["relative_asymmetry"] for row in matched_pairs]))
            details.append({
                "peak": int(point),
                "K": len(widths),
                "matched_vector": matched_vector,
                "canonical_vector": canonical_vector,
                "mean_vector": mean_vector,
                "pair_rows": pair_rows,
                "candidate_raw_asymmetry": candidate_raw,
                "candidate_relative_asymmetry": candidate_relative,
                "canonical_L": canonical_l,
                "mean_L": mean_l,
            })
            local.append(canonical_l if alignment == "canonical" else mean_l)
        evidence = np.column_stack((height, global_values, local))
        return peaks, evidence, details


class CanonicalBilateralCurveFeatures(BilateralCurveFeatures):
    def evidence(self, reference, radius=1.0):
        key = reference, radius, "canonical"
        if key not in self.evidence_cache:
            peaks, evidence, _ = self.detailed_evidence(reference, radius, "canonical")
            self.evidence_cache[key] = peaks, evidence
        return self.evidence_cache[key]


class MeanBaselineCurveFeatures(BilateralCurveFeatures):
    def evidence(self, reference, radius=1.0):
        key = reference, radius, "mean"
        if key not in self.evidence_cache:
            peaks, evidence, _ = self.detailed_evidence(reference, radius, "mean")
            self.evidence_cache[key] = peaks, evidence
        return self.evidence_cache[key]


def validate_canonical_formula(features: CanonicalBilateralCurveFeatures, reference: float, radius: float) -> None:
    audit_peaks, audit_evidence = features.evidence(reference, radius)
    canonical = engine.CurveFeatures(features.curve, features.k)
    source_peaks, source_evidence = canonical.evidence(reference, radius)
    np.testing.assert_array_equal(audit_peaks, source_peaks)
    np.testing.assert_allclose(audit_evidence, source_evidence, rtol=0.0, atol=1e-15)


def candidate_rows(backbone, dataset, setting, records, subjects, outer, all_configs, choices, mode):
    rows = []
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    for video_index, record in enumerate(records):
        sid = subject_ids[record["subject"]]
        config = all_configs[choices[sid]]
        features = CanonicalBilateralCurveFeatures(record["curve"], int(outer[sid]))
        validate_canonical_formula(features, config.reference, config.radius)
        peaks, evidence, details = features.detailed_evidence(config.reference, config.radius, "canonical")
        decoder = fair.VideoFeatures(record, features.k, backbone if mode == "native" else "metst")
        prepared = engine.Prepared(decoder, peaks, evidence)
        _, selected_details = prepared.evaluate([config], details=True)
        selected_by_peak = {item["peak"]: item for item in selected_details[0]}
        lookup = {int(peak): index for index, peak in enumerate(peaks)}
        for candidate_index, cluster in enumerate(prepared.clusters):
            peak = int(cluster[0])
            detail = details[lookup[peak]]
            global_value = float(prepared.evidence[candidate_index, 1])
            canonical_l = detail["canonical_L"]
            mean_l = detail["mean_L"]
            canonical_s = 0.5 * (global_value + canonical_l)
            mean_s = 0.5 * (global_value + mean_l)
            selected_item = selected_by_peak.get(peak)
            selected = selected_item is not None
            association = (
                "TP" if selected and selected_item["matched_gt"] >= 0
                else "FP" if selected else "not_selected"
            )
            margin = config.threshold - canonical_s
            near = 0 < margin <= NEAR_MARGIN + EPS
            interval = prepared.intervals[candidate_index]
            rows.append({
                "setting": setting,
                "backbone": backbone,
                "dataset": dataset,
                "subject": record["subject"],
                "video": record["video"],
                "video_index": video_index,
                "peak": peak,
                "onset": int(interval[0]),
                "offset": int(interval[1]),
                "K": detail["K"],
                "matched_vector": json.dumps([int(x) for x in detail["matched_vector"]]),
                "canonical_local_vector": json.dumps(detail["canonical_vector"]),
                "mean_local_vector": json.dumps(detail["mean_vector"]),
                "bilateral_pair_details": json.dumps(detail["pair_rows"]),
                "raw_asymmetry": detail["candidate_raw_asymmetry"],
                "relative_asymmetry": detail["candidate_relative_asymmetry"],
                "G": global_value,
                "L_canonical": canonical_l,
                "L_mean_diag": mean_l,
                "S_canonical": canonical_s,
                "S_mean_diag": mean_s,
                "delta_L": mean_l - canonical_l,
                "delta_S": mean_s - canonical_s,
                "strongly_penalized": int(mean_s - canonical_s >= STRONG_CUTOFF - EPS),
                "selected_tau": config.threshold,
                "selected_a0": config.reference,
                "selected_rho": config.radius,
                "selected_prediction": int(selected),
                "selected_association": association,
                "selected_matched_gt": selected_item["matched_gt"] if selected else -1,
                "gt_compatible": int(prepared.best[candidate_index] >= 0),
                "potential_gt_index": int(prepared.best[candidate_index]),
                "margin": margin,
                "near_threshold": int(near),
                "would_cross_threshold": int(near and mean_s + EPS >= config.threshold),
            })
    return rows


def paired_bootstrap(subject_rows: list[dict], tp_field: str, fp_field: str) -> dict:
    pairs = [
        (float(row[tp_field]), float(row[fp_field]))
        for row in subject_rows
        if row[tp_field] != "NA" and row[fp_field] != "NA"
    ]
    if not pairs:
        return {"n_subjects": 0, "mean_difference": "NA", "ci_low": "NA", "ci_high": "NA"}
    values = np.asarray(pairs, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    deltas = (values[draws, 0] - values[draws, 1]).mean(axis=1)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "n_subjects": len(values),
        "mean_difference": float(np.mean(values[:, 0] - values[:, 1])),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def stratum(value: float) -> str:
    if value < 0.1:
        return "[0,0.1)"
    if value < 0.25:
        return "[0.1,0.25)"
    if value < 0.5:
        return "[0.25,0.5)"
    return "[0.5,1.0]"


def aggregate_stage1(setting_rows: list[dict], subjects: list[str]):
    selected = [row for row in setting_rows if row["selected_prediction"]]
    tp = [row for row in selected if row["selected_association"] == "TP"]
    fp = [row for row in selected if row["selected_association"] == "FP"]
    setting = setting_rows[0]["setting"]
    diagnostic = {"setting": setting, "num_tp": len(tp), "num_fp": len(fp)}
    for label, group in (("tp", tp), ("fp", fp)):
        for field in ("raw_asymmetry", "relative_asymmetry", "delta_L", "delta_S"):
            mean, median = numeric_summary([float(row[field]) for row in group])
            diagnostic[f"{label}_mean_{field}"] = mean
            diagnostic[f"{label}_median_{field}"] = median
        strong = sum(int(row["strongly_penalized"]) for row in group)
        diagnostic[f"{label}_strongly_penalized_count"] = strong
        diagnostic[f"{label}_strongly_penalized_prevalence"] = safe_rate(strong, len(group))
    tp_prev = diagnostic["tp_strongly_penalized_prevalence"]
    fp_prev = diagnostic["fp_strongly_penalized_prevalence"]
    diagnostic["strongly_penalized_prevalence_difference"] = (
        tp_prev - fp_prev if tp_prev != "NA" and fp_prev != "NA" else "NA"
    )

    strata = []
    labels = ("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1.0]")
    near = [row for row in setting_rows if row["near_threshold"]]
    for label in labels:
        selected_group = [row for row in selected if stratum(float(row["relative_asymmetry"])) == label]
        near_group = [row for row in near if stratum(float(row["relative_asymmetry"])) == label]
        n_tp = sum(row["selected_association"] == "TP" for row in selected_group)
        n_fp = sum(row["selected_association"] == "FP" for row in selected_group)
        gt = sum(row["gt_compatible"] for row in near_group)
        non_gt = len(near_group) - gt
        strata.append({
            "setting": setting,
            "relative_asymmetry_stratum": label,
            "selected_TP": n_tp,
            "selected_FP": n_fp,
            "selected_precision": safe_rate(n_tp, n_tp + n_fp),
            "near_threshold_GT_compatible": gt,
            "near_threshold_non_GT_compatible": non_gt,
        })

    crossers = [row for row in near if row["would_cross_threshold"]]
    gt_cross = sum(row["gt_compatible"] for row in crossers)
    non_gt_cross = len(crossers) - gt_cross
    crossing = {
        "setting": setting,
        "near_threshold_GT_compatible": sum(row["gt_compatible"] for row in near),
        "near_threshold_non_GT_compatible": sum(not row["gt_compatible"] for row in near),
        "GT_compatible_crossers": gt_cross,
        "non_GT_compatible_crossers": non_gt_cross,
        "all_crossers": len(crossers),
        "potential_rescue_precision": safe_rate(gt_cross, len(crossers)),
    }

    subject_rows = []
    for subject in subjects:
        subject_tp = [row for row in tp if row["subject"] == subject]
        subject_fp = [row for row in fp if row["subject"] == subject]
        subject_rows.append({
            "setting": setting,
            "subject": subject,
            "num_tp": len(subject_tp),
            "num_fp": len(subject_fp),
            "tp_strong_prevalence": safe_rate(sum(row["strongly_penalized"] for row in subject_tp), len(subject_tp)),
            "fp_strong_prevalence": safe_rate(sum(row["strongly_penalized"] for row in subject_fp), len(subject_fp)),
            "tp_mean_delta_S": float(np.mean([row["delta_S"] for row in subject_tp])) if subject_tp else "NA",
            "fp_mean_delta_S": float(np.mean([row["delta_S"] for row in subject_fp])) if subject_fp else "NA",
        })
    strong_bs = paired_bootstrap(subject_rows, "tp_strong_prevalence", "fp_strong_prevalence")
    delta_bs = paired_bootstrap(subject_rows, "tp_mean_delta_S", "fp_mean_delta_S")
    bootstrap_rows = []
    for metric, values in (("strongly_penalized_prevalence_TP_minus_FP", strong_bs), ("mean_delta_S_TP_minus_FP", delta_bs)):
        bootstrap_rows.append({"setting": setting, "metric": metric, **values, "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED})
    return diagnostic, strata, crossing, bootstrap_rows, strong_bs


def diagnostic_gate(diagnostics, crossings, strong_bootstraps, score_rows):
    enrichment_settings = [
        row["setting"] for row in diagnostics
        if row["strongly_penalized_prevalence_difference"] != "NA"
        and row["strongly_penalized_prevalence_difference"] >= 0.05 - EPS
    ]
    gt_crosser_settings = [row["setting"] for row in crossings if row["GT_compatible_crossers"] >= 10]
    precision_settings = [
        row["setting"] for row in crossings
        if row["potential_rescue_precision"] != "NA" and row["potential_rescue_precision"] >= 0.20 - EPS
    ]
    bootstrap_ok = all(
        values["ci_high"] != "NA" and values["ci_high"] > 0
        for values in strong_bootstraps.values()
    )
    universal_bad = all(
        row["potential_rescue_precision"] != "NA"
        and row["potential_rescue_precision"] < 0.10
        and row["non_GT_compatible_crossers"] > row["GT_compatible_crossers"]
        for row in crossings
    )
    changes = any(abs(float(row["delta_L"])) > EPS or abs(float(row["delta_S"])) > EPS for row in score_rows)
    passed = bool(enrichment_settings and gt_crosser_settings and precision_settings and bootstrap_ok and not universal_bad and changes)
    reasons = []
    if not enrichment_settings:
        reasons.append("No setting had TP strongly-penalized prevalence at least 5 percentage points above FP")
    if not gt_crosser_settings:
        reasons.append("No setting had at least 10 near-threshold GT-compatible crossers")
    if not precision_settings:
        reasons.append("No setting reached potential rescue precision 0.20")
    if not bootstrap_ok:
        reasons.append("At least one setting had a strongly-penalized TP-FP bootstrap upper bound <= 0 or unavailable")
    if universal_bad:
        reasons.append("All settings with crossers were overwhelmingly non-GT-compatible (precision < 0.10)")
    if not changes:
        reasons.append("Mean-baseline diagnostic did not change L or S")
    return passed, {
        "enrichment_settings": enrichment_settings,
        "gt_crosser_settings": gt_crosser_settings,
        "precision_settings": precision_settings,
        "bootstrap_direction_ok": bootstrap_ok,
        "universal_non_gt_crosser_failure": universal_bad,
        "diagnostic_changes_scores": changes,
    }, reasons or ["All preregistered Diagnostic Gate B conditions were met"]


def local_contrast_report() -> str:
    return """# Canonical Bilateral Local Contrast Audit

## Source of truth

The audited implementation is `historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/unified_persistence.py`, class `CurveFeatures`. The EXP-5B replay also checks the reconstructed values against that class at absolute tolerance `1e-15` for every evaluated video/configuration.

## Exact semantics for a matched candidate-scale peak

Let the aligned peak index be `p`, the integer local radius be `w=max(1, round(rho*k))`, the smoothed curve at that physical scale be `y`, and `R=max(ptp(y), 1e-12)`.

1. Peak/center value: `C = y[p]`.
2. Left neighborhood: `y[max(0,p-w):p+1]`; it includes the center sample.
3. Right neighborhood: `y[p:min(len(y),p+w+1)]`; it also includes the center sample.
4. Left baseline: `B_L = min(left neighborhood)`.
5. Right baseline: `B_R = min(right neighborhood)`.
6. Canonical bilateral baseline: `B_can = max(B_L,B_R)`.
7. Numerator: `max(0, C-B_can)`.
8. Denominator/range normalization: `R=max(ptp(y),1e-12)`.
9. Clipping: no upper clipping.
10. Positivity constraint: yes, `max(...,0)` around the numerator.
11. Epsilon: `1e-12` only as the minimum range denominator.
12. Range normalization: yes, by the full smoothed-curve range at that scale.
13. Per-scale local evidence:

`l_a(c) = max(0, y_a[p_a] - max(B_L,a, B_R,a)) / max(ptp(y_a), 1e-12)`.

Cross-scale alignment uses tolerance `max(1,round(0.5*k))`; absent non-reference matches contribute exactly zero; the candidate local evidence is the NumPy median across distinct physical widths. The reference candidate self-matches. Final canonical score is `S=(G+L)/2`.

## Applicability

**PASS.** The higher (stricter) side minimum controls the canonical bilateral baseline through `max(B_L,B_R)`, and this operator is directly replaceable without changing the surrounding numerator, positivity constraint, denominator, range normalization, radius, alignment algorithm, missing-zero policy, or median aggregation.
"""


def stage1_report(diagnostics, crossings, bootstrap_rows, gate_pass, checks, reasons) -> str:
    lines = [
        "# EXP-5B Stage 1 Bilateral Asymmetry Audit", "",
        "## Canonical replay", "",
        "All four settings exactly reproduced selected configurations, inner F1 values, full prediction lists, subject TP/FP/FN, and pooled F1: **PASS**.", "",
        "## Diagnostic definition", "",
        "Only `B_can=max(B_L,B_R)` was replaced diagnostically by `B_mean=(B_L+B_R)/2`. Stage 1 did not reselect configurations, change thresholds, or produce alternative predictions. Candidate-level raw and relative asymmetry are the medians across that candidate's canonically aligned matched scale pairs; missing scales are excluded from asymmetry summaries but remain zero in both local-evidence vectors.", "",
        "## Selected TP/FP diagnostic", "",
        "| Setting | TP strong prev | FP strong prev | Difference | TP mean delta S | FP mean delta S |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(f"| {row['setting']} | {fmt(row['tp_strongly_penalized_prevalence'])} | {fmt(row['fp_strongly_penalized_prevalence'])} | {fmt(row['strongly_penalized_prevalence_difference'])} | {fmt(row['tp_mean_delta_S'])} | {fmt(row['fp_mean_delta_S'])} |")
    lines.extend(["", "## Near-threshold crossing", "", "| Setting | GT crossers | non-GT crossers | Potential rescue precision |", "|---|---:|---:|---:|"])
    for row in crossings:
        lines.append(f"| {row['setting']} | {row['GT_compatible_crossers']} | {row['non_GT_compatible_crossers']} | {fmt(row['potential_rescue_precision'])} |")
    lines.extend(["", "## Subject-level paired bootstrap", ""])
    for row in bootstrap_rows:
        if row["metric"] == "strongly_penalized_prevalence_TP_minus_FP":
            lines.append(f"- {row['setting']}: mean TP-FP={fmt(row['mean_difference'])}, 95% CI=[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}], paired subjects={row['n_subjects']}.")
    lines.extend([
        "", "## Gate B", "",
        f"**{'PASS' if gate_pass else 'EXP5B_DIAGNOSTIC_GATE_FAILED'}**", "",
        "; ".join(reasons) + ".", "",
        f"Stage 2 executed: **{'YES' if gate_pass else 'NO'}**.", "",
        "Canonical GLSD-v1 was not modified.",
    ])
    return "\n".join(lines) + "\n"


def stage2_group(run, all_configs):
    stats = {}
    for k in sorted(set(run["outer"].tolist()) | set(run["inner"][run["inner"] > 0].tolist())):
        needed = fair.required_subjects(k, run["outer"], run["inner"])
        stats[k] = base.evaluate_k(
            run["records"], run["subjects"], run["backbone"], k, needed,
            all_configs, run["mode"], MeanBaselineCurveFeatures,
        )
    choices, counts, selection_rows = base.select(stats, run["outer"], run["inner"], run["subjects"], all_configs)
    replay_counts, predictions = base.replay_method(
        run["records"], run["subjects"], run["outer"], run["backbone"], run["mode"],
        all_configs, choices, MeanBaselineCurveFeatures,
    )
    np.testing.assert_array_equal(replay_counts, counts)
    changes = fair.event_changes(predictions, run["predictions"])
    delta = counts.sum(axis=0) - run["counts"].sum(axis=0)
    if changes["net_TP"] != delta[0] or changes["net_FP"] != delta[1]:
        raise AssertionError(f"Stage-2 event accounting mismatch: {run['setting']}")

    selected_rows = []
    subject_ids = {subject: index for index, subject in enumerate(run["subjects"])}
    for record, pred_list in zip(run["records"], predictions):
        sid = subject_ids[record["subject"]]
        config = all_configs[choices[sid]]
        features = MeanBaselineCurveFeatures(record["curve"], int(run["outer"][sid]))
        peaks, evidence, details = features.detailed_evidence(config.reference, config.radius, "mean")
        decoder = fair.VideoFeatures(record, features.k, run["backbone"] if run["mode"] == "native" else "metst")
        prepared = engine.Prepared(decoder, peaks, evidence)
        by_peak = {row["peak"]: row for row in details}
        if {item["peak"] for item in pred_list} != {item["peak"] for item in prepared.evaluate([config], details=True)[1][0]}:
            raise AssertionError("Stage-2 selected prediction replay mismatch")
        for pred in pred_list:
            detail = by_peak[pred["peak"]]
            selected_rows.append({
                "association": "TP" if pred["matched_gt"] >= 0 else "FP",
                "delta_L": detail["mean_L"] - detail["canonical_L"],
                "delta_S": 0.5 * (detail["mean_L"] - detail["canonical_L"]),
                "relative_asymmetry": detail["candidate_relative_asymmetry"],
            })

    summary = {
        "setting": run["setting"], "row_type": "summary", "relative_asymmetry_stratum": "ALL",
        "rescued_gt": changes["GT_rescued"], "removed_fp": changes["FP_removed_exact_interval"],
        "lost_gt": changes["GT_lost"], "new_fp": changes["FP_added_exact_interval"],
    }
    for label in ("TP", "FP"):
        group = [row for row in selected_rows if row["association"] == label]
        summary[f"{label.lower()}_mean_delta_L"] = float(np.mean([row["delta_L"] for row in group])) if group else "NA"
        summary[f"{label.lower()}_median_delta_L"] = float(np.median([row["delta_L"] for row in group])) if group else "NA"
        summary[f"{label.lower()}_mean_delta_S"] = float(np.mean([row["delta_S"] for row in group])) if group else "NA"
    mechanism_rows = [summary]
    for label in ("[0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,1.0]"):
        group = [row for row in selected_rows if stratum(float(row["relative_asymmetry"])) == label]
        tp = sum(row["association"] == "TP" for row in group)
        fp = sum(row["association"] == "FP" for row in group)
        mechanism_rows.append({
            "setting": run["setting"], "row_type": "asymmetry_stratum", "relative_asymmetry_stratum": label,
            "rescued_gt": "", "removed_fp": "", "lost_gt": "", "new_fp": "",
            "tp_mean_delta_L": "", "tp_median_delta_L": "", "tp_mean_delta_S": "",
            "fp_mean_delta_L": "", "fp_median_delta_L": "", "fp_mean_delta_S": "",
            "stage2_TP": tp, "stage2_FP": fp, "stage2_precision": safe_rate(tp, tp + fp),
        })
    for row in selection_rows:
        row["setting"] = run["setting"]
    return choices, counts, selection_rows, predictions, mechanism_rows


def stage2_gate(results, mechanisms):
    canonical_mean = float(np.mean([row["canonical_f1"] for row in results]))
    mean_mean = float(np.mean([row["meanbaseline_f1"] for row in results]))
    no_sig_regression = all(row["ci_high"] >= 0 for row in results)
    noninferior = sum(row["meanbaseline_f1"] >= row["canonical_f1"] for row in results)
    improved = sum(row["meanbaseline_f1"] > row["canonical_f1"] for row in results)
    rescue_settings = [row for row in results if row["delta_tp"] > 0 or row["delta_fn"] < 0]
    rescue_fp_ok = all(row["fp_relative_change"] != "NA" and row["fp_relative_change"] <= 0.10 + EPS for row in rescue_settings)
    severe = any(row["severe_fp_inflation"] for row in results)
    asymmetry_rescue = any(
        row["row_type"] == "summary" and row["rescued_gt"] > 0 and row["tp_mean_delta_S"] != "NA" and row["tp_mean_delta_S"] > 0
        for row in mechanisms
    )
    strong = no_sig_regression and noninferior >= 3 and mean_mean > canonical_mean and bool(rescue_settings) and rescue_fp_ok and not severe
    partial = no_sig_regression and improved in (2, 3) and mean_mean > canonical_mean and asymmetry_rescue and not severe
    gate = "EXP5B_STRONG_CANDIDATE" if strong else "EXP5B_PARTIAL_SUPPORT" if partial else "EXP5B_FAILED"
    return gate, {
        "canonical_mean": canonical_mean, "meanbaseline_mean": mean_mean,
        "no_significant_regression": no_sig_regression, "noninferior_count": noninferior,
        "improved_count": improved, "tp_rescue": bool(rescue_settings),
        "rescue_setting_fp_increase_within_10pct": rescue_fp_ok,
        "severe_fp_inflation": severe, "asymmetry_related_tp_rescue": asymmetry_rescue,
        "significant_improvement": any(row["ci_low"] > 0 for row in results),
    }


def stage2_report(results, gate, checks) -> str:
    lines = ["# EXP-5B-M Mean-Baseline Local Contrast", "", "## Results", "", "| Setting | Canonical F1 | Mean-baseline F1 | Delta | 95% paired CI | FP relative change |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['setting']} | {row['canonical_f1']:.6f} | {row['meanbaseline_f1']:.6f} | {row['delta_f1']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {fmt(row['fp_relative_change'])} |")
    lines.extend(["", "## Gate", "", f"**{gate}**", "", "; ".join(f"{key}={value}" for key, value in checks.items()) + ".", "", "Only `max(B_L,B_R)` was replaced by `(B_L+B_R)/2`; all four settings used the same formula and the 90-configuration budget. Canonical GLSD-v1 was not modified."])
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = RETHINK / "caches/me_tst"
    all_configs = base.configs()
    runs, all_score_rows, replay_summary = [], [], {}
    try:
        for backbone, dataset, setting in GROUPS:
            fair.log(f"[{setting}] EXP-5B canonical exact replay")
            records, subjects, _, input_path = fair.load_data(backbone, dataset)
            outer, inner = fair.fold_priors(records, subjects, backbone)
            mode = "native" if backbone == "boostingvrme" else "fixed"
            stats, choices, counts, selection_rows, predictions = base.exact_replay(
                backbone, dataset, setting, records, subjects, outer, inner, all_configs, mode
            )
            metrics = fair.metrics(counts.sum(axis=0))
            score_rows = candidate_rows(backbone, dataset, setting, records, subjects, outer, all_configs, choices, mode)
            all_score_rows.extend(score_rows)
            replay_summary[setting] = {
                "selected_configuration": "PASS", "inner_f1": "PASS", "prediction_list": "PASS",
                "subject_counts": "PASS", "TP": metrics["TP"], "FP": metrics["FP"],
                "FN": metrics["FN"], "pooled_F1": metrics["F1"],
                "selected_per_subject": selection_rows,
                "input": str(input_path), "input_sha256": digest(Path(input_path)),
            }
            runs.append({
                "backbone": backbone, "dataset": dataset, "setting": setting, "records": records,
                "subjects": subjects, "outer": outer, "inner": inner, "mode": mode,
                "choices": choices, "counts": counts, "selection_rows": selection_rows,
                "predictions": predictions, "score_rows": score_rows,
            })
    except (AssertionError, ValueError) as error:
        failure = {"experiment": "EXP-5B", "canonical_replay": "FAIL", "status": "CANONICAL_REPLAY_FAILED", "error": str(error), "stage2_executed": False, "canonical_glsd_modified": False}
        (OUT / "protocol.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print("CANONICAL_REPLAY_FAILED")
        raise SystemExit(2) from error

    (OUT / "CANONICAL_LOCAL_CONTRAST.md").write_text(local_contrast_report(), encoding="utf-8")
    write_csv(OUT / "bilateral_diagnostic_scores.csv", all_score_rows)
    diagnostics, strata_rows, crossing_rows, bootstrap_rows = [], [], [], []
    strong_bootstraps = {}
    for run in runs:
        diagnostic, strata, crossing, boot_rows, strong_bs = aggregate_stage1(run["score_rows"], run["subjects"])
        diagnostics.append(diagnostic)
        strata_rows.extend(strata)
        crossing_rows.append(crossing)
        bootstrap_rows.extend(boot_rows)
        strong_bootstraps[run["setting"]] = strong_bs
    write_csv(OUT / "selected_asymmetry_diagnostic.csv", diagnostics)
    write_csv(OUT / "asymmetry_strata.csv", strata_rows)
    write_csv(OUT / "potential_meanbaseline_crossing.csv", crossing_rows)
    write_csv(OUT / "asymmetry_subject_bootstrap.csv", bootstrap_rows)

    gate_pass, diagnostic_checks, reasons = diagnostic_gate(diagnostics, crossing_rows, strong_bootstraps, all_score_rows)
    (OUT / "STAGE1_BILATERAL_AUDIT.md").write_text(
        stage1_report(diagnostics, crossing_rows, bootstrap_rows, gate_pass, diagnostic_checks, reasons), encoding="utf-8"
    )
    protocol = {
        "experiment": "EXP-5B", "parent": "canonical GLSD-v1", "canonical_replay": "PASS",
        "canonical_replay_details": replay_summary,
        "applicability": "PASS", "canonical_bilateral_operator": "max(B_L,B_R)",
        "diagnostic_alternative": "(B_L+B_R)/2", "candidate_asymmetry_aggregation": "median across canonically aligned matched scale pairs",
        "strongly_penalized_cutoff": STRONG_CUTOFF, "near_threshold_margin": NEAR_MARGIN,
        "bootstrap": {"unit": "subject", "paired": True, "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED, "missing_denominator": "NA"},
        "diagnostic_gate_b": "PASS" if gate_pass else "FAIL", "diagnostic_gate_checks": diagnostic_checks,
        "diagnostic_gate_reasons": reasons, "strongly_penalized_bootstrap": strong_bootstraps,
        "stage2_executed": gate_pass, "canonical_glsd_modified": False, "configuration_budget": 90,
        "forbidden_variants_tested": [], "continue_bilateral_baseline_search": False,
        "source_sha256": {
            "unified_persistence.py": digest(SIGNED / "unified_persistence.py"),
            "equiscale_fair_validation.py": digest(SIGNED / "equiscale_fair_validation.py"),
        },
    }

    if not gate_pass:
        protocol["status"] = "EXP5B_DIAGNOSTIC_GATE_FAILED"
        protocol["source_sha256"]["run_exp5b.py"] = digest(Path(__file__))
        (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        total_gt = sum(row["GT_compatible_crossers"] for row in crossing_rows)
        total_non_gt = sum(row["non_GT_compatible_crossers"] for row in crossing_rows)
        total_precision = safe_rate(total_gt, total_gt + total_non_gt)
        print("================================")
        print("EXP-5B DIAGNOSTIC VERDICT")
        print("================================")
        print("Canonical replay: PASS")
        print("Canonical bilateral operator: max(B_L,B_R)")
        print("Applicability: PASS")
        print(f"Any TP strongly-penalized enrichment >=5pp: {'YES' if diagnostic_checks['enrichment_settings'] else 'NO'}")
        print(f"Near-threshold GT crossers: {total_gt}")
        print(f"Near-threshold non-GT crossers: {total_non_gt}")
        print(f"Potential rescue precision: {fmt(total_precision)}")
        print("Subject-bootstrap direction: " + "; ".join(f"{key} CI upper={fmt(value['ci_high'])}" for key, value in strong_bootstraps.items()))
        print("Diagnostic Gate B: FAIL")
        print("Stage 2 executed: NO")
        print("Canonical GLSD modified: NO")
        print("Reason: " + "; ".join(reasons))
        print("================================")
        return

    result_rows, mechanism_rows, selected_rows = [], [], []
    for run in runs:
        fair.log(f"[{run['setting']}] gated EXP-5B-M 90-configuration search")
        choices, counts, selections, predictions, mechanisms = stage2_group(run, all_configs)
        canonical_metrics = fair.metrics(run["counts"].sum(axis=0))
        mean_metrics = fair.metrics(counts.sum(axis=0))
        low, high, positive = base.paired_f1_bootstrap(counts, run["counts"])
        fp_relative = (mean_metrics["FP"] - canonical_metrics["FP"]) / canonical_metrics["FP"] if canonical_metrics["FP"] else "NA"
        result_rows.append({
            "setting": run["setting"], "canonical_f1": canonical_metrics["F1"], "meanbaseline_f1": mean_metrics["F1"],
            "delta_f1": mean_metrics["F1"] - canonical_metrics["F1"], "ci_low": low, "ci_high": high,
            "canonical_tp": canonical_metrics["TP"], "canonical_fp": canonical_metrics["FP"], "canonical_fn": canonical_metrics["FN"],
            "mean_tp": mean_metrics["TP"], "mean_fp": mean_metrics["FP"], "mean_fn": mean_metrics["FN"],
            "delta_tp": mean_metrics["TP"] - canonical_metrics["TP"], "delta_fp": mean_metrics["FP"] - canonical_metrics["FP"],
            "delta_fn": mean_metrics["FN"] - canonical_metrics["FN"], "positive_resample_fraction": positive,
            "fp_relative_change": fp_relative, "fp_inflation": int(fp_relative != "NA" and fp_relative > 0.10),
            "severe_fp_inflation": int(fp_relative != "NA" and fp_relative > 0.25),
        })
        mechanism_rows.extend(mechanisms)
        selected_rows.extend(selections)
    write_csv(OUT / "results.csv", result_rows)
    write_csv(OUT / "meanbaseline_mechanism.csv", mechanism_rows)
    write_csv(OUT / "selected_config_per_subject.csv", selected_rows)
    gate, stage2_checks = stage2_gate(result_rows, mechanism_rows)
    (OUT / "EXP5B_ANALYSIS.md").write_text(stage2_report(result_rows, gate, stage2_checks), encoding="utf-8")
    protocol.update({"stage2_gate": gate, "stage2_checks": stage2_checks, "status": gate})
    protocol["source_sha256"]["run_exp5b.py"] = digest(Path(__file__))
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print("================================")
    print("EXP-5B FINAL VERDICT")
    print("================================")
    print("Diagnostic Gate B: PASS")
    print("Canonical replay: PASS")
    print(f"Stage-2 Gate: {gate}")
    print(f"Canonical mean F1: {stage2_checks['canonical_mean']:.6f}")
    print(f"Mean-baseline mean F1: {stage2_checks['meanbaseline_mean']:.6f}")
    for row in result_rows:
        print(f"{row['setting']}: {row['canonical_f1']:.6f} -> {row['meanbaseline_f1']:.6f} ({row['delta_f1']:+.6f})")
    print(f"TP rescue: {'YES' if stage2_checks['tp_rescue'] else 'NO'}")
    print(f"FP inflation: {'YES' if any(row['fp_inflation'] for row in result_rows) else 'NO'}")
    print(f"Severe FP inflation: {'YES' if stage2_checks['severe_fp_inflation'] else 'NO'}")
    print(f"Any significant regression: {'YES' if not stage2_checks['no_significant_regression'] else 'NO'}")
    print(f"Any significant improvement: {'YES' if stage2_checks['significant_improvement'] else 'NO'}")
    print("Configuration budget: 90")
    print("Canonical GLSD modified: NO")
    print("Continue bilateral-baseline search: NO")
    print("Recommended next step: retain canonical unless the preregistered Stage-2 gate supports promotion")


if __name__ == "__main__":
    main()
