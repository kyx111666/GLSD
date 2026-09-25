"""EXP-5A: scale-support / missing-zero diagnostic and gated support-aware evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SIGNED = ROOT / "historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme"
RETHINK = ROOT / "RethinkFuse_reproduction"
ARCHIVE = ROOT / "historical_gl_exact_fresh_reproduction/fresh_run"

sys.path.insert(0, str(SIGNED))
import equiscale_fair_validation as fair  # noqa: E402
import unified_persistence as engine  # noqa: E402


GROUPS = (
    ("metst", "sammlv", "ME-TST/SAMMLV"),
    ("metst", "casme3", "ME-TST/CAS(ME)3"),
    ("boostingvrme", "sammlv", "BoostingVRME/SAMMLV"),
    ("boostingvrme", "casme3", "BoostingVRME/CAS(ME)3"),
)
BOOTSTRAP_RESAMPLES = 10_000
SEED = 100
NEAR_MARGIN = 0.10
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    reference: float
    radius: float
    threshold: float
    family: str = "unified"
    height_weight: float = 0.0

    @property
    def identifier(self) -> str:
        return (
            f"unified|scale={self.reference:g}|height=0"
            f"|tau={self.threshold:g}|radius={self.radius:g}"
        )


def configs() -> list[Config]:
    values = [
        Config(reference, radius, threshold)
        for reference in engine.REFERENCE_SCALES
        for radius in engine.LOCAL_RADII
        for threshold in engine.THRESHOLDS
    ]
    if len(values) != 90:
        raise AssertionError("Canonical configuration budget must be 90")
    return values


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else "NA"


def fmt(value, digits=6):
    return "NA" if value == "NA" else f"{float(value):.{digits}f}"


def archive_dir(backbone: str, dataset: str) -> Path:
    mode = "native" if backbone == "boostingvrme" else "fixed"
    return ARCHIVE / backbone / "results/pure_persistence_matched_v1" / dataset / mode


def archived(backbone: str, dataset: str):
    root = archive_dir(backbone, dataset)
    selection = [row for row in load_csv(root / "outer_loso_selections.csv") if row["family"] == "pure"]
    counts_rows = [row for row in load_csv(root / "subject_counts.csv") if row["family"] == "pure"]
    counts = np.array([[int(row[key]) for key in ("TP", "FP", "FN")] for row in counts_rows])
    predictions = json.loads((root / "selected_predictions.json").read_text(encoding="utf-8"))
    return selection, counts, predictions


class AuditedCurveFeatures(engine.CurveFeatures):
    """Canonical evidence plus explicit matched/missing support metadata."""

    def detailed_evidence(self, reference: float, radius: float):
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))
        widths = list(self.effective_scales)
        reference_width = min(len(self.curve), max(1, round(reference * self.k)))
        if reference_width not in self.effective_scales:
            raise AssertionError("Reference physical width is absent from evidence scales")
        reference_position = widths.index(reference_width)
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
        details = []
        for point in peaks:
            local_vector, matched_vector = [], []
            for width, (other_peaks, _, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    local_vector.append(0.0)
                    matched_vector.append(False)
                    continue
                chosen = min(
                    indexes,
                    key=lambda index: (abs(int(other_peaks[index]) - point), -values[index]),
                )
                local_vector.append(float(values[chosen]))
                matched_vector.append(True)
            matched_values = [v for v, matched in zip(local_vector, matched_vector) if matched]
            if not matched_vector[reference_position]:
                raise AssertionError("Canonical reference candidate failed to self-match")
            k_scales = len(widths)
            matched_count = int(sum(matched_vector))
            canonical_local = float(np.median(local_vector))
            valid_median = float(np.median(matched_values)) if matched_values else 0.0
            support_local = (matched_count / k_scales) * valid_median if k_scales else 0.0
            details.append({
                "peak": int(point),
                "K": k_scales,
                "scale_widths": widths,
                "local_vector": local_vector,
                "matched_vector": matched_vector,
                "matched_scale_count": matched_count,
                "positive_evidence_scale_count": int(sum(v > 0 for v, m in zip(local_vector, matched_vector) if m)),
                "missing_scale_count": k_scales - matched_count,
                "canonical_L": canonical_local,
                "valid_median": valid_median,
                "support_L": support_local,
                "missing_zero_suppressed": int(valid_median > canonical_local + EPS),
                "hard_zero_suppressed": int(abs(canonical_local) <= EPS and valid_median > EPS),
            })
        evidence = np.column_stack((height, global_values, [row["canonical_L"] for row in details]))
        return peaks, evidence, details

    def evidence(self, reference, radius=1.0):
        key = reference, radius
        if key not in self.evidence_cache:
            peaks, evidence, _ = self.detailed_evidence(reference, radius)
            self.evidence_cache[key] = peaks, evidence
        return self.evidence_cache[key]


class SupportCurveFeatures(AuditedCurveFeatures):
    def evidence(self, reference, radius=1.0):
        key = reference, radius, "support"
        if key not in self.evidence_cache:
            peaks, canonical, details = self.detailed_evidence(reference, radius)
            support = canonical.copy()
            support[:, 2] = [row["support_L"] for row in details]
            self.evidence_cache[key] = peaks, support
        return self.evidence_cache[key]


def prepare(record, features, config: Config, backbone: str, mode: str):
    peaks, evidence = features.evidence(config.reference, config.radius)
    decoder = fair.VideoFeatures(record, features.k, backbone if mode == "native" else "metst")
    return engine.Prepared(decoder, peaks, evidence)


def evaluate_k(records, subjects, backbone, k, needed, all_configs, mode, feature_cls):
    stats = np.zeros((len(all_configs), len(subjects), 3), dtype=np.int32)
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    batches = defaultdict(list)
    for index, config in enumerate(all_configs):
        batches[(config.reference, config.radius)].append(index)
    for video_index, record in enumerate(records):
        sid = subject_ids[record["subject"]]
        if not needed[sid]:
            continue
        features = feature_cls(record["curve"], k)
        for indexes in batches.values():
            selected = [all_configs[index] for index in indexes]
            item = prepare(record, features, selected[0], backbone, mode)
            stats[indexes, sid] += item.evaluate(selected)
        if (video_index + 1) % 100 == 0:
            fair.log(f"[{backbone}] EXP-5A k={k}: {video_index + 1}/{len(records)} videos")
    return stats


def select(stats, outer, inner, subjects, all_configs):
    choices, rows = [], []
    counts = np.zeros((len(subjects), 3), dtype=np.int64)
    indexes = list(range(len(all_configs)))
    for held, subject in enumerate(subjects):
        training = engine.inner_counts(stats, held, inner)
        choice = engine.choose(training, indexes)
        config = all_configs[choice]
        choices.append(choice)
        counts[held] = stats[int(outer[held])][choice, held]
        metrics = fair.metrics(training[choice])
        rows.append({
            "subject": subject,
            "selected_a0": config.reference,
            "selected_rho": config.radius,
            "selected_tau": config.threshold,
            "config": config.identifier,
            "inner_f1": metrics["F1"],
            "inner_precision": metrics["precision"],
            "inner_fp": metrics["FP"],
            "outer_tp": int(counts[held, 0]),
            "outer_fp": int(counts[held, 1]),
            "outer_fn": int(counts[held, 2]),
        })
    return choices, counts, rows


def exact_replay(backbone, dataset, setting, records, subjects, outer, inner, all_configs, mode):
    stats = {}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist())):
        needed = fair.required_subjects(k, outer, inner)
        stats[k] = evaluate_k(records, subjects, backbone, k, needed, all_configs, mode, AuditedCurveFeatures)
    choices, counts, rows = select(stats, outer, inner, subjects, all_configs)
    archived_rows, archived_counts, archived_predictions = archived(backbone, dataset)
    if [row["subject"] for row in archived_rows] != subjects:
        raise AssertionError(f"Subject order mismatch: {setting}")
    for current, old, choice in zip(rows, archived_rows, choices):
        if all_configs[choice].identifier != old["config"]:
            raise AssertionError(f"Selected config mismatch: {setting}/{current['subject']}")
        if float(current["inner_f1"]) != float(old["inner_F1"]):
            raise AssertionError(f"Inner F1 mismatch: {setting}/{current['subject']}")
    np.testing.assert_array_equal(counts, archived_counts)

    replay_counts = np.zeros_like(counts)
    replay_predictions = []
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    for record, old_video in zip(records, archived_predictions):
        sid = subject_ids[record["subject"]]
        config = all_configs[choices[sid]]
        features = AuditedCurveFeatures(record["curve"], int(outer[sid]))
        values, predictions = prepare(record, features, config, backbone, mode).evaluate([config], details=True)
        replay_counts[sid] += values[0]
        if record["subject"] != old_video["subject"] or record["video"] != old_video["video"]:
            raise AssertionError(f"Archived video order mismatch: {setting}")
        if predictions[0] != old_video["predictions"]["pure"]:
            raise AssertionError(f"Prediction list mismatch: {setting}/{record['video']}")
        replay_predictions.append(predictions[0])
    np.testing.assert_array_equal(replay_counts, counts)
    return stats, choices, counts, rows, replay_predictions


def candidate_rows(backbone, dataset, setting, records, subjects, outer, all_configs, choices, mode):
    rows = []
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    for video_index, record in enumerate(records):
        sid = subject_ids[record["subject"]]
        config = all_configs[choices[sid]]
        features = AuditedCurveFeatures(record["curve"], int(outer[sid]))
        peaks, evidence, details = features.detailed_evidence(config.reference, config.radius)
        decoder = fair.VideoFeatures(record, features.k, backbone if mode == "native" else "metst")
        prepared = engine.Prepared(decoder, peaks, evidence)
        _, selected_details = prepared.evaluate([config], details=True)
        selected_by_peak = {item["peak"]: item for item in selected_details[0]}
        lookup = {int(peak): index for index, peak in enumerate(peaks)}
        for candidate_index, cluster in enumerate(prepared.clusters):
            peak = int(cluster[0])
            detail = details[lookup[peak]]
            global_value = float(prepared.evidence[candidate_index, 1])
            canonical_l = float(prepared.evidence[candidate_index, 2])
            support_l = detail["support_L"]
            canonical_s = 0.5 * (global_value + canonical_l)
            support_s = 0.5 * (global_value + support_l)
            selected_item = selected_by_peak.get(peak)
            selected = selected_item is not None
            association = (
                "TP" if selected and selected_item["matched_gt"] >= 0
                else "FP" if selected else "not_selected"
            )
            margin = config.threshold - canonical_s
            near = 0 < margin <= NEAR_MARGIN + EPS
            interval = prepared.intervals[candidate_index]
            row = {
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
                "matched_scale_count": detail["matched_scale_count"],
                "support_ratio": detail["matched_scale_count"] / detail["K"],
                "positive_evidence_scale_count": detail["positive_evidence_scale_count"],
                "missing_scale_count": detail["missing_scale_count"],
                "matched_vector": json.dumps([int(x) for x in detail["matched_vector"]]),
                "local_vector": json.dumps(detail["local_vector"]),
                "G": global_value,
                "L_canonical": canonical_l,
                "diagnostic_valid_median": detail["valid_median"],
                "L_support_diag": support_l,
                "S_canonical": canonical_s,
                "S_support_diag": support_s,
                "delta_L_diag": support_l - canonical_l,
                "delta_S_diag": support_s - canonical_s,
                "missing_zero_suppressed": detail["missing_zero_suppressed"],
                "hard_zero_suppressed": detail["hard_zero_suppressed"],
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
                "would_cross_threshold": int(near and support_s + EPS >= config.threshold),
            }
            rows.append(row)
    return rows


def paired_prevalence_bootstrap(subject_rows, tp_field, fp_field):
    pairs = [
        (float(row[tp_field]), float(row[fp_field]))
        for row in subject_rows
        if row[tp_field] != "NA" and row[fp_field] != "NA"
    ]
    if not pairs:
        return {"n_subjects": 0, "mean_difference": "NA", "ci_low": "NA", "ci_high": "NA", "p_fp_gt_tp": "NA"}
    values = np.asarray(pairs)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    deltas = (values[draws, 0] - values[draws, 1]).mean(axis=1)
    low, high = np.quantile(deltas, (0.025, 0.975))
    return {
        "n_subjects": len(values),
        "mean_difference": float(np.mean(values[:, 0] - values[:, 1])),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_fp_gt_tp": float(np.mean(deltas < 0)),
    }


def aggregate_stage1(setting_rows, subjects):
    selected = [row for row in setting_rows if row["selected_prediction"]]
    tp = [row for row in selected if row["selected_association"] == "TP"]
    fp = [row for row in selected if row["selected_association"] == "FP"]
    setting = setting_rows[0]["setting"]

    diagnostic = {"setting": setting, "num_tp": len(tp), "num_fp": len(fp)}
    for label, group in (("tp", tp), ("fp", fp)):
        for m in (1, 2, 3):
            count = sum(row["matched_scale_count"] == m for row in group)
            diagnostic[f"{label}_m{m}_count"] = count
            diagnostic[f"{label}_m{m}_prevalence"] = safe_rate(count, len(group))
        low = sum(row["matched_scale_count"] < row["K"] for row in group)
        diagnostic[f"{label}_low_support_count"] = low
        diagnostic[f"{label}_low_support_prevalence"] = safe_rate(low, len(group))
        for flag in ("hard_zero_suppressed", "missing_zero_suppressed"):
            count = sum(row[flag] for row in group)
            diagnostic[f"{label}_{flag}"] = count
            diagnostic[f"{label}_{flag}_prevalence"] = safe_rate(count, len(group))

    strata = []
    for m in (1, 2, 3):
        a = sum(row["matched_scale_count"] == m for row in tp)
        b = sum(row["matched_scale_count"] == m for row in fp)
        strata.append({"setting": setting, "stratum_type": "matched_scale_count", "stratum": f"m={m}", "TP": a, "FP": b, "precision": safe_rate(a, a + b)})
    for value in (0, 1):
        a = sum(row["missing_zero_suppressed"] == value for row in tp)
        b = sum(row["missing_zero_suppressed"] == value for row in fp)
        strata.append({"setting": setting, "stratum_type": "missing_zero_suppressed", "stratum": str(value), "TP": a, "FP": b, "precision": safe_rate(a, a + b)})

    near_rows = []
    near = [row for row in setting_rows if row["near_threshold"]]
    for compatible, label in ((1, "GT-compatible"), (0, "non-GT-compatible")):
        group = [row for row in near if row["gt_compatible"] == compatible]
        near_rows.append({
            "setting": setting,
            "candidate_group": label,
            "count": len(group),
            "m1_count": sum(row["matched_scale_count"] == 1 for row in group),
            "m2_count": sum(row["matched_scale_count"] == 2 for row in group),
            "m3_count": sum(row["matched_scale_count"] == 3 for row in group),
            "hard_zero_suppressed": sum(row["hard_zero_suppressed"] for row in group),
            "missing_zero_suppressed": sum(row["missing_zero_suppressed"] for row in group),
            "mean_margin": float(np.mean([row["margin"] for row in group])) if group else "NA",
            "median_margin": float(np.median([row["margin"] for row in group])) if group else "NA",
        })

    crossers = [row for row in near if row["would_cross_threshold"]]
    gt_cross = sum(row["gt_compatible"] for row in crossers)
    non_gt_cross = len(crossers) - gt_cross
    crossing = {
        "setting": setting,
        "num_gt_compatible_crossers": gt_cross,
        "num_non_gt_crossers": non_gt_cross,
        "potential_rescue_precision": safe_rate(gt_cross, len(crossers)),
    }

    subject_rows = []
    for subject in subjects:
        subject_selected = [row for row in selected if row["subject"] == subject]
        subject_tp = [row for row in subject_selected if row["selected_association"] == "TP"]
        subject_fp = [row for row in subject_selected if row["selected_association"] == "FP"]
        subject_near = [row for row in crossers if row["subject"] == subject]
        subject_rows.append({
            "setting": setting,
            "subject": subject,
            "num_tp": len(subject_tp),
            "num_fp": len(subject_fp),
            "tp_low_support_prevalence": safe_rate(sum(row["matched_scale_count"] < row["K"] for row in subject_tp), len(subject_tp)),
            "fp_low_support_prevalence": safe_rate(sum(row["matched_scale_count"] < row["K"] for row in subject_fp), len(subject_fp)),
            "tp_hard_zero_prevalence": safe_rate(sum(row["hard_zero_suppressed"] for row in subject_tp), len(subject_tp)),
            "fp_hard_zero_prevalence": safe_rate(sum(row["hard_zero_suppressed"] for row in subject_fp), len(subject_fp)),
            "gt_compatible_crossers": sum(row["gt_compatible"] for row in subject_near),
            "non_gt_crossers": sum(not row["gt_compatible"] for row in subject_near),
        })
    low_bs = paired_prevalence_bootstrap(subject_rows, "tp_low_support_prevalence", "fp_low_support_prevalence")
    hard_bs = paired_prevalence_bootstrap(subject_rows, "tp_hard_zero_prevalence", "fp_hard_zero_prevalence")
    for row in subject_rows:
        for prefix, values in (("low_support_bootstrap", low_bs), ("hard_zero_bootstrap", hard_bs)):
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
    return diagnostic, strata, near_rows, crossing, subject_rows, low_bs, hard_bs


def gate_s(diagnostics, crossings, bootstraps, score_rows):
    by_setting = {row["setting"]: row for row in diagnostics}
    cross_by_setting = {row["setting"]: row for row in crossings}
    qualifying = []
    for setting, row in by_setting.items():
        tp_low = row["tp_low_support_prevalence"]
        fp_low = row["fp_low_support_prevalence"]
        cross = cross_by_setting[setting]
        enrichment = tp_low != "NA" and fp_low != "NA" and tp_low - fp_low >= 0.05
        burden = row["tp_hard_zero_suppressed"] >= 10 or cross["num_gt_compatible_crossers"] >= 10
        precision = cross["potential_rescue_precision"] != "NA" and cross["potential_rescue_precision"] >= 0.20
        if enrichment and burden and precision:
            qualifying.append(setting)
    universal_bad = all(
        row["num_non_gt_crossers"] > row["num_gt_compatible_crossers"]
        and row["potential_rescue_precision"] != "NA"
        and row["potential_rescue_precision"] < 0.10
        for row in crossings
    )
    clear_fp_enrichment = any(
        values["ci_high"] != "NA" and values["ci_high"] < 0
        for values in bootstraps.values()
    )
    changes = any(abs(row["delta_L_diag"]) > EPS or abs(row["delta_S_diag"]) > EPS for row in score_rows)
    passed = bool(qualifying) and not universal_bad and not clear_fp_enrichment and changes
    reasons = []
    if not qualifying:
        reasons.append("No single setting jointly met low-support enrichment, burden, and rescue-precision conditions")
    if universal_bad:
        reasons.append("All settings showed dominant non-GT crossers with rescue precision below 0.10")
    if clear_fp_enrichment:
        reasons.append("A subject bootstrap clearly supported FP low-support enrichment")
    if not changes:
        reasons.append("Support-aware diagnostic was mathematically invariant")
    return passed, qualifying, reasons or ["All preregistered Gate S conditions were met"]


def paired_f1_bootstrap(candidate, baseline):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(candidate), size=(BOOTSTRAP_RESAMPLES, len(candidate)))
    def f1(values):
        totals = values[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(2 * totals[:, 0], denominator, out=np.zeros(BOOTSTRAP_RESAMPLES), where=denominator > 0)
    delta = f1(candidate) - f1(baseline)
    low, high = np.quantile(delta, (0.025, 0.975))
    return float(low), float(high), float(np.mean(delta > 0))


def replay_method(records, subjects, outer, backbone, mode, all_configs, choices, feature_cls):
    subject_ids = {subject: index for index, subject in enumerate(subjects)}
    counts = np.zeros((len(subjects), 3), dtype=np.int64)
    predictions = []
    for record in records:
        sid = subject_ids[record["subject"]]
        config = all_configs[choices[sid]]
        features = feature_cls(record["curve"], int(outer[sid]))
        values, detail = prepare(record, features, config, backbone, mode).evaluate([config], details=True)
        counts[sid] += values[0]
        predictions.append(detail[0])
    return counts, predictions


def stage2_group(backbone, dataset, setting, records, subjects, outer, inner, mode, all_configs, canonical_choices, canonical_counts, canonical_predictions):
    stats = {}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist())):
        needed = fair.required_subjects(k, outer, inner)
        stats[k] = evaluate_k(records, subjects, backbone, k, needed, all_configs, mode, SupportCurveFeatures)
    choices, counts, selection_rows = select(stats, outer, inner, subjects, all_configs)
    replay_counts, predictions = replay_method(records, subjects, outer, backbone, mode, all_configs, choices, SupportCurveFeatures)
    np.testing.assert_array_equal(replay_counts, counts)
    changes = fair.event_changes(predictions, canonical_predictions)
    delta = counts.sum(axis=0) - canonical_counts.sum(axis=0)
    if changes["net_TP"] != delta[0] or changes["net_FP"] != delta[1]:
        raise AssertionError(f"Stage-2 event accounting mismatch: {setting}")

    canonical_support = Counter()
    support_support = Counter()
    deltas = {"TP": [], "FP": []}
    for method_predictions, method_choices, feature_cls, target_counter in (
        (canonical_predictions, canonical_choices, AuditedCurveFeatures, canonical_support),
        (predictions, choices, SupportCurveFeatures, support_support),
    ):
        subject_ids = {subject: index for index, subject in enumerate(subjects)}
        for record, pred_list in zip(records, method_predictions):
            sid = subject_ids[record["subject"]]
            config = all_configs[method_choices[sid]]
            features = AuditedCurveFeatures(record["curve"], int(outer[sid]))
            _, _, details = features.detailed_evidence(config.reference, config.radius)
            detail_by_peak = {row["peak"]: row for row in details}
            for pred in pred_list:
                target_counter[detail_by_peak[pred["peak"]]["matched_scale_count"]] += 1
                if feature_cls is SupportCurveFeatures:
                    item = detail_by_peak[pred["peak"]]
                    label = "TP" if pred["matched_gt"] >= 0 else "FP"
                    delta_l = item["support_L"] - item["canonical_L"]
                    deltas[label].append((delta_l, 0.5 * delta_l))

    mechanism = {
        "setting": setting,
        "rescued_gt": changes["GT_rescued"],
        "removed_fp": changes["FP_removed_exact_interval"],
        "lost_gt": changes["GT_lost"],
        "new_fp": changes["FP_added_exact_interval"],
    }
    for m in (1, 2, 3):
        mechanism[f"canonical_selected_m{m}"] = canonical_support[m]
        mechanism[f"support_selected_m{m}"] = support_support[m]
    for label in ("TP", "FP"):
        values = deltas[label]
        mechanism[f"{label.lower()}_mean_delta_L"] = float(np.mean([x[0] for x in values])) if values else "NA"
        mechanism[f"{label.lower()}_median_delta_L"] = float(np.median([x[0] for x in values])) if values else "NA"
        mechanism[f"{label.lower()}_mean_delta_S"] = float(np.mean([x[1] for x in values])) if values else "NA"
    for row in selection_rows:
        row["setting"] = setting
    return choices, counts, selection_rows, predictions, mechanism


def stage2_gate(results):
    canonical_mean = float(np.mean([row["canonical_f1"] for row in results]))
    support_mean = float(np.mean([row["supportaware_f1"] for row in results]))
    no_sig_regression = all(row["ci_high"] >= 0 for row in results)
    noninferior = sum(row["supportaware_f1"] >= row["canonical_f1"] for row in results)
    improved = sum(row["supportaware_f1"] > row["canonical_f1"] for row in results)
    recall_rescue = any(
        (row["delta_tp"] > 0 or row["delta_fn"] < 0)
        and (row["delta_fp"] <= 0 or row["support_fp"] <= 1.10 * row["canonical_fp"])
        for row in results
    )
    fp_explosion = any(row["delta_fp"] > 10 and row["support_fp"] > 1.10 * row["canonical_fp"] for row in results)
    significant_improvement = any(row["ci_low"] > 0 for row in results)
    strong = no_sig_regression and noninferior >= 3 and support_mean > canonical_mean and recall_rescue and not fp_explosion
    partial = no_sig_regression and improved in (2, 3) and support_mean > canonical_mean and recall_rescue and not fp_explosion
    if strong:
        gate = "EXP5A_STRONG_CANDIDATE"
    elif partial:
        gate = "EXP5A_PARTIAL_SUPPORT"
    else:
        gate = "EXP5A_FAILED"
    return gate, {
        "canonical_mean": canonical_mean,
        "support_mean": support_mean,
        "no_significant_regression": no_sig_regression,
        "noninferior_count": noninferior,
        "improved_count": improved,
        "recall_rescue": recall_rescue,
        "fp_explosion": fp_explosion,
        "significant_improvement": significant_improvement,
    }


def stage1_report(diagnostics, strata, near, crossings, bootstraps, gate_pass, qualifying, reasons):
    lines = [
        "# EXP-5A Stage 1 Scale-Support Audit", "",
        "## Canonical replay", "",
        "Selected configurations, inner F1 values, full selected prediction lists, subject TP/FP/FN, and pooled F1 replayed exactly in all four settings: **PASS**.", "",
        "## Canonical scale-support semantics", "",
        "The implementation iterates over distinct physical smoothing widths. In every evaluated outer and inner fold here, K=3. The reference width is present in that map; the reference peak therefore self-matches at temporal distance zero. A non-reference width with no peak inside `max(1, round(0.5*k))` appends exactly `0.0`. A matched bilateral local contrast can also legally be exactly zero. Aggregation is `numpy.median`; for K=3 this is the second order statistic, so one positive matched value plus two missing zeros yields zero.", "",
        "## Selected TP/FP support", "",
        "| Setting | TP | FP | TP low | FP low | TP hard-zero | FP hard-zero |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(f"| {row['setting']} | {row['num_tp']} | {row['num_fp']} | {fmt(row['tp_low_support_prevalence'])} | {fmt(row['fp_low_support_prevalence'])} | {row['tp_hard_zero_suppressed']} | {row['fp_hard_zero_suppressed']} |")
    lines.extend(["", "## Near-threshold and crossing diagnostic", "", "| Setting | GT crossers | non-GT crossers | rescue precision |", "|---|---:|---:|---:|"])
    for row in crossings:
        lines.append(f"| {row['setting']} | {row['num_gt_compatible_crossers']} | {row['num_non_gt_crossers']} | {fmt(row['potential_rescue_precision'])} |")
    lines.extend(["", "Near-threshold means `0 < selected_tau - S_canonical <= 0.10`. Candidate compatibility uses the unchanged potential interval and the evaluator's inclusive-frame IoU >= 0.5 best-GT rule. It is diagnostic only and never replaces formal matching.", "", "## Subject paired bootstrap", ""])
    for setting, values in bootstraps.items():
        lines.append(f"- {setting}: TP-low minus FP-low mean={fmt(values['mean_difference'])}, 95% CI=[{fmt(values['ci_low'])}, {fmt(values['ci_high'])}], paired subjects={values['n_subjects']}, P(bootstrap mean < 0)={fmt(values['p_fp_gt_tp'])}.")
    lines.extend(["", "## Gate S", "", f"**{'PASS' if gate_pass else 'EXP5A_DIAGNOSTIC_GATE_FAILED'}**", "", f"Qualifying setting(s): {', '.join(qualifying) if qualifying else 'none'}.", "", "Reason: " + "; ".join(reasons) + ".", "", f"Stage 2 executed: **{'YES' if gate_pass else 'NO'}**.", "", "Canonical GLSD-v1 was not modified."])
    return "\n".join(lines) + "\n"


def stage2_report(results, mechanisms, gate, checks):
    lines = ["# EXP-5A-S Support-Aware Local Evidence", "", "## Results", "", "| Setting | Canonical F1 | Support-aware F1 | Delta | 95% paired CI |", "|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['setting']} | {row['canonical_f1']:.6f} | {row['supportaware_f1']:.6f} | {row['delta_f1']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |")
    lines.extend(["", "## Mechanism", ""])
    for row in mechanisms:
        lines.append(f"- {row['setting']}: rescued GT={row['rescued_gt']}, removed FP={row['removed_fp']}, lost GT={row['lost_gt']}, new FP={row['new_fp']}.")
    lines.extend(["", "## Gate", "", f"**{gate}**", "", "; ".join(f"{key}={value}" for key, value in checks.items()) + ".", "", "The only method change was `L_sup=(m/K)*median(V_valid)`. Configuration budget remained 90. Canonical GLSD-v1 was not modified, and no further missing-scale variant was tested."])
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = RETHINK / "caches/me_tst"
    all_configs = configs()
    runs = []
    all_score_rows = []
    replay_summary = {}
    try:
        for backbone, dataset, setting in GROUPS:
            fair.log(f"[{setting}] canonical exact replay")
            records, subjects, _, input_path = fair.load_data(backbone, dataset)
            outer, inner = fair.fold_priors(records, subjects, backbone)
            mode = "native" if backbone == "boostingvrme" else "fixed"
            stats, choices, counts, selection_rows, predictions = exact_replay(
                backbone, dataset, setting, records, subjects, outer, inner, all_configs, mode
            )
            metrics = fair.metrics(counts.sum(axis=0))
            score_rows = candidate_rows(backbone, dataset, setting, records, subjects, outer, all_configs, choices, mode)
            all_score_rows.extend(score_rows)
            replay_summary[setting] = {
                "selected_config": "PASS", "prediction_list": "PASS", "subject_counts": "PASS",
                "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"], "F1": metrics["F1"],
                "input": str(input_path), "input_sha256": digest(Path(input_path)),
            }
            runs.append({
                "backbone": backbone, "dataset": dataset, "setting": setting, "records": records,
                "subjects": subjects, "outer": outer, "inner": inner, "mode": mode,
                "choices": choices, "counts": counts, "selection_rows": selection_rows,
                "predictions": predictions, "score_rows": score_rows,
            })
    except (AssertionError, ValueError) as error:
        failure = {"experiment": "EXP-5A", "canonical_replay": "FAIL", "status": "CANONICAL_REPLAY_FAILED", "error": str(error), "stage2_executed": False, "canonical_glsd_modified": False}
        (OUT / "protocol.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print("CANONICAL_REPLAY_FAILED")
        raise SystemExit(2) from error

    write_csv(OUT / "support_diagnostic_scores.csv", all_score_rows)
    diagnostic_rows, strata_rows, near_rows, crossing_rows, subject_rows = [], [], [], [], []
    low_bootstraps, hard_bootstraps = {}, {}
    for run in runs:
        diagnostic, strata, near, crossing, subjects_out, low_bs, hard_bs = aggregate_stage1(run["score_rows"], run["subjects"])
        diagnostic_rows.append(diagnostic)
        strata_rows.extend(strata)
        near_rows.extend(near)
        crossing_rows.append(crossing)
        subject_rows.extend(subjects_out)
        low_bootstraps[run["setting"]] = low_bs
        hard_bootstraps[run["setting"]] = hard_bs
    write_csv(OUT / "selected_support_diagnostic.csv", diagnostic_rows)
    write_csv(OUT / "support_strata.csv", strata_rows)
    write_csv(OUT / "near_threshold_support.csv", near_rows)
    write_csv(OUT / "potential_threshold_crossing.csv", crossing_rows)
    write_csv(OUT / "support_subject.csv", subject_rows)

    gate_pass, qualifying, reasons = gate_s(diagnostic_rows, crossing_rows, low_bootstraps, all_score_rows)
    (OUT / "STAGE1_SCALE_SUPPORT_AUDIT.md").write_text(
        stage1_report(diagnostic_rows, strata_rows, near_rows, crossing_rows, low_bootstraps, gate_pass, qualifying, reasons), encoding="utf-8"
    )
    protocol = {
        "experiment": "EXP-5A", "parent": "canonical GLSD-v1", "canonical_replay": "PASS",
        "canonical_replay_details": replay_summary,
        "scale_semantics": {
            "K_observed": sorted(set(row["K"] for row in all_score_rows)),
            "reference_self_match": True,
            "unmatched_nonreference_value": 0.0,
            "matched_evidence_may_equal_zero": True,
            "median": "numpy.median",
            "K3_behavior": "second order statistic",
        },
        "near_threshold_margin": NEAR_MARGIN,
        "bootstrap": {"unit": "subject", "paired": True, "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED, "missing_denominator": "NA"},
        "gate_s": "PASS" if gate_pass else "FAIL", "qualifying_settings": qualifying, "gate_reasons": reasons,
        "low_support_bootstrap": low_bootstraps, "hard_zero_bootstrap": hard_bootstraps,
        "stage2_executed": gate_pass, "canonical_glsd_modified": False,
        "configuration_budget": 90, "forbidden_variants_tested": [],
        "source_sha256": {"unified_persistence.py": digest(SIGNED / "unified_persistence.py"), "equiscale_fair_validation.py": digest(SIGNED / "equiscale_fair_validation.py")},
    }
    protocol["source_sha256"]["run_exp5a.py"] = digest(Path(__file__))

    if not gate_pass:
        protocol["status"] = "EXP5A_DIAGNOSTIC_GATE_FAILED"
        protocol["continue_missing_scale_search"] = False
        (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        total_hard = sum(row["tp_hard_zero_suppressed"] for row in diagnostic_rows)
        total_gt = sum(row["num_gt_compatible_crossers"] for row in crossing_rows)
        total_non_gt = sum(row["num_non_gt_crossers"] for row in crossing_rows)
        precision = safe_rate(total_gt, total_gt + total_non_gt)
        print("================================")
        print("EXP-5A DIAGNOSTIC VERDICT")
        print("================================")
        print("Canonical replay: PASS")
        print("K: [3]")
        print("Reference self-match: YES")
        print(f"TP low-support enrichment: {'YES' if qualifying else 'NO'}")
        print(f"Hard-zero suppressed TP count: {total_hard}")
        print(f"Near-threshold GT-compatible crossers: {total_gt}")
        print(f"Near-threshold non-GT crossers: {total_non_gt}")
        print(f"Potential rescue precision: {fmt(precision)}")
        print("Subject-bootstrap direction: " + "; ".join(f"{key}={fmt(value['mean_difference'])}" for key, value in low_bootstraps.items()))
        print("Diagnostic Gate S: FAIL")
        print("Stage 2 executed: NO")
        print("Canonical GLSD modified: NO")
        print("Reason: " + "; ".join(reasons))
        print("================================")
        return

    result_rows, mechanism_rows, selected_rows = [], [], []
    for run in runs:
        fair.log(f"[{run['setting']}] gated EXP-5A-S 90-configuration search")
        choices, counts, selections, predictions, mechanism = stage2_group(
            run["backbone"], run["dataset"], run["setting"], run["records"], run["subjects"],
            run["outer"], run["inner"], run["mode"], all_configs, run["choices"], run["counts"], run["predictions"]
        )
        canonical_metrics = fair.metrics(run["counts"].sum(axis=0))
        support_metrics = fair.metrics(counts.sum(axis=0))
        low, high, positive = paired_f1_bootstrap(counts, run["counts"])
        result_rows.append({
            "setting": run["setting"], "canonical_f1": canonical_metrics["F1"], "supportaware_f1": support_metrics["F1"],
            "delta_f1": support_metrics["F1"] - canonical_metrics["F1"], "ci_low": low, "ci_high": high,
            "canonical_tp": canonical_metrics["TP"], "canonical_fp": canonical_metrics["FP"], "canonical_fn": canonical_metrics["FN"],
            "support_tp": support_metrics["TP"], "support_fp": support_metrics["FP"], "support_fn": support_metrics["FN"],
            "delta_tp": support_metrics["TP"] - canonical_metrics["TP"], "delta_fp": support_metrics["FP"] - canonical_metrics["FP"],
            "delta_fn": support_metrics["FN"] - canonical_metrics["FN"], "positive_resample_fraction": positive,
        })
        mechanism_rows.append(mechanism)
        selected_rows.extend(selections)
    write_csv(OUT / "results.csv", result_rows)
    write_csv(OUT / "support_mechanism.csv", mechanism_rows)
    write_csv(OUT / "selected_config_per_subject.csv", selected_rows)
    gate, checks = stage2_gate(result_rows)
    (OUT / "EXP5A_ANALYSIS.md").write_text(stage2_report(result_rows, mechanism_rows, gate, checks), encoding="utf-8")
    protocol.update({"stage2_gate": gate, "stage2_checks": checks, "support_formula": "(m/K)*median(V_valid)", "continue_missing_scale_search": False, "status": gate})
    protocol["source_sha256"]["run_exp5a.py"] = digest(Path(__file__))
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print("================================")
    print("EXP-5A FINAL VERDICT")
    print("================================")
    print("Diagnostic Gate S: PASS")
    print("Canonical replay: PASS")
    print(f"Stage-2 Gate: {gate}")
    print(f"Canonical mean F1: {checks['canonical_mean']:.6f}")
    print(f"Support-aware mean F1: {checks['support_mean']:.6f}")
    for row in result_rows:
        print(f"{row['setting']}: {row['canonical_f1']:.6f} -> {row['supportaware_f1']:.6f} ({row['delta_f1']:+.6f})")
    print(f"TP rescue observed: {'YES' if any(row['delta_tp'] > 0 for row in result_rows) else 'NO'}")
    print(f"FP explosion: {'YES' if checks['fp_explosion'] else 'NO'}")
    print(f"Any significant regression: {'YES' if not checks['no_significant_regression'] else 'NO'}")
    print(f"Any significant improvement: {'YES' if checks['significant_improvement'] else 'NO'}")
    print("Configuration budget: 90")
    print("Canonical GLSD modified: NO")
    print("Continue missing-scale search: NO")
    print("Recommended next step: retain canonical unless the preregistered Stage-2 gate supports promotion")


if __name__ == "__main__":
    main()
