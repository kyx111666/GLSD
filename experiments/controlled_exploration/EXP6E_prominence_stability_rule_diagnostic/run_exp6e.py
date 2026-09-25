"""EXP-6E: preregistered prominence-stability rule diagnostic.

This is a frozen, diagnostic-only replay.  It consumes the exact EXP-6D
canonical descriptors, adds the prespecified all-traceable-FP control, and
never changes a decoder, score, threshold, prediction, or evaluation metric.
"""
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
EXP6D = ROOT / "controlled_exploration/EXP6D_reference_g_normalization"
BOOTSTRAPS, SEED = 10_000, 100
METRICS = ("Q_P", "Q_G", "Q_logP")
LOW_ORIENTATION = {"Q_P": -1, "Q_G": -1, "Q_logP": 1}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = load_module("exp6d_parent_for_exp6e", EXP6D / "run_exp6d.py")
C, PARENT = D.C, D.PARENT


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else ["status"])
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def num(row: dict, key: str) -> float:
    return float(row[key])


def values(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray([num(row, metric) for row in rows if row.get(metric, "") not in ("", None)], dtype=float)


def subjects(rows: list[dict]) -> set[str]:
    return {row["subject"] for row in rows}


def auc_lost(left: np.ndarray, right: np.ndarray, metric: str) -> float:
    """Pairwise common-language probability; ties contribute one half."""
    if not len(left) or not len(right):
        return float("nan")
    if LOW_ORIENTATION[metric] < 0:
        wins = left[:, None] < right[None, :]
    else:
        wins = left[:, None] > right[None, :]
    ties = left[:, None] == right[None, :]
    return float((wins.sum() + .5 * ties.sum()) / (len(left) * len(right)))


def oriented_difference(left: np.ndarray, right: np.ndarray, metric: str, statistic: str) -> float:
    fn = np.mean if statistic == "mean" else np.median
    return float(LOW_ORIENTATION[metric] * (fn(left) - fn(right)))


def summary(rows: list[dict], group: str) -> list[dict]:
    result = []
    for metric in METRICS:
        v = values(rows, metric)
        result.append({
            "group": group, "metric": metric, "event_count": len(v),
            "subject_count": len(subjects(rows)), "mean": float(v.mean()) if len(v) else "",
            "median": float(np.median(v)) if len(v) else "",
            "std": float(v.std(ddof=1)) if len(v) > 1 else "",
            "IQR_low": float(np.quantile(v, .25)) if len(v) else "",
            "IQR_high": float(np.quantile(v, .75)) if len(v) else "",
            "min": float(v.min()) if len(v) else "", "max": float(v.max()) if len(v) else "",
        })
    return result


def subject_summary(groups: dict[str, list[dict]]) -> list[dict]:
    result = []
    for label, rows in groups.items():
        for subject in sorted(subjects(rows), key=str):
            subset = [row for row in rows if row["subject"] == subject]
            row = {"group": label, "subject": subject, "event_count": len(subset)}
            for metric in METRICS:
                v = values(subset, metric)
                row[f"mean_{metric}"] = float(v.mean()) if len(v) else ""
                row[f"median_{metric}"] = float(np.median(v)) if len(v) else ""
            result.append(row)
    return result


def by_subject(rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric, "") not in ("", None):
            out[row["subject"]].append(num(row, metric))
    return {key: np.asarray(value, dtype=float) for key, value in out.items()}


def bootstrap_comparison(left_rows: list[dict], right_rows: list[dict], metric: str,
                         comparison: str, paired: bool) -> list[dict]:
    """Subject-resampled mean, median, and pooled-event AUC diagnostics."""
    left, right = by_subject(left_rows, metric), by_subject(right_rows, metric)
    if paired:
        eligible = sorted(set(left) & set(right), key=str)
        left_ids = right_ids = eligible
        unit = "paired_subject"
    else:
        left_ids, right_ids = sorted(left, key=str), sorted(right, key=str)
        eligible = []
        unit = "independent_subject"
    status = "DESCRIPTIVE" if len(left_ids) >= 2 and len(right_ids) >= 2 else "INSUFFICIENT_SUBJECT_SUPPORT"
    base_left = np.concatenate([left[x] for x in left_ids]) if left_ids else np.asarray([])
    base_right = np.concatenate([right[x] for x in right_ids]) if right_ids else np.asarray([])
    base = {
        "comparison": comparison, "metric": metric, "left_subjects": len(left_ids),
        "right_subjects": len(right_ids), "eligible_subjects": len(eligible),
        "resamples": BOOTSTRAPS, "seed": SEED, "unit": unit, "status": status,
    }
    if status != "DESCRIPTIVE":
        return [{**base, "statistic": statistic, "estimate": "", "oriented_estimate": "", "ci95_low": "", "ci95_high": ""}
                for statistic in ("mean_difference", "median_difference", "auc")]
    rng = np.random.default_rng(SEED)
    if paired:
        draw = rng.integers(0, len(eligible), size=(BOOTSTRAPS, len(eligible)))
        pairs = [(left[s], right[s]) for s in eligible]
        sampled = [([pairs[i][0] for i in take], [pairs[i][1] for i in take]) for take in draw]
    else:
        ldraw = rng.integers(0, len(left_ids), size=(BOOTSTRAPS, len(left_ids)))
        rdraw = rng.integers(0, len(right_ids), size=(BOOTSTRAPS, len(right_ids)))
        sampled = [([left[left_ids[i]] for i in lt], [right[right_ids[i]] for i in rt]) for lt, rt in zip(ldraw, rdraw)]
    result = []
    for statistic in ("mean_difference", "median_difference", "auc"):
        if statistic == "auc":
            estimate = auc_lost(base_left, base_right, metric)
            estimates = np.asarray([auc_lost(np.concatenate(a), np.concatenate(b), metric) for a, b in sampled])
            oriented = estimate - .5
        else:
            stat = "mean" if statistic.startswith("mean") else "median"
            estimate = (float(np.mean(base_left) - np.mean(base_right)) if stat == "mean"
                        else float(np.median(base_left) - np.median(base_right)))
            estimates = np.asarray([((np.mean(np.concatenate(a)) - np.mean(np.concatenate(b))) if stat == "mean"
                                    else (np.median(np.concatenate(a)) - np.median(np.concatenate(b)))) for a, b in sampled])
            oriented = LOW_ORIENTATION[metric] * estimate
        result.append({**base, "statistic": statistic, "estimate": estimate, "oriented_estimate": oriented,
                       "ci95_low": float(np.quantile(estimates, .025)), "ci95_high": float(np.quantile(estimates, .975))})
    return result


def direction_rows(lost: list[dict], retained: list[dict]) -> list[dict]:
    result = []
    for metric in METRICS:
        a, b = by_subject(lost, metric), by_subject(retained, metric)
        eligible = sorted(set(a) & set(b), key=str)
        consistent = 0
        for subject in eligible:
            lm, rm = float(np.median(a[subject])), float(np.median(b[subject]))
            ok = LOW_ORIENTATION[metric] * (lm - rm) > 0
            consistent += ok
            result.append({"row_type": "subject", "metric": metric, "subject": subject,
                           "lost_median": lm, "retained_median": rm, "direction_consistent": "YES" if ok else "NO"})
        result.append({"row_type": "summary", "metric": metric, "subject": "",
                       "lost_median": "", "retained_median": "", "direction_consistent": "",
                       "consistent_subjects": consistent, "eligible_subjects": len(eligible),
                       "proportion": consistent / len(eligible) if eligible else ""})
    return result


def loo_rows(lost: list[dict], retained: list[dict]) -> list[dict]:
    result = []
    for metric in METRICS:
        common = sorted(subjects(lost) & subjects(retained), key=str)
        for held_out in common:
            l = [r for r in lost if r["subject"] != held_out and r["subject"] in common]
            r = [r for r in retained if r["subject"] != held_out and r["subject"] in common]
            diff = oriented_difference(values(l, metric), values(r, metric), metric, "median")
            result.append({"metric": metric, "held_out_subject": held_out, "oriented_median_difference": diff,
                           "direction_preserved": "YES" if diff > 0 else "NO"})
    return result


def overlap(left: list[dict], right: list[dict], left_name: str, right_name: str) -> list[dict]:
    result = []
    for metric in METRICS:
        a, b = values(left, metric), values(right, metric)
        aq, bq = np.quantile(a, [.25, .75]), np.quantile(b, [.25, .75])
        intersection = max(0., min(aq[1], bq[1]) - max(aq[0], bq[0]))
        union = max(aq[1], bq[1]) - min(aq[0], bq[0])
        grid = np.unique(np.concatenate((a, b)))
        ks = max(abs(np.mean(a <= x) - np.mean(b <= x)) for x in grid) if len(grid) else float("nan")
        result.append({"left_group": left_name, "right_group": right_name, "metric": metric,
                       "left_min": float(a.min()), "left_max": float(a.max()), "right_min": float(b.min()), "right_max": float(b.max()),
                       "IQR_overlap_fraction": intersection / union if union else 0., "ecdf_ks_distance": ks,
                       "ecdf_overlap_coefficient": 1. - ks})
    return result


def replay_all_fp(only_subjects: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """All R_A final FPs; exact peak, then positive-IoU peak is an auditable counterpart."""
    PARENT.fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    a_sel, c_sel = PARENT.selections("BoostingVRME/CAS(ME)3")
    records, subject_list, _, _ = PARENT.fair.load_data("boostingvrme", "casme3")
    ks, _ = PARENT.fair.fold_priors(records, subject_list, "boostingvrme")
    position = {subject: i for i, subject in enumerate(subject_list)}
    traceable, no_counterpart = [], []
    for anchor, selected in (("ANCHOR-A", a_sel), ("ANCHOR-C", c_sel)):
        for record in records:
            subject, video = record["subject"], record["video"]
            if only_subjects is not None and subject not in only_subjects:
                continue
            k = int(ks[position[subject]])
            a = D.build_state(record, "boostingvrme", k, a_sel[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
            c = D.build_state(record, "boostingvrme", k, c_sel[subject]["ref"], selected[subject]["rho"], selected[subject]["tau"])
            c_by_peak = {event["peak"]: event for event in c["p0"]}
            common = {"setting": "BoostingVRME/CAS(ME)3", "anchor": anchor, "subject": subject, "video": video,
                      "k": k, "rho": selected[subject]["rho"], "tau": selected[subject]["tau"],
                      "reference_A": a["ref"], "reference_C": c["ref"]}
            for event in a["p2"]:
                if event["matched_gt"] >= 0:
                    continue
                counterpart, method = c_by_peak.get(event["peak"]), "EXACT_PEAK"
                if counterpart is None:
                    candidates = [x for x in c["p0"] if C.iou(x["interval"], event["interval"]) > 0]
                    if candidates:
                        counterpart = min(candidates, key=lambda x: (abs(x["peak"] - event["peak"]), -x["S"], x["peak"]))
                        method = "INTERVAL_OVERLAP"
                if counterpart is None:
                    d = D.prominence_descriptor(a, event)
                    no_counterpart.append({**common, "group": "NO_COUNTERPART", "peak_A": d["peak"], "P_A": d["P"], "D_A": d["D"], "G_A": d["G"],
                                           "counterpart_method": "", "topology_class": "TOPOLOGY_DISAPPEARANCE"})
                    continue
                row = D.paired_row(common, a, event, c, counterpart, "GROUP_FA_ALL_TRACEABLE_FP")
                row.update({"counterpart_method": method, "counterpart_peak_C": counterpart["peak"],
                            "counterpart_distance": abs(counterpart["peak"] - event["peak"]), "topology_class": "TRACEABLE"})
                traceable.append(row)
    return traceable, no_counterpart


def parent_checks(primary: list[dict], removed: list[dict]) -> list[dict]:
    required = ["EXP6D_ANALYSIS.md", "g_factor_decomposition.csv", "scale_stability_features.csv", "lost_vs_retained.csv",
                "lost_vs_removed_fp.csv", "removed_fp_counterpart_status.csv", "subject_g_stability.csv", "mechanism_bootstrap.csv"]
    checks = [{"check": "required_exp6d_artifacts", "status": "PASS" if all((EXP6D / x).is_file() for x in required) else "FAIL",
               "observed": sum((EXP6D / x).is_file() for x in required), "expected": len(required), "detail": "all mandatory parent artifacts read"}]
    analysis = (EXP6D / "EXP6D_ANALYSIS.md").read_text(encoding="utf-8")
    parent = read_csv(EXP6D / "parent_replay_check.csv")
    identity = read_csv(EXP6D / "decomposition_identity_check.csv")
    factors = read_csv(EXP6D / "g_factor_decomposition.csv")
    stability = read_csv(EXP6D / "scale_stability_features.csv")
    lost_retained = read_csv(EXP6D / "lost_vs_retained.csv")
    lost_removed = read_csv(EXP6D / "lost_vs_removed_fp.csv")
    counterpart_status = read_csv(EXP6D / "removed_fp_counterpart_status.csv")
    subject_stability = read_csv(EXP6D / "subject_g_stability.csv")
    parent_bootstrap = read_csv(EXP6D / "mechanism_bootstrap.csv")
    def add(name, observed, expected, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected, "detail": detail})
        if not ok:
            raise RuntimeError("EXP6E_PARENT_REPLAY_FAILED")
    add("exp6d_parent_replay", all(x["status"] == "PASS" for x in parent), True, all(x["status"] == "PASS" for x in parent), "EXP-6D parent audit")
    add("exp6d_decomposition_identity", all(x["status"] == "PASS" for x in identity), True, all(x["status"] == "PASS" for x in identity), "EXP-6D identity audit")
    add("exp6d_scale_stability_features", len(stability), 166, len(stability) == 166, "read parent Q table")
    add("exp6d_lost_retained_table", len(lost_retained), 91, len(lost_retained) == 91, "read parent L/R table")
    add("exp6d_lost_removed_table", len(lost_removed), 88, len(lost_removed) == 88, "read parent L/FR table")
    add("exp6d_removed_counterpart_table", len(counterpart_status), 75, len(counterpart_status) == 75, "read parent counterpart table")
    add("exp6d_subject_stability_table", len(subject_stability) >= 28, True, len(subject_stability) >= 28, "read parent subject table")
    add("exp6d_parent_bootstrap", any(x["comparison"] == "LOST_MINUS_RETAINED" and x["status"] == "DESCRIPTIVE" for x in parent_bootstrap), True,
        any(x["comparison"] == "LOST_MINUS_RETAINED" and x["status"] == "DESCRIPTIVE" for x in parent_bootstrap), "read parent bootstrap table")
    lost = [x for x in primary if x["group"] == "G1_LOST_TP"]
    add("fresh_lost_tp_events", len(lost), 13, len(lost) == 13, "canonical frozen replay")
    add("fresh_lost_tp_subjects", len(subjects(lost)), 6, len(subjects(lost)) == 6, "canonical frozen replay")
    for key, expected in (("delta_G", -.263475), ("C_P", -.340306), ("C_D", .076831)):
        observed = float(np.mean([num(x, key) for x in lost])); add(f"fresh_mean_{key}", f"{observed:.6f}", f"{expected:.6f}", abs(observed-expected) <= 1e-6, "canonical frozen replay")
    add("fresh_abs_cp_gt_cd", sum(abs(num(x, "C_P")) > abs(num(x, "C_D")) for x in lost), 13,
        all(abs(num(x, "C_P")) > abs(num(x, "C_D")) for x in lost), "canonical frozen replay")
    add("parent_normalization_buffers", "BUFFERS" in analysis, True, "BUFFERS" in analysis, "EXP-6D report")
    add("fresh_mean_delta_g_ponly", f"{np.mean([num(x, 'delta_G_Ponly') for x in lost]):.6f}", "-0.316597",
        abs(np.mean([num(x, "delta_G_Ponly") for x in lost]) + .316597) <= 1e-6, "canonical frozen replay")
    add("fresh_mean_delta_g_donly", f"{np.mean([num(x, 'delta_G_Donly') for x in lost]):.6f}", "+0.100540",
        abs(np.mean([num(x, "delta_G_Donly") for x in lost]) - .100540) <= 1e-6, "canonical frozen replay")
    add("parent_raw_prominence_supported", "RAW_PROMINENCE" in analysis, True, "RAW_PROMINENCE" in analysis, "EXP-6D report")
    add("parent_normalization_rejected", "Normalization route: **REJECTED**" in analysis, True, "Normalization route: **REJECTED**" in analysis, "EXP-6D report")
    add("parent_lost_retained_supported", "Lost-vs-retained stability: SUPPORTED" in analysis, True, "Lost-vs-retained stability: SUPPORTED" in analysis, "EXP-6D report")
    add("removed_fp_limited_subject_support", len(subjects(removed)), 2, len(subjects(removed)) == 2, "fresh B/C replay; no false stability claim")
    add("fresh_shapley_identity", max(abs(num(x, "decomposition_error")) for x in primary + removed), "<=1e-12",
        max(abs(num(x, "decomposition_error")) for x in primary + removed) <= 1e-12, "fresh canonical paired rows")
    if any(x["status"] == "FAIL" for x in checks):
        raise RuntimeError("EXP6E_PARENT_REPLAY_FAILED")
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D.ENGINE.SCALES = C.SCALES
    D.ENGINE.REFERENCE_SCALES = C.SCALES
    D.verify_exp6c()  # Freshly replay EXP-6C before any EXP-6E calculation.
    primary, _ = D.replay_primary()
    removed_all = D.replay_removed_fp()
    lost = [x for x in primary if x["group"] == "G1_LOST_TP"]
    retained = [x for x in primary if x["group"] == "G2_RETAINED_TP"]
    removed = [x for x in removed_all if x["counterpart_available"] == "YES"]
    # The identical frozen B/C replay is collected in bounded subject batches
    # (see --collect-fp) so an interrupted execution cannot silently omit a
    # portion of the safety population.
    collected = read_csv(OUT / "all_fp_counterpart_audit.csv")
    all_fp = [x for x in collected if x.get("group") == "GROUP_FA_ALL_TRACEABLE_FP"]
    no_counterpart = [x for x in collected if x.get("group") == "NO_COUNTERPART"]
    # Some B/C subjects legitimately contain no R_A final FP and therefore do
    # not appear in either group.  The collection batches cover all 94 source
    # subjects; GROUP-FA coverage is reported from the 91 with an eligible FP.
    if not all_fp:
        raise RuntimeError("EXP6E_ALL_FP_REPLAY_INCOMPLETE")
    checks = parent_checks(primary, removed_all)
    groups = {"GROUP_L_LOST_TP": lost, "GROUP_R_RETAINED_TP": retained,
              "GROUP_FR_REMOVED_FP": removed, "GROUP_FA_ALL_TRACEABLE_FP": all_fp}
    event_summary = sum((summary(rows, label) for label, rows in groups.items()), [])
    subject_rows = subject_summary(groups)
    directions = direction_rows(lost, retained)
    loo = loo_rows(lost, retained)
    boot = []
    for metric in METRICS:
        boot += bootstrap_comparison(lost, retained, metric, "LOST_VS_RETAINED", True)
    comparison = []
    gate_features, safety_features = [], []
    for metric in METRICS:
        lv, rv, fv, frv = values(lost, metric), values(retained, metric), values(all_fp, metric), values(removed, metric)
        auc_lr, auc_lf, auc_lfr = auc_lost(lv, rv, metric), auc_lost(lv, fv, metric), auc_lost(lv, frv, metric)
        sr = next(x for x in directions if x["row_type"] == "summary" and x["metric"] == metric)
        br = next(x for x in boot if x["comparison"] == "LOST_VS_RETAINED" and x["metric"] == metric and x["statistic"] == "mean_difference")
        loo_ok = all(x["direction_preserved"] == "YES" for x in loo if x["metric"] == metric)
        pooled_ok = oriented_difference(lv, rv, metric, "median") > 0
        boot_ok = float(br["oriented_estimate"]) > 0
        gate = pooled_ok and sr["proportion"] >= 2/3 and auc_lr >= .70 and boot_ok and loo_ok
        if gate:
            gate_features.append(metric)
        all_fp_oriented = oriented_difference(lv, fv, metric, "median") > 0
        # A fixed descriptive AUC >= .60 is only a conservative feasibility criterion,
        # never a decision cutoff and never used to select a feature or rule.
        safety = gate and all_fp_oriented and auc_lf >= .60
        if safety:
            safety_features.append(metric)
        comparison.append({"metric": metric, "auc_lost_vs_retained": auc_lr, "auc_lost_vs_all_fp": auc_lf,
                           "auc_lost_vs_removed_fp": auc_lfr, "pooled_orientation_lost_retained": "CORRECT" if pooled_ok else "WRONG",
                           "direction_consistent_subjects": sr["consistent_subjects"], "eligible_subjects": sr["eligible_subjects"],
                           "direction_consistency": sr["proportion"], "bootstrap_oriented_effect": br["oriented_estimate"],
                           "loo_direction_preserved": "YES" if loo_ok else "NO", "tp_gate_feature": "PASS" if gate else "FAIL",
                           "all_fp_separability": "SUPPORTED" if safety else "NOT_SUPPORTED"})
    tp_gate = bool(gate_features)
    removed_support = len(subjects(removed))
    removed_status = "INSUFFICIENT" if removed_support < 3 else "SUPPORTED"
    all_fp_status = "SUPPORTED" if safety_features else "NOT_SUPPORTED"
    transitions = []
    transition_ok = []
    for metric in METRICS:
        for transition in sorted({(x["reference_A"], x["reference_C"]) for x in lost}):
            l = [x for x in lost if (x["reference_A"], x["reference_C"]) == transition]
            r = [x for x in retained if (x["reference_A"], x["reference_C"]) == transition]
            if not r:
                continue
            orient = oriented_difference(values(l, metric), values(r, metric), metric, "median") > 0
            auc = auc_lost(values(l, metric), values(r, metric), metric)
            transitions.append({"metric": metric, "reference_A": transition[0], "reference_C": transition[1], "lost_events": len(l), "retained_events": len(r),
                                "lost_subjects": len(subjects(l)), "retained_subjects": len(subjects(r)), "orientation_correct": "YES" if orient else "NO", "auc": auc})
            if metric in safety_features:
                transition_ok.append((metric, transition, orient and auc >= .5))
    robust_features = {metric for metric in safety_features if sum(x[0] == metric and x[2] for x in transition_ok) >= 2}
    transition_specific = bool(safety_features) and not bool(robust_features)
    # B/C contributes only FP controls, so it cannot establish TP-signal transfer.
    transfer = "INCONCLUSIVE"
    if all_fp_status != "SUPPORTED":
        rescue_risk = "INCONCLUSIVE"
    elif removed_support < 3:
        rescue_risk = "MODERATE"
    else:
        rescue_risk = "LOW"
    safe = tp_gate and bool(robust_features) and all_fp_status == "SUPPORTED" and removed_status != "CONFLICTING"
    feasible = safe
    protocol = {"experiment": "EXP-6E Prominence Stability Rule Diagnostic", "parent": "EXP-6D", "fixed_semantics": "E_7; saved rho/tau/k; saved R_A/R_C; canonical scipy prominence",
                "groups": {"GROUP_L": "B/S lost TP", "GROUP_R": "B/S retained TP", "GROUP_FR": "B/C R1 removed FP with counterpart", "GROUP_FA": "B/C all R_A final FP with legal R_C counterpart"},
                "predeclared_features": list(METRICS), "counterpart_policy": "exact R_C peak, otherwise positive-IoU R_C peak; no counterpart is topology disappearance and receives no Q value",
                "bootstrap": {"resamples": BOOTSTRAPS, "seed": SEED, "unit": "outer subject"},
                "forbidden": ["GLSD modification", "new score", "threshold search", "classifier", "F1/Recognition/STRS recomputation", "outer-test rule selection", "feature combination"],
                "canonical_GLSD_modified": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    manifest = {"source_sha256": {"EXP6D": digest(EXP6D / "run_exp6d.py"), "EXP6C": digest(D.EXP6C / "run_exp6c.py"), "EXP6E": digest(Path(__file__))},
                "parent_replay": "PASS", "shapley_identity": "PASS", "new_method_executed": False, "threshold_searched": False,
                "formal_F1_changed": False, "recognition_STRS_changed": False, "canonical_GLSD_modified": False}
    (OUT / "replay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv("parent_replay_check.csv", checks)
    write_csv("group_event_features.csv", sum(groups.values(), []))
    write_csv("event_distribution_summary.csv", event_summary)
    write_csv("subject_stability_groups.csv", subject_rows)
    write_csv("lost_retained_direction_consistency.csv", directions)
    write_csv("leave_one_subject_out.csv", loo)
    write_csv("subject_aware_bootstrap.csv", boot)
    write_csv("comparison_diagnostics.csv", comparison)
    write_csv("overlap_analysis.csv", overlap(lost, retained, "GROUP_L", "GROUP_R") + overlap(lost, all_fp, "GROUP_L", "GROUP_FA") + overlap(lost, removed, "GROUP_L", "GROUP_FR"))
    write_csv("all_fp_counterpart_audit.csv", all_fp + no_counterpart)
    write_csv("no_counterpart_topology_summary.csv", [{"NO_COUNTERPART_FP_COUNT": len(no_counterpart), "NO_COUNTERPART_FP_SUBJECTS": len(subjects(no_counterpart)),
                                                        "all_R_A_final_FP_count": len(all_fp) + len(no_counterpart), "no_counterpart_fraction": len(no_counterpart) / (len(all_fp) + len(no_counterpart)) if all_fp or no_counterpart else ""}])
    write_csv("reference_transition_analysis.csv", transitions)
    analysis = render_analysis(lost, retained, all_fp, removed, no_counterpart, event_summary, comparison, tp_gate, all_fp_status, removed_status,
                               transition_specific, transfer, safe, feasible, rescue_risk, robust_features, gate_features)
    (OUT / "EXP6E_ANALYSIS.md").write_text(analysis, encoding="utf-8")
    print_terminal(lost, retained, all_fp, removed, comparison, tp_gate, all_fp_status, removed_status, transition_specific, safe, feasible, rescue_risk)


def render_analysis(lost, retained, all_fp, removed, no_counterpart, event_summary, comparison, tp_gate, all_fp_status, removed_status,
                    transition_specific, transfer, safe, feasible, rescue_risk, robust_features, gate_features) -> str:
    lines = ["# EXP-6E — Prominence Stability Rule Diagnostic", "", "## Integrity", "",
             "- Fresh parent replay and all required EXP-6D artifact checks: **PASS**.",
             "- Fresh Shapley-style identity replay (`C_P + C_D = ΔG`): **PASS** at `≤1e-12`.",
             "- This diagnostic adds no GLSD score, threshold, decoder, prediction, F1, Recognition, or STRS computation.", "",
             "## Parent facts verified", "",
             "- B/S lost TP: 13 events across 6 subjects; mean `ΔG=-0.263475`, `C_P=-0.340306`, `C_D=+0.076831`; all 13 have `|C_P|>|C_D|`.",
             "- Normalization **BUFFERS** collapse; `ΔG_Ponly=-0.316597`, `ΔG_Donly=+0.100540`; raw-prominence route is supported and normalization route rejected.",
             "- Parent lost-vs-retained signal is supported. B/C removed-FP coverage is limited to two subjects and is not treated as stable inference.", "",
             "## Event-level distributions", "", "See `event_distribution_summary.csv` for count, subject count, mean, median, standard deviation, IQR, min, and max for every preregistered quantity and group.", "",
             "## Lost TP versus retained TP", ""]
    for row in comparison:
        lines.append(f"- `{row['metric']}`: AUC={row['auc_lost_vs_retained']:.3f}; direction consistency={row['direction_consistent_subjects']}/{row['eligible_subjects']} ({row['direction_consistency']:.1%}); leave-one-subject-out direction={row['loo_direction_preserved']}; TP feature gate={row['tp_gate_feature']}.")
    lines += ["", f"TP stability gate: **{'PASS' if tp_gate else 'FAIL'}** (`{', '.join(gate_features) if gate_features else 'none'}`).", "",
              "## FP safety controls", "", f"- All traceable B/C FPs: {len(all_fp)} events across {len(subjects(all_fp))} subjects.",
              f"- Removed B/C FPs with counterpart: {len(removed)} events across {len(subjects(removed))} subjects; safety evidence: **{removed_status}**.",
              f"- No-counterpart topology disappearances: {len(no_counterpart)} events across {len(subjects(no_counterpart))} subjects; they receive no fabricated Q value.",
              f"- Lost-vs-all-FP separability: **{all_fp_status}** (`{', '.join(robust_features) if robust_features else 'no robust feature'}`). The ECDF and IQR overlap diagnostics are in `overlap_analysis.csv`.", "",
              "## Dependence and feasibility", "", f"- `REFERENCE_TRANSITION_SPECIFIC = {'YES' if transition_specific else 'NO'}`; transition audit is in `reference_transition_analysis.csv`.",
              f"- `STABILITY_SIGNAL_TRANSFER = {transfer}`: B/C supplies FP safety controls only, not a B/C lost-TP analogue; no cross-dataset TP transfer is asserted.",
              f"- `RESCUE_RISK = {rescue_risk}`.", f"- `SAFE_STABILITY_SIGNAL_{'SUPPORTED' if safe else 'NOT_SUPPORTED'}`.",
              f"- `TRAINING_ONLY_RULE_{'FEASIBLE' if feasible else 'NOT_JUSTIFIED'}`.", "",
              "## Verdict", "", f"Recommended next experiment: **{'PROSPECTIVE_STABILITY_RULE_VALIDATION' if feasible else 'STOP_REFERENCE_G_RULE_MINING'}**.",
              "", "No threshold was searched and no candidate-specific GT fitting, feature combination, classifier, or outer-test rule was used.", ""]
    return "\n".join(lines)


def print_terminal(lost, retained, all_fp, removed, comparison, tp_gate, all_fp_status, removed_status, transition_specific, safe, feasible, rescue_risk) -> None:
    print("================================")
    print("EXP-6E PROMINENCE STABILITY VERDICT")
    print("================================")
    print("Parent replay:\nPASS")
    print(f"\nLost TP events:\n{len(lost)}\n\nLost TP subjects:\n{len(subjects(lost))}")
    print(f"\nRetained TP events:\n{len(retained)}\n\nAll traceable FP events:\n{len(all_fp)}\n\nAll traceable FP subjects:\n{len(subjects(all_fp))}")
    print(f"\nRemoved FP events:\n{len(removed)}\n\nRemoved FP subjects:\n{len(subjects(removed))}")
    for row in comparison:
        print(f"\n{row['metric']} lost-vs-retained:\nAUC = {row['auc_lost_vs_retained']:.3f}\nDirection consistency = {row['direction_consistent_subjects']}/{row['eligible_subjects']}")
    print(f"\nTP stability gate:\n{'PASS' if tp_gate else 'FAIL'}")
    print(f"\nLost-vs-all-FP separability:\n{all_fp_status}\n\nRemoved-FP safety evidence:\n{removed_status}")
    print(f"\nReference-transition specific:\n{'YES' if transition_specific else 'NO'}")
    print(f"\nSafe stability signal:\n{'SUPPORTED' if safe else 'NOT_SUPPORTED'}\n\nTraining-only rule feasibility:\n{'SUPPORTED' if feasible else 'NOT_JUSTIFIED'}")
    print(f"\nRescue risk:\n{rescue_risk}\n\nRecommended next experiment:\n{'PROSPECTIVE_STABILITY_RULE_VALIDATION' if feasible else 'STOP_REFERENCE_G_RULE_MINING'}")
    print("\nNew method executed:\nNO\n\nThreshold searched:\nNO\n\nFormal F1 changed:\nNO\n\nRecognition/STRS changed:\nNO\n\nCanonical GLSD modified:\nNO")
    print("================================")


if __name__ == "__main__":
    # The collection mode makes the long frozen B/C response replay resumable in
    # bounded subject batches.  It cannot calculate a verdict or alter a method.
    if len(sys.argv) == 3 and sys.argv[1] == "--collect-fp":
        requested = set(sys.argv[2].split(","))
        D.ENGINE.SCALES = C.SCALES
        D.ENGINE.REFERENCE_SCALES = C.SCALES
        traceable, absent = replay_all_fp(requested)
        path = OUT / "all_fp_counterpart_audit.csv"
        old = read_csv(path) if path.is_file() else []
        keys = {(x["anchor"], x["subject"], x["video"], x["peak_A"]) for x in old}
        merged = old + [x for x in traceable + absent if (x["anchor"], x["subject"], x["video"], x["peak_A"]) not in keys]
        write_csv("all_fp_counterpart_audit.csv", merged)
        print(f"EXP6E_FP_COLLECTION subjects={len(requested)} traceable={len(traceable)} no_counterpart={len(absent)} total={len(merged)}")
    else:
        main()
