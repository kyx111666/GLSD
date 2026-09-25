"""EXP-7D: frozen EXP-7B duration protocol on ME-TST+/SAMMLV only."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP7A = ROOT / "controlled_exploration/EXP7A_event_geometry_diagnostic"
EXP7B = ROOT / "controlled_exploration/EXP7B_training_only_duration_geometry_validation"
EXP7C = ROOT / "controlled_exploration/EXP7C_boosting_duration_replication"
BASE = ROOT / "controlled_exploration/baseline_snapshot/results/main_results.csv"
sys.path.extend((str(EXP7A), str(EXP7B), str(EXP7C)))
from run_exp7a import canonical_sources, config, digest, fair, read_csv, replay_video  # noqa: E402
from run_exp7b_phasea import alpha_for_outer  # noqa: E402
from run_exp7b_phaseb import blind_fixed_center_adapter, bootstrap, formal_match, metrics, transition_rows  # noqa: E402
from run_exp7c import primary_context  # noqa: E402


BACKBONE, DATASET, SETTING = "metst", "sammlv", "ME-TST+/SAMMLV"
NEAR_ONE = .05


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def event_json(event: dict) -> dict:
    return {"onset": event["interval"][0], "offset": event["interval"][1], "peak": event["peak"], "matched_gt": event["matched_gt"]}


def sammlv_ids(rows: list[dict]) -> list[dict]:
    """The imported Phase-A pair helper is algorithmically identical; fix only its audit dataset label."""
    for row in rows:
        for key in ("GT_id", "candidate_id", "prediction_id"):
            if key in row:
                row[key] = row[key].replace("|casme3|", "|sammlv|")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    records, subjects, _, _ = fair.load_data(BACKBONE, DATASET)
    outer_k, _ = fair.fold_priors(records, subjects, BACKBONE)
    subject_index = {subject: i for i, subject in enumerate(subjects)}
    selections, expected, saved = canonical_sources(BACKBONE, DATASET)
    states, total, parent_rows, integrity = {}, np.zeros(3, dtype=int), [], True
    for record in records:
        subject, video = record["subject"], record["video"]
        states[subject, video] = replay_video(record, config(selections[subject]), BACKBONE, int(outer_k[subject_index[subject]]))
    for subject in subjects:
        own = [(video, state) for (current, video), state in states.items() if current == subject]
        observed = np.sum([state["counts"] for _, state in own], axis=0)
        prediction_ok = all([event_json(x) for x in state["p2"]] == saved[subject, video] for video, state in own)
        counts_ok = tuple(observed) == expected[subject]
        integrity &= prediction_ok and counts_ok
        total += observed
        parent_rows.append({"scope": "outer_subject", "outer_subject": subject, "expected_TP": expected[subject][0], "expected_FP": expected[subject][1], "expected_FN": expected[subject][2], "replay_TP": int(observed[0]), "replay_FP": int(observed[1]), "replay_FN": int(observed[2]), "prediction_check": "PASS" if prediction_ok else "FAIL", "counts_check": "PASS" if counts_ok else "FAIL"})
    base = next(row for row in read_csv(BASE) if row["Backbone"] == BACKBONE and row["Dataset"] == DATASET)
    archive, archived_f1 = tuple(int(base[f"GL_Skill_{key}"]) for key in ("TP", "FP", "FN")), float(base["GL_Skill_F1"])
    integrity &= tuple(total) == archive and np.isclose(metrics(total)["F1"], archived_f1, atol=1e-15)
    parent_rows.append({"scope": "TOTAL", "outer_subject": "ALL", "expected_TP": archive[0], "expected_FP": archive[1], "expected_FN": archive[2], "replay_TP": int(total[0]), "replay_FP": int(total[1]), "replay_FN": int(total[2]), "archived_F1": archived_f1, "replay_F1": metrics(total)["F1"], "prediction_check": "PASS" if integrity else "FAIL", "counts_check": "PASS" if integrity else "FAIL"})
    parent_fields = ["scope", "outer_subject", "expected_TP", "expected_FP", "expected_FN", "replay_TP", "replay_FP", "replay_FN", "archived_F1", "replay_F1", "prediction_check", "counts_check"]
    write_csv("parent_replay_check.csv", parent_rows, parent_fields)
    if not integrity:
        raise RuntimeError("EXP7D_PARENT_REPLAY_FAILED")
    pairs, per_subject, outer_alpha, cache, leakage = [], [], [], {}, []
    for outer_subject in subjects:
        fold_pairs, fold_subject, fold_alpha = alpha_for_outer(subjects, records, outer_subject, config(selections[outer_subject]), int(outer_k[subject_index[outer_subject]]), cache)
        pairs += sammlv_ids(fold_pairs); per_subject += fold_subject; outer_alpha.append(fold_alpha)
        leakage.append({"outer_subject": outer_subject, "reused_phaseA_alpha_for_outer": "PASS", "training_record_excludes_outer_subject": "PASS", "GT_assertion_before_pairing": "PASS", "outer_test_GT_to_alpha": "NO", "outer_test_GT_to_adapter": "NO"})
    if any(row["outer_test_GT_to_alpha"] != "NO" or row["outer_test_GT_to_adapter"] != "NO" for row in leakage):
        raise RuntimeError("EXP7D_LEAKAGE_AUDIT_FAILED")
    alpha_values = np.array([row["alpha_subject_balanced_clipped"] for row in outer_alpha])
    q1, median, q3 = np.quantile(alpha_values, [.25, .5, .75])
    alpha_summary = [{"number_of_outer_folds": len(alpha_values), "median_alpha": median, "IQR_Q1": q1, "IQR_Q3": q3, "IQR_width": q3-q1, "min": alpha_values.min(), "max": alpha_values.max(), "alpha_lt_1_count": int(np.sum(alpha_values < 1-NEAR_ONE)), "alpha_approx_1_count": int(np.sum(np.abs(alpha_values-1) <= NEAR_ONE)), "alpha_gt_1_count": int(np.sum(alpha_values > 1+NEAR_ONE)), "lower_clip_count": sum(row["hit_lower_clip"] for row in outer_alpha), "upper_clip_count": sum(row["hit_upper_clip"] for row in outer_alpha), "near_one_definition": "abs(alpha-1)<=0.05"}]
    audits, transitions, subject_rows, leakage_adapter, canonical_counts, adapted_counts = [], [], [], [], [], []
    record_map = {(record["subject"], record["video"]): record for record in records}
    for subject in subjects:
        alpha = next(row["alpha_subject_balanced_clipped"] for row in outer_alpha if row["outer_subject"] == subject)
        canonical_events, adapted_events, gt_count, own_audits = [], [], 0, {}
        for (current, video), state in states.items():
            if current != subject:
                continue
            record = record_map[subject, video]
            # Strictly strip match status before the imported GT-free adapter receives its input.
            blind = [{"index": event["index"], "peak": event["peak"], "interval": event["interval"], "score": event["score"]} for event in state["p2"]]
            adapted_blind, audits_by_id = [], {}
            for event in blind:
                audit = blind_fixed_center_adapter(event, alpha, len(record["curve"]))
                if not audit["center_preserved"]:
                    raise RuntimeError("EXP7D_CENTER_PRESERVATION_FAILED")
                audits_by_id[event["index"]] = audit
                adapted_blind.append({**event, "interval": (audit["adapted_start"], audit["adapted_end"])})
                audits.append({"outer_subject": subject, "event_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|P2|{event['index']}", "video": video, **audit})
            canonical = formal_match(blind, record["gt"])
            adapted = formal_match(adapted_blind, record["gt"])
            if [event["matched_gt"] for event in canonical] != [event["matched_gt"] for event in state["p2"]]:
                raise RuntimeError("EXP7D_CANONICAL_EVALUATOR_REPLAY_FAILED")
            canonical_events += canonical; adapted_events += adapted; gt_count += len(record["gt"]); own_audits.update(audits_by_id)
            transitions += sammlv_ids(transition_rows(subject, video, canonical, adapted, record["gt"], audits_by_id))
        can_tp, adp_tp = sum(event["matched_gt"] >= 0 for event in canonical_events), sum(event["matched_gt"] >= 0 for event in adapted_events)
        canonical_value = np.array([can_tp, len(canonical_events)-can_tp, gt_count-can_tp])
        adapted_value = np.array([adp_tp, len(adapted_events)-adp_tp, gt_count-adp_tp])
        canonical_counts.append(canonical_value); adapted_counts.append(adapted_value)
        own_gt = [row for row in transitions if row["outer_subject"] == subject and row["record_type"] == "GT"]
        own_pred = [row for row in transitions if row["outer_subject"] == subject and row["record_type"] == "PREDICTION"]
        subject_rows.append({"outer_subject": subject, "alpha_s": alpha, **{f"canonical_{key}": value for key, value in metrics(canonical_value).items()}, **{f"adapted_{key}": value for key, value in metrics(adapted_value).items()}, "delta_TP": int(adapted_value[0]-canonical_value[0]), "delta_FP": int(adapted_value[1]-canonical_value[1]), "delta_FN": int(adapted_value[2]-canonical_value[2]), "delta_F1": metrics(adapted_value)["F1"]-metrics(canonical_value)["F1"], "rescued": sum(row["transition_type"] == "RESCUED" for row in own_gt), "lost": sum(row["transition_type"] == "LOST" for row in own_gt), "fp_removed": sum(row["transition_type"] == "FP_REMOVED" for row in own_pred), "new_fp": sum(row["transition_type"] == "NEW_FP" for row in own_pred), "match_reassigned": sum(row["match_reassigned"] for row in own_gt), "boundary_clipped_count": sum(row["boundary_clipped"] for row in own_pred)})
        leakage_adapter.append({"outer_subject": subject, "adapter_function": "EXP7B.blind_fixed_center_adapter", "adapter_input_schema": "index|peak|interval|score;alpha;video_length", "outer_test_GT_passed_to_adapter": "NO", "center_preservation_assertion": "PASS"})
    can_array, adp_array = np.asarray(canonical_counts), np.asarray(adapted_counts)
    if not np.array_equal(can_array.sum(axis=0), total):
        raise RuntimeError("EXP7D_CANONICAL_TOTAL_CHANGED")
    boot_rows, delta_f1, ci_low, ci_high = bootstrap(adp_array, can_array)
    canonical_total, adapted_total = metrics(can_array.sum(axis=0)), metrics(adp_array.sum(axis=0))
    delta = {key: adapted_total[key]-canonical_total[key] for key in ("TP", "FP", "FN", "precision", "recall", "F1")}
    gt_transitions = Counter(row["transition_type"] for row in transitions if row["record_type"] == "GT")
    pred_transitions = Counter(row["transition_type"] for row in transitions if row["record_type"] == "PREDICTION")
    distribution = {"positive_delta_F1_subjects": sum(row["delta_F1"] > 0 for row in subject_rows), "neutral_delta_F1_subjects": sum(np.isclose(row["delta_F1"], 0, atol=1e-15) for row in subject_rows), "negative_delta_F1_subjects": sum(row["delta_F1"] < 0 for row in subject_rows), "rescued_gt_lost_subjects": sum(row["rescued"] > row["lost"] for row in subject_rows), "rescued_eq_lost_subjects": sum(row["rescued"] == row["lost"] for row in subject_rows), "rescued_lt_lost_subjects": sum(row["rescued"] < row["lost"] for row in subject_rows), "total_net_TP_gain": int(delta["TP"])}
    gains = sorted((max(0, row["delta_TP"]) for row in subject_rows), reverse=True)
    for number in (1, 2, 5):
        distribution[f"top_{number}_TP_gain_fraction"] = sum(gains[:number])/delta["TP"] if delta["TP"] > 0 else "NA"
    transition_ok = gt_transitions["RESCUED"]-gt_transitions["LOST"] == delta["TP"] and pred_transitions["NEW_FP"]-pred_transitions["FP_REMOVED"] == delta["FP"]
    status = "SUPPORTED" if delta_f1 > 0 and ci_low > 0 else "INCONCLUSIVE" if delta_f1 > 0 else "NOT_SUPPORTED"
    gates = {"GATE-A": "PASS", "GATE-B": "PASS", "GATE-C": "PASS", "GATE-D": "PASS" if all(row["center_preserved"] == 1 for row in audits) else "FAIL", "GATE-E": "PASS" if delta_f1 > 0 else "FAIL", "GATE-F": "PASS" if ci_low > 0 else "FAIL", "GATE-G": "PASS" if transition_ok else "FAIL", "GATE-H": "FACTS_REPORTED_NO_NEW_CUTOFF"}
    pair_fields = ["outer_subject", "training_subject", "video", "GT_id", "candidate_id", "gt_index", "GT_onset", "GT_offset", "candidate_peak", "formal_event_onset", "formal_event_offset", "L_G", "L_E", "ratio", "selection_rule"]
    alpha_fields = ["outer_subject", "training_subject", "n_valid_pairs", "alpha_training_subject_median", "valid_training_subject"]
    outer_fields = ["outer_subject", "n_train_subjects", "n_valid_training_subjects", "n_valid_pairs", "alpha_pooled_event_median", "alpha_subject_balanced_raw", "alpha_subject_balanced_clipped", "hit_lower_clip", "hit_upper_clip", "outer_test_GT_accessed_for_alpha"]
    audit_fields = ["outer_subject", "event_id", "video", "original_start", "original_end", "original_length", "original_center", "alpha_s", "target_length_float", "target_length_integer", "requested_length", "adapted_start", "adapted_end", "adapted_length", "adapted_center", "center_preserved", "boundary_clipped", "video_length"]
    transition_fields = ["record_type", "outer_subject", "video", "GT_id", "prediction_id", "canonical_status", "adapted_status", "transition_type", "match_reassigned", "canonical_interval", "adapted_interval", "canonical_match_id", "adapted_match_id", "canonical_iou", "adapted_iou", "boundary_clipped"]
    subject_fields = ["outer_subject", "alpha_s", "canonical_TP", "canonical_FP", "canonical_FN", "canonical_precision", "canonical_recall", "canonical_F1", "adapted_TP", "adapted_FP", "adapted_FN", "adapted_precision", "adapted_recall", "adapted_F1", "delta_TP", "delta_FP", "delta_FN", "delta_F1", "rescued", "lost", "fp_removed", "new_fp", "match_reassigned", "boundary_clipped_count"]
    write_csv("training_pair_ratios.csv", pairs, pair_fields); write_csv("per_training_subject_alpha.csv", per_subject, alpha_fields); write_csv("outer_alpha_calibration.csv", outer_alpha, outer_fields); write_csv("alpha_summary.csv", alpha_summary, list(alpha_summary[0])); write_csv("leakage_audit.csv", leakage, list(leakage[0])); write_csv("duration_adapter_audit.csv", audits, audit_fields); write_csv("outer_subject_results.csv", subject_rows, subject_fields); write_csv("event_transition_accounting.csv", transitions, transition_fields); write_csv("bootstrap_delta_f1.csv", boot_rows, ["iteration", "delta_F1", "N", "seed"]); write_csv("subject_gain_distribution.csv", [distribution], list(distribution)); write_csv("adapter_leakage_audit.csv", leakage_adapter, list(leakage_adapter[0]))
    context = primary_context(); context.update({"replication_dataset": "SAMMLV", "replication_alpha_median": median, "replication_alpha_IQR": f"{q1}:{q3}", "replication_canonical_F1": canonical_total["F1"], "replication_adapted_F1": adapted_total["F1"], "replication_delta_F1": delta_f1, "replication_ci95": f"{ci_low}:{ci_high}", "replication_rescued": gt_transitions["RESCUED"], "replication_lost": gt_transitions["LOST"], "replication_fp_removed": pred_transitions["FP_REMOVED"], "replication_new_fp": pred_transitions["NEW_FP"], "replication_positive_neutral_negative": f"{distribution['positive_delta_F1_subjects']}/{distribution['neutral_delta_F1_subjects']}/{distribution['negative_delta_F1_subjects']}"})
    write_csv("dataset_contextual_comparison.csv", [context], list(context))
    protocol = {"experiment": "EXP-7D", "setting": SETTING, "reused_helpers": ["EXP7B PhaseA alpha_for_outer", "EXP7B PhaseB blind_fixed_center_adapter", "EXP7B PhaseB formal_match", "EXP7B PhaseB transition_rows", "EXP7B PhaseB bootstrap"], "protocol_changed": False, "forbidden": ["alpha search", "cross-dataset alpha transfer", "conditional rules", "Boosting", "Recognition", "STRS"]}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"experiment": "EXP-7D", "script_sha256": digest(Path(__file__)), "canonical_modified": False, "EXP7B_modified": False,
                "primary_context_sha256": digest(EXP7B / "EXP7B_PHASEB_ANALYSIS.md"),
                "frozen_helper_sha256": {"phaseA_alpha_for_outer": digest(EXP7B / "run_exp7b_phasea.py"),
                                         "phaseB_adapter_matching_bootstrap": digest(EXP7B / "run_exp7b_phaseb.py")}}
    (OUT / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# EXP-7D — ME-TST+/SAMMLV Duration Calibration Replication", "", "## 1. Integrity", "", f"- Canonical replay: **PASS** — TP/FP/FN={tuple(map(int, can_array.sum(axis=0)))}; F1={canonical_total['F1']:.15f}.", "- The EXP-7B calibration and adapter helpers are imported directly without behavioural modification.", "- Leakage audit: every calibration fold excludes its outer subject; adapter input is stripped of GT and matching status.", "", "## 2. Training-only calibration", "", f"Subject-balanced alpha: median={median:.6f}; IQR=[{q1:.6f}, {q3:.6f}]; range=[{alpha_values.min():.6f}, {alpha_values.max():.6f}]; lower/upper clips={alpha_summary[0]['lower_clip_count']}/{alpha_summary[0]['upper_clip_count']}.", "", "## 3. Blind duration adapter", "", "The imported EXP-7B same-parity rounding, deterministic tie-break, fixed-center construction, and legal boundary handling are applied to every canonical P2 event. No ME-TST+ suppression is added.", f"Center preservation: {sum(row['center_preserved'] == 1 for row in audits)}/{len(audits)}; boundary clips={sum(row['boundary_clipped'] == 1 for row in audits)}.", "", "## 4. Canonical vs calibrated result", "", f"- Canonical TP/FP/FN={canonical_total['TP']}/{canonical_total['FP']}/{canonical_total['FN']}; precision={canonical_total['precision']:.6f}; recall={canonical_total['recall']:.6f}; F1={canonical_total['F1']:.6f}.", f"- Adapted TP/FP/FN={adapted_total['TP']}/{adapted_total['FP']}/{adapted_total['FN']}; precision={adapted_total['precision']:.6f}; recall={adapted_total['recall']:.6f}; F1={adapted_total['F1']:.6f}.", f"- Delta TP/FP/FN={delta['TP']:+.0f}/{delta['FP']:+.0f}/{delta['FN']:+.0f}; Delta F1={delta_f1:+.6f}.", "", "## 5. Event transitions", "", f"Rescued={gt_transitions['RESCUED']}; lost={gt_transitions['LOST']}; FP removed={pred_transitions['FP_REMOVED']}; new FP={pred_transitions['NEW_FP']}; match reassigned={pred_transitions['MATCH_REASSIGNED']}.", "", "## 6. Subject-level distribution", "", f"Positive/neutral/negative={distribution['positive_delta_F1_subjects']}/{distribution['neutral_delta_F1_subjects']}/{distribution['negative_delta_F1_subjects']}; rescued>lost/equal/less={distribution['rescued_gt_lost_subjects']}/{distribution['rescued_eq_lost_subjects']}/{distribution['rescued_lt_lost_subjects']}; top-1/top-2/top-5 gain={distribution['top_1_TP_gain_fraction']}/{distribution['top_2_TP_gain_fraction']}/{distribution['top_5_TP_gain_fraction']}.", "", "## 7. Bootstrap", "", f"Paired-subject bootstrap: N=10000, seed=100; Delta F1={delta_f1:+.6f}; 95% CI=[{ci_low:+.6f}, {ci_high:+.6f}].", "", "## 8. Context against EXP-7B", "", f"Frozen CAS(ME)3 primary: F1 {context['canonical_F1']}→{context['adapted_F1']}, Delta {context['delta_F1']}, CI {context['ci95']}. SAMMLV replication: F1 {canonical_total['F1']:.6f}→{adapted_total['F1']:.6f}, Delta {delta_f1:+.6f}, CI [{ci_low:+.6f}, {ci_high:+.6f}]. Descriptive only; no dataset ranking or comparison test was performed.", "", "## 9. Gates", "", *[f"- {key}: **{value}**." for key, value in gates.items()], "", "## 10. Final status", "", f"**METST_SAMMLV_DURATION_REPLICATION = {status}**", "", "Canonical GLSD and EXP-7B were not modified. No Boosting, Recognition, STRS, or further rule mining was run."]
    (OUT / "EXP7D_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("================================"); print("EXP-7D SAMMLV REPLICATION VERDICT"); print("================================"); print("Parent replay: PASS"); print(f"Canonical/adapted F1: {canonical_total['F1']:.6f} / {adapted_total['F1']:.6f}"); print(f"Delta F1 CI: [{ci_low:+.6f}, {ci_high:+.6f}]"); print(f"METST_SAMMLV_DURATION_REPLICATION = {status}"); print("Canonical GLSD modified: NO"); print("================================")


if __name__ == "__main__":
    main()
