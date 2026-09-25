"""EXP-7C controlled BoostingVRME/CAS(ME)3 replication of frozen EXP-7B protocol."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP7A = ROOT / "controlled_exploration/EXP7A_event_geometry_diagnostic"
EXP7B = ROOT / "controlled_exploration/EXP7B_training_only_duration_geometry_validation"
BASE = ROOT / "controlled_exploration/baseline_snapshot/results/main_results.csv"
sys.path.extend((str(EXP7A), str(EXP7B)))
from run_exp7a import canonical_sources, config, digest, fair, iou, read_csv, replay_video  # noqa: E402
from run_exp7b_phaseb import blind_fixed_center_adapter, bootstrap, formal_match, metrics  # noqa: E402


BACKBONE, DATASET, SETTING = "boostingvrme", "casme3", "BoostingVRME/CAS(ME)3"
LOWER_CLIP, UPPER_CLIP, NEAR_ONE = .5, 1.5, .05


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def representative_pair(record: dict, state: dict, gt_index: int, outer_subject: str) -> dict | None:
    assert record["subject"] != outer_subject, "outer-test GT reached calibration pairing"
    gt = record["gt"][gt_index]
    g = (int(gt[0]), int(gt[2]))
    center = (g[0] + g[1]) / 2
    eligible = [x for x in state["p0"] if g[0] <= x["peak"] <= g[1]]
    if not eligible:
        return None
    candidate = min(eligible, key=lambda x: (-iou(x["interval"], g), abs((x["interval"][0] + x["interval"][1]) / 2 - center), x["index"]))
    lg, le = g[1] - g[0] + 1, candidate["interval"][1] - candidate["interval"][0] + 1
    return {"outer_subject": outer_subject, "training_subject": record["subject"], "video": record["video"],
            "GT_id": f"{BACKBONE}|{DATASET}|{record['subject']}|{record['video']}|{gt_index}",
            "candidate_id": f"{BACKBONE}|{DATASET}|{record['subject']}|{record['video']}|P0|{candidate['index']}",
            "peak": candidate["peak"], "formal_start": candidate["interval"][0], "formal_end": candidate["interval"][1],
            "L_G": lg, "L_E": le, "ratio": lg / le,
            "selection_rule": "max_IoU_then_min_center_distance_then_P0_order"}


def alpha_for_outer(subjects: list[str], records: list[dict], outer_subject: str, outer_config, outer_k: int, cache: dict) -> tuple[list[dict], list[dict], dict]:
    training = [record for record in records if record["subject"] != outer_subject]
    assert all(record["subject"] != outer_subject for record in training), "outer test record passed to alpha path"
    geometry_key, pairs = (outer_config.reference, outer_config.radius, outer_k), []
    for record in training:
        key = (*geometry_key, record["subject"], record["video"])
        if key not in cache:
            cache[key] = replay_video(record, outer_config, BACKBONE, outer_k)
        for gt_index in range(len(record["gt"])):
            pair = representative_pair(record, cache[key], gt_index, outer_subject)
            if pair is not None:
                pairs.append(pair)
    subject_rows, medians = [], []
    for training_subject in (s for s in subjects if s != outer_subject):
        values = [float(row["ratio"]) for row in pairs if row["training_subject"] == training_subject]
        if values:
            alpha = float(np.median(values)); medians.append(alpha)
            subject_rows.append({"outer_subject": outer_subject, "training_subject": training_subject, "n_valid_pairs": len(values),
                                 "alpha_training_subject_median": alpha, "valid_training_subject": "YES"})
        else:
            subject_rows.append({"outer_subject": outer_subject, "training_subject": training_subject, "n_valid_pairs": 0,
                                 "alpha_training_subject_median": "", "valid_training_subject": "NO"})
    if not medians:
        raise RuntimeError("EXP7C_NO_VALID_TRAINING_PAIRS")
    raw = float(np.median(medians))
    outer = {"outer_subject": outer_subject, "n_train_subjects": len(subjects) - 1, "n_valid_training_subjects": len(medians),
             "n_valid_pairs": len(pairs), "alpha_pooled_event_median": float(np.median([x["ratio"] for x in pairs])),
             "alpha_subject_balanced_raw": raw, "alpha_subject_balanced_clipped": min(UPPER_CLIP, max(LOWER_CLIP, raw)),
             "hit_lower_clip": int(raw < LOWER_CLIP), "hit_upper_clip": int(raw > UPPER_CLIP)}
    return pairs, subject_rows, outer


def conflict_matrix(events: list[dict], k: int) -> np.ndarray:
    peaks = np.array([x["peak"] for x in events], dtype=int)
    intervals = np.array([x["interval"] for x in events], dtype=int).reshape(-1, 2)
    if not len(events):
        return np.zeros((0, 0), dtype=bool)
    starts, ends = intervals.T
    # Exact archived Boosting conflict semantics: open-span overlap ratio plus peak distance.
    intersection = np.maximum(0, np.minimum(ends[:, None], ends) - np.maximum(starts[:, None], starts))
    span = np.maximum(ends[:, None], ends) - np.minimum(starts[:, None], starts)
    overlap = np.divide(intersection, span, out=np.zeros_like(intersection, dtype=float), where=span > 0)
    return (overlap >= .2) | (np.abs(peaks[:, None] - peaks) <= k)


def original_conflict(events: list[dict], k: int) -> tuple[list[dict], np.ndarray]:
    matrix = conflict_matrix(events, k)
    keep = np.ones(len(events), dtype=bool)
    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            if keep[left] and keep[right] and matrix[left, right]:
                keep[right] = False
    return [event for index, event in enumerate(events) if keep[index]], matrix


def transition_rows(subject: str, video: str, canonical: list[dict], adapted: list[dict], canonical_p1: list[dict], adapted_p1: list[dict], gt: list[list[int]], audits: dict[int, dict]) -> list[dict]:
    rows = []
    can_p2, adp_p2 = {x["index"]: x for x in canonical}, {x["index"]: x for x in adapted}
    can_gt, adp_gt = ({x["matched_gt"]: x for x in values if x["matched_gt"] >= 0} for values in (canonical, adapted))
    for gt_index in range(len(gt)):
        old, new = can_gt.get(gt_index), adp_gt.get(gt_index)
        kind = "RESCUED" if old is None and new is not None else "LOST" if old is not None and new is None else "TP_STABLE" if old else "FN_STABLE"
        rows.append({"record_type": "GT", "outer_subject": subject, "video": video, "GT_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|{gt_index}", "prediction_id": "",
                     "canonical_status": "TP" if old else "FN", "adapted_status": "TP" if new else "FN", "transition_type": kind,
                     "canonical_interval": "" if old is None else f"{old['interval'][0]}:{old['interval'][1]}", "adapted_interval": "" if new is None else f"{new['interval'][0]}:{new['interval'][1]}",
                     "canonical_match_id": "" if old is None else old["matched_gt"], "adapted_match_id": "" if new is None else new["matched_gt"],
                     "canonical_iou": "" if old is None else old["best_iou"], "adapted_iou": "" if new is None else new["best_iou"], "alpha_s": audits[next(iter(audits))]["alpha_s"] if audits else "",
                     "boundary_clipped": 0, "conflict_changed": int(old is not None and new is not None and old["index"] != new["index"])})
    all_ids = sorted({x["index"] for x in canonical_p1} | {x["index"] for x in adapted_p1})
    for index in all_ids:
        old, new, audit = can_p2.get(index), adp_p2.get(index), audits[index]
        if old is None or new is None:
            kind = "CONFLICT_RULE_CHANGED"
        elif old["matched_gt"] < 0 and new["matched_gt"] >= 0:
            kind = "FP_REMOVED"
        elif old["matched_gt"] >= 0 and new["matched_gt"] < 0:
            kind = "NEW_FP"
        elif old["matched_gt"] != new["matched_gt"]:
            kind = "MATCH_REASSIGNED"
        elif audit["boundary_clipped"]:
            kind = "BOUNDARY_CLIPPED_EVENT"
        else:
            kind = "EVENT_STABLE"
        rows.append({"record_type": "PREDICTION", "outer_subject": subject, "video": video, "GT_id": "", "prediction_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|P1|{index}",
                     "canonical_status": "CONFLICT_REMOVED" if old is None else ("TP" if old["matched_gt"] >= 0 else "FP"),
                     "adapted_status": "CONFLICT_REMOVED" if new is None else ("TP" if new["matched_gt"] >= 0 else "FP"), "transition_type": kind,
                     "canonical_interval": "" if old is None else f"{old['interval'][0]}:{old['interval'][1]}", "adapted_interval": "" if new is None else f"{new['interval'][0]}:{new['interval'][1]}",
                     "canonical_match_id": "" if old is None else old["matched_gt"], "adapted_match_id": "" if new is None else new["matched_gt"],
                     "canonical_iou": "" if old is None else old["best_iou"], "adapted_iou": "" if new is None else new["best_iou"], "alpha_s": audit["alpha_s"],
                     "boundary_clipped": audit["boundary_clipped"], "conflict_changed": int((old is None) != (new is None) or (old and new and old["matched_gt"] != new["matched_gt"]) )})
    return rows


def primary_context() -> dict:
    report = (EXP7B / "EXP7B_PHASEB_ANALYSIS.md").read_text(encoding="utf-8")
    alpha = read_csv(EXP7B / "alpha_summary.csv")[0]
    number = r"[0-9]+(?:\.[0-9]+)?"
    canonical = re.search(rf"- Canonical: TP/FP/FN=\d+/\d+/\d+; precision={number}; recall={number}; F1=({number})", report)
    adapted = re.search(rf"- Adapted: TP/FP/FN=\d+/\d+/\d+; precision={number}; recall={number}; F1=({number})", report)
    delta = re.search(rf"F1=([+-]{number})", report)
    transition = re.search(r"Rescued=(\d+); lost=(\d+); FP removed=(\d+); new FP=(\d+)", report)
    distribution = re.search(r"Positive/neutral/negative delta-F1 subjects: (\d+)/(\d+)/(\d+)", report)
    ci = re.search(r"95% CI=\[([+-][0-9.]+), ([+-][0-9.]+)\]", report)
    return {"pipeline": "ME-TST+/CAS(ME)3", "alpha_median": alpha["median"], "canonical_F1": canonical.group(1), "adapted_F1": adapted.group(1), "delta_F1": delta.group(1),
            "ci95": f"{ci.group(1)}:{ci.group(2)}", "rescued": transition.group(1), "lost": transition.group(2), "fp_removed": transition.group(3), "new_fp": transition.group(4),
            "positive_neutral_negative": "/".join(distribution.groups())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    records, subjects, _, _ = fair.load_data(BACKBONE, DATASET)
    outer_k, _ = fair.fold_priors(records, subjects, BACKBONE)
    subject_index = {s: i for i, s in enumerate(subjects)}
    selections, expected, saved = canonical_sources(BACKBONE, DATASET)
    states, replay_rows, total = {}, [], np.zeros(3, dtype=int)
    integrity = True
    for record in records:
        subject, video = record["subject"], record["video"]
        states[subject, video] = replay_video(record, config(selections[subject]), BACKBONE, int(outer_k[subject_index[subject]]))
    for subject in subjects:
        own = [(video, state) for (current, video), state in states.items() if current == subject]
        observed = np.sum([state["counts"] for _, state in own], axis=0)
        pred_ok = all([{"onset": x["interval"][0], "offset": x["interval"][1], "peak": x["peak"], "matched_gt": x["matched_gt"]} for x in state["p2"]] == saved[subject, video] for video, state in own)
        count_ok = tuple(observed) == expected[subject]; integrity &= pred_ok and count_ok; total += observed
        replay_rows.append({"scope": "outer_subject", "outer_subject": subject, "expected_TP": expected[subject][0], "expected_FP": expected[subject][1], "expected_FN": expected[subject][2],
                            "replay_TP": int(observed[0]), "replay_FP": int(observed[1]), "replay_FN": int(observed[2]), "prediction_check": "PASS" if pred_ok else "FAIL", "counts_check": "PASS" if count_ok else "FAIL"})
    base = next(row for row in read_csv(BASE) if row["Backbone"] == BACKBONE and row["Dataset"] == DATASET)
    archive, archive_f1 = tuple(int(base[f"GL_Skill_{x}"]) for x in ("TP", "FP", "FN")), float(base["GL_Skill_F1"])
    integrity &= tuple(total) == archive and np.isclose(metrics(total)["F1"], archive_f1, atol=1e-15)
    replay_rows.append({"scope": "TOTAL", "outer_subject": "ALL", "expected_TP": archive[0], "expected_FP": archive[1], "expected_FN": archive[2], "replay_TP": int(total[0]), "replay_FP": int(total[1]), "replay_FN": int(total[2]), "archived_F1": archive_f1, "replay_F1": metrics(total)["F1"], "prediction_check": "PASS" if integrity else "FAIL", "counts_check": "PASS" if integrity else "FAIL"})
    parent_fields = ["scope", "outer_subject", "expected_TP", "expected_FP", "expected_FN", "replay_TP", "replay_FP", "replay_FN", "archived_F1", "replay_F1", "prediction_check", "counts_check"]
    write_csv("parent_replay_check.csv", replay_rows, parent_fields)
    if not integrity:
        raise RuntimeError("EXP7C_PARENT_REPLAY_FAILED")
    pair_rows, alpha_rows, outer_alpha, cache, leakage = [], [], [], {}, []
    for outer in subjects:
        pairs, per_subject, fold = alpha_for_outer(subjects, records, outer, config(selections[outer]), int(outer_k[subject_index[outer]]), cache)
        pair_rows += pairs; alpha_rows += per_subject; outer_alpha.append(fold)
        leakage.append({"outer_subject": outer, "training_record_excludes_outer": "PASS", "GT_assertion_before_pairing": "PASS", "outer_test_GT_to_alpha": "NO", "outer_test_GT_to_adapter": "NO", "outer_test_GT_to_conflict": "NO"})
    if any(row["outer_test_GT_to_alpha"] != "NO" for row in leakage):
        raise RuntimeError("EXP7C_LEAKAGE_AUDIT_FAILED")
    values = np.array([row["alpha_subject_balanced_clipped"] for row in outer_alpha])
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    alpha_summary = [{"number_of_folds": len(values), "median_alpha": median, "IQR_Q1": q1, "IQR_Q3": q3, "IQR_width": q3-q1, "min": values.min(), "max": values.max(),
                      "alpha_lt_1_count": int(np.sum(values < 1-NEAR_ONE)), "alpha_approx_1_count": int(np.sum(np.abs(values-1) <= NEAR_ONE)), "alpha_gt_1_count": int(np.sum(values > 1+NEAR_ONE)),
                      "lower_clip_count": sum(x["hit_lower_clip"] for x in outer_alpha), "upper_clip_count": sum(x["hit_upper_clip"] for x in outer_alpha), "near_one_definition": "abs(alpha-1)<=0.05"}]
    audit_rows, conflict_rows, transitions, subject_rows, canonical_counts, adapted_counts = [], [], [], [], [], []
    record_map = {(r["subject"], r["video"]): r for r in records}
    for subject in subjects:
        alpha = next(x["alpha_subject_balanced_clipped"] for x in outer_alpha if x["outer_subject"] == subject)
        own_can, own_adapted, gt_total, subject_conflict = [], [], 0, []
        for (current, video), state in states.items():
            if current != subject: continue
            record, p1 = record_map[subject, video], [{"index": x["index"], "peak": x["peak"], "interval": x["interval"], "score": x["score"]} for x in state["p1"]]
            adapted_p1, audits = [], {}
            for event in p1:
                audit = blind_fixed_center_adapter(event, alpha, len(record["curve"]))
                if not audit["center_preserved"]: raise RuntimeError("EXP7C_CENTER_PRESERVATION_FAILED")
                audits[event["index"]] = audit; adapted_p1.append({**event, "interval": (audit["adapted_start"], audit["adapted_end"])})
                audit_rows.append({"outer_subject": subject, "event_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|P1|{event['index']}", "video": video, **audit})
            canonical_p2, canonical_matrix = original_conflict(p1, state["k"])
            adapted_p2, adapted_matrix = original_conflict(adapted_p1, state["k"])
            if [x["index"] for x in canonical_p2] != [x["index"] for x in state["p2"]]: raise RuntimeError("EXP7C_CONFLICT_REPLAY_FAILED")
            changed_pairs = int(np.sum(np.triu(canonical_matrix != adapted_matrix, 1)))
            conflict_rows.append({"outer_subject": subject, "video": video, "prediction_count_before_adaptation": len(p1), "prediction_count_after_interval_adaptation_before_conflict": len(adapted_p1), "canonical_prediction_count_after_conflict": len(canonical_p2), "prediction_count_after_canonical_conflict_rule": len(adapted_p2), "conflict_removed_count": len(adapted_p1)-len(adapted_p2), "conflict_changed_pair_count": changed_pairs})
            canonical, adapted = formal_match(canonical_p2, record["gt"]), formal_match(adapted_p2, record["gt"])
            own_can += canonical; own_adapted += adapted; gt_total += len(record["gt"]); subject_conflict.append(conflict_rows[-1])
            transitions += transition_rows(subject, video, canonical, adapted, canonical_p2, adapted_p2, record["gt"], audits)
        can_value = np.array([sum(x["matched_gt"] >= 0 for x in own_can), len(own_can)-sum(x["matched_gt"] >= 0 for x in own_can), gt_total-sum(x["matched_gt"] >= 0 for x in own_can)])
        adp_value = np.array([sum(x["matched_gt"] >= 0 for x in own_adapted), len(own_adapted)-sum(x["matched_gt"] >= 0 for x in own_adapted), gt_total-sum(x["matched_gt"] >= 0 for x in own_adapted)])
        canonical_counts.append(can_value); adapted_counts.append(adp_value)
        own_gt, own_pred = [x for x in transitions if x["outer_subject"] == subject and x["record_type"] == "GT"], [x for x in transitions if x["outer_subject"] == subject and x["record_type"] == "PREDICTION"]
        subject_rows.append({"outer_subject": subject, "alpha_s": alpha, **{f"canonical_{k}":v for k,v in metrics(can_value).items()}, **{f"adapted_{k}":v for k,v in metrics(adp_value).items()}, "delta_TP": int(adp_value[0]-can_value[0]), "delta_FP": int(adp_value[1]-can_value[1]), "delta_FN": int(adp_value[2]-can_value[2]), "delta_F1": metrics(adp_value)["F1"]-metrics(can_value)["F1"], "rescued": sum(x["transition_type"]=="RESCUED" for x in own_gt), "lost": sum(x["transition_type"]=="LOST" for x in own_gt), "fp_removed": sum(x["transition_type"]=="FP_REMOVED" for x in own_pred), "new_fp": sum(x["transition_type"]=="NEW_FP" for x in own_pred), "match_reassigned": sum(x["conflict_changed"] for x in own_gt), "conflict_removed": sum(x["conflict_removed_count"] for x in subject_conflict), "boundary_clipped_count": sum(x["boundary_clipped"] for x in own_pred)})
    can_array, adp_array = np.asarray(canonical_counts), np.asarray(adapted_counts)
    if not np.array_equal(can_array.sum(axis=0), total): raise RuntimeError("EXP7C_CANONICAL_TOTAL_CHANGED")
    boot_rows, delta_f1, ci_low, ci_high = bootstrap(adp_array, can_array)
    can_total, adp_total = metrics(can_array.sum(axis=0)), metrics(adp_array.sum(axis=0)); delta = {k:adp_total[k]-can_total[k] for k in ("TP","FP","FN","precision","recall","F1")}
    distribution = {"positive_delta_F1_subjects":sum(x["delta_F1"]>0 for x in subject_rows), "neutral_delta_F1_subjects":sum(np.isclose(x["delta_F1"],0,atol=1e-15) for x in subject_rows), "negative_delta_F1_subjects":sum(x["delta_F1"]<0 for x in subject_rows), "rescued_gt_lost_subjects":sum(x["rescued"]>x["lost"] for x in subject_rows), "rescued_eq_lost_subjects":sum(x["rescued"]==x["lost"] for x in subject_rows), "rescued_lt_lost_subjects":sum(x["rescued"]<x["lost"] for x in subject_rows)}
    gain = int(delta["TP"]); positive_gains=sorted((max(0,x["delta_TP"]) for x in subject_rows),reverse=True)
    distribution.update({"total_net_TP_gain":gain, **{f"top_{n}_TP_gain_fraction":sum(positive_gains[:n])/gain if gain>0 else "NA" for n in (1,2,5)}})
    status = "SUPPORTED" if delta_f1>0 and ci_low>0 else "INCONCLUSIVE" if delta_f1>0 else "NOT_SUPPORTED"
    event_gt=Counter(x["transition_type"] for x in transitions if x["record_type"]=="GT"); event_pred=Counter(x["transition_type"] for x in transitions if x["record_type"]=="PREDICTION")
    accounting_ok = event_gt["RESCUED"]-event_gt["LOST"]==delta["TP"] and np.array_equal(can_array.sum(axis=0),total)
    gates={"GATE-A":"PASS", "GATE-B":"PASS", "GATE-C":"PASS" if all(x["center_preserved"]==1 for x in audit_rows) else "FAIL", "GATE-D":"PASS" if delta_f1>0 else "FAIL", "GATE-E":"PASS" if ci_low>0 else "FAIL", "GATE-F":"PASS" if accounting_ok else "FAIL", "GATE-G":"FACTS_REPORTED_NO_NEW_CUTOFF"}
    pair_fields=["outer_subject","training_subject","video","GT_id","candidate_id","peak","formal_start","formal_end","L_G","L_E","ratio","selection_rule"]
    alpha_fields=["outer_subject","training_subject","n_valid_pairs","alpha_training_subject_median","valid_training_subject"]
    outer_fields=["outer_subject","n_train_subjects","n_valid_training_subjects","n_valid_pairs","alpha_pooled_event_median","alpha_subject_balanced_raw","alpha_subject_balanced_clipped","hit_lower_clip","hit_upper_clip"]
    audit_fields=["outer_subject","event_id","video","original_start","original_end","original_length","original_center","alpha_s","target_length_float","target_length_integer","requested_length","adapted_start","adapted_end","adapted_length","adapted_center","center_preserved","boundary_clipped","video_length"]
    conflict_fields=["outer_subject","video","prediction_count_before_adaptation","prediction_count_after_interval_adaptation_before_conflict","canonical_prediction_count_after_conflict","prediction_count_after_canonical_conflict_rule","conflict_removed_count","conflict_changed_pair_count"]
    transition_fields=["record_type","outer_subject","video","GT_id","prediction_id","canonical_status","adapted_status","transition_type","canonical_interval","adapted_interval","canonical_match_id","adapted_match_id","canonical_iou","adapted_iou","alpha_s","boundary_clipped","conflict_changed"]
    subject_fields=["outer_subject","alpha_s","canonical_TP","canonical_FP","canonical_FN","canonical_precision","canonical_recall","canonical_F1","adapted_TP","adapted_FP","adapted_FN","adapted_precision","adapted_recall","adapted_F1","delta_TP","delta_FP","delta_FN","delta_F1","rescued","lost","fp_removed","new_fp","match_reassigned","conflict_removed","boundary_clipped_count"]
    write_csv("training_pair_ratios.csv",pair_rows,pair_fields); write_csv("per_training_subject_alpha.csv",alpha_rows,alpha_fields); write_csv("outer_alpha_calibration.csv",outer_alpha,outer_fields); write_csv("alpha_summary.csv",alpha_summary,list(alpha_summary[0])); write_csv("leakage_audit.csv",leakage,list(leakage[0])); write_csv("duration_adapter_audit.csv",audit_rows,audit_fields); write_csv("conflict_rule_audit.csv",conflict_rows,conflict_fields); write_csv("outer_subject_results.csv",subject_rows,subject_fields); write_csv("event_transition_accounting.csv",transitions,transition_fields); write_csv("bootstrap_delta_f1.csv",boot_rows,["iteration","delta_F1","N","seed"]); write_csv("subject_gain_distribution.csv",[distribution],list(distribution))
    context=primary_context(); context.update({"replication_pipeline":SETTING,"replication_alpha_median":median,"replication_canonical_F1":can_total["F1"],"replication_adapted_F1":adp_total["F1"],"replication_delta_F1":delta_f1,"replication_ci95":f"{ci_low}:{ci_high}","replication_rescued":event_gt["RESCUED"],"replication_lost":event_gt["LOST"],"replication_fp_removed":event_pred["FP_REMOVED"],"replication_new_fp":event_pred["NEW_FP"],"replication_positive_neutral_negative":f"{distribution['positive_delta_F1_subjects']}/{distribution['neutral_delta_F1_subjects']}/{distribution['negative_delta_F1_subjects']}"})
    write_csv("pipeline_contextual_comparison.csv",[context],list(context))
    protocol={"experiment":"EXP-7C","setting":SETTING,"outer_LOSO":True,"alpha":"subject-balanced per-fold Boosting-only estimator","same_as_EXP7B_rounding":True,"adapter":"fixed-center same-parity on all P1 candidates before original Boosting conflict","forbidden":["ME alpha transfer","alpha search","center/boundary correction","rule change","Recognition","STRS"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest={"experiment":"EXP-7C","script_sha256":digest(Path(__file__)),"canonical_modified":False,"exp7b_modified":False,"primary_context_source_sha256":digest(EXP7B/"EXP7B_PHASEB_ANALYSIS.md")}
    (OUT/"replay_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    report=["# EXP-7C — BoostingVRME/CAS(ME)3 Duration Replication","","## 1. Integrity","",f"- Canonical replay: **PASS** — TP/FP/FN={tuple(map(int,can_array.sum(axis=0)))}; F1={can_total['F1']:.15f}.","- Stored canonical outer config, k, scores, threshold, candidate list, and original conflict priority were replayed without selection.","- Leakage audit: all calibration inputs exclude their outer subject; adapter and conflict functions accept no test GT.","","## 2. Training-only calibration","",f"Subject-balanced Boosting-only alpha: median={median:.6f}; IQR=[{q1:.6f}, {q3:.6f}]; range=[{values.min():.6f}, {values.max():.6f}]; clips lower/upper={alpha_summary[0]['lower_clip_count']}/{alpha_summary[0]['upper_clip_count']}.","","## 3. Duration adapter","","Closed interval length is `end-start+1`. EXP-7B same-parity rounding and tie-breaking are reused exactly; formal center is fixed. Boundary handling only shortens a request to the largest legal same-center, same-parity interval.",f"Center-preserved events: {sum(x['center_preserved']==1 for x in audit_rows)}/{len(audit_rows)}; boundary clips={sum(x['boundary_clipped']==1 for x in audit_rows)}.","","## 4. Boosting conflict replay","",f"The archived chronological Boosting conflict rule was reapplied after adapting all P1 intervals. No threshold, score, priority, or conflict criterion changed. Adapted conflict removals={sum(x['conflict_removed_count'] for x in conflict_rows)}; changed conflict pairs={sum(x['conflict_changed_pair_count'] for x in conflict_rows)}.","","## 5. Canonical vs adapted result","",f"- Canonical TP/FP/FN={can_total['TP']}/{can_total['FP']}/{can_total['FN']}; precision={can_total['precision']:.6f}; recall={can_total['recall']:.6f}; F1={can_total['F1']:.6f}.",f"- Adapted TP/FP/FN={adp_total['TP']}/{adp_total['FP']}/{adp_total['FN']}; precision={adp_total['precision']:.6f}; recall={adp_total['recall']:.6f}; F1={adp_total['F1']:.6f}.",f"- Delta TP/FP/FN={delta['TP']:+.0f}/{delta['FP']:+.0f}/{delta['FN']:+.0f}; Delta F1={delta_f1:+.6f}.","","## 6. Event transitions","",f"Rescued={event_gt['RESCUED']}; lost={event_gt['LOST']}; FP removed={event_pred['FP_REMOVED']}; new FP={event_pred['NEW_FP']}; match reassigned={event_pred['MATCH_REASSIGNED']}; conflict-rule changes={event_pred['CONFLICT_RULE_CHANGED']}.","","## 7. Subject-level generalization","",f"Positive/neutral/negative={distribution['positive_delta_F1_subjects']}/{distribution['neutral_delta_F1_subjects']}/{distribution['negative_delta_F1_subjects']}; rescued>lost/equal/less={distribution['rescued_gt_lost_subjects']}/{distribution['rescued_eq_lost_subjects']}/{distribution['rescued_lt_lost_subjects']}; top-1/top-2/top-5 TP concentration={distribution['top_1_TP_gain_fraction']}/{distribution['top_2_TP_gain_fraction']}/{distribution['top_5_TP_gain_fraction']}.","","## 8. Bootstrap","",f"Paired-subject bootstrap: N=10000, seed=100; Delta F1={delta_f1:+.6f}; 95% CI=[{ci_low:+.6f}, {ci_high:+.6f}].","","## 9. Context against EXP-7B primary result","",f"Frozen ME-TST+ primary F1: {context['canonical_F1']}→{context['adapted_F1']} (Delta {context['delta_F1']}, CI {context['ci95']}); Boosting replication: {can_total['F1']:.6f}→{adp_total['F1']:.6f} (Delta {delta_f1:+.6f}, CI [{ci_low:+.6f}, {ci_high:+.6f}]). This is descriptive only; no cross-pipeline ranking or significance test was performed.","","## 10. Gates","",*[f"- {k}: **{v}**." for k,v in gates.items()],"","## 11. Final replication status","",f"**BOOSTING_DURATION_REPLICATION = {status}**","","Canonical GLSD and EXP-7B were not modified. No SAMMLV, Recognition, STRS, or further rule mining was run."]
    (OUT/"EXP7C_ANALYSIS.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("================================"); print("EXP-7C BOOSTING REPLICATION VERDICT"); print("================================"); print("Parent replay: PASS"); print(f"Canonical/adapted F1: {can_total['F1']:.6f} / {adp_total['F1']:.6f}"); print(f"Delta F1 CI: [{ci_low:+.6f}, {ci_high:+.6f}]"); print(f"BOOSTING_DURATION_REPLICATION = {status}"); print("Canonical GLSD modified: NO"); print("================================")


if __name__ == "__main__": main()
