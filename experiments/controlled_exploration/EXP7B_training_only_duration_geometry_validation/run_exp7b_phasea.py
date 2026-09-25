"""EXP-7B Phase A: training-only duration calibration audit for ME-TST+/CAS(ME)3.

This script has no interval-writing or evaluation path.  It replays canonical
P0/P1/P2 only to verify the archived parent, then estimates each alpha_s from
records whose subject is explicitly different from s.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP7A = ROOT / "controlled_exploration/EXP7A_event_geometry_diagnostic"
BASE = ROOT / "controlled_exploration/baseline_snapshot/results/main_results.csv"
sys.path.insert(0, str(EXP7A))
from run_exp7a import canonical_sources, config, digest, fair, iou, read_csv, replay_video  # noqa: E402


BACKBONE = "metst"
DATASET = "casme3"
SETTING = "ME-TST+/CAS(ME)3"
LOWER_CLIP, UPPER_CLIP = 0.5, 1.5
NEAR_ONE_TOLERANCE = 0.05


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def formal_f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def event_json(event: dict) -> dict:
    return {"onset": event["interval"][0], "offset": event["interval"][1], "peak": event["peak"],
            "matched_gt": event["matched_gt"]}


def representative_pair(record: dict, state: dict, gt_index: int, outer_subject: str) -> dict | None:
    """Return one deterministic P0 representative; the assertion is the leakage guard."""
    assert record["subject"] != outer_subject, "outer-test GT reached alpha construction"
    gt = record["gt"][gt_index]
    gt_interval = (int(gt[0]), int(gt[2]))
    gt_center = (gt_interval[0] + gt_interval[1]) / 2
    eligible = [candidate for candidate in state["p0"] if gt_interval[0] <= candidate["peak"] <= gt_interval[1]]
    if not eligible:
        return None
    # Max IoU, min formal-center distance, then stable P0 order.  All intervals are closed.
    candidate = min(eligible, key=lambda item: (-iou(item["interval"], gt_interval),
                                                  abs((item["interval"][0] + item["interval"][1]) / 2 - gt_center),
                                                  item["index"]))
    event_length = candidate["interval"][1] - candidate["interval"][0] + 1
    gt_length = gt_interval[1] - gt_interval[0] + 1
    return {"outer_subject": outer_subject, "training_subject": record["subject"],
            "video": record["video"], "GT_id": f"{BACKBONE}|{DATASET}|{record['subject']}|{record['video']}|{gt_index}",
            "candidate_id": f"{BACKBONE}|{DATASET}|{record['subject']}|{record['video']}|P0|{candidate['index']}",
            "gt_index": gt_index, "GT_onset": gt_interval[0], "GT_offset": gt_interval[1],
            "candidate_peak": candidate["peak"], "formal_event_onset": candidate["interval"][0],
            "formal_event_offset": candidate["interval"][1], "L_G": gt_length, "L_E": event_length,
            "ratio": gt_length / event_length, "selection_rule": "max_IoU_then_min_center_distance_then_P0_order"}


def calibration_pairs(records: list[dict], state_by_video: dict[tuple[str, str], dict], outer_subject: str) -> list[dict]:
    """The only function that reads GT for alpha; its input is training records only."""
    assert all(record["subject"] != outer_subject for record in records), "outer-test record passed to alpha path"
    pairs = []
    for record in records:
        state = state_by_video[record["subject"], record["video"]]
        for gt_index in range(len(record["gt"])):
            pair = representative_pair(record, state, gt_index, outer_subject)
            if pair is not None:
                pairs.append(pair)
    return pairs


def alpha_for_outer(subjects: list[str], records: list[dict], outer_subject: str, outer_config, outer_k: int,
                    p0_cache: dict) -> tuple[list[dict], list[dict], dict]:
    training_subjects = [subject for subject in subjects if subject != outer_subject]
    training_records = [record for record in records if record["subject"] != outer_subject]
    # Every training P0 is replayed with *this* outer fold's archived config and k.
    # Hence neither candidate geometry nor its frozen selection can contain outer-subject GT.
    geometry_key = (outer_config.reference, outer_config.radius, outer_k)
    training_states = {}
    for record in training_records:
        cache_key = (*geometry_key, record["subject"], record["video"])
        # P0 geometry depends only on reference, radius, k, and frozen curve; threshold is not part of P0.
        if cache_key not in p0_cache:
            p0_cache[cache_key] = replay_video(record, outer_config, BACKBONE, outer_k)
        training_states[record["subject"], record["video"]] = p0_cache[cache_key]
    pairs = calibration_pairs(training_records, training_states, outer_subject)
    per_subject = []
    for subject in training_subjects:
        values = [float(pair["ratio"]) for pair in pairs if pair["training_subject"] == subject]
        if values:
            alpha = float(np.median(values))
            per_subject.append({"outer_subject": outer_subject, "training_subject": subject, "n_valid_pairs": len(values),
                                "alpha_training_subject_median": alpha, "valid_training_subject": "YES"})
        else:
            per_subject.append({"outer_subject": outer_subject, "training_subject": subject, "n_valid_pairs": 0,
                                "alpha_training_subject_median": "", "valid_training_subject": "NO"})
    valid = [float(row["alpha_training_subject_median"]) for row in per_subject if row["valid_training_subject"] == "YES"]
    if not valid:
        raise RuntimeError(f"EXP7B_PHASEA_NO_VALID_TRAINING_PAIRS outer_subject={outer_subject}")
    pooled = float(np.median([float(pair["ratio"]) for pair in pairs]))
    raw = float(np.median(valid))
    clipped = min(UPPER_CLIP, max(LOWER_CLIP, raw))
    outer = {"outer_subject": outer_subject, "n_train_subjects": len(training_subjects),
             "n_valid_training_subjects": len(valid), "n_valid_pairs": len(pairs),
             "alpha_pooled_event_median": pooled, "alpha_subject_balanced_raw": raw,
             "alpha_subject_balanced_clipped": clipped, "hit_lower_clip": int(raw < LOWER_CLIP),
             "hit_upper_clip": int(raw > UPPER_CLIP), "outer_test_GT_accessed_for_alpha": "NO"}
    return pairs, per_subject, outer


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    records, subjects, _, input_path = fair.load_data(BACKBONE, DATASET)
    outer_k, _ = fair.fold_priors(records, subjects, BACKBONE)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    selections, expected_by_subject, saved = canonical_sources(BACKBONE, DATASET)
    states, replay_rows = {}, []
    total = np.zeros(3, dtype=int)
    parent_ok = True
    for record in records:
        subject, video = record["subject"], record["video"]
        state = replay_video(record, config(selections[subject]), BACKBONE, int(outer_k[subject_index[subject]]))
        states[subject, video] = state
    for subject in subjects:
        subject_states = [state for (current, _), state in states.items() if current == subject]
        observed = np.sum([state["counts"] for state in subject_states], axis=0)
        prediction_ok = all([event_json(event) for event in states[subject, video]["p2"]] == saved[subject, video]
                            for current, video in states if current == subject)
        counts_ok = tuple(observed) == expected_by_subject[subject]
        parent_ok &= prediction_ok and counts_ok
        total += observed
        replay_rows.append({"scope": "outer_subject", "outer_subject": subject, "expected_TP": expected_by_subject[subject][0],
                            "expected_FP": expected_by_subject[subject][1], "expected_FN": expected_by_subject[subject][2],
                            "replay_TP": int(observed[0]), "replay_FP": int(observed[1]), "replay_FN": int(observed[2]),
                            "canonical_prediction_check": "PASS" if prediction_ok else "FAIL", "counts_check": "PASS" if counts_ok else "FAIL"})
    baseline = next(row for row in read_csv(BASE) if row["Backbone"] == BACKBONE and row["Dataset"] == DATASET)
    archived = tuple(int(baseline[f"GL_Skill_{key}"]) for key in ("TP", "FP", "FN"))
    archived_f1 = float(baseline["GL_Skill_F1"])
    replay_f1 = formal_f1(*map(int, total))
    total_ok = tuple(total) == archived and np.isclose(replay_f1, archived_f1, atol=1e-15)
    parent_ok &= total_ok
    replay_rows.append({"scope": "TOTAL", "outer_subject": "ALL", "expected_TP": archived[0], "expected_FP": archived[1], "expected_FN": archived[2],
                        "replay_TP": int(total[0]), "replay_FP": int(total[1]), "replay_FN": int(total[2]),
                        "archived_F1": archived_f1, "replay_F1": replay_f1, "canonical_prediction_check": "PASS" if parent_ok else "FAIL",
                        "counts_check": "PASS" if total_ok else "FAIL"})
    parent_fields = ["scope", "outer_subject", "expected_TP", "expected_FP", "expected_FN", "replay_TP", "replay_FP", "replay_FN",
                     "archived_F1", "replay_F1", "canonical_prediction_check", "counts_check"]
    write_csv("parent_replay_check.csv", replay_rows, parent_fields)
    if not parent_ok:
        raise RuntimeError("EXP7B_PARENT_REPLAY_FAILED")
    pair_rows, subject_rows, outer_rows, leakage_rows, p0_cache = [], [], [], [], {}
    for outer_subject in subjects:
        pairs, per_subject, outer = alpha_for_outer(subjects, records, outer_subject,
                                                     config(selections[outer_subject]), int(outer_k[subject_index[outer_subject]]), p0_cache)
        pair_rows.extend(pairs)
        subject_rows.extend(per_subject)
        outer_rows.append(outer)
        leakage_rows.append({"outer_subject": outer_subject, "training_record_count": sum(r["subject"] != outer_subject for r in records),
                             "outer_record_count_excluded": sum(r["subject"] == outer_subject for r in records),
                             "outer_fold_config_and_k_used_for_all_training_P0": "PASS",
                             "calibration_function_input_excludes_outer_subject": "PASS",
                             "representative_pair_assertion": "PASS", "outer_test_GT_accessed_for_alpha": "NO"})
    alpha_values = np.array([float(row["alpha_subject_balanced_clipped"]) for row in outer_rows])
    q1, median, q3 = np.quantile(alpha_values, [.25, .5, .75])
    summary = [{"setting": SETTING, "fold_count": len(outer_rows), "median": float(median), "IQR_Q1": float(q1), "IQR_Q3": float(q3),
                "IQR_width": float(q3 - q1), "min": float(np.min(alpha_values)), "max": float(np.max(alpha_values)),
                "alpha_lt_1_folds": int(np.sum(alpha_values < 1 - NEAR_ONE_TOLERANCE)),
                "alpha_approx_1_folds": int(np.sum(np.abs(alpha_values - 1) <= NEAR_ONE_TOLERANCE)),
                "alpha_gt_1_folds": int(np.sum(alpha_values > 1 + NEAR_ONE_TOLERANCE)),
                "lower_clip_frequency": int(sum(row["hit_lower_clip"] for row in outer_rows)),
                "upper_clip_frequency": int(sum(row["hit_upper_clip"] for row in outer_rows)),
                "near_one_definition": f"abs(alpha-1)<={NEAR_ONE_TOLERANCE}"}]
    pair_fields = ["outer_subject", "training_subject", "video", "GT_id", "candidate_id", "gt_index", "GT_onset", "GT_offset", "candidate_peak",
                   "formal_event_onset", "formal_event_offset", "L_G", "L_E", "ratio", "selection_rule"]
    subject_fields = ["outer_subject", "training_subject", "n_valid_pairs", "alpha_training_subject_median", "valid_training_subject"]
    outer_fields = ["outer_subject", "n_train_subjects", "n_valid_training_subjects", "n_valid_pairs", "alpha_pooled_event_median",
                    "alpha_subject_balanced_raw", "alpha_subject_balanced_clipped", "hit_lower_clip", "hit_upper_clip", "outer_test_GT_accessed_for_alpha"]
    write_csv("training_pair_ratios.csv", pair_rows, pair_fields)
    write_csv("per_training_subject_alpha.csv", subject_rows, subject_fields)
    write_csv("outer_alpha_calibration.csv", outer_rows, outer_fields)
    write_csv("alpha_summary.csv", summary, list(summary[0]))
    write_csv("leakage_audit.csv", leakage_rows, list(leakage_rows[0]))
    protocol = {"experiment": "EXP-7B Phase A", "setting": SETTING, "phase": "calibration-only", "interval_semantics": "discrete closed: L=end-start+1",
                "alpha": "per-training-subject median of L_G/L_E, then median across valid training subjects, clipped to [0.5, 1.5]",
                "outer_test_GT_in_alpha": False, "forbidden": ["duration application", "new prediction", "formal F1", "GLSD modification", "Boosting", "secondary replication"]}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"experiment": "EXP-7B Phase A", "source_sha256": {"run_exp7b_phasea.py": digest(Path(__file__)),
                "EXP7A_run": digest(EXP7A / "run_exp7a.py")}, "frozen_input": str(input_path.relative_to(ROOT)),
                "frozen_input_sha256": digest(Path(input_path)), "new_prediction_written": False, "new_F1_computed": False}
    (OUT / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# EXP-7B — Training-Only Duration Geometry Validation", "", "## Phase A only", "",
              "- Scope: ME-TST+ / CAS(ME)3 only. No duration-adjusted interval was generated, applied, matched, or evaluated.",
              "- Interval semantics are discrete closed intervals: `L = end - start + 1`.",
              f"- Canonical parent replay: **PASS** — TP/FP/FN={tuple(map(int, total))}; F1={replay_f1:.15f}, matching archived F1={archived_f1:.15f}.",
              "", "## Training-only calibration", "",
              "For each outer subject, only records from all other subjects enter `calibration_pairs`. Their P0 candidates are replayed using that outer fold's frozen config and k. For each training GT, the representative candidate has a peak in the closed GT interval and is selected by maximum IoU, then minimum formal-center distance, then P0 order.",
              "", "The calibration is subject-balanced: each training subject contributes one median `L_G/L_E`; the outer alpha is the median of those subject medians, then clipped to [0.5, 1.5]. The pooled-event median is audit-only.",
              "", "## Leakage audit", "",
              "`calibration_pairs` asserts that every input record has `subject != outer_subject`; `representative_pair` repeats that assertion before reading any GT. Every training P0 is replayed with the current outer fold's stored config and k, not another subject's fold. Runtime audit passed for every outer fold, and no outer-test GT was passed to alpha construction.",
              "", "## Alpha stability", "",
              f"- Folds: {len(outer_rows)}; median={median:.6f}; IQR=[{q1:.6f}, {q3:.6f}]; range=[{np.min(alpha_values):.6f}, {np.max(alpha_values):.6f}].",
              f"- alpha<1: {summary[0]['alpha_lt_1_folds']}; alpha≈1: {summary[0]['alpha_approx_1_folds']}; alpha>1: {summary[0]['alpha_gt_1_folds']}; clips: lower={summary[0]['lower_clip_frequency']}, upper={summary[0]['upper_clip_frequency']}.",
              "", "Phase B executed: **NO**. New formal prediction: **NO**. New formal F1: **NO**. GLSD modified: **NO**. Boosting/secondary replication: **NO**."]
    (OUT / "EXP7B_PHASEA_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("================================")
    print("EXP-7B PHASE-A CALIBRATION VERDICT")
    print("================================")
    print("Parent replay: PASS")
    print(f"Outer folds: {len(outer_rows)}")
    print(f"Subject-balanced alpha median/IQR: {median:.6f} / [{q1:.6f}, {q3:.6f}]")
    print(f"alpha<1 / alpha≈1 / alpha>1: {summary[0]['alpha_lt_1_folds']} / {summary[0]['alpha_approx_1_folds']} / {summary[0]['alpha_gt_1_folds']}")
    print("Outer-test GT used for alpha: NO")
    print("Phase B executed: NO")
    print("Formal prediction changed: NO")
    print("Formal F1 changed: NO")
    print("Canonical GLSD modified: NO")
    print("================================")


if __name__ == "__main__":
    main()
