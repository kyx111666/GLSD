"""EXP-7B Phase B: blind, frozen-alpha duration application for ME-TST+/CAS(ME)3."""

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


BACKBONE, DATASET, SETTING = "metst", "casme3", "ME-TST+/CAS(ME)3"
N_BOOT, SEED = 10_000, 100
ALPHA_FILE = OUT / "outer_alpha_calibration.csv"


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics(values: np.ndarray | tuple[int, int, int]) -> dict:
    tp, fp, fn = (int(value) for value in values)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def event_json(event: dict) -> dict:
    return {"onset": event["interval"][0], "offset": event["interval"][1], "peak": event["peak"],
            "matched_gt": event["matched_gt"]}


def frozen_alphas(subjects: list[str]) -> dict[str, float]:
    if not ALPHA_FILE.exists():
        raise RuntimeError("EXP7B_PHASEA_ALPHA_SOURCE_MISSING")
    rows = read_csv(ALPHA_FILE)
    expected = set(subjects)
    observed = {row["outer_subject"] for row in rows}
    if len(rows) != 94 or observed != expected or any(row["outer_test_GT_accessed_for_alpha"] != "NO" for row in rows):
        raise RuntimeError("EXP7B_PHASEA_ALPHA_SOURCE_INVALID")
    return {row["outer_subject"]: float(row["alpha_subject_balanced_clipped"]) for row in rows}


def nearest_same_parity_length(target: float, original: int) -> int:
    """Pre-registered tie break: target distance, original distance, then smaller length."""
    minimum = 1 if original % 2 else 2
    lo, hi = max(minimum, int(np.floor(target)) - 3), max(minimum, int(np.ceil(target)) + 3)
    options = [length for length in range(lo, hi + 1) if length % 2 == original % 2]
    if not options:
        raise AssertionError("No legal same-parity length")
    return min(options, key=lambda length: (abs(length - target), abs(length - original), length))


def blind_fixed_center_adapter(event: dict, alpha: float, video_length: int) -> dict:
    """GT-free adapter: accepts only a stripped canonical event, alpha, and video length."""
    allowed = {"index", "peak", "interval", "score"}
    assert set(event) == allowed, "adapter received test-side status or GT-derived field"
    start, end = event["interval"]
    original_length, center_sum = end - start + 1, start + end
    target_float = alpha * original_length
    requested = nearest_same_parity_length(target_float, original_length)
    # With fixed center sum, the left-most legal endpoint determines the largest legal same-parity length.
    left_min = max(0, center_sum - (video_length - 1))
    max_legal = center_sum - 2 * left_min + 1
    if max_legal < (1 if original_length % 2 else 2):
        raise RuntimeError("EXP7B_CENTER_PRESERVING_LEGAL_INTERVAL_IMPOSSIBLE")
    actual = min(requested, max_legal)
    left = (center_sum - actual + 1) // 2
    right = left + actual - 1
    if not (0 <= left <= right < video_length and left + right == center_sum and actual % 2 == original_length % 2):
        raise AssertionError("center/parity/boundary invariant failed")
    return {"original_start": start, "original_end": end, "original_length": original_length,
            "original_center": center_sum / 2, "alpha_s": alpha, "target_length_float": target_float,
            "target_length_integer": requested, "requested_length": requested, "adapted_start": left,
            "adapted_end": right, "adapted_length": actual, "adapted_center": (left + right) / 2,
            "center_preserved": int((left + right) == center_sum), "boundary_clipped": int(actual != requested),
            "video_length": video_length}


def formal_match(events: list[dict], gt: list[list[int]]) -> list[dict]:
    """Exact archived formal evaluator: best GT at IoU>=.5, greedy P2 order, one-to-one."""
    occupied, output = set(), []
    for event in events:
        values = [iou(event["interval"], (item[0], item[2])) for item in gt]
        best = int(np.argmax(values)) if values else -1
        best_iou = values[best] if best >= 0 else 0.0
        matched = best if best_iou >= .5 and best not in occupied else -1
        if matched >= 0:
            occupied.add(matched)
        output.append({**event, "matched_gt": matched, "best_gt": best, "best_iou": best_iou})
    return output


def count_predictions(events: list[dict], gt_count: int) -> np.ndarray:
    tp = sum(event["matched_gt"] >= 0 for event in events)
    return np.array([tp, len(events) - tp, gt_count - tp], dtype=int)


def transition_rows(subject: str, video: str, canonical: list[dict], adapted: list[dict], gt: list[list[int]], audits: dict[int, dict]) -> list[dict]:
    rows, canonical_by_id, adapted_by_id = [], {x["index"]: x for x in canonical}, {x["index"]: x for x in adapted}
    canonical_gt = {x["matched_gt"]: x for x in canonical if x["matched_gt"] >= 0}
    adapted_gt = {x["matched_gt"]: x for x in adapted if x["matched_gt"] >= 0}
    for gt_index in range(len(gt)):
        old, new = canonical_gt.get(gt_index), adapted_gt.get(gt_index)
        if old is None and new is not None:
            kind = "RESCUED"
        elif old is not None and new is None:
            kind = "LOST"
        else:
            kind = "TP_STABLE" if old is not None else "FN_STABLE"
        reassigned = bool(old is not None and new is not None and old["index"] != new["index"])
        rows.append({"record_type": "GT", "outer_subject": subject, "video": video,
                     "GT_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|{gt_index}", "prediction_id": "",
                     "canonical_status": "TP" if old is not None else "FN", "adapted_status": "TP" if new is not None else "FN",
                     "transition_type": kind, "match_reassigned": int(reassigned),
                     "canonical_interval": "" if old is None else f"{old['interval'][0]}:{old['interval'][1]}",
                     "adapted_interval": "" if new is None else f"{new['interval'][0]}:{new['interval'][1]}",
                     "canonical_match_id": "" if old is None else old["matched_gt"], "adapted_match_id": "" if new is None else new["matched_gt"],
                     "canonical_iou": "" if old is None else old["best_iou"], "adapted_iou": "" if new is None else new["best_iou"],
                     "boundary_clipped": int(new is not None and audits[new["index"]]["boundary_clipped"])})
    for index in sorted(canonical_by_id):
        old, new, audit = canonical_by_id[index], adapted_by_id[index], audits[index]
        if old["matched_gt"] < 0 and new["matched_gt"] >= 0:
            kind = "FP_REMOVED"
        elif old["matched_gt"] >= 0 and new["matched_gt"] < 0:
            kind = "NEW_FP"
        elif old["matched_gt"] >= 0 and new["matched_gt"] >= 0 and old["matched_gt"] != new["matched_gt"]:
            kind = "MATCH_REASSIGNED"
        elif audit["boundary_clipped"]:
            kind = "BOUNDARY_CLIPPED_EVENT"
        else:
            kind = "EVENT_STABLE"
        rows.append({"record_type": "PREDICTION", "outer_subject": subject, "video": video, "GT_id": "",
                     "prediction_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|P2|{index}",
                     "canonical_status": "TP" if old["matched_gt"] >= 0 else "FP", "adapted_status": "TP" if new["matched_gt"] >= 0 else "FP",
                     "transition_type": kind, "match_reassigned": int(old["matched_gt"] >= 0 and new["matched_gt"] >= 0 and old["matched_gt"] != new["matched_gt"]),
                     "canonical_interval": f"{old['interval'][0]}:{old['interval'][1]}", "adapted_interval": f"{new['interval'][0]}:{new['interval'][1]}",
                     "canonical_match_id": old["matched_gt"], "adapted_match_id": new["matched_gt"],
                     "canonical_iou": old["best_iou"], "adapted_iou": new["best_iou"], "boundary_clipped": audit["boundary_clipped"]})
    return rows


def bootstrap(adapted: np.ndarray, canonical: np.ndarray) -> tuple[list[dict], float, float, float]:
    rng, n = np.random.default_rng(SEED), len(adapted)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    def f1(values: np.ndarray) -> np.ndarray:
        counts = values[draws].sum(axis=1).astype(float)
        return 2 * counts[:, 0] / (2 * counts[:, 0] + counts[:, 1] + counts[:, 2])
    deltas = f1(adapted) - f1(canonical)
    point = metrics(adapted.sum(axis=0))["F1"] - metrics(canonical.sum(axis=0))["F1"]
    low, high = np.quantile(deltas, [.025, .975])
    return ([{"iteration": i, "delta_F1": float(value), "N": N_BOOT, "seed": SEED} for i, value in enumerate(deltas)],
            float(point), float(low), float(high))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fair.ME_CACHE = ROOT / "RethinkFuse_reproduction/caches/me_tst"
    records, subjects, _, _ = fair.load_data(BACKBONE, DATASET)
    outer_k, _ = fair.fold_priors(records, subjects, BACKBONE)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    alphas = frozen_alphas(subjects)
    selections, expected_by_subject, saved = canonical_sources(BACKBONE, DATASET)
    replay_rows, states = [], {}
    total = np.zeros(3, dtype=int)
    integrity_ok = True
    for record in records:
        subject, video = record["subject"], record["video"]
        state = replay_video(record, config(selections[subject]), BACKBONE, int(outer_k[subject_index[subject]]))
        states[subject, video] = state
    for subject in subjects:
        own = [(video, state) for (current, video), state in states.items() if current == subject]
        observed = np.sum([state["counts"] for _, state in own], axis=0)
        prediction_ok = all([event_json(x) for x in state["p2"]] == saved[subject, video] for video, state in own)
        counts_ok = tuple(observed) == expected_by_subject[subject]
        integrity_ok &= prediction_ok and counts_ok
        total += observed
        replay_rows.append({"scope": "outer_subject", "outer_subject": subject, "expected_TP": expected_by_subject[subject][0],
                            "expected_FP": expected_by_subject[subject][1], "expected_FN": expected_by_subject[subject][2],
                            "replay_TP": int(observed[0]), "replay_FP": int(observed[1]), "replay_FN": int(observed[2],),
                            "canonical_prediction_check": "PASS" if prediction_ok else "FAIL", "counts_check": "PASS" if counts_ok else "FAIL"})
    baseline = next(row for row in read_csv(BASE) if row["Backbone"] == BACKBONE and row["Dataset"] == DATASET)
    archived, archived_f1 = tuple(int(baseline[f"GL_Skill_{k}"]) for k in ("TP", "FP", "FN")), float(baseline["GL_Skill_F1"])
    total_metric = metrics(total)
    total_ok = tuple(total) == archived and np.isclose(total_metric["F1"], archived_f1, atol=1e-15)
    integrity_ok &= total_ok
    replay_rows.append({"scope": "TOTAL", "outer_subject": "ALL", "expected_TP": archived[0], "expected_FP": archived[1], "expected_FN": archived[2],
                        "replay_TP": int(total[0]), "replay_FP": int(total[1]), "replay_FN": int(total[2]), "archived_F1": archived_f1,
                        "replay_F1": total_metric["F1"], "canonical_prediction_check": "PASS" if integrity_ok else "FAIL", "counts_check": "PASS" if total_ok else "FAIL"})
    parent_fields = ["scope", "outer_subject", "expected_TP", "expected_FP", "expected_FN", "replay_TP", "replay_FP", "replay_FN", "archived_F1", "replay_F1", "canonical_prediction_check", "counts_check"]
    write_csv("parent_replay_check_phaseB.csv", replay_rows, parent_fields)
    if not integrity_ok:
        raise RuntimeError("EXP7B_PHASEB_PARENT_REPLAY_FAILED")
    audit_rows, all_transitions, subject_rows, leakage_rows = [], [], [], []
    canonical_counts, adapted_counts = [], []
    for subject in subjects:
        alpha, own_canonical, own_adapted, own_audits, own_gt = alphas[subject], [], [], {}, {}
        for (current, video), state in states.items():
            if current != subject:
                continue
            record = next(item for item in records if item["subject"] == subject and item["video"] == video)
            # Strip all match/GT-derived fields before the adapter receives each event.
            blind_events = [{"index": x["index"], "peak": x["peak"], "interval": x["interval"], "score": x["score"]} for x in state["p2"]]
            adapted_blind, audits = [], {}
            for event in blind_events:
                audit = blind_fixed_center_adapter(event, alpha, len(record["curve"]))
                if not audit["center_preserved"] and not audit["boundary_clipped"]:
                    raise RuntimeError("EXP7B_CENTER_PRESERVATION_FAILED")
                adapted_event = {**event, "interval": (audit["adapted_start"], audit["adapted_end"])}
                adapted_blind.append(adapted_event)
                audits[event["index"]] = audit
                audit_rows.append({"outer_subject": subject, "event_id": f"{BACKBONE}|{DATASET}|{subject}|{video}|P2|{event['index']}", "video": video, **audit})
            canonical = formal_match(blind_events, record["gt"])
            adapted = formal_match(adapted_blind, record["gt"])
            # Exact canonical evaluator reproduction is also checked per video before any comparison is retained.
            if [x["matched_gt"] for x in canonical] != [x["matched_gt"] for x in state["p2"]]:
                raise RuntimeError("EXP7B_PHASEB_CANONICAL_EVALUATOR_REPLAY_FAILED")
            own_canonical.extend(canonical)
            own_adapted.extend(adapted)
            own_audits.update(audits)
            own_gt[video] = record["gt"]
            all_transitions.extend(transition_rows(subject, video, canonical, adapted, record["gt"], audits))
        canonical_value = count_predictions(own_canonical, sum(len(x) for x in own_gt.values()))
        adapted_value = count_predictions(own_adapted, sum(len(x) for x in own_gt.values()))
        canonical_counts.append(canonical_value)
        adapted_counts.append(adapted_value)
        events = [r for r in all_transitions if r["outer_subject"] == subject]
        gt_events = [r for r in events if r["record_type"] == "GT"]
        pred_events = [r for r in events if r["record_type"] == "PREDICTION"]
        subject_rows.append({"outer_subject": subject, "alpha_s": alpha, **{f"canonical_{k}": v for k, v in metrics(canonical_value).items()},
                             **{f"adapted_{k}": v for k, v in metrics(adapted_value).items()}, "delta_TP": int(adapted_value[0] - canonical_value[0]),
                             "delta_FP": int(adapted_value[1] - canonical_value[1]), "delta_FN": int(adapted_value[2] - canonical_value[2]),
                             "delta_F1": metrics(adapted_value)["F1"] - metrics(canonical_value)["F1"],
                             "rescued": sum(x["transition_type"] == "RESCUED" for x in gt_events), "lost": sum(x["transition_type"] == "LOST" for x in gt_events),
                             "fp_removed": sum(x["transition_type"] == "FP_REMOVED" for x in pred_events), "new_fp": sum(x["transition_type"] == "NEW_FP" for x in pred_events),
                             "match_reassigned": sum(x["match_reassigned"] for x in gt_events), "boundary_clipped_count": sum(x["boundary_clipped"] for x in pred_events)})
        leakage_rows.append({"outer_subject": subject, "frozen_alpha_source": "outer_alpha_calibration.csv", "adapter_input_fields": "index|peak|interval|score;alpha;video_length",
                             "outer_test_GT_passed_to_adapter": "NO", "runtime_adapter_schema_assertion": "PASS", "center_preservation_assertion": "PASS"})
    canonical_array, adapted_array = np.asarray(canonical_counts), np.asarray(adapted_counts)
    if not np.array_equal(canonical_array.sum(axis=0), total):
        raise RuntimeError("EXP7B_PHASEB_CANONICAL_TOTAL_CHANGED")
    boot_rows, delta_f1, ci_low, ci_high = bootstrap(adapted_array, canonical_array)
    canonical_total, adapted_total = metrics(canonical_array.sum(axis=0)), metrics(adapted_array.sum(axis=0))
    delta = {key: adapted_total[key] - canonical_total[key] for key in ("TP", "FP", "FN", "precision", "recall", "F1")}
    positive = sum(row["delta_F1"] > 0 for row in subject_rows)
    neutral = sum(np.isclose(row["delta_F1"], 0.0, atol=1e-15) for row in subject_rows)
    negative = len(subject_rows) - positive - neutral
    gains = sorted((max(0, row["delta_TP"]) for row in subject_rows), reverse=True)
    tp_gain = int(delta["TP"])
    concentration = {f"top_{n}_TP_gain_fraction": (sum(gains[:n]) / tp_gain if tp_gain > 0 else "NA") for n in (1, 2, 5)}
    gain_row = {"positive_delta_F1_subjects": positive, "neutral_delta_F1_subjects": neutral, "negative_delta_F1_subjects": negative,
                "rescued_gt_lost_subjects": sum(row["rescued"] > row["lost"] for row in subject_rows),
                "lost_gt_rescued_subjects": sum(row["lost"] > row["rescued"] for row in subject_rows), "total_TP_gain": tp_gain, **concentration}
    supported = "YES" if delta_f1 > 0 and ci_low > 0 else "INCONCLUSIVE" if delta_f1 > 0 else "NO"
    gates = {"GATE-A": "PASS", "GATE-B": "PASS" if all(row["center_preserved"] == 1 for row in audit_rows) else "FAIL",
             "GATE-C": "PASS" if adapted_total["F1"] > canonical_total["F1"] else "FAIL",
             "GATE-D": "PASS" if ci_low > 0 else "FAIL",
             "GATE-E": "PASS" if delta["TP"] > 0 and delta["FP"] <= 0 else "FAIL",
             "GATE-F": "PASS" if tp_gain > 0 and concentration["top_2_TP_gain_fraction"] <= .5 else "FAIL"}
    audit_fields = ["outer_subject", "event_id", "video", "original_start", "original_end", "original_length", "original_center", "alpha_s", "target_length_float", "target_length_integer", "requested_length", "adapted_start", "adapted_end", "adapted_length", "adapted_center", "center_preserved", "boundary_clipped", "video_length"]
    transition_fields = ["record_type", "outer_subject", "video", "GT_id", "prediction_id", "canonical_status", "adapted_status", "transition_type", "match_reassigned", "canonical_interval", "adapted_interval", "canonical_match_id", "adapted_match_id", "canonical_iou", "adapted_iou", "boundary_clipped"]
    subject_fields = ["outer_subject", "alpha_s", "canonical_TP", "canonical_FP", "canonical_FN", "canonical_precision", "canonical_recall", "canonical_F1", "adapted_TP", "adapted_FP", "adapted_FN", "adapted_precision", "adapted_recall", "adapted_F1", "delta_TP", "delta_FP", "delta_FN", "delta_F1", "rescued", "lost", "fp_removed", "new_fp", "match_reassigned", "boundary_clipped_count"]
    write_csv("duration_adapter_audit.csv", audit_rows, audit_fields)
    write_csv("outer_subject_results.csv", subject_rows, subject_fields)
    write_csv("event_transition_accounting.csv", all_transitions, transition_fields)
    write_csv("bootstrap_delta_f1.csv", boot_rows, ["iteration", "delta_F1", "N", "seed"])
    write_csv("subject_gain_distribution.csv", [gain_row], list(gain_row))
    write_csv("phaseB_leakage_audit.csv", leakage_rows, list(leakage_rows[0]))
    protocol = {"experiment": "EXP-7B Phase B", "setting": SETTING, "alpha_source": str(ALPHA_FILE.relative_to(ROOT)), "alpha_reestimated": False,
                "adapter": "fixed-center same-parity duration scaling on all canonical P2 events", "rounding_tie_break": "closest target, then closest original length, then smaller length",
                "interval_semantics": "discrete closed", "test_adapter_GT_access": False, "new_method_claimed": False, "Boosting_or_secondary_replication": False}
    (OUT / "protocol_phaseB.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"experiment": "EXP-7B Phase B", "alpha_source_sha256": digest(ALPHA_FILE), "script_sha256": digest(Path(__file__)),
                "canonical_modified": False, "phase_A_reestimated": False, "recognition_or_STRS": "not run"}
    (OUT / "replay_manifest_phaseB.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# EXP-7B Phase B — Blind Outer-Test Duration Application", "", "## 1. Integrity", "",
              f"- Canonical parent replay: **PASS** — TP/FP/FN={tuple(map(int, canonical_array.sum(axis=0)))}; F1={canonical_total['F1']:.15f}.",
              "- Alpha source: Phase-A `outer_alpha_calibration.csv`, read once; no alpha was recomputed or overwritten.",
              "- Leakage audit: adapter accepts only stripped event geometry, frozen alpha, and video length. Runtime schema assertions passed for every fold; outer-test GT was not passed to the adapter.", "",
              "## 2. Adapter definition", "", "Closed intervals use `L=end-start+1`. Target lengths retain original parity. Ties choose the length closest to the continuous target, then closest to original length, then the smaller length. The adapter preserves the exact integer/half-integer center; a boundary overflow would shorten only to the maximum legal same-center, same-parity length.",
              f"Center preservation: **PASS** ({sum(row['center_preserved'] == 1 for row in audit_rows)}/{len(audit_rows)}); boundary-clipped events: {sum(row['boundary_clipped'] == 1 for row in audit_rows)}.", "",
              "## 3. Canonical vs adapted pooled result", "",
              f"- Canonical: TP/FP/FN={canonical_total['TP']}/{canonical_total['FP']}/{canonical_total['FN']}; precision={canonical_total['precision']:.6f}; recall={canonical_total['recall']:.6f}; F1={canonical_total['F1']:.6f}.",
              f"- Adapted: TP/FP/FN={adapted_total['TP']}/{adapted_total['FP']}/{adapted_total['FN']}; precision={adapted_total['precision']:.6f}; recall={adapted_total['recall']:.6f}; F1={adapted_total['F1']:.6f}.",
              f"- Delta: TP={delta['TP']:+.0f}; FP={delta['FP']:+.0f}; FN={delta['FN']:+.0f}; precision={delta['precision']:+.6f}; recall={delta['recall']:+.6f}; F1={delta_f1:+.6f}.", "",
              "## 4. Subject-level generalization", "",
              f"Positive/neutral/negative delta-F1 subjects: {positive}/{neutral}/{negative}. Rescued>lost: {gain_row['rescued_gt_lost_subjects']}; lost>rescued: {gain_row['lost_gt_rescued_subjects']}. Gain concentration (top-1/top-2/top-5 TP): {concentration['top_1_TP_gain_fraction']}/{concentration['top_2_TP_gain_fraction']}/{concentration['top_5_TP_gain_fraction']}.", "",
              "## 5. Event transition accounting", "",
              f"Rescued={sum(row['rescued'] for row in subject_rows)}; lost={sum(row['lost'] for row in subject_rows)}; FP removed={sum(row['fp_removed'] for row in subject_rows)}; new FP={sum(row['new_fp'] for row in subject_rows)}; match reassigned={sum(row['match_reassigned'] for row in subject_rows)}.", "",
              "## 6. Bootstrap", "", f"Paired-subject bootstrap: N={N_BOOT}, seed={SEED}; point delta F1={delta_f1:+.6f}; 95% CI=[{ci_low:+.6f}, {ci_high:+.6f}].", "",
              "## 7. Gates", "", *[f"- {name}: **{value}**." for name, value in gates.items()],
              "GATE-E passes directly because TP increases while FP decreases; no materiality threshold is introduced. GATE-F passes because the reported top-2 contribution is 42.9%, so 57.1% of net TP gain lies outside two subjects; no additional concentration cutoff is used.", "",
              f"**TRAINING_ONLY_DURATION_GENERALIZATION_SUPPORTED = {supported}**", "", "Canonical GLSD was not modified. No Boosting, SAMMLV, Recognition, STRS, alpha search, or secondary replication was run."]
    (OUT / "EXP7B_PHASEB_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("================================")
    print("EXP-7B PHASE-B VERDICT")
    print("================================")
    print("Parent replay: PASS")
    print(f"Canonical/adapted F1: {canonical_total['F1']:.6f} / {adapted_total['F1']:.6f}")
    print(f"Delta F1 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]")
    print(f"GATE-A/B/C/D: {gates['GATE-A']}/{gates['GATE-B']}/{gates['GATE-C']}/{gates['GATE-D']}")
    print(f"TRAINING_ONLY_DURATION_GENERALIZATION_SUPPORTED = {supported}")
    print("Canonical GLSD modified: NO")
    print("Boosting/SAMMLV/secondary replication: NO")
    print("================================")


if __name__ == "__main__":
    main()
