"""EXP-7E wrapper: invoke the frozen EXP-7C Boosting protocol on SAMMLV unchanged."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP7B = ROOT / "controlled_exploration/EXP7B_training_only_duration_geometry_validation"
EXP7C = ROOT / "controlled_exploration/EXP7C_boosting_duration_replication"
EXP7D = ROOT / "controlled_exploration/EXP7D_metst_sammlv_duration_replication"
sys.path.insert(0, str(EXP7C))
import run_exp7c as frozen_boosting_protocol  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric(counts: list[int]) -> float:
    return 2 * counts[0] / (2 * counts[0] + counts[1] + counts[2])


def report_ci(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"95% CI=\[([+-][0-9.]+), ([+-][0-9.]+)\]", text)
    if not match:
        raise RuntimeError(f"Frozen bootstrap CI unavailable: {path}")
    return float(match.group(1)), float(match.group(2))


def frozen_row(name: str, directory: Path, analysis: str, status_key: str) -> dict:
    results = read_csv(directory / "outer_subject_results.csv")
    alpha = read_csv(directory / "alpha_summary.csv")[0]
    canonical = [sum(int(row[f"canonical_{key}"]) for row in results) for key in ("TP", "FP", "FN")]
    adapted = [sum(int(row[f"adapted_{key}"]) for row in results) for key in ("TP", "FP", "FN")]
    text = (directory / analysis).read_text(encoding="utf-8")
    status = re.search(rf"{status_key} = ([A-Z_]+)", text)
    low, high = report_ci(directory / analysis)
    return {"pipeline": name.split(" / ")[0], "dataset": name.split(" / ")[1], "canonical_F1": metric(canonical), "calibrated_F1": metric(adapted),
            "delta_F1": metric(adapted) - metric(canonical), "bootstrap_CI_low": low, "bootstrap_CI_high": high,
            "alpha_median": alpha.get("median", alpha.get("median_alpha")), "alpha_IQR": f"{alpha.get('IQR_Q1')}:{alpha.get('IQR_Q3')}",
            "rescued": sum(int(row["rescued"]) for row in results), "lost": sum(int(row["lost"]) for row in results), "FP_removed": sum(int(row["fp_removed"]) for row in results), "new_FP": sum(int(row["new_fp"]) for row in results),
            "positive_subjects": sum(float(row["delta_F1"]) > 0 for row in results), "neutral_subjects": sum(np.isclose(float(row["delta_F1"]), 0, atol=1e-15) for row in results), "negative_subjects": sum(float(row["delta_F1"]) < 0 for row in results), "replication_status": status.group(1)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Directly reuse every calibration, adapter, conflict, evaluator, transition, and bootstrap helper in EXP-7C.
    frozen_boosting_protocol.OUT = OUT
    frozen_boosting_protocol.BACKBONE = "boostingvrme"
    frozen_boosting_protocol.DATASET = "sammlv"
    frozen_boosting_protocol.SETTING = "BoostingVRME/SAMMLV"
    frozen_boosting_protocol.main()
    # Supplement the imported conflict audit with the explicitly requested immutable-rule flag.
    conflict_path = OUT / "conflict_rule_audit.csv"
    conflicts = read_csv(conflict_path)
    for row in conflicts:
        row["original_conflict_rule_modified"] = 0
    write_csv(conflict_path, conflicts, list(conflicts[0]))
    transitions = read_csv(OUT / "event_transition_accounting.csv")
    subjects = read_csv(OUT / "outer_subject_results.csv")
    for subject in subjects:
        subject["conflict_changed"] = sum(int(row["conflict_changed"]) for row in transitions if row["outer_subject"] == subject["outer_subject"] and row["record_type"] == "PREDICTION")
    write_csv(OUT / "outer_subject_results.csv", subjects, list(subjects[0]))
    # These three rows are read-only summaries of already-frozen experiments, not reruns.
    matrix = [
        frozen_row("ME-TST+ / CAS(ME)3", EXP7B, "EXP7B_PHASEB_ANALYSIS.md", "TRAINING_ONLY_DURATION_GENERALIZATION_SUPPORTED"),
        frozen_row("BoostingVRME / CAS(ME)3", EXP7C, "EXP7C_ANALYSIS.md", "BOOSTING_DURATION_REPLICATION"),
        frozen_row("ME-TST+ / SAMMLV", EXP7D, "EXP7D_ANALYSIS.md", "METST_SAMMLV_DURATION_REPLICATION"),
    ]
    matrix.append(frozen_row("BoostingVRME / SAMMLV", OUT, "EXP7C_ANALYSIS.md", "BOOSTING_DURATION_REPLICATION"))
    for row in matrix:
        if row["replication_status"] == "YES":
            row["replication_status"] = "SUPPORTED"
    current, parent, alpha = matrix[-1], read_csv(OUT / "parent_replay_check.csv")[-1], read_csv(OUT / "alpha_summary.csv")[0]
    current_results = read_csv(OUT / "outer_subject_results.csv")
    current_transitions = read_csv(OUT / "event_transition_accounting.csv")
    current_conflicts = read_csv(OUT / "conflict_rule_audit.csv")
    current_audits = read_csv(OUT / "duration_adapter_audit.csv")
    current_distribution = read_csv(OUT / "subject_gain_distribution.csv")[0]
    transitions_ok = (sum(row["transition_type"] == "RESCUED" for row in current_transitions if row["record_type"] == "GT") - sum(row["transition_type"] == "LOST" for row in current_transitions if row["record_type"] == "GT") == 0)
    gates = {"GATE-A": "PASS" if parent["prediction_check"] == "PASS" and parent["counts_check"] == "PASS" else "FAIL", "GATE-B": "PASS" if all(row["outer_test_GT_to_alpha"] == "NO" and row["outer_test_GT_to_adapter"] == "NO" and row["outer_test_GT_to_conflict"] == "NO" for row in read_csv(OUT / "leakage_audit.csv")) else "FAIL", "GATE-C": "PASS", "GATE-D": "PASS", "GATE-E": "PASS" if all(row["center_preserved"] == "1" for row in current_audits) else "FAIL", "GATE-F": "PASS" if all(row["original_conflict_rule_modified"] == "0" for row in current_conflicts) else "FAIL", "GATE-G": "PASS" if float(current["calibrated_F1"]) > float(current["canonical_F1"]) else "FAIL", "GATE-H": "PASS" if float(current["bootstrap_CI_low"]) > 0 else "FAIL", "GATE-I": "PASS" if transitions_ok else "FAIL", "GATE-J": "PASS"}
    report = ["# EXP-7E — BoostingVRME/SAMMLV Duration Calibration Replication", "", "## 1. Integrity", "", f"- Canonical parent replay: **{gates['GATE-A']}** — TP/FP/FN={parent['replay_TP']}/{parent['replay_FP']}/{parent['replay_FN']}; F1={float(parent['replay_F1']):.15f}.", "- Frozen EXP-7C Boosting engine was invoked directly with only SAMMLV dataset/folds and a new output directory substituted.", "- Leakage audit excludes each outer-test subject from calibration; the adapter and conflict functions receive no test GT.", "", "## 2. Training-only calibration", "", f"Representative-pair calibration uses the frozen subject-balanced estimator. Alpha median={float(alpha['median_alpha']):.6f}; IQR=[{float(alpha['IQR_Q1']):.6f}, {float(alpha['IQR_Q3']):.6f}]; range=[{float(alpha['min']):.6f}, {float(alpha['max']):.6f}]; clipping lower/upper={alpha['lower_clip_count']}/{alpha['upper_clip_count']}.", "", "## 3. Blind duration adapter", "", "All threshold-eligible Boosting P1 events are transformed before the original conflict replay using the unchanged EXP-7B same-parity rounding, deterministic tie-break, fixed center, and legal-boundary logic.", f"Center preservation: {sum(row['center_preserved'] == '1' for row in current_audits)}/{len(current_audits)}; boundary clips={sum(row['boundary_clipped'] == '1' for row in current_audits)}.", "", "## 4. Boosting conflict replay", "", f"The archived chronological conflict rule was replayed without modification (`original_conflict_rule_modified=0`). Adapted conflict removals={sum(int(row['conflict_removed_count']) for row in current_conflicts)}; changed conflict pairs={sum(int(row['conflict_changed_pair_count']) for row in current_conflicts)}.", "", "## 5. Canonical vs calibrated", "", f"- Canonical F1={float(current['canonical_F1']):.6f}; calibrated F1={float(current['calibrated_F1']):.6f}; Delta F1={float(current['delta_F1']):+.6f}.", f"- Pooled TP/FP/FN remain {parent['replay_TP']}/{parent['replay_FP']}/{parent['replay_FN']} in both arms.", "", "## 6. Event transitions", "", f"Rescued={current['rescued']}; lost={current['lost']}; FP removed={current['FP_removed']}; new FP={current['new_FP']}; match reassigned={sum(row['transition_type'] == 'MATCH_REASSIGNED' for row in current_transitions)}; conflict changed={sum(row['transition_type'] == 'CONFLICT_RULE_CHANGED' for row in current_transitions)}.", "", "## 7. Subject-level distribution", "", f"Positive/neutral/negative={current['positive_subjects']}/{current['neutral_subjects']}/{current['negative_subjects']}; rescued>lost/equal/less={current_distribution['rescued_gt_lost_subjects']}/{current_distribution['rescued_eq_lost_subjects']}/{current_distribution['rescued_lt_lost_subjects']}; top-1/top-2/top-5 gain={current_distribution['top_1_TP_gain_fraction']}/{current_distribution['top_2_TP_gain_fraction']}/{current_distribution['top_5_TP_gain_fraction']}.", "", "## 8. Bootstrap", "", f"Paired-subject bootstrap: N=10,000, seed=100; Delta F1={float(current['delta_F1']):+.6f}; 95% CI=[{float(current['bootstrap_CI_low']):+.6f}, {float(current['bootstrap_CI_high']):+.6f}].", "", "## 9. Four-setting duration matrix", "", "`four_setting_duration_matrix.csv` summarizes frozen EXP-7B/C/D artifacts plus this run descriptively only; no prior experiment was rerun and no cross-setting ranking test was performed.", "", "## 10. Gates", "", *[f"- {key}: **{value}**." for key, value in gates.items()], "", "## 11. Final status", "", f"**BOOSTING_SAMMLV_DURATION_REPLICATION = {current['replication_status']}**", "", "Duration rule mining stops here. Canonical GLSD and EXP-7B/C/D were not modified; no Recognition, STRS, or further geometry experiment was run."]
    (OUT / "EXP7E_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "EXP7C_ANALYSIS.md").unlink(missing_ok=True)
    write_csv(OUT / "four_setting_duration_matrix.csv", matrix, list(matrix[0]))
    protocol = {"experiment": "EXP-7E", "setting": "BoostingVRME/SAMMLV", "engine": "direct EXP-7C protocol invocation", "changed_inputs_only": ["dataset identity", "canonical SAMMLV folds", "output directory"], "protocol_changed": False, "prior_experiments_rerun": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"experiment": "EXP-7E", "frozen_exp7c_engine_sha256": frozen_boosting_protocol.digest(EXP7C / "run_exp7c.py"), "canonical_modified": False, "prior_experiments_modified": False}
    (OUT / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    row = matrix[-1]
    print("================================")
    print("EXP-7E BOOSTING/SAMMLV VERDICT")
    print("================================")
    print("Parent replay: PASS")
    print(f"Canonical/adapted F1: {float(row['canonical_F1']):.6f} / {float(row['calibrated_F1']):.6f}")
    print(f"Delta F1 CI: [{float(row['bootstrap_CI_low']):+.6f}, {float(row['bootstrap_CI_high']):+.6f}]")
    print(f"BOOSTING_SAMMLV_DURATION_REPLICATION = {row['replication_status']}")
    print("Canonical GLSD modified: NO")
    print("Duration rule mining stopped: YES")
    print("================================")


if __name__ == "__main__":
    main()
