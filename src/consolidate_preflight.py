#!/usr/bin/env python3
"""Consolidate four independently executed exact replays into one hard gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transfer_common import LOCKED, f1, load_csv, write_csv, write_json


def one_generic(root: Path, setting: str) -> dict:
    candidates = list(root.glob("preflight_*/locked_replay_summary.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one preflight summary below {root}, got {candidates}")
    rows = load_csv(candidates[0])
    if len(rows) != 1:
        raise RuntimeError(f"expected one setting in {candidates[0]}")
    row = rows[0]
    expected = LOCKED[setting]
    native = tuple(int(row[f"Native_{name}"]) for name in ("TP", "FP", "FN"))
    tuned = tuple(int(row[f"GLSD_{name}"]) for name in ("TP", "FP", "FN"))
    if native != expected["native"] or tuned != expected["tuned"]:
        raise RuntimeError(f"TRANSFER_PROTOCOL_FAIL: {setting} exact replay mismatch")
    return {"setting": setting, "native": native, "tuned": tuned, "source": str(candidates[0])}


def casme3(root: Path) -> dict:
    path = root / "recovery_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    setting = "boosting_casme3"
    expected = LOCKED[setting]
    native = tuple(int(x) for x in data["native"]["full_counts"])
    tuned = tuple(int(x) for x in data["glsd_90"]["full_counts"])
    if data.get("status") != "PASS" or not data.get("all_functional_checks_passed"):
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: BoostingVRME/CAS(ME)3 replay did not PASS")
    if native != expected["native"] or tuned != expected["tuned"]:
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: BoostingVRME/CAS(ME)3 exact counts mismatch")
    return {"setting": setting, "native": native, "tuned": tuned, "source": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metst-sammlv", type=Path, required=True)
    parser.add_argument("--metst-casme3", type=Path, required=True)
    parser.add_argument("--boosting-sammlv", type=Path, required=True)
    parser.add_argument("--boosting-casme3", type=Path, required=True)
    args = parser.parse_args()
    mapping_gate = args.output_root / "audit/mapping_gate.json"
    mapping = json.loads(mapping_gate.read_text(encoding="utf-8"))
    if mapping.get("status") != "PASS":
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: mapping gate is not PASS")
    records = [
        one_generic(args.metst_sammlv, "metst_sammlv"),
        one_generic(args.metst_casme3, "metst_casme3"),
        one_generic(args.boosting_sammlv, "boosting_sammlv"),
        casme3(args.boosting_casme3),
    ]
    native_rows, tuned_rows = [], []
    for record in records:
        spec = LOCKED[record["setting"]]
        native_rows.append({
            "setting": record["setting"], "pipeline": spec["pipeline"], "dataset": spec["dataset"],
            "TP": record["native"][0], "FP": record["native"][1], "FN": record["native"][2],
            "F1": f1(record["native"]), "exact_replay": "PASS", "source": record["source"],
        })
        tuned_rows.append({
            "setting": record["setting"], "pipeline": spec["pipeline"], "dataset": spec["dataset"],
            "TP": record["tuned"][0], "FP": record["tuned"][1], "FN": record["tuned"][2],
            "F1": f1(record["tuned"]), "exact_replay": "PASS", "source": record["source"],
        })
    write_csv(args.output_root / "audit/native_replay.csv", native_rows)
    write_csv(args.output_root / "audit/tuned_glsd90_replay.csv", tuned_rows)
    gate = {
        "status": "PASS", "native_exact_replays": 4, "tuned_glsd90_exact_replays": 4,
        "mapping_gate": str(mapping_gate), "transfer_allowed": True,
    }
    write_json(args.output_root / "audit/preflight_gate.json", gate)
    print("FOUR_SETTING_EXACT_REPLAY_GATE = PASS")


if __name__ == "__main__":
    main()
