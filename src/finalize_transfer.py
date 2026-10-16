#!/usr/bin/env python3
"""Summarize the four cross-pipeline transfer experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transfer_common import DIRECTIONS, write_csv, write_json


ORDER = (
    "metst_to_boosting_sammlv",
    "metst_to_boosting_casme3",
    "boosting_to_metst_sammlv",
    "boosting_to_metst_casme3",
)


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def ci(values) -> str:
    return f"[{float(values[0]):.6f}, {float(values[1]):.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    gate = json.loads((root / "validation/preflight_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: preflight gate is not PASS")
    summary_dir = root / "summary"
    if summary_dir.exists():
        raise RuntimeError(f"refusing to overwrite summary: {summary_dir}")
    summary_dir.mkdir(parents=True)

    results = []
    for name in ORDER:
        spec = DIRECTIONS[name]
        direction_dir = root / spec["relative_output"]
        checks = json.loads((direction_dir / "protocol_checks.json").read_text(encoding="utf-8"))
        if checks.get("status") != "PASS" or checks.get("source_k_passed_to_target"):
            raise RuntimeError(f"TRANSFER_PROTOCOL_FAIL: protocol check failed for {name}")
        result = json.loads((direction_dir / "metrics.json").read_text(encoding="utf-8"))
        results.append(result)

    csv_rows, paper_rows = [], []
    for result in results:
        native = result["target_native"]
        tuned = result["target_tuned_glsd90"]
        transfer = result["transferred"]
        row = {
            "direction": result["direction"], "dataset": result["dataset"],
            "source_pipeline": result["source_pipeline"], "target_pipeline": result["target_pipeline"],
            "native_TP": native["TP"], "native_FP": native["FP"], "native_FN": native["FN"], "native_F1": native["F1"],
            "tuned_TP": tuned["TP"], "tuned_FP": tuned["FP"], "tuned_FN": tuned["FN"], "tuned_F1": tuned["F1"],
            "transfer_TP": transfer["TP"], "transfer_FP": transfer["FP"], "transfer_FN": transfer["FN"], "transfer_F1": transfer["F1"],
            "delta_vs_native": result["delta_vs_native"],
            "ci_low_vs_native": result["bootstrap_95_ci_vs_native"][0],
            "ci_high_vs_native": result["bootstrap_95_ci_vs_native"][1],
            "delta_vs_tuned": result["delta_vs_tuned"],
            "ci_low_vs_tuned": result["bootstrap_95_ci_vs_tuned"][0],
            "ci_high_vs_tuned": result["bootstrap_95_ci_vs_tuned"][1],
        }
        csv_rows.append(row)
        short = "M -> B" if result["source_pipeline"] == "ME-TST+" else "B -> M"
        paper_rows.append({
            "Transfer": f"{short} / {result['dataset'].upper()}",
            "Target Native": fmt(native["F1"]),
            "Target Tuned GLSD-90": fmt(tuned["F1"]),
            "Transferred": fmt(transfer["F1"]),
            "Delta vs Native": fmt(result["delta_vs_native"]),
            "95% CI vs Native": ci(result["bootstrap_95_ci_vs_native"]),
            "Delta vs Tuned": fmt(result["delta_vs_tuned"]),
            "95% CI vs Tuned": ci(result["bootstrap_95_ci_vs_tuned"]),
        })

    write_csv(summary_dir / "transfer_summary.csv", csv_rows)
    write_json(summary_dir / "transfer_summary.json", results)
    write_csv(summary_dir / "paper_table.csv", paper_rows)
    print("Transfer summaries saved successfully.")
    print("OUTPUT_ROOT =", root)


if __name__ == "__main__":
    main()
