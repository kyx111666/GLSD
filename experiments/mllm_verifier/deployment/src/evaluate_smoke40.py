#!/usr/bin/env python3
"""Evaluate real smoke40 verifier output against private labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/manifests/smoke40_labels_PRIVATE.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke40_evaluation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/SMOKE40_REPORT.md")
    args = parser.parse_args()
    raw = load_json(args.raw)
    if raw.get("MOCK_OUTPUT") is True or raw.get("evaluation_status") == "NOT_FOR_EVALUATION":
        print("MOCK_RESULT_NOT_EVALUABLE")
        raise RuntimeError("MOCK_RESULT_NOT_EVALUABLE")
    labels = {item["candidate_id"]: item["original_label"] for item in load_json(args.labels)["candidates"]}
    results = raw.get("results", [])
    if len(labels) != 40 or len(results) != 40 or {item["candidate_id"] for item in results} != set(labels):
        raise RuntimeError("Evaluation candidate identities are incomplete")
    table = {"TP": {"keep": 0, "reject": 0}, "FP": {"keep": 0, "reject": 0}}
    for item in results:
        if not item.get("parse_ok") or item.get("verdict") not in {"keep", "reject"}:
            raise RuntimeError(f"Invalid result for {item['candidate_id']}")
        table[labels[item["candidate_id"]]][item["verdict"]] += 1
    retention = table["TP"]["keep"] / 20
    removal = table["FP"]["reject"] / 20
    if retention >= 0.90 and removal >= 0.20:
        status = "PASS"
    elif retention >= 0.75 and removal > 0:
        status = "BORDERLINE"
    else:
        status = "FAIL"
    result = {
        "TP_kept": table["TP"]["keep"], "TP_rejected": table["TP"]["reject"],
        "FP_kept": table["FP"]["keep"], "FP_rejected": table["FP"]["reject"],
        "TP_retention": retention, "FP_removal": removal, "exploration_gate": status,
    }
    write_json(args.output, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        f"# Smoke40 Report\n\n| Original class | kept | rejected |\n|---|---:|---:|\n"
        f"| TP | {result['TP_kept']} | {result['TP_rejected']} |\n"
        f"| FP | {result['FP_kept']} | {result['FP_rejected']} |\n\n"
        f"- TP retention: {retention:.3f}\n- FP removal: {removal:.3f}\n- Exploration gate: **{status}**\n\n"
        "This is an exploration gate, not a statistical-significance conclusion.\n",
        encoding="utf-8",
    )
    print(status)


if __name__ == "__main__":
    main()

