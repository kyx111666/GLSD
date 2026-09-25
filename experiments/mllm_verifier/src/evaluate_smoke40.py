#!/usr/bin/env python3
"""Join evaluation-only labels with frozen-MLLM verdicts after inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import load_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data/smoke40_manifest.json")
    parser.add_argument("--raw", type=Path, default=root / "results/smoke40_raw.jsonl")
    parser.add_argument(
        "--determinism", type=Path, default=root / "results/determinism_gate.json"
    )
    parser.add_argument("--output", type=Path, default=root / "results/smoke40_eval.json")
    parser.add_argument("--report", type=Path, default=root / "results/SMOKE40_REPORT.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = {
        item["candidate_id"]: item["gt_match_label_for_evaluation_only"]
        for item in load_json(args.manifest)["candidates"]
    }
    label_counts = {label: sum(value == label for value in labels.values()) for label in ("TP", "FP")}
    if label_counts != {"TP": 20, "FP": 20}:
        raise RuntimeError(f"Smoke manifest must be exactly 20 TP / 20 FP: {label_counts}")
    raw = read_jsonl(args.raw)
    if len(raw) != len(labels) or len({item["candidate_id"] for item in raw}) != len(raw):
        raise RuntimeError("Inference output candidate count/identity is invalid")
    unknown = {item["candidate_id"] for item in raw} - set(labels)
    if unknown:
        raise RuntimeError(f"Unknown inference candidate IDs: {sorted(unknown)}")
    table = {"TP": {"keep": 0, "reject": 0}, "FP": {"keep": 0, "reject": 0}}
    parse_successes = 0
    inference_failures = []
    parse_failures = []
    for item in raw:
        label = labels[item["candidate_id"]]
        if item.get("inference_error"):
            inference_failures.append(item["candidate_id"])
            continue
        if not item.get("parse_ok") or item.get("verdict") not in {"keep", "reject"}:
            parse_failures.append(item["candidate_id"])
            continue
        parse_successes += 1
        table[label][item["verdict"]] += 1
    tp_retention = table["TP"]["keep"] / 20
    fp_removal = table["FP"]["reject"] / 20
    parse_rate = parse_successes / 40
    determinism = load_json(args.determinism) if args.determinism.is_file() else {"status": "MISSING"}
    if (
        tp_retention >= 0.90
        and fp_removal >= 0.20
        and parse_rate == 1.0
        and not inference_failures
        and determinism.get("status") == "PASS"
    ):
        status = "PASS"
    elif (
        parse_rate < 0.90
        or tp_retention < 0.75
        or fp_removal < 0.10
        or determinism.get("status") != "PASS"
    ):
        status = "FAIL"
    else:
        status = "BORDERLINE"
    result = {
        "TP_kept": table["TP"]["keep"],
        "TP_rejected": table["TP"]["reject"],
        "FP_kept": table["FP"]["keep"],
        "FP_rejected": table["FP"]["reject"],
        "TP_retention": tp_retention,
        "FP_removal": fp_removal,
        "JSON_parse_success_rate": parse_rate,
        "inference_failures": inference_failures,
        "parse_failures": parse_failures,
        "determinism_gate": determinism.get("status"),
        "exploratory_status": status,
        "rescued_GT": 0,
        "new_FP": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = f"""# Smoke40 Report

## Confusion-like table

| Original label | keep | reject |
|---|---:|---:|
| Original TP | {result['TP_kept']} | {result['TP_rejected']} |
| Original FP | {result['FP_kept']} | {result['FP_rejected']} |

## Metrics

- TP retention: {tp_retention:.4f}
- FP removal: {fp_removal:.4f}
- 20 TP 中 reject 数量: {result['TP_rejected']}
- 20 FP 中 reject 数量: {result['FP_rejected']}
- JSON parse success rate: {parse_rate:.4f}
- Inference failures: {len(inference_failures)}
- Parse failures: {len(parse_failures)}
- Deterministic replay: {result['determinism_gate']}

## Exploratory status

**{status}**

This is an engineering feasibility gate, not a statistical-significance claim.
The verifier only vetoes existing candidates; rescued GT and new FP are both fixed at zero.
"""
    args.report.write_text(report, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
