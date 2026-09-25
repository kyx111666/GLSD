#!/usr/bin/env python3
"""Build a balanced, development-only smoke40 manifest from canonical GLSD output."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from protocol import (
    CANONICAL_PREDICTIONS,
    FRAME_SKIP,
    FPS,
    SEED,
    canonical_json_hash,
    load_json,
    replay_gate,
)


def candidates_from_source(source: Path, visual_mapping: dict[str, dict[str, str]]) -> list[dict]:
    candidates = []
    for row in load_json(source):
        for ordinal, event in enumerate(row["predictions"]["pure"]):
            start, end = int(event["onset"]), int(event["offset"])
            visual = visual_mapping.get(str(row["video"]))
            candidates.append(
                {
                    "candidate_id": (
                        f"metst_sammlv_{row['subject']}_{row['video']}_{ordinal:03d}"
                    ),
                    "subject_id": str(row["subject"]),
                    "video_id": str(row["video"]),
                    "candidate_start": start,
                    "candidate_end": end,
                    "candidate_peak": int(event["peak"]),
                    "candidate_center": round((start + end) / 2),
                    "video_path": visual["path"] if visual else None,
                    "video_source_type": visual["source_type"] if visual else None,
                    "frame_skip": FRAME_SKIP,
                    "fps": FPS,
                    "gt_match_label_for_evaluation_only": (
                        "TP" if int(event["matched_gt"]) >= 0 else "FP"
                    ),
                }
            )
    return candidates


def choose_diverse(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    ranked = pool[:]
    rng.shuffle(ranked)
    tie_rank = {item["candidate_id"]: rank for rank, item in enumerate(ranked)}
    selected: list[dict] = []
    subject_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    remaining = ranked[:]
    while len(selected) < count and remaining:
        best = min(
            remaining,
            key=lambda item: (
                subject_counts[item["subject_id"]],
                video_counts[item["video_id"]],
                tie_rank[item["candidate_id"]],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        subject_counts[best["subject_id"]] += 1
        video_counts[best["video_id"]] += 1
    if len(selected) != count:
        raise RuntimeError(f"Need {count} candidates but only found {len(selected)}")
    return selected


def designate_development_subjects(candidates: list[dict], needed: int, seed: int) -> list[str]:
    """Freeze a subject-level dev scope; these subjects cannot be final test later."""
    by_subject: dict[str, Counter[str]] = {}
    for item in candidates:
        if not item["video_path"] or not Path(item["video_path"]).exists():
            continue
        by_subject.setdefault(item["subject_id"], Counter())[
            item["gt_match_label_for_evaluation_only"]
        ] += 1
    subjects = sorted(by_subject)
    random.Random(seed).shuffle(subjects)
    chosen: list[str] = []
    totals: Counter[str] = Counter()
    while subjects and (totals["TP"] < needed or totals["FP"] < needed):
        gains = {
            value: (
                min(totals["TP"] + by_subject[value]["TP"], needed)
                - min(totals["TP"], needed)
                + min(totals["FP"] + by_subject[value]["FP"], needed)
                - min(totals["FP"], needed)
            )
            for value in subjects
        }
        subject = max(subjects, key=lambda value: (gains[value], -subjects.index(value)))
        chosen.append(subject)
        totals.update(by_subject[subject])
        subjects.remove(subject)
    if totals["TP"] < needed or totals["FP"] < needed:
        raise RuntimeError(
            "SMOKE40_INPUT_BLOCKED: mappable canonical candidates provide only "
            f"TP={totals['TP']} and FP={totals['FP']}; need {needed} each"
        )
    return sorted(chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=CANONICAL_PREDICTIONS)
    parser.add_argument(
        "--video-source-audit",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/video_source_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/smoke40_manifest.json",
    )
    parser.add_argument("--development-subjects-file", type=Path)
    parser.add_argument("--designate-development-scope", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-label", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = replay_gate(args.source)
    source_audit = load_json(args.video_source_audit)
    if source_audit.get("source_conflicts"):
        raise RuntimeError("VIDEO_MAPPING_GATE_FAILED: ambiguous visual source mappings exist")
    candidates = candidates_from_source(args.source, source_audit["resolved_visual_mapping"])
    if args.development_subjects_file:
        dev_subjects = [
            str(item)
            for item in json.loads(args.development_subjects_file.read_text(encoding="utf-8"))[
                "development_subjects"
            ]
        ]
    elif args.designate_development_scope:
        dev_subjects = designate_development_subjects(candidates, args.per_label, args.seed)
    else:
        raise RuntimeError(
            "A development scope is mandatory. Pass --development-subjects-file or "
            "explicitly freeze one with --designate-development-scope."
        )
    eligible = [
        item
        for item in candidates
        if item["subject_id"] in dev_subjects
        and item["video_path"]
        and Path(item["video_path"]).exists()
    ]
    by_label = {
        label: [
            item
            for item in eligible
            if item["gt_match_label_for_evaluation_only"] == label
        ]
        for label in ("TP", "FP")
    }
    if any(len(by_label[label]) < args.per_label for label in by_label):
        raise RuntimeError(
            "SMOKE40_INPUT_BLOCKED: development scope has mappable "
            f"TP={len(by_label['TP'])}, FP={len(by_label['FP'])}; "
            f"need {args.per_label} each"
        )
    rng = random.Random(args.seed)
    selected = choose_diverse(by_label["TP"], args.per_label, rng)
    selected += choose_diverse(by_label["FP"], args.per_label, rng)
    rng.shuffle(selected)
    geometry = [
        [
            item["candidate_id"],
            item["subject_id"],
            item["video_id"],
            item["candidate_start"],
            item["candidate_peak"],
            item["candidate_end"],
        ]
        for item in selected
    ]
    payload = {
        "protocol": {
            "seed": args.seed,
            "sampling": "balanced 20 TP / 20 FP with subject/video diversity",
            "scope": "development-only; subjects must be excluded from future final test",
            "development_subjects": dev_subjects,
            "canonical_replay_gate": gate,
            "candidate_geometry_sha256": canonical_json_hash(geometry),
        },
        "candidates": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scope_path = args.output.parent / "development_scope.json"
    scope_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "development_subjects": dev_subjects,
                "rule": "Never use these subjects as final verifier test subjects.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {len(selected)} candidates")


if __name__ == "__main__":
    main()
