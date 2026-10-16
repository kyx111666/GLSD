#!/usr/bin/env python3
"""Validate subject, video and fold correspondence before configuration transfer."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from transfer_common import DATASETS, LOCKED, canonical_subject, field_name, load_csv, selected_config_rows, write_csv, write_json
from transfer_paths import EVIDENCE, RESPONSES, RUNNERS, CORES


def normalize_video(dataset: str, subject: str, raw_value) -> str:
    value = Path(str(raw_value).strip()).name.lower()
    while "." in value and value.rsplit(".", 1)[1] in {"avi", "mp4", "mov", "pkl", "npy", "csv"}:
        value = value.rsplit(".", 1)[0]
    if dataset == "casme3":
        value = re.sub(r"^spno\.?0*\d+[_-]", "", value, flags=re.IGNORECASE)
    else:
        value = re.sub(rf"^0*{int(subject)}[_-]", "", value)
    if not value:
        raise RuntimeError(f"empty canonical video ID from {raw_value!r}")
    return value


def video_inventory(path: Path, dataset: str) -> dict[tuple[str, str], str]:
    rows = load_csv(path)
    subject_field = field_name(rows[0], ("outer_subject", "subject", "subject_id"), "video subject")
    video_field = field_name(
        rows[0], ("video", "video_name", "video_id", "video_path"), "explicit video ID"
    )
    inventory = {}
    for row in rows:
        subject = canonical_subject(dataset, row[subject_field])
        video = normalize_video(dataset, subject, row[video_field])
        key = subject, video
        raw = str(row[video_field])
        if key in inventory and inventory[key] != raw:
            raise RuntimeError(f"ambiguous canonical video key {key}: {inventory[key]!r} vs {raw!r}")
        inventory[key] = raw
    return inventory


def required_inputs() -> list[Path]:
    paths = []
    for setting in LOCKED:
        paths.extend([
            EVIDENCE[setting],
            EVIDENCE[setting] / "outer_selected_configs.csv",
            EVIDENCE[setting] / "per_subject_counts.csv",
            EVIDENCE[setting] / "per_video_evaluation.csv",
            EVIDENCE[setting] / "run_manifest.json",
            RESPONSES[setting],
            RUNNERS[setting],
        ])
    paths.extend(CORES.values())
    return paths


def parse_subject_list(value) -> list:
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return parsed
    if ";" in text:
        return [item for item in text.split(";") if item]
    raise RuntimeError(f"cannot parse inner_subjects list: {text[:120]!r}")


def validate_selection_provenance(path: Path, dataset: str, all_subjects: set[str]) -> None:
    rows = load_csv(path)
    subject_field = field_name(rows[0], ("outer_subject", "subject", "subject_id"), "outer subject")
    inner_field = field_name(rows[0], ("inner_subjects", "training_subjects"), "inner subjects")
    count_field = field_name(rows[0], ("inner_subject_count", "training_subject_count"), "inner subject count")
    if len(rows) != len(all_subjects):
        raise RuntimeError(f"selection provenance row count mismatch: {path}")
    for row in rows:
        outer = canonical_subject(dataset, row[subject_field])
        inner = {canonical_subject(dataset, value) for value in parse_subject_list(row[inner_field])}
        expected = all_subjects - {outer}
        if outer in inner or inner != expected or int(row[count_field]) != len(expected):
            raise RuntimeError(
                f"TRANSFER_PROTOCOL_FAIL: {path} outer fold {outer} is not exact Train_s"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / "validation"
    output.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise RuntimeError("TRANSFER_PROTOCOL_FAIL: missing input file(s):\n" + "\n".join(missing))

    subject_rows = []
    video_rows = []
    for dataset in ("sammlv", "casme3"):
        left_name = f"metst_{dataset}"
        right_name = f"boosting_{dataset}"
        left_configs = selected_config_rows(EVIDENCE[left_name] / "outer_selected_configs.csv", dataset)
        right_configs = selected_config_rows(EVIDENCE[right_name] / "outer_selected_configs.csv", dataset)
        expected = DATASETS[dataset]
        if len(left_configs) != expected["subjects"] or len(right_configs) != expected["subjects"]:
            raise RuntimeError(f"TRANSFER_PROTOCOL_FAIL: {dataset} subject cardinality mismatch")
        if set(left_configs) != set(right_configs):
            raise RuntimeError(f"TRANSFER_PROTOCOL_FAIL: {dataset} subject ID sets differ")
        validate_selection_provenance(
            EVIDENCE[left_name] / "outer_selected_configs.csv", dataset, set(left_configs)
        )
        validate_selection_provenance(
            EVIDENCE[right_name] / "outer_selected_configs.csv", dataset, set(right_configs)
        )
        for subject in sorted(left_configs, key=int):
            subject_rows.append({
                "dataset": dataset,
                "canonical_subject": subject,
                "metst_subject": left_configs[subject]["source_subject_raw"],
                "boosting_subject": right_configs[subject]["source_subject_raw"],
                "mapping_status": "EXACT_CANONICAL_MATCH",
            })

        left_videos = video_inventory(EVIDENCE[left_name] / "per_video_evaluation.csv", dataset)
        right_videos = video_inventory(EVIDENCE[right_name] / "per_video_evaluation.csv", dataset)
        if len(left_videos) != expected["videos"] or len(right_videos) != expected["videos"]:
            raise RuntimeError(
                f"TRANSFER_PROTOCOL_FAIL: {dataset} video cardinality mismatch "
                f"ME-TST+={len(left_videos)}, BoostingVRME={len(right_videos)}, expected={expected['videos']}"
            )
        if set(left_videos) != set(right_videos):
            only_left = sorted(set(left_videos) - set(right_videos))[:20]
            only_right = sorted(set(right_videos) - set(left_videos))[:20]
            raise RuntimeError(
                f"TRANSFER_PROTOCOL_FAIL: {dataset} video IDs differ; "
                f"only ME-TST+={only_left}; only BoostingVRME={only_right}"
            )
        for subject, video in sorted(left_videos, key=lambda item: (int(item[0]), item[1])):
            video_rows.append({
                "dataset": dataset, "canonical_subject": subject, "canonical_video": video,
                "metst_video": left_videos[(subject, video)],
                "boosting_video": right_videos[(subject, video)], "mapping_status": "MATCH",
            })
        if dataset == "casme3" and ("15", "j") in left_videos:
            raise RuntimeError("TRANSFER_PROTOCOL_FAIL: forbidden excluded CAS(ME)3 video spNO.15_j is present")

    write_csv(output / "subject_mapping.csv", subject_rows)
    write_csv(output / "video_mapping.csv", video_rows)
    write_json(output / "mapping_gate.json", {
        "status": "PASS", "subject_mapping_complete": True,
        "video_mapping_complete": True, "casme3_spNO_15_j_excluded": True,
        "source_outer_fold_train_sets_exact": True,
        "source_test_subject_excluded_from_selection": True,
        "target_side_tuning_used": False, "transferred_fields": ["a0", "rho", "tau"],
        "forbidden_transferred_fields": sorted({"k", "event_geometry", "boundary_rule", "candidate_order", "conflict_rule"}),
    })
    print("Subject and video correspondence validated.")


if __name__ == "__main__":
    main()
