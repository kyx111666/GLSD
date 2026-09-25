#!/usr/bin/env python3
"""Recursively discover exact SAMMLV visual sources without fuzzy matching."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from protocol import CANONICAL_PREDICTIONS, load_json, ordered_frame_paths


VIDEO_EXTENSIONS = {
    ".3gp",
    ".asf",
    ".avi",
    ".dv",
    ".f4v",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpe",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ogv",
    ".qt",
    ".rm",
    ".rmvb",
    ".ts",
    ".vob",
    ".webm",
    ".wmv",
}


def canonical_video_ids(source: Path) -> list[str]:
    ids = [str(row["video"]) for row in load_json(source)]
    if len(ids) != 79 or len(set(ids)) != 79:
        raise RuntimeError(
            f"Canonical metadata must contain 79 unique video IDs; got {len(ids)}/{len(set(ids))}"
        )
    return ids


def discover_raw_videos(root: Path, canonical: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    matches: dict[str, list[str]] = defaultdict(list)
    all_video_files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        resolved = str(path.resolve())
        all_video_files.append(resolved)
        # Strict rule: the complete filename stem must equal one canonical ID.
        if path.stem in canonical:
            matches[path.stem].append(resolved)
    return {key: sorted(set(values)) for key, values in matches.items()}, sorted(all_video_files)


def discover_frame_directories(root: Path, canonical: set[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_dir() or path.name not in canonical:
            continue
        try:
            frames = ordered_frame_paths(path)
        except RuntimeError:
            continue
        if frames:
            matches[path.name].append(str(path.resolve()))
    return {key: sorted(set(values)) for key, values in matches.items()}


def unique_only(mapping: dict[str, list[str]]) -> dict[str, str]:
    return {key: values[0] for key, values in mapping.items() if len(values) == 1}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(os.environ.get("VIDEO_SEARCH_ROOT", "data")),
    )
    parser.add_argument("--source", type=Path, default=CANONICAL_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=root / "data/video_source_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = canonical_video_ids(args.source)
    canonical = set(ids)
    raw_matches, all_video_files = discover_raw_videos(args.search_root, canonical)
    frame_matches = discover_frame_directories(args.search_root, canonical)
    raw_unique = unique_only(raw_matches)
    frame_unique = unique_only(frame_matches)
    raw_ambiguous = {key: value for key, value in raw_matches.items() if len(value) > 1}
    frame_ambiguous = {key: value for key, value in frame_matches.items() if len(value) > 1}

    # Prefer an exact unique raw video. Use an exact unique decoded-frame
    # directory only when there is no raw video for that ID.
    resolved: dict[str, dict[str, str]] = {}
    source_conflicts: dict[str, dict[str, list[str]]] = {}
    for video_id in ids:
        if video_id in raw_ambiguous or video_id in frame_ambiguous:
            source_conflicts[video_id] = {
                "raw_video_candidates": raw_matches.get(video_id, []),
                "frame_directory_candidates": frame_matches.get(video_id, []),
            }
            continue
        if video_id in raw_unique:
            resolved[video_id] = {"source_type": "raw_video", "path": raw_unique[video_id]}
        elif video_id in frame_unique:
            resolved[video_id] = {
                "source_type": "frame_directory",
                "path": frame_unique[video_id],
            }

    raw_missing = [video_id for video_id in ids if video_id not in raw_unique]
    resolved_missing = [video_id for video_id in ids if video_id not in resolved]
    report = {
        "search_root": str(args.search_root.resolve()),
        "matching_policy": (
            "Exact filename stem == canonical video_id for movie files; exact directory "
            "name == canonical video_id plus image frames for decoded directories. No fuzzy matching."
        ),
        "supported_video_extensions": sorted(VIDEO_EXTENSIONS),
        "canonical_video_count": len(ids),
        "canonical_video_ids": ids,
        "all_movie_files_found": all_video_files,
        "all_movie_file_count": len(all_video_files),
        "found_videos": raw_matches,
        "raw_video_unique_mapping": raw_unique,
        "raw_video_unique_count": len(raw_unique),
        "raw_video_missing": raw_missing,
        "raw_video_missing_count": len(raw_missing),
        "ambiguous_mappings": raw_ambiguous,
        "duplicate_mappings": raw_ambiguous,
        "frame_directory_matches": frame_matches,
        "frame_directory_unique_count": len(frame_unique),
        "frame_directory_ambiguous_mappings": frame_ambiguous,
        "source_conflicts": source_conflicts,
        "resolved_visual_mapping": resolved,
        "resolved_unique_count": len(resolved),
        "resolved_missing": resolved_missing,
        "resolved_missing_count": len(resolved_missing),
        "unique_79_video_mapping": len(raw_unique) == 79 and not raw_ambiguous,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "movie_files_found": len(all_video_files),
                "raw_video_unique": len(raw_unique),
                "raw_video_missing": len(raw_missing),
                "ambiguous": len(raw_ambiguous),
                "duplicates": len(raw_ambiguous),
                "frame_directories_unique": len(frame_unique),
                "resolved_visual_sources": len(resolved),
            }
        )
    )


if __name__ == "__main__":
    main()
