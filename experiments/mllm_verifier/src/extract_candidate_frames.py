#!/usr/bin/env python3
"""Deterministically extract ordered B/C/A facial frames and contact sheets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ModuleNotFoundError:  # Optional when all sources are frame directories.
    cv2 = None

from protocol import FRAME_SKIP, FPS, SEED, canonical_json_hash, load_json, ordered_frame_paths


def uniform_three(start: int, end: int) -> list[int]:
    if end < start:
        return []
    if end - start + 1 <= 3:
        return list(range(start, end + 1))
    return [start, round((start + end) / 2), end]


def sample_temporal_indices(start: int, end: int, maximum: int) -> tuple[list[int], list[str]]:
    duration = max(end - start + 1, 1)
    ranges = [
        (start - duration, start - 1),
        (start, end),
        (end + 1, end + duration),
    ]
    labels = ["B", "C", "A"]
    selected: list[int] = []
    warnings: list[str] = []
    for label, (left, right) in zip(labels, ranges):
        clipped_left, clipped_right = max(0, left), min(maximum, right)
        values = uniform_three(clipped_left, clipped_right)
        if len(values) < 3:
            warnings.append(f"{label} interval required nearest-frame expansion")
            midpoint = min(max((clipped_left + clipped_right) // 2, 0), maximum)
            alternatives = sorted(
                (index for index in range(maximum + 1) if index not in selected),
                key=lambda index: (abs(index - midpoint), index),
            )
            for index in alternatives:
                if index not in values:
                    values.append(index)
                if len(values) == 3:
                    break
        selected.extend(values[:3])
    # The normal GLSD geometry yields ordered unique indices. Boundary cases are
    # repaired globally without inventing or duplicating frames.
    unique = sorted(set(selected))
    if len(unique) < 9:
        warnings.append("global nearest-frame expansion was required")
        center = round((start + end) / 2)
        for index in sorted(range(maximum + 1), key=lambda x: (abs(x - center), x)):
            if index not in unique:
                unique.append(index)
            if len(unique) == min(9, maximum + 1):
                break
        unique.sort()
    if len(unique) < 9:
        warnings.append(f"video permits only {len(unique)} distinct sampled frames")
    return unique[:9], warnings


def make_contact_sheet(frame_paths: list[Path], output: Path) -> None:
    labels = ["B1", "B2", "B3", "C1", "C2", "C3", "A1", "A2", "A3"][: len(frame_paths)]
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    thumb = 224
    header = 42
    label_height = 24
    canvas = Image.new("RGB", (thumb * len(images), header + thumb + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    groups = (("Before", 0, 3), ("Candidate", 3, 6), ("After", 6, 9))
    for title, left, right in groups:
        if left >= len(images):
            continue
        visible_right = min(right, len(images))
        x = ((left + visible_right) * thumb) // 2
        draw.text((x, 8), title, fill="black", font=font, anchor="ma")
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail((thumb, thumb))
        x = index * thumb + (thumb - image.width) // 2
        y = header + (thumb - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(
            (index * thumb + thumb // 2, header + thumb + 5),
            label,
            fill="black",
            font=font,
            anchor="ma",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    for image in images:
        image.close()


def probe_video(path: Path) -> tuple[int, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required when the visual source is a raw video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open raw video: {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if total_frames <= 0:
        raise RuntimeError(f"Raw video reports invalid frame count {total_frames}: {path}")
    if reported_fps <= 0:
        raise RuntimeError(f"Raw video reports invalid fps {reported_fps}: {path}")
    return total_frames, reported_fps


def decode_selected_video_frames(path: Path, indices: list[int]) -> dict[int, object]:
    """Decode only the short candidate neighborhood, not the complete video."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required when the visual source is a raw video")
    required = sorted(set(indices))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open raw video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, required[0])
    positioned = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if positioned != required[0]:
        capture.release()
        raise RuntimeError(
            f"Raw-video seek mismatch for {path}: requested {required[0]}, got {positioned}"
        )
    decoded = {}
    target_set = set(required)
    for raw_index in range(required[0], required[-1] + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Decode failed at raw frame {raw_index}: {path}")
        next_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if next_position != raw_index + 1:
            capture.release()
            raise RuntimeError(
                f"Raw-video decode position mismatch for {path}: "
                f"expected {raw_index + 1}, got {next_position}"
            )
        if raw_index in target_set:
            decoded[raw_index] = frame
    capture.release()
    missing = target_set - set(decoded)
    if missing:
        raise RuntimeError(f"Missing decoded raw frames {sorted(missing)} from {path}")
    return decoded


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data/smoke40_manifest.json")
    parser.add_argument("--frames-dir", type=Path, default=root / "data/frames")
    parser.add_argument(
        "--contact-sheets-dir", type=Path, default=root / "data/contact_sheets"
    )
    parser.add_argument(
        "--inference-manifest", type=Path, default=root / "data/inference_manifest.json"
    )
    parser.add_argument(
        "--validation-output", type=Path, default=root / "data/frame_validation.json"
    )
    parser.add_argument("--validation-sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_json(args.manifest)
    candidates = source["candidates"]
    inference_rows = []
    validation_rows = []
    for item in candidates:
        source_path = Path(item["video_path"])
        source_type = item.get("video_source_type") or (
            "frame_directory" if source_path.is_dir() else "raw_video"
        )
        frame_skip = int(item.get("frame_skip", FRAME_SKIP))
        if source_type == "raw_video":
            total_frames, reported_fps = probe_video(source_path)
            original = None
        elif source_type == "frame_directory":
            original = ordered_frame_paths(source_path)
            total_frames = len(original)
            reported_fps = float(item.get("fps", FPS))
        else:
            raise RuntimeError(f"Unknown visual source type {source_type!r}")
        maximum = (total_frames - 1) // frame_skip
        indices, warnings = sample_temporal_indices(
            int(item["candidate_start"]), int(item["candidate_end"]), maximum
        )
        if len(indices) != 9:
            raise RuntimeError(
                f"Candidate {item['candidate_id']} cannot provide 9 distinct frames: {warnings}"
            )
        candidate_dir = args.frames_dir / item["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        labels = ["B1", "B2", "B3", "C1", "C2", "C3", "A1", "A2", "A3"]
        copied = []
        raw_indices = []
        for temporal_index in indices:
            raw_indices.append(temporal_index * frame_skip)
        if any(index < 0 or index >= total_frames for index in raw_indices):
            raise RuntimeError(
                f"Converted raw frame index out of range for {item['candidate_id']}: "
                f"indices={raw_indices}, total={total_frames}"
            )
        decoded = (
            decode_selected_video_frames(source_path, raw_indices)
            if source_type == "raw_video"
            else None
        )
        for label, raw_index in zip(labels, raw_indices):
            target = candidate_dir / f"{label}.jpg"
            if source_type == "raw_video":
                if not cv2.imwrite(str(target), decoded[raw_index]):
                    raise RuntimeError(f"Failed to write decoded frame {target}")
            else:
                shutil.copy2(original[raw_index], target)
            copied.append(target.resolve())
        if raw_indices != sorted(set(raw_indices)):
            raise RuntimeError(f"Non-monotonic or duplicate frames for {item['candidate_id']}")
        contact_sheet = args.contact_sheets_dir / f"{item['candidate_id']}.jpg"
        make_contact_sheet(copied, contact_sheet)
        inference_rows.append(
            {
                "candidate_id": item["candidate_id"],
                "ordered_frame_paths": [str(path) for path in copied],
                "ordered_frame_roles": labels,
                "contact_sheet_path": str(contact_sheet.resolve()),
                "extraction_warnings": warnings,
            }
        )
        validation_rows.append(
            {
                "candidate_id": item["candidate_id"],
                "source_type": source_type,
                "raw_video_path": str(source_path.resolve()),
                "requested_cache_coordinates": indices,
                "converted_raw_frame_indices": raw_indices,
                "decoded_raw_frames": [str(path) for path in copied],
                "raw_video_total_frames": total_frames,
                "raw_video_reported_fps": reported_fps,
                "expected_dataset_fps": FPS,
                "all_indices_legal": all(0 <= index < total_frames for index in raw_indices),
                "fps_warning": abs(reported_fps - FPS) > 1.0,
            }
        )
    source_geometry = [
        [
            item["candidate_id"],
            item["candidate_start"],
            item["candidate_peak"],
            item["candidate_end"],
        ]
        for item in candidates
    ]
    output = {
        "protocol": {
            "candidate_count": len(inference_rows),
            "candidate_geometry_sha256": canonical_json_hash(source_geometry),
            "frame_order": "B1 B2 B3 C1 C2 C3 A1 A2 A3",
            "frame_skip": FRAME_SKIP,
        },
        "candidates": inference_rows,
    }
    args.inference_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.inference_manifest.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    rng = random.Random(args.seed)
    sample_ids = {
        row["candidate_id"]
        for row in rng.sample(validation_rows, min(args.validation_sample_size, len(validation_rows)))
    }
    validation_output = {
        "seed": args.seed,
        "mapping": "raw_frame_index = frame_skip * zero_based_cache_coordinate",
        "frame_skip": FRAME_SKIP,
        "expected_dataset_fps": FPS,
        "candidate_count": len(validation_rows),
        "all_indices_legal": all(row["all_indices_legal"] for row in validation_rows),
        "random_validation_candidate_ids": sorted(sample_ids),
        "random_validation_records": [
            row for row in validation_rows if row["candidate_id"] in sample_ids
        ],
        "all_candidate_records": validation_rows,
    }
    args.validation_output.write_text(
        json.dumps(validation_output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.inference_manifest} with {len(inference_rows)} candidates")


if __name__ == "__main__":
    main()
