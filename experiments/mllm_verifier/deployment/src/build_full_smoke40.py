#!/usr/bin/env python3
"""Build the locked 11-remote + 9-local TP + 20-local FP smoke40 package."""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import ROLES, load_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
EXPLORATION = ROOT.parent
PROJECT = EXPLORATION.parent
REMOTE_TP_ZIP = os.environ.get(
    "REMOTE_TP_ZIP", "data/smoke11_remote_frames.zip"
)
LOCAL_FRAME_ROOT = Path(os.environ.get("SAMMLV_VIDEO_ROOT", "data/SAMM_longvideos"))
CANONICAL = (
    PROJECT
    / "historical_gl_exact_fresh_reproduction/fresh_run/metst/results"
    / "pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json"
)
EXPECTED_CANONICAL_SHA256 = "bbf0d559db15aaa5cd1ccc69e0dac3ef49cdd6c7d5294d07b8f10b7c38858643"
FRAME_SKIP = 7
SEED = 100


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def uniform_three(start: int, end: int) -> list[int]:
    return [start, round((start + end) / 2), end]


def temporal_indices(start: int, end: int) -> list[int]:
    duration = end - start + 1
    values = []
    for left, right in (
        (start - duration, start - 1),
        (start, end),
        (end + 1, end + duration),
    ):
        values.extend(uniform_three(left, right))
    if values != sorted(set(values)) or len(values) != 9:
        raise RuntimeError(f"Unusable candidate interval [{start}, {end}]")
    return values


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb, header, footer = 160, 32, 22
    canvas = Image.new("RGB", (thumb * 9, header + thumb + footer), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for title, left, right in (("Before", 0, 3), ("Candidate", 3, 6), ("After", 6, 9)):
        draw.text((((left + right) * thumb) // 2, 7), title, fill="black", font=font, anchor="ma")
    for index, (image, role) in enumerate(zip(images, ROLES)):
        image.thumbnail((thumb, thumb))
        x = index * thumb + (thumb - image.width) // 2
        y = header + (thumb - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((index * thumb + thumb // 2, header + thumb + 4), role, fill="black", font=font, anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    for image in images:
        image.close()


def canonical_candidates() -> list[dict]:
    if sha256_file(CANONICAL) != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError("GLSD_REPLAY_GATE_FAILED: canonical predictions SHA256 changed")
    candidates = []
    for row in load_json(CANONICAL):
        for ordinal, event in enumerate(row["predictions"]["pure"]):
            candidates.append(
                {
                    "candidate_id": f"metst_sammlv_{row['subject']}_{row['video']}_{ordinal:03d}",
                    "subject_id": str(row["subject"]),
                    "video_id": str(row["video"]),
                    "start": int(event["onset"]),
                    "peak": int(event["peak"]),
                    "end": int(event["offset"]),
                    "original_label": "TP" if int(event["matched_gt"]) >= 0 else "FP",
                }
            )
    return candidates


def choose_fp(pool: list[dict]) -> list[dict]:
    """Seeded greedy selection: subject diversity, video diversity, then stable ID."""
    rng = random.Random(SEED)
    seeded = {item["candidate_id"]: rng.random() for item in sorted(pool, key=lambda x: x["candidate_id"])}
    selected: list[dict] = []
    subject_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    remaining = pool[:]
    while len(selected) < 20:
        best = min(
            remaining,
            key=lambda item: (
                subject_counts[item["subject_id"]],
                video_counts[item["video_id"]],
                seeded[item["candidate_id"]],
                item["candidate_id"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        subject_counts[best["subject_id"]] += 1
        video_counts[best["video_id"]] += 1
    return sorted(selected, key=lambda item: item["candidate_id"])


def safe_extract(zip_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            resolved = (destination / member.filename).resolve()
            if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                raise RuntimeError("REMOTE_TP_IMPORT_FAILED: unsafe ZIP member")
        archive.extractall(destination)
    root = destination / "smoke11_remote_frames"
    if not root.is_dir():
        raise RuntimeError("REMOTE_TP_IMPORT_FAILED: ZIP root missing")
    return root


def import_remote(zip_path: Path, expected_ids: set[str], staging: Path) -> tuple[list[dict], Path]:
    sidecar = EXPLORATION / "smoke11_remote_frames.zip.sha256"
    if not zip_path.is_file() or not sidecar.is_file():
        raise RuntimeError("REMOTE_TP_IMPORT_FAILED: ZIP or SHA256 sidecar missing")
    expected_zip_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(zip_path) != expected_zip_hash:
        raise RuntimeError("REMOTE_TP_IMPORT_FAILED: ZIP SHA256 mismatch")
    remote_root = safe_extract(zip_path, staging)
    audit_text = (remote_root / "audit/EXTRACTION_AUDIT.md").read_text(encoding="utf-8")
    if "REMOTE_FRAME_EXTRACTION_PASS" not in audit_text:
        raise RuntimeError("REMOTE_TP_IMPORT_FAILED: extraction PASS marker absent")
    manifest = load_json(remote_root / "audit/frame_manifest_resolved.json")
    hashes = load_json(remote_root / "audit/sha256_manifest.json")
    if {item["candidate_id"] for item in manifest["candidates"]} != expected_ids:
        raise RuntimeError("REMOTE_TP_IMPORT_FAILED: candidate identity mismatch")
    hash_map = {(item["candidate_id"], item["role"]): item["sha256"] for item in hashes}
    for item in manifest["candidates"]:
        if [frame["role"] for frame in item["frames"]] != ROLES:
            raise RuntimeError("REMOTE_TP_IMPORT_FAILED: incomplete frame mapping")
        for frame in item["frames"]:
            path = remote_root / "candidates" / item["candidate_id"] / frame["output_filename"]
            if not path.is_file() or sha256_file(path) != hash_map.get((item["candidate_id"], frame["role"])):
                raise RuntimeError("REMOTE_TP_IMPORT_FAILED: image SHA256 mismatch")
            with Image.open(path) as image:
                image.verify()
    return manifest["candidates"], remote_root


def local_record(item: dict, output_dir: Path) -> dict:
    video_dir = LOCAL_FRAME_ROOT / item["video_id"]
    source_frames = sorted(
        [*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")],
        key=natural_key,
    )
    indices = temporal_indices(item["start"], item["end"])
    raw_indices = [FRAME_SKIP * value for value in indices]
    if not source_frames or max(raw_indices) >= len(source_frames):
        raise RuntimeError(f"Local source frames incomplete for {item['candidate_id']}")
    frames = []
    for role, raw_index in zip(ROLES, raw_indices):
        source = source_frames[raw_index]
        target = output_dir / f"{role}.jpg"
        shutil.copy2(source, target)
        with Image.open(target) as image:
            image.verify()
        digest = sha256_file(target)
        frames.append(
            {
                "role": role,
                "logical_raw_frame_index": raw_index,
                "physical_file_index": raw_index + 1,
                "source_filename": source.name,
                "source_path": str(source.resolve()),
                "output_filename": target.name,
                "sha256": digest,
            }
        )
    return {
        "candidate_id": item["candidate_id"],
        "subject_id": item["subject_id"],
        "video_id": item["video_id"],
        "candidate_start": item["start"],
        "candidate_end": item["end"],
        "candidate_center": round((item["start"] + item["end"]) / 2),
        "frames": frames,
    }


def build(args: argparse.Namespace) -> None:
    all_candidates = canonical_candidates()
    remote_ids = {item["candidate_id"] for item in load_json(EXPLORATION / "data/required_remote_candidates.json")["selected_11_tp_candidates"]}
    local_pool = [item for item in all_candidates if (LOCAL_FRAME_ROOT / item["video_id"]).is_dir()]
    local_tp = sorted((item for item in local_pool if item["original_label"] == "TP"), key=lambda x: x["candidate_id"])
    local_fp_pool = [item for item in local_pool if item["original_label"] == "FP"]
    if len(local_tp) != 9 or len(local_fp_pool) != 23:
        raise RuntimeError(f"Local coverage changed: TP={len(local_tp)}, FP={len(local_fp_pool)}")
    local_fp = choose_fp(local_fp_pool)
    remote_meta = {item["candidate_id"]: item for item in all_candidates if item["candidate_id"] in remote_ids}
    if len(remote_meta) != 11 or any(item["original_label"] != "TP" for item in remote_meta.values()):
        raise RuntimeError("REMOTE_TP_IMPORT_FAILED: canonical remote identities changed")

    smoke_root = ROOT / "data/smoke40"
    candidates_root = smoke_root / "candidates"
    sheets_root = smoke_root / "contact_sheets"
    audit = ROOT / "data/audit"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    candidates_root.mkdir(parents=True)
    sheets_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="smoke40_remote_") as temp:
        remote_records, remote_root = import_remote(args.remote_tp_zip, remote_ids, Path(temp))
        remote_audit = audit / "remote"
        if remote_audit.exists():
            shutil.rmtree(remote_audit)
        shutil.copytree(remote_root / "audit", remote_audit)
        remote_by_id = {item["candidate_id"]: item for item in remote_records}
        selected = sorted([*remote_meta.values(), *local_tp, *local_fp], key=lambda x: x["candidate_id"])
        inference, labels, provenance, local_records = [], [], [], []
        for number, item in enumerate(selected, start=1):
            package_name = f"candidate_{number:03d}"
            output_dir = candidates_root / package_name
            output_dir.mkdir()
            source = "remote" if item["candidate_id"] in remote_ids else "local"
            if source == "remote":
                record = remote_by_id[item["candidate_id"]]
                normalized_frames = []
                for frame in record["frames"]:
                    source_path = remote_root / "candidates" / item["candidate_id"] / frame["output_filename"]
                    target = output_dir / frame["output_filename"]
                    shutil.copy2(source_path, target)
                    normalized_frames.append(
                        {
                            "role": frame["role"],
                            "logical_raw_frame_index": frame["logical_raw_frame_index"],
                            "physical_file_index": frame["physical_file_index"],
                            "source_filename": frame["source_filename"],
                            "source_path": frame["source_path"],
                            "output_filename": target.name,
                            "sha256": sha256_file(target),
                        }
                    )
                record = {**record, "frames": normalized_frames}
            else:
                record = local_record(item, output_dir)
                local_records.append(record)
            paths = [output_dir / f"{role}.jpg" for role in ROLES]
            make_contact_sheet(paths, sheets_root / f"{package_name}.jpg")
            inference.append(
                {
                    "candidate_id": item["candidate_id"],
                    "subject_id": item["subject_id"],
                    "video_id": item["video_id"],
                    "frame_files": [
                        {"role": role, "file": f"../smoke40/candidates/{package_name}/{role}.jpg"}
                        for role in ROLES
                    ],
                }
            )
            labels.append({"candidate_id": item["candidate_id"], "original_label": item["original_label"]})
            provenance.append(
                {
                    "candidate_id": item["candidate_id"],
                    "source": source,
                    "video_id": item["video_id"],
                    "subject_id": item["subject_id"],
                    "logical_frame_indices": [frame["logical_raw_frame_index"] for frame in record["frames"]],
                    "physical_source_filenames": [frame["source_filename"] for frame in record["frames"]],
                    "individual_sha256": [frame["sha256"] for frame in record["frames"]],
                }
            )

    manifests = ROOT / "data/manifests"
    write_json(manifests / "smoke40_inference_manifest.json", {"frame_role_schema": ROLES, "candidates": inference})
    write_json(manifests / "smoke40_labels_PRIVATE.json", {"candidates": labels})
    write_json(manifests / "selected_local_9tp.json", {"seed": SEED, "candidates": [{k: v for k, v in item.items() if k != "original_label"} for item in local_tp]})
    write_json(manifests / "selected_local_20fp.json", {"seed": SEED, "selection": "subject diversity, video diversity, seeded rank, candidate_id", "candidates": [{k: v for k, v in item.items() if k != "original_label"} for item in local_fp]})
    write_json(audit / "SMOKE40_PROVENANCE.json", {"frame_index_convention": "zero-based logical raw index; physical filename index is +1", "candidates": provenance})
    write_json(audit / "local_frame_manifest_resolved.json", {"candidates": local_records})
    local_hashes = [
        {"candidate_id": record["candidate_id"], "role": frame["role"], "sha256": frame["sha256"]}
        for record in local_records for frame in record["frames"]
    ]
    write_json(audit / "local_sha256_manifest.json", local_hashes)
    local_audit_text = (
        "# Local frame extraction audit\n\n"
        f"- Candidate count: {len(local_records)}\n- Frame occurrences: {len(local_hashes)}\n"
        "- Missing frames: 0\n- Image decode failures: 0\n- Frame index convention: `zero-based logical / 1-based physical filename`\n"
        "- Frame order: `B1 B2 B3 C1 C2 C3 A1 A2 A3`\n\n`LOCAL_FRAME_EXTRACTION_PASS`\n"
    )
    (audit / "LOCAL_EXTRACTION_AUDIT.md").write_text(local_audit_text, encoding="utf-8")
    local_audit = audit / "local"
    write_json(local_audit / "frame_manifest_resolved.json", {"candidates": local_records})
    write_json(local_audit / "sha256_manifest.json", local_hashes)
    (local_audit / "EXTRACTION_AUDIT.md").write_text(local_audit_text, encoding="utf-8")
    print("SMOKE40_BUILD_PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-tp-zip", type=Path, default=Path(REMOTE_TP_ZIP))
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as error:
        if "REMOTE_TP_IMPORT_FAILED" in str(error):
            print("REMOTE_TP_IMPORT_FAILED")
        raise
