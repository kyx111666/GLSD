#!/usr/bin/env python3
"""Verify smoke40 identities, images, extraction consistency, and create its lockfile."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from common import ROLES, load_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise RuntimeError(f"SMOKE40_INPUT_GATE_FAILED: {message}")


def main() -> None:
    manifest_path = ROOT / "data/manifests/smoke40_inference_manifest.json"
    labels_path = ROOT / "data/manifests/smoke40_labels_PRIVATE.json"
    provenance_path = ROOT / "data/audit/SMOKE40_PROVENANCE.json"
    manifest = load_json(manifest_path)
    labels = load_json(labels_path)["candidates"]
    provenance = load_json(provenance_path)["candidates"]
    rows = manifest.get("candidates", [])
    if len(rows) != 40 or len({row["candidate_id"] for row in rows}) != 40:
        fail(f"candidate count/identity invalid: {len(rows)}")
    counts = Counter(item["original_label"] for item in labels)
    if counts != {"TP": 20, "FP": 20}:
        fail(f"private label counts invalid: {dict(counts)}")
    if {row["candidate_id"] for row in rows} != {item["candidate_id"] for item in labels}:
        fail("inference and evaluation identities differ")
    sources = Counter(item["source"] for item in provenance)
    label_by_id = {item["candidate_id"]: item["original_label"] for item in labels}
    remote_tp = sum(item["source"] == "remote" and label_by_id[item["candidate_id"]] == "TP" for item in provenance)
    local_tp = sum(item["source"] == "local" and label_by_id[item["candidate_id"]] == "TP" for item in provenance)
    local_fp = sum(item["source"] == "local" and label_by_id[item["candidate_id"]] == "FP" for item in provenance)
    if (remote_tp, local_tp, local_fp) != (11, 9, 20) or sources != {"remote": 11, "local": 29}:
        fail(f"composition invalid: remote TP={remote_tp}, local TP={local_tp}, local FP={local_fp}")

    missing = corrupt = mismatches = unresolved = 0
    image_signatures: dict[str, set[tuple[str | None, str, tuple[int, int]]]] = {
        "remote": set(),
        "local": set(),
    }
    provenance_by_id = {item["candidate_id"]: item for item in provenance}
    for row in rows:
        frames = row.get("frame_files", [])
        if [item.get("role") for item in frames] != ROLES:
            unresolved += 1
            continue
        record = provenance_by_id[row["candidate_id"]]
        if len(record.get("logical_frame_indices", [])) != 9 or len(record.get("physical_source_filenames", [])) != 9:
            unresolved += 1
        resolved_paths = []
        for index, frame in enumerate(frames):
            path = (manifest_path.parent / frame["file"]).resolve()
            resolved_paths.append(path)
            if not path.is_file():
                missing += 1
                continue
            try:
                with Image.open(path) as image:
                    image_signatures[record["source"]].add(
                        (image.format, image.mode, image.size)
                    )
                    image.verify()
            except Exception:
                corrupt += 1
            if sha256_file(path) != record["individual_sha256"][index]:
                mismatches += 1
        if len(resolved_paths) != 9:
            unresolved += 1
    if missing or corrupt or mismatches or unresolved:
        fail(f"missing={missing}, corrupt={corrupt}, hash_mismatch={mismatches}, unresolved={unresolved}")

    expected_signature = {("JPEG", "L", (600, 600))}
    if image_signatures["remote"] != expected_signature or image_signatures["local"] != expected_signature:
        raise RuntimeError(
            "EXTRACTION_PROTOCOL_MISMATCH: remote/local image format expectations differ"
        )

    # Both sources were normalized to the same nine roles, JPEG names, manifest fields,
    # monotonically increasing temporal order, and zero-based logical index convention.
    for item in provenance:
        if len(item["logical_frame_indices"]) != 9 or item["logical_frame_indices"] != sorted(set(item["logical_frame_indices"])):
            raise RuntimeError("EXTRACTION_PROTOCOL_MISMATCH")
        if len(item["individual_sha256"]) != 9:
            raise RuntimeError("EXTRACTION_PROTOCOL_MISMATCH")

    tracked = {
        "smoke40_inference_manifest.json": sha256_file(manifest_path),
        "smoke40_labels_PRIVATE.json": sha256_file(labels_path),
        "verifier_prompt_v1.txt": sha256_file(ROOT / "config/verifier_prompt_v1.txt"),
        "smoke40_protocol_v1.yaml": sha256_file(ROOT / "config/smoke40_protocol_v1.yaml"),
    }
    smoke_root = ROOT / "data/smoke40"
    file_hashes = {
        str(path.relative_to(smoke_root)): sha256_file(path)
        for path in sorted(smoke_root.rglob("*")) if path.is_file()
    }
    lockfile = {
        "status": "SMOKE40_LOCKFILE_CREATED",
        "candidate_count": 40,
        "private_label_counts": {"TP": 20, "FP": 20},
        "composition": {"remote_tp": 11, "local_tp": 9, "local_fp": 20},
        "missing_frames": 0,
        "corrupt_images": 0,
        "unresolved_frame_mappings": 0,
        "hash_mismatches": 0,
        "locked_files": tracked,
        "smoke40_sorted_file_hash_manifest": file_hashes,
    }
    write_json(ROOT / "data/audit/SMOKE40_LOCKFILE.json", lockfile)
    print("SMOKE40_INPUT_GATE_PASS")
    print("SMOKE40_LOCKFILE_CREATED")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "EXTRACTION_PROTOCOL_MISMATCH" in str(error):
            print("EXTRACTION_PROTOCOL_MISMATCH")
        raise
