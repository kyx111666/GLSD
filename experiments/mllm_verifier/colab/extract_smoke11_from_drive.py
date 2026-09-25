#!/usr/bin/env python3
"""Selectively extract the 11 missing smoke40 candidates from a split Drive ZIP."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter, deque
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# ============================ USER CONFIGURATION ============================
# Change only these two paths for the normal Colab workflow.
SAMMLV_ARCHIVE_PATH = "/content/drive/MyDrive/.../SAMM_longvideos.zip"
OUTPUT_DRIVE_DIR = "/content/drive/MyDrive/..."

# Preferred: upload cloud_extraction_manifest.json when prompted.
USE_MANIFEST_UPLOAD = True
# Alternative: set USE_MANIFEST_UPLOAD=False and configure this Drive path.
MANIFEST_PATH = "/content/drive/MyDrive/.../cloud_extraction_manifest.json"

# None = accept auto-detection only when the complete directory clearly starts
# at physical index 0 or 1. Set explicitly to 0 or 1 if that is not detectable.
FRAME_FILE_INDEX_OFFSET: int | None = None

SAVE_UNIQUE_ORIGINALS = False
RUNTIME_OUTPUT_DIR = Path("/content/smoke11_remote_frames")
ARCHIVE_SELECTION_DIR = Path("/content/smoke11_archive_selection")
# ===========================================================================


TARGET_CANDIDATES = (
    "metst_sammlv_006_006_1_001",
    "metst_sammlv_006_006_1_002",
    "metst_sammlv_006_006_1_003",
    "metst_sammlv_006_006_1_004",
    "metst_sammlv_011_011_2_001",
    "metst_sammlv_011_011_2_003",
    "metst_sammlv_016_016_7_000",
    "metst_sammlv_016_016_7_003",
    "metst_sammlv_020_020_7_000",
    "metst_sammlv_020_020_7_002",
    "metst_sammlv_020_020_7_003",
)
TARGET_VIDEOS = ("006_1", "011_2", "016_7", "020_7")
DEFAULT_ROLES = ("B1", "B2", "B3", "C1", "C2", "C3", "A1", "A2", "A3")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def schema_summary(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: schema_summary(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [] if not value else [schema_summary(value[0], depth + 1)]
    return type(value).__name__


def value_from(item: dict[str, Any], names: tuple[str, ...], label: str) -> Any:
    present = [(name, item[name]) for name in names if name in item]
    if not present:
        raise ValueError(f"Missing required manifest field {label}; accepted aliases={names}")
    first = present[0][1]
    if any(value != first for _, value in present[1:]):
        raise ValueError(f"Conflicting aliases for manifest field {label}: {present}")
    return first


def integer_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(index, int) and not isinstance(index, bool) for index in value):
        raise ValueError(f"{label} must be a list of integer frame indices")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return list(value)


def extract_frame_groups(item: dict[str, Any], protocol: dict[str, Any]) -> dict[str, list[int]]:
    aliases = {
        "B": ("before_frame_indices", "before_raw_frame_indices", "before"),
        "C": ("candidate_frame_indices", "candidate_raw_frame_indices", "candidate"),
        "A": ("after_frame_indices", "after_raw_frame_indices", "after"),
    }
    groups: dict[str, list[int]] = {}
    containers = [item]
    for key in ("frames", "frame_indices", "raw_frames", "raw_frame_groups"):
        if isinstance(item.get(key), dict):
            containers.append(item[key])
    for group, names in aliases.items():
        found: list[tuple[str, Any]] = []
        for container in containers:
            found.extend((name, container[name]) for name in names if name in container)
        if found:
            parsed = [integer_list(value, name) for name, value in found]
            if any(value != parsed[0] for value in parsed[1:]):
                raise ValueError(f"Conflicting equivalent {group} frame fields")
            groups[group] = parsed[0]
    if len(groups) == 3:
        return groups

    raw = item.get("raw_frame_indices")
    if raw is not None:
        raw_indices = integer_list(raw, "raw_frame_indices")
        roles = item.get("frame_order", protocol.get("frame_order"))
        if not isinstance(roles, list) or len(roles) != len(raw_indices):
            raise ValueError("raw_frame_indices requires an equally sized manifest frame_order")
        parsed_groups = {"B": [], "C": [], "A": []}
        for role, index in zip(roles, raw_indices):
            if not isinstance(role, str) or not role or role[0].upper() not in parsed_groups:
                raise ValueError(f"Unrecognized frame role {role!r}")
            parsed_groups[role[0].upper()].append(index)
        if all(parsed_groups.values()):
            return parsed_groups
    raise ValueError("Could not map manifest fields to before/candidate/after raw frame groups")


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read manifest {manifest_path}: {exc}") from exc
    print("Manifest schema:")
    print(json.dumps(schema_summary(data), indent=2, ensure_ascii=False))
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ValueError("Manifest root must contain a candidates list")
    protocol = data.get("protocol") if isinstance(data.get("protocol"), dict) else {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for raw_item in data["candidates"]:
        if not isinstance(raw_item, dict):
            raise ValueError("Every candidate must be a JSON object")
        candidate_id = str(value_from(raw_item, ("candidate_id", "id"), "candidate_id"))
        if candidate_id in rows_by_id:
            raise ValueError(f"Duplicate candidate_id in manifest: {candidate_id}")
        if candidate_id not in TARGET_CANDIDATES:
            continue
        video_id = str(value_from(raw_item, ("video_id", "video"), "video_id"))
        subject_id = str(value_from(raw_item, ("subject_id", "subject"), "subject_id"))
        groups = extract_frame_groups(raw_item, protocol)
        roles_and_indices = [
            (f"{group}{position}", index)
            for group in ("B", "C", "A")
            for position, index in enumerate(groups[group], 1)
        ]
        rows_by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "video_id": video_id,
            "subject_id": subject_id,
            "candidate_start": int(value_from(raw_item, ("candidate_start", "start"), "candidate_start")),
            "candidate_end": int(value_from(raw_item, ("candidate_end", "end"), "candidate_end")),
            "candidate_center": int(value_from(raw_item, ("candidate_center", "center", "candidate_peak", "peak"), "candidate_center")),
            "candidate_start_raw": int(value_from(raw_item, ("candidate_start_raw", "start_raw"), "candidate_start_raw")),
            "candidate_end_raw": int(value_from(raw_item, ("candidate_end_raw", "end_raw"), "candidate_end_raw")),
            "candidate_center_raw": int(value_from(raw_item, ("candidate_center_raw", "center_raw", "peak_raw"), "candidate_center_raw")),
            "groups": groups,
            "roles_and_indices": roles_and_indices,
            "temporal_frame_indices": raw_item.get("temporal_frame_indices"),
        }
    missing = sorted(set(TARGET_CANDIDATES) - set(rows_by_id))
    if missing:
        raise ValueError(f"Manifest is missing fixed target candidates: {missing}")
    rows = [rows_by_id[candidate_id] for candidate_id in TARGET_CANDIDATES]
    if {row["video_id"] for row in rows} != set(TARGET_VIDEOS):
        raise ValueError("Target candidate video IDs do not equal the fixed four-video scope")
    return data, rows


def physical_index(path: Path | PurePosixPath) -> int:
    tokens = re.findall(r"\d+", path.stem)
    if not tokens:
        raise ValueError(f"Image filename has no numeric index: {path.name}")
    return int(tokens[-1])


def ensure_7z() -> str:
    executable = shutil.which("7z")
    if executable:
        return executable
    print("7z is not installed; installing the small CPU-only p7zip-full package...")
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "p7zip-full"], check=True)
    executable = shutil.which("7z")
    if not executable:
        raise RuntimeError("7z installation completed but the executable is unavailable")
    return executable


def archive_volumes(archive: Path) -> list[Path]:
    if not archive.is_file():
        raise FileNotFoundError(f"SAMMLV_ARCHIVE_PATH does not exist: {archive}")
    if archive.suffix.lower() != ".zip":
        raise ValueError("SAMMLV_ARCHIVE_PATH must point to the final .zip volume")
    pattern = re.compile(re.escape(archive.stem) + r"\.z\d+$", re.IGNORECASE)
    parts = sorted(
        (path for path in archive.parent.iterdir() if path.is_file() and pattern.fullmatch(path.name)),
        key=lambda path: path.suffix.lower(),
    )
    volumes = parts + [archive]
    print("Archive volumes:")
    for path in volumes:
        print(f"- {path.name} ({path.stat().st_size} bytes)")
    return volumes


def target_image_members(
    seven_zip: str,
    archive: Path,
    subject_by_video: dict[str, str],
) -> list[str]:
    accepted_parts = {
        video_id: {video_id.lower(), f"subject_{subject_by_video[video_id]}_{video_id}".lower()}
        for video_id in TARGET_VIDEOS
    }
    include_filters = [
        pattern
        for names in accepted_parts.values()
        for name in names
        for pattern in (f"-ir!*/{name}/*", f"-ir!{name}/*")
    ]
    command = [seven_zip, "l", "-slt", *include_filters, str(archive)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is None:
        raise RuntimeError("Unable to read 7z archive listing")
    selected: list[str] = []
    diagnostics: deque[str] = deque(maxlen=40)
    for line in process.stdout:
        diagnostics.append(line.rstrip())
        if not line.startswith("Path = "):
            continue
        member = line[7:].strip().replace("\\", "/")
        path = PurePosixPath(member)
        if path.suffix.lower() not in IMAGE_EXTENSIONS or "__MACOSX" in path.parts:
            continue
        lower_parts = {part.lower() for part in path.parts[:-1]}
        if any(lower_parts & names for names in accepted_parts.values()):
            if path.is_absolute() or ".." in path.parts or "\n" in member or "\r" in member:
                raise RuntimeError(f"Unsafe archive member path: {member!r}")
            selected.append(member)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"7z could not list the split archive (exit {return_code}). Ensure all .z01... parts "
            f"and the .zip file are together and unchanged. Tail:\n" + "\n".join(diagnostics)
        )
    if not selected:
        raise RuntimeError("No target JPG/PNG members were found in the archive")
    return sorted(set(selected))


def index_archive_frames(
    archive: Path,
    members: list[str],
    subject_by_video: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, dict[int, str]]]:
    prefixes: dict[str, set[str]] = {video_id: set() for video_id in TARGET_VIDEOS}
    files_by_video: dict[str, list[PurePosixPath]] = {video_id: [] for video_id in TARGET_VIDEOS}
    for member in members:
        path = PurePosixPath(member)
        lower_parts = [part.lower() for part in path.parts]
        matches: list[tuple[str, int]] = []
        for video_id in TARGET_VIDEOS:
            accepted = {video_id.lower(), f"subject_{subject_by_video[video_id]}_{video_id}".lower()}
            matches.extend((video_id, position) for position, part in enumerate(lower_parts[:-1]) if part in accepted)
        if len(matches) != 1:
            raise RuntimeError(f"Archive member must map to exactly one target video: {member}")
        video_id, position = matches[0]
        prefixes[video_id].add(PurePosixPath(*path.parts[: position + 1]).as_posix())
        files_by_video[video_id].append(path)

    resolved_directories: dict[str, str] = {}
    naming_audits: list[dict[str, Any]] = []
    frame_maps: dict[str, dict[int, str]] = {}
    for video_id in TARGET_VIDEOS:
        if len(prefixes[video_id]) != 1:
            raise RuntimeError(
                f"Expected exactly one archive directory for {video_id}, found {sorted(prefixes[video_id])}"
            )
        prefix = next(iter(prefixes[video_id]))
        files = sorted(files_by_video[video_id], key=lambda path: path.name)
        if not files:
            raise RuntimeError(f"No image members found for {video_id}")
        indexed: dict[int, str] = {}
        for path in files:
            index = physical_index(path)
            member = path.as_posix()
            if index in indexed:
                raise RuntimeError(
                    f"Duplicate physical frame index {index} in {prefix}: {indexed[index]}, {member}"
                )
            indexed[index] = member
        extensions = sorted({path.suffix.lower() for path in files})
        widths = [len(re.findall(r"\d+", path.stem)[-1]) for path in files]
        minimum = min(indexed)
        detected = 0 if minimum == 0 else 1 if minimum == 1 else None
        resolved = f"{archive}::{prefix}"
        resolved_directories[video_id] = resolved
        naming_audits.append({
            "video_id": video_id,
            "resolved_directory": resolved,
            "number_of_image_files": len(files),
            "first_5_filenames": [path.name for path in files[:5]],
            "last_5_filenames": [path.name for path in files[-5:]],
            "detected_extension": extensions[0] if len(extensions) == 1 else extensions,
            "numeric_token_rule": "last numeric token in filename stem",
            "numeric_widths": sorted(set(widths)),
            "zero_padded": len(set(widths)) == 1 and widths[0] > len(str(max(indexed))),
            "minimum_physical_index": minimum,
            "maximum_physical_index": max(indexed),
            "detected_numeric_index_convention": (
                f"{detected}-based (archive directory minimum is {minimum})"
                if detected is not None else "FRAME_INDEX_CONVENTION_UNRESOLVED"
            ),
        })
        frame_maps[video_id] = indexed
    return resolved_directories, naming_audits, frame_maps


def extracted_member_path(root: Path, member: str) -> Path:
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe archive member path: {member!r}")
    output = (root / Path(*relative.parts)).resolve()
    output.relative_to(root.resolve())
    return output


def selectively_extract_members(
    seven_zip: str,
    archive: Path,
    members: set[str],
    output_dir: Path,
) -> dict[str, Path]:
    if not members:
        raise RuntimeError("No requested manifest frames could be mapped to archive members")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    listfile = output_dir.parent / "smoke11_archive_members.txt"
    listfile.write_text("\n".join(sorted(members)) + "\n", encoding="utf-8")
    command = [
        seven_zip,
        "x",
        "-y",
        "-scsUTF-8",
        f"-o{output_dir}",
        str(archive),
        f"@{listfile}",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Selective 7z extraction failed (exit {result.returncode}):\n{result.stdout[-4000:]}")
    extracted = {member: extracted_member_path(output_dir, member) for member in members}
    missing = [member for member, path in extracted.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"7z reported success but did not extract selected members: {missing[:10]}")
    return extracted


def resolve_offset(naming_audits: list[dict[str, Any]]) -> int:
    if FRAME_FILE_INDEX_OFFSET not in (None, 0, 1):
        raise ValueError("FRAME_FILE_INDEX_OFFSET must be None, 0, or 1")
    detected = {
        audit["minimum_physical_index"]
        for audit in naming_audits
        if audit["minimum_physical_index"] in (0, 1)
    }
    unresolved = [audit["video_id"] for audit in naming_audits if audit["minimum_physical_index"] not in (0, 1)]
    if FRAME_FILE_INDEX_OFFSET is None:
        if unresolved or len(detected) != 1:
            raise RuntimeError(
                "FRAME_INDEX_CONVENTION_UNRESOLVED: set FRAME_FILE_INDEX_OFFSET explicitly "
                f"after reviewing FRAME_NAMING_AUDIT; unresolved={unresolved}, detected={sorted(detected)}"
            )
        return detected.pop()
    conflicts = [
        audit["video_id"]
        for audit in naming_audits
        if audit["minimum_physical_index"] in (0, 1)
        and audit["minimum_physical_index"] != FRAME_FILE_INDEX_OFFSET
    ]
    if conflicts:
        raise RuntimeError(
            f"Configured FRAME_FILE_INDEX_OFFSET={FRAME_FILE_INDEX_OFFSET} conflicts with directory minima for {conflicts}"
        )
    return FRAME_FILE_INDEX_OFFSET


def verify_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            bands = image.getbands()
    except Exception as exc:
        return {"ok": False, "error": f"IMAGE_DECODE_FAILED: {exc}"}
    valid_channels = len(bands) in (1, 2, 3, 4)
    return {
        "ok": width > 0 and height > 0 and valid_channels,
        "width": width,
        "height": height,
        "channels": list(bands),
        "error": None if width > 0 and height > 0 and valid_channels else "IMAGE_DECODE_FAILED: invalid dimensions/channels",
    }


def copy_as_jpeg(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, output)
        return
    with Image.open(source) as image:
        image.load()
        image.convert("RGB").save(output, format="JPEG", quality=95, subsampling=0)


def make_contact_sheet(paths: list[Path], roles: list[str], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        thumb, header, label_height = 180, 40, 24
        canvas = Image.new("RGB", (thumb * len(images), header + thumb + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for position, (image, role) in enumerate(zip(images, roles)):
            image.thumbnail((thumb, thumb))
            x = position * thumb + (thumb - image.width) // 2
            y = header + (thumb - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((position * thumb + thumb // 2, header + thumb + 5), role, fill="black", font=font, anchor="ma")
        group_positions: dict[str, list[int]] = {"B": [], "C": [], "A": []}
        for position, role in enumerate(roles):
            group_positions[role[0]].append(position)
        for group, title in (("B", "Before"), ("C", "Candidate"), ("A", "After")):
            positions = group_positions[group]
            center = (min(positions) + max(positions) + 1) * thumb // 2
            draw.text((center, 8), title, fill="black", font=font, anchor="ma")
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="JPEG", quality=92)
    finally:
        for image in images:
            image.close()


def deterministic_zip(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((path for path in source_dir.rglob("*") if path.is_file()), key=lambda item: item.relative_to(source_dir).as_posix()):
            relative = (Path(source_dir.name) / path.relative_to(source_dir)).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def audit_markdown(
    manifest_sha256: str,
    archive_volumes_used: list[Path],
    resolved_directories: dict[str, str],
    counts: dict[str, int],
    offset: int | None,
    temporal_warnings: list[dict[str, Any]],
    decode_failures: list[dict[str, Any]],
    duplicate_hash_mismatches: list[dict[str, Any]],
    status: str,
) -> str:
    directories = "\n".join(f"- `{video}` -> `{path}`" for video, path in resolved_directories.items())
    volumes = "\n".join(f"- `{path}` ({path.stat().st_size} bytes)" for path in archive_volumes_used)
    return f"""# Smoke11 remote-frame extraction audit

- Manifest SHA256: `{manifest_sha256}`
- SAMMLV archive: `{SAMMLV_ARCHIVE_PATH}`
- Candidate count: {counts['number_of_target_candidates']}
- Requested frame occurrences: {counts['requested_frame_occurrences']}
- Unique source frames: {counts['number_of_unique_requested_raw_frames']}
- Successfully extracted frame occurrences: {counts['successfully_resolved_frame_occurrences']}
- Missing frame occurrences: {counts['missing_frame_occurrences']}
- Duplicate frame occurrences: {counts['duplicate_frame_occurrences']}
- Image decode failures: {len(decode_failures)}
- Duplicate hash mismatches: {len(duplicate_hash_mismatches)}
- Frame index convention: `{offset}-based`
- Unresolved index conventions: {0 if offset in (0, 1) else 1}
- Temporal-order warnings: {len(temporal_warnings)}

## Archive volumes

{volumes}

## Resolved archive directories

{directories}

## Final status

`{status}`
"""


def acquire_manifest() -> Path:
    if USE_MANIFEST_UPLOAD:
        from google.colab import files

        print("Upload cloud_extraction_manifest.json")
        uploaded = files.upload()
        if "cloud_extraction_manifest.json" not in uploaded:
            raise RuntimeError("Expected an uploaded file named cloud_extraction_manifest.json")
        path = Path("/content/cloud_extraction_manifest.json")
        path.write_bytes(uploaded["cloud_extraction_manifest.json"])
        return path
    path = Path(MANIFEST_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Configured MANIFEST_PATH does not exist: {path}")
    return path


def run_extraction(manifest_path: Path) -> tuple[Path, Path]:
    archive = Path(SAMMLV_ARCHIVE_PATH)
    volumes = archive_volumes(archive)
    _, candidates = validate_manifest(manifest_path)
    print("[2/7] Manifest validated")

    subject_by_video = {
        video_id: next(row["subject_id"] for row in candidates if row["video_id"] == video_id)
        for video_id in TARGET_VIDEOS
    }
    seven_zip = ensure_7z()
    members = target_image_members(seven_zip, archive, subject_by_video)
    resolved_directories, naming_audits, archive_frame_maps = index_archive_frames(
        archive, members, subject_by_video
    )
    print("video_id -> resolved_archive_path")
    for video_id, directory in resolved_directories.items():
        print(f"{video_id} -> {directory}")
    print("[3/7] Video directories resolved")

    print("FRAME_NAMING_AUDIT")
    print(json.dumps(naming_audits, indent=2, ensure_ascii=False))
    offset = resolve_offset(naming_audits)
    print(f"Resolved FRAME_FILE_INDEX_OFFSET={offset}")
    print("[4/7] Frame naming audited")

    required_members = {
        member
        for candidate in candidates
        for _, logical_index in candidate["roles_and_indices"]
        if (member := archive_frame_maps[candidate["video_id"]].get(logical_index + offset)) is not None
    }
    extracted_members = selectively_extract_members(
        seven_zip, archive, required_members, ARCHIVE_SELECTION_DIR
    )
    print(f"Selectively extracted {len(extracted_members)} unique archive members")

    if RUNTIME_OUTPUT_DIR.exists():
        shutil.rmtree(RUNTIME_OUTPUT_DIR)
    candidates_dir = RUNTIME_OUTPUT_DIR / "candidates"
    originals_dir = RUNTIME_OUTPUT_DIR / "originals"
    audit_dir = RUNTIME_OUTPUT_DIR / "audit"
    contact_dir = RUNTIME_OUTPUT_DIR / "contact_sheets"
    for directory in (candidates_dir, originals_dir, audit_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_manifest: list[dict[str, Any]] = []
    sha_manifest: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []
    decode_failures: list[dict[str, Any]] = []
    temporal_warnings: list[dict[str, Any]] = []
    duplicate_hash_mismatches: list[dict[str, Any]] = []
    source_hashes: dict[tuple[str, int], str] = {}
    output_hashes: dict[tuple[str, int], str] = {}
    requested_keys: list[tuple[str, int]] = []

    for candidate in candidates:
        groups = candidate["groups"]
        ordered = max(groups["B"]) < min(groups["C"]) and max(groups["C"]) < min(groups["A"])
        frame_index_row = {
            "candidate_id": candidate["candidate_id"],
            "video_id": candidate["video_id"],
            "B_indices": groups["B"],
            "C_indices": groups["C"],
            "A_indices": groups["A"],
            "strict_temporal_order": ordered,
            "warning": None if ordered else "TEMPORAL_ORDER_WARNING",
        }
        if not ordered:
            temporal_warnings.append(frame_index_row)
        candidate_result = {
            key: candidate[key]
            for key in (
                "candidate_id", "video_id", "subject_id", "candidate_start", "candidate_end",
                "candidate_center", "candidate_start_raw", "candidate_end_raw", "candidate_center_raw",
            )
        }
        candidate_result["frames"] = []
        output_paths: list[Path] = []
        output_roles: list[str] = []
        for role, logical_index in candidate["roles_and_indices"]:
            requested_keys.append((candidate["video_id"], logical_index))
            physical = logical_index + offset
            archive_member = archive_frame_maps[candidate["video_id"]].get(physical)
            source = extracted_members.get(archive_member) if archive_member else None
            output = candidates_dir / candidate["candidate_id"] / f"{role}.jpg"
            mapping = {
                "role": role,
                "logical_raw_frame_index": logical_index,
                "physical_file_index": physical,
                "source_filename": PurePosixPath(archive_member).name if archive_member else None,
                "source_path": f"{archive}::{archive_member}" if archive_member else None,
                "extracted_source_path": str(source) if source else None,
                "output_filename": output.name,
                "output_path": str(output),
            }
            candidate_result["frames"].append(mapping)
            if source is None:
                missing_frames.append({"candidate_id": candidate["candidate_id"], "video_id": candidate["video_id"], **mapping})
                continue
            source_check = verify_image(source)
            if not source_check["ok"]:
                decode_failures.append({"candidate_id": candidate["candidate_id"], "role": role, "path": str(source), **source_check})
                continue
            source_hash = sha256_file(source)
            key = (candidate["video_id"], logical_index)
            if key in source_hashes and source_hashes[key] != source_hash:
                duplicate_hash_mismatches.append({
                    "error": "DUPLICATE_FRAME_HASH_MISMATCH", "video_id": key[0],
                    "logical_raw_frame_index": key[1], "previous_sha256": source_hashes[key], "current_sha256": source_hash,
                })
                continue
            source_hashes[key] = source_hash
            try:
                copy_as_jpeg(source, output)
            except Exception as exc:
                decode_failures.append({"candidate_id": candidate["candidate_id"], "role": role, "path": str(source), "ok": False, "error": f"IMAGE_DECODE_FAILED: {exc}"})
                continue
            output_check = verify_image(output)
            if not output_check["ok"]:
                decode_failures.append({"candidate_id": candidate["candidate_id"], "role": role, "path": str(output), **output_check})
                continue
            output_hash = sha256_file(output)
            if key in output_hashes and output_hashes[key] != output_hash:
                duplicate_hash_mismatches.append({
                    "error": "DUPLICATE_FRAME_HASH_MISMATCH", "video_id": key[0],
                    "logical_raw_frame_index": key[1], "previous_output_sha256": output_hashes[key],
                    "current_output_sha256": output_hash,
                })
                continue
            output_hashes[key] = output_hash
            mapping.update({"source_sha256": source_hash, "output_sha256": output_hash, "image_integrity": output_check})
            sha_manifest.append({
                "candidate_id": candidate["candidate_id"], "role": role,
                "source_path": f"{archive}::{archive_member}",
                "extracted_source_path": str(source), "output_path": str(output),
                "logical_raw_frame_index": logical_index, "physical_file_index": physical,
                "source_filename": source.name, "source_sha256": source_hash,
                "sha256": output_hash, "file_size_bytes": output.stat().st_size,
            })
            output_paths.append(output)
            output_roles.append(role)
            if SAVE_UNIQUE_ORIGINALS:
                original = originals_dir / candidate["video_id"] / source.name
                if not original.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, original)
        if len(output_paths) == len(candidate["roles_and_indices"]):
            make_contact_sheet(output_paths, output_roles, contact_dir / f"{candidate['candidate_id']}.jpg")
        resolved_manifest.append(candidate_result)

    requested_counter = Counter(requested_keys)
    requested_occurrences = len(requested_keys)
    successful_occurrences = len(sha_manifest)
    counts = {
        "number_of_target_candidates": len(candidates),
        "requested_frame_occurrences": requested_occurrences,
        "number_of_unique_requested_raw_frames": len(requested_counter),
        "successfully_resolved_frame_occurrences": successful_occurrences,
        "missing_frame_occurrences": len(missing_frames),
        "duplicate_frame_occurrences": sum(count - 1 for count in requested_counter.values()),
    }
    print(json.dumps(counts, indent=2))
    print("[5/7] Frames extracted")

    frame_index_audit = {
        "sammlv_archive_path": str(archive),
        "archive_volumes": [str(path) for path in volumes],
        "selectively_extracted_unique_members": len(extracted_members),
        "frame_file_index_offset": offset,
        "frame_naming_audit": naming_audits,
        "candidates": [
            {
                "candidate_id": row["candidate_id"], "video_id": row["video_id"],
                "B_indices": row["groups"]["B"], "C_indices": row["groups"]["C"], "A_indices": row["groups"]["A"],
                "strict_temporal_order": max(row["groups"]["B"]) < min(row["groups"]["C"]) and max(row["groups"]["C"]) < min(row["groups"]["A"]),
            }
            for row in candidates
        ],
        "temporal_order_warnings": temporal_warnings,
        "raw_mapping_audit_only": [
            {
                "candidate_id": row["candidate_id"],
                "manifest_temporal_frame_indices": row["temporal_frame_indices"],
                "manifest_raw_frame_indices": [index for _, index in row["roles_and_indices"]],
                "seven_times_temporal_matches": (
                    [7 * index for index in row["temporal_frame_indices"]] == [index for _, index in row["roles_and_indices"]]
                    if isinstance(row["temporal_frame_indices"], list) else None
                ),
            }
            for row in candidates
        ],
    }
    write_json(audit_dir / "frame_manifest_resolved.json", {"manifest_sha256": sha256_file(manifest_path), "candidates": resolved_manifest})
    write_json(audit_dir / "sha256_manifest.json", sha_manifest)
    write_json(audit_dir / "missing_frames.json", missing_frames)
    write_json(audit_dir / "frame_index_audit.json", frame_index_audit)

    complete_candidates = sum(
        len(row["frames"]) == sum(1 for entry in sha_manifest if entry["candidate_id"] == row["candidate_id"])
        for row in resolved_manifest
    )
    passed = (
        len(candidates) == 11 and complete_candidates == 11 and not missing_frames
        and not decode_failures and not duplicate_hash_mismatches and offset in (0, 1)
    )
    status = "REMOTE_FRAME_EXTRACTION_PASS" if passed else "REMOTE_FRAME_EXTRACTION_FAILED"
    report = audit_markdown(
        sha256_file(manifest_path), volumes, resolved_directories, counts, offset, temporal_warnings,
        decode_failures, duplicate_hash_mismatches, status,
    )
    (audit_dir / "EXTRACTION_AUDIT.md").write_text(report, encoding="utf-8")
    write_json(audit_dir / "image_decode_failures.json", decode_failures)
    write_json(audit_dir / "duplicate_hash_mismatches.json", duplicate_hash_mismatches)
    print("[6/7] Integrity audit complete")
    print(status)
    if not passed:
        raise RuntimeError(f"{status}; inspect {audit_dir / 'EXTRACTION_AUDIT.md'}")

    zip_path = RUNTIME_OUTPUT_DIR.parent / "smoke11_remote_frames.zip"
    deterministic_zip(RUNTIME_OUTPUT_DIR, zip_path)
    zip_hash = sha256_file(zip_path)
    hash_path = RUNTIME_OUTPUT_DIR.parent / "smoke11_remote_frames.zip.sha256"
    hash_path.write_text(f"{zip_hash}  {zip_path.name}\n", encoding="utf-8")

    drive_output = Path(OUTPUT_DRIVE_DIR)
    drive_output.mkdir(parents=True, exist_ok=True)
    for source in (
        zip_path,
        hash_path,
        audit_dir / "EXTRACTION_AUDIT.md",
        audit_dir / "frame_manifest_resolved.json",
        audit_dir / "sha256_manifest.json",
    ):
        shutil.copy2(source, drive_output / source.name)
    print(f"ZIP SHA256: {zip_hash}")
    print(f"Saved selected outputs to: {drive_output}")
    print("[7/7] Package created")
    return zip_path, hash_path


def main() -> None:
    from google.colab import drive

    drive.mount("/content/drive")
    print("[1/7] Drive mounted")
    manifest_path = acquire_manifest()
    run_extraction(manifest_path)


if __name__ == "__main__":
    main()
