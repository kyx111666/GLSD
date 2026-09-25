#!/usr/bin/env python3
"""Instrument the original ME-TST+ CASME3 test path before stitching.

This exporter is intentionally inference-only.  It loads one frozen checkpoint
per subject, traverses that subject's complete flattened window sequence once,
captures yhat/yhat1 immediately after forward, retains the original batch-local
``i == 0`` stitching branch, and verifies each saved video by reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


DATASET = "CASME_3"
MODEL_NAME = "ME-TST+"
WINDOW_LENGTH = 50
# Derived from the original train.py write expressions using k//2.
STRIDE = WINDOW_LENGTH // 2
BATCH_SIZE = 256
FRAME_SKIP = 1
NUM_CLASSES = 5
EXPECTED_SUBJECTS = 94
EXPECTED_VIDEOS = 462
EXPECTED_ALIGNMENT_SHA256 = "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda"
SCHEMA = "casme3_raw_prestitch_window_dump_v1"
FULL_SCHEMA = "casme3_raw_prestitch_full_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-cache-dir", type=Path, required=True)
    parser.add_argument("--alignment-cache", type=Path, required=True)
    parser.add_argument("--skip-manifest", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject")
    parser.add_argument("--video")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        help="required for a full run; must record an exact one-video smoke PASS",
    )
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN_NOT_GIT"


def cache_parts(cache_dir: Path) -> list[Path]:
    return [cache_dir / f"CASME_3_dataset_{index}.pkl" for index in range(1, 6)]


def checkpoint_path(weights_dir: Path, subject: str) -> Path:
    return weights_dir / f"subject_{subject}.pkl"


def load_alignment_manifest(path: Path) -> tuple[list[dict], list[str]]:
    if sha256(path) != EXPECTED_ALIGNMENT_SHA256:
        raise RuntimeError("verified CASME3 alignment-cache SHA-256 mismatch")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"alignment cache is {type(payload)!r}, expected dict")
    expected_top = {
        "dataset": DATASET,
        "frame_skip": FRAME_SKIP,
        "k": WINDOW_LENGTH,
        "num_subjects": EXPECTED_SUBJECTS,
        "num_videos": EXPECTED_VIDEOS,
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"alignment cache {key}={payload.get(key)!r}, expected={expected!r}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_VIDEOS:
        raise RuntimeError("alignment cache does not contain 462 ordered records")
    manifest = []
    subjects = []
    seen_subjects = set()
    per_subject_video_count = {}
    for global_video_index, record in enumerate(records):
        subject = str(record["subject"])
        video = str(record["video"])
        subject_index = int(record["subject_index"])
        subject_video_index = int(record["video_index"])
        if subject not in seen_subjects:
            subjects.append(subject)
            seen_subjects.add(subject)
        expected_video_index = per_subject_video_count.get(subject, 0)
        if subject_video_index != expected_video_index:
            raise RuntimeError(
                f"non-sequential video_index at {global_video_index}: "
                f"{subject}/{video}={subject_video_index}, expected={expected_video_index}"
            )
        if subject_index != subjects.index(subject):
            raise RuntimeError(f"subject_index mismatch at {subject}/{video}")
        score = np.asarray(record["score"])
        logits = np.asarray(record["logits"])
        if score.ndim != 1 or logits.shape != (len(score), NUM_CLASSES):
            raise RuntimeError(f"invalid compact output geometry at {subject}/{video}")
        if len(score) % STRIDE != 0 or len(score) < WINDOW_LENGTH:
            raise RuntimeError(f"stitched length is incompatible with stride at {subject}/{video}")
        expected_windows = len(score) // STRIDE - 1
        manifest.append(
            {
                "subject": subject,
                "video": video,
                "subject_index": subject_index,
                "subject_video_index": subject_video_index,
                "global_video_index": global_video_index,
                "expected_windows": expected_windows,
                "expected_stitched_length": len(score),
            }
        )
        per_subject_video_count[subject] = expected_video_index + 1
    if len(subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"alignment subject count={len(subjects)}")
    if subjects != [str(value) for value in sorted(map(int, subjects))]:
        raise RuntimeError("alignment subject order is not the original numeric LOSO order")
    return manifest, subjects


def read_skip_manifest(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or "skip_index" not in rows[0]:
        raise RuntimeError("skip manifest must contain exactly one row with skip_index")
    row = dict(rows[0])
    row["skip_index"] = int(row["skip_index"])
    if not 0 <= row["skip_index"] <= EXPECTED_VIDEOS:
        raise RuntimeError("skip_index is outside the pre-drop 463-video label manifest")
    return row


def inspect_feature_cache(parts: list[Path], manifest: list[dict], load_arrays: bool = True):
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing original CASME3 cache part(s): " + ", ".join(missing))
    if not load_arrays:
        return None, []
    videos = []
    reports = []
    for part_index, path in enumerate(parts, start=1):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"cache part {part_index} is {type(value)!r}, expected list/tuple")
        reports.append(
            {
                "part_index": part_index,
                "path": str(path),
                "sha256": sha256(path),
                "videos": len(value),
            }
        )
        videos.extend(value)
    if len(videos) != EXPECTED_VIDEOS:
        raise RuntimeError(f"five-part feature cache has {len(videos)} videos, expected 462")
    for index, (video_features, identity) in enumerate(zip(videos, manifest)):
        array = np.asarray(video_features)
        expected = (identity["expected_windows"], 10, WINDOW_LENGTH)
        if array.shape != expected or not np.isfinite(array).all():
            raise RuntimeError(
                f"feature geometry mismatch at global video {index} "
                f"{identity['subject']}/{identity['video']}: {array.shape}, expected={expected}"
            )
    return videos, reports


def inspect_feature_cache_metadata(parts: list[Path], manifest: list[dict]) -> dict:
    """Inspect one pickle part at a time without retaining the full feature cache."""
    reports = []
    cursor = 0
    errors = []
    for part_index, path in enumerate(parts, start=1):
        if not path.is_file():
            reports.append({"part_index": part_index, "path": str(path), "status": "MISSING"})
            errors.append(f"missing {path}")
            continue
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"container={type(value)!r}, expected list/tuple")
            first_shape = list(np.asarray(value[0]).shape) if value else None
            last_shape = list(np.asarray(value[-1]).shape) if value else None
            total_windows = 0
            for local_index, video_features in enumerate(value):
                global_index = cursor + local_index
                if global_index >= len(manifest):
                    raise RuntimeError(f"part exceeds the {len(manifest)}-video alignment")
                array = np.asarray(video_features)
                expected = (manifest[global_index]["expected_windows"], 10, WINDOW_LENGTH)
                if array.shape != expected or not np.isfinite(array).all():
                    identity = manifest[global_index]
                    raise RuntimeError(
                        f"geometry mismatch at {identity['subject']}/{identity['video']}: "
                        f"{array.shape}, expected={expected}"
                    )
                total_windows += len(array)
            reports.append(
                {
                    "part_index": part_index,
                    "path": str(path),
                    "sha256": sha256(path),
                    "videos": len(value),
                    "total_windows": total_windows,
                    "first_shape": first_shape,
                    "last_shape": last_shape,
                    "status": "PASS",
                }
            )
            cursor += len(value)
            del value
        except Exception as error:
            reports.append(
                {
                    "part_index": part_index,
                    "path": str(path),
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            errors.append(f"part {part_index}: {type(error).__name__}: {error}")
    if cursor != EXPECTED_VIDEOS:
        errors.append(f"observed feature videos={cursor}, expected={EXPECTED_VIDEOS}")
    return {"parts": reports, "observed_videos": cursor, "errors": errors}


def group_manifest(manifest: list[dict]) -> OrderedDict[str, list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in manifest:
        grouped.setdefault(row["subject"], []).append(row)
    return grouped


def static_preflight(args: argparse.Namespace) -> dict:
    checks = {}
    errors = []
    train_source = args.project_root / "train.py"
    main_source = args.project_root / "main.py"
    network_source = args.project_root / "network_sf.py"
    for name, path in (("train.py", train_source), ("main.py", main_source), ("network_sf.py", network_source)):
        checks[name] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"missing {path}")
    if train_source.is_file():
        compact = re.sub(r"\s+", "", train_source.read_text(encoding="utf-8"))
        required = (
            "withtorch.no_grad():",
            "yhat,yhat1=model(x)",
            "ifi==0:",
            "result_video[framecount*k//2:(framecount+2)*k//2]=yhat[i]",
            "result_video[(framecount+1)*k//2:(framecount+2)*k//2]=yhat[i][k//2:]",
        )
        missing = [token for token in required if token not in compact]
        checks["train.py"].update({"sha256": sha256(train_source), "required_inference_tokens_pass": not missing})
        if missing:
            errors.append(f"train.py original stitching tokens missing: {missing}")
    if main_source.is_file():
        compact = re.sub(r"\s+", "", main_source.read_text(encoding="utf-8"))
        required = ('ifdataset_name=="CASME_3":', "frame_skip=1", "k=50", "batch_size=256")
        missing = [token for token in required if token not in compact]
        checks["main.py"].update({"sha256": sha256(main_source), "required_geometry_tokens_pass": not missing})
        if missing:
            errors.append(f"main.py CASME3 geometry tokens missing: {missing}")
    manifest = []
    subjects = []
    try:
        manifest, subjects = load_alignment_manifest(args.alignment_cache)
        checks["alignment"] = {
            "path": str(args.alignment_cache),
            "sha256": sha256(args.alignment_cache),
            "subjects": len(subjects),
            "videos": len(manifest),
            "status": "PASS",
        }
    except Exception as error:
        errors.append(f"alignment: {type(error).__name__}: {error}")
        checks["alignment"] = {"path": str(args.alignment_cache), "status": "FAIL"}
    try:
        skip = read_skip_manifest(args.skip_manifest)
        checks["skip_manifest"] = {
            "path": str(args.skip_manifest),
            "sha256": sha256(args.skip_manifest),
            "row": skip,
            "status": "PASS",
        }
    except Exception as error:
        errors.append(f"skip manifest: {type(error).__name__}: {error}")
        checks["skip_manifest"] = {"path": str(args.skip_manifest), "status": "FAIL"}
    parts = cache_parts(args.input_cache_dir)
    if manifest:
        feature_metadata = inspect_feature_cache_metadata(parts, manifest)
        feature_status = "PASS" if not feature_metadata["errors"] else "FAIL"
        checks["feature_cache"] = {
            "directory": str(args.input_cache_dir),
            "parts": feature_metadata["parts"],
            "found_parts": sum(path.is_file() for path in parts),
            "expected_parts": 5,
            "observed_videos": feature_metadata["observed_videos"],
            "expected_videos": EXPECTED_VIDEOS,
            "inspection_mode": "one_part_at_a_time_not_retained",
            "status": feature_status,
        }
        errors.extend(f"feature cache: {message}" for message in feature_metadata["errors"])
    else:
        checks["feature_cache"] = {
            "directory": str(args.input_cache_dir),
            "found_parts": sum(path.is_file() for path in parts),
            "expected_parts": 5,
            "status": "FAIL",
        }
        errors.append("feature cache cannot be aligned without verified compact manifest")
    checkpoint_subjects = (
        [str(args.subject)] if args.smoke_only and args.subject is not None else subjects
    )
    checkpoints = [checkpoint_path(args.weights_dir, subject) for subject in checkpoint_subjects]
    missing_checkpoints = [str(path) for path in checkpoints if not path.is_file()]
    checks["checkpoints"] = {
        "directory": str(args.weights_dir),
        "expected": len(checkpoint_subjects) if checkpoint_subjects else EXPECTED_SUBJECTS,
        "found": len(checkpoints) - len(missing_checkpoints),
        "missing_preview": missing_checkpoints[:10],
        "status": "PASS" if checkpoints and not missing_checkpoints else "FAIL",
    }
    if not checkpoints or missing_checkpoints:
        errors.append(f"checkpoint discovery found {len(checkpoints)-len(missing_checkpoints)}/{len(checkpoints) or EXPECTED_SUBJECTS}")
    report = {
        "status": "STATIC-PREFLIGHT-PASS" if not errors else "STATIC-PREFLIGHT-BLOCKED",
        "dataset": DATASET,
        "geometry": {
            "window_length": WINDOW_LENGTH,
            "stride_expression": "k//2",
            "stride": STRIDE,
            "batch_size": BATCH_SIZE,
            "frame_skip": FRAME_SKIP,
        },
        "expected_outputs": EXPECTED_VIDEOS,
        "checks": checks,
        "errors": errors,
        "cuda_initialized": False,
        "backbone_forward": False,
    }
    return report


def validate_smoke_gate(path: Path | None, args: argparse.Namespace) -> dict:
    if path is None:
        raise RuntimeError("full run requires --smoke-manifest from an exact one-video smoke PASS")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": FULL_SCHEMA,
        "status": "CASME3-RAW-WINDOW-SMOKE-PASS",
        "complete": True,
        "videos_checked": 1,
        "videos_passed": 1,
        "videos_failed": 0,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"smoke manifest {key}={value.get(key)!r}, expected={expected!r}")
    if value.get("alignment_cache_sha256") != sha256(args.alignment_cache):
        raise RuntimeError("smoke/full alignment cache mismatch")
    source_checks = value.get("source_checks", {})
    for filename in ("train.py", "main.py", "network_sf.py"):
        source = args.project_root / filename
        expected_sha = source_checks.get(filename, {}).get("sha256")
        if expected_sha != sha256(source):
            raise RuntimeError(f"smoke/full source mismatch: {filename}")
    reports = value.get("video_reports", [])
    if len(reports) != 1 or not reports[0].get("passed"):
        raise RuntimeError("smoke manifest lacks one exact reconstruction PASS")
    return {"path": str(path), "sha256": sha256(path), "subject": reports[0]["subject"], "video": reports[0]["video"]}


def load_state_dict(path: Path, device) -> dict:
    import torch

    value = torch.load(path, map_location=device)
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(value.get(key), dict):
                value = value[key]
                break
    if not isinstance(value, dict):
        raise TypeError(f"unsupported checkpoint container: {type(value)!r}")
    return {str(key).removeprefix("module."): tensor for key, tensor in value.items()}


def validate_output_shapes(scores: np.ndarray, logits: np.ndarray) -> None:
    if scores.ndim != 2 or scores.shape[1] != WINDOW_LENGTH:
        raise RuntimeError(f"unexpected raw spotting shape: {scores.shape}")
    if logits.ndim != 3 or logits.shape != (len(scores), WINDOW_LENGTH, NUM_CLASSES):
        raise RuntimeError(f"unexpected raw logits shape: {logits.shape}")


def new_stitched(n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    length = (n_windows + 1) * STRIDE
    return np.zeros(length, dtype=np.float32), np.zeros((length, NUM_CLASSES), dtype=np.float32)


def forward_subject_once(model, subject: str, entries: list[dict], features: list, device):
    import torch

    subject_features = [features[row["global_video_index"]] for row in entries]
    video_num = [len(value) for value in subject_features]
    flat_windows = [window for video_features in subject_features for window in video_features]
    flat_map = [
        (row["subject_video_index"], row["global_video_index"], row["video"], video_window_id)
        for row, video_features in zip(entries, subject_features)
        for video_window_id, _window in enumerate(video_features)
    ]
    if len(flat_windows) != len(flat_map) or any(count <= 0 for count in video_num):
        raise RuntimeError(f"{subject}: invalid subject-level flattened window map")
    captured = {row["video"]: [] for row in entries}
    result_all = []
    logits_all = []
    video_index = 0
    framecount = 0
    result_video, logits_video = new_stitched(video_num[0])
    model_forward_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_index, batch_start in enumerate(range(0, len(flat_windows), BATCH_SIZE)):
            batch_stop = min(batch_start + BATCH_SIZE, len(flat_windows))
            batch = np.asarray(flat_windows[batch_start:batch_stop], dtype=np.float32)
            x = torch.as_tensor(batch)[:, None, :].to(device)
            yhat, yhat1 = model(x)
            model_forward_batches += 1

            # Required capture point: before stitching, slicing, or argmax.
            raw_scores = yhat.detach().cpu().numpy().astype(np.float32, copy=True)
            raw_logits = yhat1.detach().cpu().numpy().astype(np.float32, copy=True)
            validate_output_shapes(raw_scores, raw_logits)

            for index_within_batch in range(len(raw_scores)):
                subject_flat_index = batch_start + index_within_batch
                mapped_video_index, global_video_index, video, video_window_id = flat_map[subject_flat_index]
                if framecount == video_num[video_index]:
                    result_all.append(result_video)
                    logits_all.append(logits_video)
                    video_index += 1
                    framecount = 0
                    result_video, logits_video = new_stitched(video_num[video_index])
                if mapped_video_index != video_index or video_window_id != framecount:
                    raise RuntimeError(f"{subject}: subject/video/window order drift before write")
                global_start = video_window_id * STRIDE
                captured[video].append(
                    {
                        "global_video_index": global_video_index,
                        "subject_video_index": video_index,
                        "subject_flat_window_index": subject_flat_index,
                        "batch_index": batch_index,
                        "index_within_batch": index_within_batch,
                        "video_window_id": video_window_id,
                        "framecount_before_write": framecount,
                        "global_start": global_start,
                        "global_indices": global_start + np.arange(WINDOW_LENGTH, dtype=np.int32),
                        "raw_window_score": raw_scores[index_within_batch].copy(),
                        "raw_window_logits": raw_logits[index_within_batch].copy(),
                    }
                )
                # Exact original train.py batch-sensitive stitching branch.
                if index_within_batch == 0:
                    start, stop = framecount * STRIDE, (framecount + 2) * STRIDE
                    result_video[start:stop] = raw_scores[index_within_batch]
                    logits_video[start:stop] = raw_logits[index_within_batch]
                else:
                    start, stop = (framecount + 1) * STRIDE, (framecount + 2) * STRIDE
                    result_video[start:stop] = raw_scores[index_within_batch, STRIDE:]
                    logits_video[start:stop] = raw_logits[index_within_batch, STRIDE:]
                framecount += 1
    result_all.append(result_video)
    logits_all.append(logits_video)
    if len(result_all) != len(entries) or model_forward_batches != math.ceil(len(flat_windows) / BATCH_SIZE):
        raise RuntimeError(f"{subject}: incomplete subject traversal")
    outputs = {}
    for row, n_windows in zip(entries, video_num):
        index = row["subject_video_index"]
        outputs[row["video"]] = {
            "captured": captured[row["video"]],
            "original_fresh_result_video": result_all[index],
            "original_fresh_logits_video": logits_all[index],
            "subject_video_count": len(entries),
            "subject_total_windows": len(flat_windows),
            "target_subject_video_index": index,
            "target_num_windows": n_windows,
        }
    return outputs, model_forward_batches


def output_path(output_dir: Path, subject: str, video: str) -> Path:
    safe_video = video.replace("/", "_")
    return output_dir / f"{subject}_{safe_video}_raw_prestitch.npz"


def save_dump(path: Path, subject: str, video: str, result: dict) -> None:
    rows = result["captured"]
    arrays = {
        "schema": np.asarray(SCHEMA),
        "subject": np.asarray(subject),
        "video": np.asarray(video),
        "batch_size": np.asarray(BATCH_SIZE, dtype=np.int32),
        "window_length": np.asarray(WINDOW_LENGTH, dtype=np.int32),
        "stride": np.asarray(STRIDE, dtype=np.int32),
        "frame_skip": np.asarray(FRAME_SKIP, dtype=np.int32),
        "global_video_index": np.asarray([row["global_video_index"] for row in rows], dtype=np.int32),
        "subject_video_index": np.asarray([row["subject_video_index"] for row in rows], dtype=np.int32),
        "subject_flat_window_index": np.asarray([row["subject_flat_window_index"] for row in rows], dtype=np.int32),
        "batch_index": np.asarray([row["batch_index"] for row in rows], dtype=np.int32),
        "index_within_batch": np.asarray([row["index_within_batch"] for row in rows], dtype=np.int16),
        "video_window_id": np.asarray([row["video_window_id"] for row in rows], dtype=np.int32),
        "framecount_before_write": np.asarray([row["framecount_before_write"] for row in rows], dtype=np.int32),
        "global_start": np.asarray([row["global_start"] for row in rows], dtype=np.int32),
        "global_indices": np.stack([row["global_indices"] for row in rows]),
        "raw_window_score": np.stack([row["raw_window_score"] for row in rows]),
        "raw_window_logits": np.stack([row["raw_window_logits"] for row in rows]),
        "original_fresh_result_video": result["original_fresh_result_video"],
        "original_fresh_logits_video": result["original_fresh_logits_video"],
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def reconstruct_and_compare(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["schema"]).item()) != SCHEMA:
            raise RuntimeError("unexpected CASME3 raw dump schema")
        raw_score = np.asarray(data["raw_window_score"])
        raw_logits = np.asarray(data["raw_window_logits"])
        framecount = np.asarray(data["framecount_before_write"], dtype=int)
        batch_local = np.asarray(data["index_within_batch"], dtype=int)
        original_score = np.asarray(data["original_fresh_result_video"])
        original_logits = np.asarray(data["original_fresh_logits_video"])
        score = np.zeros_like(original_score)
        logits = np.zeros_like(original_logits)
        for row in range(len(raw_score)):
            if batch_local[row] == 0:
                start, stop = framecount[row] * STRIDE, (framecount[row] + 2) * STRIDE
                score[start:stop] = raw_score[row]
                logits[start:stop] = raw_logits[row]
            else:
                start, stop = (framecount[row] + 1) * STRIDE, (framecount[row] + 2) * STRIDE
                score[start:stop] = raw_score[row, STRIDE:]
                logits[start:stop] = raw_logits[row, STRIDE:]
        def compare(actual, expected):
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            equal = bool(np.array_equal(actual, expected))
            if equal:
                pearson = 1.0
            elif actual.size and np.std(actual) and np.std(expected):
                pearson = float(np.corrcoef(actual.reshape(-1), expected.reshape(-1))[0, 1])
            else:
                pearson = None
            return {
                "shape_match": actual.shape == expected.shape,
                "max_abs_error": float(difference.max(initial=0.0)),
                "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
                "np_allclose": bool(np.allclose(actual, expected, rtol=0.0, atol=0.0)),
                "array_equal": equal,
                "pearson": pearson,
            }
        score_result = compare(score, original_score)
        logits_result = compare(logits, original_logits)
        return {
            "subject": str(np.asarray(data["subject"]).item()),
            "video": str(np.asarray(data["video"]).item()),
            "num_windows": len(raw_score),
            "score_reconstruction": score_result,
            "logits_reconstruction": logits_result,
            "passed": score_result["array_equal"] and logits_result["array_equal"],
        }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    preflight = static_preflight(args)
    if args.static_preflight:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["status"] == "STATIC-PREFLIGHT-PASS" else 2
    if preflight["status"] != "STATIC-PREFLIGHT-PASS":
        raise RuntimeError("static preflight failed:\n" + "\n".join(preflight["errors"]))
    if args.smoke_only and (args.subject is None or args.video is None):
        raise ValueError("--smoke-only requires both --subject and --video")
    if not args.smoke_only and (args.subject is not None or args.video is not None):
        raise ValueError("--subject/--video are reserved for --smoke-only")
    if args.smoke_only and args.smoke_manifest is not None:
        raise ValueError("--smoke-manifest is only valid for the subsequent full run")
    smoke_gate = None if args.smoke_only else validate_smoke_gate(args.smoke_manifest, args)

    # Torch/CUDA is deliberately imported only after all filesystem/provenance checks.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no backbone forward was attempted")
    manifest, subjects = load_alignment_manifest(args.alignment_cache)
    grouped = group_manifest(manifest)
    features, part_reports = inspect_feature_cache(cache_parts(args.input_cache_dir), manifest)
    skip_row = read_skip_manifest(args.skip_manifest)
    target_subjects = [str(args.subject)] if args.smoke_only else subjects
    if any(subject not in grouped for subject in target_subjects):
        raise ValueError(f"unknown CASME3 subject: {target_subjects}")
    if args.smoke_only and not any(row["video"] == str(args.video) for row in grouped[target_subjects[0]]):
        raise ValueError(f"unknown CASME3 video: {args.subject}/{args.video}")

    outputs_to_write = []
    for subject in target_subjects:
        for row in grouped[subject]:
            if not args.smoke_only or row["video"] == str(args.video):
                outputs_to_write.append(output_path(args.output_dir, subject, row["video"]))
    final_manifest = args.output_dir / ("smoke_manifest.json" if args.smoke_only else "full_manifest.json")
    existing = [path for path in outputs_to_write + [final_manifest] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {len(existing)} existing output(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.project_root))
    from network_sf import METST_SF

    device = torch.device("cuda")
    video_reports = []
    subject_reports = []
    checkpoint_hashes = {}
    for subject in target_subjects:
        checkpoint = checkpoint_path(args.weights_dir, subject)
        checkpoint_hashes[subject] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
        model = METST_SF(out_channels=NUM_CLASSES).to(device)
        model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
        model.eval()
        subject_outputs, forward_batches = forward_subject_once(
            model, subject, grouped[subject], features, device
        )
        selected_entries = [
            row for row in grouped[subject]
            if not args.smoke_only or row["video"] == str(args.video)
        ]
        for row in selected_entries:
            path = output_path(args.output_dir, subject, row["video"])
            save_dump(path, subject, row["video"], subject_outputs[row["video"]])
            reconstruction = reconstruct_and_compare(path)
            report = {
                **reconstruction,
                "path": str(path),
                "relative_path": path.name,
                "sha256": sha256(path),
            }
            video_reports.append(report)
            print(f"VERIFIED {subject}/{row['video']}: array_equal={report['passed']}")
            if not report["passed"]:
                raise RuntimeError(f"RAW-WINDOW-DUMP-EQUIVALENCE-FAIL: {subject}/{row['video']}")
        subject_reports.append(
            {
                "subject": subject,
                "checkpoint_loads": 1,
                "subject_forward_passes": 1,
                "model_forward_batches": forward_batches,
                "subject_videos_forwarded": len(grouped[subject]),
                "videos_saved": len(selected_entries),
            }
        )
        del model, subject_outputs
        torch.cuda.empty_cache()
        if not args.smoke_only:
            write_json_atomic(
                final_manifest,
                {
                    "schema": FULL_SCHEMA,
                    "status": "CASME3-RAW-WINDOW-SOURCE-INCOMPLETE",
                    "complete": False,
                    "subjects_completed": len(subject_reports),
                    "videos_checked": len(video_reports),
                    "subject_reports": subject_reports,
                    "video_reports": video_reports,
                },
            )

    expected_saved = 1 if args.smoke_only else EXPECTED_VIDEOS
    all_passed = len(video_reports) == expected_saved and all(row["passed"] for row in video_reports)
    status = (
        "CASME3-RAW-WINDOW-SMOKE-PASS"
        if args.smoke_only and all_passed
        else "CASME3-RAW-WINDOW-SOURCE-READY"
        if all_passed
        else "CASME3-RAW-WINDOW-SOURCE-NOT-READY"
    )
    score_errors = [row["score_reconstruction"]["max_abs_error"] for row in video_reports]
    logits_errors = [row["logits_reconstruction"]["max_abs_error"] for row in video_reports]
    score_pearsons = [row["score_reconstruction"]["pearson"] for row in video_reports]
    logits_pearsons = [row["logits_reconstruction"]["pearson"] for row in video_reports]
    finite_score_pearsons = [value for value in score_pearsons if value is not None and np.isfinite(value)]
    finite_logits_pearsons = [value for value in logits_pearsons if value is not None and np.isfinite(value)]
    source_checks = {
        filename: {"path": str(args.project_root / filename), "sha256": sha256(args.project_root / filename)}
        for filename in ("train.py", "main.py", "network_sf.py")
    }
    result = {
        "schema": FULL_SCHEMA,
        "status": status,
        "complete": all_passed,
        "dataset": DATASET,
        "model": MODEL_NAME,
        "subjects": EXPECTED_SUBJECTS,
        "videos": EXPECTED_VIDEOS,
        "subjects_executed": len(target_subjects),
        "videos_checked": len(video_reports),
        "videos_passed": sum(row["passed"] for row in video_reports),
        "videos_failed": sum(not row["passed"] for row in video_reports),
        "worst_score_max_abs_error": max(score_errors, default=None),
        "worst_logits_max_abs_error": max(logits_errors, default=None),
        "minimum_score_pearson": min(finite_score_pearsons, default=None),
        "minimum_logits_pearson": min(finite_logits_pearsons, default=None),
        "train": False,
        "model_eval": True,
        "torch_no_grad": True,
        "training_calls": {"backward": 0, "optimizer_step": 0},
        "geometry": {"k": WINDOW_LENGTH, "stride_expression": "k//2", "stride": STRIDE, "batch_size": BATCH_SIZE, "frame_skip": FRAME_SKIP},
        "alignment_cache": str(args.alignment_cache),
        "alignment_cache_sha256": sha256(args.alignment_cache),
        "skip_manifest": str(args.skip_manifest),
        "skip_manifest_sha256": sha256(args.skip_manifest),
        "skip_manifest_row": skip_row,
        "feature_cache_parts": part_reports,
        "checkpoint_hashes": checkpoint_hashes,
        "smoke_gate": smoke_gate,
        "project_root": str(args.project_root),
        "source_commit": source_commit(args.project_root),
        "source_checks": source_checks,
        "subject_reports": subject_reports,
        "video_reports": video_reports,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0)},
    }
    write_json_atomic(final_manifest, result)
    print(json.dumps({"status": status, "videos_checked": len(video_reports), "videos_passed": result["videos_passed"]}, ensure_ascii=False, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
