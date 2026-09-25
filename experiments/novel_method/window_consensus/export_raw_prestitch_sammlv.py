#!/usr/bin/env python3
"""Dump raw ME-TST+ SAMMLV window outputs before original stitching.

This is instrumentation only.  It runs one frozen subject checkpoint in eval
mode, preserves the original subject-level window order and batch size, and
saves the requested video's raw spotting scores/logits together with the
fresh result produced by the original batch-sensitive stitching rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


BATCH_SIZE = 32
WINDOW_LENGTH = 30
STRIDE = WINDOW_LENGTH // 2
NUM_CLASSES = 5

# Frozen ordering used by the verified SAMMLV cache.
VIDEO_MANIFEST = [
    ("006", "006_1"), ("006", "006_2"), ("006", "006_3"), ("006", "006_5"),
    ("007", "007_3"), ("007", "007_4"), ("007", "007_5"), ("007", "007_6"), ("007", "007_7"),
    ("009", "009_2"), ("009", "009_3"),
    ("010", "010_2"), ("010", "010_4"),
    ("011", "011_1"), ("011", "011_2"), ("011", "011_3"), ("011", "011_4"),
    ("011", "011_5"), ("011", "011_6"), ("011", "011_7"),
    ("012", "012_3"), ("012", "012_7"),
    ("013", "013_1"), ("013", "013_7"),
    ("014", "014_1"), ("014", "014_2"), ("014", "014_3"), ("014", "014_5"),
    ("014", "014_6"), ("014", "014_7"),
    ("015", "015_5"), ("016", "016_7"),
    ("017", "017_3"), ("017", "017_6"),
    ("018", "018_1"), ("018", "018_3"), ("018", "018_5"), ("018", "018_7"),
    ("019", "019_3"), ("019", "019_4"), ("019", "019_5"),
    ("020", "020_1"), ("020", "020_4"), ("020", "020_7"),
    ("021", "021_7"),
    ("022", "022_2"), ("022", "022_3"), ("022", "022_4"), ("022", "022_5"),
    ("023", "023_1"), ("024", "024_2"),
    ("025", "025_4"), ("025", "025_5"), ("025", "025_6"),
    ("026", "026_1"), ("026", "026_2"), ("026", "026_3"), ("026", "026_5"),
    ("026", "026_6"), ("026", "026_7"),
    ("028", "028_4"),
    ("030", "030_1"), ("030", "030_5"),
    ("031", "031_3"),
    ("032", "032_3"), ("032", "032_4"), ("032", "032_6"),
    ("033", "033_1"), ("033", "033_2"),
    ("034", "034_3"), ("034", "034_7"),
    ("035", "035_1"), ("035", "035_4"), ("035", "035_5"), ("035", "035_6"), ("035", "035_7"),
    ("036", "036_7"),
    ("037", "037_3"), ("037", "037_4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="006")
    parser.add_argument("--video", default="006_1")
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


def load_features(path: Path) -> list:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        for key in ("features", "data", "final_dataset_spotting", "dataset"):
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)):
                return list(candidate)
    raise TypeError(f"unsupported input cache container: {type(value)!r}")


def load_state_dict(path: Path, device) -> dict:
    import torch

    value = torch.load(path, map_location=device)
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(value.get(key), dict):
                value = value[key]
                break
    if not isinstance(value, dict):
        raise TypeError(f"unsupported checkpoint: {type(value)!r}")
    return {
        str(key).removeprefix("module."): tensor
        for key, tensor in value.items()
    }


def manifest_groups() -> OrderedDict[str, list[tuple[int, str]]]:
    grouped: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    for global_video_index, (subject, video) in enumerate(VIDEO_MANIFEST):
        grouped.setdefault(subject, []).append((global_video_index, video))
    return grouped


def validate_output_shapes(score: np.ndarray, logits: np.ndarray) -> None:
    if score.ndim != 2 or score.shape[1] != WINDOW_LENGTH:
        raise RuntimeError(f"unexpected raw spotting shape: {score.shape}")
    if logits.ndim != 3 or logits.shape[:2] != score.shape:
        raise RuntimeError(f"unexpected raw logits shape: {logits.shape}")
    if logits.shape[2] != NUM_CLASSES:
        raise RuntimeError(f"expected {NUM_CLASSES} recognition classes, got {logits.shape[2]}")


def forward_and_instrument(
    model,
    subject: str,
    entries: list[tuple[int, str]],
    features: list,
    target_video: str,
    device,
) -> dict:
    """Run the unchanged subject-level batching/stitching and capture one video."""
    import torch

    subject_features = [features[index] for index, _ in entries]
    video_num = [len(video_features) for video_features in subject_features]
    flat_windows = [window for video_features in subject_features for window in video_features]
    flat_map = [
        (subject_video_index, global_video_index, video, video_window_id)
        for subject_video_index, ((global_video_index, video), video_features) in enumerate(
            zip(entries, subject_features)
        )
        for video_window_id, _window in enumerate(video_features)
    ]
    if len(flat_windows) != len(flat_map):
        raise RuntimeError("flat window mapping changed length")

    result_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    captured: list[dict] = []
    video_index = 0
    framecount = 0
    result_video = np.zeros((video_num[0] + 1) * STRIDE, dtype=np.float32)
    logits_video = np.zeros(((video_num[0] + 1) * STRIDE, NUM_CLASSES), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch_index, batch_start in enumerate(range(0, len(flat_windows), BATCH_SIZE)):
            batch_stop = min(batch_start + BATCH_SIZE, len(flat_windows))
            batch = np.asarray(flat_windows[batch_start:batch_stop], dtype=np.float32)
            # Match the verified path's CPU tensor construction followed by
            # transfer to the frozen device.
            x = torch.as_tensor(batch)[:, None, :].to(device)
            yhat, yhat1 = model(x)

            # Capture both tensors immediately after model forward, before any
            # stitching, overwrite, slicing, or recognition argmax.
            raw_scores = yhat.detach().cpu().numpy().astype(np.float32, copy=True)
            raw_logits = yhat1.detach().cpu().numpy().astype(np.float32, copy=True)
            validate_output_shapes(raw_scores, raw_logits)

            for index_within_batch in range(len(raw_scores)):
                subject_flat_index = batch_start + index_within_batch
                mapped_video_index, global_video_index, video, video_window_id = flat_map[
                    subject_flat_index
                ]

                if framecount == video_num[video_index]:
                    result_all.append(result_video)
                    logits_all.append(logits_video)
                    video_index += 1
                    framecount = 0
                    result_video = np.zeros(
                        (video_num[video_index] + 1) * STRIDE, dtype=np.float32
                    )
                    logits_video = np.zeros(
                        ((video_num[video_index] + 1) * STRIDE, NUM_CLASSES),
                        dtype=np.float32,
                    )

                if mapped_video_index != video_index or video_window_id != framecount:
                    raise RuntimeError(
                        "manifest/window traversal mismatch before original write: "
                        f"mapped=({mapped_video_index},{video_window_id}) "
                        f"live=({video_index},{framecount})"
                    )

                framecount_before_write = framecount
                global_start = video_window_id * STRIDE
                if video == target_video:
                    captured.append(
                        {
                            "subject": subject,
                            "video": video,
                            "global_video_index": global_video_index,
                            "subject_video_index": video_index,
                            "subject_flat_window_index": subject_flat_index,
                            "batch_index": batch_index,
                            "index_within_batch": index_within_batch,
                            "video_window_id": video_window_id,
                            "framecount_before_write": framecount_before_write,
                            "global_start": global_start,
                            "global_indices": global_start + np.arange(WINDOW_LENGTH, dtype=np.int32),
                            "raw_window_score": raw_scores[index_within_batch].copy(),
                            "raw_window_logits": raw_logits[index_within_batch].copy(),
                        }
                    )

                # Original batch-sensitive stitching rule.  Do not simplify.
                if index_within_batch == 0:
                    start_idx = framecount * STRIDE
                    end_idx = (framecount + 2) * STRIDE
                    result_video[start_idx:end_idx] = raw_scores[index_within_batch]
                    logits_video[start_idx:end_idx] = raw_logits[index_within_batch]
                else:
                    start_idx = (framecount + 1) * STRIDE
                    end_idx = (framecount + 2) * STRIDE
                    result_video[start_idx:end_idx] = raw_scores[index_within_batch, STRIDE:]
                    logits_video[start_idx:end_idx] = raw_logits[index_within_batch, STRIDE:]
                framecount += 1

    result_all.append(result_video)
    logits_all.append(logits_video)
    if len(result_all) != len(entries):
        raise RuntimeError(f"stitched videos={len(result_all)} expected={len(entries)}")
    if not captured:
        raise RuntimeError(f"target video was not captured: {subject}/{target_video}")

    target_subject_video_index = next(
        index for index, (_global_index, video) in enumerate(entries) if video == target_video
    )
    return {
        "captured": captured,
        "original_fresh_result_video": result_all[target_subject_video_index],
        "original_fresh_logits_video": logits_all[target_subject_video_index],
        "subject_video_count": len(entries),
        "subject_total_windows": len(flat_windows),
        "target_subject_video_index": target_subject_video_index,
        "target_num_windows": video_num[target_subject_video_index],
    }


def save_dump(path: Path, subject: str, video: str, result: dict) -> None:
    rows = result["captured"]
    arrays = {
        "schema": np.asarray("sammlv_raw_prestitch_window_dump_v1"),
        "subject": np.asarray(subject),
        "video": np.asarray(video),
        "batch_size": np.asarray(BATCH_SIZE, dtype=np.int32),
        "window_length": np.asarray(WINDOW_LENGTH, dtype=np.int32),
        "stride": np.asarray(STRIDE, dtype=np.int32),
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


def main() -> int:
    import torch

    args = parse_args()
    subject = str(args.subject).zfill(3)
    video = str(args.video)
    if (subject, video) not in VIDEO_MANIFEST:
        raise ValueError(f"unknown frozen SAMMLV target: {subject}/{video}")
    if not video.startswith(subject + "_"):
        raise ValueError(f"video {video!r} does not belong to subject {subject!r}")

    required = (args.project_root / "network_sf.py", args.input_cache, args.weights_dir)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run in the restored frozen GPU environment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{video}_raw_prestitch"
    output_dump = args.output_dir / f"{stem}.npz"
    output_manifest = args.output_dir / f"{stem}_manifest.json"
    for output in (output_dump, output_manifest):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {output}")

    features = load_features(args.input_cache)
    if len(features) != len(VIDEO_MANIFEST):
        raise RuntimeError(f"feature videos={len(features)} expected={len(VIDEO_MANIFEST)}")
    entries = manifest_groups()[subject]
    checkpoint = args.weights_dir / f"subject_{subject}.pkl"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(args.project_root))
    from network_sf import METST_SF

    device = torch.device("cuda")
    model = METST_SF(out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
    model.eval()
    result = forward_and_instrument(model, subject, entries, features, video, device)
    save_dump(output_dump, subject, video, result)

    manifest = {
        "schema": "sammlv_raw_prestitch_window_dump_manifest_v1",
        "status": "RAW-WINDOW-DUMP-WRITTEN-AWAITING-EQUIVALENCE-CHECK",
        "dataset": "SAMMLV",
        "model": "ME-TST+",
        "subject": subject,
        "video": video,
        "train": False,
        "model_eval": True,
        "torch_no_grad": True,
        "training_calls": {"backward": 0, "optimizer_step": 0},
        "batch_size": BATCH_SIZE,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "subject_video_count": result["subject_video_count"],
        "subject_total_windows": result["subject_total_windows"],
        "target_subject_video_index": result["target_subject_video_index"],
        "target_num_windows": result["target_num_windows"],
        "raw_window_score_shape": [result["target_num_windows"], WINDOW_LENGTH],
        "raw_window_logits_shape": [result["target_num_windows"], WINDOW_LENGTH, NUM_CLASSES],
        "stitching": "subject-level batch order; batch-local i==0 writes all 30, otherwise writes local_t[15:30]",
        "project_root": str(args.project_root),
        "source_commit": source_commit(args.project_root),
        "input_cache": str(args.input_cache),
        "input_cache_sha256": sha256(args.input_cache),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "output_dump": str(output_dump),
        "output_dump_sha256": sha256(output_dump),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
