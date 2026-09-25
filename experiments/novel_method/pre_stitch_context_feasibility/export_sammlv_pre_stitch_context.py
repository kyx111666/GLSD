#!/usr/bin/env python3
"""Export raw ME-TST+ spotting-window scores for all SAMMLV videos.

Run only in the restored Colab CUDA environment.  This performs frozen
inference and preserves both every raw window observation and the exact
batch-sensitive fresh stitching.  It does not read historical predictions.
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
STRIDE = 15
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path: Path) -> list:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        for key in ("features", "data", "final_dataset_spotting", "dataset"):
            if isinstance(value.get(key), (list, tuple)):
                return list(value[key])
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
        str(key).replace("module.", "", 1) if str(key).startswith("module.") else str(key): tensor
        for key, tensor in value.items()
    }


def groups() -> OrderedDict[str, list[tuple[int, str]]]:
    output: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    for index, (subject, video) in enumerate(VIDEO_MANIFEST):
        output.setdefault(subject, []).append((index, video))
    return output


def stitch_subject(window_scores: np.ndarray, video_num: list[int]) -> list[np.ndarray]:
    outputs = []
    video_index = 0
    frame_count = 0
    global_index = 0
    current = np.zeros((video_num[0] + 1) * STRIDE, dtype=np.float32)
    for batch_start in range(0, len(window_scores), BATCH_SIZE):
        batch_stop = min(batch_start + BATCH_SIZE, len(window_scores))
        for batch_local_index in range(batch_stop - batch_start):
            if frame_count == video_num[video_index]:
                outputs.append(current)
                video_index += 1
                frame_count = 0
                current = np.zeros((video_num[video_index] + 1) * STRIDE, dtype=np.float32)
            score = window_scores[global_index]
            if batch_local_index == 0:
                current[frame_count * STRIDE : (frame_count + 2) * STRIDE] = score
            else:
                current[(frame_count + 1) * STRIDE : (frame_count + 2) * STRIDE] = score[STRIDE:]
            frame_count += 1
            global_index += 1
    outputs.append(current)
    if len(outputs) != len(video_num):
        raise RuntimeError(f"stitched videos={len(outputs)} expected={len(video_num)}")
    return outputs


def source_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN_NOT_GIT"


def main() -> int:
    import torch

    args = arguments()
    output_cache = args.output_dir / "SAMMLV_PRE_STITCH_CONTEXT_CACHE.pkl"
    output_manifest = args.output_dir / "sammlv_pre_stitch_context_manifest.json"
    for path in (args.project_root / "network_sf.py", args.input_cache, args.weights_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_cache.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing cache: {output_cache}")
    if not torch.cuda.is_available():
        raise RuntimeError("run in the restored Colab CUDA environment")

    features = load_features(args.input_cache)
    if len(features) != len(VIDEO_MANIFEST):
        raise RuntimeError(f"feature videos={len(features)} expected=79")
    sys.path.insert(0, str(args.project_root))
    from network_sf import METST_SF

    device = torch.device("cuda")
    records = []
    checkpoint_hashes = {}
    for subject, entries in groups().items():
        checkpoint = args.weights_dir / f"subject_{subject}.pkl"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[subject] = {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        }
        model = METST_SF(out_channels=5).to(device)
        model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
        model.eval()

        subject_features = [features[index] for index, _ in entries]
        video_num = [len(value) for value in subject_features]
        flat_windows = [window for video in subject_features for window in video]
        batches = []
        with torch.no_grad():
            for start in range(0, len(flat_windows), BATCH_SIZE):
                batch = np.asarray(flat_windows[start : start + BATCH_SIZE], dtype=np.float32)
                x = torch.as_tensor(batch, device=device)[:, None, :]
                score, _unused_recognition_logits = model(x)
                score = score.detach().cpu().numpy().astype(np.float32)
                if score.ndim != 2 or score.shape[1] != WINDOW_LENGTH:
                    raise RuntimeError(f"unexpected spotting output {score.shape}")
                batches.append(score)
        window_scores = np.concatenate(batches, axis=0)
        if len(window_scores) != sum(video_num):
            raise RuntimeError("subject window count changed during forward")
        stitched = stitch_subject(window_scores, video_num)

        offset = 0
        for subject_video_index, ((global_video_index, video_id), n_windows) in enumerate(zip(entries, video_num)):
            raw = window_scores[offset : offset + n_windows]
            offset += n_windows
            window_id = np.repeat(np.arange(n_windows, dtype=np.int32), WINDOW_LENGTH)
            local_t = np.tile(np.arange(WINDOW_LENGTH, dtype=np.int16), n_windows)
            global_t = window_id * STRIDE + local_t
            raw_score = raw.reshape(-1).astype(np.float32)
            length = (n_windows + 1) * STRIDE
            context_count = np.bincount(global_t, minlength=length).astype(np.int16)
            if len(stitched[subject_video_index]) != length:
                raise RuntimeError(f"{subject}/{video_id}: stitched length mismatch")
            if context_count.min() != 1 or context_count.max() > 2:
                raise RuntimeError(f"{subject}/{video_id}: unexpected context multiplicity")
            records.append(
                {
                    "subject": subject,
                    "video_id": video_id,
                    "global_video_index": global_video_index,
                    "subject_video_index": subject_video_index,
                    "num_windows": n_windows,
                    "window_id": window_id,
                    "local_t": local_t,
                    "global_t": global_t.astype(np.int32),
                    "raw_score": raw_score,
                    "fresh_stitched_score": stitched[subject_video_index],
                    "context_count": context_count,
                }
            )
            print(f"SAVED {subject}/{video_id}: windows={n_windows}, T={length}, raw={len(raw_score)}")
        del model
        torch.cuda.empty_cache()

    cache = {
        "schema": "sammlv_pre_stitch_context_v1",
        "status": "COMPLETE",
        "dataset": "SAMMLV",
        "model": "ME-TST+",
        "inference_only": True,
        "model_eval": True,
        "torch_no_grad": True,
        "training_calls": {"model_train": 0, "backward": 0, "optimizer_step": 0},
        "historical_prediction_fields_used": [],
        "source_commit": source_commit(args.project_root),
        "project_root": str(args.project_root),
        "input_feature_cache": str(args.input_cache),
        "input_feature_cache_sha256": sha256(args.input_cache),
        "weights_dir": str(args.weights_dir),
        "checkpoint_hashes": checkpoint_hashes,
        "batch_size": BATCH_SIZE,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "stitching": "original subject-level batching; batch-local i==0 writes full window, otherwise writes local_t[15:30]",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
        },
        "num_subjects": len(groups()),
        "num_videos": len(records),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_cache.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        key: value for key, value in cache.items()
        if key not in ("records", "checkpoint_hashes")
    }
    manifest["checkpoint_hashes"] = checkpoint_hashes
    manifest["output_cache"] = str(output_cache)
    manifest["output_cache_sha256"] = sha256(output_cache)
    manifest["raw_observations"] = int(sum(len(row["raw_score"]) for row in records))
    manifest["total_stitched_T"] = int(sum(len(row["fresh_stitched_score"]) for row in records))
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
