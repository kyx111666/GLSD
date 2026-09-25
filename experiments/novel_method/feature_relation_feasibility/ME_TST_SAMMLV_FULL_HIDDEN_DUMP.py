#!/usr/bin/env python3
"""Fresh frozen ME-TST+ hidden/score dump for all 79 SAMMLV videos.

The historical compact cache is not read.  The embedded manifest contains
only the original subject/video order.  Numerical outputs are regenerated
from the input feature cache and frozen subject checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


BATCH_SIZE = 32
K = 30
HIDDEN_DIM = 384

# Order recovered from the paper-aligned SAMMLV manifest.  No cached score,
# logit, emotion prediction, or label is embedded or consumed.
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


def state_dict(path: Path, device) -> dict:
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


def stitch(window_scores: np.ndarray, window_hidden: np.ndarray, video_num: list[int]) -> list[dict]:
    """Exact original subject-level batch-sensitive stitching."""
    result = []
    videocount = 0
    framecount = 0
    score_video = np.zeros((video_num[0] + 1) * K // 2, dtype=np.float32)
    hidden_video = np.zeros((len(score_video), HIDDEN_DIM), dtype=np.float32)
    valid_video = np.zeros(len(score_video), dtype=bool)
    global_index = 0

    for batch_start in range(0, len(window_scores), BATCH_SIZE):
        batch_stop = min(batch_start + BATCH_SIZE, len(window_scores))
        for i in range(batch_stop - batch_start):
            if framecount == video_num[videocount]:
                result.append({"score": score_video, "hidden": hidden_video, "valid_mask": valid_video})
                videocount += 1
                framecount = 0
                score_video = np.zeros((video_num[videocount] + 1) * K // 2, dtype=np.float32)
                hidden_video = np.zeros((len(score_video), HIDDEN_DIM), dtype=np.float32)
                valid_video = np.zeros(len(score_video), dtype=bool)

            score = window_scores[global_index]
            hidden = window_hidden[global_index]
            if i == 0:
                start = framecount * K // 2
                stop = (framecount + 2) * K // 2
                score_video[start:stop] = score
                hidden_video[start:stop] = hidden
                valid_video[start:stop] = True
            else:
                start = (framecount + 1) * K // 2
                stop = (framecount + 2) * K // 2
                score_video[start:stop] = score[K // 2 :]
                hidden_video[start:stop] = hidden[K // 2 :]
                valid_video[start:stop] = True
            framecount += 1
            global_index += 1

    result.append({"score": score_video, "hidden": hidden_video, "valid_mask": valid_video})
    if len(result) != len(video_num):
        raise RuntimeError(f"stitch produced {len(result)} videos; expected {len(video_num)}")
    return result


def subject_groups() -> OrderedDict[str, list[tuple[int, str]]]:
    groups: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    for index, (subject, video) in enumerate(VIDEO_MANIFEST):
        groups.setdefault(subject, []).append((index, video))
    return groups


def main() -> int:
    import torch

    args = arguments()
    for path in (args.project_root / "network_sf.py", args.input_cache, args.weights_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the restored ME-TST environment")

    sys.path.insert(0, str(args.project_root))
    from network_sf import METST_SF

    features = load_features(args.input_cache)
    if len(features) != len(VIDEO_MANIFEST):
        raise RuntimeError(f"input cache has {len(features)} videos; manifest has {len(VIDEO_MANIFEST)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    records = []
    total_t = 0

    for subject, entries in subject_groups().items():
        checkpoint = args.weights_dir / f"subject_{subject}.pkl"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        subject_dir = args.output_dir / f"subject_{subject}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        expected = [subject_dir / f"{video}_frozen_hidden_score.npz" for _, video in entries]
        if not args.overwrite and all(path.exists() for path in expected):
            print(f"SKIP subject {subject}: all {len(expected)} outputs exist")
            checkpoint_hash = sha256(checkpoint)
            for local_index, ((global_index, video), output_path) in enumerate(zip(entries, expected)):
                with np.load(output_path, allow_pickle=False) as saved:
                    hidden_shape = saved["hidden"].shape
                    score_shape = saved["score"].shape
                    if "valid_mask" not in saved.files:
                        raise RuntimeError(
                            f"existing output predates valid-mask fix: {output_path}; rerun with --overwrite"
                        )
                    valid_mask = saved["valid_mask"]
                if len(hidden_shape) != 2 or hidden_shape[1] != HIDDEN_DIM or score_shape != (hidden_shape[0],):
                    raise RuntimeError(f"invalid existing output {output_path}: {hidden_shape}, {score_shape}")
                total_t += int(hidden_shape[0])
                records.append({
                    "subject": subject,
                    "video_id": video,
                    "global_video_index": global_index,
                    "subject_video_index": local_index,
                    "T_hidden": int(hidden_shape[0]),
                    "hidden_dim": int(hidden_shape[1]),
                    "T_score": int(score_shape[0]),
                    "max_window_head_error": None,
                    "max_stitched_head_error": None,
                    "num_valid_positions": int(valid_mask.sum()),
                    "num_unwritten_zero_positions": int((~valid_mask).sum()),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "output_path": str(output_path),
                    "resumed_existing_output": True,
                })
            continue

        model = METST_SF(out_channels=5).to(device)
        model.load_state_dict(state_dict(checkpoint, device), strict=True)
        model.eval()
        captured = []

        def hook(_module, _inputs, output):
            captured.append(output.detach())

        handle = model.spot_pathway.register_forward_hook(hook)
        subject_features = [features[index] for index, _ in entries]
        video_num = [len(video_features) for video_features in subject_features]
        flat = [window for video_features in subject_features for window in video_features]
        score_batches = []
        hidden_batches = []
        max_window_head_error = 0.0
        try:
            with torch.no_grad():
                for start in range(0, len(flat), BATCH_SIZE):
                    batch = np.asarray(flat[start : start + BATCH_SIZE], dtype=np.float32)
                    x = torch.as_tensor(batch, device=device)[:, None, :]
                    captured.clear()
                    model_score, _ = model(x)
                    if len(captured) != 1:
                        raise RuntimeError(f"spot_pathway hook count={len(captured)}")
                    hidden = captured[0]
                    if hidden.ndim != 3 or hidden.shape[1:] != (K, HIDDEN_DIM):
                        raise RuntimeError(f"unexpected hidden shape {tuple(hidden.shape)}")
                    rebuilt = torch.sigmoid(model.fc_spot(hidden)).squeeze(-1)
                    max_window_head_error = max(
                        max_window_head_error,
                        float(torch.max(torch.abs(rebuilt - model_score)).item()),
                    )
                    score_batches.append(model_score.cpu().numpy().astype(np.float32))
                    hidden_batches.append(hidden.cpu().numpy().astype(np.float32))
        finally:
            handle.remove()

        outputs = stitch(
            np.concatenate(score_batches, axis=0),
            np.concatenate(hidden_batches, axis=0),
            video_num,
        )
        checkpoint_hash = sha256(checkpoint)
        for local_index, ((global_index, video), output) in enumerate(zip(entries, outputs)):
            hidden = output["hidden"]
            valid_mask = output["valid_mask"]
            with torch.no_grad():
                rebuilt_score = torch.sigmoid(
                    model.fc_spot(torch.as_tensor(hidden, device=device))
                ).squeeze(-1).cpu().numpy().astype(np.float32)
            # The original batch-sensitive stitch leaves leading slots at zero
            # when a new video starts in the middle of a DataLoader batch.
            # Those positions have no corresponding hidden state/head call.
            rebuilt_score[~valid_mask] = 0.0
            head_error = float(np.max(np.abs(rebuilt_score - output["score"])))
            if head_error > 1e-6:
                raise RuntimeError(f"{subject}/{video}: stitched hidden/head error={head_error}")
            output_path = subject_dir / f"{video}_frozen_hidden_score.npz"
            np.savez(
                output_path,
                hidden=hidden,
                score=rebuilt_score,
                valid_mask=valid_mask,
                subject=np.asarray(subject),
                video_id=np.asarray(video),
            )
            total_t += len(rebuilt_score)
            records.append({
                "subject": subject,
                "video_id": video,
                "global_video_index": global_index,
                "subject_video_index": local_index,
                "T_hidden": int(hidden.shape[0]),
                "hidden_dim": int(hidden.shape[1]),
                "T_score": int(len(rebuilt_score)),
                "max_window_head_error": max_window_head_error,
                "max_stitched_head_error": head_error,
                "num_valid_positions": int(valid_mask.sum()),
                "num_unwritten_zero_positions": int((~valid_mask).sum()),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "output_path": str(output_path),
            })
            print(f"SAVED {subject}/{video}: hidden={hidden.shape}, score={rebuilt_score.shape}")
        del model
        torch.cuda.empty_cache()

    manifest = {
        "status": "COMPLETE",
        "dataset": "SAMMLV",
        "model": "ME-TST+",
        "inference_only": True,
        "model_eval": True,
        "torch_no_grad": True,
        "training_calls": {"model_train": 0, "backward": 0, "optimizer_step": 0},
        "input_cache": str(args.input_cache),
        "input_cache_sha256": sha256(args.input_cache),
        "weights_dir": str(args.weights_dir),
        "project_root": str(args.project_root),
        "batch_size": BATCH_SIZE,
        "window_length": K,
        "hidden_dim": HIDDEN_DIM,
        "num_subjects": len(subject_groups()),
        "num_videos": len(records),
        "total_T": total_t,
        "historical_compact_numeric_outputs_used": False,
        "manifest_provenance": "embedded subject/video order only",
        "records": records,
    }
    manifest_path = args.output_dir / "sammlv_full_hidden_dump_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "num_subjects", "num_videos", "total_T")}, indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
