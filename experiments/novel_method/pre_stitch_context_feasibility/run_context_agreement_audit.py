#!/usr/bin/env python3
"""Phase-0 alignment gate for the pre-stitch context-agreement audit.

Run this in the previously verified Colab ME-TST+ environment.  It performs
frozen inference for one complete SAMMLV subject, preserves every real
window-level scalar prediction, reproduces the original batch-sensitive
stitching, and compares one requested video with the historical compact score.

The scientific agreement audit is intentionally not executed unless this gate
passes.  No context is reconstructed from an already stitched curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


WINDOW_LENGTH = 30
STRIDE = 15
BATCH_SIZE = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def load_compact(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("dataset") != "SAMMLV" or "records" not in payload:
        raise ValueError("expected the complete ME-TST+ SAMMLV compact cache")
    return payload


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
    return {
        str(key).removeprefix("module."): tensor
        for key, tensor in value.items()
    }


def frozen_forward(model, flat_windows: list, device) -> np.ndarray:
    import torch

    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat_windows), BATCH_SIZE):
            batch = np.asarray(
                flat_windows[start:start + BATCH_SIZE], dtype=np.float32
            )
            x = torch.as_tensor(batch, device=device)[:, None, :]
            spotting_score, _ = model(x)
            score = spotting_score.detach().cpu().numpy().astype(np.float32)
            if score.ndim != 2 or score.shape[1] != WINDOW_LENGTH:
                raise RuntimeError(f"unexpected window score shape: {score.shape}")
            batches.append(score)
    return np.concatenate(batches, axis=0)


def stitch_and_preserve_contexts(window_scores: np.ndarray,
                                 video_num: list[int]) -> list[dict]:
    """Reproduce the original write branch and retain all theoretical overlaps."""
    outputs = []
    video_index = 0
    framecount = 0
    global_window_index = 0

    def new_video(count: int) -> dict:
        length = (count + 1) * STRIDE
        return {
            "stitched": np.zeros(length, dtype=np.float32),
            "contexts": [[] for _ in range(length)],
            "rows": [],
        }

    current = new_video(video_num[0])
    for batch_start in range(0, len(window_scores), BATCH_SIZE):
        batch_stop = min(batch_start + BATCH_SIZE, len(window_scores))
        for batch_local_i in range(batch_stop - batch_start):
            if framecount == video_num[video_index]:
                outputs.append(current)
                video_index += 1
                framecount = 0
                current = new_video(video_num[video_index])

            score = window_scores[global_window_index]
            window_id = framecount
            for local_t, value in enumerate(score):
                global_t = window_id * STRIDE + local_t
                if global_t < len(current["stitched"]):
                    current["contexts"][global_t].append(float(value))
                    current["rows"].append((window_id, local_t, global_t,
                                            float(value)))

            if batch_local_i == 0:
                start = framecount * STRIDE
                stop = (framecount + 2) * STRIDE
                current["stitched"][start:stop] = score
            else:
                start = (framecount + 1) * STRIDE
                stop = (framecount + 2) * STRIDE
                current["stitched"][start:stop] = score[STRIDE:]

            framecount += 1
            global_window_index += 1

    outputs.append(current)
    if len(outputs) != len(video_num):
        raise RuntimeError(
            f"stitch produced {len(outputs)} videos; expected {len(video_num)}"
        )
    return outputs


def alignment_metrics(reconstructed: np.ndarray,
                      existing: np.ndarray) -> dict:
    if reconstructed.shape != existing.shape:
        return {
            "T_reconstructed": len(reconstructed),
            "T_existing": len(existing),
            "shape_match": False,
            "np_allclose": False,
        }
    difference = reconstructed.astype(float) - existing.astype(float)
    correlation = (
        float(np.corrcoef(reconstructed, existing)[0, 1])
        if np.std(reconstructed) > 0 and np.std(existing) > 0 else None
    )
    return {
        "T_reconstructed": len(reconstructed),
        "T_existing": len(existing),
        "shape_match": True,
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference ** 2))),
        "Pearson_correlation": correlation,
        "np_allclose": bool(np.allclose(reconstructed, existing)),
        "np_allclose_parameters": "numpy defaults: rtol=1e-5, atol=1e-8",
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--compact-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="006")
    parser.add_argument("--video", default="006_1")
    return parser.parse_args()


def main() -> int:
    import torch

    args = arguments()
    required = [
        args.project_root / "network_sf.py",
        args.input_cache,
        args.compact_cache,
        args.weights_dir / f"subject_{args.subject}.pkl",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("run this gate in the verified Colab CUDA environment")

    compact = load_compact(args.compact_cache)
    records = compact["records"]
    features = load_features(args.input_cache)
    if len(features) != len(records):
        raise RuntimeError(
            f"feature videos={len(features)} but compact records={len(records)}"
        )
    subject_indices = [
        index for index, row in enumerate(records)
        if str(row["subject"]) == args.subject
    ]
    if not subject_indices:
        raise ValueError(f"subject not found: {args.subject}")
    subject_records = [records[index] for index in subject_indices]
    subject_features = [features[index] for index in subject_indices]
    target_local_index = next(
        (index for index, row in enumerate(subject_records)
         if str(row["video"]) == args.video),
        None,
    )
    if target_local_index is None:
        raise ValueError(f"video not found in subject: {args.video}")

    checkpoint = args.weights_dir / f"subject_{args.subject}.pkl"
    sys.path.insert(0, str(args.project_root))
    from network_sf import METST_SF

    device = torch.device("cuda")
    model = METST_SF(out_channels=5).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
    model.eval()

    video_num = [len(video_features) for video_features in subject_features]
    flat_windows = [
        window for video_features in subject_features for window in video_features
    ]
    window_scores = frozen_forward(model, flat_windows, device)
    outputs = stitch_and_preserve_contexts(window_scores, video_num)
    target = outputs[target_local_index]
    existing = np.asarray(subject_records[target_local_index]["score"],
                          dtype=np.float32)
    metrics = alignment_metrics(target["stitched"], existing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(target["rows"], dtype=np.float64)
    np.savez(
        args.output_dir / f"{args.subject}_{args.video}_pre_stitch_alignment.npz",
        window_id=rows[:, 0].astype(np.int32),
        local_t=rows[:, 1].astype(np.int16),
        global_t=rows[:, 2].astype(np.int32),
        raw_window_score=rows[:, 3].astype(np.float32),
        reconstructed_score=target["stitched"],
        existing_score=existing,
        context_count=np.asarray([len(values) for values in target["contexts"]],
                                 dtype=np.int16),
    )
    report = {
        "status": "ALIGNMENT_PASS" if metrics["np_allclose"]
        else "BLOCKED_ALIGNMENT",
        "scientific_audit_executed": False,
        "stop_reason": None if metrics["np_allclose"] else (
            "pre-stitch reconstruction is not np.allclose to historical compact score"
        ),
        "subject": args.subject,
        "video": args.video,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "input_cache_path": str(args.input_cache),
        "input_cache_sha256": sha256(args.input_cache),
        "compact_cache_path": str(args.compact_cache),
        "compact_cache_sha256": sha256(args.compact_cache),
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "window_score_shape": list(window_scores.shape),
        "alignment": metrics,
        "training_calls": {"model_train": 0, "backward": 0,
                           "optimizer_step": 0},
    }
    output = args.output_dir / "context_agreement_results.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not metrics["np_allclose"]:
        print("BLOCKED-ALIGNMENT: stop before multiplicity/agreement analysis")
        return 2
    print(
        "ALIGNMENT PASS, but full extraction/scientific audit is deliberately "
        "not bundled into this phase-0 gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
