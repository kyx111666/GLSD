#!/usr/bin/env python3
"""Export verified raw pre-stitch windows for all 29/79 SAMMLV videos.

Each subject checkpoint is loaded exactly once and each subject is traversed
exactly once in the frozen subject-level order.  The model is still invoked in
the original batches of 32; changing that batching would change the verified
stitching semantics.  This script adds no decoder or aggregation method.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

from export_raw_prestitch_sammlv import (
    BATCH_SIZE,
    NUM_CLASSES,
    STRIDE,
    VIDEO_MANIFEST,
    WINDOW_LENGTH,
    load_features,
    load_state_dict,
    manifest_groups,
    save_dump,
    sha256,
    source_commit,
    validate_output_shapes,
)
from reconstruct_original_stitch import evaluate as evaluate_reconstruction


EXPECTED_SUBJECTS = 29
EXPECTED_VIDEOS = 79


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expected_output_path(output_dir: Path, subject: str, video: str) -> Path:
    return output_dir / f"{subject}_{video}_raw_prestitch.npz"


def preflight_outputs(
    output_dir: Path,
    grouped: OrderedDict[str, list[tuple[int, str]]],
    overwrite: bool,
) -> None:
    outputs = [
        expected_output_path(output_dir, subject, video)
        for subject, entries in grouped.items()
        for _global_video_index, video in entries
    ]
    outputs.append(output_dir / "full_manifest.json")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(str(path) for path in existing[:10])
        suffix = "\n..." if len(existing) > 10 else ""
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} existing output(s):\n{preview}{suffix}"
        )


def forward_subject_once(
    model,
    subject: str,
    entries: list[tuple[int, str]],
    features: list,
    device,
) -> tuple[dict[str, dict], int]:
    """Capture every video during one frozen subject-level traversal."""
    import torch

    subject_features = [features[index] for index, _video in entries]
    video_num = [len(video_features) for video_features in subject_features]
    if not video_num or any(count <= 0 for count in video_num):
        raise RuntimeError(f"{subject}: empty subject/video window list")
    flat_windows = [window for video_features in subject_features for window in video_features]
    flat_map = [
        (subject_video_index, global_video_index, video, video_window_id)
        for subject_video_index, ((global_video_index, video), video_features) in enumerate(
            zip(entries, subject_features)
        )
        for video_window_id, _window in enumerate(video_features)
    ]
    if len(flat_windows) != len(flat_map) or len(flat_windows) != sum(video_num):
        raise RuntimeError(f"{subject}: flat window mapping changed length")

    captured_by_video: dict[str, list[dict]] = {video: [] for _index, video in entries}
    result_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    video_index = 0
    framecount = 0
    result_video = np.zeros((video_num[0] + 1) * STRIDE, dtype=np.float32)
    logits_video = np.zeros(((video_num[0] + 1) * STRIDE, NUM_CLASSES), dtype=np.float32)
    model_forward_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_index, batch_start in enumerate(range(0, len(flat_windows), BATCH_SIZE)):
            batch_stop = min(batch_start + BATCH_SIZE, len(flat_windows))
            batch = np.asarray(flat_windows[batch_start:batch_stop], dtype=np.float32)
            x = torch.as_tensor(batch)[:, None, :].to(device)
            yhat, yhat1 = model(x)
            model_forward_batches += 1

            # Verified capture point: immediately after forward and before any
            # stitching/overwrite/slicing/recognition argmax.
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
                        f"{subject}: manifest/window traversal mismatch before original write: "
                        f"mapped=({mapped_video_index},{video_window_id}) "
                        f"live=({video_index},{framecount})"
                    )

                global_start = video_window_id * STRIDE
                captured_by_video[video].append(
                    {
                        "subject": subject,
                        "video": video,
                        "global_video_index": global_video_index,
                        "subject_video_index": video_index,
                        "subject_flat_window_index": subject_flat_index,
                        "batch_index": batch_index,
                        "index_within_batch": index_within_batch,
                        "video_window_id": video_window_id,
                        "framecount_before_write": framecount,
                        "global_start": global_start,
                        "global_indices": global_start
                        + np.arange(WINDOW_LENGTH, dtype=np.int32),
                        "raw_window_score": raw_scores[index_within_batch].copy(),
                        "raw_window_logits": raw_logits[index_within_batch].copy(),
                    }
                )

                # Exact verified batch-sensitive original stitching semantics.
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
    if len(result_all) != len(entries) or len(logits_all) != len(entries):
        raise RuntimeError(
            f"{subject}: stitched videos={len(result_all)} expected={len(entries)}"
        )
    if model_forward_batches != math.ceil(len(flat_windows) / BATCH_SIZE):
        raise RuntimeError(f"{subject}: unexpected number of forward batches")

    outputs: dict[str, dict] = {}
    for subject_video_index, ((_global_index, video), n_windows) in enumerate(
        zip(entries, video_num)
    ):
        captured = captured_by_video[video]
        if len(captured) != n_windows:
            raise RuntimeError(
                f"{subject}/{video}: captured windows={len(captured)} expected={n_windows}"
            )
        outputs[video] = {
            "captured": captured,
            "original_fresh_result_video": result_all[subject_video_index],
            "original_fresh_logits_video": logits_all[subject_video_index],
            "subject_video_count": len(entries),
            "subject_total_windows": len(flat_windows),
            "target_subject_video_index": subject_video_index,
            "target_num_windows": n_windows,
        }
    return outputs, model_forward_batches


def verify_saved_video(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return evaluate_reconstruction(data)


def finite_min(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return min(finite) if finite else None


def finite_max(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return max(finite) if finite else None


def build_summary(video_reports: list[dict]) -> dict:
    def row_passed(row: dict) -> bool:
        return bool(
            row["score_reconstruction"]["array_equal"]
            and row["logits_reconstruction"]["array_equal"]
        )

    passed = [row for row in video_reports if row_passed(row)]
    failed = [row for row in video_reports if not row_passed(row)]
    score_errors = [row["score_reconstruction"]["max_abs_error"] for row in video_reports]
    logits_errors = [row["logits_reconstruction"]["max_abs_error"] for row in video_reports]
    score_pearsons = [row["score_reconstruction"]["pearson"] for row in video_reports]
    logits_pearsons = [row["logits_reconstruction"]["pearson"] for row in video_reports]
    observed_identities = {(row["subject"], row["video"]) for row in video_reports}
    expected_identities = set(VIDEO_MANIFEST)
    all_ready = (
        len(video_reports) == EXPECTED_VIDEOS
        and len(passed) == EXPECTED_VIDEOS
        and observed_identities == expected_identities
    )
    return {
        "status": (
            "SAMMLV-RAW-WINDOW-SOURCE-READY"
            if all_ready
            else "SAMMLV-RAW-WINDOW-SOURCE-NOT-READY"
        ),
        "subjects": EXPECTED_SUBJECTS,
        "videos": EXPECTED_VIDEOS,
        "videos_checked": len(video_reports),
        "videos_passed": len(passed),
        "videos_failed": len(failed),
        "failed_video_ids": [f"{row['subject']}/{row['video']}" for row in failed],
        "worst_score_max_abs_error": finite_max(score_errors),
        "worst_logits_max_abs_error": finite_max(logits_errors),
        "minimum_score_pearson": finite_min(score_pearsons),
        "minimum_logits_pearson": finite_min(logits_pearsons),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    import torch

    args = parse_args()
    grouped = manifest_groups()
    if len(grouped) != EXPECTED_SUBJECTS or len(VIDEO_MANIFEST) != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"frozen manifest changed: subjects={len(grouped)}, videos={len(VIDEO_MANIFEST)}"
        )
    for path in (args.project_root / "network_sf.py", args.input_cache, args.weights_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run in the restored frozen GPU environment")

    features = load_features(args.input_cache)
    if len(features) != EXPECTED_VIDEOS:
        raise RuntimeError(f"feature videos={len(features)} expected={EXPECTED_VIDEOS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_outputs(args.output_dir, grouped, args.overwrite)

    sys.path.insert(0, str(args.project_root))
    from network_sf import METST_SF

    device = torch.device("cuda")
    video_reports: list[dict] = []
    subject_reports: list[dict] = []
    checkpoint_hashes: dict[str, dict] = {}
    full_manifest_path = args.output_dir / "full_manifest.json"

    for subject, entries in grouped.items():
        checkpoint = args.weights_dir / f"subject_{subject}.pkl"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        checkpoint_hashes[subject] = {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        }

        # Exactly one checkpoint load and one complete subject traversal.
        model = METST_SF(out_channels=NUM_CLASSES).to(device)
        model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
        model.eval()
        subject_outputs, forward_batches = forward_subject_once(
            model, subject, entries, features, device
        )

        subject_video_reports = []
        for _global_video_index, video in entries:
            output_path = expected_output_path(args.output_dir, subject, video)
            save_dump(output_path, subject, video, subject_outputs[video])
            reconstruction = verify_saved_video(output_path)
            report = {
                **reconstruction,
                "path": str(output_path),
                "sha256": sha256(output_path),
            }
            video_reports.append(report)
            subject_video_reports.append(report)
            print(
                f"VERIFIED {subject}/{video}: {reconstruction['status']} "
                f"score_max={reconstruction['score_reconstruction']['max_abs_error']} "
                f"logits_max={reconstruction['logits_reconstruction']['max_abs_error']}"
            )

        subject_reports.append(
            {
                "subject": subject,
                "checkpoint_loads": 1,
                "subject_forward_passes": 1,
                "model_forward_batches": forward_batches,
                "videos": len(entries),
                "videos_passed": sum(
                    row["score_reconstruction"]["array_equal"]
                    and row["logits_reconstruction"]["array_equal"]
                    for row in subject_video_reports
                ),
            }
        )
        del model, subject_outputs
        torch.cuda.empty_cache()

        # Recoverable progress manifest after every completed subject.
        progress = build_summary(video_reports)
        write_manifest(
            full_manifest_path,
            {
                "schema": "sammlv_raw_prestitch_full_manifest_v1",
                **progress,
                "complete": False,
                "subjects_completed": len(subject_reports),
                "subject_reports": subject_reports,
                "video_reports": video_reports,
            },
        )

    summary = build_summary(video_reports)
    manifest = {
        "schema": "sammlv_raw_prestitch_full_manifest_v1",
        **summary,
        "complete": True,
        "dataset": "SAMMLV",
        "model": "ME-TST+",
        "train": False,
        "model_eval": True,
        "torch_no_grad": True,
        "batch_size": BATCH_SIZE,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "source_commit": source_commit(args.project_root),
        "input_cache": str(args.input_cache),
        "input_cache_sha256": sha256(args.input_cache),
        "weights_dir": str(args.weights_dir),
        "checkpoint_hashes": checkpoint_hashes,
        "subject_reports": subject_reports,
        "video_reports": video_reports,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    write_manifest(full_manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "SAMMLV-RAW-WINDOW-SOURCE-READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
