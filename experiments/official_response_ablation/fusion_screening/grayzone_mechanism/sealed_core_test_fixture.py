"""Official-response GLSD evaluation for BoostingVRME/SAMMLV.

This program never runs a backbone or training.  It consumes the recorder-made
official response cache and calls the repository's official spotting,
recognition, metric and NMS code.  Run it from the root of the official
BoostingVRME repository, for example::

    python boosting_official_glds_full.py \
        --cache /content/sammlv_official_full_responses.pkl \
        --output_dir /content/boosting_glsd_results

``--mode audit`` stops after the mandatory native exact-replay gate.  ``full``
(the default) runs the nested-LOSO GLSD evaluation only after that gate passes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences


# Locked GLSD-90 protocol.  The ordering is also the final deterministic
# selection tie-break after F1, precision and FP.
SCALES = (1.0, 1.5, 2.0)
LOCAL_RADII = (1.0, 2.0, 3.0)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)
EXPECTED_RAW = (51, 144, 108)
EXPECTED_FULL = (51, 141, 108)


@dataclass(frozen=True)
class GLSDConfig:
    """One member of the locked 3 x 3 x 10 GLSD search grid."""

    reference_scale: float
    local_radius: float
    threshold: float

    @property
    def identifier(self) -> str:
        return (
            f"reference_scale={self.reference_scale:g}|"
            f"local_radius={self.local_radius:g}|threshold={self.threshold:g}"
        )


def configuration_grid() -> list[GLSDConfig]:
    return [
        GLSDConfig(reference, radius, threshold)
        for reference in SCALES
        for radius in LOCAL_RADII
        for threshold in THRESHOLDS
    ]


def metrics(counts) -> dict:
    tp, fp, fn = (int(value) for value in counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def cache_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash an existing artifact without changing its contents."""
    return cache_sha256(path)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def official_git_commit() -> str:
    root = Path(__file__).resolve().parent
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def as_counts(value) -> tuple[int, int, int]:
    return int(value["TP"]), int(value["FP"]), int(value["FN"])


def prediction_lists(predictions) -> list:
    return [np.asarray(video_predictions).tolist() for video_predictions in predictions]


def load_cache(path: Path) -> dict:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    if cache.get("schema_version") != 1 or cache.get("dataset") != "SAMMLV":
        raise RuntimeError("cache is not a schema-v1 SAMMLV official response cache")
    config = cache.get("config", {})
    required_config = {
        "emotion_type", "frame_skip", "k", "k_p", "synergy_strategy",
        "interval_strategy", "penalty_strategy",
    }
    missing = sorted(required_config.difference(config))
    if missing:
        raise RuntimeError("cache config missing: " + ", ".join(missing))
    if config["synergy_strategy"] != 1 or config["interval_strategy"] != 1:
        raise RuntimeError("this evaluator requires official SAMMLV synergy=1 and interval_strategy=1")
    direct_recognition = cache.get("direct_recognition", {})
    if set(direct_recognition) != {"pred_list", "gt_tp_list"}:
        raise RuntimeError("cache direct-recognition audit values are missing")
    direct_counts = cache.get("direct_inference_counts", {})
    if set(direct_counts) != {"raw", "full"}:
        raise RuntimeError("cache direct-inference count audit values are missing")
    videos = cache.get("videos", [])
    assert len(videos) == 79
    required_video = {
        "subject_id", "subject_index", "video_index_within_subject", "video_name",
        "spot_response", "recognition_sequence", "gt_intervals", "emotion_labels",
        "direct_spot_predictions",
    }
    for global_index, video in enumerate(videos):
        if video.get("video_index_global") != global_index:
            raise RuntimeError(f"cache global-video ordering failure at {global_index}")
        missing = sorted(required_video.difference(video))
        if missing:
            raise RuntimeError(f"video {global_index} missing: {', '.join(missing)}")
    return cache


def group_videos(videos: list[dict]) -> tuple[list[str], list[list[dict]]]:
    """Preserve the recorder's official subject/video ordering exactly."""
    subjects: list[str] = []
    grouped: list[list[dict]] = []
    current_index = None
    for video in videos:
        subject_index = int(video["subject_index"])
        if subject_index != current_index:
            if subject_index != len(grouped):
                raise RuntimeError(f"non-contiguous subject order at video {video['video_index_global']}")
            current_index = subject_index
            subjects.append(str(video["subject_id"]))
            grouped.append([])
        if int(video["video_index_within_subject"]) != len(grouped[-1]):
            raise RuntimeError(f"non-contiguous video order for subject {subject_index}")
        grouped[-1].append(video)
    assert len(subjects) == 29
    return subjects, grouped


def official_functions():
    """Import the exact repository implementations only after cache checks."""
    from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
    import training_utils

    return MeanAveragePrecision2d, training_utils


def samples_and_emotions(grouped: list[list[dict]]) -> tuple[list, list]:
    samples = [[video["gt_intervals"] for video in subject] for subject in grouped]
    emotions = [[video["emotion_labels"] for video in subject] for subject in grouped]
    return samples, emotions


def full_counts_from_official_synergy(raw_counts, pred_list, gt_tp_list, config) -> tuple[int, int, int]:
    """The official synergy=1 neutral handling, without any new rule."""
    neutral = int(config["emotion_type"]) - 1
    tp_neutral = sum(prediction == neutral and target != -1 for prediction, target in zip(pred_list, gt_tp_list))
    fp_neutral = sum(prediction == neutral and target == -1 for prediction, target in zip(pred_list, gt_tp_list))
    tp, fp, fn = raw_counts
    return int(tp - tp_neutral), int(fp - fp_neutral), int(fn + tp_neutral)


def native_exact_replay(cache: dict, grouped: list[list[dict]]):
    """Replay the complete native official response path before GLSD is allowed."""
    MeanAveragePrecision2d, official = official_functions()
    config = cache["config"]
    final_samples, final_emotions = samples_and_emotions(grouped)
    metric_final = MeanAveragePrecision2d(num_classes=1)
    total_gt = 0
    pred_list, gt_tp_list = [], []
    pred_windows, pred_single = [], []
    decoded_predictions = []
    for subject_index, videos in enumerate(grouped):
        result_all = [np.asarray(video["spot_response"]) for video in videos]
        recognition_sequence = [np.asarray(video["recognition_sequence"]) for video in videos]
        predictions, _, total_gt, metric_video, metric_final = official.spotting(
            final_samples, subject_index, result_all, total_gt, metric_final,
            config["k_p"], config["interval_strategy"], "SAMMLV",
        )
        decoded_predictions.extend(prediction_lists(predictions))
        pred_list, _, gt_tp_list, pred_windows, pred_single = official.recognition(
            recognition_sequence, predictions, metric_video, final_emotions, subject_index,
            pred_list, gt_tp_list, final_samples, pred_windows, pred_single,
            config["frame_skip"], config["penalty_strategy"],
        )
    raw_counts = tuple(int(value) for value in official.sequence_evaluation(total_gt, metric_final))
    full_counts = full_counts_from_official_synergy(raw_counts, pred_list, gt_tp_list, config)
    direct_predictions = prediction_lists([video["direct_spot_predictions"] for video in cache["videos"]])
    return {
        "raw_counts": raw_counts,
        "full_counts": full_counts,
        "decoded_predictions_match": direct_predictions == decoded_predictions,
        "recognition_predictions_match": cache["direct_recognition"]["pred_list"] == pred_list,
        "matched_gt_sequence_match": cache["direct_recognition"]["gt_tp_list"] == gt_tp_list,
    }


class GLSDFeatures:
    """Locked label-free GLSD G/L evidence for one official spotting response."""

    def __init__(self, response, k: int):
        self.response = np.asarray(response, dtype=float)
        if self.response.ndim != 1 or len(self.response) < 3 or not np.isfinite(self.response).all():
            raise ValueError("official spotting response must be finite one-dimensional data")
        self.k = int(k)
        if self.k < 1:
            raise ValueError("k_p must be positive")
        self.scale_data = {}
        self.local_cache = {}
        physical = {}
        for scale in SCALES:
            width = min(len(self.response), max(1, round(scale * self.k)))
            if width not in physical:
                smooth = np.convolve(self.response, np.ones(width, dtype=float) / width, mode="same")
                peaks = find_peaks(smooth, distance=self.k)[0] if np.ptp(self.response) > 0 else np.empty(0, dtype=int)
                spread = max(float(np.ptp(smooth)), 1e-12)
                global_prominence = peak_prominences(smooth, peaks)[0] / spread
                physical[width] = (peaks, smooth, spread, global_prominence)
            self.scale_data[scale] = physical[width]
        # A duplicate integer smoothing width is one physical scale/vote.
        self.effective_scales = physical

    def evidence(self, reference: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
        peaks, _, _, global_values = self.scale_data[reference]
        window = max(1, round(radius * self.k))
        tolerance = max(1, round(0.5 * self.k))
        for width, (other_peaks, smooth, spread, _) in self.effective_scales.items():
            key = width, window
            if key not in self.local_cache:
                values = []
                for peak in other_peaks:
                    left = float(np.min(smooth[max(0, peak - window):peak + 1]))
                    right = float(np.min(smooth[peak:min(len(smooth), peak + window + 1)]))
                    values.append(max(0.0, float(smooth[peak]) - max(left, right)) / spread)
                self.local_cache[key] = np.asarray(values, dtype=float)
        local_values = []
        for peak in peaks:
            aligned = []
            for width, (other_peaks, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                candidates = np.flatnonzero(np.abs(other_peaks - peak) <= tolerance)
                if not len(candidates):
                    aligned.append(0.0)  # locked missing-scale policy
                    continue
                chosen = min(candidates, key=lambda index: (abs(int(other_peaks[index]) - int(peak)), -values[index]))
                aligned.append(float(values[chosen]))
            local_values.append(float(np.median(aligned)))  # locked median aggregation
        evidence = np.column_stack((global_values, np.asarray(local_values, dtype=float)))
        return np.asarray(peaks, dtype=int), evidence

    def selected_peaks(self, config: GLSDConfig) -> np.ndarray:
        peaks, evidence = self.evidence(config.reference_scale, config.local_radius)
        if not len(peaks):
            return peaks
        global_values, local_values = evidence.T
        score = (global_values + local_values) / 2.0  # locked S(c)=(G(c)+L(c))/2
        return peaks[score >= config.threshold]


@contextmanager
def official_glsd_candidate_source(official, selected_by_video: list[np.ndarray]):
    """Feed GLSD-retained peaks to the *unmodified* official spotting() decoder.

    ``spotting()`` owns the boundary search, ``peak_new`` construction, candidate
    representation, metric registration and its direct call to official ``nms()``.
    Only its candidate source is substituted: GLSD has already constructed and
    thresholded the reference-scale candidates.  The original finder is restored
    even if the official function raises.
    """
    original_find_peaks = official.find_peaks
    position = 0

    def glsd_find_peaks(signal, *args, **kwargs):
        nonlocal position
        if position >= len(selected_by_video):
            raise RuntimeError("official spotting requested more candidate lists than cached videos")
        peaks = np.asarray(selected_by_video[position], dtype=int)
        position += 1
        if np.any(peaks < 0) or np.any(peaks >= len(signal)):
            raise RuntimeError("GLSD candidate lies outside its official response")
        return peaks, {}

    official.find_peaks = glsd_find_peaks
    try:
        yield
        if position != len(selected_by_video):
            raise RuntimeError("official spotting did not consume every GLSD candidate list")
    finally:
        official.find_peaks = original_find_peaks


def decode_glsd_subject(
    grouped, final_samples, subject_index: int, glsd_config: GLSDConfig,
    official, MeanAveragePrecision2d, cache_config, with_recognition: bool,
):
    """Decode one subject through official spotting; optionally official recognition."""
    videos = grouped[subject_index]
    responses = [np.asarray(video["spot_response"]) for video in videos]
    candidates = [GLSDFeatures(response, int(cache_config["k_p"])).selected_peaks(glsd_config) for response in responses]
    metric_final = MeanAveragePrecision2d(num_classes=1)
    total_gt = 0
    with official_glsd_candidate_source(official, candidates):
        predictions, _, total_gt, metric_video, metric_final = official.spotting(
            final_samples, subject_index, responses, total_gt, metric_final,
            cache_config["k_p"], cache_config["interval_strategy"], "SAMMLV",
        )
    raw_counts = tuple(int(value) for value in official.sequence_evaluation(total_gt, metric_final))
    prediction_matches = sorted(
        metric_video.value(iou_thresholds=0.5)[0.5][0]["pred_match_gt"].items()
    )
    if not with_recognition:
        return raw_counts, predictions, None, None, prediction_matches, metric_video
    _, final_emotions = samples_and_emotions(grouped)
    sequences = [np.asarray(video["recognition_sequence"]) for video in videos]
    pred_list, _, gt_tp_list, _, _ = official.recognition(
        sequences, predictions, metric_video, final_emotions, subject_index,
        [], [], final_samples, [], [], cache_config["frame_skip"], cache_config["penalty_strategy"],
    )
    return raw_counts, predictions, pred_list, gt_tp_list, prediction_matches, metric_video


def selection_order(inner_counts: np.ndarray) -> np.ndarray:
    """Locked selection order: F1, precision, fewer FP, fixed GLSD-grid order."""
    tp, fp, fn = inner_counts.astype(float).T
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    grid_order = np.arange(len(inner_counts))
    return np.lexsort((grid_order, fp, -precision, -f1))


def choose_configuration(inner_counts: np.ndarray) -> int:
    """Return the winner under the unchanged locked selection order."""
    return int(selection_order(inner_counts)[0])


def plain_scalar(value):
    """Convert NumPy scalar labels to portable CSV/JSON/Pickle values."""
    return value.item() if isinstance(value, np.generic) else value


def auditable_prediction_evidence(
    videos, predictions, prediction_matches, metric_video, pred_list, gt_tp_list, official, cache_config,
):
    """Persist evaluator/recognition outputs without participating in evaluation.

    The supplied match map is read directly from the official metric object and
    the recognition labels are returned by the official recognition() call.
    These rows are evidence only: they never feed candidate selection, spotting,
    recognition, synergy, or metric computation.
    """
    matches_by_video = dict(prediction_matches)
    if set(matches_by_video).difference(range(len(videos))):
        raise RuntimeError("official metric returned an invalid video index")
    neutral_label = int(cache_config["emotion_type"]) - 1
    label_index = 0
    raw = np.zeros(3, dtype=np.int64)
    full = np.zeros(3, dtype=np.int64)
    csv_rows = []
    pickle_videos = []
    for video_index, (video, video_predictions) in enumerate(zip(videos, predictions)):
        video_predictions = np.asarray(video_predictions).tolist()
        matched_indices = list(matches_by_video.get(video_index, []))
        if len(matched_indices) != len(video_predictions):
            raise RuntimeError("official metric/prediction cardinality mismatch")
        evaluator_rows = metric_video.match_table[0]
        evaluator_rows = evaluator_rows[evaluator_rows["img_id"] == video_index]
        if len(evaluator_rows) != len(video_predictions):
            raise RuntimeError("official evaluator evidence cardinality mismatch")
        video_rows = []
        matched_gt = set()
        for prediction_index, (prediction, gt_index) in enumerate(zip(video_predictions, matched_indices)):
            if label_index >= len(pred_list) or label_index >= len(gt_tp_list):
                raise RuntimeError("official recognition cardinality mismatch")
            recognition_prediction = plain_scalar(pred_list[label_index])
            gt_recognition_label = plain_scalar(gt_tp_list[label_index])
            label_index += 1
            # The official metric may serialize unmatched indexes as either -1
            # or a one-element [-1] container; normalize only for persistence.
            gt_index = int(np.asarray(gt_index).reshape(-1)[0])
            if gt_index >= 0:
                matched_gt.add(gt_index)
            interval = [int(prediction[0]), int(prediction[2])]
            evaluator_iou = np.asarray(evaluator_rows.iloc[prediction_index]["iou"], dtype=float)
            best_iou = float(evaluator_iou.max()) if evaluator_iou.size else 0.0
            raw_status = "TP" if gt_index >= 0 else "FP"
            is_neutral = recognition_prediction == neutral_label
            if raw_status == "TP":
                raw += (1, 0, 0)
                if is_neutral:
                    synergy_action = "neutral_tp_to_fn"
                    final_contribution = (0, 0, 1)
                else:
                    synergy_action = "preserve_tp"
                    final_contribution = (1, 0, 0)
            else:
                raw += (0, 1, 0)
                if is_neutral:
                    synergy_action = "neutral_fp_removed"
                    final_contribution = (0, 0, 0)
                else:
                    synergy_action = "preserve_fp"
                    final_contribution = (0, 1, 0)
            full += final_contribution
            row = {
                "record_type": "prediction",
                "subject": str(video["subject_id"]),
                "subject_index": int(video["subject_index"]),
                "video": str(video["video_name"]),
                "video_index_global": int(video["video_index_global"]),
                "prediction_index": prediction_index,
                "onset": interval[0],
                "offset": interval[1],
                "peak": int(prediction[6]),
                "config_id": None,
                "matched_gt_index": gt_index,
                "best_iou": best_iou,
                "raw_status": raw_status,
                "recognition_prediction": recognition_prediction,
                "matched_gt_recognition_label": gt_recognition_label,
                "classified_neutral": bool(is_neutral),
                "synergy_action": synergy_action,
                "raw_TP_contribution": 1 if raw_status == "TP" else 0,
                "raw_FP_contribution": 1 if raw_status == "FP" else 0,
                "raw_FN_contribution": 0,
                "final_TP_contribution": final_contribution[0],
                "final_FP_contribution": final_contribution[1],
                "final_FN_contribution": final_contribution[2],
            }
            video_rows.append(row)
            csv_rows.append(row)
        for gt_index, gt in enumerate(video["gt_intervals"]):
            if gt_index in matched_gt:
                continue
            raw += (0, 0, 1)
            full += (0, 0, 1)
            row = {
                "record_type": "unmatched_gt",
                "subject": str(video["subject_id"]),
                "subject_index": int(video["subject_index"]),
                "video": str(video["video_name"]),
                "video_index_global": int(video["video_index_global"]),
                "prediction_index": None,
                "onset": int(gt[0]),
                "offset": int(gt[2]),
                "peak": int(gt[1]),
                "config_id": None,
                "matched_gt_index": gt_index,
                "best_iou": 0.0,
                "raw_status": "FN",
                "recognition_prediction": None,
                "matched_gt_recognition_label": plain_scalar(official.convertLabel(video["emotion_labels"][gt_index])),
                "classified_neutral": False,
                "synergy_action": "preserve_fn",
                "raw_TP_contribution": 0,
                "raw_FP_contribution": 0,
                "raw_FN_contribution": 1,
                "final_TP_contribution": 0,
                "final_FP_contribution": 0,
                "final_FN_contribution": 1,
            }
            video_rows.append(row)
            csv_rows.append(row)
        pickle_videos.append({
            "subject_id": str(video["subject_id"]),
            "subject_index": int(video["subject_index"]),
            "video_name": str(video["video_name"]),
            "video_index_global": int(video["video_index_global"]),
            "evidence": video_rows,
        })
    if label_index != len(pred_list) or label_index != len(gt_tp_list):
        raise RuntimeError("unconsumed official recognition records")
    return tuple(int(value) for value in raw), tuple(int(value) for value in full), csv_rows, pickle_videos


def make_run_manifest(cache_path: Path, output_dir: Path, video_count: int, subject_count: int, started_at: str) -> dict:
    """Create machine-generated provenance; hashes are never entered manually."""
    import scipy
    import sklearn

    script_path = Path(__file__).resolve()
    log_path = output_dir / "run_stdout_stderr.log"
    return {
        "cache_absolute_path": str(cache_path.resolve()),
        "cache_sha256": file_sha256(cache_path),
        "script_absolute_path": str(script_path),
        "script_sha256": file_sha256(script_path),
        "official_boostingvrme_git_commit": official_git_commit(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "command_line": list(sys.argv),
        "dataset": "SAMMLV",
        "video_count": video_count,
        "subject_count": subject_count,
        "glsd_grid_size": len(configuration_grid()),
        "glsd_grid_constants": {
            "reference_scales": list(SCALES),
            "local_radii": list(LOCAL_RADII),
            "thresholds": list(THRESHOLDS),
        },
        "native_expected_counts": {"raw": list(EXPECTED_RAW), "full": list(EXPECTED_FULL)},
        "execution_started_at_utc": started_at,
        "execution_finished_at_utc": utc_timestamp(),
        "run_stdout_stderr_path": str(log_path.resolve()),
        "run_stdout_stderr_sha256": None,
        "run_stdout_stderr_status": "PENDING_FINALIZE_LOG",
    }


def finalize_log(output_dir: Path, log_path: Path) -> None:
    """Bind the externally tee'd complete log to an already completed run."""
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("run_manifest.json is required before log finalization")
    if not log_path.is_file():
        raise RuntimeError(f"log does not exist: {log_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = file_sha256(log_path)
    digest_path = output_dir / "run_stdout_stderr.sha256"
    digest_path.write_text(f"{digest}  {log_path.name}\n", encoding="utf-8")
    manifest.update({
        "run_stdout_stderr_path": str(log_path.resolve()),
        "run_stdout_stderr_sha256": digest,
        "run_stdout_stderr_status": "FINALIZED",
        "log_finalized_at_utc": utc_timestamp(),
    })
    write_json(manifest_path, manifest)
    print(f"run_stdout_stderr.sha256 = {digest}")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path: Path, audit: dict, selections: list[dict] | None = None) -> None:
    lines = [
        "# BoostingVRME SAMMLV GLSD Full-Pipeline Audit",
        "",
        "## Cache and native gate",
        "",
        f"- Cache path: `{audit['cache_path']}`",
        f"- Cache SHA-256: `{audit['cache_sha256']}`",
        f"- Videos: {audit['video_count']} / 79.",
        f"- Subjects: {audit['subject_count']} / 29.",
        f"- Native raw: `{audit['native']['raw_counts']}`; F1 = {audit['native']['raw']['F1']:.10f}.",
        f"- Native full: `{audit['native']['full_counts']}`; F1 = {audit['native']['full']['F1']:.10f}.",
        f"- Decoded spotting predictions exact match: `{audit['native']['decoded_predictions_match']}`.",
        f"- Recognition prediction sequence exact match: `{audit['native']['recognition_predictions_match']}`.",
        f"- Matched-GT sequence exact match: `{audit['native']['matched_gt_sequence_match']}`.",
        f"- Native gate: `{audit['native']['status']}`.",
        "",
        "## Locked implementation",
        "",
        "- GLSD grid: 90 = 3 reference scales x 3 local radii x 10 thresholds.",
        "- GLSD score: `S(c) = (G(c) + L(c)) / 2`; missing-scale support is zero and multi-scale aggregation is median.",
        "- Official BoostingVRME interval geometry is reused by direct calls to `training_utils.spotting()` with `interval_strategy=1`; its native `nms()` remains in that call path.",
        "- Official recognition aggregation is reused by direct calls to `training_utils.recognition()` over each newly decoded GLSD interval.",
        "- Official neutral/result synergy is applied with the cached official `emotion_type` convention.",
        "- Nested selection pools each of the 28 non-outer subjects as an inner held-out validation subject; outer subject is asserted absent.",
    ]
    if selections is not None:
        lines.extend(["", "## Per-outer selected configuration", ""])
        for row in selections:
            lines.append(
                f"- Subject `{row['outer_subject']}`: `{row['config']}`; "
                f"inner TP/FP/FN = {row['inner_TP']}/{row['inner_FP']}/{row['inner_FN']}; "
                f"inner F1 = {row['inner_F1']:.10f}."
            )
        glsd = audit["glsd"]
        lines.extend([
            "", "## GLSD official-response result", "",
            f"- Raw: `{tuple(glsd['raw_counts'])}`; F1 = {glsd['raw']['F1']:.10f}.",
            f"- Full: `{tuple(glsd['full_counts'])}`; F1 = {glsd['full']['F1']:.10f}.",
            f"- Delta raw: {glsd['delta_raw']:.10f}.",
            f"- Delta full: {glsd['delta_full']:.10f}.",
            "", "## Verdict", "", "`BOOSTING_SAMMLV_GLSD_FULL_VALIDATED = PASS`",
        ])
    else:
        lines.extend([
            "", "## Verdict", "",
            f"`NATIVE_EXACT_REPLAY = {audit['native']['status']}`"
            + (" (audit mode; GLSD was not executed)." if audit["native"]["status"] == "PASS" else ""),
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full(
    cache: dict, cache_path: Path, output_dir: Path, native: dict, started_at: str,
) -> None:
    MeanAveragePrecision2d, official = official_functions()
    subjects, grouped = group_videos(cache["videos"])
    final_samples, _ = samples_and_emotions(grouped)
    grid = configuration_grid()
    assert len(grid) == 90

    # One raw official decode per (configuration, subject).  This table contains
    # no model fitting and no outer labels are read during a fold's selection.
    raw_table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
    for subject_index in range(len(subjects)):
        for config_index, glsd_config in enumerate(grid):
            raw_counts, _, _, _, _, _ = decode_glsd_subject(
                grouped, final_samples, subject_index, glsd_config,
                official, MeanAveragePrecision2d, cache["config"], with_recognition=False,
            )
            raw_table[config_index, subject_index] = raw_counts

    choices: list[int] = []
    selection_rows: list[dict] = []
    search_rows: list[dict] = []
    inner_fold_rows: list[dict] = []
    selection_traces: list[dict] = []
    for outer_index, outer_subject in enumerate(subjects):
        inner_indexes = [index for index in range(len(subjects)) if index != outer_index]
        inner_selection_subjects = [subjects[index] for index in inner_indexes]
        assert outer_subject not in inner_selection_subjects
        # These 28 validation subjects are the inner-LOSO folds.  Pooling their
        # validation counts is the locked pooled-inner selection objective.
        pooled_inner = raw_table[:, inner_indexes, :].sum(axis=1)
        order = selection_order(pooled_inner)
        selected = int(order[0])
        ranks = np.empty(len(grid), dtype=int)
        ranks[order] = np.arange(1, len(grid) + 1)
        choices.append(selected)
        selected_metrics = metrics(pooled_inner[selected])
        selection_rows.append({
            "outer_subject": outer_subject,
            "config_id": selected,
            "config": grid[selected].identifier,
            **asdict(grid[selected]),
            "inner_subject_count": len(inner_selection_subjects),
            "inner_subjects": inner_selection_subjects,
            "inner_TP": selected_metrics["TP"],
            "inner_FP": selected_metrics["FP"],
            "inner_FN": selected_metrics["FN"],
            "inner_precision": selected_metrics["precision"],
            "inner_recall": selected_metrics["recall"],
            "inner_F1": selected_metrics["F1"],
        })
        trace_candidates = []
        for config_index, glsd_config in enumerate(grid):
            current_metrics = metrics(pooled_inner[config_index])
            selected_flag = config_index == selected
            search_rows.append({
                "outer_subject": outer_subject,
                "config_id": config_index,
                **asdict(glsd_config),
                "pooled_inner_TP": current_metrics["TP"],
                "pooled_inner_FP": current_metrics["FP"],
                "pooled_inner_FN": current_metrics["FN"],
                "precision": current_metrics["precision"],
                "recall": current_metrics["recall"],
                "F1": current_metrics["F1"],
                "grid_order": config_index,
                "final_rank": int(ranks[config_index]),
                "selected": selected_flag,
            })
            trace_candidates.append({
                "config_id": config_index,
                **asdict(glsd_config),
                "pooled_inner_counts": list(map(int, pooled_inner[config_index])),
                "precision": current_metrics["precision"],
                "recall": current_metrics["recall"],
                "F1": current_metrics["F1"],
                "selection_key": {
                    "higher_F1": current_metrics["F1"],
                    "higher_precision": current_metrics["precision"],
                    "fewer_FP": current_metrics["FP"],
                    "fixed_grid_order": config_index,
                },
                "final_rank": int(ranks[config_index]),
                "selected": selected_flag,
            })
            for inner_index in inner_indexes:
                inner_metrics = metrics(raw_table[config_index, inner_index])
                inner_fold_rows.append({
                    "outer_subject": outer_subject,
                    "inner_validation_subject": subjects[inner_index],
                    "config_id": config_index,
                    **asdict(glsd_config),
                    "TP": inner_metrics["TP"],
                    "FP": inner_metrics["FP"],
                    "FN": inner_metrics["FN"],
                })
        selection_traces.append({
            "outer_subject": outer_subject,
            "inner_subjects": inner_selection_subjects,
            "tie_break_order": [
                "higher F1", "higher precision", "fewer FP", "fixed grid order",
            ],
            "candidates": trace_candidates,
            "winner_config_id": selected,
            "winner_rank": int(ranks[selected]),
        })

    glsd_raw = np.zeros(3, dtype=np.int64)
    glsd_full = np.zeros(3, dtype=np.int64)
    final_predictions = []
    auditable_predictions = []
    per_video_rows = []
    per_subject_rows = []
    for subject_index, videos in enumerate(grouped):
        selected = choices[subject_index]
        raw_counts, predictions, pred_list, gt_tp_list, prediction_matches, metric_video = decode_glsd_subject(
            grouped, final_samples, subject_index, grid[selected], official,
            MeanAveragePrecision2d, cache["config"], with_recognition=True,
        )
        # The selected config's separately evaluated outer raw count must replay exactly.
        assert tuple(raw_counts) == tuple(raw_table[selected, subject_index])
        full_counts = full_counts_from_official_synergy(raw_counts, pred_list, gt_tp_list, cache["config"])
        evidence_raw, evidence_full, video_rows, video_evidence = auditable_prediction_evidence(
            videos, predictions, prediction_matches, metric_video, pred_list, gt_tp_list, official, cache["config"],
        )
        assert evidence_raw == tuple(raw_counts)
        assert evidence_full == tuple(full_counts)
        glsd_raw += np.asarray(raw_counts, dtype=np.int64)
        glsd_full += np.asarray(full_counts, dtype=np.int64)
        raw_metrics = metrics(raw_counts)
        full_metrics = metrics(full_counts)
        per_subject_rows.append({
            "outer_subject": subjects[subject_index],
            "config_id": selected,
            "raw_TP": raw_metrics["TP"],
            "raw_FP": raw_metrics["FP"],
            "raw_FN": raw_metrics["FN"],
            "raw_F1": raw_metrics["F1"],
            "full_TP": full_metrics["TP"],
            "full_FP": full_metrics["FP"],
            "full_FN": full_metrics["FN"],
            "full_F1": full_metrics["F1"],
        })
        for row in video_rows:
            row["config_id"] = selected
            row["selected_config"] = grid[selected].identifier
        for evidence in video_evidence:
            evidence["selected_config_id"] = selected
            evidence["selected_config"] = asdict(grid[selected])
            for row in evidence["evidence"]:
                row["config_id"] = selected
        per_video_rows.extend(video_rows)
        auditable_predictions.extend(video_evidence)
        for video, predictions_for_video in zip(videos, predictions):
            final_predictions.append({
                "subject_id": video["subject_id"],
                "subject_index": int(video["subject_index"]),
                "video_name": video["video_name"],
                "video_index_global": int(video["video_index_global"]),
                "selected_config_id": selected,
                "selected_config": asdict(grid[selected]),
                "predictions": np.asarray(predictions_for_video).tolist(),
            })

    raw_counts = tuple(int(value) for value in glsd_raw)
    full_counts = tuple(int(value) for value in glsd_full)
    assert tuple(sum(row[f"raw_{name}_contribution"] for row in per_video_rows) for name in ("TP", "FP", "FN")) == raw_counts
    assert tuple(sum(row[f"final_{name}_contribution"] for row in per_video_rows) for name in ("TP", "FP", "FN")) == full_counts
    assert tuple(sum(row[f"raw_{name}"] for row in per_subject_rows) for name in ("TP", "FP", "FN")) == raw_counts
    assert tuple(sum(row[f"full_{name}"] for row in per_subject_rows) for name in ("TP", "FP", "FN")) == full_counts
    audit = {
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256(cache_path),
        "video_count": len(cache["videos"]),
        "subject_count": len(subjects),
        "native": {**native, "raw": metrics(native["raw_counts"]), "full": metrics(native["full_counts"]), "status": "PASS"},
        "glsd": {
            "raw_counts": raw_counts,
            "full_counts": full_counts,
            "raw": metrics(raw_counts),
            "full": metrics(full_counts),
            "delta_raw": metrics(raw_counts)["F1"] - metrics(native["raw_counts"])["F1"],
            "delta_full": metrics(full_counts)["F1"] - metrics(native["full_counts"])["F1"],
        },
    }
    json_rows = selection_rows
    csv_rows = [{**row, "inner_subjects": ";".join(row["inner_subjects"])} for row in selection_rows]
    write_json(output_dir / "outer_selected_configs.json", json_rows)
    write_csv(output_dir / "outer_selected_configs.csv", csv_rows)
    write_csv(output_dir / "glsd_search_all_90x29.csv", search_rows)
    write_csv(output_dir / "glsd_search_inner_fold_counts.csv", inner_fold_rows)
    write_json(output_dir / "outer_selection_trace.json", selection_traces)
    with (output_dir / "glsd_final_predictions.pkl").open("wb") as handle:
        pickle.dump(final_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "glsd_final_predictions_auditable.pkl").open("wb") as handle:
        pickle.dump(auditable_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_csv(output_dir / "per_video_evaluation.csv", per_video_rows)
    write_csv(output_dir / "per_subject_counts.csv", per_subject_rows)
    write_json(
        output_dir / "run_manifest.json",
        make_run_manifest(cache_path, output_dir, len(cache["videos"]), len(subjects), started_at),
    )
    write_audit(output_dir / "BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md", audit, selection_rows)

    print("==============================")
    print("GLSD official-response")
    print("==============================")
    print("Raw  : TP/FP/FN = %d/%d/%d, F1 = %.4f" % (*raw_counts, audit["glsd"]["raw"]["F1"]))
    print("Full : TP/FP/FN = %d/%d/%d, F1 = %.4f" % (*full_counts, audit["glsd"]["full"]["F1"]))
    print("Delta raw  = %.10f" % audit["glsd"]["delta_raw"])
    print("Delta full = %.10f" % audit["glsd"]["delta_full"])
    print("BOOSTING_SAMMLV_GLSD_FULL_VALIDATED = PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="recorder-produced SAMMLV official response cache")
    parser.add_argument("--output_dir", required=True, type=Path, help="directory for audit and evaluation outputs")
    parser.add_argument("--mode", choices=("audit", "full"), default="full")
    parser.add_argument(
        "--finalize_log", type=Path,
        help="hash an externally tee'd completed run log and update run_manifest.json; no evaluation is run",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_log is not None:
        if args.cache is not None:
            parser.error("--finalize_log does not accept --cache")
        finalize_log(args.output_dir, args.finalize_log)
        return
    if args.cache is None:
        parser.error("--cache is required unless --finalize_log is used")
    if not args.cache.is_file():
        parser.error(f"cache does not exist: {args.cache}")
    started_at = utc_timestamp()

    cache = load_cache(args.cache)
    subjects, grouped = group_videos(cache["videos"])
    assert len(cache["videos"]) == 79
    assert len(subjects) == 29
    native = native_exact_replay(cache, grouped)
    expected_direct_raw = as_counts(cache["direct_inference_counts"]["raw"])
    expected_direct_full = as_counts(cache["direct_inference_counts"]["full"])
    native_passed = (
        native["raw_counts"] == EXPECTED_RAW == expected_direct_raw
        and native["full_counts"] == EXPECTED_FULL == expected_direct_full
        and native["decoded_predictions_match"]
        and native["recognition_predictions_match"]
        and native["matched_gt_sequence_match"]
    )
    print("==============================")
    print("Native exact replay")
    print("==============================")
    print("Raw  : TP/FP/FN = %d/%d/%d, F1 = %.4f" % (*native["raw_counts"], metrics(native["raw_counts"])["F1"]))
    print("Full : TP/FP/FN = %d/%d/%d, F1 = %.4f" % (*native["full_counts"], metrics(native["full_counts"])["F1"]))
    print("decoded spotting predictions exact match =", native["decoded_predictions_match"])
    print("recognition prediction sequence exact match =", native["recognition_predictions_match"])
    print("matched-GT sequence exact match =", native["matched_gt_sequence_match"])
    audit = {
        "cache_path": str(args.cache),
        "cache_sha256": cache_sha256(args.cache),
        "video_count": len(cache["videos"]),
        "subject_count": len(subjects),
        "native": {
            **native,
            "raw": metrics(native["raw_counts"]),
            "full": metrics(native["full_counts"]),
            "status": "PASS" if native_passed else "FAIL",
        },
    }
    if not native_passed:
        write_audit(args.output_dir / "BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md", audit)
        print("NATIVE_EXACT_REPLAY = FAIL", file=sys.stderr)
        raise SystemExit(1)
    print("NATIVE_EXACT_REPLAY = PASS")
    if args.mode == "audit":
        write_audit(args.output_dir / "BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md", audit)
        return
    run_full(cache, args.cache, args.output_dir, native, started_at)


if __name__ == "__main__":
    main()
