"""Exact offline replay for a recorder-produced SAMMLV official response cache.

This script intentionally imports no network or checkpoint code.  It decodes only
the stored result_all/result1_all values with the original official functions.
"""

import os
import pickle
import subprocess
import sys

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(ROOT, "cache", "sammlv_official_full_responses.pkl")
AUDIT_PATH = os.path.join(ROOT, "BOOSTING_SAMMLV_FULL_REPLAY_AUDIT.md")
EXPECTED_RAW = (51, 144, 108)
EXPECTED_FULL = (51, 141, 108)


def _git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _counts_dict(counts):
    return {"TP": counts[0], "FP": counts[1], "FN": counts[2]}


def _as_counts(value):
    return (int(value["TP"]), int(value["FP"]), int(value["FN"]))


def _f1(counts):
    tp, fp, fn = counts
    return 2 * tp / (2 * tp + fp + fn)


def _predictions_as_lists(predictions):
    return [np.asarray(video_predictions).tolist() for video_predictions in predictions]


def _load_and_validate_cache():
    with open(CACHE_PATH, "rb") as file:
        cache = pickle.load(file)

    if cache.get("schema_version") != 1 or cache.get("dataset") != "SAMMLV":
        raise RuntimeError("cache is not a schema-v1 SAMMLV official response cache")
    if cache.get("official_git_commit") != _git_commit():
        raise RuntimeError(
            "official git commit mismatch: cache=%s, checkout=%s"
            % (cache.get("official_git_commit"), _git_commit())
        )

    config = cache.get("config", {})
    required_config = {"emotion_type", "frame_skip", "k", "k_p", "synergy_strategy", "interval_strategy", "penalty_strategy"}
    missing_config = sorted(required_config.difference(config))
    if missing_config:
        raise RuntimeError("cache config missing: %s" % ", ".join(missing_config))
    if config["synergy_strategy"] != 1:
        raise RuntimeError("this full replay requires the official SAMMLV synergy_strategy=1")
    direct_recognition = cache.get("direct_recognition", {})
    if set(direct_recognition) != {"pred_list", "gt_tp_list"}:
        raise RuntimeError("cache direct_recognition audit values are missing")

    videos = cache.get("videos", [])
    if len(videos) != 79:
        raise RuntimeError("cache completeness failure: expected 79 videos, got %d" % len(videos))
    for expected_index, video in enumerate(videos):
        if video.get("video_index_global") != expected_index:
            raise RuntimeError("cache order failure at global video index %d" % expected_index)
        required_video = {
            "subject_id", "subject_index", "video_index_within_subject", "video_name",
            "spot_response", "recognition_sequence", "gt_intervals", "emotion_labels",
            "direct_spot_predictions",
        }
        missing_video = sorted(required_video.difference(video))
        if missing_video:
            raise RuntimeError("video %d missing: %s" % (expected_index, ", ".join(missing_video)))
    return cache


def _group_cached_videos(videos):
    subjects = []
    current_index = None
    current_videos = None
    for video in videos:
        subject_index = video["subject_index"]
        if subject_index != current_index:
            if subject_index != len(subjects):
                raise RuntimeError("non-contiguous subject order at video %d" % video["video_index_global"])
            current_index = subject_index
            current_videos = []
            subjects.append(current_videos)
        if video["video_index_within_subject"] != len(current_videos):
            raise RuntimeError("non-contiguous video order within subject %d" % subject_index)
        current_videos.append(video)
    return subjects


def _official_replay_functions():
    """Load the official decoder/evaluator only after cache integrity passes."""
    from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
    from training_utils import recognition, sequence_evaluation, spotting

    return MeanAveragePrecision2d, recognition, sequence_evaluation, spotting


def replay(cache):
    MeanAveragePrecision2d, recognition, sequence_evaluation, spotting = _official_replay_functions()
    config = cache["config"]
    subject_videos = _group_cached_videos(cache["videos"])
    final_samples = [[video["gt_intervals"] for video in videos] for videos in subject_videos]
    final_emotions = [[video["emotion_labels"] for video in videos] for videos in subject_videos]

    metric_final = MeanAveragePrecision2d(num_classes=1)
    total_gt_spot = 0
    pred_list = []
    gt_tp_list = []
    pred_window_list = []
    pred_single_list = []
    replay_predictions = []

    for subject_index, videos in enumerate(subject_videos):
        result_all = [np.asarray(video["spot_response"]) for video in videos]
        result1_all = [np.asarray(video["recognition_sequence"]) for video in videos]
        preds, _, total_gt_spot, metric_video, metric_final = spotting(
            final_samples,
            subject_index,
            result_all,
            total_gt_spot,
            metric_final,
            config["k_p"],
            config["interval_strategy"],
            "SAMMLV",
        )
        raw_counts = sequence_evaluation(total_gt_spot, metric_final)
        replay_predictions.extend(_predictions_as_lists(preds))
        pred_list, _, gt_tp_list, pred_window_list, pred_single_list = recognition(
            result1_all,
            preds,
            metric_video,
            final_emotions,
            subject_index,
            pred_list,
            gt_tp_list,
            final_samples,
            pred_window_list,
            pred_single_list,
            config["frame_skip"],
            config["penalty_strategy"],
        )

    tp_neutral = 0
    fp_neutral = 0
    for prediction, target in zip(pred_list, gt_tp_list):
        if prediction == config["emotion_type"] - 1:
            if target == -1:
                fp_neutral += 1
            else:
                tp_neutral += 1
    full_counts = (raw_counts[0] - tp_neutral, raw_counts[1] - fp_neutral, raw_counts[2] + tp_neutral)
    direct_predictions = _predictions_as_lists(
        [video["direct_spot_predictions"] for video in cache["videos"]]
    )
    recognition_match = (
        cache["direct_recognition"]["pred_list"] == pred_list
        and cache["direct_recognition"]["gt_tp_list"] == gt_tp_list
    )
    return raw_counts, full_counts, direct_predictions == replay_predictions, recognition_match


def _write_audit(status, detail, cache=None, raw_counts=None, full_counts=None,
                 predictions_match=None, recognition_match=None):
    lines = [
        "# BoostingVRME SAMMLV Full Exact Replay Audit",
        "",
        "## Recorder modification",
        "",
        "- `train.py`: imports at lines 16–18 and recorder helpers at lines 21–73.",
        "- `train.py`: response-list initialization at line 121; test-path capture at lines 379–392; guarded write at lines 409–422.",
        "- The capture copies already-assembled `result_all[video_index]` and `result1_all[video_index]`; it does not assign to either official variable or invoke a decoder.",
        "",
        "## Cache integrity",
        "",
    ]
    if cache is None:
        lines.append("- Cache unavailable or invalid: %s" % detail)
    else:
        lines.extend([
            "- Dataset: `%s`" % cache["dataset"],
            "- Official git commit: `%s`" % cache["official_git_commit"],
            "- Config: `%s`." % cache["config"],
            "- Videos: %d / 79; ordered by validated `video_index_global`." % len(cache["videos"]),
            "- Direct inference raw: `%s`" % _as_counts(cache["direct_inference_counts"]["raw"]),
            "- Direct inference full: `%s`" % _as_counts(cache["direct_inference_counts"]["full"]),
            "- Cache replay raw: `%s`; F1 = %.10f." % (raw_counts, _f1(raw_counts)),
            "- Cache replay full: `%s`; F1 = %.10f." % (full_counts, _f1(full_counts)),
            "- F1 target check: raw `%.4f`, full `%.4f`." % (_f1(raw_counts), _f1(full_counts)),
            "- Decoded spotting predictions exact match: `%s`." % predictions_match,
            "- Recognition predictions and matched-GT sequence exact match: `%s`." % recognition_match,
            "- Count exact match: `%s`." % (
                _as_counts(cache["direct_inference_counts"]["raw"]) == raw_counts
                and _as_counts(cache["direct_inference_counts"]["full"]) == full_counts
            ),
        ])
    lines.extend(["", "## Verdict", "", "`BOOSTING_SAMMLV_FULL_EXACT_REPLAY = %s`" % status, ""])
    with open(AUDIT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    try:
        cache = _load_and_validate_cache()
        raw_counts, full_counts, predictions_match, recognition_match = replay(cache)
        direct_raw = _as_counts(cache["direct_inference_counts"]["raw"])
        direct_full = _as_counts(cache["direct_inference_counts"]["full"])
        passed = (
            direct_raw == EXPECTED_RAW == raw_counts
            and direct_full == EXPECTED_FULL == full_counts
            and predictions_match
            and recognition_match
        )
        _write_audit(
            "PASS" if passed else "FAIL",
            "replay completed",
            cache,
            raw_counts,
            full_counts,
            predictions_match,
            recognition_match,
        )
        if not passed:
            raise RuntimeError("exact replay mismatch; see %s" % AUDIT_PATH)
    except Exception as error:
        _write_audit("FAIL", str(error))
        print("BOOSTING_SAMMLV_FULL_EXACT_REPLAY = FAIL", file=sys.stderr)
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print("BOOSTING_SAMMLV_FULL_EXACT_REPLAY = PASS")


if __name__ == "__main__":
    main()
