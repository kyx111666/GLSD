"""Paper-aligned ME-TST metrics.

The functions here mirror the original repository's evaluation path:
training_utils.spotting -> sequence_evaluation -> recognition ->
train_evaluate.final_evaluation. They are kept skill-local so the project
source remains read-only.
"""

from collections import Counter

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix

from training_utils import smooth  # noqa: E402
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d  # noqa: E402


EMOTION_ID = {"negative": 0, "positive": 1, "surprise": 2, "others": 3}
EMOTION_NAME = {0: "negative", 1: "positive", 2: "surprise", 3: "others"}


def empty_paper_stats():
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "num_gt": 0,
        "num_pred": 0,
        "matched_pred_emotion_ids": [],
        "matched_gt_emotion_ids": [],
        "all_pred_emotion_ids": [],
        "all_gt_tp_emotion_ids": [],
        "matched_emotion_correct": 0,
        "matches": 0,
    }


def paper_predictions_from_score(score, k_p, p=0.55):
    """Reproduce training_utils.spotting for one video.

    Prediction rows intentionally keep the original repo layout:
    [onset, 0, offset, 0, class_id, confidence, peak].
    The original code stores confidence as 0 and the peak in the last column;
    MeanAveragePrecision2d sorting is disabled in this repository, so row order
    follows find_peaks order.
    """
    score_s = np.asarray(smooth(np.asarray(score), k_p * 2))
    if score_s.size == 0:
        return np.empty((0, 7))
    threshold = score_s.mean() + p * (float(score_s.max()) - score_s.mean())
    peaks, _ = find_peaks(score_s, height=threshold, distance=k_p)
    if len(peaks) == 0:
        return np.empty((0, 7))
    return np.asarray([[int(peak - k_p), 0, int(peak + k_p), 0, 0, 0, int(peak)] for peak in peaks])


def paper_predictions_from_detections(detections, frame_skip):
    rows = []
    for det in detections:
        onset = int(det["onset_frame"]) // frame_skip
        peak = int(det["peak_frame"]) // frame_skip
        offset = int(det["offset_frame"]) // frame_skip
        rows.append([onset, 0, offset, 0, 0, 0, peak])
    if not rows:
        return np.empty((0, 7))
    return np.asarray(rows)


def paper_gt_from_samples(samples):
    rows = []
    for sample in samples:
        rows.append([int(sample[0]), 0, int(sample[2]), 0, 0, 0, 0, int(sample[1])])
    if not rows:
        return np.empty((0, 8))
    return np.asarray(rows)


def _safe_majority_emotion(emotion, onset, offset, peak):
    start = max(0, int(onset) + 1)
    stop = max(1, int(offset) - 1)
    values = list(np.asarray(emotion)[start:stop])
    if not values:
        peak = min(max(0, int(peak)), len(emotion) - 1)
        values = [int(np.asarray(emotion)[peak])] if len(emotion) else [3]
    emotion_id, _ = Counter(int(item) for item in values).most_common(1)[0]
    return int(emotion_id)


def paper_match_one_video(preds, gt, iou_threshold=0.5):
    metric = MeanAveragePrecision2d(num_classes=1)
    metric.add(preds, gt)
    value = metric.value(iou_thresholds=iou_threshold)[iou_threshold][0]
    tp_flags = [int(item) for item in value["tp"]]
    fp_flags = [int(item) for item in value["fp"]]
    pred_match_gt = value["pred_match_gt"].get(0, [])
    return {
        "tp": int(sum(tp_flags)),
        "fp": int(sum(fp_flags)),
        "fn": int(len(gt) - sum(tp_flags)),
        "tp_flags": tp_flags,
        "fp_flags": fp_flags,
        "pred_match_gt": [int(item) for item in pred_match_gt],
    }


def add_paper_video_stats(total, preds, gt, emotion, gt_emotions, iou_threshold=0.5):
    match = paper_match_one_video(preds, gt, iou_threshold)
    total["tp"] += match["tp"]
    total["fp"] += match["fp"]
    total["fn"] += match["fn"]
    total["num_gt"] += int(len(gt))
    total["num_pred"] += int(len(preds))

    video_matches = []
    for pred_index, sample_index in enumerate(match["pred_match_gt"]):
        pred = preds[pred_index]
        pred_emotion = _safe_majority_emotion(
            emotion,
            onset=pred[0],
            offset=pred[2],
            peak=pred[-1],
        )
        total["all_pred_emotion_ids"].append(pred_emotion)
        if sample_index != -1:
            gt_name = gt_emotions[sample_index]
            gt_emotion = EMOTION_ID.get(str(gt_name), 3)
            total["all_gt_tp_emotion_ids"].append(gt_emotion)
            total["matched_pred_emotion_ids"].append(pred_emotion)
            total["matched_gt_emotion_ids"].append(gt_emotion)
            total["matched_emotion_correct"] += int(pred_emotion == gt_emotion)
            total["matches"] += 1
        else:
            gt_emotion = -1
            total["all_gt_tp_emotion_ids"].append(-1)

        video_matches.append({
            "pred_index": int(pred_index),
            "gt_index": int(sample_index),
            "pred_onset": int(pred[0]),
            "pred_peak": int(pred[-1]),
            "pred_offset": int(pred[2]),
            "pred_emotion_id": int(pred_emotion),
            "pred_emotion": EMOTION_NAME.get(int(pred_emotion), str(pred_emotion)),
            "gt_emotion_id": int(gt_emotion),
            "gt_emotion": EMOTION_NAME.get(int(gt_emotion), str(gt_emotion)) if gt_emotion != -1 else "none",
            "emotion_correct": bool(sample_index != -1 and pred_emotion == gt_emotion),
        })

    return match, video_matches


def _repo_confusion_stats(gt, pred, class_id):
    gt_binary = [1 if int(item) == int(class_id) else 0 for item in gt]
    pred_binary = [1 if int(item) == int(class_id) else 0 for item in pred]
    tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary).ravel()
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "class_id": int(class_id),
        "class_name": EMOTION_NAME.get(int(class_id), str(class_id)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def paper_recognition_summary(gt, pred, class_ids):
    per_class = []
    for class_id in class_ids:
        try:
            per_class.append(_repo_confusion_stats(gt, pred, class_id))
        except Exception:
            # Matches training_utils.recognition_evaluation, which skips
            # classes whose one-vs-rest confusion matrix cannot be raveled.
            pass

    if not per_class:
        return {
            "num_matched": len(gt),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "uf1": 0.0,
            "uar": 0.0,
            "f1_score": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "per_class": [],
        }

    macro_precision = float(np.mean([item["precision"] for item in per_class]))
    macro_recall = float(np.mean([item["recall"] for item in per_class]))
    f1_score = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if macro_precision + macro_recall else 0.0
    )
    return {
        "num_matched": len(gt),
        "tp": int(sum(item["tp"] for item in per_class)),
        "fp": int(sum(item["fp"] for item in per_class)),
        "fn": int(sum(item["fn"] for item in per_class)),
        "tn": int(sum(item["tn"] for item in per_class)),
        "uf1": round(float(np.mean([item["f1"] for item in per_class])), 4),
        "uar": round(float(np.mean([item["recall"] for item in per_class])), 4),
        "f1_score": round(float(f1_score), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "per_class": [
            {
                **item,
                "precision": round(item["precision"], 4),
                "recall": round(item["recall"], 4),
                "f1": round(item["f1"], 4),
            }
            for item in per_class
        ],
    }


def finalize_paper_stats(stats):
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_4emo = [int(item) for item in stats.get("matched_gt_emotion_ids", [])]
    pred_4emo = [int(item) for item in stats.get("matched_pred_emotion_ids", [])]
    recog_4emo = paper_recognition_summary(gt_4emo, pred_4emo, [0, 1, 2, 3])

    gt_3emo = []
    pred_3emo = []
    for gt_id, pred_id in zip(gt_4emo, pred_4emo):
        if gt_id != 3:
            gt_3emo.append(gt_id)
            pred_3emo.append(pred_id)
    recog_3emo = paper_recognition_summary(gt_3emo, pred_3emo, [0, 1, 2])

    emotion_acc = (
        stats["matched_emotion_correct"] / stats["matches"]
        if stats["matches"] else 0.0
    )
    return {
        "paper_strict": True,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "num_pred": int(stats.get("num_pred", tp + fp)),
        "num_gt": int(stats.get("num_gt", tp + fn)),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "spotting_f1": round(float(f1), 4),
        "matched_emotion_accuracy": round(float(emotion_acc), 4),
        "recognition_uf1_4emo": recog_4emo["uf1"],
        "recognition_uar_4emo": recog_4emo["uar"],
        "recognition_f1_score_4emo": recog_4emo["f1_score"],
        "recognition_num_matched_4emo": recog_4emo["num_matched"],
        "recognition_counts_4emo": {
            "tp": recog_4emo["tp"],
            "fp": recog_4emo["fp"],
            "fn": recog_4emo["fn"],
            "tn": recog_4emo["tn"],
        },
        "recognition_uf1_3emo_wo_others": recog_3emo["uf1"],
        "recognition_uar_3emo_wo_others": recog_3emo["uar"],
        "recognition_f1_score_3emo_wo_others": recog_3emo["f1_score"],
        "recognition_num_matched_3emo_wo_others": recog_3emo["num_matched"],
        "recognition_counts_3emo_wo_others": {
            "tp": recog_3emo["tp"],
            "fp": recog_3emo["fp"],
            "fn": recog_3emo["fn"],
            "tn": recog_3emo["tn"],
        },
        "strs_4emo": round(float(f1 * recog_4emo["f1_score"]), 4),
        "strs_3emo_wo_others": round(float(f1 * recog_3emo["f1_score"]), 4),
        "srts_repo_4emo": round(float(f1 * recog_4emo["f1_score"]), 4),
        "srts_repo_3emo_wo_others": round(float(f1 * recog_3emo["f1_score"]), 4),
        "srts_uf1_4emo": round(float(f1 * recog_4emo["uf1"]), 4),
        "srts_uf1_3emo_wo_others": round(float(f1 * recog_3emo["uf1"]), 4),
    }
