
#!/usr/bin/env python3

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn

from scipy.signal import (
    find_peaks,
    peak_prominences,
)

import training_utils as tu


# ============================================================
# Locked protocol
# ============================================================

DATASET = "CASME_3"

EXPECTED_SUBJECTS = 94
EXPECTED_VIDEOS = 462
EXPECTED_GT = 853

EXPECTED_NATIVE_RAW = (
    84, 793, 769
)

EXPECTED_NATIVE_FULL = (
    84, 786, 769
)

OFFICIAL_COMMIT = (
    "689a7baa5edcb5831f6f2a911e22b4673fc70ac3"
)

REFERENCE_SCALES = [
    1.0,
    1.5,
    2.0,
]

LOCAL_RADII = [
    1.0,
    2.0,
    3.0,
]

THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.65,
    0.75,
]


# ============================================================
# Basic helpers
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path):

    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def normalize(obj):

    if isinstance(obj, np.ndarray):
        return normalize(obj.tolist())

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, tuple):
        return [
            normalize(x)
            for x in obj
        ]

    if isinstance(obj, list):
        return [
            normalize(x)
            for x in obj
        ]

    if isinstance(obj, dict):
        return {
            str(k): normalize(v)
            for k, v in obj.items()
        }

    return obj


def metrics(counts):

    tp, fp, fn = [
        int(x)
        for x in counts
    ]

    p = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    r = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * tp / (
            2 * tp + fp + fn
        )
        if 2 * tp + fp + fn > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def add_counts(items):

    tp = sum(
        int(x[0])
        for x in items
    )

    fp = sum(
        int(x[1])
        for x in items
    )

    fn = sum(
        int(x[2])
        for x in items
    )

    return (
        tp,
        fp,
        fn,
    )


def write_csv(
    path,
    rows,
    fieldnames,
):

    path = Path(path)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def official_metric_counts(
    metric,
    total_gt,
):

    result = metric.value(
        iou_thresholds=0.5
    )[0.5][0]

    tp = int(
        sum(result["tp"])
    )

    fp = int(
        sum(result["fp"])
    )

    fn = int(total_gt - tp)

    return (
        tp,
        fp,
        fn,
    )


def interval_iou_inclusive(
    pred,
    gt,
):

    ps = int(pred[0])
    pe = int(pred[2])

    gs = int(gt[0])
    ge = int(gt[2])

    inter = max(
        0,
        min(pe, ge)
        - max(ps, gs)
        + 1,
    )

    if inter <= 0:
        return 0.0

    plen = pe - ps + 1
    glen = ge - gs + 1

    union = (
        plen
        + glen
        - inter
    )

    return (
        inter / union
        if union > 0
        else 0.0
    )


# ============================================================
# Load cache
# ============================================================

def load_cache(path):

    with open(path, "rb") as f:
        cache = pickle.load(f)

    assert (
        cache["dataset"]
        == DATASET
    )

    assert (
        cache["video_count"]
        == EXPECTED_VIDEOS
    )

    assert (
        cache["subject_count"]
        == EXPECTED_SUBJECTS
    )

    videos = sorted(
        cache["videos"],
        key=lambda x:
            int(
                x[
                    "video_index_global"
                ]
            ),
    )

    assert [
        int(
            v[
                "video_index_global"
            ]
        )
        for v in videos
    ] == list(
        range(
            EXPECTED_VIDEOS
        )
    )

    gt_count = sum(
        len(v["gt_intervals"])
        for v in videos
    )

    assert gt_count == EXPECTED_GT, (
        gt_count
    )

    return (
        cache,
        videos,
    )


def group_videos(videos):

    grouped = [
        []
        for _ in range(
            EXPECTED_SUBJECTS
        )
    ]

    for video in videos:

        idx = int(
            video[
                "subject_index"
            ]
        )

        grouped[idx].append(
            video
        )

    assert all(
        len(x) > 0
        for x in grouped
    )

    return grouped


# ============================================================
# Native exact replay hard gate
# ============================================================

def native_exact_replay(
    cache,
    grouped,
):

    config = cache["config"]

    final_samples = [
        [
            v["gt_intervals"]
            for v in subject
        ]
        for subject in grouped
    ]

    final_emotions = [
        [
            v["emotion_labels"]
            for v in subject
        ]
        for subject in grouped
    ]

    metric_final = (
        tu.MeanAveragePrecision2d(
            num_classes=1
        )
    )

    total_gt = 0

    pred_list = []
    gt_tp_list = []

    pred_window_list = []
    pred_single_list = []

    replay_predictions = []

    # Ensure official find_peaks
    original_find_peaks = (
        tu.find_peaks
    )

    try:

        for subject_index in range(
            EXPECTED_SUBJECTS
        ):

            subject = grouped[
                subject_index
            ]

            result_all = [
                np.asarray(
                    v["spot_response"],
                    dtype=float,
                )
                for v in subject
            ]

            result1_all = [
                np.asarray(
                    v[
                        "recognition_sequence"
                    ]
                )
                for v in subject
            ]

            (
                preds,
                _,
                total_gt,
                metric_video,
                metric_final,
            ) = tu.spotting(
                final_samples,
                subject_index,
                result_all,
                total_gt,
                metric_final,
                int(
                    config["k_p"]
                ),
                int(
                    config[
                        "interval_strategy"
                    ]
                ),
                DATASET,
            )

            replay_predictions.extend(
                [
                    normalize(x)
                    for x in preds
                ]
            )

            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                (
                    pred_list,
                    _,
                    gt_tp_list,
                    pred_window_list,
                    pred_single_list,
                ) = tu.recognition(
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
                    int(
                        config[
                            "frame_skip"
                        ]
                    ),
                    int(
                        config[
                            "penalty_strategy"
                        ]
                    ),
                )

    finally:

        tu.find_peaks = (
            original_find_peaks
        )

    raw_counts = (
        official_metric_counts(
            metric_final,
            total_gt,
        )
    )

    tp_neutral = 0
    fp_neutral = 0

    neutral_label = (
        int(
            config[
                "emotion_type"
            ]
        )
        - 1
    )

    for prediction, target in zip(
        pred_list,
        gt_tp_list,
    ):

        if (
            int(prediction)
            == neutral_label
        ):

            if int(target) == -1:
                fp_neutral += 1

            else:
                tp_neutral += 1

    full_counts = (
        raw_counts[0]
        - tp_neutral,

        raw_counts[1]
        - fp_neutral,

        raw_counts[2]
        + tp_neutral,
    )

    direct_predictions = [
        normalize(
            v[
                "direct_spot_predictions"
            ]
        )
        for subject in grouped
        for v in subject
    ]

    prediction_exact = (
        direct_predictions
        == replay_predictions
    )

    recognition_exact = (
        normalize(
            cache[
                "direct_recognition"
            ]["pred_list"]
        )
        ==
        normalize(
            pred_list
        )
    )

    gt_exact = (
        normalize(
            cache[
                "direct_recognition"
            ]["gt_tp_list"]
        )
        ==
        normalize(
            gt_tp_list
        )
    )

    passed = (
        raw_counts
        == EXPECTED_NATIVE_RAW

        and full_counts
        == EXPECTED_NATIVE_FULL

        and prediction_exact
        and recognition_exact
        and gt_exact
    )

    return {
        "passed": passed,

        "raw_counts":
            raw_counts,

        "full_counts":
            full_counts,

        "prediction_exact":
            prediction_exact,

        "recognition_exact":
            recognition_exact,

        "matched_gt_exact":
            gt_exact,
    }


# ============================================================
# Locked GLSD feature construction
# ============================================================

class GLSDFeatures:

    def __init__(
        self,
        response,
        k,
    ):

        self.x = np.asarray(
            response,
            dtype=float,
        ).reshape(-1)

        self.T = len(
            self.x
        )

        self.k = max(
            1,
            int(k),
        )

        self.scale_to_width = {}

        self.curves = {}
        self.peaks = {}
        self.ranges = {}

        self.local_cache = {}
        self.score_cache = {}

        if self.T == 0:
            return

        # Constant original curve -> no candidates
        self.constant = (
            float(
                np.ptp(
                    self.x
                )
            )
            <= 1e-12
        )

        for scale in REFERENCE_SCALES:

            width = min(
                self.T,
                max(
                    1,
                    int(
                        round(
                            scale
                            * self.k
                        )
                    ),
                ),
            )

            self.scale_to_width[
                float(scale)
            ] = int(width)

            if width in self.curves:
                continue

            kernel = (
                np.ones(
                    width,
                    dtype=float,
                )
                / float(width)
            )

            y = np.convolve(
                self.x,
                kernel,
                mode="same",
            )

            self.curves[
                width
            ] = y

            self.ranges[
                width
            ] = max(
                float(
                    np.ptp(y)
                ),
                1e-12,
            )

            if self.constant:

                peaks = np.array(
                    [],
                    dtype=int,
                )

            else:

                peaks, _ = find_peaks(
                    y,
                    distance=self.k,
                )

                peaks = np.asarray(
                    peaks,
                    dtype=int,
                )

            self.peaks[
                width
            ] = peaks

    def local_values(
        self,
        width,
        rho,
    ):

        key = (
            int(width),
            float(rho),
        )

        if key in self.local_cache:
            return self.local_cache[
                key
            ]

        y = self.curves[
            width
        ]

        peaks = self.peaks[
            width
        ]

        dynamic_range = (
            self.ranges[
                width
            ]
        )

        radius = max(
            1,
            int(
                round(
                    float(rho)
                    * self.k
                )
            ),
        )

        q = {}

        for peak in peaks:

            p = int(peak)

            left_start = max(
                0,
                p - radius,
            )

            right_end = min(
                self.T - 1,
                p + radius,
            )

            left_base = float(
                np.min(
                    y[
                        left_start:
                        p + 1
                    ]
                )
            )

            right_base = float(
                np.min(
                    y[
                        p:
                        right_end + 1
                    ]
                )
            )

            local = max(
                0.0,
                float(y[p])
                - max(
                    left_base,
                    right_base,
                ),
            )

            q[p] = (
                local
                / dynamic_range
            )

        self.local_cache[
            key
        ] = q

        return q

    def scores(
        self,
        reference_scale,
        rho,
    ):

        key = (
            float(reference_scale),
            float(rho),
        )

        if key in self.score_cache:
            return self.score_cache[
                key
            ]

        if self.T == 0 or self.constant:

            result = {
                "peaks": np.array(
                    [],
                    dtype=int,
                ),
                "G": np.array(
                    [],
                    dtype=float,
                ),
                "L": np.array(
                    [],
                    dtype=float,
                ),
                "S": np.array(
                    [],
                    dtype=float,
                ),
            }

            self.score_cache[
                key
            ] = result

            return result

        ref_width = (
            self.scale_to_width[
                float(
                    reference_scale
                )
            ]
        )

        ref_curve = (
            self.curves[
                ref_width
            ]
        )

        ref_peaks = (
            self.peaks[
                ref_width
            ]
        )

        if len(ref_peaks) == 0:

            result = {
                "peaks": ref_peaks,
                "G": np.array(
                    [],
                    dtype=float,
                ),
                "L": np.array(
                    [],
                    dtype=float,
                ),
                "S": np.array(
                    [],
                    dtype=float,
                ),
            }

            self.score_cache[
                key
            ] = result

            return result

        prominence = (
            peak_prominences(
                ref_curve,
                ref_peaks,
            )[0]
        )

        G = (
            prominence
            / self.ranges[
                ref_width
            ]
        )

        delta = max(
            1,
            int(
                round(
                    0.5
                    * self.k
                )
            ),
        )

        local_by_width = {
            width:
                self.local_values(
                    width,
                    rho,
                )

            for width in self.curves
        }

        L_values = []

        effective_widths = list(
            self.curves.keys()
        )

        for candidate in ref_peaks:

            c = int(candidate)

            aligned_values = []

            for width in effective_widths:

                peaks = (
                    self.peaks[
                        width
                    ]
                )

                q = (
                    local_by_width[
                        width
                    ]
                )

                valid = [
                    int(p)
                    for p in peaks
                    if abs(
                        int(p) - c
                    ) <= delta
                ]

                if not valid:

                    aligned_values.append(
                        0.0
                    )

                    continue

                best = min(
                    valid,
                    key=lambda p: (
                        abs(p - c),
                        -float(
                            q[p]
                        ),
                        p,
                    ),
                )

                aligned_values.append(
                    float(
                        q[best]
                    )
                )

            L_values.append(
                float(
                    np.median(
                        aligned_values
                    )
                )
            )

        L = np.asarray(
            L_values,
            dtype=float,
        )

        S = (
            np.asarray(
                G,
                dtype=float,
            )
            + L
        ) / 2.0

        result = {
            "peaks":
                np.asarray(
                    ref_peaks,
                    dtype=int,
                ),

            "G":
                np.asarray(
                    G,
                    dtype=float,
                ),

            "L":
                L,

            "S":
                S,
        }

        self.score_cache[
            key
        ] = result

        return result

    def selected_peaks(
        self,
        reference_scale,
        rho,
        tau,
    ):

        evidence = self.scores(
            reference_scale,
            rho,
        )

        peaks = (
            evidence[
                "peaks"
            ]
        )

        score = (
            evidence[
                "S"
            ]
        )

        if len(peaks) == 0:
            return []

        selected = peaks[
            score
            >= float(tau)
        ]

        return [
            int(x)
            for x in selected
        ]


# ============================================================
# Grid
# ============================================================

def build_grid():

    grid = []

    config_id = 0

    for a0 in REFERENCE_SCALES:

        for rho in LOCAL_RADII:

            for tau in THRESHOLDS:

                grid.append({
                    "config_id":
                        config_id,

                    "reference_scale":
                        float(a0),

                    "local_radius":
                        float(rho),

                    "threshold":
                        float(tau),

                    "grid_order":
                        config_id,
                })

                config_id += 1

    assert len(grid) == 90

    return grid


# ============================================================
# Official Boosting geometry adapter
# ============================================================

@contextlib.contextmanager
def patched_candidate_source(
    selected_per_video,
):

    original = (
        tu.find_peaks
    )

    state = {
        "index": 0
    }

    def fake_find_peaks(
        *_args,
        **_kwargs,
    ):

        i = state["index"]

        if i >= len(
            selected_per_video
        ):
            raise RuntimeError(
                "GLSD candidate source "
                "called too many times"
            )

        selected = (
            selected_per_video[
                i
            ]
        )

        state["index"] += 1

        return (
            np.asarray(
                selected,
                dtype=int,
            ),
            {},
        )

    tu.find_peaks = (
        fake_find_peaks
    )

    try:
        yield

    finally:

        tu.find_peaks = (
            original
        )

        if (
            state["index"]
            != len(
                selected_per_video
            )
        ):
            raise RuntimeError(
                "GLSD candidate-source "
                "call count mismatch: "
                f'{state["index"]} vs '
                f'{len(selected_per_video)}'
            )


# ============================================================
# Decode one subject under one GLSD configuration
# ============================================================

def decode_subject(
    subject_index,
    config_item,
    grouped,
    feature_bank,
    final_samples,
    final_emotions,
    cache_config,
    need_recognition=False,
):

    subject = grouped[
        subject_index
    ]

    selected_per_video = [
        feature_bank[
            subject_index
        ][video_index].selected_peaks(
            config_item[
                "reference_scale"
            ],
            config_item[
                "local_radius"
            ],
            config_item[
                "threshold"
            ],
        )
        for video_index in range(
            len(subject)
        )
    ]

    result_all = [
        np.asarray(
            v["spot_response"],
            dtype=float,
        )
        for v in subject
    ]

    metric_final = (
        tu.MeanAveragePrecision2d(
            num_classes=1
        )
    )

    total_gt = 0

    with patched_candidate_source(
        selected_per_video
    ):

        (
            preds,
            _,
            total_gt,
            metric_video,
            metric_final,
        ) = tu.spotting(
            final_samples,
            subject_index,
            result_all,
            total_gt,
            metric_final,
            int(
                cache_config[
                    "k_p"
                ]
            ),
            int(
                cache_config[
                    "interval_strategy"
                ]
            ),
            DATASET,
        )

    raw_counts = (
        official_metric_counts(
            metric_final,
            total_gt,
        )
    )

    if not need_recognition:

        return {
            "raw_counts":
                raw_counts,

            "preds":
                preds,

            "selected_peaks":
                selected_per_video,

            "metric_video":
                metric_video,
        }

    result1_all = [
        np.asarray(
            v[
                "recognition_sequence"
            ]
        )
        for v in subject
    ]

    with contextlib.redirect_stdout(
        io.StringIO()
    ):

        (
            pred_list,
            _,
            gt_tp_list,
            _,
            _,
        ) = tu.recognition(
            result1_all,
            preds,
            metric_video,
            final_emotions,
            subject_index,
            [],
            [],
            final_samples,
            [],
            [],
            int(
                cache_config[
                    "frame_skip"
                ]
            ),
            int(
                cache_config[
                    "penalty_strategy"
                ]
            ),
        )

    neutral_label = (
        int(
            cache_config[
                "emotion_type"
            ]
        )
        - 1
    )

    tp_neutral = 0
    fp_neutral = 0

    for prediction, target in zip(
        pred_list,
        gt_tp_list,
    ):

        if (
            int(prediction)
            == neutral_label
        ):

            if int(target) == -1:
                fp_neutral += 1
            else:
                tp_neutral += 1

    full_counts = (
        raw_counts[0]
        - tp_neutral,

        raw_counts[1]
        - fp_neutral,

        raw_counts[2]
        + tp_neutral,
    )

    return {
        "raw_counts":
            raw_counts,

        "full_counts":
            full_counts,

        "preds":
            preds,

        "selected_peaks":
            selected_per_video,

        "metric_video":
            metric_video,

        "recognition_pred":
            [
                int(x)
                for x in pred_list
            ],

        "recognition_gt":
            [
                int(x)
                for x in gt_tp_list
            ],
    }


# ============================================================
# Prediction-level audit extraction
# ============================================================

def subject_prediction_evidence(
    subject_index,
    subject_name,
    config_item,
    decoded,
    grouped,
    final_samples,
    cache_config,
):

    preds = decoded["preds"]

    metric_video = (
        decoded[
            "metric_video"
        ]
    )

    value = metric_video.value(
        iou_thresholds=0.5
    )[0.5][0]

    match_map = (
        value[
            "pred_match_gt"
        ]
    )

    recognition_pred = (
        decoded[
            "recognition_pred"
        ]
    )

    recognition_gt = (
        decoded[
            "recognition_gt"
        ]
    )

    neutral_label = (
        int(
            cache_config[
                "emotion_type"
            ]
        )
        - 1
    )

    prediction_rows = []
    video_rows = []
    video_prediction_records = []

    flat_recognition_index = 0

    subject_raw_tp = 0
    subject_raw_fp = 0
    subject_raw_fn = 0

    subject_full_tp = 0
    subject_full_fp = 0
    subject_full_fn = 0

    subject_videos = (
        grouped[
            subject_index
        ]
    )

    for video_index, video in enumerate(
        subject_videos
    ):

        video_preds = preds[
            video_index
        ]

        gt_list = (
            final_samples[
                subject_index
            ][video_index]
        )

        matches = list(
            match_map.get(
                video_index,
                [],
            )
        )

        if len(video_preds) == 0:
            matches = []

        if (
            len(matches)
            != len(video_preds)
        ):
            raise RuntimeError(
                "pred_match_gt length mismatch "
                f"subject={subject_index}, "
                f"video={video_index}: "
                f"{len(matches)} vs "
                f"{len(video_preds)}"
            )

        raw_tp = 0
        raw_fp = 0

        neutral_tp = 0
        neutral_fp = 0

        this_video_records = []

        for pred_index, pred in enumerate(
            video_preds
        ):

            sample_index = int(
                matches[
                    pred_index
                ]
            )

            is_tp = (
                sample_index
                != -1
            )

            if is_tp:
                raw_tp += 1
            else:
                raw_fp += 1

            if (
                flat_recognition_index
                >= len(
                    recognition_pred
                )
            ):
                raise RuntimeError(
                    "recognition index overflow"
                )

            recog_pred = int(
                recognition_pred[
                    flat_recognition_index
                ]
            )

            recog_gt = int(
                recognition_gt[
                    flat_recognition_index
                ]
            )

            flat_recognition_index += 1

            is_neutral = (
                recog_pred
                == neutral_label
            )

            if is_neutral:

                if is_tp:
                    neutral_tp += 1
                    synergy_action = (
                        "TP_TO_FN"
                    )

                else:
                    neutral_fp += 1
                    synergy_action = (
                        "REMOVE_FP"
                    )

            else:
                synergy_action = "KEEP"

            gt_index = (
                sample_index
                if is_tp
                else None
            )

            matched_gt = (
                gt_list[
                    sample_index
                ]
                if is_tp
                else None
            )

            all_ious = [
                interval_iou_inclusive(
                    pred,
                    gt,
                )
                for gt in gt_list
            ]

            best_iou = (
                max(all_ious)
                if all_ious
                else 0.0
            )

            matched_iou = (
                interval_iou_inclusive(
                    pred,
                    matched_gt,
                )
                if matched_gt
                is not None
                else 0.0
            )

            row = {
                "subject_index":
                    subject_index,

                "subject":
                    subject_name,

                "video_index_within_subject":
                    video_index,

                "video_index_global":
                    int(
                        video[
                            "video_index_global"
                        ]
                    ),

                "video":
                    str(
                        video["video"]
                    ),

                "config_id":
                    int(
                        config_item[
                            "config_id"
                        ]
                    ),

                "prediction_index":
                    pred_index,

                "onset":
                    int(pred[0]),

                "offset":
                    int(pred[2]),

                "peak":
                    int(pred[-1]),

                "matched_gt_index":
                    (
                        sample_index
                        if is_tp
                        else -1
                    ),

                "matched_gt":
                    (
                        json.dumps(
                            normalize(
                                matched_gt
                            )
                        )
                        if matched_gt
                        is not None
                        else ""
                    ),

                "best_iou_inclusive":
                    float(best_iou),

                "matched_iou_inclusive":
                    float(matched_iou),

                "raw_status":
                    (
                        "TP"
                        if is_tp
                        else "FP"
                    ),

                "recognition_prediction":
                    recog_pred,

                "recognition_target":
                    recog_gt,

                "neutral":
                    bool(
                        is_neutral
                    ),

                "synergy_action":
                    synergy_action,
            }

            prediction_rows.append(
                row
            )

            this_video_records.append(
                row
            )

        raw_fn = (
            len(gt_list)
            - raw_tp
        )

        full_tp = (
            raw_tp
            - neutral_tp
        )

        full_fp = (
            raw_fp
            - neutral_fp
        )

        full_fn = (
            raw_fn
            + neutral_tp
        )

        video_row = {
            "subject_index":
                subject_index,

            "subject":
                subject_name,

            "video_index_within_subject":
                video_index,

            "video_index_global":
                int(
                    video[
                        "video_index_global"
                    ]
                ),

            "video":
                str(
                    video["video"]
                ),

            "config_id":
                int(
                    config_item[
                        "config_id"
                    ]
                ),

            "gt_count":
                len(gt_list),

            "prediction_count":
                len(video_preds),

            "raw_tp":
                raw_tp,

            "raw_fp":
                raw_fp,

            "raw_fn":
                raw_fn,

            "neutral_tp":
                neutral_tp,

            "neutral_fp":
                neutral_fp,

            "full_tp":
                full_tp,

            "full_fp":
                full_fp,

            "full_fn":
                full_fn,
        }

        video_rows.append(
            video_row
        )

        video_prediction_records.append({
            "video_index_global":
                int(
                    video[
                        "video_index_global"
                    ]
                ),

            "subject_index":
                subject_index,

            "subject":
                subject_name,

            "video":
                str(
                    video["video"]
                ),

            "config":
                normalize(
                    config_item
                ),

            "selected_reference_peaks":
                normalize(
                    decoded[
                        "selected_peaks"
                    ][video_index]
                ),

            "decoded_predictions":
                normalize(
                    video_preds
                ),

            "prediction_evidence":
                this_video_records,
        })

        subject_raw_tp += raw_tp
        subject_raw_fp += raw_fp
        subject_raw_fn += raw_fn

        subject_full_tp += full_tp
        subject_full_fp += full_fp
        subject_full_fn += full_fn

    assert (
        flat_recognition_index
        == len(
            recognition_pred
        )
    )

    evidence_raw = (
        subject_raw_tp,
        subject_raw_fp,
        subject_raw_fn,
    )

    evidence_full = (
        subject_full_tp,
        subject_full_fp,
        subject_full_fn,
    )

    assert (
        evidence_raw
        == tuple(
            decoded[
                "raw_counts"
            ]
        )
    ), (
        evidence_raw,
        decoded[
            "raw_counts"
        ],
    )

    assert (
        evidence_full
        == tuple(
            decoded[
                "full_counts"
            ]
        )
    ), (
        evidence_full,
        decoded[
            "full_counts"
        ],
    )

    return (
        prediction_rows,
        video_rows,
        video_prediction_records,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    args = parser.parse_args()

    cache_path = Path(
        args.cache
    ).resolve()

    out = Path(
        args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = utc_now()

    # --------------------------------------------------------
    # Identity / manifest
    # --------------------------------------------------------

    script_path = Path(
        __file__
    ).resolve()

    repo_root = (
        script_path.parent
    )

    git_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    if git_commit != OFFICIAL_COMMIT:

        raise RuntimeError(
            "Official repo commit mismatch: "
            f"{git_commit}"
        )

    cache, videos = (
        load_cache(
            cache_path
        )
    )

    grouped = group_videos(
        videos
    )

    config_cache = (
        cache["config"]
    )

    k = int(
        config_cache[
            "k_p"
        ]
    )

    if k < 1:
        raise RuntimeError(
            "invalid k_p"
        )

    subject_names = [
        str(
            subject[0][
                "subject"
            ]
        )
        for subject in grouped
    ]

    gt_count = sum(
        len(
            v[
                "gt_intervals"
            ]
        )
        for v in videos
    )

    assert gt_count == EXPECTED_GT

    grid = build_grid()

    assert len(grid) == 90

    manifest = {
        "dataset":
            DATASET,

        "protocol":
            "official-response full-pipeline GLSD-90",

        "cache_absolute_path":
            str(cache_path),

        "cache_sha256":
            sha256_file(
                cache_path
            ),

        "script_absolute_path":
            str(script_path),

        "script_sha256":
            sha256_file(
                script_path
            ),

        "official_boostingvrme_git_commit":
            git_commit,

        "python_version":
            sys.version,

        "python_executable":
            sys.executable,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "scipy_version":
            scipy.__version__,

        "sklearn_version":
            sklearn.__version__,

        "command_line":
            sys.argv,

        "subject_count":
            EXPECTED_SUBJECTS,

        "video_count":
            EXPECTED_VIDEOS,

        "gt_event_count":
            EXPECTED_GT,

        "important_protocol_note":
            (
                "This official-response "
                "reconstruction contains "
                "853 GT events. It is not "
                "silently relabeled as the "
                "858-event controlled cache."
            ),

        "official_cache_k_p":
            k,

        "base_scale_source":
            (
                "fixed official response-cache "
                "k_p for apples-to-apples "
                "full-pipeline comparison"
            ),

        "glsd_grid_size":
            90,

        "glsd_grid_constants": {
            "reference_scales":
                REFERENCE_SCALES,

            "local_radii":
                LOCAL_RADII,

            "thresholds":
                THRESHOLDS,
        },

        "selection_tie_break": [
            "higher_F1",
            "higher_precision",
            "fewer_FP",
            "fixed_grid_order",
        ],

        "native_expected_counts": {
            "raw":
                list(
                    EXPECTED_NATIVE_RAW
                ),

            "full":
                list(
                    EXPECTED_NATIVE_FULL
                ),
        },

        "published_context_f1":
            0.0997,

        "execution_started_at_utc":
            started,

        "run_stdout_stderr_path":
            str(
                out
                / "run_stdout_stderr.log"
            ),

        "run_stdout_stderr_sha256":
            None,

        "run_stdout_stderr_status":
            "PENDING_FINALIZE_LOG",
    }

    # --------------------------------------------------------
    # HARD GATE: Native exact replay
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NATIVE EXACT REPLAY HARD GATE")
    print("=" * 70)

    native = native_exact_replay(
        cache,
        grouped,
    )

    print(
        "Native Raw :",
        native[
            "raw_counts"
        ],
    )

    print(
        "Native Full:",
        native[
            "full_counts"
        ],
    )

    print(
        "Spot predictions exact =",
        native[
            "prediction_exact"
        ],
    )

    print(
        "Recognition exact =",
        native[
            "recognition_exact"
        ],
    )

    print(
        "Matched-GT exact =",
        native[
            "matched_gt_exact"
        ],
    )

    if not native["passed"]:

        print(
            "NATIVE_EXACT_REPLAY = FAIL"
        )

        raise RuntimeError(
            "Native exact replay "
            "hard gate failed"
        )

    print(
        "NATIVE_EXACT_REPLAY = PASS"
    )

    # --------------------------------------------------------
    # Build fixed subject structures
    # --------------------------------------------------------

    final_samples = [
        [
            v[
                "gt_intervals"
            ]
            for v in subject
        ]
        for subject in grouped
    ]

    final_emotions = [
        [
            v[
                "emotion_labels"
            ]
            for v in subject
        ]
        for subject in grouped
    ]

    # --------------------------------------------------------
    # Precompute GLSD evidence objects
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILD GLSD FEATURE BANK")
    print("=" * 70)

    feature_bank = []

    for subject_index, subject in enumerate(
        grouped
    ):

        feature_bank.append(
            [
                GLSDFeatures(
                    v[
                        "spot_response"
                    ],
                    k,
                )
                for v in subject
            ]
        )

        if (
            (subject_index + 1)
            % 10 == 0
            or
            subject_index + 1
            == EXPECTED_SUBJECTS
        ):

            print(
                "feature subjects:",
                subject_index + 1,
                "/",
                EXPECTED_SUBJECTS,
            )

    # --------------------------------------------------------
    # 90 × 94 subject/config raw-count table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PRECOMPUTE SUBJECT × CONFIG RAW COUNTS")
    print("=" * 70)

    subject_config_counts = {
        int(c["config_id"]):
            [None]
            * EXPECTED_SUBJECTS

        for c in grid
    }

    for config_item in grid:

        cid = int(
            config_item[
                "config_id"
            ]
        )

        for subject_index in range(
            EXPECTED_SUBJECTS
        ):

            decoded = decode_subject(
                subject_index,
                config_item,
                grouped,
                feature_bank,
                final_samples,
                final_emotions,
                config_cache,
                need_recognition=False,
            )

            subject_config_counts[
                cid
            ][subject_index] = tuple(
                int(x)
                for x in decoded[
                    "raw_counts"
                ]
            )

        print(
            f"config {cid + 1:02d}/90 complete"
        )

    # --------------------------------------------------------
    # Full inner-fold evidence table:
    # 94 × 90 × 93 = 786,780
    # --------------------------------------------------------

    inner_path = (
        out
        / "glsd_search_inner_fold_counts.csv"
    )

    inner_fields = [
        "outer_subject_index",
        "outer_subject",
        "config_id",
        "reference_scale",
        "local_radius",
        "threshold",
        "inner_subject_index",
        "inner_subject",
        "tp",
        "fp",
        "fn",
    ]

    with inner_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=inner_fields,
        )

        writer.writeheader()

        for outer in range(
            EXPECTED_SUBJECTS
        ):

            for config_item in grid:

                cid = int(
                    config_item[
                        "config_id"
                    ]
                )

                for inner in range(
                    EXPECTED_SUBJECTS
                ):

                    if inner == outer:
                        continue

                    assert (
                        inner
                        != outer
                    )

                    tp, fp, fn = (
                        subject_config_counts[
                            cid
                        ][inner]
                    )

                    writer.writerow({
                        "outer_subject_index":
                            outer,

                        "outer_subject":
                            subject_names[
                                outer
                            ],

                        "config_id":
                            cid,

                        "reference_scale":
                            config_item[
                                "reference_scale"
                            ],

                        "local_radius":
                            config_item[
                                "local_radius"
                            ],

                        "threshold":
                            config_item[
                                "threshold"
                            ],

                        "inner_subject_index":
                            inner,

                        "inner_subject":
                            subject_names[
                                inner
                            ],

                        "tp":
                            tp,

                        "fp":
                            fp,

                        "fn":
                            fn,
                    })

    # --------------------------------------------------------
    # Outer selection
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTER LOSO CONFIGURATION SELECTION")
    print("=" * 70)

    search_rows = []
    selected_rows = []
    selection_trace = []

    selected_by_outer = {}

    for outer in range(
        EXPECTED_SUBJECTS
    ):

        inner_subjects = [
            i
            for i in range(
                EXPECTED_SUBJECTS
            )
            if i != outer
        ]

        assert len(
            inner_subjects
        ) == 93

        assert (
            outer
            not in inner_subjects
        )

        candidates = []

        for config_item in grid:

            cid = int(
                config_item[
                    "config_id"
                ]
            )

            pooled = add_counts(
                [
                    subject_config_counts[
                        cid
                    ][inner]
                    for inner
                    in inner_subjects
                ]
            )

            m = metrics(
                pooled
            )

            candidates.append({
                **config_item,

                "pooled_inner_tp":
                    m["tp"],

                "pooled_inner_fp":
                    m["fp"],

                "pooled_inner_fn":
                    m["fn"],

                "precision":
                    m[
                        "precision"
                    ],

                "recall":
                    m[
                        "recall"
                    ],

                "f1":
                    m["f1"],
            })

        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(
                    row["f1"]
                ),

                -float(
                    row[
                        "precision"
                    ]
                ),

                int(
                    row[
                        "pooled_inner_fp"
                    ]
                ),

                int(
                    row[
                        "grid_order"
                    ]
                ),
            ),
        )

        for rank, row in enumerate(
            ranked,
            start=1,
        ):
            row["rank"] = rank

        winner = ranked[0]

        selected_by_outer[
            outer
        ] = dict(
            winner
        )

        rank_lookup = {
            int(row["config_id"]):
                int(row["rank"])
            for row in ranked
        }

        for row in candidates:

            cid = int(
                row[
                    "config_id"
                ]
            )

            search_rows.append({
                "outer_subject_index":
                    outer,

                "outer_subject":
                    subject_names[
                        outer
                    ],

                **row,

                "final_rank":
                    rank_lookup[cid],

                "selected":
                    int(
                        cid
                        == int(
                            winner[
                                "config_id"
                            ]
                        )
                    ),
            })

        selected_rows.append({
            "outer_subject_index":
                outer,

            "outer_subject":
                subject_names[
                    outer
                ],

            "inner_subject_count":
                len(
                    inner_subjects
                ),

            "inner_subject_indices":
                json.dumps(
                    inner_subjects
                ),

            "inner_subjects":
                json.dumps(
                    [
                        subject_names[i]
                        for i
                        in inner_subjects
                    ]
                ),

            **winner,
        })

        selection_trace.append({
            "outer_subject_index":
                outer,

            "outer_subject":
                subject_names[
                    outer
                ],

            "inner_subject_indices":
                inner_subjects,

            "inner_subjects":
                [
                    subject_names[i]
                    for i
                    in inner_subjects
                ],

            "tie_break": [
                "higher_F1",
                "higher_precision",
                "fewer_FP",
                "fixed_grid_order",
            ],

            "winner_config_id":
                int(
                    winner[
                        "config_id"
                    ]
                ),

            "winner_rank":
                int(
                    winner["rank"]
                ),

            "candidates":
                ranked,
        })

        print(
            "outer",
            f"{outer + 1:02d}/94",
            "subject",
            subject_names[
                outer
            ],
            "winner config",
            winner[
                "config_id"
            ],
            "F1",
            f'{winner["f1"]:.6f}',
        )

    # --------------------------------------------------------
    # Persist selection evidence BEFORE outer test decoding
    # --------------------------------------------------------

    search_fields = [
        "outer_subject_index",
        "outer_subject",
        "config_id",
        "reference_scale",
        "local_radius",
        "threshold",
        "grid_order",
        "pooled_inner_tp",
        "pooled_inner_fp",
        "pooled_inner_fn",
        "precision",
        "recall",
        "f1",
        "rank",
        "final_rank",
        "selected",
    ]

    write_csv(
        out
        / "glsd_search_all_90x94.csv",
        search_rows,
        search_fields,
    )

    selected_fields = [
        "outer_subject_index",
        "outer_subject",
        "inner_subject_count",
        "inner_subject_indices",
        "inner_subjects",
        "config_id",
        "reference_scale",
        "local_radius",
        "threshold",
        "grid_order",
        "pooled_inner_tp",
        "pooled_inner_fp",
        "pooled_inner_fn",
        "precision",
        "recall",
        "f1",
        "rank",
    ]

    write_csv(
        out
        / "outer_selected_configs.csv",
        selected_rows,
        selected_fields,
    )

    (
        out
        / "outer_selected_configs.json"
    ).write_text(
        json.dumps(
            normalize(
                selected_rows
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        out
        / "outer_selection_trace.json"
    ).write_text(
        json.dumps(
            normalize(
                selection_trace
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Decode the 94 held-out outer subjects
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL OUTER-TEST GLSD DECODING")
    print("=" * 70)

    per_subject_rows = []
    per_video_rows = []
    per_prediction_rows = []
    final_video_records = []

    outer_raw_counts = []
    outer_full_counts = []

    for outer in range(
        EXPECTED_SUBJECTS
    ):

        config_item = (
            selected_by_outer[
                outer
            ]
        )

        decoded = decode_subject(
            outer,
            config_item,
            grouped,
            feature_bank,
            final_samples,
            final_emotions,
            config_cache,
            need_recognition=True,
        )

        raw = tuple(
            int(x)
            for x in decoded[
                "raw_counts"
            ]
        )

        full = tuple(
            int(x)
            for x in decoded[
                "full_counts"
            ]
        )

        (
            prediction_rows,
            video_rows,
            video_records,
        ) = subject_prediction_evidence(
            outer,
            subject_names[
                outer
            ],
            config_item,
            decoded,
            grouped,
            final_samples,
            config_cache,
        )

        # independent sum from video evidence
        video_raw = (
            sum(
                x["raw_tp"]
                for x in video_rows
            ),

            sum(
                x["raw_fp"]
                for x in video_rows
            ),

            sum(
                x["raw_fn"]
                for x in video_rows
            ),
        )

        video_full = (
            sum(
                x["full_tp"]
                for x in video_rows
            ),

            sum(
                x["full_fp"]
                for x in video_rows
            ),

            sum(
                x["full_fn"]
                for x in video_rows
            ),
        )

        assert video_raw == raw
        assert video_full == full

        mr = metrics(raw)
        mf = metrics(full)

        per_subject_rows.append({
            "outer_subject_index":
                outer,

            "outer_subject":
                subject_names[
                    outer
                ],

            "config_id":
                int(
                    config_item[
                        "config_id"
                    ]
                ),

            "reference_scale":
                config_item[
                    "reference_scale"
                ],

            "local_radius":
                config_item[
                    "local_radius"
                ],

            "threshold":
                config_item[
                    "threshold"
                ],

            "raw_tp":
                mr["tp"],

            "raw_fp":
                mr["fp"],

            "raw_fn":
                mr["fn"],

            "raw_precision":
                mr[
                    "precision"
                ],

            "raw_recall":
                mr["recall"],

            "raw_f1":
                mr["f1"],

            "full_tp":
                mf["tp"],

            "full_fp":
                mf["fp"],

            "full_fn":
                mf["fn"],

            "full_precision":
                mf[
                    "precision"
                ],

            "full_recall":
                mf["recall"],

            "full_f1":
                mf["f1"],
        })

        per_video_rows.extend(
            video_rows
        )

        per_prediction_rows.extend(
            prediction_rows
        )

        final_video_records.extend(
            video_records
        )

        outer_raw_counts.append(
            raw
        )

        outer_full_counts.append(
            full
        )

        print(
            "outer",
            f"{outer + 1:02d}/94",
            subject_names[
                outer
            ],
            "raw",
            raw,
            "full",
            full,
        )

    final_raw = add_counts(
        outer_raw_counts
    )

    final_full = add_counts(
        outer_full_counts
    )

    raw_metrics = metrics(
        final_raw
    )

    full_metrics = metrics(
        final_full
    )

    # --------------------------------------------------------
    # Persist final evidence
    # --------------------------------------------------------

    subject_fields = [
        "outer_subject_index",
        "outer_subject",
        "config_id",
        "reference_scale",
        "local_radius",
        "threshold",
        "raw_tp",
        "raw_fp",
        "raw_fn",
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "full_tp",
        "full_fp",
        "full_fn",
        "full_precision",
        "full_recall",
        "full_f1",
    ]

    write_csv(
        out
        / "per_subject_counts.csv",
        per_subject_rows,
        subject_fields,
    )

    video_fields = [
        "subject_index",
        "subject",
        "video_index_within_subject",
        "video_index_global",
        "video",
        "config_id",
        "gt_count",
        "prediction_count",
        "raw_tp",
        "raw_fp",
        "raw_fn",
        "neutral_tp",
        "neutral_fp",
        "full_tp",
        "full_fp",
        "full_fn",
    ]

    write_csv(
        out
        / "per_video_evaluation.csv",
        per_video_rows,
        video_fields,
    )

    prediction_fields = [
        "subject_index",
        "subject",
        "video_index_within_subject",
        "video_index_global",
        "video",
        "config_id",
        "prediction_index",
        "onset",
        "offset",
        "peak",
        "matched_gt_index",
        "matched_gt",
        "best_iou_inclusive",
        "matched_iou_inclusive",
        "raw_status",
        "recognition_prediction",
        "recognition_target",
        "neutral",
        "synergy_action",
    ]

    write_csv(
        out
        / "per_prediction_evaluation.csv",
        per_prediction_rows,
        prediction_fields,
    )

    with (
        out
        / "glsd_final_predictions.pkl"
    ).open(
        "wb"
    ) as f:

        pickle.dump(
            final_video_records,
            f,
            protocol=4,
        )

    auditable_payload = {
        "dataset":
            DATASET,

        "subject_count":
            EXPECTED_SUBJECTS,

        "video_count":
            EXPECTED_VIDEOS,

        "gt_event_count":
            EXPECTED_GT,

        "grid":
            grid,

        "selected_configs":
            selected_rows,

        "per_subject_counts":
            per_subject_rows,

        "per_video_evaluation":
            per_video_rows,

        "per_prediction_evaluation":
            per_prediction_rows,

        "final_raw_counts":
            final_raw,

        "final_full_counts":
            final_full,

        "videos":
            final_video_records,
    }

    with (
        out
        / "glsd_final_predictions_auditable.pkl"
    ).open(
        "wb"
    ) as f:

        pickle.dump(
            auditable_payload,
            f,
            protocol=4,
        )

    # --------------------------------------------------------
    # Manifest final scientific result
    # --------------------------------------------------------

    native_raw_m = metrics(
        EXPECTED_NATIVE_RAW
    )

    native_full_m = metrics(
        EXPECTED_NATIVE_FULL
    )

    delta_raw = (
        raw_metrics["f1"]
        - native_raw_m["f1"]
    )

    delta_full = (
        full_metrics["f1"]
        - native_full_m["f1"]
    )

    manifest.update({
        "execution_finished_at_utc":
            utc_now(),

        "native_exact_replay":
            normalize(
                native
            ),

        "glsd_final_raw_counts":
            list(
                final_raw
            ),

        "glsd_final_raw_metrics":
            raw_metrics,

        "glsd_final_full_counts":
            list(
                final_full
            ),

        "glsd_final_full_metrics":
            full_metrics,

        "delta_raw_vs_native":
            delta_raw,

        "delta_full_vs_native":
            delta_full,

        "glsd_full_minus_published_0_0997":
            (
                full_metrics["f1"]
                - 0.0997
            ),
    })

    manifest_path = (
        out
        / "run_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            normalize(
                manifest
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Main audit Markdown
    # --------------------------------------------------------

    audit = f"""# BoostingVRME / CAS(ME)3 GLSD Official-Response Full-Pipeline Audit

## Protocol identity

- Official repository commit: `{git_commit}`
- Cache SHA-256: `{manifest["cache_sha256"]}`
- Script SHA-256: `{manifest["script_sha256"]}`
- Subjects: {EXPECTED_SUBJECTS}
- Videos: {EXPECTED_VIDEOS}
- GT events in this official-response reconstruction: {EXPECTED_GT}
- GLSD configurations: 90
- Outer folds: 94
- Inner validation subjects per outer fold: 93

## Important protocol note

This run contains 853 GT events. It is not relabeled as the separate
858-event controlled CAS(ME)3 cache used by earlier decoder experiments.

## Native exact replay hard gate

- Raw: `{EXPECTED_NATIVE_RAW}`, F1 = `{native_raw_m["f1"]:.10f}`
- Full: `{EXPECTED_NATIVE_FULL}`, F1 = `{native_full_m["f1"]:.10f}`
- Spot predictions exact: `{native["prediction_exact"]}`
- Recognition exact: `{native["recognition_exact"]}`
- Matched-GT exact: `{native["matched_gt_exact"]}`
- Native exact replay: `PASS`

## GLSD-90

- Reference scales: `{REFERENCE_SCALES}`
- Local radii: `{LOCAL_RADII}`
- Thresholds: `{THRESHOLDS}`
- Selection: pooled inner Raw Spotting F1
- Tie-break: F1 -> precision -> fewer FP -> fixed grid order
- Outer subject excluded from its own inner pool: enforced by assertion

## GLSD final outer-test result

- Raw TP/FP/FN: `{final_raw}`
- Raw Precision: `{raw_metrics["precision"]:.10f}`
- Raw Recall: `{raw_metrics["recall"]:.10f}`
- Raw F1: `{raw_metrics["f1"]:.10f}`

- Full TP/FP/FN: `{final_full}`
- Full Precision: `{full_metrics["precision"]:.10f}`
- Full Recall: `{full_metrics["recall"]:.10f}`
- Full F1: `{full_metrics["f1"]:.10f}`

- Delta Raw vs Native: `{delta_raw:+.10f}`
- Delta Full vs Native: `{delta_full:+.10f}`
- GLSD Full minus published-context 0.0997: `{full_metrics["f1"] - 0.0997:+.10f}`

## Persisted evidence

- `glsd_search_all_90x94.csv`
- `glsd_search_inner_fold_counts.csv`
- `outer_selected_configs.csv`
- `outer_selected_configs.json`
- `outer_selection_trace.json`
- `per_subject_counts.csv`
- `per_video_evaluation.csv`
- `per_prediction_evaluation.csv`
- `glsd_final_predictions.pkl`
- `glsd_final_predictions_auditable.pkl`
- `run_manifest.json`
- `run_stdout_stderr.log`

## Runtime verdict

`BOOSTING_CASME3_GLSD_FULL_VALIDATED = PASS`

Note: stdout/stderr SHA-256 is finalized after process exit.
"""

    (
        out
        / "BOOSTING_CASME3_GLSD_FULL_AUDIT.md"
    ).write_text(
        audit,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("CASME3 GLSD OFFICIAL-RESPONSE")
    print("=" * 70)

    print(
        "Native Raw :",
        EXPECTED_NATIVE_RAW,
        "F1 =",
        f'{native_raw_m["f1"]:.10f}',
    )

    print(
        "GLSD Raw   :",
        final_raw,
        "F1 =",
        f'{raw_metrics["f1"]:.10f}',
    )

    print(
        "Native Full:",
        EXPECTED_NATIVE_FULL,
        "F1 =",
        f'{native_full_m["f1"]:.10f}',
    )

    print(
        "GLSD Full  :",
        final_full,
        "F1 =",
        f'{full_metrics["f1"]:.10f}',
    )

    print(
        "Delta Raw  =",
        f"{delta_raw:+.10f}",
    )

    print(
        "Delta Full =",
        f"{delta_full:+.10f}",
    )

    print(
        "GLSD Full - published 0.0997 =",
        f"{full_metrics['f1'] - 0.0997:+.10f}",
    )

    print()
    print(
        "BOOSTING_CASME3_GLSD_FULL_VALIDATED = PASS"
    )


if __name__ == "__main__":
    main()
