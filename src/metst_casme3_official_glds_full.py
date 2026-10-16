
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import pickle
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# Locked experiment paths
# ============================================================

from release_config import paths, ROOT as RELEASE_ROOT
_INPUTS = paths("metst_casme3")
ROOT = _INPUTS["upstream"]
LOCKED_GLSD_PATH = _INPUTS["core"]
DUMP_DIR = _INPUTS["responses"]
OUTPUT_DIR = RELEASE_ROOT / "outputs/metst_casme3"
PREREG_PATH = _INPUTS["preregistration"]

EXPECTED_GLSD_SHA256 = (
    "6f500e5af179a28e8a4f5e49a926004c1a60887a3cc27b1e9df2b9d61c94ca3d"
)

EXPECTED_NATIVE_RAW = None
EXPECTED_NATIVE_FULL = None

DATASET_NAME = "CASME_3"
METHOD_TYPE = 1
STRATEGY = 1
NATIVE_P = 0.55


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def combined_dump_sha256(paths):
    h = hashlib.sha256()

    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")

        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)

    return h.hexdigest()


def f1(counts):
    tp, fp, fn = map(float, counts)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def precision(counts):
    tp, fp, _ = map(float, counts)
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def recall(counts):
    tp, _, fn = map(float, counts)
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def metrics(counts):
    tp, fp, fn = map(int, counts)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision(counts),
        "recall": recall(counts),
        "F1": f1(counts),
    }


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Load EXACT locked GLSD core
# ============================================================

def load_locked_glsd():
    if not LOCKED_GLSD_PATH.exists():
        raise RuntimeError(
            f"locked GLSD source missing: {LOCKED_GLSD_PATH}"
        )

    actual_hash = sha256_file(
        LOCKED_GLSD_PATH
    )

    if actual_hash != EXPECTED_GLSD_SHA256:
        raise RuntimeError(
            "locked GLSD source hash mismatch\n"
            f"expected={EXPECTED_GLSD_SHA256}\n"
            f"actual  ={actual_hash}"
        )

    spec = importlib.util.spec_from_file_location(
        "boosting_glsd_locked",
        LOCKED_GLSD_PATH,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "GLSDFeatures",
        "configuration_grid",
        "selection_order",
    ]

    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(
                f"locked GLSD missing: {name}"
            )

    grid = module.configuration_grid()

    if len(grid) != 90:
        raise RuntimeError(
            f"GLSD grid changed: {len(grid)}"
        )

    print(
        "LOCKED_GLSD_SHA256 =",
        actual_hash
    )

    print(
        "LOCKED_GLSD_GRID =",
        len(grid)
    )

    print(
        "LOCKED_GLSD_CORE_PASS"
    )

    return module


# ============================================================
# Load ME-TST+ official functions
# ============================================================

def load_official_metst():
    from Utils.mean_average_precision_str.mean_average_precision import (
        MeanAveragePrecision2d,
    )

    import training_utils as official

    spotting_params = list(
        inspect.signature(
            official.spotting
        ).parameters
    )

    expected_spotting = [
        "final_samples",
        "subject_count",
        "pred_interval",
        "total_gt_spot",
        "p",
        "metric_final",
        "k_p",
    ]

    if spotting_params != expected_spotting:
        raise RuntimeError(
            "unexpected ME-TST spotting() signature\n"
            f"actual={spotting_params}"
        )

    recognition_params = list(
        inspect.signature(
            official.recognition
        ).parameters
    )

    expected_recognition = [
        "result",
        "preds",
        "metric_video",
        "final_emotions",
        "subject_count",
        "pred_list",
        "gt_tp_list",
        "final_samples",
        "pred_window_list",
        "pred_single_list",
        "frame_skip",
    ]

    if recognition_params != expected_recognition:
        raise RuntimeError(
            "unexpected ME-TST recognition() signature\n"
            f"actual={recognition_params}"
        )

    if not hasattr(official, "find_peaks"):
        raise RuntimeError(
            "training_utils.find_peaks not found"
        )

    print("METST_OFFICIAL_INTERFACE_PASS")

    return MeanAveragePrecision2d, official


# ============================================================
# Load the 29 official frozen dumps
# ============================================================

def load_records():
    paths = sorted(
        DUMP_DIR.glob("subject_*.pkl")
    )

    if len(paths) != 94:
        raise RuntimeError(
            f"expected 94 dumps, got {len(paths)}"
        )

    records = []

    for path in paths:
        with path.open("rb") as f:
            d = pickle.load(f)

        records.append(d)

    records.sort(
        key=lambda d: int(
            d["subject_count"]
        )
    )

    for i, d in enumerate(records):
        if int(d["subject_count"]) != i:
            raise RuntimeError(
                f"subject order broken at {i}"
            )

        if d["dataset_name"] != DATASET_NAME:
            raise RuntimeError(
                "dataset mismatch"
            )

        if int(d["method_type"]) != METHOD_TYPE:
            raise RuntimeError(
                "method_type mismatch"
            )

        if int(d["strategy"]) != STRATEGY:
            raise RuntimeError(
                "strategy mismatch"
            )

        n = len(d["result_all"])

        if not (
            n
            == len(d["result1_all"])
            == len(d["samples"])
            == len(d["emotions"])
            == len(d["videos"])
        ):
            raise RuntimeError(
                f"video alignment failure "
                f"for subject {d['subject_id']}"
            )

    kps = {
        int(d["k_p"])
        for d in records
    }

    frameskips = {
        int(d["frame_skip"])
        for d in records
    }

    if len(kps) != 1:
        raise RuntimeError(
            f"k_p not constant: {kps}"
        )

    if len(frameskips) != 1:
        raise RuntimeError(
            f"frame_skip not constant: "
            f"{frameskips}"
        )

    print(
        "OFFICIAL_DUMPS =",
        len(records)
    )

    print(
        "k_p =",
        next(iter(kps))
    )

    print(
        "DUMP_SCHEMA_PASS"
    )

    return paths, records


# ============================================================
# Shared metadata
# ============================================================

def final_samples(records):
    return [
        d["samples"]
        for d in records
    ]


def final_emotions(records):
    return [
        d["emotions"]
        for d in records
    ]


def subject_names(records):
    return [
        str(d["subject_id"])
        for d in records
    ]


# ============================================================
# Official Result Synergy
# ============================================================

def full_counts_from_official_synergy(
    raw_counts,
    pred_list,
    gt_tp_list,
    emotion_type=5,
):
    neutral = (
        int(emotion_type) - 1
    )

    tp_neutral = sum(
        prediction == neutral
        and target != -1
        for prediction, target
        in zip(
            pred_list,
            gt_tp_list,
        )
    )

    fp_neutral = sum(
        prediction == neutral
        and target == -1
        for prediction, target
        in zip(
            pred_list,
            gt_tp_list,
        )
    )

    tp, fp, fn = raw_counts

    return (
        int(tp - tp_neutral),
        int(fp - fp_neutral),
        int(fn + tp_neutral),
    )


# ============================================================
# Quiet official sequence evaluator
# ============================================================

def sequence_counts_quiet(
    official,
    total_gt,
    metric,
):
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        counts = (
            official.sequence_evaluation(
                total_gt,
                metric,
            )
        )

    return tuple(
        int(x)
        for x in counts
    )


# ============================================================
# Native exact replay — gate before GLSD
# ============================================================

def decode_native_subject(
    records,
    subject_index,
    MeanAveragePrecision2d,
    official,
):
    d = records[subject_index]

    samples = final_samples(records)
    emotions = final_emotions(records)

    metric_final = (
        MeanAveragePrecision2d(
            num_classes=1
        )
    )

    total_gt = 0

    preds, _, total_gt, metric_video, metric_final = (
        official.spotting(
            samples,
            subject_index,
            d["result_all"],
            total_gt,
            NATIVE_P,
            metric_final,
            int(d["k_p"]),
        )
    )

    raw_counts = sequence_counts_quiet(
        official,
        total_gt,
        metric_final,
    )

    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        (
            pred_list,
            _,
            gt_tp_list,
            _,
            _,
        ) = official.recognition(
            d["result1_all"],
            preds,
            metric_video,
            emotions,
            subject_index,
            [],
            [],
            samples,
            [],
            [],
            int(d["frame_skip"]),
        )

    full_counts = (
        full_counts_from_official_synergy(
            raw_counts,
            pred_list,
            gt_tp_list,
        )
    )

    return (
        raw_counts,
        full_counts,
        preds,
        pred_list,
        gt_tp_list,
        metric_video,
    )


def native_gate(
    records,
    MeanAveragePrecision2d,
    official,
):
    raw_total = np.zeros(
        3,
        dtype=np.int64,
    )

    full_total = np.zeros(
        3,
        dtype=np.int64,
    )

    for i in range(
        len(records)
    ):
        (
            raw,
            full,
            _,
            _,
            _,
            _,
        ) = decode_native_subject(
            records,
            i,
            MeanAveragePrecision2d,
            official,
        )

        raw_total += np.asarray(
            raw,
            dtype=np.int64,
        )

        full_total += np.asarray(
            full,
            dtype=np.int64,
        )

    raw_total = tuple(
        map(int, raw_total)
    )

    full_total = tuple(
        map(int, full_total)
    )

    print(
        "Native raw :",
        raw_total,
        "F1=%.10f" % f1(raw_total),
    )

    print(
        "Native full:",
        full_total,
        "F1=%.10f" % f1(full_total),
    )

    if EXPECTED_NATIVE_RAW is not None and raw_total != EXPECTED_NATIVE_RAW:
        raise RuntimeError(
            "native RAW gate failed: "
            f"{raw_total}"
        )

    if EXPECTED_NATIVE_FULL is not None and full_total != EXPECTED_NATIVE_FULL:
        raise RuntimeError(
            "native FULL gate failed: "
            f"{full_total}"
        )

    print(
        "METST_NATIVE_EXACT_REPLAY = PASS"
    )

    return {
        "raw_counts": raw_total,
        "full_counts": full_total,
    }


# ============================================================
# Candidate-source substitution
#
# Only find_peaks() is replaced.
# Official ME-TST spotting() still owns:
# - interval geometry peak-k_p / peak+k_p
# - GT registration
# - official IoU evaluator
# ============================================================

@contextlib.contextmanager
def metst_glsd_candidate_source(
    official,
    selected_by_video,
):
    original_find_peaks = (
        official.find_peaks
    )

    position = 0

    def glsd_find_peaks(
        signal,
        *args,
        **kwargs,
    ):
        nonlocal position

        if position >= len(
            selected_by_video
        ):
            raise RuntimeError(
                "ME-TST spotting requested "
                "too many candidate lists"
            )

        peaks = np.asarray(
            selected_by_video[position],
            dtype=int,
        )

        position += 1

        if (
            np.any(peaks < 0)
            or np.any(
                peaks >= len(signal)
            )
        ):
            raise RuntimeError(
                "GLSD candidate outside "
                "official response"
            )

        return peaks, {}

    official.find_peaks = (
        glsd_find_peaks
    )

    try:
        yield

        if position != len(
            selected_by_video
        ):
            raise RuntimeError(
                "ME-TST spotting did not "
                "consume every GLSD list"
            )

    finally:
        official.find_peaks = (
            original_find_peaks
        )


# ============================================================
# ME-TST+ adapter for one subject
# ============================================================

def decode_glsd_subject(
    records,
    subject_index,
    glsd_config,
    glsd_core,
    MeanAveragePrecision2d,
    official,
    with_recognition,
):
    d = records[subject_index]

    samples = final_samples(records)
    emotions = final_emotions(records)

    responses = [
        np.asarray(
            x,
            dtype=float,
        )
        for x in d["result_all"]
    ]

    k_p = int(
        d["k_p"]
    )

    candidates = [
        glsd_core.GLSDFeatures(
            response,
            k_p,
        ).selected_peaks(
            glsd_config
        )
        for response in responses
    ]

    metric_final = (
        MeanAveragePrecision2d(
            num_classes=1
        )
    )

    total_gt = 0

    with metst_glsd_candidate_source(
        official,
        candidates,
    ):
        (
            predictions,
            _,
            total_gt,
            metric_video,
            metric_final,
        ) = official.spotting(
            samples,
            subject_index,
            responses,
            total_gt,
            NATIVE_P,
            metric_final,
            k_p,
        )

    raw_counts = (
        sequence_counts_quiet(
            official,
            total_gt,
            metric_final,
        )
    )

    match_map = dict(
        sorted(
            metric_video.value(
                iou_thresholds=0.5
            )[0.5][0][
                "pred_match_gt"
            ].items()
        )
    )

    if not with_recognition:
        return (
            raw_counts,
            predictions,
            None,
            None,
            match_map,
            metric_video,
        )

    sequences = [
        np.asarray(x)
        for x in d["result1_all"]
    ]

    buf = io.StringIO()

    with contextlib.redirect_stdout(
        buf
    ):
        (
            pred_list,
            _,
            gt_tp_list,
            _,
            _,
        ) = official.recognition(
            sequences,
            predictions,
            metric_video,
            emotions,
            subject_index,
            [],
            [],
            samples,
            [],
            [],
            int(d["frame_skip"]),
        )

    return (
        raw_counts,
        predictions,
        pred_list,
        gt_tp_list,
        match_map,
        metric_video,
    )


# ============================================================
# Per-video audit
# ============================================================

def per_video_evidence(
    record,
    predictions,
    match_map,
    pred_list,
    gt_tp_list,
):
    rows = []
    evidence = []

    recognition_offset = 0

    for video_index, preds in enumerate(
        predictions
    ):
        preds = list(preds)

        matches = list(
            match_map.get(
                video_index,
                [],
            )
        )

        if len(matches) != len(preds):
            raise RuntimeError(
                "prediction/match length "
                f"mismatch video={video_index}"
            )

        n_pred = len(preds)

        video_pred_labels = (
            pred_list[
                recognition_offset:
                recognition_offset+n_pred
            ]
        )

        video_gt_labels = (
            gt_tp_list[
                recognition_offset:
                recognition_offset+n_pred
            ]
        )

        recognition_offset += n_pred

        if not (
            len(video_pred_labels)
            == len(video_gt_labels)
            == n_pred
        ):
            raise RuntimeError(
                "recognition alignment failure"
            )

        for match, target in zip(
            matches,
            video_gt_labels,
        ):
            if (
                (int(match) != -1)
                != (target != -1)
            ):
                raise RuntimeError(
                    "IoU match / recognition "
                    "target mismatch"
                )

        raw_tp = sum(
            int(x) != -1
            for x in matches
        )

        raw_fp = n_pred - raw_tp

        gt_n = len(
            record["samples"][
                video_index
            ]
        )

        raw_fn = gt_n - raw_tp

        neutral = 4

        neutral_tp = sum(
            pred == neutral
            and target != -1
            for pred, target
            in zip(
                video_pred_labels,
                video_gt_labels,
            )
        )

        neutral_fp = sum(
            pred == neutral
            and target == -1
            for pred, target
            in zip(
                video_pred_labels,
                video_gt_labels,
            )
        )

        final_tp = (
            raw_tp - neutral_tp
        )

        final_fp = (
            raw_fp - neutral_fp
        )

        final_fn = (
            raw_fn + neutral_tp
        )

        rows.append({
            "subject_id":
                str(record["subject_id"]),

            "video_index":
                video_index,

            "video_name":
                str(
                    record["videos"][
                        video_index
                    ]
                ),

            "raw_TP_contribution":
                int(raw_tp),

            "raw_FP_contribution":
                int(raw_fp),

            "raw_FN_contribution":
                int(raw_fn),

            "final_TP_contribution":
                int(final_tp),

            "final_FP_contribution":
                int(final_fp),

            "final_FN_contribution":
                int(final_fn),

            "predictions":
                int(n_pred),

            "neutral_TP_removed":
                int(neutral_tp),

            "neutral_FP_removed":
                int(neutral_fp),
        })

        candidate_rows = []

        for pred_index, pred in enumerate(
            preds
        ):
            candidate_rows.append({
                "prediction_index":
                    pred_index,

                "interval":
                    np.asarray(
                        pred
                    ).tolist(),

                "matched_gt_index":
                    int(
                        matches[
                            pred_index
                        ]
                    ),

                "recognition_prediction":
                    int(
                        video_pred_labels[
                            pred_index
                        ]
                    ),

                "recognition_target":
                    int(
                        video_gt_labels[
                            pred_index
                        ]
                    ),

                "neutral_removed":
                    bool(
                        video_pred_labels[
                            pred_index
                        ]
                        == neutral
                    ),
            })

        evidence.append({
            "subject_id":
                str(record["subject_id"]),

            "video_index":
                video_index,

            "video_name":
                str(
                    record["videos"][
                        video_index
                    ]
                ),

            "evidence":
                candidate_rows,
        })

    if recognition_offset != len(
        pred_list
    ):
        raise RuntimeError(
            "unused recognition predictions"
        )

    return rows, evidence


# ============================================================
# Full locked GLSD-90 experiment
# ============================================================

def run_full(
    records,
    glsd_core,
    MeanAveragePrecision2d,
    official,
    native,
    dump_paths,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    subjects = (
        subject_names(records)
    )

    grid = (
        glsd_core.configuration_grid()
    )

    if len(grid) != 90:
        raise RuntimeError(
            "grid is not 90"
        )

    # --------------------------------------------------------
    # 90 configs x 29 subjects
    # --------------------------------------------------------

    raw_table = np.zeros(
        (
            len(grid),
            len(subjects),
            3,
        ),
        dtype=np.int64,
    )

    print()
    print(
        "Starting locked "
        "90 x 94 GLSD search..."
    )

    for subject_index, subject in enumerate(
        subjects
    ):
        print(
            f"[SEARCH] "
            f"{subject_index+1:02d}/"
            f"{len(subjects)} "
            f"subject={subject}"
        )

        for config_index, config in enumerate(
            grid
        ):
            (
                raw_counts,
                _,
                _,
                _,
                _,
                _,
            ) = decode_glsd_subject(
                records,
                subject_index,
                config,
                glsd_core,
                MeanAveragePrecision2d,
                official,
                False,
            )

            raw_table[
                config_index,
                subject_index,
            ] = raw_counts

    # --------------------------------------------------------
    # Nested outer LOSO configuration selection
    # --------------------------------------------------------

    choices = []

    selection_rows = []
    search_rows = []
    inner_fold_rows = []
    selection_traces = []

    for outer_index, outer_subject in enumerate(
        subjects
    ):
        inner_indexes = [
            i
            for i in range(
                len(subjects)
            )
            if i != outer_index
        ]

        inner_subjects = [
            subjects[i]
            for i in inner_indexes
        ]

        if outer_subject in inner_subjects:
            raise RuntimeError(
                "OUTER SUBJECT LEAKAGE"
            )

        pooled_inner = (
            raw_table[
                :,
                inner_indexes,
                :,
            ].sum(
                axis=1
            )
        )

        order = (
            glsd_core.selection_order(
                pooled_inner
            )
        )

        selected = int(
            order[0]
        )

        choices.append(
            selected
        )

        ranks = np.empty(
            len(grid),
            dtype=int,
        )

        ranks[order] = np.arange(
            1,
            len(grid)+1,
        )

        selected_metrics = metrics(
            pooled_inner[
                selected
            ]
        )

        selected_config = (
            grid[selected]
        )

        config_identifier = (
            getattr(
                selected_config,
                "identifier",
                (
                    f"reference_scale="
                    f"{selected_config.reference_scale}"
                    f"|local_radius="
                    f"{selected_config.local_radius}"
                    f"|threshold="
                    f"{selected_config.threshold}"
                ),
            )
        )

        selection_rows.append({
            "outer_subject":
                outer_subject,

            "config_id":
                selected,

            "config":
                config_identifier,

            **asdict(
                selected_config
            ),

            "inner_subject_count":
                len(inner_subjects),

            "inner_subjects":
                inner_subjects,

            "inner_TP":
                selected_metrics["TP"],

            "inner_FP":
                selected_metrics["FP"],

            "inner_FN":
                selected_metrics["FN"],

            "inner_precision":
                selected_metrics[
                    "precision"
                ],

            "inner_recall":
                selected_metrics[
                    "recall"
                ],

            "inner_F1":
                selected_metrics["F1"],
        })

        trace_candidates = []

        for config_index, config in enumerate(
            grid
        ):
            m = metrics(
                pooled_inner[
                    config_index
                ]
            )

            selected_flag = (
                config_index
                == selected
            )

            row = {
                "outer_subject":
                    outer_subject,

                "config_id":
                    config_index,

                **asdict(config),

                "pooled_inner_TP":
                    m["TP"],

                "pooled_inner_FP":
                    m["FP"],

                "pooled_inner_FN":
                    m["FN"],

                "precision":
                    m["precision"],

                "recall":
                    m["recall"],

                "F1":
                    m["F1"],

                "grid_order":
                    config_index,

                "final_rank":
                    int(
                        ranks[
                            config_index
                        ]
                    ),

                "selected":
                    selected_flag,
            }

            search_rows.append(
                row
            )

            trace_candidates.append({
                **row,

                "selection_key": {
                    "higher_F1":
                        m["F1"],

                    "higher_precision":
                        m["precision"],

                    "fewer_FP":
                        m["FP"],

                    "fixed_grid_order":
                        config_index,
                },
            })

            for inner_index in (
                inner_indexes
            ):
                mi = metrics(
                    raw_table[
                        config_index,
                        inner_index,
                    ]
                )

                inner_fold_rows.append({
                    "outer_subject":
                        outer_subject,

                    "inner_validation_subject":
                        subjects[
                            inner_index
                        ],

                    "config_id":
                        config_index,

                    **asdict(config),

                    "TP": mi["TP"],
                    "FP": mi["FP"],
                    "FN": mi["FN"],
                })

        selection_traces.append({
            "outer_subject":
                outer_subject,

            "inner_subjects":
                inner_subjects,

            "tie_break_order": [
                "higher F1",
                "higher precision",
                "fewer FP",
                "fixed grid order",
            ],

            "winner_config_id":
                selected,

            "winner_rank":
                int(
                    ranks[selected]
                ),

            "candidates":
                trace_candidates,
        })

    print()
    print(
        "NESTED_LOSO_SELECTION_PASS"
    )

    # --------------------------------------------------------
    # Evaluate each outer subject using ONLY its inner-selected
    # configuration
    # --------------------------------------------------------

    glsd_raw = np.zeros(
        3,
        dtype=np.int64,
    )

    glsd_full = np.zeros(
        3,
        dtype=np.int64,
    )

    final_predictions = []
    auditable_predictions = []
    per_video_rows = []
    per_subject_rows = []

    for subject_index, record in enumerate(
        records
    ):
        selected = (
            choices[
                subject_index
            ]
        )

        config = (
            grid[selected]
        )

        (
            raw_counts,
            predictions,
            pred_list,
            gt_tp_list,
            match_map,
            metric_video,
        ) = decode_glsd_subject(
            records,
            subject_index,
            config,
            glsd_core,
            MeanAveragePrecision2d,
            official,
            True,
        )

        # Selected outer decode must exactly replay
        # the precomputed raw-table entry.
        if tuple(raw_counts) != tuple(
            raw_table[
                selected,
                subject_index,
            ]
        ):
            raise RuntimeError(
                "selected outer replay mismatch"
            )

        full_counts = (
            full_counts_from_official_synergy(
                raw_counts,
                pred_list,
                gt_tp_list,
            )
        )

        video_rows, video_evidence = (
            per_video_evidence(
                record,
                predictions,
                match_map,
                pred_list,
                gt_tp_list,
            )
        )

        evidence_raw = (
            sum(
                x[
                    "raw_TP_contribution"
                ]
                for x in video_rows
            ),
            sum(
                x[
                    "raw_FP_contribution"
                ]
                for x in video_rows
            ),
            sum(
                x[
                    "raw_FN_contribution"
                ]
                for x in video_rows
            ),
        )

        evidence_full = (
            sum(
                x[
                    "final_TP_contribution"
                ]
                for x in video_rows
            ),
            sum(
                x[
                    "final_FP_contribution"
                ]
                for x in video_rows
            ),
            sum(
                x[
                    "final_FN_contribution"
                ]
                for x in video_rows
            ),
        )

        if evidence_raw != tuple(
            raw_counts
        ):
            raise RuntimeError(
                "per-video RAW audit mismatch"
            )

        if evidence_full != tuple(
            full_counts
        ):
            raise RuntimeError(
                "per-video FULL audit mismatch"
            )

        glsd_raw += np.asarray(
            raw_counts,
            dtype=np.int64,
        )

        glsd_full += np.asarray(
            full_counts,
            dtype=np.int64,
        )

        rm = metrics(
            raw_counts
        )

        fm = metrics(
            full_counts
        )

        per_subject_rows.append({
            "outer_subject":
                subjects[
                    subject_index
                ],

            "config_id":
                selected,

            "raw_TP": rm["TP"],
            "raw_FP": rm["FP"],
            "raw_FN": rm["FN"],
            "raw_F1": rm["F1"],

            "full_TP": fm["TP"],
            "full_FP": fm["FP"],
            "full_FN": fm["FN"],
            "full_F1": fm["F1"],
        })

        identifier = getattr(
            config,
            "identifier",
            str(asdict(config)),
        )

        for row in video_rows:
            row[
                "config_id"
            ] = selected

            row[
                "selected_config"
            ] = identifier

        for evidence in (
            video_evidence
        ):
            evidence[
                "selected_config_id"
            ] = selected

            evidence[
                "selected_config"
            ] = asdict(config)

        per_video_rows.extend(
            video_rows
        )

        auditable_predictions.extend(
            video_evidence
        )

        for video_index, preds in enumerate(
            predictions
        ):
            final_predictions.append({
                "subject_id":
                    str(
                        record[
                            "subject_id"
                        ]
                    ),

                "subject_index":
                    subject_index,

                "video_index":
                    video_index,

                "video_name":
                    str(
                        record[
                            "videos"
                        ][video_index]
                    ),

                "selected_config_id":
                    selected,

                "selected_config":
                    asdict(config),

                "predictions":
                    np.asarray(
                        preds
                    ).tolist(),
            })

    raw_counts = tuple(
        map(
            int,
            glsd_raw,
        )
    )

    full_counts = tuple(
        map(
            int,
            glsd_full,
        )
    )

    # --------------------------------------------------------
    # Aggregate integrity gates
    # --------------------------------------------------------

    raw_from_subjects = tuple(
        sum(
            row[
                f"raw_{name}"
            ]
            for row
            in per_subject_rows
        )
        for name in (
            "TP",
            "FP",
            "FN",
        )
    )

    full_from_subjects = tuple(
        sum(
            row[
                f"full_{name}"
            ]
            for row
            in per_subject_rows
        )
        for name in (
            "TP",
            "FP",
            "FN",
        )
    )

    raw_from_videos = tuple(
        sum(
            row[
                f"raw_{name}"
                "_contribution"
            ]
            for row
            in per_video_rows
        )
        for name in (
            "TP",
            "FP",
            "FN",
        )
    )

    full_from_videos = tuple(
        sum(
            row[
                f"final_{name}"
                "_contribution"
            ]
            for row
            in per_video_rows
        )
        for name in (
            "TP",
            "FP",
            "FN",
        )
    )

    if not (
        raw_counts
        == raw_from_subjects
        == raw_from_videos
    ):
        raise RuntimeError(
            "RAW aggregate audit failed"
        )

    if not (
        full_counts
        == full_from_subjects
        == full_from_videos
    ):
        raise RuntimeError(
            "FULL aggregate audit failed"
        )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    csv_selection = []

    for row in selection_rows:
        r = dict(row)

        r[
            "inner_subjects"
        ] = ";".join(
            r["inner_subjects"]
        )

        csv_selection.append(r)

    write_json(
        OUTPUT_DIR
        / "outer_selected_configs.json",
        selection_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "outer_selected_configs.csv",
        csv_selection,
    )

    write_csv(
        OUTPUT_DIR
        / "glsd_search_all_90x29.csv",
        search_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "glsd_search_inner_fold_counts.csv",
        inner_fold_rows,
    )

    write_json(
        OUTPUT_DIR
        / "outer_selection_trace.json",
        selection_traces,
    )

    with (
        OUTPUT_DIR
        / "glsd_final_predictions.pkl"
    ).open(
        "wb"
    ) as f:
        pickle.dump(
            final_predictions,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with (
        OUTPUT_DIR
        / "glsd_final_predictions_auditable.pkl"
    ).open(
        "wb"
    ) as f:
        pickle.dump(
            auditable_predictions,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    write_csv(
        OUTPUT_DIR
        / "per_video_evaluation.csv",
        per_video_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_subject_counts.csv",
        per_subject_rows,
    )

    manifest = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "dataset":
            DATASET_NAME,

        "pipeline":
            "ME-TST+",

        "method_type":
            METHOD_TYPE,

        "strategy":
            STRATEGY,

        "subjects":
            len(records),

        "videos":
            sum(
                len(d["result_all"])
                for d in records
            ),

        "dump_combined_sha256":
            combined_dump_sha256(
                dump_paths
            ),

        "locked_glsd_source":
            str(
                LOCKED_GLSD_PATH
            ),

        "locked_glsd_sha256":
            sha256_file(
                LOCKED_GLSD_PATH
            ),

        "glsd_grid_size":
            len(grid),

        "selection":
            (
                "outer LOSO; pooled "
                "28 inner subjects"
            ),

        "tie_break": [
            "higher F1",
            "higher precision",
            "fewer FP",
            "fixed grid order",
        ],

        "native_raw_counts":
            list(
                native[
                    "raw_counts"
                ]
            ),

        "native_full_counts":
            list(
                native[
                    "full_counts"
                ]
            ),

        "glsd_raw_counts":
            list(raw_counts),

        "glsd_full_counts":
            list(full_counts),

        "python":
            platform.python_version(),

        "preregistration":
            str(PREREG_PATH),
    }

    write_json(
        OUTPUT_DIR
        / "run_manifest.json",
        manifest,
    )

    audit_lines = [
        "# ME-TST+ CAS(ME)3 GLSD Official-Response Audit",
        "",
        "## Native exact gate",
        "",
        f"- Native raw: `{native['raw_counts']}`; F1 = {f1(native['raw_counts']):.10f}.",
        f"- Native full: `{native['full_counts']}`; F1 = {f1(native['full_counts']):.10f}.",
        "- Native exact replay: `PASS`.",
        "",
        "## Locked GLSD",
        "",
        f"- Shared GLSD source SHA-256: `{sha256_file(LOCKED_GLSD_PATH)}`.",
        "- Grid: 3 reference scales × 3 local radii × 10 thresholds = 90.",
        "- Score: `S(c)=(G(c)+L(c))/2`.",
        "- Missing-scale support: zero.",
        "- Multi-scale aggregation: median.",
        "- Selection: outer LOSO with pooled 28-subject inner validation.",
        "- Tie-break: F1 → precision → fewer FP → fixed grid order.",
        "- No new tuning or post-hoc configuration search.",
        "",
        "## GLSD official-response result",
        "",
        f"- Raw: `{raw_counts}`; F1 = {f1(raw_counts):.10f}.",
        f"- Full: `{full_counts}`; F1 = {f1(full_counts):.10f}.",
        f"- Delta raw: `{f1(raw_counts)-f1(native['raw_counts']):.10f}`.",
        f"- Delta full: `{f1(full_counts)-f1(native['full_counts']):.10f}`.",
        "",
        "## Verdict",
        "",
        "`METST_CASME3_GLSD_FULL_VALIDATED = PASS`",
        "",
    ]

    (
        OUTPUT_DIR
        / "METST_CASME3_GLSD_FULL_AUDIT.md"
    ).write_text(
        "\n".join(
            audit_lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "ME-TST+ GLSD official-response"
    )

    print(
        "=" * 60
    )

    print(
        "Native raw : "
        "%d/%d/%d  F1=%.4f"
        % (
            *native["raw_counts"],
            f1(
                native[
                    "raw_counts"
                ]
            ),
        )
    )

    print(
        "Native full: "
        "%d/%d/%d  F1=%.4f"
        % (
            *native["full_counts"],
            f1(
                native[
                    "full_counts"
                ]
            ),
        )
    )

    print(
        "GLSD raw   : "
        "%d/%d/%d  F1=%.4f"
        % (
            *raw_counts,
            f1(raw_counts),
        )
    )

    print(
        "GLSD full  : "
        "%d/%d/%d  F1=%.4f"
        % (
            *full_counts,
            f1(full_counts),
        )
    )

    print(
        "Delta raw  = %.10f"
        % (
            f1(raw_counts)
            - f1(
                native[
                    "raw_counts"
                ]
            )
        )
    )

    print(
        "Delta full = %.10f"
        % (
            f1(full_counts)
            - f1(
                native[
                    "full_counts"
                ]
            )
        )
    )

    print(
        "OUTPUT_DIR =",
        OUTPUT_DIR,
    )

    print(
        "METST_CASME3_GLSD_FULL_VALIDATED = PASS"
    )


# ============================================================
# Main
# ============================================================

def main():
    if not PREREG_PATH.exists():
        raise RuntimeError(
            "preregistration file missing"
        )

    prereg = json.loads(
        PREREG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        prereg.get("glsd", {})
        .get("protocol")
        != "locked GLSD-90"
    ):
        raise RuntimeError(
            "preregistration protocol mismatch"
        )

    glsd_core = (
        load_locked_glsd()
    )

    (
        MeanAveragePrecision2d,
        official,
    ) = load_official_metst()

    dump_paths, records = (
        load_records()
    )

    native = native_gate(
        records,
        MeanAveragePrecision2d,
        official,
    )

    run_full(
        records,
        glsd_core,
        MeanAveragePrecision2d,
        official,
        native,
        dump_paths,
    )


if __name__ == "__main__":
    main()