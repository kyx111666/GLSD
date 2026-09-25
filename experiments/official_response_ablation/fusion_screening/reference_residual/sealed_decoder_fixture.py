"""Test-only source excerpts from supplied GLSD+ME-TST notebook; synthetic inputs.
Production imports the Drive-sealed runner, never this fixture.
"""
import contextlib
import io
import numpy as np

NATIVE_P = 0.55

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

