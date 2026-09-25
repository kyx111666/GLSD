"""Versioned one-to-one event matching adapter; no source-file modification.

Uses the existing prediction order, IoU threshold and IoU ranking. For soft
mode, pick the first unused GT in that ranking; greedy mode considers only
the top GT, as before. Supports ordinary, non-difficult events. Crowd reuse
is intentionally disallowed for this one-to-one event evaluation protocol.
The nested GT-index format expected by recognition is preserved.
"""
from contextlib import contextmanager
import hashlib
import inspect

import numpy as np

PROTOCOL = "event_one_to_one_v1_prediction_order"


def check_box_one_to_one(iou, difficult, crowd, order, matched_ind,
                         iou_threshold, mpolicy="greedy"):
    if mpolicy not in ("greedy", "soft"):
        raise ValueError(f"Unsupported matching policy: {mpolicy}")
    if np.any(np.asarray(difficult) != 0):
        raise ValueError("This event protocol requires non-difficult GT; inspect annotations")
    used = set()
    for item in matched_ind:
        for value in np.asarray(item).reshape(-1):
            if int(value) >= 0:
                used.add(int(value))
    limit = 1 if mpolicy == "greedy" else len(order)
    for idx in order[:limit]:
        idx = int(idx)
        if iou[idx] < iou_threshold:
            break
        if idx not in used:
            return 1, [idx]
    return 0, []


@contextmanager
def install(metric_class):
    """Patch the actual function binding consumed by _evaluate_class, restore on exit."""
    original_evaluate = metric_class._evaluate_class
    scope = original_evaluate.__globals__
    original_check = scope["check_box"]
    source = inspect.getsource(original_check)
    info = dict(protocol=PROTOCOL, metric_class=metric_class.__name__,
                metric_source=inspect.getsourcefile(metric_class),
                legacy_check_box_source=source,
                legacy_check_box_sha256=hashlib.sha256(source.encode()).hexdigest(),
                prediction_order="unchanged evaluator input order; no confidence resorting",
                iou="unchanged caller threshold, normally >=0.5",
                assignment="at most one GT per prediction and one prediction per GT within each video",
                score_comparability="revised evaluation; do not pool with legacy official counts")

    def checked_evaluate(self, *args, **kwargs):
        result = original_evaluate(self, *args, **kwargs)
        tp = np.asarray(result[3])
        matches = result[-1]
        if not np.all((tp == 0) | (tp == 1)):
            raise RuntimeError("A prediction received more than one TP")
        positive_count = 0
        for video, items in matches.items():
            used = set()
            for ids in items:
                positive = [int(x) for x in np.asarray(ids).reshape(-1) if int(x) >= 0]
                if len(positive) > 1 or used.intersection(positive):
                    raise RuntimeError(f"Non-unique GT matching in video {video}")
                used.update(positive)
                positive_count += len(positive)
        if int(tp.sum()) != positive_count:
            raise RuntimeError("TP array and recognition match map disagree")
        class_id = args[0] if args else kwargs["class_id"]
        if int(tp.sum()) > int(self.class_counter[:, class_id].sum()):
            raise RuntimeError("TP exceeds the evaluator's registered GT count")
        return result

    scope["check_box"] = check_box_one_to_one
    metric_class._evaluate_class = checked_evaluate
    try:
        yield info
    finally:
        metric_class._evaluate_class = original_evaluate
        scope["check_box"] = original_check
