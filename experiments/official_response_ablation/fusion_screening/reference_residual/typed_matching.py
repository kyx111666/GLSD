"""Audited aggregate adapter plus unmodified native video matching.

Changes are scoped to a context manager; sealed files are never edited.
"""
from contextlib import contextmanager
import hashlib
import inspect

import numpy as np
from one_to_one_evaluator import check_box_one_to_one

PROTOCOL = 'aggregate_v1_video_native_checked_v1'


def normalize_map(mapping):
    result = {}
    for video, items in mapping.items():
        ids = [np.asarray(item).reshape(-1).tolist() for item in items]
        if any(len(item) != 1 or int(item[0]) != item[0] or item[0] < -1 for item in ids):
            raise RuntimeError('invalid prediction-to-GT map')
        result[int(video)] = [int(item[0]) for item in ids]
    return result


@contextmanager
def install(metric_class):
    original = metric_class._evaluate_class
    scope = original.__globals__
    old_check = scope['check_box']
    sample = old_check(np.array([.8]), np.array([0]), np.array([0]), np.array([0]), [], .5)
    if sample[0] == 'tp':
        kind = 'string_scalar'
    elif sample[0] == 1:
        kind = 'numeric_list'
    else:
        raise RuntimeError('unsupported evaluator check_box return contract')
    source = inspect.getsource(old_check)
    info = dict(protocol=PROTOCOL, return_contract=kind,
                source=inspect.getsourcefile(metric_class),
                old_check_box_sha256=hashlib.sha256(source.encode()).hexdigest(),
                old_value_signature=str(inspect.signature(metric_class.value)),
                effective_policy='caller policy unchanged; aggregate adapter only; native video check_box retained')

    def checked(self, class_id, iou_threshold, recall_thresholds, mpolicy='greedy'):
        result = original(self, class_id, iou_threshold, recall_thresholds, mpolicy=mpolicy)
        tp = np.asarray(result[3])
        fp = np.asarray(result[4])
        mapping = normalize_map(result[-1])
        if sum(map(len, mapping.values())) != len(tp):
            raise RuntimeError('missing prediction match entries, including unmatched predictions')
        positive = 0
        for ids in mapping.values():
            used = [i for i in ids if i >= 0]
            if len(used) != len(set(used)):
                raise RuntimeError('GT reused in one video')
            positive += len(used)
        if not np.all((tp == 0) | (tp == 1)) or positive != int(tp.sum()):
            raise RuntimeError('TP/map mismatch')
        if tp.shape != fp.shape or not np.all(tp + fp == 1):
            raise RuntimeError('prediction missing TP/FP classification')
        if positive > int(self.class_counter[:, class_id].sum()):
            raise RuntimeError('TP exceeds registered GT')
        return result

    checked.return_contract = kind
    if kind == 'numeric_list':
        scope['check_box'] = check_box_one_to_one
    metric_class._evaluate_class = checked
    try:
        yield info
    finally:
        metric_class._evaluate_class = original
        scope['check_box'] = old_check


def synthetic_probe(classes):
    cases = [([(0, 9)], [(0, 9)], (1, 0)),
             ([(20, 29)], [(0, 9)], (0, 1)),
             ([(0, 9), (0, 9)], [(0, 9)], (1, 1)),
             ([(0, 19)], [(0, 9), (10, 19)], (1, 0)),
             ([(0, 19), (0, 19)], [(0, 9), (10, 19)], (2, 0)),
             ([], [(0, 9)], (0, 0)), ([(0, 9)], [], (0, 1))]
    rows = []
    for ci, cls in enumerate(classes):
        for policy in ('greedy', 'soft'):
            for case, (pred, gt, expected) in enumerate(cases):
                if case == 4 and policy == 'greedy':
                    expected = (1, 1)
                metric = cls(num_classes=1)
                p = np.array([[a, 0, b, 0, 0, 1, 0, 0] for a, b in pred], float).reshape(-1, 8)
                if cls._evaluate_class.return_contract == 'string_scalar':
                    p = p[:, :7]
                g = np.array([[a, 0, b, 0, 0, 0, 0, 0] for a, b in gt], float).reshape(-1, 8)
                metric.add(p, g)
                v = metric.value(iou_thresholds=.5, mpolicy=policy)[.5][0]
                actual = (int(sum(v['tp'])), int(sum(v['fp'])))
                if actual != expected:
                    raise RuntimeError('matching probe failed: %s != %s' % (actual, expected))
                rows.append(dict(class_index=ci, policy=policy, case=case,
                                 counts=actual, mapping=normalize_map(v['pred_match_gt'])))
    # Cross-class equality is expected only when the caller explicitly uses the same policy.
    reference = {(r['policy'], r['case']): r['mapping'] for r in rows if r['class_index'] == 0}
    if any(r['mapping'] != reference[r['policy'], r['case']] for r in rows):
        raise RuntimeError('aggregate/video assignment mismatch under identical policy')
    return rows
