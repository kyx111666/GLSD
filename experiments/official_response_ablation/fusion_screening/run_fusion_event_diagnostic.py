"""Replay selected S_beta configs and export official per-event assignments.

Capture helpers adapted from the existing run_p8_threshold_control.py.
No search, new configuration selection, backbone training, or evaluator changes.
"""
import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_full_fusion_tuning as tuning
import run_one_to_one_full_tuning as previous
import one_to_one_evaluator as evaluator
from structure_expansion import ExpandedCore

shared = tuning.shared

if not hasattr(pd.DataFrame, "append"):
    def dataframe_append(self, other, **kwargs):
        return pd.concat([self, other], **kwargs)
    pd.DataFrame.append = dataframe_append

@contextlib.contextmanager
def capture_registered_inputs(metric_class):
    """Observe public add calls without assuming evaluator-owned array buffers."""
    original_add = metric_class.add
    captured = {}

    def add(self, preds, gt, *args, **kwargs):
        pred_copy, gt_copy = np.array(preds, copy=True), np.array(gt, copy=True)
        result = original_add(self, preds, gt, *args, **kwargs)
        # Keep the instance alive so object IDs cannot be recycled during decode.
        entry = captured.setdefault(id(self), (self, []))
        entry[1].append((pred_copy, gt_copy))
        return result

    metric_class.add = add
    try:
        yield captured
    finally:
        metric_class.add = original_add


@contextlib.contextmanager
def capture_decoder_inputs(metric_class, official):
    """Observe both the supplied aggregate and spotting-owned video evaluators."""
    classes = [metric_class]
    if official is not None:
        video_class = official.spotting.__globals__.get('MeanAveragePrecision2d')
        if not isinstance(video_class, type) or not callable(getattr(video_class, 'add', None)):
            raise RuntimeError('cannot resolve official spotting video evaluator class')
        if video_class not in classes:
            classes.append(video_class)
    with contextlib.ExitStack() as stack:
        captures = [stack.enter_context(capture_registered_inputs(cls)) for cls in classes]
        yield captures


def event_rows(runner, record, subject, predictions, matches, labels, targets, registered, raw, full):
    """Preserve official assignment; derive each full contribution using sealed synergy."""
    if len(predictions) != len(record['videos']) or len(registered) != len(predictions):
        raise RuntimeError('video/GT identity alignment failed')
    rows, gt_rows, offset = [], [], 0
    for vi, preds in enumerate(predictions):
        video_id = '%s/video_%d' % (subject, vi)
        registered_preds, gt = registered[vi]
        if not np.array_equal(np.asarray(preds), registered_preds):
            raise RuntimeError('prediction order differs from registered evaluator input')
        if len(gt) != len(record['samples'][vi]):
            raise RuntimeError('registered GT count differs from record')
        for gi, target in enumerate(gt):
            sample = record['samples'][vi][gi]
            if target[0] != sample[0] or target[2] != sample[2]:
                raise RuntimeError('GT index does not map to source annotation coordinates')
            gt_rows.append(dict(gt_id='%s/gt_%d' % (video_id, gi), video_id=video_id,
                video_name=str(record['videos'][vi]), gt_index=gi, registered_gt=np.asarray(target).tolist(),
                source_sample=shared.serializable(record['samples'][vi][gi])))
        mapping = matches.get(vi, [])
        if len(mapping) != len(preds):
            raise RuntimeError('prediction/match length mismatch')
        for pi, (pred, ids) in enumerate(zip(preds, mapping)):
            ids = np.asarray(ids).reshape(-1).astype(int).tolist()
            if len(ids) != 1 or ids[0] < -1 or ids[0] >= len(gt):
                raise RuntimeError('invalid one-to-one GT ID')
            gi = ids[0]
            label, target = int(labels[offset]), int(targets[offset])
            offset += 1
            if (gi >= 0) != (target != -1):
                raise RuntimeError('recognition target disagrees with spotting match')
            single = (1, 0, 0) if gi >= 0 else (0, 1, 0)
            contribution = tuple(runner.full_counts_from_official_synergy(single, [label], [target]))
            permitted = ((1,0,0), (0,0,1)) if gi >= 0 else ((0,1,0), (0,0,0))
            if contribution not in permitted:
                raise RuntimeError('unsupported full-stage event semantics')
            rows.append(dict(video_id=video_id, video_name=str(record['videos'][vi]),
                prediction_index=pi, interval=np.asarray(pred).tolist(),
                matched_gt_id=None if gi < 0 else '%s/gt_%d' % (video_id, gi),
                recognition_prediction=label, recognition_target=target,
                raw_TP=int(gi >= 0), raw_FP=int(gi < 0),
                full_TP=int(contribution[0]), full_FP=int(contribution[1]),
                full_retained=bool(contribution[0]+contribution[1])))
    if offset != len(labels) or offset != len(targets):
        raise RuntimeError('recognition labels not exhausted')
    for stage, expected in (('raw', raw), ('full', full)):
        tp = sum(r[stage+'_TP'] for r in rows)
        fp = sum(r[stage+'_FP'] for r in rows)
        if (tp, fp, len(gt_rows)-tp) != tuple(expected):
            raise RuntimeError('event ledger does not reconstruct %s counts' % stage)
        ids = [r['matched_gt_id'] for r in rows if r[stage+'_TP']]
        if len(ids) != len(set(ids)):
            raise RuntimeError('duplicate GT event ID')
    return rows, gt_rows


def detailed_decode(context, core, si, config):
    runner, _, metric_class, official, records, _, subjects, _ = context
    core.trace, core.capture = [], True
    before = core.calls
    with capture_decoder_inputs(metric_class, official) as captures, contextlib.redirect_stdout(io.StringIO()):
        raw, predictions, labels, targets, matches, metric = runner.decode_glsd_subject(
            records, si, config, core, metric_class, official, True)
        full = runner.full_counts_from_official_synergy(raw, labels, targets)
    if core.calls == before:
        raise RuntimeError('fusion scorer bypassed')
    previous.validate(raw, full, previous.subject_gt(context, si), "diagnostic replay")
    captured = [entry[id(metric)] for entry in captures if id(metric) in entry]
    if len(captured) != 1:
        raise RuntimeError('returned video evaluator has no captured add calls')
    rows, gt = event_rows(runner, records[si], subjects[si], predictions, matches,
                         labels, targets, captured[0][1], raw, full)
    traces = list(core.trace)
    if len(traces) != len(records[si]['result_all']):
        raise RuntimeError('candidate trace/video count mismatch')
    for vi, (trace, response) in enumerate(zip(traces, records[si]['result_all'])):
        digest = hashlib.sha256(np.ascontiguousarray(np.asarray(response, dtype=float)).tobytes()).hexdigest()
        if trace['response_sha256'] != digest:
            raise RuntimeError('candidate trace/video order mismatch')
        trace.update(video_id='%s/video_%d' % (subjects[si], vi),
                     video_name=str(records[si]['videos'][vi]))
    return dict(subject=subjects[si], method=config.method, config=asdict(config),
                raw_counts=list(map(int, raw)), full_counts=list(map(int, full)),
                events=rows, ground_truth=gt, candidates=traces)


def compare_records(left, right):
    if left['ground_truth'] != right['ground_truth']:
        raise RuntimeError('Compared methods have different GT identities')
    result = dict(subject=left['subject'], compared_variant=left['variant'],
        compared=left['method'], reference_variant=right['variant'], reference=right['method'])
    for stage in ('raw', 'full'):
        a = {e['matched_gt_id'] for e in left['events'] if e[stage+'_TP']}
        b = {e['matched_gt_id'] for e in right['events'] if e[stage+'_TP']}
        result.update({stage+'_common_gt': len(a & b), stage+'_added_gt_ids': sorted(a-b),
                       stage+'_lost_gt_ids': sorted(b-a), stage+'_both_missed_gt': len(left['ground_truth'])-len(a|b)})
        delta = np.asarray(left[stage+'_counts']) - np.asarray(right[stage+'_counts'])
        if len(a-b)-len(b-a) != int(delta[0]):
            raise RuntimeError('Event identities do not explain aggregate delta')
        for key, value in zip(('TP','FP','FN'), delta):
            result[stage+'_delta_'+key] = int(value)
    return result


def summarize(records, output):
    by = {(r['variant'], r['subject'], r['method']): r for r in records}
    subjects = sorted({r['subject'] for r in records})
    comparisons = []
    for subject in subjects:
        for variant in ('original', 'expanded'):
            left = by[variant, subject, 'DominantEvidence']
            for method in ('G', 'L', 'EqualMean', 'WeightedMean'):
                comparisons.append(compare_records(left, by[variant, subject, method]))
            comparisons.append(compare_records(by[variant, subject, 'G'], by[variant, subject, 'L']))
        for method in tuning.METHODS:
            comparisons.append(compare_records(by['expanded', subject, method], by['original', subject, method]))
    shared.write_json(output/'event_differences.json', comparisons)
    groups = {}
    for row in comparisons:
        key = tuple(row[k] for k in ('compared_variant','compared','reference_variant','reference'))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        for stage in ('raw','full'):
            summary.append(dict(zip(('compared_variant','compared','reference_variant','reference'),key),
                metric_stage=stage, common_gt=sum(r[stage+'_common_gt'] for r in rows),
                added_gt=sum(len(r[stage+'_added_gt_ids']) for r in rows),
                lost_gt=sum(len(r[stage+'_lost_gt_ids']) for r in rows),
                both_missed_gt=sum(r[stage+'_both_missed_gt'] for r in rows),
                **{'delta_'+k:sum(r[stage+'_delta_'+k] for r in rows) for k in ('TP','FP','FN')}))
    shared.write_csv(output/'complementarity_summary.csv', summary)
    # Preserve method-specific assignment for shared predictions. Matching can
    # change when another prediction is added; do not transfer TP labels.
    score_rows = []
    for record in records:
        features = {(c['video_id'], int(p)): (g,l,s,bool(keep))
                    for c in record['candidates']
                    for p,g,l,s,keep in zip(c['peaks'],c['G'],c['L'],c['score'],c['retained'])}
        for event in record['events']:
            peak = int(event['interval'][6])
            key = event['video_id'], peak
            if key not in features or not features[key][3]:
                raise RuntimeError('Decoded prediction is not a retained candidate')
            g,l,s,_ = features[key]
            score_rows.append(dict(variant=record['variant'],subject=record['subject'],method=record['method'],
                video_id=event['video_id'],video_name=event['video_name'],peak=peak,
                a0=record['config']['reference_scale'],rho=record['config']['local_radius'],
                threshold=record['config']['threshold'],beta=record['config']['beta'],
                G=g,L=l,score=s,matched_gt_id=event['matched_gt_id'],
                raw_TP=event['raw_TP'],raw_FP=event['raw_FP'],full_TP=event['full_TP'],full_FP=event['full_FP'],
                recognition_prediction=event['recognition_prediction']))
    shared.write_csv(output/'selected_event_scores.csv', score_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference_path = Path(__file__).with_name('structure_selected_replay_reference.json')
    reference = json.loads(reference_path.read_text())
    if reference['matching_protocol'] != evaluator.PROTOCOL:
        raise RuntimeError('Matching protocol differs from saved run')
    previous.configure_threshold_grid('refined')
    previous.configure_structure_grid('expanded')
    ora, helper = shared.load_helper()
    spec = ora.SPECS['metst_sammlv']
    ora.verify_sealed_inputs(spec)
    for key in ('source','core'):
        if ora.sha256(spec[key]) != reference[key+'_sha256']:
            raise RuntimeError('Sealed source differs from structure run: '+key)
    output = args.output.resolve()
    for key in ('dump','evidence','run_evidence','results','locked_results'):
        protected = spec[key].resolve()
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise RuntimeError('Output overlaps sealed input')
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output/'manifest.json', dict(
        source_archive=reference['source_archive'],source_archive_sha256=reference['source_archive_sha256'],
        created_utc=datetime.now(timezone.utc).isoformat(),matching_protocol=evaluator.PROTOCOL,
        mode='replay already selected parameters; no search or model training',
        reference_sha256=ora.sha256(reference_path),
        code_sha256={p.name:ora.sha256(p) for p in (Path(__file__),Path(tuning.__file__),
            Path(previous.__file__),Path(evaluator.__file__),Path(shared.__file__),helper,
            Path(__file__).with_name('structure_expansion.py'))}))
    context = ora.metst_context(spec)
    if list(map(str, context[6])) != reference['subjects']:
        raise RuntimeError('Subject order differs from saved run')
    if ora.context_video_and_gt_counts('metst',context) != (79,159):
        raise RuntimeError('Unexpected dataset inventory')
    core = tuning.FusionCore(ExpandedCore(context[1], tuning.SCALES))
    records, checks = [], []
    with evaluator.install(context[2]) as info, gzip.open(output/'event_records.jsonl.gz','wt',encoding='utf8') as ledger:
        shared.write_json(output/'matching_protocol.json', info)
        for index, saved in enumerate(reference['records']):
            si = reference['subjects'].index(saved['subject'])
            config = tuning.Config(**saved['config'])
            result = detailed_decode(context, core, si, config)
            result['variant'] = saved['variant']
            for stage in ('raw_counts','full_counts'):
                if result[stage] != saved[stage]:
                    raise RuntimeError('Saved count replay failed: '+str((saved['variant'],saved['subject'],saved['method'],stage)))
            reconstructed = [[] for _ in saved['predictions']]
            for event in result['events']:
                vi = int(event['video_id'].rsplit('_',1)[1])
                reconstructed[vi].append(event['interval'])
            if reconstructed != saved['predictions']:
                raise RuntimeError('Saved predictions replay failed')
            records.append(result)
            checks.append(dict(variant=saved['variant'],subject=saved['subject'],method=saved['method'],
                raw_full_counts_match=True,predictions_match=True,event_counts_reconstruct=True))
            ledger.write(json.dumps(shared.serializable(result),allow_nan=False)+'\n')
            if (index+1)%29 == 0:
                ledger.flush()
                print(f"Selected configuration replay: {index+1}/{len(reference['records'])} PASS",flush=True)
    summarize(records, output)
    shared.write_csv(output/'replay_checks.csv',checks)
    shared.write_json(output/'completion.json',dict(completed=True,replays=len(checks),subjects=29,videos=79,gt=159))
    print('EVENT_DIAGNOSTIC_REPLAY = PASS; OUTPUT =',output,flush=True)


if __name__ == '__main__':
    main()
