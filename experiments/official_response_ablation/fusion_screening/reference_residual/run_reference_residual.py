"""Fixed-structure reference-residual development screening on frozen responses.

Every decoder call uses full recognition and verifies both matching maps.
No old search counts are reused. No backbone is trained or invoked.
"""
import argparse
import contextlib
from dataclasses import asdict, replace
import gzip
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import run_p8_phase2 as phase
import run_p8_one_to_one as previous
import run_p8_threshold_control as ledger
import typed_matching as matching
from residual_features import FusionCore, METHODS, LOCKED

shared = phase.shared
PROTOCOL = 'reference_residual_fixed_a2_r2_native_video_v1'


def map_audit_path(path):
    """Map Colab paths recorded by the passed audit to this host's roots."""
    path = Path(path)
    colab_drive = Path('/content/drive/MyDrive')
    data_root = Path(os.environ.get('GLSD_DATA_ROOT', str(colab_drive)))
    metst_root = Path(os.environ.get('ME_TST_ROOT', '/content/ME-TST'))
    try:
        return data_root / path.relative_to(colab_drive)
    except ValueError:
        pass
    try:
        return metst_root / path.relative_to('/content/ME-TST')
    except ValueError:
        return path


def grids():
    return {m: [phase.Config(m, i, 2., 2., t) for i, t in enumerate(phase.TAUS)] for m in METHODS}


def decode(context, core, si, config, capture=False):
    runner, _, metric_class, official, records, _, subjects, _ = context
    core.capture, core.trace = capture, []
    core.last_context = dict(subject=subjects[si], config=asdict(config))
    before = core.calls
    try:
        with ledger.capture_decoder_inputs(metric_class, official) as captures, contextlib.redirect_stdout(io.StringIO()):
            raw, predictions, labels, targets, maps, video_metric = runner.decode_glsd_subject(
                records, si, config, core, metric_class, official, True)
            full = runner.full_counts_from_official_synergy(raw, labels, targets)
            core.last_context.update(raw=raw, full=full, predictions=predictions,
                                     video_matches=maps, recognition_labels=labels, recognition_targets=targets)
            entries = [entry for group in captures for entry in group.values()]
            returned = [items for instance, items in entries if instance is video_metric]
            if len(returned) != 1:
                raise RuntimeError('cannot identify returned video evaluator')
            core.last_context['registered_inputs'] = [dict(
                role='video' if instance is video_metric else 'aggregate', inputs=items)
                for instance, items in entries]
            expected_map = matching.normalize_map(maps)
            # Compare GT identities, not just total TP. Both stages must see all videos/predictions.
            for instance, items in entries:
                if len(items) != len(predictions):
                    raise RuntimeError('evaluator/video registration mismatch')
                for (pred, gt), expected_pred, (_, expected_gt) in zip(items, predictions, returned[0]):
                    if len(pred) != len(expected_pred):
                        raise RuntimeError('evaluator prediction count differs')
                    if len(pred) and not np.array_equal(pred[:, :5], np.asarray(expected_pred)[:, :5]):
                        raise RuntimeError('evaluator prediction coordinates/order changed')
                    if not np.array_equal(gt, expected_gt):
                        raise RuntimeError('aggregate/video GT order differs')
                values = instance.value(iou_thresholds=.5)[.5][0]
                core.last_context['last_checked_map'] = values['pred_match_gt']
                if matching.normalize_map(values['pred_match_gt']) != expected_map:
                    raise RuntimeError('aggregate/video matched GT identities differ')
        if core.calls == before:
            raise RuntimeError('fusion scorer bypassed')
        previous.validate(raw, full, core.expected_gt[si])
        events, gt = ledger.event_rows(runner, records[si], subjects[si], predictions, maps,
                                       labels, targets, returned[0], raw, full)
        traces = list(core.trace)
        if capture:
            if len(traces) != len(records[si]['result_all']):
                raise RuntimeError('feature trace/video cardinality mismatch')
            for vi, (trace, response) in enumerate(zip(traces, records[si]['result_all'])):
                digest = hashlib.sha256(np.ascontiguousarray(np.asarray(response, float)).tobytes()).hexdigest()
                if digest != trace['response_sha256']:
                    raise RuntimeError('feature trace/video order mismatch')
                trace.update(video_id='%s/video_%d' % (subjects[si], vi), video_name=str(records[si]['videos'][vi]))
        return dict(subject=subjects[si], method=config.method, config=asdict(config),
                    raw_counts=list(map(int, raw)), full_counts=list(map(int, full)),
                    events=events, ground_truth=gt, candidates=traces)
    finally:
        core.capture = False


def export_features(context, core, output):
    summaries = []
    with gzip.open(output/'features.jsonl.gz', 'wt', encoding='utf8') as stream:
        for si, subject in enumerate(context[6]):
            record = context[4][si]
            deltas = []
            for vi, response in enumerate(record['result_all']):
                feature = core.GLSDFeatures(response, record['k_p'])
                data = feature.residual_evidence(2., 2.)
                deltas.extend(data['D'])
                row = dict(subject=subject, video_id='%s/video_%d' % (subject, vi),
                           video_name=str(record['videos'][vi]), response_sha256=feature.key[2],
                           k=record['k_p'], a0=2., rho=2., **data)
                stream.write(json.dumps(shared.serializable(row), allow_nan=False)+'\n')
            d = np.asarray(deltas)
            summaries.append(dict(subject=subject, candidates=len(d),
                negative=int(np.sum(d < -1e-12)), zero=int(np.sum(np.abs(d) <= 1e-12)),
                positive=int(np.sum(d > 1e-12)),
                absolute_delta_median=float(np.median(np.abs(d))) if len(d) else 0.,
                absolute_delta_p95=float(np.quantile(np.abs(d), .95)) if len(d) else 0.))
            print('feature export', si+1, '/', len(context[6]), flush=True)
    shared.write_csv(output/'feature_summary.csv', summaries)


def probe(context, core, output, name):
    indices = list(dict.fromkeys([0]+[i for i, s in enumerate(context[6]) if str(s).lstrip('0') == '6']))
    rows, elapsed = [], []
    for si in indices:
        for method in METHODS:
            start = time.monotonic()
            rows.append(decode(context, core, si, phase.Config(method, 0, 2., 2., .65), True))
            elapsed.append(time.monotonic()-start)
    with gzip.open(output/'probe_event_records.jsonl.gz', 'wt', encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(shared.serializable(row), allow_nan=False)+'\n')
    removed = sum(e['raw_FP']-e['full_FP'] for row in rows for e in row['events'])
    shared.write_json(output/'probe.json', dict(passed=True, setting=name,
        subjects=[context[6][i] for i in indices], calls=len(rows),
        median_seconds_per_timed_decode=float(np.median(elapsed)),
        recognition_removed_FP_in_probe=int(removed),
        all_prediction_entries_present=True, aggregate_video_gt_maps_equal=True,
        full_event_ledger_reconstructs_counts=True,
        time_note='small-sample timings only; subject length varies; no runtime guarantee'))
    print('REFERENCE_RESIDUAL_PROBE = PASS; median decode seconds=%.3f' % np.median(elapsed), flush=True)


def search(context, core, configs, path):
    subjects = list(context[6])
    identity = json.dumps([asdict(c) for c in configs], sort_keys=True)
    shape = (len(configs), len(subjects), 3)
    raw, full, done = np.zeros(shape, np.int64), np.zeros(shape, np.int64), np.zeros(len(subjects), bool)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved['configs'].item() != identity or saved['subjects'].tolist() != subjects:
                raise RuntimeError('checkpoint identity mismatch')
            raw, full, done = (saved[k].copy() for k in ('raw', 'full', 'done'))
        if raw.shape != shape or full.shape != shape or done.shape != (len(subjects),):
            raise RuntimeError('checkpoint shape mismatch')
        for array in (raw, full):
            if np.any(array[:, done] < 0) or not np.all(array[:, done, 0]+array[:, done, 2] == np.asarray(core.expected_gt)[done]):
                raise RuntimeError('checkpoint count conservation failed')
    for si, subject in enumerate(subjects):
        if done[si]:
            continue
        for config in configs:
            record = decode(context, core, si, config)
            raw[config.config_id, si], full[config.config_id, si] = record['raw_counts'], record['full_counts']
        done[si] = True
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, raw=raw, full=full, done=done,
                configs=np.asarray(identity), subjects=np.asarray(subjects, dtype=str))
        tmp.replace(path)
        print(configs[0].method, si+1, '/', len(subjects), 'saved', flush=True)
    return raw, full


def comparisons(counts):
    draws = np.random.default_rng(100).integers(0, len(counts['G']), (10000, len(counts['G'])))
    def boot(a):
        x = a[draws].sum(1)
        denom = 2*x[:, 0]+x[:, 1]+x[:, 2]
        return np.divide(2*x[:, 0], denom, out=np.zeros(len(x)), where=denom > 0)
    target, rows = counts['RefResidual'], []
    for method, array in counts.items():
        if method == 'RefResidual':
            continue
        delta = boot(target)-boot(array)
        rows.append(dict(compared='RefResidual', reference=method,
            delta_F1=shared.metrics(target.sum(0))['F1']-shared.metrics(array.sum(0))['F1'],
            CI_low=float(np.quantile(delta, .025)), CI_high=float(np.quantile(delta, .975)),
            seed=100, resamples=10000, interpretation='development; conditional on selected predictions; no retuning'))
    return rows


def screen(context, core, output):
    all_grids = grids()
    tables = {method: search(context, core, configs, output/('search_counts_%s.npz' % method))
              for method, configs in all_grids.items()}
    counts = {method: [] for method in (*METHODS, *LOCKED)}
    selections, subject_rows = [], []
    event_totals = {method: dict(added_gt=0, lost_gt=0, delta_FP=0) for method in counts if method != 'RefResidual'}
    with gzip.open(output/'event_records.jsonl.gz', 'wt', encoding='utf8') as events, \
         gzip.open(output/'event_differences.jsonl.gz', 'wt', encoding='utf8') as differences:
        for si, subject in enumerate(context[6]):
            configs = {}
            for method in METHODS:
                winner, pooled = shared.choose(tables[method][1], si)
                configs[method] = all_grids[method][winner]
                selections.append(dict(subject=subject, **asdict(configs[method]),
                    selection='inner_full', inner_F1=shared.metrics(pooled[winner])['F1']))
            configs.update({method: replace(configs['RefResidual'], method=method) for method in LOCKED})
            selected = {}
            for method, config in configs.items():
                record = decode(context, core, si, config, True)
                if method in tables:
                    for j, stage in enumerate(('raw', 'full')):
                        if not np.array_equal(record[stage+'_counts'], tables[method][j][config.config_id, si]):
                            raise RuntimeError('selected counts do not reproduce search tensor')
                else:
                    selections.append(dict(subject=subject, **asdict(config), selection='locked_RefResidual', inner_F1=''))
                selected[method] = record
                counts[method].append(record['full_counts'])
                events.write(json.dumps(shared.serializable(record), allow_nan=False)+'\n')
                for stage in ('raw', 'full'):
                    subject_rows.append(dict(subject=subject, method=method, stage=stage,
                        **shared.metrics(record[stage+'_counts'])))
            for method in event_totals:
                delta = ledger.event_delta(selected['RefResidual'], selected[method])
                differences.write(json.dumps(shared.serializable(delta), allow_nan=False)+'\n')
                event_totals[method]['added_gt'] += len(delta['added_gt_ids'])
                event_totals[method]['lost_gt'] += len(delta['lost_gt_ids'])
                event_totals[method]['delta_FP'] += delta['delta_FP']
            print('selected event replay', si+1, '/', len(context[6]), flush=True)
    counts = {m: np.asarray(rows, np.int64) for m, rows in counts.items()}
    summary = [dict(method=m, selection='locked_RefResidual' if m in LOCKED else 'inner_full',
                    configs=0 if m in LOCKED else 19, **shared.metrics(rows.sum(0))) for m, rows in counts.items()]
    for filename, rows in [('summary_full.csv', summary), ('selected_configs.csv', selections),
                           ('per_subject_counts.csv', subject_rows), ('paired_comparisons.csv', comparisons(counts))]:
        shared.write_csv(output/filename, rows)
    f = {row['method']: row['F1'] for row in summary}
    positive = f['RefResidual'] > max(f[m] for m in ('G', 'L', 'Mean', 'no_local', 'no_G'))
    shared.write_json(output/'decision.json', dict(stage='fixed-structure development screening',
        exceeds_G_L_Mean_and_locked_removals=positive, event_deltas=event_totals,
        next_step='inspect event/threshold evidence before any expansion' if positive else 'do not expand automatically; inspect failure mechanism',
        not_SOTA_validation=True, no_claim_of_independent_external_validation=True))
    print('REFERENCE_RESIDUAL_SCREEN = PASS (execution completed; not a claim of efficacy)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'screen'), required=True)
    parser.add_argument('--setting', choices=('sammlv', 'casme3'), default='sammlv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit', type=Path, required=True,
                        help='Completed stage-0 SAMMLV audit directory, read only')
    args = parser.parse_args()
    audit_completion = json.loads((args.audit/'completion.json').read_text())
    if not audit_completion.get('completed') or audit_completion.get('status') != 'PASS':
        raise RuntimeError('stage-0 evaluation audit has not passed')
    if args.setting != 'sammlv':
        raise RuntimeError('This package is for SAMMLV screening; CAS follows mechanism review')
    ora, helper = shared.load_helper()
    name = shared.SETTINGS[args.setting]
    spec, output = ora.SPECS[name], args.output.resolve()
    for protected in ora.SPECS.values():
        for key in ('source', 'core', 'dump', 'cache', 'evidence', 'run_evidence', 'results', 'locked_results'):
            if key in protected and phase.overlaps(output, Path(protected[key]).resolve()):
                raise RuntimeError('output overlaps sealed input')
    ora.verify_sealed_inputs(spec)
    context = ora.metst_context(spec)  # Original historical Native gate, before any patch.
    video_class = context[3].spotting.__globals__['MeanAveragePrecision2d']
    classes = list(dict.fromkeys([context[2], video_class]))
    paths = [spec['source'], spec['core'], spec['evidence']/'evidence_bundle_sha256.txt',
             spec['run_evidence']/'run_manifest.json']
    paths += sorted(p for p in spec['dump'].rglob('*') if p.is_file())
    for obj in [*classes, context[3]]:
        path = Path(inspect.getsourcefile(obj))
        paths.append(path)
    for cls in classes:
        paths.append(Path(inspect.getsourcefile(cls._evaluate_class.__globals__['check_box'])))
    audit_manifest = json.loads((args.audit/'run_manifest.json').read_text())
    for path, digest in {**audit_manifest['inputs'], **audit_manifest['dump_sha256']}.items():
        mapped = map_audit_path(path)
        if not mapped.exists() or ora.sha256(mapped) != digest:
            raise RuntimeError('input differs from passed stage-0 audit: '+str(mapped))
    contracts = json.loads((args.audit/'contracts.json').read_text())
    for cls, role in [(context[2], 'aggregate'), (video_class, 'video')]:
        expected = next(row for row in contracts if row['role'] == role)
        actual = inspect.getsource(cls._evaluate_class.__globals__['check_box'])
        if hashlib.sha256(actual.encode()).hexdigest() != expected['check_box_sha256']:
            raise RuntimeError('check_box differs from passed audit: '+role)
    paths += [args.audit/'run_manifest.json', args.audit/'completion.json', args.audit/'contracts.json']
    identity = dict(protocol=PROTOCOL, evaluation=matching.PROTOCOL, mode=args.mode, setting=name,
        runtime=dict(python=sys.version, numpy=np.__version__, pandas=getattr(sys.modules.get('pandas'), '__version__', None)),
        inputs={str(p): ora.sha256(p) for p in paths},
        code={p.name: ora.sha256(p) for p in Path(__file__).parent.glob('*.py')},
        subjects=list(context[6]), structure=dict(a0=2., rho=2., thresholds=list(phase.TAUS)))
    if output.exists():
        if not args.resume or not (output/'run_manifest.json').exists() or json.loads((output/'run_manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('resume requires identical code/runtime/input/protocol/mode; use a new directory')
        if (output/'completion.json').exists():
            print('Already completed:', output, flush=True)
            return
    else:
        if args.resume:
            raise RuntimeError('resume directory missing')
        output.mkdir(parents=True)
        shared.write_json(output/'run_manifest.json', dict(identity=identity))
    print('OUTPUT =', output, flush=True)
    shared.write_json(output/'protocol.json', dict(protocol=PROTOCOL, evaluation=matching.PROTOCOL,
        grids={m: [asdict(c) for c in cs] for m, cs in grids().items()},
        selection='pooled other-subject full F1; precision; fewer FP; fixed config index',
        ablations='no_local=G; no_G=clip(L-L0); no_negative=clip(G+max(L-L0,0)); locked to complete method',
        scope='frozen-response post-processing development; a0=2/rho=2 informed by earlier development',
        bootstrap='paired subjects, 10000, seed100; conditional on selected predictions',
        old_counts_reused=False, matching_change='audited aggregate one-to-one adapter; native video check_box and caller policies unchanged'))
    core = FusionCore(context[1])
    core.expected_gt = previous.gt_counts(context)
    if sum(core.expected_gt) != spec['gt'] or sum(len(r['videos']) for r in context[4]) != spec['videos']:
        raise RuntimeError('video/GT inventory mismatch')
    try:
        with contextlib.ExitStack() as stack:
            info = [stack.enter_context(matching.install(cls)) for cls in classes]
            shared.write_json(output/'matching_protocol.json', info)
            export_features(context, core, output)
            probe(context, core, output, name)
            if args.mode == 'screen':
                screen(context, core, output)
    except Exception:
        shared.write_json(output/'failure_case.json', shared.serializable(dict(
            traceback=traceback.format_exc(), context=getattr(core, 'last_context', {}))))
        raise
    shared.write_json(output/'completion.json', dict(completed=True, mode=args.mode, setting=name,
        protocol=PROTOCOL, evaluation=matching.PROTOCOL))
    print('OUTPUT =', output, flush=True)


if __name__ == '__main__':
    main()
