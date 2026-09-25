"""Fixed-structure time-shift screening. No backbone training or inference."""
import argparse
import contextlib
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
import zipfile

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT/'vendor'
sys.path.insert(0, str(VENDOR))
import run_standalone as runtime
import numpy as np
import pandas as pd
import scipy
import sklearn
import shift_features as features

warnings.filterwarnings('ignore', message='The frame.append method is deprecated.*', category=FutureWarning)

engine, matching = runtime.engine, runtime.typed_matching
Config = engine.phase.Config
PROTOCOL = 'shift_control_fixed_a2_r3_six_scales_tau_minus100_200_v1'
EXPECTED = {'sammlv': (29, 79, 159, 5, 7), 'casme3': (94, 462, 853, 17, 1)}
ABLATIONS = {'no_global': 'D', 'no_local': 'G', 'no_shift': 'G_plus_L'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    runtime.write_json(tmp, value)
    tmp.replace(path)


def protocol():
    return dict(protocol=PROTOCOL, scales=features.SCALES, a0=features.A0, rho=features.RHO,
        methods=features.METHODS, thresholds=features.TAUS, threshold_count=301,
        configurations=2107, extra_all_reject_configuration=False,
        comparison='score >= tau; tau=2 can retain a score exactly equal to 2; no clipping',
        formulas=dict(Full='G+L-B', G='G', L='L', D='L-B', G_plus_L='G+L',
                      VideoMeanB='G+L-mean_candidates_in_this_video(B)', RefResidualUnclipped='G+L-L0'),
        controls='All integer q<=t<n-q with abs(t-p)>2q; fixed candidate L0, jointly shifted non-reference physical scales; q=w+delta+ceil(m/2)',
        fallback='single physical scale or no eligible time: B=L, D=0, Full=G exactly',
        selection='Exclude outer subject; maximize pooled other-subject full F1, precision, fewer FP, smaller threshold index',
        primary_ablations='Full structure fixed; D/G/G_plus_L independently reselect tau on outer training subjects',
        locked_ablations='Full structure AND tau inherited; decision diagnostics only',
        baseline_scope='G/L/D independently select tau in the singleton fixed structure domain; not independent multi-structure strong-baseline search',
        history='SAM-developed structure; SAM and CAS both development-exposed; no claim of untouched test data',
        historical_counts_reused=False, reason='Old structure/grid/clipping domains differ; no compatible new-mask count cache established',
        evaluation='Bundled official full decoder and neutral filtering, prediction-order one-to-one IoU=.5, typed aggregate/video interfaces',
        no_supervised_fit=True, no_backbone_run=True,
        efficacy='Execution completion is not evidence of superiority; full must beat G+L, VideoMeanB and unclipped residual before considering expansion')


def annotation_exception(record, vi, dataset):
    samples, emotions = record['samples'][vi], record['emotions'][vi]
    if len(samples) == len(emotions):
        return None
    detail = dict(subject=str(record['subject_id']), video=str(record['videos'][vi]),
        video_index=vi, samples=engine.shared.serializable(samples),
        emotions=engine.shared.serializable(emotions))
    if (dataset == 'casme3' and detail['subject'] == '216' and detail['video'] == 'e'
            and detail['samples'] == [[462, 470, 492]]
            and detail['emotions'] == ['others', 'surprise']):
        return dict(**detail, issue='known_extra_emotion_216_e',
                    action='preserve_original_labels_and_indexing',
                    scope='raw spotting and full neutral rejection only', correct_emotion='unresolved')
    raise RuntimeError('GT/emotion mismatch: '+json.dumps(detail))


def context_from_dump(dump, dataset):
    ns, nv, ng, k, frame_skip = EXPECTED[dataset]
    source = 'metst_official_glds_full.py' if dataset == 'sammlv' else 'metst_casme3_official_glds_full.py'
    runner = runtime.load_module(VENDOR/'sealed'/source, 'shift_sealed_runner')
    base = runtime.load_module(VENDOR/'sealed/boosting_official_glds_full_LOCKED.py', 'shift_sealed_core')
    base.SCALES = features.SCALES
    runner.ROOT, runner.DUMP_DIR = VENDOR/'author_runtime', dump
    runner.LOCKED_GLSD_PATH = VENDOR/'sealed/boosting_official_glds_full_LOCKED.py'
    metric, official = runner.load_official_metst()
    paths, records = runner.load_records()
    subjects = list(map(str, runner.subject_names(records)))
    if len(subjects) != ns or len(set(subjects)) != ns or sum(len(r['videos']) for r in records) != nv:
        raise RuntimeError('Official subject/video cardinality mismatch')
    context = (runner, base, metric, official, records, paths, subjects, None)
    core = features.FusionCore(base)
    core.expected_gt = engine.previous.gt_counts(context)
    if sum(core.expected_gt) != ng:
        raise RuntimeError(f'Expected {ng} GT; legacy CAS 858-GT input is incompatible')
    exceptions, response_hashes, missing_apex = [], [], []
    for si, r in enumerate(records):
        if int(r['k_p']) != k or int(r['frame_skip']) != frame_skip:
            raise RuntimeError('Unexpected k_p/frame_skip')
        for vi, (x, rec, samples) in enumerate(zip(r['result_all'], r['result1_all'], r['samples'])):
            x, rec = np.asarray(x), np.asarray(rec)
            if x.ndim != 1 or len(x) < 3 or rec.shape != x.shape:
                raise RuntimeError('Expected aligned one-dimensional response and recognition labels')
            if not np.isfinite(x).all() or not np.isfinite(rec).all() or not np.isin(rec, np.arange(5)).all():
                raise RuntimeError('Nonfinite response or invalid recognition labels (expected 0..4)')
            for sample in samples:
                s = np.asarray(sample)
                if (s.shape != (3,) or not np.isfinite(s).all() or not np.all(s == s.astype(int))
                        or not 0 <= s[0] <= s[2] or not (s[1] == 0 or s[0] <= s[1] <= s[2])):
                    raise RuntimeError('Invalid onset/apex/offset annotation')
                if s[1] == 0 and s[0] > 0:
                    missing_apex.append(dict(subject=subjects[si], video=str(r['videos'][vi]),
                        sample=s.tolist(), action='preserve author missing-apex sentinel; spotting uses onset/offset'))
            exception = annotation_exception(r, vi, dataset)
            if exception:
                exceptions.append(exception)
            response_hashes.append(dict(video_id=f'{subjects[si]}/video_{vi}', video_name=str(r['videos'][vi]),
                response_sha256=hashlib.sha256(np.ascontiguousarray(x, dtype=float).tobytes()).hexdigest(),
                recognition_sha256=hashlib.sha256(np.ascontiguousarray(rec, dtype=np.int64).tobytes()).hexdigest(), n=len(x)))
    summary = dict(dataset=dataset, subjects=ns, videos=nv, ground_truth=ng, k_p=k,
        frame_skip=frame_skip, method_type=1, strategy=1, per_subject_gt=core.expected_gt,
        annotation_exceptions=exceptions, missing_apex=missing_apex,
        emotion_classification_validated=False, response_hashes=response_hashes)
    return context, core, summary


def config(method, index):
    return Config(method, index, features.A0, features.RHO, features.TAUS[index])


def mask_key(context, core, si, cfg):
    r = context[4][si]
    return tuple((len(d['peaks']), np.packbits(features.score(d, cfg.method) >= cfg.threshold).tobytes())
        for d in (core.GLSDFeatures(x, r['k_p']).shift_evidence() for x in r['result_all']))


def checkpoint_digest(raw, full, done):
    return hashlib.sha256(raw.tobytes()+full.tobytes()+done.tobytes()).hexdigest()


def search(context, core, output, indices=None):
    """One subject checkpoint, caching identical complete subject masks across methods."""
    subjects = context[6]
    shape = (len(features.METHODS), len(features.TAUS), len(subjects), 3)
    raw, full = np.zeros(shape, np.int64), np.zeros(shape, np.int64)
    done = np.zeros(len(subjects), bool)
    path = output/'search_counts.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if z['subjects'].tolist() != subjects or z['methods'].tolist() != list(features.METHODS) or not np.array_equal(z['taus'], features.TAUS):
                raise RuntimeError('Checkpoint subject/method/grid identity mismatch')
            raw, full, done = (z[key].copy() for key in ('raw', 'full', 'done'))
            if raw.shape != shape or full.shape != shape or done.shape != (len(subjects),):
                raise RuntimeError('Checkpoint shape mismatch')
            if raw.dtype != np.int64 or full.dtype != np.int64 or done.dtype != bool:
                raise RuntimeError('Checkpoint dtype mismatch')
            if z['digest'].item() != checkpoint_digest(raw, full, done):
                raise RuntimeError('Checkpoint digest mismatch')
        for tensor in (raw, full):
            if np.any(tensor < 0) or not np.all(tensor[:, :, done, 0]+tensor[:, :, done, 2] == np.asarray(core.expected_gt)[done]):
                raise RuntimeError('Checkpoint GT conservation mismatch')
        if np.any(full[..., :2] > raw[..., :2]):
            raise RuntimeError('Checkpoint neutral filtering invariant failed')
    for si in (range(len(subjects)) if indices is None else indices):
        if done[si]:
            continue
        started, cache = time.monotonic(), {}
        for mi, method in enumerate(features.METHODS):
            for ti in range(len(features.TAUS)):
                cfg = config(method, ti)
                key = mask_key(context, core, si, cfg)
                if key not in cache:
                    row = engine.decode(context, core, si, cfg)
                    cache[key] = row['raw_counts'], row['full_counts']
                raw[mi, ti, si], full[mi, ti, si] = cache[key]
        done[si] = True
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, raw=raw, full=full, done=done, subjects=np.asarray(subjects),
                methods=np.asarray(features.METHODS), taus=np.asarray(features.TAUS),
                digest=np.asarray(checkpoint_digest(raw, full, done)))
        tmp.replace(path)
        write_json(output/f'search_subject_{si:03d}.json', dict(subject=subjects[si],
            grid_entries=2107, unique_subject_masks=len(cache), seconds=time.monotonic()-started))
        print(f'SEARCH {si+1}/{len(subjects)} subject={subjects[si]} unique_masks={len(cache)} seconds={time.monotonic()-started:.2f}', flush=True)
        core.cache.clear()
    return raw, full, done


def probe(context, core, output, grid=False):
    indices = list(dict.fromkeys([0]+[i for i, s in enumerate(context[6]) if s.lstrip('0') == '6']))
    if len(indices) == 1 and len(context[6]) > 1:
        indices.append(1)
    elapsed, counts, removed = [], [], 0
    with gzip.open(output/'probe_event_records.jsonl.gz', 'wt') as stream:
        for si in indices:
            for method in features.METHODS:
                for tau in (-.25, .65, 1.25):
                    started = time.monotonic()
                    row = engine.decode(context, core, si, config(method, features.TAUS.index(tau)), True)
                    elapsed.append(time.monotonic()-started)
                    stream.write(json.dumps(engine.shared.serializable(row), allow_nan=False)+'\n')
                    removed += sum(not e['full_retained'] for e in row['events'])
                    counts.append(dict(subject=row['subject'], method=method, tau=tau,
                                       raw=row['raw_counts'], full=row['full_counts']))
    if removed <= 0:
        raise RuntimeError('Real probe did not exercise neutral rejection')
    if grid:
        raw, full, done = search(context, core, output, indices=indices[:1])
        # Replaying all seven methods at a threshold confirms cached count mapping.
        for mi, method in enumerate(features.METHODS):
            cfg = config(method, 165)
            row = engine.decode(context, core, indices[0], cfg, True)
            if row['raw_counts'] != raw[mi, 165, indices[0]].tolist() or row['full_counts'] != full[mi, 165, indices[0]].tolist():
                raise RuntimeError('Real cached-grid replay mismatch')
    result = dict(status='PASS', scope='Real-input integration only; no outer performance estimate or formula selection',
        subjects=[context[6][i] for i in indices], real_decodes=len(counts),
        median_seconds_per_decode=float(np.median(elapsed)), recognition_rejections_across_probe_calls=removed,
        full_grid_one_subject=grid, counts=counts,
        checks=['seven scoring routes used by official decoder', 'negative and >1 thresholds',
                'numeric/list and string/scalar matching', 'prediction order and GT identity equality',
                'raw/full event ledger conservation', 'real neutral filtering exercised'])
    write_json(output/'probe.json', result)
    return result


def export_features(context, core, output):
    with gzip.open(output/'features.jsonl.gz', 'wt') as stream:
        for si, subject in enumerate(context[6]):
            r = context[4][si]
            for vi, x in enumerate(r['result_all']):
                f = core.GLSDFeatures(x, r['k_p'])
                row = dict(subject=subject, video_id=f'{subject}/video_{vi}', video_name=str(r['videos'][vi]),
                           response_sha256=f.key[2], k=r['k_p'], **f.shift_evidence())
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            core.cache.clear()


def select_and_replay(context, core, output, raw, full):
    selections, counts, diagnostic = [], [], []
    with gzip.open(output/'event_records.jsonl.gz', 'wt') as stream, gzip.open(output/'event_differences.jsonl.gz', 'wt') as diffs:
        for si, subject in enumerate(context[6]):
            winners = {m: engine.shared.choose(full[mi], si) for mi, m in enumerate(features.METHODS)}
            choices = [(m, m, winners[m][0], 'fixed_structure_independent_threshold') for m in features.METHODS]
            choices += [('ablation_retuned_'+name, m, winners[m][0], 'fixed_structure_retuned_threshold') for name, m in ABLATIONS.items()]
            choices += [('ablation_locked_'+name, m, winners['Full'][0], 'Full_structure_and_threshold_locked') for name, m in ABLATIONS.items()]
            records = {}
            for name, method, ti, selection in choices:
                mi = features.METHODS.index(method)
                row = engine.decode(context, core, si, config(method, ti), True)
                for stage, tensor in (('raw', raw), ('full', full)):
                    if row[stage+'_counts'] != tensor[mi, ti, si].tolist():
                        raise RuntimeError('Selected replay does not match search counts')
                    counts.append(dict(subject=subject, method=name, stage=stage, **engine.shared.metrics(row[stage+'_counts'])))
                row.update(method=name, scoring_method=method, selection=selection)
                records[name] = row
                stream.write(json.dumps(engine.shared.serializable(row), allow_nan=False)+'\n')
                pooled = full[mi, ti].sum(0)-full[mi, ti, si]
                selections.append(dict(subject=subject, method=name, scoring_method=method, config_id=ti,
                    a0=features.A0, rho=features.RHO, threshold=features.TAUS[ti], selection=selection,
                    training_subject_count=len(context[6])-1, **{'inner_'+k:v for k,v in engine.shared.metrics(pooled).items()}))
                if selection == 'Full_structure_and_threshold_locked':
                    values = [s for v in row['candidates'] for s in v['score']]
                    retained = sum(sum(v['retained']) for v in row['candidates'])
                    upper = 2 if method == 'G_plus_L' else 1
                    diagnostic.append(dict(subject=subject, method=name, threshold=features.TAUS[ti],
                        theoretical_upper=upper, threshold_above_upper=features.TAUS[ti] > upper,
                        candidates=len(values), retained=retained, all_rejected=retained == 0,
                        score_min=min(values) if values else None, score_max=max(values) if values else None))
            for name, row in records.items():
                if name != 'Full':
                    delta = engine.ledger.event_delta(records['Full'], row)
                    # Record actual event additions/losses as well as GT/candidate deltas.
                    def event_keys(r, stage):
                        return {(e['video_id'], tuple(e['interval'])) for e in r['events'] if stage == 'raw' or e['full_retained']}
                    for stage in ('raw', 'full'):
                        a, b = event_keys(records['Full'], stage), event_keys(row, stage)
                        delta[stage+'_added_events'] = sorted(a-b)
                        delta[stage+'_lost_events'] = sorted(b-a)
                    diffs.write(json.dumps(engine.shared.serializable(delta), allow_nan=False)+'\n')
            core.cache.clear()
            print('SELECTED_REPLAY', si+1, '/', len(context[6]), flush=True)
    engine.shared.write_csv(output/'selected_configs.csv', selections)
    engine.shared.write_csv(output/'per_subject_counts.csv', counts)
    engine.shared.write_csv(output/'locked_range_diagnostics.csv', diagnostic)
    summaries = []
    for name in dict.fromkeys(r['method'] for r in selections):
        for stage in ('raw', 'full'):
            values = np.asarray([[r[k] for k in ('TP','FP','FN')] for r in counts if r['method'] == name and r['stage'] == stage])
            summaries.append(dict(method=name, stage=stage, **engine.shared.metrics(values.sum(0))))
    engine.shared.write_csv(output/'summary_raw_full.csv', summaries)
    for name, prefix in [('fixed_structure_baselines', None), ('ablations_retuned', 'ablation_retuned_'), ('ablations_locked_diagnostic', 'ablation_locked_')]:
        rows = [r for r in summaries if r['stage'] == 'full' and (r['method'] == 'Full' or
                (r['method'] in features.METHODS if prefix is None else r['method'].startswith(prefix)))]
        engine.shared.write_csv(output/(name+'.csv'), rows)


def pack(output):
    archive = Path(str(output)+'.zip')
    tmp = archive.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix != '.tmp':
                z.write(p, str(Path(output.name)/p.relative_to(output)))
    tmp.replace(archive)
    print('RESULT_ZIP =', archive, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=EXPECTED, required=True)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'probe', 'screen'), default='screen')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-grid', action='store_true', help='Probe all 2107 grid entries for one real subject')
    args = parser.parse_args()
    dump, output = args.dump.resolve(), args.output.resolve()
    if not dump.is_dir():
        raise RuntimeError('Missing official dump folder: '+str(dump))
    for protected in (ROOT, dump):
        if output == protected or output in protected.parents or protected in output.parents:
            raise RuntimeError('Output must not overlap package or inputs')
    context, core, input_summary = context_from_dump(dump, args.dataset)
    identity = dict(protocol=protocol(), dataset=args.dataset, mode=args.mode, probe_grid=args.probe_grid,
        inputs={p.name:sha(p) for p in context[5]},
        code={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py'))},
        runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                     sklearn=sklearn.__version__, platform=platform.platform()))
    identity = engine.shared.serializable(identity)
    if output.exists():
        if not args.resume or not (output/'run_manifest.json').is_file():
            raise RuntimeError('Use a fresh output or --resume on this exact run')
        if json.loads((output/'run_manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('Resume input/code/runtime/protocol differs; choose a fresh output')
        if (output/'completion.json').exists():
            from verify_results import verify
            verify(output)
            pack(output)
            print('Already completed and reverified', flush=True)
            return
    else:
        if args.resume:
            raise RuntimeError('Resume output does not exist')
        output.mkdir(parents=True)
        write_json(output/'run_manifest.json', dict(identity=identity))
    write_json(output/'protocol.json', protocol())
    write_json(output/'input_summary.json', input_summary)
    classes = [context[2], context[3].spotting.__globals__['MeanAveragePrecision2d']]
    try:
        with contextlib.ExitStack() as stack:
            contracts = [stack.enter_context(matching.install(cls)) for cls in classes]
            if [c['return_contract'] for c in contracts] != ['numeric_list', 'string_scalar']:
                raise RuntimeError('Unexpected matching interfaces')
            write_json(output/'matching_protocol.json', contracts)
            write_json(output/'audit_synthetic.json', matching.synthetic_probe(classes))
            if args.mode == 'audit':
                write_json(output/'input_audit_complete.json', dict(status='PASS', scope='input and evaluator software contracts only'))
                return
            probe(context, core, output, grid=args.probe_grid)
            if args.mode == 'probe':
                return
            export_features(context, core, output)
            raw, full, done = search(context, core, output)
            if not done.all():
                raise RuntimeError('Incomplete search')
            select_and_replay(context, core, output, raw, full)
        from verify_results import verify
        audit = verify(output)
        write_json(output/'result_audit.json', audit)
        write_json(output/'completion.json', dict(completed=True, protocol=PROTOCOL, status='PASS',
            scope='Full fixed-structure screening completed and audited; not a claim of efficacy or SOTA'))
        print('SCREEN_COMPLETE; see separate baseline/retuned/locked result tables', flush=True)
    except Exception:
        write_json(output/'failure_case.json', dict(traceback=traceback.format_exc(), context=getattr(core, 'last_context', {})))
        raise
    finally:
        pack(output)


if __name__ == '__main__':
    main()
