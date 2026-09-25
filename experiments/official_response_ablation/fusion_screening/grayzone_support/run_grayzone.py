"""Grayzone relative support v1. Frozen ME-TST responses; pooled full LOSO.

Inherited decoder/evidence/evaluator sources are immutable package snapshots.
No historical P8 counts or currently running P8 outputs are loaded.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import contextlib
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
import run_p8_threshold_control as ledger

phase, previous, evaluator = ledger.phase, ledger.previous, ledger.evaluator
shared = phase.shared
PROTOCOL = 'grayzone_relative_support_v1_full_loso'
SCALES, RADII = phase.SCALES, phase.RADII
TAUS = tuple(i / 40 for i in range(1, 41))
# Dense singles plus *all* effective thresholds of this run's P8 grid.
# A single >1 threshold represents the known identical reject-all endpoint.
SINGLE_TAUS = tuple(sorted(set(TAUS + tuple(2**(1/8)*t for t in TAUS if 2**(1/8)*t <= 1) + (1.1,))))
PAIRS = tuple(((h-w)/20, h/20) for h in range(2, 21) for w in (1, 2, 4) if h > w)
ETAS = (.25, .5, .75, 1.)
ABLATIONS = ('Gray_high_only', 'Gray_no_support')


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    reference_scale: float
    local_radius: float
    threshold: float = 0.
    branch: str = ''
    tau_low: float = 0.
    tau_high: float = 0.
    eta: float = 0.

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def grids():
    result = {}
    for m in ('G', 'L', 'Mean', 'P8_control'):
        ts = SINGLE_TAUS if m in ('G', 'L') else TAUS
        radii = (1.,) if m == 'G' else RADII
        values = [(a, r, t) for a in SCALES for r in radii for t in ts]
        result[m] = [Config(m, i, *v) for i, v in enumerate(values)]
    values = [(a, r, b, lo, hi, eta) for a in SCALES for r in RADII
              for b in ('G', 'L') for lo, hi in PAIRS for eta in ETAS]
    result['Gray'] = [Config('Gray', i, a, r, branch=b, tau_low=lo, tau_high=hi, eta=e)
                      for i, (a, r, b, lo, hi, e) in enumerate(values)]
    return result


def masks(values, c):
    values = np.asarray(values, dtype=float)
    phase.score(values, 'Mean')  # inherited finite/range/shape validation; no clipping
    g, l = values.T
    if c.method in ('G', 'L', 'Mean', 'P8_control'):
        scores = phase.score(values, 'GM_p8' if c.method == 'P8_control' else c.method)
        return scores >= c.threshold, dict(score=scores.tolist())
    if c.method not in ('Gray', *ABLATIONS):
        raise ValueError(c.method)
    if c.branch not in ('G', 'L') or not 0 < c.tau_low < c.tau_high <= 1 or not np.isfinite(c.eta) or c.eta < 0:
        raise ValueError('invalid grayzone configuration')
    b, r = (g, l) if c.branch == 'G' else (l, g)
    strong = b >= c.tau_high
    gray = (b >= c.tau_low) & ~strong
    supported = r >= c.eta*b
    keep = strong | (gray & supported)
    if c.method == 'Gray_high_only':
        keep = strong
    elif c.method == 'Gray_no_support':
        keep = b >= c.tau_low
    return keep, dict(branch=c.branch, strong=strong.tolist(), gray_entered=gray.tolist(),
                      support_condition=supported.tolist(), gray_passed=(gray & supported).tolist())


class FeatureView(phase.FeatureView):
    def selected_peaks(self, config):
        peaks, values = self.evidence(config.reference_scale, config.local_radius)
        keep, details = masks(values, config)
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.key[0],
                peaks=peaks.tolist(), G=values[:, 0].tolist(), L=values[:, 1].tolist(),
                retained=keep.tolist(), **details))
        return peaks[keep]


class FusionCore(phase.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def verify_candidate_only_config(runner):
    """Fail closed before memoizing: config must only enter selected_peaks()."""
    fn = runner.decode_glsd_subject
    parameter = list(inspect.signature(fn).parameters)[2]
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == parameter and isinstance(n.ctx, ast.Load)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'selected_peaks' and len(n.args) == 1 and n.args[0] in loads]
    if len(loads) != 1 or len(calls) != 1:
        raise RuntimeError('sealed config is not exclusively used by selected_peaks; cannot cache by candidate identity')
    return dict(config_argument=parameter, config_only_in_selected_peaks=True,
                decoder_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest())


def candidate_key(context, core, si, c):
    record = context[4][si]
    # Video boundaries and ordered peaks are part of identity, not just a bag of peaks.
    return tuple(tuple(int(p) for p in core.GLSDFeatures(x, record['k_p']).selected_peaks(c))
                 for x in record['result_all'])


def save_npz(path, **values):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def search(context, core, configs, output):
    subjects = list(context[6])
    identity = json.dumps([asdict(c) for c in configs], sort_keys=True)
    shape = (len(configs), len(subjects), 3)
    path = output/'search_counts.npz'
    raw, full = np.zeros(shape, np.int64), np.zeros(shape, np.int64)
    done = np.zeros(shape[:2], bool)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if z['configs'].item() != identity or z['subjects'].tolist() != subjects:
                raise RuntimeError('search checkpoint identity mismatch')
            raw, full, done = (z[k].copy() for k in ('raw', 'full', 'done'))
        if raw.shape != shape or full.shape != shape or done.shape != shape[:2] or done.dtype != bool:
            raise RuntimeError('search checkpoint shape/type mismatch')
        if raw.dtype != np.int64 or full.dtype != np.int64:
            raise RuntimeError('noninteger search checkpoint')
        for i, si in zip(*np.where(done)):
            previous.validate(raw[i, si], full[i, si], core.expected_gt[si])
    def checkpoint():
        save_npz(path, raw=raw, full=full, done=done, subjects=np.asarray(subjects), configs=np.asarray(identity))
    for si, subject in enumerate(subjects):
        if done[:, si].all():
            continue
        core.cache.clear()
        memo, evaluations, aliases = {}, 0, 0
        last = time.monotonic()
        for i, c in enumerate(configs):
            core.capture = False
            key = candidate_key(context, core, si, c)
            if done[i, si]:
                pair = (raw[i, si].copy(), full[i, si].copy())
                if key in memo and any(not np.array_equal(x, y) for x, y in zip(pair, memo[key])):
                    raise RuntimeError('checkpoint equivalent predictions disagree')
                memo[key] = pair
                continue
            if key not in memo:
                # Evaluate the complete official path. Ledger guards also validate video/aggregate agreement.
                rec = ledger.detailed_decode(context, core, si, c)
                memo[key] = rec['raw_counts'], rec['full_counts']
                evaluations += 1
            else:
                aliases += 1
            raw[i, si], full[i, si] = memo[key]
            done[i, si] = True
            if (i+1) % 100 == 0 or time.monotonic()-last > 45:
                checkpoint()
                print('search', subject, si+1, '/', len(subjects), i+1, '/', len(configs),
                      'new decodes', evaluations, 'equivalent masks', aliases, flush=True)
                last = time.monotonic()
        checkpoint()
        shared.write_json(output/('search_subject_%03d.json' % si), dict(subject=subject,
            complete=True, distinct_candidate_sequences=len(memo), configurations=len(configs),
            new_decodes_this_attempt=evaluations, aliases_this_attempt=aliases))
    return raw, full


def fp_ids(record):
    # Stable event identity across configurations, including duplicate intervals.
    multiplicities = Counter()
    result = {}
    for r in record['events']:
        key = json.dumps([r['video_id'], r['interval']], separators=(',', ':'))
        occurrence = multiplicities[key]
        multiplicities[key] += 1
        identity = key + '#' + str(occurrence)
        if r['full_FP']:
            result[identity] = r
    return result


def event_delta(fusion, reference):
    delta = ledger.event_delta(fusion, reference)
    a, b = fp_ids(fusion), fp_ids(reference)
    delta.update(added_fp_events=[a[k] for k in sorted(a.keys()-b.keys())],
                 removed_fp_events=[b[k] for k in sorted(b.keys()-a.keys())])
    if len(delta['added_fp_events'])-len(delta['removed_fp_events']) != delta['delta_FP']:
        raise RuntimeError('FP identity delta does not reconstruct counts')
    return delta


def paired(counts):
    n = len(counts['Gray'])
    draws = np.random.default_rng(100).integers(0, n, (10000, n))
    def sampled(a):
        a = a[draws].sum(1)
        d = 2*a[:, 0]+a[:, 1]+a[:, 2]
        return np.divide(2*a[:, 0], d, out=np.zeros(len(d)), where=d > 0)
    gray = sampled(counts['Gray'])
    return [dict(compared='Gray', reference=m,
                 delta_F1=shared.metrics(counts['Gray'].sum(0))['F1']-shared.metrics(a.sum(0))['F1'],
                 CI_low=float(np.quantile(gray-sampled(a), .025)),
                 CI_high=float(np.quantile(gray-sampled(a), .975)), resamples=10000, seed=100,
                 interpretation='conditional on selected predictions; no reselection or development-bias correction')
            for m, a in counts.items() if m != 'Gray']


def selected_record(context, core, si, c, output):
    path = output/('selected_%03d_%s.json.gz' % (si, c.method))
    if path.exists():
        with gzip.open(path, 'rt') as f:
            r = json.load(f)
        if r['subject'] != context[6][si] or r['config'] != asdict(c):
            raise RuntimeError('event checkpoint identity mismatch')
        return r
    r = ledger.detailed_decode(context, core, si, c)
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wt', encoding='utf8') as f:
        json.dump(shared.serializable(r), f, allow_nan=False)
    tmp.replace(path)
    return r


def report(context, core, all_grids, tables, native, output):
    choices, selections, counts, per_subject = {}, [], {}, []
    for m, cs in all_grids.items():
        choices[m] = []
        for si, subject in enumerate(context[6]):
            wi, pooled = shared.choose(tables[m][1], si)
            choices[m].append(cs[wi])
            selections.append(dict(subject=subject, selected_method=m, **asdict(cs[wi]),
                selection='inner_full', inner_F1=shared.metrics(pooled[wi])['F1'],
                inner_TP=int(pooled[wi, 0]), inner_FP=int(pooled[wi, 1]), inner_FN=int(pooled[wi, 2])))
    # Stronger one-module control: choose G versus L inside each training fold as well.
    single_cs = all_grids['G']+all_grids['L']
    single_full = np.concatenate((tables['G'][1], tables['L'][1]))
    choices['Single_selected'] = []
    for si, subject in enumerate(context[6]):
        wi, pool = shared.choose(single_full, si)
        choices['Single_selected'].append(single_cs[wi])
        selections.append(dict(subject=subject, selected_method='Single_selected', **asdict(single_cs[wi]),
            selection='inner_full', inner_F1=shared.metrics(pool[wi])['F1'],
            inner_TP=int(pool[wi, 0]), inner_FP=int(pool[wi, 1]), inner_FN=int(pool[wi, 2])))
    mechanisms = []
    replay = output/'selected_events'
    replay.mkdir(exist_ok=True)
    with gzip.open(output/'event_records.jsonl.gz', 'wt', encoding='utf8') as out, \
         gzip.open(output/'event_differences.jsonl.gz', 'wt', encoding='utf8') as diff:
        for si, subject in enumerate(context[6]):
            core.cache.clear()
            recs = {}
            configs = {m: cs[si] for m, cs in choices.items() if m != 'Single_selected'}
            configs.update({m: replace(configs['Gray'], method=m) for m in ABLATIONS})
            for m, c in configs.items():
                r = selected_record(context, core, si, c, replay)
                if m in tables:
                    for j, stage in enumerate(('raw', 'full')):
                        if not np.array_equal(r[stage+'_counts'], tables[m][j][c.config_id, si]):
                            raise RuntimeError('selected replay count mismatch: '+m)
                else:
                    selections.append(dict(subject=subject, selected_method=m, **asdict(c),
                        selection='locked_Gray', inner_F1='', inner_TP='', inner_FP='', inner_FN=''))
                recs[m] = r
            # Exact already replayed single chosen using training subjects only.
            recs['Single_selected'] = dict(recs[choices['Single_selected'][si].method], method='Single_selected')
            for m, r in recs.items():
                counts.setdefault(m, []).append(r['full_counts'])
                out.write(json.dumps(shared.serializable(r), allow_nan=False)+'\n')
                for stage in ('raw', 'full'):
                    per_subject.append(dict(subject=subject, method=m, evaluation=stage,
                                            **shared.metrics(r[stage+'_counts'])))
            for m in recs:
                if m != 'Gray':
                    d = event_delta(recs['Gray'], recs[m])
                    diff.write(json.dumps(shared.serializable(d), allow_nan=False)+'\n')
                    if m in ABLATIONS:
                        mechanisms.append(d)
            traces = recs['Gray']['candidates']
            gray_entered = sum(sum(t['gray_entered']) for t in traces)
            gray_passed = sum(sum(t['gray_passed']) for t in traces)
            shared.write_json(replay/('mechanism_%03d.json' % si), dict(subject=subject,
                branch=configs['Gray'].branch, gray_entered=gray_entered, gray_passed=gray_passed,
                gray_rejected=gray_entered-gray_passed,
                same_as_high_candidates=gray_passed == 0,
                same_as_low_candidates=gray_entered == gray_passed))
            print('selected event replay', si+1, '/', len(context[6]), flush=True)
    counts = {m: np.asarray(a, dtype=np.int64) for m, a in counts.items()}
    counts['Native'] = native[1]
    for si, subject in enumerate(context[6]):
        for j, stage in enumerate(('raw', 'full')):
            per_subject.append(dict(subject=subject, method='Native', evaluation=stage,
                                    **shared.metrics(native[j][si])))
    summary = [dict(method=m, selection='locked_Gray' if m in ABLATIONS else 'none' if m == 'Native' else 'inner_full',
        configurations=len(single_cs) if m == 'Single_selected' else len(all_grids.get(m, [])),
        **shared.metrics(a.sum(0))) for m, a in counts.items()]
    for filename, rows in [('summary_full.csv', summary), ('selected_configs.csv', selections),
                           ('per_subject_counts.csv', per_subject), ('paired_comparisons.csv', paired(counts))]:
        shared.write_csv(output/filename, rows)
    frequency = []
    for field in ('branch', 'reference_scale', 'local_radius', 'tau_low', 'tau_high', 'eta'):
        for value, n in Counter(getattr(c, field) for c in choices['Gray']).items():
            frequency.append(dict(parameter=field, value=value, selected_subjects=n))
    shared.write_csv(output/'selection_frequency.csv', frequency)
    branch_rows = []
    for branch in ('G', 'L'):
        idx = [i for i,c in enumerate(choices['Gray']) if c.branch == branch]
        if idx:
            for m in ('Gray', *ABLATIONS):
                branch_rows.append(dict(main_branch=branch, supporting_branch='L' if branch=='G' else 'G',
                    method=m, subjects=len(idx), **shared.metrics(counts[m][idx].sum(0))))
    shared.write_csv(output/'mechanism_by_branch.csv', branch_rows)
    f = {r['method']: r['F1'] for r in summary}
    module_changes = {m: dict(added_GT=sum(len(d['added_gt_ids']) for d in mechanisms if d['reference'] == m),
        lost_GT=sum(len(d['lost_gt_ids']) for d in mechanisms if d['reference'] == m),
        delta_FP=sum(d['delta_FP'] for d in mechanisms if d['reference'] == m)) for m in ABLATIONS}
    shared.write_json(output/'decision.json', dict(stage='frozen-response development experiment',
        delta_F1={m: f['Gray']-v for m, v in f.items() if m != 'Gray'},
        exceeds_tuned_G_and_L=all(f['Gray'] > f[m] for m in ('G', 'L')),
        exceeds_training_selected_single=f['Gray'] > f['Single_selected'],
        exceeds_both_locked_controls=all(f['Gray'] > f[m] for m in ABLATIONS),
        mechanism_changes=module_changes,
        note='Point criteria only. Examine GT/FP identities, per-branch folds and conditional CIs; not independent confirmation or SOTA proof.'))


def run_setting(ora, context, name, output, mode):
    output.mkdir(exist_ok=True)
    core = FusionCore(context[1])
    core.expected_gt = previous.gt_counts(context)
    all_grids = grids()
    protocol = dict(method=PROTOCOL, evaluation=evaluator.PROTOCOL,
        selection='exclude outer subject; pool training full TP/FP/FN; maximize F1, precision, fewer FP, fixed index',
        structure=dict(reference_scales=SCALES, local_radii=RADII), thresholds=TAUS,
        single_thresholds=SINGLE_TAUS, gray_pairs=PAIRS, etas=ETAS,
        budget={m: len(cs) for m, cs in all_grids.items()}, equal_budget=False,
        branch_selection='joint with a0/rho/low/high/eta on training subjects only',
        eta_zero='not searched; equivalent single endpoint is evaluated by Gray_no_support',
        nonmonotone=True, formula='B>=high OR (low<=B<high AND R>=eta*B)',
        ablations={'Gray_high_only': 'B>=high at locked full parameters', 'Gray_no_support': 'B>=low at locked full parameters'},
        comparison_P8='P8_control is a fresh comparator; unrelated to the user current P8 job',
        scope='frozen-response decoder LOSO development; not nested backbone CV',
        cache=verify_candidate_only_config(context[0]))
    shared.write_json(output/'protocol.json', protocol)
    if not (output/'probe.json').exists():
        cases = []
        for m in all_grids:
            c = all_grids[m][len(all_grids[m])//2]
            cases.append(ledger.detailed_decode(context, core, 0, c))
        c = all_grids['Gray'][len(all_grids['Gray'])//2]
        cases.extend(ledger.detailed_decode(context, core, 0, replace(c, method=m)) for m in ABLATIONS)
        if name == 'metst_sammlv':
            regression = previous.probe_case(context, True)
            if regression['raw'] != [3, 16, 0] or regression['full'] != [3, 15, 0]:
                raise RuntimeError('subject037 one-to-one regression failed')
            shared.write_json(output/'subject037_regression.json', regression)
        shared.write_json(output/'probe.json', dict(passed=True, records=cases, evidence='runtime compatibility only'))
    print(name, 'GRAYZONE_PROBE = PASS', flush=True)
    if mode == 'probe':
        return
    native_path = output/'native_counts.npz'
    if native_path.exists():
        with np.load(native_path, allow_pickle=False) as z:
            if z['subjects'].tolist() != list(context[6]):
                raise RuntimeError('native checkpoint subjects mismatch')
            native = (z['raw'].copy(), z['full'].copy())
    else:
        pairs = []
        for si in range(len(context[6])):
            with contextlib.redirect_stdout(io.StringIO()):
                pairs.append(ora.native_subject_counts('metst', context, si))
        native = tuple(np.asarray([p[j] for p in pairs], dtype=np.int64) for j in (0, 1))
        save_npz(native_path, raw=native[0], full=native[1], subjects=np.asarray(context[6]))
    if any(a.shape != (len(context[6]), 3) for a in native):
        raise RuntimeError('native checkpoint shape mismatch')
    for si, gt in enumerate(core.expected_gt):
        previous.validate(native[0][si], native[1][si], gt)
    configs = [c for cs in all_grids.values() for c in cs]
    raw, full = search(context, core, configs, output)
    tables, offset = {}, 0
    for m, cs in all_grids.items():
        tables[m] = raw[offset:offset+len(cs)], full[offset:offset+len(cs)]
        offset += len(cs)
    report(context, core, all_grids, tables, native, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=('sammlv', 'casme3'), required=True)
    parser.add_argument('--mode', choices=('probe', 'full'), default='full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ora, _ = shared.load_helper()
    name, output = shared.SETTINGS[args.setting], args.output.resolve()
    spec = ora.SPECS[name]
    for s in ora.SPECS.values():
        for key in ('dump', 'cache', 'evidence', 'run_evidence', 'results', 'locked_results'):
            if key in s and phase.overlaps(output, Path(s[key]).resolve()):
                raise RuntimeError('output overlaps sealed input')
    allowed = Path('/content/drive/MyDrive/GLSD_GRAYZONE_SUPPORT').resolve()
    if output == allowed or allowed not in output.parents:
        raise RuntimeError('output must be a new child of GLSD_GRAYZONE_SUPPORT; protects existing P8 runs')
    ora.verify_sealed_inputs(spec)
    paths = [p for p in ora.required_paths(spec) if p.is_file()]
    paths += sorted(p for p in spec['dump'].rglob('*') if p.is_file())
    import os
    checkout = Path(os.environ.get('ME_TST_ROOT', '/content/ME-TST'))
    if not checkout.is_dir():
        raise RuntimeError('restore the existing ME-TST checkout first')
    paths += sorted(p for p in checkout.rglob('*.py') if '.git' not in p.parts)
    identity = dict(protocol=PROTOCOL, evaluation=evaluator.PROTOCOL, setting=name, mode=args.mode,
        input_sha256={str(p): ora.sha256(p) for p in paths},
        code_sha256={p.name: ora.sha256(p) for p in Path(__file__).parent.glob('*.py')},
        runtime=dict(python=sys.version, numpy=np.__version__, pandas=getattr(sys.modules.get('pandas'), '__version__', None)))
    if output.exists():
        if not args.resume or not (output/'run_manifest.json').exists():
            raise RuntimeError('output exists; use explicit resume or a new output')
        if json.loads((output/'run_manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('resume code/input/runtime/protocol mismatch')
    else:
        if args.resume:
            raise RuntimeError('resume directory missing')
        output.mkdir(parents=True)
        shared.write_json(output/'run_manifest.json', dict(identity=identity))
    print('OUTPUT =', output, flush=True)
    preflight = output/'legacy_preflight'
    preflight.mkdir(exist_ok=True)
    if not (preflight/'locked_replay.json').exists():
        shared.write_json(preflight/'locked_replay.json', ora.run_preflight_setting(name, preflight))
    context = ora.metst_context(spec)  # native historical gate BEFORE patching aggregate evaluator
    if len(set(context[6])) != spec['subjects'] or ora.context_video_and_gt_counts('metst', context) != (spec['videos'], spec['gt']):
        raise RuntimeError('official population mismatch; CASME3 must contain 853 GT')
    with evaluator.install(context[2]) as info:
        video_class = context[3].spotting.__globals__['MeanAveragePrecision2d']
        info['video_evaluator_is_aggregate'] = video_class is context[2]
        info['adapter_target'] = 'aggregate evaluator; spotting video binding retained'
        info['source_hashes'] = {str(p): ora.sha256(p) for p in set(
            [Path(inspect.getsourcefile(x)) for x in (context[2], video_class, context[3])])}
        matching_path = output/'matching_protocol.json'
        if matching_path.exists() and json.loads(matching_path.read_text()) != info:
            raise RuntimeError('official evaluator/recognition source changed')
        shared.write_json(matching_path, info)
        if (output/'completion.json').exists():
            print('Already completed and identity verified:', output, flush=True)
            return
        run_setting(ora, context, name, output/name, args.mode)
    shared.write_json(output/'completion.json', dict(completed=True, mode=args.mode,
        setting=name, protocol=PROTOCOL, evaluation=evaluator.PROTOCOL))
    print('GRAYZONE_%s = PASS' % args.mode.upper(), flush=True)


if __name__ == '__main__':
    main()
