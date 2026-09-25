"""Joint p/a0/rho/tau continuation on the existing corrected frozen responses.

Reuse p=1/8 and baseline counts; compute only missing configurations. The
existing aggregate evaluator adapter and dual-evaluator event observer are
imported unchanged. No backbone training or new feature definition.
"""
import argparse
import copy
import csv
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

import run_p8_threshold_control as prior

phase, shared, evaluator = prior.phase, prior.shared, prior.evaluator
PROTOCOL = 'gm_joint_p_structure_v1'
POWERS = ('1', '2', '4', '8', 'inf')


@dataclass(frozen=True)
class Config(phase.Config):
    p: str = '8'


def factor(p):
    return 1.0 if p == 'inf' else 2.0 ** (1.0 / float(p))


def grids():
    old = phase.grids()
    result = {m: [Config(**asdict(c), p='1' if m == 'Mean' else '8') for c in old[m]]
              for m in ('G', 'L', 'Mean')}
    result['P8_rho1'] = [Config('P8_rho1', i, c.reference_scale, c.local_radius, c.threshold)
                         for i, c in enumerate(c for c in old['GM_p8'] if c.local_radius == 1)]
    result['G_rescaled_p8'] = [replace(c, method='G_rescaled_p8') for c in result['G']]
    for p in ('1', '2', '4', 'inf'):
        result['G_rescaled_p' + p] = [replace(c, method='G_rescaled_p' + p, p=p)
                                     for c in result['G']]
    result['GM_joint'] = [Config('GM_joint', i, a, r, t, p) for i, (p, a, r, t) in enumerate(
        (p, a, r, t) for p in POWERS for a in phase.SCALES for r in phase.RADII for t in phase.TAUS)]
    # Original thresholds first, then each p's effective G thresholds. Exact
    # float identity: no approximate deduplication and no clipping above one.
    union = []
    seen = set()
    for p in ('inf', '1', '2', '4', '8'):
        for a in phase.SCALES:
            for t in phase.TAUS:
                threshold = factor(p) * t
                if (a, threshold) not in seen:
                    seen.add((a, threshold))
                    union.append(Config('G_union', len(union), a, 1.0, threshold, 'inf'))
    result['G_union'] = union
    return result


def score_keep(values, c):
    values = np.asarray(values, dtype=float)
    # Preserve the existing finite/nonnegative feature validation.
    phase.score(values, 'G')
    g, l = values.T
    if c.method in ('G', 'G_union'):
        scores = g
    elif c.method == 'L':
        scores = l
    elif c.method.startswith('G_rescaled_p') or c.method in ('GM_no_L', 'GM_no_G'):
        p = c.method.removeprefix('G_rescaled_p') if c.method.startswith('G_rescaled_p') else c.p
        branch = l if c.method == 'GM_no_G' else g
        return branch / factor(p), branch >= factor(p) * c.threshold
    elif c.method in ('Mean', 'P8_rho1', 'GM_joint'):
        p = '1' if c.method == 'Mean' else '8' if c.method == 'P8_rho1' else c.p
        scores = shared.generalized_mean(g, l, float(p))
    else:
        raise ValueError(c.method)
    return scores, scores >= c.threshold


class FeatureView(phase.FeatureView):
    def selected_peaks(self, config):
        peaks, values = self.evidence(config.reference_scale, config.local_radius)
        scores, keep = score_keep(values, config)
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.key[0],
                peaks=peaks.tolist(), G=values[:, 0].tolist(), L=values[:, 1].tolist(),
                score=scores.tolist(), retained=keep.tolist()))
        return peaks[keep]


class FusionCore(prior.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def load_counts(path, configs, subjects, gts):
    with np.load(path, allow_pickle=False) as z:
        if json.loads(z['configs'].item()) != [asdict(c) for c in configs]:
            raise RuntimeError('reused configuration identity mismatch: ' + str(path))
        if z['subjects'].tolist() != list(subjects) or not z['done'].all():
            raise RuntimeError('reused subject/completion mismatch: ' + str(path))
        arrays = tuple(z[k].copy() for k in ('raw', 'full'))
    for a in arrays:
        if a.shape != (len(configs), len(subjects), 3) or not np.issubdtype(a.dtype, np.integer):
            raise RuntimeError('invalid reused counts shape/dtype')
        if np.any(a < 0) or not np.all(a[:, :, 0] + a[:, :, 2] == np.asarray(gts)):
            raise RuntimeError('invalid reused counts/GT conservation')
    return arrays


def load_reuse(old, threshold, name, subjects, gts):
    tables = {m: load_counts(old/name/('search_counts_' + m + '.npz'), cs, subjects, gts)
              for m, cs in phase.grids().items()}
    # The latest completed threshold-control run declares exactly which old
    # tensor bytes it reused; identity verification in main checks those hashes.
    mapped = load_counts(threshold/name/'search_counts_G_rescaled.npz',
                         prior.grids()['G_rescaled'], subjects, gts)
    for m in ('G', 'L', 'Mean', 'GM_p8'):
        latest = load_counts(threshold/name/('search_counts_' + m + '.npz'),
                             prior.grids()[m], subjects, gts)
        old_arrays = tables[m]
        if m == 'GM_p8':
            ix = [i for i, c in enumerate(phase.grids()[m]) if c.local_radius == 1]
            old_arrays = tuple(a[ix] for a in old_arrays)
        if any(not np.array_equal(a, b) for a, b in zip(latest, old_arrays)):
            raise RuntimeError('old/latest reused counts disagree: ' + m)
    tables['G_rescaled_p8'] = mapped
    return tables


def save_counts(path, configs, arrays, subjects):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, raw=arrays[0], full=arrays[1], done=np.ones(len(subjects), bool),
                            subjects=np.asarray(subjects, dtype=str),
                            configs=np.asarray(json.dumps([asdict(c) for c in configs], sort_keys=True)))
    temporary.replace(path)


def g_reuse_map(old_tables):
    reused = {}
    for c in phase.grids()['G']:
        reused[c.reference_scale, c.threshold] = tuple(a[c.config_id] for a in old_tables['G'])
        key = c.reference_scale, factor('8') * c.threshold
        value = tuple(a[c.config_id] for a in old_tables['G_rescaled_p8'])
        if key in reused and any(not np.array_equal(a, b) for a, b in zip(reused[key], value)):
            raise RuntimeError('equivalent G thresholds have inconsistent saved counts')
        reused[key] = value
    return reused


def assemble_counts(context, core, old_tables, all_grids, output):
    subjects = context[6]
    tables = {m: old_tables[m] for m in ('G', 'L', 'Mean', 'G_rescaled_p8')}
    ix = [i for i, c in enumerate(phase.grids()['GM_p8']) if c.local_radius == 1]
    tables['P8_rho1'] = tuple(a[ix] for a in old_tables['GM_p8'])
    pieces = []
    for p in POWERS:
        if p in ('1', '8'):
            pieces.append(old_tables['Mean' if p == '1' else 'GM_p8'])
            continue
        configs = [replace(c, config_id=i) for i, c in enumerate(c for c in all_grids['GM_joint'] if c.p == p)]
        core.capture = False
        pieces.append(phase.search_counts(context, core, configs, output/('new_counts_p' + p + '.npz')))
    tables['GM_joint'] = tuple(np.concatenate([part[j] for part in pieces], axis=0) for j in (0, 1))
    reused = g_reuse_map(old_tables)
    missing = [c for c in all_grids['G_union'] if (c.reference_scale, c.threshold) not in reused]
    if missing:
        core.capture = False
        new = phase.search_counts(context, core, [replace(c, config_id=i) for i, c in enumerate(missing)],
                                  output/'new_counts_G_union.npz')
        for i, c in enumerate(missing):
            reused[c.reference_scale, c.threshold] = tuple(a[i] for a in new)
    tables['G_union'] = tuple(np.stack([reused[c.reference_scale, c.threshold][j]
                                      for c in all_grids['G_union']]) for j in (0, 1))
    for p in ('1', '2', '4', 'inf'):
        method = 'G_rescaled_p' + p
        tables[method] = tuple(np.stack([reused[c.reference_scale, factor(p)*c.threshold][j]
                                        for c in all_grids[method]]) for j in (0, 1))
    for m, cs in all_grids.items():
        for a in tables[m]:
            if a.shape != (len(cs), len(subjects), 3) or np.any(a < 0):
                raise RuntimeError('assembled count shape/nonnegative check failed: ' + m)
            if not np.all(a[:, :, 0] + a[:, :, 2] == np.asarray(core.expected_gt)):
                raise RuntimeError('assembled GT conservation failed: ' + m)
        save_counts(output/('search_counts_' + m + '.npz'), cs, tables[m], subjects)
    return tables


def event_signature(c):
    """Equivalent scorer identity, retaining rho so recorded L stays correct."""
    p = getattr(c, 'p', '8')
    method, t = c.method, c.threshold
    if method in ('G', 'G_union', 'G_expanded'):
        scorer = 'G'
    elif method.startswith('G_rescaled_p'):
        scorer = 'G'
        t *= factor(method.removeprefix('G_rescaled_p'))
    elif method in ('G_rescaled', 'p8_no_L', 'GM_no_L'):
        scorer = 'G'
        t *= factor(p if method == 'GM_no_L' else '8')
    elif method in ('p8_no_G', 'GM_no_G'):
        scorer = 'L'
        t *= factor(p if method == 'GM_no_G' else '8')
    elif method == 'L':
        scorer = 'L'
    else:
        scorer = ('GM', '1' if method == 'Mean' else p)
    return c.reference_scale, c.local_radius, scorer, t


def relabel_record(record, config):
    """Reuse an equivalent decoded event set, recomputing displayed scores."""
    result = copy.deepcopy(record)
    result.update(method=config.method, config=asdict(config))
    for trace in result['candidates']:
        scores, keep = score_keep(np.column_stack((trace['G'], trace['L'])), config)
        if keep.tolist() != trace['retained']:
            raise RuntimeError('cached event candidate selection mismatch')
        trace['score'] = scores.tolist()
    return result


def load_event_cache(path):
    cache = {}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for line in f:
            r = json.loads(line)
            c = Config(**r['config'])
            key = r['subject'], event_signature(c)
            if key in cache:
                for field in ('raw_counts', 'full_counts', 'events', 'ground_truth'):
                    if cache[key][field] != r[field]:
                        raise RuntimeError('equivalent saved event records disagree: ' + field)
            cache[key] = r
    return cache


def run_probe(context, core, old_tables, output):
    subjects = context[6]
    # One representative subject, with all new scorer branches and both
    # inherited endpoints. Full runs perform this once before filling holes.
    si = 0
    records = []
    for method, power in (('Mean', '1'), ('GM_p8', '8')):
        winner, _ = shared.choose(old_tables[method][1], si)
        base = phase.grids()[method][winner]
        c = Config('GM_joint', winner, base.reference_scale, base.local_radius, base.threshold, power)
        r = prior.detailed_decode(context, core, si, c)
        for j, stage in enumerate(('raw', 'full')):
            if not np.array_equal(r[stage + '_counts'], old_tables[method][j][winner, si]):
                raise RuntimeError('endpoint full replay failed: p=' + power)
        records.append(r)
    anchor = Config('GM_joint', 0, 2.0, 2.0, .55, '2')
    for power in ('2', '4', 'inf'):
        c = replace(anchor, p=power)
        for method in ('GM_joint', 'GM_no_L', 'GM_no_G'):
            records.append(prior.detailed_decode(context, core, si, replace(c, method=method)))
    mapping_checks = 0
    for response in context[4][si]['result_all']:
        feature = core.GLSDFeatures(response, context[4][si]['k_p'])
        for power in POWERS:
            for c in grids()['G_rescaled_p' + power]:
                direct = replace(c, method='G_union', threshold=factor(power)*c.threshold)
                if not np.array_equal(feature.selected_peaks(c), feature.selected_peaks(direct)):
                    raise RuntimeError('rescaled/mapped G candidate mismatch')
                mapping_checks += 1
    with gzip.open(output/'probe_event_records.jsonl.gz', 'wt', encoding='utf8') as f:
        for r in records:
            f.write(json.dumps(shared.serializable(r), allow_nan=False) + '\n')
    shared.write_json(output/'probe.json', dict(passed=True, subject=subjects[si],
        endpoint_replays=2, new_branch_replays=9, mapped_G_checks=mapping_checks, event_counts_verified=True,
        evaluator_observer='unchanged threshold-control dual-class add observer'))
    print('JOINT_SCORER_EVENT_PROBE = PASS', flush=True)
    return records


def paired(counts):
    aliases = {('GM_p8' if m == 'GM_joint' else m): a for m, a in counts.items()}
    rows = prior.paired(aliases)
    for r in rows:
        r['compared'] = 'GM_joint'
    return rows


def run_setting(context, old, threshold, name, output, mode):
    output.mkdir(exist_ok=True)
    subjects = context[6]
    core = FusionCore(context[1])
    core.expected_gt = prior.previous.gt_counts(context)
    old_tables = load_reuse(old, threshold, name, subjects, core.expected_gt)
    all_grids = grids()
    g_missing = sum((c.reference_scale, c.threshold) not in g_reuse_map(old_tables) for c in all_grids['G_union'])
    shared.write_json(output/'protocol.json', dict(protocol=PROTOCOL, evaluation_protocol=evaluator.PROTOCOL,
        stage='development after prior outer-result inspection; no independent confirmation claim',
        p=POWERS, scales=phase.SCALES, radii=phase.RADII, thresholds=phase.TAUS,
        selection='p,a0,rho,tau chosen jointly on pooled inner full counts excluding outer subject',
        tie_break=['F1 descending', 'precision descending', 'FP ascending', 'fixed configuration index'],
        p_order=POWERS, configurations={m: [asdict(c) for c in cs] for m, cs in all_grids.items()},
        ablations='GM_no_L / GM_no_G lock all complete-method parameters, no reselection',
        G_control='union of original T and 2^(1/p) T for every p, exact deduplication, no clipping',
        budgets={m: len(cs) for m, cs in all_grids.items()}, budgets_are_equal=False,
        new_fusion_configs=513, new_G_configs=g_missing, old_fusion_configs_reused=342,
        decoding='existing aggregate adapter only; existing separate video evaluator unchanged',
        old_source=str(old), threshold_source=str(threshold)))
    print(name, '855 fusion configs: 342 reused, 513 missing; G missing:', g_missing, flush=True)
    # Successful probe is persisted under the run identity, so a resumed run
    # does not repeat it. It is not a request for a new user approval.
    if not (output/'probe.json').exists():
        run_probe(context, core, old_tables, output)
    elif not json.loads((output/'probe.json').read_text())['passed']:
        raise RuntimeError('previous probe did not pass')
    if mode == 'probe':
        return
    tables = assemble_counts(context, core, old_tables, all_grids, output)
    cache = load_event_cache(threshold/name/'event_records.jsonl.gz')
    replay_dir = output/'selected_events'
    replay_dir.mkdir(exist_ok=True)
    chosen, selected_rows = {}, []
    for m, cs in all_grids.items():
        chosen[m] = []
        for si, subject in enumerate(subjects):
            wi, pooled = shared.choose(tables[m][1], si)
            chosen[m].append(cs[wi])
            selected_rows.append(dict(subject=subject, **asdict(cs[wi]),
                                      inner_F1=shared.metrics(pooled[wi])['F1'], selection='inner_full'))
    counts = {m: [] for m in (*all_grids, 'GM_no_L', 'GM_no_G')}
    per_subject, contributions = [], []
    with gzip.open(output/'event_records.jsonl.gz', 'wt', encoding='utf8') as ledger, \
         gzip.open(output/'event_differences.jsonl.gz', 'wt', encoding='utf8') as differences:
        for si, subject in enumerate(subjects):
            configs = {m: cs[si] for m, cs in chosen.items()}
            configs.update({m: replace(configs['GM_joint'], method=m) for m in ('GM_no_L', 'GM_no_G')})
            recs = {}
            for m, c in configs.items():
                path = replay_dir/('%03d_%s.json.gz' % (si, m))
                key = subject, event_signature(c)
                if path.exists():
                    with gzip.open(path, 'rt', encoding='utf8') as f:
                        r = json.load(f)
                    if r['subject'] != subject or r['config'] != asdict(c):
                        raise RuntimeError('selected event checkpoint identity mismatch')
                elif key in cache:
                    r = relabel_record(cache[key], c)
                    r['record_origin'] = 'equivalent_saved_event_record'
                else:
                    r = prior.detailed_decode(context, core, si, c)
                    r['record_origin'] = 'official_selected_replay'
                if m in tables:
                    for j, stage in enumerate(('raw', 'full')):
                        if not np.array_equal(r[stage + '_counts'], tables[m][j][c.config_id, si]):
                            raise RuntimeError('selected event count mismatch: ' + m)
                if not path.exists():
                    tmp = path.with_suffix('.tmp')
                    with gzip.open(tmp, 'wt', encoding='utf8') as f:
                        json.dump(shared.serializable(r), f, allow_nan=False)
                    tmp.replace(path)
                cache[key] = r
                recs[m] = r
                counts[m].append(r['full_counts'])
                ledger.write(json.dumps(shared.serializable(r), allow_nan=False) + '\n')
                if m.startswith('GM_no'):
                    selected_rows.append(dict(subject=subject, **asdict(c), inner_F1='', selection='locked_GM_joint'))
                for stage in ('raw', 'full'):
                    per_subject.append(dict(setting=name, subject=subject, method=m, evaluation=stage,
                                            **shared.metrics(r[stage + '_counts'])))
            for m in configs:
                if m != 'GM_joint':
                    delta = prior.event_delta(recs['GM_joint'], recs[m])
                    differences.write(json.dumps(delta) + '\n')
                    if m.startswith('GM_no'):
                        contributions.append(delta)
            print('selected event ledger', name, si + 1, '/', len(subjects), flush=True)
    counts = {m: np.asarray(a, dtype=np.int64) for m, a in counts.items()}
    summaries = [dict(setting=name, method=m, selection='locked_GM_joint' if m.startswith('GM_no') else 'inner_full',
                      configs=len(all_grids[m]) if m in all_grids else 0, **shared.metrics(a.sum(0)))
                 for m, a in counts.items()]
    with (old/(name + '_audit')/'native_per_subject_counts.csv').open() as f:
        native = [r for r in csv.DictReader(f) if r['evaluation'] == 'full']
    if [r['subject'] for r in native] != list(subjects):
        raise RuntimeError('native subject identity mismatch')
    native_counts = np.asarray([[int(r[k]) for k in ('TP', 'FP', 'FN')] for r in native])
    for i, a in enumerate(native_counts):
        prior.previous.validate(a, a, core.expected_gt[i])
    summaries.append(dict(setting=name, method='Native', selection='none', configs=0,
                          **shared.metrics(native_counts.sum(0))))
    for filename, rows in [('summary_full.csv', summaries), ('selected_configs.csv', selected_rows),
                           ('per_subject_counts.csv', per_subject), ('paired_comparisons.csv', paired(counts))]:
        shared.write_csv(output/filename, rows)
    f1 = {r['method']: r['F1'] for r in summaries}
    shared.write_json(output/'decision.json', dict(stage='development',
        delta_F1={m: f1['GM_joint'] - f for m, f in f1.items() if m != 'GM_joint'},
        exceeds_independent_controls=all(f1['GM_joint'] > f1[m] for m in
            ('G', 'L', 'Mean', 'G_union', *('G_rescaled_p' + p for p in POWERS))),
        exceeds_both_locked_ablations=all(f1['GM_joint'] > f1[m] for m in ('GM_no_L', 'GM_no_G')),
        module_event_changes={m: dict(added_GT=sum(len(r['added_gt_ids']) for r in contributions if r['reference'] == m),
                                     lost_GT=sum(len(r['lost_gt_ids']) for r in contributions if r['reference'] == m),
                                     delta_FP=sum(r['delta_FP'] for r in contributions if r['reference'] == m))
                              for m in ('GM_no_L', 'GM_no_G')},
        uncertainty='paired bootstrap conditional on selected predictions; no reselection or exploration correction'))


def verify_manifests(ora, old, threshold, name, runtime, input_hashes):
    old_manifest = json.loads((old/'run_manifest.json').read_text())['identity']
    latest = json.loads((threshold/'run_manifest.json').read_text())['identity']
    for root in (old, threshold):
        c = json.loads((root/'completion.json').read_text())
        if not c['completed'] or c['mode'] != 'full':
            raise RuntimeError('reuse requires completed full outputs: ' + str(root))
    if (old_manifest['protocol'] != evaluator.PROTOCOL or latest['protocol'] != prior.PROTOCOL
            or latest['evaluation'] != evaluator.PROTOCOL or latest['setting'] != name):
        raise RuntimeError('reuse protocol/setting mismatch')
    if old_manifest['runtime'] != runtime or latest['runtime'] != runtime:
        raise RuntimeError('runtime differs from saved counts; restore previous Colab runtime')
    if old_manifest['inputs'][name] != input_hashes or latest['inputs'] != input_hashes:
        raise RuntimeError('frozen input inventory/hash changed')
    for mapping in (old_manifest['code_sha256'], latest['code']):
        for filename, digest in mapping.items():
            if ora.sha256(Path(__file__).with_name(filename)) != digest:
                raise RuntimeError('inherited source hash mismatch: ' + filename)
    # Validate every relevant old result referenced by the latest manifest.
    for original, digest in latest['reuse'].items():
        p = Path(original)
        if p.name in ('run_manifest.json', 'completion.json'):
            target = old/p.name
        elif p.parent.name in (name, name + '_audit'):
            target = old/p.parent.name/p.name
        else:
            continue
        if ora.sha256(target) != digest:
            raise RuntimeError('old/latest provenance hash mismatch: ' + str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('probe', 'full'), default='full')
    p.add_argument('--setting', choices=('sammlv', 'casme3'), required=True)
    p.add_argument('--reuse', type=Path, required=True)
    p.add_argument('--threshold-reuse', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    ora, _ = shared.load_helper()
    name = shared.SETTINGS[args.setting]
    old, threshold, output = (x.resolve() for x in (args.reuse, args.threshold_reuse, args.output))
    protected = [old, threshold]
    for spec in ora.SPECS.values():
        protected += [Path(spec[k]).resolve() for k in ('dump', 'cache', 'evidence', 'run_evidence', 'results', 'locked_results') if k in spec]
    if any(phase.overlaps(output, root) for root in protected):
        raise RuntimeError('output overlaps existing inputs/results')
    spec = ora.SPECS[name]
    ora.verify_sealed_inputs(spec)
    inputs = [spec['source'], spec['core'], spec['evidence']/'evidence_bundle_sha256.txt', spec['run_evidence']/'run_manifest.json']
    inputs += sorted(x for x in spec['dump'].rglob('*') if x.is_file())
    hashes = {str(x): ora.sha256(x) for x in inputs}
    runtime = dict(python=sys.version, numpy=np.__version__, pandas=getattr(sys.modules.get('pandas'), '__version__', None))
    verify_manifests(ora, old, threshold, name, runtime, hashes)
    reuse_files = [old/'run_manifest.json', old/'completion.json', old/(name + '_audit')/'matching_protocol.json',
                   old/(name + '_audit')/'native_per_subject_counts.csv', threshold/'run_manifest.json',
                   threshold/'completion.json', threshold/'matching_protocol.json', threshold/name/'event_records.jsonl.gz']
    reuse_files += [old/name/('search_counts_' + m + '.npz') for m in phase.METHODS]
    reuse_files += [threshold/name/('search_counts_' + m + '.npz') for m in (*phase.METHODS, 'G_rescaled')]
    identity = dict(protocol=PROTOCOL, evaluation=evaluator.PROTOCOL, mode=args.mode, setting=name, runtime=runtime,
                    inputs=hashes, reuse={str(x): ora.sha256(x) for x in reuse_files},
                    code={x.name: ora.sha256(x) for x in Path(__file__).parent.glob('*.py')})
    if output.exists():
        if not args.resume or not (output/'run_manifest.json').is_file():
            raise RuntimeError('use new output or explicit resume')
        if json.loads((output/'run_manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('resume identity mismatch: code/input/runtime/protocol/reuse')
        if (output/'completion.json').exists():
            print('Already completed:', output, flush=True)
            return
    else:
        if args.resume:
            raise RuntimeError('resume directory missing')
        output.mkdir(parents=True)
        shared.write_json(output/'run_manifest.json', dict(identity=identity))
    print('OUTPUT =', output, flush=True)
    context = ora.metst_context(spec)
    with evaluator.install(context[2]) as info:
        info['metric_source_sha256'] = ora.sha256(Path(inspect.getsourcefile(context[2])))
        for path in (old/(name + '_audit')/'matching_protocol.json', threshold/'matching_protocol.json'):
            previous = json.loads(path.read_text())
            for key in ('protocol', 'metric_source_sha256', 'legacy_check_box_sha256'):
                if previous[key] != info[key]:
                    raise RuntimeError('evaluator identity changed: ' + key)
        shared.write_json(output/'matching_protocol.json', info)
        run_setting(context, old, threshold, name, output/name, args.mode)
    shared.write_json(output/'completion.json', dict(completed=True, protocol=PROTOCOL, mode=args.mode, setting=name))
    print('JOINT_GM_%s = PASS' % args.mode.upper(), flush=True)


if __name__ == '__main__':
    main()
