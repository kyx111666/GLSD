"""ME-TST+ development experiment: tune structure/tau with p fixed to eight.

Frozen-response post-processing LOSO only. Existing backbone/checkpoint and k
provenance is inherited; this does not establish nested backbone training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

import run_generalized_mean_screening as shared

SCALES = (1.0, 1.5, 2.0)
RADII = (0.5, 1.0, 2.0)
TAUS = tuple(i / 20 for i in range(1, 20))
METHODS = ("G", "L", "Mean", "GM_p8")


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    reference_scale: float
    local_radius: float
    threshold: float

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def grids():
    result = {}
    for method in METHODS:
        # rho cannot change G; do not pad its search with duplicate configs.
        radii = (1.0,) if method == "G" else RADII
        tuples = [(a, r, t) for a in SCALES for r in radii for t in TAUS]
        result[method] = [Config(method, i, *values) for i, values in enumerate(tuples)]
    return result


def score(values, method):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("evidence must have shape (candidates, 2)")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1 + 1e-12):
        raise ValueError("G/L must be finite and nonnegative, at most 1+1e-12; no clipping")
    g, l = values.T
    if method == "G":
        return g
    if method == "L":
        return l
    if method == "Mean":
        return (g + l) / 2.0
    if method == "GM_p8":
        return shared.generalized_mean(g, l, 8)
    raise ValueError(method)


class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evidence(self, scale, radius):
        key = (scale, radius)
        if key not in self.cache:
            peaks, values = self.base.evidence(scale, radius)
            peaks = np.asarray(peaks, dtype=int).copy()
            values = np.asarray(values, dtype=float).copy()
            mean = score(values, "Mean")
            if len(peaks) != len(values):
                raise RuntimeError("candidate/evidence length mismatch")
            for tau in TAUS:
                legacy = SimpleNamespace(reference_scale=scale, local_radius=radius, threshold=tau)
                if not np.array_equal(peaks[mean >= tau], self.base.selected_peaks(legacy)):
                    raise RuntimeError("arithmetic mean hook does not exactly replay sealed core")
            for (other_scale, _), (old_peaks, old_values) in self.cache.items():
                if other_scale == scale:
                    if not np.array_equal(peaks, old_peaks) or not np.array_equal(values[:, 0], old_values[:, 0]):
                        raise RuntimeError("G/candidate pool depends on rho; G-only grid is invalid")
            peaks.setflags(write=False)
            values.setflags(write=False)
            self.cache[key] = (peaks, values)
        return self.cache[key]

    def selected_peaks(self, config):
        peaks, values = self.evidence(config.reference_scale, config.local_radius)
        scores = score(values, config.method)
        keep = scores >= config.threshold
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(
                response_sha256=self.key[2], k=self.key[0],
                feature_call_index=len(self.owner.trace),
                peaks=peaks.tolist(), G=values[:, 0].tolist(), L=values[:, 1].tolist(),
                score=scores.tolist(), retained=keep.tolist(),
                provenance="subject plus response hash and peak; no verified video/event match IDs",
            ))
        return peaks[keep]


class FusionCore(shared.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def checked_decode(context, core, subject_index, config, capture=False):
    raw, full, predictions, traces = shared.decode(context, core, subject_index, config, True, capture)
    for stage, counts in (("raw", raw), ("full", full)):
        if counts is None or len(counts) != 3 or min(counts) < 0:
            raise RuntimeError("invalid official %s counts: %s" % (stage, counts))
        if hasattr(core, 'expected_gt'):
            gt = core.expected_gt[subject_index]
            if counts[0] + counts[2] != gt or counts[0] > gt:
                raise RuntimeError("%s GT conservation failed: counts=%s, GT=%s" % (stage, counts, gt))
    return raw, full, predictions, traces


def search_counts(context, core, configs, path):
    """Atomic subject checkpoints allow --resume without reusing phase-1 stats."""
    subjects = context[6]
    identity = json.dumps([asdict(c) for c in configs], sort_keys=True)
    shape = (len(configs), len(subjects), 3)
    raw, full = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    done = np.zeros(len(subjects), dtype=bool)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved['configs'].item() != identity or saved['subjects'].tolist() != list(subjects):
                raise RuntimeError("checkpoint configs/subjects mismatch")
            raw, full, done = saved['raw'], saved['full'], saved['done']
        if raw.shape != shape or full.shape != shape or done.shape != (len(subjects),):
            raise RuntimeError("checkpoint shape mismatch")
    for si, subject in enumerate(subjects):
        if done[si]:
            continue
        for config in configs:
            try:
                r, f, _, _ = checked_decode(context, core, si, config)
            except Exception as exc:
                raise RuntimeError("subject=%s config=%s" % (subject, config.identifier)) from exc
            raw[config.config_id, si], full[config.config_id, si] = r, f
        done[si] = True
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, raw=raw, full=full, done=done,
                                subjects=np.asarray(subjects, dtype=str), configs=np.asarray(identity))
        tmp.replace(path)
        print("  %s: %d/%d subjects saved" % (configs[0].method, si + 1, len(subjects)), flush=True)
    return raw, full


def bootstrap(name, counts, subjects):
    draws = np.random.default_rng(100).integers(0, len(subjects), size=(10000, len(subjects)))
    def sampled(values):
        sums = values[draws].sum(axis=1)
        d = 2 * sums[:, 0] + sums[:, 1] + sums[:, 2]
        return np.divide(2 * sums[:, 0], d, out=np.zeros(len(draws)), where=d > 0)
    fusion = sampled(counts['GM_p8'])
    rows = []
    for reference in ('G', 'L', 'Mean'):
        delta = fusion - sampled(counts[reference])
        rows.append(dict(setting=name, compared='GM_p8', reference=reference,
                         delta_F1=shared.metrics(counts['GM_p8'].sum(0))['F1'] - shared.metrics(counts[reference].sum(0))['F1'],
                         CI_low=float(np.quantile(delta, .025)), CI_high=float(np.quantile(delta, .975)),
                         resamples=10000, seed=100, unit='subject', evaluation='full'))
    return rows


def run_setting(ora, name, output, prepared_context=None, evaluation_protocol='legacy_official'):
    output.mkdir(exist_ok=True)
    context = prepared_context if prepared_context is not None else ora.metst_context(ora.SPECS[name])
    subjects = context[6]
    if len(subjects) != len(set(subjects)):
        raise RuntimeError('duplicate subject IDs')
    core = FusionCore(context[1])
    if evaluation_protocol != 'legacy_official':
        core.expected_gt = [sum(len(v) for v in videos) for videos in context[0].final_samples(context[4])]
        if len(core.expected_gt) != len(subjects):
            raise RuntimeError('GT/subject cardinality mismatch')
    summary, per_subject, selected, search, boundaries = [], [], [], [], []
    full_by_method = {}
    all_grids = grids()
    shared.write_json(output / 'protocol.json', dict(
        stage='development; p=8 chosen after inspecting phase-1 results',
        evaluation_protocol=evaluation_protocol,
        scales=SCALES, radii=RADII, thresholds=TAUS, p=8,
        methods={m: [asdict(c) for c in cs] for m, cs in all_grids.items()},
        primary_selector='pooled full counts excluding outer subject',
        secondary_selector='pooled raw counts excluding outer subject',
        reported_metric='outer full counts; official recognition/result synergy',
        tie_break=['F1 descending', 'precision descending', 'FP ascending', 'fixed grid index'],
        scope='post-processing LOSO on existing official frozen responses; upstream provenance inherited',
        search_budget='G:57; L/Mean/GM_p8:171 each. Not a matched-budget comparison.',
    ))
    with gzip.open(output / 'selected_predictions.jsonl.gz', 'wt', encoding='utf-8') as preds, \
         gzip.open(output / 'candidate_trace.jsonl.gz', 'wt', encoding='utf-8') as trace_file:
        for method, configs in all_grids.items():
            print('%s %s: %d configs x %d subjects (full evaluation)' % (name, method, len(configs), len(subjects)), flush=True)
            raw, full = search_counts(context, core, configs, output / ('search_counts_%s.npz' % method))
            method_full = []
            for selector, table in (('full', full), ('raw', raw)):
                totals = {'raw': np.zeros(3, dtype=np.int64), 'full': np.zeros(3, dtype=np.int64)}
                for outer, subject in enumerate(subjects):
                    winner, pooled = shared.choose(table, outer)
                    config = configs[winner]
                    r, f = tuple(raw[winner, outer]), tuple(full[winner, outer])
                    if selector == 'full':
                        rr, ff, predictions, traces = checked_decode(context, core, outer, config, capture=True)
                        if r != rr or f != ff:
                            raise RuntimeError('selected configuration failed exact raw/full replay')
                        method_full.append(f)
                        record = dict(setting=name, method=method, subject=subject, config=asdict(config),
                                      raw_counts=r, full_counts=f, predictions=predictions)
                        preds.write(json.dumps(shared.serializable(record), allow_nan=False) + '\n')
                        for trace in traces:
                            trace_file.write(json.dumps(dict(setting=name, subject=subject, method=method,
                                                             config=asdict(config), **trace), allow_nan=False) + '\n')
                    selected.append(dict(setting=name, selector=selector, subject=subject, **asdict(config),
                                         inner_subjects=len(subjects)-1, inner_F1=shared.metrics(pooled[winner])['F1']))
                    for c, inner in zip(configs, pooled):
                        search.append(dict(setting=name, selector=selector, outer_subject=subject,
                                           **asdict(c), selected=int(c.config_id == winner), **shared.metrics(inner)))
                    for stage, counts in (('raw', r), ('full', f)):
                        totals[stage] += counts
                        per_subject.append(dict(setting=name, selector=selector, method=method, subject=subject,
                                                evaluation=stage, **shared.metrics(counts)))
                for stage, counts in totals.items():
                    summary.append(dict(setting=name, selector=selector, method=method, evaluation=stage,
                                        configs=len(configs), **shared.metrics(counts)))
                print('  %s, inner %s selection: outer full F1=%.6f' % (method, selector, shared.metrics(totals['full'])['F1']), flush=True)
            full_by_method[method] = np.asarray(method_full, dtype=np.int64)
            # Preserve completed methods even if a later method fails.
            for filename, rows in [('summary_all.csv', summary), ('per_subject_counts.csv', per_subject),
                                   ('selected_configs.csv', selected), ('inner_search.csv', search)]:
                shared.write_csv(output / filename, rows)
            primary = [r for r in selected if r['method'] == method and r['selector'] == 'full']
            for field, domain in [('reference_scale', SCALES), ('local_radius', RADII), ('threshold', TAUS)]:
                if field == 'local_radius' and method == 'G':
                    continue
                counts = Counter(r[field] for r in primary)
                for value in domain:
                    boundaries.append(dict(setting=name, method=method, parameter=field, value=value,
                                           selected_subjects=counts[value], total_subjects=len(subjects),
                                           is_boundary=int(value in (min(domain), max(domain)))))
    shared.write_csv(output / 'selection_frequency.csv', boundaries)
    comparisons = bootstrap(name, full_by_method, subjects)
    shared.write_csv(output / 'paired_comparisons.csv', comparisons)
    shared.write_json(output / 'completion.json', dict(completed=True, subjects=len(subjects),
                                                       sealed_mean_replay='PASS for all accessed evidence/thresholds'))
    return [r for r in summary if r['selector'] == 'full' and r['evaluation'] == 'full'], comparisons


def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=('both', 'sammlv', 'casme3'), default='both')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume only this stage-2 output with unchanged code/input hashes')
    args = parser.parse_args()
    ora, helper_path = shared.load_helper()
    names = list(shared.SETTINGS.values()) if args.setting == 'both' else [shared.SETTINGS[args.setting]]
    if args.check_inputs:
        for name in names:
            ora.verify_sealed_inputs(ora.SPECS[name])
            print(name, 'input paths/manifest PASS', flush=True)
        print('P8_PHASE2_INPUTS = PASS (official replay runs at experiment start)', flush=True)
        return
    if args.output is None:
        parser.error('--output is required')
    output = args.output.resolve()
    for spec in ora.SPECS.values():
        for key in ('dump', 'cache', 'evidence', 'run_evidence', 'results', 'locked_results'):
            if key in spec and overlaps(output, Path(spec[key]).resolve()):
                raise RuntimeError('output overlaps protected input: %s' % spec[key])
    input_hashes = {}
    for name in names:
        spec = ora.SPECS[name]
        ora.verify_sealed_inputs(spec)
        paths = [spec['source'], spec['core'], spec['evidence'] / 'evidence_bundle_sha256.txt',
                 spec['run_evidence'] / 'run_manifest.json']
        paths += sorted(p for p in spec['dump'].rglob('*') if p.is_file())
        print('Hashing frozen inputs:', name, len(paths), 'files', flush=True)
        input_hashes[name] = {str(p): ora.sha256(p) for p in paths}
    code_paths = [Path(__file__), Path(shared.__file__), helper_path,
                  Path(__file__).with_name('colab_p8_entry.py')]
    identity = dict(settings=names, input_sha256=input_hashes,
                    code_sha256={p.name: ora.sha256(p) for p in code_paths},
                    runtime=dict(python=sys.version, numpy=np.__version__,
                                 pandas=getattr(sys.modules.get('pandas'), '__version__', None)))
    if output.exists():
        if not args.resume or not (output / 'run_manifest.json').is_file():
            raise RuntimeError('output exists; use a new directory or explicit --resume')
        prior = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
        if prior['identity'] != identity:
            raise RuntimeError('resume refused: code or sealed inputs changed')
        if (output / 'completion.json').exists():
            print('Already completed:', output, flush=True)
            return
    else:
        if args.resume:
            raise RuntimeError('--resume requires an existing stage-2 output')
        output.mkdir(parents=True)
        shared.write_json(output / 'run_manifest.json', dict(identity=identity,
            created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, numpy=np.__version__))
    start = time.monotonic()
    preflight = output / 'preflight'
    preflight.mkdir(exist_ok=True)
    replay = [ora.run_preflight_setting(name, preflight) for name in names]
    shared.write_csv(preflight / 'locked_replay_summary.csv', replay)
    print('P8_PHASE2_LOCKED_REPLAY = PASS', flush=True)
    rows, paired = [], []
    for name in names:
        result, comparisons = run_setting(ora, name, output / name)
        rows.extend(result)
        paired.extend(comparisons)
        shared.write_csv(output / 'summary_full.csv', rows)
        shared.write_csv(output / 'paired_comparisons.csv', paired)
    comparison = []
    for name in names:
        f = {r['method']: r['F1'] for r in rows if r['setting'] == name}
        comparison.append(dict(setting=name, p8_exceeds_both_singles=f['GM_p8'] > max(f['G'], f['L']),
                               delta_vs_G=f['GM_p8']-f['G'], delta_vs_L=f['GM_p8']-f['L'],
                               delta_vs_Mean=f['GM_p8']-f['Mean']))
    shared.write_json(output / 'decision.json', dict(
        stage='development', comparisons=comparison,
        both_datasets_point_criterion_met=len(names) == 2 and all(r['p8_exceeds_both_singles'] for r in comparison),
        interpretation='point comparison only; not proof of significance or independent confirmation'))
    shared.write_json(output / 'completion.json', dict(completed=True, settings=names, elapsed_seconds=time.monotonic()-start))
    print('P8_PHASE2_EXECUTION = PASS', flush=True)
    print('OUTPUT =', output, flush=True)


if __name__ == '__main__':
    main()
