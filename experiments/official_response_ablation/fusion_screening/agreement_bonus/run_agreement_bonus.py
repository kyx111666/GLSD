"""Fixed-structure agreement-bonus development screen on frozen official responses."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_generalized_mean_screening as shared

A0, RHO = 1.5, 2.0
LAMBDAS = (0., .1, .25, .5, .75, 1.)
GAMMAS = (0., 1., 2., 4.)
TAUS = tuple(i / 20 for i in range(1, 20))

@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    threshold: float
    lam: float = 0.
    gamma: float = 0.
    reference_scale: float = A0
    local_radius: float = RHO

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)

def scores(g, l, c):
    g, l = np.asarray(g, float), np.asarray(l, float)
    if g.shape != l.shape or not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError('G/L must have matching shapes and finite values')
    if np.any(g < 0) or np.any(g > 1) or np.any(l < 0) or np.any(l > 1):
        raise ValueError('G/L must lie in [0,1]; no clipping')
    if c.method == 'G': return g
    if c.method == 'L': return l
    if c.method == 'Mean': return (g + l) / 2
    if c.method != 'Fusion': raise ValueError(c.method)
    if not 0 <= c.lam <= 1 or c.gamma < 0: raise ValueError('invalid fusion parameters')
    maximum, minimum = np.maximum(g, l), np.minimum(g, l)
    agreement = np.ones_like(g) if c.gamma == 0 else (1 - np.abs(g-l)) ** c.gamma
    return maximum + c.lam * agreement * minimum * (1 - maximum)

def grids():
    configs = []
    for method in ('G', 'L', 'Mean'):
        for tau in TAUS:
            configs.append(Config(method, len(configs), tau))
    pairs = [(0., 0.)] + [(lam, gamma) for lam in LAMBDAS[1:] for gamma in GAMMAS]
    for lam, gamma in pairs:
        for tau in TAUS:
            configs.append(Config('Fusion', len(configs), tau, lam, gamma))
    subsets = {m: [c.config_id for c in configs if c.method == m] for m in ('G','L','Mean')}
    fusion = [c for c in configs if c.method == 'Fusion']
    subsets['Max'] = [c.config_id for c in fusion if c.lam == 0]
    subsets['Bonus_joint'] = [c.config_id for c in fusion if c.gamma == 0]
    subsets['Full_joint'] = [c.config_id for c in fusion]
    for lam, gamma in pairs:
        subsets[f'fixed_lambda{lam:g}_gamma{gamma:g}'] = [c.config_id for c in fusion if c.lam == lam and c.gamma == gamma]
    return configs, subsets

class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        peaks, values = base.evidence(A0, RHO)
        self.peaks, self.values = np.array(peaks, int), np.array(values, float)
        if self.values.shape != (len(self.peaks), 2): raise RuntimeError('expected G,L evidence')
        g, l = self.values.T
        for tau in TAUS:
            expected = base.selected_peaks(SimpleNamespace(reference_scale=A0, local_radius=RHO, threshold=tau))
            actual = self.peaks[scores(g, l, Config('Mean', 0, tau)) >= tau]
            if not np.array_equal(expected, actual): raise RuntimeError('Mean hook fails sealed scorer replay')
    def __getattr__(self, name): return getattr(self.base, name)
    def selected_peaks(self, c):
        if c.reference_scale != A0 or c.local_radius != RHO: raise RuntimeError('structure changed')
        self.owner.calls += 1
        g, l = self.values.T
        score = scores(g, l, c)
        keep = score >= c.threshold
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.key[0], peaks=self.peaks.tolist(),
                G=g.tolist(), L=l.tolist(), score=score.tolist(), retained=keep.tolist(),
                identity_note='response hash is not a verified video/event ID'))
        return self.peaks[keep]

class FusionCore(shared.FusionCore):
    def GLSDFeatures(self, response, k):
        arr = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), arr.shape, hashlib.sha256(arr.tobytes()).hexdigest())
        if key not in self.cache: self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]

def run_setting(ora, name, out):
    out.mkdir()
    context = ora.metst_context(ora.SPECS[name])
    subjects = context[6]
    if len(subjects) != len(set(subjects)): raise RuntimeError('duplicate subjects')
    core = FusionCore(context[1])
    configs, subsets = grids()
    shared.write_json(out/'protocol.json', dict(setting=name, fixed_a0=A0, fixed_rho=RHO,
        configs=[asdict(c) for c in configs], selection_subsets=subsets,
        selection='pooled raw counts excluding held subject; F1, precision, fewer FP, config order',
        reporting='held subject official full result synergy; pooled event F1',
        scope='decoder subject holdout on frozen official responses; no backbone retraining or k re-estimation',
        development_only=True, equal_search_budget=False,
        event_mechanism='unavailable until verified video/event identity is exposed',
        response_hashes=[dict(path=str(p), sha256=ora.sha256(p)) for p in context[5]]))
    table = np.zeros((len(configs), len(subjects), 3), dtype=np.int64)
    for i, subject in enumerate(subjects):
        print(f'{name}: search subject {i+1}/{len(subjects)} ({subject}), {len(configs)} configurations', flush=True)
        for c in configs:
            raw, _, _, _ = shared.decode(context, core, i, c, False)
            table[c.config_id, i] = raw
    np.savez_compressed(out/'search_counts.npz', raw_counts=table, subjects=np.asarray(subjects, str))
    summary, selections, per_subject, search_rows = [], [], [], []
    counts_by_variant = {}
    with gzip.open(out/'selected_predictions.jsonl.gz','wt') as pred, gzip.open(out/'candidate_trace.jsonl.gz','wt') as traces:
        for variant, allowed in subsets.items():
            full_counts, raw_counts = [], []
            for i, subject in enumerate(subjects):
                winner, pooled = shared.choose(table, i, allowed)
                c = configs[winner]
                raw, full, predictions, trace = shared.decode(context, core, i, c, True, capture=not variant.startswith('fixed_'))
                if raw != tuple(table[winner,i]): raise RuntimeError('search/replay mismatch')
                full_counts.append(full); raw_counts.append(raw)
                selections.append(dict(variant=variant, subject=subject, **asdict(c), inner_raw_F1=shared.metrics(pooled[winner])['F1']))
                for stage, count in [('raw', raw),('full',full)]:
                    per_subject.append(dict(variant=variant,subject=subject,metric_stage=stage,**shared.metrics(count)))
                pred.write(json.dumps(shared.serializable(dict(variant=variant,subject=subject,config=asdict(c),raw=raw,full=full,predictions=predictions)))+'\n')
                for t in trace: traces.write(json.dumps(dict(variant=variant,subject=subject,config=asdict(c),**t))+'\n')
            for stage, counts in [('raw',raw_counts),('full',full_counts)]:
                summary.append(dict(setting=name,variant=variant,metric_stage=stage,configs=len(allowed),**shared.metrics(np.sum(counts,axis=0))))
            counts_by_variant[variant] = np.asarray(full_counts)
            print(f'{variant}: full F1={shared.metrics(np.sum(full_counts,axis=0))["F1"]:.6f}',flush=True)
    for i, subject in enumerate(subjects):
        _, pooled = shared.choose(table, i)
        for c in configs:
            search_rows.append(dict(outer_subject=subject,**asdict(c),**shared.metrics(pooled[c.config_id])))
    # Use shared paired subject bootstrap without changing its implementation.
    bootstrap_counts = {k:v for k,v in counts_by_variant.items() if not k.startswith('fixed_') and k != 'Full_joint'}
    bootstrap_counts['GM_joint'] = counts_by_variant['Full_joint']
    comparisons = shared.paired_comparisons(name, bootstrap_counts, subjects)
    for row in comparisons: row['compared'] = 'Full_joint'
    disagreement = []
    for view in core.cache.values():
        g,l = view.values.T
        d = np.abs(g-l)
        for label, mask in [('low',d<.1),('medium',(d>=.1)&(d<.3)),('high',d>=.3)]:
            disagreement.append(dict(response_sha256=view.key[2],k=view.key[0],group=label,count=int(mask.sum()),
                mean_G=float(g[mask].mean()) if mask.any() else None,mean_L=float(l[mask].mean()) if mask.any() else None))
    for filename, rows in [('screening_summary.csv',summary),('selected_configs.csv',selections),
        ('per_subject_counts.csv',per_subject),('search_all.csv',search_rows),('paired_comparisons.csv',comparisons),
        ('disagreement_analysis.csv',disagreement),
        ('fusion_heatmap.csv',[r for r in summary if r['variant'].startswith('fixed_') and r['metric_stage']=='full'])]:
        shared.write_csv(out/filename,rows)
    shared.write_json(out/'limitations.json',dict(event_rescue_counts_available=False,
        reason='sealed hook lacks verified video/event identity; candidate traces are not TP/FP transition counts',
        heatmap='development diagnostic; do not report best outer heatmap cell as unbiased test performance',
        frozen_structure_negative_result='does not rule out other a0/rho settings'))
    shared.write_json(out/'completion.json',dict(completed=True,setting=name))
    return [r for r in summary if r['metric_stage']=='full']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=('sammlv','casme3','both'), default='sammlv')
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    ora, helper_path=shared.load_helper()
    names=list(shared.SETTINGS.values()) if args.setting=='both' else [shared.SETTINGS[args.setting]]
    inventory={name:[dict(path=str(p),exists=p.exists()) for p in ora.required_paths(ora.SPECS[name])] for name in names}
    if args.check_inputs:
        print(json.dumps(inventory,indent=2))
        if any(not r['exists'] for rows in inventory.values() for r in rows): raise SystemExit(2)
        print('AGREEMENT_INPUT_PATHS = PASS (runtime/replay not yet checked)'); return
    for name in names: ora.verify_sealed_inputs(ora.SPECS[name])
    out=(args.output or Path('/content/drive/MyDrive/GLSD_AGREEMENT_BONUS')/datetime.now(timezone.utc).strftime('screen_%Y%m%dT%H%M%S_%fZ')).resolve()
    for name in names:
        for key in ('dump','evidence','run_evidence','results','locked_results'):
            protected=ora.SPECS[name][key].resolve()
            if out==protected or protected in out.parents or out in protected.parents: raise RuntimeError('output overlaps protected evidence')
    out.mkdir(parents=True,exist_ok=False)
    shared.write_json(out/'run_manifest.json',dict(settings=names,input_inventory=inventory,
        source_hashes={p.name:ora.sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    preflight=out/'preflight'; preflight.mkdir()
    shared.write_csv(preflight/'locked_replay_summary.csv',[ora.run_preflight_setting(name,preflight) for name in names])
    print('NATIVE_AND_LOCKED_GLSD_REPLAY = PASS',flush=True)
    rows=[]
    for name in names: rows.extend(run_setting(ora,name,out/name))
    shared.write_csv(out/'screening_summary_full.csv',rows)
    shared.write_json(out/'completion.json',dict(completed=True,settings=names,development_only=True))
    print('AGREEMENT_BONUS_EXECUTION = PASS',flush=True)
    print('OUTPUT =',out,flush=True)

if __name__=='__main__': main()
