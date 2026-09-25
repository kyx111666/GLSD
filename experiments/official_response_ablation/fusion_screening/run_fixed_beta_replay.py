"""Conditional fixed-beta full replay using existing search counts; no new grid."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

import run_fusion_event_diagnostic as events

tuning, shared = events.tuning, events.shared
BETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def conditional_choices(table, grid, subjects):
    if table.shape != (len(grid), len(subjects), 3) or np.any(table < 0):
        raise ValueError('Invalid saved count table')
    rows = []
    for beta in BETAS:
        indexes = [i for i, c in enumerate(grid) if c['beta'] == beta]
        if not indexes:
            raise ValueError(f'Missing beta {beta}')
        subset = table[indexes]
        for si, subject in enumerate(subjects):
            winner, pooled = shared.choose(subset, si)
            ci = indexes[winner]
            rows.append(dict(label=f'FixedBeta_{beta:.1f}', subject=str(subject),
                config=grid[ci], expected_raw=table[ci, si].tolist(),
                inner_raw_counts=pooled[winner].tolist(), inner_subject_count=len(subjects)-1))
    return rows


def summary_and_pairs(records, subjects, output):
    labels = list(dict.fromkeys(r['label'] for r in records))
    by = {(r['label'], r['subject']): r for r in records}
    summaries, per_subject = [], []
    full_tables = {}
    for label in labels:
        for stage in ('raw', 'full'):
            table = np.asarray([by[label, s][stage+'_counts'] for s in subjects], dtype=np.int64)
            summaries.append(dict(label=label, metric_stage=stage, **shared.metrics(table.sum(axis=0))))
            for subject, counts in zip(subjects, table):
                per_subject.append(dict(label=label, subject=subject, metric_stage=stage, **shared.metrics(counts)))
            if stage == 'full':
                full_tables[label] = table
    shared.write_csv(output/'summary_raw_full.csv', summaries)
    shared.write_csv(output/'per_subject_counts.csv', per_subject)
    for stage in ('raw_counts', 'full_counts'):
        for subject in subjects:
            if by['FixedBeta_0.5', subject][stage] != by['Expanded_EqualMean', subject][stage]:
                raise RuntimeError('Fixed beta=.5 differs from same-range EqualMean')
    rng = np.random.default_rng(shared.SEED)
    draws = rng.integers(0, len(subjects), size=(shared.RESAMPLES, len(subjects)))
    distributions = {}
    for label, table in full_tables.items():
        totals = table[draws].sum(axis=1)
        tp, fp, fn = totals.T
        distributions[label] = np.divide(2*tp, 2*tp+fp+fn,
            out=np.zeros(len(tp)), where=(2*tp+fp+fn)>0)
    paired = []
    for beta in BETAS:
        target = f'FixedBeta_{beta:.1f}'
        for reference in [f'Expanded_{m}' for m in tuning.METHODS] + ['Original_EqualMean']:
            lo, hi = np.quantile(distributions[target]-distributions[reference], [.025,.975])
            paired.append(dict(target=target, reference=reference, metric_stage='full',
                delta_F1=shared.metrics(full_tables[target].sum(0))['F1']-shared.metrics(full_tables[reference].sum(0))['F1'],
                CI_low=float(lo), CI_high=float(hi), seed=shared.SEED,resamples=shared.RESAMPLES))
    shared.write_csv(output/'paired_comparisons.csv', paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    reference_path = directory/'structure_selected_replay_reference.json'
    table_path = directory/'fixed_beta_search_input.npz'
    reference = json.loads(reference_path.read_text())
    with np.load(table_path, allow_pickle=False) as saved:
        table = saved['raw_counts']
        subjects = saved['subjects'].tolist()
        grid = json.loads(saved['configs'].item())
        archive_hash = saved['source_archive_sha256'].item()
    if subjects != reference['subjects'] or archive_hash != reference['source_archive_sha256']:
        raise RuntimeError('Search tensor and replay reference have different provenance')
    if reference['matching_protocol'] != events.evaluator.PROTOCOL:
        raise RuntimeError('Matching protocol differs')
    choices = conditional_choices(table, grid, subjects)
    events.previous.configure_threshold_grid('refined')
    events.previous.configure_structure_grid('expanded')
    if grid != [asdict(c) for c in tuning.grids()['DominantEvidence']]:
        raise RuntimeError('Saved grid differs from declared expanded grid')
    ora, helper = shared.load_helper()
    spec = ora.SPECS['metst_sammlv']
    ora.verify_sealed_inputs(spec)
    for key in ('source','core'):
        if ora.sha256(spec[key]) != reference[key+'_sha256']:
            raise RuntimeError('Sealed source differs: '+key)
    output = args.output.resolve()
    for key in ('dump','evidence','run_evidence','results','locked_results'):
        protected = spec[key].resolve()
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise RuntimeError('Output overlaps sealed input')
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output/'manifest.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_archive=reference['source_archive'],
        source_archive_sha256=archive_hash, matching_protocol=events.evaluator.PROTOCOL,
        betas=BETAS, scope='SAMMLV expanded structure; fixed beta, select remaining parameters from other subjects',
        selection='pooled raw F1, precision, fewer FP, fixed saved grid order; exclude held subject',
        interpretation='development diagnostic; no automatic best-beta adoption or independent confirmation',
        source_sha256={p.name:ora.sha256(p) for p in [*directory.glob('*.py'),reference_path,table_path,helper]}))
    shared.write_json(output/'fixed_beta_selected_configs.json', choices)
    context = ora.metst_context(spec)
    if list(map(str, context[6])) != subjects or ora.context_video_and_gt_counts('metst',context)!=(79,159):
        raise RuntimeError('Dataset identity differs')
    core = tuning.FusionCore(events.ExpandedCore(context[1], tuning.SCALES))
    controls = [r for r in reference['records'] if r['variant']=='expanded' or
                (r['variant']=='original' and r['method']=='EqualMean')]
    jobs = [dict(label=('Expanded_' if r['variant']=='expanded' else 'Original_')+r['method'],
                 subject=r['subject'],config=r['config'],expected_raw=r['raw_counts'],saved=r) for r in controls] + choices
    records, checks = [], []
    with events.evaluator.install(context[2]) as info, gzip.open(output/'event_records.jsonl.gz','wt',encoding='utf8') as ledger:
        shared.write_json(output/'matching_protocol.json', info)
        for index, job in enumerate(jobs):
            si = subjects.index(job['subject'])
            result = events.detailed_decode(context,core,si,tuning.Config(**job['config']))
            if result['raw_counts'] != job['expected_raw']:
                raise RuntimeError('Saved search raw counts differ: '+job['label']+'/'+job['subject'])
            if 'saved' in job:
                saved = job['saved']
                predictions = [[] for _ in saved['predictions']]
                for event in result['events']:
                    predictions[int(event['video_id'].rsplit('_',1)[1])].append(event['interval'])
                if result['full_counts']!=saved['full_counts'] or predictions!=saved['predictions']:
                    raise RuntimeError('Existing control replay differs')
            result['label'] = job['label']
            records.append(result)
            checks.append(dict(label=job['label'],subject=job['subject'],raw_search_replay=True,
                existing_full_prediction_replay=True if 'saved' in job else 'new_full_evaluation',
                event_counts_reconstruct=True))
            ledger.write(json.dumps(shared.serializable(result),allow_nan=False)+'\n')
            if (index+1)%29==0:
                ledger.flush()
                print(f"Fixed-beta/control replay: {index+1}/{len(jobs)} PASS",flush=True)
    summary_and_pairs(records,subjects,output)
    shared.write_csv(output/'replay_checks.csv',checks)
    shared.write_json(output/'completion.json',dict(completed=True,replays=len(jobs),subjects=29,videos=79,gt=159,
        fixed_beta_half_equals_expanded_mean=True))
    print('FIXED_BETA_FULL_REPLAY = PASS; OUTPUT =',output,flush=True)


if __name__=='__main__':
    main()
