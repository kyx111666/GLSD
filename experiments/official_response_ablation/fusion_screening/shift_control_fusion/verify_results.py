"""Reconstruct selected results from saved event identities and training counts."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def metric(counts):
    tp, fp, fn = map(int, counts)
    return dict(TP=tp, FP=fp, FN=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def verify(output):
    output = Path(output)
    summary = json.loads((output/'input_summary.json').read_text())
    expected_subjects, expected_gt = summary['subjects'], summary['ground_truth']
    with np.load(output/'search_counts.npz', allow_pickle=False) as z:
        raw, full = z['raw'].copy(), z['full'].copy()
        subjects, methods, taus = z['subjects'].tolist(), z['methods'].tolist(), z['taus'].tolist()
        done=z['done'].copy()
        if z['digest'].item() != hashlib.sha256(raw.tobytes()+full.tobytes()+done.tobytes()).hexdigest():
            raise RuntimeError('Checkpoint digest mismatch')
        if not done.all():
            raise RuntimeError('Incomplete checkpoint')
    if len(subjects) != expected_subjects or len(taus) != 301 or len(methods) != 7:
        raise RuntimeError('Search domain mismatch')
    for tensor in (raw, full):
        if tensor.shape != (7, 301, len(subjects), 3) or np.any(tensor < 0):
            raise RuntimeError('Invalid counts tensor')
        if not np.all(tensor[..., 0]+tensor[..., 2] == summary['per_subject_gt']):
            raise RuntimeError('GT conservation failed')
    selections = read_csv(output/'selected_configs.csv')
    if len(selections) != 13*len(subjects):
        raise RuntimeError('Expected 7 independent rows + 3 retuned + 3 locked per subject')
    configs = {(s['subject'], s['method']):s for s in selections}
    if len(configs) != len(selections):
        raise RuntimeError('Duplicate selected row')
    names=methods+[prefix+name for prefix in ('ablation_retuned_', 'ablation_locked_')
                   for name in ('no_global','no_local','no_shift')]
    if set(configs) != {(s,m) for s in subjects for m in names}:
        raise RuntimeError('Unexpected method/ablation identities')
    for row in selections:
        si, mi, ti = subjects.index(row['subject']), methods.index(row['scoring_method']), int(row['config_id'])
        if float(row['threshold']) != taus[ti] or float(row['a0']) != 2 or float(row['rho']) != 3:
            raise RuntimeError('Selected structure/threshold mismatch')
        # Independently reconstruct exclusion and the complete tie-break chain.
        inner = full[mi, :, [i for i in range(len(subjects)) if i != si], :].sum(0)
        winner = min(range(301), key=lambda i: (-metric(inner[i])['F1'], -metric(inner[i])['precision'], int(inner[i,1]), i))
        if row['method'].startswith('ablation_locked_'):
            if ti != int(configs[row['subject'], 'Full']['config_id']):
                raise RuntimeError('Locked ablation changed Full threshold')
        elif ti != winner:
            raise RuntimeError('Outer exclusion or threshold tie-break failed')
        for k, v in metric(inner[ti]).items():
            if not np.isclose(float(row['inner_'+k]), v, atol=1e-12, rtol=0):
                raise RuntimeError('Training-side metric mismatch')
    count_rows = read_csv(output/'per_subject_counts.csv')
    counts_csv = {(r['subject'], r['method'], r['stage']):[int(r[k]) for k in ('TP','FP','FN')] for r in count_rows}
    if len(counts_csv) != len(selections)*2 or len(count_rows) != len(counts_csv):
        raise RuntimeError('Missing or duplicate per-subject count rows')
    records, event_count = {}, 0
    with gzip.open(output/'event_records.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['subject'], row['method']
            if key in records or key not in configs:
                raise RuntimeError('Unexpected event-record identity')
            selected = configs[key]
            si, mi, ti = subjects.index(key[0]), methods.index(selected['scoring_method']), int(selected['config_id'])
            gt_ids = {g['gt_id'] for g in row['ground_truth']}
            if len(gt_ids) != len(row['ground_truth']) or len(gt_ids) != summary['per_subject_gt'][si]:
                raise RuntimeError('GT identities missing or duplicated')
            for stage, tensor in (('raw',raw), ('full',full)):
                tp = sum(e[stage+'_TP'] for e in row['events'])
                fp = sum(e[stage+'_FP'] for e in row['events'])
                matched = [e['matched_gt_id'] for e in row['events'] if e[stage+'_TP']]
                if len(set(matched)) != tp or not set(matched) <= gt_ids:
                    raise RuntimeError('Invalid one-to-one GT assignment')
                values = [tp, fp, len(gt_ids)-tp]
                if values != row[stage+'_counts'] or values != tensor[mi,ti,si].tolist() or values != counts_csv[key+(stage,)]:
                    raise RuntimeError('Event/checkpoint/CSV count mismatch')
            for e in row['events']:
                if e['full_TP'] != e['raw_TP']*int(e['recognition_prediction'] != 4) or e['full_FP'] != e['raw_FP']*int(e['recognition_prediction'] != 4):
                    raise RuntimeError('Neutral filtering mismatch')
            for candidate in row['candidates']:
                if candidate['retained'] != [s >= taus[ti] for s in candidate['score']]:
                    raise RuntimeError('Threshold mask mismatch')
                if candidate['a0'] != 2 or candidate['rho'] != 3:
                    raise RuntimeError('Candidate structure mismatch')
                g,l,l0,b=(np.asarray(candidate[k],float) for k in ('G','L','L0','B'))
                expected={'Full':g+l-b,'G':g,'L':l,'D':l-b,'G_plus_L':g+l,
                          'VideoMeanB':g+l-(b.mean() if len(b) else 0.),'RefResidualUnclipped':g+l-l0}[selected['scoring_method']]
                if selected['scoring_method']=='Full':
                    abstain=np.array([bool(r) for r in candidate['abstention']],dtype=bool)
                    expected[abstain]=g[abstain]
                if not np.allclose(candidate['score'],expected,atol=1e-12,rtol=0):
                    raise RuntimeError('Saved score formula mismatch')
                selected_peaks=[p for p,keep in zip(candidate['peaks'],candidate['retained']) if keep]
                event_peaks=[int(e['interval'][6]) for e in row['events'] if e['video_id']==candidate['video_id']]
                if selected_peaks != event_peaks:
                    raise RuntimeError('Candidate-to-decoder order mismatch')
            records[key] = row
            event_count += len(row['events'])
    if len(records) != len(selections):
        raise RuntimeError('Missing selected event record')
    summary_rows=read_csv(output/'summary_raw_full.csv')
    if len(summary_rows)!=26 or {(r['method'],r['stage']) for r in summary_rows}!={(m,s) for m in names for s in ('raw','full')}:
        raise RuntimeError('Incomplete or duplicated summary rows')
    for row in summary_rows:
        name, stage = row['method'], row['stage']
        values = np.sum([counts_csv[s, name, stage] for s in subjects], axis=0)
        if values[0]+values[2] != expected_gt:
            raise RuntimeError('Pooled GT mismatch')
        for k, v in metric(values).items():
            if not np.isclose(float(row[k]), v, atol=1e-12, rtol=0):
                raise RuntimeError('Pooled metric mismatch (do not average subject F1)')
    seen_differences = set()
    with gzip.open(output/'event_differences.jsonl.gz', 'rt') as stream:
        for line in stream:
            d = json.loads(line)
            key = d['subject'], d['reference']
            if key in seen_differences:
                raise RuntimeError('Duplicate event difference')
            seen_differences.add(key)
            a, b = records[d['subject'], 'Full'], records[key]
            sa = {e['matched_gt_id'] for e in a['events'] if e['full_TP']}
            sb = {e['matched_gt_id'] for e in b['events'] if e['full_TP']}
            if sorted(sa-sb) != d['added_gt_ids'] or sorted(sb-sa) != d['lost_gt_ids']:
                raise RuntimeError('GT delta mismatch')
            if [d['delta_'+k] for k in ('TP','FP','FN')] != (np.array(a['full_counts'])-b['full_counts']).tolist():
                raise RuntimeError('FP/TP/FN delta mismatch')
            for stage in ('raw','full'):
                def event_keys(r):
                    return {(e['video_id'],tuple(e['interval'])) for e in r['events'] if stage=='raw' or e['full_retained']}
                ka,kb=event_keys(a),event_keys(b)
                for name,expected in ((stage+'_added_events',ka-kb),(stage+'_lost_events',kb-ka)):
                    if {(v,tuple(interval)) for v,interval in d[name]} != expected:
                        raise RuntimeError('Decoded event delta mismatch')
    if len(seen_differences) != 12*len(subjects):
        raise RuntimeError('Missing event differences')
    return dict(status='PASS', scope='Result integrity, not efficacy', subjects=len(subjects),
        selected_records=len(records), event_entries=event_count,
        checks=['Complete 301-threshold domain', 'Outer subject excluded from selection',
                'F1/precision/FP/index tie-break', 'Separate retuned and locked ablations',
                'One-to-one GT identities', 'Event/checkpoint/CSV raw/full conservation',
                'Threshold masks', 'Pooled-count summaries', 'Added/lost GT and FP deltas'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    print(json.dumps(verify(p.parse_args().output), indent=2))
