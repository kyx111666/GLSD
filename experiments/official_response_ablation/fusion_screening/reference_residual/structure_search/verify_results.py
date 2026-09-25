"""Independent tensor/selection/event audit; optional previous-Colab regression."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def verify(root, previous=None):
    def rows(name):
        with (root / name).open() as f:
            return list(csv.DictReader(f))

    def jsonl(name):
        with gzip.open(root / name, 'rt') as f:
            return [json.loads(line) for line in f]

    def metric(c):
        tp, fp, fn = map(int, c)
        return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.

    def winner(table, outer, allowed):
        inner = table[:, [i for i in range(29) if i != outer]].sum(axis=1)
        def key(i):
            tp, fp, fn = map(int, inner[i])
            return (metric(inner[i]), tp/(tp+fp) if tp+fp else 0., -fp, -i)
        return max(allowed, key=key), inner

    def count(row):
        return np.array([int(row[k]) for k in ('TP', 'FP', 'FN')])

    completion = json.loads((root / 'completion.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    scales, radii, thresholds = [protocol[k] for k in ('scales', 'radii', 'thresholds')]
    structure_count = len(scales)*len(radii)
    expected_configs = len(scales)*len(thresholds)*(1+4*len(radii))
    assert completion['completed'] and completion['configs'] == expected_configs
    methods = ['G', 'L', 'Mean', 'PositiveResidual', 'RefResidual']
    tables = {m: dict(np.load(root / ('search_counts_'+m+'.npz'))) for m in methods}
    subjects = list(tables['G']['subjects'])
    assert len(set(subjects)) == 29
    configs = {m: json.loads(str(t['configs'])) for m, t in tables.items()}
    for m, t in tables.items():
        expected = [(a,r,t) for a in scales for r in ([1.] if m=='G' else radii) for t in thresholds]
        n = len(expected)
        assert [(c['reference_scale'], c['local_radius'], c['threshold']) for c in configs[m]] == expected
        assert t['done'].all() and list(t['subjects']) == subjects
        for stage in ('raw', 'full'):
            assert t[stage].shape == (n, 29, 3) and (t[stage] >= 0).all()
            gt = t[stage][..., 0]+t[stage][..., 2]
            assert (gt == gt[0]).all() and np.all(gt.sum(1) == 159)
    selected = {(r['subject'], r['method']): r for r in rows('selected_configs.csv')}
    per = {(r['subject'], r['method'], r['stage']): count(r) for r in rows('per_subject_counts.csv')}
    assert len(selected) == 203 and len(per) == 406
    for si, s in enumerate(subjects):
        for m in methods:
            win, pooled = winner(tables[m]['full'], si, range(len(configs[m])))
            row = selected[s, m]
            assert int(row['config_id']) == int(row['search_config_id']) == win
            assert abs(float(row['inner_F1'])-metric(pooled[win])) < 1e-14
            c = configs[m][win]
            for column, field in [('a0', 'reference_scale'), ('rho', 'local_radius'), ('threshold', 'threshold')]:
                assert float(row[column]) == c[field]
        for name, m in [('locked_no_local', 'G'), ('locked_no_negative', 'PositiveResidual')]:
            row, ref = selected[s, name], selected[s, 'RefResidual']
            assert int(row['config_id']) == -1 and row['scoring_method'] == m
            assert all(row[k] == ref[k] for k in ('a0', 'rho', 'threshold'))
        for m in [*methods, 'locked_no_local', 'locked_no_negative']:
            row = selected[s, m]
            cm, ci = row['scoring_method'], int(row['search_config_id'])
            c = configs[cm][ci]
            assert c['reference_scale'] == float(row['a0']) and c['threshold'] == float(row['threshold'])
            assert cm == 'G' or c['local_radius'] == float(row['rho'])
            for stage in ('raw', 'full'):
                assert np.array_equal(per[s, m, stage], tables[cm][stage][ci, si])
    for row in rows('summary_full.csv'):
        c = sum(per[s, row['method'], 'full'] for s in subjects)
        assert np.array_equal(count(row), c) and abs(float(row['F1'])-metric(c)) < 1e-14
    ledger = jsonl('event_records.jsonl.gz')
    assert len(ledger) == 203
    for row in ledger:
        for stage in ('raw', 'full'):
            ids = [e['matched_gt_id'] for e in row['events'] if e[stage+'_TP']]
            assert len(ids) == len(set(ids))
            c = [len(ids), sum(e[stage+'_FP'] for e in row['events']), len(row['ground_truth'])-len(ids)]
            assert c == row[stage+'_counts'] and np.array_equal(c, per[row['subject'], row['method'], stage])
    index = {(r['subject'], r['method']): r for r in ledger}
    for d in jsonl('event_differences.jsonl.gz'):
        a, b = [index[d['subject'], d[k]] for k in ('compared', 'reference')]
        ag, bg = [{e['matched_gt_id'] for e in r['events'] if e['full_TP']} for r in (a, b)]
        assert set(d['added_gt_ids']) == ag-bg and set(d['lost_gt_ids']) == bg-ag
        assert np.array_equal([d['delta_'+k] for k in ('TP','FP','FN')],
                              np.array(a['full_counts'])-np.array(b['full_counts']))
    cs = rows('conditional_structure_selections.csv')
    assert len(cs) == 29*5*structure_count
    totals = {}
    for r in cs:
        si, m = subjects.index(r['subject']), r['method']
        allowed = [i for i,c in enumerate(configs[m]) if c['reference_scale'] == float(r['a0'])
                   and (m == 'G' or c['local_radius'] == float(r['rho']))]
        win, pooled = winner(tables[m]['full'], si, allowed)
        assert win == int(r['config_id']) and configs[m][win]['threshold'] == float(r['threshold'])
        k = (m, r['a0'], r['rho'])
        totals[k] = totals.get(k, np.zeros(3, dtype=int))+tables[m]['full'][win, si]
    for r in rows('conditional_structure_summary.csv'):
        c = totals[r['method'], r['a0'], r['rho']]
        assert np.array_equal(count(r), c) and abs(float(r['F1'])-metric(c)) < 1e-14
    features = jsonl('features.jsonl.gz')
    assert len(features) == 79*structure_count
    seen = {}
    for v in features:
        key = (v['video_id'], v['a0'])
        x = (v['peaks'], v['G'])
        assert key not in seen or seen[key] == x
        seen[key] = x
        local = np.asarray(v['per_scale_support'])
        assert np.array_equal(np.median(local, axis=1), v['L'])
        assert np.array_equal(local[:, v['physical_widths'].index(v['reference_width'])], v['L0'])
        assert np.array_equal(np.array(v['L'])-v['L0'], v['D'])
        assert np.all(local[np.array(v['missing_support'])] == 0.)
    checks = [f'PASS: {expected_configs} complete configurations, 29 subjects and GT conservation',
              'PASS: 145 independent joint structure/threshold selections exclude outer subject',
              'PASS: 58 locked ablations inherit the correct full-method structure and threshold',
              'PASS: 203 raw/full event ledgers and all GT identity deltas reconstruct counts',
              f'PASS: {len(cs)} conditional threshold selections and {5*structure_count} diagnostic totals agree',
              f'PASS: {len(features)} feature records; G/candidates invariant across rho; L/L0/D reconstruct']
    if previous:
        with zipfile.ZipFile(previous) as z:
            prefix = next(n.rsplit('/',1)[0]+'/' for n in z.namelist() if n.endswith('/completion.json'))
            for m in methods:
                old = np.load(io.BytesIO(z.read(prefix+'search_counts_'+m+'.npz')))
                assert list(old['subjects']) == subjects
                old_configs = json.loads(str(old['configs']))
                ids = [next(i for i,c in enumerate(configs[m]) if c['reference_scale']==2.
                       and (m=='G' or c['local_radius']==2.) and c['threshold']==oc['threshold'])
                       for oc in old_configs]
                assert len(ids) == 19
                for stage in ('raw', 'full'):
                    assert np.array_equal(tables[m][stage][ids], old[stage]), (m,stage)
        checks.append('PASS: all 5 methods at a0=2/rho=2 exactly reproduce previous Colab raw/full tensors')
    return dict(status='PASS', checks=checks, source=str(root.name),
                previous_comparison=Path(previous).name if previous else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.previous), ensure_ascii=False, indent=2))
