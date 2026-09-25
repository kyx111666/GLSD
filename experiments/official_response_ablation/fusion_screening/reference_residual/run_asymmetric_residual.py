"""Nested SAMMLV screen for the negative-residual strength.

The reference scale and local radius stay fixed at (2, 2).  Each asymmetric
residual strength gets its own full-F1 threshold under the existing outer
subject protocol.  This is a development screen, not a final benchmark.
"""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

import run_reference_residual as base
from residual_features import LOCKED

PROTOCOL = 'reference_residual_negative_strength_a2_r2_native_video_v1'
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ('G', 'PositiveResidual', 'RefResidual') + tuple(
    'AsymResidual_l' + str(x).replace('.', 'p') for x in LAMBDAS)


def grids():
    result = {}
    for method in METHODS:
        result[method] = [base.phase.Config(method, i, 2., 2., tau)
                          for i, tau in enumerate(base.phase.TAUS)]
    return result


def comparisons(counts):
    draws = np.random.default_rng(100).integers(0, len(counts['G']),
                                                 (10000, len(counts['G'])))

    def boot(array):
        x = array[draws].sum(1)
        denom = 2*x[:, 0] + x[:, 1] + x[:, 2]
        return np.divide(2*x[:, 0], denom, out=np.zeros(len(x)), where=denom > 0)

    target = counts['PositiveResidual']
    rows = []
    for method, array in counts.items():
        if method == 'PositiveResidual':
            continue
        delta = boot(target) - boot(array)
        rows.append(dict(
            compared='PositiveResidual', reference=method,
            delta_F1=base.shared.metrics(target.sum(0))['F1'] -
                     base.shared.metrics(array.sum(0))['F1'],
            CI_low=float(np.quantile(delta, .025)),
            CI_high=float(np.quantile(delta, .975)), seed=100, resamples=10000,
            interpretation='development; conditional on selected predictions; no retuning'))
    return rows


def screen(context, core, output):
    all_grids = grids()
    tables = {method: base.search(context, core, configs,
                                  output/('search_counts_%s.npz' % method))
              for method, configs in all_grids.items()}
    counts = {method: [] for method in METHODS}
    selections, subject_rows = [], []
    event_totals = {method: dict(added_gt=0, lost_gt=0, delta_FP=0)
                    for method in METHODS if method != 'PositiveResidual'}
    with gzip.open(output/'event_records.jsonl.gz', 'wt', encoding='utf8') as events, \
         gzip.open(output/'event_differences.jsonl.gz', 'wt', encoding='utf8') as differences:
        for si, subject in enumerate(context[6]):
            configs = {}
            for method in METHODS:
                winner, pooled = base.shared.choose(tables[method][1], si)
                configs[method] = all_grids[method][winner]
                selections.append(dict(subject=subject, **asdict(configs[method]),
                    selection='inner_full',
                    inner_F1=base.shared.metrics(pooled[winner])['F1']))
            selected = {}
            for method, config in configs.items():
                record = base.decode(context, core, si, config, True)
                for j, stage in enumerate(('raw', 'full')):
                    if not np.array_equal(record[stage+'_counts'],
                                          tables[method][j][config.config_id, si]):
                        raise RuntimeError('selected counts do not reproduce search tensor')
                selected[method] = record
                counts[method].append(record['full_counts'])
                events.write(json.dumps(base.shared.serializable(record),
                                        allow_nan=False)+'\n')
                for stage in ('raw', 'full'):
                    subject_rows.append(dict(subject=subject, method=method,
                        stage=stage, **base.shared.metrics(record[stage+'_counts'])))
            for method in event_totals:
                delta = base.ledger.event_delta(selected[method],
                                                selected['PositiveResidual'])
                differences.write(json.dumps(base.shared.serializable(delta),
                                             allow_nan=False)+'\n')
                event_totals[method]['added_gt'] += len(delta['added_gt_ids'])
                event_totals[method]['lost_gt'] += len(delta['lost_gt_ids'])
                event_totals[method]['delta_FP'] += delta['delta_FP']
            print('selected event replay', si+1, '/', len(context[6]), flush=True)
    counts = {m: np.asarray(rows, np.int64) for m, rows in counts.items()}
    summary = [dict(method=m, selection='inner_full', configs=19,
                    **base.shared.metrics(rows.sum(0)))
               for m, rows in counts.items()]
    for filename, rows in [('summary_full.csv', summary),
                           ('selected_configs.csv', selections),
                           ('per_subject_counts.csv', subject_rows),
                           ('paired_comparisons.csv', comparisons(counts))]:
        base.shared.write_csv(output/filename, rows)
    positive = (base.shared.metrics(counts['PositiveResidual'].sum(0))['F1'] >
                base.shared.metrics(counts['G'].sum(0))['F1'])
    base.shared.write_json(output/'decision.json', dict(
        stage='negative-residual-strength development screening',
        positive_residual_beats_G=positive,
        event_deltas=event_totals,
        next_step='inspect lambda-minus and event evidence before a0/rho search',
        not_SOTA_validation=True, no_claim_of_independent_external_validation=True))
    print('ASYMMETRIC_RESIDUAL_SCREEN = PASS (execution completed; not efficacy)',
          flush=True)


def main():
    base.METHODS = METHODS
    base.LOCKED = ()
    base.PROTOCOL = PROTOCOL
    base.grids = grids
    base.comparisons = comparisons
    base.screen = screen
    base.main()


if __name__ == '__main__':
    main()
