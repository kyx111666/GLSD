"""Select GLSD configurations from per-subject search counts."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import run_full_fusion_tuning as full


def load_counts(directory):
    tables = {}
    subjects = None
    for method, size in (("EqualMean", 90), ("G", 30), ("L", 90)):
        with np.load(directory / f"search_counts_{method}.npz", allow_pickle=False) as data:
            counts = data['raw_counts']
            names = data['subjects']
        if names.ndim != 1 or len(names) < 2 or len(set(names.tolist())) != len(names):
            raise ValueError(f'{method}: expected at least two distinct subjects')
        if counts.shape != (size, len(names), 3):
            raise ValueError(f'{method}: invalid count dimensions')
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
            raise ValueError(f'{method}: counts must be finite nonnegative integers')
        if subjects is not None and not np.array_equal(names, subjects):
            raise ValueError('subject ordering differs between count files')
        subjects = names
        tables[method] = counts
    return tables, subjects


def select(tables, subjects):
    rows = []
    grid = full.grids()['EqualMean']
    for outer, subject in enumerate(subjects):
        winner, pooled, details = full.choose_mean_synergy(
            tables['EqualMean'], tables['G'], tables['L'], outer)
        rows.append(dict(subject=str(subject), **asdict(grid[winner]),
                         inner_F1=full.shared.metrics(pooled[winner])['F1'],
                         epsilon=full.MEAN_SELECTION_EPSILON,
                         synergy_margin=details['selected_synergy_margin']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output file')
    tables, subjects = load_counts(args.counts)
    rows = select(tables, subjects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full.shared.write_csv(args.output, rows)
    print(f'Selected configurations for {len(rows)} subjects: {args.output}')


if __name__ == '__main__':
    main()
