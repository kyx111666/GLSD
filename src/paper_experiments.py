"""Search, threshold sensitivity and decoder timing on frozen responses."""
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
import csv
import importlib
import io
import time
from types import SimpleNamespace

import numpy as np
import official_response_component_ablation as ora
import run_full_fusion_tuning as fusion
import one_to_one_evaluator as evaluator
from release_config import paths


class Experiment:
    """Adapt the four upstream response formats to the same experiment loop."""
    def __init__(self, setting):
        self.setting = setting
        self.kind = 'metst' if setting.startswith('metst_') else 'boosting'
        self.cas_boosting = setting == 'boosting_casme3'
        if self.cas_boosting:
            self.runner = importlib.import_module(paths(setting)['source'].stem)
            self.cache, videos = self.runner.load_cache(paths(setting)['responses'])
            self.grouped = self.runner.group_videos(videos)
            self.subjects = [str(group[0]['subject']) for group in self.grouped]
            self.samples = [[v['gt_intervals'] for v in group] for group in self.grouped]
            self.emotions = [[v['emotion_labels'] for v in group] for group in self.grouped]
            self.official = self.runner.tu
            self.metric = self.official.MeanAveragePrecision2d
            self.base = importlib.import_module('boosting_official_glds_full_LOCKED')
        else:
            self.context = (ora.metst_context(ora.SPECS[setting]) if self.kind == 'metst'
                            else ora.boosting_context(ora.SPECS[setting]))
            if self.kind == 'metst':
                (self.runner, self.base, self.metric, self.official, self.records,
                 _, self.subjects, _) = self.context
            else:
                (self.runner, self.cache, self.grouped, self.subjects,
                 self.metric, self.official, _) = self.context
                self.base = SimpleNamespace(GLSDFeatures=self.runner.GLSDFeatures)
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError('duplicate subject identifiers')
        self.core = fusion.FusionCore(self.base)

    def responses(self, index):
        if self.kind == 'metst':
            record = self.records[index]
            return record['result_all'], int(record['k_p'])
        return [v['spot_response'] for v in self.grouped[index]], int(self.cache['config']['k_p'])

    def decode(self, index, config, recognition=True):
        with redirect_stdout(io.StringIO()):
            if not self.cas_boosting:
                return ora.decode_component(self.kind, self.context, self.core, index, config, recognition)
            # The CAS(ME)3 runner accepts scalar feature arguments; scoring uses
            # the same G/L feature view as the other three pipeline adapters.
            class View:
                def __init__(self, feature):
                    self.feature = feature
                def selected_peaks(self, reference, radius, threshold):
                    return self.feature.selected_peaks(replace(config, reference_scale=reference,
                                                               local_radius=radius, threshold=threshold))
            responses, k = self.responses(index)
            bank = [[] for _ in self.subjects]
            bank[index] = [View(self.core.GLSDFeatures(response, k)) for response in responses]
            result = self.runner.decode_subject(index, asdict(config), self.grouped, bank,
                                                self.samples, self.emotions, self.cache['config'],
                                                need_recognition=recognition)
            return tuple(result['raw_counts']), tuple(result['full_counts']) if recognition else None

    def native(self, index):
        with redirect_stdout(io.StringIO()):
            if not self.cas_boosting:
                return ora.native_subject_counts(self.kind, self.context, index)
            from boosting_casme3_official_phase2 import native_subject_counts
            return native_subject_counts(self.runner, self.cache['config'], self.grouped,
                                         self.samples, self.emotions, index)

    def predictions(self, index, config=None):
        """Decode event intervals without recognition or metric computation."""
        responses, k = self.responses(index)
        samples = [[[] for _ in responses]]
        if config is None:
            from contextlib import nullcontext
            source = nullcontext()
        else:
            # Fresh features on every timed call: do not time a warmed search cache.
            selected = [self.base.GLSDFeatures(response, k).selected_peaks(config) for response in responses]
            source = (self.runner.metst_glsd_candidate_source(self.official, selected)
                      if self.kind == 'metst' else
                      importlib.import_module('boosting_official_glds_full_LOCKED').official_glsd_candidate_source(self.official, selected))
        with source:
            if self.kind == 'metst':
                return self.official.spotting(samples, 0, responses, 0, self.runner.NATIVE_P,
                                              NullMetric(), k)[0]
            return self.official.spotting(samples, 0, responses, 0, NullMetric(), k,
                                          self.cache['config']['interval_strategy'],
                                          ('CASME_3' if self.cas_boosting else 'SAMMLV'))[0]


class NullMetric:
    def __init__(self, *args, **kwargs):
        pass
    def add(self, *args, **kwargs):
        pass


@contextmanager
def without_metrics(official):
    original = official.MeanAveragePrecision2d
    official.MeanAveragePrecision2d = NullMetric
    try:
        yield
    finally:
        official.MeanAveragePrecision2d = original


def selected_configs(path, subjects):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    grid = fusion.grids()['EqualMean']
    selected = {}
    for row in rows:
        if row.get('method', 'EqualMean') not in ('EqualMean', 'GLSD', 'GL'):
            continue
        subject = str(row.get('subject', row.get('outer_subject', '')))
        if subject in selected:
            raise ValueError(f'duplicate selection for {subject}')
        if all(row.get(k) for k in ('reference_scale', 'local_radius', 'threshold')):
            config = fusion.Config('EqualMean', int(row.get('config_id', 0)),
                                   *[float(row[k]) for k in ('reference_scale', 'local_radius', 'threshold')])
        else:
            index = int(row['config_id'])
            if not 0 <= index < len(grid):
                raise ValueError('configuration index outside GLSD grid')
            config = grid[index]
        if config.reference_scale not in fusion.SCALES or config.local_radius not in fusion.RADII or not np.isfinite(config.threshold):
            raise ValueError('invalid selected parameters')
        selected[subject] = config
    if set(selected) != set(subjects):
        raise ValueError('selection subjects differ from response subjects')
    return selected


def write_counts(output, filename, setting, method, subjects, pairs, **extra):
    rows, summary = [], []
    for stage, position in (('raw', 0), ('full', 1)):
        counts = [pair[position] for pair in pairs]
        for subject, count in zip(subjects, counts):
            rows.append(dict(setting=setting, method=method, subject=subject, stage=stage,
                             **extra, **fusion.shared.metrics(count)))
        summary.append(dict(setting=setting, method=method, stage=stage, **extra,
                            **fusion.shared.metrics(np.sum(counts, axis=0))))
    fusion.shared.write_csv(output / (filename + '_subjects.csv'), rows)
    return summary


def search(experiment, output):
    tables, choices, summaries = {}, [], []
    subjects = experiment.subjects
    with evaluator.install(experiment.metric):
        native = [experiment.native(i) for i in range(len(subjects))]
        summaries += write_counts(output, 'native', experiment.setting, 'Native', subjects, native)
        for method in ('G', 'L', 'EqualMean'):
            grid = fusion.grids()[method]
            table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
            for i, subject in enumerate(subjects):
                for config in grid:
                    table[config.config_id, i] = experiment.decode(i, config, False)[0]
                print(f'{method}: {i + 1}/{len(subjects)} subjects', flush=True)
            tables[method] = table
            np.savez_compressed(output / f'search_counts_{method}.npz', raw_counts=table,
                                subjects=np.asarray(subjects, dtype=str))
            results = []
            selected = []
            for i, subject in enumerate(subjects):
                if method == 'EqualMean':
                    winner, pooled, _ = fusion.choose_mean_synergy(table, tables['G'], tables['L'], i)
                else:
                    winner, pooled = fusion.shared.choose(table, i)
                config = grid[winner]
                pair = experiment.decode(i, config)
                if tuple(pair[0]) != tuple(table[winner, i]):
                    raise RuntimeError('selected raw counts differ from search counts')
                results.append(pair)
                selected.append(config)
                choices.append(dict(subject=subject, **asdict(config),
                                    inner_F1=fusion.shared.metrics(pooled[winner])['F1']))
            summaries += write_counts(output, method, experiment.setting, method, subjects, results)
            if method == 'EqualMean':
                for label, component in (('without_L', 'G'), ('without_G', 'L')):
                    pairs = [experiment.decode(i, replace(config, method=component))
                             for i, config in enumerate(selected)]
                    summaries += write_counts(output, label, experiment.setting, label, subjects, pairs)
    fusion.shared.write_csv(output / 'selected_configs.csv', choices)
    fusion.shared.write_csv(output / 'summary.csv', summaries)


def threshold(experiment, output, selections):
    configs = selected_configs(selections, experiment.subjects)
    summaries = []
    with evaluator.install(experiment.metric):
        native = [experiment.native(i) for i in range(len(experiment.subjects))]
        summaries += write_counts(output, 'native', experiment.setting, 'Native', experiment.subjects, native,
                                 threshold_offset=0.0)
        for offset in (-0.05, 0.0, 0.05):
            pairs = []
            for i, subject in enumerate(experiment.subjects):
                config = configs[subject]
                changed = replace(config, threshold=max(0.0, round(config.threshold + offset, 10)))
                pairs.append(experiment.decode(i, changed))
            summaries += write_counts(output, f'threshold_{offset:+.2f}', experiment.setting,
                                      'GLSD', experiment.subjects, pairs, threshold_offset=offset)
    fusion.shared.write_csv(output / 'threshold_sensitivity.csv', summaries)


def runtime(experiment, output, selections, repeats=10):
    configs = selected_configs(selections, experiment.subjects)
    video_count = sum(len(experiment.responses(i)[0]) for i in range(len(experiment.subjects)))
    times = {'Native': [], 'GLSD': []}
    def execute(method):
        for i, subject in enumerate(experiment.subjects):
            experiment.predictions(i, configs[subject] if method == 'GLSD' else None)
    with without_metrics(experiment.official):
        for method in times:
            execute(method)
        for repeat in range(repeats):
            for method in (('Native', 'GLSD') if repeat % 2 == 0 else ('GLSD', 'Native')):
                start = time.perf_counter()
                execute(method)
                times[method].append(time.perf_counter() - start)
    times['Extra'] = [g - n for g, n in zip(times['GLSD'], times['Native'])]
    rows = [dict(setting=experiment.setting, method=method, repeats=repeats, videos=video_count,
                 milliseconds_per_video_mean=float(np.mean(values)) * 1000 / video_count,
                 milliseconds_per_video_median=float(np.median(values)) * 1000 / video_count,
                 milliseconds_per_video_std=float(np.std(values)) * 1000 / video_count)
            for method, values in times.items()]
    fusion.shared.write_csv(output / 'runtime.csv', rows)
    fusion.shared.write_csv(output / 'runtime_repeats.csv',
                           [dict(method=method, repeat=i + 1, seconds=value)
                            for method, values in times.items() for i, value in enumerate(values)])


def run(command, setting, output, selections=None):
    experiment = Experiment(setting)
    if command != 'search' and selections is None:
        selections = paths(setting)['locked_results'] / 'outer_selected_configs.csv'
    output.mkdir(parents=True, exist_ok=False)
    if command == 'search':
        search(experiment, output)
    elif command == 'threshold':
        threshold(experiment, output, selections)
    elif command == 'runtime':
        runtime(experiment, output, selections)
    else:
        raise ValueError(command)
