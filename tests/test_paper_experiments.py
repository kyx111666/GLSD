"""Tests for search-to-selection, fixed-parameter perturbations and timing."""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import csv
import unittest

import numpy as np
import paper_experiments as paper
import run_full_fusion_tuning as fusion
import boosting_official_glds_full_LOCKED as base


class ToyExperiment:
    setting = 'metst_sammlv'
    subjects = ['a', 'b', 'c']
    metric = object
    def __init__(self):
        self.calls = []
    def native(self, index):
        return (1, 2, 2), (1, 1, 2)
    def decode(self, index, config, recognition=True):
        self.calls.append((index, config, recognition))
        counts = (2, config.config_id % 3, 1)
        return counts, counts if recognition else None


class PaperExperimentsTests(unittest.TestCase):
    def test_search_creates_inputs_accepted_by_selector(self):
        from select_glsd import load_counts, select
        with TemporaryDirectory() as directory, patch.object(paper.evaluator, 'install', return_value=nullcontext()):
            output = Path(directory)
            with patch('builtins.print'):
                paper.search(ToyExperiment(), output)
            tables, subjects = load_counts(output)
            selected = select(tables, subjects)
            configs = paper.selected_configs(output / 'selected_configs.csv', subjects)
            self.assertEqual([configs[s].config_id for s in subjects],
                             [row['config_id'] for row in selected])
            self.assertTrue((output / 'without_G_subjects.csv').is_file())
            self.assertTrue((output / 'without_L_subjects.csv').is_file())

    def test_threshold_changes_only_threshold_and_keeps_fold_assignment(self):
        experiment = ToyExperiment()
        with TemporaryDirectory() as directory, patch.object(paper.evaluator, 'install', return_value=nullcontext()):
            output = Path(directory)
            selections = output / 'choices.csv'
            fusion.shared.write_csv(selections, [dict(subject=s, config_id=30+i) for i, s in enumerate(experiment.subjects)])
            original = paper.selected_configs(selections, experiment.subjects)
            paper.threshold(experiment, output, selections)
            for offset, calls in zip((-.05, 0, .05), (experiment.calls[:3], experiment.calls[3:6], experiment.calls[6:])):
                for index, config, recognition in calls:
                    before = original[experiment.subjects[index]]
                    self.assertEqual(config.reference_scale, before.reference_scale)
                    self.assertEqual(config.local_radius, before.local_radius)
                    self.assertAlmostEqual(config.threshold, max(0, before.threshold + offset))
                    self.assertTrue(recognition)

    def test_timing_alternates_order_and_excludes_loading(self):
        experiment = ToyExperiment()
        experiment.official = SimpleNamespace(MeanAveragePrecision2d=object)
        experiment.responses = lambda i: ([np.zeros(32)], 3)
        seen = []
        experiment.predictions = lambda i, config: seen.append((i, config is not None))
        with TemporaryDirectory() as directory:
            output = Path(directory)
            selections = output / 'choices.csv'
            fusion.shared.write_csv(selections, [dict(subject=s, config_id=0) for s in experiment.subjects])
            paper.runtime(experiment, output, selections, repeats=2)
            self.assertEqual(len(seen), 18)
            self.assertEqual([flag for i, flag in seen[6:] if i == 0], [False, True, True, False])
            self.assertIs(experiment.official.MeanAveragePrecision2d, object)
            with (output / 'runtime.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_boosting_hook_does_not_recurse(self):
        core = fusion.FusionCore(SimpleNamespace(GLSDFeatures=base.GLSDFeatures))
        response = np.sin(np.arange(120) / 7) + 1
        with patch.object(base, 'GLSDFeatures', core.GLSDFeatures):
            peaks = base.GLSDFeatures(response, 5).selected_peaks(fusion.grids()['EqualMean'][0])
        self.assertGreater(len(peaks), 0)

    def test_metric_binding_restored_on_failure(self):
        official = SimpleNamespace(MeanAveragePrecision2d=object)
        with self.assertRaises(RuntimeError):
            with paper.without_metrics(official):
                raise RuntimeError('test')
        self.assertIs(official.MeanAveragePrecision2d, object)


if __name__ == '__main__':
    unittest.main()
