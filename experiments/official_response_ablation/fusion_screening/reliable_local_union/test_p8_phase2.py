"""Portable protocol tests. Synthetic counts are not research evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import run_p8_phase2 as phase


class FakeFeatures:
    def __init__(self, response, k):
        self.response = np.asarray(response)

    def evidence(self, scale, radius):
        if len(self.response) == 0:
            return np.empty(0, int), np.empty((0, 2))
        return np.array([1, 3]), np.array([[.8 / scale, .6 * radius / 2], [.4 / scale, .9 * radius / 2]])

    def selected_peaks(self, c):
        peaks, values = self.evidence(c.reference_scale, c.local_radius)
        return peaks[values.mean(axis=1) >= c.threshold]


BASE = SimpleNamespace(GLSDFeatures=FakeFeatures)


class Tests(unittest.TestCase):
    def test_grid_no_dummy_rho_for_G(self):
        grids = phase.grids()
        self.assertEqual({m: len(g) for m, g in grids.items()}, dict(G=57, L=171, Mean=171, GM_p8=171))
        for method, configs in grids.items():
            self.assertEqual([c.config_id for c in configs], list(range(len(configs))))
            self.assertEqual({c.threshold for c in configs}, set(phase.TAUS))
        self.assertEqual({c.local_radius for c in grids['G']}, {1.0})

    def test_structure_cache_and_legacy_replay_including_empty(self):
        core = phase.FusionCore(BASE)
        view = core.GLSDFeatures([1, 2, 3, 4], 2)
        for configs in phase.grids().values():
            for c in configs:
                got = view.selected_peaks(c)
                peaks, values = view.base.evidence(c.reference_scale, c.local_radius)
                np.testing.assert_array_equal(got, peaks[phase.score(values, c.method) >= c.threshold])
        self.assertEqual(len(view.cache), 9)
        self.assertIs(core.GLSDFeatures([1, 2, 3, 4], 2), view)
        empty = core.GLSDFeatures([], 2)
        self.assertEqual(len(empty.selected_peaks(phase.grids()['Mean'][0])), 0)

    def test_score_equations_and_bad_inputs(self):
        x = np.array([[.9, .1], [.55, .55], [0, 0]])
        np.testing.assert_array_equal(phase.score(x, 'Mean'), x.mean(1))
        np.testing.assert_allclose(phase.score(x, 'GM_p8'), ((x[:, 0]**8 + x[:, 1]**8)/2)**(1/8))
        for bad in (np.array([[np.nan, 0]]), np.array([[-1e-14, 0]]), np.array([[1.1, .1]])):
            with self.assertRaises(ValueError):
                phase.score(bad, 'GM_p8')

    def test_full_selector_heldout_poison_and_tie(self):
        raw = np.array([[[5, 0, 0], [4, 0, 0]], [[0, 0, 5], [1, 9, 3]]])
        full = raw[::-1].copy()
        self.assertEqual(phase.shared.choose(raw, 0)[0], 0)
        self.assertEqual(phase.shared.choose(full, 0)[0], 1)
        old, pool = phase.shared.choose(full, 0)
        full[:, 0] = [[99999, 0, 0], [0, 99999, 99999]]
        winner, after = phase.shared.choose(full, 0)
        self.assertEqual(winner, old)
        np.testing.assert_array_equal(pool, after)
        self.assertEqual(phase.shared.choose(np.ones((3, 2, 3), int), 0)[0], 0)

    def test_checkpoint_resume_after_interruption(self):
        configs = phase.grids()['G'][:2]
        context = (None, None, None, None, None, None, ['a', 'b'], None)
        calls = []
        def evaluate(context, core, si, c):
            calls.append((si, c.config_id))
            if si == 1:
                raise RuntimeError('simulate runtime interruption')
            return (1, 2, 3), (1, 1, 3), None, []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'counts.npz'
            with patch.object(phase, 'checked_decode', side_effect=evaluate):
                with self.assertRaises(RuntimeError):
                    phase.search_counts(context, None, configs, path)
            with np.load(path) as data:
                self.assertEqual(data['done'].tolist(), [True, False])
            calls.clear()
            def resume(context, core, si, c):
                calls.append((si, c.config_id))
                return (2, 3, 4), (2, 1, 4), None, []
            with patch.object(phase, 'checked_decode', side_effect=resume):
                raw, full = phase.search_counts(context, None, configs, path)
            self.assertEqual(calls, [(1, 0), (1, 1)])
            np.testing.assert_array_equal(full[:, 0], [[1, 1, 3], [1, 1, 3]])

    def test_minimal_official_interface_integration(self):
        class Runner:
            def decode_glsd_subject(self, records, si, config, core, metric, official, recognition):
                self.assert_recognition = recognition
                count = len(core.GLSDFeatures([1, 2, 3, 4], 2).selected_peaks(config))
                raw = (count, 2, 3-count)
                return raw, {'synthetic': si}, [], [], None

            def full_counts_from_official_synergy(self, raw, pred_list, gt_list):
                return (raw[0], 1, raw[2])
        runner = Runner()
        context = (runner, BASE, None, None, [], [], ['a', 'b', 'c'], None)
        ora = SimpleNamespace(SPECS={'synthetic': {}}, metst_context=lambda spec: context)
        tiny = {m: [phase.Config(m, i, 1, 1, t) for i, t in enumerate((.2, .7))] for m in phase.METHODS}
        with tempfile.TemporaryDirectory() as directory, patch.object(phase, 'grids', return_value=tiny):
            output = Path(directory) / 'result'
            rows, comparisons = phase.run_setting(ora, 'synthetic', output)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(comparisons), 3)
            self.assertTrue(all(r['selector'] == r['evaluation'] == 'full' for r in rows))
            self.assertTrue(json.loads((output/'completion.json').read_text())['completed'])
            self.assertTrue(runner.assert_recognition)
            self.assertTrue((output / 'selected_predictions.jsonl.gz').exists())

    def test_protected_path_overlap(self):
        self.assertTrue(phase.overlaps(Path('/a/b'), Path('/a')))
        self.assertTrue(phase.overlaps(Path('/a'), Path('/a/b')))
        self.assertFalse(phase.overlaps(Path('/a/b'), Path('/a/c')))


if __name__ == '__main__':
    unittest.main()
