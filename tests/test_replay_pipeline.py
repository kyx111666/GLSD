"""Replay output compatibility and transfer validation checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import boosting_casme3_official_response_preflight as replay
import consolidate_preflight as gates
import finalize_transfer as summaries
from transfer_common import DIRECTIONS


class ReplayPipelineTests(unittest.TestCase):
    def manifest(self):
        return dict(status='PASS', all_functional_checks_passed=True,
                    native=dict(full_counts=[84, 786, 769]),
                    glsd_90=dict(full_counts=[120, 1110, 733]))

    def test_replay_output_is_readable_by_transfer_validation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'replay'
            replay.write_replay_outputs(output, self.manifest(), np.zeros((2, 3, 3), dtype=int),
                                       [], [], SimpleNamespace(normalize=lambda x: x))
            result = gates.casme3(output)
            self.assertEqual(result['native'], (84, 786, 769))
            self.assertEqual(result['tuned'], (120, 1110, 733))
            with (output / 'replay_manifest.json').open() as handle:
                manifest = json.load(handle)
            self.assertEqual(len(manifest['replay_artifacts']), 3)

    def test_transfer_validation_rejects_incorrect_replay_counts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.manifest()
            manifest['glsd_90']['full_counts'][1] += 1
            (output / 'replay_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, 'exact counts mismatch'):
                gates.casme3(output)

    def test_native_validation_failure_stops_replay_without_outputs(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            evidence = base / 'evidence'
            evidence.mkdir()
            for filename in ('run_manifest.json', 'per_subject_counts.csv', 'outer_selected_configs.json',
                             'outer_selection_trace.json', 'glsd_search_inner_fold_counts.csv'):
                (evidence / filename).write_text('{}')
            runner_path, cache_path = base / 'runner.py', base / 'cache.pkl'
            runner_path.write_text('')
            cache_path.write_bytes(b'fixture')
            args = SimpleNamespace(runner=runner_path, cache=cache_path, official_repo_root=base,
                                   evidence_dir=evidence, output=base / 'output')
            runner = SimpleNamespace(load_cache=Mock(return_value=({}, [])), group_videos=Mock(return_value=[]))
            with patch.object(replay, 'validate_manifest', return_value={}), \
                 patch.object(replay, 'import_runner', return_value=runner), \
                 patch.object(replay, 'validate_real_runner_api'), \
                 patch.object(replay, 'validate_universe', return_value=(['1'], 1)), \
                 patch.object(replay, 'validate_native', side_effect=RuntimeError('native count mismatch')), \
                 patch.object(replay, 'validate_grid') as grid:
                with self.assertRaisesRegex(RuntimeError, 'native count mismatch'):
                    replay.run_preflight(args)
                grid.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_four_direction_summary_preserves_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'validation').mkdir()
            (root / 'validation/preflight_gate.json').write_text(json.dumps(dict(status='PASS')))
            for direction in summaries.ORDER:
                target = root / DIRECTIONS[direction]['relative_output']
                target.mkdir(parents=True)
                (target / 'protocol_checks.json').write_text(json.dumps(dict(status='PASS', source_k_passed_to_target=False)))
                result = dict(direction=direction, dataset=DIRECTIONS[direction]['dataset'],
                              source_pipeline='ME-TST+', target_pipeline='BoostingVRME',
                              delta_vs_native=.02, delta_vs_tuned=.01,
                              bootstrap_95_ci_vs_native=[-.01,.04], bootstrap_95_ci_vs_tuned=[-.02,.03])
                result.update({name:dict(TP=10, FP=3, FN=2, F1=.8)
                               for name in ('target_native','target_tuned_glsd90','transferred')})
                (target / 'metrics.json').write_text(json.dumps(result))
            with patch.object(sys, 'argv', ['finalize_transfer', '--output-root', str(root)]), patch('builtins.print'):
                summaries.main()
            results = json.loads((root / 'summary/transfer_summary.json').read_text())
            self.assertEqual(len(results), 4)
            self.assertEqual({row['direction'] for row in results}, set(summaries.ORDER))
            self.assertEqual({path.name for path in (root / 'summary').iterdir()},
                             {'transfer_summary.csv', 'transfer_summary.json', 'paper_table.csv'})


if __name__ == '__main__':
    unittest.main()
