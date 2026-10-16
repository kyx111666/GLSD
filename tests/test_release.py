"""Regression checks for filesystem configuration and public entry points."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import release_config

class ReleaseTests(unittest.TestCase):
    def test_paths_are_relative_to_config_not_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'inputs.json'
            config.write_text(json.dumps({'metst_sammlv': {'responses': 'records'}}))
            with patch.dict(os.environ, {'GLSD_CONFIG': str(config)}):
                self.assertEqual(release_config.paths('metst_sammlv')['responses'], (Path(directory) / 'records').resolve())
                self.assertEqual(release_config.paths('metst_sammlv')['source'].parent, ROOT / 'src')

    def test_absolute_and_home_paths(self):
        self.assertEqual(release_config.resolve(Path('/unused'), '/tmp/data'), Path('/tmp/data').resolve())
        self.assertEqual(release_config.resolve(Path('/unused'), '~/data'), (Path.home() / 'data').resolve())

    def test_missing_inputs_fail_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'inputs.json'
            config.write_text('{}')
            output = Path(directory) / 'result'
            result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'check', '--config', str(config), '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Missing external inputs', result.stdout)
            self.assertFalse(output.exists())

    def test_existing_main_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'inputs.json'
            config.write_text('{}')
            result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'main', '--config', str(config), '--setting', 'boosting_sammlv', '--output', directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('not overwritten', result.stderr)
            self.assertEqual(config.read_text(), '{}')

    def test_upstream_cannot_shadow_the_release_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            upstream = base / 'upstream'
            upstream.mkdir()
            (upstream / 'training_utils.py').write_text('')
            (upstream / 'boosting_official_glds_full.py').write_text('raise RuntimeError("UPSTREAM_RUNNER_SHADOWED")')
            config = base / 'inputs.json'
            config.write_text(json.dumps({'boosting_sammlv': {'upstream': str(upstream), 'responses': 'missing.pkl'}}))
            result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'main', '--config', str(config), '--setting', 'boosting_sammlv', '--output', str(base / 'result')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('cache does not exist', result.stderr)
            self.assertNotIn('UPSTREAM_RUNNER_SHADOWED', result.stderr)

    def test_example_works_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / 'examples/decode_response.py')], cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['selected_peaks'], [36, 85])

if __name__ == '__main__':
    unittest.main()
