#!/usr/bin/env python3
"""Run GLSD experiments with explicit local input paths."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from release_config import SETTINGS, paths

DIRECTIONS = ('metst_to_boosting_sammlv', 'boosting_to_metst_sammlv',
              'metst_to_boosting_casme3', 'boosting_to_metst_casme3')

def invoke(name, argv):
    sys.argv = [name + '.py', *map(str, argv)]
    importlib.import_module(name).main()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'main', 'search', 'threshold', 'runtime', 'replay', 'ablation',
                        'transfer-audit', 'transfer-gate', 'transfer', 'transfer-summary'))
    parser.add_argument('--config', type=Path, required=True,
                        help='input JSON; paths inside it are relative to its own directory')
    parser.add_argument('--setting', choices=SETTINGS)
    parser.add_argument('--selections', type=Path, help='selected configuration CSV for threshold/runtime')
    parser.add_argument('--direction', choices=DIRECTIONS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe-only', action='store_true', help='ablation alpha=0.5 alignment only')
    args = parser.parse_args()
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f'configuration does not exist: {config}')
    os.environ['GLSD_CONFIG'] = str(config)
    # Parse before importing an upstream implementation or creating output directories.
    try:
        json.loads(config.read_text())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    selections = args.selections.resolve() if args.selections else None
    setting = args.setting
    if args.command == 'transfer':
        if not args.direction:
            parser.error('transfer requires --direction')
        target, dataset = args.direction.split('_to_')[1].rsplit('_', 1)
        setting = target + '_' + dataset
    if args.command in ('main', 'search', 'threshold', 'runtime', 'replay', 'ablation') and not setting:
        parser.error(f'{args.command} requires --setting')
    if args.command == 'ablation' and not setting.startswith('metst_'):
        parser.error('Table 3 ablation supports the two ME-TST+ settings')
    if args.command == 'check':
        missing = []
        for name in ([setting] if setting else SETTINGS):
            p = paths(name)
            required = [p['upstream'] / 'training_utils.py', p['upstream'] / 'Utils',
                        p['responses'], p['evidence'] / 'evidence_bundle_sha256.txt',
                        p['run_evidence'] / 'run_manifest.json',
                        p['run_evidence'] / 'per_subject_counts.csv',
                        p['locked_results'] / 'outer_selected_configs.csv', p['source']]
            if name.startswith('metst_'):
                required += [p['core'], p['preregistration']]
            else:
                required += [p['evidence'] / 'glsd_search_inner_fold_counts.csv']
            for path in required:
                if not path.exists(): missing.append(f'{name}: {path}')
        core = ROOT / 'src/boosting_official_glds_full_LOCKED.py'
        if hashlib.sha256(core.read_bytes()).hexdigest() != '6f500e5af179a28e8a4f5e49a926004c1a60887a3cc27b1e9df2b9d61c94ca3d':
            parser.error('frozen core checksum mismatch')
        if missing:
            print('Missing external inputs:\n' + '\n'.join(missing))
            return 2
        print('Input paths and frozen core checksum: PASS (no dataset evaluation performed)')
        return 0
    if args.output is None:
        parser.error(f'{args.command} requires --output')
    output = args.output.resolve()
    if args.command in ('main', 'search', 'threshold', 'runtime', 'replay', 'ablation') and output.exists():
        parser.error('use a new output directory; existing results are not overwritten')
    if setting:
        p = paths(setting)
        if not (p['upstream'] / 'training_utils.py').is_file():
            parser.error(f'missing upstream training_utils.py in {p["upstream"]}')
        sys.path.insert(1, str(p['upstream']))
        os.chdir(p['upstream'])
    # Shared replay adapter supplies the original pandas compatibility shim.
    importlib.import_module('official_response_component_ablation')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command in ('search', 'threshold', 'runtime'):
        importlib.import_module('paper_experiments').run(args.command, setting, output, selections)
    elif args.command == 'main':
        if setting.startswith('metst_'):
            runner = importlib.import_module(p['source'].stem)
            runner.OUTPUT_DIR = output
            runner.main()
        else:
            invoke(p['source'].stem, ['--cache', p['responses'], '--output_dir', output])
    elif args.command == 'replay':
        if setting == 'boosting_casme3':
            invoke('boosting_casme3_official_response_preflight', [
                '--preflight-only', '--runner', p['source'], '--official-repo-root', p['upstream'],
                '--cache', p['responses'], '--evidence-dir', p['evidence'], '--output', output])
        else:
            invoke('official_response_component_ablation', [
                '--setting', setting, '--preflight-only', '--output-root', output])
    elif args.command == 'ablation':
        invoke('run_locked_alpha_ablation', ['--setting', setting.removeprefix('metst_'),
               '--output', output] + (['--probe-only'] if args.probe_only else []))
    elif args.command == 'transfer-audit':
        invoke('transfer_audit', ['--output-root', output])
    elif args.command == 'transfer-gate':
        argv = ['--output-root', output]
        for name in SETTINGS:
            argv += ['--' + name.replace('_', '-'), output / 'preflight' / name]
        invoke('consolidate_preflight', argv)
    elif args.command == 'transfer':
        invoke('run_transfer_target', ['--direction', args.direction, '--output-root', output])
    elif args.command == 'transfer-summary':
        invoke('finalize_transfer', ['--output-root', output])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
