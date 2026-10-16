"""Filesystem configuration for the published experiment entry points."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('metst_sammlv', 'metst_casme3', 'boosting_sammlv', 'boosting_casme3')
RUNNERS = {
    'metst_sammlv': 'metst_official_glds_full.py',
    'metst_casme3': 'metst_casme3_official_glds_full.py',
    'boosting_sammlv': 'boosting_official_glds_full.py',
    'boosting_casme3': 'boosting_official_glds_full_casme3.py',
}

def resolve(base, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()

def settings():
    config = os.environ.get('GLSD_CONFIG')
    if config:
        path = Path(config).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        base = path.parent
    else:
        data, base = {}, ROOT
    result = {}
    for name in SETTINGS:
        pipeline = name.split('_')[0]
        raw = data.get(name, {})
        defaults = {
            'upstream': f'external/{"ME-TST" if pipeline == "metst" else "BoostingVRME"}',
            'responses': f'data/{name}/' + ('responses' if pipeline == 'metst' else 'responses.pkl'),
            'evidence': f'data/{name}/evidence',
            'preregistration': f'data/{name}/preregistration.json',
        }
        paths = {k: resolve(base, raw.get(k, v)) for k, v in defaults.items()}
        paths['run_evidence'] = resolve(base, raw['run_evidence']) if 'run_evidence' in raw else paths['evidence']
        paths['locked_results'] = resolve(base, raw['locked_results']) if 'locked_results' in raw else paths['run_evidence']
        paths['source'] = ROOT / 'src' / RUNNERS[name]
        paths['core'] = ROOT / 'src/boosting_official_glds_full_LOCKED.py'
        result[name] = paths
    return result

def paths(name):
    return settings()[name]

def configure_specs(specs):
    for name, spec in specs.items():
        p = paths(name)
        for key in ('source', 'evidence', 'run_evidence', 'locked_results'):
            spec[key] = p[key]
        if spec['kind'] == 'metst':
            spec['dump'], spec['core'] = p['responses'], p['core']
        else:
            spec['cache'] = p['responses']
