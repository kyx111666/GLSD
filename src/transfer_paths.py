"""Configured response, evidence and runner locations."""
from release_config import settings
_PATHS = settings()
EVIDENCE = {name: p['run_evidence'] for name, p in _PATHS.items()}
RESPONSES = {name: p['responses'] for name, p in _PATHS.items()}
RUNNERS = {name: p['source'] for name, p in _PATHS.items()}
CORES = {name: p['core'] for name, p in _PATHS.items() if name.startswith('metst_')}
