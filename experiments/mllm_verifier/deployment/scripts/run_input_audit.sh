#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" src/build_full_smoke40.py "$@"
"$PYTHON_BIN" src/verify_label_leakage.py
"$PYTHON_BIN" src/verify_smoke40_inputs.py
"$PYTHON_BIN" src/run_qwen3vl_verifier.py --dry-run --mock-output

