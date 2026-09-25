#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r requirements_audit.txt
echo "CPU_AUDIT_ENV_READY"

