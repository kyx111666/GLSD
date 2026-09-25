#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# TEMPLATE ONLY. Review the host driver and install the matching PyTorch build.
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip
.venv-gpu/bin/python -m pip install torch transformers accelerate qwen-vl-utils

# Install only when --precision int8 or int4 is intended and supported:
# .venv-gpu/bin/python -m pip install bitsandbytes

echo "GPU_ENV_TEMPLATE_COMPLETE"

