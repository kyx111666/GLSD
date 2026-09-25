#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f config/model_config.yaml ]]; then
  echo "Copy and review config/model_config_template.yaml as config/model_config.yaml first." >&2
  exit 1
fi

.venv-gpu/bin/python src/run_qwen3vl_verifier.py \
  --config config/model_config.yaml \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --precision bf16

