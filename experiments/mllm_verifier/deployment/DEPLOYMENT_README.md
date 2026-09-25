# Smoke40 Qwen3-VL deployment package

This package freezes the 20 TP + 20 FP smoke40 input protocol without running a model. The model-facing manifest contains no evaluation labels. `smoke40_labels_PRIVATE.json` is evaluation-only and must never be passed to the inference entry point.

## Stage A — Mac / CPU input audit

The committed data package has already been built and audited. To rebuild it from the original local frames and remote ZIP, set the ZIP path if necessary and run:

```bash
export REMOTE_TP_ZIP=/path/to/smoke11_remote_frames.zip
bash scripts/setup_cpu_audit_env.sh
PYTHON_BIN=.venv-audit/bin/python bash scripts/run_input_audit.sh
```

Expected terminal markers:

```text
SMOKE40_INPUT_GATE_PASS
LABEL_LEAKAGE_GATE_PASS
SMOKE40_LOCKFILE_CREATED
MOCK_DRY_RUN_PASS
```

The builder verifies the ZIP against the existing sidecar SHA256, checks the remote extraction PASS marker, all 11 candidate identities, all 99 frame mappings, and every remote image hash before it imports data. A failed remote check stops with `REMOTE_TP_IMPORT_FAILED`.

## Stage B — future GPU server

Upload the complete `deployment/` directory. Review rather than blindly run the environment template because the correct PyTorch build depends on the server driver:

```bash
bash scripts/setup_gpu_env_TEMPLATE.sh
cp config/model_config_template.yaml config/model_config.yaml
.venv-gpu/bin/python src/run_qwen3vl_verifier.py \
  --config config/model_config.yaml \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --precision int4
```

Supported explicit precision choices are `bf16`, `fp16`, `int8`, and `int4`. There is no silent fallback. The 24/32GB deployment path is the same 8B checkpoint with int4 or int8 weight loading; candidates, prompt, frames, and evaluator remain unchanged. Using a smaller checkpoint is a separate `MODEL_VARIANT_CHANGED` experiment and must not be merged with the 8B result.

`precision_mode: auto` in the YAML selects bf16 when the GPU reports bf16 support and otherwise fp16, and prints the selected mode. For a locked hardware run, set one of the four explicit modes in the YAML or pass `--precision`; the CLI value takes precedence.

Before inference, rerun `src/verify_label_leakage.py` and `src/verify_smoke40_inputs.py`. Compare `data/audit/SMOKE40_LOCKFILE.json` after transfer; any changed image or locked input must stop the run.

## Stage C — evaluation and replay

Only real results may be evaluated:

```bash
.venv-gpu/bin/python src/evaluate_smoke40.py --raw results/smoke40_raw.json
.venv-gpu/bin/python src/deterministic_replay.py \
  --first results/replay_first.json \
  --second results/replay_second.json
```

The evaluator deliberately rejects dry-run output with `MOCK_RESULT_NOT_EVALUABLE`. It writes `results/SMOKE40_REPORT.md`; PASS/BORDERLINE/FAIL is an exploration gate only, not a statistical-significance conclusion.

## Scientific lock

- ME-TST + SAMMLV + GLSD `predictions.pure` is unchanged.
- Candidate geometry and the 20/20 protocol are unchanged.
- The prompt requests transient localized facial-change verification, not emotion classification.
- The verifier is veto-only and has zero trainable parameters.
- No checkpoint or CUDA component is bundled here.
