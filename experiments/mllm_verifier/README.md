# Frozen Qwen3-VL Candidate Verifier — Feasibility Exploration

This directory is a self-contained, removable experiment for testing whether
a frozen Qwen3-VL model can veto false-positive candidates already produced by
the locked ME-TST+ / SAMMLV GLSD-90 decoder. It does not generate candidates,
change intervals, train parameters, fuse GLSD scores, or modify the parent
project.

Current status: **blocked before smoke40 construction** because the local raw
SAMMLV frame set maps to only 9 canonical GLSD TP and 23 canonical GLSD FP.
The protocol requires 20 TP and 20 FP. Real inference is separately blocked by
the absent Qwen3-VL checkpoint and Transformers runtime. See
`PROJECT_AUDIT.md` for exact evidence.

## Isolation and rollback

Every new file, copied frame, contact sheet, log, and result is contained below
`mllm_verifier_exploration/`. Deleting this directory rolls back the entire
exploration. The parent workspace is not a Git repository, so no Git branch
was created.

## Data separation

- `data/smoke40_manifest.json` is evaluation-only and contains TP/FP labels.
- `data/inference_manifest.json` contains only candidate IDs and visual-frame
  paths. The model runner recursively rejects GT/TP/FP/matching/score/threshold
  keys before it imports or loads the model.
- The inference output never contains TP/FP labels.
- Only `evaluate_smoke40.py` joins verdicts with labels.

## Environment

The lightweight audit, sampling, extraction, parsing, and evaluation scripts
need Python 3.11+, Pillow, and (for the independent workbook audit only)
pandas/openpyxl. Real Qwen3-VL inference additionally needs a compatible
PyTorch build, Transformers with Qwen3-VL support, Accelerate, and
qwen-vl-utils. Optional 4/8-bit loading requires a platform-supported
bitsandbytes installation.

Do not substitute a different MLLM. The intended checkpoint is:

```text
Qwen/Qwen3-VL-8B-Instruct
```

All model parameters are forced to `requires_grad=False`, inference uses
`model.eval()` and `torch.no_grad()`, and the runner aborts unless trainable
parameters equal zero.

## Commands

Run the fast protocol tests:

```bash
/opt/miniconda3/envs/torch-mac/bin/python -m unittest discover -s mllm_verifier_exploration/tests -v
```

After the missing SAMMLV directories are restored, explicitly freeze a
subject-level development scope and build the balanced manifest:

```bash
/opt/miniconda3/envs/torch-mac/bin/python mllm_verifier_exploration/src/build_smoke40.py \
  --designate-development-scope
```

The selected development subjects are written to
`data/development_scope.json` and must be excluded from any future final
verifier test. To reuse an already frozen scope, pass
`--development-subjects-file data/development_scope.json` instead.

Extract nine ordered frames and a contact sheet for every candidate:

```bash
/opt/miniconda3/envs/torch-mac/bin/python mllm_verifier_exploration/src/extract_candidate_frames.py
```

Validate the sanitized manifest, frame order, prompt, and leakage gate without
loading the checkpoint:

```bash
/opt/miniconda3/envs/torch-mac/bin/python mllm_verifier_exploration/src/run_qwen3vl_verifier.py --dry-run
```

If the balanced manifest does not yet exist, the same command records an
interface-only dry run and explicitly marks manifest/frame/leakage runtime
checks as pending; it does not report a full dry-run pass.

Run frozen inference after the requested checkpoint and dependencies are
available:

```bash
/path/to/qwen3vl/python mllm_verifier_exploration/src/run_qwen3vl_verifier.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dtype bf16
```

For a supported inference-only quantized environment, replace the final option
with `--quantization 8bit` or `--quantization 4bit`. The fixed generation
configuration uses seed 100, `do_sample=False`, no confidence score, and no
chain-of-thought request. Five seed-fixed candidates are inferred twice for
Gate D.

Evaluate only after inference completes:

```bash
/opt/miniconda3/envs/torch-mac/bin/python mllm_verifier_exploration/src/evaluate_smoke40.py
```

`locked_v1.yaml` is intentionally absent. Under the protocol it may only be
created after smoke40 achieves PASS and the result is first reported to the
user. No development-pool experiment is implemented or run at this stage.
