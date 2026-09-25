# Google Colab smoke40 inference

This notebook runs the already locked 40-candidate deployment package. It does not select candidates, extract frames, change the prompt, train parameters, or read source videos.

## First run

1. Upload the entire `deployment/` folder to Google Drive.
2. Open `colab/colab_run_smoke40_qwen3vl.ipynb` in Google Colab.
3. Choose **Runtime → Change runtime type → GPU**.
4. Edit only:

   ```text
   DEPLOYMENT_ROOT
   RESULTS_ROOT
   ```

5. Choose **Runtime → Run all**.
6. Wait for:

   ```text
   TECHNICAL_PREFLIGHT_PASS
   ```

7. The notebook then runs all 40 candidates formally. The two preflight candidates are inferred again; their preflight verdicts are never reused.
8. Open `SMOKE40_REPORT.md` under `RESULTS_ROOT` when the run finishes.

The model is always `Qwen/Qwen3-VL-8B-Instruct`, the seed is always 100, the batch size is always 1, and all model parameters remain frozen.

## Hardware policy

Precision is selected before inference, only from total GPU VRAM:

| Total VRAM | Mode | Uniform maximum image resolution |
|---:|---|---:|
| at least 35 GiB | `bf16` | 1024 × 1024 pixels |
| 20–35 GiB | `int4` (NF4) | 768 × 768 pixels |
| below 20 GiB | `int4_low_memory` (NF4) | 512 × 512 pixels |

This supports typical A100, L4, and T4 Colab runtimes. Every candidate still uses the same nine manifest-ordered frames: `B1 B2 B3 C1 C2 C3 A1 A2 A3`.

If BF16 model loading fails specifically because of GPU out-of-memory, the declared fallback is BF16 → INT4. If INT4 also runs out of memory, the notebook stops with `COLAB_HARDWARE_INSUFFICIENT`. It never switches to a smaller or different model.

## Colab interruption and resume

Reconnect to a GPU and run the notebook again. A formal candidate is skipped only when its saved result has the same model, precision, prompt hash, protocol hash, input-lock hash, preprocessing hash, and generation-config hash.

If the new GPU selects a different precision, the notebook stops with `FORMAL_PRECISION_MISMATCH_RESTART_REQUIRED`. It will not mix precision modes. To start a new uniform run, point `RESULTS_ROOT` at a new empty folder (or manually preserve and move the old result folder first).

## Gates and outputs

Before dependencies or model loading, the notebook recalculates the manifest, prompt, protocol, and all 360 candidate-image hashes, compares them with `SMOKE40_LOCKFILE.json`, and reruns the label-leakage gate. Any mismatch stops the run with `SMOKE40_LOCKFILE_VERIFICATION_FAILED`.

Private labels are first read only after `FORMAL_SMOKE40_INFERENCE_COMPLETE`, by the existing `src/evaluate_smoke40.py` evaluator.

The final output under `RESULTS_ROOT` includes:

```text
formal/per_candidate/
formal/smoke40_raw.jsonl
formal/inference_summary.json
deterministic_replay/
runtime_config.json
environment.json
runtime_metadata.json
smoke40_evaluation.json
SMOKE40_REPORT.md
SMOKE40_RESULT_LOCK.json
```

The notebook stops after printing:

```text
SMOKE40_EXPLORATION_COMPLETE
```

It does not continue to any larger experiment, dataset, fusion, prompt revision, or paper modification.

## Independent GPU server

`colab_run_smoke40_qwen3vl.py` contains the same functions called by every notebook stage. It can be imported from another Python entry point on a GPU server. Its command-line mode is intentionally limited to a non-inference validation run:

```bash
python colab/colab_run_smoke40_qwen3vl.py \
  --deployment-root /path/to/deployment \
  --results-root /tmp/smoke40_mock_check \
  --mock-dry-run
```

Mock output is marked `NOT_FOR_EVALUATION` and is never accepted by the formal completion gate.
