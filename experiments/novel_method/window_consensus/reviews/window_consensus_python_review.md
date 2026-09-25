# Window Consensus Instrumentation Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-06  
> **Scripts reviewed**: `export_raw_prestitch_sammlv.py`, `reconstruct_original_stitch.py`

## Pass Items

1. ✅ `export_raw_prestitch_sammlv.py:25-28` freezes `batch_size=32`, `window_length=30`, `stride=15`, and five recognition classes; none is exposed as a tunable CLI option.
2. ✅ `export_raw_prestitch_sammlv.py:152-161` flattens every video belonging to the selected subject in the frozen manifest order before batching, preserving cross-video batch position.
3. ✅ `export_raw_prestitch_sammlv.py:173-186` uses `model.eval()` and `torch.no_grad()`, and copies both `yhat` and `yhat1` immediately after forward and before stitching, slicing, overwrite, or argmax.
4. ✅ `export_raw_prestitch_sammlv.py:189-246` records `batch_index`, batch-local index, subject-flat index, video window id, pre-write framecount, global mapping, raw score, and raw logits before applying the original `i==0`/`i!=0` write branches.
5. ✅ `export_raw_prestitch_sammlv.py:270-295` writes portable NPZ arrays with the required `[N,30]` score and `[N,30,5]` logits layout, plus fresh stitched score/logits from the same forward.
6. ✅ `export_raw_prestitch_sammlv.py:304-337` validates the frozen target, input-cache cardinality, checkpoint existence, CUDA availability, and strict checkpoint loading before inference.
7. ✅ `reconstruct_original_stitch.py:35-88` rejects missing keys, altered geometry, inconsistent global coordinates, and inconsistent subject-level batch indices before reconstruction.
8. ✅ `reconstruct_original_stitch.py:91-116` reconstructs solely from raw windows plus saved `framecount_before_write` and `index_within_batch`; it does not use any alternative aggregation rule.
9. ✅ `reconstruct_original_stitch.py:131-179` reports shape, maximum/mean absolute error, exact `np.allclose`, `array_equal`, and Pearson for both spotting scores and raw logits, with the required failure status.
10. ✅ `reconstruct_original_stitch.py:182-214` includes a deterministic synthetic test beginning mid-batch, exercising batch boundaries without importing torch or running a backbone.
11. ✅ Static execution passed: both scripts compiled with `py_compile`; `--self-test` returned exact zero error and `RAW-WINDOW-INSTRUMENTATION-READY` for score and logits.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `reconstruct_original_stitch.py:14-20` | First draft did not require `subject_flat_window_index` although validation used it. | Added it to the required schema and added explicit `batch_index` consistency validation. | fixed |
| 2 | `export_raw_prestitch_sammlv.py:180` | First draft constructed the tensor directly on CUDA instead of mirroring the verified CPU-construction-then-transfer expression. | Changed to `torch.as_tensor(batch)[:, None, :].to(device)`. | fixed |

## Constraint Direction Review

There are no optimization inequality constraints in these instrumentation scripts. Equality checks enforce frozen geometry and metadata consistency only; no direction requires human confirmation.

## Remaining Risks

- The real `006/006_1` smoke has not been executed because this local workspace does not contain the restored CUDA/Mamba runtime, original `network_sf.py`, frozen input cache, and subject checkpoint together.
- The fixed 79-video manifest must match the restored GPU input cache exactly. The exporter deliberately stops if its top-level video count differs.
- PyTorch/Mamba numerical equivalence to any historical compact cache is outside this stage. This stage's mandatory check compares raw-dump reconstruction with the fresh original stitching generated during the same forward.

## Run Instructions

Follow `../README_RUN_CN.md`: run the exporter once for `006/006_1` in the restored GPU environment, then run `reconstruct_original_stitch.py` on the generated NPZ. Stop immediately if it returns `RAW-WINDOW-DUMP-EQUIVALENCE-FAIL`.

## Expected Outputs

- `006_006_1_raw_prestitch.npz`
- `006_006_1_raw_prestitch_manifest.json`
- `006_006_1_reconstruction.json`

## Recommended Next Step

Execute only the documented one-video GPU equivalence smoke. Do not proceed to Window Consensus or another method experiment in this stage.
