# SAMMLV Full Raw Pre-Stitch Exporter Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-06  
> **Script reviewed**: `export_raw_prestitch_sammlv_full.py`

## Pass Items

1. ✅ `export_raw_prestitch_sammlv_full.py:22-36` imports `VIDEO_MANIFEST`, frozen geometry, checkpoint/cache helpers, NPZ writer, and reconstruction evaluator directly from the already verified single-video implementation; it does not duplicate or alter that exporter.
2. ✅ `export_raw_prestitch_sammlv_full.py:87-100` constructs every subject's windows by selecting manifest entries, preserving video order, then flattening exactly once before batching.
3. ✅ `export_raw_prestitch_sammlv_full.py:111-124` keeps `model.eval()`, `torch.no_grad()`, batch size 32, and captures `yhat/yhat1` immediately after forward and before any stitching or argmax.
4. ✅ `export_raw_prestitch_sammlv_full.py:127-168` records each video's full required window metadata while retaining the subject-flat batch index and batch-local index across video boundaries.
5. ✅ `export_raw_prestitch_sammlv_full.py:171-182` reproduces the verified batch-sensitive branch exactly: batch-local index zero writes all 30 values; all other windows write only positions 15–29.
6. ✅ `export_raw_prestitch_sammlv_full.py:194-211` separates the single subject traversal into per-video payloads compatible with the existing `save_dump` contract, without re-forwarding a video.
7. ✅ `export_raw_prestitch_sammlv_full.py:214-216,325-333` saves each video with the verified v1 NPZ writer and immediately reopens it with `allow_pickle=False` for the existing exact reconstruction evaluator.
8. ✅ `export_raw_prestitch_sammlv_full.py:229-265` requires all 79 unique frozen video identities and both score/logits `array_equal=true` before emitting `SAMMLV-RAW-WINDOW-SOURCE-READY`.
9. ✅ `export_raw_prestitch_sammlv_full.py:276-294` gates execution on exactly 29 subjects / 79 videos, exact input-cache cardinality, required files, and CUDA availability.
10. ✅ `export_raw_prestitch_sammlv_full.py:306-322` contains exactly one checkpoint deserialization and one complete subject traversal inside each subject-loop iteration; per-subject reporting records one checkpoint load and one subject pass separately from the necessary batch-forward count.
11. ✅ Static tests passed: Python compilation succeeded; imported manifest resolved to 29 subjects / 79 videos; frozen constants resolved to 32/30/15/5; synthetic summary tests accepted 79/79 and rejected a single logits failure.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `export_raw_prestitch_sammlv_full.py:238-263` | Initial summary code could compare failed rows by dictionary equality and could call `max()` on `None` if a shape failure produced no numeric error. | Added an explicit pass predicate and finite-only min/max helpers. | fixed |
| 2 | `export_raw_prestitch_sammlv_full.py:242-248` | Count-only readiness could theoretically accept duplicated video identities. | Added exact identity-set equality against the frozen 79-video manifest. | fixed |
| 3 | `export_raw_prestitch_sammlv_full.py:13` | Unused `hashlib` import. | Removed. | fixed |

## Constraint Direction Review

No optimization inequalities are present. Equality and cardinality conditions only enforce the frozen manifest, geometry, traversal, and exact reconstruction gate.

## Remaining Risks

- The full CUDA run was intentionally not executed in this review. Runtime compatibility with the restored PyTorch/Mamba environment remains to be confirmed there.
- “One forward per subject” is implemented as one complete subject traversal. The model is necessarily invoked once per frozen batch of 32, and `model_forward_batches` makes this explicit in `full_manifest.json`.
- `--overwrite` replaces existing files in the selected output directory. Use a dedicated empty `raw_prestitch_sammlv/` directory for the first full run.

## Run Instructions

From the restored ME-TST+ project root:

```bash
python my_method/window_consensus/export_raw_prestitch_sammlv_full.py \
  --project-root /content/metst_01984_exact \
  --input-cache /content/metst_01984_exact/cache/ME-TST+/SAMMLV_dataset.pkl \
  --weights-dir "/content/drive/MyDrive/ME-TST_复现结果备份/weights/SAMMLV_4emo" \
  --output-dir "/content/drive/MyDrive/ME-TST_复现结果备份/raw_prestitch_sammlv"
```

## Expected Outputs

- 79 files named `{subject}_{video}_raw_prestitch.npz`
- `full_manifest.json`
- Final status only if every video passes: `SAMMLV-RAW-WINDOW-SOURCE-READY`

## Recommended Next Step

Run only this full frozen exporter in the already validated GPU environment. Do not launch any aggregation or decoder experiment as part of the export run.
