# SAMMLV Full Hidden Dump Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-03  
> **Script reviewed**: `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py`

## Pass Items

1. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:28-59` embeds exactly 79 unique `(subject, video)` entries covering 29 subjects；AST 检查已验证数量与唯一性。
2. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:212-214` loads one frozen subject checkpoint with `strict=True` and immediately calls `model.eval()`；脚本中不存在 `model.train()`。
3. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:227-245` wraps every backbone forward in `torch.no_grad()` and hooks only `spot_pathway`；静态搜索未发现 `backward()` 或 `optimizer.step()`。
4. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:109-151` implements batch-local `i` reset with `BATCH_SIZE=32` and the original full-first/half-following stitching，并保存逐帧 `valid_mask`；跨 batch/跨视频 regression test 验证第二个视频恰有 15 个原始未写入零位置。
5. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:237-243` enforces `[B,30,384]` and confirms each window's `fc_spot + sigmoid` equals the model score before accepting it.
6. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:255-271` rebuilds score from stitched hidden and saves fresh hidden/score together per video；no historical compact score/logit is read.
7. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:178-210` implements subject-level resume and validates existing `.npz` shapes before counting them in the final manifest.
8. ✅ `ME_TST_SAMMLV_FULL_HIDDEN_DUMP.py:291-315` writes a JSON manifest with input-cache hash, checkpoint hashes, dimensions, per-video paths, counts and explicit zero-training provenance.
9. ✅ `python3 -m py_compile` completed successfully on 2026-09-03.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | original resume branch | completed subjects were skipped without rebuilding their manifest records | load and validate existing output shapes, then reconstruct records | fixed |
| 2 | original top-level imports | importing the helper on Mac required unavailable torch even for a NumPy stitching test | moved torch imports into runtime functions | fixed |
| 3 | original stitched hidden reconstruction | a video beginning mid-batch has 15 unwritten score-zero positions, while `sigmoid(fc_spot(zero_hidden))≈0.49944` | track `valid_mask` during the exact stitch and restore those positions to zero after the head | fixed |

## Remaining Risks

- GPU/Mamba execution can only be tested inside the restored Colab environment.
- The embedded list is a metadata manifest recovered from the old cache. It contains no old numerical output, but its ordering must remain paired with the exact 79-entry ME-TST+ input cache.
- If Colab disconnects during one subject, rerunning without `--overwrite` reruns that entire incomplete subject. A corrupt completed `.npz` will be reported; rerun with `--overwrite` after identifying it.

## Run Instructions

Use `/content/me_tst_hidden_recovery/metst310_cu128/bin/python` and pass the exact success-backup project/cache plus the complete `SAMMLV_4emo` weights directory.

## Expected Outputs

- 79 `subject_*/<video>_frozen_hidden_score.npz` files
- `sammlv_full_hidden_dump_manifest.json`

## Recommended Next Step

Validate the final manifest reports 29 subjects, 79 videos and `max_stitched_head_error <= 1e-6` for every newly generated record before starting Feature Relation Audit.
