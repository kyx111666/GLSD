# Colab Recovery Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-02  
> **Script reviewed**: `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py`

## Pass Items

1. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:379` calls `model.eval()` before inference; static search found no `model.train()` call.
2. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:391` and `:416` wrap both model forward and hidden-head reconstruction in `torch.no_grad()`; static search found no `backward()` or optimizer execution.
3. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:386` hooks the output of `spot_pathway`, and `:400-403` validates `[B,30,384]` before applying the same model's `fc_spot + sigmoid`.
4. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:280-310` reproduces the original subject-level stitching condition in which batch-local `i` resets every 32 windows, including video transitions inside a batch.
5. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:348-373` maps the input feature list to compact-record order, runs all subject-006 windows, and selects only video `006_1` for reporting; it does not start a full-dataset experiment.
6. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:166-180` selectively extracts only a matching member from the historical zip rather than unpacking the full 6.45 GB project backup.
7. ✅ `ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py:424-446` saves `[T,384]` hidden and a portable JSON containing subject/video, checkpoint hash/path, cache path, `T_hidden`, `T_score`, max/mean absolute error, RMSE, Pearson correlation and `np.allclose`.
8. ✅ `python3 -m py_compile` completed successfully on 2026-09-02；另用 `video_num=[31,3]` 构造跨 batch/跨视频样例，确认 stitched score 与 stitched hidden 第 0 维逐点一致，输出长度为 `(480,)` 和 `(60,)`。

## Failed / Repaired Items

No syntax or static protocol defect was found. No modeling-route change was made.

## Remaining Risks

- The actual GPU forward cannot be validated on the Mac; binary imports and checkpoint compatibility remain a Colab runtime check.
- The Drive connector cannot inspect members inside `ME-TST_0.1984_Success_Backup.zip`; the script performs this check in Colab and reports `BLOCKED-ASSET` if the input cache is not a member.
- The verified Python 3.10/CUDA 12.8 archive is from the later Blackwell CASME3 recovery record, whereas an older SAMMLV log mentions a Python 3.8 conda environment. The script does not alter checkpoint weights, but this environment provenance difference must be recorded with the eventual alignment result.
- `sammlv_strategy1_outputs.pkl` is currently local and was not found by Drive filename search; it must be made Colab-visible before the numeric check.

## Run Instructions

```bash
python /content/drive/MyDrive/<synced-path>/ME_TST_HIDDEN_FEATURE_COLAB_RECOVERY.py \
  --compact-cache /content/drive/MyDrive/<path>/sammlv_strategy1_outputs.pkl
```

## Expected Outputs

- `/content/me_tst_hidden_recovery/output/SAMMLV_subject_006_006_1_spot_pathway_Tx384.npy`
- `/content/me_tst_hidden_recovery/output/one_video_hidden_score_alignment.json`
- `/content/me_tst_hidden_recovery/recovery_status.json`

## Recommended Next Step

Run exactly this one-video Colab smoke test. Only after `np_allclose == true` should a full SAMMLV hidden dump be considered.
