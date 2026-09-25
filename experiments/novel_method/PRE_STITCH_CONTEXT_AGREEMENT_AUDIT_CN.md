# Pre-Stitch Context Agreement Audit

**日期：** 2026-09-03  
**数据范围：** ME-TST+ × SAMMLV source-side  
**当前状态：** `BLOCKED-ALIGNMENT`  
**科学 GO/NO-GO：** 未执行，不得判定

## 1. Gate 结论

本实验要求在任何 Agreement 分析之前满足：

```text
真实 raw window predictions
→ 原 batch-sensitive stitching
→ reconstructed_score[T]
≈ existing compact score[T]
```

现有 one-video frozen-forward 结果没有通过这一硬门槛：

| Field | Value |
|---|---:|
| subject | 006 |
| video | 006_1 |
| T reconstructed | 1245 |
| T existing compact | 1245 |
| max_abs_error | 0.0018059015274047852 |
| mean_abs_error | 0.00004485700401821385 |
| RMSE | 0.00016291876412085786 |
| Pearson correlation | 0.9999997005557986 |
| `np.allclose` | **False** |

来源是用户在 2026-09-03 提供的 Colab one-video recovery 输出；对应 checkpoint SHA-256 为 `9509f4e5689b45f5cb753bd2796b909c2bfad4a3ab83f0be4b44b26cd3fa5c3c`，compact cache SHA-256 为 `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`。

虽然误差很小、相关系数很高，但请求明确把 `np.allclose` 作为 alignment 输出，并规定不能高精度对齐就 `STOP / BLOCKED-ALIGNMENT`。因此不能把高相关近似当作 PASS。

## 2. 为什么后续 full hidden dump 的 COMPLETE 不能解除 blocker

通过 Google Drive 读取的 `sammlv_full_hidden_dump_manifest.json` 显示：

- status=`COMPLETE`；
- 29 subjects、79 videos、`total_T=44295`；
- batch size=32、window length=30；
- `model_eval=true`、`torch_no_grad=true`；
- 006_1 的 window-head error 和 stitched-head error 均为 0；
- **`historical_compact_numeric_outputs_used=false`**。

这证明的是：fresh hidden 经同一个 fresh `fc_spot` 可以重建 fresh stitched score。它没有比较历史 compact score，因此不证明本任务要求的：

```text
fresh raw pre-stitch → historical existing_score
```

同时，每视频 NPZ 只有 stitched hidden、score 和 valid_mask，没有 raw window scores。故不能从该 dump 继续 Agreement audit。

## 3. 当前已完成与未完成

| 模块 | 状态 | 说明 |
|---|---|---|
| 原 stitching 路径 | PASS | window=30、stride=15、subject batch=32、batch-local `i` 重置已还原 |
| `(window_id, local_t)→global_t` | PASS | `global_t=window_id*15+local_t` |
| one-video frozen forward | 已有历史实际运行 | 006/006_1，T=1245 |
| existing compact alignment | **FAIL** | `np.allclose=False` |
| raw pre-stitch full SAMMLV cache | NOT CREATED | STOP rule 阻止全量导出 |
| multiplicity | NOT RUN | alignment 后置 |
| native candidate pool | NOT RUN | alignment 后置 |
| Height / Agreement / combined AP | NOT RUN | alignment 后置 |
| height-conditioned test | NOT RUN | alignment 后置 |
| native-missed subset | NOT RUN | alignment 后置 |
| matched-FP | NOT RUN | alignment 后置 |
| subject bootstrap | NOT RUN | alignment 后置 |
| context-count/boundary confound | NOT RUN | alignment 后置 |
| CASME3 / BoostingVRME | NOT RUN | source-side 尚未通过 |

## 4. 可执行 alignment gate

已生成：

`my_method/pre_stitch_context_feasibility/run_context_agreement_audit.py`

它只执行当前允许的 Phase-0 gate：

1. 从 compact cache 取得 subject/video 原顺序；
2. 取 subject 006 的完整视频窗口，不能只取单视频；
3. 加载 `subject_006.pkl`；
4. `model.eval()` 和 `torch.no_grad()`；
5. 保存每个真实 `window_id/local_t/global_t/raw_window_score`；
6. 复制 batch size 32 的原覆盖式 stitching；
7. 与 compact 006_1 比较；
8. `np.allclose=False` 时写入 `BLOCKED_ALIGNMENT` 并以 return code 2 停止。

Colab 中的命令模板：

```bash
/content/me_tst_hidden_recovery/metst310_cu128/bin/python \
  /content/drive/MyDrive/run_context_agreement_audit.py \
  --project-root /content/metst_01984_exact \
  --input-cache /content/metst_01984_exact/cache/ME-TST+/SAMMLV_dataset.pkl \
  --weights-dir '/content/drive/MyDrive/ME-TST_复现结果备份/weights/SAMMLV_4emo' \
  --compact-cache /content/drive/MyDrive/sammlv_strategy1_outputs.pkl \
  --output-dir /content/drive/MyDrive/PRE_STITCH_CONTEXT_AUDIT \
  --subject 006 \
  --video 006_1
```

脚本会保存 one-video raw mapping NPZ，即使 alignment 失败也保留用于诊断；但不会生成 full context cache，也不会运行科学指标。

## 5. 为什么没有继续“修到 allclose”

可能原因包括不同环境下的低层数值差异、compact 与 fresh dump 的生成环境/代码快照不完全相同，或历史 compact 来自另一条数值分支。当前证据不能唯一定位。

为了得到 PASS 而放宽 `allclose`、对 fresh score 做校准、改 batch、改 stitching、从 stitched hidden 反造 contexts，都会改变预先规定的 gate。本任务禁止这样做。因此这里报告最小 blocker，而不是搜索能通过的配置。

## 6. 当前唯一合法结论

> **BLOCKED-ALIGNMENT：真实 pre-stitch observations 尚未证明能重建当前 existing compact score；按照预设 STOP rule，context multiplicity、Agreement、Height-conditioned、missed-GT、matched-FP、bootstrap 和 confound 实验均未运行。当前既不能判 GO，也不能判科学 NO-GO。**

只有当同一脚本在已验证 Colab 环境中对 006/006_1 得到 `np.allclose=True`，才允许另行实现全 SAMMLV pre-stitch cache 和后续科学审计。在此之前不处理 CASME3、BoostingVRME、decoder 或 Skill。
