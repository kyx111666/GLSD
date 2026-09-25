# ME-TST+ Pre-Stitch Path Audit

**日期：** 2026-09-03  
**范围：** ME-TST+ × SAMMLV，`train=False` frozen inference  
**路径还原状态：** `PASS`  
**数值对齐状态：** `BLOCKED-ALIGNMENT`

## 1. 原生路径结论

原生 pre-stitch 路径可以从项目中的 `compare_paper_aligned_strategies.py`、`training_utils.py`、`tta.py` 以及已经通过的 full-hidden-dump manifest 可靠还原。它不是窗口平均：原实现按窗口顺序把选定片段**覆盖写入**最终数组，并且使用每个 DataLoader batch 内重置的局部索引 `i` 决定写整窗还是后半窗。

| 项目 | 还原结果 |
|---|---|
| window input temporal length | 30 |
| window stride | 15（`k/2`） |
| window-level spotting score shape | `[B, 30]`；单窗 `[30]` |
| companion logits shape | `[B, 30, C]`，本实验禁止使用 |
| subject batch size | 32 |
| global mapping of every raw context | `global_t = video_window_id × 15 + local_t`，`local_t=0..29` |
| theoretical overlap count | 视频两端通常 1；内部通常 2；精确分布须由真实 window dump 核验 |
| final length | `(number_of_video_windows + 1) × 15` |
| compact branch | 原始、非 mirror 的 subject-level batch-sensitive stitching；mirror 另存为 `mirror_score` |
| batch-size dependency | **有** |

## 2. 原 stitching 的精确规则

一个 subject 的所有视频窗口先按原视频顺序 flatten，再按 batch size 32 forward。设：

- `framecount`：当前视频内的 window id；
- `i`：当前 DataLoader batch 内索引，每个 batch 从 0 重新开始；
- `k=30`，`step=15`；
- `yhat[i]`：当前 window 的 `[30]` spotting prediction。

原规则是：

```python
if i == 0:
    result[framecount*15 : (framecount+2)*15] = yhat[i][0:30]
else:
    result[(framecount+1)*15 : (framecount+2)*15] = yhat[i][15:30]
```

因此：

1. 一般窗口只把后半窗写入最终 score；
2. batch 的第一个窗口会写完整 30 点；
3. 同一视频跨 batch 时，新的 `i=0` 整窗写入会覆盖前一窗口已经写入的 15 点；
4. 如果新视频恰在 batch 中途开始，`i` 不会随视频重置，首窗只写后半部分，视频开头 15 点可能保持初始化零；
5. 所以同一组 window predictions 在不同 batch size 下可能得到不同 stitched score；
6. 不能逐视频单独 forward 后声称复现了当前 compact cache，必须按 subject 原顺序 flatten。

## 3. Pre-stitch observations 的真实定义

无论最终 stitching 选择哪一半，每个真实 window output 的全部 30 个 scalar predictions 都有自然坐标：

```text
(video_window_id, local_t)
→ global_t = video_window_id*15 + local_t
```

这会使内部位置通常同时拥有：

```text
window w     的 local_t=15..29
window w+1   的 local_t=0..14
```

两份真实预测。它们正是本假设要研究的 contexts。现有 compact `score[T]` 只保留覆盖写入后的一个值，无法逆推出被丢弃的另一值。

## 4. DataLoader 与 video transition

在原实现中：

- batch 是对 subject 内所有视频窗口 flatten 后切片；
- batch-local `i` 在每 32 windows 后重置；
- `framecount` 在视频切换时重置；
- batch-local `i` 在视频切换时**不重置**。

这解释了此前 fresh full hidden dump 中总计 720 个 `unwritten_zero_positions`，以及部分视频开头 15 点没有相应 hidden/head prediction 的现象。这不是 pre-stitch context 缺失：首窗的原始前半预测真实存在，只是原 stitching branch 没写入最终数组。

## 5. 当前资产状态

| 资产 | 状态 | 证据 |
|---|---|---|
| 原 stitching 源码 | PRESENT | `senior_original/me_tst_video/me-tst-video/compare_paper_aligned_strategies.py` 的 `forward_subject_stitched_outputs` |
| 原 spotting decoder | PRESENT | `senior_original/.../root_dependencies/training_utils.py` 的 `spotting` |
| SAMMLV compact cache | PRESENT LOCAL | SHA-256 `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569` |
| frozen input cache | DRIVE/COLAB | `/content/metst_01984_exact/cache/ME-TST+/SAMMLV_dataset.pkl`；manifest hash `051880f8793484621baa968e664e2b5c226b6a0ce6ffc08e65b3330a80abe1a5` |
| 29 subject checkpoints | DRIVE/COLAB | `MyDrive/ME-TST_复现结果备份/weights/SAMMLV_4emo/subject_{id}.pkl` |
| subject 006 checkpoint | DRIVE/COLAB | SHA-256 `9509f4e5689b45f5cb753bd2796b909c2bfad4a3ab83f0be4b44b26cd3fa5c3c` |
| verified environment | DRIVE/COLAB | `metst310_cu128`，此前 full dump 使用 |
| fresh full hidden dump | DRIVE PRESENT | 29 subjects、79 videos、`total_T=44295`、status COMPLETE |
| raw pre-stitch window scores | **NOT FOUND IN SEARCHED ASSETS** | Drive `SAMMLV_FULL` 中每视频 NPZ 只有 stitched hidden/score/valid_mask |

“NOT FOUND IN SEARCHED ASSETS”只表示当前可检索 Drive 文件和本地工作区没有该 dump，不声称 Google Drive 任意位置绝对不存在。

## 6. 为什么现有 hidden dump 不能直接使用

Drive 中的 `SAMMLV_FULL/subject_006/006_1_frozen_hidden_score.npz` 保存：

```text
hidden[T,384]
score[T]
valid_mask[T]
```

它是在 stitching 后保存的。被原规则覆盖或忽略的 window half 已经丢失，无法恢复成：

```text
window_id / local_t / global_t / raw_window_score
```

从 stitched hidden 重新滑窗或人为添加 noise 都不是真实 context，故本审计没有这样做。

## 7. Phase-0 结论

`(window_id, local_t) → global_t` 已可靠还原，Phase 0 路径审计通过。但已有资产没有保存 raw pre-stitch observations，必须在同一 Colab frozen 环境重新 forward，并首先通过 existing compact alignment gate。

该 alignment 已有一次实际结果为 `np.allclose=False`，详见 `PRE_STITCH_CONTEXT_AGREEMENT_AUDIT_CN.md`。因此当前状态是 `BLOCKED-ALIGNMENT`，不允许继续 Agreement 科学实验。
