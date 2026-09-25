# Frozen Temporal Feature Tap Audit（Phase 0）

> 2026-09-02 更正：本文的“缺失/blocked”只描述当时 Mac 当前工作区，不能外推为历史 Colab/Google Drive 资产不存在。Drive 恢复审计与新的 one-video 脚本见 `COLAB_RECOVERY_ASSET_AUDIT_CN.md`。

> 状态：**Phase-0 STOP / blocked**  
> 科学假设判定：**尚未评估**。当前证据不能诚实地判 GO，也不能声称 relation 已被证明 NO-GO。

## 1. 审计结论

官方 ME-TST+ 结构中存在一个合适且明确的 primary feature tap，但当前项目无法执行可信 frozen forward，也没有任何现成 hidden feature cache。BoostingVRME 的模型来源和 checkpoint 在当前交付中也不完整。因此按照“不得伪造 feature、不得从 scalar score 反构 feature”的限制，本实验在 Phase 0 停止。

## 2. Feature tap 表

| Backbone | Feature tap | 单窗口 tensor shape | 与 score 对齐 | 已缓存 | 是否需 frozen forward |
|---|---|---|---|---|---|
| ME-TST+ | `METST_SF.spot_pathway` 输出、`fc_spot` 之前 | `[B,T,384]` | 单窗口严格一一对应；全视频需复制原 batch-sensitive stitching | 否 | 是 |
| ME-TST+ secondary（不运行） | `Stem` 输出 | `[B,128,T]` | 与窗口时间位置对应 | 否 | 是 |
| BoostingVRME | 无法在当前交付中认证 | 未知 | 未验证 | 否 | 是，但模型与 checkpoint 缺失 |

Primary tap 选择理由：它是最靠近 spotting head、同时仍保留 384 维 temporal representation 的位置，满足“与最终 inference 关系最直接、避免 layer zoo”的要求。secondary tap 只记录，不测试。

官方 [`network_sf.py`](https://github.com/zizheng-guo/ME-TST/blob/main/network_sf.py) 的核心路径为：

```text
x -> Stem -> spot_pathway -> x_spot[B,T,384]
                         -> fc_spot -> sigmoid -> score[B,T]
```

审计对应官方 commit：[`ae85bcdf5163d1fb3d8acec6208d25ab5621afd4`](https://github.com/zizheng-guo/ME-TST/commit/ae85bcdf5163d1fb3d8acec6208d25ab5621afd4)。

## 3. Temporal alignment

在单个 sliding window 内，`spot_pathway` 输出的每个 temporal index 直接送入 `fc_spot`，因此 hidden `[B,T,384]` 与 score `[B,T]` 一一对应。

但现有 compact score 不是简单地逐窗口平均，而是按照原项目的 overlap stitching 生成；学长的 paper-aligned 脚本还复制了原代码在每个 DataLoader batch 重置局部 `i` 的行为。若导出 hidden feature，必须对 384 维 hidden tensor使用完全相同的 batch size、视频顺序和 stitching 分支，否则即使长度相同，也不能证明 `feature[t]` 与当前 `score[t]` 是同一次 forward 的同一 temporal index。

所以可靠验收必须同时满足：

```text
T_feature == T_score
AND
fc_spot(stitched hidden) 经相同 stitching 得到的 score 与现有 cache 数值一致
```

第二项目前无法执行。

## 4. 当前 cache 实际字段

### ME-TST+

两份 cache 只包含：

- `score[T]`；
- recognition `logits[T,5]`；
- hard `emotion[T]`；
- mirror score/logits；
- subject、video、GT metadata。

其中 `[T,5]` 是 recognition head 输出，不是 spotting head 之前的 hidden representation，不能用于本次 feature relation audit。

### BoostingVRME

两份 cache 只包含：

- `score[T]`；
- hard `emotion_pred[T]`；
- subject、video、GT metadata。

不存在 logits、hidden state、embedding 或 checkpoint identifier。

| Cache | Subjects | Videos | GT | Score time points | Hidden `[T,D]` |
|---|---:|---:|---:|---:|---|
| ME-TST+ × SAMMLV | 29 | 79 | 159 | 44,295 | 无 |
| ME-TST+ × CASME3 | 94 | 462 | 858 | 1,316,600 | 无 |
| BoostingVRME × SAMMLV | 29 | 79 | 159 | 44,295 | 无 |
| BoostingVRME × CASME3 | 94 | 462 | 858 | 1,316,600 | 无 |

完整字段、dtype、shape、路径和 SHA-256 已保存到 `outputs/feature_relation_results.json`。

## 5. Frozen forward readiness

当前项目缺少：

1. 官方 `network_sf.py` 本地源码；
2. `cache/ME-TST+/SAMMLV_dataset.pkl`；
3. 五个 `cache/ME-TST+/CASME_3_dataset_*.pkl`；
4. SAMMLV 与 CASME3 全部 subject-specific checkpoints；
5. checkpoint SHA-256 manifest；
6. 可运行的 `mamba_ssm` 环境。

官方 GitHub 的 SAMMLV 权重共有 29 个，每个约 67 MB，合计约 1.94 GB；这只能说明公开资产理论上存在，不等于当前 compact cache 的完整生成链已经在本机恢复。CASME3 输入和 checkpoints 由官方 README 指向外部 Drive，当前工作区没有这些文件。

当前 `.venv` 状态：

| Component | 状态 |
|---|---|
| PyTorch | 已安装 |
| CUDA | 不可用 |
| MPS | 不可用 |
| `einops` | 缺失 |
| `timm` | 缺失 |
| `mamba_ssm` | 缺失 |

因此不能在当前 Mac 环境中通过 `model.eval()` + `torch.no_grad()` 重建可信 hidden feature。

## 6. BoostingVRME 特别问题

本地 `build_b2_curve_cache.py` 引用了交付根目录的 `network.METST_SF`、输入 cache 和 `weights/.../subject_*.pkl`，但这些依赖没有随当前 BoostingVRME 目录保存。现有 curve cache 也没有 checkpoint identifier/hash。

因此 BoostingVRME feature tap、hidden dimension、feature-score alignment 均无法认证。按照要求，不能从 scalar curve 反构 feature，也不能把 ME-TST+ 的结构假设套到 BoostingVRME 上。

## 7. Phase-0 gate

| Gate | 结果 |
|---|---|
| 找到 ME-TST+ 最后一个多维 spotting representation | PASS |
| 单窗口 hidden-score 结构对应关系明确 | PASS |
| 当前 ME-TST+ cache 已含 hidden `[T,D]` | FAIL |
| 当前项目可重跑 ME-TST+ frozen forward | FAIL |
| 可验证全视频 stitched hidden-score 一致性 | FAIL |
| BoostingVRME feature tap 可认证 | FAIL |
| 四组 relation audit 可执行 | FAIL |

Phase 0 不通过，因此不进入 self-similarity、LOSO probe、matched-FP 或 locked transfer。

## 8. 恢复实验所需最小条件

1. Linux/CUDA 环境和可工作的 `mamba_ssm`；
2. 固定到上述官方 commit 或与 compact cache 实际生成时完全一致的源码；
3. 两数据集官方输入 feature cache；
4. 全部 subject checkpoints 及 SHA-256；
5. 先导出一个视频的 hidden feature；
6. 使用原 batch size 和 stitching 重建 score，并与 compact cache 逐元素比较；
7. 只有该 smoke alignment 通过，才允许全量 dump。

## 9. Phase-0 判定

**当前项目执行判定：STOP。Frozen temporal feature relation 的科学假设尚未被测试；不得把资产与运行环境缺失写成“relation 没有增量”的 NO-GO 证据。**
