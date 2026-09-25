# Frozen Temporal Feature Relation Audit

> 2026-09-02 更正：本文没有在 Mac 上执行 forward，并不表示既往 Colab 链路不存在。Google Drive 已发现历史代码、SAMMLV subject 权重、成功备份和验证环境；恢复状态以 `COLAB_RECOVERY_ASSET_AUDIT_CN.md` 为准。

> 状态：**未进入 Phase 1；Phase 0 blocked**  
> Relation metrics：**未计算**

## 1. Feature location、shape 与 alignment

ME-TST+ 的计划 primary tap 已定位为 `METST_SF.spot_pathway` 输出、`fc_spot` 之前的 `[B,T,384]`。单窗口与 score `[B,T]` 一一对应；全视频必须复刻当前 compact score 的 batch-sensitive overlap stitching。详细依据见 `FEATURE_TAP_AUDIT_CN.md`。

BoostingVRME 在当前交付中缺失可认证的 model source/checkpoint，feature tap 和对齐均不可验证。

## 2. Frozen forward protocol

计划协议只能是：

```python
model.eval()
with torch.no_grad():
    score, logits = model(x)
```

同时 hook `spot_pathway` 输出，并与 score 使用同一 stitching。禁止 backward、optimizer、train mode 和任何权重更新。

该 forward 没有执行，因为当前工作区没有输入 feature cache、subject checkpoints 和兼容的 `mamba_ssm`/CUDA runtime。

## 3. Candidate 与 relation protocol

以下方案已由用户规范固定，但因没有 hidden cache 而没有运行：

- 与前三个 audit 相同的 native-smoothed local-peak candidate pool；
- Height baseline：per-video robust normalized peak height；
- `L=k_p`、`L=2k_p` 两档窗口；
- hidden vector L2 normalization；
- self-similarity matrix deterministic resize 到一个固定尺寸；
- 只允许 transition magnitude 与 center-context dissimilarity 两个 deterministic scalar；
- self-similarity flatten 后的 subject-LOSO LogisticRegression；
- held-out subject、scaler 与 probe 严格隔离；
- within-subject height matching；
- audit-side native FP budget；
- SAMMLV→CASME3 locked transfer。

## 4. 未计算的结果

| 项目 | 状态 | 原因 |
|---|---|---|
| ROC-AUC / PR-AUC | N/A | 无 aligned hidden feature |
| Subject bootstrap CI | N/A | 无 probe OOF prediction |
| Height-conditioned relation | N/A | 无 relation representation |
| Native missed-GT analysis | N/A | 无 relation score |
| Matched-FP recovery | N/A | 无 relation score |
| SAMMLV→CASME3 locked transfer | N/A | 无 source/target hidden cache |
| Cross-backbone consistency | N/A | BoostingVRME tap 不可认证 |

## 5. 为什么不能用现有 logits 继续

ME-TST+ 的 `[T,5]` logits 是 recognition head 输出。前一个 Joint Evidence Audit 已单独研究 recognition evidence；把它重新命名为 hidden temporal feature 并构造 self-similarity，既不符合本次 tap 定义，也会重复已经 NO-GO 的信息源。BoostingVRME 甚至没有连续 logits。

同样，不能对 scalar score 做复制、延迟嵌入或伪高维展开；那仍然只包含已被 morphology audit 否定的 scalar 信息。

## 6. GO / NO-GO 状态

本次没有产生任何 relation ranking、PR-AUC、matched-FP 或 transfer 结果，因此不能支持：

> Frozen temporal feature relations contain event-discriminative information beyond the final scalar spotting score.

也不能反向声称 hidden relation 已被实验证明没有增量。

**Phase-0 STOP：当前项目不具备执行 Frozen Temporal Feature Relation Audit 所需的可信 hidden feature 与 frozen-forward 条件。科学 GO/NO-GO 暂不判定。**

继续前必须先满足 `FEATURE_TAP_AUDIT_CN.md` 的最小恢复条件；否则应直接回到已有正向 PoC，而不是在现有 scalar cache 上继续发明 inference heuristic。
