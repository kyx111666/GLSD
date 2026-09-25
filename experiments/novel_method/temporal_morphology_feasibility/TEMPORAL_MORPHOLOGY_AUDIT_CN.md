# Temporal Morphology Information Audit

> 结论：**NO-GO**  
> 本实验是 frozen-output cache 上的信息诊断，不是新 decoder，也不是最终 Method 实现。

## 1. Audit motivation

前两个审计已经否定 PSED/persistence 和 Spotting–Recognition Joint Evidence。与此同时，native missed GT 中仍有大量事件区间包含局部峰，因此本审计只回答一个更窄的问题：

> 当峰高相近时，真实事件峰与背景峰的局部 spotting 曲线形状，是否包含超出 peak height 的稳定独立信息？

若完整局部 patch 的线性 probe 都不能稳定超过峰高，就不再继续设计 slope、curvature、symmetry、area、wavelet 等 morphology feature zoo。

## 2. Cache information

四个输入均为只读 frozen cache；脚本不写回原始文件。

| Backbone | Dataset | Subjects | Videos | GT | `k_p` | Candidates | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| ME-TST+ | SAMMLV | 29 | 79 | 159 | 5 | 5,025 | `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569` |
| ME-TST+ | CASME3 | 94 | 462 | 858 | 17 | 163,520 | `9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda` |
| BoostingVRME | SAMMLV | 29 | 79 | 159 | 6 | 4,528 | `abcda6477dcbf89a2083f6fdfe673b1686ca3cc100e62147b81a6806c67b05bc` |
| BoostingVRME | CASME3 | 94 | 462 | 858 | 11 | 136,750 | `9775aa254149717b27e1ea8449a5fdf4e16a26b0a407ce501d6532dcef00a14a` |

完整绝对路径保存在机器可读结果的 `metadata.input_path` 字段。

## 3. Candidate protocol

1. 对 frozen spotting score 使用宽度 `2*k_p` 的 moving average，与前两个 audit 一致。
2. 使用 `scipy.signal.find_peaks` 提取平滑曲线的全部局部峰。
3. candidate generation 不读取 GT。
4. Height、Raw Patch、Shape Patch 使用完全相同的 candidate pool。
5. 主标签：峰位置落入任一 GT onset–offset。
6. 次标签：以峰为中心的 `±k_p` 区间与任一 GT 的 IoU ≥ 0.5。
7. 记录 native TP peak、native FP peak、native missed-GT local peak 与普通背景峰。

本审计内的 native TP/FP/FN 来自与前两个 audit 相同的轻量匹配器，作用是定义 missed subset 和 FP budget，不应替代项目的 paper-aligned 原生指标。四组 audit-side 计数分别为 `53/184/106`、`81/912/777`、`54/158/105`、`80/830/778`。

## 4. Local patch 与 normalization

只测试两档预先限定的 normalized scale：

- native scale：`L=k_p`；
- wider scale：`L=2*k_p`。

边界采用 edge-value padding，随后确定性插值到 31 点。没有搜索绝对窗口长度。

三种表示为：

- `raw_patch`：保留幅值的局部平滑曲线；
- `shape_minmax`：`(X-min(X))/(max(X)-min(X)+1e-8)`；
- `relative_l2`：减去中心峰值后进行 L2 normalization。

后两种表示用于尽量移除 amplitude/height。主 shape-only 结论使用 `shape_minmax`，`relative_l2` 作为同方向核验。

## 5. Probe protocol 与 LOSO leakage control

诊断 probe 固定为：

```text
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=2000,
    random_state=20260902,
)
```

每个 backbone × dataset 独立执行 subject-level LOSO：held-out subject 的候选只用于测试；`StandardScaler` 与 LogisticRegression 都只在其余 subjects 上拟合。所有主表 probe 指标来自 LOSO out-of-fold prediction。该 probe 仅是信息上限诊断，不是 proposed inference algorithm。

## 6. Height / Raw Patch / Shape Patch

### 6.1 主尺度 `L=k_p`

下表为主标签上的 pooled ROC-AUC / PR-AUC。

| Backbone × Dataset | Height robust | Raw Patch Probe | Shape-only Probe | Relative Probe |
|---|---:|---:|---:|---:|
| ME-TST+ × SAMMLV | **0.7327 / 0.2370** | 0.7023 / 0.1507 | 0.6321 / 0.0698 | 0.6063 / 0.0491 |
| ME-TST+ × CASME3 | **0.6580 / 0.0622** | 0.5600 / 0.0512 | 0.5813 / 0.0299 | 0.5839 / 0.0285 |
| BoostingVRME × SAMMLV | **0.7444 / 0.2033** | 0.7247 / 0.1679 | 0.6624 / 0.0825 | 0.6260 / 0.0619 |
| BoostingVRME × CASME3 | **0.6580 / 0.0634** | 0.5278 / 0.0531 | 0.5734 / 0.0256 | 0.5720 / 0.0252 |

结论非常直接：四组中 Raw Patch 均未超过 Height；去幅值后的 Shape-only PR-AUC 在四组中均大幅下降。Raw Patch 也没有显示“完整曲线明显优于标量峰高”的上限证据。

### 6.2 稍宽尺度 `L=2k_p`

| Backbone × Dataset | Height robust | Raw Patch Probe | Shape-only Probe |
|---|---:|---:|---:|
| ME-TST+ × SAMMLV | **0.7327 / 0.2370** | 0.7015 / 0.1487 | 0.6309 / 0.0536 |
| ME-TST+ × CASME3 | **0.6580 / 0.0622** | 0.5729 / 0.0509 | 0.5970 / 0.0328 |
| BoostingVRME × SAMMLV | **0.7444 / 0.2033** | 0.7194 / 0.1613 | 0.6345 / 0.0641 |
| BoostingVRME × CASME3 | **0.6580 / 0.0634** | 0.5349 / 0.0523 | 0.5962 / 0.0309 |

扩大到 `2k_p` 没有改变方向，也没有出现可支持继续搜索 window 的信号。

## 7. Subject-level bootstrap

以下为 `shape_minmax - height_robust` 的 subject-balanced PR-AUC delta；CI 使用被试为重采样单元，1,000 次 bootstrap。

| Backbone × Dataset | Mean delta | 95% CI |
|---|---:|---:|
| ME-TST+ × SAMMLV | -0.2127 | [-0.3087, -0.1280] |
| ME-TST+ × CASME3 | -0.1046 | [-0.1447, -0.0664] |
| BoostingVRME × SAMMLV | -0.1412 | [-0.2064, -0.0717] |
| BoostingVRME × CASME3 | -0.1042 | [-0.1406, -0.0734] |

四个区间均完整位于 0 以下。结果不是由某一个大被试或 CASME3 的候选数量主导。

## 8. Height-conditioned analysis

在每个 subject 内执行 1:1、无放回的 peak-height nearest matching，caliper 固定为 0.15 robust-z。

| Backbone × Dataset | Matched pairs | Mean |Δheight| | Height ROC/PR | Shape ROC/PR |
|---|---:|---:|---:|---:|
| ME-TST+ × SAMMLV | 88 | 0.0183 | 0.5003 / 0.5111 | 0.4923 / 0.4907 |
| ME-TST+ × CASME3 | 2,398 | 0.0115 | 0.4999 / 0.5005 | 0.5032 / 0.5072 |
| BoostingVRME × SAMMLV | 92 | 0.0274 | 0.4999 / 0.5059 | 0.5293 / 0.5116 |
| BoostingVRME × CASME3 | 1,932 | 0.0111 | 0.4998 / 0.5005 | 0.4893 / 0.4896 |

Height matching 成功把 Height 降至随机附近。Shape-only 只在 ME-TST+×CASME3 出现约 `+0.0067` PR-AUC、在 BoostingVRME×SAMMLV 出现约 `+0.0057` PR-AUC；另外两组下降。这些微小且相互不一致的变化没有转化为全候选排序或同 FP 恢复收益。

## 9. Native missed-GT subset

只保留“native missed GT 内的峰”与普通背景峰，在 `L=k_p` 上比较：

| Backbone × Dataset | Height ROC/PR | Raw Patch ROC/PR | Shape ROC/PR |
|---|---:|---:|---:|
| ME-TST+ × SAMMLV | **0.6217 / 0.0443** | 0.5858 / 0.0379 | 0.5349 / 0.0278 |
| ME-TST+ × CASME3 | **0.6388 / 0.0421** | 0.5342 / 0.0337 | 0.5670 / 0.0263 |
| BoostingVRME × SAMMLV | **0.6492 / 0.0541** | 0.6215 / 0.0540 | 0.6005 / 0.0431 |
| BoostingVRME × CASME3 | **0.6449 / 0.0440** | 0.5092 / 0.0365 | 0.5612 / 0.0232 |

即使只考察最相关的 missed-GT 场景，shape-only 仍未超过峰高。

## 10. Matched-FP missed-GT recovery

主尺度使用各 cache 的 audit-side native FP 作为预算。Precision 定义为 `recovered/(recovered+FP)`，Recall 为 `recovered/all native missed GT`。

| Backbone | Dataset | Evidence | Recovered missed GT | FP | Precision | Recall |
|---|---|---|---:|---:|---:|---:|
| ME-TST+ | SAMMLV | Height | **7** | 184 | 0.0366 | 0.0660 |
| ME-TST+ | SAMMLV | Raw Patch | 4 | 184 | 0.0213 | 0.0377 |
| ME-TST+ | SAMMLV | Shape Patch | 4 | 184 | 0.0213 | 0.0377 |
| ME-TST+ | CASME3 | Height | **66** | 912 | 0.0675 | 0.0849 |
| ME-TST+ | CASME3 | Raw Patch | 53 | 912 | 0.0549 | 0.0682 |
| ME-TST+ | CASME3 | Shape Patch | 36 | 912 | 0.0380 | 0.0463 |
| BoostingVRME | SAMMLV | Height | 7 | 158 | 0.0424 | 0.0667 |
| BoostingVRME | SAMMLV | Raw Patch | 8 | 158 | 0.0482 | 0.0762 |
| BoostingVRME | SAMMLV | Shape Patch | **12** | 158 | 0.0706 | 0.1143 |
| BoostingVRME | CASME3 | Height | 73 | 830 | 0.0808 | 0.0938 |
| BoostingVRME | CASME3 | Raw Patch | **76** | 830 | 0.0839 | 0.0977 |
| BoostingVRME | CASME3 | Shape Patch | 21 | 830 | 0.0247 | 0.0270 |

BoostingVRME×SAMMLV 的 Shape Patch 是唯一明显局部正例，但在同 backbone 的 CASME3 上从 73 降到 21，且 `L=2k_p` 时 SAMMLV 也降为 5。因此它不满足跨数据集、跨尺度稳定性，不能支持 GO。

## 11. SAMMLV → CASME3 locked transfer

迁移时，probe 在全部 SAMMLV candidates 上拟合；threshold 仅从 SAMMLV LOSO OOF prediction 选择，然后原样应用到 CASME3。CASME3 GT 未用于模型或 threshold 选择。

| Backbone | Evidence | Target PR-AUC | Locked target F1 |
|---|---|---:|---:|
| ME-TST+ | Height | **0.0622** | **0.1187** |
| ME-TST+ | Raw Patch | 0.0514 | 0.0225 |
| ME-TST+ | Shape Patch | 0.0225 | 0.0346 |
| BoostingVRME | Height | **0.0634** | **0.1159** |
| BoostingVRME | Raw Patch | 0.0525 | 0.0934 |
| BoostingVRME | Shape Patch | 0.0235 | 0.0397 |

Shape-only 的方向在 locked transfer 后没有保留；两个 backbone 上都显著弱于 Height。

## 12. Cross-backbone interpretation

四组数据呈现相同的核心现象：

1. 峰高已经吸收了局部曲线中最主要的可分信息；
2. Raw Patch 不能超过 Height，说明即使允许 amplitude，完整线性 patch 也没有更高的稳定上限；
3. 去 amplitude 后 PR-AUC 系统性下降；
4. height-conditioned 小涨点方向不一致；
5. matched-FP 的单点改善不能迁移；
6. 两个 backbone 的 locked transfer 均失败。

自动严格判据共 14 项，仅 ME-TST+×CASME3 的 height-conditioned PR-AUC 微幅改善一项通过。

## 13. GO / NO-GO

本结果满足多项预定 NO-GO 条件：morphology < Height、matched-FP 在主 backbone 两数据集均不改善、迁移后消失、第二 backbone 不稳定且总体反向。

**NO-GO：local temporal morphology 没有提供足够稳定的独立信息，不建议继续从 frozen spotting curve 设计 morphology decoder。**

这个结论只针对当前四组 frozen scalar spotting curve 和受限的线性信息诊断。它不否定更早层的时空 feature、ROI feature 或重新训练模型中可能存在形态信息；但按照本审计目标，不应继续在这些标量曲线上扩展 handcrafted morphology heuristic。

## 14. Reproduction

从项目根目录运行：

```bash
.venv/bin/python my_method/temporal_morphology_feasibility/run_morphology_audit.py
```

输出：

- `my_method/temporal_morphology_feasibility/outputs/temporal_morphology_results.json`

环境版本、输入路径、hash、seed、patch 长度、normalization、probe 配置和所有分项指标均保存在 JSON 中。
