# Scale-Consistency Feasibility Audit

**日期：** 2026-09-03  
**实验性质：** 新候选机制的 cache-only、training-free falsification test  
**最终结论：** `NO-GO`

> **NO-GO：inference-scale consistency 没有提供足够稳定的独立信息，不建议继续实现完整 EquiScale 或基于它包装 Skill。**

## 1. 本次究竟测试了什么

本次没有复现或补写历史 EquiScale，也没有实现完整事件 decoder。唯一待检验假设是：

> 真实事件相关峰是否比噪声峰在固定的 temporal-scale perturbation 下更稳定，以及这种 cross-scale support 是否在 peak height 之外提供独立信息。

候选生成在查看 GT 前固定为：

```text
frozen original score[T]
→ 5 个预设尺度
→ 每尺度相同平滑、阈值和 find_peaks
→ 固定半径跨尺度关联
→ support / height / height×support
```

实验没有使用 mirror、recognition logits、hidden feature、model forward、训练、参数搜索、scale grid、rescue 或新 NMS。GT 只在候选全部生成以后用于打标签和评估。

## 2. 数据与可复现性

| Dataset | Cache path | SHA-256 | Subjects | Videos | GT | Native `k_p` | Fixed scales |
|---|---|---|---:|---:|---:|---:|---|
| SAMMLV | `caches/me_tst/sammlv_strategy1_outputs.pkl` | `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569` | 29 | 79 | 159 | 5 | {3, 4, 5, 6, 8} |
| CASME3 | `caches/me_tst/casme3_strategy1_outputs.pkl` | `9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda` | 94 | 462 | 858 | 17 | {11, 14, 17, 21, 26} |

比例在运行前固定为 `{2k/3, 4k/5, k, 5k/4, 3k/2}`，使用 round-half-up 取最近正整数并去重。两次完整重跑生成了逐字节相同的 JSON；运行前后两个输入 cache 的 SHA-256 均未变化。

主要产物：

- 脚本：`my_method/scale_consistency_feasibility/run_scale_consistency_audit.py`
- 汇总：`my_method/scale_consistency_feasibility/outputs/scale_consistency_results.json`
- 逐 cluster：`outputs/sammlv_clusters.csv`、`outputs/casme_3_clusters.csv`

## 3. 固定实现细节

每个尺度 `a` 使用：

1. width=`2a` 的 `np.convolve(..., mode="same")` moving average；
2. `mean(s_a) + 0.55 × (max(s_a)-mean(s_a))` 阈值；
3. `scipy.signal.find_peaks`；
4. minimum distance=`a`。

跨尺度 association 的歧义按以下确定性规则一次性固定，没有搜索：

1. 合并五尺度 peaks，并按 `(time, scale)` 排序；
2. 相邻 peak 时间差不超过 `floor(k/2)` 时，以 single-link 方式进入同一连通分量；
3. 一个尺度在一个分量中最多保留一个 peak：选择离原分量时间中位数最近者；平局时依次选择更高 peak 和更早时间；
4. cluster representative 是保留成员位置中位数的 round-half-up 整数；
5. support `m` 是 distinct scale 数，`M=m/5`；
6. 主 Height `H` 固定为保留成员中最大的、未经 q05/q99 校准的平滑 peak score；
7. 固定联合证据 `J=H×M`。

Primary positive 是 representative 落在任一 GT 的闭区间 `[onset, offset]` 内。Secondary label 是 representative 的 `±k` 区间与任一 GT 达到 IoU≥0.5。ROC-AUC 使用 `roc_auc_score`，报告的 PR-AUC 是 `average_precision_score`（AP）。

## 4. Hypothesis A：Scale complementarity

| Dataset | Native hit GT | Union hit GT | Native-missed recovered by ≥1 | ≥2 | ≥3 | ≥4 | 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV | 54 | 65 | 11 | 7 | 0 | 0 | 0 |
| CASME3 | 127 | 188 | 61 | 35 | 6 | 3 | 0 |

解释：

- non-native scales 的确带来 complementary coverage：SAMMLV 恢复 11/105 个 native-missed GT，CASME3 恢复 61/731 个；
- 但补回比例只有约 10.5% 和 8.3%；
- 强一致性很少：SAMMLV 的 native-missed GT 没有任何一个获得 ≥3 个非 native scale 支持；CASME3 只有 6 个达到 ≥3、3 个达到 ≥4；
- 因而 Hypothesis A 只能判为“有限支持”，不能单独构成 GO。

这里的 hit 定义是“某尺度生成的 peak 落入 GT”，不是正式 MEGC IoU 一对一 TP；因此这些数字不应与论文 Spotting TP 混用。

## 5. Hypothesis B：support 与事件可靠性

### 5.1 SAMMLV

| Support | #clusters | #positive | #negative | Positive rate |
|---:|---:|---:|---:|---:|
| 1/5 | 86 | 13 | 73 | 0.1512 |
| 2/5 | 50 | 5 | 45 | 0.1000 |
| 3/5 | 28 | 3 | 25 | 0.1071 |
| 4/5 | 54 | 11 | 43 | 0.2037 |
| 5/5 | 147 | 40 | 107 | 0.2721 |

Positive 的平均 support 为 3.833，negative 为 3.225。高 support 档的可靠性较高，但 1→2→3 档并不单调。

### 5.2 CASME3

| Support | #clusters | #positive | #negative | Positive rate |
|---:|---:|---:|---:|---:|
| 1/5 | 318 | 27 | 291 | 0.0849 |
| 2/5 | 148 | 21 | 127 | 0.1419 |
| 3/5 | 111 | 17 | 94 | 0.1532 |
| 4/5 | 143 | 17 | 126 | 0.1189 |
| 5/5 | 708 | 100 | 608 | 0.1412 |

Positive 的平均 support 为 3.780，negative 为 3.508。但 positive rate 在 3/5 后下降，5/5 仍只有 0.1412，没有清晰的 support 越高、event reliability 越高的关系。

## 6. Beyond Peak Height Test

| Dataset | Evidence | ROC-AUC | PR-AUC/AP |
|---|---|---:|---:|
| SAMMLV | Height | 0.6076 | 0.3266 |
| SAMMLV | Support | 0.6036 | 0.2501 |
| SAMMLV | Height×Support | 0.6214 | 0.3349 |
| CASME3 | Height | 0.5884 | 0.1655 |
| CASME3 | Support | 0.5435 | 0.1385 |
| CASME3 | Height×Support | 0.5807 | 0.1578 |

核心差值：

- SAMMLV：`AP(H×M)-AP(H)=+0.00828`，只是小幅增加；
- CASME3：`AP(H×M)-AP(H)=-0.00761`，联合证据比 Height 更差。

因此“support 可在两个数据集一致改善 Height”的核心要求失败。

## 7. Height-conditioned support

在每个 subject 内用 Hungarian assignment 做无放回最近 Height 配对；没有使用 caliper，也没有丢弃较难匹配的已分配 pair。

| Dataset | Matched pairs | Mean `|ΔH|` | Median `|ΔH|` | Height ROC-AUC | Support ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SAMMLV | 63 | 0.0566 | 0.0161 | 0.5185 | 0.5161 |
| CASME3 | 178 | 0.0458 | 0.0178 | 0.5100 | 0.5152 |

Height 已如预期接近随机，而 Support ROC-AUC 在两个数据集也只有约 0.516。也就是说，一旦控制 peak height，当前 scale support 基本接近随机，未显示有意义的独立判别力。这是 NO-GO 的主要证据之一。

## 8. Support threshold 机制曲线

### 8.1 SAMMLV

| Support rule | TP-related | FP/noise | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| ≥1 | 72 | 293 | 0.1973 | 1.0000 | 0.3295 |
| ≥2 | 59 | 220 | 0.2115 | 0.8194 | 0.3362 |
| ≥3 | 54 | 175 | 0.2358 | 0.7500 | 0.3588 |
| ≥4 | 51 | 150 | 0.2537 | 0.7083 | 0.3736 |
| =5 | 40 | 107 | 0.2721 | 0.5556 | 0.3653 |

SAMMLV precision 随阈值上升，但 F1 在 5/5 回落；这是一定程度的 reliability signal，但没有通过 height-conditioned 独立性和 bootstrap。

### 8.2 CASME3

| Support rule | TP-related | FP/noise | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| ≥1 | 182 | 1246 | 0.1275 | 1.0000 | 0.2261 |
| ≥2 | 155 | 955 | 0.1396 | 0.8516 | 0.2399 |
| ≥3 | 134 | 828 | 0.1393 | 0.7363 | 0.2343 |
| ≥4 | 117 | 734 | 0.1375 | 0.6429 | 0.2265 |
| =5 | 100 | 608 | 0.1412 | 0.5495 | 0.2247 |

CASME3 从 ≥2 开始 precision 基本停滞，F1 持续下降，没有系统性的 support-reliability 曲线。这里没有选择任何“最佳档”。

## 9. Matched-FP candidate recovery

主结果使用原 native decoder 的固定审计等价口径：每个 native peak 生成 `±k` 区间，按视频进行 IoU≥0.5 的 greedy one-to-one matching。该 cache 下重建的 native 预算为 SAMMLV 184 FP、CASME3 912 FP。它是本次统一重建值，不替代论文正式报告数字。

| Dataset | Evidence | GT-related | Native missed recovered | FP |
|---|---|---:|---:|---:|
| SAMMLV | Height | 46 | 3 | 184 |
| SAMMLV | Height×Support | 52 | 3 | 184 |
| CASME3 | Height | 87 | 20 | 912 |
| CASME3 | Height×Support | 89 | 17 | 912 |

在相同 FP 预算下，联合证据多保留了少量已可匹配 GT（SAM +6，CAS +2），但没有多恢复 native-missed GT：SAMMLV 持平为 3，CASME3 从 20 降至 17。故它不支持“consistency 在不增加 FP 时恢复更多遗漏事件”。

机器可读 JSON 还保留了基于 primary containment label 的补充预算。补充口径同样显示 native-missed recovery 下降：SAM 5→3，CAS 28→24。

## 10. Subject-level bootstrap

以 subject 为抽样单位，有放回抽取与原 subject 数相同的 subject，固定 seed=`20260903`，重复 1000 次。每次计算：

`AP(H×M)-AP(H)`。

| Dataset | Mean delta | 95% CI | 解释 |
|---|---:|---|---|
| SAMMLV | +0.00775 | [-0.01174, 0.03167] | CI 跨 0，不能排除负效应 |
| CASME3 | -0.00790 | [-0.01980, 0.00142] | 均值负向，CI 绝大部分在负区间 |

两个数据集方向不一致，且没有一个数据集得到明确的正区间。

## 11. 预设 GO 条件逐项检查

| # | 预设证据 | 结果 |
|---:|---|---|
| 1 | non-native scales 覆盖一定 native-missed GT | **有限满足**：SAM 11，CAS 61，但强多尺度支持很少 |
| 2 | true clusters support 明显高于 noise | **弱满足/不足**：均值较高，但差距小且 CAS 曲线不稳定 |
| 3 | positive rate 随 support 清楚上升 | **不满足**：两数据集均非单调，CAS 尤其不成立 |
| 4 | `PR-AUC(H×M)>PR-AUC(H)` | **不满足**：只在 SAM 小幅成立，CAS 负向 |
| 5 | matched-height 后 Support 仍有判别力 | **不满足**：ROC-AUC 约 0.516，接近随机 |
| 6 | matched-FP 恢复/保留更多关键 GT | **不满足**：总体 TP 略增，但 missed recovery 持平或下降 |
| 7 | SAMMLV 与 CASME3 方向一致 | **不满足**：核心 AP delta 相反 |
| 8 | subject bootstrap 不显示系统性负效应 | **不满足**：SAM CI 跨 0，CAS 均值负且 CI 主要为负 |

只有 complementarity 现象得到有限支持；决定“独立信息是否存在”的核心检验 4、5、6、7、8 均未通过。

## 12. 最终判定与停止边界

**NO-GO：inference-scale consistency 没有提供足够稳定的独立信息，不建议继续实现完整 EquiScale 或基于它包装 Skill。**

这个结论并不是说不同尺度产生完全相同的 peaks。它们确实补到了少量 GT；问题是 support 主要没有超出 Height 的信息：控制 Height 后接近随机，固定乘法在 CASME3 下降，同 FP 下也未恢复更多 native-missed GT。

按照实验开始前的约束，当前不继续尝试：

- 新 scale ratios；
- 新 support threshold；
- 新 association radius；
- mirror averaging；
- recognition 或 boundary fusion；
- 可学习 fusion weight；
- 完整 EquiScale decoder 或 Skill 包装。

## 13. 验证清单与限制

- [x] 两个数据集使用同一预设 scale ratios；
- [x] 每尺度除 `a` 外使用完全相同候选生成规则；
- [x] candidate generation 不读取 GT；
- [x] 只访问 `record['score']` 数值输入，未访问 mirror/logits/hidden；
- [x] 未执行训练、模型 forward 或参数搜索；
- [x] subject-level bootstrap 固定种子并完成 1000 次；
- [x] 输入 cache 运行前后 SHA-256 不变；
- [x] 两次运行 JSON 哈希一致；
- [x] 保存逐 cluster CSV，汇总数字可追溯；
- [x] 明确区分 primary containment 与 secondary IoU label。

局限性：single-link association 可能被时间上连续的 peaks 桥接；Height 使用未校准平滑分数；cluster primary label 不执行一对一 GT matching。它们都是运行前固定的简单 probe 规则。由于主结论已经在两个数据集、height matching、matched-FP 和 bootstrap 上共同失败，没有理由在本阶段通过修改这些规则挽救候选。
