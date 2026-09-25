# 旧 Mean 重算冻结协议

本轮目标是判断旧的 `EqualMean` 是否能在统一评价协议下同时满足主结果和模块消融要求。正式结果只接受同一次运行生成的文件；历史 Table 2、旧 Table 3 和其他路线的最高单元格不能拼接。

## 运行入口

在已经配置好的 ME-TST+ 官方响应 Colab 环境中，使用：

```text
run_full_fusion_tuning.py
```

建议先分别运行 `sammlv` 和 `casme3`，确认两个输出目录均完整后再考虑 BoostingVRME。输出目录必须是新的 Drive 目录，例如：

```text
MyDrive/GLSD_MEAN_RETAKE/<UTC-run>/
```

本机没有官方响应输入，因此本机只能做语法和合成测试，不能把本地运行当作论文实验结果。

## 比较方法

主搜索同时包含：

- `G`
- `L`
- `EqualMean`：`(G+L)/2`
- `WeightedMean`
- `DominantEvidence`

其中 `a0={1,1.5,2}`、`rho={1,2,3}`、阈值网格和权重网格由 runner 固定。所有方法都在同一外层留一被试、内层 pooled 计数选择参数；留出被试只用于最终 full 评价。G 的 `rho` 是无效维度，代码只保留 `rho=1` 的表示，不用重复行制造搜索机会。

## 模块消融

每个外层折先保存 `EqualMean` 自己选出的 `(a0,rho,tau)`，然后在完全相同参数下运行：

- `Mean_w/o_L`：把分数替换成 G；
- `Mean_w/o_G`：把分数替换成 L。

这两行不重新调参，输出到：

```text
locked_ablation_summary.csv
locked_ablation_per_subject_counts.csv
locked_ablation_selected_configs.csv
locked_ablation_comparisons.csv
```

`summary_raw_full.csv` 中的 G/L/EqualMean 等行仍然是各自独立调参结果；它们回答“独立方法谁更强”。锁参文件回答“在最终 Mean 的同一组参数下，去掉一个模块是否下降”。论文中必须把这两个问题分开标注。

## 运行前检查

1. 先通过 Native 和旧 GLSD exact replay；失败时停止，不解释任何新分数。
2. 确认 `completion.json`、`protocol.json`、`selected_configs.csv`、`per_subject_counts.csv` 和 `locked_ablation_*` 都存在。
3. 核对 SAMMLV 为 29/79/159，CAS(ME)3 为 94/462/853；CAS 不得混入 858 GT 版本。
4. 核对输出中的 evaluator、输入哈希、代码哈希和运行时一致。
5. 先看四个方法的 pooled full F1 和逐被试 TP/FP/FN，再决定是否进入超参数敏感性和 BoostingVRME。

## 结果判定

只有在 `EqualMean` 的独立调参结果达到 Table 2 目标，并且锁参 `EqualMean` 高于 `Mean_w/o_L` 和 `Mean_w/o_G` 时，才把 Mean 作为论文最终方法候选。若独立调参 Mean 仍低于强单模块，不能用锁参消融掩盖这一点；应转向已准备好的参考残差或时间错位路线，并沿用同一评价协议。

在这一步之前不修改论文 3.5、Table 2、Table 3、摘要或结论。
