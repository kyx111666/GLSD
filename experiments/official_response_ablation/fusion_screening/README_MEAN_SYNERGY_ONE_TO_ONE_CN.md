# Mean synergy 一对一重算协议

本轮是对上一轮 GLSD-90 一对一结果的独立开发重算。旧结果保留在原目录，本轮使用新的输出根目录 `GLSD_MEAN_SYNERGY_ONE_TO_ONE`，不能与旧结果混合。

## 冻结选参规则

只有 `EqualMean` 使用新的选参规则；G、L、WeightedMean、DominantEvidence 仍按原来的内层 pooled raw F1、precision、较少 FP、固定配置顺序选择。Mean 的规则在每个外层被试上只使用其他被试的 raw 计数：

1. 在 90 个 Mean 配置中计算内层 pooled F1，记最大值为 `F1_best`。
2. 保留满足 `F1 >= F1_best - 0.002` 的近最优配置。
3. 将 Mean 配置与同一 `(a0, rho, tau)` 的 G、L 配置对应，计算两个边际：`F1_Mean-F1_G` 和 `F1_Mean-F1_L`。
4. 最大化两个边际中的较小者 `min(margin_G, margin_L)`。
5. 如仍并列，按 Mean F1、precision、较少 FP、固定配置索引排序。

`0.002`、边际定义、同结构映射和 tie-break 在 SAMMLV 与 CAS(ME)3 中完全相同。外层被试只在最终 full 评价中使用。

## 模块消融

每个外层折锁定上述 Mean 选中的 `(a0,rho,tau)`，再运行：

- `Mean_w/o_L`：只保留 G；
- `Mean_w/o_G`：只保留 L。

它们不重新调参。输出中的 `mean_selection_audit.csv` 记录近最优候选数、协同边际和选中配置，便于审计规则是否按冻结协议执行。

## 证据边界

这是一轮在看完上一轮结果后提出的开发验证，因此不能把它与上一轮结果混表，也不能称为预注册实验。只有本轮两个数据集都完整结束，并且同时检查 `completion.json`、输入人口、匹配协议、选择审计和锁参消融后，才能决定是否用于论文。

## 超参数消融

这条新规则中最适合做超参数消融的是近最优容差 `epsilon`。不要改变 GLSD-90 的
`a0/rho/tau` 搜索空间；它们是统一的搜索维度。可预先冻结并比较：

```text
epsilon ∈ {0.000, 0.001, 0.002, 0.003, 0.005}
```

其中 `epsilon=0.002` 是主方法，其他值是敏感性消融。所有值必须使用相同的一对一
协议、外层折、内层被试集合、G/L/Mean 搜索网格和 tie-break。超参数表只比较
Full Mean 的最终 full F1、TP/FP/FN 和 paired bootstrap；模块锁参消融单独对主方法
`epsilon=0.002` 报告即可。

运行 `MEAN_SYNERGY_HPARAM_ABLATION_COLAB_CELL.py`，先保持 `PROBE_ONLY=True`，通过后
改为 `False`。每个 epsilon 会单独保存目录和 ZIP。运行入口支持：

```text
--mean-epsilon <value>
```

主规则不传该参数时默认使用 `0.002`。本轮的 epsilon 扫描属于开发验证，不能看完结果
后再挑一个值冒充预注册设置；若某个替代值更好，应按预先声明的共同规则重新确定主
方法，或如实把它作为敏感性结果报告。
