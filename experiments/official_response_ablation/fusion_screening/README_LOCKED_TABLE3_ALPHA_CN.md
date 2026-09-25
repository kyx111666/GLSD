# Table 3 锁定配置 alpha 消融

这版运行使用 Table 1 官方证据中的 `outer_selected_configs.csv`。每个外层被试的
`(a0, rho, tau)` 固定不变，只替换

\[
S_\alpha=\alpha G+(1-\alpha)L,
\qquad \alpha\in\{0,0.3,0.5,0.7,1\}.
\]

因此：

- `alpha=0.5` 必须逐被试复现 Table 1 的 GLSD-90 full TP/FP/FN；
- `alpha=0` 是锁参的 `L-only`（w/o `G`）；
- `alpha=1` 是锁参的 `G-only`（w/o `L`）；
- `alpha=0.3` 和 `alpha=0.7` 是只改变融合权重的超参数敏感性结果。

## Colab 顺序

1. 继续使用原来的 ME-TST+ Colab，先挂载 Drive 并完成 `/content/ME-TST` 初始化。
2. 打开 `LOCKED_TABLE3_ALPHA_COLAB_CELL.py`，保持 `PROBE_ONLY = True`。
3. 运行 cell，上传 `metst_locked_table3_alpha.zip`。
4. 必须看到 `LOCKED_TABLE3_ALPHA_PROBE = PASS`，并确认输出目录中的
   `alpha_0p5_alignment.json` 显示 `mismatches: []`。
5. 将 `PROBE_ONLY = False`，重新运行同一个 cell。`SETTING` 先设为 `sammlv`，完成后再设为 `casme3`。

full 运行完成后，读取各数据集目录下的 `alpha_summary.csv` 和
`alpha_per_subject_counts.csv`。只有 alpha=0.5 对齐检查通过，才把这五列写入 Table 3。
