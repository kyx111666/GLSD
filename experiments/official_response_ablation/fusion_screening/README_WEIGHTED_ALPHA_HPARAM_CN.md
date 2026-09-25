# S_alpha 加权融合超参数消融

本实验补充论文 §3.5 的权重敏感性分析，固定融合规则

\[
S_\alpha(c)=\alpha G(c)+(1-\alpha)L(c).
\]

主实验和模块消融继续使用 `alpha=0.5`。本包只扫描预先声明的内部权重
`{0.1, 0.3, 0.5, 0.7, 0.9}`；`alpha=0` 和 `alpha=1` 分别退化为 L-only/G-only，已由模块消融覆盖。

每个 alpha 单独进行 nested LOSO：固定 alpha 后，在相同的
`a0={1,1.5,2}`、`rho={1,2,3}`、`tau` 网格中，用外层留出被试之外的 raw TP/FP/FN 选择
`(a0,rho,tau)`，再在留出被试上运行完整 official recognition/result-synergy 评价。所有 alpha
使用相同的 90 个结构配置、相同的一对一匹配协议和相同的 tie-break。

主要输出：

- `alpha_summary.csv`：每个 alpha 的 raw/full pooled TP/FP/FN 与 F1；
- `alpha_per_subject_counts.csv`：逐被试 raw/full 计数；
- `alpha_selected_configs.csv`：每折选中的 `(a0,rho,tau)`；
- `alpha_paired_comparisons.csv`：以 `alpha=0.5` 为目标的 subject-level paired bootstrap；
- `protocol.json`、`run_manifest.json`、`completion.json`：协议、来源哈希和完成标志。

`PASS` 只表示运行完成，不表示某个 alpha 优于其他值。读取结果前先检查配置数、外层被试数、
`matching_protocol`、raw/full 阶段和 `completion.json`。
