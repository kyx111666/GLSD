# SAMMLV 固定 beta 的 full 对照回放

本轮检验：固定融合形式，是否比逐折选择 beta 更稳定？不是新参数网格，也不是重新训练。

## 操作

1. 在已配置好的 ME-TST+ Colab notebook 中确保 Drive 挂载。
2. 新建一个 cell，粘贴 `FIXED_BETA_COLAB_CELL.py` 全文。
3. 运行并上传 `metst_fixed_beta_replay.zip`。只需要这个新包，不需要上传原结果，不要重跑旧搜索 cell。
4. 完成后下载打印路径对应的 `oto_fixed_beta_*.zip` 并发回。

运行输出位于 `MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING/oto_fixed_beta_<UTC>/`。旁边有同名日志及结果 ZIP。
文件名带自动上传后缀不影响执行。回放规模接近刚完成的事件诊断，通常是分钟级；不是上一轮一小时的全网格搜索。

## 方法与选择

完整报告 beta={0.5,0.6,0.7,0.8,0.9,1.0}，不只回放事后表现较好的 beta。
每个 beta 固定后，在已有扩展结构计数中，用其他 28 被试 pooled raw F1、precision、较少 FP、原配置顺序选择 a0/rho/tau。
计算所需的原始搜索计数和配置清单随包提供，保留原 ZIP 的 SHA256；Colab 只完成对应配置的真实 decoder/recognition 回放。
本轮沿用扩展范围 a0={0.5,1,1.5,2,2.5,3,4}、rho={0.5,1,1.5,2,3,4}、tau=0.05:0.05:0.95。
L 支持尺度、gamma、事件几何、一对一评价协议保持与结构扩展实验一致。

对照包括扩展范围的 G、L、EqualMean、WeightedMean、可变 beta 的 DominantEvidence，另展示原结构范围 EqualMean 作为历史参考，明确区分范围。
174 次现有对照回放 + 174 次固定 beta 回放 = 348 次 subject-config 回放。
所有回放核对已有 raw 计数；现有对照还核对 full 和预测。beta=.5 的逐折 raw/full 结果必须等于同范围 EqualMean。
保留逐事件候选分数、匹配和识别信息，继续使用已跑通的事件诊断工具。

## 输出

- `summary_raw_full.csv`：12 行方法/对照标签各自的 raw/full 汇总，共 24 行。
- `fixed_beta_selected_configs.json`：六个固定 beta 的逐折参数、训练侧 pooled raw 计数及留出 raw 参考值。
- `per_subject_counts.csv`：逐被试 raw/full TP、FP、FN。
- `paired_comparisons.csv`：六个 beta 对五个同范围方法及一个历史均值对照的探索性 bootstrap 区间。
- `event_records.jsonl.gz`：逐事件 GT、匹配、候选证据、识别过滤和 full 贡献。
- `replay_checks.csv`、`completion.json`：执行与一致性核验。

## 解释边界

这些都是已反复用于开发的 SAMMLV 数据上的诊断结果。不能根据最高的外层分数直接宣布某个 beta 是独立验证后的最优方法。
固定 beta=1 是 max(G,L)，beta=.5 是等权均值。若 max 更强，首先说明固定简单机制可能比当前逐折选择流程更合适，不能把 max 本身包装为新的复杂融合创新。
若 fixed beta 在 full 阶段也改善，需要进一步确认改善来自真事件/误报的合理变化，并保留同范围强单模块及均值对照，再决定是否冻结方案做 CASME3 验证。
本地测试验证折排除、已有计数、raw/full 分离与端点一致性；真实 full 结果待本轮 Colab 运行。
