# 下一轮：参考尺度与局部半径扩展

本轮问题：原来的参考尺度与局部半径范围，是否限制了 G/L 证据和融合效果？
这是开发实验，不预先承诺融合胜出，不自动启动 CAS(ME)3。

## Colab 操作

1. 继续使用已配置好的 ME-TST+ notebook 和原 Drive 输入，确保 Drive 已挂载、原依赖安装完成。
2. 新建一个代码 cell，粘贴 `STRUCTURE_TUNING_COLAB_CELL.py` 全文，保留 `SETTING = "sammlv"`。
3. 运行该 cell，上传 `metst_one_to_one_structure_tuning.zip`。不要上传旧的 fusion_screening/full/refined 包。
4. 该 cell 一次完成上传、独立目录解压、子进程运行、日志保存和结果打包。上传文件名自动增加 `(1)` 等后缀不影响运行。无需再运行旧上传或旧实验 cell。
5. 完成后从 Drive 下载打印出来的 `oto_structure_*.zip`，回传分析。

输出位置：`MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING/oto_structure_<UTC>/`。
如刚重启 Colab，应先恢复原有 ME-TST+ 环境；本 cell 不负责重新安装环境。

## 搜索范围

- 参考尺度 a0：0.5、1、1.5、2、2.5、3、4。
- 局部半径 rho：0.5、1、1.5、2、3、4。
- 阈值 tau：0.05 至 0.95，步长 0.05。
- DominantEvidence 的 beta：0.5、0.6、0.7、0.8、0.9、1。
- WeightedMean 的 alpha：0、0.2、0.4、0.6、0.8、1。
- 对齐容差 gamma：保持 0.5；L 的支持尺度保持 1、1.5、2；候选最小距离、事件几何和 recognition/result synergy 保持原实现。

新 a0 只新增参考曲线、候选和对应 G；不会把所有新增尺度加入 L 的 median。
参考平滑宽度为 `min(N, max(1, round(a0*k)))`，局部单侧窗口为 `max(1, round(rho*k))`。
相同整数参考宽度共享平滑结果，相同整数参考宽度/局部窗口共享证据。
名义配置行保留，用于审计；不能把它们都说成独立有效配置。G 不重复搜索无作用的 rho。
相同选参分数下保留旧结构优先，再按固定 grid 顺序，避免只因插入新数值改变旧配置的平局顺序。

| 方法 | 名义配置数 |
|---|---:|
| G | 133 |
| L | 798 |
| EqualMean | 798 |
| WeightedMean | 4788 |
| DominantEvidence | 4788 |
| 合计 | 11305 |

搜索主体约为上轮 2451 配置的 4.6 倍；按上轮约 13 分钟粗估约 1 小时，实际受 Colab CPU、Drive I/O 和特征计算影响。GPU 并非主要加速来源。

## 评价与复现

继续使用 `event_one_to_one_v1_prediction_order`；先保留旧 Native/GLSD 原评价回放，再检查 subject 037 的一对一匹配修复。
所有方法均在每个留出折的其他被试上汇总 raw TP/FP/FN 选参，最终报告留出被试 full 计数；这不等于重新训练 backbone 的完整 nested CV。

包内 `refined_reference_sammlv.json` 来自用户提供的上一轮 `oto_refined_20260922T062648_649126Z-20260922T064252Z-1-001.zip`。
每个方法完成搜索后，在原结构子网格核对：全部 raw 搜索计数哈希、逐折选中参数、逐折 raw/full 计数。
任何不一致停止该次运行，不将不一致解释为方法提升。该文件只用于复现检查，不用于选择新参数。

## 一次搜索、四组结构范围

在同一搜索计数表上，分别只用其他被试重新选参，输出：

- `original`：a0={1,1.5,2}，rho={1,2,3}，复现上一轮。
- `a0_only`：只扩大参考尺度。
- `rho_only`：只扩大局部半径。
- `expanded`：两者都扩大，本轮预先指定的主结果。

这些是搜索空间对照，各自重选其他超参数，不是固定参数的单因素消融。G 的 `rho_only` 应与 `original` 相同。
不要根据外层结果从四组中逐被试挑最好值，或将四组最高者自动当作确认后的最终方法。

重要输出：

- `one_to_one_summary_full.csv`：两者都扩大的主结果，含 Native。
- `metst_sammlv/structure_subset_summary.csv`：四组范围 × 五种方法，含 raw/full。
- `structure_subset_selected_configs.csv`、`structure_subset_per_subject_counts.csv`：各组逐折参数和计数。
- `previous_refined_replay.json`：五种方法的旧范围完整计数/参数复现检查。
- `structure_feature_diagnostics.csv`：响应哈希、实际整数窗口、候选数、G/L 均值、L=0 比例。哈希不是视频标签，L=0 也不能直接等同于对齐缺失。
- `selected_predictions.jsonl.gz`、`structure_subset_predictions.jsonl.gz`：主结果与子网格选择的预测。
- `paired_comparisons.csv`：扩展范围中新融合相对各对照的探索性被试 bootstrap 区间。

## 如何决定后续

优先看扩大范围是否提高各方法自身的留出 full F1、TP/FP/FN，以及新融合相对同范围 G/L/均值的差值。
只因单模块下降而产生的排名变化，不算新融合自身改善。
如果所有方法都改善但新融合仍不优于均值，说明结构扩展有用，尚不能证明新融合有增量。
如果训练侧提高而留出侧下降，进一步看逐折参数稳定性；不自动继续扩大网格。
是否进入 CAS(ME)3，由本轮主结果与结构对照共同决定。

本地测试使用真实封存特征实现和合成响应；真实 Colab 官方响应的结果仍待用户运行。
