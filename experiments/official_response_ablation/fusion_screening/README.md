# ME-TST+ 六方法融合筛选：Colab 操作

本入口仅跑固定 a0=1.5、rho=2、原对齐容差 gamma=0.5 的筛选。六种方法为 G、L、EqualMean、WeightedMean、DualOR、CrossSupported；配置数分别为 9、9、9、45、81、243。输入是当前 ME-TST+ 官方响应，输出保留原官方 result synergy 后的 full 评价结果，raw 结果单独保存。没有新方法的实际成绩随包提供。

## 1. 打开原来的 ME-TST+ Colab

使用能够运行旧官方响应消融的同一个 notebook/runtime，并先完成它原有的依赖、官方仓库和路径设置。不要运行生成响应、训练 backbone 或覆盖旧结果的单元。

本包不包含你的响应数据、标注、模型权重或官方仓库。两个源 runner 与锁定 core 仍从原 Drive 证据目录加载。

也可打开本包的 `fusion_screening_colab.ipynb`，但新的 runtime 仍需要先完成原 notebook 的环境设置。仅挂载 Drive 不代表官方评价器的 Python 依赖已经安装。

## 2. 上传脚本包并检查输入

在 Colab 上传 `metst_fusion_screening.zip`，使用配套 notebook 的上传单元解压到 `/content/glsd_fusion_screening/`。包内含新筛选脚本和原封不动的 `official_response_component_ablation.py` 依赖。

输入位置：

| 设置 | Drive 中的官方响应目录 | 被试/视频/GT |
|---|---|---|
| SAMMLV | `ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1` | 29 / 79 / 159 |
| CAS(ME)3 | `ME-TST_OFFICIAL_DUMP/CASME_3_method1_strategy1` | 94 / 462 / 853 |

锁定 runner/core/结果位于 `GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE` 与 `GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE`，具体文件名继承原消融脚本。

先运行只读检查：

```python
import subprocess, sys
subprocess.run([
    sys.executable, '/content/glsd_fusion_screening/run_fusion_screening.py',
    '--setting', 'both', '--check-inputs',
], check=True)
```

它只核查输入路径，不执行解码。缺文件时会逐项打印位置并停止，不能用较早的 controlled cache 替代。

## 3. 运行首轮筛选

运行 notebook 的 screening 单元。它自动先对两个数据集重放 Native 和旧 GLSD，然后开始六方法搜索，不需要另跑一次 preflight。默认新建目录：

```text
/content/drive/MyDrive/GLSD_FUSION_SCREENING/screen_<UTC时间>/
```

每个数据集的旧结果必须先精确重现：

| 设置 | Native full TP/FP/FN | 旧 GLSD full TP/FP/FN |
|---|---|---|
| SAMMLV | 52 / 171 / 107 | 47 / 121 / 112 |
| CAS(ME)3 | 76 / 736 / 777 | 90 / 898 / 763 |

如果输入、官方库导入、原结果重放或均值替换的一致性检查失败，会报错停止，不继续写成成功结果。保存日志并定位第一个错误即可；不要为了通过检查修改旧计数或 GT 数量。

新脚本拒绝已有输出目录，也拒绝写入源响应与旧证据目录。运行原消融脚本的 `--setting all` 会涉及 BoostingVRME；本次使用的是新筛选入口，其 `both` 仅代表 ME-TST+ 的两个数据集。

## 4. 选参与计数

每种方法先计算配置×被试的 raw TP/FP/FN 表。对于外层被试 s，只汇总其他被试的 raw 计数选择配置，使用原有 F1、precision、较少 FP、固定配置顺序的规则。之后对 s 使用选中配置，经原官方识别/result synergy 路径得到 full 计数。

这是在现有冻结官方响应上的按被试留出解码器选择，不重新训练 backbone，也不重新估计输入记录中的 k。初次读取每条响应时核对新 EqualMean 与锁定 core 的候选选择完全一致；G/L 只缓存，不重写计算公式。

跨被试汇总先加 TP/FP/FN，再算 F1。不同方法的搜索预算不同，输出如实记录配置数，不能标为等预算实验。

## 5. 完成后看三个文件

1. `screening_summary_full.csv`：每个数据集六行，包含方法、配置数、TP/FP/FN、Precision、Recall、F1。
2. `paired_comparisons.csv`：CrossSupported 相对其他五种方法的 F1 差值及被试配对 bootstrap 95% 区间（10,000 次，seed=100）。这些是探索性区间，未做多重比较校正。
3. `decision.json`：两数据集的事实判断和预算建议，不自动启动下一轮实验。

`PROMISING_REVIEW_BEFORE_FORMAL` 表示两个数据集上新方法点估计不低于 G/L/加权均值，并超过均值与 OR，值得结合区间和错误变化评估下一步。`STOP_EXPANSION_REVIEW_FUSION` 表示两个数据集都低于更强单模块且低于加权均值，当前条件下没有足够理由扩大搜索。其余情况为 `MIXED_REVIEW_COUNTS_AND_CIS`，先检查代价与收益；它不等于失败。单独跑一个数据集时为 `WAIT_FOR_OTHER_DATASET`。

这些标签是时间投入建议，不是方法不可能有效、统计显著或最终论文结论。固定结构参数的负结果不能否定所有 a0/rho 条件；一个数据集的提升也不能替代另一个的检查。

## 6. 完整记录

各数据集子目录还包含：

- `protocol.json`：全部配置、固定参数、输入代码哈希、评价阶段与选择规则。
- `per_subject_counts.csv`：每种方法、每个留出被试的 raw/full 计数。
- `selected_configs.csv`：每个方法每个外层折所选参数。
- `search_all.csv`：排除对应外层被试后，每个配置的训练内部汇总计数。
- `search_counts_<method>.npz`：原始配置×被试×三计数数组，可重建选参并用于后续参数分析。
- `selected_predictions.jsonl.gz`：选中配置的原官方预测输出与计数。
- `candidate_decisions.jsonl.gz`：候选峰、G/L、是否通过，以及双阈值分支。`feature_call_index` 是该次解码的特征调用序号，不能未经核对就当作官方视频 ID。
- `completion.json`：该数据集实际执行完成的标志。

`SCREENING_EXECUTION = PASS` 只表示运行与协议检查完成，不代表新方法胜出。根目录没有 `completion.json` 时不能当作两数据集完整结果；已完成的方法记录和日志仍会保留。

## 验证范围

本机可验证融合逻辑、端点退化、测试被试不参与选择、确定性 tie-break、配对区间以及与本地锁定 G/L core 的均值一致性。Colab 的官方 ME-TST+ runner、响应与实际成绩需要在那里运行后验证，不能把本地协议测试当作真实筛选。

2026-09-22 已通过 8 项本地协议测试，notebook 的 JSON 与所有代码单元语法检查通过。尚未运行 Colab 官方输入预检查、旧结果重放或本轮实际筛选；这些不能标为已完成。
