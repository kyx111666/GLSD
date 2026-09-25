# ME-TST+ 一对一匹配修复与完整调参

## 阈值网格修订入口

`metst_one_to_one_refined_tuning.zip` 支持新增参数 `--threshold-grid refined`：
所有方法统一使用 tau={0.05,0.10,...,0.95}，共 19 点，其他设置不变。
G=57、L=171、EqualMean=171、WeightedMean=1026、DominantEvidence=1026，共2451配置。
使用 `REFINED_TUNING_COLAB_CELL.py` 的完整内容作为新 cell，上传 refined ZIP。
默认仍只跑 SAMMLV，保存到同一输出根下的 `oto_refined_<UTC>/` 新目录。
出错案例按 (a0,rho,tau) 定位，不依赖网格修改前的 config_id。
不提供该参数时保持 legacy 十点阈值网格。仅阈值网格叫 legacy，匹配仍为一对一。

2026-09-22。新评价协议：`event_one_to_one_v1_prediction_order`。

## 已确认问题

用户 Colab 的 `Utils/mean_average_precision_str/mean_average_precision/utils.py`
中，`check_box` 注释了 GT 已匹配检查及成功后的 break。它允许一个 GT 被多个
预测重复计为 TP，也允许一个预测累计多个 TP。实际出错配置为 SAMMLV 被试 037，
L-only，a0=1、rho=2、tau=0.05（config_id=10），raw=(4,15,-1)。

本包使用独立的一对一事件评价协议，保留预测遍历顺序、IoU 阈值及原 IoU 排序。
soft 模式匹配排序中第一个未用 GT；greedy 模式只考虑最高 IoU GT。每个预测最多
贡献一个 TP，同一视频内每个 GT 最多被匹配一次。保持识别流程所需嵌套索引格式。
不复用 crowd GT；非零 difficult 标记会明确报错。本修复不是最大二分匹配。

## 运行

在现有已配好的 ME-TST+ Colab 中粘贴 `ONE_TO_ONE_COLAB_CELL.py` 全文到一个新 cell，
运行后选择 `metst_one_to_one_full_tuning.zip`。无需先上传旧包或重建环境，文件名
带 (1)/(2) 后缀亦可。默认先跑 SAMMLV；之后可设置 SETTING="casme3"。

输出：`MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING/oto_<UTC>/`。
脚本依次：

1. 使用未修改评价器精确重放 Native 和旧 GLSD，保存在 legacy_preflight。
2. SAMMLV 下记录 037 的旧 raw 计数，再只在当前进程内启用一对一匹配。
3. 重放 037 的 raw/full，检查非负计数、GT 守恒及预测事件未被修复改变。
4. 统一重评 Native，然后以相同新评价器运行全部方法的 inner 搜索和 outer full 评价。
5. 重新核对 EqualMean 的评分替换与未修改均值 scorer 在同一新评价器下的一致性。

完整参数范围和配置数不变：G=30、L=90、EqualMean=90、WeightedMean=540、
DominantEvidence=540。不会删除或跳过导致旧评价器负 FN 的配置。
全部评分、候选构造、事件几何、raw 选参、full 结果换算和 outer 被试排除不变。

## 如何看结果

- `one_to_one_summary_full.csv`：统一新评价协议下的 Native 和五方法结果。
- `paired_comparisons.csv`：S_beta 与四个融合/单模块对照的探索性被试配对区间。
- `metst_sammlv_evaluation_audit/subject037_*.json`：故障案例前后计数及预测。
- 各数据集的参数选择、搜索计数、候选记录等，沿用 full-tuning 输出。

旧数字仅供旧协议重放，不与新数字混合比较。旧数据中的非负 FN 不能排除重复
匹配。这个问题不能证明融合有效或无效；须在统一修正后的协议下重新比较。
旧论文及原生方法若受该问题影响，也需重新核算后再决定报告口径。

## 验证边界

本机可复现同类官方 STR evaluator 的合成负 FN，并测试一对一计数、嵌套匹配
索引、空预测、跨视频隔离、异常后恢复、beta 端点及 held-out 选择隔离。
真实 037 的修复后数值、ME-TST recognition 兼容性和完整调参仍需 Colab 验证。
PASS 只表示代码执行和相应检查完成，不代表新融合性能胜出。
