# 参考尺度残差共同结构搜索

SAMMLV冻结官方响应实验；不训练、不运行骨干网络。作者评价运行时随包提供，沿用上一轮核验通过的接口处理，无需下载完整ME-TST仓库或运行历史哈希审计。

## 本轮预定协议

- 主方法：RefResidual = clip(G + L - L0, 0, 1)，不搜索负残差权重。
- 独立基线：G、L、Mean、PositiveResidual；各自内层选参。
- a0 = {1, 1.5, 2}；rho = {1, 2, 3}；tau = {0.05, 0.10, ..., 0.95}。
- G不受rho影响：57个有效配置；其他四个方法各171个，总741个。有效搜索数不同，如实报告，不通过重复G配置伪造等预算。
- 每个外层被试完全排除后，按其余被试汇总full F1选择；依次以precision、较少FP、固定配置索引打破平局。索引按a0、rho、tau升序。
- 汇总外层TP/FP/FN后计算F1，不平均被试F1。
- locked_no_local恢复G；locked_no_negative使用PositiveResidual。两者继承完整RefResidual的a0/rho/tau，不独立调参。独立L-only仍保留。
- 固定物理尺度集合、alignment、median、missing=0、解码与识别。改变a0会改变候选池；同一a0下各方法共用候选，rho不改变G或候选池。
- 保留28项合成接口审计、87次真实解码初检；随后每次搜索/复放仍验证两评价器GT身份一致和事件计数。
- 原始响应与旧实验输出只读，新目录存结果。哈希记录仅用于来源与本次中断续跑，不作为历史启动门槛。

## Colab

打开配套两格notebook，或将 `COLAB_STRUCTURE_CELLS.py` 的两格追加到当前笔记本末尾。先运行Cell 1并上传 `glsd_structure_ready.zip`，再运行Cell 2。CPU运行时足够；不要为了本轮重跑旧环境重建、训练或负权重搜索。

Drive输入：`MyDrive/ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1`，29份subject_*.pkl。

输出：`MyDrive/GLSD_RESIDUAL_STRUCTURE/sammlv_时间戳/`，自动下载同名ZIP。中断后，将Cell 2的STRUCT_RESUME设为之前打印的本轮OUTPUT路径；续跑会核对本次代码/输入/运行时，并跳过已经保存的完整被试。若环境版本改变，使用新输出目录。

## 本地命令

Python 3.10+，安装requirements.txt，然后：

```sh
python run_structure.py --dump /path/to/SAMMLV_method1_strategy1 --output /path/to/new_result
```

## 如何读结果

`summary_full.csv`是预定主比较；`selected_configs.csv`是各被试内层选择；`event_records.jsonl.gz`和`event_differences.jsonl.gz`保存GT和预测变化。`paired_comparisons.csv`是条件于已选预测的被试bootstrap，不重新调参，不解释为独立确认。

`conditional_structure_summary.csv`保存九个固定结构下各自内层选阈值的45行诊断。不能从这张表事后挑最好的结构冒充独立选参成绩。G在三个rho诊断行里重复只是显示方便，实际只搜索一次。

`features.jsonl.gz`含79视频×9结构的G、L、L0、逐尺度支持和匹配峰；便于查明丢失GT来自候选变化、缺失尺度还是残差幅度。每折实际消融记录保持相同结构，可解释负校正的直接作用。

判断目标：共同结构选参后，完整残差是否超过独立G/L/Mean/PositiveResidual，锁参负校正是否减少FP且没有抵消收益的TP损失，收益是否集中在少数被试。结果允许支持或否定当前搜索范围内的机制，不能证明所有连续参数下绝无更优配置，也不承诺SOTA。
