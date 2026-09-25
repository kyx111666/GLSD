# 方案 5 公平对照复验

本实验比较 ME-TST 与 BoostingVRME 上多尺度候选、Height 和 Support 的独立作用。
代码只读取已有分数缓存；结果各自写入 `metst/results/equiscale_fair_validation` 和
`boostingvrme/results/equiscale_fair_validation`。此前的方案 5 和方案 11 结果保留。

## 运行

```powershell
& 'D:\Anaconda3\envs\ME-TST\python.exe' 'D:\workspace\a\metst\equiscale_fair_validation.py'
& 'D:\Anaconda3\envs\ME-TST\python.exe' 'D:\workspace\a\boostingvrme\equiscale_fair_validation.py'
```

两个入口共用 `boostingvrme/equiscale_fair_validation.py` 的算法实现。
可用 `--dataset sammlv` 或 `--dataset casme3` 选择数据集，`--audit-only` 只检查原始
基线复现与各折时长参数。匹配签名的计数缓存支持续跑；签名不匹配时会停止，防止混用旧结果。

```powershell
& 'D:\Anaconda3\envs\ME-TST\python.exe' -m unittest discover -s 'D:\workspace\a\boostingvrme' -p 'test_equiscale_fair_validation.py' -v
& 'D:\Anaconda3\envs\ME-TST\python.exe' 'D:\workspace\a\boostingvrme\verify_equiscale_fair_results.py'
```

## 对照设置

| 方法 | 候选与筛选 |
| - | - |
| Native（训练折时长） | 原始阈值 0.55、平滑 2k、峰距 k；时长 k 只从训练被试估计 |
| 调参单尺度 | 包含多尺度网格使用的全部实际平滑宽度和峰距组合 |
| 多尺度合并 | 关联各尺度候选，全部交给同一边界解码器 |
| 多尺度 Height-only | 同一候选池，按 Height 的稳健 z-score 筛选 |
| 多尺度 Height+Support | 同一候选池，联合选择生成参数、Height/Support 权重和筛选阈值 |
| 同候选同设置加入 Support | 固定 Height-only 选中的候选生成、聚合方式和阈值，只再选择融合权重 |
| 仅融合子池 | 排除纯 Height 的辅助结果，保留用于完整汇报 |

主检验是“同候选同设置加入 Support”相对 Height-only 的增量；联合调参是次要检验。
两者均包含权重 1.0，即纯 Height，作为可选配置。所有方法都汇报，不能事后只挑正向子池。

候选阈值统一为 `0.4, 0.5, 0.55, 0.6`。沿用前轮六个尺度集合，平滑倍率为
`1, 1.5, 2, 2.5`，峰距倍率为 `0.75, 1, 1.25`，关联半径为 `0.25, 0.5, 0.75` 倍 k。
Height 聚合为 max、mean 或 top2 mean，最终 z-score 阈值为 0 或 0.5，Height 权重为
`0.25, 0.5, 0.75, 1.0`。共 324 个单尺度配置、864 组多尺度候选生成设置、21,924 个计数配置。

## 验证与解释边界

ME-TST 的 k 保留原始平均时长规则，BoostingVRME 保留中位时长规则。外层测试时排除测试被试；
内层验证时同时排除外层测试被试与内层验证被试。ME-TST 使用固定 `peak +/- k` 区间，
BoostingVRME 使用原始区间延展及 NMS，同一模型内各方法保持一致。

聚类以最高相对高度的候选为中心，关联容差范围内其他尺度最近的未使用候选。每尺度最多一票。
同尺度的另一个候选不会被直接丢弃；取整后 `(平滑宽度, 峰距)` 相同的尺度只计一票。
这修正了旧实验的聚类规则，因此新旧结果差值不能归因于单一参数或仅仅增加对照。

每个被试的最终配置在其他被试的内层计数上选择。区间重新生成后，逐视频匹配结果必须与计数缓存完全一致。
救回 GT 指新方法匹配到而参考方法未匹配到的标注事件；损失 GT 定义相反。
新增与删除 FP 按视频、起点、峰和终点的精确预测标识比较，同时报告 FP 净变化。

95% 区间采用 10,000 次被试配对 bootstrap，固定随机种子 20260906。该区间条件于已有曲线和
各折选出的配置，不涵盖网络重新训练和多轮方法设计的不确定性。

已有曲线按原模型的被试 LOSO 方式生成，缺少每个内外层组合重新训练的模型预测。
其他折模型可能训练过当前外层测试被试，所以本实验仅是冻结曲线上的后处理嵌套验证。
数据集也已参与之前的方法设计，不能当作全流程严格嵌套实验或未接触的外部验证。
BoostingVRME 沿用已有 PyTorch Mamba 参考实现曲线，尚未验证与官方二进制内核逐值等价。

## 文件

每个数据集的 `PROTOCOL.json` 在计算之前保存搜索空间、规则、输入和代码 SHA256。
`report.json` 保存所有方法及配对差值、事件变化和选择频次；
`outer_loso_selections.csv` 保存逐被试配置与训练 k；
`selected_predictions.json` 保存各方法逐视频最终区间和匹配 GT；
`stats_k*.npz` 保存逐配置、逐被试计数，供复核与续跑。
