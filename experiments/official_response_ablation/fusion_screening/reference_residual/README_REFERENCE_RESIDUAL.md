# 参考尺度残差：SAMMLV 固定结构筛查

在刚刚通过评价核验的同一个 P8 Colab 笔记本末尾追加两格：上传 `metst_reference_residual.zip`，再运行 SAMMLV screen。两格在 `REFERENCE_RESIDUAL_COLAB_CELLS.py` 和配套接续 notebook 中。

## 为什么暂时固定 a0=2、rho=2

这是控制变量的开发筛查。a0=2 是此前 SAMMLV 的 p8 与独立 L 都使用的参考尺度；rho=2 是独立 L 选中的窗口，也保留了9个历史 L-hit/p8-miss GT对应峰的较强局部支持。这些都是看过开发结果后的选择，不是新残差公式的最优参数证据。固定结构先回答同一组候选和局部证据下，残差评分是否比G、L、Mean有用。

结构固定不意味着阈值固定：每种方法都有19个阈值，各折在其他被试汇总的full计数上独立选择阈值，平局沿用precision、FP、固定索引顺序。外层汇总计数计算F1，不平均被试F1。此阶段不代替独立联合选择a0/rho/tau的最终比较；若机制有支持，再按预先确定的共同结构空间运行后续比较和CAS复核。

## 实际执行

- 数据：SAMMLV，29被试、79视频、159 GT，读取已封存响应；不训练、不重新生成响应。
- 结构：a0=2、rho=2；保留原尺度集合、物理宽度去重、对齐、median、missing=0、候选和解码器。
- 分数：G、L、Mean、RefResidual=clip(G+L-L0)、PositiveResidual=clip(G+max(L-L0,0))。
- 阈值：0.05到0.95，步长0.05；共95个方法配置。所有计数重新计算，不混用旧结果。
- 锁参消融：no_local=G；no_G=clip(L-L0)；no_negative=PositiveResidual。全部继承RefResidual每折选中的参数，三者不重新选参。
- L0：读取封存core已有的local_cache。逐候选校验G/L、reference身份、local支持及median；导出每尺度支持、匹配峰、匹配距离、缺失标记、spread和局部对比度。
- 开始搜索前自动运行一次新评分链路probe；它验证新L0接口及五种评分，不要求用户重新执行上一轮阶段0的三个诊断配置。
- 每个解码调用检查两类评价器的GT匹配身份、识别覆盖和事件账本。若出现策略差异或接口异常，保存failure_case.json并停止，不跳过配置，也不自动强制统一soft策略。
- 本轮结果属于开发筛查，不能直接宣称正式Table 3成立。特别是粗阈值网格优势仍需结合候选排序与G阈值覆盖解释。

## 评价调用链

沿用两次真实阶段0回放通过的调用方式：汇总评价器安装原 `check_box_one_to_one`，其调用方默认soft；普通视频check_box保持原函数，其调用方默认greedy。两类只增加结果检查包装。普通视频的返回类型和匹配算法不变。

默认核验依据：`/content/drive/MyDrive/GLSD_EVALUATOR_AUDIT/sammlv_20260922T091343_286158Z`。加载后检查PASS标记，并逐项核对runner、core、training_utils、评价器文件、check_box函数与响应dump哈希。成功标记只用来确定已核验代码身份，不代表未运行配置已经通过。

## 输出与恢复

结果在 `/content/drive/MyDrive/GLSD_REFERENCE_RESIDUAL/screen_sammlv_时间戳/`，结束或报错后自动下载ZIP。关键产物：features.jsonl.gz、feature_summary.csv、search_counts_*.npz、summary_full.csv、selected_configs.csv、per_subject_counts.csv、event_records.jsonl.gz、event_differences.jsonl.gz、paired_comparisons.csv、protocol.json、matching_protocol.json、run_manifest.json。

固定输入身份的完整被试计数按方法保存断点。发生会话中断时，在运行格把 `RESIDUAL_RESUME=None` 改成此前实际的 `screen_sammlv_时间戳` 完整Drive目录；保持运行包、mode、核验依据和运行环境一致。正式进入一个新实验时使用None。身份不符时拒绝恢复。

## 本地验证范围

八项回归覆盖符号校正/正确去模块、封存core中间量与退化尺度、双接口完整性、保留原视频匹配及soft/greedy差异、事件账本与FP移除、合成全筛查/断点，以及使用所提供notebook解码函数和回传真实spotting/recognition函数的合成链路回放。测试包括非空列表与空数组的官方返回形式。两个fixture文件仅用于测试，生产调用只加载Drive封存runner和真实training_utils。

本地测试使用合成响应与识别序列，不能替代完整SAMMLV运行；本轮尚无新融合成绩。
