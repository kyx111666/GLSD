# GLSD 灰区支持 rho3 独立全量实验

本目录是下一轮有限 `rho` 验证的独立交付。它沿用冻结的 ME-TST+ 响应、原有 Colab 环境、候选构造、事件几何、识别响应和 `event_one_to_one_v1_prediction_order` 评价；本地没有官方响应输入，因此这里没有宣称新实验已经运行。

本轮把局部窗口半径 `rho` 设为 `{0.5, 1, 2, 3}`。G 单模块保留原有 `rho=1`，L、Mean、P8_control 和 Gray 都在四个 rho 上选参，所以 rho3 机会对主方法和单模块是对称的。G、L、Mean、P8_control、Gray 的配置行数分别为 `231、924、480、480、5088`，合计 `7203`。G/L 使用原有单模块阈值集合，Mean/P8_control 使用 `0.025` 步长阈值，Gray 保留两条主分支、53 个 `(low, high)` 对和四个 `eta`。

每个外层被试的参数只用其余被试的 pooled full TP/FP/FN 选择，排序为 F1、precision、较少 FP、固定配置索引；随后在留出被试上重新执行完整 recognition/result synergy 和 one-to-one 匹配。`Single_selected` 在 G 与 L 的联合候选中按相同规则选择，作为“合理调参的 G/L 单模块”对照。`Gray_high_only` 与 `Gray_no_support` 使用同一折 Gray 选中的全部参数，只改变保留规则，并重新完整评价；它们不是重新调参的弱基线。

`P8_control` 是本 ZIP 内按同一 rho 网格重新计算的比较项，只用于本轮统一对照，不代表用户并行运行中的当前 P8，也不读取历史 P8 计数。输出中会记录本轮输入、代码、运行时和官方评价器绑定哈希；旧 Gray v1 目录不会作为本轮计数输入。

## 运行入口

在已经配置好的原 ME-TST Colab 中使用以下任一方式：

1. 打开 `Colab_灰区rho3接续.ipynb`，选择对应数据集的完整 cell。
2. 或把 `GRAYZONE_RHO3_SAMMLV_COMPLETE_CELL.py` / `GRAYZONE_RHO3_CASME3_COMPLETE_CELL.py` 的全文复制到一个新 cell。
3. 上传同目录的 `metst_grayzone_rho3_v1.zip`，不要上传旧 `metst_grayzone_support_v1.zip`。

ZIP 内的明确入口文件是 `run_grayzone_rho3.py`、`colab_grayzone_rho3_entry.py`、`run_colab_grayzone_rho3.py` 和 `test_grayzone_rho3.py`；`run_grayzone.py` 只承载同一份实现，避免维护两套 runner。

完整 cell 会先检查 ZIP 哈希、继承源哈希、现有 ME-TST checkout 和冻结输入，再运行包内关键测试。`GRAY_MODE='probe'` 只验证运行入口；正式实验保持 `GRAY_MODE='full'`。中断后，将 `GRAY_RESUME` 改为同一轮 Drive 输出目录并再次运行，不能把已有目录改作另一份实验。

结果写入新的 `MyDrive/GLSD_GRAYZONE_RHO3/<mode>_<setting>_<UTC>/` 子目录，并额外下载 `<output>_grayzone_rho3.zip`。当前 P8 使用同一 Colab 内核时应排队，或使用另一个已经配置好的会话；本轮不会停止、修改或读取当前 P8 的输出。

## 必看输出

- `completion.json`：确认本轮是否完整结束、协议和配置总数。
- `protocol.json`：确认 rho、网格预算、选择和评价口径。
- `summary_full.csv`：最终 full 计数汇总。它只有在 Colab 全量运行后才存在有意义的新结果。
- `selected_configs.csv`：每折选择和两个锁参消融的配置来源。
- `per_subject_counts.csv`：raw/full 的逐被试计数。
- `event_records.jsonl.gz`：保留原预测顺序、one-to-one 匹配和 full synergy 事件账本。
- `event_differences.jsonl.gz`：Gray 相对各比较项的 GT/FP 身份差集。
- `mechanism_by_branch.csv`：按 G 主分支和 L 主分支拆分的模块贡献。
- `paired_comparisons.csv`：以同一外层被试为单位的探索性 paired bootstrap；不进行重新选参，也不能单独作为显著性证明。

诊断包里的几何 eligibility、候选覆盖率和 rho 反事实不能替代这里的 full F1。没有新的 Colab ZIP 前，不要把本地测试、rho3 反事实或 v1 结果写进论文结果表。
