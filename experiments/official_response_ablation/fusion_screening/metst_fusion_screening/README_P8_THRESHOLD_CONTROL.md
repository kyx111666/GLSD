# P8 阈值对照与事件级消融：Colab 接续

本包是待运行的新实验，未生成新的真实实验结果。沿用冻结官方响应，不训练、不生成 backbone 响应。旧运行包和旧结果保持原样。

## 在上次笔记本中运行

1. 下载 `metst_p8_threshold_control.zip`。
2. 将 `Colab_P8_阈值对照与事件消融接续.ipynb` 的四个代码单元格依次复制到上次已配置好环境的笔记本末尾。也可按 `P8_THRESHOLD_CONTROL_COLAB_CELLS.py` 中的 CELL 1–4 分隔复制。
3. CELL 1 上传新 zip。默认读取上轮目录：
   `/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z`
   如实际目录不同，只改 `P8_CONTROL_REUSE`，指向包含 `run_manifest.json`、`completion.json` 和两个数据集子目录的 full 根目录；不要选 probe。
4. CELL 2 跑 SAMMLV probe。看到 `P8_THRESHOLD_PROBE = PASS` 后运行 CELL 3。
5. CELL 3 跑 SAMMLV full。看到 `P8_THRESHOLD_FULL = PASS` 并显示表格后运行 CELL 4。
6. CELL 4 跑 CAS(ME)3 full，内部先做该数据集的小规模 probe，再开始新增阈值搜索。

无需重新执行上一轮 full 搜索。若 Colab runtime 已断开，只恢复原依赖、`/content/ME-TST` 和 Drive 挂载，不重跑训练或响应生成。本包严格检查 Python/NumPy/pandas 与上轮 manifest 一致；不一致会明确停止，不自动安装/升级，也不静默混用旧计数。

每阶段自动下载一个结果 zip，Drive 同时保留全部结果。运行期间不要关闭/重置 runtime。下载弹窗不影响已保存到 Drive 的结果。

## 本轮内容

| 方法 | 配置数 | 计算来源 |
|---|---:|---|
| G | 57 | 兼容性验证后复用上轮计数 |
| L、Mean | 各 171 | 同上 |
| GM_p8 | 57 | 上轮 p8 网格的 rho=1 子集，重新执行内层选择 |
| G_rescaled | 57 | 新增搜索，使用 G >= 2^(1/8) tau |
| G_expanded | 114 | 合并原 G 与映射 G 的计数，独立内层选择 |
| p8_no_L、p8_no_G | 无独立选参 | 每折锁定完整 p8 参数，分别去掉 L/G，回放评价 |
| Native | 0 | 复用修正评价器下的 Native 计数 |

p=8、rho=1；a0={1,1.5,2}；tau={0.05,...,0.95}。映射后超过 1 的阈值不截断。重标度 G 的比较采用等价的映射形式，避免浮点边界在两种写法之间产生不一致。

统一评价器 `event_one_to_one_v1_prediction_order`。每折按其余被试的汇总 full F1 选参；平局依次按 precision 高、FP 少、固定配置索引小处理。扩展 G 先排原网格、再排映射网格。外层汇总 full TP/FP/FN 后计算 F1。各方法预算不同，不称为等预算实验。

每个数据集只新增 57 组映射 G 的搜索。复用前验证旧代码、响应文件清单和哈希、配置、被试顺序、完成状态、计数守恒、运行时及评价器身份；probe 和所有选中配置还会实际回放，要求 raw/full 计数与缓存完全一致。

## 输出

新根目录：`/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/`。

每次运行新建 `probe_sammlv_时间`、`full_sammlv_时间` 或 `full_casme3_时间`，互不覆盖。各 full 目录中数据集子目录包含：

- `summary_full.csv`：各方法最终 TP/FP/FN、precision、recall、F1。
- `paired_comparisons.csv`：p8 对各对照的 F1 差、被试配对 bootstrap 区间。条件于已选预测，不重新选参，也不校正多轮探索。
- `decision.json`：p8−扩展 G、p8−L、完整−去 L，以及完整−去 G。
- `selected_configs.csv`：逐折内层选中参数和内层 F1；消融参数保存在事件记录中。
- `per_subject_counts.csv`：逐被试各方法 raw/full 计数（Native 计数来源在旧目录）。
- `search_counts_*.npz`：本轮实际使用的搜索张量。
- `event_records.jsonl.gz`：逐被试逐方法的候选、原始预测、full 保留状态、识别标签及 GT 匹配 ID。
- `event_differences.jsonl.gz`：完整 p8 相对各对照新增/丢失 GT ID、候选差异和 TP/FP/FN 差。
- `probe.json`、`probe_event_records.jsonl.gz`：当前真实数据 probe 的检查结果与事件记录。

GT ID 由被试、视频索引、GT 索引组成；同时保存视频名、注册 GT 坐标和源标注。candidate trace 通过逐视频响应哈希验证对应关系。full 是官方识别/result synergy 后的预测保留状态，不重新匹配 GT；每个事件贡献调用封存 synergy 函数计算，并核对事件账本可以精确还原 raw/full 计数。

## 中断恢复

不要把旧 full 或 probe 路径填入新实验的 resume。只填写该次失败的新输出目录，mode 和 setting 保持一致，代码不要改动：

```python
P8_CONTROL_MODE = 'full'
P8_CONTROL_SETTING = 'sammlv'  # CAS 用 casme3
P8_CONTROL_RESUME = '/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_sammlv_实际时间'
entry = control_dir / 'run_colab_p8_threshold.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
```

映射 G 搜索按被试恢复；中断的当前被试会重算。事件导出中断后从头回放选中配置，但已完成搜索不重做。正常开始下一个阶段时把 `P8_CONTROL_RESUME=None`。

## 运行后发回什么

先发 probe 下载包；通过后发 SAMMLV full 包，再发 CAS full 包。若报错，提供末尾 traceback 和新目录旁的 `log_时间.txt`。不要为了继续而绕过计数、事件 ID 或兼容性断言。

本轮属于开发诊断；p8 和 rho=1 均来自已有结果。最终判断同时看阈值覆盖充分的独立基线与锁参消融，不将点估计提升等同于独立确认。

## 本地交付校验（2026-09-22）

- 22 项回归测试通过，包含原 14 项及新增 8 项。
- 使用真实一对一评价器的合成案例验证事件 ID、重复预测和候选视频对应。
- 阈值映射边界及大于 1 的阈值检查通过。
- 小型流水线测试确认只调用一次新增网格搜索、两项消融继承完整方法参数。
- 新复用加载器读取用户上轮完整结果，复算固定 rho=1 的计数为 SAMMLV 44/101/115、CAS(ME)3 94/940/759，与既有诊断一致。这是旧计数复算，不是新实验结果。
- 打包引用的七个旧脚本哈希均与上轮 manifest 一致。

尚未在 Colab 运行真实输入上的新 probe 或 full 搜索；以实际运行输出为准。

## Cell 2 的 gt_all 接口修复

首次 Colab probe 报 `AttributeError: MeanAveragePrecision2d ... gt_all`。原因是事件导出误用了另一种评价器的内部缓存接口，未进入新增网格搜索。

现在通过暂时观察评价器公开 `add(preds, gt)` 调用，复制实际注册的预测与 GT；按返回的 video metric 实例获取记录，并在正常结束或异常时恢复原 add。传给原 add 的输入、返回值和匹配协议均不改变。预测顺序、标注坐标、GT ID 和 raw/full 守恒检查仍保留。

已加入无 gt_all/pred_all 的评价器回归测试。重新下载新 ZIP，并将当前 Cell 1 换为新接续笔记本中的 Cell 1（旧 Cell 1 固定了旧包哈希，不能用来加载新版）；运行新 Cell 1 后，原 Cell 2–4 可以继续使用。Cell 2 保持 P8_CONTROL_RESUME=None，使用新 probe 目录，不恢复失败的旧 probe。

也可直接复制 `P8_THRESHOLD_REPAIR_CELL.py` 到当前笔记本的新单元格运行：它上传修复包并重新执行 SAMMLV probe。此后直接运行原 Cell 3、Cell 4。

## 双评价器调用链修复

第二次 probe 的 `returned video evaluator has no captured add calls` 源于解码器传入的 `_str` 汇总评价器与 `training_utils.spotting` 全局绑定的视频评价器不是同一个类。现在从 `official.spotting.__globals__` 解析其实际视频评价器，同时观察两个类的 add；只记录输入，不改变任何匹配函数，退出时恢复两者。

新增双评价器集成回归测试。另用用户原 notebook 中提取的原始解码函数、本地官方 spotting/recognition 函数及两种真实评价器完成了两视频合成输入回放，事件记录重建 raw/full 均为 2/0/0。该验证不使用真实 Colab 响应，不是实验结果。

重新下载当前 ZIP 和修复单元格代码即可在当前 Colab 运行，不重启、不重做旧搜索。真实 probe 仍以 Colab 执行为准。
