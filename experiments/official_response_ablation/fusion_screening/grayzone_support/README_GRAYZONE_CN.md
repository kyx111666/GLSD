# 主分支 + 灰区相对支持 v1：运行与协议

2026-09-22。本轮交付代码，尚未在用户 Colab 上执行真实官方实验。用户当前 P8 仍是另一运行批次，确切 manifest 尚未收到；这里没有读取或判断其成绩。S_beta/max 已放弃，本包没有该路线的搜索。

## 直接运行

- 上传包：`metst_grayzone_support_v1.zip`。
- SAMMLV：把 `GRAYZONE_SAMMLV_COMPLETE_CELL.py` 全文复制进原有、已配置的 Colab 新 cell，执行并只上传上述 ZIP。
- CASME3：把 `GRAYZONE_CASME3_COMPLETE_CELL.py` 全文复制到另一新 cell，执行并上传同一 ZIP。
- 或使用 `Colab_灰区支持接续.ipynb` 中两个对应的完整代码 cell。它是接续文件，不负责从零安装 ME-TST。
- 默认直接 `full`：自动做来源回放、关键路径检查后进入全量搜索，不需要另跑一个 probe 或再次确认。
- 每个数据集输出到 `MyDrive/GLSD_GRAYZONE_SUPPORT/full_<setting>_<UTC>/`。结束出现 `GRAYZONE_FULL = PASS`，自动下载结果 ZIP，同时保存 ZIP 到该 Drive 目录。请回传两个 `*_grayzone.zip`。
- 两个 cell 各自独立上传、独立子进程、独立代码解压路径和输出目录；原冻结响应、旧结果、当前 P8 文件均不写入。

**P8 正在占用的 Colab 内核不能同时执行另一个 cell。** 本地代码准备已可与 P8 并行；若要实验计算也并行，在另一已配置同一 ME-TST 环境的 Colab 会话执行这里的 cell、共享只读 Drive 响应。只有一个可用会话时，将新 cell 排在 P8 后运行，保留 P8 任务。此包不会停止或接管 P8。

中断恢复：在相应完整 cell 顶部把 `GRAY_RESUME = None` 改为日志中本轮实际输出目录的字符串，保持 setting/mode/ZIP/原环境不变，再运行同一完整 cell。每 100 个配置或约 45 秒原子保存一次计数（单次官方解码期间无法保存）；已完成的配置不重算。选中事件也逐折保存。若更换代码/输入/运行环境，恢复会拒绝，应使用新输出而不能混接。仅做环境检查时可将 `GRAY_MODE` 改成 `probe`；probe 输出不能作为 full 恢复目录。

## 实现和评价核对

复用 `metst_fusion_screening` 下的原 helper、G/L feature wrapper、one-to-one adapter 和事件导出器，按字节复制到本包；继承文件的 SHA256 记录在 `SOURCE_AUDIT.json`，打包文件哈希记录在 `package_manifest.json`。实际 Colab 仍从 helper 的既有 Drive 路径加载 **封存 core/runner**，不会拿本地相似 core 替代。

SAMMLV 沿用 `ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1` 和 `GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE`；CASME3 沿用 `ME-TST_OFFICIAL_DUMP/CASME_3_method1_strategy1` 和 `GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE`。运行前记录所需封存文件、完整 dump、当前 ME-TST checkout Python 源码的哈希。helper 的旧 manifest 检查本身不是对全部封存 checksum 条目的逐条认证；所以本包另外执行旧 Native/GLSD 的计数及逐被试回放，并保存本次输入的内容哈希。首次运行的哈希是本次身份快照，不伪称本地已核验不可访问的 Drive 内容。

运行时检查 29 被试 / 79 视频 / 159 GT，以及 94 / 462 / **853 GT**。858 GT 缓存不兼容。

G 是参考尺度的归一化全局 prominence；L 从 core 原始 `evidence(a0,rho)` 获取。原支持尺度、median 聚合、对齐容差、峰顺序、事件区间、recognition/result synergy 均由封存实现负责。每个访问的结构会核对均值 hook 与封存 core 的保留峰完全相同；还会验证 G 和候选池不随 rho 改变。

主协议为 `event_one_to_one_v1_prediction_order`。仅安装到传给 sealed runner 的汇总 `_str` 评价器；`official.spotting` 自身绑定的视频评价器保持原有行为，导出器观察两个类的 `add` 输入。保存这两个实际绑定及源文件哈希，绝不声称两者已统一替换。

每个**实际完整解码**均重建事件账本：检查 raw/full TP/FP/FN 守恒、GT ID 唯一、视频匹配与 recognition target 一致、逐事件贡献加总等于汇总计数。不一致直接停止并保留 checkpoint；不截负 FN、不跳过配置、不拿旧候选 TP 标签相加。SAMMLV 还自动检查 037 历史故障点修正为 raw=(3,16,0)、full=(3,15,0)。

## 本轮冻结参数与比较

方法版本 `grayzone_relative_support_v1_full_loso`：

`keep = B >= high OR (low <= B < high AND R >= eta * B)`。

一个外层留出被试对应一个主分支 B，R 是另一支。六项参数 **B/a0/rho/low/high/eta 联合选**，选择只用其他被试 pooled **full** F1；平局依次 precision 高、FP 少、固定配置索引小。外层汇总 full TP/FP/FN 再算 F1，raw 同步保存但不用于主选参。

| 项目 | 本轮固定范围 | 理由 |
|---|---|---|
| a0 | 1, 1.5, 2 | core 原生参考尺度，共同用于全部方法 |
| rho | .5, 1, 2 | 沿用已验证原生结构；G 不重复无效 rho |
| 主分支 B | G, L | 每折训练侧联合选择，不看测试标签手选 |
| high | .10, .15, …, 1.00 | 覆盖低阈值 CASME3 与高阈值 SAMMLV |
| high−low | .05, .10, .20，且 low>0 | 限定灰区宽度，不搜索任意巨大双阈值组合；共 53 个合法阈值对 |
| eta | .25, .50, .75, 1.00 | 弱相对支持至同等强度支持；两数据集共用规则 |
| Mean / P8_control 的 tau | .025, .050, …, 1.000 | 40 点，每个结构独立选参 |
| G / L 的 tau | 上述 40 点 ∪ 2^(1/8)×上述 40 点中 ≤1 的值 ∪ {1.1} | 更细单支阈值并覆盖 P8 零另一支时的有效阈值；大于1统一为拒绝全部端点，共77点 |

本范围是本轮开发协议，不是已证明最优超参数。相比交接中仅偏向 SAMMLV 的初步低/高网格，现在覆盖 `.05–1.0`；不沿用 beta 路线的大规模结构扩展，不运行后自动扩网格。

相对支持条件在灰区内对 B **非单调**：R 固定，提高 B 可能让 `R>=eta*B` 失效，直到 B 达 high 后重新保留。此为原方案有意保留的语义，并有边界测试；不是单调融合分数。`eta=0` 退化为 B>=low，本轮不将其重复加入 Gray 搜索，而用 `Gray_no_support` 锁参端点和完整独立 G/L 对照评价。

| 方法 | 配置行数 | 选参 |
|---|---:|---|
| G | 231 | 3×77；rho 固定为1，仅作无作用占位 |
| L | 693 | 3×3×77 |
| Mean | 360 | 3×3×40 |
| P8_control | 360 | p固定8；3×3×40 |
| Gray | 3816 | 3×3×2×53×4 |
| Single_selected | 924 个 G/L 候选配置 | 复用上述计数，在训练侧同时选 G 或 L；不增加解码 |
| Native | 无 | 同一本轮修正协议重新计数 |
| Gray_high_only / Gray_no_support | 无独立搜索 | 两者都锁定各折 Gray 的六项参数 |

共 5460 个搜索配置行。**不是等预算实验**；名义行数也不等于实际不同预测数（例如拒绝全部可跨结构重合）。每被试记录不同的有序候选序列数与本次实际调用次数。不会把无效 rho 或 eta=0 重复网格作为新增搜索预算。

`P8_control` 是本包在相同结构/阈值/full 评价下重新计算的对照，不能当作用户正在运行的那轮 P8。此包不要求旧 P8 结果，也不读取旧 P8 的搜索计数；后续只有核对当前 P8 的 manifest 后才讨论合表。这次重算针对新的统一比较协议，不重跑 backbone。

## 消融与成功判断

- `Gray_high_only`：仅 B>=high。检验 R 支持允许的灰区恢复是否改善取舍。
- `Gray_no_support`：仅 B>=low。检验支持条件是否过滤误报而保留真事件。
- 两者都重新执行 full 路径；分别保存相对完整 Gray 的新增/丢失 GT ID、新增/消失 FP 事件和候选差集。因重匹配，新增候选不自动等于新增 TP。
- 这两项不笼统称作 w/o G 或 w/o L。`mechanism_by_branch.csv` 按当折 B 汇总，B=G 时支持模块是 L，B=L 时支持模块是 G。
- 目标是 Gray 同时超过合理独立调参 G/L，并优于训练侧选支的 Single_selected；再用两个锁参对照和事件差集解释增量。完整方法若与单支等价、第二支只增加 FP，不能算双模块贡献成立。
- 新增 GT 不是必要条件：不丢 GT 而减少 FP 也可构成贡献。但如果只改善高阈值对照、却不优于低阈值或强单模块，则不能单凭“恢复了事件”宣称成功。
- bootstrap：被试配对重采样10000次，seed=100，条件于选定预测、不重选参、不校正反复开发偏差。两套数据都曾被用于开发；冻结响应 decoder LOSO 不等于 backbone 完整 nested CV。
- `decision.json` 仅记录点估计条件与事件差异，不自动宣布统计可信、SOTA 或论文成功。运行失败时首先定位兼容性；性能失败时先分析原因，不默认无限扩参。

## 输出与后续核验

每次结果 ZIP 包含完整输出目录：

- 根目录：`run_manifest.json`、`matching_protocol.json`、`completion.json`、`legacy_preflight/`。
- `metst_<setting>/protocol.json`：冻结网格和评价选择说明。
- `search_counts.npz`：每配置×被试 raw/full TP/FP/FN、完成掩码、配置 JSON 和被试顺序。任一留出折内所有候选配置的 pooled 分数都可由排除该被试求和重建。
- `selected_configs.csv`：每折主分支、六参数、内层 full F1/TP/FP/FN、锁参来源。
- `summary_full.csv`、`per_subject_counts.csv`、`paired_comparisons.csv`、`decision.json`。
- `event_records.jsonl.gz`：各选中 G/L/Mean/P8_control/Gray/Single_selected 与两个锁参对照的全部候选 G/L、灰区进入/支持通过/实际保留掩码、最终预测、GT ID、识别标签及 raw/full 贡献。Native 保存逐被试计数，不输出此候选账本。
- `event_differences.jsonl.gz`：完整 Gray 相对各对照的 GT/FP 身份差集；FP 以视频/完整预测行/同值出现序号定位，匹配身份变化仍可见。
- `selection_frequency.csv`、`mechanism_by_branch.csv`、`selected_events/mechanism_*.json`：选参分布、主支分组贡献、退化成高/低阈值单支的情况。
- `search_subject_*.json`：名义配置数、不同有序候选序列数、该恢复尝试中的实际解码与复用次数。

缓存只复用同一被试下每视频保留峰列表及顺序完全一致的 raw/full 计数。运行前用 AST 确认 sealed decoder 的 config 只流入 `selected_peaks`；否则停止。跨尺度但相同实际峰序列因事件几何相同可以复用。每个最终选中配置重新导出事件并逐项对照搜索计数，不能从其他配置搬 TP 标签。

## 本地验证范围

本地环境 `/opt/miniconda3/envs/torch-mac/bin/python`，未安装新依赖。30项测试通过（8项灰区新增关键路径 + 22项继承回归），覆盖阈值边界/非单调行为、eta0/高低阈值退化、网格/强单支对照、测试被试隔离、缓存恢复、事件重匹配差集、双评价器绑定与账本、负计数防护。测试使用合成响应/计数及本地真实评价器类，**不是官方数据上的新结果**。ZIP 的 CRC/文件哈希、cell/notebook 语法、隔离解压 CLI 导入会在交付前核验并记入 `VALIDATION.json`。

运行真实性的最后检查在 Colab 自动执行；本地没有 `/content/drive`，因此没有声称完成 SAMMLV/CASME3 官方实验，也未修改论文结果表。
