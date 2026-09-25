# 时间错位对照融合：首轮固定结构运行包

本包实现冻结的 `S=G+L−B`，仅替换候选评分，复用作者 ME-TST+ 完整解码和 recognition neutral 过滤。无需重训或重跑 backbone。实验入口只有 `run_shift.py`；`vendor` 是按字节复制的已核验依赖，里面的历史实验入口不运行。

本次交付完成软件测试和 SAM 真实接入验证。**没有执行正式 29 被试/94 被试全量筛查，没有新方法正式 F1 或优于基线的结论。** 两被试 integration fixture 是输出链测试，不能当作性能结果。CAS 原始响应不在本地，本包不使用旧 858-GT 缓存代替它。

## 固定协议

- `A={1,1.5,2,2.5,3,4}, a0=2, rho=3`；物理宽度去重，missing=0，距离优先、同距局部支持高者优先、全平局原顺序。结构来自 SAM 开发记录，不是独立确认。
- `w=max(1,round(rho*k)), delta=max(1,round(.5*k)), q=w+delta+ceil(max_width/2)`。每个候选固定自己的 L0，在所有整数 `q<=t<n-q` 且 `abs(t-p)>2q` 上共同查询非参考尺度，然后取 median 的均值 B。没有抽样、循环移位或新学习权重；按时间块精确计算。
- 单物理尺度/空对照：`B=L,D=0,S=G`，输出明确退化原因；无候选输出空集。B 不是背景标签、误报概率或 p-value；对照时刻数不是独立样本量。
- 七评分：Full、G、L、D=L−B、G+L、G+L−当前视频候选B均值、未裁剪 G+L−L0。
- 每方法阈值 `j/100,j=-100,...,200`，301 个，共 2,107 配置。`>=`，不裁剪；tau=2 仍保留恰好得分 2 的候选，没有偷偷增加全拒绝配置。相同被试的完整候选掩码相同才复用解码计数，实际唯一集合数另存。
- 外层留一被试；排除该被试后，以其余被试 pooled full F1、precision、较少 FP、较小配置索引依次选阈值。汇总 TP/FP/FN 计算 F1，不平均被试 F1。所有特征无标签，不做上游嵌套重训。

三类输出必须分开解释：

| 文件 | 含义 |
|---|---|
| `fixed_structure_baselines.csv` | 七方法在首轮单点结构内各自选阈值；G/L/D 不是扩大结构域后的独立强基线 |
| `ablations_retuned.csv` | 主要消融：固定 Full 结构，对 D、G、G+L 分别重选阈值；本轮与对应基线数值重合属正常 |
| `ablations_locked_diagnostic.csv` | 完全继承 Full 结构和阈值，仅作决策诊断 |

去全局是 D，仍保留参考候选生成；去局部同时移除 L 和 B，得到 G；去错位校正得到 G+L。`locked_range_diagnostics.csv` 记录 tau>1 导致 G/D 全拒绝的范围问题，不把全零作为主要模块贡献证据。

## Colab 接续

上传 `shift_control_fusion_ready.zip`，使用 `Colab_时间错位对照融合_首轮接续.ipynb`，或把 `COLAB_CELLS.py` 的四段代码依次放到现有笔记本末尾。复用已挂载 Drive；只在依赖缺失或不兼容时安装。CPU 即可，不运行旧训练/搜索 cells。

第二格设置 `DATASET` 和 `DUMP`，默认：

| 数据集 | 响应目录 | 严格输入 |
|---|---|---|
| SAM | `/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1` | 29 被试、79 视频、159 GT、k=5、frame_skip=7 |
| CAS | `/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/CASME_3_method1_strategy1` | 94 被试、462 视频、853 GT、k=17、frame_skip=1 |

两者 method_type=1、strategy=1，recognition 是与响应等长的一维类别 0..4，neutral=4。输入是直接含 `subject_*.pkl` 的原始可信响应目录。不要用最终 G/L 表或旧缓存替代原始响应。

SAM 两条样本的 apex=0 是已有缺失值标记，保留并审计；不改 onset/offset。CAS 仅精确放行 216/e、`[[462,470,492]]`、`['others','surprise']` 的已记录多余情绪标签，保留作者索引；其他错配停止。这不表示目标情绪类别已核实。

输出为 Drive 上独立的 `GLSD_SHIFT_CONTROL_FIXED_V1/{sammlv|casme3}/{input_audit|probe|screen}`。SAM 完整结果复核后，再切 CAS；不根据少量 probe 选择公式。CAS 沿用同一结构和网格，在 CAS 的训练被试内独立选阈值，不宣称未接触测试，也不把它写成冻结 SAM 阈值的跨库迁移。

第三格支持重跑接续：输入、代码、运行环境及协议身份必须完全一致。每完成一个被试原子写入完整七方法计数；中途打断只需重做当前被试。选参/事件导出中断后会重新导出，复用已完成搜索。若升级代码或 Colab 环境变化，使用新的 RUN_ROOT，不删除身份检查或强行拼接旧检查点。

## 本地命令

```bash
python test_shift.py
python run_shift.py --dataset sammlv --dump /path/to/SAMMLV_method1_strategy1 --output /path/to/shift_probe --mode probe --probe-grid
python run_shift.py --dataset sammlv --dump /path/to/SAMMLV_method1_strategy1 --output /path/to/shift_screen --mode screen
# 同一运行续跑时加 --resume；CAS 改 --dataset casme3 和对应 dump。
python verify_results.py /path/to/shift_screen
```

输出必须与代码目录和输入目录分离。不要同时启动两个进程写同一个输出目录。首次 probe 含七方法×三个阈值×两个真实被试；`--probe-grid` 另走一个被试的全 2,107 配置并验证缓存回放，不选公式、不输出方法排名。正式 screen 自动重复必要接入检查后再运行全量。`--mode audit` 只核对输入和两种评价接口，不能代替真实 probe。

## 输出和回传

`screen.zip` 自动保存于输出目录旁，异常时也尽量打包已有检查点。回传这个 ZIP，不必回传原始响应。正常完成须有 `completion.json` 和 `result_audit.json`；仅有 probe PASS 不代表正式筛查完成。

- `features.jsonl.gz`：逐候选 G/L/L0/B/D/S、对照数、退化原因、物理宽度、a0/rho 和响应身份。
- `search_counts.npz`：七方法×301阈值×被试的 raw/full TP/FP/FN、完成标志和完整性摘要。
- `selected_configs.csv`：各外层训练选择、训练指标、评分方法及消融口径。
- `event_records.jsonl.gz`：选中配置重新完整解码的预测、recognition、GT 身份、raw/full 贡献及保留掩码。
- `event_differences.jsonl.gz`：Full 相对每条控制的新增/丢失 GT、候选、解码事件及 FP 变化。
- `run_manifest.json`、`input_summary.json`、`matching_protocol.json`：输入/代码/环境、标注例外、numeric/list 与 string/scalar 接口身份。

独立审计复算训练侧选择和 tie-break，核对事件/检查点/CSV 守恒、阈值掩码、GT 增减与 pooled F1。保留旧强 L、可靠局部分支和残差的独立研究地位；本轮没有兼容的旧计数可直接替换新分数网格，不重跑旧探索。先看 Full 是否超过重选阈值的 G+L、视频常数B和未裁剪残差；未超过则分析机制，不自动加新参数。本轮不能单凭固定结构胜出宣称达到强基线或 SOTA 验收。
