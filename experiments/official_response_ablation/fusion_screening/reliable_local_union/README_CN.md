# 可靠局部证据并联融合：运行与接续说明

这是交接设计的实现包，协议 `reliable_local_union_v1`。仅用于 ME-TST+ / SAMMLV 与 CAS(ME)3 冻结响应的后处理开发实验。沿用已定公式和搜索空间，不训练、不生成响应、不扩展 BoostingVRME。**完整新方法实验尚未运行，没有新主结果 F1；本地实际验证见 VALIDATION.md。**

## 在现有 Colab 后面接续

打开 `Colab_可靠局部证据并联融合_接续.ipynb`，把四个代码格按顺序复制到当前已配置的 Colab 末尾；也可从 `RELIABLE_LOCAL_UNION_COLAB_CELLS.py` 的四个 `# %%` 段复制。只需要传输本轮新运行包 `metst_reliable_local_union.zip`，已有官方响应、封存证据和旧 ETA 结果继续使用 Drive 原文件。

1. **Cell 1**：安装到 `/content/glsd_reliable_local_union`，验证包内每个文件的 SHA-256。若 `/content/metst_reliable_local_union.zip` 已存在则直接使用，否则选择本轮新 ZIP。不会覆盖 `/content/glsd_eta_gm`。
2. **Cell 2**：SAM 输入/运行时/代码检查、全部结构的无标签特征检查和固定配置真实解码 probe。末尾应有 `RELIABLE_LOCAL_UNION_PROBE = PASS`。小样本计数只用于接口核验，不用来挑方法、阈值或修改 R。
3. **Cell 3**：SAM 完整开发实验。内部先做同样的 probe，再补算缺失配置、逐折内层选参和外层事件导出。全流程通过才写 `mode=full` 的 `completion.json`。
4. **Cell 4**：CAS 使用同一公式与协议，内部先执行 CAS 自己的 probe。SAM 不达标也不扩网格、不看外层改 R；CAS 是预定开发评价，不能称独立确认。

默认旧结果目录（可在 Cell 1 的 `RLU_REUSE_PATHS` 修正为现有目录）：

```text
/content/drive/MyDrive/GLSD_ETA_GM/full_sammlv_20260922T110018_905231Z
/content/drive/MyDrive/GLSD_ETA_GM/full_casme3_20260922T111208_876534Z
```

结果只写入 `/content/drive/MyDrive/GLSD_RELIABLE_LOCAL_UNION/` 的独立 `probe_*`、`full_*` 目录。日志逐行写入该根目录的 `log_*.txt`，终端打印 CSV/JSON 文本，不依赖 DataFrame 展示组件。CPU 即可，沿用已有 Python 环境，不在 cells 中升级依赖。

## 输入与评价兼容性门槛

运行前必须具备原 `/content/ME-TST` 环境、Drive 上原封存 runner/core/证据清单，以及两数据集相应的官方 `subject_*.pkl`。路径沿用原 helper 的 SPECS。CAS 不允许用其他近似缓存顶替。

生产入口逐项检查：

- 旧 ETA 为该数据集已完成的 full，协议及源码文件清单完全一致；21 个继承 Python 文件直接取自旧 `metst_eta_gm.zip`，保持原字节。
- 输入路径清单和全部文件 SHA-256 与旧 ETA 完全一致；运行时 Python/NumPy/pandas 与旧清单完全一致。已知旧运行时为 Python 3.13.15、NumPy 2.1.3、pandas 2.2.3，仍以清单中的完整版本字符串为准。
- 保留原 `metst_context` 的 native 兼容性 gate，不重新搜索旧方法。聚合 `_str` 评价器只安装旧 numeric-list 一对一适配器；视频评价器保留原 string-scalar 接口。两类公共 `add` 都由原观察器捕获并恢复，不假设 `gt_all`。
- 增加整个作者评价依赖与 `training_utils.py` 的固定参考哈希，记录两个评价器签名、源码与 SciPy/scikit-learn 版本。参考来自已在本地核验的原作者运行时；不自动替换 Colab 文件。
- 旧计数张量必须配置清单、被试顺序、完成标志、整数形状、非负性、TP+FN 守恒全部通过。probe 回放旧 G/L/Mean 与 max 端点；完整运行中每条已选事件再次与张量核对。

如果失败，日志会指出具体身份差异。先恢复原环境/输入或纠正路径；不要删检查、改旧清单、冒用其他协议或重新运行旧网格。若原环境确实无法恢复，另行制定并版本化全体对照重评协议，本包不会自动执行它。本地跨运行时 probe 通过不等于生产运行时门槛已经通过。

## 公式与冻结网格

保持封存 `GLSDFeatures` 的候选顺序、G、逐尺度局部分数、missing=0、跨物理尺度 median、预测几何、解码及识别。A={1,1.5,2}，重复整数宽度只算一个物理尺度，固定 δ=max(1,round(0.5k))。在现有容差中匹配最近峰，同距离优先局部分数更高者。R 直接读这些尺度缓存；逐尺度 median 必须与封存 L 精确相等。

B 排除参考物理尺度。B 为空或任一非参考尺度缺失时 R=0；否则 R=exp[-mean(d/k)]，参考峰自身零距离不进平均。不额外要求匹配尺度局部分数过线。

```text
L_rel = R * L
Full = max(G, beta * L_rel)
keep = Full >= tau
```

β∈{0.5,1,2}，不裁剪分数。G 已过线的候选不会被 R 否决；局部分支可接受低 G 的可靠候选。评分保留集包含同参数 G，但最终匹配/识别的 TP 或 F1 不保证单调增加；该公式也不能删除 G 已接受的误报。

T={0.05,0.10,…,0.95}，U=T∪T/2∪2T，共39个不同有理值，最大1.9。阈值不裁剪。固定索引顺序为 a0、rho、beta、tau；U 先保留 T 原顺序，再追加升序的新值，以保留旧 L 阈值的同分优先顺序。尺度与 rho 均按 {1,1.5,2}、{0.5,1,2} 排序。没有外层结果驱动的网格改变。

|方法|独立配置数|复用旧计数|新计算配置|
|---|---:|---:|---:|
|Full|513|0|513|
|G|57|57|0|
|L（扩展 U）|351|171|180|
|L_rel（扩展 U）|351|0|351|
|NoR：max(G,βL)|513|171|342|
|Mean|171|171|0|
|L_old（原 T）|171|171|0|
|5 种 G_rescaled_p 控制|各57|全部|0|
|G_union|258|258|0|

总计2670个配置行，1284行复用、1386行新算。NoR 的 β=1 与旧 ETA 中 η=.5、p=∞ 的 max 完全相同，预测只消费有序峰而不消费分数，所以复用该171配置；有真实端点回放检查。原 L 旧171结果另列 `L_old`，绝不覆盖旧目录。

新搜索共1386×29=40,194（SAM）或1386×94=130,284（CAS）个配置-被试解码。除此之外还有输入 gate、probe 和每被试16种方法的已选事件回放（SAM464、CAS1504条）。不进行动态候选集合去重，避免未经证明的下游等价复用。搜索预算不匹配，实际配置/复用/新计数覆盖数导出至 `search_budget.csv`。这些数字是运行协议的工作量，不是“已完成”的实验记录。

## 选参与消融

每个外层被试均被排除；在其余被试上汇总 full TP/FP/FN，依次按 F1 高、precision 高、FP 少、固定配置索引选参。外层最后汇总计数计算 F1，绝不平均被试 F1。Full、G、L、L_rel、NoR、Mean 及强 G 各自独立选参。

锁定 Full 每折参数后另导出：

- `Full_no_L`：S=G，保留 a0、tau，rho/beta不影响评分。
- `Full_no_G`：S=βL_rel；只去 G **评分分支**，保留参考尺度候选生成。
- `Full_no_R`：S=max(G,βL)，保留全部有效参数。

独立调参 NoR 与锁参 Full_no_R 分开报告。Full 必须与独立 L_rel 比较。事件贡献按 GT ID 和完整预测区间比对，明确新增/丢失 GT、增加/移除 FP，不把相同 TP 总数当相同命中。

## 输出与验收

每个运行根目录包含 `run_manifest.json`、`matching_protocol.json`、`completion.json`。只有 `completed=true, mode=full` 才代表完整新实验完成。数据集子目录包括：

- `protocol.json` / `compatibility.json`：公式、网格、选参和输入兼容性。
- `feature_audit.csv/json`：全部视频结构的候选数、缺失支持与 B 为空比例。
- `probe.json` / `probe_event_records.jsonl.gz`：固定 probe 及真实 raw/full 事件核验。
- `new_counts_*.npz`：仅新配置、逐被试原子 checkpoint；`search_counts_*.npz`：完整独立搜索张量。
- `summary_full.csv`、`selected_configs.csv`、`per_subject_counts.csv`、`search_budget.csv`：汇总、每折参数与预算。NPZ 的 configs/subjects/raw/full 足以独立重建每折内层赢家。
- `selected_events/*.json.gz` / `event_records.jsonl.gz`：预测区间、识别标签、raw/full TP/FP、matched_gt_id、GT账本及完整候选 trace。
- `candidate_records.jsonl.gz`：每个参考候选一行，含 G/L/R/L_rel、评分、实际保留状态、G/可靠L/both 接受来源、尺度宽度、逐尺度匹配峰、距离、k、预测区间及 raw/full 标签。距离-1表示缺失；未解码候选标签为 null。分支来源描述相同参数下的 Full 通路，实际方法的 retained 单独记录，避免把单支诊断误读为 Full 决策。
- `event_differences.jsonl.gz`：Full 对各方法的 added/lost GT IDs 和 added/removed FP 事件。
- `selected_degeneracy.csv`：各方法已选结构的退化比例。
- `paired_comparisons.csv`：种子100、10000次被试配对 bootstrap；条件于已选预测，不校正重复开发或重新调参。
- `decision.json`：所有独立强单支、三类锁参消融及事件代价。点估计/区间需共同审阅，文件不会宣称 SOTA。

进入 Table 2 仍需两数据集完整评价、Full 超过独立强单支（尤其 L_rel）、锁参模块有净贡献且不只是阈值覆盖的微弱差异。本轮没有文献查新，不声称创新性已确认；冻结响应的选择也不能证明上游训练隔离。

## 断点续跑与下载

中断后重跑同一 cell：若 Drive 下只有一个同数据集同模式目录，会自动 `--resume`；若多个，设置 `RLU_RESUME` 为确切目录。成功目录会打印 Already completed，不重复新搜索。运行中不要修改包；代码、输入、运行时、评价链、旧计数或 mode 变化均拒绝续跑。probe 与 full 使用不同目录。

搜索 checkpoint 以被试为单位，失败时最多重做当前方法的未完成被试；已选事件按方法-被试单独保存。只在全流程通过后写 completion。失败日志保存在 Drive，不把半成品当结果。

下载失败只运行以下下载代码，或从 Drive 下载目录；不要删除完成目录：

```python
from google.colab import files
files.download(archive)  # 当前 cell 已打印 RESULT_ZIP；也可填该路径
```

CLI 等价入口（在原 `/content/ME-TST` 环境执行）：

```bash
python /content/glsd_reliable_local_union/colab_reliable_entry.py \
  --mode probe --setting sammlv \
  --reuse /content/drive/MyDrive/GLSD_ETA_GM/full_sammlv_20260922T110018_905231Z \
  --output /content/drive/MyDrive/GLSD_RELIABLE_LOCAL_UNION/probe_sammlv_manual
```

完整运行将 `--mode` 改为 `full` 并使用新的 full 输出目录；恢复既有运行加 `--resume`。`--mode check` 仅核对兼容性（仍沿用 native gate），不执行新 probe 或搜索。

## 本地维护验证

`test_reliable_local_union.py` 检查公式、去重尺度、匹配打平、缺失处理、等效阈值、无剪裁、继承计数复用、外层排除、事件ID、锁参与续跑。`validate_local.py` 接收既有 `server_standalone` 素材目录、两份旧 ETA ZIP 和 SAM dump ZIP，执行本地真实响应 probe，不执行 full。

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python validate_local.py --assets /path/to/existing/server_standalone \
  --sam-results /path/to/sam_eta.zip --cas-results /path/to/cas_eta.zip \
  --sam-dump /path/to/SAMMLV_dump.zip --output /path/to/new_validation_directory
python build_package.py
```

旧评价器合成测试需要原 ME-TST 运行时在 PYTHONPATH，真实 probe 需要 numpy/pandas/scipy/scikit-learn。构建不会修改继承源码；包不包含官方响应或旧实验结果。
