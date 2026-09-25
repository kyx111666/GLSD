# 下一轮：对齐容差 eta 的公平联合选参

状态：运行包已实现；真实 Colab probe/full 尚未运行。目标是检验结构参数能否使完整融合超过强独立模块，同时在锁参消融中体现两个模块贡献。本轮仍仅 ME-TST+ 两套冻结响应；Table 2 SOTA 尚需最终方法及可比实验协议验证，不由本包自动宣告。

## 当前 notebook 怎么接

下载 `metst_eta_gm.zip`；将 `Colab_ETA_对齐容差接续.ipynb` 四个代码单元格复制到当前已配置好的 Colab 末尾，按顺序运行：

1. 上传本轮新 ZIP。必须使用本轮 Cell 1，旧上传格包含旧包哈希。
2. SAMMLV probe。它会在全部29被试的响应上检查全部 a0/rho 的原 eta=.5 一致性和三种 eta 的特征敏感性，并在首个被试实际解码旧配置及新评分/消融分支。输出 `ETA_GM_PROBE = PASS` 才继续；这是技术检查通过，不表示性能达标。
3. SAMMLV full。先自行执行相同检查，然后新搜索并导出完整结果。
4. CAS(ME)3 full。内部先做自己的特征检查和 probe，再搜索。

不用重启环境，不重跑旧网格或 backbone。首次full新建目录，ETA_RESUME=None。新根目录 `/content/drive/MyDrive/GLSD_ETA_GM/`，代码 `/content/glsd_eta_gm/`；所有旧结果只读。

Cell1已经填好上轮联合搜索目录：

- `/content/drive/MyDrive/GLSD_JOINT_GM/full_sammlv_20260922T091538_299118Z`
- `/content/drive/MyDrive/GLSD_JOINT_GM/full_casme3_20260922T095610_977764Z`

如实际位置不同，仅修改 `ETA_REUSE_PATHS`，指向包含completion.json的full根目录。Python/NumPy/pandas、封存输入、继承代码、评价器身份严格验证。不同运行环境会停止，不自动安装或升级。

## 这轮只改变什么

唯一新增结构参数：

`eta ∈ {0.5, 0.25, 0.75}`，`tolerance=max(1,round(eta*k))`。

使用Python round，不改取整方式。eta=.5先排列，以便完全平局时优先原定义。SAMMLV k=5时三者按以上顺序得到2/1/4；实际每个数据集/被试的k与有效容差写入诊断。

通过 AST 定位封存 evidence 函数中的唯一 tolerance 赋值，确认原表达式确为 `max(1,round(0.5*self.k))` 后，仅把系数替换为本轮eta变量。若源码结构不同则报错，绝不凭记忆重写G/L。每个访问的a0/rho均对比eta=.5与原函数的peak/G/L，要求逐元素完全一致。

尺度集合、参考尺度定义、局部窗口、median、missing=0、候选位置、全局G、官方几何、识别均保持原定义；只允许L因跨尺度匹配容差变化而变化。真实变化在alignment_sensitivity中量化。

## 搜索及对照

- p={1,2,4,8,inf}，a0={1,1.5,2}，rho={0.5,1,2}，tau={.05,.10,...,.95}。
- 完整GM_eta：3 eta × 855 = 2565配置，所有参数由每折其余被试的汇总inner full F1共同选择。
- L_eta：3 eta ×171 = 513配置，同步独立调eta/a0/rho/tau，不能只给完整方法新参数。
- Mean_eta：3 eta ×171 =513配置，从p=1融合张量直接取得，不重复解码。
- 原GM_joint、原G/L/Mean、P8_rho1、各p重标度G、G_union全部保留，旧张量直接复用。
- GM_eta_no_L/GM_eta_no_G锁定完整方法每折所选eta/p/a0/rho/tau，无独立选参；导出GT新增/丢失、FP变化。

eta=.5的855融合和171 L计数已有，直接复用。新增eta=.25/.75各855融合+171 L，共2052配置/数据集；Mean复用新p=1计数；G不受eta影响，不重算G网格。各方法预算不同，不称为等预算。

平局顺序：F1高、precision高、FP少、固定配置索引小。新网格eta顺序为.5/.25/.75，随后p顺序1/2/4/8/inf，再a0/rho/tau升序。outer被试只用于评价，不用于挑eta或p。

## 评价与旧记录

沿用已经跑通的双评价器调用链：context传入汇总评价器使用既有一对一适配器，spotting内部视频评价器行为不变；事件观察同时捕获两个类的add输入，退出后恢复。未改变评价协议。

只复用eta=.5且原评分/结构/阈值等价的旧事件记录；新eta的选中配置实际回放，以记录准确的新L。所有事件记录必须重建raw/full计数，GT ID唯一，搜索计数与回放一致。任何不一致都会停止，不跳过、不截断负FN。

## 输出与判断

- `alignment_sensitivity.csv/json`：全候选的L改变/增加/减少数量、有效容差；无标签挑选。实例数按视频/a0/rho累计，不是独立事件数。eta改变不保证L单调增大。
- `summary_full.csv`、`paired_comparisons.csv`、`selected_configs.csv`、`per_subject_counts.csv`。
- `event_records.jsonl.gz`、`event_differences.jsonl.gz`，以及逐方法事件恢复文件。
- `search_counts_*.npz` 全网格；`new_counts_eta*_*.npz` 新搜索断点。
- `decision.json` 分别报告是否超过独立强对照、是否超过两项锁参消融、事件贡献及选中eta分布。只判断点估计，不自动声称显著/SOTA。

结果表用静态文本输出，避开上轮Colab dataframe reference报错。出现 `ETA_GM_FULL = PASS` 后，计算和Drive保存已完成；随后的ZIP下载失败不需要重跑搜索。

本轮仍是看过历史outer后的开发实验。置信区间条件于选中预测，不重新选参，不校正多轮探索。

## 中断恢复

仅恢复本轮失败目录，保持mode/setting/代码一致，不填旧联合目录或probe目录：

```python
ETA_MODE = 'full'
ETA_SETTING = 'sammlv'  # CAS 用 casme3
ETA_RESUME = '/content/drive/MyDrive/GLSD_ETA_GM/full_sammlv_实际时间'
entry = eta_dir / 'run_colab_eta_gm.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
```

每个eta/p（或L）按被试保存。已经完成的被试不重算；中断的当前被试重算。事件逐方法原子保存，恢复复用。开始下一数据集时ETA_RESUME=None。

## 暂不做的扩展

当前未修改尺度集合、median或missing策略；未启动Boosting实验。若eta几乎不改变分数，或full仍无法兼顾强基线与模块贡献，依据本轮实际事件记录选择下一机制，不能事后从表里挑最好eta作为独立验证。

## 本地交付检查（2026-09-22）

- PASS：继承的22项p8、8项联合搜索测试及7项新增eta测试，共37项通过。
- PASS：原容差表达式AST匹配，eta=.5逐元素复现；不识别的封存表达式拒绝运行。
- PASS：合成例中eta改变跨尺度支持和L，候选与G保持完全一致。
- PASS：GM_eta/L_eta/Mean_eta网格及2052个新增配置核对；Mean从p=1复用，eta=.5不重算。
- PASS：inner赢家不受该折outer计数扰动；eta改变拒绝旧配置checkpoint。
- PASS：双真实评价器类上的eta评分与事件追溯测试通过，原add接口恢复。
- PASS：完整导出/事件恢复测试通过，两个消融锁定eta/p/a0/rho/tau。
- PASS：用最新真实结果包核对继承源码哈希、全部复用张量，以及319条SAMMLV/1034条CAS历史选中记录；计数和GT账本一致。
- PASS：最终ZIP逐文件哈希校验、独立解压测试和四个notebook代码单元格语法检查。

以上是真实旧产物的复用验证及合成代码测试，不是新eta在真实响应上的性能结果。
