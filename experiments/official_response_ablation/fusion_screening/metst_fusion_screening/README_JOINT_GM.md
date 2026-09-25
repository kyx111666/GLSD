# 广义均值联合选参：Colab 增量接续

状态：代码与本地验证包；真实 Colab probe/full 尚未运行。范围仅 ME-TST+ / SAMMLV、CAS(ME)3，读取已有冻结响应，不训练、不重新生成响应。

## 直接运行

下载 `metst_joint_gm.zip`，将 `Colab_GM_联合选参接续.ipynb` 的四个代码单元格复制到现有已配置的 Colab 笔记本末尾：

1. 上传本轮 ZIP。不要使用旧 p8 运行包的上传单元格。
2. SAMMLV probe：出现 `JOINT_GM_PROBE = PASS` 后继续。
3. SAMMLV full：出现 `JOINT_GM_FULL = PASS` 后继续。
4. CAS(ME)3 full：内部先 probe 再搜索。

每个 full 自动做一次当前新路径的 probe。运行包采用独立 `/content/glsd_joint_gm`，输出根目录 `/content/drive/MyDrive/GLSD_JOINT_GM/`，保留所有旧结果。每阶段完成自动下载 ZIP。

默认旧结果路径已填写：

- `/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE/full_20260922T053538_684804Z`
- `/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_sammlv_20260922T070420_025128Z`
- `/content/drive/MyDrive/GLSD_P8_THRESHOLD_CONTROL/full_casme3_20260922T071214_000063Z`

如实际位置不同，只改上传单元格里的 `JOINT_OLD` / `JOINT_THRESHOLD_PATHS`。目录应直接包含 run_manifest.json 和 completion.json，不能指向 probe。运行时检查代码、输入清单/哈希、运行时版本、协议和历史复用链；不兼容则停止，不静默混用。

## 搜索定义

p={1,2,4,8,inf}，a0={1,1.5,2}，rho={0.5,1,2}，tau={0.05,0.10,...,0.95}。保持当前 G/L 特征、alignment、尺度集合、median、missing=0 不变。

- GM_joint 共855配置，按上述 p 顺序排列；每个 p 内按 a0/rho/tau 升序。
- p=1 的 Mean 与 p=8 全网格各171组复用，共342组。
- 新增 p=2/4/inf 各171组，共513组。第一阶段结果不混入本轮：已提供且兼容的来源为修正协议后的第二阶段及阈值对照结果。
- G_union 独立选参，阈值为 T 与各 p 的 2^(1/p)T 并集；p=inf对应T。不截断超过1的阈值，按实际浮点阈值精确去重，共258组。
- G 原网格57组与 p8映射57组已完成；本轮只补144组G阈值。
- 每个数据集总新增657配置；SAMMLV 29人、CAS94人。实际耗时取决于Colab和解码，不按之前57组搜索的耗时估算本轮。

G不重复搜索rho。每个p的重标度G（各57组）从上述计数直接构建，分别独立选择，额外不解码整网格。各方法预算不同，不能称等预算。

## 选择、消融与事件导出

每折排除outer被试，汇总其余人的full计数，联合选p/a0/rho/tau。平局依次F1高、precision高、FP少、固定配置索引小。完整方法主结果为联合选择的 GM_joint，不是看完outer汇总后挑固定p。

G_union阈值顺序为先原T，再p=1/2/4/8映射，保留首次出现的精确阈值。不能事后根据成绩改平局策略。

对照包括 Native、G、L、Mean、上一轮P8_rho1、每个p独立重标度G和G_union。GM_no_L / GM_no_G锁定每折完整GM_joint的p/a0/rho/tau，不重新选参。去L为2^(-1/p)G，去G同理；p=inf时为原分支分数。使用映射形式比较避免单独缩放引入边界差异。

沿用已跑通的双评价器事件观察接口：从实际add调用捕获预测/GT；只在context传入的汇总类安装已有一对一适配器，spotting自身视频评价器匹配行为不变。不是新增评价协议。候选、识别、匹配索引和GT ID均沿用官方调用链。

已有事件记录只有在评分规则、结构参数、保留候选完全等价且来源身份验证通过时复用，并重算显示分数。新选中配置回放官方解码；计数必须与搜索张量一致。事件差异包含新增/丢失GT、候选差异及计数差。

## 恢复

只恢复本轮联合搜索目录，不填旧p8/probe目录。mode、setting和代码必须一致：

```python
JOINT_MODE = 'full'
JOINT_SETTING = 'sammlv'  # CAS 用 casme3
JOINT_RESUME = '/content/drive/MyDrive/GLSD_JOINT_GM/full_sammlv_实际时间'
entry = joint_dir / 'run_colab_joint_gm.py'
exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'))
```

新搜索每个p/被试保存checkpoint；中断的当前被试重算，已完成被试不重算。事件回放逐被试逐方法原子保存，恢复不重复已完成记录。成功的当前run probe有记录后不重复。开始下一数据集时JOINT_RESUME=None。

## 查看结果

数据集目录输出：

- summary_full.csv：所有方法的full计数/F1。
- selected_configs.csv：每折p/a0/rho/tau，独立选参和锁参消融均列出。
- paired_comparisons.csv：联合方法相对各对照的F1差及条件bootstrap区间。
- decision.json：独立对照/锁参消融是否均超过、GT增减和FP净变化；仅描述点估计，不是显著性或独立确认结论。
- event_records.jsonl.gz / event_differences.jsonl.gz：完整事件记录及比较。
- search_counts_*.npz：本轮全部网格；new_counts_*.npz：实际新增搜索断点。
- selected_events/：事件回放checkpoint。
- protocol.json、probe.json、probe_event_records.jsonl.gz；根目录身份与完成记录。

主判断：GM_joint能否超过强独立G/L对照，并且相对GM_no_L/GM_no_G都有正向净收益；同时核对具体GT事件。若仍只有L误报增加，不继续无界扩大网格，下一步考虑L的使用机制。

本轮是看过既有结果后的开发实验；bootstrap不重新选参，不校正多轮探索。不能把这轮当作独立确认或保证SOTA。不要将p=1/8旧计数复算称为新实验。

## 本地验证记录（2026-09-22）

- PASS：855组联合网格、258组G阈值并集、513+144组新增配置数量核对。
- PASS：各p的评分、零分支消融、映射G及大于1的阈值语义核对。
- PASS：增量组装只搜索缺失网格，p=1/8沿用旧计数；每个p的重标度G从并集索引准确取数。
- PASS：outer计数扰动不改变该折inner赢家，平局顺序固定。
- PASS：双评价器真实类上的新p分支与事件导出合成测试通过；原add绑定恢复。
- PASS：搜索中断恢复不重算已完成被试，p变更拒绝复用checkpoint；事件恢复不重复已保存回放。
- PASS：用户两套实际结果包的manifest、旧代码哈希、张量和事件缓存验证；145/470条历史选中记录与新复用入口输出一致。
- PASS：本地8项新增测试与22项继承测试通过；打包哈希、代码单元格语法检查通过。

这里的真实包检查只验证已有保存产物，没有执行新p的真实数据评价。Colab中的新probe/full仍未运行。
