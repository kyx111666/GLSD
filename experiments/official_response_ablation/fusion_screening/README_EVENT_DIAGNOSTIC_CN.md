# 下一步：已选配置的 G/L 事件互补性诊断

目的：用真实事件 ID 区分 G/L 各自独有的正确检测，以及 S_beta 新增、丢失的真事件和误报。
来源：用户的 `oto_structure_20260922T071507_272951Z.zip`。包内已包含原范围/扩展范围的已选配置、原预测和 raw/full 计数；不需要再次上传结果 ZIP。

## 在 Colab 操作

继续使用已配置环境的 ME-TST+ notebook，确保 Drive 已挂载。新建一个 cell，复制 `EVENT_DIAGNOSTIC_COLAB_CELL.py` 全文，运行后只上传 `metst_fusion_event_diagnostic.zip`。
文件名带 `(1)` 等后缀不影响识别。该 cell 会上传、独立目录解压、子进程运行并打包结果；不需要运行旧搜索 cell。
输出在 `MyDrive/GLSD_ONE_TO_ONE_FULL_TUNING/oto_events_<UTC>/`，及其旁边的同名 ZIP 和日志。
运行完把 `oto_events_*.zip` 发回。只回放 2 个范围 × 5 方法 × 29 被试 = 290 个已选配置，远少于上一轮全网格；通常应为分钟级，具体由环境决定。

## 保持与核验

- 不搜索、不重选参数，不训练模型；使用已冻结响应和本轮一对一评价协议。
- 检查当前 sealed runner/core 的 SHA256 与结构扩展结果一致。
- 每个回放配置必须复现原预测及 raw/full TP/FP/FN。
- 从评价器公开 add 调用捕获实际使用的 GT 和预测，保持预测顺序。
- 逐事件沿用 decoder 返回的匹配、识别标签和原 full synergy；逐事件计数必须还原原 raw/full 总计数。
- 对候选按响应哈希验证视频对应关系；真实事件 ID 为 subject/video_index/gt_index。
- 不给跨方法共享预测强行复用 TP 标签；匹配结果可能随其他预测变化。

## 输出

- `complementarity_summary.csv`：分别报告原范围、扩展范围的 G 对 L，以及 S_beta 对四个对照；另有每个方法扩展前后的变化。`added_gt`/`lost_gt` 是按真实事件 ID 计算；G 对 L 时即 G 独有/L 独有的正确检测。
- `event_differences.json`：逐被试新增、丢失事件的 ID。
- `selected_event_scores.csv`：每个已预测事件的 G/L、分数、阈值、局部半径、raw/full TP/FP 和识别标签。
- `event_records.jsonl.gz`：完整候选、匹配和 GT 明细，支持检查漏检事件附近是否有候选、是否被融合拒绝。
- `replay_checks.csv`：290 个回放配置的预测/计数核验。
- `manifest.json`、`matching_protocol.json`、`completion.json`：数据来源、代码指纹、评价协议与完成状态。

G/L 各自使用之前选中的配置；这是实际 pipeline 的互补性诊断，不能把所有差异都归因于融合公式。不同 rho 的 L 分数要分开解释。
G 和 L 的已检测 GT 并集只能反映潜在覆盖，不是可部署的 oracle 融合 F1；实际合并预测还会带来误报、冲突和重新匹配。
本包不自动选新方法、不修改论文、不启动 CASME3。先根据真实互补性决定后续实验。

本地已通过实际评价器上的合成事件捕获、识别过滤、异常恢复和互补统计测试；真实 Colab 回放仍待运行。
