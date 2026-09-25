# ICASSP 2027 十六天冲刺安排

官方完整论文截止：2026-09-16。技术内容限 4 页，第 5 页仅用于参考文献等。

| 日期 | 必须完成 | Go/No-Go |
|---|---|---|
| 9/1 | 基线失败审计、候选 PoC、确定新颖性风险 | 已完成初版 |
| 9/2 | 老师确认 Skill 是 Agent Skill、作者关系、mirror cache 可用性；冻结 SkillME 问题定义 | 含义不一致则立即改题 |
| 9/3 | 实现 SkillME 的 Inspect/Route/Decode/Verify 最小闭环与正式 evaluator | 必须先复现 native 数值 |
| 9/4 | 运行 EquiScale、SkillME 和信号处理消融 | 任一数据集低于 baseline 则停 |
| 9/5 | subject-level paired bootstrap、逐被试胜负、失败案例 | 无稳定证据则收缩算法主张 |
| 9/6 | 建立 20–30 个 Skill 任务变体；运行 no-skill、README、SkillME 对照 | 没有重复试验则不能主张 Skill 可靠性 |
| 9/7 | Skill消融、参数敏感性、运行时/token成本；检查协议和数据泄漏 | 口径不一致则禁止写结论 |
| 9/8 | Skill与方法图表定稿；固定所有数字和Skill版本 | 此后不再改算法与Skill |
| 9/9–9/11 | 完成 4 页初稿：问题、方法、实验、限制 | 9/11 必须有完整 PDF |
| 9/12 | 内部审稿：新颖性、统计、可复现、表达 | 汇总阻断项 |
| 9/13–9/14 | 修稿、压页、补引用、伦理与贡献说明 | 不新增实验分支 |
| 9/15 | 模板、匿名、引用、图中文字、PDF eXpress/提交系统检查 | 上传可提交版本 |
| 9/16 | 提交，至少提前 8 小时完成上传 | 不押截止前最后一分钟 |

## Skill 定位

SkillME 从第一天起就是方法的程序控制层，而不是最后的包装。其输入为冻结 score/logits cache 或原始视频任务，输出为事件 JSON、评测报告和可审计执行轨迹。EquiScale 是 Skill 内的确定性工具；必须用 script-only 与 SkillME 对照分离两者贡献。
