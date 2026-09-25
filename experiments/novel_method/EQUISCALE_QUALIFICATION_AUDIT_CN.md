# EquiScale 证据与资格审计

**审计日期：** 2026-09-03  
**审计对象：** 当前项目中名为 EquiScale 的 frozen inference 后处理候选  
**最终状态：** `NOT-QUALIFIED`

> 本结论判断的是“现有历史证据能否支持把 EquiScale 当作论文保底主方案”，不是断言该算法在重新实现和严格实验后一定无效。

## 1. 审计边界与判定标准

本次只审计既有资产，不修改 EquiScale，不重新选择参数，不设计新 inference idea，也不将其他方法的结果算作 EquiScale 结果。只有在历史实现和精确配置完整时，才允许做确定性的原配置复跑。

资格要求至少包括：

1. 可定位的原始实现和固定算法定义；
2. 可定位的逐视频或逐被试原始结果及配置；
3. 与同一 frozen output、同一 GT、同一 evaluator 得到的 native baseline 对齐；
4. 两个目标数据集上的关键指标没有相互冲突的结论；
5. 参数选择不依赖目标测试 GT，且存在合理的邻域稳定性证据；
6. 能区分 EquiScale 本身与 mirror TTA、普通多窗口投票等既有组件的贡献。

本次完成的显式核验项包括：

- [x] 全项目及历史 ZIP 按 `EquiScale/equi_scale/SkillME` 名称追溯；
- [x] EquiScale 现存规范逐项还原输入、处理和输出；
- [x] SAMMLV/CASME3 compact cache 元数据、mirror 字段和哈希核验；
- [x] Markdown 中声称的 TP/FP/FN、Spotting F1、Recognition F1、STRS 算术复核；
- [x] 论文 native baseline、当地复现 baseline 和 EquiScale 声称结果分开比较；
- [x] 旧 `grid_b2_kp3.py` 与 EquiScale 的算法差异及测试集选择风险核验；
- [x] 参数来源、稳定性、机制消融、跨 backbone 和 Skill 可执行性核验；
- [x] AGF/PSED/SCED 等结果与 EquiScale 证据隔离，未混用。

## 2. 名称与历史来源

| 项目 | 审计结果 |
|---|---|
| Name origin | 当前证据显示它是 2026-09-01 才出现的新标签/方案名；未在 6—7 月原始代码和备份 ZIP 中发现同名记录 |
| First file | `my_method/METHOD_AUDIT_20260901_CN.md` 与 `my_method/METHOD_CANDIDATES_CN.md`（当前文件时间并列最早） |
| First recoverable date | 2026-09-01 20:44:18 |
| Original implementation | **NOT FOUND** |
| Original result file | **NOT FOUND**；只找到后写入 Markdown 的汇总数字 |

项目中存在较早的概念前驱 `senior_original/me_tst_video/me-tst-video/grid_b2_kp3.py`（2026-06-27）。它搜索 3 个窗口的组合，默认在 `k_p=3,4,5,6,7,8` 中选取，并可调整阈值、最小尺度支持数和 NMS。它不是 EquiScale 的原实现，原因是：

- 旧脚本使用“三个被选窗口”，EquiScale 规范使用“五个固定比例窗口”；
- 旧脚本默认至少两尺度支持，EquiScale 要求至少 3/5；
- 旧脚本没有 EquiScale 规范中的 mirror 分支平均和置信度加权边界；
- 旧脚本直接利用全体 SAMMLV GT 对候选配置评分和排序；
- 未找到它声称应生成的 `kp3_grid_summary.json/csv`。

因此，合理表述只能是：**EquiScale 可能吸收了旧多窗口 B2 grid 的思想，但没有证据证明二者是同一历史实现，也没有证据说明名称和固定比例是在独立于测试结果的情况下预先确定的。**

## 3. 当前可恢复的算法定义

唯一完整的现存定义来自 `my_method/skills/skillme-inference/references/equiscale.md`，可还原为：

1. 输入原视频逐帧 spotting score `s[T]`、logits `z[T,C]`，以及水平镜像分支的 `s_m[T]`、`z_m[T,C]` 和基础半窗 `k`；
2. 分支平均：`s_bar=(s+s_m)/2`，`z_bar=(z+z_m)/2`；
3. 构造尺度集合 `K={2k/3, 4k/5, k, 5k/4, 3k/2}`，取正整数并去重；
4. 每个尺度做宽度 `2a` 的移动平均；
5. 使用 Moilanen 阈值 `mean + 0.55*(max-mean)` 找峰，峰间距离为 `a`；
6. 每条尺度曲线用 q05/q99 做校准；
7. 按时间距离不超过 `floor(k/2)` 聚类峰；
8. 只保留至少来自 3 个不同尺度的簇；
9. 对成员区间 `[peak-a, peak+a]` 做置信度加权，得到 onset/peak/offset；
10. 在最终区间内聚合平均 logits，按 Strategy 1 与 result synergy 得到 emotion；
11. 输出事件的 subject、video、onset、peak、offset、emotion、confidence 和尺度支持证据。

但该规范没有完全固定以下可影响结果的细节：峰簇采用传递闭包还是中心分配、冲突峰如何归属、加权边界的整数舍入、总权重为零的处理、同尺度重复成员处理，以及最终 confidence 的精确定义。由于没有找到原始 Python 实现，现在补齐这些规则会构成**新实现**，不能冒充历史 EquiScale 的确定性复现。

## 4. 是否严格 inference-only

| 检查项 | 结论 |
|---|---|
| Backbone retraining | 否 |
| Backbone 参数更新 | 否 |
| 新增可学习模块 | 否 |
| `train=True` / backward / optimizer | 不需要，且应禁止 |
| 能否直接使用 frozen compact output | 能；前提是同时含原分支和 mirror 分支 |
| 是否必须 frozen re-forward | 不必须；仅在缺少 mirror/logits 时才需重新 frozen forward |

所以，**按概念定义，EquiScale 是 inference-only 后处理**。这一点成立，但 inference-only 身份本身不等于有效性、创新性或已具备论文证据。

## 5. 可用 cache 与证据溯源

| 数据集 | 文件 | SHA-256 | 关键元数据 |
|---|---|---|---|
| SAMMLV | `caches/me_tst/sammlv_strategy1_outputs.pkl` | `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569` | 29 subjects，79 videos，159 GT，`k_p=5`，含 score/logits/mirror_score/mirror_logits |
| CASME3 | `caches/me_tst/casme3_strategy1_outputs.pkl` | `9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda` | 94 subjects，462 videos，858 GT，`k_p=17`，含 mirror 分支 |

这些 cache 足以支持未来重新实现后处理，但**不能证明 Markdown 中的 EquiScale 数字确实由这些精确文件产生**。目前未找到 EquiScale 的：

- 执行脚本或 commit；
- 参数 manifest；
- 输入文件哈希记录；
- 逐视频/逐被试预测；
- evaluator 输出；
- JSON/CSV/PKL/log 结果；
- bootstrap 或随机种子记录。

## 6. 已声称结果与算术复核

以下数字只出现在 `NEW_THREAD_HANDOFF_CN.md` 和 `METHOD_AUDIT_20260901_CN.md` 等二级 Markdown 汇总中，不能当作原始实验文件。

| 数据集 | TP | FP | FN | Precision | Recall | Spotting F1 | Recognition F1 | STRS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV EquiScale（声称） | 45 | 151 | 114 | 0.2296 | 0.2830 | 0.2535 | 0.9118 | 0.2312 |
| CASME3 EquiScale（声称） | 86 | 722 | 772 | 0.1064 | 0.1002 | 0.1032 | 0.5267 | 0.0544 |

算术核验通过：TP/FP/FN 可重算出表中 precision、recall 和 Spotting F1，Spotting F1 × Recognition F1 也约等于 STRS。**这只能证明汇总数字彼此自洽，不能证明实验实际运行过、配置正确或比较公平。**

## 7. Baseline 对齐审计

项目汇总 `reports/ME_TST_AGF_result_comparison.md` 同时列出了论文值和当地复现值：

| 数据集/来源 | GT | TP/FP/FN | Spotting F1 | Recognition F1 | STRS |
|---|---:|---|---:|---:|---:|
| SAMMLV 论文 native | 159 | 52/171/107 | 0.2723 | 0.6787 | 0.1848 |
| SAMMLV 当地复现 native | 159 | 55/173/104 | 0.2842 | 0.6980 | 0.1984 |
| SAMMLV EquiScale（声称） | 159 | 45/151/114 | 0.2535 | 0.9118 | 0.2312 |
| CASME3 论文 native | 858 | 76/733/782 | 0.0912 | 0.5338 | 0.0487 |
| CASME3 当地复现 native | **554** | 60/522/494 | 0.1056 | 0.5870 | 0.0620 |
| CASME3 EquiScale（声称） | 858 | 86/722/772 | 0.1032 | 0.5267 | 0.0544 |

关键问题：

1. EquiScale 汇总表采用的是论文 native STRS（SAMMLV 0.1848、CASME3 写作 0.0488），不是当地复现 native；
2. 当前无法证明论文值与 EquiScale 值来自同一 frozen output、同一输入 hash 和同一 evaluator；
3. CASME3 当地复现只覆盖 554 GT，而论文/EquiScale 声称覆盖 858 GT，不能直接比较；
4. CASME3 native 在不同文档中出现 0.0487/0.0488 的舍入或来源差异，虽小，但进一步说明比较记录未被一个原始 manifest 固化。

因此没有一组同时满足“同 frozen cache、同完整 GT、同 evaluator、可回溯输出”的 EquiScale-vs-native 配对证据。

## 8. 双数据集方向判断

即使暂时接受 Markdown 数字，并以论文 native 为参照，结论仍不是干净的 `POSITIVE`：

| 数据集 | Δ Spotting F1 | Δ Recognition F1 | Δ STRS | 数据集结论 |
|---|---:|---:|---:|---|
| SAMMLV | **-0.0188** | +0.2331 | +0.0464 | **MIXED**：联合分数升高，但关键 spotting 指标下降 |
| CASME3 | +0.0120 | **-0.0071** | +0.0057（按 0.0487） | **MIXED**：联合分数升高，但 recognition 指标下降 |

若以当地复现为参照：

- SAMMLV 仍是 MIXED：Spotting F1 0.2842→0.2535，Recognition F1 0.6980→0.9118；
- CASME3 表面上三个指标都更低，但 GT 554 与 858 不一致，故该对比无效，不能据此判负或判正。

所以，“PoC 正向”的来源只是 **STRS 单一乘积指标高于论文表格值**；它不等于所有关键指标正向，也没有经过严格配对验证。

## 9. 参数来源与测试集泄漏风险

| 参数/组件 | 当前值 | 可追溯来源 | 风险判断 |
|---|---|---|---|
| 基础 `k_p` | SAM 5；CAS 17 | cache/原方法元数据 | 推理时不读 GT，但为数据集相关参数 |
| 五个比例 | 2/3, 4/5, 1, 5/4, 3/2 | 2026-09-01 规范首次明确 | 选择历史未知；不能证明预注册或由开发集决定 |
| Moilanen 系数 | 0.55 | 原 ME-TST 阈值规则 | 低新增风险 |
| q05/q99 校准 | 固定分位点 | 早期 B2 `postprocess.py` 已存在 | 非 EquiScale 独有；选择依据未固化 |
| 支持阈值 | 3/5 | 2026-09-01 规范 | 合理多数票，但没有开发集/理论来源记录 |
| 聚类距离 | `floor(k/2)` | 2026-09-01 规范 | 来源未知，缺邻域稳定性 |
| mirror 平均 | 两分支均值 | 既有 B3/TTA 思路 | 非新贡献，需单独消融 |
| 加权边界 | confidence-weighted | 仅有文字规范 | 精确公式/舍入未完全固定 |

旧 `grid_b2_kp3.py` 明确用目标 SAMMLV 全体 GT 对多个 `k_p` 组合、支持阈值和后处理配置评分并排序。如果后来的五比例或 3/5 规则参考了这类结果，就存在 test-set researcher selection bias。当前无法证明发生了直接继承，也无法证明完全独立，因此应记为：**高风险、来源未证实，而不是已确认泄漏。**

## 10. 稳定性与邻域审计

未找到 EquiScale 对以下邻域的固定网格结果：

- 比例集合的小扰动；
- 2/5、3/5、4/5 支持阈值；
- 聚类距离变化；
- q05/q99 分位变化；
- 有/无 mirror；
- 不同 subject/video 子集或 paired bootstrap。

旧三窗口 grid 的输出文件也未找到，而且它不是 EquiScale。因此，现阶段既不能证明 EquiScale 只在一个点有效，也不能证明它形成稳定平台；正确状态是 **UNVERIFIED**。

## 11. 机制审计

二级汇总给出的 STRS 为：

| 方法 | SAMMLV | CASME3 |
|---|---:|---:|
| Native（论文值） | 0.1848 | 0.0488 |
| Mirror average + fixed decoder | 0.2285 | 0.0493 |
| EquiScale | 0.2312 | 0.0544 |

若暂时接受这些值：

- SAMMLV 总增益为 +0.0464，其中 mirror average + fixed decoder 已贡献 +0.0437；EquiScale 相对该行只再增加 +0.0027。也就是说，约 94% 的总 STRS 差值在进入尺度共识前已经出现；
- CASME3 中 mirror 行只比 native 高 +0.0005，而 EquiScale 再高 +0.0051；但缺少 scale-only、vote-only、boundary-only 原始消融，不能把增益可靠归因给某个机制。

对候选解释的判断：

| 假设 | 结论 |
|---|---|
| A. 仅生成更多候选/提高召回 | SAMMLV TP 52→45、recall 下降，不支持；CASME3 TP 76→86，局部支持 |
| B. 多尺度一致性过滤 FP | 可能，但没有同候选集前后记录，未验证 |
| C. 边界聚合提高 IoU | 可能，但没有边界误差或 IoU 分层结果，未验证 |
| D. q05/q99 稳定跨主体幅值 | 可能，但没有有/无校准消融，未验证 |
| E. mirror averaging 是主要来源 | 对 SAMMLV 的汇总数强烈支持；对 CASME3 不支持为主要来源 |

## 12. 创新性与审稿风险

按当前定义，EquiScale 更接近：

- horizontal-flip test-time averaging；
- 固定多窗口 decoder ensemble；
- 跨尺度峰值投票/一致性过滤；
- 置信度加权事件解码。

它不是自适应尺度学习，也没有引入新训练目标或新时序表征。审稿人很可能将其概括为“普通 TTA + 多尺度平滑 + majority vote + weighted averaging”。在没有清晰理论、新的一致性量、严格消融和跨 backbone 迁移证据时，独立方法创新性风险较高。

## 13. Skill 化资格

接口层面可以把统一的 `score/logits/mirror/k_p` 输入封装成后处理 Skill，但目前还不具备论文级 Skill 资格：

- 没有可执行且版本冻结的 EquiScale decoder；
- 算法细节仍有影响复现的歧义；
- 没有同参数跨 backbone 结果；
- 没有 BoostingVRME 上的 EquiScale 结果；
- 现有正向数字没有逐样本证据链；
- 某些 backbone 若没有 mirror/logits，需要重新 frozen forward，不能只依赖一条 score 曲线。

结论：**可作为待实现的实验接口草案，不可宣称为已验证、可迁移的 inference Skill。**

## 14. 为什么没有执行“最小历史复跑”

本任务允许的复跑必须是“精确历史实现 + 精确历史配置”的确定性重放。目前同时缺少：

1. 原始 EquiScale Python 实现；
2. 原始逐样本结果/manifest；
3. 对若干解码细节的完整定义；
4. 能证明汇总数字对应哪一版 cache/evaluator 的记录。

此外，当前交付中的原 evaluator 依赖源码也不完整，直接另写 evaluator 会引入协议漂移。此时现场补写 decoder/evaluator 再运行，得到的是“2026-09-03 新实现”的结果，不是历史 PoC 复现，违反本审计的限制。因此本次没有用猜测代码制造新的 EquiScale 证据。

## 15. 发现的不一致项

1. `METHOD_CANDIDATES_CN.md` 将候选标为 `PENDING`，而后续 handoff 使用“正向 PoC”措辞；两者资格状态不一致。
2. EquiScale 数字仅在 Markdown 中重复，没有对应原始结果文件，却被用于方法排序。
3. EquiScale 表格引用论文 baseline，而不是同 cache 的当地复现 baseline，配对关系未建立。
4. CASME3 当地复现只含 554 GT，EquiScale 声称含 858 GT，不能互相作为 baseline。
5. CASME3 native STRS 在文档中出现 0.0487 与 0.0488 两种写法。
6. SAMMLV 的 EquiScale STRS 上升伴随 Spotting F1 下降；CASME3 的 STRS 上升伴随 Recognition F1 下降，不能概括为关键指标全面正向。
7. EquiScale 的镜像平均和多窗口思想已有前驱，而增量贡献没有独立原始消融。
8. 旧三窗口 grid 使用目标 GT 排序配置，但 EquiScale 固定参数的选择历史缺失，无法排除研究者选择偏差。

## 16. 最终判定

**NOT-QUALIFIED：未找到可信的 EquiScale 正向证据，不应该继续把它作为论文保底主方案。**

判定原因不是“STRS 没有数字提升”，而是：原始实现与原始结果缺失、baseline 未严格配对、双数据集关键指标均为 MIXED、参数来源与稳定性未证实、主要增益可能来自既有 mirror TTA，且尚无跨 backbone 的 Skill 迁移证据。

若未来找回原始 EquiScale 脚本、逐视频输出、运行 manifest 和同 cache native baseline，可以重新启动资格审计；在此之前，应把它标为历史探索性候选，而不是已验证方案。
