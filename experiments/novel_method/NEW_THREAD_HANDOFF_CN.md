# SkillME / ME-TST+ 新会话交接（当前唯一可信版本）

更新时间：2026-09-05

> 新会话必须先阅读本文。本文记录当前研究目标、工程恢复、正式科学结论、禁止重启的失败路线，以及最新 boundary oracle 进度。旧对话中的临时候选方案不能覆盖本文中的已审计结论。

## 1. 用户最终目标

- 投稿目标：ICASSP 2027；完整论文截止日期按当前计划为 2026-09-16。
- 基础模型：ME-TST / ME-TST+ 微表情 spotting + recognition。
- 冻结 backbone，不重新训练、不微调、不修改网络权重。
- 论文创新必须位于 inference / post-processing，目标是 training-free、可复现、计算成本低。
- 理想最终形态：把验证有效的 inference 方法封装成 Agent Skill，并设计跨 backbone / 数据集迁移实验；**但在确定性 inference 算法尚未科学 GO 前，禁止先包装 Skill 或写论文 Method。**
- 用户倾向自己在 Colab 执行 GPU/长实验；助手负责代码、命令、审计、结果解释与排错。Mac 本机不用于重建 ME-TST CUDA/Mamba 环境。

## 2. 当前可信数据协议

### SAMMLV frozen cache

- 路径：`caches/me_tst/sammlv_strategy1_outputs.pkl`
- SHA-256：`3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`
- 29 subjects、79 videos、159 GT。
- `k_p=5`，`frame_skip=7`，`fps=200`。
- cache 中包含 `score[T]`、`logits[T,5]`、`emotion[T]`、GT samples 等。

### Frozen-forward 恢复状态

- Colab 上已成功恢复 ME-TST+ frozen forward 和完整 SAMMLV hidden dump。
- full hidden dump：29 subjects、79 videos、`total_T=44295`、hidden dim=384、batch size=32。
- hidden→fc_spot→sigmoid→原 stitching 最大误差约 `1.788e-7`；未使用 historical compact 数值输出。
- fresh score 与 historical compact score 虽默认 `np.allclose=False`，但已通过 `PASS-DECODER-EQUIVALENT`：aggregate TP/FP/FN 完全一致，可将 fresh forward 用作同源科学分析。
- historical compact 只作为 reproduction/alignment reference；后续不得混用 fresh raw context 与 historical score。

## 3. 当前正式 baseline

### Author Native

- TP/FP/FN：`53 / 184 / 106`
- Spotting F1：`0.267677`

### Nested pooled-F1 Tuned Native

- Outer subject LOSO；每折只用 train subjects 的 pooled Spotting F1 选择配置。
- TP/FP/FN：`49 / 143 / 110`
- Spotting Precision：`0.255208`
- Spotting Recall：`0.308176`
- Spotting F1：`0.2792022792`
- Recognition F1：`0.6990`
- STRS：`0.1951623932`
- 主要收益来自更严格的 native threshold，表现为减少 41 FP、同时损失 4 TP。
- RGR-1 报告保存了每个 outer fold 的最终配置：28 折为 `c_s=2.0, p_s=0.65, c_d=1.25, c_b=1.25`；subject `037` 为 `c_s=2.0, p_s=0.65, c_d=1.25, c_b=1.0`。
- 当前后续诊断一律以该 Tuned Native 为正式 baseline，不得重新选择配置。

## 4. 已完成且必须冻结的科学结论

以下方向已经得到可信 NO-GO / STOP，不得在没有新的外部科学依据时换名重启、扩大 grid 或追加 heuristic：

| 方向 | 当前结论 |
|---|---|
| 历史 EquiScale / 早期双数据集 PoC | 资格审计未找到可信原始实现和逐样本结果；旧“正向结果”不得作为论文证据 |
| Scale consistency / EquiScale | `NO-GO` |
| PSED / persistence / prominence | `NO-GO` |
| Temporal morphology | `NO-GO` |
| Joint spotting-recognition evidence / SCED | `NO-GO` |
| Pre-stitch Context Agreement | 完整 SAMMLV 科学审计结果 `NO-GO`；Agreement 未超过 height，bootstrap CI 不支持增益 |
| MSCR multi-scale candidate rescue | `NO-GO-MSCR-SOURCE` |
| MSCR post-hoc | `MSCR-EVIDENCE-NONSEPARABLE` |
| Recognition weak-candidate separability | `GO-RECOGNITION-SEPARABILITY`，但阳性极少，只是 feasibility signal |
| RGR / RGR-1 | `NO-GO-RGR`；RGR-1 F1=`0.272`，低于 Tuned Native `0.279202` |
| RGR-1 temporal failure diagnostic | `TEMPORAL-EVIDENCE-WEAK`，只读诊断，不支持继续自动扩展 |
| Native failure-mode audit | `GO-BOUNDARY`，只是错误类型诊断，不是方法结果 |
| TCB boundary heuristic | `NO-GO-TCB` |
| LVB local-valley boundary | `NO-GO-LVB` |
| Hidden representation localization probe | `NO-GO-HREP-SOURCE` |
| Boundary predictability audit | `NO-GO-BOUNDARY-INFORMATION` |

特别注意：`GO-RECOGNITION-SEPARABILITY` 和 `GO-BOUNDARY` 都只是中间诊断信号；后续正式 nested experiment 已证明对应的 RGR、TCB、LVB 等具体方法失败，不能把早期 GO 改写成最终方法有效。

## 5. 最新进度：Tuned Native Boundary-Error Oracle Diagnostic

目录：`results/tuned_native_boundary_oracle_diagnostic/`

### Integrity

- 精确复现 Tuned Native anchor：`49 / 143 / 110`，F1=`0.2792022792`。
- 使用同一 frozen cache、RGR-1 已保存 outer-fold 配置、现有 `tuned_native_decode`、`match_events`、`interval_iou`。
- 未训练、未 forward backbone、未改 score/threshold/emotion、未新增/删除 candidate、未移动 peak、未运行新 decoder。

### FP cohorts

| Cohort | Count | 占 Native FP |
|---|---:|---:|
| B1 Peak-In-GT Near Miss | 1 | 0.70% |
| B2 Peak-Near-GT Near Miss | 11 | 7.69% |
| B3 Overlap Near Miss | 0 | 0.00% |
| B4 Pure FP | 131 | 91.61% |

### Fixed-peak boundary oracle

- 只固定现有 candidate peak，GT 仅用于事后穷举合法 onset/offset。
- one-to-one oracle 可恢复：11 个 prediction → 11 个唯一 unmatched GT。
- 分布于 9 个 subjects：`009, 012, 017, 019, 020, 026, 031, 032, 037`。
- 占 Native FN：10.00%；占 Native FP：7.69%。
- Oracle upper bound：`60 / 132 / 99`，Spotting F1=`0.3418803419`，相对 Tuned Native 绝对增加 `0.062678`。
- 最小 boundary change：median=5，p25=2.5，p75=9.5，max=12。
- Native TP IoU：median=`0.733333`，p10=`0.557692`，IoU≥0.7 比例=`0.5714`，peak-in-GT 比例=1.0。
- 最终证据等级：`BOUNDARY-EVIDENCE-STRONG`。

### 必须保留的限制

- 这是 **oracle upper bound，不是模型结果**，不能写入主结果表冒充可部署算法。
- 11 个可恢复案例中 10 个是 `PEAK_OUTSIDE_GT`，只有 1 个 peak 真正在 GT 内。
- 因此结果证明的是：允许边界包住一个很近但略偏离 GT 的 fixed peak 时，存在明显理论 headroom；它不证明已有 peak 已精确定位，也不证明能够从 score/hidden 中预测正确边界。
- 既有 TCB、LVB、HREP 和 boundary predictability 均已 NO-GO，所以不能因为 oracle 强就直接宣称 adaptive boundary 方法可行。

## 6. 当前唯一研究判断点

最新 oracle 已完成并按任务要求停止。当前没有已验证成功的 deployable inference method。

下一步必须由用户/老师作人工判断：

1. 是否仍在严格剩余时间内尝试一个**预注册、极小范围、完全 GT-free 的 boundary reconstruction feasibility gate**；或
2. 根据 TCB/LVB/HREP/boundary-information 的连续 NO-GO，停止该论文方法路线并重新确定课题范围。

在人工决定前禁止：

- 自动实现 AEBR 或任何 adaptive boundary decoder；
- 搜索新 boundary threshold / width / feature；
- 复活 TCB、LVB、FWHM、half-height、valley、morphology 等 handcrafted heuristic；
- 跑 CASME3、BoostingVRME 或迁移实验；
- 包装 Skill；
- 写论文 Method 或把 oracle 写成模型提升。

## 7. 新会话首先应阅读的文件

1. `my_method/NEW_THREAD_HANDOFF_CN.md`（本文）
2. `results/tuned_native_boundary_oracle_diagnostic/BOUNDARY_ORACLE_DIAGNOSTIC_REPORT_CN.md`
3. `results/tuned_native_boundary_oracle_diagnostic/report.json`
4. `results/tuned_native_boundary_oracle_diagnostic/oracle_recoverable_cases.csv`
5. `results/rgr1_sammlv_nested/report.json`
6. `my_method/multi_scale_candidate_rescue/RGR_NESTED_LOSO_SAMMLV_REPORT_CN.md`
7. `my_method/adaptive_boundary_tcb_feasibility/ADAPTIVE_BOUNDARY_TCB_FEASIBILITY_CN.md`
8. `my_method/adaptive_boundary_lvb_feasibility/ADAPTIVE_BOUNDARY_LVB_FEASIBILITY_CN.md`
9. `my_method/hidden_event_localization_predictability_audit/HIDDEN_EVENT_LOCALIZATION_PREDICTABILITY_AUDIT_CN.md`
10. `my_method/boundary_predictability_audit/BOUNDARY_PREDICTABILITY_AUDIT_CN.md`

## 8. 给新会话助手的执行原则

- 先区分 engineering recovery、oracle diagnostic、scientific feasibility 和正式 nested result，禁止混为一谈。
- 任何新主张必须有逐样本输出、subject-level protocol、无 test-GT selection、明确 hash 与可复现脚本。
- 新方法必须超过合法的 Tuned Native `F1=0.279202`，不能只超过 Author Native。
- 不得从完整 test result 反向挑方法或阈值。
- 已 NO-GO 的方向不能通过改名、加参数或扩大搜索重新启动。
- 如果用户只是提问，先解释现状，不要自动运行实验。
