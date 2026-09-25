# Tuned Native Boundary Oracle：实现与数据审查

## 审查结论

**PASS。** 未发现会改变候选、peak、正式 matcher 或 anchor 的实现问题。该程序只重放已锁定的 outer-fold Tuned Native 配置，并在 anchor 通过后执行 retrospective GT oracle。

## 显式通过项

1. **PASS — 数据源锁定。** 程序验证输入是 `SAMMLV` 且 `k_p=5`，并核对 source report 与 frozen cache 的 SHA-256；不一致会直接终止（`run_boundary_oracle_diagnostic.py:321-334`）。
2. **PASS — 配置不重选。** 每个 outer fold 的 `selected_strong_config` 直接从已完成 RGR-1 报告读取，且要求配置 subject 集合与 cache subject 集合完全一致（`run_boundary_oracle_diagnostic.py:335-342`）。
3. **PASS — 正式解码与 matcher 复用。** 代码直接调用现有 `tuned_native_decode` 和 `match_events`，未复制或修改 native candidate generation / matching（`run_boundary_oracle_diagnostic.py:30-35, 356-363`）。
4. **PASS — 硬 anchor gate。** 仅当 TP/FP/FN/event count 精确等于 `49/143/110/192` 时才进入 oracle；失败路径只写 incomplete report 并退出（`run_boundary_oracle_diagnostic.py:441-460`）。
5. **PASS — candidate identity 保持。** prediction 的 onset/peak/offset、emotion、confidence、outer fold 和实际 `k_native` 被原样记录；未添加、删除或移动 candidate（`run_boundary_oracle_diagnostic.py:373-390`）。
6. **PASS — 全量 candidate×GT 可审计。** 每个 prediction 与同视频每个 GT 的 IoU、距离、边界误差及 nearest 标记均导出，nearest 排序严格为 IoU、peak distance、center distance（`run_boundary_oracle_diagnostic.py:129-138, 398-413`）。
7. **PASS — oracle search 严格固定 peak。** 穷举所有合法整数 onset/offset，要求区间包含原 peak，并以最小总 boundary change 为主排序（`run_boundary_oracle_diagnostic.py:153-202`）。
8. **PASS — one-to-one oracle。** 使用带 dummy assignment 的最大基数分配；在最大可恢复数量下依次偏好更小 boundary change、更高 achievable IoU 和更高原 confidence（`run_boundary_oracle_diagnostic.py:205-240`）。
9. **PASS — 输出隔离。** 若目标目录已经存在，程序拒绝覆盖；所有结果只写入新目录（`run_boundary_oracle_diagnostic.py:314-318, 634-749`）。
10. **PASS — 无训练/forward。** 静态检索未发现 `model.train`、`backward`、`optimizer` 或模型 forward；运行只读取 pickle、JSON 并做 cache-level 解码与统计。

## 独立验证记录

- Python compile：PASS。
- Tuned Native anchor：PASS，`49/143/110`，F1=`0.2792022792022792`。
- candidate identity：192 个；正式 TP=49、FP=143。
- candidate×GT 明细：452 行、192 个唯一 prediction；每个有 GT 的 prediction 恰有一个 `nearest_all_gt`。
- cohort：`1+11+0+131=143`，互斥且覆盖全部 Native FP。
- oracle：11 个 prediction、11 个唯一 unmatched GT；不存在重复分配。
- 对 11 个 oracle case 另行使用双重 Python 循环穷举全部合法区间，11/11 的最小 boundary change、oracle interval 和 IoU 与结果文件完全一致。
- 所有 oracle interval 都包含原 peak，且 IoU>=0.5。
- 三次运行的全部 CSV 内容确定性一致；JSON 仅 timestamp 不同。

## 解释限制（非实现缺陷）

11 个可恢复案例中 10 个属于 `PEAK_OUTSIDE_GT`，仅 1 个属于 peak-in-GT。因而 `60/132/99` 是允许边界包住邻近 peak 的严格 oracle 上界，不能解读为“11 个 peak 都已精确定位”，也不能作为正式模型结果。
