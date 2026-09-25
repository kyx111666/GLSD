# RGR-1 Python 实现复核

结论：通过。实现与锁定协议一致，subject 006 smoke test、29-fold 全量运行和独立确定性重跑均通过。

## 明确通过项

1. **PASS — 冻结输入与 recognition 校验。** `run_rgr1_nested_loso.py:89-113` 仅从 cache 读取 score/logits/emotion，检查 `[T,5]`、时间对齐和有限值，并沿 axis 1 计算 softmax；没有训练、反向传播或模型 forward。
2. **PASS — R2-mean 定义正确。** `run_rgr1_nested_loso.py:128-133` 在候选实际闭区间 `[onset, offset]` 上取最大 non-neutral 概率的均值；`run_rgr1_nested_loso.py:102-103` 明确排除 neutral id 4。
3. **PASS — weak pool 与 rescue 约束正确。** `run_rgr1_nested_loso.py:136-166` 只降低 threshold，使用 peak identity 构造 `low - strong`，并断言 strong 属于 low、rescued 属于 low 且不属于 strong。
4. **PASS — Low-threshold control 与 RGR 使用同一 weak source。** `run_rgr1_nested_loso.py:169-171` 通过同一 decoder 和同一 Delta-p 接收全部 weak candidates，没有独立重选 weak threshold。
5. **PASS — Nested LOSO 无 outer-test 泄漏。** `run_rgr1_nested_loso.py:311-316` 显式从全部 subject 中删除 held-out subject，并在选择 strong/RGR 参数前执行断言；每折保存 train subject 清单和所选配置（`run_rgr1_nested_loso.py:356-366`）。
6. **PASS — 参数空间与 tie-break 固定。** `run_rgr1_nested_loso.py:34-40` 固定 4×5 RGR grid；`run_rgr1_nested_loso.py:195-209` 以 pooled training-subject Spotting F1 选择，并严格按 smaller Delta-p、higher gamma-R 破同分。
7. **PASS — 既有 baseline/evaluator 被复用。** `run_rgr1_nested_loso.py:27-31` 直接导入现有 Author/Tuned decoder 和 spotting evaluator；full run 还在 `run_rgr1_nested_loso.py:412-415` 校验 Author Native 必须等于 53/184/106。
8. **PASS — 完整性与 provenance 输出齐全。** `run_rgr1_nested_loso.py:429-480` 记录失败 folds、cache hash、协议、grid、数据规模及所有 invariants；任何 full fold 失败都会令 `incomplete=true` 并返回非零状态。
9. **PASS — 所需结果文件全部隔离写入。** `run_rgr1_nested_loso.py:482-493` 生成 report、summary、outer-fold、frequency、rescued candidates 和 weak diagnostics 六个文件，不覆盖旧 baseline 目录。
10. **PASS — 可重复性。** 正式目录与 `/private/tmp/rgr1_sammlv_full_recheck` 的五个 CSV SHA-256 完全一致；去除 timestamp 后 report.json 完全一致。

## 运行核验

- subject 006 smoke：Tuned Native = 6/8/5，F1 = 0.48；所有运行时断言通过。
- full：29/29 folds 完成，`errors=[]`，`incomplete=false`。
- Author Native：53/184/106；Tuned Native：49/143/110，均与既有锚点一致。
- trace 自洽：86 weak candidates，24 selected，2 rescued TP，22 rescued FP。

## 非阻塞说明

subject 006 的 GT 仅含 negative 类，既有 recognition evaluator 会跳过无法展开为 2×2 的单类 confusion matrix，因此 smoke 的单折 Rec F1 为 0；全量 pooled recognition 含多类，能够正常计算。本说明不影响 spotting smoke gate。
