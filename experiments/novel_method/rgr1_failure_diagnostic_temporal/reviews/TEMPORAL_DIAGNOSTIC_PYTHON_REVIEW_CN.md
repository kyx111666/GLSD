# RGR-1 Temporal Failure Diagnostic Python Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-05
> **Script reviewed**: `my_method/rgr1_failure_diagnostic_temporal/run_temporal_failure_diagnostic.py`

## Pass Items

1. ✅ `run_temporal_failure_diagnostic.py:347-373` 将原 RGR-1 trace 的 86 个 candidate identity 设为权威集合，并同时校验 trace 行数、identity 唯一性、cache SHA-256 和恢复后的 7 个正例；没有重新选择 weak membership。
2. ✅ `run_temporal_failure_diagnostic.py:76-80,149-152` 对 `[T,5]` logits 沿 class axis 做稳定 softmax，并只在 `{0,1,2,3}` 上取最大值，正确排除 neutral id 4。
3. ✅ `run_temporal_failure_diagnostic.py:155-181` 在 candidate 实际区间内提取两条同 index 轨迹，处理边界 clip、常数序列 correlation NaN，并逐条校验重算 R2-mean 与原 trace confidence 一致。
4. ✅ `run_temporal_failure_diagnostic.py:167-195` 严格实现任务书预定义的 rise、fall、peak contrast、range、rise-fall、peak distance、normalized distance 和 Pearson correlation；没有新增 scalar feature 或组合分数。
5. ✅ `run_temporal_failure_diagnostic.py:100-105` 对预定义 lower-is-TP 的 distance 指标使用负方向计算 AUC，其余指标保持 higher-is-TP；没有按标签搜索方向。
6. ✅ `run_temporal_failure_diagnostic.py:211-249` 对每个连续 descriptor 输出 TP/FP 描述统计、ROC-AUC、PR-AUC，并逐一移除 7 个 positive 计算 49 条 LOO 结果。
7. ✅ `run_temporal_failure_diagnostic.py:252-267` 严格按 R2-mean 最高 5、contrast 最高 5、distance 最小 5 的固定规则选择 FP，并在去重时保留全部 selection reasons。
8. ✅ `run_temporal_failure_diagnostic.py:375-393` 轨迹数据只保留全部 7 个 TP 和固定规则选出的 13 个去重 FP；未人工挑选“漂亮案例”。
9. ✅ `run_temporal_failure_diagnostic.py:399-416` 证据等级首先执行用户给定的 STRONG 数值门槛；没有 descriptor 通过时，仅依据单 descriptor 的描述性表现评为 WEAK，没有组合 descriptor 或运行新 decoder。
10. ✅ `run_temporal_failure_diagnostic.py:457-472` 将任务要求的八个文件全部写入新的隔离目录，没有覆盖原 RGR-1 结果。

## Failed / Repaired Items

无。

## Remaining Risks

- 正例仅 7 个，来自 4 个 subject；所有 AUC 都应按 small-n retrospective diagnostic 解读。
- 原 weak trace 对未被 RGR 接收的候选没有保存 Low-control TP/FP 标签，因此 `run_temporal_failure_diagnostic.py:113-135` 使用原 fold-selected Tuned Native 和既有 matcher 恢复标签；86 个 identity 未改变，且恢复结果精确命中 7 TP/79 FP 锚点。
- 两个 candidate interval 接近序列边界并发生有效 index clip；clip 数量已记录在 report.json。

## Reproducibility Checks

- 语法编译：PASS。
- 正式运行：PASS；alignment=PASS，86=7+79，errors=[]，incomplete=false。
- 独立输出目录重跑：PASS。
- 六个数值 CSV 与独立重跑逐字节一致；去掉路径和 timestamp 后 report.json 完全一致。

## Run Instructions

```bash
cd <workspace-root>/RethinkFuse_reproduction
PYTHONPATH=/private/tmp/hrep_pydeps python3 my_method/rgr1_failure_diagnostic_temporal/run_temporal_failure_diagnostic.py
```

## Expected Outputs

- `results/rgr1_failure_diagnostic_temporal/report.json`
- `results/rgr1_failure_diagnostic_temporal/descriptor_summary.csv`
- `results/rgr1_failure_diagnostic_temporal/candidate_descriptors.csv`
- `results/rgr1_failure_diagnostic_temporal/leave_one_positive_auc.csv`
- `results/rgr1_failure_diagnostic_temporal/tp_case_details.csv`
- `results/rgr1_failure_diagnostic_temporal/selected_fp_case_details.csv`
- `results/rgr1_failure_diagnostic_temporal/trajectory_data.csv`
- `results/rgr1_failure_diagnostic_temporal/TEMPORAL_DIAGNOSTIC_REPORT_CN.md`

## Recommended Next Step

停止自动实验，等待人工根据 `TEMPORAL-EVIDENCE-WEAK` 及逐例轨迹作后续判断。
