# ME-TST+ Native Inference Trace

## 结论

原生 spotting 路径没有 NMS、候选级 filter、冲突消解或 interval 合并。`result_synergy` 不是 spotting decoder；它在 IoU matching 与 recognition 之后，按 neutral 标签调整最终计数。

## Exact pipeline

| Stage | 输入 | 原生操作 | 输出 |
|---|---|---|---|
| S0 | frozen `score[T]` | 不修改 | raw score |
| S1 | raw score | `np.convolve` moving average，宽度 `2*k_p`，`mode=same` | smoothed score |
| S2 | smoothed score | `mean + 0.55*(max-mean)` | 每视频 threshold |
| S3 | smoothed score | `scipy.signal.find_peaks(height=threshold, distance=k_p)` | thresholded peaks |
| S4 | peak | `[peak-k_p, peak+k_p]`，不在生成阶段裁边；confidence 字段固定为 0 | spotting events |
| S5 | event + emotion sequence | 区间内部多数票 | event emotion |
| S6 | 已完成 IoU matching 的 event | strategy-1 `result_synergy` 移除 neutral TP/FP 并相应增加 FN | final reported counts |

## 不存在的 stage

- native NMS：不存在。
- native downstream candidate filter：不存在。
- native conflict resolution：不存在；只有 evaluator 的一对一 IoU matching。
- mirror：cache 中虽有字段，但本 audit 完全不读取。

## 数据集参数

| Dataset | k_p | smoothing width | interval duration | strategy |
|---|---:|---:|---:|---|
| SAMMLV | 5 | 10 | 11 | recognition_mainly_with_neutral_synergy |
| CASME_3 | 17 | 34 | 35 | recognition_mainly_with_neutral_synergy |

## 代码证据

- `training_utils.py:9-13,15-41`：smoothing、Moilanen threshold、find_peaks、固定区间。
- `paper_metrics.py:39-55`：paper-aligned 等价 candidate 实现。
- `paper_metrics.py:79-87`：区间情绪多数票。
- `compare_paper_aligned_strategies.py:243-284`：matching 后的 neutral result_synergy。

本文件在任何 GT failure 分类计算之前由审计脚本写出。
