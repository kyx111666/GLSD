# MSCR Post-hoc Diagnostic Audit

最终状态：**MSCR-EVIDENCE-NONSEPARABLE**

## 1. 审计边界

- 仅执行 pooled-F1 selection objective 与 weak-candidate separability trace。
- MSCR formula、candidate generation、全部网格、outer folds、evaluator 与 Native preservation 未改变。
- 未运行 CASME3、recognition、hidden、训练模型或新方法。

## 2. Diagnostic A：Pooled-F1 Selection

| Method | TP | FP | FN | Precision | Recall | F1 | Rescue TP/FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| AuthorNative | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | 0/0 |
| TunedNative | 49 | 143 | 110 | 0.255208 | 0.308176 | 0.279202 | 0/0 |
| LowRescue | 54 | 226 | 105 | 0.192857 | 0.339623 | 0.246014 | 1/42 |
| HeightOnly | 53 | 187 | 106 | 0.220833 | 0.333333 | 0.265664 | 0/3 |
| ScaleOnly | 53 | 188 | 106 | 0.219917 | 0.333333 | 0.265000 | 0/4 |
| MSCR | 53 | 185 | 106 | 0.222689 | 0.333333 | 0.267003 | 0/1 |

Native event exact preservation：237/237。
MSCR-AuthorNative F1 delta：-0.000674；预固定明显改善阈值：0.005。

## 3. Diagnostic B：True LowRescue cases

### 007/007_6 peak=180

- H=0.000000，M=0.000000，R=0.000000，gamma=0.90。
- per-scale nearest peaks：`{"0.8": 184, "1.25": 189}`。
- per-scale distances：`{"0.8": 4, "1.25": 9}`。
- selected config：`{"c_r": 0.5, "delta_p": 0.05, "gamma": 0.9, "lambda": 0.25, "scale_family": "A"}`。
- MSCR direct rejection reason：`NOT_IN_SELECTED_MSCR_WEAK_POOL`。

### 011/011_6 peak=379

- H=0.469764，M=0.500000，R=0.492441，gamma=0.90。
- per-scale nearest peaks：`{"0.8": 379, "1.25": 414}`。
- per-scale distances：`{"0.8": 0, "1.25": 35}`。
- selected config：`{"c_r": 0.5, "delta_p": 0.05, "gamma": 0.9, "lambda": 0.25, "scale_family": "A"}`。
- MSCR direct rejection reason：`R_BELOW_GAMMA`。

## 4. Weak-FP evidence distribution

| Evidence | mean | median | p10 | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| H | 0.312632 | 0.036911 | 0.000000 | 0.000000 | 0.571332 | 0.849940 |
| M | 0.311628 | 0.300000 | 0.000000 | 0.100000 | 0.500000 | 0.600000 |

## 5. Separability

- H：ROC-AUC=0.418605，PR-AUC=0.051634。
- M：ROC-AUC=0.424419，PR-AUC=0.057937。
- positive prevalence=0.044444。
- locked [H,M] separability rule passed=False。

- 007/007_6 peak=180：joint FP dominance=1.000000，nearest FP distance=0.000000，H/M≥FP-p75=False/False。
- 011/011_6 peak=379：joint FP dominance=0.162791，nearest FP distance=0.043547，H/M≥FP-p75=False/True。

## 6. Final interpretation

**MSCR-EVIDENCE-NONSEPARABLE：pooled-F1 selection 未逆转 source-side NO-GO，且两个 true rescue candidates 与 weak FP 在 H/M 上未满足预固定 separability 条件。**

本报告只解释现有失败，不提出或运行新的 fusion、threshold、feature 或 classifier。
