# MSCR Nested Subject-LOSO SAMMLV Report

最终状态：**NO-GO-MSCR-SOURCE**

## 1. 输入与协议

- cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl`；SHA-256：`3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`。
- 数据：29 subjects / 79 videos / 159 GT；k_p=5。
- 算法版本：`MSCR-SAMMLV-v1-locked-2026-09-04`；仅 frozen score，未运行 model forward、CUDA、hidden 或 recognition。
- 每个 outer fold 只用其余 subjects 的 GT 选择参数；test subject GT 只参与最终评价。
- tie-break：MacroSubjectF1、pooled F1、FP、rescue event 数、固定 lexicographic config rank。

## 2. Author Native gate

- TP/FP/FN = 53/184/106。
- native peak/event count = 237/237。
- Precision/Recall/F1 = 0.223629/0.333333/0.267677。
- baseline status：`PASS`。

## 3. Nested outer-LOSO pooled metrics

| Ablation | TP | FP | FN | Precision | Recall | F1 | rescue TP/FP | rescue precision | recovered GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AuthorNative | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | 0/0 | 0.000000 | 0 |
| TunedNative | 48 | 174 | 111 | 0.216216 | 0.301887 | 0.251969 | 0/0 | 0.000000 | 0 |
| LowRescue | 55 | 227 | 104 | 0.195035 | 0.345912 | 0.249433 | 2/43 | 0.044444 | 2 |
| HeightOnly | 53 | 187 | 106 | 0.220833 | 0.333333 | 0.265664 | 0/3 | 0.000000 | 0 |
| ScaleOnly | 53 | 188 | 106 | 0.219917 | 0.333333 | 0.265000 | 0/4 | 0.000000 | 0 |
| MSCR | 53 | 187 | 106 | 0.220833 | 0.333333 | 0.265664 | 0/3 | 0.000000 | 0 |

## 4. Native preservation

- Full MSCR：237/237 = 1.000000；onset/peak/offset tuple exact match。
- MSCR 仅新增 rescue events；无删除、移动、边界修改或额外 NMS。

## 5. Subject stability

- MSCR vs Author Native：improved/equal/worse = 0/28/1。
- MSCR vs Tuned Native：improved/equal/worse = 4/22/3。

## 6. Subject bootstrap

- MSCR_minus_AuthorNative：mean ΔF1=-0.002013，95% CI=[-0.005875, 0.000000]。
- MSCR_minus_TunedNative：mean ΔF1=0.015051，95% CI=[-0.001111, 0.038788]。

## 7. Parameter stability

- MSCR：`{'scale_family': {'A': 29}, 'delta_p': {'0.05': 29}, 'c_r': {'0.5': 29}, 'lambda': {'0.25': 29}, 'gamma': {'0.45': 1, '0.75': 1, '0.9': 27}}`
- TunedNative：`{'c_s': {'1.5': 1, '2.0': 28}, 'p': {'0.55': 26, '0.65': 3}, 'c_d': {'1.25': 29}, 'c_b': {'1.0': 1, '1.25': 28}}`
- LowRescue：`{'delta_p': {'0.05': 2, '0.1': 27}}`
- HeightOnly：`{'delta_p': {'0.05': 29}, 'gamma': {'0.9': 29}}`
- ScaleOnly：`{'scale_family': {'A': 29}, 'delta_p': {'0.05': 29}, 'c_r': {'0.5': 29}, 'gamma': {'0.45': 1, '0.75': 1, '0.9': 27}}`
- Full MSCR 的 family、Delta_p、c_r、lambda 在 29 folds 完全一致；gamma 27/29 选择 0.90。因此没有随机漂移，主要表现为选择器稳定倾向于几乎不救回 candidate。

## 8. GO 判定

- MSCR > Author Native：False。
- MSCR > Tuned Native：True。
- rescued TP > 0：False。
- rescue FP 明显低于 LowRescue（严格更少且至少低 20%）：True。

## 9. 结论

**NO-GO-MSCR-SOURCE：nested outer-subject LOSO 下 Full MSCR 未超过 Author Native；本次已完成合法 train-subject tuning，不能将失败归因于未调参。**

按任务书，本阶段不进入 CASME3、recognition/STRS 或 Skill。
