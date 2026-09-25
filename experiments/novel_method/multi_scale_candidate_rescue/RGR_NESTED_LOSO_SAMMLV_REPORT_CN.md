# Recognition-Guided Rescue (RGR) SAMMLV Nested LOSO Report

最终状态：**NO-GO-RGR**

## 1. 锁定协议

- R2-mean 是上一阶段 recognition separability audit 后锁定的 primary evidence，不是根据本次 outer test result 选择。
- 29-fold outer subject LOSO；Stage A 与 Stage B 均只用 train subjects 的 pooled Spotting F1 选参。
- Stage A 的 conservative events 在 Stage B 中只允许原样保留；RGR 只可加入 `P_low - P_high`。
- R1-mean 是独立 secondary control，未与 R2 在 outer result 后择优。
- 未使用 MSCR/multi-scale、未训练、未 model forward、未运行 CASME3。

## 2. Recognition 输入核验

- logits shape=`[T, 5]`；79/79 records 对齐，T 范围为 60–1680。
- softmax axis=1；class order=['negative', 'positive', 'surprise', 'others', 'neutral']；neutral id=4。

## 3. Outer pooled metrics

| Baseline | TP | FP | FN | Precision | Recall | F1 | weak | rescued | rescue TP/FP | rescue precision | recovered missed GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 Author Native | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | 0 | 0 | 0/0 | 0.000000 | 0 |
| A1 Tuned Native | 49 | 143 | 110 | 0.255208 | 0.308176 | 0.279202 | 0 | 0 | 0/0 | 0.000000 | 0 |
| A2 LowRescue | 51 | 170 | 108 | 0.230769 | 0.320755 | 0.268421 | 29 | 29 | 2/27 | 0.068966 | 2 |
| A3 R1-mean Rescue | 49 | 164 | 110 | 0.230047 | 0.308176 | 0.263441 | 53 | 21 | 0/21 | 0.000000 | 0 |
| A4 R2-mean RGR | 49 | 164 | 110 | 0.230047 | 0.308176 | 0.263441 | 86 | 21 | 0/21 | 0.000000 | 0 |

## 4. LowRescue vs RGR

- LowRescue rescue precision=0.068966；RGR=0.000000。
- LowRescue rescue TP/FP=2/27；RGR=0/21。

## 5. Subject stability 与 bootstrap

- improved/equal/worse=0/20/9。
- leave-best-contributor-out pooled ΔF1=-0.016147。
- bootstrap mean ΔF1=-0.015720，95% CI=[-0.027706, -0.006480]。

## 6. Selected parameter frequencies

- Tuned Native: `{"c_b": {"1.0": 1, "1.25": 28}, "c_d": {"1.25": 29}, "c_s": {"2.0": 29}, "p_H": {"0.65": 29}}`
- LowRescue: `{"delta_p": {"0.05": 28, "0.1": 1}}`
- R1-mean Rescue: `{"delta_p": {"0.05": 20, "0.1": 1, "0.2": 8}, "gamma_R": {"0.35": 8, "0.55": 20, "0.65": 1}}`
- R2-mean RGR: `{"delta_p": {"0.05": 1, "0.2": 28}, "gamma_R": {"0.35": 11, "0.55": 17, "0.65": 1}}`

## 7. GO checks

- RGR_F1_ge_Tuned_plus_0.005: False
- RGR_rescue_TP_gt_0: False
- RGR_rescue_precision_gt_LowRescue: False
- gain_not_single_subject: False

## 8. 结论

**NO-GO-RGR：RGR 未超过 Tuned Native。按 stop rule 停止，不扩展后续方法。**

## 9. 解释边界

- 上一阶段 R2-mean separability 的固定 45 条 pool 来自 Author Native/MSCR diagnostic；本轮 W 来自每折 Conservative Tuned Native（主要为 p_H=0.65），因此是不同的候选分布。前一阶段的排序 GO 不保证在本轮 W 上成功。
- R2-mean 虽在本轮 outer test 前锁定，但它是在同一 SAMMLV 数据集的前置 audit 中从九种 evidence 里识别出来的；所以本实验是第一阶段 feasibility test，不是完全独立的 confirmatory generalization test。
