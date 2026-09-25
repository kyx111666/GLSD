# GLSD Controlled Exploration EXP-1C

## Controlled change

本实验只改变跨尺度局部证据聚合。对三个对齐证据降序排列
`l_(1) >= l_(2) >= l_(3)` 后，使用
`L_top2 = (l_(1) + l_(2)) / 2`。锁定 baseline 仍使用 median，EXP-1B 使用 max。

融合保持 `S(c) = (G(c) + L(c)) / 2`。候选、尺度、局部半径、阈值网格、
区间协议、evaluator、nested LOSO、tie-break 和 90-config budget 均未改变；
canonical GLSD 与 baseline snapshot 未修改。

## F1 comparison

| Setting | Median F1 | Top-2 F1 | Max F1 | Top-2 − Median | Max − Median | Top-2 − Max | Top-2 paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.292683 | 0.258258 | +0.004395 | -0.030030 | +0.034425 | [-0.011078, +0.019518] |
| ME-TST/CAS(ME)3 | 0.094164 | 0.101426 | 0.101130 | +0.007263 | +0.006966 | +0.000297 | [-0.000124, +0.016117] |
| BoostingVRME/SAMMLV | 0.283784 | 0.292208 | 0.268571 | +0.008424 | -0.015212 | +0.023636 | [-0.020339, +0.041559] |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.107211 | 0.107747 | -0.005677 | -0.005141 | -0.000536 | [-0.011790, -0.000230] |

Top-2 相对 median 在 3/4 setting 上提升，平均 F1 变化为 `+0.003601`，
最差变化为 `-0.005677`。相比之下，max 仅在 1/4 setting 上提升，平均变化
为 `-0.010854`，最差变化为 `-0.030030`。Top-2 的跨 setting F1 变化标准差
也更小（0.005554 vs. 0.013573）。不过，前三项相对 median 的提升区间均跨零；
BoostingVRME/CAS(ME)3 的下降区间不跨零。

## FP/FN change

| Setting | Top-2 ΔFP | Top-2 ΔFN | Max ΔFP | Max ΔFN |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | -5 | 0 | +5 | +5 |
| ME-TST/CAS(ME)3 | -146 | 0 | -178 | +2 |
| BoostingVRME/SAMMLV | +9 | -3 | +49 | -5 |
| BoostingVRME/CAS(ME)3 | -11 | +7 | +119 | -1 |

Top-2 没有复现 max 在 BoostingVRME 上的大幅 FP 膨胀：SAMMLV 的 FP 增量从
`+49` 降至 `+9`，CAS(ME)3 从 `+119` 变为 `-11`。代价是 CAS(ME)3 上少检出
7 个 TP（FN `+7`），使该 setting 的 F1 显著下降。ME-TST 两个 setting 保持 TP/FN
不变，同时分别减少 5 和 146 个 FP；BoostingVRME/SAMMLV 则以 9 个额外 FP
换得 3 个额外 TP（FN `-3`）。

## Candidate mechanism summary

在固定候选上，三个输入时总有 `median = l_(2) <= L_top2 <= max = l_(1)`。
因此 Top-2 的局部证据机制确实位于 consensus-seeking median 与 winner-take-all max
之间：它允许一个强尺度提高分数，但必须由第二强尺度共同支撑，抑制单尺度尖峰直接主导。

Nested LOSO 的重新选参会打破“最终 FP/FN 必然逐项位于 median 与 max 之间”的关系。
Top-2 相对 baseline 改变了 29/29、94/94、27/29、90/94 个 subject-level 配置，
明显多于 max 在两个 BoostingVRME setting 上的 7/29 与 19/94。因此最终行为不仅来自
点位分数插值，也来自阈值/尺度补偿。ME-TST 的补偿主要表现为等 TP 下减 FP；
BoostingVRME/SAMMLV 获得温和 recall 增益；BoostingVRME/CAS(ME)3 则转为偏保守，
FP 略降但 FN 增加。

## Decision

Top-2 在经验上比 max 给出更稳定的折中：3/4 setting 优于 median，平均变化为正，
最差退化和跨 setting 波动都显著小于 max，并避免了 max 的 BoostingVRME FP 激增。
但它不是跨 backbone/dataset 一致改进；BoostingVRME/CAS(ME)3 的 recall 损失得到 paired bootstrap 支持。
结论是：Top-2 是比 max 更可信的 controlled-exploration 候选，但当前证据不足以替换锁定的
median baseline。
