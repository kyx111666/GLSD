# Adaptive Boundary TCB Feasibility Audit

## 1. Scope and locked rule

本实验只读取既有 frozen score cache。native smoothing、threshold、peak indices 与 confidence 完全不变；TCB 只把固定 `peak±k_p` 换成包含该 peak 的 maximal contiguous suprathreshold component。没有训练、forward、CUDA、参数搜索、candidate recovery、NMS 或 recognition redesign。

主结论使用 **raw spotting**，以隔离 boundary 本身；依赖区间 emotion vote 的 result-synergy/recognition 只作 secondary report。

## 2. Native baseline and candidate invariance

| Dataset | Native raw TP/FP/FN | Native final TP/FP/FN | Peaks | Exact-list videos | Jaccard |
|---|---:|---:|---:|---:|---:|
| SAMMLV | 53/184/106 | 52/171/107 | 237 | 79/79 | 1.0 |
| CASME_3 | 81/912/777 | 76/727/782 | 993 | 462/462 | 1.0 |

## 3. Full raw spotting comparison

| Dataset | Boundary | TP | FP | FN | Precision | Recall | F1 | ΔF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV | Native ±k | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | 0.000000 |
| SAMMLV | TCB | 40 | 197 | 119 | 0.168776 | 0.251572 | 0.202020 | -0.065657 |
| CASME_3 | Native ±k | 81 | 912 | 777 | 0.081571 | 0.094406 | 0.087520 | 0.000000 |
| CASME_3 | TCB | 77 | 916 | 781 | 0.077543 | 0.089744 | 0.083198 | -0.004322 |

## 4. Mechanism evidence

| Dataset | Native FN-C | Converted | Remained | IoU worse | Native TP preserved | Native TP lost | NetGain | Near-GT FP converted/remained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV | 14 | 1 (0.071429) | 13 | 10 | 39 | 14 | -13 | 1/11 |
| CASME_3 | 123 | 17 (0.138211) | 106 | 57 | 60 | 21 | -4 | 14/100 |

## 5. Duration

| Dataset | Group | Mean | Median | IQR | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|
| SAMMLV | Native prediction | 11.000000 | 11.000000 | 0.000000 | 11.000000 | 11.000000 |
| SAMMLV | TCB prediction | 8.430380 | 8.000000 | 4.000000 | 4.000000 | 12.000000 |
| SAMMLV | valid GT | 11.452830 | 12.000000 | 5.000000 | 7.000000 | 15.000000 |
| CASME_3 | Native prediction | 35.000000 | 35.000000 | 0.000000 | 35.000000 | 35.000000 |
| CASME_3 | TCB prediction | 29.109768 | 31.000000 | 6.000000 | 21.000000 | 34.000000 |
| CASME_3 | valid GT | 35.647608 | 22.000000 | 19.000000 | 10.000000 | 55.000000 |

## 6. Paired IoU and directional error

配对 cohort 固定为 native max-IoU>0 的同一 peak，并始终使用该 peak 的 native-best GT；不会为 TCB 重新挑更有利的 GT。

| Dataset | Native IoU mean/median/IQR | TCB IoU mean/median/IQR | Mean ΔIoU | Median ΔIoU | Improved/equal/worse | Median onset N→T | Median offset N→T |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV | 0.628915/0.666667/0.245089 | 0.481510/0.535897/0.245061 | -0.147404 | -0.138095 | 0.196970/0.015152/0.787879 | 2.000000→3.000000 | 2.000000→3.000000 |
| CASME_3 | 0.434548/0.428571/0.341506 | 0.410620/0.424242/0.336957 | -0.023928 | 0.013095 | 0.514019/0.000000/0.485981 | 5.000000→6.500000 | 16.000000→13.000000 |

- SAMMLV IoU≥0.3/0.5/0.7：Native 0.878788/0.803030/0.424242；TCB 0.787879/0.621212/0.151515。
- CASME_3 IoU≥0.3/0.5/0.7：Native 0.686916/0.378505/0.130841；TCB 0.668224/0.369159/0.112150。

## 7. Subject stability

| Dataset | Improved | Equal | Worse | Bootstrap mean ΔF1 | 95% CI |
|---|---:|---:|---:|---:|---:|
| SAMMLV | 1 | 16 | 12 | -0.065470 | [-0.105640, -0.030027] |
| CASME_3 | 11 | 67 | 16 | -0.004393 | [-0.018628, 0.009753] |

## 8. Secondary result-synergy / recognition

| Dataset | Final spotting F1 N→T | Recognition F1 N→T |
|---|---:|---:|
| SAMMLV | 0.272251→0.211082 | 0.650214→0.529744 |
| CASME_3 | 0.091511→0.084423 | 0.423883→0.425472 |

## 9. Gate checks

### SAMMLV

- spotting_F1_improved: False
- FN_C_converted: True
- near_GT_FP_converted: True
- net_boundary_gain_positive: False
- paired_mean_IoU_improved: False
- onset_or_offset_median_error_improved: False
- bootstrap_not_systematically_negative: False

### CASME_3

- spotting_F1_improved: False
- FN_C_converted: True
- near_GT_FP_converted: True
- net_boundary_gain_positive: False
- paired_mean_IoU_improved: False
- onset_or_offset_median_error_improved: True
- bootstrap_not_systematically_negative: True

## 10. Final feasibility decision

**NO-GO-TCB**

简单 threshold-connected boundary 未能在两个数据集一致兑现 boundary oracle headroom。本结论只否定 TCB，不否定此前观测到的 boundary failure mode，也不授权自动搜索新的 boundary heuristic。

## 11. Data and interpretation limits

- SAMMLV cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl`
- SAMMLV SHA-256：`3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`
- SAMMLV subjects/videos/GT/k_p：29/79/159/5
- SAMMLV data-quality warnings：2
- CASME_3 cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl`
- CASME_3 SHA-256：`9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda`
- CASME_3 subjects/videos/GT/k_p：94/462/858/17
- CASME_3 data-quality warnings：7
- TCB 没有任何可调参数；bootstrap 只量化 subject 重采样不确定性，不参与构造 boundary。
- GT 仅用于 evaluator、配对诊断和 GO/NO-GO gate，从未用于生成或修改 prediction。
- 主结果是 raw spotting；result synergy 会因 interval 内 emotion vote 改变，故只作二级结果。
