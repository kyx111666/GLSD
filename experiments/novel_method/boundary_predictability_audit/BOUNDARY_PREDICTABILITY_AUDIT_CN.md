# Boundary Predictability Information Audit

## 1. Locked diagnostic

本实验不是正式 Method。它只使用现有 frozen score；native smoothing、threshold 和 peaks 完全不变。输入仅为 smoothed peak height 或固定 `[p-2k_p,p+2k_p]` 的 41 点 patch，Ridge `alpha=1.0`，严格 subject-LOSO。patch 逐样本 min-max normalization，随后 scaler 只在训练 subjects 拟合。

OOF 连续 offset 用于 MAE；转换区间严格按 `onset=p-d_left`、`offset=p+d_right`，使用 `np.rint` 后 clip 到视频范围。若产生 `onset>offset`，不做后验修复且 IoU 记 0。没有 duration cap、GT correction 或超参数搜索。

## 2. Cache and association audit

| Dataset | SHA-256 | Subjects/Videos/GT | k_p | Native TP/FP/FN | Peaks | Assoc rows/unique candidates/multi-target |
|---|---|---:|---:|---:|---:|---:|
| SAMMLV | `3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569` | 29/79/159 | 5 | 53/184/106 | 237 | 69/66/3 |
| CASME_3 | `9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda` | 94/462/858 | 17 | 81/912/777 | 993 | 214/207/7 |

- SAMMLV excluded associations：`{}`。
- CASME_3 excluded associations：`{'invalid_or_out_of_range_gt': 2}`。

Association 沿用 Failure-Mode Audit：A 为 native greedy raw TP；B 为 FN-C `diagnostic_peak→该GT` 与 FP-B `closest_gt_id`。同一 peak 若确定性地关联多个 GT，保留为不同 association pair，并在同一 subject fold 内，绝不跨 LOSO 泄漏。

## 3. Target distributions

| Dataset | Cohort | N | d_left mean/median | d_right mean/median | GT duration mean/median |
|---|---|---:|---:|---:|---:|
| SAMMLV | A_NATIVE_TP | 53 | 5.660377/6.000000 | 5.339623/5.000000 | 12.000000/12.000000 |
| SAMMLV | B_LOCALIZATION | 16 | 7.000000/9.000000 | 2.562500/-1.000000 | 10.562500/10.000000 |
| SAMMLV | B_FNC | 14 | 6.000000/8.500000 | 3.357143/-1.000000 | 10.357143/9.500000 |
| SAMMLV | B_NEAR_FP | 12 | 8.250000/9.000000 | 1.250000/-1.500000 | 10.500000/11.000000 |
| SAMMLV | COMBINED | 69 | 5.971014/6.000000 | 4.695652/5.000000 | 11.666667/12.000000 |
| CASME_3 | A_NATIVE_TP | 81 | 15.641975/17.000000 | 17.271605/15.000000 | 33.913580/29.000000 |
| CASME_3 | B_LOCALIZATION | 133 | 21.902256/17.000000 | 19.481203/0.000000 | 42.383459/16.000000 |
| CASME_3 | B_FNC | 121 | 17.462810/16.000000 | 21.991736/0.000000 | 40.454545/16.000000 |
| CASME_3 | B_NEAR_FP | 113 | 24.000000/17.000000 | 20.035398/0.000000 | 45.035398/16.000000 |
| CASME_3 | COMBINED | 214 | 19.532710/17.000000 | 18.644860/8.000000 | 39.177570/23.000000 |

## 4. OOF offset error — combined cohort

| Dataset | Predictor | Left mean/median AE | Right mean/median AE |
|---|---|---:|---:|
| SAMMLV | Native_fixed | 2.942029/2.000000 | 3.289855/3.000000 |
| SAMMLV | Peak_only_Ridge | 2.881464/1.742840 | 3.343090/2.619012 |
| SAMMLV | Patch_Ridge | 3.722172/2.778436 | 4.160772/3.345153 |
| CASME_3 | Native_fixed | 12.504673/5.000000 | 22.953271/16.000000 |
| CASME_3 | Peak_only_Ridge | 15.105180/8.969852 | 23.693609/17.148505 |
| CASME_3 | Patch_Ridge | 21.733774/12.952857 | 26.099360/18.758562 |

## 5. Paired IoU — combined cohort

| Dataset | Predictor | Mean/median IoU | Mean/median ΔIoU | Improved/equal/worse | IoU≥.5/.7 | Invalid intervals |
|---|---|---:|---:|---:|---:|---:|
| SAMMLV | Native_fixed | 0.607269/0.642857 | 0/0 | 0/1/0 | 0.768116/0.405797 | 0 |
| SAMMLV | Peak_only_Ridge | 0.617655/0.692308 | 0.010386/0.036842 | 0.594203/0.043478/0.362319 | 0.739130/0.492754 | 0 |
| SAMMLV | Patch_Ridge | 0.549753/0.588235 | -0.057516/-0.023810 | 0.333333/0.144928/0.521739 | 0.652174/0.347826 | 0 |
| CASME_3 | Native_fixed | 0.432932/0.428571 | 0/0 | 0/1/0 | 0.378505/0.130841 | 0 |
| CASME_3 | Peak_only_Ridge | 0.403449/0.400000 | -0.029483/-0.030451 | 0.350467/0.009346/0.640187 | 0.317757/0.112150 | 0 |
| CASME_3 | Patch_Ridge | 0.327484/0.298936 | -0.105448/-0.096639 | 0.266355/0.004673/0.728972 | 0.266355/0.074766 | 9 |

### 5.1 Cohort-specific Patch evidence

| Dataset | Cohort | Native/Patch mean IoU | Mean/median ΔIoU | Native/Patch IoU≥.5 | Bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| SAMMLV | A_NATIVE_TP | 0.726533/0.647988 | -0.078545/-0.047619 | 1.000000/0.811321 | [-0.148438, -0.018927] |
| SAMMLV | B_LOCALIZATION | 0.212208/0.224350 | 0.012142/-0.001316 | 0.000000/0.125000 | [-0.092193, 0.085886] |
| SAMMLV | B_FNC | 0.219435/0.196876 | -0.022559/-0.001316 | 0.000000/0.071429 | [-0.104081, 0.070834] |
| SAMMLV | B_NEAR_FP | 0.220414/0.249072 | 0.028658/0.000000 | 0.000000/0.166667 | [-0.111604, 0.129000] |
| CASME_3 | A_NATIVE_TP | 0.671468/0.468336 | -0.203132/-0.196078 | 1.000000/0.543210 | [-0.251812, -0.149397] |
| CASME_3 | B_LOCALIZATION | 0.287658/0.241702 | -0.045957/-0.061404 | 0.000000/0.097744 | [-0.078818, -0.007635] |
| CASME_3 | B_FNC | 0.294411/0.238105 | -0.056306/-0.061404 | 0.000000/0.082645 | [-0.090724, -0.017323] |
| CASME_3 | B_NEAR_FP | 0.290963/0.248754 | -0.042209/-0.061404 | 0.000000/0.106195 | [-0.081698, 0.000392] |

## 6. Native TP preservation and FN-C rescue

| Dataset | TP preserved/lost | FN-C rescued/remained | NetGain |
|---|---:|---:|---:|
| SAMMLV | 43/10 | 1/13 | -9 |
| CASME_3 | 44/37 | 10/111 | -27 |

## 7. Subject bootstrap

| Dataset | Cohort | Mean ΔIoU | 95% CI |
|---|---|---:|---:|
| SAMMLV | A_NATIVE_TP | -0.080418 | [-0.148438, -0.018927] |
| SAMMLV | B_LOCALIZATION | 0.009824 | [-0.092193, 0.085886] |
| SAMMLV | COMBINED | -0.058453 | [-0.101906, -0.021236] |
| CASME_3 | A_NATIVE_TP | -0.200943 | [-0.251812, -0.149397] |
| CASME_3 | B_LOCALIZATION | -0.046108 | [-0.078818, -0.007635] |
| CASME_3 | COMBINED | -0.105181 | [-0.136674, -0.072600] |

## 8. Locked checks

### SAMMLV

- patch_left_and_right_mean_MAE_below_native: False
- patch_left_and_right_mean_MAE_below_peak_only: False
- paired_mean_and_median_IoU_improved: False
- IoU_ge_0.5_fraction_improved: False
- FN_C_rescue_exceeds_native_TP_loss: False
- bootstrap_CI_strictly_positive: False
- Internal status (`>=5/6`): False

### CASME_3

- patch_left_and_right_mean_MAE_below_native: False
- patch_left_and_right_mean_MAE_below_peak_only: False
- paired_mean_and_median_IoU_improved: False
- IoU_ge_0.5_fraction_improved: False
- FN_C_rescue_exceeds_native_TP_loss: False
- bootstrap_CI_strictly_positive: False
- Internal status (`>=5/6`): False

## 9. Locked SAMMLV→CASME3 diagnostic

Status: **NOT_RUN_INTERNAL_GATE_FAILED**.

未运行：两个数据集内部 LOSO 没有同时通过预先锁定的 positive gate，因此按协议禁止启动跨数据集 probe。

## 10. Final decision

**NO-GO-BOUNDARY-INFORMATION**

当前 frozen score 上的低容量监督 probe 没有提供足够稳定的 boundary predictability。按预设协议停止整个 score-level adaptive-boundary 方向，不再尝试 nonlinear probe 或 feature zoo。

原 cache 未修改；未运行 backbone forward、CUDA、hidden feature、recognition、candidate recovery 或任何新 boundary heuristic。
