# Hidden Representation Event-Localization Predictability Audit

最终状态：**NO-GO-HREP-SOURCE**

## 1. 冻结资产与协议

- hidden source：`/private/tmp/hrep_sammlv_hidden`；tree SHA-256：`f5d75b8d545b73be83ac346e0b1626c087d89f15bc972c535b3dde3c3f118761`。
- manifest：`/private/tmp/hrep_sammlv_hidden/sammlv_full_hidden_dump_manifest.json`；SHA-256：`70a66db739e45688390adb624a959aa76529d0ff899e09a6e5b68f567005a293`。
- 完整性：29 subjects / 79 videos / total T=44295 / hidden dim=384。
- frozen 记录：model.eval=True，torch.no_grad=True，inference_only=True；batch=32，window=30。
- valid_mask 无效位置：720；涉及无效 hidden patch 的 association 同时从三种 predictor 排除。
- 本审计没有导入或运行 ME-TST，没有训练或微调 backbone。唯一拟合为 Ridge(alpha=1.0)。

## 2. Candidate 与 association

- fresh native TP/FP/FN/peaks：`[53, 184, 106, 237]`。
- eligible association rows：67；unique candidates：64；multi-target candidates：3。
- exclusions：`{'invalid_hidden_patch': 2}`。
- fresh/historical peak-list 数值差异：`[{'prediction_id': 'SAMMLV:023:023_1:1', 'historical_peak': 92, 'fresh_peak': 91}]`；HREP 始终使用 fresh peak。
- A 使用 fresh native greedy TP；B 沿用已锁定 FN-C diagnostic 与 FP-B closest-GT association。所有重复关联留在同一 subject LOSO fold。

## 3. 固定表示与预测器

- Hidden：`[p-2k_p,p+2k_p]` hidden 线性重采样至 41 点，计算每点与 `h_p` 的 cosine，得到唯一 41-D trajectory。
- Score negative control：同源 fresh smoothed score 的 41 点 patch，逐 patch min-max normalization。
- target：`r_on=(onset-p)/k_p`，`r_off=(offset-p)/k_p`；严格 subject-LOSO；scaler 仅 fit 训练 subjects。
- 输出直接 `np.rint` 后 clip；不 swap、不 clamp duration、不强制包含 peak；onset>offset 记 invalid、IoU=0。

## 4. Signed target 分布

| Cohort | N | y_on mean/median/IQR/p10/p90/min/max | y_off mean/median/IQR/p10/p90/min/max |
|---|---:|---:|---:|
| A_NATIVE_TP | 51 | -5.666667/-6.000000/2.000000/-8.000000/-3.000000/-10.000000/-1.000000 | 5.392157/5.000000/3.000000/2.000000/8.000000/0.000000/10.000000 |
| B_LOCALIZATION | 16 | -7.000000/-9.000000/13.000000/-15.500000/3.500000/-19.000000/5.000000 | 2.562500/-1.000000/13.750000/-4.000000/13.000000/-5.000000/17.000000 |
| B_FNC | 14 | -6.000000/-8.500000/11.500000/-14.100000/4.100000/-19.000000/5.000000 | 3.357143/-1.000000/15.250000/-3.700000/13.000000/-4.000000/17.000000 |
| B_NEAR_FP | 12 | -8.250000/-9.000000/8.500000/-15.900000/1.000000/-19.000000/5.000000 | 1.250000/-1.500000/7.250000/-4.000000/12.700000/-5.000000/13.000000 |
| COMBINED | 67 | -5.985075/-6.000000/3.000000/-10.000000/-1.600000/-19.000000/5.000000 | 4.716418/5.000000/5.000000/-1.400000/10.000000/-5.000000/17.000000 |

## 5. Cohort 结果

| Cohort | N | Predictor | onset mean/median AE | offset mean/median AE | mean/median IoU | IoU≥.3/.5/.7 | invalid |
|---|---:|---|---:|---:|---:|---:|---:|
| A_NATIVE_TP | 51 | native | 1.686275/2.000000 | 2.039216/2.000000 | 0.729475/0.733333 | 1.000000/1.000000/0.549020 | 0 |
| A_NATIVE_TP | 51 | score | 2.436268/1.888090 | 2.867237/2.379183 | 0.652460/0.687500 | 0.941176/0.823529/0.431373 | 0 |
| A_NATIVE_TP | 51 | hidden | 2.550465/1.851594 | 3.034349/2.049360 | 0.646002/0.687500 | 0.941176/0.823529/0.450980 | 0 |
| B_LOCALIZATION | 16 | native | 6.875000/6.500000 | 7.562500/8.000000 | 0.212208/0.186957 | 0.312500/0.000000/0.000000 | 0 |
| B_LOCALIZATION | 16 | score | 7.380761/6.716603 | 8.296955/7.675657 | 0.224015/0.133333 | 0.437500/0.125000/0.000000 | 0 |
| B_LOCALIZATION | 16 | hidden | 6.585128/6.230853 | 7.971878/8.625990 | 0.220792/0.153453 | 0.375000/0.062500/0.000000 | 0 |
| B_FNC | 14 | native | 6.571429/6.000000 | 7.500000/8.000000 | 0.219435/0.186957 | 0.357143/0.000000/0.000000 | 0 |
| B_FNC | 14 | score | 7.295628/6.716603 | 8.720157/7.675657 | 0.208398/0.133333 | 0.428571/0.071429/0.000000 | 0 |
| B_FNC | 14 | hidden | 6.006407/5.538797 | 7.753596/7.774435 | 0.242284/0.221569 | 0.428571/0.071429/0.000000 | 0 |
| B_NEAR_FP | 12 | native | 6.916667/6.000000 | 7.250000/8.000000 | 0.220414/0.231579 | 0.333333/0.000000/0.000000 | 0 |
| B_NEAR_FP | 12 | score | 6.797250/6.322361 | 7.537742/7.320266 | 0.244917/0.200000 | 0.500000/0.166667/0.000000 | 0 |
| B_NEAR_FP | 12 | hidden | 6.927168/6.230853 | 8.303707/8.928416 | 0.197337/0.150735 | 0.333333/0.000000/0.000000 | 0 |
| COMBINED | 67 | native | 2.925373/2.000000 | 3.358209/3.000000 | 0.605949/0.642857 | 0.835821/0.761194/0.417910 | 0 |
| COMBINED | 67 | score | 3.617042/2.619961 | 4.163886/3.145686 | 0.550144/0.600000 | 0.820896/0.656716/0.328358 | 0 |
| COMBINED | 67 | hidden | 3.513967/2.391549 | 4.213461/2.819107 | 0.544459/0.583333 | 0.805970/0.641791/0.343284 | 0 |

## 6. Paired IoU

| Cohort | Comparison | mean/median ΔIoU | improved/equal/worse |
|---|---|---:|---:|
| A_NATIVE_TP | Hidden_minus_Native | -0.083473/-0.053030 | 19/4/28 |
| A_NATIVE_TP | Hidden_minus_Score | -0.006457/0.000000 | 24/2/25 |
| B_LOCALIZATION | Hidden_minus_Native | 0.008584/-0.007130 | 6/1/9 |
| B_LOCALIZATION | Hidden_minus_Score | -0.003223/0.000000 | 7/2/7 |
| B_FNC | Hidden_minus_Native | 0.022849/-0.007130 | 6/0/8 |
| B_FNC | Hidden_minus_Score | 0.033886/0.000000 | 6/2/6 |
| B_NEAR_FP | Hidden_minus_Native | -0.023077/-0.015931 | 3/1/8 |
| B_NEAR_FP | Hidden_minus_Score | -0.047580/0.000000 | 5/2/5 |
| COMBINED | Hidden_minus_Native | -0.061489/-0.023529 | 25/5/37 |
| COMBINED | Hidden_minus_Score | -0.005685/0.000000 | 31/4/32 |

## 7. Safety / rescue

- Native TP preserved/lost：42/9，preservation=0.823529。
- FN-C rescued/remained：1/13。
- NetGain = FN-C rescued - Native TP lost = **-8**。
- Near-FP converted/remained：0/12；与 FN-C 重合 pair=10，未重复计入 NetGain。

## 8. Subject bootstrap

| Cohort | mean ΔIoU Hidden-Native | 95% CI |
|---|---:|---:|
| A_NATIVE_TP | -0.083984 | [-0.146333, -0.021521] |
| B_LOCALIZATION | 0.007254 | [-0.055548, 0.064169] |
| COMBINED | -0.061194 | [-0.098425, -0.019515] |

## 9. 预固定 GO gate

含糊项在看结果前操作化为：Native TP loss≤10%，invalid≤5%，10 项中至少 7 项通过；若 combined bootstrap CI 上界<0，则直接 NO-GO。

| Check | Pass |
|---|---:|
| 1_hidden_both_endpoint_mean_AE_below_native | False |
| 2_hidden_both_endpoint_mean_AE_below_score_and_mean_IoU_above_score | False |
| 3_hidden_mean_IoU_above_native | False |
| 4_hidden_median_IoU_not_below_native | False |
| 5_hidden_IoU_ge_0.5_fraction_above_native | False |
| 6_native_TP_loss_at_most_10_percent | False |
| 7_FN_C_rescue_exceeds_native_TP_loss | False |
| 8_B_LOCALIZATION_mean_delta_positive | True |
| 9_combined_bootstrap_lower_CI_nonnegative | False |
| 10_hidden_invalid_interval_at_most_5_percent | True |

通过 2/10；hard negative=True。

## 10. 结论

**NO-GO-HREP-SOURCE：ME-TST+ frozen pre-head temporal representation 在固定低容量、subject-LOSO localization probe 下未显示足够稳定的 event-boundary information。结合 score-level audits，停止当前 frozen-backbone inference refinement 路线。**

本实验与此前 hidden-relation classification/ranking audit 不重复：此前问 hidden 能否区分 true/noise candidate；本实验在 candidate anchor 固定后，仅问 hidden 能否预测 signed onset/offset。
