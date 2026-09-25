# ME-TST+ Native Inference Failure-Mode Audit

## 1. 审计性质

本报告是 cache-only GT diagnostic/oracle analysis，不是 Method 结果。没有训练、re-forward、CUDA、新 threshold、新 NMS、新 decoder 或参数搜索。原始 cache 全程只读。

## 2. Native pipeline

`score → moving-average(2*k_p) → Moilanen p=0.55 → find_peaks(distance=k_p) → [peak-k_p,peak+k_p] → IoU matching → emotion vote → neutral result_synergy`。native candidate pipeline 不含 NMS/filter。完整证据见 `NATIVE_INFERENCE_TRACE_CN.md`。

## 3. SAMMLV

- Cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl`
- SHA-256：`3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`
- Subjects / videos / GT：29 / 79 / 159
- Native k_p：5
- Payload fields：`dataset, fps, frame_skip, include_mirror, k, k_p, k_p_macro, k_p_micro, method_type, num_gt, num_subjects, num_videos, ratio, records, strategy, strategy_name`
- Record fields：`emotion, gt_emotions, logits, mirror_logits, mirror_score, samples, score, subject, subject_index, video, video_index`
- Data-quality warnings：2 条；原始 native geometry 保留，畸形区间不用于科学 boundary/duration 归因。

### Native metrics

| Endpoint | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| raw spotting | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 |
| final result_synergy | 52 | 171 | 107 | 0.233184 | 0.327044 | 0.272251 |

### FN decomposition

| Category | Count | % all FN |
|---|---:|---:|
| FN-A_NO_PEAK | 3 | 2.80% |
| FN-B_SUB_THRESHOLD | 89 | 83.18% |
| FN-C_BOUNDARY | 14 | 13.08% |
| FN-D_SUPPRESSED | 1 | 0.93% |
| FN-E_OTHER | 0 | 0.00% |

### FP decomposition

| Category | Count | % all FP |
|---|---:|---:|
| FP-A_BACKGROUND | 159 | 92.98% |
| FP-B_NEAR_GT_LOCALIZATION | 12 | 7.02% |
| FP-C_DUPLICATE | 0 | 0.00% |
| FP-D_OTHER | 0 | 0.00% |

### Stage-wise recall ceiling

| Stage | TP ceiling | FN remaining | Recall ceiling |
|---|---:|---:|---:|
| all_local_peaks | 111 | 48 | 0.698113 |
| thresholded | 53 | 106 | 0.333333 |
| pre_final | 53 | 106 | 0.333333 |
| native_final | 52 | 107 | 0.327044 |

### Boundary diagnostics

- `peak inside GT but IoU<0.5`：1
- Native event duration：11

| Group | median GT duration | median ratio | median center error | median best IoU |
|---|---:|---:|---:|---:|
| matched_TP_GT | 12.000000 | 1.090909 | 1.500000 | 0.723810 |
| FN_GT | 11.000000 | 1.000000 | 4.500000 | 0.000000 |
| FN_C_boundary_only | 9.500000 | 0.863636 | 6.750000 | 0.186957 |

### Threshold diagnostics

| Group | n | median margin | IQR |
|---|---:|---:|---:|
| fn_b_margin | 89 | -0.285807 | [-0.427575, -0.186673] |
| final_background_fp_margin | 159 | 0.162925 | [0.074725, 0.258040] |
| subthreshold_background_margin | 3353 | -0.420022 | [-0.525658, -0.290111] |

- FN-B 中仅移除 threshold、保留 native interval 即可恢复：55

### Duplicate / suppression

- GT with >1 correct thresholded candidates：0
- Duplicate FP：0
- Correct candidate removed by result_synergy：1
- Native NMS/filter：not present

### Recognition secondary diagnosis

- Correct / wrong on final matched spotting：40 / 12
- Recognition F1 (repo definition)：0.650214
- Neutral TP / FP removed：1 / 13

### Oracle headroom

| Oracle | TP | FP | FN | F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|
| Boundary_oracle | 66 | 161 | 93 | 0.341969 | 0.069718 |
| Threshold_oracle | 107 | 171 | 52 | 0.489703 | 0.217451 |
| Suppression_oracle | 53 | 171 | 106 | 0.276762 | 0.004511 |
| Duplicate_oracle | 52 | 171 | 107 | 0.272251 | 0.000000 |

### Failure ranking

- #1 background_FP：159
- #2 threshold：89
- #3 boundary：26

## 3. CASME_3

- Cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl`
- SHA-256：`9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda`
- Subjects / videos / GT：94 / 462 / 858
- Native k_p：17
- Payload fields：`dataset, fps, frame_skip, include_mirror, k, k_p, k_p_macro, k_p_micro, method_type, num_gt, num_subjects, num_videos, ratio, records, strategy, strategy_name`
- Record fields：`emotion, gt_emotions, logits, mirror_logits, mirror_score, samples, score, subject, subject_index, video, video_index`
- Data-quality warnings：7 条；原始 native geometry 保留，畸形区间不用于科学 boundary/duration 归因。

### Native metrics

| Endpoint | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| raw spotting | 81 | 912 | 777 | 0.081571 | 0.094406 | 0.087520 |
| final result_synergy | 76 | 727 | 782 | 0.094645 | 0.088578 | 0.091511 |

### FN decomposition

| Category | Count | % all FN |
|---|---:|---:|
| FN-A_NO_PEAK | 17 | 2.17% |
| FN-B_SUB_THRESHOLD | 636 | 81.33% |
| FN-C_BOUNDARY | 123 | 15.73% |
| FN-D_SUPPRESSED | 5 | 0.64% |
| FN-E_OTHER | 1 | 0.13% |

### FP decomposition

| Category | Count | % all FP |
|---|---:|---:|
| FP-A_BACKGROUND | 613 | 84.32% |
| FP-B_NEAR_GT_LOCALIZATION | 114 | 15.68% |
| FP-C_DUPLICATE | 0 | 0.00% |
| FP-D_OTHER | 0 | 0.00% |

### Stage-wise recall ceiling

| Stage | TP ceiling | FN remaining | Recall ceiling |
|---|---:|---:|---:|
| all_local_peaks | 354 | 504 | 0.412587 |
| thresholded | 81 | 777 | 0.094406 |
| pre_final | 81 | 777 | 0.094406 |
| native_final | 76 | 782 | 0.088578 |

### Boundary diagnostics

- `peak inside GT but IoU<0.5`：46
- Native event duration：35

| Group | median GT duration | median ratio | median center error | median best IoU |
|---|---:|---:|---:|---:|
| FN_GT | 20.000000 | 0.571429 | 11.500000 | 0.000000 |
| FN_C_boundary_only | 16.000000 | 0.457143 | 13.000000 | 0.309735 |
| matched_TP_GT | 29.000000 | 0.828571 | 6.000000 | 0.652101 |

### Threshold diagnostics

| Group | n | median margin | IQR |
|---|---:|---:|---:|
| fn_b_margin | 636 | -0.208225 | [-0.312278, -0.093642] |
| final_background_fp_margin | 613 | 0.130461 | [0.041492, 0.223841] |
| subthreshold_background_margin | 46668 | -0.246093 | [-0.348597, -0.130379] |

- FN-B 中仅移除 threshold、保留 native interval 即可恢复：267

### Duplicate / suppression

- GT with >1 correct thresholded candidates：0
- Duplicate FP：0
- Correct candidate removed by result_synergy：5
- Native NMS/filter：not present

### Recognition secondary diagnosis

- Correct / wrong on final matched spotting：52 / 23
- Recognition F1 (repo definition)：0.423883
- Neutral TP / FP removed：5 / 185

### Oracle headroom

| Oracle | TP | FP | FN | F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|
| Boundary_oracle | 199 | 624 | 659 | 0.236764 | 0.145253 |
| Threshold_oracle | 343 | 727 | 515 | 0.355809 | 0.264298 |
| Suppression_oracle | 81 | 727 | 777 | 0.097239 | 0.005728 |
| Duplicate_oracle | 76 | 727 | 782 | 0.091511 | 0.000000 |

### Failure ranking

- #1 threshold：636
- #2 background_FP：613
- #3 boundary：237

## 4. Cross-Dataset Bottleneck

为把任务中的定性 GO 条件落实为可复核的唯一输出，审计脚本使用固定的保守 operational rules；完整阈值和逐数据集布尔检查保存在 JSON。它们是审计决策规则，不是调参搜索，也不是性能结果。

- Boundary eligible：True
- Suppression eligible：False
- Threshold eligible：False

## 5. 最终唯一 decision

**GO-BOUNDARY**

两个数据集都存在 thresholded peak 已形成、但固定 ±k_p interval 未达到 IoU=0.5 的同向 bottleneck，且 near-GT FP 与 boundary oracle headroom 同时达到预固定门槛。下一阶段只允许设计 training-free adaptive event-boundary inference。

## 6. 限制

- 所有 oracle 都使用 GT，是不可能上界，不是正式方法结果。
- result_synergy 是 evaluator 的 matching 后计数调整，不是可独立观察的 native NMS。
- all-local-peak ceiling 仍保留 native `distance=k_p`，只移除 height threshold。
- FN-B 是 earliest-failure 分类；Threshold oracle 只计入放过 threshold 后以 native interval 已能 IoU≥0.5 的子集，避免把后续 boundary failure 也归功于 threshold。
- threshold lowering 会引入多少额外 background peak 不能由 final background FP 单独回答，因此同时报告了 subthreshold background local-peak 分布。
