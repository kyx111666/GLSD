# Adaptive Boundary LVB Feasibility Audit

## 1. Locked experiment

LVB 使用未修改的 native smoothed curve 和完全相同的 native peaks。每侧 boundary 是相邻 native peaks 所限定区域内最近的 `find_peaks(-g)` local minimum；若无 local minimum，仅该侧使用 clipped native `p±k_p` fallback。没有任何可调参数、GT boundary fitting、candidate recovery、peak shift、NMS 或 cap。

TCB 数字和逐事件区间只读取既有 audit 文件，没有重新优化。主 endpoint 为 raw spotting。

## 2. Baseline and candidate invariance

| Dataset | Native TP/FP/FN | Native F1 | Peaks N/L | Exact videos | Jaccard |
|---|---:|---:|---:|---:|---:|
| SAMMLV | 53/184/106 | 0.267677 | 237/237 | 79/79 | 1.000000 |
| CASME_3 | 81/912/777 | 0.087520 | 993/993 | 462/462 | 1.000000 |

## 3. Full spotting metrics

| Dataset | Boundary | TP | FP | FN | P | R | F1 | Δ vs Native |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV | Native | 53 | 184 | 106 | 0.223629 | 0.333333 | 0.267677 | 0.000000 |
| SAMMLV | TCB_reference | 40 | 197 | 119 | 0.168776 | 0.251572 | 0.202020 | -0.065657 |
| SAMMLV | LVB | 29 | 208 | 130 | 0.122363 | 0.182390 | 0.146465 | -0.121212 |
| CASME_3 | Native | 81 | 912 | 777 | 0.081571 | 0.094406 | 0.087520 | 0.000000 |
| CASME_3 | TCB_reference | 77 | 916 | 781 | 0.077543 | 0.089744 | 0.083198 | -0.004322 |
| CASME_3 | LVB | 35 | 958 | 823 | 0.035247 | 0.040793 | 0.037817 | -0.049703 |

## 4. Duration and extreme intervals

| Dataset | Boundary | Mean | Median | IQR | p10 | p90 | Min | Max | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMMLV | Native | 11.000000 | 11.000000 | 0.000000 | 11.000000 | 11.000000 | 11.000000 | 11.000000 | 11.000000 | 11.000000 |
| SAMMLV | TCB_reference | 8.430380 | 8.000000 | 4.000000 | 4.000000 | 12.000000 | N/A | N/A | N/A | N/A |
| SAMMLV | LVB | 21.679325 | 23.000000 | 11.000000 | 9.000000 | 31.000000 | 4.000000 | 41.000000 | 34.000000 | 37.000000 |
| SAMMLV | valid_GT | 11.452830 | 12.000000 | 5.000000 | 7.000000 | 15.000000 | 5.000000 | 16.000000 | 15.000000 | 16.000000 |
| CASME_3 | Native | 35.000000 | 35.000000 | 0.000000 | 35.000000 | 35.000000 | 35.000000 | 35.000000 | 35.000000 | 35.000000 |
| CASME_3 | TCB_reference | 29.109768 | 31.000000 | 6.000000 | 21.000000 | 34.000000 | N/A | N/A | N/A | N/A |
| CASME_3 | LVB | 39.276939 | 33.000000 | 37.000000 | 10.000000 | 73.800000 | 3.000000 | 126.000000 | 80.000000 | 93.080000 |
| CASME_3 | valid_GT | 35.647608 | 22.000000 | 19.000000 | 10.000000 | 55.000000 | 3.000000 | 3456.000000 | 71.000000 | 219.840000 |

- SAMMLV duration >2×native：119 (0.502110)。
- SAMMLV fallback left/right/both：9/7/0。
- CASME_3 duration >2×native：129 (0.129909)。
- CASME_3 fallback left/right/both：76/12/0。

## 5. FN-C / FP-B conversion and Native TP safety

| Dataset | FN-C converted/remained/worse | FP-B converted/remained/worse | Native TP preserved/lost | Requested NetGain | Unique repair NetGain | Actual ΔTP |
|---|---:|---:|---:|---:|---:|---:|
| SAMMLV | 4/10/4 | 3/9/3 | 25/28 | -21 | -24 | -24 |
| CASME_3 | 12/111/90 | 11/103/86 | 23/58 | -35 | -46 | -46 |

`FN-C converted + NearGT converted − NativeTP lost` 按任务原式报告；由于 FN-C 与 near-GT FP 可能是同一错误的两面，同时给出 overlap 去重后的 unique repair NetGain 与实际 ΔTP，防止重复计数。

## 6. Paired IoU

固定 cohort：native max-IoU>0 的同一 peak，固定 native-best GT，不为 LVB 重选 GT。

| Dataset | Boundary | Mean | Median | IQR | IoU≥.3/.5/.7 |
|---|---|---:|---:|---:|---:|
| SAMMLV | Native | 0.628915 | 0.666667 | 0.245089 | 0.878788/0.803030/0.424242 |
| SAMMLV | TCB_reference | 0.481510 | 0.535897 | 0.245061 | 0.787879/0.621212/0.151515 |
| SAMMLV | LVB | 0.460859 | 0.442391 | 0.259626 | 0.803030/0.424242/0.090909 |
| CASME_3 | Native | 0.434548 | 0.428571 | 0.341506 | 0.686916/0.378505/0.130841 |
| CASME_3 | TCB_reference | 0.410620 | 0.424242 | 0.336957 | 0.668224/0.369159/0.112150 |
| CASME_3 | LVB | 0.279740 | 0.251667 | 0.298100 | 0.420561/0.158879/0.042056 |

- SAMMLV LVB−Native mean/median ΔIoU：-0.168056/-0.200092；improved/equal/worse：0.242424/0.000000/0.757576。
- CASME_3 LVB−Native mean/median ΔIoU：-0.154808/-0.158156；improved/equal/worse：0.200935/0.000000/0.799065。

## 7. Onset / offset direction

| Dataset | Boundary | Median onset error | Median offset error |
|---|---|---:|---:|
| SAMMLV | Native | 2.000000 | 2.000000 |
| SAMMLV | TCB_reference | 3.000000 | 3.000000 |
| SAMMLV | LVB | 5.500000 | 6.000000 |
| CASME_3 | Native | 5.000000 | 16.000000 |
| CASME_3 | TCB_reference | 6.500000 | 13.000000 |
| CASME_3 | LVB | 16.500000 | 19.000000 |

## 8. Subject stability

| Dataset | Improved | Equal | Worse | Bootstrap mean ΔF1 | 95% CI |
|---|---:|---:|---:|---:|---:|
| SAMMLV | 3 | 10 | 16 | -0.118779 | [-0.164909, -0.073284] |
| CASME_3 | 4 | 53 | 37 | -0.049989 | [-0.068735, -0.031892] |

## 9. Gate checks

### SAMMLV

- F1_improved: False
- FN_C_converted: True
- native_TP_loss_below_unique_repairs: False
- paired_mean_and_median_IoU_improved: False
- onset_or_offset_median_error_improved: False
- extreme_interval_fraction_le_0_10: False
- bootstrap_not_systematically_negative: False

### CASME_3

- F1_improved: False
- FN_C_converted: True
- native_TP_loss_below_unique_repairs: False
- paired_mean_and_median_IoU_improved: False
- onset_or_offset_median_error_improved: False
- extreme_interval_fraction_le_0_10: False
- bootstrap_not_systematically_negative: False

## 10. Final decision

**NO-GO-LVB**

local-valley boundary 无法在两个数据集稳定兑现 boundary oracle headroom。按预先约束停止继续搜索新的 boundary heuristic，并重新评估 inference-only 项目可行性。该结论不否定此前 GO-BOUNDARY 的诊断事实。

## 11. Provenance and limits

- SAMMLV cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl`
- SAMMLV SHA-256：`3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569`
- SAMMLV subjects/videos/GT/k_p：29/79/159/5
- SAMMLV data-quality warnings：2
- CASME_3 cache：`<workspace-root>/RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl`
- CASME_3 SHA-256：`9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda`
- CASME_3 subjects/videos/GT/k_p：94/462/858/17
- CASME_3 data-quality warnings：7
- 原 cache 未修改；GT 只用于 evaluator 和诊断，不用于 boundary 生成。
- LVB 不含 duration cap；极端区间只报告，不后验修复。
- 稳健性结论仅限本次固定 LVB feasibility test，不外推到尚未实现的其他 boundary 方法。
