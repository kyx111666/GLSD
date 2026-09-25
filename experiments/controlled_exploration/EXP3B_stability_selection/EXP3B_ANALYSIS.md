# EXP-3B Stability-Aware Configuration Selection

## 1. Hypothesis

Pure pooled-F1 argmax may be sensitive to inner-subject sampling variation. EXP-3B tests whether the fixed one-standard-error score selects a more stable operating point without changing the decoder or candidate configurations.

## 2. Protocol

For every existing EXP-2C configuration and outer fold, subject-level F1 is computed on each inner validation subject. Selection maximizes `mean_f1 - std_f1/sqrt(m)`, with sample standard deviation (`ddof=1`). The penalty coefficient is fixed at 1 for all datasets and backbones, was not searched, and adds no configurations. Pooled inner F1 is retained for diagnosis. Exact StableScore ties use the frozen EXP-2C post-score tie-break: higher pooled precision, fewer pooled FP, then lower grid order.

All four settings use the same rule. The 210 configurations, frozen responses, 246 outer folds, inner/outer LOSO splits, decoder, evaluator, GT matching, bootstrap (N=10,000; seed=100), and event geometry are unchanged.

## 3. Main Results

| Setting | 2C | 3B | Delta | 95% CI vs 2C |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.291525 | 0.223325 | -0.068200 | [-0.097759, -0.035772] |
| ME-TST/CAS(ME)3 | 0.104548 | 0.079339 | -0.025209 | [-0.037515, -0.013664] |
| BoostingVRME/SAMMLV | 0.277778 | 0.280822 | 0.003044 | [-0.001131, 0.007786] |
| BoostingVRME/CAS(ME)3 | 0.121530 | 0.109507 | -0.012022 | [-0.020445, -0.002617] |

Four-setting mean F1: EXP-2C `0.198845`; EXP-3B `0.173248`.

## 4. TP/FP/FN

| Setting | 2C TP/FP/FN | 3B TP/FP/FN | Delta TP/FP/FN |
|---|---:|---:|---:|
| ME-TST/SAMMLV | 43/93/116 | 45/199/114 | +2/+106/-2 |
| ME-TST/CAS(ME)3 | 100/955/758 | 72/885/786 | -28/-70/+28 |
| BoostingVRME/SAMMLV | 40/89/119 | 41/92/118 | +1/+3/-1 |
| BoostingVRME/CAS(ME)3 | 116/935/742 | 110/1041/748 | -6/+106/+6 |

## 5. Selection Stability

| Setting | Change rate | Mean std 2C | Mean std 3B | Mean SE 2C | Mean SE 3B | Effect |
|---|---:|---:|---:|---:|---:|---|
| ME-TST/SAMMLV | 100.0% | 0.278394 | 0.277323 | 0.052611 | 0.052409 | LOWER_VARIABILITY |
| ME-TST/CAS(ME)3 | 98.9% | 0.172931 | 0.193279 | 0.017932 | 0.020042 | NO_STABILITY_EFFECT |
| BoostingVRME/SAMMLV | 24.1% | 0.249869 | 0.248505 | 0.047221 | 0.046963 | LOWER_VARIABILITY |
| BoostingVRME/CAS(ME)3 | 100.0% | 0.186059 | 0.183931 | 0.019293 | 0.019073 | LOWER_VARIABILITY |

Across all 246 outer folds, mean selected-config std changed from `0.199450` to `0.206125`, and mean SE from `0.025993` to `0.026661`.

## 6. Configuration Changes

| Setting | Same | Changed | a0 changed | r changed | tau changed |
|---|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0 | 29 | 27 | 29 | 29 |
| ME-TST/CAS(ME)3 | 1 | 93 | 21 | 16 | 93 |
| BoostingVRME/SAMMLV | 22 | 7 | 4 | 0 | 7 |
| BoostingVRME/CAS(ME)3 | 0 | 94 | 93 | 94 | 92 |

## 7. BoostingVRME/SAMMLV Analysis

F1 changed from `0.277778` to `0.280822`. Counts changed from `40/89/119` to `41/92/118`. EXP-3B recovered TP and reduced FN, but increased FP. This satisfies Gate A condition 4(b) as a local repair because F1 increased with TP recovery and FN reduction; however, the paired CI crosses zero and F1 remains below the canonical baseline.

### Secondary selection-margin analysis

Folds are labeled `generalizing` when EXP-2C outer subject-F1 is at least the canonical baseline outer subject-F1; otherwise they are `poor-generalizing`. This zero-parameter label is descriptive only.

| Setting | Fold class | n | Mean margin | Median margin |
|---|---|---:|---:|---:|
| ME-TST/SAMMLV | generalizing | 24 | 0.008938 | 0.009157 |
| ME-TST/SAMMLV | poor-generalizing | 5 | 0.012894 | 0.013829 |
| ME-TST/CAS(ME)3 | generalizing | 87 | 0.000577 | 0.000600 |
| ME-TST/CAS(ME)3 | poor-generalizing | 7 | 0.000636 | 0.000633 |
| BoostingVRME/SAMMLV | generalizing | 23 | 0.000765 | 0.000455 |
| BoostingVRME/SAMMLV | poor-generalizing | 6 | 0.002874 | 0.002436 |
| BoostingVRME/CAS(ME)3 | generalizing | 86 | 0.000168 | 0.000158 |
| BoostingVRME/CAS(ME)3 | poor-generalizing | 8 | 0.000526 | 0.000330 |

## 8. Gate

EXP3B_FAILED

- No paired-CI regression vs EXP-2C: NO
- Settings non-decreasing vs EXP-2C: 1/4
- CAS(ME)3 gains preserved: NO
- BoostingVRME/SAMMLV repaired: YES
- Four-setting mean at least EXP-2C: NO
- Average inner variability reduced: NO
- One fixed rule used for every setting: YES

## 9. Recommendation

B. Retain EXP-2C

No additional stability penalty, selection objective, or configuration search is authorized by this experiment.
