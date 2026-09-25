# EXP-4A Alignment Tolerance Search

## 1. Hypothesis

EXP-4A tests whether the fixed cross-scale peak-alignment tolerance is limiting canonical GLSD-v1. It is a setting-only experiment.

## 2. Protocol

The sole added parameter is `delta = max(1, round(gamma*k))`, with `gamma in {0.25, 0.50, 0.75}`. The canonical reference scales, radii, ten thresholds, median aggregation, missing-scale zero, equal G/L fusion, event geometry, outer/inner LOSO, and tie-breaks are unchanged. The main search budget is 270 configurations.

## 3. Baseline Replay

`gamma=0.50` recovered every canonical outer-fold selected configuration, inner F1, and TP/FP/FN exactly for all four settings: **PASS**.

## 4. Main Results

| Setting | Canonical | EXP-4A | Delta | CI |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.288288 | +0.000000 | [+0.000000, +0.000000] |
| ME-TST/CAS(ME)3 | 0.094164 | 0.092977 | -0.001187 | [-0.003350, +0.000000] |
| BoostingVRME/SAMMLV | 0.283784 | 0.283784 | +0.000000 | [+0.000000, +0.000000] |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.109680 | -0.003208 | [-0.007480, +0.000000] |

## 5. Fixed-Gamma Diagnostic

| Setting | gamma=.25 | gamma=.50 | gamma=.75 | Selected |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.249221 | 0.288288 | 0.265861 | 0.288288 |
| ME-TST/CAS(ME)3 | 0.092022 | 0.094164 | 0.092034 | 0.092977 |
| BoostingVRME/SAMMLV | 0.271429 | 0.283784 | 0.283784 | 0.283784 |
| BoostingVRME/CAS(ME)3 | 0.108811 | 0.112888 | 0.111268 | 0.109680 |

## 6. Selected Gamma Distribution

- ME-TST/SAMMLV: .25=0 (0.000), .50=29 (1.000), .75=0 (0.000).
- ME-TST/CAS(ME)3: .25=1 (0.011), .50=90 (0.957), .75=3 (0.032).
- BoostingVRME/SAMMLV: .25=2 (0.069), .50=27 (0.931), .75=0 (0.000).
- BoostingVRME/CAS(ME)3: .25=3 (0.032), .50=91 (0.968), .75=0 (0.000).

## 7. TP/FP/FN

- ME-TST/SAMMLV: canonical 48/126/111 -> EXP-4A 48/126/111 (delta +0/+0/+0).
- ME-TST/CAS(ME)3: canonical 96/1085/762 -> EXP-4A 94/1070/764 (delta -2/-15/+2).
- BoostingVRME/SAMMLV: canonical 42/95/117 -> EXP-4A 42/95/117 (delta +0/+0/+0).
- BoostingVRME/CAS(ME)3: canonical 120/1148/738 -> EXP-4A 115/1124/743 (delta -5/-24/+5).

## 8. Alignment Mechanism

- ME-TST/SAMMLV: matched rate 0.679170 -> 0.679170; missing rate 0.320830 -> 0.320830; reuse 0 -> 0; collisions 0 -> 0.
  - No outer fold selected the stricter gamma=0.25 setting.
  - No outer fold selected the wider gamma=0.75 setting.
- ME-TST/CAS(ME)3: matched rate 0.766459 -> 0.776097; missing rate 0.233541 -> 0.223903; reuse 18 -> 455; collisions 18 -> 455.
  - On 1 fold(s) selecting the stricter gamma=0.25, matched rate changed 0.833039 -> 0.560954; fold-pooled changes were TP +0, FP +0, FN +0.
  - On 3 fold(s) selecting the wider gamma=0.75, matched rate changed 0.696167 -> 0.877502; fold-pooled changes were TP -2, FP -15, FN +2.
- BoostingVRME/SAMMLV: matched rate 0.737790 -> 0.730114; missing rate 0.262210 -> 0.269886; reuse 88 -> 87; collisions 88 -> 87.
  - On 2 fold(s) selecting the stricter gamma=0.25, matched rate changed 0.817259 -> 0.747462; fold-pooled changes were TP +0, FP +0, FN +0.
  - No outer fold selected the wider gamma=0.75 setting.
- BoostingVRME/CAS(ME)3: matched rate 0.750555 -> 0.727009; missing rate 0.249445 -> 0.272991; reuse 2301 -> 2170; collisions 2301 -> 2170.
  - On 3 fold(s) selecting the stricter gamma=0.25, matched rate changed 0.738143 -> 0.376108; fold-pooled changes were TP -5, FP -24, FN +5.
  - No outer fold selected the wider gamma=0.75 setting.

## 9. Candidate Transition

- ME-TST/SAMMLV: rescued GT=0, removed FP=0, lost GT=0, new FP=0; mean delta L=0.0, median delta L=0.0 over 3759 shared candidates.
- ME-TST/CAS(ME)3: rescued GT=1, removed FP=50, lost GT=3, new FP=35; mean delta L=0.00010870206493113404, median delta L=0.0 over 49472 shared candidates.
- BoostingVRME/SAMMLV: rescued GT=0, removed FP=0, lost GT=0, new FP=0; mean delta L=-1.6342873698660475e-05, median delta L=0.0 over 3583 shared candidates.
- BoostingVRME/CAS(ME)3: rescued GT=0, removed FP=24, lost GT=5, new FP=0; mean delta L=0.000444139396630608, median delta L=0.0 over 69376 shared candidates.

Observed matched/missing changes above—not the theoretical direction alone—are the basis for interpretation. Narrower selected tolerances are credited only where matched rate actually falls; wider tolerances only where it rises.

## 10. Gate

**EXP4A_FAILED**

Pre-registered checks: canonical_mean=0.19478098263507845; selected_mean=0.19368245456648558; no_significant_regression=True; significant_improvement=False; fp_explosion=False; noninferior_count=2; clear_mechanism_improvement=False.

## 11. Recommendation

**B. Retain canonical GLSD-v1**

Canonical GLSD-v1 was not modified. No finer gamma grid, EXP-4B, or automatic canonical upgrade was run.
