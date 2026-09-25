# EXP-3A Rank-Calibrated Structural Evidence

## 1. Hypothesis

EXP-3A tests whether cross-response-source differences are caused by score-calibration mismatch. It does not introduce new structural evidence: candidate generation, G, L, nearest-3 evidence, median aggregation, and event geometry are inherited unchanged from EXP-2C.

## 2. Protocol

Within each video candidate set, ties receive average rank `R`. The fixed transforms are `G_rank=(R(G)-0.5)/n` and `L_rank=(R(L)-0.5)/n`, followed by the unchanged equal fusion `S_rank=(G_rank+L_rank)/2`. For `n=1`, every calibrated score is 0.5; for `n=0`, the prediction remains empty. No GT or cross-video statistic enters calibration.

The search remains 7 reference scales × 3 radii × 10 locked EXP-2C thresholds = 210 configurations. Outer LOSO, inner LOSO, tie-breaks, subject splits, seed 100, and 10,000 subject-level paired bootstrap resamples are unchanged.

## 3. Calibration Sanity

The no-GT preflight audited 84 setting/reference/radius rows. Candidate ranking preservation: **PASS**. Constant or singleton score vectors have undefined Spearman correlation and are counted separately; every defined per-video Spearman value is summarized in `calibration_sanity.csv`.

## 4. Main Results

| Setting | Baseline | 2C | 3A | 3A-2C | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.291525 | 0.139963 | -0.151562 | [-0.223300, -0.091237] |
| ME-TST/CAS(ME)3 | 0.094164 | 0.104548 | 0.037680 | -0.066867 | [-0.085002, -0.050596] |
| BoostingVRME/SAMMLV | 0.283784 | 0.277778 | 0.173141 | -0.104637 | [-0.151951, -0.062997] |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.121530 | 0.038914 | -0.082616 | [-0.104788, -0.059588] |

## 5. TP/FP/FN

- ME-TST/SAMMLV: EXP-2C 43/93/116 → EXP-3A 76/851/83 (ΔTP +33, ΔFP +758, ΔFN -33).
- ME-TST/CAS(ME)3: EXP-2C 100/955/758 → EXP-3A 244/11849/614 (ΔTP +144, ΔFP +10894, ΔFN -144).
- BoostingVRME/SAMMLV: EXP-2C 40/89/119 → EXP-3A 78/664/81 (ΔTP +38, ΔFP +575, ΔFN -38).
- BoostingVRME/CAS(ME)3: EXP-2C 116/935/742 → EXP-3A 283/13404/575 (ΔTP +167, ΔFP +12469, ΔFN -167).

For BoostingVRME/SAMMLV, the mechanism classification is **D. increased FP (despite any recall gain)**.

## 6. Threshold Behavior

- ME-TST/SAMMLV: EXP-2C `0.55:29`; EXP-3A `0.75:29`.
- ME-TST/CAS(ME)3: EXP-2C `0.5:91, 0.55:3`; EXP-3A `0.75:94`.
- BoostingVRME/SAMMLV: EXP-2C `0.6:8, 0.65:21`; EXP-3A `0.75:29`.
- BoostingVRME/CAS(ME)3: EXP-2C `0.3:2, 0.6:82, 0.65:10`; EXP-3A `0.75:94`.

Threshold concentration is diagnostic only; it is not used as evidence of superiority and the threshold grid was not adapted to ranked scores.

## 7. Score Distribution

The transform replaces within-video magnitudes with average-rank percentiles separately for G and L. `score_distribution_before_after.csv` reports pooled no-GT preflight summaries, while `selected_candidate_fraction.csv` checks whether the locked thresholds behave like a nearly fixed top-fraction selector.
- ME-TST/SAMMLV: mean selected fraction 0.058287 → 0.250435; potential top-fraction behavior: YES.
- ME-TST/CAS(ME)3: mean selected fraction 0.024440 → 0.240167; potential top-fraction behavior: YES.
- BoostingVRME/SAMMLV: mean selected fraction 0.056408 → 0.237731; potential top-fraction behavior: YES.
- BoostingVRME/CAS(ME)3: mean selected fraction 0.016263 → 0.222642; potential top-fraction behavior: YES.

## 8. Failure Analysis

- ME-TST/SAMMLV declined by -0.151562. The count change was ΔTP +33, ΔFP +758, ΔFN -33; the paired CI was [-0.223300, -0.091237].
- ME-TST/CAS(ME)3 declined by -0.066867. The count change was ΔTP +144, ΔFP +10894, ΔFN -144; the paired CI was [-0.085002, -0.050596].
- BoostingVRME/SAMMLV declined by -0.104637. The count change was ΔTP +38, ΔFP +575, ΔFN -38; the paired CI was [-0.151951, -0.062997].
- BoostingVRME/CAS(ME)3 declined by -0.082616. The count change was ΔTP +167, ΔFP +12469, ΔFN -167; the paired CI was [-0.104788, -0.059588].

## 9. Gate

**EXP3A_FAILED**

Pre-registered checks: no_significant_regression=NO; at_least_three_noninferior=NO; cas_gains_preserved=NO; boostingvrme_sammlv_repaired=NO; mean_f1_noninferior=NO; global_parameter_free_formula=YES.

## 10. Recommendation

**B. Retain EXP-2C**

Canonical GLSD-v1 remains unchanged. No additional calibration variant or threshold expansion was run.
