# EXP-2B Analysis

## Outcome

Expanding the local-radius search from `R={1,2,3}` to `R={1,2,3,4,5}`
increased the inner-LOSO search budget from 90 to 150 configurations. Scale,
median aggregation, equal-weight fusion, thresholds, interval adapters, fold
priors, evaluator, and selection tie-breaks remained unchanged. Exact replay
passed for all 246 outer folds.

| Setting | Baseline F1 | Expanded F1 | Delta F1 | Paired 95% CI |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.272727 | -0.015561 | [-0.035340, -0.001137] |
| ME-TST/CAS(ME)3 | 0.094164 | 0.090954 | -0.003210 | [-0.009213, 0.000000] |
| BoostingVRME/SAMMLV | 0.283784 | 0.283784 | 0.000000 | [0.000000, 0.000000] |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.112888 | 0.000000 | [0.000000, 0.000000] |

The paired bootstrap is conditional on the frozen response curves and saved
outer-fold selections. It is supporting uncertainty evidence, not a claim about
end-to-end backbone retraining.

## Selected-radius distribution

| Setting | Baseline distribution | Expanded distribution | Newly added radii selected |
|---|---|---|---:|
| ME-TST/SAMMLV | r2: 29 | r2: 25, r4: 4 | 4/29 (13.79%) |
| ME-TST/CAS(ME)3 | r1: 3, r2: 88, r3: 3 | r1: 2, r2: 86, r3: 1, r4: 4, r5: 1 | 5/94 (5.32%) |
| BoostingVRME/SAMMLV | r1: 29 | r1: 29 | 0/29 (0%) |
| BoostingVRME/CAS(ME)3 | r1: 93, r3: 1 | r1: 93, r3: 1 | 0/94 (0%) |

All nine folds that changed configuration selected a newly added radius. Their
inner-LOSO F1 was slightly higher than the baseline choice, so these were not
configuration-order ties. The outer-fold results nevertheless failed to
generalize: ME-TST lost three true positives on SAMMLV, while on CAS(ME)3 it
lost four true positives and removed twelve false positives.

## Answers to the research questions

### 1. Is the original local structural context too narrow?

No, not under the locked evaluation protocol. New radii were selected in only
9 of 246 outer folds (3.66%), exclusively for ME-TST, and their use did not
improve any setting. The unweighted mean setting F1 decreased from 0.194781 to
0.190088 (`-0.004693`).

### 2. Does CAS(ME)3 benefit?

No. For ME-TST/CAS(ME)3, five folds selected radius 4 or 5, but pooled F1 fell
by 0.003210. Its paired interval reaches zero at the upper bound and no bootstrap
resample has a positive delta, supporting a non-benefit conclusion without
overstating a strictly negative population effect. For BoostingVRME/CAS(ME)3,
no fold selected a new radius and the result was exactly unchanged.

Pooling both CAS(ME)3 backbones, F1 changed from 0.103721 to 0.102193
(`-0.001528`). Thus there is no backbone-consistent or pooled CAS(ME)3 gain.

### 3. Is any degradation clearly supported?

ME-TST/SAMMLV shows the clearest degradation: F1 falls by 0.015561 and the
paired 95% interval excludes zero. The four folds selecting radius 4 improved
their inner-selection objective but reduced pooled outer performance, which is
consistent with selection overfit from enlarging the search space rather than
missing useful local context in the baseline grid.

### 4. Recommendation

Retain `radius={1,2,3}` as the locked primary search space. EXP-2B does not
justify adding radii 4 and 5: the configuration budget grows by 66.7%, two
settings remain unchanged, one declines slightly, and one declines with a
paired interval excluding zero. Report EXP-2B as a controlled sensitivity check
showing that wider local context does not improve CAS(ME)3.

