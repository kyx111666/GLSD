# EXP-2A Analysis

## Outcome

Expanding `A` from 3 to 7 values increased the inner-LOSO search budget from 90
to 210 configurations. Exact replay passed for all 246 outer folds.

| Setting | Baseline F1 | Expanded F1 | Delta F1 | Paired 95% CI |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.261398 | -0.026890 | [-0.048646, -0.012079] |
| ME-TST/CAS(ME)3 | 0.094164 | 0.105207 | +0.011043 | [0.003076, 0.019869] |
| BoostingVRME/SAMMLV | 0.283784 | 0.313725 | +0.029942 | [0.005872, 0.058308] |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.118449 | +0.005561 | [-0.002023, 0.012721] |

The paired bootstrap is conditional on the frozen response curves and the saved
outer-fold selections. It is supporting uncertainty evidence, not a claim about
end-to-end backbone retraining.

## Answers to the research questions

### 1. Did F1 improve?

Partly. Expanded search improved three of four settings, but it caused a material
decrease on ME-TST/SAMMLV. The unweighted mean of the four setting F1 values rose
from 0.194781 to 0.199695 (`+0.004914`), but this average hides substantial
cross-setting heterogeneity. The paired intervals exclude zero for the ME-TST
decrease, the ME-TST/CAS(ME)3 increase, and the BoostingVRME/SAMMLV increase;
the BoostingVRME/CAS(ME)3 interval crosses zero.

### 2. Is the gain concentrated in one dataset?

Yes in net terms. Pooling counts across the two backbones within each dataset:

- SAMMLV changes from 0.286169 to 0.286614 (`+0.000446`): essentially neutral,
  because the BoostingVRME gain is offset by the ME-TST loss.
- CAS(ME)3 changes from 0.103721 to 0.111873 (`+0.008152`): both backbones improve.

The largest single-setting gain occurs on BoostingVRME/SAMMLV, so the pattern is
not simply “only CAS(ME)3 benefits”; rather, the consistent dataset-level benefit
is concentrated on CAS(ME)3.

### 3. Does this show that the original scale space was insufficient?

It shows dataset- and backbone-dependent insufficiency, not general
insufficiency. On CAS(ME)3, every one of the 94 outer folds selects a newly added
reference scale: ME-TST selects `1.25`, while BoostingVRME selects `0.75`. Both
settings improve, which is direct evidence that the original reference-scale
choices omitted useful regions for CAS(ME)3.

On SAMMLV, all 29 folds retain an original reference scale (`2.0` for ME-TST and
`1.0` for BoostingVRME). The opposite F1 movements there also show that enlarging
the multiscale evidence set changes the median local evidence even when the
selected reference scale is unchanged. Therefore EXP-2A does not support a claim
that a denser scale set is uniformly better.

### 4. Should expanded search enter the paper?

Not as the new universal primary search on the strength of EXP-2A alone. The
210-configuration search costs 2.33 times the baseline budget and produces a
clear, paired-bootstrap-supported degradation on one of four settings. Replacing
the locked 90-grid would weaken the paper's unified-setting story and would
require rerunning all matched-budget ablations, transfer tests, robustness checks,
and reporting the post-hoc expansion transparently.

The useful paper-level role is a controlled sensitivity result: it supports a
qualified statement that CAS(ME)3 benefits from finer/smaller scales, while a
single expanded set is not uniformly robust across backbones and datasets. If a
future primary expanded grid is desired, it should be fixed prospectively and
validated on independent data before replacing the 90-configuration search.

