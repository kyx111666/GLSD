# GLSD Controlled Exploration EXP-1B

## Controlled change

The sole algorithmic change is the cross-scale local-evidence aggregation:

- baseline: `L(c) = median(l1, l1.5, l2)`
- EXP-1B: `L_max(c) = max(l1, l1.5, l2)`

Fusion remains `S(c) = (G(c) + L(c)) / 2`. The candidate set, scale set,
radius grid, threshold search, LOSO protocol, evaluator, interval protocol,
configuration budget, and selection tie-break are unchanged. The canonical
GLSD implementation was not modified.

## Main result

| Setting | Baseline F1 | Max-L F1 | Delta F1 | Paired subject-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.258258 | -0.030030 | [-0.064577, -0.006644] |
| ME-TST/CAS(ME)3 | 0.094164 | 0.101130 | +0.006966 | [-0.001612, +0.016768] |
| BoostingVRME/SAMMLV | 0.283784 | 0.268571 | -0.015212 | [-0.044402, +0.015445] |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.107747 | -0.005141 | [-0.012078, +0.001229] |

## Does it improve CAS(ME)3?

Not consistently. ME-TST/CAS(ME)3 improves by 0.006966 F1, but its paired
bootstrap interval crosses zero. BoostingVRME/CAS(ME)3 decreases by 0.005141
F1, and its interval also crosses zero. The direction therefore depends on the
backbone, so EXP-1B does not support adopting strongest-scale aggregation as a
general GLSD replacement.

## Does it create many false positives?

It does for BoostingVRME, but not for ME-TST:

| Setting | Baseline FP | Max-L FP | Delta FP | Delta TP |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 126 | 131 | +5 | -5 |
| ME-TST/CAS(ME)3 | 1085 | 907 | -178 | -2 |
| BoostingVRME/SAMMLV | 95 | 144 | +49 | +5 |
| BoostingVRME/CAS(ME)3 | 1148 | 1267 | +119 | +1 |

The CAS(ME)3 total does not show an aggregate FP explosion (2233 to 2174),
because the ME-TST reduction masks the BoostingVRME increase. The latter is
mechanistically unfavorable: 119 additional FP purchase only one additional
TP. On SAMMLV, FP rises from 221 to 275 across the two backbones, with both F1
scores decreasing.

## Does the mechanism change?

Yes. Median aggregation requires support from the multi-scale consensus,
whereas max aggregation lets one strong scale dominate. Threshold selection
partly compensates for the resulting upward shift in local evidence, but the
compensation is backbone-dependent:

- ME-TST/CAS(ME)3 changes the selected configuration for all 94 subjects and
  collapses to `scale=1.5, tau=0.55, radius=2`. This suppresses 178 FP while
  losing two TP, producing the only CAS(ME)3 point-estimate gain.
- BoostingVRME/CAS(ME)3 changes the selected configuration for only 19 of 94
  subjects; 78 subjects retain `scale=1, tau=0.3, radius=1`. Strongest-scale
  support therefore passes many more candidates without enough threshold
  compensation, adding 119 FP for one TP and lowering F1.
- ME-TST/SAMMLV changes all 29 subject-level configurations and loses both TP
  and F1. BoostingVRME/SAMMLV changes 7 of 29 configurations, gains five TP,
  but adds 49 FP and also loses F1.

Thus the fusion rule is unchanged, but the local-evidence mechanism changes
from consensus-seeking to winner-take-all. The observed effect is not a stable
recall improvement; it is mostly a backbone-dependent shift in candidate
acceptance and threshold compensation.

## Decision

Do not replace median aggregation with strongest-scale aggregation globally.
The isolated ME-TST/CAS(ME)3 gain is small, uncertain under paired bootstrap,
and does not transfer to the other backbone or to SAMMLV.
