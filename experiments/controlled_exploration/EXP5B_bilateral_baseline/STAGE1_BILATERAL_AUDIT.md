# EXP-5B Stage 1 Bilateral Asymmetry Audit

## Canonical replay

All four settings exactly reproduced selected configurations, inner F1 values, full prediction lists, subject TP/FP/FN, and pooled F1: **PASS**.

## Diagnostic definition

Only `B_can=max(B_L,B_R)` was replaced diagnostically by `B_mean=(B_L+B_R)/2`. Stage 1 did not reselect configurations, change thresholds, or produce alternative predictions. Candidate-level raw and relative asymmetry are the medians across that candidate's canonically aligned matched scale pairs; missing scales are excluded from asymmetry summaries but remain zero in both local-evidence vectors.

## Selected TP/FP diagnostic

| Setting | TP strong prev | FP strong prev | Difference | TP mean delta S | FP mean delta S |
|---|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.104167 | 0.103175 | 0.000992 | 0.020442 | 0.018570 |
| ME-TST/CAS(ME)3 | 0.135417 | 0.100461 | 0.034956 | 0.015372 | 0.013254 |
| BoostingVRME/SAMMLV | 0.309524 | 0.326316 | -0.016792 | 0.040371 | 0.043189 |
| BoostingVRME/CAS(ME)3 | 0.425000 | 0.418118 | 0.006882 | 0.050137 | 0.052591 |

## Near-threshold crossing

| Setting | GT crossers | non-GT crossers | Potential rescue precision |
|---|---:|---:|---:|
| ME-TST/SAMMLV | 0 | 12 | 0.000000 |
| ME-TST/CAS(ME)3 | 2 | 18 | 0.100000 |
| BoostingVRME/SAMMLV | 4 | 28 | 0.125000 |
| BoostingVRME/CAS(ME)3 | 8 | 85 | 0.086022 |

## Subject-level paired bootstrap

- ME-TST/SAMMLV: mean TP-FP=0.013214, 95% CI=[-0.108333, 0.145726], paired subjects=20.
- ME-TST/CAS(ME)3: mean TP-FP=0.017624, 95% CI=[-0.035699, 0.085916], paired subjects=50.
- BoostingVRME/SAMMLV: mean TP-FP=-0.122180, 95% CI=[-0.351143, 0.103136], paired subjects=19.
- BoostingVRME/CAS(ME)3: mean TP-FP=-0.028999, 95% CI=[-0.158933, 0.101711], paired subjects=49.

## Gate B

**EXP5B_DIAGNOSTIC_GATE_FAILED**

No setting had TP strongly-penalized prevalence at least 5 percentage points above FP; No setting had at least 10 near-threshold GT-compatible crossers; No setting reached potential rescue precision 0.20.

Stage 2 executed: **NO**.

Canonical GLSD-v1 was not modified.
