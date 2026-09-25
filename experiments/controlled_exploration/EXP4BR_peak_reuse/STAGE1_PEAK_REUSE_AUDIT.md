# EXP-4B-R Stage 1 Peak-Reuse Audit

## Verdict

`EXP4BR_DIAGNOSTIC_GATE_FAILED`

Canonical selected configurations, prediction lists, and subject TP/FP/FN counts replayed exactly in all four settings. Canonical GLSD-v1 was not modified.

## Primary selected-prediction diagnostic

| Setting | TP | FP | TP reuse prev. | FP reuse prev. | FP-TP | FP with reuse |
|---|---|---|---|---|---|---|
| ME-TST/SAMMLV | 48 | 126 | 0.000000 | 0.000000 | 0.000000 | 0 |
| ME-TST/CAS(ME)3 | 96 | 1085 | 0.000000 | 0.000000 | 0.000000 | 0 |
| BoostingVRME/SAMMLV | 42 | 95 | 0.023810 | 0.021053 | -0.002757 | 2 |
| BoostingVRME/CAS(ME)3 | 120 | 1148 | 0.041667 | 0.025261 | -0.016405 | 29 |

## Subject-paired bootstrap

Subjects lacking either TP or FP were retained with NA fields but excluded from the paired bootstrap; missing values were not replaced by zero.

| setting | eligible_subjects | mean_difference | ci_low | ci_high |
|---|---|---|---|---|
| ME-TST/SAMMLV | 20 | 0.000000 | 0.000000 | 0.000000 |
| ME-TST/CAS(ME)3 | 50 | 0.000000 | 0.000000 | 0.000000 |
| BoostingVRME/SAMMLV | 19 | -0.043860 | -0.157895 | 0.026316 |
| BoostingVRME/CAS(ME)3 | 49 | -0.008232 | -0.059394 | 0.027965 |

## Burden strata

| setting | candidate_reuse_count_stratum | tp_count | fp_count | precision |
|---|---|---|---|---|
| ME-TST/SAMMLV | 0 | 48 | 126 | 0.275862 |
| ME-TST/SAMMLV | 1 | 0 | 0 | NA |
| ME-TST/SAMMLV | >=2 | 0 | 0 | NA |
| ME-TST/CAS(ME)3 | 0 | 96 | 1085 | 0.081287 |
| ME-TST/CAS(ME)3 | 1 | 0 | 0 | NA |
| ME-TST/CAS(ME)3 | >=2 | 0 | 0 | NA |
| BoostingVRME/SAMMLV | 0 | 41 | 93 | 0.305970 |
| BoostingVRME/SAMMLV | 1 | 1 | 2 | 0.333333 |
| BoostingVRME/SAMMLV | >=2 | 0 | 0 | NA |
| BoostingVRME/CAS(ME)3 | 0 | 115 | 1119 | 0.093193 |
| BoostingVRME/CAS(ME)3 | 1 | 5 | 29 | 0.147059 |
| BoostingVRME/CAS(ME)3 | >=2 | 0 | 0 | NA |

## Secondary all-candidate diagnostic

`not_selected` is descriptive only and combines threshold-rejected and conflict-suppressed reference candidates. It is not interpreted as a true-negative class.

| setting | candidate_group | candidate_count | reuse_prevalence | mean_reuse_count | median_reuse_count | mean_L | median_L | mean_S | median_S |
|---|---|---|---|---|---|---|---|---|---|
| ME-TST/SAMMLV | selected | 174 | 0.000000 | 0.000000 | 0.000000 | 0.810188 | 0.827024 | 0.848909 | 0.854511 |
| ME-TST/SAMMLV | not_selected | 3585 | 0.000000 | 0.000000 | 0.000000 | 0.047610 | 0.003032 | 0.048488 | 0.003283 |
| ME-TST/CAS(ME)3 | selected | 1181 | 0.000000 | 0.000000 | 0.000000 | 0.719129 | 0.721146 | 0.751592 | 0.764477 |
| ME-TST/CAS(ME)3 | not_selected | 49583 | 0.000726 | 0.000726 | 0.000000 | 0.009150 | 0.000340 | 0.009297 | 0.000335 |
| BoostingVRME/SAMMLV | selected | 137 | 0.021898 | 0.021898 | 0.000000 | 0.505656 | 0.487940 | 0.731702 | 0.733491 |
| BoostingVRME/SAMMLV | not_selected | 3446 | 0.054266 | 0.055427 | 0.000000 | 0.028008 | 0.001280 | 0.049583 | 0.003526 |
| BoostingVRME/CAS(ME)3 | selected | 1268 | 0.026814 | 0.026814 | 0.000000 | 0.396870 | 0.381190 | 0.599447 | 0.611938 |
| BoostingVRME/CAS(ME)3 | not_selected | 68108 | 0.067085 | 0.069302 | 0.000000 | 0.003045 | 0.000045 | 0.005163 | 0.000095 |

## Reuse contribution to L

| setting | association | candidate_count_with_reuse | reused_evidence_value_count | mean_reused_L_scale_value | median_reused_L_scale_value | mean_candidate_overall_L | median_candidate_overall_L | mean_candidate_S | median_candidate_S | candidates_where_reuse_changes_L |
|---|---|---|---|---|---|---|---|---|---|---|
| ME-TST/SAMMLV | TP | 0 | 0 | NA | NA | NA | NA | NA | NA | 0 |
| ME-TST/SAMMLV | FP | 0 | 0 | NA | NA | NA | NA | NA | NA | 0 |
| ME-TST/CAS(ME)3 | TP | 0 | 0 | NA | NA | NA | NA | NA | NA | 0 |
| ME-TST/CAS(ME)3 | FP | 0 | 0 | NA | NA | NA | NA | NA | NA | 0 |
| BoostingVRME/SAMMLV | TP | 1 | 1 | 0.142993 | 0.142993 | 0.696633 | 0.696633 | 0.843354 | 0.843354 | 0 |
| BoostingVRME/SAMMLV | FP | 2 | 2 | 0.233441 | 0.233441 | 0.383485 | 0.383485 | 0.688289 | 0.688289 | 0 |
| BoostingVRME/CAS(ME)3 | TP | 5 | 5 | 0.300095 | 0.304421 | 0.280794 | 0.304421 | 0.548813 | 0.536073 | 3 |
| BoostingVRME/CAS(ME)3 | FP | 29 | 29 | 0.157025 | 0.125600 | 0.207723 | 0.185706 | 0.469593 | 0.436535 | 14 |

## Gate R

Qualifying CAS(ME)3 setting(s): none.

Stage 2 executed: **NO**. The preregistered gate did not pass, so the peak-reuse route is closed and no alignment variant was run.

```text
================================
EXP-4B-R DIAGNOSTIC VERDICT
================================

Canonical replay:
PASS

Peak reuse confirmed:
YES

ME-TST/CAS(ME)3
FP reuse prevalence: 0.000000
TP reuse prevalence: 0.000000

BoostingVRME/CAS(ME)3
FP reuse prevalence: 0.025261
TP reuse prevalence: 0.041667

Any CAS(ME)3 FP enrichment >= 5pp: NO

Subject-bootstrap direction supported: NO

Diagnostic Gate R:
FAIL

Stage 2 executed:
NO

Canonical GLSD modified:
NO

Reason:
No CAS(ME)3 setting satisfied every preregistered Gate R condition jointly.

================================
```
