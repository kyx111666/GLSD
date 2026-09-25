# EXP-5A Stage 1 Scale-Support Audit

## Canonical replay

Selected configurations, inner F1 values, full selected prediction lists, subject TP/FP/FN, and pooled F1 replayed exactly in all four settings: **PASS**.

## Canonical scale-support semantics

The implementation iterates over distinct physical smoothing widths. In every evaluated outer and inner fold here, K=3. The reference width is present in that map; the reference peak therefore self-matches at temporal distance zero. A non-reference width with no peak inside `max(1, round(0.5*k))` appends exactly `0.0`. A matched bilateral local contrast can also legally be exactly zero. Aggregation is `numpy.median`; for K=3 this is the second order statistic, so one positive matched value plus two missing zeros yields zero.

## Selected TP/FP support

| Setting | TP | FP | TP low | FP low | TP hard-zero | FP hard-zero |
|---|---:|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 48 | 126 | 0.187500 | 0.190476 | 0 | 0 |
| ME-TST/CAS(ME)3 | 96 | 1085 | 0.041667 | 0.087558 | 0 | 0 |
| BoostingVRME/SAMMLV | 42 | 95 | 0.000000 | 0.021053 | 0 | 0 |
| BoostingVRME/CAS(ME)3 | 120 | 1148 | 0.033333 | 0.036585 | 0 | 4 |

## Near-threshold and crossing diagnostic

| Setting | GT crossers | non-GT crossers | rescue precision |
|---|---:|---:|---:|
| ME-TST/SAMMLV | 0 | 0 | NA |
| ME-TST/CAS(ME)3 | 0 | 1 | 0.000000 |
| BoostingVRME/SAMMLV | 0 | 1 | 0.000000 |
| BoostingVRME/CAS(ME)3 | 0 | 3 | 0.000000 |

Near-threshold means `0 < selected_tau - S_canonical <= 0.10`. Candidate compatibility uses the unchanged potential interval and the evaluator's inclusive-frame IoU >= 0.5 best-GT rule. It is diagnostic only and never replaces formal matching.

## Subject paired bootstrap

- ME-TST/SAMMLV: TP-low minus FP-low mean=0.067002, 95% CI=[-0.103994, 0.266173], paired subjects=20, P(bootstrap mean < 0)=0.247500.
- ME-TST/CAS(ME)3: TP-low minus FP-low mean=-0.058186, 95% CI=[-0.092634, -0.019530], paired subjects=50, P(bootstrap mean < 0)=0.997600.
- BoostingVRME/SAMMLV: TP-low minus FP-low mean=-0.016291, 95% CI=[-0.041353, 0.000000], paired subjects=19, P(bootstrap mean < 0)=0.880000.
- BoostingVRME/CAS(ME)3: TP-low minus FP-low mean=0.016190, 95% CI=[-0.048744, 0.095426], paired subjects=49, P(bootstrap mean < 0)=0.361100.

## Gate S

**EXP5A_DIAGNOSTIC_GATE_FAILED**

Qualifying setting(s): none.

Reason: No single setting jointly met low-support enrichment, burden, and rescue-precision conditions; A subject bootstrap clearly supported FP low-support enrichment.

Stage 2 executed: **NO**.

Canonical GLSD-v1 was not modified.
