# EXP-7B — Training-Only Duration Geometry Validation

## Phase A only

- Scope: ME-TST+ / CAS(ME)3 only. No duration-adjusted interval was generated, applied, matched, or evaluated.
- Interval semantics are discrete closed intervals: `L = end - start + 1`.
- Canonical parent replay: **PASS** — TP/FP/FN=(96, 1085, 762); F1=0.094163805787151, matching archived F1=0.094163805787151.

## Training-only calibration

For each outer subject, only records from all other subjects enter `calibration_pairs`. Their P0 candidates are replayed using that outer fold's frozen config and k. For each training GT, the representative candidate has a peak in the closed GT interval and is selected by maximum IoU, then minimum formal-center distance, then P0 order.

The calibration is subject-balanced: each training subject contributes one median `L_G/L_E`; the outer alpha is the median of those subject medians, then clipped to [0.5, 1.5]. The pooled-event median is audit-only.

## Leakage audit

`calibration_pairs` asserts that every input record has `subject != outer_subject`; `representative_pair` repeats that assertion before reading any GT. Every training P0 is replayed with the current outer fold's stored config and k, not another subject's fold. Runtime audit passed for every outer fold, and no outer-test GT was passed to alpha construction.

## Alpha stability

- Folds: 94; median=0.850000; IQR=[0.835714, 0.850000]; range=[0.800000, 0.903226].
- alpha<1: 94; alpha≈1: 0; alpha>1: 0; clips: lower=0, upper=0.

Phase B executed: **NO**. New formal prediction: **NO**. New formal F1: **NO**. GLSD modified: **NO**. Boosting/secondary replication: **NO**.
