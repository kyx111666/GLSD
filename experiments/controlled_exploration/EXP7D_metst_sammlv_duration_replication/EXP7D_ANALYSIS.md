# EXP-7D — ME-TST+/SAMMLV Duration Calibration Replication

## 1. Integrity

- Canonical replay: **PASS** — TP/FP/FN=(48, 126, 111); F1=0.288288288288288.
- The EXP-7B calibration and adapter helpers are imported directly without behavioural modification.
- Leakage audit: every calibration fold excludes its outer subject; adapter input is stripped of GT and matching status.

## 2. Training-only calibration

Subject-balanced alpha: median=1.090909; IQR=[1.090909, 1.090909]; range=[1.090909, 1.090909]; lower/upper clips=0/0.

## 3. Blind duration adapter

The imported EXP-7B same-parity rounding, deterministic tie-break, fixed-center construction, and legal boundary handling are applied to every canonical P2 event. No ME-TST+ suppression is added.
Center preservation: 174/174; boundary clips=0.

## 4. Canonical vs calibrated result

- Canonical TP/FP/FN=48/126/111; precision=0.275862; recall=0.301887; F1=0.288288.
- Adapted TP/FP/FN=48/126/111; precision=0.275862; recall=0.301887; F1=0.288288.
- Delta TP/FP/FN=+0/+0/+0; Delta F1=+0.000000.

## 5. Event transitions

Rescued=0; lost=0; FP removed=0; new FP=0; match reassigned=0.

## 6. Subject-level distribution

Positive/neutral/negative=0/29/0; rescued>lost/equal/less=0/29/0; top-1/top-2/top-5 gain=NA/NA/NA.

## 7. Bootstrap

Paired-subject bootstrap: N=10000, seed=100; Delta F1=+0.000000; 95% CI=[+0.000000, +0.000000].

## 8. Context against EXP-7B

Frozen CAS(ME)3 primary: F1 0.094164→0.107896, Delta +0.013732, CI +0.003611:+0.023646. SAMMLV replication: F1 0.288288→0.288288, Delta +0.000000, CI [+0.000000, +0.000000]. Descriptive only; no dataset ranking or comparison test was performed.

## 9. Gates

- GATE-A: **PASS**.
- GATE-B: **PASS**.
- GATE-C: **PASS**.
- GATE-D: **PASS**.
- GATE-E: **FAIL**.
- GATE-F: **FAIL**.
- GATE-G: **PASS**.
- GATE-H: **FACTS_REPORTED_NO_NEW_CUTOFF**.

## 10. Final status

**METST_SAMMLV_DURATION_REPLICATION = NOT_SUPPORTED**

Canonical GLSD and EXP-7B were not modified. No Boosting, Recognition, STRS, or further rule mining was run.
