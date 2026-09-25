# EXP-7B Phase B — Blind Outer-Test Duration Application

## 1. Integrity

- Canonical parent replay: **PASS** — TP/FP/FN=(96, 1085, 762); F1=0.094163805787151.
- Alpha source: Phase-A `outer_alpha_calibration.csv`, read once; no alpha was recomputed or overwritten.
- Leakage audit: adapter accepts only stripped event geometry, frozen alpha, and video length. Runtime schema assertions passed for every fold; outer-test GT was not passed to the adapter.

## 2. Adapter definition

Closed intervals use `L=end-start+1`. Target lengths retain original parity. Ties choose the length closest to the continuous target, then closest to original length, then the smaller length. The adapter preserves the exact integer/half-integer center; a boundary overflow would shorten only to the maximum legal same-center, same-parity length.
Center preservation: **PASS** (1181/1181); boundary-clipped events: 0.

## 3. Canonical vs adapted pooled result

- Canonical: TP/FP/FN=96/1085/762; precision=0.081287; recall=0.111888; F1=0.094164.
- Adapted: TP/FP/FN=110/1071/748; precision=0.093141; recall=0.128205; F1=0.107896.
- Delta: TP=+14; FP=-14; FN=-14; precision=+0.011854; recall=+0.016317; F1=+0.013732.

## 4. Subject-level generalization

Positive/neutral/negative delta-F1 subjects: 14/76/4. Rescued>lost: 14; lost>rescued: 4. Gain concentration (top-1/top-2/top-5 TP): 0.2857142857142857/0.42857142857142855/0.6428571428571429.

## 5. Event transition accounting

Rescued=20; lost=6; FP removed=20; new FP=6; match reassigned=0.

## 6. Bootstrap

Paired-subject bootstrap: N=10000, seed=100; point delta F1=+0.013732; 95% CI=[+0.003611, +0.023646].

## 7. Gates

- GATE-A: **PASS**.
- GATE-B: **PASS**.
- GATE-C: **PASS**.
- GATE-D: **PASS**.
- GATE-E: **PASS**.
- GATE-F: **PASS**.
GATE-E passes directly because TP increases while FP decreases; no materiality threshold is introduced. GATE-F passes because the reported top-2 contribution is 42.9%, so 57.1% of net TP gain lies outside two subjects; no additional concentration cutoff is used.

**TRAINING_ONLY_DURATION_GENERALIZATION_SUPPORTED = YES**

Canonical GLSD was not modified. No Boosting, SAMMLV, Recognition, STRS, alpha search, or secondary replication was run.
