# EXP-7E — BoostingVRME/SAMMLV Duration Calibration Replication

## 1. Integrity

- Canonical parent replay: **PASS** — TP/FP/FN=42/95/117; F1=0.283783783783784.
- Frozen EXP-7C Boosting engine was invoked directly with only SAMMLV dataset/folds and a new output directory substituted.
- Leakage audit excludes each outer-test subject from calibration; the adapter and conflict functions receive no test GT.

## 2. Training-only calibration

Representative-pair calibration uses the frozen subject-balanced estimator. Alpha median=1.027778; IQR=[1.027778, 1.063492]; range=[1.000000, 1.063492]; clipping lower/upper=0/0.

## 3. Blind duration adapter

All threshold-eligible Boosting P1 events are transformed before the original conflict replay using the unchanged EXP-7B same-parity rounding, deterministic tie-break, fixed center, and legal-boundary logic.
Center preservation: 137/137; boundary clips=0.

## 4. Boosting conflict replay

The archived chronological conflict rule was replayed without modification (`original_conflict_rule_modified=0`). Adapted conflict removals=0; changed conflict pairs=0.

## 5. Canonical vs calibrated

- Canonical F1=0.283784; calibrated F1=0.283784; Delta F1=+0.000000.
- Pooled TP/FP/FN remain 42/95/117 in both arms.

## 6. Event transitions

Rescued=0; lost=0; FP removed=0; new FP=0; match reassigned=0; conflict changed=0.

## 7. Subject-level distribution

Positive/neutral/negative=0/29/0; rescued>lost/equal/less=0/29/0; top-1/top-2/top-5 gain=NA/NA/NA.

## 8. Bootstrap

Paired-subject bootstrap: N=10,000, seed=100; Delta F1=+0.000000; 95% CI=[+0.000000, +0.000000].

## 9. Four-setting duration matrix

`four_setting_duration_matrix.csv` summarizes frozen EXP-7B/C/D artifacts plus this run descriptively only; no prior experiment was rerun and no cross-setting ranking test was performed.

## 10. Gates

- GATE-A: **PASS**.
- GATE-B: **PASS**.
- GATE-C: **PASS**.
- GATE-D: **PASS**.
- GATE-E: **PASS**.
- GATE-F: **PASS**.
- GATE-G: **FAIL**.
- GATE-H: **FAIL**.
- GATE-I: **PASS**.
- GATE-J: **PASS**.

## 11. Final status

**BOOSTING_SAMMLV_DURATION_REPLICATION = NOT_SUPPORTED**

Duration rule mining stops here. Canonical GLSD and EXP-7B/C/D were not modified; no Recognition, STRS, or further geometry experiment was run.
