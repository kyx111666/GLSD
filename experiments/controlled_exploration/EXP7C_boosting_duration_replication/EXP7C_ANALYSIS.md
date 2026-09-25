# EXP-7C — BoostingVRME/CAS(ME)3 Duration Replication

## 1. Integrity

- Canonical replay: **PASS** — TP/FP/FN=(120, 1148, 738); F1=0.112888052681091.
- Stored canonical outer config, k, scores, threshold, candidate list, and original conflict priority were replayed without selection.
- Leakage audit: all calibration inputs exclude their outer subject; adapter and conflict functions accept no test GT.

## 2. Training-only calibration

Subject-balanced Boosting-only alpha: median=1.250000; IQR=[1.250000, 1.250000]; range=[1.217391, 1.310345]; clips lower/upper=0/0.

## 3. Duration adapter

Closed interval length is `end-start+1`. EXP-7B same-parity rounding and tie-breaking are reused exactly; formal center is fixed. Boundary handling only shortens a request to the largest legal same-center, same-parity interval.
Center-preserved events: 1278/1278; boundary clips=62.

## 4. Boosting conflict replay

The archived chronological Boosting conflict rule was reapplied after adapting all P1 intervals. No threshold, score, priority, or conflict criterion changed. Adapted conflict removals=10; changed conflict pairs=0.

## 5. Canonical vs adapted result

- Canonical TP/FP/FN=120/1148/738; precision=0.094637; recall=0.139860; F1=0.112888.
- Adapted TP/FP/FN=129/1139/729; precision=0.101735; recall=0.150350; F1=0.121355.
- Delta TP/FP/FN=+9/-9/-9; Delta F1=+0.008467.

## 6. Event transitions

Rescued=19; lost=10; FP removed=19; new FP=10; match reassigned=0; conflict-rule changes=0.

## 7. Subject-level generalization

Positive/neutral/negative=14/75/5; rescued>lost/equal/less=14/75/5; top-1/top-2/top-5 TP concentration=0.2222222222222222/0.3333333333333333/0.6666666666666666.

## 8. Bootstrap

Paired-subject bootstrap: N=10000, seed=100; Delta F1=+0.008467; 95% CI=[-0.000798, +0.018157].

## 9. Context against EXP-7B primary result

Frozen ME-TST+ primary F1: 0.094164→0.107896 (Delta +0.013732, CI +0.003611:+0.023646); Boosting replication: 0.112888→0.121355 (Delta +0.008467, CI [-0.000798, +0.018157]). This is descriptive only; no cross-pipeline ranking or significance test was performed.

## 10. Gates

- GATE-A: **PASS**.
- GATE-B: **PASS**.
- GATE-C: **PASS**.
- GATE-D: **PASS**.
- GATE-E: **FAIL**.
- GATE-F: **PASS**.
- GATE-G: **FACTS_REPORTED_NO_NEW_CUTOFF**.

## 11. Final replication status

**BOOSTING_DURATION_REPLICATION = INCONCLUSIVE**

Canonical GLSD and EXP-7B were not modified. No SAMMLV, Recognition, STRS, or further rule mining was run.
