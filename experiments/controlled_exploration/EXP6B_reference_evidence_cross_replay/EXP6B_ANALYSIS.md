# EXP-6B — Reference × Evidence Cross-Replay

## Integrity gates

- Exact EXP-2A/2C count replay: **PASS**.
- Identity cells (A: R_A+E_7; C: R_C+E_3): **PASS**.
- Every same-reference E_7/E_3 pair has exact candidate peak identity, exact pre-threshold geometry, and `G_7 == G_3` within `1e-12`; see `evidence_isolation_check.csv` and `score_transition.csv`.
- `k_A == k_C`: **PASS**. k is derived from the same outer-training duration statistic, so no k effect is attributed.
- All counterfactual cells are diagnostic only: no outer-test cell was selected and no canonical GLSD or STRS result was changed.

## Cross-replay pooled results

The detailed 8-cell pooled table is `cross_replay_results.csv`; the subject-level values and all P0/P1/P2/oracle counts are in `cross_replay_subject.csv`.

## Direct answers

### 1–4. BoostingVRME / SAMMLV

The fixed-reference evidence comparison produces no TP→B: Anchor-A R_A=0, R_C=0; Anchor-C R_A=0, R_C=0. Instead it recovers B→TP (A: R_A=3, R_C=1; C: R_A=4, R_C=1). Thus nearest-3 does not itself damage B/S recall at either fixed reference.
The fixed-E_7 reference comparison produces TP→B=7 under Anchor-A and 6 under Anchor-C (with B→TP=0 and 2). It therefore explains the historical eight 2A→2C TP→B events principally as a **reference-conditioned scoring/event-set effect**, with the remaining difference attributable to the saved rho/tau operating-point change rather than a nearest-3 TP loss.
Anchor-A effects: evidence@R_A=-0.020622, evidence@R_C=-0.004834, reference@E_7=-0.027012, reference@E_3=-0.011224, interaction=+0.015788. Anchor-C has the same direction (reference@E_7=-0.005632; evidence@R_A=-0.013924), so configuration-selection coupling is present only as a magnitude modifier, not the primary mechanism. B/S verdict: **REFERENCE-DOMINANT**.

### 5. ME-TST+ / SAMMLV

R_A and R_C are foldwise identical here, so the reference factor collapses exactly. E_7→E_3 preserves TP (43→43) and removes more FP than it adds (Anchor-A 127→119; Anchor-C 101→93). This is a pure evidence/threshold effect that increases precision and pooled F1 (+0.006515 and +0.007697).

### 6–7. CAS(ME)3

For M/C, R_A=R_C, so no reference-only candidate or FP reduction exists in this EXP-2A/2C contrast (R1=0). For B/C, reference-only E_7 removes 195 candidate-events across the pooled repeated subject-video cells (75,262→75,067) and has exact R1 final-FP lineage of 29 under Anchor-A and 46 under Anchor-C. The cross-replay therefore separates this reference-pool suppression from E_7→E_3 score filtering; it does not claim that the larger canonical→2C historical reduction arose from nearest-3.

### 8–10. Factor interpretation

E_7→E_3 affects both TP and FP depending on setting: it is precision-improving in M/S, recall-improving but FP-increasing in B/S, and generally FP-increasing in the two CAS(ME)3 settings. There is B/S interaction in F1 (+0.015788 under Anchor-A, +0.002812 under Anchor-C), but it is smaller than the fixed-E_7 reference loss at Anchor-A and does not reverse the reference verdict. EXP-2C's observed benefit must consequently be described setting-specifically: evidence for M/S; reference-pool behavior for B/S and B/C; no outer-test-derived composite method is claimed.

## Interpretation limits

Difference-in-differences is a mechanism diagnostic, not a statistical causal proof. Bootstrap CIs are descriptive paired-subject stability summaries for the predeclared four factor contrasts only.

## Recommended single next experiment

**REFERENCE_SCALE_CANDIDATE_DIAGNOSTIC** for B/S. It should inspect why R_C turns retained B/S TP into B under fixed all-7 evidence; no new scale count, aggregation, fusion, threshold, calibration, or selection is proposed.
