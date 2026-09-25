# EXP-4B Stage 1 Alignment Audit

## Verdict

`EXP4B_DIAGNOSTIC_GATE_FAILED`

Implementation Applicability Gate: **PASS**. Canonical alignment is not one-to-one because a non-reference peak may support multiple reference candidates, and removing that reuse can change candidate-local evidence. However, the task-defined candidate-scale collision is structurally absent, so Gate D's FP collision-enrichment condition fails.

Stage 2 executed: **NO**.

## Canonical code semantics

The audited implementation is `historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/unified_persistence.py`.

1. **PEAK_REUSE is possible.** `CurveFeatures.evidence` loops independently over reference candidates (lines 99–109). It does not mark a selected non-reference peak as used, so the same peak can be selected by several candidates.
2. **CANDIDATE_COLLISION is impossible in the canonical implementation.** For a candidate-scale pair, admissible peaks are collected at line 103, but exactly one `chosen` index is produced by `min(...)` at line 107 and exactly one value is appended at line 108. Therefore actual `n_match(c,a)` is 0 or 1.
3. **Local evidence rule.** The selected peak is nearest in time; ties use larger normalized bilateral local contrast (`-values[i]`). A missing scale contributes zero. The per-scale values, including the unchanged reference-scale self-match, are aggregated by the median at line 109. There is no overwrite, last-assignment behavior, or multi-peak aggregation.
4. **EXP-4A audit naming.** `run_exp4a.py:219–251` first makes one chosen-peak assignment per candidate. Its `collision_count` is the number of non-reference peaks whose assignment multiplicity is greater than one; its `reuse_count` is the number of assignments beyond the first, summed across those reused peaks. Thus both quantities describe one-peak-to-many-candidate reuse, not multiple-peaks-to-one-candidate collision. They are equal in the recorded settings because every reused peak has degree exactly 2; a degree-3 reused peak would contribute 1 to `collision_count` but 2 to `reuse_count`.

## Prediction-to-candidate linkage

Available and verified. The canonical evaluator emits each kept prediction's exact reference `peak`; that peak is a key in the selected reference-candidate array. Replayed prediction lists and per-subject TP/FP/FN were required to equal the archived canonical `pure` outputs exactly before a prediction was included.

## Collision diagnostic

| Setting | TP | FP | TP collision prev. | FP collision prev. | Difference | FP collision predictions |
|---|---:|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 48 | 126 | 0.000000 | 0.000000 | 0.000000 | 0 |
| ME-TST/CAS(ME)3 | 96 | 1085 | 0.000000 | 0.000000 | 0.000000 | 0 |
| BoostingVRME/SAMMLV | 42 | 95 | 0.000000 | 0.000000 | 0.000000 | 0 |
| BoostingVRME/CAS(ME)3 | 120 | 1148 | 0.000000 | 0.000000 | 0.000000 | 0 |

All collision burdens and excess-match summaries are zero because canonical candidate-scale cardinality is at most one. `prevalence_ratio` is `NA` because TP collision prevalence is zero; no division by zero was performed.

For transparency, `collision_diagnostic.csv` also reports peak-reuse association prevalence. This auxiliary reuse statistic does not replace the preregistered collision definition and was not used to pass Gate D.

## Subject-level uncertainty

Per subject, TP and FP collision prevalence are computed only when the corresponding denominator is positive. A subject missing either TP or FP is recorded as `NA` and excluded from the paired bootstrap; zeros are not imputed. The bootstrap resamples complete-case subjects with replacement, computes the mean paired subject difference, uses 10,000 draws, and seed 100. Because all eligible subject differences are exactly zero, every setting's 95% CI is [0, 0]. Details and exclusions are in `collision_subject_bootstrap.csv`.

## Gate D

- Implementation Applicability Gate: PASS (peak reuse exists and can change L under unique assignment).
- CAS(ME)3 FP collision enrichment by at least 0.05: NO.
- Same-setting FP collision prevalence greater than TP with at least 20 FP collision predictions: NO.
- Both CAS(ME)3 settings strongly opposite: NO (both are exactly tied at zero).
- Reuse can affect L: YES; task-defined candidate collision itself is absent.

Gate D: **FAIL**. Under the preregistered definitions, Stage 2 is not authorized. No EXP-4B alignment modification, baseline replay for Stage 2, configuration search, or EXP-4C run was performed. Canonical GLSD-v1 was not modified.

## Auxiliary all-candidate reuse audit

| Setting | Reused non-reference peaks | Reuse excess assignments | Max reuse degree | Candidate-scale collisions |
|---|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0 | 0 | 1 | 0 |
| ME-TST/CAS(ME)3 | 18 | 18 | 2 | 0 |
| BoostingVRME/SAMMLV | 97 | 97 | 2 | 0 |
| BoostingVRME/CAS(ME)3 | 2377 | 2377 | 2 | 0 |

These are the assignments made by the canonical evidence code, including its local-contrast tie-break. EXP-4A's separate audit helper used peak-array index (earlier time) as the distance-tie-break instead; it reported 0/18/88/2301 reused peaks for the four settings, versus 0/18/97/2377 here. This audit-helper discrepancy does not affect EXP-4A predictions or the present candidate-collision result. Equality of reused-peak and excess-assignment totals confirms that every observed reused peak had degree 2.

## Final terminal verdict

```text
================================
EXP-4B DIAGNOSTIC VERDICT
================================

Implementation applicability:
PASS

CAS(ME)3 FP collision enrichment:
NO

Diagnostic Gate D:
FAIL

Stage 2 executed:
NO

Reason:
Canonical code permits peak reuse but never assigns multiple peaks to one candidate-scale pair; therefore task-defined collision burden is zero and the preregistered FP-enrichment thresholds cannot be met.

================================
```
