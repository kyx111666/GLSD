# EXP-5B Python Code Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-18
> **Script reviewed**: `controlled_exploration/EXP5B_bilateral_baseline/run_exp5b.py`

## Pass Items

1. ✅ `run_exp5b.py:84-94` reconstructs the canonical `max(B_L,B_R)` numerator and the sole permitted arithmetic-mean alternative while retaining the same center, neighborhoods, positivity constraint, and per-scale range denominator.
2. ✅ `run_exp5b.py:119-176` preserves canonical alignment tolerance, missing-scale zero insertion, reference self-match, and NumPy median aggregation. Every one of the 127,482 emitted candidate rows had at least one matched scale pair.
3. ✅ `run_exp5b.py:199-204` compares the audited reconstruction against the canonical source implementation for every evaluated selected video/configuration at absolute tolerance `1e-15`; the completed run passed.
4. ✅ `run_exp5b.py:207-274` keeps Stage 1 on canonical selected configurations and predictions, uses the fixed window `0 < tau-S <= 0.10`, the evaluator-derived IoU-compatible label, fixed strong-penalty cutoff `delta_S >= 0.05`, and no alternative prediction call. Output audit found 127,482 rows, 116,976 changed scores, no negative deltas, and exact `delta_S=delta_L/2` throughout.
5. ✅ `run_exp5b.py:278-296,362-380` implements 10,000-resample seed-100 subject-paired bootstrap and excludes a subject whenever either TP or FP denominator is absent instead of filling the missing prevalence with zero.
6. ✅ `run_exp5b.py:384-427` translates Diagnostic Gate B with fixed thresholds: 5 percentage-point enrichment, at least 10 GT-compatible crossers in a setting, rescue precision at least 0.20, positive bootstrap upper bounds, non-invariance, and the all-setting non-GT risk check.
7. ✅ `run_exp5b.py:606-618` performs exact replay of selected configuration, inner F1, full prediction lists, subject counts, and pooled counts/F1 against the frozen archive. All four settings passed with TP/FP totals matching the selected candidate audit.
8. ✅ `run_exp5b.py:668-691` writes the failed gate to `protocol.json` and returns before the Stage-2 loop. The completed artifact audit confirmed that `results.csv`, `meanbaseline_mechanism.csv`, `selected_config_per_subject.csv`, and `EXP5B_ANALYSIS.md` are absent.
9. ✅ The executed script SHA-256 matches `protocol.json`, the configuration budget is 90, the forbidden-variant list is empty, and `canonical_glsd_modified` is false.

## Failed / Repaired Items

No post-run defect required repair.

## Constraint Direction Review

There are no physical optimization constraints. The relevant directional diagnostic gates are listed below and match the preregistration.

| File:line | Direction | LHS | RHS | Expected meaning |
|---|---|---|---|---|
| `run_exp5b.py:236` | `0 < x <=` | `tau-S_canonical` | `0.10` | fixed near-threshold window |
| `run_exp5b.py:262` | `>=` | `delta_S` | `0.05` | fixed strongly-penalized cutoff |
| `run_exp5b.py:388` | `>=` | `TP_prev-FP_prev` | `0.05` | required TP enrichment |
| `run_exp5b.py:390` | `>=` | `GT-compatible crossers` | `10` | required rescue burden |
| `run_exp5b.py:393` | `>=` | `potential rescue precision` | `0.20` | required selectivity |
| `run_exp5b.py:396` | `>` | `bootstrap CI upper bound` | `0` | negative direction not clearly supported |

## Remaining Risks

- Candidate-level asymmetry strata require aggregation of the preregistered matched-pair quantity. The implementation uses the median across canonically aligned matched scale pairs and records this choice in both the Stage-1 report and protocol; the full pair-level values remain in `bilateral_diagnostic_scores.csv`.
- The experiment intentionally depends on frozen OOF caches and archived canonical outputs at repository-relative locations.
- The Stage-2 branch is implemented but was not executed because Diagnostic Gate B failed; it was syntax-checked and statically inspected, not empirically exercised.

## Run Instructions

```bash
python3 controlled_exploration/EXP5B_bilateral_baseline/run_exp5b.py
```

## Expected Outputs for This Run

- `CANONICAL_LOCAL_CONTRAST.md`
- `STAGE1_BILATERAL_AUDIT.md`
- `selected_asymmetry_diagnostic.csv`
- `asymmetry_strata.csv`
- `potential_meanbaseline_crossing.csv`
- `asymmetry_subject_bootstrap.csv`
- `bilateral_diagnostic_scores.csv`
- `protocol.json`

Stage-2 outputs must remain absent because Diagnostic Gate B failed.

## Recommended Next Step

Close the bilateral-baseline route as preregistered and retain canonical GLSD-v1. Do not run EXP-5C automatically.
