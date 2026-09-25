# EXP-5A Python Code Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-18
> **Script reviewed**: `controlled_exploration/EXP5A_scale_support/run_exp5a.py`

## Pass Items

1. ✅ `run_exp5a.py:55-64` constructs exactly the preregistered 90 configurations from three reference scales, three radii, and ten thresholds; the script asserts the budget.
2. ✅ `run_exp5a.py:118-174` reproduces canonical cross-scale matching, distinguishes matched zero evidence from a missing scale, uses `numpy.median`, and asserts that the reference candidate self-matches.
3. ✅ `run_exp5a.py:250-281` independently reselects canonical outer-fold configurations and compares selected configuration, inner F1, subject TP/FP/FN, and every prediction list against the frozen archive. The completed run passed all four settings.
4. ✅ `run_exp5a.py:286-354` links each final selected prediction back to its source peak and records `K`, matched, positive-evidence, and missing counts separately. Post-run assertions verified `matched + missing = K` and `positive <= matched` for all 127,482 candidate rows.
5. ✅ `run_exp5a.py:309-353` implements the fixed near-threshold margin `0 < tau-S <= 0.10`, uses the unchanged evaluator-derived potential interval label (`prepared.best`), and computes crossings without changing the Stage-1 prediction set.
6. ✅ `run_exp5a.py:357-376` implements a 10,000-resample, seed-100 paired subject bootstrap and excludes subjects with either missing denominator instead of replacing missing prevalence with zero.
7. ✅ `run_exp5a.py:463-497` requires enrichment, burden, and rescue precision in the same setting and separately checks the all-setting FP-risk condition, subject-bootstrap direction, and mathematical non-invariance.
8. ✅ `run_exp5a.py:726-751` stops before the Stage-2 loop when Gate S fails. The final artifact audit confirmed that `results.csv`, `support_mechanism.csv`, `selected_config_per_subject.csv`, and `EXP5A_ANALYSIS.md` were not created.
9. ✅ `run_exp5a.py:33-35,357-376` fixes all stochastic and diagnostic constants (`N=10,000`, `seed=100`, margin `0.10`) and saves portable CSV/JSON/Markdown outputs.
10. ✅ `run_exp5a.py:721-729` records the 90-configuration budget, an empty forbidden-variant list, canonical non-modification, and a SHA-256 digest of the executed script in `protocol.json`.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `run_exp5a.py:118-174` | Reference self-match was initially documented but not asserted at runtime. | Added physical-width presence and self-match assertions. | fixed |
| 2 | `run_exp5a.py:578-612` | Dormant Stage-2 partial gate initially admitted cases outside the preregistered strict 2/4 or 3/4 improvement condition. | Added strict improvement count and explicit partial predicate. | fixed |
| 3 | `run_exp5a.py:320-354` | Candidate audit rows initially omitted selected `a0` and `rho`. | Added `selected_a0` and `selected_rho` for full traceability. | fixed |

## Constraint Direction Review

No optimization inequality constraints are present. The only directional gates are direct translations of the preregistration: near-threshold margin at `run_exp5a.py:312`, Gate S enrichment/burden/precision at `run_exp5a.py:471-475`, and the bootstrap FP-enrichment test at `run_exp5a.py:482-485`.

## Remaining Risks

- The experiment intentionally depends on the repository's frozen OOF caches and archived canonical outputs; portability requires preserving those relative paths.
- Candidate compatibility is diagnostic best-GT IoU compatibility, not a replacement for one-to-one final evaluator matching; this distinction is stated in the audit report.
- Stage 2 is implemented but was not executed because Gate S failed, so its dormant branch was statically reviewed rather than empirically exercised in this run.

## Run Instructions

```bash
python3 controlled_exploration/EXP5A_scale_support/run_exp5a.py
```

## Expected Outputs for This Run

- `STAGE1_SCALE_SUPPORT_AUDIT.md`
- `selected_support_diagnostic.csv`
- `support_strata.csv`
- `near_threshold_support.csv`
- `potential_threshold_crossing.csv`
- `support_subject.csv`
- `support_diagnostic_scores.csv`
- `protocol.json`

Stage-2 outputs must remain absent because Gate S failed.

## Recommended Next Step

Close the missing-scale route as preregistered and retain canonical GLSD-v1.
