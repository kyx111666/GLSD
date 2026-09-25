# MSCR Post-hoc Diagnostic Python Review

> **Status**: passed  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-04  
> **Script reviewed**: `run_mscr_posthoc_diagnostic.py`

## Pass Items

1. ✅ `run_mscr_posthoc_diagnostic.py:48-61` uses pooled train-subject F1 as the sole primary objective and applies the locked tie-break order: fewer FP, fewer rescue events, then fixed configuration index.
2. ✅ `run_mscr_posthoc_diagnostic.py:52-56` evaluates a configuration only when the record subject belongs to `train_subjects`; held-out GT is not read during configuration selection.
3. ✅ `run_mscr_posthoc_diagnostic.py:63-107` reuses the unchanged configuration grids and decoder functions from the reviewed MSCR implementation, and preserves all 237/237 Author Native event tuples.
4. ✅ `run_mscr_posthoc_diagnostic.py:128-171` computes H, per-scale nearest peak/distance/alignment, M, R and the direct rejection reason without adding any feature or learned model.
5. ✅ `run_mscr_posthoc_diagnostic.py:174-217` traces the original nested-LowRescue weak pools and labels a true rescue only when its matched GT belongs to the Native unmatched set. The resulting trace contains 45 candidates: 2 true rescues and 43 weak FPs.
6. ✅ `run_mscr_posthoc_diagnostic.py:226-256` computes H-only and M-only ROC-AUC/PR-AUC plus fixed two-dimensional dominance statistics; it fits no classifier.
7. ✅ `run_mscr_posthoc_diagnostic.py:26-28,322-331` records the interpretation thresholds before status assignment and returns exactly one of the three user-authorized statuses.
8. ✅ `run_mscr_posthoc_diagnostic.py:351-354` writes all four requested machine-readable files and the Chinese report.
9. ✅ Independent checks reproduced every F1 from TP/FP/FN, verified every `R=lambda*H+(1-lambda)*M`, verified M as the mean per-scale alignment, and matched all trace counts to JSON.
10. ✅ Two complete executions produced byte-identical hashes for the report and all diagnostic CSV/JSON outputs.

## Failed / Repaired Items

None after review.

## Constraint Direction Review

No optimization constraints or learned decision boundaries are introduced. The only diagnostic comparisons are the locked `R >= gamma`, F1-gain threshold, percentile checks, and joint-dominance threshold.

## Remaining Risks

- Separability is based on only two positive weak candidates; ROC-AUC and PR-AUC are therefore descriptive diagnostics, not a stable population estimate.
- Evidence values for the LowRescue pool are evaluated under each fold's originally selected Full-MSCR configuration. A candidate outside that MSCR weak pool is explicitly marked `NOT_IN_SELECTED_MSCR_WEAK_POOL` rather than silently discarded.
- The 0.005 protocol-issue F1 margin and joint H/M separability rule were operationalized before execution and are recorded in the report; they are diagnostic conventions, not newly optimized parameters.

## Run Instructions

```bash
PYTHONPATH=/private/tmp/hrep_pydeps python3 RethinkFuse_reproduction/my_method/multi_scale_candidate_rescue/run_mscr_posthoc_diagnostic.py \
  --cache RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl \
  --original-selected-configs RethinkFuse_reproduction/my_method/multi_scale_candidate_rescue/outputs/mscr_selected_configs.csv \
  --output-root RethinkFuse_reproduction/my_method/multi_scale_candidate_rescue
```

## Expected Outputs

- `MSCR_POSTHOC_DIAGNOSTIC_CN.md`
- `outputs/mscr_pooled_objective_results.csv`
- `outputs/mscr_weak_candidate_trace.csv`
- `outputs/mscr_true_rescue_cases.csv`
- `outputs/mscr_diagnostic.json`

## Recommended Next Step

Stop at the requested post-hoc interpretation; do not extend MSCR or run CASME3.
