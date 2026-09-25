# RGR Nested LOSO Python Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-05
> **Script reviewed**: `run_rgr_nested_loso.py`

## Pass Items

1. ✅ `run_rgr_nested_loso.py:41-45` implements a stable row-wise softmax; `run_rgr_nested_loso.py:48-72` validates `[T,5]`, finite logits and temporal alignment for all 79 records before creating R1/R2.
2. ✅ `run_rgr_nested_loso.py:72-79` builds exactly the locked 81-config Stage-A Cartesian grid and maps `p_H` to the unchanged native decoder parameter.
3. ✅ `run_rgr_nested_loso.py:82-85` implements the requested inclusive `peak-k_p : peak+k_p` clipped mean aggregation, with no center/max/margin/temperature alternatives.
4. ✅ `run_rgr_nested_loso.py:92-100` generates `P_low` using locked Stage-A smoothing/distance and defines weak candidates strictly as `P_low - P_high`.
5. ✅ `run_rgr_nested_loso.py:101-114` only adds accepted rescue events and explicitly aborts if any Stage-A onset/peak/offset tuple is absent from the final event set.
6. ✅ `run_rgr_nested_loso.py:128-138` performs Stage-A selection only on the supplied train-subject set using pooled F1, then FP/rescue-count/config-index tie-breaks.
7. ✅ `run_rgr_nested_loso.py:141-160` restricts Stage B to four Delta-p values and five gamma values (20 RGR configs), while LowRescue uses only the four Delta-p values.
8. ✅ `run_rgr_nested_loso.py:196-209` performs a paired 1,000-repeat subject bootstrap with fixed seed `20260905` and aggregates TP/FP/FN before calculating each F1 difference.
9. ✅ `run_rgr_nested_loso.py:392-405` records the pre-locked R2-mean evidence, exact grids, no-test-GT selection declaration, frozen scope, subject stability and all GO checks in machine-readable JSON.
10. ✅ `run_rgr_nested_loso.py:409-414` writes every requested CSV/JSON/Markdown artifact rather than relying on console output.
11. ✅ Independent rerun produced byte-identical SHA-256 hashes for all four machine-readable outputs.
12. ✅ Cross-run check confirmed Stage-A selected configurations match the prior pooled-F1 Tuned Native audit in 29/29 folds and reproduce 49/143/110 exactly.

## Failed / Repaired Items

None.

## Constraint Direction Review

No optimization inequality constraints are present. The only decision inequalities are the task-specified `R_rec >= gamma_R` and GO thresholds; their directions match the experiment specification.

## Remaining Risks

- R2-mean was identified in a previous audit using SAMMLV labels, so this is a stage-one feasibility experiment rather than a fully independent confirmation dataset.
- The prior separability audit used an Author-Native-derived weak pool; this experiment uses fold-specific Conservative-Tuned-Native weak pools. Their candidate distributions are not interchangeable.
- The Stage-B grid intentionally contains no “rescue none” option. This follows the task exactly, but means nested selection must choose one of the specified rescue configurations even when all reduce train pooled F1.

## Run Instructions

```bash
cd <workspace-root>/RethinkFuse_reproduction/my_method/multi_scale_candidate_rescue
PYTHONPATH=/private/tmp/hrep_pydeps python3 run_rgr_nested_loso.py --cache ../../caches/me_tst/sammlv_strategy1_outputs.pkl --output-root .
```

## Expected Outputs

- `RGR_NESTED_LOSO_SAMMLV_REPORT_CN.md`
- `outputs/rgr_outer_metrics.csv`
- `outputs/rgr_selected_configs.csv`
- `outputs/rgr_rescue_trace.csv`
- `outputs/rgr_summary.json`

## Recommended Next Step

Honor `NO-GO-RGR` and the user-defined stop rule; do not extend the decoder or run CASME3 automatically.
