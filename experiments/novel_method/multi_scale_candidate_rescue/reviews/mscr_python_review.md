# MSCR Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-04  
> **Script reviewed**: `run_mscr_nested_loso.py`

## Pass Items

1. ✅ `run_mscr_nested_loso.py:108-114` implements the Author Native `MA(score, 2*k_p) → mean+0.55(max-mean) → find_peaks(distance=k_p) → [p-k_p,p+k_p]` chain. The executable baseline reproduced TP/FP/FN `53/184/106` and F1 `0.2676767677`.
2. ✅ `run_mscr_nested_loso.py:125-169` implements the locked MSCR sequence: low-threshold weak pool, explicit `P_L ∪ P_H`, three fixed scale families, soft radius alignment, height proximity, fixed convex score, and additive rescue events without a second NMS.
3. ✅ `run_mscr_nested_loso.py:228-239` creates exactly 324 Full-MSCR configurations and 81 Tuned-Native configurations; the executable assertions passed.
4. ✅ `run_mscr_nested_loso.py:249-265` precomputes score-derived event geometry without reading GT. `run_mscr_nested_loso.py:267-288` evaluates configuration quality only for the supplied outer-train subject list.
5. ✅ `run_mscr_nested_loso.py:408-419` excludes the held-out subject from `train` before selection and evaluates only the selected configuration on that held-out subject. The output contains exactly 29 selected configurations for each of the five tuned methods.
6. ✅ `run_mscr_nested_loso.py:90-103` supplies one common deterministic greedy raw spotting evaluator for Author Native, Tuned Native, LowRescue, HeightOnly, ScaleOnly, and Full MSCR.
7. ✅ `run_mscr_nested_loso.py:168` constructs Full MSCR by adding rescued events to untouched native event dictionaries. The output audit confirmed exact onset/peak/offset preservation for `237/237` native events.
8. ✅ `run_mscr_nested_loso.py:317-333` performs 1,000 subject-level bootstrap draws with fixed seed `20260904`. Two complete executions produced byte-identical report, JSON, and CSV SHA-256 hashes.
9. ✅ `run_mscr_nested_loso.py:503-509` saves every requested portable artifact: outer-subject metrics, selected configurations, rescue trace, summary JSON, Tuned-Native metrics, LowRescue metrics, and the Chinese report.
10. ✅ Independent post-run checks confirmed: 29 unique outer subjects, 145 selection rows, pooled counts equal sums of subject rows, all F1 values match TP/FP/FN formulas, MSCR trace totals equal summary rescue totals, and the final status follows `MSCR F1 <= Author Native F1`.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `run_mscr_nested_loso.py:249-288` | Initial draft precomputed GT metrics for every subject/config before outer selection, mathematically unused but inconsistent with the strict “held-out GT evaluated once” protocol. | Replaced it with GT-free score-geometry precomputation; train GT is now accessed only inside `select_config`, and test GT only for the selected configuration. | fixed |
| 2 | `run_mscr_nested_loso.py:305-311` | Initial trace marked every rescue TP as a native-missed-GT recovery. | Trace now checks membership in the native unmatched-GT set. | fixed |

## Constraint Direction Review

No optimization inequalities are implemented. Threshold comparisons have the prescribed direction: low/high peak height uses `>=` internally through `find_peaks(height=...)`, and rescue selection uses `R >= gamma` at `run_mscr_nested_loso.py:151-160`.

## Remaining Risks

- Two historical SAMMLV GT intervals are invalid or outside their score sequence. They are deliberately retained because removing them would break the locked Author Native reproduction; they remain unmatched under the common evaluator.
- The explicit union of independently detected low- and high-threshold peak sets can contain nearby cross-set peaks. No extra NMS is applied, exactly as required by the task, but this contributes to the FP behavior being tested.
- “Clearly lower FP cost” was operationalized before reading results as strictly fewer MSCR rescue FPs and at least a 20% reduction versus LowRescue. This definition is recorded in the JSON and report.

## Run Instructions

```bash
PYTHONPATH=/private/tmp/hrep_pydeps python3 RethinkFuse_reproduction/my_method/multi_scale_candidate_rescue/run_mscr_nested_loso.py \
  --cache RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl \
  --output-root RethinkFuse_reproduction/my_method/multi_scale_candidate_rescue
```

Required packages: NumPy and SciPy. No CUDA, Torch, hidden feature, recognition, or learned estimator is used.

## Expected Outputs

- `MSCR_NESTED_LOSO_SAMMLV_REPORT_CN.md`
- `outputs/mscr_outer_subject_metrics.csv`
- `outputs/mscr_selected_configs.csv`
- `outputs/mscr_rescue_event_trace.csv`
- `outputs/mscr_summary.json`
- `outputs/tuned_native_outer_metrics.csv`
- `outputs/low_rescue_outer_metrics.csv`

## Recommended Next Step

Because the locked source-side result is `NO-GO-MSCR-SOURCE`, stop before CASME3, recognition/STRS, and Skill packaging.
