# Context Agreement Scientific Audit Python Review

> **Status**: passed_with_runtime_blocker  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-03  
> **Scripts reviewed**: `export_sammlv_pre_stitch_context.py`, `run_context_agreement_scientific_audit.py`

## Pass Items

1. ✅ `export_sammlv_pre_stitch_context.py:163-199` requires CUDA, loads each subject checkpoint with `strict=True`, calls `model.eval()`, and performs every forward inside `torch.no_grad()`.
2. ✅ `export_sammlv_pre_stitch_context.py:191-199` uses only the spotting output and explicitly discards recognition logits; no hidden hook, mirror, optimizer, or learned feature is used.
3. ✅ `export_sammlv_pre_stitch_context.py:115-139` reproduces subject-level, batch-local stitching: batch-local index zero writes the full 30-step window and all other windows write local positions 15:30.
4. ✅ `export_sammlv_pre_stitch_context.py:205-232` preserves every real observation with exact `window_id/local_t/global_t/raw_score` mappings and independently stores `fresh_stitched_score` and `context_count`.
5. ✅ `export_sammlv_pre_stitch_context.py:238-282` records source commit, all checkpoint hashes, input-cache hash, batch/window/stride/stitching, environment, zero-training provenance, and final cache SHA-256.
6. ✅ `run_context_agreement_scientific_audit.py:375-412` reconstructs the entire subject-level fresh stitched output from raw windows before analysis and requires exact equality; corrupted or mismapped raw caches are rejected.
7. ✅ `run_context_agreement_scientific_audit.py:415-489` executes the full-reproduction TP/FP/FN gate before candidate construction and writes `BLOCKED-FULL-REPRODUCTION` without Agreement results when counts differ.
8. ✅ `run_context_agreement_scientific_audit.py:65-69, 491-544` uses only fresh stitched scores and the fixed native candidate path: box smoothing `2*k_p`, Moilanen `p=0.55`, and `find_peaks(distance=k_p)`; GT is attached only after peaks exist.
9. ✅ `run_context_agreement_scientific_audit.py:525-538` fixes primary disagreement as population `std`; MAD is stored only as a secondary descriptor and excluded from combined evidence and GO checks.
10. ✅ `run_context_agreement_scientific_audit.py:579-588` fixes combined evidence as per-video robust Height minus per-video robust Disagreement. No coefficient, classifier, learned fusion, threshold search, or alternative smoothing is implemented.
11. ✅ `run_context_agreement_scientific_audit.py:165-191, 634-675` applies within-subject one-to-one Height matching, fixed context-count strata, and stride-defined boundary exclusion separately.
12. ✅ `run_context_agreement_scientific_audit.py:201-237, 241-263, 678-755` saves the native-missed subset, fixed native-FP-budget ranking, 1000-draw subject bootstrap, JSON, candidate CSV, and subject CSV.
13. ✅ Both scripts compile, expose command-line help, and completed an end-to-end synthetic regression covering 29 subjects and 79 videos, including raw→stitch verification and all requested output files.

## Failed / Repaired Items

| # | Area | Issue | Action | Status |
|---|---|---|---|---|
| 1 | Fresh cache integrity | Initial analyzer trusted the stored stitched score | Added an independent subject-level raw→stitch exact-equality reconstruction | fixed |
| 2 | Boundary audit | Initial summary only reported distance distributions | Added a two-sample KS statistic/p-value and fixed stride boundary exclusion | fixed |
| 3 | Subject direction | Initial subject CSV only reported means | Added median, standard deviation, IQR, and direction-count summary | fixed |

## Constraint Direction Review

No optimization constraints or physical inequality constraints are present. Decision comparisons such as `context_count >= 2`, IoU `>= 0.5`, and GO checks are direct transcriptions or explicit pre-run operationalizations documented in the JSON.

## Remaining Risks

- **Runtime blocker**: the real raw pre-stitch cache does not yet exist. The Mac cannot execute the ME-TST CUDA/Mamba forward; the extractor must run in the restored Colab environment. Synthetic results are code tests only and must never be cited scientifically.
- The historical compact cache is read numerically only for the full-reproduction sanity comparison and non-numerically for GT. Candidate features and scores are derived solely from the fresh cache.
- With window length 30 and stride 15, theoretical raw multiplicity is at most 2. The requested `count=3` and `count>=4` strata are expected to be empty and will be reported as such, not silently omitted.
- The fixed candidate pool is thresholded native peaks as explicitly specified. Consequently, matched-FP recovery may be less informative than earlier all-local-maxima audits; the script does not broaden the pool.
- GO is operationalized before real data as: all four core checks plus at least 7 of 9 total checks. The core checks are height-conditioned AUC > 0.55, positive combined ΔAP, strictly positive bootstrap lower CI, and strictly greater missed recovery at fixed FP.

## Run Instructions

See `PRE_STITCH_CONTEXT_AGREEMENT_COLAB_RUNBOOK_CN.md`.

## Expected Outputs

- `SAMMLV_PRE_STITCH_CONTEXT_CACHE.pkl`
- `sammlv_pre_stitch_context_manifest.json`
- `outputs/PRE_STITCH_CONTEXT_AGREEMENT_AUDIT_CN.md`
- `outputs/context_agreement_results.json`
- `outputs/sammlv_context_candidates.csv`
- `outputs/sammlv_context_subject_metrics.csv`

## Recommended Next Step

Run the two scripts in the restored Colab environment. Do not interpret or package a method until the real JSON exists and passes the full-reproduction gate.
