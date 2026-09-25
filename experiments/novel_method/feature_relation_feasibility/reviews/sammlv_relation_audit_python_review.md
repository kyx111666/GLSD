# SAMMLV Relation Audit Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-03  
> **Script reviewed**: `run_sammlv_relation_audit.py`

## Pass Items

1. ✅ `run_sammlv_relation_audit.py:293-321` requires a COMPLETE 79-video fresh manifest, rejects historical-numeric provenance, and checks `hidden.shape == (T,384)` plus `valid_mask.shape == score.shape` for every video.
2. ✅ `run_sammlv_relation_audit.py:104-120` copies only `subject`, `video`, `samples`, `k_p`, and `frame_skip` from the historical cache; historical `score`, `logits`, `emotion`, and mirror predictions are only listed as present and are not assigned to experiment arrays.
3. ✅ `run_sammlv_relation_audit.py:323-336` generates candidates from the fresh smoothed score without GT, then excludes candidates whose common ±`2*k_p` relation window touches an unwritten `valid_mask=False` position.
4. ✅ `run_sammlv_relation_audit.py:130-139` L2-normalizes each hidden vector, constructs cosine self-similarity, computes exactly the two declared deterministic scalars, and deterministically resizes the matrix to the fixed 21×21 representation.
5. ✅ `run_sammlv_relation_audit.py:142-164` performs subject-LOSO: `StandardScaler` and `LogisticRegression` are fit only on non-held-out subjects, with predictions written only for the held-out subject.
6. ✅ `run_sammlv_relation_audit.py:224-250` implements within-subject 1:1 nearest peak-height matching without replacement with the fixed 0.15 caliper.
7. ✅ `run_sammlv_relation_audit.py:253-280` applies an identical native-FP budget to height, transition, center-context, and relation-probe rankings.
8. ✅ `run_sammlv_relation_audit.py:186-221` bootstraps at the subject level and reports 95% intervals and paired deltas against peak height.
9. ✅ `run_sammlv_relation_audit.py:486-495` saves portable JSON, CSV, and Chinese Markdown outputs instead of relying on notebook display state.
10. ✅ `python3 -m py_compile run_sammlv_relation_audit.py` completed successfully on 2026-09-03.

## Failed / Repaired Items

No code defect requiring a modeling change was found during static review.

## Constraint Direction Review

No optimization inequality constraints are present. The fixed gates are data-integrity checks (`COMPLETE`, 79 videos, aligned shapes, and valid relation windows), and their rejection directions match the audit protocol.

## Remaining Risks

- A full numerical run was not possible on the Mac because the fresh 79-video `.npz` dump is in Google Drive and the local Python lacks SciPy/scikit-learn. This is an execution-location warning, not a ME-TST environment blocker; the audit is CPU-only and should run in standard Colab Python.
- The source OOF best-F1 threshold is saved only for a later locked-transfer experiment. It is explicitly diagnostic and must not be described as an unbiased final decoder score.
- This run can establish or falsify a SAMMLV-side signal only. CASME3 locked transfer and a second backbone remain unverified.

## Run Instructions

```bash
python run_sammlv_relation_audit.py \
  --fresh-root /content/drive/MyDrive/ME_TST_FRESH_FROZEN_OUTPUT/SAMMLV_FULL \
  --gt-cache /content/drive/MyDrive/sammlv_strategy1_outputs.pkl \
  --output-dir /content/drive/MyDrive/ME_TST_FRESH_FROZEN_OUTPUT/SAMMLV_RELATION_AUDIT
```

## Expected Outputs

- `sammlv_relation_audit_results.json`
- `sammlv_relation_candidates.csv`
- `SAMMLV_RELATION_AUDIT_CN.md`

## Recommended Next Step

Run the script in Colab and inspect the saved Phase-1 results. Do not design the final Skill unless SAMMLV evidence is positive and CASME3 locked transfer is subsequently completed.
