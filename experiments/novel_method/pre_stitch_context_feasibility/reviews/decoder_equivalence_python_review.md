# Decoder Equivalence Python Code Review

> **Status**: passed_with_warnings  
> **Reviewer**: python-code-reviewer  
> **Date**: 2026-09-03  
> **Script reviewed**: `pre_stitch_context_feasibility/run_decoder_equivalence_audit.py`

## Pass Items

1. ✅ `run_decoder_equivalence_audit.py:40-42` reproduces the native box convolution with `mode="same"`; the width passed at line 46 is exactly `2*k_p`.
2. ✅ `run_decoder_equivalence_audit.py:45-66` independently computes smoothing, Moilanen threshold, `find_peaks(distance=k_p)`, and `peak±k_p` events for each input; it matches `training_utils.py:9-29`.
3. ✅ `run_decoder_equivalence_audit.py:162-175` loads historical and fresh arrays separately, checks identical shapes, and performs no score calibration, scaling, or overwrite.
4. ✅ `run_decoder_equivalence_audit.py:245-258` records max error, MAE, RMSE, Pearson, Spearman, five requested quantiles, and preserves `np_allclose_default=false`.
5. ✅ `run_decoder_equivalence_audit.py:272-281` evaluates exact, ±1, and ±2 one-to-one peak matching without altering detector parameters; real-data exact Jaccard is 1.0.
6. ✅ `run_decoder_equivalence_audit.py:283-300` records complete event lists and all requested event IoU statistics; real-data event boundaries are 6/6 exact.
7. ✅ `run_decoder_equivalence_audit.py:122-150, 300` uses inclusive interval IoU and one-to-one matching at IoU >= 0.5; both real inputs produce TP/FP/FN = 4/2/1.
8. ✅ `run_decoder_equivalence_audit.py:197-224, 301-315` records every inconsistent candidate and native margin; the real run finds zero inconsistent candidates and zero threshold-side disagreements.
9. ✅ `run_decoder_equivalence_audit.py:316-323` keeps the PASS gate limited to exact peaks, exact spotting boundaries, and identical TP/FP/FN, and explicitly records that recognition is outside this gate.
10. ✅ `run_decoder_equivalence_audit.py:323-327` records an empty `scientific_audits_run` list and writes a portable JSON artifact instead of only printing results.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | GT match output | Generic event-pair field names appeared as `old_index/fresh_index` for prediction-vs-GT rows | Converted them to `pred_index/gt_index` | fixed |
| 2 | Event audit | Initial result only asserted exact boundaries | Added controlled-emotion signatures and an explicit fresh-logit provenance flag | fixed |

## Remaining Risks

- The fresh NPZ contains no recognition logits/emotion sequence. The script deliberately uses the same archived historical emotion stream for both decodes to isolate score-only spotting decisions (`run_decoder_equivalence_audit.py:169-175`). Independent recognition equivalence is not claimed.
- The restored original `MeanAveragePrecision2d` source is absent from the local snapshot. The audit uses the project’s transparent inclusive-IoU greedy one-to-one evaluator used by `compare_sammlv.py`; for this video all four matched IoUs are strictly above 0.5, so no equality-edge ambiguity affects TP/FP/FN.
- The result is a one-video engineering gate, not evidence of whole-dataset decoder equivalence.

## Run Instructions

```bash
RethinkFuse_reproduction/.venv/bin/python \
  RethinkFuse_reproduction/my_method/pre_stitch_context_feasibility/run_decoder_equivalence_audit.py \
  --historical-cache RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl \
  --fresh-npz /path/to/006_1_frozen_hidden_score.npz \
  --output-json RethinkFuse_reproduction/my_method/pre_stitch_context_feasibility/outputs/decoder_equivalence_results.json
```

## Expected Output

- `pre_stitch_context_feasibility/outputs/decoder_equivalence_results.json`

## Recommended Next Step

- Preserve this engineering result and, only under a separate authorization, use a fully fresh raw-context → fresh-stitch → fresh-score pipeline for Context Agreement Audit.

