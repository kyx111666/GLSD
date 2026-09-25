# Python Code Review — Overlap-Aware Aggregation Feasibility

## Scope

- Reviewed implementation: `my_method/window_consensus/run_overlap_aggregation_feasibility.py`
- Reviewed outputs: `results/overlap_aggregation_feasibility_sammlv/`
- Review type: protocol compliance, numerical consistency, provenance/anchor gates, and output reconciliation.

## Review verdict

**PASS.** No blocking correctness issue remains. The implementation is limited to the three requested aggregation branches and reuses the frozen decoder/evaluator.

## Explicit pass items

1. **PASS — Frozen inputs and provenance are gated.** Lines 173–257 verify the raw manifest status/cardinality, exact reconstruction result for every NPZ, compact-cache SHA-256, cache metadata, video identities, per-file hashes, curve lengths, and the 49/143/110 O0 anchor before O1/O2 can run.
2. **PASS — O0 is the saved/reconstructed original curve.** Lines 224–235 require reconstruction status `RAW-WINDOW-INSTRUMENTATION-READY`, read `original_fresh_result_video`, and decode it with the frozen subject configuration. Output diagnostics independently confirm all 79 O0 curves have Pearson=1, MAE=0, and maximum absolute difference=0.
3. **PASS — O1 matches the specified uniform overlap mean.** Lines 138–145 flatten `(global_index, raw_score)` samples, sum with `numpy.bincount`, count contributors, and divide the two. An uncovered-frame guard prevents silent invalid output.
4. **PASS — O2 matches the fixed center-weight formula.** Lines 147–158 use exactly `1 - abs(local_index - 14.5) / 15.0` for local indices 0…29, with no searched or learned parameter, and normalize by the summed weights.
5. **PASS — Decoder and evaluator are shared across O0/O1/O2.** Lines 305–312 replace only the score curve and call the same imported `tuned_decode` and evaluator with the same per-subject Final Strong Native configuration and `k_p=5`.
6. **PASS — Recognition logits and forbidden method branches are absent.** Aggregation reads only `raw_window_score`, `global_indices`, and the saved original score. The script contains no model/backbone forward, training, classifier, rescue, pruning, morphology, adaptive threshold, or adaptive smoothing path.
7. **PASS — Subject comparisons and leave-one-subject sensitivity are correctly paired.** Lines 338–378 compare each method against O0 within the same subject, and recompute aggregate F1 after excluding exactly one shared subject.
8. **PASS — Output tables reconcile.** Independent verification found 3 summary rows, 87 subject rows (29×3), and 237 curve rows (79×3). Subject totals reproduce O0=49/143/110, O1=51/136/108, and O2=46/147/113.
9. **PASS — Coverage totals are internally consistent.** The O0 per-video diagnostic rows sum to 2,370 frames covered once and 41,925 covered twice, totaling 44,295 frames; maximum overlap is 2 and no other overlap count occurs.
10. **PASS — Decision direction is correct.** O1 changes F1 by +0.015595, exceeding the report-only 0.001 materiality tolerance, so `OVERLAP-AGGREGATION-SIGNAL-POSITIVE` follows. The tolerance does not alter aggregation or decoding.

## Issues found and resolved

1. The host system Python lacked SciPy, which is required by the already verified decoder dependency. The run was executed with the repository's existing `.venv` (NumPy 1.26.4, SciPy 1.11.4); no dependency, method, or protocol was changed.
2. The initial report exposed curve diagnostics only per video. Pooled all-frame Pearson, MAE, and maximum absolute difference were added to the summary CSV and Markdown report, while retaining the 237-row per-video diagnostic table.

## Constraint and direction audit

- There is no optimization objective or tunable inequality in O1/O2.
- O1 divides the score sum by the positive overlap count.
- O2 divides the weighted score sum by the positive weight sum.
- Higher F1 is correctly treated as better; positive `Delta_F1_vs_O0` means improvement.
- FP and FN deltas preserve their raw signed direction, so negative FP/FN values mean reductions.

## Reproduction command

```bash
RethinkFuse_reproduction/.venv/bin/python \
  RethinkFuse_reproduction/my_method/window_consensus/run_overlap_aggregation_feasibility.py
```

The default raw source is `<sammlv-raw-root>`; an alternate verified source can be supplied with `--raw-dir`.

## Remaining limitations

- This audit evaluates only uniform mean and the fixed center-weighted rule. It does not evaluate Window-Consensus features.
- Bootstrap was intentionally omitted because the requested subject-level and leave-one-subject diagnostics already exist and the protocol marked bootstrap optional.
