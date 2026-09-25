# Canonical consensus operators — Python review

Review target: `my_method/window_consensus/run_canonical_consensus_operators.py`

## Verdict

PASS. The implementation is consistent with the frozen paired-nested protocol and the generated artifacts passed an independent numerical audit.

## Explicit pass items

1. **PASS — operator scope is exact.** Lines 25–30 enumerate only O0, O1, O3, O4, and O5. Lines 85–90 implement arithmetic, geometric, harmonic, and minimum consensus without an extra parameter.
2. **PASS — raw-score and overlap assumptions are guarded.** Lines 67–77 reject wrong window shape, non-finite/out-of-range sigmoid scores, invalid global indices, and overlap counts outside one or two.
3. **PASS — single-view behavior is exact.** Lines 93–97 assert bitwise equality between every non-O0 operator and the sole raw score on one-view frames.
4. **PASS — arithmetic mean is cross-checked against the previously verified implementation.** Lines 117–125 compare O1 and coverage counts with `aggregate_raw` for all 79 videos and abort on any mismatch.
5. **PASS — paired identities and order are preserved.** Lines 145–151 require exactly 79 unique video identities and identical ordering across all operators.
6. **PASS — the O0 gate precedes all experimental branches.** Lines 264–279 run provenance, cardinality, historical fold-config, and 49/143/110 checks; lines 281–284 run O1/O3/O4/O5 only after those checks pass.
7. **PASS — search budget is identical.** Lines 263 and 269–284 reuse the same frozen 81-config grid and the same `precompute`/`nested_select` pathway independently for every operator.
8. **PASS — decision and bootstrap are protocol-bound.** Lines 294–320 select the best non-O0 operator from nested outer results, run the fixed 1000-subject paired bootstrap, and require all declared GO gates.
9. **PASS — prohibited method families are absent.** Lines 394–405 record that no weighted mean, aggregation tuning, classifier, recognition, morphology, W2 verifier, pruning, rescue, grid expansion, or backbone forward occurred.
10. **PASS — artifact cardinality and arithmetic were independently verified.** The audit reproduced 5 summary rows, 145 outer rows, 145 selected-config rows, 11,745 inner-score rows, and 60 frequency rows; all per-method TP/FP/FN sums and F1 values matched. Every fold selection was independently re-ranked using exact rational F1, then FP, then config index. The 1000-repeat bootstrap mean and percentile interval matched exactly.

## Constraint direction

The implementation is intentionally one-way constrained: it can only apply the four pre-registered symmetric operators and can only choose among the frozen Native decoder configurations. There is no code path for power-mean, alpha, lambda, learned weighting, recognition, pruning, rescue, or another experiment stage.

## Failed checks and repairs

No implementation defect was found. An initial ad-hoc audit command referenced output columns as `TP/FP/FN` instead of their actual `inner_TP/inner_FP/inner_FN` names; the audit command was corrected and then passed. This did not change experiment code or artifacts.

## Remaining risks

- The script imports the previously verified paired-nested implementation dynamically; provenance hashes and the O0 historical fold gate mitigate accidental drift, but the experiment still depends on that local module remaining available.
- The raw source default is an absolute local path. Reproduction on another machine should pass `--raw-dir` explicitly.
- The requested decision language provides only GO/NO-GO. The implementation therefore maps any failure of the full GO conjunction to NO-GO, which is stated in the report.

## Run and outputs

Run from the repository root:

```bash
.venv/bin/python my_method/window_consensus/run_canonical_consensus_operators.py
```

Primary report:

`results/canonical_consensus_operators_sammlv/CANONICAL_CONSENSUS_OPERATORS_PAIRED_NESTED_SAMMLV_CN.md`

Machine-readable outputs are under:

`results/canonical_consensus_operators_sammlv/outputs/`

The task stops at the reported decision; no follow-on aggregation search is implemented or launched.
