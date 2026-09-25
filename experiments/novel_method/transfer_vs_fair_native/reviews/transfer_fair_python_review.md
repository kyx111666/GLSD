# Python Review — Transfer vs Fair Native

## Scope

- `my_method/transfer_vs_fair_native/run_comparison.py`
- `my_method/native_config_transfer/run_transfer.py`
- Generated result artifacts under `results/transfer_vs_fair_native/` and `results/native_config_transfer/`

## Review findings

1. **PASS — locked inputs are read-only and outputs are isolated.** `run_comparison.py:21-24` separates the two locked input roots from `results/transfer_vs_fair_native`; `run_transfer.py:33-34` similarly writes only to `results/native_config_transfer`.
2. **PASS — Task A checks one-to-one video and subject alignment.** `run_comparison.py:135-166` rejects duplicate keys, unequal transfer/Fair/cache video sets, or unequal outer-subject sets before calculating any metric.
3. **PASS — Task A verifies GT and cache identity rather than trusting filenames.** `run_comparison.py:159-161,174-181` compares the target bundle SHA with Fair `PROTOCOL.json` and checks every Fair GT list against the target cache record.
4. **PASS — stored `matched_gt` and aggregate reports are not trusted.** `run_comparison.py:70-84` removes stored matching labels; `:183-211` reruns the locked evaluator, checks stored-vs-recounted predictions, and gates all four exact aggregate F1 values at `1e-15`.
5. **PASS — bootstrap is genuinely subject-paired.** `run_comparison.py:96-124` creates one `N×N` draw matrix per setting and applies the identical draw indices to both subject count arrays, accumulating TP/FP/FN before computing pooled F1.
6. **PASS — requested statistical outputs are complete.** `run_comparison.py:112-124` records point delta, bootstrap mean, percentile CI, `P(delta>0)`, zero crossing, resample count, seed, and paired unit.
7. **PASS — Native transfer uses source selections only.** `run_transfer.py:123-128,150-180` loads the source `single_tuned` outer selection, verifies source/target subject identity, freezes one source tuple per held-out subject, and applies it directly to target records.
8. **PASS — Native transfer does not invent a shared decoder.** `run_transfer.py:23-30,176-187` reuses `VideoFeatures` and the existing Fair Native grid/decoder; target backbone remains the argument that selects its existing interval/NMS branch.
9. **PASS — configuration portability is executable, not assumed.** `run_transfer.py:130-137,249-257` checks all 324 runtime tuples against the signed source manifest and requires the ME-TST+/Boosting manifests to be byte-identical.
10. **PASS — target duration isolation and default decoder equivalence are gated.** `run_transfer.py:125-126,171-178` recomputes target outer-fold `k` under the locked target rule, while `:185-207` requires exact per-subject replay of the target fold-Native default.
11. **PASS — deterministic reproducibility smoke test.** Both modules compile successfully under `/opt/miniconda3/bin/python`; two complete executions reproduced identical SHA256 output artifacts.
12. **PASS — numerical direction is correct.** Both scripts define delta as first method minus comparator (`run_comparison.py:112-120`; `run_transfer.py:108-119`), matching the paper-facing hypotheses.

## Issues

No blocking correctness issue found. The Native transfer is a supplementary configuration-replay control: its A-class validity applies to the selectable `(smooth,p,distance)` tuple, not to equivalence of ME-TST+ fixed intervals and BoostingVRME adaptive interval/NMS geometry.

## Final status

`PASS`
