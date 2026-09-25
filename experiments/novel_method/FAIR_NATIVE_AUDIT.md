# FAIR Native Baseline Audit

> **Scope:** code and locked-artifact audit only. No full Fair Tuned Native experiment was run and no GLSD source/result was modified.
> **Date:** 2026-09-08

## Audit conclusion

The repository contains both the original Native inference logic and the current GLSD execution chain. The current Table-2-style `H`/height-only control is **not yet proven to be a strict Fair Tuned Native baseline**. It reuses the height evidence and the unified decoder, but its selected configuration can change smoothing/reference scale, threshold, and the fold duration prior. The original Native decoder is a fixed protocol (`2*k_p` smoothing, threshold fraction `0.55`, peak distance `k_p`, fixed interval construction). Therefore the planned Fair Tuned Native experiment must be implemented as a separate audit-preserving runner that calls the original Native decoder or an exact extracted wrapper, with its parameter space explicitly documented before execution.

## Located implementation and data artifacts

| Item | Located evidence | Finding |
|---|---|---|
| ME-TST+ Native decoder | `my_method/native_failure_mode_audit/run_native_failure_mode_audit.py:71-117` (`smooth`, `native_decode`) | Moving average width `2*k_p`; threshold `mean + 0.55*(max-mean)`; `find_peaks(..., distance=k_p)` and thresholded peaks. |
| ME-TST+ Native interval/matching | `my_method/native_failure_mode_audit/run_native_failure_mode_audit.py:120-134` and `:76-80` | Fixed `[peak-k_p, peak+k_p]` interval and chronological greedy IoU matching at 0.5. |
| BoostingVRME Native decoder/interval | Historical `historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/tune_equiscale.py:120-174`; current `my_method/gl_saliency_skill/adapters/boostingvrme.py:12-76` | Historical `native_predictions()` smooths with `round(config.smooth*k_p)`, thresholds at `config.threshold`, uses distance `round(config.distance*k_p)`, constructs `source_interval`, then applies source NMS. Current adapter preserves the interval and conflict geometry. |
| Current GLSD Native/H/G/L/GL | `my_method/gl_saliency_skill/evidence.py:34-58`, `:157-169`; `decoder.py:18-83`; `skill.py:18-57` | Native is `Config("native", 2.0, 1.0, 0.55)`; H is the `height` family; G/L/GL are the corresponding evidence families; GL fusion is `0.5*(G+L)`. |
| Current Nested LOSO selection | `my_method/gl_saliency_skill/selection.py:22-68` | Inner TP/FP/FN are pooled before F1; tie-break is F1, precision, fewer FP, then deterministic config index. |
| Frozen caches and GT | `my_method/gl_saliency_skill/benchmark.py:69-82`; `caches/me_tst/*.pkl`; signed Boosting curve caches under `historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/curve_cache/` | Four frozen response caches are loaded without backbone execution. Adapters map score/GT schemas. |
| Subject splits | `my_method/gl_saliency_skill/benchmark.py:98-111`, signed `fold_duration_priors.json` | Subject order and outer/inner priors are checked against signed artifacts. |
| k_out/k_in | `benchmark.py:85-111` | `k_out(s)` excludes held-out subject; `k_in(s,v)` excludes outer subject and inner validation subject. ME-TST uses mean duration; Boosting uses median duration; final transform is `int((value+1)/2)`, lower bounded at 1. |
| Unified evaluator | `my_method/gl_saliency_skill/evaluation.py:7-60` | Inclusive interval IoU and chronological greedy one-to-one GT matching; no second-best rematch. |
| Subject-paired bootstrap | `my_method/gl_saliency_skill/benchmark.py:249-282` (current benchmark) and `my_method/strict_cross_backbone_transfer/run.py:76-101` | Fixed seed 100, 10,000 subject resamples, pooled-count F1 per resample. |

## A. Is current H strictly equivalent to a tuned Native scoring baseline?

**Answer: NO / NOT ESTABLISHED.** The current H control is a height-evidence control in the GLSD decoder, not a demonstrated wrapper around each backbone's complete Native decoder. The following comparison is required:

| Component | Original Native | Current H/height control | Equivalence status |
|---|---|---|---|
| Candidate generation | ME: smooth width `2*k_p`, `find_peaks(distance=k_p)`; Boosting: native adapter's curve geometry | `CurveFeatures.evidence(reference, radius)` creates peaks after smoothing width `round(reference*k)`; H selects reference from `{1,1.5,2}` and uses the GLSD `k` | Only the special `reference=2` case is algebraically close; not equivalent for tuned H in general. |
| Smoothing | Fixed `2*k_p` | Reference-dependent width and fold-dependent `k` | **Different when H selects another reference or prior.** |
| Peak detection | Native thresholded peaks with distance `k_p` | Peaks are generated before scoring; threshold is applied to height scores | Same family of `find_peaks`, but candidate pool and scale can differ. |
| Score definition | Native threshold on smoothed response (`0.55` fraction of range above mean) | Height evidence `(smooth[peak]-mean)/(smooth.max()-mean)` | Numerically related but not a strict proof of identical score path at all boundaries. |
| Score threshold | Fixed `0.55` | Tuned over the GLSD threshold grid | **Different.** |
| Duration prior `k` | Author/native `k_p` in the fixed decoder | Fold-specific `k_out` for outer scoring and `k_in` for inner selection | **Different protocol unless explicitly frozen to Native `k_p`.** |
| Interval construction | ME fixed ±`k_p`; Boosting curve-dependent `source_interval` | GLSD adapter with supplied fold `k` | Same adapter class, but potentially different scale; not automatically equivalent. |
| Conflict/NMS | ME native has no interval conflict matrix; Boosting native adapter uses source conflicts | `decode_candidates()` applies adapter conflict suppression after score thresholding | Must be verified per backbone; not a universal equivalence claim. |
| Output ordering | Native peak order from `find_peaks` | Adapter geometry order (stable peak order for ME; interval/start order for Boosting) | Equivalent only if the adapter ordering reproduces native ordering for the tested records. |
| Evaluator | `greedy_native_match()` at inclusive IoU ≥ 0.5 | `evaluation.evaluate()` at inclusive IoU ≥ 0.5, no second-best rematch | Semantically equivalent for the locked spotting metric; still requires smoke replay. |

Thus H can be reported as a **height-only GLSD-chain control**, but it must not be described as a strict fair tuned Native baseline until the new runner passes default replay and its parameter provenance is frozen.

## Required pre-run implementation gates

1. Extract the actual Native parameter space from the two real decoders; do not force it to 90 configurations.
2. Save `configs/fair_tuned_native_me_tst.json` and `configs/fair_tuned_native_boosting.json` before any full run, including source lines and total configuration count.
3. Add a default Native equivalence test against the locked four Native totals and per-video predictions.
4. Verify that the inner selector consumes only training/inner-validation subject cells and pools TP/FP/FN before F1.
5. Freeze deterministic ordering and record the selected configuration for every outer subject.
6. Stop immediately on any default replay mismatch; do not tune around it.

## Cache/GT/split leakage audit

- Frozen cache loading is label-free in the decoder (`skill.py:31-57`; GT is passed only to `evaluation.evaluate`).
- Signed `stats_k*.npz`, subject order, and fold priors are checked by `benchmark.signed_context()` (`benchmark.py:114-137`).
- Current GLSD selection excludes the held outer subject through `inner_counts()` (`selection.py:31-37`); the same invariant must be reused by Fair Tuned Native.
- Existing locked GLSD results remain untouched. This audit does not authorize overwriting them.
- No dedicated `fair_tuned_native` runner or pre-run Native configuration manifest currently exists in the repository; these are required deliverables before execution.

## Historical Native parameter source

The BoostingVRME historical fair-Native implementation exposes `NativeConfig(smooth, threshold, distance)` and `native_configs()` in `historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/tune_equiscale.py:66-74,300-307`. Its decoder is `native_predictions()` at `:165-174`. The ME-TST Native path is explicitly fixed in `native_failure_mode_audit/run_native_failure_mode_audit.py:18,112-117` and does not expose the same three-parameter grid. This asymmetry is why a new Fair Tuned Native manifest must be backbone-specific and must not be manufactured to match the 90 GLSD configurations.

## Newly supplied historical Native archive

The supplied `Native_decoder_实验汇总.zip` materially resolves the previous protocol ambiguity. Its README and signed artifacts provide:

- BoostingVRME historical `tuned_native` grid: **64 configurations** (`smooth` 4 × `threshold` 4 × `distance` 4), implemented by `boostingvrme/tune_equiscale.py:300-306` and used by `results/equiscale_tuning/*/report.json`.
- A shared `fair_equiscale_v1` post-processing protocol for both ME-TST and BoostingVRME, with a **324-configuration single-scale pool**. The pool is explicitly the distinct `(smooth*scale, distance*scale)` components of the multi-scale candidate pool, with `p` values `{0.4,0.5,0.55,0.6}`. This is implemented at `boostingvrme/equiscale_fair_validation.py:97-100` and documented in `README_equiscale_fair_validation.md`.
- Complete `PROTOCOL.json`, `fold_duration_priors.json`, `outer_loso_selections.csv`, `selected_predictions.json`, `stats_k*.npz`, `subject_counts.csv`, and `VERIFICATION.json` for both backbones and both datasets.
- The four fair-validation protocols all use `fair_equiscale_v1`, exclude outer and inner validation subjects from duration/configuration selection, use chronological inclusive-IoU greedy matching, and report `selected_prediction_replay=true` and `held_subject_excluded_from_k_and_objective=true`.

The archive ZIP SHA256 is `b4c6eac05b17a650292232946eb9e160fd86efd765d5f248d803ce3bc5234117`. Its key source hashes match the signed GLSD archive: `tune_equiscale.py`=`bc3aa386570fa3de8d9d56c0fd2a57e9c12d7b594765c7095fbbb40ee205263d` and `equiscale_fair_validation.py`=`c04c98aa3ba887459e03482886c183a76002341d8ad2656d49d08d9f2b8c206c`.

The archive therefore supplies a defensible Native tuning protocol. It does **not** contain the raw curve caches, weights, or a fresh rerun environment; its README explicitly says those inputs are external. The historical reports are evidence for protocol/results provenance, not a substitute for the requested current-cache default-equivalence smoke test.

## Current status

`FAIR_NATIVE_AUDIT_COMPLETE; FAIR_TUNED_NATIVE_COMPLETE`

Locked default Native anchors in `results/final_gl_skill/main_results.csv` are: ME-TST+/SAMMLV `53/184/106, F1=0.2676767676767677`; ME-TST+/CASME3 `81/912/777, F1=0.08752025931928688`; BoostingVRME/SAMMLV `49/145/110, F1=0.2776203966005666`; BoostingVRME/CASME3 `93/818/765, F1=0.10514414923685698`.

The supplied archive resolved the ME-TST protocol ambiguity at the historical experiment level. The path-adapted runner was executed on the current frozen caches under `results/fair_tuned_native_v2/`; default Native smoke replay and independent per-video replay passed for all four settings. Consolidated outputs are under `results/fair_tuned_native/`. No GLSD source or prior result was overwritten.
