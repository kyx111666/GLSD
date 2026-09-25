# FINAL-GL-SKILL Python implementation review

Status: **PASS WITH NON-BLOCKING LIMITATIONS**

Scope: implementation fidelity, evaluation semantics, leakage boundaries, portability, tests, and generated artifacts. No algorithm changes were made during this review.

## Concrete pass items

1. **Public API and zero-parameter contract pass.** `GLSaliencySkill` exposes one backbone-independent `decode(temporal_scores, temporal_scale, metadata)` entry point and declares `trainable_parameters == 0` (`skill.py`, lines 13–34). It rejects non-historical weighted unified configurations (`skill.py`, lines 17–21).
2. **Historical algorithm-source guard pass.** The benchmark verifies all four historical source SHA256 values, the frozen-cache SHA256, the exact signed configuration grid, per-fold duration priors, and every `stats_k*.npz` signature before analysis (`benchmark.py`, lines 114–137).
3. **Exact selection and replay gate pass.** Per-subject `config_id`, serialized config, `k`, and `inner_F1` are compared to the exact-fresh artifacts; every per-video event is replayed; aggregate TP/FP/FN and F1 are then checked with tolerance `1e-12` (`benchmark.py`, lines 144–226). All four backbone/dataset groups report PASS with zero F1 difference.
4. **Matching semantics pass.** Evaluation uses inclusive interval IoU and the historical chronological, no-second-best-rematch rule at IoU `>= 0.5` (`evaluation.py`, lines 8–47). Aggregate F1 is recomputed directly from TP/FP/FN (`evaluation.py`, lines 50–58).
5. **Subject-level bootstrap pass.** Resampling is paired by subject, uses exactly 10,000 draws and seed 100, and reports percentile 95% intervals plus whether zero is included (`benchmark.py`, lines 236–260).
6. **Adapter isolation pass.** The ME-TST adapter only loads/maps the frozen cache and provides fixed-interval geometry (`adapters/metst.py`, lines 14–59). The BoostingVRME adapter only loads/maps its historical curve cache, supplies native interval/conflict geometry, and provides the historical temporal scale (`adapters/boostingvrme.py`, lines 14–115). Repository search found no backbone condition or adapter import in the G/L evidence core.
7. **No test-label access in inference API pass.** `GLSaliencySkill.decode` receives scores, scale, and interval metadata only (`skill.py`, lines 27–48). Ground truth is consumed only by the external evaluator and post-hoc mechanism analysis. Inner selection excludes both the outer subject and held inner-validation subject.
8. **Regression-test pass.** Ten unit tests pass, covering the full historical grid, the 90 pure-G/L subset, elementwise G/L evidence parity, fixed/native interval decoding, fusion, tie-breaking, empty curves, no-GT API behavior, zero trainable parameters, and rejection of weighted evidence. `compileall` also passes.
9. **Artifact completeness pass.** The final result directory contains the requested combined report, main table, ablation, bootstrap, threshold robustness, mechanism summary, complexity table, portability audit, and prediction regression, plus candidate-level analysis and run metadata.

## Constraint and direction checks

- Candidate acceptance remains `score >= threshold`; no inequality reversal was found.
- Matching remains `IoU >= 0.5`; no strictness change was found.
- LOSO selection retains the historical tie-break order: higher F1, higher precision, fewer FP, lower grid ID.
- `S = (G + L) / 2` is used for pure G+L; no learned or backbone-specific weight exists.

## Non-blocking limitations

- Peak RAM is the peak Python allocation observed by `tracemalloc`, not total process RSS. This should be labeled explicitly in paper tables.
- Cache loaders use Python pickle and therefore assume trusted, signed local artifacts; they should not load untrusted external pickle files.
- The candidate-level audit is intentionally large because it preserves every candidate needed for the requested mechanism trace.

## Verification commands

```text
.venv/bin/python -m compileall -q my_method/gl_saliency_skill
.venv/bin/python -m unittest my_method.gl_saliency_skill.tests.test_skill
.venv/bin/python -m my_method.gl_saliency_skill.benchmark
```

Observed: compile pass; 10/10 tests pass; benchmark status `FINAL_GL_SKILL_COMPLETE`; regression status `ALL_FOUR_EXACT_REPLAY`.
