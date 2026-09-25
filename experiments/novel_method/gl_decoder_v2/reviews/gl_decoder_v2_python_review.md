# GL-Decoder-v2 Python Code Review

> **Status**: passed_with_warnings
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-06
> **Scripts reviewed**: `decoder.py`, `evidence.py`, `selection.py`, `evaluation.py`, `bootstrap.py`, `audit.py`, both cache adapters, `run_gl_decoder_v2.py`, and `tests/test_gl_decoder_v2.py`

## Pass Items

1. ✅ `evidence.py:53-59` exposes score, temporal scale, reference scale, and radius only; ground truth is absent from the evidence API. This is also asserted by `tests/test_gl_decoder_v2.py:92-95`.
2. ✅ `decoder.py:48-55` implements the primary score as the exact arithmetic mean `(G+L)/2`; `tests/test_gl_decoder_v2.py:34-39` verifies equality directly.
3. ✅ `run_gl_decoder_v2.py:72-82` reads the recovered protocol grids, checks formula/grid agreement across all four historical protocol files, and asserts exactly 90 configurations before any experiment result is inspected.
4. ✅ `selection.py:26-39` explicitly excludes the held-out subject, pools training TP/FP/FN, ranks with exact rational F1, and applies the frozen deterministic tie-break. Isolation and tie behavior pass at `tests/test_gl_decoder_v2.py:46-59`.
5. ✅ `evaluation.py:17-29` reproduces the shared Native curve/threshold/peak convention, while `evaluation.py:32-60` implements chronological best-unmatched greedy matching with inclusive IoU >= 0.5.
6. ✅ The real-cache regression tests at `tests/test_gl_decoder_v2.py:72-82` passed exact Native counts for ME-TST SAMMLV `(53,184,106)`, ME-TST CASME_3 `(81,912,777)`, current BoostingVRME SAMMLV `(54,158,105)`, and current BoostingVRME CASME_3 `(80,830,778)`.
7. ✅ `run_gl_decoder_v2.py:573-579` enforces the Phase-A D stop before the Boosting execution block at `run_gl_decoder_v2.py:581-585`. The actual run produced `phase_b_status=NOT_RUN_PER_PROTOCOL` and no Phase-B directory.
8. ✅ `bootstrap.py:10-34` fixes the requested seed at 100 by default, performs 10,000 subject-level resamples, and labels results `recomputed_v2`; it is reachable only in report output if the Phase-A gate permits Phase B.
9. ✅ Both `adapters/me_tst.py:11-26` and `adapters/boostingvrme.py:11-27` only map cache schemas to the same record contract. No backbone-specific G, L, fusion, threshold, or selection logic exists in either adapter.
10. ✅ `python -m compileall -q my_method/gl_decoder_v2` completed successfully, and all 12 unit/regression tests passed under the project `.venv` (Python 3.10, NumPy 1.26.4, SciPy 1.11.4).

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `adapters/me_tst.py:14-23`, `adapters/boostingvrme.py:14-24` | Initial adapters assumed per-record `gt/k_p` and a dictionary Boosting schema. | Mapped the actual verified cache schemas (`samples`, payload-level `k_p`, list-valued `subject_curves`) without changing decoder semantics. | fixed |
| 2 | `run_gl_decoder_v2.py:149-155` | Verified control report names CASME_3 while cache task names use CASME3. | Added a deterministic underscore-only dataset-key normalization. | fixed |
| 3 | `run_gl_decoder_v2.py:113-160` and output-writing block | Initial artifacts did not expose every requested canonical field/file name. | Added explicit recovered/unresolved fields, historical SHA/targets, candidate-generation spec, and the exact requested Phase-A combined CSV/JSON names. No algorithm or grading condition changed. | fixed |

## Constraint Direction Review

No mathematical inequality constraints are implemented. The only threshold inequality is `evidence >= threshold` in `decoder.py:88`, matching the frozen decoder definition.

## Remaining Risks

- The exact historical G normalization, L construction/alignment, candidate engine, temporal-scale estimator, and tie-break remain unavailable. They are explicitly marked `UNRESOLVED_FROM_HISTORICAL_ARTIFACTS`; the Phase-A D result confirms that v2 must not be represented as the historical executable method.
- The cache inputs are trusted pickle files. Their SHA256 identities are recorded and checked, but pickle loading is not safe for untrusted external files.
- The system Python 3.14 lacks SciPy. Runs must use the existing project `.venv`; no dependency was installed or changed.
- Test 10 is primarily an architectural assertion. The stronger evidence is that the runner instantiates one `GLDecoder` and adapters contain no decoder logic.
- Phase-B artifacts are intentionally absent because Phase A graded D. Creating them would violate the experiment protocol.

## Run Instructions

```bash
cd <workspace-root>/RethinkFuse_reproduction
.venv/bin/python -m unittest my_method.gl_decoder_v2.tests.test_gl_decoder_v2
.venv/bin/python -m my_method.gl_decoder_v2.run_gl_decoder_v2
```

## Expected Outputs

- `results/gl_decoder_v2/reconstruction_spec.json`
- `results/gl_decoder_v2/reconstruction_assumptions.json`
- `results/gl_decoder_v2/gl_v2_configs.json`
- `results/gl_decoder_v2/candidate_generation_spec.json`
- `results/gl_decoder_v2/phase_a_metst_historical_audit/report.json`
- `results/gl_decoder_v2/phase_a_metst_historical_audit/config_agreement.csv`
- `results/gl_decoder_v2/phase_a_metst_historical_audit/prediction_agreement.csv`
- `results/gl_decoder_v2/combined_report.json`
- No `phase_b_current_cache/` directory when Phase A is D.

## Recommended Next Skill

No further modeling or robustness skill should run: the user-defined D-level stop condition has been reached.
