# EXP-6B Python review

> **Status**: passed_with_warnings
> **Reviewer**: python-code-reviewer
> **Scripts reviewed**: `controlled_exploration/EXP6B_reference_evidence_cross_replay/run_exp6b.py`

## Pass Items

1. ✅ `run_exp6b.py:99-106` reads only the saved EXP-2A/EXP-2C fold-selection CSVs and asserts identical subject sets; it does not call a selector.
2. ✅ `run_exp6b.py:42-51,109-118` imports EXP-2C's archived `DecoupledCurveFeatures` and uses the archived base feature class for EXP-2A all-7 evidence.
3. ✅ `run_exp6b.py:123-135` preserves the archived threshold, chronological conflict, and greedy matching logic while exposing P0/P1/P2 rather than replacing the evaluator.
4. ✅ `run_exp6b.py:168-172` fails closed if same-reference candidate/event identity or G equality (tolerance `1e-12`) is violated.
5. ✅ `run_exp6b.py:235-244,289-292` checks both identity cells against retained EXP-6A prediction lists and stops on either count replay or prediction identity failure.
6. ✅ `run_exp6b.py:198-206` implements only the predeclared paired subject bootstrap (`N=10000`, `seed=100`).
7. ✅ `run_exp6b.py:333-349` saves CSV/JSON/Markdown artifacts and records source checksums plus `canonical_modified=false`, `new_method_executed=false`, and `outer_test_cell_selected=false`.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `run_exp6b.py:315-326` | Early aggregation did not preserve both reference GT contrasts in the summary. | Re-derived all four contrasts from the retained unified GT identity. | fixed |

## Remaining Risks

- `run_exp6b.py:111-112` mutates module-level scale globals, but only inside the standalone diagnostic process. Do not import this script into a process that runs another GLSD experiment concurrently.
- Exact EXP-2A/2C prediction lists are retained through EXP-6A's replay artifact; original EXP-2A/2C files supply the formal saved per-subject counts and configurations.

## Run Instructions

```bash
python3 controlled_exploration/EXP6B_reference_evidence_cross_replay/run_exp6b.py
```

## Expected Outputs

- `cross_replay_results.csv`, `cross_replay_subject.csv`, `factor_contrasts.csv`, and `factor_contrast_bootstrap.csv`
- candidate/score/GT/FP lineage CSVs, identity checks, protocol/manifest, and `EXP6B_ANALYSIS.md`

## Recommended Next Step

- `REFERENCE_SCALE_CANDIDATE_DIAGNOSTIC` (the predeclared follow-up for the B/S reference-dominant finding).
