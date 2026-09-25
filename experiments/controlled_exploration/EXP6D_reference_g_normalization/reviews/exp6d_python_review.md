# EXP-6D Python Code Review

> **Status**: passed_with_warnings
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-18
> **Scripts reviewed**: `controlled_exploration/EXP6D_reference_g_normalization/run_exp6d.py`

## Pass Items

1. ✅ `run_exp6d.py:73-93` calls canonical `peak_prominences` only after confirming the peak is present exactly once in the reference peak set; the reconstructed raw prominence is checked against canonical normalized `G * D` at `1e-12`.
2. ✅ `run_exp6d.py:101-123` implements both prescribed two-order Shapley contributions and raises `EXP6D_DECOMPOSITION_FAILED` if `C_P + C_D != ΔG` beyond `1e-12`.
3. ✅ `run_exp6d.py:126-157` fail-closes on the required EXP-6C parent facts, including the 7/6 TP→B split, six subjects, reported deltas, 13 G-dominant losses, 11 shifts, 75 B/C R1 FPs, and retained-TP ΔG.
4. ✅ `run_exp6d.py:187-213` uses the required 10,000 subject-level bootstrap replicates with fixed seed 100 and writes `INSUFFICIENT_SUBJECT_SUPPORT` when a comparison has fewer than two usable subject units.
5. ✅ `run_exp6d.py:216-248` replays only saved B/S selection cells (`E_7`, saved `k`, `rho`, `tau`, `R_A`, and `R_C`) and only retains the predeclared G1/G2/G3 diagnostic groups.
6. ✅ `run_exp6d.py:251-274` tracks only pre-existing B/C R1 removed FPs, selects an overlapping R_C counterpart deterministically, and leaves all retention quantities empty for a disappeared counterpart rather than assigning zero.
7. ✅ `run_exp6d.py:309-317` writes the required CSV/JSON audit artifacts, records prohibited operations in `protocol.json`, and records `new_method_executed=false`, `alternative_G_tested=false`, `formal_F1_recomputed=false`, and `canonical_GLSD_modified=false` in `replay_manifest.json`.
8. ✅ A clean `python3 -m py_compile` and a complete execution of the script succeeded. `parent_replay_check.csv` has zero failures and `decomposition_identity_check.csv` has 168/168 PASS rows.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `run_exp6d.py:217` | Initial execution referenced a non-existent `PARENT.ENGINE` alias. | Replaced it with the already imported canonical `ENGINE` module before the final execution. | fixed |
| 2 | `run_exp6d.py:200-201` | Initial paired-bootstrap indexing treated a 2-D indexed array as 3-D. | Materialized the sampled `(B, n, 2)` array and used `sampled[..., 0/1]`. | fixed |

## Remaining Risks

- `run_exp6d.py:206-209`: the lost-vs-removed-FP bootstrap has only two B/C counterpart subjects. It is correctly marked descriptive, but its apparent separation should not be treated as population-level significance or converted to a rule without a new held-out validation protocol.
- `run_exp6d.py:239-247`: none of the 11 shifted lost TPs retains the exact old peak at R_C. Response-scale versus candidate-switch prominence effects are therefore intentionally NA, not estimable.

## Run Instructions

```bash
python3 -m py_compile controlled_exploration/EXP6D_reference_g_normalization/run_exp6d.py
python3 controlled_exploration/EXP6D_reference_g_normalization/run_exp6d.py
```

## Expected Outputs

- `EXP6D_ANALYSIS.md`, factor/stability CSVs, counterpart lineage, subject table, bootstrap table, and identity check in `controlled_exploration/EXP6D_reference_g_normalization/`
- `protocol.json` and `replay_manifest.json` recording immutable diagnostic scope and source digests

## Recommended Next Step

- `PROMINENCE_STABILITY_RULE_DIAGNOSTIC` only as a separate, pre-registered diagnostic. Do not modify GLSD or test an alternative denominator from this result.
