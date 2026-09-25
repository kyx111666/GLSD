# EXP-6C Python Review

## Scope

Reviewed `run_exp6c.py` as a frozen EXP-2A/EXP-2C reference-scale diagnostic.

## Result: PASS

- The script imports the EXP-6B replay implementation and uses its evaluator labels and E_7 replay semantics; it does not reimplement a competing decoder.
- `verify_parent_artifacts()` fails closed on the required parent replay, identity, same-reference isolation, B/S all-7 TP→B counts, fixed-reference E_7→E_3 zero TP→B counts, M/S reference collapse, B/C candidate deletion, and B/C R1 FP lineage.
- `rho`, `tau`, `k`, and reference choices come only from saved EXP-2A/EXP-2C artifacts. The GT labels are used after replay only to organize diagnostic representatives; no selection or mutation path consumes them.
- Global evidence is decomposed using the actual reference response range and the exact identity `raw_prominence = G * range`. Local correspondence follows the existing radius/tolerance and effective-width deduplication; no matching window is introduced.
- The subject bootstrap is restricted to the three predeclared score deltas with 10,000 resamples and seed 100. It emits `INSUFFICIENT_SUBJECT_SUPPORT` rather than a significance claim when paired subject support is inadequate.
- The manifest explicitly records that no new method, outer-test rule, or canonical GLSD change was executed.

## Verification run

`python3 -m py_compile controlled_exploration/EXP6C_reference_scale_candidate_diagnostic/run_exp6c.py` and the frozen replay completed successfully. See `identity_check.csv` for the machine-readable integrity rows.
