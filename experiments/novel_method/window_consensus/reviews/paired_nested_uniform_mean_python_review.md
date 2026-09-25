# Paired Nested Uniform Mean — Python Code Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-06
> **Scripts reviewed**: `my_method/window_consensus/run_paired_nested_uniform_mean.py`

## Pass Items

1. ✅ `run_paired_nested_uniform_mean.py:29-32,65-72` defines exactly the historical 3×3×3×3 Native grid and asserts 81 candidates; there is no grid expansion.
2. ✅ `run_paired_nested_uniform_mean.py:95-121` builds a strictly paired 79-video record set and obtains Uniform Mean only through the already reviewed `aggregate_raw`; Original and Uniform records are sorted and identity-checked one-to-one.
3. ✅ `run_paired_nested_uniform_mean.py:136-198` excludes the held subject from every config's pooled selection counts, uses exact rational F1, fewer FP, and fixed config index for deterministic selection, then evaluates only the selected config on the held subject.
4. ✅ `run_paired_nested_uniform_mean.py:201-230` blocks unless B0 reproduces 49/143/110 and all 29 selected config indices and values exactly match the historical Final Strong Native report.
5. ✅ `run_paired_nested_uniform_mean.py:255-279` implements a paired 1,000-repetition subject bootstrap with shared sampled subject indices and the fixed seed 20260905.
6. ✅ `run_paired_nested_uniform_mean.py:302-322` independently rebuilds the frozen-decoder O0/O1 subject counts used for the first requested bootstrap and rechecks the O0 anchor.
7. ✅ `run_paired_nested_uniform_mean.py:395-412` completes all raw provenance, B0 aggregate, and 29-fold historical configuration gates before B1 precomputation begins.
8. ✅ `run_paired_nested_uniform_mean.py:481-490` records identical search budgets, subject-disjoint selection, tie-break order, and absence of outer-test selection leakage in machine-readable output.
9. ✅ `run_paired_nested_uniform_mean.py:505-513` records that Uniform Mean and the Native grid were not modified and that recognition, weighting, Window-Consensus features, and backbone forward were not used.
10. ✅ Independent output reconciliation verified 2 summary rows, 58 outer rows, 58 selected-config rows, and 4,698 inner-score rows (=2×29×81). Every independently re-ranked fold selected the saved config, and outer rows sum to B0=49/143/110 and B1=49/144/110.
11. ✅ Both bootstrap outputs were independently recomputed from saved subject rows/source subject metrics and match the saved means and percentile intervals to machine precision.

## Failed / Repaired Items

None. No mathematical or implementation repair was required after execution.

## Constraint Direction Review

| File:line | Direction | LHS | RHS | Expected meaning |
|---|---|---|---|---|
| `run_paired_nested_uniform_mean.py:146` | `!=` | inner subject | outer held subject | outer-test subject is excluded from selection pooling |
| `run_paired_nested_uniform_mean.py:148` | minimize `-F1` | rank primary | exact pooled F1 | larger pooled Spotting F1 ranks first |
| `run_paired_nested_uniform_mean.py:148` | minimize | rank secondary | pooled FP | fewer FP breaks an exact F1 tie |
| `run_paired_nested_uniform_mean.py:235-239` | `>` / `<` | subject ΔF1 | 0 | positive is improved and negative is worse |

These are selection/reporting comparisons rather than optimization constraints; their directions match the locked protocol.

## Remaining Risks

- The experiment has only 29 bootstrap units. Both percentile intervals are consequently wide; the report preserves rather than suppresses this uncertainty.
- B1's selected `p=0.65` and `c_d=1.25` occur at grid endpoints in all folds. The user explicitly prohibited expansion, so no additional grid was tested.
- The default raw source is an absolute path outside the repository. SHA/provenance gates prevent silent substitution, and `--raw-dir` supports relocating the same verified source.

## Run Instructions

```bash
.venv/bin/python my_method/window_consensus/run_paired_nested_uniform_mean.py
```

## Expected Outputs

- `results/paired_nested_overlap_aggregation_sammlv/PAIRED_NESTED_OVERLAP_AGGREGATION_SAMMLV_CN.md`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_summary.csv`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_outer_metrics.csv`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_selected_configs.csv`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_parameter_frequency.csv`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_inner_scores.csv`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_bootstrap.json`
- `results/paired_nested_overlap_aggregation_sammlv/outputs/paired_nested_summary.json`

## Recommended Next Skill

- `result-report-generator` for evidence-only reporting. No next experiment is recommended or launched.
