# Selective Window-Consensus Veto — Python Code Review

> **Status**: passed_with_warnings
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-06
> **Scripts reviewed**: `my_method/window_consensus/run_selective_window_consensus_veto.py`

## Pass Items

1. ✅ `run_selective_window_consensus_veto.py:24-38` fixes the anchor, 29 subjects, 79 videos, `k_p=5`, bootstrap count/seed, exactly four delta values, exactly four tau values, and only C0–C3.
2. ✅ `run_selective_window_consensus_veto.py:94-109` derives Native events and signed margin from the previously audited compact frozen cache and separately asserts that raw-dump O0 and cache O0 event identities match for every video.
3. ✅ `run_selective_window_consensus_veto.py:113-138` reuses the verified causal deletion labels and Window-Pair C2 extractor; the run blocks on any ambiguous causal label.
4. ✅ `run_selective_window_consensus_veto.py:149-154` blocks unless C0 is exactly 49/143/110 and the candidate bank reconciles to exactly 49 KEEP plus 143 PRUNE events.
5. ✅ `run_selective_window_consensus_veto.py:158-167` exposes only 4 C1 configs, 4 C2 configs, and the prescribed 16 Cartesian C3 configs; no grid expansion exists.
6. ✅ `run_selective_window_consensus_veto.py:170-191` isolates the controls correctly: C1 uses margin only, C2 uses pair agreement only with single-view forced keep, and C3 requires pair coverage plus both `H_s<=delta` and `A_ratio<tau`.
7. ✅ `run_selective_window_consensus_veto.py:194-245` keeps original Native event intervals unchanged, deletes only selected existing events, reevaluates with the verified event evaluator, and asserts exact TP/FP deletion reconciliation.
8. ✅ `run_selective_window_consensus_veto.py:248-257` implements deterministic selection using exact rational F1, then fewer FP, fewer TP lost, smaller pruning rate, smaller delta, and smaller tau.
9. ✅ `run_selective_window_consensus_veto.py:269-310` excludes the held subject before pooling all selection metrics; the outer-test subject is evaluated only after its configuration is selected at lines 312–340.
10. ✅ `run_selective_window_consensus_veto.py:367-415` calculates subject improved/equal/worse, leave-one-subject aggregate sensitivity, and the fixed 1000-repeat paired subject bootstrap from frozen OOF counts.
11. ✅ `run_selective_window_consensus_veto.py:534-549` applies the requested STRONG gates and assigns WEAK only when C3 improves over C0 but misses at least one required STRONG gate; the preferred 6.16 ratio is reported separately.
12. ✅ `run_selective_window_consensus_veto.py:636-651` saves portable CSV/JSON/Markdown outputs; results are not console-only.
13. ✅ Independent artifact reconciliation passed: 4 summary rows, 116 outer rows, 116 selected-config rows, 696 inner-score rows, 768 event-trace rows, and 24 parameter-frequency rows. Every fold/config selection was independently reranked and matched.
14. ✅ Independent provenance comparison found exact matches for all 192 `H_s` values against the prior pruning audit and all 192 `A_ratio`/coverage records against the prior Window-Pair audit. The 1000 bootstrap mean and interval also reproduced exactly.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | initial implementation, candidate-bank margin source | `H_s` was initially recomputed from the raw-dump fresh O0 curve and differed from the previous compact-cache audit by at most `6.27e-6` | restored the verified compact-cache margin pathway and added a 79/79 raw/cache event-identity gate at lines 94–109 | fixed |

The repair did not change the final counts, selected configurations, bootstrap result, or decision.

## Constraint Direction Review

| File:line | Direction | LHS | RHS | Expected meaning |
|---|---|---|---|---|
| `run_selective_window_consensus_veto.py:151` | `==` | C0 TP/FP/FN | `49/143/110` | frozen anchor must match exactly |
| `run_selective_window_consensus_veto.py:172` | `≤` | `H_s` | selected `delta` | event is in the low-margin zone |
| `run_selective_window_consensus_veto.py:173` | `<` | `A_ratio` | selected `tau` | two views have insufficient relative agreement |
| `run_selective_window_consensus_veto.py:181-182` | `AND` | pair, low margin, low agreement | all required | C3 deletes only when every safety condition holds |
| `run_selective_window_consensus_veto.py:270` | `!=` | training subject | held subject | held subject cannot influence parameter selection |
| `run_selective_window_consensus_veto.py:535-541` | `>`, `<`, `≥` | C3 decision metrics | locked thresholds | implements the user-specified STRONG gates |

The inequality directions match the supplied protocol and were not altered during review.

## Remaining Risks

- C3 selected both upper grid endpoints (`delta=0.20`, `tau=0.80`) in 29/29 folds. This is reported as a stability warning only; the stop rule forbids expanding the grid.
- The paired bootstrap mean is positive, but its 95% interval crosses zero (`[-0.003663, 0.040053]`).
- C3 retains 46 TP, missing the pre-registered STRONG threshold of 47 by one TP; this is why the final status is WEAK-GO despite positive pooled and leave-one-subject results.
- C1 is interpreted as a pure margin-only control over all Native events, while C2 is a pure agreement-only control over two-view events. This interpretation is explicitly persisted in the report and JSON.
- The default raw-dump location is machine-specific; `--raw-dir` can relocate the same verified source, whose manifest hash is saved.

## Run Instructions

```bash
.venv/bin/python my_method/window_consensus/run_selective_window_consensus_veto.py
```

## Expected Outputs

- `results/selective_window_consensus_veto_sammlv/SELECTIVE_WINDOW_CONSENSUS_VETO_SAMMLV_CN.md`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_summary.csv`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_outer_metrics.csv`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_selected_configs.csv`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_inner_scores.csv`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_event_trace.csv`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_parameter_frequency.csv`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_bootstrap.json`
- `results/selective_window_consensus_veto_sammlv/outputs/selective_veto_summary.json`

## Recommended Next Skill

None. The development-gate protocol requires stopping without method modification or grid expansion.
