# Window-Pair Consensus Separability — Python Code Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-06
> **Scripts reviewed**: `my_method/window_consensus/run_window_pair_consensus_audit.py`

## Pass Items

1. ✅ `run_window_pair_consensus_audit.py:157-200` reconstructs the B0 candidate bank with each subject's frozen Final Strong Native config, uses the verified evaluator, applies the imported causal deletion label function, and blocks unless the aggregate is exactly 49/143/110.
2. ✅ `run_window_pair_consensus_audit.py:104-110` counts actual occurrences of the candidate global peak in `global_indices`, excludes one-view candidates as `SINGLE-VIEW-CANDIDATE`, and admits only exactly two-view candidates to pair features.
3. ✅ `run_window_pair_consensus_audit.py:112-153` implements only C1–C5 from the two raw score rows: exact peak disagreement, prescribed ratio with `1e-8`, threshold-free local peak support, normalized nearest-peak disagreement, and common-global-frame Pearson correlation.
4. ✅ `run_window_pair_consensus_audit.py:91-101` runs `scipy.signal.find_peaks` independently within each raw window with no height threshold and restricts candidates to global `peak±k_p`; the deterministic tie is nearest distance then earlier global frame.
5. ✅ `run_window_pair_consensus_audit.py:218-255` excludes missing values rather than imputing them in scalar diagnostics, retains raw ROC/AP direction, and computes semantic-direction ROC separately for interpretation.
6. ✅ `run_window_pair_consensus_audit.py:263-272,286-307` computes W2 medians from the LOSO training matrix only, applies them to train/test, then fits `StandardScaler` and the exact fixed liblinear balanced LR only on training candidates.
7. ✅ `run_window_pair_consensus_audit.py:275-335` excludes the held subject before all preprocessing and fitting and verifies exactly one OOF probability for every primary candidate and each of W0/W1/W2.
8. ✅ `run_window_pair_consensus_audit.py:382-420` generates one shared 1000×29 subject-index matrix for W0/W1/W2 and reports the paired W2−W0 PR-AUC distribution with seed 20260906.
9. ✅ `run_window_pair_consensus_audit.py:423-442` applies the declared STRONG/WEAK/NO-GO gate without tuning a model or probability threshold; the scalar-useful threshold affects interpretation only.
10. ✅ `run_window_pair_consensus_audit.py:607-618` records that GT is absent from features and that no decoder, recognition, stitched height, morphology, weighting, aggregation search, classifier tuning, threshold tuning, or backbone forward was used.
11. ✅ Independent output reconciliation verified 192 candidates = 49 KEEP + 143 PRUNE, zero ambiguous interactions, 186 two-view primary candidates, 6 single-view exclusions, 558 OOF rows (=186×3), 87 fold diagnostics, and 87 subject-metric rows.
12. ✅ Independent recomputation from the saved OOF table reproduced all W0/W1/W2 ROC-AUC and PR-AUC values and the 1000-subject-bootstrap mean/CI to machine precision.

## Failed / Repaired Items

None. No implementation repair was required after the diagnostic run.

## Constraint Direction Review

| File:line | Direction | LHS | RHS | Expected meaning |
|---|---|---|---|---|
| `run_window_pair_consensus_audit.py:96` | `≤` | distance from raw-view peak | `k_p` | only local peaks inside candidate ±k_p count as support |
| `run_window_pair_consensus_audit.py:132` | `≥` | common aligned frames | 5 | C5 requires at least five shared valid frames |
| `run_window_pair_consensus_audit.py:182` | `==` | coverage status | `PAIR` | primary set requires exactly two raw views |
| `run_window_pair_consensus_audit.py:279` | `!=` | training subject | held subject | held subject is excluded from scaler/LR fitting |
| `run_window_pair_consensus_audit.py:426` | `≥` | semantic-oriented scalar ROC | 0.60 | fixed auxiliary useful-separation interpretation |
| `run_window_pair_consensus_audit.py:430-433` | `≥`, `>`, `>` | W2 metrics/delta | locked decision thresholds | implements the user's STRONG decision conditions |

The inequality directions match the stated definitions. They were reviewed but not changed.

## Remaining Risks

- Only 186 candidates enter the primary analysis, with 47 KEEP labels; nine subjects have undefined within-subject PR-AUC because their candidate labels are single-class or absent.
- C3 is nearly constant (both views support a local peak for almost every candidate), limiting its possible ranking value.
- The auxiliary `oriented ROC≥0.60` definition was declared before execution because the protocol used the qualitative phrase “useful separation.” It changes only the decision interpretation, not any reported raw metric or model input.
- The default raw source is an absolute path. The existing per-file SHA and reconstruction gates protect provenance, while `--raw-dir` supports relocating the same verified source.

## Run Instructions

```bash
.venv/bin/python my_method/window_consensus/run_window_pair_consensus_audit.py
```

## Expected Outputs

- `results/window_pair_consensus_separability_sammlv/WINDOW_PAIR_CONSENSUS_SEPARABILITY_AUDIT_SAMMLV_CN.md`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_candidate_features.csv`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_scalar_diagnostics.csv`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_multivariate_metrics.csv`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_oof_predictions.csv`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_subject_metrics.csv`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_fold_diagnostics.csv`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_bootstrap.json`
- `results/window_pair_consensus_separability_sammlv/outputs/window_pair_summary.json`

## Recommended Next Skill

- None in this task. The protocol requires stopping after the separability decision.
