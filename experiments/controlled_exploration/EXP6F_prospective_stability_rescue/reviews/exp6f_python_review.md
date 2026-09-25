# EXP-6F Python review

Status: passed with documented experimental outcome.

- The implementation limits the decision feature to canonical raw-prominence `Q_P`; `Q_G`, log-prominence, combinations, margins, and altered matching radii are absent.
- Both outer baseline configurations are reselected from nested count tables and checked against archived EXP-2A/2C selections before outer prediction decoding.
- Eta candidates are `OFF` plus the five specified unlabeled training-candidate quantiles.  Inner pooled TP/FP/FN selection uses the stated F1, precision, FP, rescue-count, OFF, and eta-order tie-break sequence.
- Candidate matching is deterministic: canonical `max(1, round(0.5*k))`, then temporal distance, higher raw C prominence, and temporal position.
- The full conflict rule remains chronological and native for BoostingVRME; rescued events retain the original R_A interval.
- The paired bootstrap contains 10,000 subject-resamples with seed 100.  A separate output audit recomputed all reported F1 values from TP/FP/FN and reconciled subject totals with every pooled setting.
- ME-TST+ STRS is re-evaluated from the frozen Strategy-1 emotion cache and archived evaluator.  Small compatibility shims only restore removed `sklearn.metrics.confusion_matrix` and pandas `DataFrame.append` APIs; canonical M/S independently replayed exactly as 48/126/111, Recognition 0.6937, STRS 0.2000.

Outcome note: the B/C paired 95% CI is entirely negative, so the implementation correctly reports `PROMINENCE_RESCUE_VALIDATED = NO`, `STRONG_VALIDATION = NO`, and `REJECT`; no second rescue rule was tested.
