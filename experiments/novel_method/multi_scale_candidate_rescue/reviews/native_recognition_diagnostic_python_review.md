# Python implementation review: native attribution + recognition audit

Review status: **PASS with statistical limitation noted**. No blocking implementation defect found.

## Explicit pass items

1. **PASS — grids remain locked.** `run_native_parameter_recognition_audit.py:38-50` constructs exactly the four requested three-value single-factor grids and the existing 3×3×3×3 Full Tuned Native grid; all other native values stay at the Author setting.
2. **PASS — nested LOSO has no outer-label leakage.** `run_native_parameter_recognition_audit.py:87-98` removes the held subject before selection and evaluates that subject only after selecting on pooled train-subject counts.
3. **PASS — tie-break is protocol-exact and deterministic.** `run_native_parameter_recognition_audit.py:67-75` ranks by pooled F1, fewer FP, fewer rescue events, then fixed config index.
4. **PASS — baseline identity is guarded.** `run_native_parameter_recognition_audit.py:101-107` aborts unless Author Native is exactly 53/184/106.
5. **PASS — softmax is numerically stable.** `run_native_parameter_recognition_audit.py:119-123` subtracts the row maximum before exponentiation.
6. **PASS — cached logits are validated before use.** `run_native_parameter_recognition_audit.py:136-149` checks five-class shape, temporal score/logit alignment, and finite values for all 79 records.
7. **PASS — requested recognition evidence is implemented without feature expansion.** `run_native_parameter_recognition_audit.py:151-175` computes only R1, R2, R3 and center/mean/max over the fixed peak±k_p window.
8. **PASS — exact weak-pool identity is enforced.** `run_native_parameter_recognition_audit.py:177-179` aborts unless the reused trace contains exactly 45 rows and 2 positives.
9. **PASS — decision rule was frozen and applied literally.** `run_native_parameter_recognition_audit.py:181-199` requires both ROC-AUC ≥ 0.60 and PR-AUC ≥ 1.5×prevalence for at least one evidence.
10. **PASS — deterministic rerun.** All four required machine-readable artifacts were regenerated independently and had byte-identical SHA-256 hashes.

## Cross-result checks

- Full Tuned Native selections matched the prior pooled-objective diagnostic in all 29/29 outer folds.
- Full Tuned Native reproduced TP/FP/FN = 49/143/110 and F1 = 0.2792022792.
- Required artifacts exist and parse; the recognition trace has 45 rows, nine evidence columns, and two positive labels.

## Non-blocking limitation

Only two positive weak candidates exist. ROC-AUC and especially PR-AUC are therefore high-variance descriptive feasibility statistics. The GO status must not be represented as a demonstrated decoder improvement or generalization result.
