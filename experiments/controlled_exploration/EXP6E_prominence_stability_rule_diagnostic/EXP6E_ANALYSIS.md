# EXP-6E — Prominence Stability Rule Diagnostic

## Integrity

- Fresh parent replay and all required EXP-6D artifact checks: **PASS**.
- Fresh Shapley-style identity replay (`C_P + C_D = ΔG`): **PASS** at `≤1e-12`.
- This diagnostic adds no GLSD score, threshold, decoder, prediction, F1, Recognition, or STRS computation.

## Parent facts verified

- B/S lost TP: 13 events across 6 subjects; mean `ΔG=-0.263475`, `C_P=-0.340306`, `C_D=+0.076831`; all 13 have `|C_P|>|C_D|`.
- Normalization **BUFFERS** collapse; `ΔG_Ponly=-0.316597`, `ΔG_Donly=+0.100540`; raw-prominence route is supported and normalization route rejected.
- Parent lost-vs-retained signal is supported. B/C removed-FP coverage is limited to two subjects and is not treated as stable inference.

## Event-level distributions

See `event_distribution_summary.csv` for count, subject count, mean, median, standard deviation, IQR, min, and max for every preregistered quantity and group.

## Lost TP versus retained TP

- `Q_P`: AUC=0.890; direction consistency=5/5 (100.0%); leave-one-subject-out direction=YES; TP feature gate=PASS.
- `Q_G`: AUC=0.951; direction consistency=4/5 (80.0%); leave-one-subject-out direction=YES; TP feature gate=PASS.
- `Q_logP`: AUC=0.890; direction consistency=5/5 (100.0%); leave-one-subject-out direction=YES; TP feature gate=PASS.

TP stability gate: **PASS** (`Q_P, Q_G, Q_logP`).

## FP safety controls

- All traceable B/C FPs: 1802 events across 91 subjects.
- Removed B/C FPs with counterpart: 75 events across 2 subjects; safety evidence: **INSUFFICIENT**.
- No-counterpart topology disappearances: 0 events across 0 subjects; they receive no fabricated Q value.
- Lost-vs-all-FP separability: **SUPPORTED** (`Q_G, Q_P, Q_logP`). The ECDF and IQR overlap diagnostics are in `overlap_analysis.csv`.

## Dependence and feasibility

- `REFERENCE_TRANSITION_SPECIFIC = NO`; transition audit is in `reference_transition_analysis.csv`.
- `STABILITY_SIGNAL_TRANSFER = INCONCLUSIVE`: B/C supplies FP safety controls only, not a B/C lost-TP analogue; no cross-dataset TP transfer is asserted.
- `RESCUE_RISK = MODERATE`.
- `SAFE_STABILITY_SIGNAL_SUPPORTED`.
- `TRAINING_ONLY_RULE_FEASIBLE`.

## Verdict

Recommended next experiment: **PROSPECTIVE_STABILITY_RULE_VALIDATION**.

No threshold was searched and no candidate-specific GT fitting, feature combination, classifier, or outer-test rule was used.
