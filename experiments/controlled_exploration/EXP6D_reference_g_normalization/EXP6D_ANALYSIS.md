# EXP-6D — Reference-G Normalization / Scale-Stability Diagnostic

## Integrity

- Parent EXP-6C facts and fresh canonical frozen-cell replay: **PASS**.
- The Shapley-style identity `C_P + C_D = ΔG` passes at `≤1e-12` for every comparable row.
- This is read-only: no GLSD/prediction/decoder/threshold/configuration/F1/Recognition/STRS change was made.

## Answers

1. B/S lost TP has 13 events across 6 subjects (Anchor-A=7, Anchor-C=6). Mean ΔG=-0.263475; mean C_P=-0.340306; mean C_D=0.076831.
2. The dominant source is **RAW_PROMINENCE**: `|C_P|>|C_D|` for 100.0% and `|C_D|>|C_P|` for 0.0%; median C_P=-0.267680, median C_D=0.107036. Range normalization therefore **BUFFERS** collapse rather than being its primary cause.
3. Holding the old denominator (`P_C/D_A`) gives mean ΔG_Ponly=-0.316597; the collapse remains. Changing only the denominator (`P_A/D_C`) gives mean ΔG_Donly=0.100540.
4. Same-peak lost TPs: 2/13; mean ΔP=-0.219947 and mean C_P=-0.220119 (YES). Exact `P_A/P_C`, `D_A/D_C`, and `G_A/G_C` are in `same_peak_smoothing_only.csv` (P 0.870600→0.650654; D 0.999813→0.998626; G 0.870763→0.651549, P 0.870600→0.650654; D 0.999813→0.998626; G 0.870763→0.651549). Shifted cases: 11/13, mean ΔP=-0.191622; peak-shift association=YES.
5. Shift/smoothing decomposition: old R_A peak remains a legal R_C peak in 0/11 shifted events. Therefore neither a response-scale nor candidate-switch prominence component is numerically identifiable for this subset under canonical peak semantics; every such field is NA rather than evaluating a non-peak.
6. Lost-vs-retained stability: SUPPORTED by predeclared subject-aware bootstrap (see `mechanism_bootstrap.csv`), not a cutoff search.
7. B/C removed FP: 75 exact R1 events; counterpart available=75, topology-disappeared=0. Lost-vs-removed-FP separability is YES on the counterpart-available subset; disappeared FPs remain a separate topology class.
8. Normalization route: **REJECTED**. Raw-prominence instability: **SUPPORTED**. Safe scale-stability signal: **SUPPORTED**. Recommended next experiment: **PROMINENCE_STABILITY_RULE_DIAGNOSTIC**.

## Limits

The representative TP is GT-centric only for diagnosis. No diagnostic quantity became a new score or decision rule; no test-label-guided feature engineering or threshold search was performed.
