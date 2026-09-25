# EXP-6C — Reference-Scale Candidate Diagnostic

## Integrity

- Parent EXP-6B replay, identity cells, same-reference isolation, B/S TP→B counts, and B/C R1 FP lineage were rechecked from retained artifacts and fresh frozen replays: **PASS**.
- All primary comparisons use fixed all-7 evidence (`E_7`), saved foldwise `rho`/`tau`, and `S=(G+L)/2`. No rule, scale, threshold, fusion, or decoder was selected from outer-test labels.

## Answers

1. B/S has 13 TP→B events (Anchor-A=7; Anchor-C=6) across 6 outer subjects. The gate verdict is **REFERENCE_G_COLLAPSE_SUPPORTED**.
2. Mean TP→B changes are Δraw-prominence=-0.195979, Δrange=-0.055314, ΔG=-0.263475, ΔL=-0.051339, ΔS=-0.157407; the G-source label is **RAW_PROMINENCE**. Exact per-event values are retained in `tp_to_b_detail.csv` and `G_decomposition.csv`.
3. G-dominant/L-dominant/other TP losses are 13/0/0; all scale-level correspondence and median drivers are in `L_alignment_decomposition.csv` (no alignment tolerance was searched).
4. Systematic representative peak shift: **YES** (11/13); candidate multiplicity loss: **NO** (4/13).
5. Retained TP (78 events) has mean ΔG=-0.015650, ΔL=0.000000, ΔS=-0.007825; compare with the lost-TP values above rather than inferring from a handful of examples.
6. B→TP reverse control contains 2 events; its event-level score decomposition is in `b_to_tp_control.csv`, which tests whether the reference effect is directionally mirrored without fitting a rule.
7. B/C has 75 exact R1 removed final FPs. Its dominant topology label is **FP-PEAK-SHIFTED**; each candidate's R_A raw/G/L/S evidence and R_C potential-event check are in `bc_removed_fp_reference.csv`.
8. Lost-TP versus removed-FP structural separability is **NO**. `All four R_A raw-prominence/G/L/S ranges overlap, so these diagnostics do not support a simple unified deterministic correction.`
9. Subject-aware predeclared bootstrap rows are in `mechanism_bootstrap.csv`; rows marked `INSUFFICIENT_SUBJECT_SUPPORT` are intentionally not interpreted as significance.
10. A GLSD modification is **not justified by this diagnostic alone**. Recommended next experiment: **REFERENCE_G_NORMALIZATION_DIAGNOSTIC**.

## Limits

The GT-centric representative is diagnostic only: formal evaluation, prediction sets, Recognition F1, and STRS were not recomputed or changed. The B/C FP transition records R1 candidate disappearance; it does not test candidate union, dual references, ensembles, adaptive scale selection, new weights, calibration, or a new threshold.
