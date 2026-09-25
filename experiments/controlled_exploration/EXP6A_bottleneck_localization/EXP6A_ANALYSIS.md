# EXP-6A — End-to-End FN / Event Bottleneck Localization

## Integrity gates

- Exact replay: **PASS**.
- FN A/B/C/D accounting: **PASS**.
- The replay uses saved outer-fold selections and frozen temporal responses; it does not retrain, reselection, alter the evaluator, or execute a new method.

## Canonical FN localization

- ME-TST+/SAMMLV: A=48, B=63, C=0, D=0; P0/P1/P2/formal=111/48/48/48.
- ME-TST+/CAS(ME)3: A=486, B=276, C=0, D=0; P0/P1/P2/formal=372/96/96/96.
- BoostingVRME/SAMMLV: A=40, B=77, C=0, D=0; P0/P1/P2/formal=119/42/42/42.
- BoostingVRME/CAS(ME)3: A=386, B=352, C=0, D=0; P0/P1/P2/formal=472/120/120/120.

SAMMLV is score-filter dominated in the canonical replay (M/S B=63 of 111 FN; B/S B=77 of 117), while CAS(ME)3 has a large pre-threshold A component (M/C A=486 of 762; B/C A=386 of 738). ME-TST+ has C=0 in both settings, as expected from its no-extra-suppression adapter. Boosting has C=0 canonically and only 3 C cases across the two expanded variants, so its chronological native conflict rule is not the observed bottleneck.

Across canonical A misses, 607/960 have a reference peak inside the GT but formal interval IoU < 0.5; 353/960 have no peak inside GT. Thus the A evidence is predominantly geometry-limited, not simply candidate absent. Canonical B=768; its P0→P1 oracle losses equal B per setting, so score filtering is also a strong, separate bottleneck.

### Canonical B-score diagnostic

- ME-TST+/SAMMLV: B=63, margin≤0.05=3, margin≤0.10=4, mean/median=0.492/0.568, lower G/L/tie=23/26/14.
- ME-TST+/CAS(ME)3: B=276, margin≤0.05=3, margin≤0.10=7, mean/median=0.354/0.395, lower G/L/tie=48/154/74.
- BoostingVRME/SAMMLV: B=77, margin≤0.05=2, margin≤0.10=6, mean/median=0.463/0.564, lower G/L/tie=1/69/7.
- BoostingVRME/CAS(ME)3: B=352, margin≤0.05=11, margin≤0.10=22, mean/median=0.274/0.299, lower G/L/tie=14/300/38.

Most B margins are not merely near-threshold cases, and L is usually the lower component outside M/S. These are descriptive diagnostics only; no tau is changed.

## EXP-2C explanation

### M/S — TP −5, FP −33, F1 +0.003237

All five lost canonical TPs become **B** in 2C (TP→B=5; no TP→A/C/D). The 2C score/evidence rule filters 35 canonical FP by R2 lineage while 91 persist; two newly generated final FP offset this, giving net FP −33. The precision gain from fewer FP outweighs the five lost TPs, which explains the higher pooled Raw Spotting F1.

### M/C — TP +4, FP −130

The net TP gain is driven by A→TP=9 and B→TP=3, against TP→A=2 and TP→B=6. Of canonical FP, R1=996 and R2=9 disappear; R1 dominates, so the observed net FP reduction is principally candidate-pool/reference change rather than threshold filtering.

### B/S — the focal loss

Canonical→2C has TP→B=4 and B→TP=2, yielding net TP −2 with no TP→A/C/D. More directly, 2A→2C has TP→B=8 and no compensating B→TP: the 2A→2C drop is entirely **B / score-threshold stage**, not conflict or final matching. This supports reference/evidence coupling as the next diagnostic focus, rather than a reference-candidate or Boosting-conflict explanation.

### B/C — TP −4, FP −213

FP lineage is R1=878, R2=42, R3=1, persistent=227. R1 dominates the removed canonical FP, so the large net reduction is mainly candidate-pool shrinkage; the small R2/R3 counts do not support filtering or conflict as the primary cause. The TP change includes TP→A=1, TP→B=7, TP→C=1, partly offset by A→TP=4 and B→TP=1.

## Oracle and gates

`oracle_coverage.csv` reports all exact differences. For every canonical setting oracle_P2 equals formal_TP, so formal matching contributes 0 and is not worth further investigation. P1→P2 conflict loss is 0 for all four canonical settings; it is 1 and 2 only for expanded Boosting/CAS(ME)3, with no cross-setting recurrence.

- Reference-candidate bottleneck: **MIXED** — true no-peak A cases exist, but most canonical A cases have a peak inside GT and fail geometry.
- Event-geometry bottleneck: **SUPPORTED** — 607/960 canonical A events are peak-present but formal-IoU-incompatible, recurring in all four settings.
- Scoring/filter bottleneck: **SUPPORTED** — canonical B=768, P0→P1 loss=768, and the B/S 2A→2C TP loss is entirely TP→B.
- Boosting conflict bottleneck: **NOT SUPPORTED** — no canonical C and only three expanded C cases.
- Final matching major bottleneck: **NO** — oracle_P2−formal_TP=0 for every canonical setting.

**Recommended single next experiment: `REFERENCE_EVIDENCE_CROSS_REPLAY`**. It is the gate-consistent follow-up for the B/S TP→B transitions and P0→P1 score loss. It should use the existing EXP-2A/2C frozen responses and selections; it is not executed in EXP-6A.

## Recognition / STRS

Formal prediction sets are replayed, not modified. Recognition F1 and STRS therefore remain the archived values and were not recomputed. Event, prediction, and GT identities are retained in the CSV/JSON artifacts for a later same-origin recognition replay if an event set is genuinely changed.
