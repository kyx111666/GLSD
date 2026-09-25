# EXP-2B Development Log

### 2026-09-17 — Iteration 1: freeze the controlled radius-expansion protocol

**Reason:** Test whether the original three-value local structural context is
too narrow, with particular attention to CAS(ME)3.

**Change:**

- Added `README.md` defining radius as the sole experimental change, the
  90-versus-150 configuration comparison, the locked evaluator, and the output
  contract.

**Expected effect:** Inner LOSO may select radii 4 or 5 and improve outer-fold
F1 if longer local context captures useful event structure missed by the
baseline radius set.

**Document sync:** protocol specification yes | model design unchanged | main GLSD unchanged

### 2026-09-17 — Iteration 2: implement isolated expanded-radius runner

**Reason:** Execute the locked EXP-2B protocol without changing the production
GLSD implementation.

**Change:**

- Added `run_exp2b.py` with the 150-configuration grid, nested-LOSO selection,
  exact outer-count replay, archived-baseline comparison, selected-radius
  distribution, and paired subject bootstrap.

**Expected effect:** Produce auditable evidence for whether radii 4 or 5 are
selected and whether their use improves pooled outer F1, especially on CAS(ME)3.

**Document sync:** README yes | model design unchanged | main GLSD unchanged

### 2026-09-17 — Iteration 2 result

The 150-configuration nested-LOSO run completed for all four settings; exact
outer-count replay passed for all 246 folds.

| Setting | Baseline F1 | Expanded F1 | Delta |
|---|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.272727 | -0.015561 |
| ME-TST/CAS(ME)3 | 0.094164 | 0.090954 | -0.003210 |
| BoostingVRME/SAMMLV | 0.283784 | 0.283784 | 0.000000 |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.112888 | 0.000000 |

**Conclusion:** Wider local context does not benefit CAS(ME)3 or any other
locked setting. Keep the baseline radius set. Nine folds selected a new radius,
but the inner-selection improvements did not translate to outer-F1 gains.

### 2026-09-17 — Iteration 3: add radius-distribution and CAS(ME)3 analysis

**Reason:** The experiment requires an explicit selected-radius comparison and
special attention to whether CAS(ME)3 benefits.

**Change:**

- Added `analysis.md` with F1 deltas, paired intervals, selected-radius
  distributions, pooled dataset comparisons, and a CAS(ME)3-specific decision.

**Expected effect:** Make the controlled-exploration decision auditable without
requiring manual joins across result files.

**Document sync:** README yes | results yes | main GLSD unchanged
