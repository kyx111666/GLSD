# EXP-2A Development Log

### 2026-09-17 — Iteration 1: freeze the controlled scale-expansion protocol

**Reason:** Test whether the original three-value GLSD scale space is too coarse.

**Change:**

- Added `README.md` defining the sole experimental change and the 90-versus-210
  configuration comparison.

**Expected effect:** Inner LOSO may select intermediate or boundary scales and
improve outer-fold F1 if the original scale grid omitted useful smoothing widths.

**Document sync:** protocol specification yes | model design unchanged | main GLSD unchanged

### 2026-09-17 — Iteration 2: implement isolated expanded-grid runner

**Reason:** Execute the locked EXP-2A protocol without changing the production
GLSD implementation.

**Change:**

- Added `run_exp2a.py` with the 210-configuration grid, nested-LOSO selection,
  exact outer-count replay, archived-baseline comparison, selected-scale
  distribution, and paired subject bootstrap.

**Expected effect:** Produce auditable evidence for whether newly introduced
reference scales are selected and whether their use improves pooled outer F1.

**Document sync:** README yes | model design unchanged | main GLSD unchanged

### 2026-09-17 — Iteration 2 result

The 210-configuration nested-LOSO run completed for all four settings; exact
outer-count replay passed for all 246 folds.

| Setting | Baseline F1 | Expanded F1 | Delta |
|---|---:|---:|---:|
| ME-TST/SAMMLV | 0.288288 | 0.261398 | -0.026890 |
| ME-TST/CAS(ME)3 | 0.094164 | 0.105207 | +0.011043 |
| BoostingVRME/SAMMLV | 0.283784 | 0.313725 | +0.029942 |
| BoostingVRME/CAS(ME)3 | 0.112888 | 0.118449 | +0.005561 |

**Conclusion:** The expansion helps consistently at dataset level only on
CAS(ME)3 and is not robust enough to replace the locked 90-configuration grid as
the universal primary paper search.

### 2026-09-17 — Iteration 3: make scale-distribution comparison explicit

**Reason:** The required scale-distribution artifact should directly compare the
90-configuration baseline with the 210-configuration expansion.

**Change:**

- Updated `run_exp2a.py` so `selected_scale_distribution.csv` reports baseline
  and expanded counts/fractions side by side for all seven scale values.

**Expected effect:** The required CSV is self-contained and no longer requires
joining the per-fold audit file to recover the baseline distribution.

**Document sync:** README yes | model design unchanged | main GLSD unchanged
