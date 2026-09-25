# EXP-6F — Training-Only Prominence-Stability Rescue Validation

**Status:** `POST_HOC_DEVELOPED_NESTED_EVALUATED_VARIANT`; this is not independent prospective or untouched external validation.

## Integrity

- EXP-6E parent facts: PASS.
- Canonical and EXP-2C frozen baseline replay: PASS.
- Nested eta selection: PASS; Q_P only; no trainable parameters; outer test labels or F1 were never used to tune eta.

## Primary results vs EXP-2C

| Setting | 2C F1 | EXP6F F1 | ΔF1 | TP/FP/FN | 95% CI |
|---|---:|---:|---:|---:|---:|
| ME-TST/SAMMLV | 0.291525 | 0.291525 | +0.000000 | 43.0/93.0/116.0 | [+0.000000, +0.000000] |
| ME-TST/CAS(ME)3 | 0.104548 | 0.104548 | +0.000000 | 100.0/955.0/758.0 | [+0.000000, +0.000000] |
| BoostingVRME/SAMMLV | 0.277778 | 0.294118 | +0.016340 | 45.0/102.0/114.0 | [-0.001812, +0.036648] |
| BoostingVRME/CAS(ME)3 | 0.121530 | 0.120896 | -0.000633 | 116.0/945.0/742.0 | [-0.001307, -0.000124] |

## Rescue accounting

- Historical B/S lost TP recovered: 8/13.
- New rescued TP/FP: 5.0/23.0; rescue precision=0.179.
- B/C removed FP reintroduced: 0; total B/C rescued FP=10.
- eta OFF folds: 192; active folds: 54.

## Recognition / STRS

| Setting | Method | Spot F1 | Recognition F1 | STRS |
|---|---|---:|---:|---:|
| ME-TST/SAMMLV | EXP6F | 0.291525 | 0.7254 | 0.2115 |
| ME-TST/SAMMLV | canonical_replay | 0.288288 | 0.6937 | 0.2000 |
| ME-TST/CAS(ME)3 | EXP6F | 0.104548 | 0.5307 | 0.0555 |
| ME-TST/CAS(ME)3 | canonical_replay | 0.094164 | 0.5374 | 0.0506 |

## Verdict

- `PROMINENCE_RESCUE_VALIDATED`: **NO**.
- `STRONG_VALIDATION`: **NO**.
- Recommended status: **REJECT**.
- Additional method cost: second reference branch YES; learned parameters 0; new scalar eta selected training-only; additional candidate matching; decoder-only runtime in `runtime.csv`.
