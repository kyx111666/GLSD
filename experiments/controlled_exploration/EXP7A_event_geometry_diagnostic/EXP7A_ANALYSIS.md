# EXP-7A — Event Geometry Failure Decomposition

## Integrity

- Parent canonical replay: **PASS**. Fresh P2 lists exactly match the archived canonical predictions and per-subject TP/FP/FN counts.
- Parent FN accounting: **PASS**. A/B are {'ME-TST+/SAMMLV': (48, 63), 'ME-TST+/CAS(ME)3': (486, 276), 'BoostingVRME/SAMMLV': (40, 77), 'BoostingVRME/CAS(ME)3': (386, 352)}; canonical A total is 960; peak-present geometry-limited A is 607.
- This is diagnostic only: no GLSD change, new geometry, prediction set, F1, Recognition, or STRS was produced.

## Primary population

The fresh primary population contains **607** canonical A events with at least one reference peak inside the closed GT interval and no P0 event at IoU ≥ 0.5. The representative candidate is deterministically the in-GT P0 candidate with greatest IoU (then nearest GT center).

## Setting decomposition

- M/S: n=18; dominant GEO-BOTH-SINGLE; width-only=18; shift-only=17.
- M/C: n=258; dominant GEO-WIDTH; width-only=256; shift-only=56.
- B/S: n=23; dominant GEO-BOTH-SINGLE; width-only=16; shift-only=17.
- B/C: n=308; dominant GEO-WIDTH; width-only=248; shift-only=101.

The small-SAMMLV and large-CAS(ME)3 settings do not have identical topology, but width-only capacity recurs in every setting (16/18 or more) and dominates both CAS(ME)3 settings.

## Oracle capacity (not new predictions)

- Width-only repairable: 538/607 (88.6%).
- Shift-only repairable: 191/607 (31.5%).
- Both-single: 160/607 (26.4%); coupled: 36/607 (5.9%); hard: 2/607 (0.3%).
- At least one single-parameter oracle repairs 569/607 (93.7%); allowing joint center+width adds the 36 coupled cases, reaching 605/607 (99.7%) diagnostic capacity.
- These are GT-based upper-bound geometry capacities only. They do not imply recovered TPs, because modified intervals could alter false positives, overlap, matching, and recognition.

## Pipeline-specific findings

- ME-TST+ k-based width tendency: **TOO_WIDE** (157 too-large versus 117 too-small diagnostics). Peak-edge effect: **SUPPORTED** (169/276).
- Boosting boundary asymmetry: **NOT_SUPPORTED** (mean signed normalized asymmetry B/S=-0.033, B/C=-0.003). Candidate peak closer to GT center than its formal interval center: **NO** (139/331); boundary search therefore is not the shared explanation.
- Formal-output observable geometry signal versus TP controls: **SUPPORTED** (subject-aware bootstrap, N=10,000, seed=100).

## Gates

- WIDTH_GEOMETRY_SUPPORTED: **YES**.
- CENTER_SHIFT_GEOMETRY_SUPPORTED: **NO**.
- BOOSTING_BOUNDARY_GEOMETRY_SUPPORTED: **NO**.
- GEOMETRY_FAILURE_MIXED: **NO**.

**Recommended single next experiment: `TRAINING_ONLY_DURATION_GEOMETRY_VALIDATION`.**

## Controls and scope

`tp_geometry_control.csv` contains the same geometry quantities for formal matched TPs. A-no-peak and B controls are held as categorical controls to preserve the distinction between absent reference candidates, threshold loss, and the primary geometry-limited A population.
