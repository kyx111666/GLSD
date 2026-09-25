# CASME3 Raw Pre-Stitch Preparation — Python Code Review

> **Status**: passed_with_warnings
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-06
> **Scripts reviewed**: `export_raw_prestitch_casme3_full.py`, `validate_selective_veto_casme3.py`

## Pass Items

1. ✅ `export_raw_prestitch_casme3_full.py:30-34` fixes CASME3 geometry to `k=50`, `batch_size=256`, `frame_skip=1`, and derives `stride` as `WINDOW_LENGTH // 2`; static source checks at lines 270–370 verify these tokens against the actual original `main.py` and `train.py`.
2. ✅ Static provenance read the original `<source-root>/ME-TST-main/train.py` and confirmed the test path contains `torch.no_grad`, `yhat,yhat1=model(x)`, batch-local `i==0`, full-window writes, and `k//2:` tail writes. Its SHA-256 is persisted by the preflight.
3. ✅ `export_raw_prestitch_casme3_full.py:88-157` requires the verified compact alignment SHA and checks 94 subjects, 462 ordered videos, sequential per-subject video indices, stitched lengths, and expected raw-window counts.
4. ✅ `export_raw_prestitch_casme3_full.py:203-259` inspects original feature-cache pickle parts one at a time, validates every present video as `(expected_windows,10,50)`, records part hashes/shapes/counts, and releases each part instead of retaining the full cache during static inspection.
5. ✅ `export_raw_prestitch_casme3_full.py:381-408` makes a successful exact one-video smoke manifest mandatory before any full run and rechecks the alignment and source hashes.
6. ✅ `export_raw_prestitch_casme3_full.py:436-510` preserves subject-level flatten order, invokes `model.eval()` and `torch.no_grad()`, captures `yhat/yhat1` immediately after forward, and then applies the original batch-local stitching branch.
7. ✅ `export_raw_prestitch_casme3_full.py:532-559` writes all required identity, batch, window, global mapping, raw score/logit, and same-run stitched arrays into each video NPZ.
8. ✅ `export_raw_prestitch_casme3_full.py:561-611` reconstructs score and raw logits using only saved raw windows plus `framecount_before_write` and `index_within_batch`, requiring exact `array_equal` semantics.
9. ✅ `export_raw_prestitch_casme3_full.py:631-769` keeps smoke/full execution separate, loads each selected subject checkpoint once, performs one complete subject traversal, saves only after exact reconstruction, and emits READY only when all 462 videos pass.
10. ✅ `validate_selective_veto_casme3.py:117-157` refuses validation before a complete 94-subject/462-video exact raw-source manifest and verifies one checkpoint load/subject traversal per subject.
11. ✅ `validate_selective_veto_casme3.py:216-269` retains compact-cache Final Strong Native events/margins, uses raw views only for agreement, requires raw/cache event-identity equality, and blocks unless the baseline is exactly 124/1148/734.
12. ✅ `validate_selective_veto_casme3.py:33-42,345-373` fixes Frozen Cross-Dataset to `(delta,tau)=(0.20,0.80)` and limits nested calibration to the original 4×4 grid with pooled outer-train-subject F1 selection.
13. ✅ Static AST inspection found no top-level Torch import and no `.train()`, `.backward()`, or optimizer `.step()` call in either script. `py_compile` passed for both scripts; the validation script was not executed.
14. ✅ Static Final Strong artifact audit found 94 selected configs and 94 outer-subject rows with exact subject mapping; summing saved rows reproduced 124/1148/734 and F1 `0.11643192488262911`.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | exporter full-run entry | full export did not require documentary evidence of a passed smoke run | added mandatory `--smoke-manifest` validation before Torch/CUDA import | fixed |
| 2 | exporter static feature inspection | the original preflight path could retain all five cache parts simultaneously | added one-part-at-a-time metadata/shape inspection with immediate release | fixed |
| 3 | validation entry | validation loaded the compact cache before rejecting an absent/incomplete raw source | added `precheck_raw_manifest()` as the first operation after argument parsing | fixed |
| 4 | local execution assets | CASME3 feature part 5, skip manifest, and all 94 CASME3 checkpoints are absent | no code workaround allowed; GPU execution remains blocked pending restoration of exact original assets | unresolved blocker |

## Constraint Direction Review

| File:line | Direction | LHS | RHS | Expected meaning |
|---|---|---|---|---|
| `export_raw_prestitch_casme3_full.py:32` | `==` | stride | `k//2` | derive overlap geometry from original source |
| `export_raw_prestitch_casme3_full.py:108-109` | `==` | subject/video indices | manifest traversal indices | preserve original 94-subject/462-video order |
| `export_raw_prestitch_casme3_full.py:499` | `==` | batch-local index | 0 | only the first item of each DataLoader batch writes a full window |
| `export_raw_prestitch_casme3_full.py:381-408` | `==` | smoke status/counts/hashes | exact PASS/source provenance | full run is forbidden without a matching successful smoke |
| `validate_selective_veto_casme3.py:195-202` | `==` | peak coverage count | 1 or 2 | single view is kept; pair view permits agreement calculation |
| `validate_selective_veto_casme3.py:289-290` | `≤`, `<` | `H_s`, `A_ratio` | selected `delta`, `tau` | prune only low-margin, low-agreement pair events |
| `validate_selective_veto_casme3.py:357` | `!=` | training subject | held subject | outer-test GT cannot influence nested selection |

The directions match the user protocol and were not changed during review.

## Remaining Risks / Blockers

- Static preflight is currently BLOCKED: original feature-cache parts 1–4 contain 349 videos and 39,622 windows; part 5 with the remaining 113 videos is missing.
- `<source-root>/ME-TST-main/weights/CASME_3_4emo` contains 0/94 required subject checkpoints.
- The original `skipped_manifest.csv` for 463→462 label alignment was not found.
- Raw-logit stitching is an instrumentation-preserving 2D analogue of the original batch branch; the original application converts logits to class IDs before its own 1D recognition stitching. The exporter saves both raw logits and their exact instrumentation reconstruction as requested, without changing the application path.
- The full exporter intentionally requires significant CPU/GPU memory for the original cached windows. Static inspection is streaming, but actual subject inference must retain the original input order and is reserved for the restored GPU environment.

## Run Instructions

See `my_method/window_consensus/README_CASME3_RUN_CN.md`. Run static preflight first, then one-video smoke, then full export with the generated smoke manifest. Do not run `validate_selective_veto_casme3.py` until the full manifest reports 462/462 exact PASS.

## Expected Outputs

- Smoke: one video NPZ plus `smoke_manifest.json`, status `CASME3-RAW-WINDOW-SMOKE-PASS`.
- Full: 462 video NPZ files plus `full_manifest.json`, status `CASME3-RAW-WINDOW-SOURCE-READY` only after 462/462 exact score/logit reconstruction.
- Later validation (not run now): frozen-transfer and same-16-grid nested Selective Veto CSV/JSON/Markdown outputs.

## Recommended Next Skill

None in this task. Restore the exact missing CASME3 execution assets and rerun static preflight in the GPU environment; do not alter the method or grid.
