# PROJECT AUDIT — Frozen MLLM Candidate Verifier

Audit date: 2026-09-16  
Scope: ME-TST+ / SAMMLV / locked 90-configuration GLSD only  
Status: `AUDIT_COMPLETE_WITH_INPUT_BLOCKERS`

This directory is an isolated feasibility workspace. No existing GLSD source,
configuration, result, paper file, or conclusion was modified.

## Selected canonical sources

Only the source chain used by the current paper-facing GLSD-90 result was
selected. The later expanded-grid exploration is explicitly excluded.

```text
GLSD result source:
  <workspace-root>/historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json
  prediction key: predictions.pure
  SHA256: bbf0d559db15aaa5cd1ccc69e0dac3ef49cdd6c7d5294d07b8f10b7c38858643

Final-result regression evidence:
  <workspace-root>/RethinkFuse_reproduction/results/final_gl_skill/prediction_regression.json
  status: PASS; predictions_exact=true; TP_FP_FN_exact=true

Frozen ME-TST+ response cache:
  <workspace-root>/RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl
  SHA256: 3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569

GT source used by the GLSD result:
  embedded `gt` arrays in the selected-predictions file, originating from the
  paper-aligned SAMMLV loader and frozen response cache.

Independent GT provenance check:
  <data-root>/SAMM_LongVideos_V3_Release.xlsx
  <data-root>/SAMM_Micro_FACS_Codes_v2.xlsx
  The original ME-TST+ merge/filter/downsampling rules reproduce all 159 GT
  intervals in all 79 result rows exactly (79/79 video rows, 0 mismatches).

video source:
  <sammlv-video-root>/<video_id>/*.jpg
  Current local coverage: 40 directories total, but only 17 of the 79 videos
  in the canonical GLSD result are present.

candidate coordinate format:
  zero-based, inclusive [onset, peak, offset] on the ME-TST+ downsampled
  temporal axis. The requested manifest fields map start=onset, end=offset,
  center=round((onset+offset)/2); peak is retained separately.

fps source:
  200 fps from the frozen SAMMLV cache metadata and the paper-aligned runtime.
  `frame_skip=7`; temporal coordinate t maps to the raw ordered image at
  zero-based list position 7*t (normally filename frame number 7*t+1).

evaluation implementation:
  <workspace-root>/RethinkFuse_reproduction/my_method/gl_saliency_skill/evaluation.py
  SHA256: fb9995d42f65792d1d124adea0da65b560f12e1dc49f958b1e7e15a778b8c434
  Inclusive interval IoU; threshold >=0.5; predictions processed in saved
  chronological order; each prediction considers its highest-IoU GT only;
  an already matched GT is not replaced by a second-best GT.

face crop availability:
  Available raw SAMMLV image directories already contain deterministic,
  centered 600x600 grayscale facial crops. The first frame of all 40 locally
  available video directories was checked and all 40 were 600x600. They will be reused directly.
  No new face detector or learned alignment stage is needed.

decoded frame cache:
  The available SAMMLV source is already decoded as ordered JPEG frames.
  No separate complete 79-video aligned/decoded cache was found.
```

## Gate A — Original GLSD replay

The canonical `predictions.pure` events were re-evaluated from their saved
intervals and embedded GT using the locked evaluator semantics.

| Videos | Subjects | GT | Candidates | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 79 | 29 | 159 | 174 | 48 | 126 | 111 | 0.275862 | 0.301887 | 0.2882882882882883 |

- Saved-versus-recomputed candidate match assignments: 0 mismatches.
- Candidate geometries `(subject, video, onset, peak, offset)`: 174/174 unique.
- Exact agreement with the current paper-facing ME-TST+/SAMMLV GLSD result:
  **PASS**.

Therefore `GLSD_REPLAY_GATE_FAILED` is **not** triggered.

## Gate B — Candidate preservation contract

The exploration must copy the 174 canonical candidate geometries unchanged.
Only `candidate_id` and later `keep/reject` metadata may be appended. Frame
extraction converts a temporal coordinate to a raw frame path but does not
rewrite the candidate interval.

Status: **PASS at source audit; enforced again by scripts before inference.**

## Gate C — Label leakage contract

The canonical result contains GT and `matched_gt`, so it must never be passed
to the model runner. The planned data flow uses two files:

1. `smoke40_manifest.json`: evaluation-only manifest containing TP/FP labels.
2. `inference_manifest.json`: sanitized frame-only manifest. The inference
   runner rejects forbidden keys recursively before model loading.

Status: **design-enforced; runtime gate pending a valid smoke40 manifest.**

## Gate D — Determinism contract

Five candidates must be run twice with seed 100, greedy decoding, sampling
disabled, and identical verdicts. This cannot yet be executed because the
checkpoint and inference dependencies are absent.

Status: **PENDING**.

## Raw-frame coverage blocker

Of the 79 canonical result videos, only 17 have a matching local raw-frame
directory. Those 17 videos cover 9 original GLSD TP and 23 original GLSD FP.
The required balanced smoke set needs 20 TP and 20 FP, so it cannot be built
without inventing samples, duplicating candidates, or silently changing the
GLSD result source. All three would violate the protocol.

Required data to unblock:

```text
The remaining SAMMLV frame directories needed by the canonical 79-video
ME-TST+/SAMMLV subset, placed under:
<sammlv-video-root>/<video_id>/*.jpg
```

The builder must see at least 20 mappable TP and 20 mappable FP candidates.

## Qwen3-VL environment blocker

Requested checkpoint: `Qwen/Qwen3-VL-8B-Instruct` (pretrained, frozen).

Current host:

- Apple M4 MacBook Air, 24 GB unified memory.
- Base Python 3.14 has no PyTorch, Transformers, Accelerate, bitsandbytes,
  qwen-vl-utils, or OpenCV.
- Existing `torch-mac` Conda environment uses Python 3.11 and PyTorch 2.10.0,
  but lacks Transformers, Accelerate, bitsandbytes, and qwen-vl-utils.
- No local Qwen3-VL checkpoint directory was found.
- The installed PyTorch build reports neither CUDA nor MPS availability.

No alternative vision-language model will be substituted. No model download
or dependency installation is performed by this audit.

## Audit decision

The locked GLSD result, GT, coordinate convention, evaluator, and available
face-frame format are identified and internally consistent. The experiment is
blocked before smoke40 construction by incomplete raw-frame coverage, and
blocked before real inference by the absent requested checkpoint/runtime.

Safe next action: complete the independent scripts and dry-run their gates;
do not create a nonconforming smoke40 result and do not continue to a
development-pool experiment.
