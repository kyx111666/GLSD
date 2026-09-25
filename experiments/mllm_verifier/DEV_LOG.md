# Development Log — Frozen MLLM Candidate Verifier

Created and last updated: 2026-09-16  
Implementation contract: the user-provided feasibility protocol; the parent
project has no `docs/implementation.md` or `docs/user_requirements.md`.

## Progress

| Module | File | Status | Note |
|---|---|---|---|
| Project audit | `PROJECT_AUDIT.md` | Done | Canonical GLSD/GT/video/evaluator sources identified |
| Replay and leakage gates | `src/protocol.py` | Done | Static and unit-tested |
| Smoke sampling | `src/build_smoke40.py` | Blocked | Only 9 mappable TP are locally available |
| Frame extraction | `src/extract_candidate_frames.py` | Ready | Awaiting valid smoke40 manifest |
| Output parsing | `src/parse_output.py` | Done | Strict schema plus one syntax-only repair |
| Frozen inference | `src/run_qwen3vl_verifier.py` | Blocked | Checkpoint/runtime absent |
| Evaluation | `src/evaluate_smoke40.py` | Ready | Awaiting real inference output |
| Locked development config | `config/locked_v1.yaml` | Not created | Allowed only after smoke40 PASS |
| Dependency list | `requirements.txt` | Done | Intentionally excludes PyTorch family |

## Log entries

### 2026-09-16 — Audit and isolated scaffold

- Selected the paper-facing GLSD-90 ME-TST+/SAMMLV `predictions.pure` output.
- Replayed 48/126/111 and F1 0.2882882882882883 exactly.
- Verified all 159 embedded GT intervals against the two SAMMLV workbooks.
- Confirmed frame coordinates are zero-based after frame skip 7 at 200 fps.
- Confirmed available 600x600 JPEGs are already centered facial crops.
- Found only 17/79 canonical videos locally, covering 9 TP and 23 FP.
- Added fail-closed sampling, frame extraction, label-leakage, frozen-inference,
  parsing, determinism, and evaluation code without changing the parent tree.
- Fast validation: Python compilation and 5/5 unit tests passed; the real
  builder stopped with `SMOKE40_INPUT_BLOCKED` at 9 TP / 23 FP as intended.
- Interface-only dry run completed and recorded the fixed model ID, prompt
  SHA256, seed, generation settings, and pending runtime gates. Final local
  review found no evaluation-label field access in the inference runner.

## Known blockers

- Restore enough canonical SAMMLV video directories to provide at least 20
  mappable TP and 20 mappable FP in a frozen development-only subject scope.
- Provide the exact Qwen3-VL-8B-Instruct checkpoint and a compatible inference
  environment. The existing PyTorch build has neither CUDA nor MPS available.

## Run instructions

The canonical commands and their outputs are maintained in `README.md`. No
full smoke command is run until both blockers are resolved. The fast unit-test
command verifies replay, leakage rejection, output parsing, and frame sampling.
