# Final pre-GPU audit

Audit date: 2026-09-16  
Scope: frozen ME-TST + SAMMLV + GLSD smoke40 deployment input only

## Outcome

```text
SMOKE40_BUILD_PASS
LABEL_LEAKAGE_GATE_PASS
SMOKE40_INPUT_GATE_PASS
SMOKE40_LOCKFILE_CREATED
MOCK_DRY_RUN_PASS
SMOKE40_GPU_DEPLOYMENT_READY
MLLM_INFERENCE_NOT_RUN
```

## Input accounting

| Check | Result |
|---|---:|
| Candidates | 40 / 40 |
| Private evaluation TP | 20 |
| Private evaluation FP | 20 |
| Remote TP | 11 |
| Local TP | 9 |
| Local FP | 20 |
| Candidate images | 360 / 360 |
| Contact sheets | 40 / 40 |
| Missing frames | 0 |
| Corrupt images | 0 |
| Unresolved mappings | 0 |
| Hash mismatches | 0 |

All 400 files below `data/smoke40/` are listed in sorted order in `data/audit/SMOKE40_LOCKFILE.json`. All 360 model input frames from both sources are 600 × 600 grayscale JPEG images. Both sources use the ordered `B1 B2 B3 C1 C2 C3 A1 A2 A3` schema, zero-based logical raw indices, one-based physical source filenames, and strictly increasing temporal order.

## Remote import gate

- Configurable input: `REMOTE_TP_ZIP` or `--remote-tp-zip`.
- Imported archive SHA256: `9ab20a4c2c98781d59c45cdee3c9c250376bad402075be9f307225270d072e65`.
- Sidecar SHA256 matched.
- `REMOTE_FRAME_EXTRACTION_PASS` was present.
- Candidate identities: 11 / 11.
- Remote frame mappings: 99 / 99.
- Remote image SHA256 checks: 99 / 99.

## Locked inputs

| Input | SHA256 |
|---|---|
| `smoke40_inference_manifest.json` | `6e57702436afe5103ff68c5630082023f18d7a514a799b7cdf11633d78f44a5f` |
| `smoke40_labels_PRIVATE.json` | `85745f8080c45f669acfe9a04b1652cbcc345e0c4738de9e8188da94bb49dea6` |
| `verifier_prompt_v1.txt` | `39dc599039e1dce3c91b9bee832ced34fd4214fc1d38cb8af015d48b3b04c175` |
| `smoke40_protocol_v1.yaml` | `ac36b43e96007c50af920553538fd7cccb9fb2691e16c04ebfc2ddcbe745c22f` |

## Separation and negative tests

- The inference manifest contains only candidate identity, subject/video identity, frame files, and B/C/A roles.
- Candidate directories and contact sheets use neutral sequential names.
- The leakage gate found zero forbidden terms in model-facing metadata, paths, and prompt text.
- Passing the private labels file to the model runner was rejected with `LABEL_LEAKAGE_GATE_FAILED`.
- The mock output is marked `MOCK_OUTPUT: true` and `NOT_FOR_EVALUATION`.
- The evaluator rejected that output with `MOCK_RESULT_NOT_EVALUABLE`.

## Dry-run coverage

The CPU mock run parsed the 40-row inference manifest, decoded all 360 images, loaded the locked prompt, parsed the strict five-field JSON schema, and serialized 40 results. It did not import PyTorch, Transformers, bitsandbytes, or qwen-vl-utils, and did not load or run a model.

Future supported explicit weight-loading modes are `bf16`, `fp16`, `int8`, and `int4`. Unsupported quantized environments fail explicitly; there is no silent precision fallback. A checkpoint other than Qwen3-VL-8B-Instruct is a separate model-variant experiment.

