# Strict Cross-Backbone Transfer Python Code Review

> **Status**: passed
> **Reviewer**: python-code-reviewer
> **Date**: 2026-09-06
> **Scripts reviewed**: `my_method/strict_cross_backbone_transfer/run.py`, `my_method/strict_cross_backbone_transfer/tests/test_protocol.py`

## Pass Items

1. ✅ `run.py:51-64` obtains selections through the signed context and existing historical `select_all`; no replacement grid or selection rule is implemented in the transfer runner.
2. ✅ `run.py:158-170` obtains the transferred configuration exclusively from `source["selected"]`; `target["selected"]` is confined to the separately named Target-Tuned comparator and agreement analysis.
3. ✅ `run.py:164-179` supplies the target fold-protocol `k_p` and target adapter to the unchanged `GLSaliencySkill`; target ground truth is passed only to `evaluate` after decoding.
4. ✅ `run.py:234-273` records one source selection for every held-out subject, explicitly constructs the source training-subject set without that subject, and records config ID, scale, radius, threshold, and source inner F1.
5. ✅ `run.py:76-101` implements subject-paired bootstrap with shared resample indices, exactly 10,000 draws, fixed seed 100, bootstrap mean, percentile 95% CI, and a zero-crossing flag.
6. ✅ `run.py:117-122` freezes the interpretation tolerance at `1e-12` before inspecting results and assigns positive/neutral/negative labels solely from transferred-minus-native F1.
7. ✅ `run.py:362-396` locks the historical 90-config pool, validates Native and Target-Tuned F1 against FINAL-GL-SKILL at `1e-15`, and fails if any Skill Python-file SHA changes during execution.
8. ✅ `run.py:406-450` automatically verifies held-subject exclusion, source/target subject alignment, absence of GT in the decoder API, absence of G/L or selection logic in adapters, unchanged Skill manifest, and zero trainable parameters.
9. ✅ Independent recount from `per_video_transferred_predictions.json` exactly reproduces transferred TP/FP/FN for all four rows in `transfer_main_results.csv`: 47/119/112, 125/1045/733, 39/115/120, and 101/1123/757.
10. ✅ Mechanism accounting is internally closed in all four groups: retained+lost equals Native TP, retained+rescued equals Transferred TP, and Native FP−removed+new equals Transferred FP.
11. ✅ Four protocol unit tests and Python `compileall` pass; the full runner completed with `STRICT_CROSS_BACKBONE_SKILL_TRANSFER_COMPLETE`.

## Failed / Repaired Items

| # | File:line | Issue | Action taken | Status |
|---|---|---|---|---|
| 1 | `run.py:171-231` | Per-video counts were unnecessarily evaluated twice | Reused the first evaluation result; no numerical or algorithmic change | fixed |
| 2 | `run.py:7` | Unused `asdict` import | Removed | fixed |

## Constraint Direction Review

There is no optimization model or physical inequality constraint in this experiment. The relevant predeclared decision/gate comparisons are:

| File:line | Direction | LHS | RHS | Meaning |
|---|---|---|---|---|
| `run.py:118` | `>` | transfer ΔF1 | `1e-12` | label positive transfer |
| `run.py:120` | `<` | transfer ΔF1 | `-1e-12` | label negative transfer |
| `run.py:98` | `<=` | CI low, 0 | `0, CI high` | report whether paired CI contains zero |
| `run.py:385-391` | `<=` | comparator F1 difference | `1e-15` | require agreement with FINAL-GL-SKILL |

## Remaining Risks

- Target `k_p` is the target fold-protocol value, not part of the transferred grid configuration. This is declared in the report and follows the requested allowance for target adapter-provided `k_p`.
- Signed historical inputs include trusted local pickle caches; as before, these loaders should not be used on untrusted pickle files.
- Bootstrap intervals quantify subject-paired uncertainty for each fixed direction; no multiple-comparison correction is applied or claimed.

## Run Instructions

```text
.venv/bin/python -m unittest my_method.strict_cross_backbone_transfer.tests.test_protocol
.venv/bin/python -m my_method.strict_cross_backbone_transfer.run
```

## Expected Outputs

- `results/final_gl_skill_transfer/combined_report.json`
- `results/final_gl_skill_transfer/transfer_main_results.csv`
- `results/final_gl_skill_transfer/transfer_config_agreement.csv`
- `results/final_gl_skill_transfer/transfer_bootstrap.csv`
- `results/final_gl_skill_transfer/transfer_mechanism.csv`
- `results/final_gl_skill_transfer/transfer_leakage_audit.json`
- `results/final_gl_skill_transfer/per_subject_source_configs.csv`
- `results/final_gl_skill_transfer/per_video_transferred_predictions.json`
- `results/final_gl_skill_transfer/transfer_threshold_robustness.csv`

## Recommended Next Skill

None. The user explicitly requested stopping after this experiment.
