# Strict cross-backbone GL Skill transfer

This experiment transfers only the source-backbone outer-LOSO selected historical pure G+L configuration. The target backbone supplies its frozen cache, adapter geometry, and fold-protocol `k_p`; no target performance metric participates in transferred configuration selection.

Run from the project root:

```text
.venv/bin/python -m my_method.strict_cross_backbone_transfer.run
```

Inputs are the signed historical stats/cache artifacts already verified by FINAL-GL-SKILL. Outputs are written to `results/final_gl_skill_transfer/`. The script snapshots every Python file in `my_method/gl_saliency_skill/` before and after execution and fails if any hash changes.
