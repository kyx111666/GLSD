# FINAL-GL-SKILL

This package is a backbone-agnostic, training-free refactor of the exactly
reproduced historical Global-Local Saliency Decoder. The sole algorithm truth
is the signed historical implementation in `a(1).zip`; GL-v2 is not used.

The public API is:

```python
skill = GLSaliencySkill(config)
events = skill.decode(
    temporal_scores=score,
    temporal_scale=k_p,
    metadata={"interval_adapter": adapter},
)
```

`evidence.py` preserves historical G, L, normalization, alignment, smoothing,
grid order, and pure 90-config family. `decoder.py` applies the historical
candidate threshold and suppression. Adapters only load frozen cache schemas
and provide fixed or native interval geometry; neither adapter computes G/L.

Run from `RethinkFuse_reproduction` with the existing scientific environment:

```bash
.venv/bin/python -m unittest my_method.gl_saliency_skill.tests.test_skill
.venv/bin/python -m my_method.gl_saliency_skill.benchmark
```

The benchmark first enforces four exact regression gates. Any mismatch exits
with `SKILL_REFACTOR_REGRESSION_MISMATCH` before main, ablation, bootstrap,
robustness, mechanism, or complexity outputs are produced.

Dependencies: Python 3.10, NumPy, SciPy. Trainable parameters: zero.
