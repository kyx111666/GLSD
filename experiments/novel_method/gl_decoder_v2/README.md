# GL-Decoder-v2

**THIS IS A RECONSTRUCTED REPRODUCIBLE IMPLEMENTATION.**

**NOT A CLAIM OF BYTE-IDENTICAL HISTORICAL ENGINE REPRODUCTION.**

This directory contains one deterministic, training-free decoder shared by
ME-TST and BoostingVRME. The backbone-specific modules only adapt cache schemas.
All experiment outputs are written to `RethinkFuse_reproduction/results/gl_decoder_v2`.

Run from the repository root:

```bash
python3 -m my_method.gl_decoder_v2.run_gl_decoder_v2
python3 -m unittest my_method.gl_decoder_v2.tests.test_gl_decoder_v2
```

The runner first writes the frozen reconstruction specification, verifies the
native ME-TST hash/count gate, and only then evaluates v2. Phase B is executed
only when the predeclared Phase-A grade is A, B, or C.
