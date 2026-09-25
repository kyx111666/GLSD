# ME-TST+ safeguarded fusion screening

This package is for the already prepared ME-TST+ Colab environment. It does not train a backbone or regenerate responses.

It evaluates, on both ME-TST+ datasets at fixed `a0=1.5`, `rho=2`: G, L, EqualMean, WeightedMean, DominantEvidence, and SafeSelect.

DominantEvidence uses `beta*max(G,L)+(1-beta)*min(G,L)` with beta in `{0.5,0.6,0.7,0.8,0.9,1.0}`. The threshold grid is the official GLSD-90 grid: `{0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.75}`.

SafeSelect chooses among the five methods above on pooled inner raw counts, with the existing F1, precision, fewer-FP, fixed-order tie break. Its pooled search contains 150 configurations. The outer subject is evaluated only with the selected configuration. This is an inner-fold fallback safeguard; it does not mathematically guarantee outer-test dominance.

Upload the safeguarded package, then replace the script path in the notebook's final run cell with `run_safeguarded_fusion.py`. Use a new output directory.
