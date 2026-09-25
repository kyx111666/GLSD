# Canonical Bilateral Local Contrast Audit

## Source of truth

The audited implementation is `historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/unified_persistence.py`, class `CurveFeatures`. The EXP-5B replay also checks the reconstructed values against that class at absolute tolerance `1e-15` for every evaluated video/configuration.

## Exact semantics for a matched candidate-scale peak

Let the aligned peak index be `p`, the integer local radius be `w=max(1, round(rho*k))`, the smoothed curve at that physical scale be `y`, and `R=max(ptp(y), 1e-12)`.

1. Peak/center value: `C = y[p]`.
2. Left neighborhood: `y[max(0,p-w):p+1]`; it includes the center sample.
3. Right neighborhood: `y[p:min(len(y),p+w+1)]`; it also includes the center sample.
4. Left baseline: `B_L = min(left neighborhood)`.
5. Right baseline: `B_R = min(right neighborhood)`.
6. Canonical bilateral baseline: `B_can = max(B_L,B_R)`.
7. Numerator: `max(0, C-B_can)`.
8. Denominator/range normalization: `R=max(ptp(y),1e-12)`.
9. Clipping: no upper clipping.
10. Positivity constraint: yes, `max(...,0)` around the numerator.
11. Epsilon: `1e-12` only as the minimum range denominator.
12. Range normalization: yes, by the full smoothed-curve range at that scale.
13. Per-scale local evidence:

`l_a(c) = max(0, y_a[p_a] - max(B_L,a, B_R,a)) / max(ptp(y_a), 1e-12)`.

Cross-scale alignment uses tolerance `max(1,round(0.5*k))`; absent non-reference matches contribute exactly zero; the candidate local evidence is the NumPy median across distinct physical widths. The reference candidate self-matches. Final canonical score is `S=(G+L)/2`.

## Applicability

**PASS.** The higher (stricter) side minimum controls the canonical bilateral baseline through `max(B_L,B_R)`, and this operator is directly replaceable without changing the surrounding numerator, positivity constraint, denominator, range normalization, radius, alignment algorithm, missing-zero policy, or median aggregation.
