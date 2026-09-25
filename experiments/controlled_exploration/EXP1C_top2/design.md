# GLSD Controlled Exploration EXP-1C — Design Lock

## Research question

Test whether median aggregation is overly conservative, max aggregation is overly aggressive,
and Top-2 aggregation provides a more stable trade-off between them.

## Sole algorithmic change

For aligned local evidence sorted as `l_(1) >= l_(2) >= l_(3)`, define

`L_top2 = (l_(1) + l_(2)) / 2`.

The baseline remains the locked median-based GLSD. EXP-1C is written to a separate directory
and does not replace the canonical implementation or baseline results.

## Frozen controls

- Fusion remains `S = (G + L) / 2`.
- Four settings, candidate generation, scale set, local-radius grid, threshold grid,
  interval protocol, evaluator, nested LOSO selection, tie-break, and 90-configuration
  budget remain unchanged.
- Evaluation reports F1 against the locked median baseline, TP/FP/FN changes,
  paired subject-bootstrap uncertainty, selected-configuration changes, and a mechanism summary.

