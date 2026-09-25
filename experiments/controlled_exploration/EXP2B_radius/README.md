# EXP-2B: GLSD Local-Radius Expansion

## Question

Is the locked GLSD local structural context `radius={1,2,3}` too narrow?

## Controlled change

- Baseline radius set: `R={1,2,3}`.
- Expanded radius set: `R_new={1,2,3,4,5}`.
- The scale set and searched reference scales remain `A={1,1.5,2}`.
- Local evidence aggregation remains `median`.
- Global evidence `G`, equal-weight fusion `(G+L)/2`, thresholds, interval
  adapters, fold priors, evaluator, and selection tie-breaks are unchanged.
- This is a configuration-space expansion, not a new model.

## Configuration budgets

- Baseline: `3 scales x 3 radii x 10 thresholds = 90` configurations.
- Expanded: `3 scales x 5 radii x 10 thresholds = 150` configurations.

## Evaluation

Run the same decoder-level nested LOSO protocol on the four locked
backbone/dataset settings. Compare pooled outer-fold Raw Spotting F1 against the
archived 90-configuration baseline. Report a paired subject bootstrap as
supporting uncertainty evidence, conditional on frozen response curves and the
selected configurations.

## Required outputs

- `results.csv`: baseline and expanded pooled counts/F1 and F1 delta by setting.
- `selected_radius_distribution.csv`: baseline-versus-expanded outer-fold
  selected-radius counts and fractions, including zero-count radii.
- `outer_fold_selections.csv`: per-fold selected configuration and outer counts.
- `bootstrap.csv`: paired subject-bootstrap interval for each F1 delta.
- `analysis.md`: interpretation, with explicit attention to CAS(ME)3.

