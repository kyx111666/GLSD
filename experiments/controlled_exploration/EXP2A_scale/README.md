# EXP-2A: GLSD Scale-Space Expansion

## Question

Is the locked GLSD scale set `A={1,1.5,2}` too coarse?

## Controlled change

- Baseline scale set: `A={1,1.5,2}`.
- Expanded scale set: `A_new={0.75,1.0,1.25,1.5,1.75,2.0,2.5}`.
- The expanded values are used both to build the multiscale evidence and as the
  reference-scale choices in inner-LOSO configuration selection.
- Local evidence aggregation remains `median`.
- Global evidence `G`, equal-weight fusion `(G+L)/2`, local radii, thresholds,
  interval adapters, fold priors, evaluator, and selection tie-breaks are unchanged.
- This is a configuration-space expansion, not a new model.

## Configuration budgets

- Baseline: `3 scales × 3 radii × 10 thresholds = 90` configurations.
- Expanded: `7 scales × 3 radii × 10 thresholds = 210` configurations.

## Evaluation

Run the same decoder-level nested LOSO protocol on the four locked
backbone/dataset settings. Compare pooled outer-fold Raw Spotting F1 against the
archived 90-configuration baseline. A paired subject bootstrap is reported as a
supporting uncertainty analysis, conditional on the selected configurations and
frozen response curves.

## Required outputs

- `results.csv`: baseline and expanded pooled counts/F1 by setting.
- `selected_scale_distribution.csv`: baseline-versus-expanded outer-fold selected
  reference-scale counts and fractions, including zero-count scales.
