# Persistence + Morphology Fusion

This directory contains a standalone, read-only post-processing experiment for
the strict ME-TST score/logit caches. It does not modify the original project
or the reference ME-TST evaluation code included under
`experiments/third_party/metst_plus/`.

## What Is Tested

The script compares four candidate-scoring ablations against the native
ME-TST post-processing baseline:

1. `height_only`
2. `morphology_only`
3. `height_persistence`
4. `height_morphology`
5. `height_persistence_morphology`

All variants use the same frozen per-frame spotting scores. Candidates are
local maxima on a smoothed score curve and retain the native fixed interval
`[peak-k_p, peak+k_p]`. The evidence terms are:

- `Height`: robust per-video z-score of the smoothed peak value.
- `Persistence`: robust per-video z-score of one-dimensional peak prominence.
- `Morphology`: an unsupervised local-shape score formed from half-prominence
  width compatibility, left/right symmetry, local area, and local stability.

The candidate pool can lower the original threshold from `0.55` to `0.45` or
`0.35`, so the test can expose possible recovery of originally sub-threshold
events. It does not train or alter the ME-TST backbone.

## Validation Protocol

The experiment reads only the strict paper-aligned caches:

- `SAMMLV`: 29 subjects, 79 videos, 159 GT events
- `CASME_3`: 94 subjects, 462 videos, 858 GT events

For every outer held-out subject, the fusion family, smoothing multiplier,
candidate threshold, morphology radius, evidence weights, and final candidate
score threshold are selected using only the other subjects. Because the fusion
has no learned subject-specific parameters, inner LOSO consists of applying
each deterministic configuration to every outer-train subject and selecting
the aggregate paper-style Spotting F1. The held-out subject is evaluated once
with that selected configuration.

Matching uses the repository's paper-style chronological greedy matching at
IoU `0.5`. Selection uses raw Spotting F1; reports include both raw and
`result_synergy` spotting counts. The script also validates its native
implementation against the strict-cache reference TP/FP/FN counts before
running the ablations.

## Run

Use the bundled Python runtime available in this workspace:

```powershell
$py = 'python3'
& $py D:\workspace\a\metst\persistence_morphology_fusion.py
```

Results are written to `results/<dataset>/report.json`,
`results/<dataset>/outer_loso_selections.csv`, and `results/combined_report.json`.

## Scope Notes

- This is an initial, training-free fusion test, not a claim that the full
  Persistence or Morphology research directions have been exhausted.
- The reported `subthreshold_proxy` is explicitly an operational proxy based
  on the frozen score curve. It is not a replacement for the document's more
  detailed failure-mode audit.
