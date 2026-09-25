# Experiment index

This index is a map of the source tree. It is intentionally brief; each
experiment directory keeps its own protocol or README when one exists.

## Main decoder studies

- `experiments/glsd_phase0/`: candidate generation, boundary refinement, dense
  segment decoding, sparse event reconstruction, GESO, ENPD, and persistence
  controls.
- `experiments/controlled_exploration/EXP1A_mean` through `EXP1C_top2`:
  alternative cross-scale aggregation rules.
- `experiments/controlled_exploration/EXP2A_scale` through `EXP2D_five_scale_evidence`:
  scale and evidence-neighborhood controls.
- `experiments/controlled_exploration/EXP3A_rank_calibration` and
  `EXP3B_stability_selection`: calibration and selection objectives.
- `experiments/controlled_exploration/EXP4A_alignment_tolerance` and the
  `EXP4B*` directories: alignment and peak-reuse diagnostics.
- `experiments/controlled_exploration/EXP5*`: support and bilateral-baseline
  controls.
- `experiments/controlled_exploration/EXP6*`: bottleneck, reference evidence,
  normalization, and stability diagnostics.
- `experiments/controlled_exploration/EXP7*`: duration and event-geometry
  replication checks.

## Official-response analyses

`experiments/official_response_ablation/` contains the locked component
ablation entry points and the later fusion-screening branches. The source is
kept in separate subdirectories because these branches use different frozen
inputs and selection budgets.

## Independent method work

`experiments/novel_method/` contains decoder candidates and feasibility audits.
These are research branches, not all of them are part of the final GLSD
protocol. They are included to preserve the complete experimental record while
keeping the final method code easy to find.

## Baselines and optional verifier

- `experiments/baselines/boostingvrme/`: official BoostingVRME source needed by
  the corresponding replay and training scripts.
- `experiments/baselines/external/`: SOFTNet-SpotME and MEAN Spot-then-recognize
  source snapshots.
- `experiments/mllm_verifier/`: optional frame-level verification code. It is
  independent of the main GLSD decoder and requires a separate transformer
  environment.
