# Reproduction

## Inputs

Copy `configs/inputs.example.json` to `configs/inputs.json`. Each setting defines:

| Field | Required content |
| --- | --- |
| `upstream` | Matching ME-TST+ or BoostingVRME checkout containing `training_utils.py` and `Utils/` |
| `responses` | ME-TST+: directory of `subject_*.pkl` official dumps. BoostingVRME: official response-cache pickle |
| `evidence` | Original evidence bundle root, including `evidence_bundle_sha256.txt` |
| `run_evidence` | Directory containing original `run_manifest.json`, `per_subject_counts.csv` and per-video evaluation files |
| `locked_results` | Directory containing `outer_selected_configs.csv` |
| `preregistration` | ME-TST+ main runners: original preregistration JSON naming protocol `locked GLSD-90` |

`run_evidence` and `locked_results` default to the evidence directory when omitted. Boosting replay also uses `glsd_search_inner_fold_counts.csv`; CAS(ME)3 preflight reads the original saved selection JSON and available detailed evidence. Transfer requires saved per-video inventories and source outer-fold subject lists. Keep the response-cache schema and subject ordering consistent with the supplied experiment metadata.

SAMMLV contains 29 subjects, 79 videos and 159 events. CAS(ME)3 contains 94 subjects, 462 videos and 853 events.

Caches must contain the original recognition outputs and metadata as well as temporal responses: final scores use the upstream recognition/result-synergy stage. Raw spotting counts and final full counts must not be interchanged. The repository excludes videos, annotations, weights, response dumps and personal execution records. Dataset access and response generation are external prerequisites, not performed by this release.

Run `python run.py check --config configs/inputs.json` to list missing inputs and verify the frozen core checksum.

## Main experiment

The search evaluates G, L and equal-mean GLSD on the declared grids, selects parameters with the held-out subject excluded, and evaluates Native and the locked component ablations. Run each setting in its matching upstream environment, with a new output directory:

```bash
python run.py search --config configs/inputs.json --setting metst_sammlv --output outputs/main/metst_sammlv
python run.py search --config configs/inputs.json --setting metst_casme3 --output outputs/main/metst_casme3
python run.py search --config configs/inputs.json --setting boosting_sammlv --output outputs/main/boosting_sammlv
python run.py search --config configs/inputs.json --setting boosting_casme3 --output outputs/main/boosting_casme3
```

Use a separate output directory for each run.

Outputs include `search_counts_G.npz`, `search_counts_L.npz`, `search_counts_EqualMean.npz`, `selected_configs.csv`, per-subject raw/full counts and `summary.csv`. Raw counts measure spotting; full counts include the upstream recognition/result-synergy step.

## Locked-alpha ablation

For each ME-TST+ setting, first check that alpha=0.5 reproduces the saved Table 1 counts, then run all five weights in a separate output directory:

```bash
python run.py ablation --config configs/inputs.json --setting metst_sammlv --probe-only --output outputs/alpha_probe_sammlv
python run.py ablation --config configs/inputs.json --setting metst_sammlv --output outputs/alpha_sammlv
python run.py ablation --config configs/inputs.json --setting metst_casme3 --probe-only --output outputs/alpha_probe_casme3
python run.py ablation --config configs/inputs.json --setting metst_casme3 --output outputs/alpha_casme3
```

Weights are 0, 0.3, 0.5, 0.7 and 1. The saved per-fold scale, radius and threshold remain fixed; no additional tuning is performed. The matching adapter's alpha=0.5 alignment gate is retained and must pass.

## Cross-pipeline transfer

First audit subject/video correspondence:

```bash
python run.py transfer-audit --config configs/inputs.json --output outputs/transfer
```

Run four replays, each in its matching upstream environment:

```bash
python run.py replay --config configs/inputs.json --setting metst_sammlv --output outputs/transfer/preflight/metst_sammlv
python run.py replay --config configs/inputs.json --setting metst_casme3 --output outputs/transfer/preflight/metst_casme3
python run.py replay --config configs/inputs.json --setting boosting_sammlv --output outputs/transfer/preflight/boosting_sammlv
python run.py replay --config configs/inputs.json --setting boosting_casme3 --output outputs/transfer/preflight/boosting_casme3
python run.py transfer-gate --config configs/inputs.json --output outputs/transfer
```

The gate requires four exact Native and four exact tuned-GLSD replays. Then execute each direction in its **target** pipeline environment:

```bash
python run.py transfer --config configs/inputs.json --direction metst_to_boosting_sammlv --output outputs/transfer
python run.py transfer --config configs/inputs.json --direction boosting_to_metst_sammlv --output outputs/transfer
python run.py transfer --config configs/inputs.json --direction metst_to_boosting_casme3 --output outputs/transfer
python run.py transfer --config configs/inputs.json --direction boosting_to_metst_casme3 --output outputs/transfer
python run.py transfer-summary --config configs/inputs.json --output outputs/transfer
```

Only `(a0, rho, tau)` transfers. The target retains its own temporal scale, responses, event boundaries, ordering, conflict handling and recognition interface. Bootstrap uses 10,000 paired subject resamples with seed 100.

## Near-optimal GLSD selection

The selection command reads `search_counts_EqualMean.npz`, `search_counts_G.npz` and `search_counts_L.npz`. Each file contains `raw_counts` with shape `(configurations, subjects, 3)` in TP, FP, FN order, and a `subjects` array. EqualMean and L use 90 configurations; G uses 30. Configuration order is scale, radius, then threshold, with G independent of radius.

```bash
python src/select_glsd.py --counts outputs/search --output outputs/selected_configs.csv
```

The command excludes each held-out subject, pools counts from the remaining subjects, and selects a configuration using epsilon=0.002. Its CSV contains the held-out subject, selected configuration and inner-fold score. It selects from existing search counts; it does not generate responses or evaluate held-out predictions. See [selection protocol](PROTOCOL.md).

## Threshold sensitivity

Pass the selected configurations from the corresponding search. Use the same command for each of the four settings:

```bash
python run.py threshold --config configs/inputs.json --setting metst_sammlv \
  --selections outputs/main/metst_sammlv/selected_configs.csv \
  --output outputs/threshold/metst_sammlv
```

The output contains Native scores, per-subject GLSD counts at offsets -0.05, 0 and +0.05, and `threshold_sensitivity.csv`. When `--selections` is omitted, the command reads `outer_selected_configs.csv` from the configured `locked_results` directory.

## Runtime

```bash
python run.py runtime --config configs/inputs.json --setting metst_sammlv \
  --selections outputs/main/metst_sammlv/selected_configs.csv \
  --output outputs/runtime/metst_sammlv
```

Use the corresponding setting and selection file for the other three experiments. `runtime.csv` reports mean, median and standard deviation in milliseconds per video for Native, GLSD and the paired difference. `runtime_repeats.csv` contains the individual measurements. Timing covers response-to-event decoding with metric accumulation disabled; loading, selection, recognition and backbone inference are excluded.
