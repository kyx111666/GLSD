# Official-response component ablation — Colab patch

Use [official_response_component_ablation.py](official_response_component_ablation.py) only in the final Colab environments represented by the supplied notebooks. It performs its Drive checks after mount, refuses missing or mismatched sealed inputs, and writes a timestamped new directory below:

```text
/content/drive/MyDrive/official_response_ablation/
```

It reads only the four official dump/response locations named in the notebooks. It does not reference a historical controlled cache, run a backbone, regenerate a response, or write into any locked evidence directory.

Start with the supplied notebook's **preflight-only** command. It directly replays Native and each sealed GLSD-90 outer-fold selection using the locked evaluator, verifies the expected complete-count totals and cardinalities, and writes `locked_replay_summary.csv`, per-subject replay counts, and `preflight_stage_audit.md`. It starts neither ablation search nor bootstrap. A nonzero exit code is a failed gate.

Only after that Colab preflight passes should the command be changed deliberately to omit `--preflight-only`. The later ablation computes H/G/L/GL with 30 configurations per score mode (`a0={1.0,1.5,2.0}` × the locked ten thresholds; `rho=2` fixed for L/GL). It reuses the sealed program's candidate source, event geometry, post-processing, official recognition/result synergy, evaluator, and selection order. Configuration selection pools only non-outer subjects. `GL` is explicitly named **matched-budget-GL-30** in future evidence and is not the locked **GLSD-90** method.

For every setting it writes the requested ablation files, full 30-configuration selection evidence (`ablation_search_all.csv` and `ablation_inner_fold_counts.csv`), and paired subject bootstrap outputs. Bootstrap compares the sealed Native and sealed locked-GLSD per-subject **full** counts, because the cited official final F1 values include official result synergy. It uses 10,000 identical subject resamples for both methods, seed 100, and recomputes F1 after summing TP/FP/FN.

Run it in Colab with stdout/stderr captured as part of the new evidence:

```bash
python /content/official_response_component_ablation.py \
  --setting all \
  --preflight-only \
  --output-root /content/drive/MyDrive/official_response_ablation \
  2>&1 | tee /content/drive/MyDrive/official_response_ablation/preflight_launcher.log
```

The provided notebook patch creates a run-specific preflight log under the new `official_response_ablation/` tree.

## BoostingVRME / CAS(ME)3 dedicated preflight

Do not route BoostingVRME / CAS(ME)3 through the generic Boosting/SAMMLV call
shape.  Use the fail-closed dedicated phase-one program instead:

```bash
python /content/boosting_casme3_official_response_preflight.py \
  --preflight-only \
  --runner /content/BoostingVRME/boosting_official_glds_full_casme3.py \
  --official-repo-root /content/BoostingVRME \
  --cache /content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/official_response_cache/casme3_official_full_responses.pkl \
  --evidence-dir /content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1 \
  --output /content/drive/MyDrive/official_response_ablation/boosting_casme3_preflight_v1
```

The program first requires the current cache SHA-256 to match the historical
manifest exactly.  A current/historical runner SHA mismatch is recorded but
does not stop execution: the program then replays the full 90 x 94 raw table,
reconstructs all 94 outer selections and 93-subject inner folds, replays the
selected configurations through official recognition/result synergy, and
compares all available sealed search, selection, per-subject, trace, and final
prediction evidence.  It requires exactly 94 subjects, 462 videos, 853 GT
events, and rejects the older 858-event controlled cache.

`outer_selected_configs.json` is a required CAS(ME)3 selection artifact.  The
program will not substitute a SAMMLV selection file or infer selections while
silently skipping that comparison.  `per_video_evaluation.csv`,
`per_prediction_evaluation.csv`, `outer_selected_configs.csv`, and
`glsd_search_all_90x94.csv` are compared exactly when present.

The historical `run_manifest.json` is read-only.  A new
`recovery_manifest.json` is written below the requested output directory.  The
program does not expose component-ablation or bootstrap execution in phase one.
For the expected uploaded-runner mismatch case, continue only when the final
three lines are exactly:

```text
BOOSTING_CASME3_FUNCTIONAL_PREFLIGHT = PASS
CACHE_PROVENANCE = EXACT_MATCH
RUNNER_PROVENANCE = SHA_MISMATCH_FUNCTIONALLY_REPLAYED
```

## BoostingVRME official-response Phase 2

Keep these files together in `/content/official_response_ablation/`:

- `phase2_common.py`
- `boosting_sammlv_official_phase2.py`
- `boosting_casme3_official_phase2.py`
- `boosting_casme3_official_response_preflight.py`

Run the dependency-free protocol smoke tests before a formal run:

```bash
python /content/official_response_ablation/boosting_sammlv_official_phase2.py --smoke-test
python /content/official_response_ablation/boosting_casme3_official_phase2.py --smoke-test
```

Each formal output directory must not already exist.  The SAMMLV command first
replays Native and the complete sealed GLSD-90 nested selection.  The CAS(ME)3
command first invokes the dedicated functional preflight and proceeds only
after its recovery manifest records PASS.

```bash
mkdir -p /content/drive/MyDrive/official_response_phase2

python /content/official_response_ablation/boosting_sammlv_official_phase2.py \
  --runner /content/BoostingVRME/boosting_official_glds_full.py \
  --official-repo-root /content/BoostingVRME \
  --cache /content/drive/MyDrive/GLSD_BoostingVRME_Full/sammlv_official_full_responses.pkl \
  --evidence-dir /content/drive/MyDrive/GLSD_BoostingVRME_Full/results_evidence_v2 \
  --output /content/drive/MyDrive/official_response_phase2/boosting_sammlv_v1

python /content/official_response_ablation/boosting_casme3_official_phase2.py \
  --runner /content/BoostingVRME/boosting_official_glds_full_casme3.py \
  --official-repo-root /content/BoostingVRME \
  --cache /content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/official_response_cache/casme3_official_full_responses.pkl \
  --evidence-dir /content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1 \
  --preflight-script /content/official_response_ablation/boosting_casme3_official_response_preflight.py \
  --output /content/drive/MyDrive/official_response_phase2/boosting_casme3_v1
```

`component_pooled_full_summary.csv` is the paper-facing component result.
`component_pooled_raw_summary.csv` is audit evidence only.  `G+L` is labeled
`matched-budget-G+L-30`; the locked `GLSD-90` name is reserved for preflight
and the Native-vs-GLSD paired bootstrap.
