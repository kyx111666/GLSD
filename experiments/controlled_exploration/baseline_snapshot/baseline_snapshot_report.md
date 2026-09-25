# GLSD-v1 baseline snapshot report

Snapshot date: 2026-09-17 (Asia/Shanghai)  
Status: **FINAL / LOCKED**  
Experiment execution during snapshot: **none**

## Locked method and evaluation

- Method: `GLSD-v1`, `S(c)=(G(c)+L(c))/2`.
- Local evidence aggregation: median across aligned multi-scale evidence.
- Scales / reference scales: `A={1, 1.5, 2}`.
- Local radii: `r={1, 2, 3}`.
- Threshold search: original locked paper grid `{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75}`; 90 pure G+L configurations.
- Evaluation: outer LOSO; inner configuration selection excludes the outer subject and the held inner-validation subject.
- Selection tie-break: higher F1, higher precision, fewer FP, lower configuration ID.
- Subject-level paired bootstrap: 10,000 resamples, seed 100.

## Code commit / path

- Git commit: **unavailable**. Neither the workspace nor `RethinkFuse_reproduction` contains resolvable Git repository metadata, so no commit hash is asserted.
- Authoritative source path: `<workspace-root>/RethinkFuse_reproduction/my_method/gl_saliency_skill/`.
- Frozen snapshot path: `<workspace-root>/controlled_exploration/baseline_snapshot/code/gl_saliency_skill/`.
- Source/config checksum manifest: `source_checksums.sha256`.
- SHA-256 of source/config checksum manifest: `83c968937be583c386fc25adf717cf7fef0059ca63a73abd7bf180d788bdafa0`.
- Historical-source and cache guards remain enforced by `benchmark.py`; the saved regression artifact reports `ALL_FOUR_EXACT_REPLAY`.

## Config path

- Canonical copied baseline config: `<workspace-root>/controlled_exploration/baseline_snapshot/config/baseline_config.json`.
- Four signed historical protocols: `<workspace-root>/controlled_exploration/baseline_snapshot/config/protocols/`.
- Canonical config SHA-256: `6bb5b71dbb94eec4ad2e800d467ba947c5663a03374422aa5598ed70acb7bfee`.

## Result path and checksum

- Original FINAL/LOCKED result path: `<workspace-root>/RethinkFuse_reproduction/results/final_gl_skill/`.
- Frozen baseline result path: `<workspace-root>/controlled_exploration/baseline_snapshot/results/`.
- Per-file checksum manifest: `<workspace-root>/controlled_exploration/baseline_snapshot/result_checksums.sha256`.
- SHA-256 of result checksum manifest: `30a535631603733fe5e5af5dc728dea4c8ba4250f4d899177e43f63b7cb562ba`.
- Copy verification: all 11 baseline result files are byte-identical to the original FINAL/LOCKED result directory; every recorded SHA-256 verifies successfully.

## Four-setting baseline F1

Values are `GL_Skill_F1` from the frozen `results/main_results.csv`.

| Backbone | Dataset | Baseline F1 |
|---|---|---:|
| ME-TST | SAMMLV | 0.2882882882882883 |
| ME-TST | CAS(ME)3 | 0.09416380578715057 |
| BoostingVRME | SAMMLV | 0.28378378378378377 |
| BoostingVRME | CAS(ME)3 | 0.11288805268109126 |

## Controlled-exploration lock

Every later exploration must be created in a new directory, must not overwrite any file under this baseline snapshot, and must emit the same 11-file result set with the same schemas as `baseline_snapshot/results/`. This snapshot contains no exploration output and no newly run experiment.
