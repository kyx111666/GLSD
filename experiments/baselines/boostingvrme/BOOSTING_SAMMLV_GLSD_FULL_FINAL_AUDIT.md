# BoostingVRME / SAMMLV GLSD Full-Pipeline Final Audit

> **Verdict: FAIL — strict validation evidence is insufficient.**
>
> This is not a finding that the reported GLSD counts are numerically wrong. The
> available artifacts consistently report the requested Native and GLSD counts.
> It is a finding that the claimed `BOOSTING_SAMMLV_GLSD_FULL_VALIDATED = PASS`
> cannot be independently certified without re-running the experiment, which
> was expressly prohibited for this audit.

## Scope and evidence policy

- No backbone, Native replay, GLSD decoder, recognition, IoU evaluator, or new
  experiment was run for this audit.
- No source or result artifact was modified.
- The audit used the copied Colab result artifacts in
  [`results `](results%20/) and the static implementation in
  [`boosting_official_glds_full.py`](boosting_official_glds_full.py).
- The final-prediction pickle was opened read-only for schema/count checks only;
  it was not passed through an evaluator.

## Passed checks

1. **Cache identity matches.** The local cache
   [`cache/sammlv_official_full_responses.pkl`](cache/sammlv_official_full_responses.pkl)
   has SHA-256 `36417abdcec34e63074f1c200907694da2ebaad843fc4126e70e26fceccf5ce2`,
   exactly matching the copied Colab audit at
   [`results /BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:6`](results%20/BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:6).

2. **Native exact-replay result is internally consistent.** The Colab audit
   records Raw `51/144/108`, Full `51/141/108`, and all three exact-match
   booleans as `True` at
   [`results /BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md`](results%20/BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:9), lines 9–14.
   The executable hard-gates those exact counts and booleans before `run_full()`
   at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:526), lines 526–560.

3. **Dataset cardinalities agree across code and artifacts.** The result audit
   states 79 videos and 29 subjects at
   [`results /BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md`](results%20/BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:7), lines 7–8.
   Read-only inspection found 79 contiguous `video_index_global` prediction
   records spanning 29 subjects in
   [`results /glsd_final_predictions.pkl`](results%20/glsd_final_predictions.pkl).
   The program asserts these cardinalities at
   [`boosting_official_glds_full.py`](boosting_official_glds_full.py:117), lines 117, 151, and 524–525.

4. **No outer-subject membership leakage is present in the persisted choices.**
   All 29 rows in
   [`results /outer_selected_configs.json`](results%20/outer_selected_configs.json)
   contain exactly 28 inner subjects, and read-only comparison found no row
   whose `outer_subject` appears in `inner_subjects`. The same outer/config
   order is present in
   [`results /outer_selected_configs.csv`](results%20/outer_selected_configs.csv).
   The runtime selection code constructs the complement of the outer index and
   asserts the exclusion at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:427), lines 427–433.

5. **The selected configuration is a valid member of the locked grid.** All 29
   persisted selections use `config_id=37`, which maps in the declared fixed
   order to `(reference_scale=1.5, local_radius=1.0, threshold=0.6)`; this
   agrees with every row of
   [`results /outer_selected_configs.csv`](results%20/outer_selected_configs.csv).
   The locked scale/radius/threshold constants and 90-member construction are at
   [`boosting_official_glds_full.py`](boosting_official_glds_full.py:33), lines 33–64,
   with the hard `assert len(grid) == 90` at
   [`boosting_official_glds_full.py`](boosting_official_glds_full.py:411), lines 411–412.

6. **The static GLSD implementation changes candidate sourcing only.**
   `GLSDFeatures.selected_peaks()` computes only reference candidates and
   `S=(G+L)/2` at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:213), lines 213–272.
   The temporary substitution is limited to `training_utils.find_peaks` and is
   restored in `finally` at
   [`boosting_official_glds_full.py`](boosting_official_glds_full.py:275), lines 275–304.
   The resulting peaks are then supplied to the unmodified official
   `spotting()` at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:317), lines 317–322.

7. **Official geometry, NMS, recognition, IoU path, and synergy are statically
   preserved.** Official `spotting()` contains the SAMMLV boundary search,
   `peak_new`, official `nms()`, and metric registration at
   [`training_utils.py`](training_utils.py:32), lines 32–120; its NMS rule is at
   [`training_utils.py`](training_utils.py:21), lines 21–29. The GLSD path calls that
   function directly at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:317), lines 317–322,
   invokes official `recognition()` on the newly decoded intervals at
   [`boosting_official_glds_full.py`](boosting_official_glds_full.py:325), lines 325–330,
   and applies the locked neutral count conversion at
   [`boosting_official_glds_full.py`](boosting_official_glds_full.py:169), lines 169–175.
   Official recognition's interval slice and aggregation remain in
   [`training_utils.py`](training_utils.py:172), lines 172–220.

8. **Reported final values agree across the copied result artifacts.** The
   Colab audit reports GLSD Raw `44/84/115` and Full `44/82/115` at
   [`results /BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md`](results%20/BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:61), lines 61–64,
   followed by the claimed PASS at line 66. The 79 prediction records reference
   only valid selected config ids `[0, 89]` and contiguous global video ids.

## Blocking evidence gaps

| # | Severity | Finding | Evidence location | Consequence |
|---|---|---|---|---|
| 1 | BLOCKING | The 90 × 29 raw-count table used to select each configuration is held only in memory as `raw_table`; it is never written to an output artifact. | Built at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:414), lines 414–423; pooled and selected at line 427, lines 427–449; output code writes only the chosen rows at line 494, lines 494–500. | The audit can verify the grid definition and that the persisted winner is in it, but cannot independently prove that all 90 configurations were evaluated or that config 37 won each fold under F1 → precision → fewer-FP → grid-order. |
| 2 | BLOCKING | No executable/source identity is bound to the Colab output. The copied result audit records the cache SHA but no SHA-256/Git commit for `boosting_official_glds_full.py`, no locked official-repository commit, and no Colab stdout log. | Cache identity is recorded at [`results /BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:5-6`](results%20/BOOSTING_SAMMLV_GLSD_FULL_AUDIT.md:5); the claimed verdict is at line 66. | Static review can show the current script follows the protocol, but cannot prove this exact source revision produced the copied result files. |
| 3 | BLOCKING | `glsd_final_predictions.pkl` persists intervals but not per-video IoU matches, per-subject raw/full counts, or recognition/neutral decision records. The final counts therefore cannot be independently recomputed from persisted artifacts without invoking the official evaluator, which this audit was forbidden to run. | [`results /glsd_final_predictions.pkl`](results%20/glsd_final_predictions.pkl); the evaluator and recognition calls are runtime-only at [`boosting_official_glds_full.py`](boosting_official_glds_full.py:317), lines 317–330. | The reported Raw `44/84/115` and Full `44/82/115` are consistent claims, not independently reproduced audit evidence. |

## Leakage conclusion

The inspected code and persisted selection membership provide positive evidence
that the outer held-out subject is excluded from its own configuration-selection
pool. There is **no observed outer-test membership leakage**. However, because
the 90 × 29 selection statistics were not persisted, the strict audit cannot
independently certify that the recorded per-fold winner was selected from the
complete evaluated grid during the Colab run.

## Required evidence for a future PASS (no rerun implied)

The completed Colab run must retain, alongside the existing four result files:

1. `glsd_raw_table_90x29.npz` or lossless CSV/JSON containing every
   `(config_id, outer-validation-subject, TP, FP, FN)` value used by selection;
2. a per-outer selection trace including all tie-break keys and rank; and
3. a manifest containing cache SHA-256, script SHA-256, official repository Git
   commit, Python/package versions, command line, and stdout/stderr log digest.

With those existing-run artifacts available, this audit can be upgraded without
retraining or changing any scientific logic.

## Final verdict

`BOOSTING_SAMMLV_GLSD_FULL_FINAL_AUDIT = FAIL`

The reported values remain:

- Native: Raw `51/144/108`; Full `51/141/108`.
- GLSD: Raw `44/84/115`; Full `44/82/115`.

They are **not approved as strictly validated** until the blocking provenance
and complete-selection evidence above are supplied.
