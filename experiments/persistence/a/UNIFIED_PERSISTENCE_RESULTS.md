# Unified Persistence: Four-Setting Results

The primary comparison uses each backbone's native interval decoder. All four settings use the same formula and search grid; configurations are selected separately from each outer fold's training subjects.

| Backbone | Dataset | Original native F1 | Train-k native F1 | Unified F1 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metst | sammlv | 0.267677 | 0.267677 | 0.282282 | 47 | 127 | 112 |
| metst | casme3 | 0.087520 | 0.086580 | 0.093992 | 97 | 1109 | 761 |
| boostingvrme | sammlv | 0.277620 | 0.277620 | 0.283784 | 42 | 95 | 117 |
| boostingvrme | casme3 | 0.105144 | 0.105144 | 0.111378 | 116 | 1109 | 742 |

ME-TST/CASME3 has a different training-only duration prior in three outer folds. The original native result is also shown so the baseline change is explicit.

## Paired Uncertainty

| Backbone | Dataset | Delta F1 vs train-k native | Subject-paired 95% CI | Net TP | Net FP |
| --- | --- | ---: | --- | ---: | ---: |
| metst | sammlv | +0.014606 | [-0.016130, +0.042533] | -6 | -57 |
| metst | casme3 | +0.007412 | [-0.008127, +0.023314] | +17 | +199 |
| boostingvrme | sammlv | +0.006163 | [-0.025079, +0.037954] | -7 | -50 |
| boostingvrme | casme3 | +0.006234 | [-0.005723, +0.017352] | +23 | +291 |

All four point estimates improve, but every interval crosses zero. These are exploratory improvements, not evidence of established significance or independent generalization.

## Unified Rule

`S = w*H + (1-w)*(G+L)/2`; accept when `S >= tau`.

- H is continuous reference-curve height `(peak-mean)/(max-mean)`, clipped below at zero.
- G is global prominence divided by the reference smoothed curve's range.
- L is the median aligned local prominence across smoothing scales 1, 1.5, and 2 times k. Each scale uses its own range; missing matches contribute zero.
- Local radii are selected from k, 2k, and 3k. Reference smoothing is selected from k, 1.5k, and 2k.
- Height weights are 0, 0.25, 0.5, and 0.75. Thresholds are 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, and 0.75.
- There is no P-based candidate gate, high/low routing, strict/recovery priority, or global-peak snapping.
- Peak spacing is k. Rounded duplicate smoothing scales count once. Constant input produces no peaks.
- The unified pool has 360 configurations. Including native and ablations, 511 configurations are recorded.

## Complete Ablations

The primary family was unified for every setting; no per-dataset best-family substitution is used.

| Backbone | Dataset | Native | Unified | Persistence only | Height only | Global only | Local only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metst | sammlv | 0.267677 | 0.282282 | 0.288288 | 0.261905 | 0.263473 | 0.285714 |
| metst | casme3 | 0.086580 | 0.093992 | 0.094164 | 0.085854 | 0.100292 | 0.094819 |
| boostingvrme | sammlv | 0.277620 | 0.283784 | 0.283784 | 0.285714 | 0.291262 | 0.284722 |
| boostingvrme | casme3 | 0.105144 | 0.111378 | 0.112888 | 0.106434 | 0.112652 | 0.102641 |

## Verification and Scope

Duration priors exclude the outer subject, and additionally exclude the inner validation subject during inner selection. Configurations maximize aggregate inner raw F1; ties use precision, fewer FP, then grid order.

The verification script independently recomputes selected configurations, recounts every saved prediction using the old scalar IoU matcher, checks input/code hashes, and replays the label-free unified detector.

Frozen OOF backbone curves are reused. Inner-specific backbone predictions are unavailable, so this is not fully nested network retraining. Both datasets informed method design. Bootstrap uses 10,000 subject-paired draws and conditions on the selected configurations and frozen curves. Only raw spotting is evaluated.

Cached labels are preserved, including pre-existing malformed/out-of-bounds CASME3 intervals. BoostingVRME uses the existing reference-kernel curve cache; this work does not validate binary-kernel equivalence.

## Run

```powershell
& 'D:\Anaconda3\envs\ME-TST\python.exe' -B 'D:\workspace\a\boostingvrme\unified_persistence.py' --backbone both --dataset both
& 'D:\Anaconda3\envs\ME-TST\python.exe' -B -m unittest discover -s 'D:\workspace\a\boostingvrme' -p 'test_unified_persistence.py' -v
& 'D:\Anaconda3\envs\ME-TST\python.exe' -B 'D:\workspace\a\boostingvrme\verify_unified_persistence.py'
```

Each project writes `results/unified_persistence_final/<dataset>/<intervals>/`. `report.json`, per-subject selections/counts, selected predictions, and independent verification are included. BoostingVRME additionally has a fixed-interval sensitivity analysis.

The original scripts and results are preserved. `unified_persistence_v1` records the first, one-radius attempt that failed on ME-TST/SAMMLV; `unified_persistence_v2` records the wider-radius experiment. The final run adds a constant-input guard and reproduces v2's results.
