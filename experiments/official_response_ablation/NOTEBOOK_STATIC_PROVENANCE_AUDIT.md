# Static provenance audit of the final Colab notebooks

Audit basis: the supplied notebooks, read without executing any cell. Drive paths are provenance declarations; this Mac audit does not call `Path.exists()` on `/content/drive/...` and makes no negative inference from their absence locally.

## Confirmed chains

| Setting | Official input declared by notebook | Sealed result/evidence declared by notebook | Final full F1 recorded |
|---|---|---|---|
| ME-TST+ / SAMMLV | `/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/SAMMLV_method1_strategy1` (29 `subject_*.pkl`; `result_all` and `result1_all`) | `/content/drive/MyDrive/GLSD_METST_OFFICIAL/results_v1`; `/content/drive/MyDrive/GLSD_METST_OFFICIAL/SAMMLV_FINAL_EVIDENCE` | Native `52/171/107`, .2723; GLSD `47/121/112`, .2875 |
| ME-TST+ / CAS(ME)3 | `/content/drive/MyDrive/ME-TST_OFFICIAL_DUMP/CASME_3_method1_strategy1` (94 dumps) | `/content/drive/MyDrive/GLSD_METST_OFFICIAL/CASME3/results_v1`; `/content/drive/MyDrive/GLSD_METST_OFFICIAL/CASME3_FINAL_EVIDENCE` | Native `76/736/777`, .0912912913; GLSD `90/898/763`, .0977729495 |
| BoostingVRME / SAMMLV | `/content/drive/MyDrive/GLSD_BoostingVRME_Full/sammlv_official_full_responses.pkl` | `/content/drive/MyDrive/GLSD_BoostingVRME_Full/results_evidence_v2` | Native `51/141/108`, .2906; GLSD `44/82/115`, .3088 |
| BoostingVRME / CAS(ME)3 | `/content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/official_response_cache/casme3_official_full_responses.pkl` | `/content/drive/MyDrive/GLSD_BoostingVRME_Full/CASME3/glsd_full_evidence_v1` | Native `84/786/769`, .0975043529; GLSD `120/1110/733`, .1152184349 |

## Static evidence details

- The ME-TST notebook records official dump patches only around already assembled response tensors and archives the locked source, `results_v1`, pre-registration, final lock JSON, and an evidence-bundle SHA-256 list.
- The ME-TST CAS(ME)3 final lock explicitly records `94` subjects, `462` aligned videos, and `853` GT events, distinct from the earlier 858-event controlled cache.
- The BoostingVRME SAMMLV notebook requires `glsd_search_all_90x29.csv`, `glsd_search_inner_fold_counts.csv`, `outer_selection_trace.json`, `per_subject_counts.csv`, a manifest, a stdout/stderr log, and a SHA-256 evidence bundle in `results_evidence_v2`.
- The BoostingVRME CAS(ME)3 notebook independently checks `94 × 90 = 8,460` outer/config rows, `94 × 90 × 93 = 786,780` inner rows, 94 selected rows, 462 video rows, 853 GT events, and the F1 → precision → fewer-FP → grid-order tie-break. Its recorded output is `BOOSTING_CASME3_GLSD_FULL_FINAL_AUDIT = PASS`.

## Patch design

The accompanying Colab runner uses those locations only after Drive mount. It creates a timestamped `official_response_ablation/` run directory, fails closed on absent/mismatched sealed inputs, uses the official decoder/evaluator and event geometry through the locked sources, and changes only the candidate score. It never accesses a historical controlled cache.
