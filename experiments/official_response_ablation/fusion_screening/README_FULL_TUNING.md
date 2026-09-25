# Full-structure ME-TST+ tuning

Use this after the existing ME-TST+ environment and official-response package are loaded in Colab. Upload `run_full_fusion_tuning.py` into the existing `/content/glsd_fusion_screening/` directory; keep the existing `run_fusion_screening.py` and `official_response_component_ablation.py` beside it.

The search varies `a0={1,1.5,2}` and `rho={1,2,3}`. It evaluates G, L, EqualMean, WeightedMean and DominantEvidence. Effective grids are G=30, L=90, EqualMean=90, WeightedMean=540 and DominantEvidence=540, for 1,290 configurations per dataset. All outer selection uses pooled inner raw counts; held-out subjects are evaluated through the unchanged official full result-synergy path.

Run `--setting both` with a new output root such as `GLSD_FULL_FUSION_TUNING`. The script first replays Native and locked GLSD. It also checks that fresh full EqualMean-90 selection reproduces the locked per-subject full counts before continuing.

This is a development experiment on the already explored ME-TST+ settings. It does not claim independent confirmation and does not automatically expand the grid.
