# Agreement-bonus Colab screen
Reuse the functioning ME-TST+ Colab kernel and official repository/runtime.
Run the three cells in Colab_新融合AgreementBonus_SAMMLV.ipynb in order.
Upload metst_agreement_bonus.zip in cell 1. No inference or training.
Fixed a0=1.5, rho=2.0; SAMMLV only by default.
456 unique configurations: G/L/Mean 19 each, fusion 399 (21 pairs x 19 tau).
Six primary variants: G, L, Mean, Max, Bonus_joint, Full_joint.
Outer subject is excluded from pooled raw decoder selection. Report held-subject full official counts.
This is development on frozen responses, not a new backbone nested training experiment.
Native/locked GLSD replay and exact Mean hook checks must pass before interpreting results.
Fixed-pair heatmap is exploratory; main result is Full_joint with internal selection.
Full_joint includes lambda=0 and gamma=0 fallback configurations; inspect their selection frequency.
Event rescue TP/FP accounting is unavailable without verified video/event identity.
Local tests cover formula bounds, holdout isolation, grid and a synthetic adapter run.
No real Colab official experiment has been executed locally.
The included generalized-mean module is an unchanged utility dependency; its main is never invoked.
