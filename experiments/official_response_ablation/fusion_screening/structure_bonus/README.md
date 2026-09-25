# SAMMLV structure-matched simple-bonus development experiment

Reuse the already verified server environment and /content symbolic links. No new data, training or inference.
Run python3 start_server.py to validate and launch in background. New outputs go to glsd_agreement/results/structure_bonus_sammlv_<timestamp>.

Methods: G, L, Mean, SimpleBonus. SimpleBonus=G+L-GL, computed using max+min*(1-max) to retain previous floating-point endpoint behavior. No lambda or gamma search.
a0={1,1.5,2}; rho={1,2,3}; tau={0.05,0.10,...,0.95}. G radius is deduplicated to 1.
Effective budgets: G=57, L=171, Mean=171, SimpleBonus=171; total 570.
Selection: pool all other subjects' raw TP/FP/FN; tie-break F1, precision, fewer FP, fixed grid order. Reporting: held-subject full official recognition/result synergy.
Frozen response decoder leave-subject-out selection only; not new backbone nested training. All k/prior handling is inherited.

Gates: exact Native and historical GLSD replay; every structure's Mean mask equals sealed core; search raw counts replay; selected Mean replays sealed scorer with the same newly chosen configs.
The new Mean search uses 19 thresholds. It is NOT expected to match GLSD-90 selections (10 thresholds); the historical result is only a separate replay reference.
The protocol's fixed_gamma denotes inherited peak alignment tolerance, NOT the removed fusion gamma.

Outputs: structure_summary_full.csv, paired_comparisons.csv, per_subject_counts.csv, selected_configs.csv, selected_parameter_frequency.csv, search_counts_<method>.npz, search_all.csv, candidate_decisions.jsonl.gz, selected_predictions.jsonl.gz, protocol.json, completion.json.
Bootstrap: paired subjects, 10000 draws, seed 100. Development intervals only.
Candidate traces lack verified video/event IDs; do not claim rescued/lost TP event accounting.

Local tests passed: formula/grid, structure cache/mean gate, outer exclusion, synthetic official adapter flow. Real server official replay and experiment are pending.
