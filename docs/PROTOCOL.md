# GLSD parameter selection

GLSD combines normalized global prominence G and aligned local saliency L with equal weight. Local saliency uses support scales 1, 1.5 and 2. The parameter grid contains 90 combinations:

- Reference scale: 1, 1.5, 2.
- Local radius: 1, 2, 3.
- Threshold: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.75.

For each held-out subject, pool TP, FP and FN across the remaining subjects. Retain equal-mean configurations whose F1 is within 0.002 of the largest inner-fold F1. Among these configurations, maximize the smaller F1 improvement over the corresponding G-only and L-only configurations. Break ties by higher equal-mean F1, higher precision, fewer false positives, then grid order. The held-out subject never contributes to selection.

`run.py search` computes the per-subject search counts, applies this selection rule and evaluates each held-out subject. G-only and L-only configurations are independently selected by inner-fold F1, precision, fewer false positives and grid order. The locked component ablations reuse the GLSD configuration and remove one score component at a time. `select_glsd.py` can also apply the epsilon rule to saved search counts.

For locked-weight ablation, keep each fold's selected reference scale, radius and threshold fixed and vary only the fusion weight over 0, 0.3, 0.5, 0.7 and 1.

For cross-pipeline transfer, transfer the selected reference scale, radius and threshold. The target pipeline retains its temporal scale, response sequence, event boundaries, candidate ordering, conflict handling and recognition interface.

For threshold sensitivity, vary each selected threshold by -0.05, 0 and +0.05, with a lower bound of zero. Keep the subject assignment, reference scale and local radius fixed.

Runtime is measured after loading responses and selected configurations. Each method receives a warm-up run followed by ten timed passes, alternating Native/GLSD execution order. Each GLSD pass reconstructs its features. The upstream spotting function supplies event boundaries and conflict handling. Metric accumulation is disabled for both methods; recognition, backbone inference, input loading and parameter search are outside the timed region. Report Native, GLSD and their paired time difference per video.
