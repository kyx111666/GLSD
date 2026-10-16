"""Score a synthetic response with the unchanged frozen GLSD core."""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from boosting_official_glds_full_LOCKED import GLSDConfig, GLSDFeatures

t = np.arange(120, dtype=float)
response = np.exp(-((t - 35) / 6)**2) + 0.65 * np.exp(-((t - 85) / 9)**2)
config = GLSDConfig(reference_scale=1.5, local_radius=2.0, threshold=0.3)
features = GLSDFeatures(response, k=5)
peaks, evidence = features.evidence(config.reference_scale, config.local_radius)
print(json.dumps({'candidate_peaks': peaks.tolist(), 'global_local_scores': evidence.tolist(),
                  'selected_peaks': features.selected_peaks(config).tolist()}, indent=2))
