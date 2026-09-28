# Third-party and baseline boundary

The top-level [MIT License](../LICENSE) applies only to original GLSD source
code, configuration, documentation, and tests. It does not relicense copied
baseline or third-party material.

The following directories are kept so that the experiment entry points remain
inspectable, but they are not presented as original GLSD code:

- `experiments/baselines/boostingvrme/` includes upstream source files from the
  working archive. Its local `LICENSE.txt` and README remain the governing
  notices for that component.
- `experiments/baselines/external/SoftNet-SpotME/` and
  `experiments/baselines/external/MEAN_Spot-then-recognize/` retain their local
  README and dependency files. Their datasets, weights, and image assets are
  not included. No additional license is granted here; users should consult
  the respective upstream projects before reusing or redistributing these
  files.
- `experiments/third_party/metst_plus/` contains only the small evaluation
  helpers needed by selected replay scripts. It is separated from the method
  code so that its provenance remains visible. No additional license is
  asserted for this directory.

The repository does not redistribute original videos, annotations, response
caches, trained weights, or other private data. Users are responsible for
following the upstream terms and obtaining any permissions required for
third-party components.
