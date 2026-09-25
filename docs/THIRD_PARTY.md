# Third-party and baseline boundary

The package keeps several baseline source trees so that the experiment entry
points remain inspectable. They are not presented as original GLSD code.

- `experiments/baselines/boostingvrme/` includes the upstream source files that
  were present in the working archive. Its local `LICENSE.txt` and README are
  retained.
- `experiments/baselines/external/SoftNet-SpotME/` and
  `experiments/baselines/external/MEAN_Spot-then-recognize/` retain their local
  README and dependency files. Their datasets, weights, and image assets are
  not included.
- `experiments/third_party/metst_plus/` contains only the small evaluation
  helpers needed by selected replay scripts. It is separated from the method
  code so that its provenance is visible.

Before public release, confirm the upstream license terms and add a top-level
license for the complete repository. No new license terms are asserted here.
