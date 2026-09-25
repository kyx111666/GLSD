# Reproduction notes

## 1. Cache-level decoder replay

Use Python 3.10 with `requirements/cache_reproduction.txt`. Supply the frozen
temporal response caches and the matching annotations expected by the selected
experiment. The scripts record the input paths and hashes in their output
manifest when that protocol requires provenance.

The first checks to run are:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 experiments/glsd_phase0/run_dense_segment_phase0_v2.py
```

The second command is an example entry point. It will stop with a missing-input
error until the required response caches and annotation files are available.

## 2. Controlled exploration

The `controlled_exploration` scripts replay a frozen parent protocol and write
their own tables. Run one experiment at a time from the repository root. Do
not reuse an output directory from a previous run when the script documents a
fresh-run requirement.

The baseline snapshot under
`experiments/controlled_exploration/baseline_snapshot/code/` is source only;
the result tables used during the original analysis are not part of this
release.

## 3. Official-response ablation

The programs under `experiments/official_response_ablation/` are designed for
the supplied Colab environments. They expect official response dumps and
sealed evaluation files mounted by the notebook. Run the preflight command
described in the local README before starting a full ablation. The package does
not contain Drive paths, response dumps, or model weights.

## 4. Baseline training

The baseline implementations have their own legacy dependency pins. Create a
separate environment for each baseline and follow its local README. Do not
install the legacy pins into the cache-level environment; the NumPy, SciPy,
pandas, and scikit-learn versions are intentionally different.

## 5. Data and outputs

Expected private inputs include original videos, frame annotations, official
response caches, recognition outputs, and model weights. Generated files should
be written to a run-specific `results/` or `outputs/` directory and are ignored
by the repository-level `.gitignore`.

Reported values must be regenerated from the matching protocol and input
version. A local unit-test pass only verifies implementation invariants; it is
not evidence that a full dataset replay has completed.
