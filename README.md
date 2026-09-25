# GLSD

Reference implementation, notebooks, and reproducibility materials for the
Global-Local Saliency Decoder (GLSD) study on long-video micro-expression
spotting.

The repository is organised around experiment families rather than one large
training script. The main implementation operates on frozen temporal response
curves and keeps each upstream detector's native interval construction rules.

## Repository map

| Directory | Contents |
| --- | --- |
| `experiments/glsd_phase0/` | Boundary, dense-segment, sparse-event, persistence, and decoder diagnostics. |
| `experiments/controlled_exploration/` | EXP-1 through EXP-7 controlled studies and their protocol files. |
| `experiments/official_response_ablation/` | Official-response component ablations and fusion screening programs. |
| `experiments/cross_pipeline_transfer/` | Cross-backbone transfer checks and preflight utilities. |
| `experiments/persistence/` | Persistence and morphology selection programs for the two main backbones. |
| `experiments/novel_method/` | Independent decoder candidates, audits, and feasibility studies. |
| `experiments/baselines/` | Baseline implementations kept separate from the GLSD code. |
| `experiments/mllm_verifier/` | Optional frame-level verification experiments and Colab entry points. |
| `notebooks/` | Cleaned Colab notebooks with execution output and runtime metadata removed. |
| `tests/` | Local protocol and decoder tests that do not require the private datasets. |
| `docs/` | Scope, reproduction notes, and the experiment index. |

## Quick start

Create a small environment for the cache-level decoder checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cache_reproduction.txt
python -m unittest discover -s tests -p 'test_*.py'
```

The tests exercise deterministic candidate generation, matching, and selection
logic. Full experiments require the corresponding response caches, annotations,
and (for baseline training) model weights. Those inputs are intentionally not
stored in this repository.

For a particular experiment, run the entry point from the repository root. For
example:

```bash
python3 experiments/glsd_phase0/run_dense_segment_phase0_v2.py
python3 experiments/controlled_exploration/EXP2C_reference_evidence_decoupling/run_exp2c.py
```

The scripts write new results beside their experiment directory or to the
output path specified by their command-line arguments. Do not place private
response caches, videos, labels, or model weights under version control.

## Reproduction boundary

The public package contains source code, configuration, protocol descriptions,
and tests. It does not contain original videos, annotations, response caches,
trained weights, local Drive folders, result archives, or paper build output.
See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the required external inputs and
the difference between cache-level replay and full backbone training.

The files under `experiments/baselines/` retain their original upstream layout
where practical. Use their local README and requirements file in a separate
environment; the legacy pins are not compatible with the cache-level
environment above.

## Release hygiene

Notebook execution state has been removed. Generated results, caches, logs,
virtual environments, archives, and platform-specific files are ignored by
`.gitignore`. A release manifest is written to `FILE_MANIFEST.sha256` after the
final source review.

No top-level software license has been selected yet. Add the intended license
before publishing the repository if redistribution terms are required. Review
[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) for the baseline code boundary.
