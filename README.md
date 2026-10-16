# GLSD

Code for **Global-Local Saliency Decoding for Micro-Expression Spotting in Long Videos**.

GLSD selects peaks from frozen temporal responses using normalized global prominence and aligned local multi-scale saliency. ME-TST+ and BoostingVRME retain their own event boundaries, candidate order and conflict handling.

## Quick start

Use a Python 3.10+ environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/decode_response.py
```

The example scores a synthetic response and requires no dataset or upstream model. It prints candidate peaks, global/local scores and retained peaks.

## Experiments

| Experiment | Command |
| --- | --- |
| Native, G, L and GLSD; locked component ablation | `python run.py search ...` |
| Replay saved Native/GLSD configurations | `python run.py replay ...` |
| Locked fusion-weight ablation | `python run.py ablation ...` |
| Cross-pipeline configuration transfer | `python run.py transfer ...` |
| Threshold sensitivity (τ ± 0.05) | `python run.py threshold ...` |
| Decoder runtime | `python run.py runtime ...` |

All paths are supplied through `configs/inputs.example.json`; copy it to `configs/inputs.json` and fill in your local inputs. Relative paths resolve from the configuration file, independently of the shell's working directory.

```bash
python run.py check --config configs/inputs.json
python run.py search --config configs/inputs.json \
  --setting boosting_sammlv --output outputs/boosting_sammlv
```

Settings are `metst_sammlv`, `metst_casme3`, `boosting_sammlv` and `boosting_casme3`. Full experiments require the corresponding official response dumps/caches, upstream evaluation code and original experiment metadata. They do not retrain the response generators.

See [reproduction instructions](docs/REPRODUCE.md) for input formats and experiment commands, and [the selection protocol](docs/PROTOCOL.md) for the GLSD parameter-selection rule.

## Layout

- `src/`: decoder implementations, experiment runners and evaluation helpers.
- `run.py`: portable experiment launcher.
- `configs/`: input configuration template.
- `examples/`: synthetic response example.
- `tests/`: decoder, evaluator, transfer and launcher regression checks.
- `docs/`: reproduction, protocol and upstream dependencies.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_locked_alpha_ablation.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_release.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_selection.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_paper_experiments.py'
PYTHONPATH=src python tests/test_transfer_protocol.py
```

The evaluator regression suite also needs the upstream BoostingVRME evaluator:

```bash
GLSD_BOOSTING_ROOT=external/BoostingVRME PYTHONPATH=src \
  python -m unittest discover -s tests -p 'test_*.py'
```

## License

Original GLSD code is released under the [MIT License](LICENSE). Upstream software is obtained separately and retains its own license; see [third-party dependencies](docs/THIRD_PARTY.md).
