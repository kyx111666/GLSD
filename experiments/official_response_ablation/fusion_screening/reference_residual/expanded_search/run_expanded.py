"""Expanded-scale shared structure search.

This wrapper reuses the audited structure-search driver while changing only
the declared physical scale set, reference scales and local-radius grid.
The sealed core is configured in memory; no frozen source file is edited.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / 'run_structure.py').exists():
    sys.path.insert(0, str(HERE.parent / 'structure_search'))
import run_structure as base

# Every method uses this same enlarged scale set.  Thus a0 expansion does not
# silently change only the proposed method's inputs.
base.SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
base.RADII = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
base.TAUS = tuple(i / 20 for i in range(1, 20))
base.PROTOCOL = 'reference_residual_expanded_scales_a16_r6_v1'
original_write_json = base.runtime.write_json


def write_expanded_metadata(path, data):
    if path.name == 'protocol.json':
        data = dict(data, physical_scales=base.SCALES,
                    changed='physical scale voting set expanded for all methods; reference and radius search expanded',
                    unchanged='alignment tolerance, median operation, missing=0, decoder, recognition, matching',
                    diagnostic='Each fixed structure chooses tau in inner subjects only; do not choose best outer row')
    original_write_json(path, data)


def expanded_context(dump):
    runner = base.runtime.load_module(base.runtime.ROOT/'sealed/metst_official_glds_full.py',
                                      'expanded_sealed_runner')
    core_module = base.runtime.load_module(base.runtime.ROOT/'sealed/boosting_official_glds_full_LOCKED.py',
                                           'expanded_sealed_core')
    # The sealed core reads this module constant while constructing scale_data.
    # This is an in-memory protocol setting, with exact original smoothing,
    # find_peaks, spread, alignment and missing=0 operations retained.
    core_module.SCALES = base.SCALES
    runner.ROOT = base.runtime.ROOT/'author_runtime'
    runner.DUMP_DIR = dump
    runner.LOCKED_GLSD_PATH = base.runtime.ROOT/'sealed/boosting_official_glds_full_LOCKED.py'
    metric, official = runner.load_official_metst()
    paths, records = runner.load_records()
    subjects = [str(s) for s in runner.subject_names(records)]
    if len(set(subjects)) != 29 or sum(len(r['videos']) for r in records) != 79:
        raise RuntimeError('Expected 29 distinct SAMMLV subjects and 79 videos')
    context = (runner, core_module, metric, official, records, paths, subjects, None)
    core = base.runtime.residual_features.FusionCore(core_module)
    core.expected_gt = base.engine.previous.gt_counts(context)
    if sum(core.expected_gt) != 159:
        raise RuntimeError('Expected 159 ground-truth events')
    return context, core


base.runtime.context_from_dump = expanded_context
base.runtime.write_json = write_expanded_metadata

if __name__ == '__main__':
    base.main()
