"""SAMMLV shared structure search using the verified frozen-response runtime."""
import argparse
import contextlib
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
import zipfile

# Source checkout and delivered ZIP use the same driver and runtime.
HERE = Path(__file__).resolve().parent
if not (HERE / 'run_standalone.py').exists():
    sys.path.insert(0, str(HERE.parent / 'server_standalone'))
import run_standalone as runtime
import numpy as np
import pandas as pd

engine = runtime.engine
PROTOCOL = 'reference_residual_shared_structure_a123_r123_full_v1'
SCALES = (1., 1.5, 2.)
RADII = (1., 2., 3.)
TAUS = tuple(i / 20 for i in range(1, 20))
METHODS = ('G', 'L', 'Mean', 'PositiveResidual', 'RefResidual')
LOCKED = {'locked_no_local': 'G', 'locked_no_negative': 'PositiveResidual'}


def grids():
    return {m: [engine.phase.Config(m, i, a, r, t)
                for i, (a, r, t) in enumerate((a, r, t) for a in SCALES
                    for r in ((1.,) if m == 'G' else RADII) for t in TAUS)]
            for m in METHODS}


def config_key(c):
    return (c.method, c.reference_scale, 1. if c.method == 'G' else c.local_radius, c.threshold)


def export_features(context, core, output):
    summaries = []
    with gzip.open(output / 'features.jsonl.gz', 'wt', encoding='utf8') as stream:
        for si, subject in enumerate(context[6]):
            record = context[4][si]
            for a in SCALES:
                for r in RADII:
                    deltas = []
                    for vi, response in enumerate(record['result_all']):
                        feature = core.GLSDFeatures(response, record['k_p'])
                        data = feature.residual_evidence(a, r)
                        deltas.extend(data['D'])
                        row = dict(subject=subject, video_id=f'{subject}/video_{vi}',
                            video_name=str(record['videos'][vi]), k=record['k_p'],
                            response_sha256=feature.key[2], a0=a, rho=r, **data)
                        stream.write(json.dumps(engine.shared.serializable(row), allow_nan=False)+'\n')
                    d = np.asarray(deltas)
                    summaries.append(dict(subject=subject, a0=a, rho=r, candidates=len(d),
                        negative=int(np.sum(d < -1e-12)), zero=int(np.sum(np.abs(d) <= 1e-12)),
                        positive=int(np.sum(d > 1e-12)),
                        absolute_delta_median=float(np.median(np.abs(d))) if len(d) else 0.))
            print('structure features', si+1, '/ 29', flush=True)
    engine.shared.write_csv(output / 'feature_summary.csv', summaries)


def conditional_summary(context, all_grids, tables, output):
    """Predeclared 9-structure diagnostics; never choose a winner from this table."""
    summaries, selections = [], []
    for a in SCALES:
        for r in RADII:
            for m, grid in all_grids.items():
                allowed = [c.config_id for c in grid if c.reference_scale == a
                           and (m == 'G' or c.local_radius == r)]
                selected_counts = []
                for si, subject in enumerate(context[6]):
                    winner, pooled = engine.shared.choose(tables[m][1], si, allowed=allowed)
                    selected_counts.append(tables[m][1][winner, si])
                    selections.append(dict(subject=subject, method=m, a0=a, rho=r,
                        threshold=grid[winner].threshold, config_id=winner,
                        inner_F1=engine.shared.metrics(pooled[winner])['F1']))
                summaries.append(dict(method=m, a0=a, rho=r,
                    selection='fixed structure; other-subject full F1 selects tau',
                    **engine.shared.metrics(np.asarray(selected_counts).sum(0))))
    engine.shared.write_csv(output / 'conditional_structure_summary.csv', summaries)
    engine.shared.write_csv(output / 'conditional_structure_selections.csv', selections)


def screen(context, core, output):
    all_grids = grids()
    tables = {m: engine.search(context, core, grid, output / ('search_counts_'+m+'.npz'))
              for m, grid in all_grids.items()}
    lookup = {config_key(c): c.config_id for grid in all_grids.values() for c in grid}
    names = (*METHODS, *LOCKED)
    counts = {m: [] for m in names}
    selections, subject_rows = [], []
    with gzip.open(output / 'event_records.jsonl.gz', 'wt', encoding='utf8') as stream, \
         gzip.open(output / 'event_differences.jsonl.gz', 'wt', encoding='utf8') as diffs:
        for si, subject in enumerate(context[6]):
            selected, inner_scores = {}, {}
            for m in METHODS:
                winner, pooled = engine.shared.choose(tables[m][1], si)
                selected[m] = all_grids[m][winner]
                inner_scores[m] = engine.shared.metrics(pooled[winner])['F1']
            for name, method in LOCKED.items():
                selected[name] = replace(selected['RefResidual'], method=method, config_id=-1)
            records = {}
            for name, config in selected.items():
                search_id = lookup[config_key(config)]
                record = engine.decode(context, core, si, config, True)
                for stage_index, stage in enumerate(('raw', 'full')):
                    if not np.array_equal(record[stage+'_counts'], tables[config.method][stage_index][search_id, si]):
                        raise RuntimeError('Selected replay differs from matching search config')
                    subject_rows.append(dict(subject=subject, method=name, stage=stage,
                        **engine.shared.metrics(record[stage+'_counts'])))
                record['method'], record['scoring_method'] = name, config.method
                selections.append(dict(subject=subject, method=name, scoring_method=config.method,
                    config_id=config.config_id, search_config_id=search_id,
                    a0=config.reference_scale, rho=config.local_radius, threshold=config.threshold,
                    inner_F1=inner_scores.get(name),
                    selection='locked_RefResidual' if name in LOCKED else 'inner_full'))
                records[name] = record
                counts[name].append(record['full_counts'])
                stream.write(json.dumps(engine.shared.serializable(record), allow_nan=False)+'\n')
            for name in names:
                if name != 'RefResidual':
                    delta = engine.ledger.event_delta(records['RefResidual'], records[name])
                    diffs.write(json.dumps(engine.shared.serializable(delta), allow_nan=False)+'\n')
            print('selected structure replay', si+1, '/ 29', flush=True)
    counts = {m: np.asarray(c, np.int64) for m, c in counts.items()}
    summary = [dict(method=m, selection='locked_RefResidual' if m in LOCKED else 'inner_full',
                    configs=0 if m in LOCKED else len(all_grids[m]),
                    **engine.shared.metrics(c.sum(0))) for m, c in counts.items()]
    for filename, rows in [('summary_full.csv', summary), ('selected_configs.csv', selections),
                          ('per_subject_counts.csv', subject_rows),
                          ('paired_comparisons.csv', engine.comparisons(counts))]:
        engine.shared.write_csv(output / filename, rows)
    conditional_summary(context, all_grids, tables, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    dump, output = args.dump.resolve(), args.output.resolve()
    if not dump.is_dir():
        raise RuntimeError('Missing response directory: '+str(dump))
    for protected in (dump, runtime.ROOT):
        if output == protected or protected in output.parents or output in protected.parents:
            raise RuntimeError('Output must be separate from code and inputs')
    context, core = runtime.context_from_dump(dump)
    import scipy, sklearn
    identity = dict(protocol=PROTOCOL,
        runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                     scipy=scipy.__version__, sklearn=sklearn.__version__, platform=platform.platform()),
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in context[5]},
        code={str(p.relative_to(runtime.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.ROOT.rglob('*.py')},
        structure_driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grids={m: [asdict(c) for c in grid] for m, grid in grids().items()})
    identity = json.loads(json.dumps(identity))
    if output.exists():
        if not args.resume or not (output / 'run_manifest.json').exists():
            raise RuntimeError('Use a fresh output directory, or --resume for a previous structure run')
        if json.loads((output / 'run_manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('Code, inputs or runtime changed; use a fresh output directory')
        if (output / 'completion.json').exists():
            print('Already completed:', output, flush=True)
            return
    else:
        if args.resume:
            raise RuntimeError('Resume directory does not exist')
        output.mkdir(parents=True)
        runtime.write_json(output / 'run_manifest.json', dict(identity=identity))
    runtime.write_json(output / 'protocol.json', dict(protocol=PROTOCOL,
        primary='RefResidual', formula='clip(G + L - L0, 0, 1)',
        methods=METHODS, scales=SCALES, radii=RADII, thresholds=TAUS,
        config_counts={m: len(grid) for m, grid in grids().items()},
        selection='Each method separately: exclude outer subject; pooled inner full F1, precision, fewer FP, config index',
        tie_order='a0 ascending, rho ascending, tau ascending',
        ablations='locked_no_local=G and locked_no_negative=PositiveResidual inherit RefResidual a0/rho/tau',
        candidates='Shared across methods at equal a0; may change across a0; G independent of rho',
        unchanged='physical scales, alignment tolerance, median, missing=0, decoder, recognition, matching',
        diagnostic='Each of 9 fixed structures chooses tau in inner subjects only; do not select best row with outer F1',
        scope='SAMMLV development after repeated use; no independent confirmation claim',
        bootstrap='paired subjects; conditional on selected predictions; no retuning',
        weights_searched=False, previous_audit_required=False, old_counts_reused=False, training=False))
    video = context[3].spotting.__globals__['MeanAveragePrecision2d']
    classes = [context[2], video]
    try:
        with contextlib.ExitStack() as stack:
            contracts = [stack.enter_context(runtime.typed_matching.install(cls)) for cls in classes]
            if [c['return_contract'] for c in contracts] != ['numeric_list', 'string_scalar']:
                raise RuntimeError('Unexpected aggregate/video contracts')
            runtime.write_json(output / 'matching_protocol.json', contracts)
            runtime.audit(context, core, output, classes)
            export_features(context, core, output)
            summary = screen(context, core, output)
        runtime.write_json(output / 'completion.json', dict(completed=True, protocol=PROTOCOL,
            evaluation_audit='PASS', subjects=29, videos=79, ground_truth=159,
            configs=sum(len(g) for g in grids().values()), selected_records=29*(len(METHODS)+len(LOCKED))))
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
        print('STRUCTURE_SEARCH = PASS (execution complete; inspect efficacy separately)', flush=True)
    except Exception:
        runtime.write_json(output / 'failure_case.json', dict(traceback=traceback.format_exc(),
            context=getattr(core, 'last_context', {})))
        raise
    finally:
        archive = output.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file():
                    z.write(p, str(Path(output.name) / p.relative_to(output)))
        print('RESULT_ZIP =', archive, flush=True)


if __name__ == '__main__':
    main()
