"""P8 phase-2 under a separately versioned one-to-one event protocol.

Probe the real failing case before searching. No legacy count reuse.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import inspect
import io
import json
from pathlib import Path
import sys

import numpy as np
import run_p8_phase2 as phase
import one_to_one_evaluator as evaluator

shared = phase.shared


def gt_counts(context):
    return [sum(len(video) for video in subject) for subject in context[0].final_samples(context[4])]


def validate(raw, full, gt):
    for counts in (raw, full):
        if counts is None:
            continue
        if len(counts) != 3 or any(not np.isfinite(v) or v < 0 or int(v) != v for v in counts):
            raise RuntimeError('invalid counts: %s' % (counts,))
        if counts[0] + counts[2] != gt or counts[0] > gt:
            raise RuntimeError('GT conservation failed: counts=%s GT=%s' % (counts, gt))


def probe_case(context, recognition):
    runner, base, metric, official, records, _, subjects, _ = context
    si = next(i for i, s in enumerate(subjects) if str(s).lstrip('0') == '37')
    config = next(c for c in phase.grids()['L'] if
                  (c.reference_scale, c.local_radius, c.threshold) == (1.0, 2.0, 0.05))
    core = phase.FusionCore(base)
    # Bypass only the count guard to RECORD the original anomaly, never select on it.
    with contextlib.redirect_stdout(io.StringIO()):
        raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
            records, si, config, core, metric, official, recognition)
        full = runner.full_counts_from_official_synergy(raw, pred_list, gt_list) if recognition else None
    if core.calls == 0:
        raise RuntimeError('failing-case scorer was bypassed')
    return shared.serializable(dict(subject=subjects[si], GT=gt_counts(context)[si],
        config=phase.asdict(config), raw=raw, full=full, predictions=predictions))


def corrected_native(ora, context, name, output):
    rows = []
    subjects, gts = context[6], gt_counts(context)
    for si, subject in enumerate(subjects):
        with contextlib.redirect_stdout(io.StringIO()):
            raw, full = ora.native_subject_counts('metst', context, si)
        validate(raw, full, gts[si])
        for stage, counts in (('raw', raw), ('full', full)):
            rows.append(dict(setting=name, subject=subject, method='Native', evaluation=stage,
                             **shared.metrics(counts)))
    shared.write_csv(output / 'native_per_subject_counts.csv', rows)
    full_rows = [r for r in rows if r['evaluation'] == 'full']
    counts = np.asarray([[r[k] for k in ('TP', 'FP', 'FN')] for r in full_rows]).sum(0)
    return dict(setting=name, selector='none', method='Native', evaluation='full', configs=0,
                **shared.metrics(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'full'), default='probe')
    parser.add_argument('--setting', choices=('sammlv', 'casme3', 'both'), default='sammlv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ora, helper = shared.load_helper()
    names = list(shared.SETTINGS.values()) if args.setting == 'both' else [shared.SETTINGS[args.setting]]
    root = Path('/content/drive/MyDrive/GLSD_P8_ONE_TO_ONE')
    output = (args.output or root / (args.mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
    for spec in ora.SPECS.values():
        for key in ('dump', 'cache', 'evidence', 'run_evidence', 'results', 'locked_results'):
            if key in spec and phase.overlaps(output, Path(spec[key]).resolve()):
                raise RuntimeError('output overlaps sealed input')
    paths = [Path(__file__), Path(phase.__file__), Path(shared.__file__), helper,
             Path(evaluator.__file__), Path(__file__).with_name('colab_p8_entry.py'),
             Path(__file__).with_name('colab_p8_one_to_one_entry.py')]
    identity = dict(mode=args.mode, settings=names, protocol=evaluator.PROTOCOL,
                    code_sha256={p.name: ora.sha256(p) for p in paths},
                    runtime=dict(python=sys.version, numpy=np.__version__,
                                 pandas=getattr(sys.modules.get('pandas'), '__version__', None)), inputs={})
    for name in names:
        spec = ora.SPECS[name]
        ora.verify_sealed_inputs(spec)
        inputs = [spec['source'], spec['core'], spec['evidence'] / 'evidence_bundle_sha256.txt',
                  spec['run_evidence'] / 'run_manifest.json']
        inputs += sorted(p for p in spec['dump'].rglob('*') if p.is_file())
        print('Hashing frozen inputs:', name, flush=True)
        identity['inputs'][name] = {str(p): ora.sha256(p) for p in inputs}
    if output.exists():
        if not args.resume or not (output / 'run_manifest.json').is_file():
            raise RuntimeError('use a new output directory or explicit --resume')
        if json.loads((output / 'run_manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('resume refused: evaluation protocol, mode, code, input or runtime mismatch')
        if (output / 'completion.json').exists():
            print('Already completed:', output, flush=True)
            return
    else:
        if args.resume:
            raise RuntimeError('resume path does not exist')
        output.mkdir(parents=True)
        shared.write_json(output / 'run_manifest.json', dict(identity=identity))
    print('OUTPUT =', output, flush=True)
    preflight = output / 'legacy_preflight'
    preflight.mkdir(exist_ok=True)
    shared.write_csv(preflight / 'locked_replay_summary.csv',
                     [ora.run_preflight_setting(n, preflight) for n in names])
    print('LEGACY_REPLAY = PASS (historical provenance only)', flush=True)
    summaries, comparisons = [], []
    for name in names:
        # Load context/native gate BEFORE patching the official class.
        context = ora.metst_context(ora.SPECS[name])
        audit = output / (name + '_audit')
        audit.mkdir(exist_ok=True)
        before = None
        if name == 'metst_sammlv':
            before = probe_case(context, False)
            shared.write_json(audit / 'subject037_legacy.json', before)
            print('SUBJECT037_LEGACY_RAW =', before['raw'], 'GT =', before['GT'], flush=True)
        with evaluator.install(context[2]) as info:
            metric_path = Path(inspect.getsourcefile(context[2]))
            info['metric_source_sha256'] = ora.sha256(metric_path)
            prior_info = audit / 'matching_protocol.json'
            if args.resume and prior_info.exists():
                previous = json.loads(prior_info.read_text())
                for key in ('metric_source_sha256', 'legacy_check_box_sha256', 'protocol'):
                    if previous[key] != info[key]:
                        raise RuntimeError('resume refused: evaluator changed: %s' % key)
            shared.write_json(audit / 'matching_protocol.json', info)
            if before is not None:
                after = probe_case(context, True)
                validate(after['raw'], after['full'], after['GT'])
                if before['predictions'] != after['predictions']:
                    raise RuntimeError('matching patch changed prediction events')
                shared.write_json(audit / 'subject037_one_to_one.json', after)
                print('SUBJECT037_ONE_TO_ONE = PASS; raw=', after['raw'], 'full=', after['full'], flush=True)
                print('PREDICTIONS_UNCHANGED = PASS', flush=True)
            summaries.append(corrected_native(ora, context, name, audit))
            print(name, 'CORRECTED_NATIVE = PASS', flush=True)
            if args.mode == 'full':
                result, paired = phase.run_setting(ora, name, output / name,
                    prepared_context=context, evaluation_protocol=evaluator.PROTOCOL)
                summaries.extend(result)
                comparisons.extend(paired)
                shared.write_csv(output / 'summary_full.csv', summaries)
                shared.write_csv(output / 'paired_comparisons.csv', comparisons)
    if args.mode == 'probe':
        shared.write_csv(output / 'native_summary_full.csv', summaries)
    else:
        details = []
        for name in names:
            f = {r['method']: r['F1'] for r in summaries if r['setting'] == name}
            details.append(dict(setting=name, p8_exceeds_both_singles=f['GM_p8'] > max(f['G'], f['L']),
                                delta_vs_G=f['GM_p8']-f['G'], delta_vs_L=f['GM_p8']-f['L']))
        shared.write_json(output / 'decision.json', dict(stage='development', protocol=evaluator.PROTOCOL,
                                                         comparisons=details, not_comparable_to_legacy_counts=True))
    shared.write_json(output / 'completion.json', dict(completed=True, mode=args.mode,
                                                       protocol=evaluator.PROTOCOL, settings=names))
    print('P8_ONE_TO_ONE_%s = PASS' % args.mode.upper(), flush=True)
    print('OUTPUT =', output, flush=True)


if __name__ == '__main__':
    main()
