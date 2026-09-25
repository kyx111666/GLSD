"""Threshold controls and locked-parameter event ablations on frozen ME-TST responses."""
import argparse
import contextlib
import csv
from dataclasses import asdict, replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import run_p8_phase2 as phase
import run_p8_one_to_one as previous
import one_to_one_evaluator as evaluator

shared = phase.shared
FACTOR = 2.0 ** (1.0 / 8.0)
PROTOCOL = 'p8_rho1_threshold_control_v1'


def grids():
    old = phase.grids()
    result = {m: old[m] for m in ('G', 'L', 'Mean')}
    result['GM_p8'] = [replace(c, config_id=i) for i, c in enumerate(
        c for c in old['GM_p8'] if c.local_radius == 1)]
    result['G_rescaled'] = [replace(c, method='G_rescaled') for c in old['G']]
    # Original grid first: unchanged tie preference; mapped grid second.
    result['G_expanded'] = [replace(c, method='G_expanded', config_id=i) for i, c in enumerate(
        old['G'] + [replace(c, threshold=FACTOR*c.threshold) for c in old['G']])]
    return result


class FeatureView(phase.FeatureView):
    def selected_peaks(self, config):
        peaks, values = self.evidence(config.reference_scale, config.local_radius)
        if config.method in ('G_rescaled', 'p8_no_L'):
            scores = values[:, 0] / FACTOR
            # Canonical mapped comparison ensures identical floating-point boundary semantics.
            keep = values[:, 0] >= FACTOR * config.threshold
        elif config.method == 'p8_no_G':
            scores = values[:, 1] / FACTOR
            keep = values[:, 1] >= FACTOR * config.threshold
        else:
            scores = phase.score(values, 'G' if config.method == 'G_expanded' else config.method)
            keep = scores >= config.threshold
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.key[0],
                peaks=peaks.tolist(), G=values[:, 0].tolist(), L=values[:, 1].tolist(),
                score=scores.tolist(), retained=keep.tolist()))
        return peaks[keep]


class FusionCore(phase.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


@contextlib.contextmanager
def capture_registered_inputs(metric_class):
    """Observe public add calls without assuming evaluator-owned array buffers."""
    original_add = metric_class.add
    captured = {}

    def add(self, preds, gt, *args, **kwargs):
        pred_copy, gt_copy = np.array(preds, copy=True), np.array(gt, copy=True)
        result = original_add(self, preds, gt, *args, **kwargs)
        # Keep the instance alive so object IDs cannot be recycled during decode.
        entry = captured.setdefault(id(self), (self, []))
        entry[1].append((pred_copy, gt_copy))
        return result

    metric_class.add = add
    try:
        yield captured
    finally:
        metric_class.add = original_add


@contextlib.contextmanager
def capture_decoder_inputs(metric_class, official):
    """Observe both the supplied aggregate and spotting-owned video evaluators."""
    classes = [metric_class]
    if official is not None:
        video_class = official.spotting.__globals__.get('MeanAveragePrecision2d')
        if not isinstance(video_class, type) or not callable(getattr(video_class, 'add', None)):
            raise RuntimeError('cannot resolve official spotting video evaluator class')
        if video_class not in classes:
            classes.append(video_class)
    with contextlib.ExitStack() as stack:
        captures = [stack.enter_context(capture_registered_inputs(cls)) for cls in classes]
        yield captures


def event_rows(runner, record, subject, predictions, matches, labels, targets, registered, raw, full):
    """Preserve official assignment; derive each full contribution using sealed synergy."""
    if len(predictions) != len(record['videos']) or len(registered) != len(predictions):
        raise RuntimeError('video/GT identity alignment failed')
    rows, gt_rows, offset = [], [], 0
    for vi, preds in enumerate(predictions):
        video_id = '%s/video_%d' % (subject, vi)
        registered_preds, gt = registered[vi]
        if not np.array_equal(np.asarray(preds), registered_preds):
            raise RuntimeError('prediction order differs from registered evaluator input')
        if len(gt) != len(record['samples'][vi]):
            raise RuntimeError('registered GT count differs from record')
        for gi, target in enumerate(gt):
            sample = record['samples'][vi][gi]
            if target[0] != sample[0] or target[2] != sample[2]:
                raise RuntimeError('GT index does not map to source annotation coordinates')
            gt_rows.append(dict(gt_id='%s/gt_%d' % (video_id, gi), video_id=video_id,
                video_name=str(record['videos'][vi]), gt_index=gi, registered_gt=np.asarray(target).tolist(),
                source_sample=shared.serializable(record['samples'][vi][gi])))
        mapping = matches.get(vi, [])
        if len(mapping) != len(preds):
            raise RuntimeError('prediction/match length mismatch')
        for pi, (pred, ids) in enumerate(zip(preds, mapping)):
            ids = np.asarray(ids).reshape(-1).astype(int).tolist()
            if len(ids) != 1 or ids[0] < -1 or ids[0] >= len(gt):
                raise RuntimeError('invalid one-to-one GT ID')
            gi = ids[0]
            label, target = int(labels[offset]), int(targets[offset])
            offset += 1
            if (gi >= 0) != (target != -1):
                raise RuntimeError('recognition target disagrees with spotting match')
            single = (1, 0, 0) if gi >= 0 else (0, 1, 0)
            contribution = tuple(runner.full_counts_from_official_synergy(single, [label], [target]))
            permitted = ((1,0,0), (0,0,1)) if gi >= 0 else ((0,1,0), (0,0,0))
            if contribution not in permitted:
                raise RuntimeError('unsupported full-stage event semantics')
            rows.append(dict(video_id=video_id, video_name=str(record['videos'][vi]),
                prediction_index=pi, interval=np.asarray(pred).tolist(),
                matched_gt_id=None if gi < 0 else '%s/gt_%d' % (video_id, gi),
                recognition_prediction=label, recognition_target=target,
                raw_TP=int(gi >= 0), raw_FP=int(gi < 0),
                full_TP=int(contribution[0]), full_FP=int(contribution[1]),
                full_retained=bool(contribution[0]+contribution[1])))
    if offset != len(labels) or offset != len(targets):
        raise RuntimeError('recognition labels not exhausted')
    for stage, expected in (('raw', raw), ('full', full)):
        tp = sum(r[stage+'_TP'] for r in rows)
        fp = sum(r[stage+'_FP'] for r in rows)
        if (tp, fp, len(gt_rows)-tp) != tuple(expected):
            raise RuntimeError('event ledger does not reconstruct %s counts' % stage)
        ids = [r['matched_gt_id'] for r in rows if r[stage+'_TP']]
        if len(ids) != len(set(ids)):
            raise RuntimeError('duplicate GT event ID')
    return rows, gt_rows


def detailed_decode(context, core, si, config):
    runner, _, metric_class, official, records, _, subjects, _ = context
    core.trace, core.capture = [], True
    before = core.calls
    with capture_decoder_inputs(metric_class, official) as captures, contextlib.redirect_stdout(io.StringIO()):
        raw, predictions, labels, targets, matches, metric = runner.decode_glsd_subject(
            records, si, config, core, metric_class, official, True)
        full = runner.full_counts_from_official_synergy(raw, labels, targets)
    if core.calls == before:
        raise RuntimeError('fusion scorer bypassed')
    previous.validate(raw, full, core.expected_gt[si])
    captured = [entry[id(metric)] for entry in captures if id(metric) in entry]
    if len(captured) != 1:
        raise RuntimeError('returned video evaluator has no captured add calls')
    rows, gt = event_rows(runner, records[si], subjects[si], predictions, matches,
                         labels, targets, captured[0][1], raw, full)
    traces = list(core.trace)
    if len(traces) != len(records[si]['result_all']):
        raise RuntimeError('candidate trace/video count mismatch')
    for vi, (trace, response) in enumerate(zip(traces, records[si]['result_all'])):
        digest = hashlib.sha256(np.ascontiguousarray(np.asarray(response, dtype=float)).tobytes()).hexdigest()
        if trace['response_sha256'] != digest:
            raise RuntimeError('candidate trace/video order mismatch')
        trace.update(video_id='%s/video_%d' % (subjects[si], vi),
                     video_name=str(records[si]['videos'][vi]))
    return dict(subject=subjects[si], method=config.method, config=asdict(config),
                raw_counts=list(map(int, raw)), full_counts=list(map(int, full)),
                events=rows, ground_truth=gt, candidates=traces)


def event_delta(fusion, reference):
    def matched(record):
        return {r['matched_gt_id'] for r in record['events'] if r['full_TP']}
    def candidates(record):
        return {(r['video_id'], p) for r in record['candidates']
                for p, keep in zip(r['peaks'], r['retained']) if keep}
    a, b = matched(fusion), matched(reference)
    ca, cb = candidates(fusion), candidates(reference)
    delta = np.asarray(fusion['full_counts']) - np.asarray(reference['full_counts'])
    if len(a-b)-len(b-a) != delta[0]:
        raise RuntimeError('GT set delta does not match TP delta')
    return dict(subject=fusion['subject'], compared=fusion['method'], reference=reference['method'],
                added_gt_ids=sorted(a-b), lost_gt_ids=sorted(b-a),
                added_candidates=sorted(ca-cb), removed_candidates=sorted(cb-ca),
                delta_TP=int(delta[0]), delta_FP=int(delta[1]), delta_FN=int(delta[2]))


def load_old_counts(root, name, subjects, gts):
    result = {}
    for method, configs in phase.grids().items():
        with np.load(root/name/('search_counts_%s.npz' % method), allow_pickle=False) as z:
            if json.loads(z['configs'].item()) != [asdict(c) for c in configs]:
                raise RuntimeError('old config identity mismatch')
            if z['subjects'].tolist() != list(subjects) or not z['done'].all():
                raise RuntimeError('old subject identity/completion mismatch')
            arrays = tuple(z[k].copy() for k in ('raw','full'))
        for a in arrays:
            if a.shape != (len(configs),len(subjects),3) or not np.issubdtype(a.dtype,np.integer):
                raise RuntimeError('invalid reused count shape/type')
            if np.any(a < 0) or not np.all(a[:,:,0]+a[:,:,2] == np.asarray(gts)):
                raise RuntimeError('invalid reused counts')
        if method == 'GM_p8':
            indices = [c.config_id for c in configs if c.local_radius == 1]
            arrays = tuple(a[indices] for a in arrays)
        result[method] = arrays
    return result


def check_mapping(context, core, indices):
    checked = 0
    for si in indices:
        d = context[4][si]
        for response in d['result_all']:
            feature = core.GLSDFeatures(response, d['k_p'])
            for c in grids()['G_rescaled']:
                a = feature.selected_peaks(c)
                b = feature.selected_peaks(replace(c, method='G_expanded', threshold=FACTOR*c.threshold))
                if not np.array_equal(a,b):
                    raise RuntimeError('mapped/rescaled G mismatch')
                checked += 1
    return checked


def paired(counts):
    subjects = len(counts['GM_p8'])
    draws = np.random.default_rng(100).integers(0, subjects, size=(10000,subjects))
    def sampled(a):
        s = a[draws].sum(1)
        d = 2*s[:,0]+s[:,1]+s[:,2]
        return np.divide(2*s[:,0],d,out=np.zeros(len(d)),where=d>0)
    p8 = sampled(counts['GM_p8'])
    rows = []
    for name, a in counts.items():
        if name == 'GM_p8':
            continue
        d = p8-sampled(a)
        rows.append(dict(compared='GM_p8',reference=name,
            delta_F1=shared.metrics(counts['GM_p8'].sum(0))['F1']-shared.metrics(a.sum(0))['F1'],
            CI_low=float(np.quantile(d,.025)),CI_high=float(np.quantile(d,.975)),
            resamples=10000,seed=100,interpretation='conditional on selected predictions; no retuning or exploration correction'))
    return rows


def run_setting(context, old_root, name, output, mode):
    output.mkdir(exist_ok=True)
    core = FusionCore(context[1])
    core.expected_gt = previous.gt_counts(context)
    subjects = context[6]
    tables = load_old_counts(old_root,name,subjects,core.expected_gt)
    all_grids = grids()
    shared.write_json(output/'protocol.json',dict(protocol=PROTOCOL,evaluation_protocol=evaluator.PROTOCOL,
        stage='development; p8 and rho=1 chosen after earlier results',
        selection='pooled inner full F1; outer full evaluation; frozen responses',
        tie_break=['F1 descending','precision descending','FP ascending','config index'],
        configurations={m:[asdict(c) for c in cs] for m,cs in all_grids.items()},
        ablations='p8_no_L and p8_no_G inherit each fold complete p8 parameters; no retuning',
        search_budget='G/GM_p8/G_rescaled:57; G_expanded:114; L/Mean:171',
        rescaled_comparison='canonical G >= 2^(1/8)*tau; never clip thresholds',
        old_counts_source=str(old_root)))
    indices = [0]
    if name == 'metst_sammlv':
        indices += [i for i,s in enumerate(subjects) if str(s).lstrip('0') == '37']
    checked = check_mapping(context,core,indices)
    probe_records = []
    for si in indices:
        for method in ('G','L','Mean','GM_p8'):
            winner,_ = shared.choose(tables[method][1],si)
            c = all_grids[method][winner]
            r = detailed_decode(context,core,si,c)
            if any(not np.array_equal(r[k+'_counts'], tables[method][j][winner,si])
                   for j,k in enumerate(('raw','full'))):
                raise RuntimeError('old selected counts failed exact replay')
            probe_records.append(r)
        for method in ('p8_no_L','p8_no_G'):
            probe_records.append(detailed_decode(context,core,si,replace(c,method=method)))
    if name == 'metst_sammlv':
        case = previous.probe_case(context,True)
        if case['raw'] != [3,16,0] or case['full'] != [3,15,0]:
            raise RuntimeError('subject037 known regression failed')
        shared.write_json(output/'subject037_regression.json',case)
    with gzip.open(output/'probe_event_records.jsonl.gz','wt',encoding='utf8') as f:
        for r in probe_records:
            f.write(json.dumps(shared.serializable(r),allow_nan=False)+'\n')
    shared.write_json(output/'probe.json',dict(passed=True,subjects=[subjects[i] for i in indices],
        mapped_comparisons=checked,selected_replays=len(indices)*4,event_ledger_reconstructs_counts=True))
    print(name,'THRESHOLD_EVENT_PROBE = PASS',flush=True)
    if mode == 'probe':
        return
    # The only new search: 57 mapped thresholds. Old grids are never rerun.
    core.capture = False
    tables['G_rescaled'] = phase.search_counts(context,core,all_grids['G_rescaled'],output/'search_counts_G_rescaled.npz')
    tables['G_expanded'] = tuple(np.concatenate((a,b),axis=0)
                                 for a,b in zip(tables['G'],tables['G_rescaled']))
    for m in ('G','L','Mean','GM_p8','G_expanded'):
        with (output/('search_counts_%s.npz'%m)).open('wb') as f:
            np.savez_compressed(f,raw=tables[m][0],full=tables[m][1],done=np.ones(len(subjects),bool),
                subjects=np.asarray(subjects,dtype=str),configs=np.asarray(json.dumps([asdict(c) for c in all_grids[m]],sort_keys=True)))
    chosen, selected_rows, summaries, per_subject = {},[],[],[]
    for m,cs in all_grids.items():
        chosen[m] = []
        for si,s in enumerate(subjects):
            winner,pooled = shared.choose(tables[m][1],si)
            chosen[m].append(cs[winner])
            selected_rows.append(dict(subject=s,**asdict(cs[winner]),inner_F1=shared.metrics(pooled[winner])['F1']))
    counts = {m:[] for m in (*all_grids,'p8_no_L','p8_no_G')}
    with gzip.open(output/'event_records.jsonl.gz','wt',encoding='utf8') as ledger, \
         gzip.open(output/'event_differences.jsonl.gz','wt',encoding='utf8') as differences:
        for si,s in enumerate(subjects):
            records = {}
            configs = {m:cs[si] for m,cs in chosen.items()}
            configs.update({m:replace(configs['GM_p8'],method=m) for m in ('p8_no_L','p8_no_G')})
            for m,c in configs.items():
                r = detailed_decode(context,core,si,c)
                if m in tables:
                    for j,stage in enumerate(('raw','full')):
                        if not np.array_equal(r[stage+'_counts'],tables[m][j][c.config_id,si]):
                            raise RuntimeError('selected prediction replay mismatch')
                records[m] = r
                counts[m].append(r['full_counts'])
                ledger.write(json.dumps(shared.serializable(r),allow_nan=False)+'\n')
                for stage in ('raw','full'):
                    per_subject.append(dict(setting=name,subject=s,method=m,evaluation=stage,
                                            **shared.metrics(r[stage+'_counts'])))
            for m in configs:
                if m != 'GM_p8':
                    differences.write(json.dumps(event_delta(records['GM_p8'],records[m]))+'\n')
            print('event replay',name,si+1,'/',len(subjects),flush=True)
    counts = {m:np.asarray(v,dtype=np.int64) for m,v in counts.items()}
    for m,a in counts.items():
        summaries.append(dict(setting=name,method=m,selection='locked_p8' if m.startswith('p8_no') else 'inner_full',
                              configs=0 if m.startswith('p8_no') else len(all_grids[m]),**shared.metrics(a.sum(0))))
    native_path = old_root/(name+'_audit')/'native_per_subject_counts.csv'
    with native_path.open() as f:
        native = [r for r in csv.DictReader(f) if r['evaluation']=='full']
    if [r['subject'] for r in native] != list(subjects):
        raise RuntimeError('native subject mismatch')
    native_counts = np.asarray([[int(r[k]) for k in ('TP','FP','FN')] for r in native])
    for si,a in enumerate(native_counts):
        previous.validate(a,a,core.expected_gt[si])
    summaries.append(dict(setting=name,method='Native',selection='none',configs=0,**shared.metrics(native_counts.sum(0))))
    for filename,rows in [('summary_full.csv',summaries),('selected_configs.csv',selected_rows),
                           ('per_subject_counts.csv',per_subject),('paired_comparisons.csv',paired(counts))]:
        shared.write_csv(output/filename,rows)
    f1={r['method']:r['F1'] for r in summaries}
    shared.write_json(output/'decision.json',dict(stage='development',
        delta_vs_expanded_G=f1['GM_p8']-f1['G_expanded'],delta_vs_L=f1['GM_p8']-f1['L'],
        delta_vs_no_L=f1['GM_p8']-f1['p8_no_L'],delta_vs_no_G=f1['GM_p8']-f1['p8_no_G'],
        exceeds_expanded_G_L_Mean=all(f1['GM_p8']>f1[m] for m in ('G_expanded','L','Mean'))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('probe','full'),required=True)
    p.add_argument('--setting',choices=('sammlv','casme3'),required=True)
    p.add_argument('--reuse',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    ora,helper=shared.load_helper()
    name=shared.SETTINGS[args.setting]
    old=args.reuse.resolve(); output=args.output.resolve()
    if phase.overlaps(old,output):
        raise RuntimeError('new output overlaps old results')
    for spec in ora.SPECS.values():
        for key in ('dump','cache','evidence','run_evidence','results','locked_results'):
            if key in spec and phase.overlaps(output,Path(spec[key]).resolve()):
                raise RuntimeError('output overlaps sealed input')
    manifest=json.loads((old/'run_manifest.json').read_text())['identity']
    completion=json.loads((old/'completion.json').read_text())
    if manifest['protocol']!=evaluator.PROTOCOL or manifest['mode']!='full' or not completion['completed']:
        raise RuntimeError('reuse requires a completed corrected full run')
    for filename,digest in manifest['code_sha256'].items():
        if ora.sha256(Path(__file__).with_name(filename))!=digest:
            raise RuntimeError('old source hash mismatch: '+filename)
    runtime=dict(python=sys.version,numpy=np.__version__,pandas=getattr(sys.modules.get('pandas'),'__version__',None))
    if manifest['runtime']!=runtime:
        raise RuntimeError('reuse runtime differs; restore the previous Colab runtime before continuing')
    ora.verify_sealed_inputs(ora.SPECS[name])
    # Compare complete input inventory as well as contents.
    spec=ora.SPECS[name]
    inputs=[spec['source'],spec['core'],spec['evidence']/'evidence_bundle_sha256.txt',spec['run_evidence']/'run_manifest.json']
    inputs+=sorted(x for x in spec['dump'].rglob('*') if x.is_file())
    input_hashes={str(x):ora.sha256(x) for x in inputs}
    if input_hashes!=manifest['inputs'][name]:
        raise RuntimeError('sealed input hashes/inventory changed')
    reuse_paths=[old/'run_manifest.json',old/'completion.json',old/(name+'_audit')/'matching_protocol.json',
                 old/(name+'_audit')/'native_per_subject_counts.csv']
    reuse_paths += [old/name/('search_counts_%s.npz'%m) for m in phase.METHODS]
    identity=dict(protocol=PROTOCOL,evaluation=evaluator.PROTOCOL,mode=args.mode,setting=name,runtime=runtime,
        inputs=input_hashes,reuse={str(x):ora.sha256(x) for x in reuse_paths},
        code={x.name:ora.sha256(x) for x in Path(__file__).parent.glob('*.py')})
    if output.exists():
        if not args.resume or json.loads((output/'run_manifest.json').read_text())['identity']!=identity:
            raise RuntimeError('resume requires identical code/input/protocol/runtime/reuse identity')
        if (output/'completion.json').exists():
            print('Already completed:',output); return
    else:
        if args.resume: raise RuntimeError('resume directory missing')
        output.mkdir(parents=True)
        shared.write_json(output/'run_manifest.json',dict(identity=identity))
    print('OUTPUT =',output,flush=True)
    context=ora.metst_context(spec)  # Historical native gate before installing corrected evaluator.
    with evaluator.install(context[2]) as info:
        import inspect
        old_info=json.loads((old/(name+'_audit')/'matching_protocol.json').read_text())
        info['metric_source_sha256']=ora.sha256(Path(inspect.getsourcefile(context[2])))
        for key in ('protocol','metric_source_sha256','legacy_check_box_sha256'):
            if old_info[key]!=info[key]: raise RuntimeError('evaluator identity changed: '+key)
        shared.write_json(output/'matching_protocol.json',info)
        run_setting(context,old,name,output/name,args.mode)
    shared.write_json(output/'completion.json',dict(completed=True,protocol=PROTOCOL,mode=args.mode,setting=name))
    print('P8_THRESHOLD_%s = PASS'%args.mode.upper(),flush=True)


if __name__=='__main__': main()
