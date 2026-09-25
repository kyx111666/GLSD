"""Frozen reliable-local-union development experiment; prior ETA results are read-only."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import run_eta_gm as eta
import reliable_features as features

prior, phase, shared, evaluator, joint = eta.prior, eta.phase, eta.shared, eta.evaluator, eta.joint
Config, FusionCore, grids, LOCKED, PROTOCOL = (features.Config, features.FusionCore,
    features.grids, features.LOCKED, features.PROTOCOL)
ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    shared.write_json(tmp,data)
    tmp.replace(path)


def all_grids():
    gs = grids()
    for m,cs in joint.grids().items():
        if m == 'G_union' or m.startswith('G_rescaled_p'):
            gs[m] = [Config(**asdict(c)) for c in cs]
    return gs


def verify_reuse(root,name,runtime,inputs):
    old = json.loads((root/'run_manifest.json').read_text())['identity']
    done = json.loads((root/'completion.json').read_text())
    if (old['protocol'] != eta.PROTOCOL or old['evaluation'] != evaluator.PROTOCOL
        or old['setting'] != name or old['mode'] != 'full'
        or not done['completed'] or done['mode'] != 'full' or done['setting'] != name
        or done['protocol'] != eta.PROTOCOL):
        raise RuntimeError('requires completed ETA full run for this dataset')
    if old['runtime'] != runtime: raise RuntimeError('reuse runtime changed; restore original Colab runtime')
    if old['inputs'] != inputs: raise RuntimeError('frozen input inventory/hash changed')
    inherited = json.loads((ROOT/'inherited_manifest.json').read_text())
    if set(old['code']) != set(inherited):raise RuntimeError('inherited source inventory mismatch')
    for n,h in old['code'].items():
        if inherited.get(n) != h or sha(ROOT/n) != h:
            raise RuntimeError('inherited code identity mismatch: '+n)
    return old


def load_reuse(root,name,subjects,gts):
    gs = eta.grids()
    wanted = ('G','L','Mean','GM_eta','G_union',*('G_rescaled_p'+p for p in joint.POWERS))
    return {m:joint.load_counts(root/name/('search_counts_'+m+'.npz'),gs[m],subjects,gts) for m in wanted}


def reuse_mapping(tables):
    """Only identical scorers: old G/L/Mean, strong G, and eta=.5,p=inf,beta=1 NoR."""
    old = eta.grids()
    mapping = {}
    for m in tables:
        if m == 'GM_eta':
            cs = [c for c in old[m] if c.eta == .5 and c.p == 'inf']
            target = 'NoR'
        else:
            cs, target = old[m], m
        for c in cs:
            key = target,c.reference_scale,c.local_radius,c.threshold,1.,c.p if target.startswith('G_rescaled_p') else 'inf'
            mapping[key] = tuple(a[c.config_id] for a in tables[m])
            if target == 'L': mapping[('L_old',*key[1:])] = mapping[key]
    return mapping


def config_key(c):
    return c.method,c.reference_scale,c.local_radius,c.threshold,c.beta,c.p


def validate_arrays(arrays,configs,subjects,gts):
    for a in arrays:
        if (a.shape != (len(configs),len(subjects),3) or not np.issubdtype(a.dtype,np.integer)
            or np.any(a<0) or not np.all(a[:,:,0]+a[:,:,2] == np.asarray(gts))):
            raise RuntimeError('invalid tensor shape/type/count/GT conservation')


def assemble(context,core,old,output):
    known = reuse_mapping(old)
    tables, budget = {}, []
    for m,cs in all_grids().items():
        missing = [c for c in cs if config_key(c) not in known]
        if missing:
            core.capture = False
            new_cs = [replace(c,config_id=i) for i,c in enumerate(missing)]
            new = phase.search_counts(context,core,new_cs,output/('new_counts_'+m+'.npz'))
            validate_arrays(new,new_cs,context[6],core.expected_gt)
            for i,c in enumerate(missing): known[config_key(c)] = tuple(a[i] for a in new)
        arrays = tuple(np.stack([known[config_key(c)][j] for c in cs]) for j in (0,1))
        validate_arrays(arrays,cs,context[6],core.expected_gt)
        joint.save_counts(output/('search_counts_'+m+'.npz'),cs,arrays,context[6])
        tables[m] = arrays
        budget.append(dict(method=m,configurations=len(cs),reused=len(cs)-len(missing),
            new_configurations=len(missing),new_search_subject_decodes=len(missing)*len(context[6]),
            dynamic_candidate_set_deduplication=False))
    shared.write_csv(output/'search_budget.csv',budget)
    return tables


def detailed_decode(context,core,si,c):
    r = prior.detailed_decode(context,core,si,c)
    by = {}
    for trace in r['candidates']:
        for i,p in enumerate(trace['peaks']):
            key = trace['video_id'],p
            by[key] = (trace,i)
    for e in r['events']:
        # The unchanged official candidate representation stores original peak in column 6.
        p = e['interval'][6]
        key = e['video_id'],p
        if key not in by: raise RuntimeError('prediction peak not present in candidate trace')
        t,i = by[key]
        if not t['retained'][i]: raise RuntimeError('prediction from rejected candidate')
        e.update(candidate_id=e['video_id']+'/peak_'+str(int(p)),peak=int(p))
        for f in ('G','L','R','L_rel','score','G_accept','reliable_L_accept','both_accept','full_accept_source'):
            e[f] = t[f][i]
    eta.validate_record(r)
    return r


def candidate_rows(r):
    events = {}
    for e in r['events']: events.setdefault((e['video_id'],e['peak']),[]).append(e)
    for t in r['candidates']:
        for i,p in enumerate(t['peaks']):
            es = events.get((t['video_id'],p),[])
            yield dict(subject=r['subject'],method=r['method'],config=r['config'],
                candidate_id=t['video_id']+'/peak_'+str(p),video_id=t['video_id'],video_name=t['video_name'],
                peak=p,response_sha256=t['response_sha256'],k=t['k'],
                effective_widths=t['effective_widths'],reference_width=t['reference_width'],
                tolerance=t['tolerance'],B_empty=t['B_empty'],
                **{f:t[f][i] for f in ('G','L','R','L_rel','score','retained','G_accept','reliable_L_accept',
                    'both_accept','full_accept_source','matched_peaks','distances','per_scale_local')},
                events=es,decoded=bool(es),
                raw_label=('TP' if any(e['raw_TP'] for e in es) else 'FP') if es else None,
                full_label=('TP' if any(e['full_TP'] for e in es) else 'FP' if any(e['full_FP'] for e in es) else 'filtered') if es else None,
                matched_gt_ids=[e['matched_gt_id'] for e in es],prediction_intervals=[e['interval'] for e in es])


def event_delta(full,other):
    d = prior.event_delta(full,other)
    def fps(r):
        return Counter(json.dumps([e['video_id'],e['interval']],sort_keys=True)
                       for e in r['events'] if e['full_FP'])
    a,b = fps(full),fps(other)
    d['added_FP_events'] = [json.loads(x) for x in sorted((a-b).elements())]
    d['removed_FP_events'] = [json.loads(x) for x in sorted((b-a).elements())]
    if len(d['added_FP_events'])-len(d['removed_FP_events']) != d['delta_FP']:
        raise RuntimeError('FP identity delta mismatch')
    return d


def feature_audit(context,core,output):
    rows=[]
    for si,s in enumerate(context[6]):
        record=context[4][si]
        for vi,response in enumerate(record['result_all']):
            f=core.GLSDFeatures(response,record['k_p'])
            for a in phase.SCALES:
                for rho in phase.RADII:
                    d=f.reliable_evidence(a,rho)
                    nonref=np.delete(d['distances'],d['reference_index'],axis=1)
                    rows.append(dict(subject=s,video_id=f'{s}/video_{vi}',response_sha256=f.key[2],
                        a0=a,rho=rho,k=f.k,reference_width=d['reference_width'],
                        physical_scales=len(d['widths']),candidates=len(d['peaks']),
                        B_empty=d['B_empty'],missing_nonreference=int(np.any(nonref<0,axis=1).sum()),
                        R_zero=int((d['R']==0).sum()),R_sum=float(d['R'].sum())))
    shared.write_csv(output/'feature_audit.csv',rows)
    n=sum(r['candidates'] for r in rows)
    deg=sum(r['candidates'] for r in rows if r['B_empty'])
    report=dict(passed=True,videos=sum(len(r['result_all']) for r in context[4]),
        video_structure_instances=len(rows),candidate_structure_instances=n,B_empty_candidates=deg,
        B_empty_candidate_ratio=deg/n if n else 0.,
        B_empty_video_structure_ratio=sum(r['B_empty'] for r in rows)/len(rows),
        missing_nonreference=sum(r['missing_nonreference'] for r in rows),
        checks=['sealed mean replay at all T','exact per-scale median equals L',
                'reference physical scale located by object identity','same G and peaks across rho'],
        scope='features only; no F1 or model selection')
    atomic_json(output/'feature_audit.json',report)
    return report


def probe(context,core,old,output):
    records=[]
    # Fixed subjects and fixed new configurations; probe F1 is never used for selection.
    indices=sorted(set([0]+[i for i,s in enumerate(context[6]) if str(s).lstrip('0')=='37']))
    replay_count=0
    for si in indices:
        for m in ('G','L','Mean'):
            w,_=shared.choose(old[m][1],si)
            oc=eta.grids()[m][w]
            c=Config(m,w,oc.reference_scale,oc.local_radius,oc.threshold)
            r=detailed_decode(context,core,si,c)
            for j,stage in enumerate(('raw','full')):
                if r[stage+'_counts']!=old[m][j][w,si].tolist(): raise RuntimeError('old replay failed: '+m)
            records.append(r);replay_count+=1
        # Prove max endpoint reuse with actual official decoding.
        oc=next(c for c in eta.grids()['GM_eta'] if c.eta==.5 and c.p=='inf' and c.reference_scale==2 and c.local_radius==2 and c.threshold==.55)
        c=Config('NoR',0,2.,2.,.55)
        r=detailed_decode(context,core,si,c)
        for j,stage in enumerate(('raw','full')):
            if r[stage+'_counts']!=old['GM_eta'][j][oc.config_id,si].tolist():raise RuntimeError('NoR/max endpoint replay failed')
        records.append(r);replay_count+=1
        for beta in features.BETAS:
            for m in ('Full','L_rel','NoR',*LOCKED):
                records.append(detailed_decode(context,core,si,Config(m,0,2.,2.,.55,beta)))
    regression=None
    if any(str(s).lstrip('0')=='37' for s in context[6]) and len(context[6])==29:
        si=next(i for i,s in enumerate(context[6]) if str(s).lstrip('0')=='37')
        regression=detailed_decode(context,core,si,Config('L',0,1.,2.,.05))
        if regression['raw_counts']!=[3,16,0] or regression['full_counts']!=[3,15,0]:
            raise RuntimeError('subject037 known regression failed')
        records.append(regression)
    with gzip.open(output/'probe_event_records.jsonl.gz','wt',encoding='utf8') as f:
        for r in records:f.write(json.dumps(shared.serializable(r),allow_nan=False)+'\n')
    report=dict(passed=True,subjects=[context[6][i] for i in indices],historical_replays=replay_count,
        actual_subject_decodes=len(records),event_counts_verified=True,
        subject037_regression='PASS' if regression else 'not applicable',
        scope='implementation probe only; no method selection or new experiment F1')
    atomic_json(output/'probe.json',report)
    print('RELIABLE_LOCAL_UNION_PROBE = PASS',flush=True)
    return report


def selected_results(context,core,tables,output):
    gs=all_grids(); chosen={}; selected=[]
    for m,cs in gs.items():
        chosen[m]=[]
        for si,s in enumerate(context[6]):
            w,pool=shared.choose(tables[m][1],si); c=cs[w];chosen[m].append(c)
            selected.append(dict(subject=s,**asdict(c),inner_F1=shared.metrics(pool[w])['F1'],selection='inner_full'))
    counts={m:[] for m in (*gs,*LOCKED)};per=[];deltas=[];deg=[]
    checkpoint=output/'selected_events';checkpoint.mkdir(exist_ok=True)
    with gzip.open(output/'event_records.jsonl.gz','wt') as ledger, gzip.open(output/'candidate_records.jsonl.gz','wt') as candidates, gzip.open(output/'event_differences.jsonl.gz','wt') as differences:
        for si,s in enumerate(context[6]):
            configs={m:cs[si] for m,cs in chosen.items()}
            configs.update({m:replace(configs['Full'],method=m) for m in LOCKED})
            records={}
            for m,c in configs.items():
                path=checkpoint/f'{si:03d}_{m}.json.gz'
                if path.exists():
                    with gzip.open(path,'rt') as f:r=json.load(f)
                    if r['subject']!=s or r['config']!=asdict(c):raise RuntimeError('event checkpoint identity mismatch')
                else:
                    r=detailed_decode(context,core,si,c)
                    tmp=path.with_suffix('.tmp')
                    with gzip.open(tmp,'wt') as f:json.dump(shared.serializable(r),f,allow_nan=False)
                    tmp.replace(path)
                eta.validate_record(r)
                if m in tables:
                    for j,stage in enumerate(('raw','full')):
                        if r[stage+'_counts']!=tables[m][j][c.config_id,si].tolist():raise RuntimeError('selected tensor replay mismatch: '+m)
                records[m]=r;counts[m].append(r['full_counts'])
                ledger.write(json.dumps(shared.serializable(r),allow_nan=False)+'\n')
                n=empty=0
                for row in candidate_rows(r):
                    candidates.write(json.dumps(shared.serializable(row),allow_nan=False)+'\n');n+=1;empty+=row['B_empty']
                deg.append(dict(subject=s,method=m,candidates=n,B_empty_candidates=empty,B_empty_ratio=empty/n if n else 0.))
                if m in LOCKED:selected.append(dict(subject=s,**asdict(c),inner_F1='',selection='locked_Full'))
                for stage in ('raw','full'):per.append(dict(subject=s,method=m,evaluation=stage,**shared.metrics(r[stage+'_counts'])))
            for m in configs:
                if m!='Full':
                    d=event_delta(records['Full'],records[m]);deltas.append(d);differences.write(json.dumps(d)+'\n')
            print('selected event audit',si+1,'/',len(context[6]),flush=True)
    counts={m:np.asarray(a,dtype=np.int64) for m,a in counts.items()}
    rows=[dict(method=m,selection='locked_Full' if m in LOCKED else 'inner_full',configs=len(gs[m]) if m in gs else 0,**shared.metrics(a.sum(0))) for m,a in counts.items()]
    paired=prior.paired({('GM_p8' if m=='Full' else m):a for m,a in counts.items()})
    for r in paired:r['compared']='Full'
    for fn,data in [('summary_full.csv',rows),('selected_configs.csv',selected),('per_subject_counts.csv',per),('paired_comparisons.csv',paired),('selected_degeneracy.csv',deg)]:shared.write_csv(output/fn,data)
    f1={r['method']:r['F1'] for r in rows}
    controls=['G','L','L_rel','G_union']+[m for m in gs if m.startswith('G_rescaled_p')]
    atomic_json(output/'decision.json',dict(stage='development; not independent confirmation',table2='not evaluated',
        exceeds_independent_single_branches=all(f1['Full']>f1[m] for m in controls),
        exceeds_locked_ablations=all(f1['Full']>f1[m] for m in LOCKED),
        delta_F1={m:f1['Full']-v for m,v in f1.items() if m!='Full'},
        event_contributions={m:dict(added_GT=sum(len(d['added_gt_ids']) for d in deltas if d['reference']==m),
            lost_GT=sum(len(d['lost_gt_ids']) for d in deltas if d['reference']==m),
            added_FP=sum(len(d['added_FP_events']) for d in deltas if d['reference']==m),
            removed_FP=sum(len(d['removed_FP_events']) for d in deltas if d['reference']==m),
            delta_FP=sum(d['delta_FP'] for d in deltas if d['reference']==m)) for m in counts if m!='Full'},
        uncertainty='paired subject bootstrap conditional on selected predictions; no retuning/exploration correction',
        stop_rule='do not expand grids or change R after outer inspection; CAS remains prespecified development evaluation'))


def check_resume(output,identity,resume):
    if output.exists():
        if not resume or not (output/'run_manifest.json').is_file() or json.loads((output/'run_manifest.json').read_text())['identity']!=identity:
            raise RuntimeError('resume requires identical code/input/runtime/reuse/evaluation/mode identity')
        if (output/'completion.json').is_file():
            done=json.loads((output/'completion.json').read_text())
            if not done.get('completed') or any(done.get(k)!=identity.get(k) for k in ('protocol','mode','setting')):
                raise RuntimeError('completion identity mismatch')
            return True
        return False
    if resume:raise RuntimeError('resume output missing')
    output.mkdir(parents=True)
    atomic_json(output/'run_manifest.json',dict(identity=identity))
    return False


def runtime_identity():
    import pandas, scipy, sklearn
    return dict(python=sys.version,numpy=np.__version__,pandas=pandas.__version__), dict(scipy=scipy.__version__,sklearn=sklearn.__version__)


def decoder_identity(context):
    classes=[context[2],context[3].spotting.__globals__['MeanAveragePrecision2d']]
    paths={Path(inspect.getsourcefile(context[3])).resolve()}
    for cls in classes:
        directory=Path(inspect.getsourcefile(cls)).resolve().parent
        paths.update(directory.rglob('*.py'))
    files={str(p):sha(p) for p in sorted(paths)}
    return dict(files=files,classes=[dict(module=c.__module__,name=c.__name__,add_signature=str(inspect.signature(c.add)),value_signature=str(inspect.signature(c.value))) for c in classes])


def verify_runtime_sources(context):
    expected=json.loads((ROOT/'official_runtime_reference.json').read_text())
    official_root=Path(inspect.getsourcefile(context[3])).resolve().parent
    for name,digest in expected.items():
        path=official_root/name
        if not path.is_file() or sha(path)!=digest:raise RuntimeError('official decode/evaluation source differs from verified reference: '+str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('check','probe','full'),default='probe')
    p.add_argument('--setting',choices=('sammlv','casme3'),required=True)
    p.add_argument('--reuse',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    ora,_=shared.load_helper();name=shared.SETTINGS[args.setting]
    root,output=args.reuse.resolve(),args.output.resolve()
    protected=[root,ROOT]
    for spec in ora.SPECS.values():protected.extend(Path(spec[k]).resolve() for k in ('dump','cache','evidence','run_evidence','results','locked_results') if k in spec)
    if any(phase.overlaps(output,x) for x in protected):raise RuntimeError('output overlaps protected code/input/old results')
    spec=ora.SPECS[name];ora.verify_sealed_inputs(spec)
    paths=[spec['source'],spec['core'],spec['evidence']/'evidence_bundle_sha256.txt',spec['run_evidence']/'run_manifest.json']
    paths+=sorted(x for x in spec['dump'].rglob('*') if x.is_file())
    inputs={str(x):sha(x) for x in paths}
    runtime,extra=runtime_identity();verify_reuse(root,name,runtime,inputs)
    # Same historical native gate as the verified ETA call chain; no search/backbone rerun.
    context=ora.metst_context(spec);verify_runtime_sources(context)
    chain=decoder_identity(context)
    old_info=json.loads((root/'matching_protocol.json').read_text())
    with evaluator.install(context[2]) as info:
        info['metric_source_sha256']=sha(Path(inspect.getsourcefile(context[2])))
        for k in ('protocol','metric_source_sha256','legacy_check_box_sha256'):
            if info[k]!=old_info[k]:raise RuntimeError('aggregate evaluator identity mismatch: '+k)
        reused=[root/'run_manifest.json',root/'completion.json',root/'matching_protocol.json']
        reused += [root/name/('search_counts_'+m+'.npz') for m in ('G','L','Mean','GM_eta','G_union',*('G_rescaled_p'+p for p in joint.POWERS))]
        identity=dict(protocol=PROTOCOL,evaluation=evaluator.PROTOCOL,mode=args.mode,setting=name,
            runtime=runtime,extra_runtime=extra,inputs=inputs,decoder=chain,
            reuse={str(x):sha(x) for x in reused},code={x.name:sha(x) for x in ROOT.glob('*.py')},
            references={x:sha(ROOT/x) for x in ('inherited_manifest.json','official_runtime_reference.json')})
        if check_resume(output,identity,args.resume):print('Already completed:',output,flush=True);return
        atomic_json(output/'matching_protocol.json',dict(**info,decoder=chain))
        out=output/name;out.mkdir(exist_ok=True)
        core=FusionCore(context[1]);core.expected_gt=prior.previous.gt_counts(context)
        old=load_reuse(root,name,context[6],core.expected_gt)
        atomic_json(out/'protocol.json',dict(protocol=PROTOCOL,stage='development',
            formula='R=0 if B empty or missing; else exp(-mean(d_nonref/k)); L_rel=R*L; S=max(G,beta*L_rel)',
            eta=.5,beta=features.BETAS,U=[float(t) for t in features.U],
            configs={m:[asdict(c) for c in cs] for m,cs in all_grids().items()},
            tie_break=['pooled inner full F1 descending','precision descending','FP ascending','fixed config index'],
            grid_order='a0,rho,beta,tau; U starts with T then sorted new rational thresholds',
            budget_equal=False,upstream_training_isolation='not established',
            ablations='Full_no_L, Full_no_G (scoring branch only), Full_no_R inherit Full parameters'))
        atomic_json(out/'compatibility.json',dict(passed=True,subjects=len(context[6]),gt=sum(core.expected_gt),
            historical_tensor_methods=list(old),historical_runtime_matches=True,official_source_reference_matches=True))
        if args.mode!='check':
            if not (out/'feature_audit.json').exists():feature_audit(context,core,out)
            if not (out/'probe.json').exists():probe(context,core,old,out)
            if args.mode=='full':selected_results(context,core,assemble(context,core,old,out),out)
        atomic_json(output/'completion.json',dict(completed=True,protocol=PROTOCOL,mode=args.mode,setting=name))
        print('RELIABLE_LOCAL_UNION_'+args.mode.upper()+' = PASS',flush=True)

if __name__=='__main__':main()
