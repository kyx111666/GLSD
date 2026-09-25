"""Finite rho=3 extension; immutable old counts, exact full replay and locked interventions."""
import argparse
from dataclasses import asdict,replace
import gzip
import inspect
import json
import os
from pathlib import Path
import sys
import numpy as np
import run_grayzone as prior

shared=prior.shared
PROTOCOL='grayzone_rho3_extension_v1_full_loso'


def grids():
    result={}
    for m,cs in prior.grids().items():
        added=[] if m=='G' else [replace(c,local_radius=3.) for c in cs if c.local_radius==1.]
        result[m]=[replace(c,config_id=i) for i,c in enumerate(cs+added)]
    return result


def load_old(path,subjects=None):
    with np.load(path,allow_pickle=False) as z:
        configs=json.loads(z['configs'].item());raw=z['raw'].copy();full=z['full'].copy()
        done=z['done'].copy();saved_subjects=z['subjects'].tolist()
    expected=[asdict(c) for cs in prior.grids().values() for c in cs]
    if configs!=expected or (subjects is not None and saved_subjects!=list(subjects)):
        raise RuntimeError('old search configuration/subject identity mismatch')
    shape=(len(expected),len(saved_subjects),3)
    if raw.shape!=shape or full.shape!=shape or done.shape!=shape[:2] or done.dtype!=bool or not done.all():
        raise RuntimeError('old search incomplete or malformed')
    if len(saved_subjects)!=len(set(saved_subjects)):raise RuntimeError('duplicate old subjects')
    gt=full[0,:,0]+full[0,:,2]
    for x in (raw,full):
        if not np.issubdtype(x.dtype,np.integer) or np.any(x<0) or not np.all(x[:,:,0]+x[:,:,2]==gt):
            raise RuntimeError('invalid old counts')
    return raw,full,saved_subjects,gt


def seed_checkpoint(old_tables,subjects,output):
    """Per-method old prefix preserved; prior.search decodes only uncompleted new rows."""
    all_grids=grids();old_grids=prior.grids()
    configs=[c for cs in all_grids.values() for c in cs]
    shape=(len(configs),len(subjects),3)
    raw,full=np.zeros(shape,np.int64),np.zeros(shape,np.int64)
    done=np.zeros(shape[:2],bool);oi=ni=0
    for m,cs in old_grids.items():
        n=len(cs);raw[ni:ni+n]=old_tables[0][oi:oi+n];full[ni:ni+n]=old_tables[1][oi:oi+n]
        done[ni:ni+n]=True
        oi+=n;ni+=len(all_grids[m])
    path=output/'search_counts.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if z['configs'].item()!=json.dumps([asdict(c) for c in configs],sort_keys=True) or z['subjects'].tolist()!=list(subjects):
                raise RuntimeError('extended checkpoint identity mismatch')
            if z['done'].shape!=done.shape or not z['done'][done].all():raise RuntimeError('reused rows not complete')
            if not np.array_equal(z['raw'][done],raw[done]) or not np.array_equal(z['full'][done],full[done]):
                raise RuntimeError('reused old rows changed')
    else:
        prior.save_npz(path,raw=raw,full=full,done=done,subjects=np.asarray(subjects),
            configs=np.asarray(json.dumps([asdict(c) for c in configs],sort_keys=True)))
    return configs


def records_from(path):
    with gzip.open(path,'rt',encoding='utf8') as f:rows=[json.loads(x) for x in f]
    result={(r['subject'],r['method']):r for r in rows}
    if len(result)!=len(rows):raise RuntimeError('duplicate old event records')
    return result


def replay_old(context,core,old_records,old_tables,output):
    output.mkdir(exist_ok=True)
    offset=0;methods={}
    for m,cs in prior.grids().items():
        methods[m]=(cs,old_tables[0][offset:offset+len(cs)],old_tables[1][offset:offset+len(cs)])
        offset+=len(cs)
    for si,subject in enumerate(context[6]):
        core.cache.clear()
        for m,(cs,raw,full) in methods.items():
            wi,_=shared.choose(full,si);c=cs[wi];saved=old_records[subject,m]
            if saved['config']!=asdict(c):raise RuntimeError('old selected config differs from training-only choice')
            r=prior.selected_record(context,core,si,c,output)
            for key in ('events','ground_truth','raw_counts','full_counts'):
                if r[key]!=saved[key]:raise RuntimeError('old replay changed '+subject+' '+m+' '+key)
            for j,stage in enumerate(('raw','full')):
                if not np.array_equal(r[stage+'_counts'],(raw,full)[j][wi,si]):raise RuntimeError('old tensor/ledger mismatch')
            if len(r['candidates'])!=len(saved['candidates']):raise RuntimeError('old trace video count mismatch')
            for a,b in zip(r['candidates'],saved['candidates']):
                for key in ('peaks','retained','video_id','response_sha256'):
                    if a[key]!=b[key]:raise RuntimeError('old trace changed')
                for key in ('G','L'):np.testing.assert_allclose(a[key],b[key],rtol=1e-13,atol=1e-14)
        print('old selected full replay',si+1,'/',len(context[6]),flush=True)


def bootstrap_pair(a,b,label,reference):
    draws=np.random.default_rng(100).integers(0,len(a),(10000,len(a)))
    def sampled(x):
        n=x[draws].sum(1);den=2*n[:,0]+n[:,1]+n[:,2]
        return np.divide(2*n[:,0],den,out=np.zeros(len(den)),where=den>0)
    d=sampled(a)-sampled(b)
    return dict(compared=label,reference=reference,delta_F1=shared.metrics(a.sum(0))['F1']-shared.metrics(b.sum(0))['F1'],
        CI_low=float(np.quantile(d,.025)),CI_high=float(np.quantile(d,.975)),resamples=10000,seed=100,
        interpretation='conditional on selected predictions; no refitting or development correction')


def intervention_report(context,core,old_records,old_tables,new_tables,output):
    out=output/'locked_intervention';out.mkdir(exist_ok=True)
    counts={};selected=[];per=[];changes=[]
    old_grids=prior.grids();old_cs=[c for cs in old_grids.values() for c in cs]
    def signature(c):return tuple(v for k,v in asdict(c).items() if k!='config_id')
    old_lookup={signature(c):i for i,c in enumerate(old_cs)}
    new_gray=grids()['Gray'];new_lookup={signature(c):i for i,c in enumerate(new_gray)}
    replay=out/'selected_events';replay.mkdir(exist_ok=True)
    with gzip.open(out/'event_records.jsonl.gz','wt',encoding='utf8') as ledger,gzip.open(out/'event_differences.jsonl.gz','wt',encoding='utf8') as delta_file:
        for si,s in enumerate(context[6]):
            core.cache.clear();base=prior.Config(**old_records[s,'Gray']['config'])
            recs={'Gray_original':old_records[s,'Gray'],'L_original':old_records[s,'L']}
            for m in ('Gray','L'):
                with gzip.open(output/'selected_events'/('selected_%03d_%s.json.gz'%(si,m)),'rt') as f:
                    recs[m+'_extended']=json.load(f)
            configs={'Gray_locked_rho2':replace(base,local_radius=2.),'Gray_locked_rho3':replace(base,local_radius=3.),
                     'rho3_high_only':replace(base,local_radius=3.,method='Gray_high_only'),
                     'rho3_no_support':replace(base,local_radius=3.,method='Gray_no_support')}
            for label,c in configs.items():
                folder=replay/label;folder.mkdir(exist_ok=True)
                r=prior.selected_record(context,core,si,c,folder)
                if label in ('Gray_locked_rho2','Gray_locked_rho3'):
                    ix=old_lookup[signature(c)] if c.local_radius==2 else new_lookup[signature(c)]
                    table=old_tables if c.local_radius==2 else new_tables['Gray']
                    for j,stage in enumerate(('raw','full')):
                        if not np.array_equal(r[stage+'_counts'],table[j][ix,si]):raise RuntimeError('locked intervention/tensor mismatch')
                recs[label]=r
            for label,r in recs.items():
                row=dict(r,method=label,record_origin='verified_old_ledger' if label.endswith('original') else 'full_replay')
                ledger.write(json.dumps(shared.serializable(row),allow_nan=False)+'\n')
                counts.setdefault(label,[]).append(r['full_counts'])
                selected.append(dict(subject=s,reported_method=label,**r['config']))
                for stage in ('raw','full'):per.append(dict(subject=s,method=label,evaluation=stage,**shared.metrics(r[stage+'_counts'])))
            for a,b in [('Gray_locked_rho3','Gray_original'),('Gray_locked_rho3','Gray_locked_rho2'),
                ('Gray_locked_rho3','rho3_high_only'),('Gray_locked_rho3','rho3_no_support'),
                ('Gray_locked_rho3','L_original'),('Gray_locked_rho3','L_extended'),
                ('Gray_extended','Gray_original'),('Gray_extended','L_original')]:
                d=prior.event_delta(dict(recs[a],method=a),dict(recs[b],method=b));changes.append(d)
                delta_file.write(json.dumps(shared.serializable(d),allow_nan=False)+'\n')
            print('locked rho intervention replay',si+1,'/',len(context[6]),flush=True)
    counts={m:np.asarray(a,dtype=np.int64) for m,a in counts.items()}
    summaries=[dict(method=m,**shared.metrics(a.sum(0))) for m,a in counts.items()]
    pairs=[bootstrap_pair(counts[a],counts[b],a,b) for a,b in dict.fromkeys((d['compared'],d['reference']) for d in changes)]
    for filename,rows in [('summary_full.csv',summaries),('selected_configs.csv',selected),('per_subject_counts.csv',per),('paired_comparisons.csv',pairs)]:
        shared.write_csv(out/filename,rows)
    totals=[]
    for a,b in dict.fromkeys((d['compared'],d['reference']) for d in changes):
        ds=[d for d in changes if d['compared']==a and d['reference']==b]
        totals.append(dict(compared=a,reference=b,added_GT=sum(len(d['added_gt_ids']) for d in ds),
            lost_GT=sum(len(d['lost_gt_ids']) for d in ds),added_FP_events=sum(len(d['added_fp_events']) for d in ds),
            removed_FP_events=sum(len(d['removed_fp_events']) for d in ds),delta_FP=sum(d['delta_FP'] for d in ds)))
    shared.write_csv(out/'event_change_summary.csv',totals)
    f={r['method']:r['F1'] for r in summaries}
    decision=json.loads((output/'decision.json').read_text())
    decision.update(protocol=PROTOCOL,prior_result_controls={m:f['Gray_extended']-f[m] for m in ('Gray_original','L_original')},
        locked_rho3_delta_vs_original=f['Gray_locked_rho3']-f['Gray_original'],
        locked_rho3_beats_old_and_extended_L=all(f['Gray_locked_rho3']>f[m] for m in ('L_original','L_extended')),
        extended_Gray_beats_old_L=f['Gray_extended']>f['L_original'],
        intervention_scope='original outer-selected Gray parameters with rho replaced; no retuning; development intervention')
    shared.write_json(output/'decision.json',decision)


def run(context,old,output):
    output.mkdir(exist_ok=True);name='metst_'+SETTING_FROM_CONTEXT(context)
    old_tables=load_old(old/name/'search_counts.npz',context[6])
    old_records=records_from(old/name/'event_records.jsonl.gz')
    core=prior.FusionCore(context[1]);core.expected_gt=prior.previous.gt_counts(context)
    if list(old_tables[3])!=list(core.expected_gt):raise RuntimeError('old/current GT mismatch')
    proof=prior.verify_candidate_only_config(context[0])
    shared.write_json(output/'protocol.json',dict(protocol=PROTOCOL,evaluation=prior.evaluator.PROTOCOL,
        selection='training pooled full F1, precision, fewer FP, fixed index; outer excluded',
        old_radii=list(prior.RADII),new_radii=[.5,1.,2.,3.],a0=list(prior.SCALES),
        unchanged='Gray pairs, eta, branch choice, G/L thresholds, gamma=.5, median, missing=0, support scales, peak spacing, event geometry, recognition',
        configurations={m:len(cs) for m,cs in grids().items()},new_configurations=1743,reused_configurations=5460,
        equal_budget=False,cache=proof,P8_control='same Gray package comparator; not the separate user P8 run',
        locked='original selected Gray -> rho2/rho3, keep other parameters; rho3 high-only/no-support controls',
        development='rho3 proposed after diagnostic; not independent confirmation or new backbone CV'))
    replay_old(context,core,old_records,old_tables,output/'old_replays')
    configs=seed_checkpoint(old_tables,context[6],output)
    raw,full=prior.search(context,core,configs,output)
    tables={};offset=0
    for m,cs in grids().items():
        tables[m]=raw[offset:offset+len(cs)],full[offset:offset+len(cs)];offset+=len(cs)
    with np.load(old/name/'native_counts.npz',allow_pickle=False) as z:
        if z['subjects'].tolist()!=list(context[6]):raise RuntimeError('native subject identity mismatch')
        native=z['raw'].copy(),z['full'].copy()
    for si,gt in enumerate(core.expected_gt):prior.previous.validate(native[0][si],native[1][si],gt)
    prior.save_npz(output/'native_counts.npz',raw=native[0],full=native[1],subjects=np.asarray(context[6]))
    prior.report(context,core,grids(),tables,native,output)
    intervention_report(context,core,old_records,old_tables,tables,output)


def SETTING_FROM_CONTEXT(context):
    # Only used after hard population verification in main; no silent guessing.
    return {29:'sammlv',94:'casme3'}[len(context[6])]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--setting',choices=('sammlv','casme3'),required=True)
    p.add_argument('--reuse',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();old=args.reuse.resolve();out=args.output.resolve()
    allowed=Path('/content/drive/MyDrive/GLSD_RHO3_EXTENSION')
    if allowed not in out.parents or prior.phase.overlaps(old,out):raise RuntimeError('output must be new child under GLSD_RHO3_EXTENSION')
    ora,_=shared.load_helper();name='metst_'+args.setting;spec=ora.SPECS[name]
    comp=json.loads((old/'completion.json').read_text());old_id=json.loads((old/'run_manifest.json').read_text())['identity']
    if not comp['completed'] or comp['mode']!='full' or comp['protocol']!=prior.PROTOCOL or old_id['setting']!=name or old_id['evaluation']!=prior.evaluator.PROTOCOL:
        raise RuntimeError('reuse requires the completed original Gray full run for this setting')
    runtime=dict(python=sys.version,numpy=np.__version__,pandas=getattr(sys.modules.get('pandas'),'__version__',None))
    if runtime!=old_id['runtime']:raise RuntimeError('runtime differs from reused counts; restore original Colab environment')
    for n,h in old_id['code_sha256'].items():
        if ora.sha256(Path(__file__).with_name(n))!=h:raise RuntimeError('inherited code changed: '+n)
    ora.verify_sealed_inputs(spec)
    for n,h in old_id['input_sha256'].items():
        if not Path(n).is_file() or ora.sha256(Path(n))!=h:raise RuntimeError('frozen input changed: '+n)
    checkout=Path(os.environ.get('ME_TST_ROOT','/content/ME-TST'))
    current={str(x) for x in spec['dump'].rglob('*') if x.is_file()}
    current|={str(x) for x in checkout.rglob('*.py') if '.git' not in x.parts}
    previous={n for n in old_id['input_sha256'] if Path(n).is_relative_to(spec['dump']) or Path(n).is_relative_to(checkout)}
    if current!=previous:raise RuntimeError('frozen dump/checkout inventory changed')
    files=[old/'completion.json',old/'run_manifest.json',old/'matching_protocol.json']+[old/name/n for n in ('search_counts.npz','event_records.jsonl.gz','native_counts.npz')]
    identity=dict(protocol=PROTOCOL,setting=name,evaluation=prior.evaluator.PROTOCOL,runtime=runtime,
        reuse={str(x):ora.sha256(x) for x in files},inputs=old_id['input_sha256'],
        code={x.name:ora.sha256(x) for x in Path(__file__).parent.glob('*.py')})
    if out.exists():
        if not args.resume or json.loads((out/'run_manifest.json').read_text())['identity']!=identity:raise RuntimeError('resume identity mismatch')
    else:
        if args.resume:raise RuntimeError('resume path missing')
        out.mkdir(parents=True);shared.write_json(out/'run_manifest.json',dict(identity=identity))
    context=ora.metst_context(spec)
    if len(set(context[6]))!=spec['subjects'] or ora.context_video_and_gt_counts('metst',context)!=(spec['videos'],spec['gt']):raise RuntimeError('population mismatch')
    with prior.evaluator.install(context[2]) as info:
        previous_info=json.loads((old/'matching_protocol.json').read_text())
        for key in ('protocol','legacy_check_box_sha256'):
            if info[key]!=previous_info[key]:raise RuntimeError('matching identity changed')
        info['source_hashes']={str(Path(inspect.getsourcefile(x))):ora.sha256(Path(inspect.getsourcefile(x)))
            for x in (context[2],context[3],context[3].spotting.__globals__['MeanAveragePrecision2d'])}
        if info['source_hashes']!=previous_info['source_hashes']:raise RuntimeError('actual evaluator binding/source changed')
        shared.write_json(out/'matching_protocol.json',info)
        if (out/'completion.json').exists():print('Already complete:',out,flush=True);return
        run(context,old,out/name)
    shared.write_json(out/'completion.json',dict(completed=True,setting=name,protocol=PROTOCOL,
        subjects=spec['subjects'],videos=spec['videos'],GT=spec['gt'],new_configurations=1743,reused_configurations=5460))
    print('RHO3_EXTENSION = PASS',out,flush=True)


if __name__=='__main__':main()
