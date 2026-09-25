"""Alignment-tolerance continuation. Frozen responses; joint inner selection only."""
import argparse
import ast
from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import run_joint_gm as joint

phase, shared, prior, evaluator = joint.phase, joint.shared, joint.prior, joint.evaluator
PROTOCOL = 'gm_alignment_eta_v1'
ETAS = (.5, .25, .75)  # Original definition first for exact selection ties.


@dataclass(frozen=True)
class Config(joint.Config):
    eta: float = .5


def grids():
    result = {m: [Config(**asdict(c)) for c in cs] for m, cs in joint.grids().items()}
    for method, source in (('GM_eta', 'GM_joint'), ('L_eta', 'L'), ('Mean_eta', 'Mean')):
        result[method] = [replace(Config(**asdict(c), eta=eta), method=method, config_id=i)
            for i, (eta, c) in enumerate((eta,c) for eta in ETAS for c in joint.grids()[source])]
    return result


def alignment_method(base_class):
    """Change only the verified tolerance assignment in the sealed evidence method."""
    source = textwrap.dedent(inspect.getsource(base_class.evidence))
    tree = ast.parse(source)
    expected = ast.dump(ast.parse('max(1, round(0.5 * self.k))', mode='eval').body)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'tolerance' for t in n.targets)]
    if len(assignments) != 1 or ast.dump(assignments[0].value) != expected:
        raise RuntimeError('sealed tolerance expression differs; refuse to reconstruct feature definition')
    assignments[0].value = ast.parse('max(1, round(self._experiment_eta * self.k))', mode='eval').body
    ast.fix_missing_locations(tree)
    namespace = dict(base_class.evidence.__globals__)
    exec(compile(tree, '<sealed evidence: tolerance coefficient only>', 'exec'), namespace)
    return namespace['evidence'], hashlib.sha256(source.encode()).hexdigest()


def score_keep(values, c):
    aliases = {'GM_eta':'GM_joint','L_eta':'L','Mean_eta':'Mean',
               'GM_eta_no_L':'GM_no_L','GM_eta_no_G':'GM_no_G'}
    return joint.score_keep(values, replace(c, method=aliases.get(c.method,c.method)))


class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        self.cache = {}
        self.baselines = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evidence(self, scale, radius, eta=.5):
        pair = scale, radius
        if pair not in self.baselines:
            original_peaks, original = self.base.evidence(scale, radius)
            self.base._experiment_eta = .5
            try:
                peaks, values = self.owner.adapted(self.base, scale, radius)
            finally:
                del self.base._experiment_eta
            if not np.array_equal(peaks,original_peaks) or not np.array_equal(values,original):
                raise RuntimeError('eta=.5 must exactly replay sealed evidence')
            self.baselines[pair] = (np.array(peaks,copy=True), np.array(values,copy=True))
        key = scale, radius, eta
        if key not in self.cache:
            if eta == .5:
                peaks, values = self.baselines[pair]
            else:
                self.base._experiment_eta = eta
                try:
                    peaks, values = self.owner.adapted(self.base,scale,radius)
                finally:
                    del self.base._experiment_eta
            phase.score(values,'G')
            old_peaks, old_values = self.baselines[pair]
            if not np.array_equal(peaks,old_peaks) or not np.array_equal(values[:,0],old_values[:,0]):
                raise RuntimeError('alignment change modified candidate pool or G')
            self.cache[key] = (np.array(peaks,copy=True),np.array(values,copy=True))
        return self.cache[key]

    def selected_peaks(self,c):
        peaks, values = self.evidence(c.reference_scale,c.local_radius,c.eta)
        scores, keep = score_keep(values,c)
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2],k=self.key[0],eta=c.eta,
                tolerance=max(1,round(c.eta*self.key[0])),peaks=peaks.tolist(),
                G=values[:,0].tolist(),L=values[:,1].tolist(),score=scores.tolist(),retained=keep.tolist()))
        return peaks[keep]


class FusionCore(prior.FusionCore):
    def __init__(self,base):
        super().__init__(base)
        self.adapted,self.evidence_sha256 = alignment_method(base.GLSDFeatures)

    def GLSDFeatures(self,response,k):
        array=np.ascontiguousarray(np.asarray(response,dtype=float))
        key=(int(k),array.shape,hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key]=FeatureView(self.base.GLSDFeatures(response,k),key,self)
        return self.cache[key]


def load_reuse(root,name,subjects,gts):
    return {m:joint.load_counts(root/name/('search_counts_'+m+'.npz'),cs,subjects,gts)
            for m,cs in joint.grids().items()}


def sensitivity(context,core,output):
    """All videos/structures, no new labels and no candidate-level TP inference."""
    rows=[]
    for si,subject in enumerate(context[6]):
        record=context[4][si]
        for vi,response in enumerate(record['result_all']):
            feature=core.GLSDFeatures(response,record['k_p'])
            for a in phase.SCALES:
                for rho in phase.RADII:
                    peaks,baseline=feature.evidence(a,rho,.5)
                    for eta in ETAS:
                        _,values=feature.evidence(a,rho,eta)
                        diff=values[:,1]-baseline[:,1]
                        rows.append(dict(subject=subject,video_index=vi,video_name=str(record['videos'][vi]),
                            a0=a,rho=rho,eta=eta,k=record['k_p'],tolerance=max(1,round(eta*record['k_p'])),
                            candidates=len(peaks),L_changed=int(np.count_nonzero(diff)),
                            L_increased=int(np.count_nonzero(diff>0)),L_decreased=int(np.count_nonzero(diff<0)),
                            max_abs_change=float(np.max(np.abs(diff))) if len(diff) else 0))
        print('alignment feature check',si+1,'/',len(context[6]),flush=True)
    shared.write_csv(output/'alignment_sensitivity.csv',rows)
    summary={str(eta):dict(candidate_structure_instances=sum(r['candidates'] for r in rows if r['eta']==eta),
        changed=sum(r['L_changed'] for r in rows if r['eta']==eta),
        increased=sum(r['L_increased'] for r in rows if r['eta']==eta),
        decreased=sum(r['L_decreased'] for r in rows if r['eta']==eta),
        tolerances=sorted({(int(r['k']),int(r['tolerance'])) for r in rows if r['eta']==eta})) for eta in ETAS}
    shared.write_json(output/'alignment_sensitivity.json',summary)
    print('ALIGNMENT_SENSITIVITY =',summary,flush=True)
    return summary


def probe(context,core,tables,output):
    records=[]
    si=0
    for m in ('L','Mean','GM_joint'):
        w,_=shared.choose(tables[m][1],si)
        c=Config(**asdict(joint.grids()[m][w]))
        r=prior.detailed_decode(context,core,si,c)
        for j,stage in enumerate(('raw','full')):
            if not np.array_equal(r[stage+'_counts'],tables[m][j][w,si]):
                raise RuntimeError('original eta selected replay failed: '+m)
        records.append(r)
    for eta in (.25,.75):
        for p in ('1','8','inf'):
            for m in ('GM_eta','GM_eta_no_L','GM_eta_no_G','L_eta'):
                c=Config(m,0,2.,2.,.55,p,eta)
                records.append(prior.detailed_decode(context,core,si,c))
    with gzip.open(output/'probe_event_records.jsonl.gz','wt',encoding='utf8') as f:
        for r in records:f.write(json.dumps(shared.serializable(r),allow_nan=False)+'\n')
    shared.write_json(output/'probe.json',dict(passed=True,original_replays=3,new_replays=24,
        sealed_evidence_sha256=core.evidence_sha256,feature_replay='all accessed eta=.5 structures exact'))
    print('ETA_SCORER_EVENT_PROBE = PASS',flush=True)


def assemble(context,core,tables,output):
    gs=grids();pieces={m:[tables[old]] for m,old in [('GM_eta','GM_joint'),('L_eta','L'),('Mean_eta','Mean')]}
    for eta in (.25,.75):
        parts=[]
        for p in joint.POWERS:
            cs=[replace(c,config_id=i) for i,c in enumerate(c for c in gs['GM_eta'] if c.eta==eta and c.p==p)]
            core.capture=False
            part=phase.search_counts(context,core,cs,output/f'new_counts_eta{eta}_p{p}.npz')
            parts.append(part)
            if p=='1':pieces['Mean_eta'].append(part)
        pieces['GM_eta'].append(tuple(np.concatenate([part[j] for part in parts]) for j in (0,1)))
        cs=[replace(c,config_id=i) for i,c in enumerate(c for c in gs['L_eta'] if c.eta==eta)]
        core.capture=False
        pieces['L_eta'].append(phase.search_counts(context,core,cs,output/f'new_counts_eta{eta}_L.npz'))
    for m,ps in pieces.items():tables[m]=tuple(np.concatenate([part[j] for part in ps]) for j in (0,1))
    for m,cs in gs.items():
        for a in tables[m]:
            if a.shape!=(len(cs),len(context[6]),3) or np.any(a<0) or not np.all(a[:,:,0]+a[:,:,2]==core.expected_gt):
                raise RuntimeError('assembled counts inconsistent: '+m)
        joint.save_counts(output/('search_counts_'+m+'.npz'),cs,tables[m],context[6])
    return tables


def validate_record(record):
    gt={g['gt_id'] for g in record['ground_truth']}
    if len(gt)!=len(record['ground_truth']):raise RuntimeError('duplicate GT ledger IDs')
    for stage in ('raw','full'):
        ids=[e['matched_gt_id'] for e in record['events'] if e[stage+'_TP']]
        tp=len(ids);fp=sum(e[stage+'_FP'] for e in record['events'])
        if len(set(ids))!=tp or not set(ids)<=gt or [tp,fp,len(gt)-tp]!=record[stage+'_counts']:
            raise RuntimeError('event ledger count/GT inconsistency')


def run_setting(context,root,name,output,mode):
    output.mkdir(exist_ok=True)
    core=FusionCore(context[1]);core.expected_gt=prior.previous.gt_counts(context)
    tables=load_reuse(root,name,context[6],core.expected_gt);gs=grids()
    shared.write_json(output/'protocol.json',dict(protocol=PROTOCOL,evaluation=evaluator.PROTOCOL,
        stage='development after prior outer inspection',etas=ETAS,p=joint.POWERS,
        parameters='joint inner full selection of eta,p,a0,rho,tau; L selects eta,a0,rho,tau independently',
        unchanged='sealed G/L except tolerance coefficient; scales/median/missing policy/geometry unchanged',
        tie_break=['F1 descending','precision descending','FP ascending','config index'],
        eta_order=ETAS,p_order=joint.POWERS,configs={m:[asdict(c) for c in cs] for m,cs in gs.items()},
        new_configurations=2052,reused_GM_configurations=855,budgets_equal=False,
        evaluator='existing context aggregate adapter and unchanged spotting-owned video evaluator',
        ablation='GM_eta_no_L/GM_eta_no_G inherit complete parameters; no reselection'))
    if not (output/'alignment_sensitivity.json').exists():sensitivity(context,core,output)
    if not (output/'probe.json').exists():probe(context,core,tables,output)
    if mode=='probe':return
    tables=assemble(context,core,tables,output)
    old_cache=joint.load_event_cache(root/name/'event_records.jsonl.gz')
    replay_dir=output/'selected_events';replay_dir.mkdir(exist_ok=True)
    selected=[];chosen={}
    for m,cs in gs.items():
        chosen[m]=[]
        for si,s in enumerate(context[6]):
            w,pool=shared.choose(tables[m][1],si);c=cs[w];chosen[m].append(c)
            selected.append(dict(subject=s,**asdict(c),inner_F1=shared.metrics(pool[w])['F1'],selection='inner_full'))
    counts={m:[] for m in (*gs,'GM_eta_no_L','GM_eta_no_G')};per=[];deltas=[]
    with gzip.open(output/'event_records.jsonl.gz','wt',encoding='utf8') as ledger, \
         gzip.open(output/'event_differences.jsonl.gz','wt',encoding='utf8') as diff:
        for si,s in enumerate(context[6]):
            configs={m:cs[si] for m,cs in chosen.items()}
            configs.update({m:replace(configs['GM_eta'],method=m) for m in ('GM_eta_no_L','GM_eta_no_G')})
            records={}
            for m,c in configs.items():
                path=replay_dir/f'{si:03d}_{m}.json.gz'
                if path.exists():
                    with gzip.open(path,'rt') as f:r=json.load(f)
                    if r['subject']!=s or r['config']!=asdict(c):raise RuntimeError('event checkpoint identity mismatch')
                else:
                    # Reuse only unchanged eta=.5 scorers. New eta always records its actual L values.
                    aliases={'GM_eta':'GM_joint','L_eta':'L','Mean_eta':'Mean','GM_eta_no_L':'GM_no_L','GM_eta_no_G':'GM_no_G'}
                    old_c=replace(c,method=aliases.get(m,m))
                    key=s,joint.event_signature(old_c)
                    if c.eta==.5 and key in old_cache:
                        r=joint.relabel_record(old_cache[key],old_c)
                        r.update(method=m,config=asdict(c),record_origin='verified_eta_half_saved_record')
                    else:
                        r=prior.detailed_decode(context,core,si,c);r['record_origin']='official_selected_replay'
                    validate_record(r)
                    tmp=path.with_suffix('.tmp')
                    with gzip.open(tmp,'wt') as f:json.dump(shared.serializable(r),f,allow_nan=False)
                    tmp.replace(path)
                validate_record(r)
                if m in tables:
                    for j,stage in enumerate(('raw','full')):
                        if r[stage+'_counts']!=tables[m][j][c.config_id,si].tolist():raise RuntimeError('selected replay mismatch '+m)
                records[m]=r;counts[m].append(r['full_counts'])
                ledger.write(json.dumps(shared.serializable(r),allow_nan=False)+'\n')
                if m.startswith('GM_eta_no'):
                    selected.append(dict(subject=s,**asdict(c),inner_F1='',selection='locked_GM_eta'))
                for stage in ('raw','full'):per.append(dict(subject=s,method=m,evaluation=stage,**shared.metrics(r[stage+'_counts'])))
            for m in configs:
                if m!='GM_eta':
                    d=prior.event_delta(records['GM_eta'],records[m]);deltas.append(d);diff.write(json.dumps(d)+'\n')
            print('eta selected events',si+1,'/',len(context[6]),flush=True)
    counts={m:np.asarray(v,dtype=np.int64) for m,v in counts.items()}
    rows=[dict(setting=name,method=m,selection='locked_GM_eta' if m.startswith('GM_eta_no') else 'inner_full',
               configs=len(gs[m]) if m in gs else 0,**shared.metrics(a.sum(0))) for m,a in counts.items()]
    with (root/name/'summary_full.csv').open() as f:native=next(r for r in csv.DictReader(f) if r['method']=='Native')
    rows.append({k:native[k] for k in rows[0]})
    # prior.paired uses GM_p8 as its compared key; alias only for computation.
    comparisons=prior.paired({('GM_p8' if m=='GM_eta' else m):a for m,a in counts.items()})
    for r in comparisons:r['compared']='GM_eta'
    for fn,data in [('summary_full.csv',rows),('selected_configs.csv',selected),('per_subject_counts.csv',per),('paired_comparisons.csv',comparisons)]:shared.write_csv(output/fn,data)
    f1={r['method']:float(r['F1']) for r in rows}
    controls=['G','L','L_eta','Mean','Mean_eta','G_union']+['G_rescaled_p'+p for p in joint.POWERS]
    shared.write_json(output/'decision.json',dict(stage='development',table2_SOTA='not evaluated in this experiment',
        exceeds_independent_controls=all(f1['GM_eta']>f1[m] for m in controls),
        exceeds_locked_ablations=all(f1['GM_eta']>f1[m] for m in ('GM_eta_no_L','GM_eta_no_G')),
        delta_F1={m:f1['GM_eta']-v for m,v in f1.items() if m!='GM_eta'},
        module_events={m:dict(added_GT=sum(len(d['added_gt_ids']) for d in deltas if d['reference']==m),
            lost_GT=sum(len(d['lost_gt_ids']) for d in deltas if d['reference']==m),
            delta_FP=sum(d['delta_FP'] for d in deltas if d['reference']==m)) for m in ('GM_eta_no_L','GM_eta_no_G')},
        selected_eta_counts=dict(Counter(str(c.eta) for c in chosen['GM_eta'])),
        uncertainty='conditional bootstrap; no retuning or exploration correction'))


def verify_reuse(ora,root,name,runtime,inputs):
    completion=json.loads((root/'completion.json').read_text())
    old=json.loads((root/'run_manifest.json').read_text())['identity']
    if not completion['completed'] or completion['mode']!='full' or completion['setting']!=name:
        raise RuntimeError('requires completed joint full run for requested setting')
    if old['protocol']!=joint.PROTOCOL or old['evaluation']!=evaluator.PROTOCOL or old['setting']!=name:
        raise RuntimeError('joint source protocol mismatch')
    if old['runtime']!=runtime or old['inputs']!=inputs:raise RuntimeError('runtime or frozen inputs changed')
    for n,h in old['code'].items():
        if ora.sha256(Path(__file__).with_name(n))!=h:raise RuntimeError('inherited code mismatch: '+n)
    return old


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('probe','full'),default='probe');p.add_argument('--setting',choices=('sammlv','casme3'),required=True)
    p.add_argument('--reuse',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();ora,_=shared.load_helper();name=shared.SETTINGS[args.setting]
    root=args.reuse.resolve();output=args.output.resolve()
    protected=[root]
    for spec in ora.SPECS.values():protected += [Path(spec[k]).resolve() for k in ('dump','cache','evidence','run_evidence','results','locked_results') if k in spec]
    if any(phase.overlaps(output,x) for x in protected):raise RuntimeError('output overlaps protected data')
    spec=ora.SPECS[name];ora.verify_sealed_inputs(spec)
    paths=[spec['source'],spec['core'],spec['evidence']/'evidence_bundle_sha256.txt',spec['run_evidence']/'run_manifest.json']
    paths+=sorted(x for x in spec['dump'].rglob('*') if x.is_file())
    inputs={str(x):ora.sha256(x) for x in paths}
    runtime=dict(python=sys.version,numpy=np.__version__,pandas=getattr(sys.modules.get('pandas'),'__version__',None))
    verify_reuse(ora,root,name,runtime,inputs)
    reused=[root/'run_manifest.json',root/'completion.json',root/'matching_protocol.json',root/name/'event_records.jsonl.gz',root/name/'summary_full.csv']
    reused += [root/name/('search_counts_'+m+'.npz') for m in joint.grids()]
    identity=dict(protocol=PROTOCOL,evaluation=evaluator.PROTOCOL,mode=args.mode,setting=name,runtime=runtime,inputs=inputs,
        reuse={str(x):ora.sha256(x) for x in reused},code={x.name:ora.sha256(x) for x in Path(__file__).parent.glob('*.py')})
    if output.exists():
        if not args.resume or not (output/'run_manifest.json').exists() or json.loads((output/'run_manifest.json').read_text())['identity']!=identity:
            raise RuntimeError('resume requires identical code/input/protocol/runtime/reuse')
        if (output/'completion.json').exists():print('Already completed:',output,flush=True);return
    else:
        if args.resume:raise RuntimeError('resume output missing')
        output.mkdir(parents=True);shared.write_json(output/'run_manifest.json',dict(identity=identity))
    print('OUTPUT =',output,flush=True)
    context=ora.metst_context(spec)
    with evaluator.install(context[2]) as info:
        info['metric_source_sha256']=ora.sha256(Path(inspect.getsourcefile(context[2])))
        previous=json.loads((root/'matching_protocol.json').read_text())
        for k in ('protocol','metric_source_sha256','legacy_check_box_sha256'):
            if info[k]!=previous[k]:raise RuntimeError('evaluator mismatch '+k)
        shared.write_json(output/'matching_protocol.json',info)
        run_setting(context,root,name,output/name,args.mode)
    shared.write_json(output/'completion.json',dict(completed=True,protocol=PROTOCOL,mode=args.mode,setting=name))
    print('ETA_GM_%s = PASS'%args.mode.upper(),flush=True)

if __name__=='__main__':main()
