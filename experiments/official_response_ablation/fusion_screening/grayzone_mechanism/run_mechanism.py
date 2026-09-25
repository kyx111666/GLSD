"""Export all candidates and scale evidence using current frozen official sources.
A diagnostic run, not a new fusion parameter sweep or a spotting-result table.
"""
import argparse
import contextlib
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import numpy as np
import run_grayzone as prior
from mechanism_features import RADII,candidate_rows
from analyze_export import analyze

PROTOCOL='glsd_scale_conditional_diagnostic_v1'


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def geometry_decode(context,core,si,a0):
    runner,_,metric,official,records,_,subjects,_=context
    # Resolve official inclusive-coordinate IoU before add is temporarily observed.
    compute_iou=metric.add.__globals__['compute_match_table'].__globals__['compute_iou']
    config=prior.Config('G',0,a0,1.,0.)
    core.trace=[];core.capture=True
    with prior.ledger.capture_decoder_inputs(metric,official) as captured,contextlib.redirect_stdout(io.StringIO()):
        raw,predictions,_,_,matches,video_metric=runner.decode_glsd_subject(records,si,config,core,metric,official,False)
    prior.previous.validate(raw,None,core.expected_gt[si])
    registered=[c[id(video_metric)][1] for c in captured if id(video_metric) in c]
    if len(registered)!=1:raise RuntimeError('cannot identify actual video geometry inputs')
    registered=registered[0]
    if len(predictions)!=len(core.trace) or len(registered)!=len(predictions):raise RuntimeError('video alignment')
    result=[]
    for vi,(p,(registered_p,gt),t) in enumerate(zip(predictions,registered,core.trace)):
        p=np.asarray(p);gt=np.asarray(gt)
        if not np.array_equal(p,registered_p):raise RuntimeError('prediction order mismatch')
        peaks=t['peaks']
        if len(p)!=len(peaks) or (len(p) and p[:,6].tolist()!=peaks):raise RuntimeError('all-candidate geometry changes/removes anchors')
        expected_hash=hashlib.sha256(np.ascontiguousarray(np.asarray(records[si]['result_all'][vi],float)).tobytes()).hexdigest()
        if t['response_sha256']!=expected_hash:raise RuntimeError('response order mismatch')
        # No candidate TP labels inferred from these pairwise geometric overlaps.
        overlap=compute_iou(p[:,:4],gt[:,:4]) if len(p) and len(gt) else np.zeros((len(p),len(gt)))
        vids='%s/video_%d'%(subjects[si],vi)
        result.append(dict(video_id=vids,video_name=str(records[si]['videos'][vi]),response_sha256=expected_hash,
            predictions=p.tolist(),registered_gt=gt.tolist(),peaks=peaks,
            eligible_gt_ids=[[vids+'/gt_'+str(i) for i in np.flatnonzero(row>=.5)] for row in overlap],
            best_iou=overlap.max(1).tolist() if len(gt) else [0.]*len(p),
            covered_GT=int(np.any(overlap>=.5,axis=0).sum()),GT=len(gt)))
    return result


def verify_prior_replay(context,core,si,old_records):
    subject=context[6][si]
    for m in ('L','Gray'):
        saved=old_records[subject,m];c=prior.Config(**saved['config'])
        actual=prior.ledger.detailed_decode(context,core,si,c)
        for key in ('raw_counts','full_counts','events','ground_truth'):
            if actual[key]!=saved[key]:raise RuntimeError('prior selected replay changed: '+subject+' '+m+' '+key)
        for a,b in zip(actual['candidates'],saved['candidates']):
            for key in ('video_id','peaks','retained','response_sha256'):
                if a[key]!=b[key]:raise RuntimeError('prior candidate replay changed')
            for key in ('G','L'):np.testing.assert_allclose(a[key],b[key],rtol=1e-13,atol=1e-14)


def export_subject(context,core,si,old_records):
    record=context[4][si];subject=context[6][si]
    core.cache.clear();verify_prior_replay(context,core,si,old_records)
    features=[];geometry=[];coverage=[]
    for a0 in (1.,1.5,2.):
        geo=geometry_decode(context,core,si,a0)
        geometry.append(dict(a0=a0,videos=geo))
        coverage.append(dict(subject=subject,a0=a0,candidates=sum(len(v['peaks']) for v in geo),
            GT=sum(v['GT'] for v in geo),geometry_covered_GT=sum(v['covered_GT'] for v in geo)))
        for vi,response in enumerate(record['result_all']):
            feature=core.GLSDFeatures(response,record['k_p']).base
            for rho in RADII:
                row=candidate_rows(feature,a0,rho)
                if row['peaks']!=geo[vi]['peaks']:raise RuntimeError('rho changes candidates')
                row.update(video_id=geo[vi]['video_id'],video_name=geo[vi]['video_name'],
                    geometry_eligible=[bool(x) for x in geo[vi]['eligible_gt_ids']],
                    response_sha256=geo[vi]['response_sha256'])
                features.append(row)
    return dict(subject=subject,old_L_Gray_exact_replay=True,features=features,geometry=geometry,coverage=coverage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting',choices=('sammlv','casme3'),required=True)
    p.add_argument('--reuse',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();old=args.reuse.resolve();output=args.output.resolve()
    allowed=Path('/content/drive/MyDrive/GLSD_MECHANISM_DIAGNOSTIC')
    if output==allowed or allowed not in output.parents or prior.phase.overlaps(old,output):raise RuntimeError('new output must be under GLSD_MECHANISM_DIAGNOSTIC')
    ora,_=prior.shared.load_helper();name='metst_'+args.setting;spec=ora.SPECS[name]
    complete=json.loads((old/'completion.json').read_text());old_id=json.loads((old/'run_manifest.json').read_text())['identity']
    if not complete['completed'] or complete['mode']!='full' or complete['protocol']!=prior.PROTOCOL or old_id['setting']!=name:
        raise RuntimeError('reuse must be this dataset completed Gray full run')
    for n,h in old_id['code_sha256'].items():
        if ora.sha256(Path(__file__).with_name(n))!=h:raise RuntimeError('inherited source mismatch: '+n)
    ora.verify_sealed_inputs(spec)
    # Revalidate every frozen dump, official code and evidence input in the completed run.
    for namepath,digest in old_id['input_sha256'].items():
        path=Path(namepath)
        if not path.is_file() or ora.sha256(path)!=digest:raise RuntimeError('frozen input changed/missing: '+namepath)
    current_dumps={str(p) for p in spec['dump'].rglob('*') if p.is_file()}
    previous_dumps={n for n in old_id['input_sha256'] if Path(n).is_relative_to(spec['dump'])}
    if current_dumps!=previous_dumps:raise RuntimeError('dump inventory changed')
    ledger_path=old/name/'event_records.jsonl.gz'
    with gzip.open(ledger_path,'rt') as f:rr=[json.loads(line) for line in f]
    old_records={(r['subject'],r['method']):r for r in rr}
    paths=[old/'completion.json',old/'run_manifest.json',old/'matching_protocol.json',ledger_path]
    identity=dict(protocol=PROTOCOL,setting=name,old_run={str(x):ora.sha256(x) for x in paths},
        code={x.name:ora.sha256(x) for x in Path(__file__).parent.glob('*.py')},
        runtime=dict(python=sys.version,numpy=np.__version__))
    if output.exists():
        if not args.resume or json.loads((output/'run_manifest.json').read_text())!=identity:raise RuntimeError('resume identity mismatch')
    else:
        if args.resume:raise RuntimeError('missing resume directory')
        output.mkdir(parents=True);write_json(output/'run_manifest.json',identity)
    parts=output/'subjects';parts.mkdir(exist_ok=True)
    context=ora.metst_context(spec)
    if ora.context_video_and_gt_counts('metst',context)!=(spec['videos'],spec['gt']):raise RuntimeError('population mismatch')
    core=prior.FusionCore(context[1]);core.expected_gt=prior.previous.gt_counts(context)
    with prior.evaluator.install(context[2]) as info:
        previous_info=json.loads((old/'matching_protocol.json').read_text())
        for key in ('protocol','legacy_check_box_sha256'):
            if info[key]!=previous_info[key]:raise RuntimeError('evaluation identity mismatch')
        write_json(output/'matching_protocol.json',info)
        for si,subject in enumerate(context[6]):
            path=parts/('%03d.json.gz'%si)
            if path.exists():
                with gzip.open(path,'rt') as f:saved=json.load(f)
                if saved['subject']!=subject or not saved['old_L_Gray_exact_replay']:raise RuntimeError('subject checkpoint mismatch')
            else:
                value=export_subject(context,core,si,old_records)
                tmp=path.with_suffix('.tmp')
                with gzip.open(tmp,'wt',encoding='utf8') as f:json.dump(value,f,allow_nan=False)
                tmp.replace(path)
            print('scale export',args.setting,si+1,'/',len(context[6]),subject,flush=True)
    # Clear evidence caches before the numeric diagnostic stage.
    core.cache.clear();del old_records,rr
    print('Export complete; fitting training-only conditional diagnostics',flush=True)
    analyze(output)
    write_json(output/'completion.json',dict(completed=True,protocol=PROTOCOL,setting=name,
        subjects=spec['subjects'],videos=spec['videos'],GT=spec['gt'],purpose='mechanism diagnosis, not full F1'))
    print('MECHANISM_DIAGNOSTIC = PASS',flush=True)


if __name__=='__main__':main()
