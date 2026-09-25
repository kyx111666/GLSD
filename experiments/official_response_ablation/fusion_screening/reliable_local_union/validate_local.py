"""Local delivery probe only: audit saved artifacts and real SAM responses, never full search.

Local runtime differs from Colab, so this does NOT authorize cross-runtime count
reuse for a full experiment. The production entry retains its strict runtime gate.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import warnings
import zipfile
import numpy as np
import pandas as pd
import run_reliable_local_union as run


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def unpack(path,dest):
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            p=(dest/n).resolve()
            if not p.is_relative_to(dest.resolve()):raise RuntimeError('unsafe ZIP path')
        z.extractall(dest)
    manifests=list(dest.glob('*/run_manifest.json'))
    if len(manifests)!=1:raise RuntimeError('ambiguous result root')
    return manifests[0].parent


def artifact_audit(root,setting):
    name='metst_'+setting
    old=json.loads((root/'run_manifest.json').read_text())['identity']
    # Audit recorded identity against shipped source; active local runtime is recorded separately.
    run.verify_reuse(root,name,old['runtime'],old['inputs'])
    with np.load(root/name/'search_counts_G.npz',allow_pickle=False) as z:
        subjects=z['subjects'].tolist();gts=z['full'][0,:,0]+z['full'][0,:,2]
    tables=run.load_reuse(root,name,subjects,gts)
    checked=0
    for m,table in tables.items():
        for si,s in enumerate(subjects):
            winner,_=run.shared.choose(table[1],si)
            path=root/name/'selected_events'/f'{si:03d}_{m}.json.gz'
            with gzip.open(path,'rt') as f:r=json.load(f)
            run.eta.validate_record(r)
            if r['config']!=asdict(run.eta.grids()[m][winner]):raise RuntimeError('saved winner mismatch')
            for j,stage in enumerate(('raw','full')):
                if r[stage+'_counts']!=table[j][winner,si].tolist():raise RuntimeError('saved tensor/event mismatch')
            checked+=1
    return tables,old,dict(dataset=setting,passed=True,subjects=len(subjects),GT=int(sum(gts)),
        audited_saved_selected_records=checked,artifact_code_matches=True,
        active_runtime_reuse_authorized=False,archive_only=(setting=='casme3'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True,help='existing server_standalone directory (sealed + author_runtime)')
    p.add_argument('--sam-results',type=Path,required=True);p.add_argument('--cas-results',type=Path,required=True)
    p.add_argument('--sam-dump',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.assets.resolve()/'author_runtime'))
    if not hasattr(pd.DataFrame,'append'):
        def append(self,other,ignore_index=False,verify_integrity=False,sort=False):
            if isinstance(other,dict):other=pd.DataFrame([other])
            elif isinstance(other,pd.Series):other=other.to_frame().T
            return pd.concat([self,other],ignore_index=ignore_index,verify_integrity=verify_integrity,sort=sort)
        pd.DataFrame.append=append
    warnings.filterwarnings('ignore',category=FutureWarning)
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);reports=[]
        for setting,archive in [('sammlv',a.sam_results),('casme3',a.cas_results)]:
            dest=temp/setting;dest.mkdir();root=unpack(archive,dest)
            tables,old,report=artifact_audit(root,setting);reports.append(report)
            if setting=='sammlv':samroot,samtables,samold=root,tables,old
            print('SAVED_ARTIFACT_AUDIT',setting,report,flush=True)
        assets=a.assets.resolve();runner_path=assets/'sealed/metst_official_glds_full.py';core_path=assets/'sealed/boosting_official_glds_full_LOCKED.py'
        for path in (runner_path,core_path):
            expected={v for k,v in samold['inputs'].items() if Path(k).name==path.name}
            if expected!={run.sha(path)}:raise RuntimeError('sealed source hash mismatch')
        dump=temp/'dump';dump.mkdir();dump_count=0
        with zipfile.ZipFile(a.sam_dump) as z:
            for n in z.namelist():
                if not n.endswith('.pkl'):continue
                filename=Path(n).name;body=z.read(n)
                expected={v for k,v in samold['inputs'].items() if Path(k).name==filename}
                if expected!={hashlib.sha256(body).hexdigest()}:raise RuntimeError('dump bytes differ: '+filename)
                path=dump/filename
                if path.exists():raise RuntimeError('duplicate dump name')
                path.write_bytes(body);dump_count+=1
        if dump_count!=29:raise RuntimeError('SAM dump inventory mismatch')
        runner=load(runner_path,'reliable_probe_sealed_runner');base=load(core_path,'reliable_probe_sealed_core')
        runner.DUMP_DIR=dump;runner.ROOT=assets/'author_runtime';runner.LOCKED_GLSD_PATH=core_path
        metric,official=runner.load_official_metst();paths,records=runner.load_records()
        subjects=[str(s) for s in runner.subject_names(records)]
        context=(runner,base,metric,official,records,paths,subjects,None)
        run.verify_runtime_sources(context);chain=run.decoder_identity(context)
        core=run.FusionCore(base);core.expected_gt=run.prior.previous.gt_counts(context)
        source_info=json.loads((samroot/'matching_protocol.json').read_text())
        with run.evaluator.install(metric) as info:
            info['metric_source_sha256']=run.sha(Path(run.inspect.getsourcefile(metric)))
            for k in ('protocol','metric_source_sha256','legacy_check_box_sha256'):
                if info[k]!=source_info[k]:raise RuntimeError('aggregate identity mismatch: '+k)
            feature=run.feature_audit(context,core,out)
            probe=run.probe(context,core,samtables,out)
            exported_candidates=0
            with gzip.open(out/'probe_event_records.jsonl.gz','rt') as src, gzip.open(out/'probe_candidate_records.jsonl.gz','wt') as dest:
                for line in src:
                    for row in run.candidate_rows(json.loads(line)):
                        dest.write(json.dumps(run.shared.serializable(row),allow_nan=False)+'\n');exported_candidates+=1
            # Broader compatibility replay of saved controls; no threshold/config search.
            selected_replays=0; max_feature_abs_error=0.
            with gzip.open(out/'compatibility_event_records.jsonl.gz','wt') as ledger:
                for si,s in enumerate(subjects):
                    for m in ('G','L','Mean','G_union',*('G_rescaled_p'+p for p in run.joint.POWERS)):
                        w,_=run.shared.choose(samtables[m][1],si);oldc=run.eta.grids()[m][w]
                        c=run.Config(m,w,oldc.reference_scale,oldc.local_radius,oldc.threshold,p=oldc.p)
                        r=run.detailed_decode(context,core,si,c)
                        with gzip.open(samroot/'metst_sammlv'/'selected_events'/f'{si:03d}_{m}.json.gz','rt') as f:saved=json.load(f)
                        for stage in ('raw','full'):
                            if r[stage+'_counts']!=saved[stage+'_counts']:raise RuntimeError('actual historical control count mismatch')
                        for now,prev in zip(r['events'],saved['events']):
                            if any(now[k]!=v for k,v in prev.items()):raise RuntimeError('actual historical event identity mismatch')
                        if len(r['events'])!=len(saved['events']):raise RuntimeError('historical event length mismatch')
                        for now,prev in zip(r['candidates'],saved['candidates']):
                            for k in ('response_sha256','peaks','G','L','retained'):
                                if k in ('G','L'):
                                    error=float(np.max(np.abs(np.asarray(now[k])-prev[k]))) if now[k] else 0.
                                    max_feature_abs_error=max(max_feature_abs_error,error)
                                    np.testing.assert_allclose(now[k],prev[k],rtol=0,atol=1e-12)
                                elif now[k]!=prev[k]:raise RuntimeError('historical candidate replay mismatch '+k)
                        ledger.write(json.dumps(run.shared.serializable(r),allow_nan=False)+'\n');selected_replays+=1
                    print('local control compatibility',si+1,'/',len(subjects),flush=True)
        runtime,extra=run.runtime_identity()
        report=dict(status='PASS',stage='local implementation validation only',
            inputs={str(p):run.sha(p) for p in (a.sam_results,a.cas_results,a.sam_dump,runner_path,core_path)},
            code={p.name:run.sha(p) for p in run.ROOT.glob('*.py')},runtime=runtime,extra_runtime=extra,
            original_colab_runtime=samold['runtime'],same_runtime_as_colab=runtime==samold['runtime'],
            artifact_audits=reports,SAM_dump_files=dump_count,feature_audit=feature,probe=probe,
            actual_historical_selected_control_replays=selected_replays,decoder=chain,
            historical_G_L_max_abs_error=max_feature_abs_error,historical_feature_tolerance=1e-12,
            actual_probe_candidate_rows_exported=exported_candidates,
            full_experiment_run=False,new_full_F1=None,CAS_real_response_probe='NOT RUN: matching dump unavailable locally',
            checks=['inherited source bytes match both saved runs','both saved count tensors and selected events agree',
                'SAM response/core/runner hashes match saved input identity','new feature G/L exact against current sealed core; historical G/L within 1e-12; historical peaks and keep exact',
                'reliability feature audit all SAM structures','new scorer and locked-branch actual decoder probes',
                'subject037 regression raw/full preserved','historical selected control event IDs exactly replay',
                'no backbone or full grid search performed'])
        run.atomic_json(out/'validation_report.json',report)
        print('LOCAL_DELIVERY_VALIDATION = PASS',flush=True)

if __name__=='__main__':main()
