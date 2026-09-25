"""Locked GLSD evidence audit; no writes to the Skill or previous results."""
from pathlib import Path
import json
import pickle
import hashlib
import numpy as np
from my_method.gl_saliency_skill.benchmark import load_bundle, signed_context, write_json, write_csv
from my_method.gl_saliency_skill.evidence import configuration_grid
from my_method.gl_saliency_skill.selection import choose, inner_counts, indexed_pools
from my_method.gl_saliency_skill.skill import GLSaliencySkill
from my_method.gl_saliency_skill.evaluation import evaluate, metrics
from my_method.strict_cross_backbone_transfer.run import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/glsd_evidence_audit_v1'

def manifest():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'my_method/gl_saliency_skill').rglob('*.py'))}

def decode(record, config, k, adapter):
    return GLSaliencySkill(config).decode(record.score, int(k), {'interval_adapter': adapter})

def total(counts):
    return metrics(dict(zip(('TP','FP','FN'), counts.sum(axis=0))))

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    before = manifest()
    configs = configuration_grid()
    pools = indexed_pools(configs)
    pools['H'] = [i for i,c in enumerate(configs) if c.family == 'height']
    write_json(OUT/'locked_plan.json', {'pools': {m:len(p) for m,p in pools.items()},
        'seed':100, 'resamples':10000, 'tolerance':1e-12,
        'scope':'frozen-output evaluation, not end-to-end nested model training',
        'skill_manifest':before})
    contexts = {}
    main_rows, transfer_rows, cis, selections, predictions, checks, provenance = [],[],[],[],[],[],[]
    for dataset in ('sammlv','casme3'):
        for backbone in ('metst','boostingvrme'):
            bundle = load_bundle(backbone,dataset)
            _, protocol, outer, inner, stats, mode = signed_context(bundle, configs)
            chosen = {}
            for held,subject in enumerate(bundle.subjects):
                training = inner_counts(stats,held,inner)
                chosen[subject] = {m:choose(training,p) for m,p in pools.items()}
                # Poison every held-subject cell: historical aggregation must not read it.
                poisoned = {k:v.copy() for k,v in stats.items()}
                for v in poisoned.values():
                    v[:,held,:] = 987654321
                np.testing.assert_array_equal(training,inner_counts(poisoned,held,inner))
                checks.append({'backbone':backbone,'dataset':dataset,'subject':subject,
                               'held_stats_poison_invariance':True})
                for m,i in chosen[subject].items():
                    selections.append({'backbone':backbone,'dataset':dataset,'subject':subject,
                        'method':m,'config_id':i,'config':configs[i].identifier,'k':int(outer[held])})
            counts = {m:np.zeros((len(bundle.subjects),3),dtype=np.int64)
                      for m in ('Author Native','Fold-k Native','H','G','L','GL')}
            lookup = {s:i for i,s in enumerate(bundle.subjects)}
            for record in bundle.records:
                s = lookup[record.subject]
                for method in counts:
                    i = 0 if 'Native' in method else chosen[record.subject][method]
                    k = bundle.legacy_temporal_scale if method=='Author Native' else outer[s]
                    events = decode(record,configs[i],k,bundle.interval_adapter)
                    c,_ = evaluate(events,record.ground_truth)
                    counts[method][s] += [c[t] for t in ('TP','FP','FN')]
                    if method=='GL':
                        # GT is not an input to the decoder; evaluating other labels cannot mutate it.
                        snapshot=json.dumps(events,sort_keys=True)
                        evaluate(events,[])
                        assert snapshot==json.dumps(events,sort_keys=True)
            for m,c in counts.items():
                main_rows.append({'backbone':backbone,'dataset':dataset,'method':m,**total(c)})
            for m in ('Author Native','Fold-k Native','H','G','L'):
                cis.append({'setting':f'{backbone}/{dataset}','comparison':f'GL-{m}',
                            **paired_bootstrap(counts['GL'],counts[m])})
            with Path(bundle.cache_path).open('rb') as f:
                package=pickle.load(f)
            provenance.append({'backbone':backbone,'dataset':dataset,'path':bundle.cache_path,
                'sha256':protocol['input_sha256'],
                'metadata':{k:v for k,v in package.items() if k not in ('records','curves')},
                'record_fields':list(package['records'][0]),
                'checkpoint_to_training_subject_mapping':'UNVERIFIED',
                'direct_test_leakage':'UNVERIFIED',
                'end_to_end_nested_LOSO':'NOT_ESTABLISHED'})
            # No target training statistics enter the following transfer stage.
            contexts[backbone,dataset]=(bundle,outer,chosen,counts)
            del stats,poisoned
            print('fair comparisons complete',backbone,dataset,flush=True)
    transfer_counts={}
    for dataset in ('sammlv','casme3'):
        for source,target in (('metst','boostingvrme'),('boostingvrme','metst')):
            sb,_,source_selected,_=contexts[source,dataset]
            tb,outer,target_selected,target_counts=contexts[target,dataset]
            assert sb.subjects==tb.subjects
            # Freeze source-only configuration IDs in an independent mapping.
            frozen={s:dict(v) for s,v in source_selected.items()}
            counts={m:np.zeros((len(tb.subjects),3),dtype=np.int64) for m in ('H','G','L','GL')}
            for record in tb.records:
                s=tb.subjects.index(record.subject)
                for method in counts:
                    i=frozen[record.subject][method]
                    events=decode(record,configs[i],outer[s],tb.interval_adapter)
                    c,details=evaluate(events,record.ground_truth)
                    counts[method][s]+=[c[t] for t in ('TP','FP','FN')]
                    predictions.append({'source':source,'target':target,'dataset':dataset,
                        'subject':record.subject,'video':record.video,'method':method,
                        'config_id':i,'k':int(outer[s]),'counts':c,'predictions':details})
            setting=f'{source}->{target}/{dataset}'
            for m,c in counts.items():
                transfer_rows.append({'source':source,'target':target,'dataset':dataset,
                    'method':m,**total(c),'delta_native':total(c)['F1']-total(target_counts['Author Native'])['F1']})
            for m in ('H','G','L'):
                cis.append({'setting':setting,'comparison':f'Transferred GL-{m}',
                            **paired_bootstrap(counts['GL'],counts[m])})
            transfer_counts[setting]={m:v.tolist() for m,v in counts.items()}
            print('component transfer complete',setting,flush=True)
    # Existing primary and transfer GL results are exact regression anchors.
    import csv
    prior=list(csv.DictReader((ROOT/'results/final_gl_skill/main_results.csv').open()))
    for r in prior:
        n=next(x for x in main_rows if x['backbone']==r['Backbone'] and x['dataset']==r['Dataset'] and x['method']=='GL')
        assert all(n[k]==int(r['GL_Skill_'+k]) for k in ('TP','FP','FN'))
        assert n['F1']==float(r['GL_Skill_F1'])
    prior=list(csv.DictReader((ROOT/'results/final_gl_skill_transfer/transfer_main_results.csv').open()))
    for r in prior:
        n=next(x for x in transfer_rows if x['source']==r['Source'] and x['target']==r['Target'] and x['dataset']==r['Dataset'] and x['method']=='GL')
        assert all(n[k]==int(r['Transferred_'+k]) for k in ('TP','FP','FN'))
        assert n['F1']==float(r['Transferred_GL_F1'])
    assert before==manifest()
    write_csv(OUT/'fair_baselines.csv',main_rows)
    write_csv(OUT/'component_transfer.csv',transfer_rows)
    write_csv(OUT/'paired_bootstrap.csv',cis)
    write_csv(OUT/'selections.csv',selections)
    write_json(OUT/'transfer_predictions.json',predictions)
    write_json(OUT/'cache_provenance.json',provenance)
    write_json(OUT/'audit.json',{'held_poison_checks':checks,'primary_and_transfer_GL_exact':True,
        'skill_unchanged':True,'manifest':before,
        'backbone_training_provenance':'UNVERIFIED; not implied by decoder isolation'})
    write_json(OUT/'subject_counts.json',{'subjects':{d:contexts['metst',d][0].subjects for d in ('sammlv','casme3')},
        'fair':{f'{b}/{d}':{m:v.tolist() for m,v in ctx[3].items()} for (b,d),ctx in contexts.items()},
        'transfer':transfer_counts})
    print('EVIDENCE_COMPARISONS_COMPLETE',flush=True)

if __name__=='__main__':
    main()
