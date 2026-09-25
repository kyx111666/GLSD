#!/usr/bin/env python3
"""Verify saved FIRST3 results, selections and LR inference without new tuning."""
import csv
import hashlib
import json
import pickle
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import expit
from run_first3_nested import ROOT, METHODS, IDEAS, SOURCE, CACHE, K_P, EPS
from run_mscr_nested_loso import tuned_native_decode, match_events, interval_iou

OUT=ROOT/'results/first3_candidate_recovery_nested/outputs'


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def main():
    report=json.loads((OUT/'first3_summary.json').read_text())
    with CACHE.open('rb') as f:records=pickle.load(f)['records']
    source=json.loads(SOURCE.read_text())
    configs={f['subject']:f['selected_strong_config'] for f in source['outer_folds']}
    subjects=sorted(configs)
    common=read('common_weak_candidate_trace.csv')
    pool={(r['outer_subject'],r['subject'],r['video'],float(r['Delta_p']),int(r['peak'])):r for r in common}
    passed=[]
    assert report['completed_outer_folds']==29 and not report['incomplete']
    for path,digest in report['provenance'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    passed.append('PASS 1: frozen cache, saved config, protocol and experiment source hashes match; 29 folds complete.')
    native={}
    for rec in records:
        s,v=str(rec['subject']),str(rec['video']);c=configs[s]
        native[(s,v)]=tuned_native_decode({'score':rec['score']},dict(c_s=c['c_s'],p=c['p_s'],c_d=c['c_d'],c_b=c['c_b']))['events']
    stems=('low_rescue','psed','recognition_joint','morphology')
    for method,stem in zip(METHODS,stems):
        predictions=read(f'{stem}_prediction_trace.csv')
        counts=Counter(TP=0,FP=0,FN=0,rescue_TP=0,rescue_FP=0)
        high_count=0
        for rec in records:
            s,v=str(rec['subject']),str(rec['video'])
            rows=[r for r in predictions if r['subject']==s and r['video']==v]
            high=[{k:int(r[k]) if k!='source' else r[k] for k in ('onset','peak','offset','source')}
                  for r in rows if r['source']=='tuned_native']
            assert high==native[(s,v)]
            high_count+=len(high)
            matched,missed=match_events(rows,rec['samples'])
            assert matched==[int(r['matched_gt']) for r in rows]
            counts.update(TP=sum(m>=0 for m in matched),FP=sum(m<0 for m in matched),FN=len(missed),
                rescue_TP=sum(m>=0 and r['source']=='rescue' for m,r in zip(matched,rows)),
                rescue_FP=sum(m<0 and r['source']=='rescue' for m,r in zip(matched,rows)))
        assert high_count==192
        assert all(counts[k]==report['aggregate_metrics'][method][k] for k in counts)
    passed.append('PASS 2: official matching on all saved events reproduces every method count and rescue TP/FP; all 192 high events exactly preserved.')
    for method,stem,size in zip(METHODS,stems,(4,60,100,288)):
        scores=read(f'{stem}_inner_config_scores.csv');selections=read(f'{stem}_selected_configs.csv')
        assert len(scores)==29*size and len(selections)==29
        for selection in selections:
            held=selection['outer_subject']
            assert set(selection['train_subjects'].split(';'))==set(subjects)-{held}
            candidates=[r for r in scores if r['outer_subject']==held]
            def rank(row):
                c=json.loads(row['config_json']);tp,fp,fn=map(int,[row['TP'],row['FP'],row['FN']])
                return (-Fraction(2*tp,2*tp+fp+fn),fp,int(row['rescued_candidate_count']),
                        c['Delta_p'],-c.get('gamma',0),tuple(c[k] for k in sorted(c)))
            best=json.loads(min(candidates,key=rank)['config_json'])
            assert all(selection[k]==str(v) for k,v in best.items())
    passed.append('PASS 3: independent exact-F1 tie-break replay matches all 116 selections; grids exactly 4/60/100/288 and test excluded.')
    pool_hashes=json.loads((OUT/'common_pool_hashes.json').read_text())
    assert len(pool_hashes)==29*4 and all(len(set(v.values()))==1 for v in pool_hashes.values())
    for stem in stems:
        trace=read(f'{stem}_weak_selection_trace.csv')
        for r in trace:
            shared=pool[(r['outer_subject'],r['outer_subject'],r['video'],float(r['Delta_p']),int(r['peak']))]
            assert all(shared[k]==r[k] for k in ('onset','peak','offset'))
    passed.append('PASS 4: selected weak events are members of the same shared Delta-specific pool; all 116 pool fingerprints agree across methods.')
    audits=read('morphology_inner_fit_audit.csv')
    assert len(audits)==29*28*4*4*2*3
    degenerate=[]
    for row in audits:
        held,val=row['outer_subject'],row['inner_validation_subject']
        assert set(row['inner_training_subjects'].split(';'))==set(subjects)-{held,val}
        assert held!=val
        if row['single_class_constant']=='True':
            assert {held,val}=={'007','013'} and float(row['Delta_p'])==.05
            assert int(row['train_positive_count'])==int(row['max_iterations_used'])==0
            degenerate.append(row)
        else:
            assert 0<int(row['train_positive_count'])<int(row['train_candidate_count'])
            assert 0<int(row['max_iterations_used'])<2000
    assert len(degenerate)==48
    passed.append('PASS 5: 77,952 inner fit records exclude both held subjects; 48 predefined all-negative cases use constant zero; remaining 77,904 LR fits converged.')
    # Independently reconstruct [H, patch] and replay saved scaler/LR state.
    models=json.loads((OUT/'morphology_fitted_models.json').read_text())
    test_rows=read('morphology_weak_selection_trace.csv')
    probabilities={(r['outer_subject'],r['video'],int(r['peak'])):r for r in test_rows}
    max_error=0.
    for state in models:
        held=state['outer_subject'];config=state['selected_config'];c=configs[held]
        assert set(state['train_subjects'])==set(subjects)-{held}
        x_train=[];y_train=[]
        for rec in records:
            s,v=str(rec['subject']),str(rec['video'])
            decoded=tuned_native_decode({'score':rec['score']},dict(c_s=c['c_s'],p=c['p_s'],c_d=c['c_d'],c_b=c['c_b']))
            curve=decoded['curve'];radius=max(1,int(round(config['L']*K_P)))
            weak=[r for r in common if r['outer_subject']==held and r['subject']==s and r['video']==v and float(r['Delta_p'])==config['Delta_p']]
            _,missed=match_events(decoded['events'],rec['samples'])
            for r in weak:
                peak=int(r['peak']);pad=np.pad(curve,(radius,radius),mode='edge')
                patch=pad[peak:peak+2*radius+1]
                patch=np.interp(np.linspace(0,1,31),np.linspace(0,1,len(patch)),patch)
                if config['normalization']=='raw-z':patch=(patch-patch.mean())/(patch.std()+EPS)
                else:
                    patch=patch-patch[15];patch=patch/(np.linalg.norm(patch)+EPS)
                x=np.r_[float(r['H']),patch]
                if s==held:
                    value=float(expit(((x-np.asarray(state['scaler_mean']))/state['scaler_scale'])@np.asarray(state['coef'])[0]+state['intercept'][0]))
                    row=probabilities[(held,v,peak)]
                    error=abs(value-float(row['evidence']));max_error=max(max_error,error)
                    assert error<1e-12
                    assert (value>=config['gamma'])==(row['selected']=='True')
                else:
                    x_train.append(x)
                    y_train.append(int(any(interval_iou(r,rec['samples'][j])>=.5 for j in missed)))
        x_train=np.array(x_train)
        np.testing.assert_allclose(x_train.mean(axis=0),state['scaler_mean'],rtol=1e-12,atol=1e-14)
        scale=x_train.std(axis=0);scale[scale==0]=1.
        np.testing.assert_allclose(scale,state['scaler_scale'],rtol=1e-12,atol=1e-14)
        assert len(y_train)==state['train_candidates'] and sum(y_train)==state['train_positive']
    passed.append(f'PASS 6: all 29 selected LR models replay independently; max probability error {max_error:.3g}; scaler stats and training class counts match outer-training data only.')
    samples=json.loads((OUT/'bootstrap_subject_indices.json').read_text());indices=np.array(samples['indices'])
    np.testing.assert_array_equal(indices,np.random.default_rng(20260902).integers(0,29,size=(1000,29)))
    baseline_rows=read('baseline_outer_metrics.csv')
    baseline={r['outer_subject']:r for r in baseline_rows if r['method']=='Tuned Native'}
    for method,stem in zip(IDEAS,stems[1:]):
        outer={r['outer_subject']:r for r in read(f'{stem}_outer_metrics.csv')}
        values=[]
        for draws in indices:
            vals=[]
            for rows in (outer,baseline):
                tp,fp,fn=[sum(int(rows[subjects[i]][k]) for i in draws) for k in ('TP','FP','FN')]
                vals.append(2*tp/(2*tp+fp+fn))
            values.append(vals[0]-vals[1])
        np.testing.assert_allclose(np.percentile(values,[2.5,97.5]),report['bootstrap'][method]['ci95'],rtol=0,atol=1e-15)
        assert abs(float(np.mean(values))-report['bootstrap'][method]['mean_delta'])<1e-15
    passed.append('PASS 7: independently replayed 1,000 common subject-bootstrap draws and all three mean deltas / confidence intervals exactly.')
    audit={'status':'PASS','checks':passed,'single_class_constant_inner_cases':48,
           'normal_inner_LR_fits':77904,'outer_refits':29,'max_probability_replay_error':max_error}
    (OUT/'independent_artifact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    text='# FIRST3 saved-result verification\n\n'+'\n\n'.join(passed)+'\n'
    (OUT/'INDEPENDENT_ARTIFACT_AUDIT.md').write_text(text)
    print(text)


if __name__=='__main__':main()
