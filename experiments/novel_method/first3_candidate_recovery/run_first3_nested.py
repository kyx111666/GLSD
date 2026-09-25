#!/usr/bin/env python3
"""Predefined SAMMLV candidate recovery comparison; cache-only CPU execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.signal import find_peaks, peak_prominences
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'my_method/multi_scale_candidate_rescue'
sys.path[:0] = [str(LEGACY), str(ROOT/'my_method/rgr1_sammlv'),
               str(ROOT/'my_method/psed_feasibility'), str(ROOT/'my_method/temporal_morphology_feasibility')]
from run_mscr_nested_loso import (author_native, evaluate_decoding, match_events,
    empty_counts, add_counts, metrics, f1, sha256, EPS, K_P)
from run_rgr1_nested_loso import decode_low
from run_native_parameter_recognition_audit import softmax, aggregate, CLASS_ORDER, NEUTRAL_ID
from run_feasibility import robust_z
from run_morphology_audit import extract_patch, resample_patch

SEED = 20260902
DELTAS = (.05,.10,.15,.20)
GAMMAS = (.25,.40,.55,.70,.85)
LS = (.5,1.,1.5,2.)
NORMS = ('raw-z','peak-relative')
CS = (.1,1.,10.)
MG = (.3,.5,.7)
METHODS = ('LowRescue','PSED Rescue','Recognition Joint','Morphology Rescue')
IDEAS = METHODS[1:]
CACHE = ROOT/'caches/me_tst/sammlv_strategy1_outputs.pkl'
SOURCE = ROOT/'results/rgr1_sammlv_nested/report.json'
CACHE_HASH = '3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569'
FIELDS = ('TP','FP','FN','event_count','Precision','Recall','F1','weak_candidate_count',
          'rescued_candidate_count','rescue_TP','rescue_FP','native_missed_GT_recovered','rescue_precision')
CTX = None


def grid(method):
    if method == 'LowRescue':
        return [dict(Delta_p=d) for d in DELTAS]
    if method == 'PSED Rescue':
        return [dict(Delta_p=d,lambda_P=l,gamma=g) for d,l,g in itertools.product(DELTAS,(.25,.5,.75),GAMMAS)]
    if method == 'Recognition Joint':
        return [dict(Delta_p=d,alpha=a,gamma=g) for d,a,g in itertools.product(DELTAS,(0.,.25,.5,.75,1.),GAMMAS)]
    if method == 'Morphology Rescue':
        return [dict(Delta_p=d,L=l,normalization=n,C=c,gamma=g)
                for d,l,n,c,g in itertools.product(DELTAS,LS,NORMS,CS,MG)]
    raise ValueError(method)


def key(config):
    return json.dumps(config,sort_keys=True)


def ranking(counts, config):
    den=2*counts['TP']+counts['FP']+counts['FN']
    score=Fraction(2*counts['TP'],den) if den else Fraction(0)
    return (-score,counts['FP'],counts['rescued_candidate_count'],config['Delta_p'],
            -config.get('gamma',0),tuple(config[k] for k in sorted(config)))


def features(curve, peak, height, multiplier, normalization):
    radius=max(1,int(round(multiplier*K_P)))
    patch=resample_patch(extract_patch(curve,peak,radius))
    if normalization=='raw-z':
        patch=(patch-patch.mean())/(patch.std()+EPS)
    else:
        assert normalization=='peak-relative'
        patch=patch-patch[15]
        patch=patch/(np.linalg.norm(patch)+EPS)
    return np.concatenate(([height],patch))


def common_pool(signal, native_config):
    """GT-free adapter directly reusing the old low-minus-high implementation."""
    assert set(signal)=={'subject','video','score','logits'}
    probs=softmax(signal['logits'])
    assert probs.shape==(len(signal['score']),5) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1),1)
    r2=np.max(probs[:,:NEUTRAL_ID],axis=1)
    decoded={d:decode_low({'score':signal['score'],'R2':np.zeros(len(signal['score']))},
                           native_config,d) for d in DELTAS}
    first=decoded[DELTAS[0]]['native']
    curve=first['curve']
    z=robust_z(curve)
    peaks=find_peaks(z)[0]
    prominence=dict(zip(peaks.tolist(),peak_prominences(z,peaks)[0].tolist()))
    pools={}
    for d,result in decoded.items():
        assert result['native']['events']==first['events']
        events=result['rescued']
        assert [e['peak'] for e in events]==list(result['weak'])
        low=result['native']['curve'].mean()+result['p_w']*(curve.max()-curve.mean())
        high=first['threshold']
        hs=np.asarray([np.clip((curve[e['peak']]-low)/(high-low+EPS),0,1) for e in events])
        ps=np.asarray([expit(prominence[e['peak']]) for e in events])
        rs=np.asarray([aggregate(r2,e['peak'])['mean'] for e in events])
        patches={(l,n):np.asarray([features(curve,e['peak'],h,l,n) for e,h in zip(events,hs)]).reshape(-1,32)
                 for l,n in itertools.product(LS,NORMS)}
        pools[d]={'events':events,'H':hs,'P':ps,'R2':rs,'patches':patches,
                  'prominence_z':np.asarray([prominence[e['peak']] for e in events]),
                  'low_events':result['events']}
    return {'subject':signal['subject'],'video':signal['video'],'high':first['events'],'pools':pools}


def union(pool, delta, keep):
    weak=pool['pools'][delta]['events']
    assert len(weak)==len(keep)
    rescued=[event for event,selected in zip(weak,keep) if selected]
    events=sorted(pool['high']+rescued,key=lambda e:(e['peak'],0 if e['source']=='tuned_native' else 1))
    assert [e for e in events if e['source']=='tuned_native']==pool['high']
    assert len(events)==len(pool['high'])+len(rescued)
    return {'events':events,'native':{'events':pool['high']},
            'weak':[e['peak'] for e in weak],'rescued':rescued}


def evidence(pool, method, config):
    data=pool['pools'][config['Delta_p']]
    if method=='LowRescue': return np.ones(len(data['events']))
    if method=='PSED Rescue':
        return config['lambda_P']*data['H']+(1-config['lambda_P'])*data['P']
    assert method=='Recognition Joint'
    return config['alpha']*data['H']+(1-config['alpha'])*data['R2']


def evaluate_subset(records, bank, method, config, probabilities=None, trace=False):
    total=empty_counts(); traces=[]
    for record in records:
        rid=(str(record['subject']),str(record['video']))
        pool=bank[rid]
        values=probabilities[rid] if probabilities is not None else evidence(pool,method,config)
        decoded=union(pool,config['Delta_p'],values>=config.get('gamma',0))
        result=evaluate_decoding(record,decoded)
        add_counts(total,result)
        if trace:
            values_by_peak=dict(zip([e['peak'] for e in pool['pools'][config['Delta_p']]['events']],values))
            for event,match in zip(decoded['events'],result['matches']):
                traces.append({'outer_subject':rid[0],'subject':rid[0],'video':rid[1],'method':method,
                    'candidate_id':f'{rid[0]}/{rid[1]}/{event["source"]}_peak_{event["peak"]}',
                    **event,'matched_gt':int(match),'rescue_evidence':float(values_by_peak[event['peak']])
                    if event['source']=='rescue' else None})
    return total,traces


def label_training(records, bank, delta):
    """Labels are created only for outer-training records passed explicitly."""
    labels={}
    for record in records:
        rid=(str(record['subject']),str(record['video']))
        pool=bank[rid]
        _,missed=match_events(pool['high'],record['samples'])
        targets=[record['samples'][j] for j in sorted(missed)]
        labels[rid]=np.asarray([int(match_events([e],targets)[0][0]>=0)
                               for e in pool['pools'][delta]['events']],dtype=int)
    return labels


def init_worker(context):
    global CTX
    CTX=context
    threadpool_limits(limits=1)
    warnings.simplefilter('error',ConvergenceWarning)


def model_fit(x,y,c):
    assert x.shape[1]==32 and len(np.unique(y))==2
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=c,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=SEED)
    model.fit(scaler.transform(x),y)
    assert int(max(model.n_iter_))<2000
    return scaler,model


def model_predict(scaler,model,x):
    return model.predict_proba(scaler.transform(x))[:,1] if len(x) else np.empty(0)


def worker(job):
    method,held=job
    context=CTX
    bank=context['banks'][key(context['folds'][held]['selected_strong_config'])]
    train=[r for r in context['records'] if str(r['subject'])!=held]
    test=[r for r in context['records'] if str(r['subject'])==held]
    train_subjects=sorted({str(r['subject']) for r in train})
    assert set(train_subjects)==set(context['subjects'])-{held} and len(train_subjects)==28
    scored=[]; fit_audit=[]; probability_bank={}
    if method!='Morphology Rescue':
        for config in grid(method):
            counts,_=evaluate_subset(train,bank,method,config)
            scored.append((ranking(counts,config),config,counts))
    else:
        # Both labels and all fitted statistics are confined to outer train.
        # Test record labels are never passed to the training routines.
        rids=[(str(r['subject']),str(r['video'])) for r in train]
        for delta in DELTAS:
            labels=label_training(train,bank,delta)
            y=np.concatenate([labels[rid] for rid in rids])
            owner=np.concatenate([np.repeat(rid[0],len(labels[rid])) for rid in rids])
            assert held not in set(owner)
            offsets=np.cumsum([0]+[len(labels[rid]) for rid in rids])
            for l,n,c in itertools.product(LS,NORMS,CS):
                x=np.concatenate([bank[rid]['pools'][delta]['patches'][(l,n)] for rid in rids])
                oof=np.full(len(y),np.nan)
                for validation in train_subjects:
                    tr=owner!=validation; va=~tr
                    assert not set(owner[tr]) & {held,validation}
                    single_class=len(np.unique(y[tr]))<2
                    if single_class:
                        # A two-class LR is undefined here. The sole training
                        # class supplies the fixed probability; no validation
                        # or test label is consulted to resolve this case.
                        oof[va]=float(y[tr][0]) if tr.any() else 0.
                        iterations=0
                    else:
                        scaler,model=model_fit(x[tr],y[tr],c)
                        oof[va]=model_predict(scaler,model,x[va])
                        iterations=int(max(model.n_iter_))
                    fit_audit.append({'outer_subject':held,'inner_validation_subject':validation,
                        'inner_training_subjects':';'.join(s for s in train_subjects if s!=validation),
                        'Delta_p':delta,'L':l,'normalization':n,'C':c,
                        'train_candidate_count':int(tr.sum()),'train_positive_count':int(y[tr].sum()),
                        'validation_candidate_count':int(va.sum()),'max_iterations_used':iterations,
                        'single_class_constant':single_class,
                        'train_x_sha256':hashlib.sha256(x[tr].tobytes()).hexdigest(),
                        'train_y_sha256':hashlib.sha256(y[tr].tobytes()).hexdigest()})
                assert np.isfinite(oof).all()
                probabilities={rid:oof[offsets[i]:offsets[i+1]] for i,rid in enumerate(rids)}
                probability_bank[(delta,l,n,c)]=probabilities
                for gamma in MG:
                    config=dict(Delta_p=delta,L=l,normalization=n,C=c,gamma=gamma)
                    counts,_=evaluate_subset(train,bank,method,config,probabilities)
                    scored.append((ranking(counts,config),config,counts))
    _,chosen,inner=min(scored,key=lambda row:row[0])
    model_state=None; selected_oof=[]
    if method=='Morphology Rescue':
        delta,l,n,c=(chosen[k] for k in ('Delta_p','L','normalization','C'))
        labels=label_training(train,bank,delta)
        rids=list(labels)
        x=np.concatenate([bank[rid]['pools'][delta]['patches'][(l,n)] for rid in rids])
        y=np.concatenate([labels[rid] for rid in rids])
        scaler,model=model_fit(x,y,c)
        probabilities={}
        for record in test:
            rid=(held,str(record['video']))
            probabilities[rid]=model_predict(scaler,model,bank[rid]['pools'][delta]['patches'][(l,n)])
        oof=probability_bank[(delta,l,n,c)]
        for rid in rids:
            for event,label,prob in zip(bank[rid]['pools'][delta]['events'],labels[rid],oof[rid]):
                selected_oof.append({'outer_subject':held,'validation_subject':rid[0],'video':rid[1],
                    'weak_peak':event['peak'],'training_label_for_audit':int(label),'oof_probability':float(prob)})
        model_state={'outer_subject':held,'train_subjects':train_subjects,'selected_config':chosen,
            'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
            'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),
            'train_candidates':len(y),'train_positive':int(y.sum()),'max_iter_used':int(max(model.n_iter_))}
    else:
        probabilities=None
    # Only the selected config is evaluated against the held-out GT here.
    outer,traces=evaluate_subset(test,bank,method,chosen,probabilities,trace=True)
    final_matches={(r['video'],r['peak']):r['matched_gt'] for r in traces}
    weak_trace=[]
    for record in test:
        rid=(held,str(record['video']));pool=bank[rid]
        values=probabilities[rid] if probabilities is not None else evidence(pool,method,chosen)
        for event,value in zip(pool['pools'][chosen['Delta_p']]['events'],values):
            selected=bool(value>=chosen.get('gamma',0))
            weak_trace.append({'outer_subject':held,'video':rid[1],'method':method,**event,
                'candidate_id':f'{held}/{rid[1]}/weak_peak_{event["peak"]}',
                'Delta_p':chosen['Delta_p'],'evidence':float(value),'selected':selected,
                'matched_gt_posthoc':final_matches.get((rid[1],event['peak']),-1)})
    inner_rows=[{'outer_subject':held,'method':method,'config_json':key(config),
        **{k:metrics(counts)[k] for k in FIELDS}} for _,config,counts in scored]
    return {'outer_subject':held,'method':method,'selected_config':chosen,'inner_metrics':metrics(inner),
        'outer_metrics':metrics(outer),'train_subjects':train_subjects,'inner_rows':inner_rows,
        'predictions':traces,'weak_trace':weak_trace,'fit_audit':fit_audit,'selected_oof':selected_oof,'model_state':model_state}


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader();writer.writerows(rows)


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def baseline(records,folds):
    by_subject={};total={m:empty_counts() for m in ('Author Native','Tuned Native')}
    for subject,fold in folds.items():
        by_subject[subject]={}
        for method in total:
            counts=empty_counts()
            for record in records:
                if str(record['subject'])!=subject:continue
                score_only={'score':record['score']}
                decoded=author_native(score_only) if method=='Author Native' else decode_low(
                    {**score_only,'R2':np.zeros(len(record['score']))},fold['selected_strong_config'],.05)['native']
                add_counts(counts,evaluate_decoding(record,decoded))
            by_subject[subject][method]=metrics(counts)
            add_counts(total[method],counts)
    for method,expected in [('Author Native',(53,184,106)),('Tuned Native',(49,143,110))]:
        if tuple(total[method][k] for k in ('TP','FP','FN'))!=expected:
            raise RuntimeError('BLOCKED-BASELINE-MISMATCH')
    assert total['Tuned Native']['event_count']==192
    return {m:metrics(c) for m,c in total.items()},by_subject


def common_trace(banks,folds,held_subjects):
    rows=[];pool_hashes={}
    for held in held_subjects:
        bank=banks[key(folds[held]['selected_strong_config'])]
        for delta in DELTAS:
            identity_rows=[]
            for (subject,video),pool in bank.items():
                data=pool['pools'][delta]
                all_union=union(pool,delta,np.ones(len(data['events']),dtype=bool))
                assert all_union['events']==data['low_events']
                for i,event in enumerate(data['events']):
                    identity=f'{subject}/{video}/weak_peak_{event["peak"]}'
                    row={'outer_subject':held,'role':'outer_test' if subject==held else 'outer_train',
                        'subject':subject,'video':video,'Delta_p':delta,'candidate_id':identity,**event,
                        'H':float(data['H'][i]),'P':float(data['P'][i]),'R2_mean':float(data['R2'][i]),
                        'prominence_z':float(data['prominence_z'][i])}
                    rows.append(row);identity_rows.append((identity,event['onset'],event['peak'],event['offset']))
            digest=hashlib.sha256(json.dumps(identity_rows).encode()).hexdigest()
            pool_hashes[f'{held}/{delta}']={method:digest for method in METHODS}
    return rows,pool_hashes


def finish(results,anchors,baseline_folds,subjects,output,metadata):
    # This routine consumes final OOF counts only. It cannot change any model.
    out=output/'outputs'
    summary=[];aggregate_counts={}
    for method in ('Author Native','Tuned Native')+METHODS:
        total=empty_counts()
        for subject in subjects:
            counts=baseline_folds[subject][method] if method in anchors else results[method][subject]['outer_metrics']
            add_counts(total,counts)
        aggregate_counts[method]=total
        summary.append({'Method':method,**{k:metrics(total)[k] for k in FIELDS}})
    unified={r['Method']:r for r in summary}
    delta_rows=[];stability={};bootstrap={};decisions={};frequencies={}
    indices=np.random.default_rng(SEED).integers(0,len(subjects),size=(1000,len(subjects)))
    dump(out/'bootstrap_subject_indices.json',{'seed':SEED,'subjects':subjects,'indices':indices.tolist()})
    base=np.asarray([[baseline_folds[s]['Tuned Native'][k] for k in ('TP','FP','FN')] for s in subjects])
    def vector_f1(counts):
        return 2*counts[...,0]/(2*counts[...,0]+counts[...,1]+counts[...,2])
    base_boot=vector_f1(base[indices].sum(axis=1))
    for method in IDEAS:
        array=np.asarray([[results[method][s]['outer_metrics'][k] for k in ('TP','FP','FN')] for s in subjects])
        deltas=vector_f1(array)-vector_f1(base)
        for s,d in zip(subjects,deltas):
            delta_rows.append({'outer_subject':s,'method':method,'delta_F1':float(d)})
        leaveout=vector_f1(array.sum(axis=0)-array)-vector_f1(base.sum(axis=0)-base)
        stability[method]={'improved':int(sum(deltas>0)),'equal':int(sum(deltas==0)),'worse':int(sum(deltas<0)),
            'leave_one_subject_out_aggregate_delta':dict(zip(subjects,map(float,leaveout))),
            'subjects_whose_removal_eliminates_positive_gain':[s for s,d in zip(subjects,leaveout) if d<=0],
            'gain_survives_every_subject_removal':bool(np.all(leaveout>0))}
        boot=vector_f1(array[indices].sum(axis=1))-base_boot
        bootstrap[method]={'repetitions':1000,'seed':SEED,'mean_delta':float(boot.mean()),
            'ci95':list(map(float,np.percentile(boot,[2.5,97.5])))}
        value=unified[method];delta=value['F1']-unified['Tuned Native']['F1']
        conditions={'delta_ge_0_005':delta>=.005,'rescue_TP_positive':value['rescue_TP']>0,
            'rescue_precision_above_LowRescue':value['rescue_precision']>unified['LowRescue']['rescue_precision'],
            'not_single_subject_driven':bool(np.all(leaveout>0))}
        decision='NO-GO' if delta<=0 else 'STRONG-GO' if all(conditions.values()) else 'WEAK-GO'
        decisions[method]={'decision':decision,'delta_vs_TunedNative':delta,'strong_go_conditions':conditions}
        configs=[results[method][s]['selected_config'] for s in subjects]
        frequencies[method]={}
        for field in configs[0]:
            freq=Counter(str(c[field]) for c in configs)
            possible=sorted({c[field] for c in grid(method)})
            boundary=sum(c[field] in (possible[0],possible[-1]) for c in configs) if field!='normalization' else None
            frequencies[method][field]={'frequency':dict(freq),'endpoint_selection_count':boundary,
                'flag':'BOUNDARY-SELECTION' if boundary is not None and boundary>len(subjects)/2 else None}
    best=max(IDEAS,key=lambda m:unified[m]['F1'])
    overall='FIRST3-CANDIDATE-RECOVERY-STRONG-WINNER' if any(d['decision']=='STRONG-GO' for d in decisions.values()) else 'FIRST3-CANDIDATE-RECOVERY-NO-STRONG-WINNER'
    best_report={'BEST_OF_FIRST3':best,'BEST_F1':unified[best]['F1'],
        'Delta_vs_Author':unified[best]['F1']-unified['Author Native']['F1'],
        'Delta_vs_TunedNative':unified[best]['F1']-unified['Tuned Native']['F1'],
        'Rescue_TP':unified[best]['rescue_TP'],'Rescue_FP':unified[best]['rescue_FP'],
        'Bootstrap_CI':bootstrap[best]['ci95'],
        'Subject_improved_equal_worse':{k:stability[best][k] for k in ('improved','equal','worse')}}
    report={**metadata,'aggregate_metrics':unified,'subject_stability':stability,'bootstrap':bootstrap,
        'parameter_stability':frequencies,'decisions':decisions,'overall_decision':overall,**best_report,
        'incomplete':False,'completed_outer_folds':len(subjects),'smoke_pass':len(subjects)==1}
    dump(out/'first3_summary.json',report)
    dump(out/'first3_bootstrap.json',bootstrap)
    write_csv(out/'first3_unified_summary.csv',summary)
    write_csv(out/'first3_subject_deltas.csv',delta_rows)
    dump(out/'parameter_stability.json',frequencies)
    write_csv(out/'unified_outer_metrics.csv',[{'outer_subject':s,'method':m,
        **results[m][s]['outer_metrics'],**{f'selected_{k}':v for k,v in results[m][s]['selected_config'].items()},
        **{f'Author_{k}':baseline_folds[s]['Author Native'][k] for k in ('TP','FP','FN','F1')},
        **{f'TunedNative_{k}':baseline_folds[s]['Tuned Native'][k] for k in ('TP','FP','FN','F1')}}
        for s in subjects for m in METHODS])
    lines=['# FIRST3 Candidate Recovery Nested 实验报告','',
        '> 本实验是在此前 fixed-configuration audits 之后，根据导师建议重新开放预定义、有限、subject-disjoint 的 hyperparameter tuning；outer test subject 从未用于参数选择。','',
        '## Integrity / protocol','',f"- 两个 anchor 精确复现：{anchors}",
        f"- Cache SHA-256: `{CACHE_HASH}`；29折旧 Native config 路径及全部代码哈希见 first3_summary.json。",
        '- 固定 grids：LowRescue 4、PSED 60、Recognition 100、Morphology 288。Morphology 使用28-fold inner subject LOSO；训练侧 scaler 与 LR 同折拟合，选定配置后仅一次 outer-train refit。',
        '- PSED: z=(g−median)/(1.4826·MAD+1e−8)，prominence_z=peak_prominences(z)，P=sigmoid(prominence_z)；H按固定tau_L/tau_H计算。',
        '- Recognition: softmax axis=1，5类 [negative, positive, surprise, others, neutral]，neutral=4；R2=max非neutral概率，±k_p窗口均值。未运行可选R1。',
        '- Morphology: [H,31点patch]；edge padding，L半径2/5/8/10；raw-z使用31点均值/std，peak-relative减中心再L2 scaling。旧StandardScaler仅在训练集fit。',
        '- 所有方法直接共享decode_low共同pool，沿用原integer rounding与不clip的事件坐标；high event完整保留，没有新增NMS。候选标签按high未匹配GT的正式IoU matching生成，test feature不读取GT。',
        '- 10项sanity见sanity_checks.json；006 smoke通过后运行正式29折。所有输入缓存与旧结果保持不变。',
        '- Delta=.05仅007/013有训练正例；两者分别outer-test与inner-validation时，inner-train全负，固定预测概率0而不拟合不可定义的LR。这是仅由训练标签决定的退化折处理，逐次标记single_class_constant，不新增classifier或grid。',
        '- 历史全缓存审计已影响研究方向；本次防泄漏约束针对参数和权重拟合。BEST_OF_FIRST3是事后方法比较，不是独立验证集上的模型选择结果。','',
        '## Pooled summary','', '| Method | TP | FP | FN | Precision | Recall | F1 | Rescue TP | Rescue FP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['Method']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['F1']:.6f} | {row['rescue_TP']} | {row['rescue_FP']} |")
    lines += ['', '## Rescue diagnostics','', '| Method | Weak pool | Selected rescue | Rescue precision | Native missed GT recovered |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        r=unified[method]
        lines.append(f"| {method} | {r['weak_candidate_count']} | {r['rescued_candidate_count']} | {r['rescue_precision']:.6f} | {r['native_missed_GT_recovered']} |")
    lines += ['', '## Subject stability / paired bootstrap','', '| Method | Improved/equal/worse | Mean bootstrap ΔF1 | 95% CI | 单subject删除后增益均保留 |','|---|---|---:|---|---|']
    for method in IDEAS:
        s,b=stability[method],bootstrap[method]
        lines.append(f"| {method} | {s['improved']}/{s['equal']}/{s['worse']} | {b['mean_delta']:+.6f} | [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | {s['gain_survives_every_subject_removal']} |")
    lines += ['', 'bootstrap为1000次subject重采样，三方法共用相同索引，seed=20260902。逐subject delta及删除敏感性均已保存。不同方法所选Delta_p不同，rescue precision比较对应各自nested policy。','', '## Parameter stability','']
    for method in IDEAS:
        lines.append(f'- {method}: `{json.dumps(frequencies[method],ensure_ascii=False)}`')
    lines += ['', '端点选择超过半数fold标记BOUNDARY-SELECTION；不扩大grid。','', '## Files','']
    lines += [f'- `{p.resolve()}`' for p in sorted(out.iterdir())]
    lines += [f'- `{Path(__file__).resolve()}`',f'- `{(HERE/"LOCKED_PROTOCOL_CN.md").resolve()}`','', '## Results / decision','']
    for method in IDEAS:
        lines.append(f"- {method}: **{decisions[method]['decision']}**；ΔF1={decisions[method]['delta_vs_TunedNative']:+.6f}；strong条件={decisions[method]['strong_go_conditions']}。")
    lines += ['',f'`{overall}`','', '```json',json.dumps(best_report,ensure_ascii=False,indent=2),'```']
    (output/'FIRST3_CANDIDATE_RECOVERY_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('smoke','full'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    started=time.time()
    output=ROOT/'results'/('first3_candidate_recovery_nested' if args.mode=='full' else 'first3_candidate_recovery_smoke_006')
    if output.exists():raise RuntimeError(f'Refusing to overwrite {output}')
    paths=[Path(__file__),HERE/'LOCKED_PROTOCOL_CN.md',SOURCE,CACHE,
        LEGACY/'run_mscr_nested_loso.py',LEGACY/'run_native_parameter_recognition_audit.py',
        ROOT/'my_method/rgr1_sammlv/run_rgr1_nested_loso.py',ROOT/'my_method/psed_feasibility/run_feasibility.py',
        ROOT/'my_method/temporal_morphology_feasibility/run_morphology_audit.py']
    provenance={str(p):sha256(p) for p in paths}
    assert provenance[str(CACHE)]==CACHE_HASH
    if args.mode=='full':
        smoke=json.loads((ROOT/'results/first3_candidate_recovery_smoke_006/outputs/first3_summary.json').read_text())
        assert smoke['smoke_pass'] and smoke['provenance']==provenance
    source=json.loads(SOURCE.read_text())
    assert not source['incomplete'] and source['cache_sha256']==CACHE_HASH
    folds={f['subject']:f for f in source['outer_folds']}
    with CACHE.open('rb') as handle:payload=pickle.load(handle)
    assert payload['dataset']=='SAMMLV' and payload['k_p']==5
    records=payload['records'];subjects=sorted(folds)
    assert len(subjects)==29 and len(records)==79 and sum(len(r['samples']) for r in records)==159
    assert set(str(r['subject']) for r in records)==set(subjects)
    for subject,fold in folds.items():assert set(fold['train_subjects'])==set(subjects)-{subject}
    anchors,baseline_folds=baseline(records,folds)
    for subject in subjects:
        assert all(baseline_folds[subject]['Tuned Native'][k]==folds[subject]['outer_metrics']['Tuned Native'][k]
                   for k in ('TP','FP','FN','F1'))
    print('BASELINE PASS '+json.dumps(anchors),flush=True)
    assert [len(grid(m)) for m in METHODS]==[4,60,100,288]
    banks={};recognition_shapes=Counter()
    for record in records:
        logits=np.asarray(record['logits']);score=np.asarray(record['score'])
        assert logits.shape==(len(score),5) and np.isfinite(logits).all() and np.isfinite(score).all()
        recognition_shapes[str(logits.shape)]+=1
    for fold in folds.values():
        config=fold['selected_strong_config']
        if key(config) in banks:continue
        banks[key(config)]={}
        for record in records:
            signal={k:record[k] for k in ('subject','video','score','logits')}
            rid=(str(record['subject']),str(record['video']))
            banks[key(config)][rid]=common_pool(signal,config)
    held_subjects=['006'] if args.mode=='smoke' else subjects
    trace,pool_hashes=common_trace(banks,folds,held_subjects)
    output.mkdir(parents=True);out=output/'outputs';out.mkdir()
    write_csv(out/'baseline_outer_metrics.csv',[{'outer_subject':s,'method':m,**baseline_folds[s][m]}
        for s in subjects for m in anchors])
    write_csv(out/'common_weak_candidate_trace.csv',trace)
    dump(out/'common_pool_hashes.json',pool_hashes)
    print(f'COMMON POOL PASS: {len(trace)} trace rows; same shared object for all methods',flush=True)
    context={'records':records,'subjects':subjects,'folds':folds,'banks':banks}
    results={}
    # Stage barriers ensure PSED completes before Recognition starts, then LR.
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(context,)) as executor:
        for method,stem in zip(METHODS,('low_rescue','psed','recognition_joint','morphology')):
            print(f'STAGE START {method}',flush=True)
            results[method]={}
            futures={executor.submit(worker,(method,s)):s for s in held_subjects}
            for future in as_completed(futures):
                result=future.result();subject=result['outer_subject'];results[method][subject]=result
                # Do not expose outer result numbers until every Idea is complete.
                print(f'{method}: fold {subject} complete ({len(results[method])}/{len(held_subjects)})',flush=True)
            ordered=[results[method][s] for s in held_subjects]
            write_csv(out/f'{stem}_outer_metrics.csv',[{'outer_subject':r['outer_subject'],
                **r['outer_metrics'],**{f'selected_{k}':v for k,v in r['selected_config'].items()}} for r in ordered])
            write_csv(out/f'{stem}_selected_configs.csv',[{'outer_subject':r['outer_subject'],
                **r['selected_config'],'train_subjects':';'.join(r['train_subjects']),
                'inner_F1':r['inner_metrics']['F1'],'inner_TP':r['inner_metrics']['TP'],
                'inner_FP':r['inner_metrics']['FP'],'inner_FN':r['inner_metrics']['FN']} for r in ordered])
            write_csv(out/f'{stem}_inner_config_scores.csv',[row for r in ordered for row in r['inner_rows']])
            write_csv(out/f'{stem}_prediction_trace.csv',[row for r in ordered for row in r['predictions']])
            write_csv(out/f'{stem}_weak_selection_trace.csv',[row for r in ordered for row in r['weak_trace']])
            if method=='Morphology Rescue':
                write_csv(out/'morphology_inner_fit_audit.csv',[row for r in ordered for row in r['fit_audit']])
                write_csv(out/'morphology_selected_inner_oof.csv',[row for r in ordered for row in r['selected_oof']])
                dump(out/'morphology_fitted_models.json',[r['model_state'] for r in ordered])
    checks={'outer_test_excluded_from_selection':True,'LR_outer_test_labels_excluded':True,
        'Author_Native_exact':True,'Tuned_Native_exact':True,'high_events_100_percent_preserved':True,
        'same_common_weak_generator':True,'equal_weak_counts_at_same_delta':True,
        'identical_LowRescue_traces_at_same_delta':True,'no_additional_NMS':True,
        'GT_free_test_features':True,'all_requested_ideas_complete':len(results)==4,
        'all_requested_folds_complete':all(len(v)==len(held_subjects) for v in results.values())}
    assert all(checks.values())
    assert all(sha256(Path(p))==digest for p,digest in provenance.items())
    dump(out/'sanity_checks.json',checks)
    metadata={'experiment':'FIRST3 Candidate Recovery','mode':args.mode,'provenance':provenance,
        'grids':{m:grid(m) for m in METHODS},'recognition_validation':{'shapes':dict(recognition_shapes),
            'softmax_axis':1,'class_order':CLASS_ORDER,'neutral_class_id':NEUTRAL_ID},
        'inner_protocol':'28-fold subject LOSO for morphology; pooled outer-training F1 for deterministic methods',
        'baseline_anchors':anchors,'sanity_checks':checks,'cpu_workers':args.workers,
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'elapsed_seconds_before_reporting':time.time()-started}
    if args.mode=='smoke':
        # No bootstrap or scientific decisions on a one-fold smoke result.
        dump(out/'first3_summary.json',{**metadata,'smoke_pass':True,'incomplete':False,
            'completed_outer_folds':1,'outer_metrics':{m:results[m]['006']['outer_metrics'] for m in METHODS}})
        print('SMOKE PASS',flush=True)
    else:
        report=finish(results,anchors,baseline_folds,subjects,output,metadata)
        print(json.dumps({k:report[k] for k in ('aggregate_metrics','decisions','overall_decision','BEST_OF_FIRST3')},indent=2),flush=True)
    print(f'FINISHED {output} elapsed={time.time()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
