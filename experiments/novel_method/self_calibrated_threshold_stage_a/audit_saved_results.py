#!/usr/bin/env python3
"""Validate Stage A saved predictions, held-out exclusion and paired statistics."""
import csv
import json
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks

import run_stage_a as stage


def read_csv(name):
    with (stage.OUT/"outputs"/name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    output=stage.OUT/"outputs"
    summary=json.loads((output/"quantile_stage_a_summary.json").read_text())
    boots=json.loads((output/"quantile_bootstrap.json").read_text())
    selected=read_csv("quantile_selected_q.csv")
    traces=read_csv("quantile_peak_survival.csv")
    configs,_=stage.selected_sources()
    data={}
    for name,cfg in stage.DATASETS.items():
        _,records,subjects,_=stage.native.load_payload(cfg["cache"])
        data[name]=(records,subjects,{(str(r['subject']),str(r['video'])):i for i,r in enumerate(records)})
    for path,digest in summary['input_sha256'].items():
        assert stage.native.sha256(stage.Path(path))==digest

    @lru_cache(None)
    def independent(name,config_tuple,index,q):
        record=data[name][0][index]
        cs,p,cd,cb=config_tuple
        kp=stage.DATASETS[name]['k_p']
        width,dist,radius=[max(1,round(v*kp)) for v in (cs,cd,cb)]
        g=np.convolve(np.asarray(record['score'],dtype=np.float64),np.ones(width)/width,mode='same')
        legal=find_peaks(g,distance=dist)[0]
        fallback=q is not None and len(legal)<2
        tau=float(np.quantile(g[legal],q,method='linear')) if q is not None and not fallback else float(g.mean()+p*(g.max()-g.mean()))
        # Independent route: run the existing detector with the derived height.
        keep=find_peaks(g,height=tau,distance=dist)[0]
        events=[stage.native.event(peak,radius,'audit') for peak in keep]
        return tau,legal,keep,events,stage.native.evaluate(record,events),fallback

    for row in selected:
        name,held=row['dataset'],row['outer_subject']
        records,subjects,_=data[name]
        config=tuple(float(row[k]) for k in ('c_s','p','c_d','c_b'))
        assert config==stage.config_id(configs[name][held])
        assert set(json.loads(row['train_subjects']))==set(subjects)-{held}
        stored=json.loads(row['all_q_train_metrics'])
        assert tuple(x['q'] for x in stored)==stage.Q_VALUES
        # Re-sum by subtracting held-out counts from totals, unlike the run's
        # explicit train-index iteration. No different q or setting is evaluated.
        reproduced=[]
        for q,expected in zip(stage.Q_VALUES,stored):
            total=Counter(); held_total=Counter()
            for i,record in enumerate(records):
                counts=independent(name,config,i,q)[4]
                total.update(counts)
                if str(record['subject'])==held:
                    held_total.update(counts)
            train={k:total[k]-held_total[k] for k in stage.COUNT_KEYS}
            assert all(train[k]==expected[k] for k in stage.COUNT_KEYS)
            assert abs(stage.native.f1(train)-expected['F1'])<1e-14
            reproduced.append(train)
        best=min(range(6),key=lambda i:stage.q_rank(reproduced[i],i))
        assert float(row['selected_q'])==stage.Q_VALUES[best]

    grouped={}
    for row in traces:
        name,held,method=row['dataset'],row['outer_subject'],row['method']
        index=data[name][2][held,row['video']]
        config=stage.AUTHOR_CONFIG if method==stage.METHODS[0] else configs[name][held]
        q=float(row['selected_q']) if row['selected_q'] else None
        tau,legal,kept,events,counts,fallback=independent(name,stage.config_id(config),index,q)
        assert abs(tau-float(row['threshold']))<1e-14
        assert len(legal)==int(row['legal_peak_count'])
        assert kept.tolist()==json.loads(row['peak_times'])
        assert [[e['onset'],e['offset']] for e in events]==json.loads(row['intervals'])
        assert (row['fallback_to_native']=='True')==fallback
        assert all(counts[k]==int(row[k]) for k in stage.COUNT_KEYS)
        grouped.setdefault((name,held,method),Counter()).update(counts)
    assert len(traces)==1623 and len(selected)==123

    for name,(_,subjects,_) in data.items():
        rows=read_csv(f'quantile_outer_metrics_{name.lower()}.csv')
        assert len(rows)==3*(len(subjects)+1)
        for row in rows:
            if row['row_type']=='outer_subject':
                assert all(grouped[name,row['outer_subject'],row['method']][k]==int(row[k]) for k in stage.COUNT_KEYS)
        matrices=[]
        indices=np.asarray(boots[name]['sampled_subject_indices'])
        assert indices.shape==(1000,len(subjects))
        for method in stage.METHODS[1:]:
            matrix=np.asarray([[grouped[name,s,method][k] for k in ('TP','FP','FN')] for s in subjects])
            matrices.append(matrix)
        def f1(x):
            den=2*x[...,0]+x[...,1]+x[...,2]
            return np.divide(2*x[...,0],den,out=np.zeros_like(den,dtype=float),where=den!=0)
        delta=f1(matrices[1][indices].sum(axis=1))-f1(matrices[0][indices].sum(axis=1))
        np.testing.assert_allclose(delta,boots[name]['replicate_delta_F1'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(np.quantile(delta,[.025,.975]),boots[name]['ci95'],rtol=0,atol=1e-14)
        for i,record in enumerate(summary['datasets'][name]['leave_one_subject_out']):
            assert record['omitted_subject']==subjects[i]
            value=f1(matrices[1].sum(axis=0)-matrices[1][i])-f1(matrices[0].sum(axis=0)-matrices[0][i])
            assert abs(value-record['delta_F1'])<1e-14
    assert stage.decision(summary['datasets'],boots)[0]==summary['decision']
    result=dict(status='PASS', checks={
        'input_files_unchanged':True, 'all_123_frozen_configs_and_splits':True,
        'all_738_train_q_counts_and_selections':True, 'all_1623_saved_video_predictions':True,
        'outer_counts_conserved':True, 'paired_1000_bootstrap_replayed_per_dataset':True,
        'all_123_leave_one_subject_out_deltas':True, 'decision_matches_evidence':True})
    stage.dump(stage.OUT/'SAVED_RESULTS_AUDIT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
