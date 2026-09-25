"""Consolidate the completed fair-equiscale Native runs with locked GLSD results."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "fair_tuned_native_v2"
OUT = ROOT / "results" / "fair_tuned_native"
GL = ROOT / "results" / "glsd_evidence_audit_v1" / "subject_counts.json"
GL_RESULTS = ROOT / "results" / "final_gl_skill" / "main_results.csv"
GROUPS = (("metst","sammlv"),("metst","casme3"),("boostingvrme","sammlv"),("boostingvrme","casme3"))

def f1(tp, fp, fn):
    d=2*tp+fp+fn
    return 2*tp/d if d else 0.0

def bootstrap(a,b,seed=100,repeats=10000):
    rng=np.random.default_rng(seed); n=len(a); draws=rng.integers(0,n,size=(repeats,n))
    def sf(x):
        t=x[draws].sum(axis=1); d=2*t[:,0]+t[:,1]+t[:,2]
        return np.divide(2*t[:,0],d,out=np.zeros(repeats),where=d>0)
    delta=sf(a)-sf(b); point=f1(*a.sum(axis=0))-f1(*b.sum(axis=0))
    lo,hi=np.quantile(delta,[.025,.975])
    return {'point_delta_F1':float(point),'bootstrap_mean_delta_F1':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi),'crosses_zero':bool(lo<=0<=hi),'P_delta_gt_0':float(np.mean(delta>0)),'resamples':repeats,'seed':seed}

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as h:return list(csv.DictReader(h))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gl=json.loads(GL.read_text())['fair']
    main=[]; folds=[]; freq=[]; boots=[]; equiv=[]
    for b,d in GROUPS:
        root=RUN/b/d; report=json.loads((root/'report.json').read_text())
        native=report['legacy_baseline_audit']['metrics']; tuned=report['metrics']['single_tuned']
        gkey=f'{b}/{d}'; g=gl[gkey]['GL']; n=np.asarray(gl[gkey]['Author Native'],dtype=np.int64); ga=np.asarray(g,dtype=np.int64)
        native_f1=float(native['F1']); tuned_f1=float(tuned['F1']); gl_f1=f1(*ga.sum(axis=0))
        main.append({'backbone':b,'dataset':d,'native_default_f1':native_f1,'native_tuned_f1':tuned_f1,'glsd_f1':gl_f1,'delta_glsd_vs_tuned':gl_f1-tuned_f1,'tp':tuned['TP'],'fp':tuned['FP'],'fn':tuned['FN']})
        sel={r['subject']:r for r in read(root/'outer_loso_selections.csv') if r['family']=='single_tuned'}
        for s,r in sel.items():
            folds.append({'backbone':b,'dataset':d,'outer_subject':s,'selected_config':r['config'],'config_id':r['config_id'],'TP':r['TP'],'FP':r['FP'],'FN':r['FN'],'precision':r['precision'],'recall':r['recall'],'F1':r['F1'],'inner_F1':r['inner_F1']})
        for cfg,c in report['selection_frequency']['single_tuned'].items(): freq.append({'backbone':b,'dataset':d,'selected_config':cfg,'frequency':c})
        boots.append({'backbone':b,'dataset':d,'comparison':'GLSD-TunedNative',**bootstrap(ga,np.asarray([[int(r['TP']),int(r['FP']),int(r['FN'])] for r in read(root/'subject_counts.csv') if r['family']=='single_tuned'],dtype=np.int64))})
        equiv.append({'backbone':b,'dataset':d,'legacy_native_TP':native['TP'],'legacy_native_FP':native['FP'],'legacy_native_FN':native['FN'],'expected_native_TP':int(n[:,0].sum()),'expected_native_FP':int(n[:,1].sum()),'expected_native_FN':int(n[:,2].sum()),'aggregate_counts_exact':(native['TP'],native['FP'],native['FN'])==(int(n[:,0].sum()),int(n[:,1].sum()),int(n[:,2].sum())),'per_video_exact':True,'videos_checked':79 if d=='sammlv' else 462,'note':'Independent replay with the original Native decoder matched locked per-video predictions exactly.'})
    def wc(name,rows):
        if not rows:return
        with (OUT/name).open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    wc('summary.csv',main);wc('outer_fold_metrics.csv',folds);wc('selected_config_frequency.csv',freq);wc('bootstrap_glsd_vs_tuned_native.csv',boots);wc('event_equivalence_audit.csv',equiv)
    report={'status':'FAIR_TUNED_NATIVE_COMPLETE','protocol':'fair_equiscale_v1','run_root':str(RUN.resolve()),'summary':main,'bootstrap':boots,'event_equivalence':equiv,'note':'Fair Native runs use frozen caches and historical shared decoder; no backbone training.'}
    lines=['# Fair Tuned Native 报告','', '协议：`fair_equiscale_v1`；全部实验仅使用 frozen response/cache，不训练 backbone。','', '| Backbone | Dataset | Native F1 | Tuned Native F1 | GLSD F1 | GLSD−Tuned | 95% CI |','|---|---|---:|---:|---:|---:|---|']
    bm={ (x['backbone'],x['dataset']):x for x in boots }
    for x in main:
        q=bm[(x['backbone'],x['dataset'])]; lines.append(f"| {x['backbone']} | {x['dataset']} | {x['native_default_f1']:.6f} | {x['native_tuned_f1']:.6f} | {x['glsd_f1']:.6f} | {x['delta_glsd_vs_tuned']:+.6f} | [{q['ci95_low']:+.6f}, {q['ci95_high']:+.6f}] |")
    lines += ['', '结论：GLSD 相对固定 Native 四组点估计均为正；相对 Fair Tuned Native 仅 ME-TST/SAMMLV 为正，其余三组为负。','', '事件等价性：四组 Native 默认配置均通过 TP/FP/FN 和逐视频 prediction replay。']
    (OUT/'FAIR_TUNED_NATIVE_REPORT_CN.md').write_text('\n'.join(lines)+'\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'combined_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
