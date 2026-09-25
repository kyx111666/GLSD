#!/usr/bin/env python3
"""ENPD V1: nested-LOSO empirical-null primitive peak screening."""
import csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/enpd_v1"
sys.path.insert(0,str(ROOT))
import run_boundary_phase0_audit as native
import run_dense_decoder_phase0 as dense

TAUS=(-.5,0.,.5,1.,1.5,2.,3.)
QUANTILES=(.70,.80,.85,.90,.95,.975)
ENPD_Q=(.01,.02,.05,.10,.20)
METHODS=("Native","Height","Quantile","ENPD")

def write_csv(path,rows):
    if not rows: rows=[{"note":"no rows"}]
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def f1(c): return 2*c[0]/(2*c[0]+c[1]+c[2]) if 2*c[0]+c[1]+c[2] else 0.
def precision(c): return c[0]/(c[0]+c[1]) if c[0]+c[1] else 0.
def add(a,b): return tuple(a[i]+b[i] for i in range(3))

def choose(grid,counts):
    best_i=0; best=(-1.,-1.,float("-inf"))
    for i,c in enumerate(counts):
        key=(f1(c),precision(c),-c[1])
        if key>best: best=key; best_i=i
    return grid[best_i]

def decode(video,selected):
    peaks=np.asarray(video["primitive"],int)[np.asarray(selected,bool)]
    heights=np.asarray(video["heights"],float)[np.asarray(selected,bool)]
    if video["backbone"]=="ME-TST":
        return [{"onset":max(0,int(p)-video["k"]),"offset":min(video["T"]-1,int(p)+video["k"]),"peak":int(p)} for p in sorted(peaks)]
    candidates=[]
    for p,h in zip(peaks,heights):
        on,off=native.fair.source_interval(video["raw_curve"],int(p),video["k"])
        candidates.append((int(on),int(off),int(p),float(h)))
    candidates.sort(key=lambda x:(x[0],x[2])); kept=[]
    for cand in candidates:
        conflict=False
        for old in kept:
            inter=max(0,min(cand[1],old[1])-max(cand[0],old[0])); span=max(cand[1],old[1])-min(cand[0],old[0]); ov=inter/span if span else 0.
            if ov>=.2 or abs(cand[2]-old[2])<=video["k"]: conflict=True; break
        if not conflict: kept.append(cand)
    return [{"onset":a,"offset":b,"peak":p} for a,b,p,_ in kept]

def metric_counts(pred,gt):
    pairs=native.match(pred,gt); tp=len(pairs); return (tp,len(pred)-tp,len(gt)-tp),pairs

def rank_percentile(h):
    h=np.asarray(h,float); order=np.sort(h); return np.searchsorted(order,h,side="right")/len(h) if len(h) else h

def tail_counts(sorted_values,h):
    h=np.asarray(h,float); return len(sorted_values)-np.searchsorted(sorted_values,h,side="left")

def empirical_pvalues(h,total_sorted,excluded_sorted):
    count=tail_counts(total_sorted,h).astype(np.int64); n=len(total_sorted)
    for arr in excluded_sorted: count-=tail_counts(arr,h); n-=len(arr)
    return (1+count)/(n+1),n

def bh_mask(pvals,q):
    p=np.asarray(pvals,float); m=len(p)
    if not m:return np.zeros(0,bool)
    order=np.argsort(p,kind="stable"); passed=p[order]<=q*np.arange(1,m+1)/m
    if not passed.any():return np.zeros(m,bool)
    cutoff=p[order[np.flatnonzero(passed)[-1]]]; return p<=cutoff

def bootstrap(videos,a,b,n=10000,seed=100):
    subjects=sorted({v["subject"] for v in videos}); idx={s:i for i,s in enumerate(subjects)}
    ca=np.zeros((len(subjects),3),int); cb=np.zeros_like(ca)
    for v in videos:
        for arr,m in ((ca,a),(cb,b)):
            c,_=metric_counts(v["predictions"][m],v["gt"]); arr[idx[v["subject"]]]+=c
    rng=np.random.default_rng(seed); draws=rng.integers(0,len(subjects),size=(n,len(subjects)))
    aa=ca[draws].sum(axis=1); bb=cb[draws].sum(axis=1)
    fa=2*aa[:,0]/np.maximum(1,2*aa[:,0]+aa[:,1]+aa[:,2]); fb=2*bb[:,0]/np.maximum(1,2*bb[:,0]+bb[:,1]+bb[:,2]); d=fa-fb
    point=f1(tuple(ca.sum(axis=0)))-f1(tuple(cb.sum(axis=0))); lo,hi=np.quantile(d,[.025,.975])
    return {"point_estimate":point,"bootstrap_mean":float(d.mean()),"ci95_low":float(lo),"ci95_high":float(hi),"ci_crosses_zero":bool(lo<=0<=hi),"n_resamples":n,"seed":seed}

def entropy(values):
    c=Counter(values); p=np.asarray(list(c.values()),float)/len(values); h=float(-(p*np.log(p)).sum()); return h,h/math.log(len(c)) if len(c)>1 else 0.,c.most_common(1)[0][1]/len(values)

def prepare_case(bb,ds,cache_path,target_path,pmf):
    k,source=native.group_audit(bb,ds,cache_path,target_path,pmf); videos=[]; exact=True; baseline=[0,0,0]
    for subject,video,gt,native_pred,cands,vk,source_curve,threshold,expected in source:
        exp=expected["predictions"].get("original_native",expected["predictions"].get("native",[]))
        exact &= sorted((p["onset"],p["offset"],p["peak"]) for p in native_pred)==sorted((p["onset"],p["offset"],p["peak"]) for p in exp)
        c,_=metric_counts(native_pred,gt)
        for i in range(3):baseline[i]+=c[i]
        raw=np.asarray(source_curve,float); response=np.convolve(raw,np.ones(2*k)/(2*k),mode="same") if bb=="BoostingVRME" else raw
        primitive=dense.primitive_metst(response) if bb=="ME-TST" else native.fair.find_peaks(response,distance=1)[0].astype(int)
        z=dense.z_curve(response); heights=z[primitive]; final_selected=[c for c in cands if c["selected"]]; matched={j for _,j in native.match(native_pred,gt)}; gen=set()
        for gi,g in enumerate(gt):
            if gi in matched:continue
            if any(g[0]<=p["peak"]<=g[2] for p in final_selected):continue
            if any(g[0]<=p["peak"]<=g[2] for p in cands):continue
            gen.add(gi)
        null=np.asarray([h for p,h in zip(primitive,heights) if not any(g[0]<=p<=g[2] for g in gt)],float)
        videos.append({"backbone":bb,"dataset":ds,"subject":str(subject),"video":str(video),"gt":gt,"generation_miss_gt":gen,"k":k,"T":len(response),"raw_curve":np.asarray(source_curve,float),"primitive":primitive,"heights":heights,"null":null,"native_candidates":len(cands),"predictions":{"Native":native_pred}})
    return k,videos,exact,tuple(baseline)

def main():
    OUT.mkdir(parents=True,exist_ok=True); pmf=native.load_pmf(); all_cases=[]; provenance=[]; null_audit=[]; config_rows=[]; volume_rows=[]; result_rows=[]; mechanism_rows=[]; bootstrap_rows=[]; prediction_rows=[]
    for bb,ds,cache_path,target_path in native.CASES:
        k,videos,exact,baseline=prepare_case(bb,ds,cache_path,target_path,pmf); subjects=sorted({v["subject"] for v in videos}); by_subject=defaultdict(list)
        for v in videos:by_subject[v["subject"]].append(v)
        subject_null={s:np.sort(np.concatenate([v["null"] for v in by_subject[s]]) if by_subject[s] else np.array([])) for s in subjects}; total_null=np.sort(np.concatenate(list(subject_null.values())))
        for v in videos: provenance.append({"backbone":bb,"dataset":ds,"subject":v["subject"],"video":v["video"],"k":k,"T":v["T"],"primitive_count":len(v["primitive"]),"primitive_peaks":[int(x) for x in v["primitive"]],"normalized_peak_heights":[float(x) for x in v["heights"]],"curve_source":str(cache_path),"semantics":"historical Native 2*k response; historical plateau semantics; distance=1; no height gate"})
        # Fixed-grid subject-level counts for leakage-free outer selection of B1/B2.
        fixed_predictions={"Height":{},"Quantile":{}}
        fixed_subject_counts={"Height":{},"Quantile":{}}
        for method,grid in (("Height",TAUS),("Quantile",QUANTILES)):
            for cfg in grid:
                fixed_subject_counts[method][cfg]={s:(0,0,0) for s in subjects}; fixed_predictions[method][cfg]={}
                for v in videos:
                    mask=v["heights"]>=cfg if method=="Height" else rank_percentile(v["heights"])>=cfg
                    pred=decode(v,mask); fixed_predictions[method][cfg][(v["subject"],v["video"])]=(pred,int(mask.sum()))
                    c,_=metric_counts(pred,v["gt"]); fixed_subject_counts[method][cfg][v["subject"]]=add(fixed_subject_counts[method][cfg][v["subject"]],c)
        for outer in subjects:
            training=[s for s in subjects if s!=outer]
            selected={}
            for method,grid in (("Height",TAUS),("Quantile",QUANTILES)):
                pooled=[]
                for cfg in grid:
                    c=(0,0,0)
                    for s in training:c=add(c,fixed_subject_counts[method][cfg][s])
                    pooled.append(c)
                selected[method]=choose(grid,pooled)
            # Nested training-subject-excluded calibration for ENPD q selection.
            q_counts=[(0,0,0) for _ in ENPD_Q]
            inner_null_sizes=[]
            for inner in training:
                excluded=[subject_null[outer],subject_null[inner]]; subject_size=len(total_null)-len(subject_null[outer])-len(subject_null[inner]); inner_null_sizes.append(subject_size)
                for v in by_subject[inner]:
                    pv,_=empirical_pvalues(v["heights"],total_null,excluded)
                    for qi,q in enumerate(ENPD_Q): q_counts[qi]=add(q_counts[qi],metric_counts(decode(v,bh_mask(pv,q)),v["gt"])[0])
            selected["ENPD"]=choose(ENPD_Q,q_counts)
            outer_null_size=len(total_null)-len(subject_null[outer]); null_audit.append({"backbone":bb,"dataset":ds,"outer_subject":outer,"null_peak_count":outer_null_size,"training_subject_count":len(training),"training_video_count":sum(len(by_subject[s]) for s in training),"inner_excluded_null_min":min(inner_null_sizes) if inner_null_sizes else 0,"inner_excluded_null_median":float(np.median(inner_null_sizes)) if inner_null_sizes else 0,"inner_excluded_null_max":max(inner_null_sizes) if inner_null_sizes else 0})
            config_rows.append({"backbone":bb,"dataset":ds,"outer_subject":outer,"height_tau":selected["Height"],"quantile_q":selected["Quantile"],"enpd_q":selected["ENPD"],"outer_null_size":outer_null_size})
            for v in by_subject[outer]:
                for method in ("Height","Quantile"):
                    pred,post_gate=fixed_predictions[method][selected[method]][(v["subject"],v["video"])]; v["predictions"][method]=pred; v[method+"_post_gate"]=post_gate
                pv,_=empirical_pvalues(v["heights"],total_null,[subject_null[outer]]); mask=bh_mask(pv,selected["ENPD"]); v["predictions"]["ENPD"]=decode(v,mask); v["ENPD_post_gate"]=int(mask.sum())
        # Evaluate group and mechanism.
        totals={m:(0,0,0) for m in METHODS}
        for v in videos:
            for method in METHODS: totals[method]=add(totals[method],metric_counts(v["predictions"][method],v["gt"])[0])
            native_pairs=metric_counts(v["predictions"]["Native"],v["gt"])[1]; native_gt={j for _,j in native_pairs}; native_fp={v["predictions"]["Native"][i]["peak"] for i in range(len(v["predictions"]["Native"])) if i not in {x for x,_ in native_pairs}}
            for method in ("Height","Quantile","ENPD"):
                pairs=metric_counts(v["predictions"][method],v["gt"])[1]; mgt={j for _,j in pairs}; matched_i={i for i,_ in pairs}; mfp={v["predictions"][method][i]["peak"] for i in range(len(v["predictions"][method])) if i not in matched_i}
                mechanism_rows.append({"backbone":bb,"dataset":ds,"method":method,"subject":v["subject"],"video":v["video"],"retained_native_TP":len(native_gt&mgt),"rescued_GT":len(mgt-native_gt),"lost_native_GT":len(native_gt-mgt),"removed_native_FP":len(native_fp-mfp),"new_FP":len(mfp-native_fp),"rescued_generation_miss_gt":len(mgt&v["generation_miss_gt"])})
            for method in METHODS:
                for p in v["predictions"][method]: prediction_rows.append({"backbone":bb,"dataset":ds,"subject":v["subject"],"video":v["video"],"method":method,"onset":p["onset"],"offset":p["offset"],"peak":p["peak"]})
        n=totals["Native"]
        result_rows.append({"Backbone":bb,"Dataset":ds,"Native_TP":n[0],"Native_FP":n[1],"Native_FN":n[2],"Native_F1":f1(n),**{f"{m}_{key}":val for m in ("Height","Quantile","ENPD") for key,val in zip(("TP","FP","FN","F1","Delta_F1"),(*totals[m],f1(totals[m]),f1(totals[m])-f1(n)))}})
        for method in METHODS:
            primitive=sum(len(v["primitive"]) for v in videos)
            post_gate=sum((v["native_candidates"] if method=="Native" else v[method+"_post_gate"]) for v in videos)
            post_conflict=sum(len(v["predictions"][method]) for v in videos)
            volume_rows.append({"backbone":bb,"dataset":ds,"method":method,"primitive_peak_count":primitive,"post_gate_peak_count":post_gate,"post_conflict_prediction_count":post_conflict,"post_conflict_over_native_predictions":post_conflict/sum(len(v["predictions"]["Native"]) for v in videos)})
        for a,b in (("Height","Native"),("Quantile","Native"),("ENPD","Native"),("ENPD","Height"),("ENPD","Quantile")):
            bootstrap_rows.append({"backbone":bb,"dataset":ds,"comparison":f"{a}-{b}",**bootstrap(videos,a,b)})
        all_cases.append({"backbone":bb,"dataset":ds,"native_exact_replay":exact and baseline==n,"baseline":n,"primitive_count":sum(len(v["primitive"]) for v in videos),"null_pool_min":min(r["null_peak_count"] for r in null_audit if r["backbone"]==bb and r["dataset"]==ds),"null_pool_median":float(np.median([r["null_peak_count"] for r in null_audit if r["backbone"]==bb and r["dataset"]==ds])),"null_pool_max":max(r["null_peak_count"] for r in null_audit if r["backbone"]==bb and r["dataset"]==ds)})
    if not all(c["native_exact_replay"] for c in all_cases):
        report={"status":"ENPD_V1_NATIVE_REPLAY_MISMATCH","cases":all_cases}; (OUT/"combined_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return
    write_csv(OUT/"enpd_v1_results.csv",result_rows); write_csv(OUT/"enpd_v1_mechanism.csv",mechanism_rows); write_csv(OUT/"enpd_v1_candidate_volume.csv",volume_rows); write_csv(OUT/"enpd_v1_bootstrap.csv",bootstrap_rows); write_csv(OUT/"enpd_v1_selected_configs.csv",config_rows); write_csv(OUT/"enpd_v1_predictions.csv",prediction_rows)
    (OUT/"primitive_candidate_provenance.json").write_text(json.dumps({"candidate_identity":"deterministic historical primitive extrema","videos":provenance},ensure_ascii=False,indent=2)); (OUT/"null_pool_audit.json").write_text(json.dumps({"wording":"BH-style empirical-null screening; no theoretical FDR claim","outer_folds":null_audit},ensure_ascii=False,indent=2))
    stability=[]
    for bb,ds in {(r["backbone"],r["dataset"]) for r in config_rows}:
        rows=[r for r in config_rows if r["backbone"]==bb and r["dataset"]==ds]
        for field in ("height_tau","quantile_q","enpd_q"):
            vals=[r[field] for r in rows]; h,hn,mc=entropy(vals); stability.append({"backbone":bb,"dataset":ds,"config":field,"distribution":dict(Counter(map(str,vals))),"entropy_nats":h,"normalized_entropy":hn,"most_common_proportion":mc})
    mech_summary=[]
    for bb,ds,m in sorted({(r["backbone"],r["dataset"],r["method"]) for r in mechanism_rows}):
        rows=[r for r in mechanism_rows if (r["backbone"],r["dataset"],r["method"])==(bb,ds,m)]; mech_summary.append({"backbone":bb,"dataset":ds,"method":m,**{k:sum(r[k] for r in rows) for k in ("retained_native_TP","rescued_GT","lost_native_GT","removed_native_FP","new_FP","rescued_generation_miss_gt")}})
    wins_native=sum(r["ENPD_Delta_F1"]>0 for r in result_rows); wins_height=sum(r["ENPD_F1"]>r["Height_F1"] for r in result_rows); ge_quant=sum(r["ENPD_F1"]>=r["Quantile_F1"] for r in result_rows); strong=wins_native==4 and wins_height>=3 and ge_quant>=3 and sum(r["ENPD_Delta_F1"]>=.01 for r in result_rows)>=2
    negatives=[r["ENPD_Delta_F1"] for r in result_rows if r["ENPD_Delta_F1"]<0]; go=wins_native>=3 and (not negatives or min(negatives)>-.003) and sum(r["ENPD_F1"]>r["Height_F1"] and r["ENPD_F1"]>r["Quantile_F1"] for r in result_rows)>=2
    cas=[r for r in result_rows if r["Dataset"]=="CASME3"]; borderline=wins_native==2 and all(r["ENPD_Delta_F1"]>=.01 for r in cas) and all(m["new_FP"]<3*m["rescued_generation_miss_gt"]+1 for m in mech_summary if m["method"]=="ENPD" and m["dataset"]=="CASME3")
    decision="STRONG_GO" if strong else "GO" if go else "BORDERLINE" if borderline else "STOP"
    files={p.name:str(p) for p in OUT.iterdir() if p.is_file()}; files["combined_report.json"]=str(OUT/"combined_report.json")
    report={"status":"ENPD_V1_NATIVE_REPLAY_EXACT","decision":decision,"wording":"BH-style empirical-null screening; no theoretical FDR-control claim","trainable_model_parameters":0,"v2_not_designed":True,"cases":all_cases,"results":result_rows,"mechanism_summary":mech_summary,"candidate_volume":volume_rows,"bootstrap":bootstrap_rows,"configuration_stability":stability,"enpd_wins_vs_native":wins_native,"enpd_wins_vs_height":wins_height,"enpd_ge_quantile":ge_quant,"files":files}
    (OUT/"combined_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
