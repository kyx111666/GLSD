#!/usr/bin/env python3
"""Dense Temporal Decoding Phase 0: generation-miss opportunity audit only."""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/dense_decoder_phase0"
sys.path.insert(0,str(ROOT))
import run_boundary_phase0_audit as native

def write_csv(path,rows):
    if not rows: rows=[{"note":"no rows"}]
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def sha_curve(x): return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()

def primitive_metst(x):
    peaks=[]; i=1
    while i<len(x)-1:
        if x[i]>x[i-1]:
            left=i
            while i+1<len(x) and x[i]==x[i+1]: i+=1
            if i<len(x)-1 and x[i]>x[i+1]: peaks.append((left+i)//2)
        i+=1
    return np.asarray(peaks,dtype=int)

def empirical_percentiles(video,values):
    ordered=np.sort(np.asarray(video,float)); n=len(ordered)
    return np.asarray([np.searchsorted(ordered,v,side="right")/n for v in values]) if n else np.array([])

def z_curve(x):
    med=float(np.median(x)); mad=float(np.median(np.abs(x-med))); return (x-med)/(mad+1e-12)

def signal_features(x,z,on,off):
    on=max(0,int(on)); off=min(len(x)-1,int(off))
    if on>off: return {"max_z":None,"mean_z":None,"median_z":None,"positive_z_area":None,"max_raw_percentile":None,"mean_raw_percentile":None}
    seg=z[on:off+1]; raw=x[on:off+1]; pct=empirical_percentiles(x,raw)
    return {"max_z":float(np.max(seg)),"mean_z":float(np.mean(seg)),"median_z":float(np.median(seg)),"positive_z_area":float(np.maximum(seg,0).sum()),"max_raw_percentile":float(np.max(pct)),"mean_raw_percentile":float(np.mean(pct))}

def contrast_features(x,on,off,k):
    on=max(0,int(on)); off=min(len(x)-1,int(off))
    if on>off: return {"max_minus_median_bg":None,"mean_minus_mean_bg":None,"mean_minus_median_bg":None,"local_normalized_contrast":None}
    lo=max(0,on-k); hi=min(len(x)-1,off+k); inside=x[on:off+1]
    bg=np.concatenate((x[lo:on],x[off+1:hi+1]))
    if not len(bg): return {"max_minus_median_bg":None,"mean_minus_mean_bg":None,"mean_minus_median_bg":None,"local_normalized_contrast":None}
    d1=float(np.max(inside)-np.median(bg)); d2=float(np.mean(inside)-np.mean(bg)); d3=float(np.mean(inside)-np.median(bg))
    context=x[lo:hi+1]; mad=float(np.median(np.abs(context-np.median(context))))
    return {"max_minus_median_bg":d1,"mean_minus_mean_bg":d2,"mean_minus_median_bg":d3,"local_normalized_contrast":d3/(mad+1e-12)}

def gt_overlaps(start,end,gt): return any(max(start,g[0])<=min(end,g[2]) for g in gt)

def controls_for(gt_event,all_gt,T,rng,limit=3):
    duration=gt_event[2]-gt_event[0]+1
    if duration<=0 or duration>T: return []
    valid=[s for s in range(0,max(0,T-duration+1)) if not gt_overlaps(s,s+duration-1,all_gt)]
    if not valid: return []
    chosen=rng.choice(valid,size=min(limit,len(valid)),replace=False)
    return [(int(s),int(s+duration-1)) for s in sorted(chosen)]

def roc_auc(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); pos=int(y.sum()); neg=len(y)-pos
    if not pos or not neg:return None
    order=np.argsort(s); ranks=np.empty(len(s),float); i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        ranks[order[i:j]]=(i+j+1)/2; i=j
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def average_precision(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); pos=int(y.sum())
    if not pos:return None
    order=np.argsort(-s,kind="stable"); yy=y[order]; return float(np.sum(np.cumsum(yy)[yy==1]/(np.flatnonzero(yy==1)+1))/pos)

def summary(values):
    a=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    if not len(a): return {"n":0,"mean":None,"median":None,"q1":None,"q3":None,"p10":None,"p90":None}
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"q1":float(np.quantile(a,.25)),"q3":float(np.quantile(a,.75)),"p10":float(np.quantile(a,.10)),"p90":float(np.quantile(a,.90))}

def main():
    OUT.mkdir(parents=True,exist_ok=True); pmf=native.load_pmf()
    provenance=[]; recover=[]; losses=[]; sig_rows=[]; contrast_rows=[]; summaries=[]; ranks=[]; oracle_rows=[]; distributions=[]; groups=[]
    for bb,ds,cache_path,target_path in native.CASES:
        k,videos=native.group_audit(bb,ds,cache_path,target_path,pmf); file_sha=sha_file(cache_path); rng=np.random.default_rng(100)
        group_exact=True; group_counts=[0,0,0]; target_counts=[0,0,0]; gen_count=0; gen_inside=0; stage_counts=Counter(); affected=defaultdict(int); affected_v=set(); all_video=[]
        for subject,video,gt,preds,cands,vk,source_curve,threshold,expected in videos:
            exp=expected["predictions"].get("original_native",expected["predictions"].get("native",[]))
            if sorted((p["onset"],p["offset"],p["peak"]) for p in preds)!=sorted((p["onset"],p["offset"],p["peak"]) for p in exp): group_exact=False
            pairs=native.match(preds,gt); matched={j for _,j in pairs}; tp=len(pairs); c=(tp,len(preds)-tp,len(gt)-tp); tc=(sum(int(p.get("matched_gt",-1))>=0 for p in exp),len(exp)-sum(int(p.get("matched_gt",-1))>=0 for p in exp),len(gt)-sum(int(p.get("matched_gt",-1))>=0 for p in exp))
            for j in range(3): group_counts[j]+=c[j]; target_counts[j]+=tc[j]
            x=np.asarray(source_curve,float)
            if bb=="BoostingVRME": x=np.convolve(x,np.ones(2*k)/(2*k),mode="same")
            primitive=primitive_metst(x) if bb=="ME-TST" else native.fair.find_peaks(x,distance=1)[0].astype(int)
            if bb=="ME-TST":
                thresh=float(x.mean()+.55*(x.max()-x.mean())); distance_survivors=pmf.local_maxima(x,-np.inf,k); gate_survivors=np.asarray([p for p in primitive if x[p]>=thresh],int); native_peaks=pmf.local_maxima(x,thresh,k)
            else:
                thresh=float(x.mean()+.55*(x.max()-x.mean())); distance_survivors=native.fair.find_peaks(x,distance=k)[0].astype(int); gate_survivors=np.asarray([p for p in distance_survivors if x[p]>=thresh],int); native_peaks=gate_survivors
            selected=[c for c in cands if c["selected"]]; pre=cands; z=z_curve(x)
            provenance.append({"backbone":bb,"dataset":ds,"subject":str(subject),"video":str(video),"curve_source":str(cache_path),"cache_sha256":file_sha,"curve_sha256_float64":sha_curve(x),"T":len(x),"native_k":k,"native_candidate_peaks":[int(p) for p in native_peaks],"native_final_peaks":[int(p["peak"]) for p in preds],"response":"historical Native 2*k smoothed temporal response"})
            gt_types=[]
            for gi,g in enumerate(gt):
                if gi in matched: typ="NATIVE_MATCHED"
                elif any(g[0]<=p["peak"]<=g[2] for p in selected): typ="BOUNDARY_MISS_STRICT"
                elif any(g[0]<=p["peak"]<=g[2] for p in pre): typ="SELECTION_MISS_STRICT"
                else: typ="GENERATION_MISS_STRICT"
                gt_types.append(typ)
                sf=signal_features(x,z,int(g[0]),int(g[2])); cf=contrast_features(x,int(g[0]),int(g[2]),k)
                sig_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"gt_id":gi,"gt_type":typ,"onset":g[0],"apex":g[1],"offset":g[2],**sf})
                contrast_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"gt_id":gi,"gt_type":typ,"context_k":k,**cf})
                if typ!="GENERATION_MISS_STRICT": continue
                gen_count+=1
                flags={f"within_{frac:.2f}k":bool(np.any((primitive>=max(0,g[0]-frac*k))&(primitive<=min(len(x)-1,g[2]+frac*k)))) for frac in (0,.25,.5,1.)}
                inside=flags["within_0.00k"]; gen_inside+=int(inside)
                recover.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"gt_id":gi,"onset":g[0],"apex":g[1],"offset":g[2],"k":k,"primitive_peak_inside":int(inside),"primitive_peak_within_0.25k":int(flags["within_0.25k"]),"primitive_peak_within_0.50k":int(flags["within_0.50k"]),"primitive_peak_within_1.00k":int(flags["within_1.00k"]),"primitive_peaks_inside":";".join(map(str,primitive[(primitive>=g[0])&(primitive<=g[2])]))})
                if inside:
                    affected[str(subject)]+=1; affected_v.add((str(subject),str(video)))
                    if bb=="ME-TST":
                        if np.any((gate_survivors>=g[0])&(gate_survivors<=g[2])): stage="distance_suppression"
                        else: stage="threshold_gate"
                    else:
                        if np.any((distance_survivors>=g[0])&(distance_survivors<=g[2])): stage="threshold_gate"
                        else: stage="distance_suppression"
                    stage_counts[stage]+=1; losses.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"gt_id":gi,"loss_stage":stage,"stage_order":"threshold_then_distance" if bb=="ME-TST" else "distance_then_threshold","primitive_inside_count":int(np.sum((primitive>=g[0])&(primitive<=g[2]))),"threshold":thresh,"boundary_validity_is_candidate_stage":False,"smoothing_scale_changed":False,"plateau_semantics_changed":False})
                for con,onoff in enumerate(controls_for(g,gt,len(x),rng)):
                    on,off=onoff; c_sf=signal_features(x,z,on,off); c_cf=contrast_features(x,on,off,k)
                    all_video.append({"kind":"control","max_z":c_sf["max_z"],"mean_z":c_sf["mean_z"],"local_normalized_contrast":c_cf["local_normalized_contrast"]})
                all_video.append({"kind":"generation_miss","max_z":sf["max_z"],"mean_z":sf["mean_z"],"local_normalized_contrast":cf["local_normalized_contrast"]})
            total_match=len(native.maximum_edges([{"peak":int(p)} for p in primitive],gt)); gen_indices=[i for i,t in enumerate(gt_types) if t=="GENERATION_MISS_STRICT"]; gen_gt=[gt[i] for i in gen_indices]; gen_match=len(native.maximum_edges([{"peak":int(p)} for p in primitive],gen_gt))
            oracle_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"num_gt":len(gt),"num_primitive_peaks":len(primitive),"all_gt_coverable":total_match,"generation_miss_count":len(gen_gt),"generation_miss_newly_coverable":gen_match})
        for typ in ("NATIVE_MATCHED","BOUNDARY_MISS_STRICT","SELECTION_MISS_STRICT","GENERATION_MISS_STRICT"):
            ss=[r for r in sig_rows if r["backbone"]==bb and r["dataset"]==ds and r["gt_type"]==typ]; cc=[r for r in contrast_rows if r["backbone"]==bb and r["dataset"]==ds and r["gt_type"]==typ]
            for metric in ("max_z","mean_z","median_z","positive_z_area","max_raw_percentile","mean_raw_percentile"):
                summaries.append({"backbone":bb,"dataset":ds,"gt_type":typ,"feature":metric,**summary([r[metric] for r in ss])})
            for metric in ("max_minus_median_bg","mean_minus_mean_bg","mean_minus_median_bg","local_normalized_contrast"):
                summaries.append({"backbone":bb,"dataset":ds,"gt_type":typ,"feature":metric,**summary([r[metric] for r in cc])})
        gen_sig=[r for r in sig_rows if r["backbone"]==bb and r["dataset"]==ds and r["gt_type"]=="GENERATION_MISS_STRICT"]
        valid_gen_sig=[r for r in gen_sig if r["max_z"] is not None]
        threshold_rates={f"max_z_ge_{v:g}":sum(r["max_z"]>=v for r in valid_gen_sig)/len(valid_gen_sig) if valid_gen_sig else 0 for v in (0,.5,1.,1.5)}
        threshold_rates["valid_generation_segments"]=len(valid_gen_sig); threshold_rates["unavailable_generation_segments"]=len(gen_sig)-len(valid_gen_sig)
        for metric in ("max_z","mean_z","local_normalized_contrast"):
            usable=[r for r in all_video if r[metric] is not None and np.isfinite(r[metric])]; y=[int(r["kind"]=="generation_miss") for r in usable]; s=[r[metric] for r in usable]
            ranks.append({"backbone":bb,"dataset":ds,"diagnostic_label":"GT-ASSISTED DIAGNOSTIC — NOT A METHOD","feature":metric,"positive_generation_miss":sum(y),"background_controls":len(y)-sum(y),"ROC_AUC":roc_auc(y,s),"PR_AUC":average_precision(y,s),"control_seed":100,"max_controls_per_GT":3})
        vals=sorted(affected.values(),reverse=True); total=sum(vals)
        distributions.append({"backbone":bb,"dataset":ds,"recoverable_generation_misses":total,"affected_subjects":len(vals),"affected_videos":len(affected_v),"top1_subject_share":vals[0]/total if vals else 0,"top3_subject_share":sum(vals[:3])/total if vals else 0,"median_recoverable_per_affected_subject":float(np.median(vals)) if vals else 0})
        oracle_group=[r for r in oracle_rows if r["backbone"]==bb and r["dataset"]==ds]
        groups.append({"backbone":bb,"dataset":ds,"native_exact_replay":group_exact and group_counts==target_counts,"native_TP_FP_FN":group_counts,"generation_miss":gen_count,"primitive_inside":gen_inside,"primitive_inside_ratio":gen_inside/gen_count if gen_count else 0,"within_0.25k":sum(r["primitive_peak_within_0.25k"] for r in recover if r["backbone"]==bb and r["dataset"]==ds),"within_0.50k":sum(r["primitive_peak_within_0.50k"] for r in recover if r["backbone"]==bb and r["dataset"]==ds),"within_1.00k":sum(r["primitive_peak_within_1.00k"] for r in recover if r["backbone"]==bb and r["dataset"]==ds),"loss_stages":dict(stage_counts),"max_z_threshold_rates":threshold_rates,"primitive_oracle_all_gt_coverable":sum(r["all_gt_coverable"] for r in oracle_group),"primitive_oracle_recall_ceiling":sum(r["all_gt_coverable"] for r in oracle_group)/sum(r["num_gt"] for r in oracle_group),"primitive_oracle_generation_miss_newly_coverable":sum(r["generation_miss_newly_coverable"] for r in oracle_group)})
    if not all(g["native_exact_replay"] for g in groups):
        report={"status":"DENSE_DECODER_PHASE0_NATIVE_REPLAY_MISMATCH","groups":groups}; (OUT/"combined_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,indent=2)); return
    write_csv(OUT/"primitive_peak_recoverability.csv",recover); write_csv(OUT/"generation_loss_stage.csv",losses); write_csv(OUT/"gt_signal_features.csv",sig_rows); write_csv(OUT/"gt_background_contrast.csv",contrast_rows); write_csv(OUT/"generation_signal_summary.csv",summaries); write_csv(OUT/"generation_rank_separation.csv",ranks); write_csv(OUT/"primitive_peak_oracle.csv",oracle_rows); write_csv(OUT/"generation_opportunity_distribution.csv",distributions)
    (OUT/"curve_provenance.json").write_text(json.dumps({"diagnostic_curve_rule":"Historical Native 2*k smoothed temporal response; no resmoothing search","videos":provenance},ensure_ascii=False,indent=2))
    ratios=[g["primitive_inside_ratio"] for g in groups]; auc_signal=[max(r["ROC_AUC"] or 0 for r in ranks if r["backbone"]==g["backbone"] and r["dataset"]==g["dataset"]) for g in groups]
    distributed=sum(d["affected_subjects"]>=3 for d in distributions)>=3; both_backbones=all(any(g["backbone"]==bb and g["primitive_inside_ratio"]>=.25 for g in groups) for bb in ("ME-TST","BoostingVRME")); strong=sum(v>=.25 for v in ratios)>=3 and distributed and sum(a>=.65 for a in auc_signal)>=1 and both_backbones
    go=sum(v>=.20 for v in ratios)>=2 and all(v>=.05 for v in ratios); stop=sum(v<.10 for v in ratios)>=3 and sum(a<.60 for a in auc_signal)>=3
    decision="STRONG_GO" if strong else "GO" if go else "STOP" if stop else "BORDERLINE"
    files={p.name:str(p) for p in OUT.iterdir() if p.is_file()}; files["combined_report.json"]=str(OUT/"combined_report.json")
    report={"status":"DENSE_DECODER_PHASE0_NATIVE_REPLAY_EXACT","decision":decision,"semi_markov_not_designed":True,"geso_not_run":True,"boundary_refiner_not_run":True,"groups":groups,"rank_separation":ranks,"distribution":distributions,"decision_notes":{"strong_auc_descriptive_threshold":0.65,"stop_auc_indistinguishable_threshold":0.60,"thresholds_used_only_to operationalize qualitative criterion; diagnostics were not tuned":True},"files":files}
    (OUT/"combined_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
