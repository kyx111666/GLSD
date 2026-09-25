#!/usr/bin/env python3
"""GESO Phase 1: native replay, conflict audit, and exact-set comparison.

This script is deliberately self-contained and does not import or inspect GLSD
artifacts.  It operates only on the four frozen Native score caches.
"""
from __future__ import annotations

import csv, hashlib, json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "geso_phase1"
METST_POST = ROOT / "RethinkFuse_reproduction/senior_original/me_tst_video/me-tst-video/postprocess.py"
BOOST_SRC = ROOT / "RethinkFuse_reproduction/senior_original/boostingvrme/experiments/evaluate_boosting_local_pair_fuse.py"
CACHES = {
    ("ME-TST", "SAMMLV"): ROOT / "RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl",
    ("ME-TST", "CASME3"): ROOT / "RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl",
    ("BoostingVRME", "SAMMLV"): ROOT / "RethinkFuse_reproduction/caches/boostingvrme/sammlv_curves.pkl",
    ("BoostingVRME", "CASME3"): ROOT / "RethinkFuse_reproduction/caches/boostingvrme/casme3_curves.pkl",
}

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def smooth(x, w): return np.convolve(np.asarray(x, float), np.ones(int(w))/int(w), mode="same")

def iou(a,b):
    l=max(int(a[0]),int(b[0])); r=min(int(a[1]),int(b[1])); inter=max(0,r-l)
    union=max(int(a[1]),int(b[1]))-min(int(a[0]),int(b[0]))
    return inter/union if union else 0.0

def nms_greedy(cands, distance, thr):
    kept=[]
    for c in sorted(cands, key=lambda z:(-z["weight"], z["id"])):
        if any(abs(c["peak"]-k["peak"]) <= distance or (thr > 0 and iou((c["start"],c["end"]),(k["start"],k["end"])) >= thr) for k in kept): continue
        kept.append(c)
    return kept

def metst_candidates(records, kp):
    out=[]
    for r in records:
        s=smooth(r["score"], kp*2); mean=float(s.mean()); th=mean+.55*(float(s.max())-mean)
        peaks,_=find_peaks(s,height=th,distance=kp)
        cs=[]
        for j,p in enumerate(peaks):
            p=int(p); cs.append({"id":j,"start":max(0,p-kp),"end":min(len(s)-1,p+kp),"peak":p,"weight":float(s[p])})
        out.append(cs) # Native ME-TST has find_peaks(distance=kp), no later NMS.
    return out

def boosting_candidates(curves, kp):
    out=[]
    for sr in curves:
        for score in sr["score"]:
            s=smooth(score,kp*2); mean=float(s.mean()); th=mean+.55*(float(s.max())-mean)
            peaks,_=find_peaks(s,height=th,distance=kp); cs=[]
            for j,p in enumerate(peaks):
                p=int(p); l0=max(0,p-kp); r0=min(len(score)-1,p+kp); raw=np.asarray(score,float)
                t1=raw.mean()/1.5; t2=raw.mean(); start=next((i-1 for i in range(l0+1,p) if raw[i]>raw[i-1] and raw[i]>t1),l0)
                end=next((i+1 for i in range(r0-1,p,-1) if raw[i]>raw[i+1] and raw[i]>t1),r0)
                le=max(0,p-kp*5); re=min(len(raw)-1,p+kp*5)
                for i in range(start-1,le-1,-1):
                    if i>0 and raw[i]<t2 and raw[i-1]<t2: start=i+1; break
                for z in range(end+1,re+1):
                    if z<len(raw)-1 and raw[z]<t2 and raw[z+1]<t2: end=z-1; break
                p2=int(np.argmax(raw[start:end+1])+start)
                cs.append({"id":j,"start":int(start),"end":int(end),"peak":p2,"weight":float(s[p])})
            out.append(cs)  # retain the true pre-NMS pool; Native NMS is applied below.
    return out

def components(cs, distance, thr):
    n=len(cs); adj=[set() for _ in cs]
    for i in range(n):
        for j in range(i+1,n):
            if abs(cs[i]["peak"]-cs[j]["peak"])<=distance or (thr > 0 and iou((cs[i]["start"],cs[i]["end"]),(cs[j]["start"],cs[j]["end"]))>=thr):
                adj[i].add(j); adj[j].add(i)
    seen=set(); comps=[]
    for i in range(n):
        if i in seen: continue
        st=[i]; seen.add(i); cc=[]
        while st:
            x=st.pop(); cc.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        comps.append((cc,adj))
    return adj, comps

def exact_component(cs, inds, adj):
    best=(0.0,())
    m=len(inds)
    for mask in range(1<<m):
        sel=tuple(inds[k] for k in range(m) if mask>>k&1)
        if any(b in adj[a] for q,a in enumerate(sel) for b in sel[q+1:]): continue
        w=sum(cs[i]["weight"] for i in sel)
        if w>best[0]+1e-12 or (abs(w-best[0])<=1e-12 and sel<best[1]): best=(w,sel)
    return list(best[1]),best[0]

def match(preds,gts):
    used=set(); tp=0
    for p in preds:
        vals=[(iou((p["start"],p["end"]),(g[0],g[2])),j) for j,g in enumerate(gts) if j not in used]
        if vals:
            v,j=max(vals)
            if v>=.5: tp+=1; used.add(j)
    return tp,len(preds)-tp,len(gts)-tp

def f1(tp,fp,fn):
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0

def boot(rows, seed=100, n=10000):
    rng=np.random.default_rng(seed); keys=list(rows); vals=[]
    for _ in range(n):
        samp=[rows[keys[i]] for i in rng.integers(0,len(keys),len(keys))]
        a=sum(x[0] for x in samp); b=sum(x[1] for x in samp); c=sum(x[2] for x in samp)
        d=sum(x[3] for x in samp); e=sum(x[4] for x in samp); g=sum(x[5] for x in samp)
        vals.append(f1(d,e,g)-f1(a,b,c))
    return float(np.mean(vals)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prov={"experiment":"GESO Phase 1","glsd_used":False,"solver":"deterministic exhaustive maximum-weight independent set per conflict component","solver_settings":{"tie_break":"lexicographic candidate id","bootstrap_seed":100,"bootstrap_resamples":10000},"pipelines":{}}
    allrows=[]; struct=[]; fail=[]; overlaps=[]; oracle=[]; mech=[]; bootrows=[]
    for (backbone,dataset),p in CACHES.items():
        cache=pickle.load(open(p,"rb")); key=f"{backbone}/{dataset}"
        if backbone=="ME-TST":
            records=cache["records"]; kp=int(cache["k_p"]); pools=metst_candidates(records,kp); vids=[(r["subject"],r["video"],r["samples"]) for r in records]; dist=kp; thr=0.0
            native_final=[nms_greedy(cs,dist,thr) for cs in pools]
        else:
            curves=cache["subject_curves"]; kp=int(cache["k_p"]); pools=boosting_candidates(curves,kp); vids=[]
            for sr in curves:
                for v,s in zip(sr["videos"],sr["samples"]): vids.append((sr["subject"],v,s))
            dist=kp; thr=.2; native_final=[nms_greedy(cs,dist,thr) for cs in pools]
        prov["pipelines"][key]={"cache":str(p),"cache_sha256":sha(p),"native_source":str(METST_POST if backbone=="ME-TST" else BOOST_SRC),"candidate_score":"smoothed peak score; Native Moilanen p=0.55","interval":"ME-TST peak +/- k_p; BoostingVRME native interval strategy","conflict":"peak distance <= k_p OR interval IoU >= 0.2 for BoostingVRME; ME-TST find_peaks distance is the Native candidate constraint","k_p":kp,"n_videos":len(vids)}
        changed_v=0; changed_c=0; gain=0; total_components=0; diff_components=0; sizes=[]; no_conf=0; conf_vid=0
        per_subject=defaultdict(lambda:[0,0,0,0,0,0]); native_total=[0,0,0]; exact_total=[0,0,0]
        for vi,(sub,video,gts) in enumerate(vids):
            cs=pools[vi]; adj,comps=components(cs,dist,thr); total_components+=len(comps); sizes += [len(x[0]) for x in comps]; no_conf += int(not any(len(x[0])>1 for x in comps)); conf_vid += int(any(len(x[0])>1 for x in comps))
            gs=[]; ew=0
            for inds,_ in comps:
                sel,w=exact_component(cs,inds,adj); gs += sel; ew += w
                greedy=set(i for i,c in enumerate(nms_greedy([cs[i] for i in inds],dist,thr)) for _ in [0] if c["id"]==inds[i]) if False else None
                local_g=nms_greedy([cs[i] for i in inds],dist,thr); local_g_ids={c["id"] for c in local_g}; exact_ids={cs[i]["id"] for i in sel}
                if local_g_ids!=exact_ids: diff_components+=1; changed_c+=len(local_g_ids.symmetric_difference(exact_ids)); gain += ew-sum(c["weight"] for c in local_g)
                fail.append({"backbone":backbone,"dataset":dataset,"subject":sub,"video":video,"component_size":len(inds),"greedy_ids":sorted(local_g_ids),"exact_ids":sorted(exact_ids),"greedy_weight":sum(c["weight"] for c in local_g),"exact_weight":ew,"different":local_g_ids!=exact_ids})
            native=native_final[vi]; exact=[cs[i] for i in sorted(gs)]
            nt=match(native,gts); et=match(exact,gts); native_total=np.add(native_total,nt); exact_total=np.add(exact_total,et)
            if [(c["start"],c["end"],c["peak"]) for c in native] != [(c["start"],c["end"],c["peak"]) for c in exact]: changed_v+=1
            per_subject[sub]=[per_subject[sub][i]+x for i,x in enumerate((*nt,*et))]
            for g in gts: overlaps.append((backbone,dataset,sub,video,g))
            OUT.joinpath("per_video").mkdir(exist_ok=True)
            (OUT/"per_video"/f"{backbone}_{dataset}_{sub}_{video}.json").write_text(json.dumps({"backbone":backbone,"dataset":dataset,"subject":sub,"video":video,"native":[{k:c[k] for k in ("id","start","end","peak","weight")} for c in native],"exact":[{k:c[k] for k in ("id","start","end","peak","weight")} for c in exact]},indent=2))
        nf=f1(*native_total); ef=f1(*exact_total); allrows.append([backbone,dataset,*native_total,nf,*exact_total,ef,ef-nf])
        bootrows.append((backbone,dataset,boot(per_subject)))
        struct.append({"backbone":backbone,"dataset":dataset,"pre_nms_candidates":int(sum(len(x) for x in pools)),"conflict_edges":int(sum(len(a) for _,a in [components(cs,dist,thr) for cs in pools] for z in a for _ in [])),"conflict_components":total_components,"component_size_distribution":dict((str(s),sizes.count(s)) for s in sorted(set(sizes))),"videos_no_conflict":no_conf,"videos_with_conflict":conf_vid,"max_component_size":max(sizes or [0]),"greedy_exact_components":total_components-diff_components,"greedy_different_components":diff_components,"affected_videos":changed_v,"total_candidate_changes":changed_c,"total_weight_gain":gain})
    # conflict edge count is recomputed plainly for readable audit output
    for row in struct:
        edge=0
        for (bb,ds),p in CACHES.items():
            if bb==row["backbone"] and ds==row["dataset"]:
                cache=pickle.load(open(p,"rb")); kp=int(cache["k_p"]); pools=metst_candidates(cache["records"],kp) if bb=="ME-TST" else boosting_candidates(cache["subject_curves"],kp)
                for cs in pools:
                    adj,_=components(cs,kp,0.0 if bb=="ME-TST" else .2); edge+=sum(len(x) for x in adj)//2
        row["conflict_edges"]=edge
    (OUT/"native_nms_provenance.json").write_text(json.dumps(prov,indent=2,ensure_ascii=False))
    for name,rows in [("conflict_structure.csv",struct),("greedy_failure_audit.csv",fail)]:
        keys=list(rows[0]) if rows else []; 
        with open(OUT/name,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT/"geso_phase1_results.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Backbone","Dataset","Native TP","Native FP","Native FN","Native F1","ExactSet TP","ExactSet FP","ExactSet FN","ExactSet F1","Delta F1"]); w.writerows(allrows)
    with open(OUT/"geso_phase1_bootstrap.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Backbone","Dataset","Mean Delta F1","CI 2.5%","CI 97.5%","Resamples","Seed"]); [w.writerow([a,b,*c,10000,100]) for a,b,c in bootrows]
    for bb,ds,g in bootrows: pass
    (OUT/"gt_overlap_audit.csv").write_text("backbone,dataset,subject,video,gt_start,gt_peak,gt_end\n"+"\n".join(",".join(map(str,(bb,ds,s,v,*g))) for bb,ds,s,v,g in overlaps)+"\n")
    (OUT/"same_pool_oracle.csv").write_text("backbone,dataset,status\n"+"\n".join(f"{a},{b},NOT_COMPUTED_GT_ASSISTED_ORACLE_REQUIRES_EXPLICIT_GT_OBJECTIVE" for a,b in CACHES)+"\n")
    (OUT/"geso_phase1_mechanism.csv").write_text("backbone,dataset,note\n"+"\n".join(f"{a},{b},Phase1 exact-set outputs are in per_video; replay gate is reported in combined_report.json" for a,b in CACHES)+"\n")
    report={"status":"REPLAY_REQUIRED_BEFORE_GESO","native_replay":"not asserted until existing final-prediction artifacts are explicitly mapped to each cache","results":allrows,"structure":struct,"bootstrap":bootrows,"note":"GT-assisted oracle intentionally not used to set any method parameter."}
    (OUT/"combined_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,"item") else str(x)))
    print(json.dumps(report,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,"item") else str(x)))
if __name__=="__main__": main()
