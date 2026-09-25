#!/usr/bin/env python3
"""Strict, post-hoc Boundary Refinement Phase-0 audit.

This script only replays the frozen Native adapters and measures oracle
boundary opportunity.  It does not run or import any GESO optimizer.
"""
import csv, importlib.util, json, pickle, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/boundary_opportunity_audit"
METST_SRC = ROOT / "historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py"
BOOST_ROOT = ROOT / "historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme"
sys.path.insert(0, str(BOOST_ROOT))
import equiscale_fair_validation as fair

CASES = [
 ("ME-TST", "SAMMLV", ROOT/"RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl", ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json"),
 ("ME-TST", "CASME3", ROOT/"RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl", ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/casme3/fixed/selected_predictions.json"),
 ("BoostingVRME", "SAMMLV", BOOST_ROOT/"curve_cache/sammlv_curves.pkl", ROOT/"pure_persistence_matched_v1/sammlv/native/selected_predictions.json"),
 ("BoostingVRME", "CASME3", BOOST_ROOT/"curve_cache/casme_3_curves.pkl", ROOT/"pure_persistence_matched_v1/casme3/native/selected_predictions.json"),
]

def write_csv(path, rows):
    if not rows: rows=[{"note":"no rows"}]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load_pmf():
    spec=importlib.util.spec_from_file_location("historical_metst_native_boundary", METST_SRC)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def me_raw_peaks(curve, threshold):
    """The pre-distance part of the exact historical local_maxima function."""
    v=np.asarray(curve, dtype=float); peaks=[]; i=1
    while i < len(v)-1:
        if v[i] > v[i-1]:
            left=i
            while i+1 < len(v) and v[i] == v[i+1]: i += 1
            if i < len(v)-1 and v[i] > v[i+1]:
                p=(left+i)//2
                if v[p] >= threshold: peaks.append(int(p))
        i += 1
    return peaks

def iou(a,b):
    l=max(int(a[0]),int(b[0])); r=min(int(a[1]),int(b[1])); inter=max(0,r-l+1)
    den=(int(a[1])-int(a[0])+1)+(int(b[1])-int(b[0])+1)-inter
    return inter/den if den else 0.0

def match(preds, gt):
    used=set(); pairs=[]
    for pi,p in enumerate(preds):
        vals=[iou((p["onset"],p["offset"]),(g[0],g[2])) for g in gt]
        gi=int(np.argmax(vals)) if vals else -1
        if gi>=0 and vals[gi]>=.5 and gi not in used: used.add(gi); pairs.append((pi,gi))
    return pairs

def fixed_interval(n, peak, k): return max(0,peak-k), min(n-1,peak+k)

def group_audit(bb, ds, cache_path, target_path, pmf):
    cache=pickle.load(cache_path.open("rb")); target=json.load(target_path.open())
    rows=[]; videos=[]; expected=[]
    if bb == "ME-TST":
        k=int(cache["k_p"])
        for rec, exp in zip(cache["records"], target):
            curve=pmf.moving_average(np.asarray(rec["score"],float),2*k)
            threshold=float(curve.mean()+.55*(curve.max()-curve.mean()))
            raw=me_raw_peaks(curve,threshold)
            native=pmf.native_detections(rec,cache)[0]
            selected={int(p["peak"]):i for i,p in enumerate(native)}
            candidates=[]
            for cid,p in enumerate(raw):
                on,off=fixed_interval(len(rec["emotion"]),p,k)
                candidates.append({"candidate_id":cid,"peak":p,"onset":on,"offset":off,"score":float(curve[p]),"selected":p in selected,"status":"selected_final" if p in selected else "native_distance_selection_rejected","source_interval":"fixed_peak_pm_k"})
            videos.append((rec["subject"],rec["video"],rec["samples"],native,candidates,k,curve,threshold,exp))
    else:
        k=fair.duration_k([{"subject":str(r["subject"]),"gt":r["gt"]} for r in cache["records"]],set(),"boostingvrme")
        conf=fair.Config("single",2.0,.55,1.0)
        for rec,curve,exp in zip(cache["records"],cache["curves"],target):
            vf=fair.VideoFeatures({"curve":np.asarray(curve,float),"gt":rec["gt"]},k,"boostingvrme")
            clusters=vf.clusters(conf); prep=vf.prepare(clusters); native=prep.evaluate([conf],details=True)[1][0]
            selected={(int(p["peak"]),int(p["onset"]),int(p["offset"])) for p in native}
            candidates=[]
            for cid,(peak,_,height,*_) in enumerate(prep.clusters):
                on,off=map(int,prep.intervals[cid]); key=(int(peak),on,off)
                candidates.append({"candidate_id":cid,"peak":int(peak),"onset":on,"offset":off,"score":float(height),"selected":key in selected,"status":"selected_final" if key in selected else "native_greedy_conflict_suppressed","source_interval":"source_interval(curve, peak, k)"})
            videos.append((rec["subject"],rec["video"],rec["gt"],native,candidates,k,np.asarray(curve,float),None,exp))
    return k, videos

def maximum_edges(preds,gt):
    adj=[[j for j,g in enumerate(gt) if g[0] <= p["peak"] <= g[2]] for p in preds]
    owner={}
    def dfs(i,seen):
        for j in adj[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or dfs(owner[j],seen): owner[j]=i; return True
        return False
    for i in range(len(preds)): dfs(i,set())
    return [(i,j) for j,i in sorted(owner.items())]

def main():
    OUT.mkdir(parents=True,exist_ok=True); pmf=load_pmf(); all_groups=[]; cand_rows=[]; match_rows=[]; decomp=[]; oracle=[]; near=[]; geom=[]; errtypes=[]; dist=[]
    for bb,ds,cp,tp in CASES:
        k,videos=group_audit(bb,ds,cp,tp,pmf); group_name=f"{bb}/{ds}"; gcounts=[0,0,0]; target_counts=[0,0,0]; mismatch=[]; affected=defaultdict(int); affected_vid=Counter(); group_decomp=Counter(); group_near=Counter(); ggeom=defaultdict(list)
        for subject,video,gt,preds,cands,vk,curve,threshold,exp in videos:
            exp_preds=exp["predictions"].get("original_native",exp["predictions"].get("native",[]))
            got_t=sorted((int(p["onset"]),int(p["offset"]),int(p["peak"])) for p in preds); exp_t=sorted((int(p["onset"]),int(p["offset"]),int(p["peak"])) for p in exp_preds)
            if got_t != exp_t: mismatch.append({"subject":subject,"video":video,"replayed":got_t,"expected":exp_t})
            pairs=match(preds,gt); matched_gt={j for _,j in pairs}; target_pairs=[int(p.get("matched_gt",-1)) for p in exp_preds]
            tp=len(pairs); counts=(tp,len(preds)-tp,len(gt)-tp); ttp=sum(x>=0 for x in target_pairs); tcounts=(ttp,len(exp_preds)-ttp,len(gt)-ttp)
            for z in range(3): gcounts[z]+=counts[z]; target_counts[z]+=tcounts[z]
            bypi={i:j for i,j in pairs}
            selected=[c for c in cands if c["selected"]]; pre=cands
            for c in cands:
                cand_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"candidate_id":c["candidate_id"],"peak_frame":c["peak"],"native_start":c["onset"],"native_end":c["offset"],"native_duration":c["offset"]-c["onset"]+1,"native_score":c["score"],"selected_final":int(c["selected"]),"selection_status":c["status"],"k":vk,"source_interval":c["source_interval"],"nearest_gt_id":min(range(len(gt)),key=lambda j:abs(c["peak"]-np.clip(c["peak"],gt[j][0],gt[j][2]))) if gt else -1,"native_iou_nearest":max([iou((c["onset"],c["offset"]),(x[0],x[2])) for x in gt] or [0.0])})
            for pi,gi in pairs: match_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"record_type":"matched_prediction","candidate_id":next((c["candidate_id"] for c in selected if c["peak"]==preds[pi]["peak"]),-1),"gt_id":gi,"pred_onset":preds[pi]["onset"],"pred_offset":preds[pi]["offset"],"peak":preds[pi]["peak"],"gt_onset":gt[gi][0],"gt_apex":gt[gi][1],"gt_offset":gt[gi][2],"iou":iou((preds[pi]["onset"],preds[pi]["offset"]),(gt[gi][0],gt[gi][2]))})
            for pi,p in enumerate(preds):
                if pi not in bypi: match_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"record_type":"unmatched_prediction","candidate_id":next((c["candidate_id"] for c in selected if c["peak"]==p["peak"]),-1),"gt_id":-1,"pred_onset":p["onset"],"pred_offset":p["offset"],"peak":p["peak"],"gt_onset":"","gt_apex":"","gt_offset":"","iou":0.0})
            for gi,g in enumerate(gt):
                if gi in matched_gt: continue
                final_inside=[p for p in selected if g[0]<=p["peak"]<=g[2]]; pre_inside=[c for c in pre if g[0]<=c["peak"]<=g[2]]
                typ="BOUNDARY_MISS_STRICT" if final_inside else ("SELECTION_MISS_STRICT" if pre_inside else "GENERATION_MISS_STRICT")
                group_decomp[typ]+=1; decomp.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"gt_id":gi,"gt_onset":g[0],"gt_apex":g[1],"gt_offset":g[2],"error_type":typ,"final_selected_peak_inside":len(final_inside),"pre_final_candidate_peak_inside":len(pre_inside)})
                if typ=="BOUNDARY_MISS_STRICT": affected[subject]+=1; affected_vid[(subject,video)]+=1
                match_rows.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"record_type":"unmatched_gt","candidate_id":-1,"gt_id":gi,"pred_onset":"","pred_offset":"","peak":"","gt_onset":g[0],"gt_apex":g[1],"gt_offset":g[2],"iou":0.0})
                for source,arr in [("final_selected",selected),("pre_final_candidate",pre)]:
                    d=min([0 if g[0]<=c["peak"]<=g[2] else min(abs(c["peak"]-g[0]),abs(c["peak"]-g[2])) for c in arr] or [None])
                    for frac in (.25,.5,1.0): group_near[(source,frac)]+=int(d is not None and d<=frac*vk)
                if final_inside:
                    for p in final_inside: ggeom["BOUNDARY_MISS_STRICT"].append((p,g,gi,subject,video))
            for pi,gi in pairs: ggeom["TP"].append((preds[pi],gt[gi],gi,subject,video))
        total_gt=sum(len(x[2]) for x in videos); omtp=sum(len(maximum_edges([c for c in vs if c["selected"]],gt)) for *_,gt,vs,__,___ in []) if False else 0
        # Oracle and geometry are computed from each video to keep identities local.
        otp=0; opred=0
        for subject,video,gt,preds,cands,*_ in videos:
            sel=[c for c in cands if c["selected"]]; edges=maximum_edges(sel,gt); otp+=len(edges); opred+=len(sel)
        f1=lambda c: 2*c[0]/(2*c[0]+c[1]+c[2]) if 2*c[0]+c[1]+c[2] else 0
        base_f1=f1(gcounts)
        oracle_counts=(otp, opred-otp, total_gt-otp)
        oracle.append({"backbone":bb,"dataset":ds,"baseline_TP":gcounts[0],"baseline_FP":gcounts[1],"baseline_FN":gcounts[2],"oracle_TP":oracle_counts[0],"oracle_FP":oracle_counts[1],"oracle_FN":oracle_counts[2],"baseline_F1":base_f1,"oracle_F1":f1(oracle_counts),"delta_F1":f1(oracle_counts)-base_f1,"recoverable_FN":gcounts[2]-oracle_counts[2],"recoverable_FN_ratio":(gcounts[2]-oracle_counts[2])/gcounts[2] if gcounts[2] else 0})
        for subset,items in ggeom.items():
            metric_values=defaultdict(list)
            for p,g,gi,s,v in items:
                pd=p["offset"]-p["onset"]+1; gd=g[2]-g[0]+1; oe=p["onset"]-g[0]; fe=p["offset"]-g[2]
                cat="Over-wide" if pd>gd and p["onset"]<=g[0] and p["offset"]>=g[2] else ("Over-short" if pd<gd and p["onset"]>=g[0] and p["offset"]<=g[2] else ("Shift-left" if oe<0 and fe<0 else ("Shift-right" if oe>0 and fe>0 else "Mixed/other")))
                geom.append({"backbone":bb,"dataset":ds,"subset":subset,"subject":s,"video":v,"gt_id":gi,"pred_duration":pd,"gt_duration":gd,"duration_ratio":pd/gd if gd else None,"onset_error":oe,"offset_error":fe,"center_error":(p["onset"]+p["offset"]-g[0]-g[2])/2,"peak_relative_position":(p["peak"]-g[0])/gd if gd else None,"iou":iou((p["onset"],p["offset"]),(g[0],g[2])),"category":cat})
                metric_values["pred_duration"].append(pd); metric_values["gt_duration"].append(gd); metric_values["duration_ratio"].append(pd/gd if gd else 0); metric_values["onset_error"].append(oe); metric_values["offset_error"].append(fe); metric_values["center_error"].append((p["onset"]+p["offset"]-g[0]-g[2])/2); metric_values["iou"].append(iou((p["onset"],p["offset"]),(g[0],g[2])))
            for metric,values in metric_values.items():
                geom.append({"backbone":bb,"dataset":ds,"subset":subset,"subject":"__SUMMARY__","video":"","gt_id":"","pred_duration":"","gt_duration":"","duration_ratio":"","onset_error":"","offset_error":"","center_error":"","peak_relative_position":"","iou":"","category":json.dumps({"metric":metric,"n":len(values),"median":float(np.median(values)),"mean":float(np.mean(values)),"q1":float(np.quantile(values,.25)),"q3":float(np.quantile(values,.75))},ensure_ascii=False)})
        for typ,n in group_decomp.items(): errtypes.append({"backbone":bb,"dataset":ds,"error_type":typ,"count":n,"fraction_of_native_fn":n/gcounts[2] if gcounts[2] else 0})
        for source in ("final_selected","pre_final_candidate"):
            near.append({"backbone":bb,"dataset":ds,"source":source,"native_fn":gcounts[2],"le_0.25k":group_near[(source,.25)],"le_0.50k":group_near[(source,.5)],"le_1.00k":group_near[(source,1.0)]})
        affected_sub=len(affected); affected_v=len(affected_vid); vals=sorted(affected.values(),reverse=True); total_b=sum(vals)
        dist.append({"backbone":bb,"dataset":ds,"subjects_with_boundary_miss":affected_sub,"videos_with_boundary_miss":affected_v,"median_recoverable_per_affected_subject":float(np.median(vals)) if vals else 0,"max_recoverable_one_subject":max(vals) if vals else 0,"top1_subject_share":(vals[0]/total_b if vals else 0),"top3_subject_share":(sum(vals[:3])/total_b if vals else 0)})
        all_groups.append({"backbone":bb,"dataset":ds,"videos":len(videos),"mismatched_videos":len(mismatch),"exact_replay":not mismatch,"native_TP_FP_FN":gcounts,"target_TP_FP_FN":target_counts,"decomposition":dict(group_decomp),"mismatch_examples":mismatch[:3],"k":k})
    status="BOUNDARY_AUDIT_NATIVE_REPLAY_MISMATCH" if any(not g["exact_replay"] or g["native_TP_FP_FN"]!=g["target_TP_FP_FN"] for g in all_groups) else "BOUNDARY_AUDIT_NATIVE_REPLAY_EXACT"
    write_csv(OUT/"boundary_candidate_records.csv",cand_rows); write_csv(OUT/"native_matching_exact.csv",match_rows); write_csv(OUT/"fn_error_decomposition.csv",decomp); write_csv(OUT/"strict_boundary_oracle.csv",oracle); write_csv(OUT/"peak_near_gt_audit.csv",near); write_csv(OUT/"boundary_geometry.csv",geom); write_csv(OUT/"boundary_error_types.csv",errtypes); write_csv(OUT/"boundary_distribution.csv",dist)
    provenance={"ME-TST":{"source_file":str(METST_SRC),"function":"native_detections / local_maxima","lines":"313-330; 134-163","start_formula":"max(0, peak-k_p)","end_formula":"min(len(emotion)-1, peak+k_p)","k_role":"fixed half-width from cache k_p","source_interval":False,"clipping":True,"metadata":"smooth width 2*k_p; threshold mean+0.55*(max-mean); local maxima with historical distance rule"},"BoostingVRME":{"source_file":str(BOOST_ROOT/"tune_equiscale.py"),"function":"source_interval; native_predictions; fair.VideoFeatures.prepare/evaluate","lines":"130-146; 165-174; equiscale_fair_validation.py 249-317","start_formula":"source_interval: peak-k then curve-derived left boundary and below-mean expansion","end_formula":"source_interval: peak+k then curve-derived right boundary and below-mean expansion","k_role":"dataset duration prior; CASME3/SAMMLV exact native k","source_interval":True,"clipping":True,"metadata":"single-scale smoothing 2*k; height threshold .55; scipy find_peaks distance k; chronological interval-start greedy conflict suppression"}}
    (OUT/"native_boundary_rule_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    strict_go=all(x["delta_F1"]>=.010 for x in oracle) and sum(x["delta_F1"]>=.010 for x in oracle)>=3 and sum(x["recoverable_FN"]>=.05*x["baseline_FN"] for x in oracle)>=2
    go=(sum(x["delta_F1"]>=.010 for x in oracle)>=2 and all(x["delta_F1"]>.003 for x in oracle)) or (sum(x["delta_F1"]>=.005 for x in oracle)>=3 and sum(e["error_type"]=="BOUNDARY_MISS_STRICT" and e["count"]>0 for e in errtypes)>=1)
    stop=sum(x["delta_F1"]<.005 for x in oracle)>=3 or sum(e["error_type"]=="BOUNDARY_MISS_STRICT" and e["count"]>0 for e in errtypes)==0
    decision="BOUNDARY_AUDIT_NATIVE_REPLAY_MISMATCH" if status.endswith("MISMATCH") else ("STRONG_GO" if strict_go else "GO" if go else "STOP" if stop else "BORDERLINE")
    files={p.name:str(p) for p in OUT.iterdir() if p.is_file()}; files["combined_report.json"]=str(OUT/"combined_report.json")
    report={"status":status,"decision":decision,"geso_not_run":True,"boundary_refiner_not_designed":True,"groups":all_groups,"strict_oracle":oracle,"relaxed_peak_near_gt":near,"distribution":dist,"geometry_category_definition":"Descriptive fixed categories; no threshold optimization. Priority: Over-wide, Over-short, Shift-left, Shift-right, Mixed/other.","files":files}
    (OUT/"combined_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
