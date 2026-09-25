#!/usr/bin/env python3
"""Boundary Refiner V1: post-selection interval-only, no optimizer/NMS changes."""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/boundary_refiner_v1"; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
import run_boundary_phase0_audit as audit

def write_csv(path, rows):
    if not rows: rows=[{"note":"no rows"}]
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def prom_bounds(x,p):
    if not np.isfinite(x).all() or p<=0 or p>=len(x)-1: return None
    h=float(x[p]); left_min=h; right_min=h; i=p
    while i>0:
        i-=1; z=float(x[i])
        if z>h: break
        left_min=min(left_min,z)
    i=p
    while i<len(x)-1:
        i+=1; z=float(x[i])
        if z>h: break
        right_min=min(right_min,z)
    level=h-.5*(h-max(left_min,right_min))
    def cross(a,b):
        if x[a]==x[b]: return float(a)
        return float(a)+(level-float(x[a]))/(float(x[b])-float(x[a]))
    l=None
    for i in range(p-1,-1,-1):
        if x[i]<=level<=x[i+1] or x[i]>=level>=x[i+1]: l=cross(i,i+1); break
    r=None
    for i in range(p,len(x)-1):
        if x[i]<=level<=x[i+1] or x[i]>=level>=x[i+1]: r=cross(i,i+1); break
    if l is None or r is None: return None
    return int(np.clip(round(l),0,p)), int(np.clip(round(r),p,len(x)-1))

def cp_bounds(x,p,k,native):
    if not np.isfinite(x).all(): return native
    R=2*int(k); lo=max(0,p-R); hi=min(len(x)-1,p+R); eps=1e-12
    vals=x[lo:hi+1]
    mad=float(np.median(np.abs(vals-np.median(vals))))+eps
    def score(a,b,c,d):
        bg=x[a:b]; ev=x[c:d]
        if len(bg)==0 or len(ev)==0: return None
        contrast=float(np.mean(ev)-np.mean(bg)); balance=math.sqrt(len(bg)*len(ev)/(len(bg)+len(ev)))
        return balance*contrast/mad,contrast
    best_l=None
    for b in range(lo+1,p):
        q=score(lo,b,b,p+1)
        if q is not None and q[1]>0 and (best_l is None or q[0]>best_l[0]): best_l=(q[0],b)
    best_r=None
    for b in range(p+1,hi+1):
        q=score(p,b+1,b+1,hi+1)
        if q is not None and q[1]>0 and (best_r is None or q[0]>best_r[0]): best_r=(q[0],b)
    return (best_l[1] if best_l else native[0], best_r[1] if best_r else native[1])

def counts(preds,gt):
    pairs=audit.match(preds,gt); tp=len(pairs); return (tp,len(preds)-tp,len(gt)-tp),pairs
def f1(c): return 2*c[0]/(2*c[0]+c[1]+c[2]) if 2*c[0]+c[1]+c[2] else 0.0
def refine_preds(preds,curve,k,method):
    out=[]; fallbacks=0
    for p in preds:
        native=(int(p["onset"]),int(p["offset"])); peak=int(p["peak"])
        if method=="B0_Native": interval=native
        elif method=="B1_HalfProm": interval=prom_bounds(curve,peak) or native; fallbacks += int(interval==native)
        else: interval=cp_bounds(curve,peak,k,native); fallbacks += int(interval==native)
        q=dict(p); q["onset"],q["offset"]=map(int,interval); out.append(q)
    return out,fallbacks

def bootstrap(groups, a, b, seed=100, n=10000):
    rng=np.random.default_rng(seed); subs=sorted(set(x["subject"] for x in groups)); vals=[]
    by=defaultdict(lambda: defaultdict(lambda: [0,0,0]))
    for x in groups:
        for method in (a,b):
            c,_=counts(x[method],x["gt"])
            for j in range(3): by[x["subject"]][method][j]+=c[j]
    for _ in range(n):
        chosen=rng.choice(subs,size=len(subs),replace=True); ca=[0,0,0]; cb=[0,0,0]
        for s in chosen:
            for dest,method in ((ca,a),(cb,b)):
                for j in range(3): dest[j]+=by[s][method][j]
        vals.append(f1(ca)-f1(cb))
    arr=np.asarray(vals); point=0.0
    ca=[0,0,0]; cb=[0,0,0]
    for x in groups:
        for dest,method in ((ca,a),(cb,b)):
            c,_=counts(x[method],x["gt"])
            for j in range(3): dest[j]+=c[j]
    point=f1(ca)-f1(cb)
    return {"point_estimate":point,"bootstrap_mean":float(arr.mean()),"ci95_low":float(np.quantile(arr,.025)),"ci95_high":float(np.quantile(arr,.975)),"ci_crosses_zero":bool(np.quantile(arr,.025)<=0<=np.quantile(arr,.975)),"n_resamples":n,"seed":seed}

def main():
    pmf=audit.load_pmf(); groups=[]; result_rows=[]; ret_rows=[]; boot_rows=[]; mech_rows=[]; geometry=[]; interval_rows=[]; gate=[]; fallbacks=defaultdict(int)
    for bb,ds,cp,tp in audit.CASES:
        k,videos=audit.group_audit(bb,ds,cp,tp,pmf); g=[]; target_mismatch=False
        for subject,video,gt,native,cands,vk,curve,threshold,exp in videos:
            exp_preds=exp["predictions"].get("original_native",exp["predictions"].get("native",[]))
            if sorted((p["onset"],p["offset"],p["peak"]) for p in native)!=sorted((p["onset"],p["offset"],p["peak"]) for p in exp_preds): target_mismatch=True
            # Native candidate generation uses a 2*k moving average for both
            # families; Boosting's source_interval remains raw-curve logic,
            # but V1 boundary evidence uses the corresponding response.
            refine_curve=np.convolve(curve,np.ones(max(1,2*int(k)),dtype=float)/max(1,2*int(k)),mode="same") if bb=="BoostingVRME" else curve
            methods={}; intervals=[]
            for method in ("B0_Native","B1_HalfProm","B2_ChangePoint"):
                methods[method],fb=refine_preds(native,refine_curve,k,method); fallbacks[(bb,ds,method)]+=fb
                for cid,p in enumerate(methods[method]): intervals.append({"backbone":bb,"dataset":ds,"subject":subject,"video":video,"candidate_id":cid,"method":method,"peak":p["peak"],"start":p["onset"],"end":p["offset"],"native_start":native[cid]["onset"],"native_end":native[cid]["offset"],"changed":int((p["onset"],p["offset"])!=(native[cid]["onset"],native[cid]["offset"]))})
            interval_rows.extend(intervals)
            g.append({"subject":subject,"video":video,"gt":gt,**methods})
        base=[]
        for method in ("B0_Native","B1_HalfProm","B2_ChangePoint"):
            c=[0,0,0]
            for x in g:
                q,_=counts(x[method],x["gt"])
                for j in range(3): c[j]+=q[j]
            base.append((method,c))
        native_c=base[0][1]; oracle=next(x for x in audit.group_audit(bb,ds,cp,tp,pmf)[1:2] if False) if False else None
        # Phase-0 oracle is recomputed from the frozen selected identities.
        otp=op=0
        for x in g:
            op+=len(x["B0_Native"]); otp+=len(audit.maximum_edges([{"peak":p["peak"]} for p in x["B0_Native"]],x["gt"]))
        oracle_delta=f1((otp,op-otp,sum(len(x["gt"]) for x in g)-otp))-f1(native_c)
        total_predictions=sum(len(x["B0_Native"]) for x in g)
        row={"Backbone":bb,"Dataset":ds,"Native_TP":native_c[0],"Native_FP":native_c[1],"Native_FN":native_c[2],"Native_F1":f1(native_c),"HalfProm_TP":base[1][1][0],"HalfProm_FP":base[1][1][1],"HalfProm_FN":base[1][1][2],"HalfProm_F1":f1(base[1][1]),"HalfProm_Delta_F1":f1(base[1][1])-f1(native_c),"ChangePoint_TP":base[2][1][0],"ChangePoint_FP":base[2][1][1],"ChangePoint_FN":base[2][1][2],"ChangePoint_F1":f1(base[2][1]),"ChangePoint_Delta_F1":f1(base[2][1])-f1(native_c),"Native_prediction_count":total_predictions,"HalfProm_prediction_count":sum(len(x["B1_HalfProm"]) for x in g),"ChangePoint_prediction_count":sum(len(x["B2_ChangePoint"]) for x in g),"prediction_count_equal":int(all(len(x["B0_Native"])==len(x["B1_HalfProm"])==len(x["B2_ChangePoint"]) for x in g)),"native_exact_replay":int(not target_mismatch)}
        result_rows.append(row); ret_rows.append({"Backbone":bb,"Dataset":ds,"oracle_delta_F1":oracle_delta,"HalfProm_delta_F1":row["HalfProm_Delta_F1"],"HalfProm_retention":row["HalfProm_Delta_F1"]/oracle_delta if oracle_delta else None,"ChangePoint_delta_F1":row["ChangePoint_Delta_F1"],"ChangePoint_retention":row["ChangePoint_Delta_F1"]/oracle_delta if oracle_delta else None})
        for a,b in (("B1_HalfProm","B0_Native"),("B2_ChangePoint","B0_Native"),("B2_ChangePoint","B1_HalfProm")):
            z=bootstrap(g,a,b); boot_rows.append({"Backbone":bb,"Dataset":ds,"comparison":f"{a}-{b}",**z})
        for method in ("B1_HalfProm","B2_ChangePoint"):
            for x in g:
                native_pair_list=counts(x["B0_Native"],x["gt"])[1]; refined_pair_list=counts(x[method],x["gt"])[1]
                native_pairs={j for _,j in native_pair_list}; refined_pairs={j for _,j in refined_pair_list}
                native_matched_candidates={i for i,_ in native_pair_list}; refined_matched_candidates={i for i,_ in refined_pair_list}
                changed=sum((a["onset"],a["offset"])!=(b["onset"],b["offset"]) for a,b in zip(x["B0_Native"],x[method]))
                mech_rows.append({"Backbone":bb,"Dataset":ds,"method":method,"subject":x["subject"],"video":x["video"],"retained_matched_gt":len(native_pairs&refined_pairs),"newly_matched_gt":len(refined_pairs-native_pairs),"lost_matched_gt":len(native_pairs-refined_pairs),"native_FP_converted_to_TP":len(refined_matched_candidates-native_matched_candidates),"native_TP_converted_to_FP":len(native_matched_candidates-refined_matched_candidates),"intervals_changed":changed,"intervals_unchanged":len(x[method])-changed,"start_only_changed":sum(a["onset"]!=b["onset"] and a["offset"]==b["offset"] for a,b in zip(x["B0_Native"],x[method])),"end_only_changed":sum(a["onset"]==b["onset"] and a["offset"]!=b["offset"] for a,b in zip(x["B0_Native"],x[method])),"both_changed":sum(a["onset"]!=b["onset"] and a["offset"]!=b["offset"] for a,b in zip(x["B0_Native"],x[method]))})
        for method in ("B0_Native","B1_HalfProm","B2_ChangePoint"):
            for x in g:
                pairs=counts(x[method],x["gt"])[1]
                for pi,gi in pairs:
                    p=x[method][pi]; q=x["gt"][gi]; pd=p["offset"]-p["onset"]+1; gd=q[2]-q[0]+1
                    geometry.append({"Backbone":bb,"Dataset":ds,"method":method,"subject":x["subject"],"video":x["video"],"gt_id":gi,"iou":audit.iou((p["onset"],p["offset"]),(q[0],q[2])),"duration_ratio":pd/gd,"onset_abs_error":abs(p["onset"]-q[0]),"offset_abs_error":abs(p["offset"]-q[2]),"center_error":(p["onset"]+p["offset"]-q[0]-q[2])/2})
    write_csv(OUT/"boundary_v1_results.csv",result_rows); write_csv(OUT/"boundary_v1_oracle_retention.csv",ret_rows); write_csv(OUT/"boundary_v1_bootstrap.csv",boot_rows); write_csv(OUT/"boundary_v1_mechanism.csv",mech_rows); write_csv(OUT/"boundary_v1_geometry.csv",geometry); write_csv(OUT/"boundary_v1_intervals.csv",interval_rows)
    # No Native identity or count mutation is possible in this harness; still assert it explicitly.
    counts_equal=all(bool(r["prediction_count_equal"]) and r["Native_prediction_count"]==r["HalfProm_prediction_count"]==r["ChangePoint_prediction_count"] for r in result_rows)
    positive=sum(r["ChangePoint_Delta_F1"]>0 for r in result_rows); high=[r for r in ret_rows if r["Dataset"]=="CASME3" or (r["Backbone"]=="BoostingVRME" and r["Dataset"]=="SAMMLV")]; retention20=sum(r["ChangePoint_retention"]>=.2 for r in high)
    losses=sum(r["lost_matched_gt"] for r in mech_rows if r["method"]=="B2_ChangePoint"); status="STRONG_GO" if positive>=3 and retention20>=2 and all(r["ChangePoint_Delta_F1"]>-.003 for r in result_rows if r["Dataset"]=="SAMMLV" and r["Backbone"]=="ME-TST") else ("GO" if positive>=3 and sum(r["ChangePoint_Delta_F1"]>.005 for r in result_rows)>=2 and losses<sum(r["Native_TP"] for r in result_rows) else ("BORDERLINE" if positive==2 else "STOP"))
    mechanism_summary=[]
    for bb,ds,method in sorted({(r["Backbone"],r["Dataset"],r["method"]) for r in mech_rows}):
        rows=[r for r in mech_rows if (r["Backbone"],r["Dataset"],r["method"])==(bb,ds,method)]
        keys=("retained_matched_gt","newly_matched_gt","lost_matched_gt","native_FP_converted_to_TP","native_TP_converted_to_FP","intervals_changed","intervals_unchanged","start_only_changed","end_only_changed","both_changed")
        mechanism_summary.append({"Backbone":bb,"Dataset":ds,"method":method,**{key:sum(r[key] for r in rows) for key in keys}})
    report={"status":status,"native_exact_replay":all(bool(r["native_exact_replay"]) for r in result_rows),"prediction_counts_unchanged":counts_equal,"geso_not_run":True,"v2_not_designed":True,"results":result_rows,"oracle_retention":ret_rows,"bootstrap":boot_rows,"mechanism_summary":mechanism_summary,"fallback_counts":{f"{a}/{b}/{c}":v for (a,b,c),v in fallbacks.items()},"change_point_positive_settings":positive,"high_opportunity_retention_ge_20pct":retention20,"change_point_lost_native_matches":losses,"files":{p.name:str(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/"combined_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
