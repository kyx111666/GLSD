#!/usr/bin/env python3
"""GESO prerequisite gate using exact historical Native adapters."""
import importlib.util, json, pickle, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results/geso_phase1"
METST_SRC=ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py"
BOOST_ROOT=ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme"; sys.path.insert(0,str(BOOST_ROOT))
import equiscale_fair_validation as fair

def load_module(path):
    spec=importlib.util.spec_from_file_location("historical_metst_native",path); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def main():
    pmf=load_module(METST_SRC)
    cases=[
      ("ME-TST","SAMMLV",ROOT/"RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json"),
      ("ME-TST","CASME3",ROOT/"RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/casme3/fixed/selected_predictions.json"),
      ("BoostingVRME","SAMMLV",BOOST_ROOT/"curve_cache/sammlv_curves.pkl",ROOT/"pure_persistence_matched_v1/sammlv/native/selected_predictions.json"),
      ("BoostingVRME","CASME3",BOOST_ROOT/"curve_cache/casme_3_curves.pkl",ROOT/"pure_persistence_matched_v1/casme3/native/selected_predictions.json")]
    groups=[]
    for bb,ds,cp,tp in cases:
        c=pickle.load(open(cp,"rb")); target=json.load(open(tp)); got=[]
        if bb=="ME-TST":
            for r in c["records"]: got.append(pmf.native_detections(r,c)[0])
        else:
            k=fair.duration_k([{"subject":str(r["subject"]),"gt":r["gt"]} for r in c["records"]],set(),"boostingvrme")
            conf=fair.Config("single",2.0,.55,1.0)
            for r,curve in zip(c["records"],c["curves"]):
                vf=fair.VideoFeatures({"curve":curve,"gt":r["gt"]},k,"boostingvrme")
                got.append(vf.prepare(vf.clusters(conf)).evaluate([conf],details=True)[1][0])
        mism=[]
        source_counts=[0,0,0]; target_counts=[0,0,0]
        for output,expected in zip(got,target):
            exp=expected["predictions"].get("original_native",expected["predictions"].get("native",[]))
            a=sorted((int(x["onset"]),int(x["offset"]),int(x["peak"])) for x in output); b=sorted((int(x["onset"]),int(x["offset"]),int(x["peak"])) for x in exp)
            src_tp=sum(int(x.get("matched_gt",-1))>=0 for x in output) if bb!="ME-TST" else sum(x>=0 for x in pmf.match_detections(output, expected["gt"])[0])
            tgt_tp=sum(int(x.get("matched_gt",-1))>=0 for x in exp)
            source_counts[0]+=src_tp; source_counts[1]+=len(output)-src_tp; source_counts[2]+=len(expected["gt"])-src_tp
            target_counts[0]+=tgt_tp; target_counts[1]+=len(exp)-tgt_tp; target_counts[2]+=len(expected["gt"])-tgt_tp
            if a!=b: mism.append({"subject":expected["subject"],"video":expected["video"],"replayed":a,"expected":b})
        groups.append({"backbone":bb,"dataset":ds,"cache":str(cp),"target":str(tp),"videos":len(got),"mismatched_videos":len(mism),"exact_replay":not mism,"source_TP_FP_FN":source_counts,"target_TP_FP_FN":target_counts,"metrics_exact":source_counts==target_counts,"first_mismatches":mism[:5]})
    status="NATIVE_FINAL_SELECTION_EXACT_REPLAY" if all(x["exact_replay"] and x["metrics_exact"] for x in groups) else "NATIVE_FINAL_SELECTION_PROVENANCE_UNRESOLVED"
    report={"status":status,"phase":"Native final-selection provenance recovery","groups":groups,"geso_not_run":True}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"native_replay_gate_v2.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
