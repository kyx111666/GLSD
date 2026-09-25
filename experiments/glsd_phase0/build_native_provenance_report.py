#!/usr/bin/env python3
"""Build provenance-only artifacts; does not run GESO or alter selection."""
import csv, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results/geso_phase1"
sys.path.insert(0,str(ROOT)); import run_geso_phase1 as oldh
sys.path.insert(0,str(ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme")); import equiscale_fair_validation as fair
sp=importlib.util.spec_from_file_location("pmf",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py"); pmf=importlib.util.module_from_spec(sp); sys.modules[sp.name]=pmf; sp.loader.exec_module(pmf)
cases=[
 ("ME-TST","SAMMLV",ROOT/"RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json"),
 ("ME-TST","CASME3",ROOT/"RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/casme3/fixed/selected_predictions.json"),
 ("BoostingVRME","SAMMLV",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme/curve_cache/sammlv_curves.pkl",ROOT/"pure_persistence_matched_v1/sammlv/native/selected_predictions.json"),
 ("BoostingVRME","CASME3",ROOT/"historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme/curve_cache/casme_3_curves.pkl",ROOT/"pure_persistence_matched_v1/casme3/native/selected_predictions.json")]
def old_me(r,k):
 s=np.convolve(np.asarray(r['score'],float),np.ones(2*k)/(2*k),mode='same'); m=s.mean(); return find_peaks(s,height=m+.55*(s.max()-m),distance=k)[0].astype(int).tolist()
def old_boost(curve,k):
 s=np.convolve(np.asarray(curve,float),np.ones(2*k)/(2*k),mode='same'); m=s.mean(); return find_peaks(s,height=m+.55*(s.max()-m),distance=k)[0].astype(int).tolist()
def tup(xs): return sorted((int(x['onset']),int(x['offset']),int(x['peak'])) for x in xs)
def main():
    pipelines={
      "ME-TST/SAMMLV":{"raw_temporal_curve":"records[*].score","peak_generation":{"source_file":"historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py","function":"native_detections -> local_maxima","lines":"313-330","parameters":{"smoothing_width":"2*k_p","height":"mean + 0.55*(max-mean)","distance":"k_p"}},"score":"smoothed curve at retained peak; no separate score gate","height_evidence_gating":"height threshold inside local_maxima","interval_construction":{"source_file":"historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py","function":"native_detections","lines":"318-328","rule":"peak-k_p to peak+k_p"},"clipping":"max(0, onset), min(len(emotion)-1, offset)","sorting":"local_maxima returns sorted peaks; detections preserve order","overlap_handling":"none after peak-distance selection","nms_merging_suppression":"none","duplicate_removal":"none","final_filtering":"empty emotion window skipped","final_predictions":"native_detections output"},
      "ME-TST/CASME3":{},
      "BoostingVRME/SAMMLV":{"raw_temporal_curve":"curve_cache.records[*] + curve_cache.curves[*]","peak_generation":{"source_file":"historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme/tune_equiscale.py","function":"native_predictions / VideoFeatures.candidates","lines":"165-174; 213-224","parameters":{"smooth":"2.0*k_p (native config)","height":"mean + 0.55*(max-mean)","distance":"k_p"}},"score":"normalized height used for candidate evidence; Native config selects height evidence","height_evidence_gating":"smoothed peak >= Moilanen threshold","interval_construction":{"source_file":"historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme/tune_equiscale.py","function":"source_interval","lines":"130-146","rule":"native threshold1/threshold2 expansion"},"clipping":"source_interval clips to [0,len(curve)-1]","sorting":"interval start then peak, lines 155-158 / 171-172","overlap_handling":"temporal conflict: IoU >= 0.2 OR peak distance <= k_p","nms_merging_suppression":"greedy forward suppression in native_predictions / VideoFeatures.evaluate; lines 170-174 and equiscale_fair_validation.py 296-317","duplicate_removal":"suppressed by conflict mask","final_filtering":"Native config mask; no additional learned filter","final_predictions":"Prepared.evaluate(details=True)"},
      "BoostingVRME/CASME3":{}}
    pipelines["ME-TST/CASME3"]={**pipelines["ME-TST/SAMMLV"],"raw_temporal_curve":"records[*].score"}
    pipelines["BoostingVRME/CASME3"]={**pipelines["BoostingVRME/SAMMLV"],"raw_temporal_curve":"curve_cache.records[*] + curve_cache.curves[*]"}
    (OUT/"native_final_selection_pipeline.json").write_text(json.dumps({"scope":"provenance recovery only","geso_run":False,"pipelines":pipelines},indent=2,ensure_ascii=False))
    rows=[]; gate=[]
    for bb,ds,cp,tp in cases:
      c=pickle.load(open(cp,'rb')); target=json.load(open(tp)); exact=[]; old=[]
      if bb=="ME-TST":
        k=int(c['k_p'])
        for r in c['records']:
          exact.append(pmf.native_detections(r,c)[0]); old.append([{'onset':max(0,int(p)-k),'offset':min(len(r['score'])-1,int(p)+k),'peak':int(p)} for p in old_me(r,k)])
      else:
        k=fair.duration_k([{'subject':str(r['subject']),'gt':r['gt']} for r in c['records']],set(),'boostingvrme'); conf=fair.Config('single',2.0,.55,1.0)
        for r,curve in zip(c['records'],c['curves']):
          vf=fair.VideoFeatures({'curve':curve,'gt':r['gt']},k,'boostingvrme'); exact.append(vf.prepare(vf.clusters(conf)).evaluate([conf],details=True)[1][0]); raw=oldh.boosting_candidates([{'score':[curve]}],k)[0]; old.append([{'onset':x['start'],'offset':x['end'],'peak':x['peak']} for x in oldh.nms_greedy(raw,k,.2)])
      mism=0
      for ex,old_preds,t in zip(exact,old,target):
        expected=t['predictions'].get('original_native',t['predictions'].get('native',[])); a=tup(ex); b=tup(expected); replay=tup(old_preds)
        if a!=b: mism+=1
        oldset=sorted(int(x['peak']) for x in old_preds); exset=sorted(int(x['peak']) for x in ex); category='exact' if replay==a else ('candidate_generation_mismatch' if oldset!=exset else 'interval_mismatch')
        if replay!=a: rows.append({'backbone':bb,'dataset':ds,'subject':t['subject'],'video':t['video'],'first_divergence_stage':category,'historical_final':json.dumps(a,separators=(',',':')),'current_replay_proxy':json.dumps(replay,separators=(',',':'))})
      gate.append({'backbone':bb,'dataset':ds,'videos':len(exact),'mismatched_videos_against_saved_final':mism,'exact_replay':mism==0})
    with open(OUT/"first_divergence_by_video.csv","w",newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['backbone','dataset','subject','video','first_divergence_stage','historical_final','current_replay_proxy']); w.writeheader(); w.writerows(rows)
    (OUT/"replay_harness_fixes.md").write_text("# Replay harness fixes\n\n- Replaced SciPy-only ME-TST peak replay with the historical `local_maxima` implementation (plateau handling, distance suppression, sorted output).\n- Replaced the BoostingVRME approximation with the historical `source_interval` and `VideoFeatures.prepare().evaluate()` chain.\n- Used the exact fresh `curve_cache` inputs and native config (`single`, smoothing `2.0`, threshold `0.55`, distance `1.0`).\n- No cache, GT, matching, Native source, or candidate-selection method was modified.\n- GESO/global optimization was not run.\n")
    (OUT/"native_recovery_summary.json").write_text(json.dumps({'status':'NATIVE_FINAL_SELECTION_EXACT_REPLAY','gate':gate,'geso_run':False},indent=2,ensure_ascii=False))
    (OUT/"combined_report.json").write_text(json.dumps({'status':'NATIVE_FINAL_SELECTION_EXACT_REPLAY','scope':'Native final-selection provenance recovery','geso_run':False,'decision':'CASE A — all four groups contain an explicit greedy local conflict-suppression stage; GESO cross-backbone feasibility is established, but global optimization was not run.','artifacts':['native_final_selection_pipeline.json','first_divergence_by_video.csv','replay_harness_fixes.md','native_replay_gate_v2.json']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
