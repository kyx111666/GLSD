import csv, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/geso_phase1'
sys.path.insert(0,str(ROOT)); import run_geso_phase1 as old
sys.path.insert(0,str(ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme')); import equiscale_fair_validation as fair
sp=importlib.util.spec_from_file_location('pmf2',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py'); pmf=importlib.util.module_from_spec(sp); sys.modules[sp.name]=pmf; sp.loader.exec_module(pmf)
cases=[('ME-TST','SAMMLV',ROOT/'RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl',ROOT/'RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json'),('ME-TST','CASME3',ROOT/'RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl',ROOT/'RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/casme3/fixed/selected_predictions.json'),('BoostingVRME','SAMMLV',ROOT/'RethinkFuse_reproduction/caches/boostingvrme/sammlv_curves.pkl',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme/curve_cache/sammlv_curves.pkl',ROOT/'pure_persistence_matched_v1/sammlv/native/selected_predictions.json'),('BoostingVRME','CASME3',ROOT/'RethinkFuse_reproduction/caches/boostingvrme/casme3_curves.pkl',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme/curve_cache/casme_3_curves.pkl',ROOT/'pure_persistence_matched_v1/casme3/native/selected_predictions.json')]
def ts(xs): return sorted((int(x.get('onset',x.get('start'))),int(x.get('offset',x.get('end'))),int(x['peak'])) for x in xs)
def main():
 rows=[]
 for bb,ds,current_cp,hist_cp,target_cp in cases:
  cur=pickle.load(open(current_cp,'rb')); hist=pickle.load(open(hist_cp,'rb')); targets=json.load(open(target_cp)); k=int(hist['k_p']) if bb=='ME-TST' else fair.duration_k([{'subject':str(r['subject']),'gt':r['gt']} for r in hist['records']],set(),'boostingvrme')
  if bb=='ME-TST': hist_out=[pmf.native_detections(r,hist)[0] for r in hist['records']]; cur_out=[old.nms_greedy(x,k,0) for x in old.metst_candidates(cur['records'],k)]; cur_curves=[r['score'] for r in cur['records']]; hist_curves=[r['score'] for r in hist['records']]
  else:
   conf=fair.Config('single',2.0,.55,1.0); hist_out=[]
   for r,c in zip(hist['records'],hist['curves']):
    v=fair.VideoFeatures({'curve':c,'gt':r['gt']},k,'boostingvrme'); hist_out.append(v.prepare(v.clusters(conf)).evaluate([conf],details=True)[1][0])
   cur_out=[]; cur_curves=[]
   for sr in cur['subject_curves']:
    for c in sr['score']: cur_out.append([{'onset':x['start'],'offset':x['end'],'peak':x['peak']} for x in old.nms_greedy(old.boosting_candidates([{'score':[c]}],k)[0],k,.2)]); cur_curves.extend(sr['score'])
   hist_curves=hist['curves']
  for ex,co,hc,cc,t in zip(hist_out,cur_out,hist_curves,cur_curves,targets):
   expected=t['predictions'].get('original_native',t['predictions'].get('native',[])); a=ts(ex); b=ts(co)
   if a==b: continue
   if len(hc)!=len(cc) or not np.array_equal(np.asarray(hc),np.asarray(cc)): stage='raw_temporal_curve/cache_mismatch'
   elif bb=='ME-TST' and sorted(x['peak'] for x in ex)!=sorted(x['peak'] for x in co): stage='candidate_generation_mismatch'
   elif a!=b: stage='interval_construction_or_native_suppression_mismatch'
   rows.append({'backbone':bb,'dataset':ds,'subject':t['subject'],'video':t['video'],'first_divergence_stage':stage,'historical_final':json.dumps(a,separators=(',',':')),'current_replay_proxy':json.dumps(b,separators=(',',':'))})
 with open(OUT/'first_divergence_by_video.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['backbone','dataset','subject','video','first_divergence_stage','historical_final','current_replay_proxy']); w.writeheader(); w.writerows(rows)
 print(json.dumps({'rows':len(rows),'by_stage':{s:sum(r['first_divergence_stage']==s for r in rows) for s in sorted(set(r['first_divergence_stage'] for r in rows))}},ensure_ascii=False))
if __name__=='__main__': main()
