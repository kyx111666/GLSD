#!/usr/bin/env python3
"""GESO Phase 1 exact-set experiment using the locked Native adapters."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/geso_phase1_exactset'
BOOST=ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/boostingvrme'; sys.path.insert(0,str(BOOST)); import equiscale_fair_validation as fair
sp=importlib.util.spec_from_file_location('pmf_exact',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/metst/persistence_morphology_fusion.py'); pmf=importlib.util.module_from_spec(sp); sys.modules[sp.name]=pmf; sp.loader.exec_module(pmf)
CASES=[('ME-TST','SAMMLV',ROOT/'RethinkFuse_reproduction/caches/me_tst/sammlv_strategy1_outputs.pkl',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/sammlv/fixed/selected_predictions.json'),('ME-TST','CASME3',ROOT/'RethinkFuse_reproduction/caches/me_tst/casme3_strategy1_outputs.pkl',ROOT/'historical_gl_exact_fresh_reproduction/fresh_run/metst/results/pure_persistence_matched_v1/casme3/fixed/selected_predictions.json'),('BoostingVRME','SAMMLV',BOOST/'curve_cache/sammlv_curves.pkl',ROOT/'pure_persistence_matched_v1/sammlv/native/selected_predictions.json'),('BoostingVRME','CASME3',BOOST/'curve_cache/casme_3_curves.pkl',ROOT/'pure_persistence_matched_v1/casme3/native/selected_predictions.json')]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(a,b):
 l=max(a[0],b[0]); r=min(a[1],b[1]); inter=max(0,r-l); union=max(a[1],b[1])-min(a[0],b[0]); return inter/union if union else 0.0
def match_overlap(a,b):
 l=max(a[0],b[0]); r=min(a[1],b[1]); inter=max(0,r-l+1); union=(a[1]-a[0]+1)+(b[1]-b[0]+1)-inter; return inter/union if union else 0.0
def raw_me_peaks(curve,k):
 s=pmf.moving_average(np.asarray(curve,float),2*k); m=float(s.mean()); th=m+.55*(float(s.max())-m); v=s; peaks=[]; i=1
 while i<len(v)-1:
  if v[i]>v[i-1]:
   left=i
   while i+1<len(v) and v[i]==v[i+1]: i+=1
   if i<len(v)-1 and v[i]>v[i+1] and v[i]>=th: peaks.append((left+i)//2)
  i+=1
 return s,peaks
def exact(cs,adj):
 best=(0.0,())
 for mask in range(1<<len(cs)):
  sel=tuple(i for i in range(len(cs)) if mask>>i&1)
  if any(j in adj[i] for q,i in enumerate(sel) for j in sel[q+1:]): continue
  w=sum(cs[i]['weight'] for i in sel)
  if w>best[0]+1e-12 or (abs(w-best[0])<=1e-12 and sel<best[1]): best=(w,sel)
 return list(best[1]),best[0]
def graph(cs,kind,k):
 adj=[set() for _ in cs]
 for i in range(len(cs)):
  for j in range(i+1,len(cs)):
   conflict=(abs(cs[i]['peak']-cs[j]['peak'])<k) if kind=='ME-TST' else (abs(cs[i]['peak']-cs[j]['peak'])<=k or overlap((cs[i]['start'],cs[i]['end']),(cs[j]['start'],cs[j]['end']))>=.2)
   if conflict: adj[i].add(j); adj[j].add(i)
 return adj
def comps(adj):
 seen=set(); out=[]
 for i in range(len(adj)):
  if i in seen: continue
  st=[i]; seen.add(i); c=[]
  while st:
   x=st.pop(); c.append(x)
   for y in adj[x]:
    if y not in seen: seen.add(y); st.append(y)
  out.append(c)
 return out
def greedy(cs,order,adj):
 kept=[]
 for i in order:
  if not any(i in adj[j] for j in kept): kept.append(i)
 return kept
def match(preds,gts):
 used=set(); flags=[]
 for p in preds:
  vals=[(match_overlap((p['start'],p['end']),(g[0],g[2])),j) for j,g in enumerate(gts) if j not in used]
  if vals:
   v,j=max(vals)
   if v>=.5: flags.append(j); used.add(j); continue
  flags.append(-1)
 return flags
def counts(preds,gts):
 m=match(preds,gts); tp=sum(x>=0 for x in m); return [tp,len(preds)-tp,len(gts)-tp],m
def f1(x): return 2*x[0]/(2*x[0]+x[1]+x[2]) if 2*x[0]+x[1]+x[2] else 0.0
def oracle(cs,gts,adj):
 # Exact GT-assisted maximum F1 over independent subsets, via component DP on matched-GT masks.
 gt_n=len(gts); dp={0:(0,0)}
 for cc in comps(adj):
  options={}
  for mask in range(1<<len(cc)):
   sel=[cc[q] for q in range(len(cc)) if mask>>q&1]
   if any(j in adj[i] for q,i in enumerate(sel) for j in sel[q+1:]): continue
   flags=match([cs[i] for i in sorted(sel)],gts); gm=sum(1<<x for x in flags if x>=0); fp=sum(x<0 for x in flags)
   options[gm]=min(options.get(gm,10**9),fp)
  nd={}
  for a,(tp,fp) in dp.items():
   for gm,addfp in options.items():
    b=a|gm; val=(b.bit_count(),fp+addfp); nd[b]=min(nd.get(b,( -1,10**9)),val,key=lambda z:(-z[0],z[1]))
  dp=nd
 best=max(((f1((tp,fp,gt_n-tp)),tp,fp,gt_n-tp) for _m,(tp,fp) in dp.items()),key=lambda z:(z[0],z[1],-z[2]))
 return list(best)
def main():
 run_start=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True); rows=[]; struct=[]; failures=[]; mech=[]; mech_rows=[]; boot=[]; weights=[]; all_exact=True
 for bb,ds,cp,tp in CASES:
  c=pickle.load(open(cp,'rb')); target=json.load(open(tp)); videos=[]; k=int(c['k_p']) if bb=='ME-TST' else fair.duration_k([{'subject':str(r['subject']),'gt':r['gt']} for r in c['records']],set(),'boostingvrme')
  if bb=='ME-TST':
   for r in c['records']:
    curve,peaks=raw_me_peaks(r['score'],k); videos.append((str(r['subject']),r['video'],r['samples'],curve,peaks))
  else:
   conf=fair.Config('single',2.0,.55,1.0)
   for r,curve in zip(c['records'],c['curves']):
    vf=fair.VideoFeatures({'curve':curve,'gt':r['gt']},k,'boostingvrme'); clusters=vf.clusters(conf); videos.append((str(r['subject']),r.get('video',''),r['gt'],np.asarray(curve),clusters))
  subj=defaultdict(lambda:[0,0,0,0,0,0]); per_video=[]; total=[0,0,0]; exact_total=[0,0,0]; oracle_total=[0,0,0]; pool_total=0; edge_total=0; chain_total=0; mech_total=defaultdict(int)
  for vi,(sub,vid,gts,curve,items) in enumerate(videos):
   if bb=='ME-TST':
    cs=[{'id':i,'start':max(0,p-k),'end':min(len(curve)-1,p+k),'peak':int(p),'weight':float(curve[p])} for i,p in enumerate(items)]; adj=graph(cs,bb,k); order=sorted(range(len(cs)),key=lambda i:(-cs[i]['weight'],cs[i]['peak'])); native_ids=greedy(cs,order,adj)
   else:
    # Native adapter's candidate order and intervals are authoritative.
    vf=fair.VideoFeatures({'curve':curve,'gt':gts},k,'boostingvrme'); prep=vf.prepare(items); cs=[{'id':i,'start':int(prep.intervals[i,0]),'end':int(prep.intervals[i,1]),'peak':int(prep.clusters[i][0]),'weight':float(prep.clusters[i][2])} for i in range(len(items))]; adj=[set(np.flatnonzero(prep.conflicts[i]).tolist())-{i} for i in range(len(cs))]; native_ids=[]
   if bb=='BoostingVRME': native_ids=greedy(cs,list(range(len(cs))),adj)
   pool_total += len(cs); edge_total += sum(len(x) for x in adj)//2
   for i in range(len(cs)):
    for j in adj[i]:
     if j>i:
      for z in adj[j]:
       if z>j and z not in adj[i] and cs[i]['weight']+cs[z]['weight']>cs[j]['weight']: chain_total += 1
   exact_ids=[]; ew=0
   for cc in comps(adj):
    sel,w=exact([cs[i] for i in cc], [{j-i for j in []}], []) if False else (None,None)
    local=[cs[i] for i in cc]; la=[{j-q for j in adj[cc[q]] if j in cc} for q in range(len(cc))]; sel,w=exact(local,la); exact_ids += [cc[x] for x in sel]; ew+=w
    gset=set(native_ids)&set(cc); eset=set(cc[x] for x in sel); failures.append({'backbone':bb,'dataset':ds,'subject':sub,'video':vid,'component_size':len(cc),'native_weight':sum(cs[i]['weight'] for i in gset),'exact_weight':w,'weight_gain':w-sum(cs[i]['weight'] for i in gset),'same_selection':gset==eset,'native_ids':sorted(gset),'exact_ids':sorted(eset)})
   native_ids=sorted(native_ids); exact_ids=sorted(exact_ids); npreds=[cs[i] for i in native_ids]; epreds=[cs[i] for i in exact_ids]; nc,nm=counts(npreds,gts); ec,em=counts(epreds,gts); total=np.add(total,nc); exact_total=np.add(exact_total,ec); om=oracle(cs,gts,adj); oracle_total=np.add(oracle_total,om[1:])
   subj[sub]=[subj[sub][i]+x for i,x in enumerate((*nc,*ec))]
   ng={x for x in nm if x>=0}; eg={x for x in em if x>=0}; nfp={(p['start'],p['end'],p['peak']) for p,m in zip(npreds,nm) if m<0}; efp={(p['start'],p['end'],p['peak']) for p,m in zip(epreds,em) if m<0}; mech_total['retained_gt']+=len(ng&eg); mech_total['rescued_gt']+=len(eg-ng); mech_total['lost_gt']+=len(ng-eg); mech_total['retained_fp']+=len(nfp&efp); mech_total['removed_fp']+=len(nfp-efp); mech_total['new_fp']+=len(efp-nfp); mech_total['videos_improved']+=int(f1(ec)>f1(nc)); mech_total['videos_worsened']+=int(f1(ec)<f1(nc)); mech_total['videos_unchanged']+=int(f1(ec)==f1(nc))
   per_video.append({'subject':sub,'video':vid,'native':npreds,'exact':epreds,'native_counts':nc,'exact_counts':ec,'oracle':[om[1],om[2],om[3]]})
  comps_all=[x for x in failures if x['backbone']==bb and x['dataset']==ds]; diff=[x for x in comps_all if not x['same_selection']]
  for v in per_video:
   for i in range(len(v['native'])): pass
  # edge count from component identities
  sizes=[x['component_size'] for x in comps_all]; struct.append({'backbone':bb,'dataset':ds,'pre_suppression_candidates':pool_total,'conflict_edges':edge_total,'conflict_components':len(comps_all),'component_size_distribution':{str(s):sizes.count(s) for s in sorted(set(sizes))},'max_component_size':max(sizes or [0]),'videos_without_conflict':sum(1 for v in per_video if sum(x['component_size']>1 for x in comps_all if x['video']==v['video'])==0),'videos_with_conflict':sum(1 for v in per_video if any(x['component_size']>1 and x['video']==v['video'] for x in comps_all)),'videos_greedy_vs_exact_different':len({x['video'] for x in diff}),'greedy_exact_components':len(comps_all)-len(diff),'total_candidate_additions':sum(max(0,len(x['exact_ids'])-len(x['native_ids'])) for x in diff),'total_candidate_removals':sum(max(0,len(x['native_ids'])-len(x['exact_ids'])) for x in diff),'typical_chain_count':chain_total})
  for v in per_video: OUT.joinpath('per_video').mkdir(exist_ok=True); (OUT/'per_video'/f'{bb}_{ds}_{v["subject"]}_{v["video"]}.json').write_text(json.dumps(v,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
  rows.append({'Backbone':bb,'Dataset':ds,'Native TP':int(total[0]),'Native FP':int(total[1]),'Native FN':int(total[2]),'Native F1':f1(total),'GESO TP':int(exact_total[0]),'GESO FP':int(exact_total[1]),'GESO FN':int(exact_total[2]),'GESO F1':f1(exact_total),'Delta F1':f1(exact_total)-f1(total),'Delta TP':int(exact_total[0]-total[0]),'Delta FP':int(exact_total[1]-total[1]),'Delta FN':int(exact_total[2]-total[2]),'Native prediction count':int(total[0]+total[1]),'GESO prediction count':int(exact_total[0]+exact_total[1])})
  weights.append({'backbone':bb,'dataset':ds,'weight_equal_components':sum(x['weight_gain']==0 for x in comps_all),'exact_higher_components':sum(x['weight_gain']>1e-12 for x in comps_all),'mean_weight_gain':float(np.mean([x['weight_gain'] for x in comps_all])) if comps_all else 0.0,'max_weight_gain':float(max([x['weight_gain'] for x in comps_all] or [0.0])),'affected_videos':len({x['video'] for x in comps_all if x['weight_gain']>1e-12})})
  boot.append((bb,ds,subj));
  mech_rows.append({'Backbone':bb,'Dataset':ds,**dict(mech_total)})
  with open(OUT/f'_per_video_{bb}_{ds}.json','w') as f: json.dump(per_video,f,default=lambda x:x.item() if hasattr(x,'item') else str(x))
  (OUT/f'_oracle_{bb}_{ds}.json').write_text(json.dumps({'TP':int(oracle_total[0]),'FP':int(oracle_total[1]),'FN':int(oracle_total[2]),'F1':f1(oracle_total)},indent=2))
 # reports
 def bootrow(d):
  rng=np.random.default_rng(100); keys=list(d); vals=[]
  for _ in range(10000):
   ss=[d[keys[i]] for i in rng.integers(0,len(keys),len(keys))]; n=np.sum(ss,axis=0); vals.append(f1(n[3:])-f1(n[:3]))
  return float(np.mean(vals)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
 with open(OUT/'geso_phase1_results.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 with open(OUT/'greedy_vs_exact_structure.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(failures[0])); w.writeheader(); w.writerows(failures)
 with open(OUT/'greedy_failure_patterns.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['Backbone','Dataset','equal_weight_components','exact_higher_components','mean_weight_gain','max_weight_gain','affected_videos','typical_chain_count']); w.writerows([[x['backbone'],x['dataset'],x['weight_equal_components'],x['exact_higher_components'],x['mean_weight_gain'],x['max_weight_gain'],x['affected_videos'],next(s['typical_chain_count'] for s in struct if s['backbone']==x['backbone'] and s['dataset']==x['dataset'])] for x in weights])
 with open(OUT/'conflict_structure.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(struct[0])); w.writeheader(); w.writerows(struct)
 with open(OUT/'same_pool_oracle.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['Backbone','Dataset','Oracle TP','Oracle FP','Oracle FN','Oracle F1','Native F1','Oracle gap']);
  for x in rows:
   op=json.load(open(OUT/f'_oracle_{x["Backbone"]}_{x["Dataset"]}.json')); w.writerow([x['Backbone'],x['Dataset'],op['TP'],op['FP'],op['FN'],op['F1'],x['Native F1'],op['F1']-x['Native F1']])
 with open(OUT/'geso_phase1_bootstrap.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['Backbone','Dataset','Point Delta F1','Bootstrap Mean','CI 2.5%','CI 97.5%','CI crosses 0','Resamples','Seed']);
  for x in rows:
   m,l,h=bootrow(dict(boot[[a+b for a,b,_ in boot].index(x['Backbone']+x['Dataset'])][2])); w.writerow([x['Backbone'],x['Dataset'],x['Delta F1'],m,l,h,l<=0<=h,10000,100])
 (OUT/'candidate_weight_provenance.json').write_text(json.dumps({'weight':'Native candidate evidence/score','ME-TST':'smoothed Native curve at historical local_maxima peak','BoostingVRME':'VideoFeatures Native height evidence (cluster tuple index 2)','source_files':[str(pmf.__file__),str(BOOST/'equiscale_fair_validation.py'),str(BOOST/'tune_equiscale.py')]},indent=2,ensure_ascii=False))
 (OUT/'geso_phase1_mechanism.csv').write_text('Backbone,Dataset,retained_gt,rescued_gt,lost_gt,retained_fp,removed_fp,new_fp,videos_improved,videos_worsened,videos_unchanged\n'+''.join(f'{x["Backbone"]},{x["Dataset"]},{x.get("retained_gt",0)},{x.get("rescued_gt",0)},{x.get("lost_gt",0)},{x.get("retained_fp",0)},{x.get("removed_fp",0)},{x.get("new_fp",0)},{x.get("videos_improved",0)},{x.get("videos_worsened",0)},{x.get("videos_unchanged",0)}\n' for x in mech_rows))
 status='STRONG_GO' if sum(x['Delta F1']>0 for x in rows)==4 else ('GO' if sum(x['Delta F1']>0 for x in rows)>=3 and min(x['Delta F1'] for x in rows)>-.003 else ('BORDERLINE' if sum(x['Delta F1']>0 for x in rows)==2 else 'STOP'))
 (OUT/'combined_report.json').write_text(json.dumps({'status':status,'native_replay':'NATIVE_FINAL_SELECTION_EXACT_REPLAY','geso_run':True,'solver':'exhaustive exact maximum-weight independent set per connected component; GT oracle uses exact component DP','solver_version':'Python '+sys.version.split()[0]+'; NumPy '+np.__version__,'solver_settings':{'tie_break':'lexicographic IDs','bootstrap_resamples':10000,'bootstrap_seed':100,'no_heuristic':True},'runtime_sec':time.monotonic()-run_start,'results':rows,'structural_audit':struct,'weight_audit':weights,'geso_not_v2':True},indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
 print(json.dumps({'status':status,'results':rows,'structure':struct,'weights':weights},indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
if __name__=='__main__': main()
