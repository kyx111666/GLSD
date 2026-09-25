from pathlib import Path
import csv, json, pickle, sys
from collections import Counter
import numpy as np
from my_method.gl_saliency_skill.benchmark import load_bundle, signed_context
from my_method.gl_saliency_skill.evidence import configuration_grid
from my_method.gl_saliency_skill.selection import choose, inner_counts, indexed_pools
from my_method.gl_saliency_skill.skill import GLSaliencySkill
from my_method.gl_saliency_skill.evaluation import evaluate, metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/glsd_str_evaluation'; OUT.mkdir(parents=True,exist_ok=True)
GROUPS=(('metst','sammlv'),('metst','casme3'),('boostingvrme','sammlv'),('boostingvrme','casme3'))
LOCKED={('metst','sammlv'):(48,126,111),('metst','casme3'):(96,1085,762),('boostingvrme','sammlv'):(42,95,117),('boostingvrme','casme3'):(120,1148,738)}
NATIVE={('metst','sammlv'):(53,184,106),('metst','casme3'):(81,912,777),('boostingvrme','sammlv'):(49,145,110),('boostingvrme','casme3'):(93,818,765)}

def f1(c):
 t,p,n=c; return 2*t/(2*t+p+n) if 2*t+p+n else 0.
def rec_f1(gt,pred):
 # paper_metrics macro precision/recall harmonic F1, classes 0/1/2
 ps=[]; rs=[]
 for k in (0,1,2):
  tp=sum(a==k and b==k for a,b in zip(gt,pred)); fp=sum(a!=k and b==k for a,b in zip(gt,pred)); fn=sum(a==k and b!=k for a,b in zip(gt,pred))
  ps.append(tp/(tp+fp) if tp+fp else 0.); rs.append(tp/(tp+fn) if tp+fn else 0.)
 mp=float(np.mean(ps)); mr=float(np.mean(rs)); return 2*mp*mr/(mp+mr) if mp+mr else 0.
def majority(arr,on,off,peak):
 vals=list(np.asarray(arr)[max(0,int(on)+1):max(1,int(off)-1)])
 if not vals and len(arr): vals=[arr[min(max(0,int(peak)),len(arr)-1)]]
 return int(Counter(int(x) for x in vals).most_common(1)[0][0]) if vals else 3
def raw_records(backbone,dataset):
 if backbone=='metst':
  p=ROOT/'caches/me_tst'/f'{dataset}_strategy1_outputs.pkl'; d=pickle.load(open(p,'rb')); return {(str(x['subject']),str(x['video'])):x for x in d['records']}
 return {}
def run_group(backbone,dataset):
 bundle=load_bundle(backbone,dataset); configs=configuration_grid(); pools=indexed_pools(configs); pools['H']=[i for i,c in enumerate(configs) if c.family=='height']
 _,_,outer,inner,stats,_=signed_context(bundle,configs); selected={}
 for held,s in enumerate(bundle.subjects):
  selected[s]={m:choose(inner_counts(stats,held,inner),pool) for m,pool in pools.items()}
 lookup={s:i for i,s in enumerate(bundle.subjects)}; raw=raw_records(backbone,dataset)
 rows=[]; totals=Counter(); native_tot=Counter(); rg=[]; rn=[]; subject_pairs={'glsd':{},'native':{}}
 for rec in bundle.records:
  si=lookup[rec.subject]
  for tag in ('glsd','native'):
   cfg=configs[selected[rec.subject]['GL']] if tag=='glsd' else configs[0]; k=int(outer[si]) if tag=='glsd' else bundle.legacy_temporal_scale
   ev=GLSaliencySkill(cfg).decode(rec.score,k,{'interval_adapter':bundle.interval_adapter}); c,det=evaluate(ev,rec.ground_truth); (totals if tag=='glsd' else native_tot).update(c)
   if backbone=='metst':
    src=raw[(rec.subject,rec.video)]; gt_names=src['gt_emotions'];
    for e in det:
     j=e['matched_gt'];
     if j>=0:
      gid={'negative':0,'positive':1,'surprise':2,'others':3}.get(str(gt_names[j]),3)
      pair=(gid,majority(src['emotion'],e['onset'],e['offset'],e['peak']))
      if gid!=3:
       rg.append(pair); subject_pairs[tag].setdefault(rec.subject,[]).append(pair)
       if tag=='native': rn.append(pair)
   rows.append((rec.subject,tag,c))
 return bundle,totals,native_tot,rg,rn,rows,subject_pairs

summary=[]; per=[]; audit=['# GLSD Spot-then-Recognize evaluation audit','', 'Evaluator: frozen GLSD decoder + project chronological IoU evaluator (`gl_saliency_skill.evaluation.evaluate`), IoU >= 0.5, inclusive endpoints, one GT match, no second-best rematching. Recognition reuses paper-aligned majority-vote and macro precision/recall harmonic F1 over classes negative/positive/surprise; `others` excluded. STRS = spotting F1 × Recognition F1.','']
for b,d in GROUPS:
 bundle,tot,nt,rg,rn,rows,subject_pairs=run_group(b,d); obs=(tot['TP'],tot['FP'],tot['FN']); lock=LOCKED[(b,d)]; replay=obs==lock
 if not replay: raise RuntimeError(f'locked replay mismatch {b}/{d}: {obs} != {lock}')
 rf=rec_f1(*zip(*rg)) if rg else None; nrf=rec_f1(*zip(*rn)) if rn else None
 sf=f1(obs); nsf=f1(NATIVE[(b,d)]); strs=sf*rf if rf is not None else None; nstrs=nsf*nrf if nrf is not None else None
 summary.append(dict(backbone=b,dataset=d,spot_tp=obs[0],spot_fp=obs[1],spot_fn=obs[2],spot_f1=sf,recognition_f1='' if rf is None else rf,strs='' if strs is None else strs,native_spot_f1=nsf,native_recognition_f1='' if nrf is None else nrf,native_strs='' if nstrs is None else nstrs,replay='PASS'))
 for s in bundle.subjects:
  for tag in ('glsd','native'):
   c=Counter(); [c.update(x[2]) for x in rows if x[0]==s and x[1]==tag]; pairs=subject_pairs[tag].get(s,[]); sr=rec_f1(*zip(*pairs)) if pairs else None; sf0=f1((c['TP'],c['FP'],c['FN'])); per.append(dict(backbone=b,dataset=d,subject=s,method=tag,spot_tp=c['TP'],spot_fp=c['FP'],spot_fn=c['FN'],spot_f1=sf0,recognition_f1='' if sr is None else sr,strs='' if sr is None else sf0*sr))
 audit += [f'- {b}/{d}: GLSD replay {obs} vs locked {lock}: **PASS**; recognition mapping: '+('ME-TST逐帧 frozen emotion available.' if b=='metst' else 'UNAVAILABLE: Boosting cache lacks frozen emotion_pred/logits; no STRS fabricated.'), '']
with open(OUT/'summary.csv','w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
with open(OUT/'per_subject.csv','w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
audit += ['No GLSD formula, definitions, thresholds, configs, geometry, NMS, prediction, or training changed; no retraining or parameter search occurred. BoostingVRME Recognition F1/STRS are not identifiable from the signed cache and are reported as unavailable.']
(OUT/'audit.md').write_text('\n'.join(audit),encoding='utf-8')
print(json.dumps(summary,indent=2))
