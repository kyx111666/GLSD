"""LOSO evidence diagnostics on geometry eligibility, not spotting performance.
All structures reported. No outer result is used to choose structure or a variant.
"""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from mechanism_features import binary_logloss,fit_predict,logit_scores


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(output):
    output=Path(output);groups=defaultdict(list);coverage=[]
    for p in sorted((output/'subjects').glob('*.json.gz')):
        with gzip.open(p,'rt') as f:subject=json.load(f)
        coverage.extend(subject['coverage'])
        for v in subject['features']:
            base=v['scales']['0.5']
            for i,peak in enumerate(v['peaks']):
                groups[v['a0'],v['rho']].append(dict(subject=subject['subject'],video_id=v['video_id'],peak=peak,
                    y=int(v['geometry_eligible'][i]),G=v['G'][i],L=v['L'][i],
                    L_gamma025=v['scales']['0.25']['median'][i],L_gamma075=v['scales']['0.75']['median'][i],
                    L_gamma1=v['scales']['1.0']['median'][i],L_mean=base['mean'][i],
                    L_matched=base['matched_only_median'][i],
                    missing=sum(not b for b in base['matched'][i]),
                    median_zero=float(v['L'][i]==0),
                    nearest_outside=any(d>base['tolerance'] for d in base['nearest_distance'][i])))
    summaries=[];folds=[];bins=[];effects=[]
    for (a,r),rows in sorted(groups.items()):
        subjects=np.array([x['subject'] for x in rows]);unique=sorted(set(subjects))
        y=np.array([x['y'] for x in rows]);l=np.array([x['L'] for x in rows]);g=np.array([x['G'] for x in rows])
        pred={m:np.zeros(len(y)) for m in ('L_only','L_plus_G')}
        for s in unique:
            tr=subjects!=s;te=~tr
            if not tr.any() or not te.any():raise RuntimeError('empty training/heldout pool')
            for m,xx in [('L_only',logit_scores(l)[:,None]),('L_plus_G',logit_scores(np.column_stack((l,g))))]:
                pp,info=fit_predict(xx[tr],y[tr],subjects[tr],xx[te]);pred[m][te]=pp
                folds.append(dict(a0=a,rho=r,subject=s,model=m,heldout_candidates=int(te.sum()),
                    geometry_positives=int(y[te].sum()),logloss=float(binary_logloss(y[te],pp).mean()),
                    coefficients=json.dumps(info,sort_keys=True)))
            # Descriptive conditional test: L-bin boundaries and G median from training only.
            edges=np.unique(np.quantile(l[tr],[.2,.4,.6,.8]))
            trainbin=np.searchsorted(edges,l[tr],side='right');testbin=np.searchsorted(edges,l[te],side='right')
            for bi in range(len(edges)+1):
                ti=trainbin==bi;ei=testbin==bi
                if not ti.any() or not ei.any():continue
                cut=float(np.median(g[tr][ti]));yy=y[te][ei];gg=g[te][ei]
                for high in (False,True):
                    keep=gg>=cut if high else gg<cut
                    bins.append(dict(a0=a,rho=r,subject=s,L_bin=bi,G_group='high' if high else 'low',
                        train_L_edges=json.dumps(edges.tolist()),train_G_median=cut,candidates=int(keep.sum()),
                        geometry_positives=int(yy[keep].sum())))
        loss0=binary_logloss(y,pred['L_only']);loss1=binary_logloss(y,pred['L_plus_G'])
        delta=np.array([(loss1[subjects==s]-loss0[subjects==s]).mean() for s in unique])
        draws=np.random.default_rng(100).integers(0,len(unique),(10000,len(unique)))
        boot=delta[draws].mean(1)
        summaries.append(dict(a0=a,rho=r,candidates=len(y),geometry_positives=int(y.sum()),
            subjects=len(unique),L_logloss=float(np.mean([loss0[subjects==s].mean() for s in unique])),
            LG_logloss=float(np.mean([loss1[subjects==s].mean() for s in unique])),
            delta_LG_minus_L=float(delta.mean()),CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975)),
            improved_subjects=int((delta<0).sum()),interpretation='negative logloss delta favors G; geometric diagnostic, not full F1; no multiplicity correction'))
        for label in (0,1):
            yy=y==label
            for name in ('L_gamma025','L_gamma075','L_gamma1','L_mean','L_matched'):
                alt=np.array([row[name] for row in rows]);d=alt-l
                effects.append(dict(a0=a,rho=r,geometry_label=label,variant=name,candidates=int(yy.sum()),
                    mean_delta_L=float(d[yy].mean()) if yy.any() else None,
                    fraction_increase_ge005=float((d[yy]>=.05).mean()) if yy.any() else None,
                    fraction_decrease_ge005=float((d[yy]<=-.05).mean()) if yy.any() else None,
                    default_missing_fraction=float(np.mean([row['missing']>0 for row in rows if row['y']==label])) if yy.any() else None))
        print('conditional diagnosis',a,r,'complete',flush=True)
    radius_rows=[]
    for a in sorted({k[0] for k in groups}):
        pools={r:{(x['subject'],x['video_id'],x['peak']):x for x in groups[a,r]} for r in (.5,1.,2.,3.)}
        for r0,r1 in ((.5,1.),(1.,2.),(2.,3.)):
            assert pools[r0].keys()==pools[r1].keys()
            for label in (0,1):
                keys=[k for k,v in pools[r0].items() if v['y']==label]
                ds=np.array([pools[r1][k]['L']-pools[r0][k]['L'] for k in keys])
                radius_rows.append(dict(a0=a,rho_from=r0,rho_to=r1,geometry_label=label,candidates=len(keys),
                    mean_delta_L=float(ds.mean()) if len(ds) else None,
                    fraction_increase_ge005=float((ds>=.05).mean()) if len(ds) else None))
    write_csv(output/'radius_effects.csv',radius_rows)
    write_csv(output/'conditional_summary.csv',summaries);write_csv(output/'conditional_folds.csv',folds)
    write_csv(output/'heldout_L_bins_G_groups.csv',bins);write_csv(output/'scale_effects.csv',effects)
    write_csv(output/'candidate_GT_coverage.csv',coverage)
    (output/'analysis_notes.json').write_text(json.dumps(dict(
        primary='fixed L-only versus L+G ridge logistic; training-only scaling and equal-subject weights',
        label='any official IoU>=0.5 for a candidate before assignment and recognition; can duplicate GT; NOT TP',
        secondary='same training-defined L quantile bins, G split at training-bin median',
        variants='one-factor evidence changes only, all listed; no selected winner or counterfactual F1',
        boundary='no support-set/peak-distance/event-width change; a0 fixed per reported analysis',
        inference='subject bootstrap conditional on fitted OOF predictions, no refitting or development correction',
        next='Only a subsequent complete rematched/full evaluation can establish method gains.'),indent=2)+'\n')
    return summaries
