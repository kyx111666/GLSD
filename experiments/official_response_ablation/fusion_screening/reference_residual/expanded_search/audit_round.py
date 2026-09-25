"""Audit the completed three-arm development round and its G-score ablation."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'structure_search'))
from verify_results import verify


def overlap(first,second):
    checks={}
    for m in ['G','L','Mean','PositiveResidual','RefResidual']:
        a=np.load(first/('search_counts_'+m+'.npz'))
        b=np.load(second/('search_counts_'+m+'.npz'))
        key=lambda c:tuple(c[k] for k in ['reference_scale','local_radius','threshold'])
        ix={key(c):i for i,c in enumerate(json.loads(str(b['configs'])))}
        ids=[ix[key(c)] for c in json.loads(str(a['configs']))]
        assert list(a['subjects'])==list(b['subjects'])
        for stage in ['raw','full']:np.testing.assert_array_equal(a[stage],b[stage][ids])
        checks[m]=len(ids)
    return checks


def main():
    audits={name:verify(ROOT/name) for name in ['local_run','local_fine','original_fine']}
    audits['coarse_fine_overlap']=overlap(ROOT/'local_run',ROOT/'local_fine')
    audits['original_coarse_fine_overlap']=overlap(ROOT.parent/'structure_search/local_run',ROOT/'original_fine')
    supplement=ROOT/'no_global_supplement'
    records=[json.loads(l) for l in gzip.open(supplement/'event_records.jsonl.gz','rt')]
    selected={r['subject']:r for r in csv.DictReader((ROOT/'local_fine/selected_configs.csv').open()) if r['method']=='RefResidual'}
    assert len(records)==29 and {r['subject'] for r in records}==set(selected)
    totals=np.zeros(3,dtype=int)
    for r in records:
        c=r['config'];s=selected[r['subject']]
        assert c['method']=='no_G'
        assert (c['reference_scale'],c['local_radius'],c['threshold'])==tuple(float(s[k]) for k in ['a0','rho','threshold'])
        for v in r['candidates']:
            score=np.clip(np.asarray(v['L'])-np.asarray(v['L0']),0,1)
            np.testing.assert_array_equal(score,v['score'])
            np.testing.assert_array_equal(score>=c['threshold'],v['retained'])
        for stage in ['raw','full']:
            tp=sum(e[stage+'_TP'] for e in r['events']);fp=sum(e[stage+'_FP'] for e in r['events'])
            assert r[stage+'_counts']==[tp,fp,len(r['ground_truth'])-tp]
        totals+=r['full_counts']
    summary=next(csv.DictReader((supplement/'summary_full.csv').open()))
    assert totals.tolist()==[int(summary[k]) for k in ['TP','FP','FN']]
    audits['no_global_supplement']=dict(status='PASS',subjects=29,full_counts=totals.tolist(),
        checks=['inherited full-method config','score exactly clip(L-L0)','candidate masks agree','event counts reconstruct'])
    (ROOT/'round_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    # Preserve original result metadata; make its inherited wording correction explicit.
    addendum=dict(applies_to=['local_run','local_fine'],
        corrections={'unchanged':'alignment, median operation, missing=0, decoder, recognition, matching',
                     'physical_scales':[1,1.5,2,2.5,3,4],'fixed_structures':36},
        explanation='Original generic driver metadata retained two old descriptive strings; grids and feature records contain actual expanded definitions.',
        radii=[.5,1,2,3,4,6],
        threshold_fine=[.01,.99,.01],threshold_coarse=[.05,.95,.05],
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        score_changes_after_run=False)
    (ROOT/'metadata_addendum.json').write_text(json.dumps(addendum,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'expanded_round_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['local_run','local_fine','original_fine','no_global_supplement']:
            for p in sorted((ROOT/name).rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        for name in ['RESULTS_FINE.md','round_audit.json','metadata_addendum.json']:
            z.write(ROOT/name,name)
    print('ROUND_AUDIT = PASS; evidence archive includes all three arms and supplementary ablation')


if __name__=='__main__':main()
