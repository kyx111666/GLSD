"""Attribute retained L events using their own labels and Gray candidate masks.
No counterfactual F1, no propagation of TP labels between configurations.
"""
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
INPUTS={'sammlv':'full_sammlv_20260922T125802_413296Z_grayzone.zip',
        'casme3':'full_casme3_20260922T134807_975740Z_grayzone.zip'}


def load_records(path):
    with zipfile.ZipFile(path) as z:
        members=[n for n in z.namelist() if n.endswith('/event_records.jsonl.gz')]
        assert len(members)==1
        return [json.loads(s) for s in gzip.decompress(z.read(members[0])).splitlines()]


def analyze(records):
    by={(r['subject'],r['method']):r for r in records}
    rows=[]
    missed_coverage=[]
    for subject in sorted({s for s,m in by}):
        l,g=by[subject,'L'],by[subject,'Gray']
        gc={t['video_id']:t for t in g['candidates']}
        lc={t['video_id']:t for t in l['candidates']}
        ghit={e['matched_gt_id'] for e in g['events'] if e['full_TP']}
        truth={t['gt_id']:t for t in g['ground_truth']}
        # Verify current saved geometry before testing unretained anchors.
        for r in (l,g):
            trace={t['video_id']:t for t in r['candidates']}
            for e in r['events']:
                p=e['interval'];k=trace[e['video_id']]['k']
                assert p[0]==p[6]-k and p[2]==p[6]+k and p[1]==p[3]==0
        for e in l['events']:
            if not e['full_TP'] or e['matched_gt_id'] in ghit:continue
            target=truth[e['matched_gt_id']]['registered_gt'];t=gc[e['video_id']];k=t['k']
            eligible=[]
            for i,p in enumerate(t['peaks']):
                inter=max(0,min(p+k,target[2])-max(p-k,target[0])+1)
                iou=inter/(2*k+1+target[2]-target[0]+1-inter)
                if iou>=.5:eligible.append(i)
            if not eligible:reason='no_Gray_candidate_reaches_IoU_0.5'
            elif any(t['retained'][i] for i in eligible):
                associated=[x for x in g['events'] if x['matched_gt_id']==e['matched_gt_id'] and x['raw_TP']]
                reason='recognition_filtered_raw_TP' if associated and not any(x['full_TP'] for x in associated) else 'eligible_retained_but_no_full_TP'
            elif all(t[g['config']['branch']][i]<g['config']['tau_low'] for i in eligible):reason='all_eligible_below_main_low'
            else:reason='eligible_gray_candidates_support_rejected'
            missed_coverage.append(dict(subject=subject,gt_id=e['matched_gt_id'],reason=reason,
                eligible_peaks=[t['peaks'][i] for i in eligible],reference_scale=g['config']['reference_scale']))
        for e in l['events']:
            vid,peak=e['video_id'],int(e['interval'][6])
            lt,gt=lc[vid],gc[vid]
            assert lt['response_sha256']==gt['response_sha256'] and lt['k']==gt['k']
            li=lt['peaks'].index(peak)
            assert lt['retained'][li]
            row=dict(subject=subject,video_name=lt['video_name'],video_id=vid,peak=peak,
                source='L independently selected; labels valid only in this source configuration',
                source_class='TP' if e['full_TP'] else 'FP' if e['full_FP'] else 'full_removed',
                gt_id=e['matched_gt_id'],L_source_G=lt['G'][li],L_source_L=lt['L'][li],
                L_a0=l['config']['reference_scale'],L_rho=l['config']['local_radius'],L_tau=l['config']['threshold'],
                Gray_a0=g['config']['reference_scale'],Gray_rho=g['config']['local_radius'],
                Gray_branch=g['config']['branch'],low=g['config']['tau_low'],high=g['config']['tau_high'],eta=g['config']['eta'],
                source_GT_missed=bool(e['full_TP'] and e['matched_gt_id'] not in ghit),
                source_peak_in_gray=peak in gt['peaks'],Gray_G=None,Gray_L=None,reason=None)
            if peak not in gt['peaks']:
                row['reason']='reference_candidate_changed'
            else:
                i=gt['peaks'].index(peak);row['Gray_G']=gt['G'][i];row['Gray_L']=gt['L'][i]
                b=gt[g['config']['branch']][i]
                if b<g['config']['tau_low']:reason='below_main_low'
                elif not gt['retained'][i]:reason='gray_support_rejected'
                else:reason='source_peak_retained'
                row['reason']=reason
            rows.append(row)
    summary=dict(missed_GT_candidate_coverage=missed_coverage,coverage_counts=dict(Counter(r['reason'] for r in missed_coverage)),source_events=len(rows),missed_GT_reasons=dict(Counter(r['reason'] for r in rows if r['source_GT_missed'])),
        all_source_classes={label:dict(Counter(r['reason'] for r in rows if r['source_class']==label)) for label in ('TP','FP','full_removed')},
        note='Attribution concerns the L source event anchor, not all alternative candidates covering the same GT. Candidate changes and full rematching are not score-gate errors.')
    return rows,summary


if __name__=='__main__':
    summaries={}
    for setting,name in INPUTS.items():
        input_root = Path(os.environ.get('GRAYZONE_INPUT_ROOT', 'data'))
        rows,summary=analyze(load_records(input_root/name))
        with (ROOT/(setting+'_L_event_attribution.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summaries[setting]=summary
    (ROOT/'saved_attribution_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
