"""Adversarial checks on the existing transfer runner, without editing it."""
import copy
import json
from my_method.glsd_evidence_audit import ROOT,OUT,manifest
from my_method.gl_saliency_skill.evidence import configuration_grid
from my_method.gl_saliency_skill.benchmark import write_json,write_csv
from my_method.strict_cross_backbone_transfer.run import load_context,run_direction,TRANSFER_PAIRS

def main():
    configs=configuration_grid(); before=manifest()
    previous=json.loads((ROOT/'results/final_gl_skill_transfer/per_video_transferred_predictions.json').read_text())
    contexts={(b,d):load_context(b,d,configs) for b in ('metst','boostingvrme') for d in ('sammlv','casme3')}
    checks=[]
    for source,target,dataset in TRANSFER_PAIRS:
        src=contexts[source,dataset]
        tgt=dict(contexts[target,dataset])
        # Remove target training-performance metadata and replace comparator choices.
        tgt.pop('protocol');tgt.pop('selection_rows');tgt.pop('inner');tgt.pop('signed_root')
        tgt['selected']={s:{'GL':0} for s in tgt['bundle'].subjects}
        result=run_direction(src,tgt,configs)
        expected=[r for r in previous if r['source']==source and r['target']==target and r['dataset']==dataset]
        assert result['predictions']==expected,(source,target,dataset)
        checks.append({'source':source,'target':target,'dataset':dataset,
            'target_performance_metadata_removed':True,'target_comparator_configs_replaced_with_native':True,
            'transferred_per_video_predictions_exact':True,'videos':len(expected)})
        print('isolation passed',source,target,dataset,flush=True)
    assert before==manifest()
    write_json(OUT/'transfer_adversarial_audit.json',{'checks':checks,'skill_unchanged':True,
        'scope':'postprocessing implementation isolation only; does not establish checkpoint training provenance'})
    import csv
    with (OUT/'fair_baselines.csv').open() as f: rows=list(csv.DictReader(f))
    net=[]
    for b in ('metst','boostingvrme'):
        for d in ('sammlv','casme3'):
            lookup={r['method']:r for r in rows if r['backbone']==b and r['dataset']==d}
            n,g=lookup['Author Native'],lookup['GL']
            net.append({'backbone':b,'dataset':d,**{f'delta_{k}':int(g[k])-int(n[k]) for k in ('TP','FP','FN')},
                'delta_precision':float(g['precision'])-float(n['precision']),
                'delta_recall':float(g['recall'])-float(n['recall']),
                'delta_F1':float(g['F1'])-float(n['F1'])})
    write_csv(OUT/'mechanism_net_changes.csv',net)

if __name__=='__main__':main()
