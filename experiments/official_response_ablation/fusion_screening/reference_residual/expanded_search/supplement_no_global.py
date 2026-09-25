"""Replay missing G-score ablation at preselected RefResidual configurations.

Read existing selections; write a separate supplement without editing results.
"""
import argparse
import contextlib
import csv
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_expanded as expanded
base=expanded.base


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dump',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    protocol=json.loads((args.result/'protocol.json').read_text())
    base.SCALES=tuple(protocol['scales'])
    ctx,core=expanded.expanded_context(args.dump)
    selected={r['subject']:r for r in csv.DictReader((args.result/'selected_configs.csv').open()) if r['method']=='RefResidual'}
    expected={r['subject']:[int(r[k]) for k in ('TP','FP','FN')] for r in csv.DictReader((args.result/'per_subject_counts.csv').open()) if r['method']=='RefResidual' and r['stage']=='full'}
    if (args.output/'protocol.json').exists():
        raise RuntimeError('Supplement already complete; use a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];counts=[];primary=[]
    with contextlib.ExitStack() as stack, gzip.open(args.output/'event_records.jsonl.gz','wt') as stream:
        for cls in [ctx[2],ctx[3].spotting.__globals__['MeanAveragePrecision2d']]:
            stack.enter_context(base.runtime.typed_matching.install(cls))
        for si,s in enumerate(ctx[6]):
            r=selected[s]
            c=base.engine.phase.Config('RefResidual',int(r['config_id']),float(r['a0']),float(r['rho']),float(r['threshold']))
            original=base.engine.decode(ctx,core,si,c,True)
            assert original['full_counts']==expected[s]
            nc=base.engine.phase.Config('no_G',-1,c.reference_scale,c.local_radius,c.threshold)
            removed=base.engine.decode(ctx,core,si,nc,True)
            removed['method']='locked_no_global'
            stream.write(json.dumps(base.engine.shared.serializable(removed),allow_nan=False)+'\n')
            for stage in ['raw','full']:
                rows.append(dict(subject=s,method='locked_no_global',stage=stage,**base.engine.shared.metrics(removed[stage+'_counts'])))
            counts.append(removed['full_counts']);primary.append(original['full_counts'])
    counts=np.array(counts);primary=np.array(primary)
    base.engine.shared.write_csv(args.output/'summary_full.csv',[dict(method='locked_no_global',**base.engine.shared.metrics(counts.sum(0)))])
    base.engine.shared.write_csv(args.output/'per_subject_counts.csv',rows)
    comparisons=base.engine.comparisons({'RefResidual':primary,'G':counts})
    for row in comparisons:row['reference']='locked_no_global'
    base.engine.shared.write_csv(args.output/'paired_comparisons.csv',comparisons)
    base.runtime.write_json(args.output/'protocol.json',dict(
        formula='clip(L-L0,0,1)',selection='inherit preselected RefResidual a0,rho,tau; no tuning',
        semantics='remove G score only; reference candidate construction retained',source_result=args.result.name,
        primary_replay_subjects=29,completion='PASS'))
    print('NO_GLOBAL_SUPPLEMENT = PASS',base.engine.shared.metrics(counts.sum(0)),flush=True)


if __name__=='__main__':main()
