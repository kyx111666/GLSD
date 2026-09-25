"""Local implementation validation; its two-subject mini-screen is NOT a study result."""
import argparse
import contextlib
import gzip
import json
from pathlib import Path
import numpy as np
import run_shift as run
import shift_features as sf
from verify_results import verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump',type=Path,required=True)
    p.add_argument('--prototype',type=Path,required=True)
    p.add_argument('--historical-trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    context,core,summary=run.context_from_dump(args.dump,'sammlv')
    original=run.runtime.load_module(args.prototype,'shift_prototype_validation')
    saved={}
    with gzip.open(args.historical_trace,'rt') as h:
        for line in h:
            row=json.loads(line)
            if row['method']=='RefResidual':
                for v in row['candidates']:saved[v['video_id']]=v
    max_error=0.; videos=0; candidates=0; widths=[]
    for si,s in enumerate(context[6]):
        r=context[4][si]
        for vi,x in enumerate(r['result_all']):
            f=core.GLSDFeatures(x,r['k_p'])
            d=f.shift_evidence()
            old=saved[f'{s}/video_{vi}']
            if old['response_sha256']!=f.key[2] or old['k']!=r['k_p']:
                raise RuntimeError('Historical input response identity mismatch')
            peak,gl,l0,b,delta,counts=original.compute(f.base,2.,3.)
            np.testing.assert_array_equal(d['peaks'],peak)
            for key,expected in [('G',gl[:,0]),('L',gl[:,1]),('L0',l0),('B',b),('D',delta)]:
                np.testing.assert_allclose(d[key],expected,rtol=0,atol=1e-12)
                if len(expected):max_error=max(max_error,float(np.max(np.abs(np.asarray(d[key])-expected))))
            np.testing.assert_array_equal(d['control_count'],counts)
            videos+=1;candidates+=len(peak);widths.append(d['physical_widths'])
    if videos!=79:raise RuntimeError('Real feature validation incomplete')
    run.write_json(args.output/'production_feature_equivalence.json',dict(status='PASS',
        scope='New production path on fixed a0=2/rho=3; no performance inference',videos=videos,candidates=candidates,
        max_absolute_error=max_error,prototype_sha256=run.sha(args.prototype),historical_trace_sha256=run.sha(args.historical_trace),
        checks=['All 79 response hashes and k match historical trace','Production matches scalar prototype at fixed structure',
                'G/L/L0 replay','Exact eligible-control counts','Candidate order and physical scale votes']))
    # Real inputs, deliberately only two subjects: exercise complete output plumbing.
    # No formal completion marker, no claim that these thresholds estimate LOSO performance.
    small=list(context);small[4]=context[4][:2];small[5]=context[5][:2];small[6]=context[6][:2]
    small=tuple(small)
    mini_core=sf.FusionCore(context[1]);mini_core.expected_gt=core.expected_gt[:2]
    out=args.output/'integration_only_two_subjects';out.mkdir()
    run.write_json(out/'NOT_PERFORMANCE.json',dict(scope='Two-subject software integration fixture, not valid study performance; do not rank methods or choose formula'))
    mini_summary=dict(summary,subjects=2,videos=sum(len(r['videos']) for r in small[4]),
                      ground_truth=sum(mini_core.expected_gt),per_subject_gt=mini_core.expected_gt)
    run.write_json(out/'input_summary.json',mini_summary)
    with contextlib.ExitStack() as stack:
        classes=[small[2],small[3].spotting.__globals__['MeanAveragePrecision2d']]
        for cls in classes:stack.enter_context(run.matching.install(cls))
        run.write_json(out/'matching_software_tests.json',run.matching.synthetic_probe(classes))
        # Interrupt/resume boundary is between subjects, using the exact production checkpoint.
        run.search(small,mini_core,out,indices=[0])
        raw,full,done=run.search(small,mini_core,out)
        if not done.all():raise RuntimeError('Real resume did not finish')
        run.select_and_replay(small,mini_core,out,raw,full)
    result=verify(out)
    # Corrupt a saved selection, confirm independent audit rejects it, then restore.
    path=out/'selected_configs.csv';content=path.read_text()
    import csv,io
    rows=list(csv.DictReader(io.StringIO(content)));rows[0]['threshold']='9.99'
    run.engine.shared.write_csv(path,rows)
    try:
        try:verify(out)
        except RuntimeError:pass
        else:raise RuntimeError('Verifier accepted deliberately corrupted selection')
    finally:path.write_text(content)
    verify(out)
    run.write_json(args.output/'real_validation.json',dict(status='PASS',scope='Software and real integration only, not performance',
        prototype_videos=videos, mini_screen_subjects=2, selected_records=result['selected_records'],
        checks=result['checks']+['Real checkpoint resume','Verifier rejects corrupted selection']))
    print('REAL_VALIDATION_PASS: no performance conclusion')


if __name__=='__main__':main()
