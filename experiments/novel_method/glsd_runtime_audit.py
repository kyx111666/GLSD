"""Time existing native functions and unchanged GLSD, with equivalence gate."""
import importlib.util
import sys
import time
import tracemalloc
import csv
import numpy as np
from my_method.glsd_evidence_audit import ROOT, OUT, manifest, decode
from my_method.gl_saliency_skill.benchmark import load_bundle, write_json, write_csv
from my_method.gl_saliency_skill.evidence import configuration_grid
from my_method.native_failure_mode_audit.run_native_failure_mode_audit import native_decode

def main():
    before=manifest()
    path=ROOT.parent/'historical_gl_exact_fresh_reproduction/source_archive/a/boostingvrme/tune_equiscale.py'
    spec=importlib.util.spec_from_file_location('runtime_historical_native',path)
    native=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=native
    spec.loader.exec_module(native)
    configs=configuration_grid()
    with (OUT/'selections.csv').open() as f:
        selected={(r['backbone'],r['dataset'],r['subject']):r for r in csv.DictReader(f) if r['method']=='GL'}
    rows=[]
    for dataset in ('sammlv','casme3'):
        for backbone in ('metst','boostingvrme'):
            b=load_bundle(backbone,dataset)
            def original(r):
                if backbone=='metst':
                    peaks=native_decode(r.score,b.legacy_temporal_scale)[3]
                    return [(max(0,int(p)-b.legacy_temporal_scale),int(p),min(len(r.score)-1,int(p)+b.legacy_temporal_scale)) for p in peaks]
                return [(on,p,off) for on,off,p in native.native_predictions(r.score,b.legacy_temporal_scale,native.NativeConfig(2.,.55,1.))]
            for r in b.records:
                ref=decode(r,configs[0],b.legacy_temporal_scale,b.interval_adapter)
                assert original(r)==[(e['onset'],e['peak'],e['offset']) for e in ref],(backbone,dataset,r.video)
            def execute(method):
                for r in b.records:
                    if method=='Native': original(r)
                    else:
                        s=selected[backbone,dataset,r.subject]
                        decode(r,configs[int(s['config_id'])],int(s['k']),b.interval_adapter)
            times={'Native':[],'GLSD':[]}
            for m in times: execute(m)
            for repeat in range(10):
                for m in (('Native','GLSD') if repeat%2==0 else ('GLSD','Native')):
                    start=time.perf_counter();execute(m);times[m].append(time.perf_counter()-start)
            peaks={}
            for m in times:
                tracemalloc.start();execute(m);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                peaks[m]=peak/2**20
            for m,v in {**times,'Extra':[g-n for g,n in zip(times['GLSD'],times['Native'])]}.items():
                rows.append({'backbone':backbone,'dataset':dataset,'method':m,'repeats':10,
                    'group_seconds_mean':float(np.mean(v)),'group_seconds_median':float(np.median(v)),
                    'group_seconds_std':float(np.std(v)),'videos':len(b.records),
                    'samples':sum(len(r.score) for r in b.records),
                    'seconds_per_video_mean':float(np.mean(v))/len(b.records),
                    'python_allocation_peak_MiB':peaks.get(m),
                    'memory_measurement_repeats':1,'native_predictions_exact':True})
            print('runtime completed',backbone,dataset,flush=True)
    assert before==manifest()
    write_csv(OUT/'runtime_corrected.csv',rows)
    write_json(OUT/'runtime_scope.json',{'timing':'10 repeats without tracemalloc, cache loading and selection excluded',
        'memory':'separate single tracemalloc run; not RSS',
        'metst_native':'existing native_decode also computes all_peaks for audit, so not a minimal optimized lower bound',
        'boosting_native':'historical tune_equiscale.native_predictions',
        'GLSD':'unchanged public API including internal candidate audit allocation',
        'skill_unchanged':before==manifest()})

if __name__=='__main__': main()
