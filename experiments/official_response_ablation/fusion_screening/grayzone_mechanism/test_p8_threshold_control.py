"""Critical-path regression checks; synthetic tests are not experiment evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import run_p8_threshold_control as run
from test_p8_phase2 import BASE


class Tests(unittest.TestCase):
    def test_detailed_decode_with_real_one_to_one_metric(self):
        from test_p8_matching import metric, counts, MeanAveragePrecision2d
        class Runner:
            def decode_glsd_subject(self, records, si, c, core, metric_class, official, recognition):
                selected=core.GLSDFeatures(records[si]['result_all'][0],2).selected_peaks(c)
                intervals=[(0,9) for _ in selected]
                m=metric(intervals,[(0,9)])
                raw,matches=counts(m)
                targets=[1 if ids[0]>=0 else -1 for ids in matches.get(0,[])]
                return raw,[m.pred_all[0]], [1]*len(intervals),targets,matches,m
            def full_counts_from_official_synergy(self,raw,labels,targets):
                return raw
        records=[dict(result_all=[[1,2,3,4]],k_p=2,videos=['v'],samples=[[[0,4,9]]])]
        context=(Runner(),BASE,MeanAveragePrecision2d,None,records,[],['037'],None)
        core=run.FusionCore(BASE); core.expected_gt=[1]
        with run.evaluator.install(MeanAveragePrecision2d):
            result=run.detailed_decode(context,core,0,run.phase.Config('GM_p8',0,1,1,.1))
        self.assertEqual(result['full_counts'],[1,1,0])
        self.assertEqual(result['events'][0]['matched_gt_id'],'037/video_0/gt_0')
        self.assertIsNone(result['events'][1]['matched_gt_id'])
        self.assertEqual(result['candidates'][0]['video_id'],'037/video_0')

    def test_bufferless_evaluator_capture_and_restore(self):
        import test_p8_matching  # Runtime compatibility and local source path.
        from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
        original = MeanAveragePrecision2d.add
        runner = SimpleNamespace(full_counts_from_official_synergy=lambda raw, labels, targets: raw)
        record = dict(videos=['v'], samples=[[[0,4,9]]])
        preds = np.array([[0,0,9,0,0,1,4]], float)
        gt = np.array([[0,0,9,0,0,0,0,4]], float)
        # This older bufferless class uses string match codes; test its add API
        # unchanged. Corrected numeric matching is exercised separately above.
        with run.contextlib.nullcontext():
            with run.capture_registered_inputs(MeanAveragePrecision2d) as captured:
                metric = MeanAveragePrecision2d(num_classes=1)
                metric.add(preds,gt)
                self.assertFalse(hasattr(metric,'gt_all'))
                self.assertFalse(hasattr(metric,'pred_all'))
                matches = metric.value(iou_thresholds=0.5)[0.5][0]['pred_match_gt']
            self.assertIs(MeanAveragePrecision2d.add,original)
            rows, targets = run.event_rows(runner,record,'s',[preds],matches,[1],[1],
                                          captured[id(metric)][1],(1,0,0),(1,0,0))
        self.assertEqual(rows[0]['matched_gt_id'],'s/video_0/gt_0')
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with run.capture_registered_inputs(MeanAveragePrecision2d):
                raise RuntimeError('deliberate')
        self.assertIs(MeanAveragePrecision2d.add,original)

    def test_distinct_aggregate_and_video_evaluators(self):
        from test_p8_matching import MeanAveragePrecision2d as Aggregate
        from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d as Video
        # Mirror the sealed call chain: spotting resolves its own global class.
        scope = {'MeanAveragePrecision2d': Video, 'np': np}
        exec("def spotting(metric_final, preds, gt):\n"
             "    video = MeanAveragePrecision2d(num_classes=1)\n"
             "    video.add(preds, gt)\n"
             "    metric_final.add(np.column_stack((preds, np.zeros(len(preds)))), gt)\n"
             "    return video\n", scope)
        official = SimpleNamespace(spotting=scope['spotting'])
        class Runner:
            def decode_glsd_subject(self, records, si, c, core, metric_class, official, recognition):
                peaks = core.GLSDFeatures(records[si]['result_all'][0],2).selected_peaks(c)
                preds = np.array([[0,0,9,0,0,1,4] for _ in peaks],float).reshape(-1,7)
                gt = np.array([[0,0,9,0,0,0,0,4]],float)
                aggregate = metric_class(num_classes=1)
                video = official.spotting(aggregate,preds,gt)
                values = aggregate.value(iou_thresholds=0.5)[0.5][0]
                raw = (int(sum(values['tp'])),int(sum(values['fp'])),1-int(sum(values['tp'])))
                matches = video.value(iou_thresholds=0.5)[0.5][0]['pred_match_gt']
                targets = [1 if np.asarray(x).reshape(-1)[0]>=0 else -1 for x in matches.get(0,[])]
                return raw,[preds],[1]*len(preds),targets,matches,video
            def full_counts_from_official_synergy(self,raw,labels,targets): return raw
        record = dict(result_all=[[1,2,3,4]],k_p=2,videos=['v'],samples=[[[0,4,9]]])
        context = (Runner(),BASE,Aggregate,official,[record],[],['s'],None)
        core = run.FusionCore(BASE); core.expected_gt=[1]
        original_a,original_v = Aggregate.add,Video.add
        with run.evaluator.install(Aggregate):
            result = run.detailed_decode(context,core,0,run.phase.Config('G',0,1,1,.5))
        self.assertEqual(result['full_counts'],[1,0,0])
        self.assertEqual(result['events'][0]['matched_gt_id'],'s/video_0/gt_0')
        self.assertIs(Aggregate.add,original_a)
        self.assertIs(Video.add,original_v)

    def test_threshold_mapping_including_boundaries(self):
        core=run.FusionCore(BASE)
        f=core.GLSDFeatures([1,2,3,4],2)
        values=np.column_stack(([.0,.5,run.FACTOR*.7,1.], [0.,0.,.8,1.]))
        f.cache[(1.,1.)]=(np.arange(4),values)
        for tau in (*run.phase.TAUS, .5/run.FACTOR):
            c=run.phase.Config('G_rescaled',0,1,1,tau)
            self.assertTrue(np.array_equal(f.selected_peaks(c),f.selected_peaks(run.replace(c,method='G_expanded',threshold=run.FACTOR*tau))))
        self.assertEqual(len(f.selected_peaks(run.replace(c,threshold=.95))),0)
        g=run.grids()
        self.assertEqual({k:len(v) for k,v in g.items()},dict(G=57,L=171,Mean=171,GM_p8=57,G_rescaled=57,G_expanded=114))
        self.assertTrue(all(c.local_radius==1 for c in g['GM_p8']))
        self.assertGreater(max(c.threshold for c in g['G_expanded']),1)

    def test_event_ids_and_neutral_full_filter(self):
        def synergy(raw,labels,targets):
            tp,fp,fn=raw
            for l,t in zip(labels,targets):
                if l==4:
                    if t==-1: fp-=1
                    else: tp-=1; fn+=1
            return tp,fp,fn
        runner=SimpleNamespace(full_counts_from_official_synergy=synergy)
        record=dict(videos=['v'],samples=[[[0,4,9],[20,24,29]]])
        metric=SimpleNamespace(gt_all=[np.array([[0,0,9,0,0,0,0,1],[20,0,29,0,0,0,0,1]])],
                               pred_all=[np.array([[0,9],[20,29],[40,49]])])
        rows,gt=run.event_rows(runner,record,'037',[[[0,9],[20,29],[40,49]]],{0:[[0],[1],[-1]]},
                              [1,4,4],[1,1,-1],list(zip(metric.pred_all,metric.gt_all)),(2,1,0),(1,0,1))
        self.assertEqual(rows[0]['matched_gt_id'],'037/video_0/gt_0')
        self.assertFalse(rows[1]['full_retained'])
        self.assertEqual(len(gt),2)
        metric.pred_all=[np.array([[0,9]])]
        with self.assertRaisesRegex(RuntimeError,'recognition target'):
            run.event_rows(runner,record,'037',[[[0,9]]],{0:[[0]]},[1],[-1],list(zip(metric.pred_all,metric.gt_all)),(1,0,1),(1,0,1))

    def test_event_delta_identifies_swapped_gt_with_same_tp(self):
        def record(gt):
            return dict(subject='s',method='m',full_counts=[1,0,1],
                        events=[dict(matched_gt_id=gt,full_TP=1)],candidates=[])
        d=run.event_delta(record('s/v/g1'),record('s/v/g0'))
        self.assertEqual(d['delta_TP'],0)
        self.assertEqual(d['added_gt_ids'],['s/v/g1'])
        self.assertEqual(d['lost_gt_ids'],['s/v/g0'])

    def test_reuse_subgrid_preserves_order_and_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'synthetic').mkdir()
            for m,cs in run.phase.grids().items():
                a=np.zeros((len(cs),2,3),dtype=np.int64); a[:,:,2]=2
                np.savez(root/'synthetic'/('search_counts_%s.npz'%m),raw=a,full=a,done=[True,True],
                         subjects=['a','b'],configs=json.dumps([run.asdict(c) for c in cs]))
            tables=run.load_old_counts(root,'synthetic',['a','b'],[2,2])
            self.assertEqual(tables['GM_p8'][0].shape,(57,2,3))
            with self.assertRaisesRegex(RuntimeError,'subject'):
                run.load_old_counts(root,'synthetic',['b','a'],[2,2])

    def test_full_pipeline_reuses_grids_and_locks_ablations(self):
        from test_p8_phase2 import BASE
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); old=root/'old'; out=root/'out'; old.mkdir(); out.mkdir()
            (old/'synthetic_audit').mkdir()
            (old/'synthetic_audit'/'native_per_subject_counts.csv').write_text('subject,evaluation,TP,FP,FN\na,full,1,0,1\nb,full,1,0,1\n')
            grids={m:[run.phase.Config(m,0,1,1,.5)] for m in run.grids()}
            grids['G_expanded'].append(run.phase.Config('G_expanded',1,1,1,run.FACTOR*.5))
            old_tables={m:(np.array([[[1,0,1],[1,0,1]]]),)*2 for m in ('G','L','Mean','GM_p8')}
            context=(SimpleNamespace(final_samples=lambda records:[[[(0,9),(20,29)]]]*2),BASE,None,None,[],[],['a','b'],None)
            seen=[]
            def detailed(ctx,core,si,c):
                seen.append(c)
                return dict(subject=ctx[6][si],method=c.method,config=run.asdict(c),raw_counts=[1,0,1],full_counts=[1,0,1],
                            events=[dict(matched_gt_id=ctx[6][si]+'/v/g0',full_TP=1)],ground_truth=[],candidates=[])
            with patch.object(run,'grids',return_value=grids),patch.object(run,'load_old_counts',return_value=old_tables), \
                 patch.object(run,'check_mapping',return_value=1),patch.object(run,'detailed_decode',side_effect=detailed), \
                 patch.object(run.phase,'search_counts',return_value=old_tables['G']) as search:
                run.run_setting(context,old,'synthetic',out,'full')
            self.assertEqual(search.call_count,1)
            self.assertEqual(search.call_args.args[2][0].method,'G_rescaled')
            self.assertTrue(all((c.reference_scale,c.local_radius,c.threshold)==(1,1,.5) for c in seen if c.method.startswith('p8_no')))
            self.assertTrue((out/'paired_comparisons.csv').exists())
            self.assertEqual(json.loads((out/'decision.json').read_text())['delta_vs_expanded_G'],0)

if __name__=='__main__': unittest.main()
