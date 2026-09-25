#!/usr/bin/env python3
"""Retrospective geometry only; consumes saved predictions, never decodes."""
import csv
import json
import pickle
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'my_method/multi_scale_candidate_rescue'))
from run_mscr_nested_loso import interval_iou, match_events, moving_average, sha256

OUT = ROOT / 'results/pcbr1_constraint_ladder_diagnostic'
ORACLE = ROOT / 'results/tuned_native_boundary_oracle_diagnostic'
PCBR = ROOT / 'results/pcbr1_sammlv_nested'
CACHE = ROOT / 'caches/me_tst/sammlv_strategy1_outputs.pkl'
ALPHAS = (.3, .4, .5, .6, .7)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def exact_iou(left, right, gt_left, gt_right):
    return Fraction(max(0, min(right, gt_right)-max(left, gt_left)+1),
                    max(right, gt_right)-min(left, gt_left)+1)


def search(case, length, level):
    p, k, nl, nr, gl, gr = (int(case[x]) for x in ('pred_peak', 'k_native', 'pred_onset',
        'pred_offset', 'nearest_gt_onset', 'nearest_gt_offset'))
    lo, hi = max(0, p-2*k), min(length-1, p+2*k)
    feasible, maximum, count = [], Fraction(0), 0
    peaks = range(max(0, p-k), min(length-1, p+k)+1) if level == 'O3' else (p,)
    for q in peaks:
        # O3 uses the intersection of original ±2k and relocated-peak ±2k.
        # The native k remains fixed; no symmetric-width requirement is added.
        low, high = max(lo, q-2*k), min(hi, q+2*k)
        lefts = range(max(low, nl-k), nl+1) if level == 'O1' else range(low, q+1)
        rights = range(nr, min(high, nr+k)+1) if level == 'O1' else range(q, high+1)
        for left in lefts:
            for right in rights:
                count += 1
                iou = exact_iou(left, right, gl, gr)
                maximum = max(maximum, iou)
                if iou >= Fraction(1, 2):
                    # Minimal peak displacement first for O3; otherwise minimal
                    # total boundary displacement, then IoU and integer order.
                    rank = (abs(q-p), abs(left-nl)+abs(right-nr), -iou, left, right, q)
                    feasible.append((rank, q, left, right, iou))
    row = {'candidate_id': case['candidate_id'], 'subject': case['subject'], 'video': case['video'],
        'level': level, 'recoverable': bool(feasible), 'native_onset': nl, 'native_offset': nr,
        'native_peak': p, 'k': k, 'range_left': lo, 'range_right': hi,
        'target_gt_id': case['nearest_gt_id'], 'max_achievable_iou': float(maximum),
        'searched_interval_peak_combinations': count, 'feasible_combination_count': len(feasible),
        'status': 'RECOVERABLE' if feasible else f'{level}_GEOMETRICALLY_INFEASIBLE',
        'oracle_onset': None, 'oracle_offset': None, 'oracle_peak': None, 'oracle_iou': None,
        'left_delta': None, 'right_delta': None, 'requires_shrink': None,
        'requires_shift_like_asymmetric_change': None, 'left_expansion': None, 'right_expansion': None,
        'minimum_peak_shift': None, 'signed_peak_shift': None, 'peak_shift_direction': None}
    if feasible:
        _, q, left, right, iou = min(feasible, key=lambda item: item[0])
        row.update(oracle_onset=left, oracle_offset=right, oracle_peak=q, oracle_iou=float(iou),
            left_delta=left-nl, right_delta=right-nr, requires_shrink=left>nl or right<nr,
            requires_shift_like_asymmetric_change=(left-nl)*(right-nr)>0,
            left_expansion=max(0, nl-left), right_expansion=max(0, right-nr),
            minimum_peak_shift=abs(q-p), signed_peak_shift=q-p,
            peak_shift_direction='LEFT' if q<p else 'RIGHT' if q>p else 'NONE')
        if level == 'O1':
            row['minimum_left_expansion_over_all_feasible'] = min(nl-x[2] for x in feasible)
            row['minimum_right_expansion_over_all_feasible'] = min(x[3]-nr for x in feasible)
            row['minimum_total_expansion'] = nl-left+right-nr
    return row


def stats(values):
    return {'count': len(values), **{name: float(np.percentile(values, percentile)) if values else None
        for name, percentile in (('p25', 25), ('median', 50), ('p75', 75))}}


def main():
    if OUT.exists():
        raise RuntimeError(f'Refusing to overwrite {OUT}')
    # Hash every file in the three existing result directories, not only the
    # files used below, to check the read-only guarantee after analysis.
    inputs = [p for directory in (ORACLE, PCBR, ROOT/'results/rgr1_sammlv_nested')
              for p in directory.rglob('*') if p.is_file()] + [CACHE]
    hashes = {str(p): sha256(p) for p in inputs}
    assert hashes[str(CACHE)] == '3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569'
    report = json.loads((PCBR/'report.json').read_text())
    assert report['completed_fold_count'] == 29 and not report['incomplete']
    assert report['cache_sha256'] == hashes[str(CACHE)]
    with CACHE.open('rb') as handle:
        payload = pickle.load(handle)
    records = {(str(r['subject']), str(r['video'])): r for r in payload['records']}
    native = read_csv(PCBR/'native_predictions.csv')
    nested = read_csv(PCBR/'nested_predictions.csv')
    fixed = read_csv(PCBR/'fixed_alpha_predictions.csv')
    assert len(native) == len(nested) == 192 and len(fixed) == 960
    native_lookup = {r['candidate_id']: r for r in native}
    nested_lookup = {r['candidate_id']: r for r in nested}
    fixed_lookup = {(r['candidate_id'], float(r['alpha'])): r for r in fixed}
    assert set(native_lookup) == set(nested_lookup) and len(native_lookup) == 192
    # Anchor checked using existing saved intervals and official matching only.
    for rows in (native, nested):
        totals = Counter(TP=0, FP=0, FN=0)
        for key, record in records.items():
            predictions = [r for r in rows if (r['subject'], r['video']) == key]
            matches, missed = match_events(predictions, record['samples'])
            assert matches == [int(r['matched_gt']) for r in predictions]
            totals.update(TP=sum(m>=0 for m in matches), FP=sum(m<0 for m in matches), FN=len(missed))
        assert dict(totals) == dict(TP=49, FP=143, FN=110)
    cases = read_csv(ORACLE/'oracle_recoverable_cases.csv')
    assert len(cases) == 11 and len({c['candidate_id'] for c in cases}) == 11
    assert len({c['nearest_gt_id'] for c in cases}) == 11
    trajectories = {}
    for row in read_csv(ORACLE/'trajectory_data.csv'):
        trajectories.setdefault(row['candidate_id'], {})[int(row['absolute_index'])] = float(row['smoothed_spotting_score'])
    oracle_rows = {level: [] for level in ('O1','O2','O3')}
    taxonomy, support = [], []
    for case in cases:
        identity = case['candidate_id']
        record = records[(case['subject'],case['video'])]
        n = native_lookup[identity]
        assert int(n['matched_gt']) == int(nested_lookup[identity]['matched_gt']) == -1
        assert [int(n[key]) for key in ('onset','peak','offset')] == [int(case[key]) for key in ('pred_onset','pred_peak','pred_offset')]
        gt = record['samples'][int(case['nearest_gt_id'].rsplit('_',1)[1])]
        assert int(gt[0]) == int(case['nearest_gt_onset']) and int(gt[2]) == int(case['nearest_gt_offset'])
        assert interval_iou({'onset':case['oracle_onset'],'offset':case['oracle_offset']},gt) >= .5
        assert int(case['oracle_onset']) <= int(case['pred_peak']) <= int(case['oracle_offset'])
        rows = {level: search(case,len(record['score']),level) for level in oracle_rows}
        assert not rows['O1']['recoverable'] or rows['O2']['recoverable']
        assert not rows['O2']['recoverable'] or rows['O3']['recoverable']
        for level, row in rows.items(): oracle_rows[level].append(row)
        kind = ('TYPE-A: OUTWARD-SUFFICIENT' if rows['O1']['recoverable'] else
                'TYPE-B: ASYMMETRIC-BOUNDARY' if rows['O2']['recoverable'] else
                'TYPE-C: PEAK-ANCHOR' if rows['O3']['recoverable'] else 'TYPE-D: LOCAL-UNRECOVERABLE')
        taxonomy.append({'candidate_id':identity,'subject':case['subject'],'taxonomy':kind,
            'O0_recoverable':True, **{f'{level}_recoverable':row['recoverable'] for level,row in rows.items()},
            'O2_minus_O1':rows['O2']['recoverable'] and not rows['O1']['recoverable'],
            'O3_minus_O2':rows['O3']['recoverable'] and not rows['O2']['recoverable'],
            'O3_minimum_peak_shift':rows['O3']['minimum_peak_shift'],
            'O3_peak_shift_direction':rows['O3']['peak_shift_direction']})
        if not rows['O1']['recoverable']: continue
        o1 = rows['O1']; p,k = int(case['pred_peak']),int(case['k_native'])
        config = json.loads(case['tuned_native_config'])
        curve = moving_average(record['score'],max(1,int(round(config['c_s']*5))))
        lo,hi = o1['range_left'],o1['range_right']
        assert all(curve[i] == trajectories[identity][i] for i in range(lo,hi+1))
        b,sp = float(min(curve[lo:hi+1])),float(curve[p])
        assert sp > b
        q = (curve-b)/(sp-b)
        left,right = o1['oracle_onset'],o1['oracle_offset']
        need_left,need_right = left<int(n['onset']),right>int(n['offset'])
        left_min,right_min = float(min(q[left:p+1])),float(min(q[p:right+1]))
        bridge = min([v for v,need in ((left_min,need_left),(right_min,need_right)) if need])
        # A rebound flag describes an interior minimum below the target endpoint;
        # a low monotone tail alone is not called a valley.
        rebound = ((need_left and len(q[left:p+1])>2 and min(q[left+1:p])<q[left]-1e-12)
                   or (need_right and len(q[p:right+1])>2 and min(q[p+1:right])<q[right]-1e-12))
        for alpha in ALPHAS:
            actual = fixed_lookup[(identity,alpha)]
            al,ar = int(actual['onset']),int(actual['offset'])
            reach_left,reach_right = al<=left,ar>=right
            theory_left,theory_right = (not need_left or left_min>=alpha),(not need_right or right_min>=alpha)
            assert (reach_left,reach_right) == (theory_left,theory_right)
            support.append({'candidate_id':identity,'subject':case['subject'],'alpha':alpha,
                'actual_pcbr_onset':al,'actual_pcbr_offset':ar,
                'required_O1_onset':left,'required_O1_offset':right,
                'reached_required_left':reach_left,'reached_required_right':reach_right,
                'both_required_boundaries_reached':reach_left and reach_right,
                'bridge_min_q':bridge,'left_path_min_q':left_min,'right_path_min_q':right_min,
                'left_expansion_required':need_left,'right_expansion_required':need_right,
                'local_background':b,'peak_score':sp,'normalized_support_defined':True,
                'theoretical_alpha_can_reach_required_boundaries':theory_left and theory_right,
                'interior_valley_rebound_on_required_path':bool(rebound),
                'actual_iou_to_oracle_gt':interval_iou(actual,gt),
                'actual_geometric_recovery':interval_iou(actual,gt)>=.5,
                'actual_formal_recovery':int(actual['matched_gt'])==int(case['nearest_gt_id'].rsplit('_',1)[1]),
                'selected_in_case_outer_fold':alpha==report['outer_folds'][next(i for i,f in enumerate(report['outer_folds']) if f['outer_subject']==case['subject'])]['selected_alpha']})
    counts = {level:sum(r['recoverable'] for r in rows) for level,rows in oracle_rows.items()}
    ladder = [dict(Oracle='O0 Original',peak_movable='NO',boundary_inward='YES',boundary_outward='YES',range='free valid',recoverable=11)]
    for level,name,peak,inward in (('O1','PCBR-1 constraint','NO','NO'),('O2','Asymmetric local','NO','YES'),('O3','Peak relocation','YES ±k','YES')):
        ladder.append(dict(Oracle=f'{level} {name}',peak_movable=peak,boundary_inward=inward,boundary_outward='YES',range='original ±2k',recoverable=counts[level]))
    types = Counter(r['taxonomy'] for r in taxonomy)
    new_o3 = [r['minimum_peak_shift'] for r in oracle_rows['O3'] if next(t for t in taxonomy if t['candidate_id']==r['candidate_id'])['O3_minus_O2']]
    assert all(sha256(Path(p))==digest for p,digest in hashes.items())
    result = {'diagnostic':'Constraint-Ladder Oracle Diagnostic','source_file_sha256':hashes,
        'script_sha256':sha256(Path(__file__)), 'all_existing_inputs_unchanged':True,
        'anchor':dict(TP=49,FP=143,FN=110,F1=98/351),'PCBR_nested_recoveries':0,
        'scope':'Exactly original 11 candidate-target GT pairs; not whole-dataset performance',
        'O0_reused_not_recomputed':True,'constraint_ladder':ladder,'taxonomy_counts':dict(types),
        'O2_minus_O1':[r['candidate_id'] for r in taxonomy if r['O2_minus_O1']],
        'O3_minus_O2':[r['candidate_id'] for r in taxonomy if r['O3_minus_O2']],
        'O3_peak_shift_all_recoverable':stats([r['minimum_peak_shift'] for r in oracle_rows['O3'] if r['recoverable']]),
        'O3_peak_shift_new_recoveries_only':stats(new_o3),
        'fixed_alpha_O1_formal_recoveries':{str(a):sum(r['actual_formal_recovery'] for r in support if r['alpha']==a) for a in ALPHAS},
        'interpretation':'LOCAL-SCORE-SUPPORT-FAILURE',
        'interpretation_basis':'O1 has 5 feasible cases, none recovered by nested-selected alpha=.70. Geometry also excludes 6 cases: 3 require inward freedom, 3 require peak movement within the local range. Fixed controls do recover some TYPE-A cases.',
        'definitions':{'O3':'q in original p±k; l/r in intersection of original p±2k, relocated q±2k, and valid sequence; l<=q<=r; zero shift allowed',
            'oracle_tie_break':'minimal |q-p|, minimal total boundary displacement, higher exact IoU, lower onset, offset, peak',
            'shift_like':'both boundary deltas have the same nonzero sign',
            'bridge_min_q':'minimum q on peak-to-required-boundary paths ONLY for sides requiring expansion; candidate Native envelope already covers the other side',
            'reach_caveat':'reaching a selected minimum-change oracle envelope is not sufficient for TP if the actual opposite boundary overexpands; actual IoU and formal match are separately recorded',
            'locality_caveat':'TYPE-C diagnoses peak containment under fixed local range, not globally: O0 already recovered all with fixed peaks'},
        'no_new_decoder':True,'no_nested_LOSO':True,'no_training_or_forward':True,'no_alpha_tuning':True}
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_csv('constraint_ladder_summary.csv',ladder)
    write_csv('case_level_taxonomy.csv',taxonomy)
    for level,name in (('O1','o1_constrained_oracle.csv'),('O2','o2_asymmetric_oracle.csv'),('O3','o3_peak_relocation_oracle.csv')):
        write_csv(name,oracle_rows[level])
    write_csv('pcbr1_support_failure.csv',support)
    lines=['# Constraint-Ladder Oracle Diagnostic','', '**LOCAL-SCORE-SUPPORT-FAILURE**','',
        '严格限定原 11 个 candidate—GT 对。全部输入只读，文件哈希前后相同。复核 saved Native / PCBR 区间的正式 matching，均为 49/143/110；候选数量均为 192。没有执行 decoder、Nested LOSO、训练、backbone forward 或 alpha 调参。','',
        '## Constraint ladder','', '| Oracle | Peak movable | Boundary inward | Boundary outward | Range | Recoverable |',
        '|---|---|---|---|---|---:|']
    lines += [f"| {r['Oracle']} | {r['peak_movable']} | {r['boundary_inward']} | {r['boundary_outward']} | {r['range']} | {r['recoverable']} |" for r in ladder]
    lines += ['', 'O0 直接复用已有结果。O1/O2/O3 采用闭区间整数穷举、精确分数 IoU≥0.5；k 沿用各 candidate Native k。O3 新 peak q∈原 p±k，边界范围为原 p±2k 与 q±2k 及有效视频区间的交集；允许零位移。', '',
        '## Case-level taxonomy','', '| Candidate | Type | O1 | O2 | O3 | 最小 peak shift | Direction |','|---|---|---|---|---|---:|---|']
    lines += [f"| {r['candidate_id']} | {r['taxonomy']} | {r['O1_recoverable']} | {r['O2_recoverable']} | {r['O3_recoverable']} | {r['O3_minimum_peak_shift']} | {r['O3_peak_shift_direction']} |" for r in taxonomy]
    lines += ['',f'Taxonomy: {dict(types)}。TYPE-D 为 0。', '',
        'O1 witness 按总 expansion 最小、再 IoU 最大选择；CSV 同时提供各侧独立最小 expansion（两侧独立最小值不一定构成同一个可行区间）。O2 提供有符号 boundary delta、是否 shrink，以及两侧 delta 同号的 shift-like 标记。O3 首先最小化 peak 位移，然后最小化边界改动。','',
        f"O3 全部 11 个 recovery 的 peak shift 统计：{result['O3_peak_shift_all_recoverable']}。",
        f"仅 O3−O2 新增 3 个 recovery：{result['O3_peak_shift_new_recoveries_only']}。数值单位为 cache index，不是原视频 frame。",'',
        '## TYPE-A support analysis','',
        '| Candidate | O1 onset/offset | left/right expansion | bridge_min_q | 路径内部 valley 后回升 | 能到达 witness 的现有 alpha | 正式恢复的现有 alpha |',
        '|---|---|---|---:|---|---|---|']
    for row in oracle_rows['O1']:
        if not row['recoverable']: continue
        ss=[r for r in support if r['candidate_id']==row['candidate_id']]
        lines.append(f"| {row['candidate_id']} | {row['oracle_onset']} / {row['oracle_offset']} | {row['left_expansion']} / {row['right_expansion']} | {ss[0]['bridge_min_q']:.6f} | {ss[0]['interior_valley_rebound_on_required_path']} | {[r['alpha'] for r in ss if r['both_required_boundaries_reached']]} | {[r['alpha'] for r in ss if r['actual_formal_recovery']]} |")
    lines += ['', '所有 actual PCBR 区间均读取既有 fixed_alpha_predictions.csv。s(t) 使用原 moving_average 与该折原参数重建，并与已有 oracle trajectory 逐点精确一致；未运行 peak 搜索或 decoder。每个 TYPE-A × 5 alpha 的边界、到达标记、q、实际 IoU、formal match 见 pcbr1_support_failure.csv。','',
        'bridge_min_q 只检查真正需要扩展的一侧路径。低 q 可能是衰减尾部，只有内部下降后回升才标记 valley；不能把所有阈值截断都称作 valley。到达 witness 也不保证实际区间的 IoU≥0.5，因此另外核查实际 IoU。','',
        f"现有 fixed-alpha 的 TYPE-A formal recoveries：{result['fixed_alpha_O1_formal_recoveries']}。nested 所选 0.70 全部失败，但不能说所有 fixed-alpha 都失败。",'',
        '## Interpretation','',
        '选择 LOCAL-SCORE-SUPPORT-FAILURE：5 个 case 在 PCBR-1 几何约束下已经可达，但实际 nested-selected support rule 未恢复其中任何一个。其余 6 个确有几何限制：3 个必须放开 inward 边界，另外 3 个在固定局部范围内需要移动 peak。这个标签概括主结果，不表示 11 个 case 都由同一原因造成。','',
        'O2=8，O3 新增 3，不能将 fixed peak 概括为全部案例的主要瓶颈；这些 TYPE-C 的结论还依赖 ±2k 局部范围，因为 O0 在自由范围内已经以 fixed peak 恢复了它们。较低 fixed-alpha 已恢复部分案例，但这不构成重新选择 alpha 的依据。所有 oracle 数量都是原 11 个目标对的事后几何结果，不是新模型性能。','',
        '## Files','']
    lines += [f'- `{(OUT/name).resolve()}`' for name in ('report.json','constraint_ladder_summary.csv','case_level_taxonomy.csv','o1_constrained_oracle.csv','o2_asymmetric_oracle.csv','o3_peak_relocation_oracle.csv','pcbr1_support_failure.csv','CONSTRAINT_LADDER_REPORT_CN.md')]
    lines += [f'- `{Path(__file__).resolve()}`', '', '诊断完成并停止。']
    (OUT/'CONSTRAINT_LADDER_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'counts':counts,'taxonomy':dict(types),'peak_shift':result['O3_peak_shift_new_recoveries_only'],
        'support':[r for r in support if r['alpha']==.7],'interpretation':result['interpretation']},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
