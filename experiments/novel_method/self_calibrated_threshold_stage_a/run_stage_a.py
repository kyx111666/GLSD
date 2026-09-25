#!/usr/bin/env python3
"""Frozen Native settings + local-peak quantile height threshold; Stage A only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/self_calibrated_threshold_stage_a"
NATIVE_SOURCE = ROOT / "my_method/expanded_native_fairness_audit/run_expanded_native_fairness_audit.py"
spec = importlib.util.spec_from_file_location("native_only", NATIVE_SOURCE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

SAMM_SELECTED = ROOT / "results/rgr1_sammlv_nested/report.json"
CAS_SELECTED = ROOT / "results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv"
CAS_METRICS = CAS_SELECTED.with_name("final_strong_native_outer_metrics.csv")
Q_VALUES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
COUNT_KEYS = ("TP", "FP", "FN", "event_count")
METHODS = ("Author Native", "Final Strong Native", "Quantile Threshold Decoder")
AUTHOR_CONFIG = dict(c_s=2.0, p=0.55, c_d=1.0, c_b=1.0)
SEED = 20260905
BOOTSTRAP_REPS = 1000
TIE_TOLERANCE = 0.001
REGIME_GAP = 0.15
DATASETS = {
    "SAMMLV": dict(cache=native.SAMMLV_CACHE, sha=native.EXPECTED_SAMMLV_SHA,
                   subjects=29, videos=79, gt=159, k_p=5,
                   author=(53, 184, 106), strong=(49, 143, 110),
                   author_f1=.267677, strong_f1=.279202),
    "CASME3": dict(cache=native.CASME3_CACHE, sha=native.EXPECTED_CASME3_SHA,
                   subjects=94, videos=462, gt=858, k_p=17,
                   author=(81, 912, 777), strong=(124, 1148, 734),
                   author_f1=.087520, strong_f1=.116432),
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def add(total, counts):
    for key in COUNT_KEYS:
        total[key] += counts[key]


def counts_sum(rows):
    total = native.counts_empty()
    for row in rows:
        add(total, row)
    return total


def config_id(config):
    return tuple(config[k] for k in ("c_s", "p", "c_d", "c_b"))


def integer_config(config, kp):
    return dict(smoothing_width=max(1, round(config["c_s"] * kp)),
                peak_distance=max(1, round(config["c_d"] * kp)),
                boundary_radius=max(1, round(config["c_b"] * kp)))


def peak_geometry(score, config, kp):
    ints = integer_config(config, kp)
    curve = native.moving_average(score, ints["smoothing_width"])
    peaks = find_peaks(curve, distance=ints["peak_distance"])[0].astype(int)
    return curve, peaks, ints


def choose_peaks(curve, peaks, config, q=None):
    fallback = q is not None and len(peaks) < 2
    if q is None or fallback:
        tau = native.threshold(curve, config["p"])
    else:
        tau = float(np.quantile(curve[peaks], q, method="linear"))
    return peaks[curve[peaks] >= tau], tau, fallback


def events_for(kept, radius, source):
    return [native.event(peak, radius, source) for peak in kept]


def q_rank(counts, index):
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return (-Fraction(2 * counts["TP"], denominator) if denominator else Fraction(0),
            counts["FP"], index)


def selected_sources():
    samm = json.loads(SAMM_SELECTED.read_text())
    configs = {"SAMMLV": {}, "CASME3": {}}
    reference = {"SAMMLV": {}, "CASME3": {}}
    subjects = {row["subject"] for row in samm["outer_folds"]}
    for row in samm["outer_folds"]:
        s = row["subject"]
        assert set(row["train_subjects"]) == subjects - {s}
        c = row["selected_strong_config"]
        configs["SAMMLV"][s] = dict(c_s=c["c_s"], p=c["p_s"], c_d=c["c_d"], c_b=c["c_b"])
        reference["SAMMLV"][s] = {
            METHODS[0]: row["outer_metrics"]["Author Native"],
            METHODS[1]: row["outer_metrics"]["Tuned Native"],
        }
    with CAS_SELECTED.open(newline="") as handle:
        for row in csv.DictReader(handle):
            configs["CASME3"][row["outer_subject"]] = {k: float(row[k]) for k in ("c_s", "p", "c_d", "c_b")}
    with CAS_METRICS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["outer_subject"] != "aggregate":
                reference["CASME3"][row["outer_subject"]] = {METHODS[1]: {k: int(row[k]) for k in COUNT_KEYS}}
    return configs, reference


def gate():
    configs, reference = selected_sources()
    data, gate_rows = {}, {}
    for name, expected in DATASETS.items():
        assert native.sha256(expected["cache"]) == expected["sha"], f"{name} SHA mismatch"
        payload, records, subjects, observed = native.load_payload(expected["cache"])
        assert observed == {k: expected[k] for k in ("subjects", "videos", "gt", "k_p")}, (name, observed)
        assert (payload["num_subjects"], payload["num_videos"], payload["num_gt"]) == (expected["subjects"], expected["videos"], expected["gt"])
        assert set(configs[name]) == set(subjects)
        by_subject = {s: [] for s in subjects}
        anomalies = []
        for index, record in enumerate(records):
            scores = np.asarray(record["score"])
            assert scores.ndim == 1 and len(scores) and np.isfinite(scores).all()
            by_subject[str(record["subject"])].append(index)
            for gt_index, gt in enumerate(record["samples"]):
                if int(gt[2]) < int(gt[0]):
                    anomalies.append(dict(subject=str(record["subject"]), video=str(record["video"]), gt_index=gt_index,
                                          GT=list(map(int, gt)), handling="Retained unchanged in existing evaluator"))
        baseline = {method: {} for method in METHODS[:2]}
        for subject in subjects:
            for method in METHODS[:2]:
                total = native.counts_empty()
                for index in by_subject[subject]:
                    record = records[index]
                    events = native.author_decode(record, expected["k_p"]) if method == METHODS[0] else native.tuned_decode(record, configs[name][subject], expected["k_p"])
                    add(total, native.evaluate(record, events))
                baseline[method][subject] = total
                ref = reference[name].get(subject, {}).get(method)
                if ref is not None:
                    assert all(total[k] == ref[k] for k in COUNT_KEYS), (name, subject, method, total, ref)
        aggregates = {method: native.metrics(counts_sum(baseline[method].values())) for method in METHODS[:2]}
        for method, key in zip(METHODS[:2], ("author", "strong")):
            actual = aggregates[method]
            assert tuple(actual[k] for k in COUNT_KEYS[:3]) == expected[key], (name, method, actual)
            assert abs(actual["F1"] - expected[key + "_f1"]) <= 5e-7
        gate_rows[name] = dict(cache_path=str(expected["cache"]), cache_sha256=expected["sha"], **observed,
                               anchors=aggregates, frozen_config_frequency=dict(Counter(str(config_id(c)) for c in configs[name].values())),
                               existing_GT_anomalies=anomalies, anchor_exact=True)
        data[name] = dict(records=records, subjects=subjects, by_subject=by_subject,
                          configs=configs[name], baseline=baseline, kp=expected["k_p"])
    return data, gate_rows


def self_tests():
    config = dict(c_s=1.0, p=.55, c_d=1.0, c_b=1.0)
    # Positive affine scaling preserves quantile keep decisions, including ties.
    curve = np.array([0., 1., 0., 2., 0., 2., 0., 4., 0.])
    peaks = find_peaks(curve)[0]
    for q in Q_VALUES:
        kept, tau, fallback = choose_peaks(curve, peaks, config, q)
        transformed, tau2, _ = choose_peaks(3 * curve + 7, peaks, config, q)
        assert np.array_equal(kept, transformed) and np.isclose(tau2, 3 * tau + 7) and not fallback
    # Zero/one legal peak: exactly the Native fallback; equality at threshold survives.
    for curve in (np.zeros(8), np.array([0., 2., 0.])):
        peaks = find_peaks(curve)[0]
        for q in Q_VALUES:
            kept, tau, fallback = choose_peaks(curve, peaks, config, q)
            assert fallback and tau == native.threshold(curve, config["p"])
            assert np.array_equal(kept, find_peaks(curve, height=tau, distance=1)[0])
    tied = np.array([0., 2., 0., 2., 0.])
    assert len(choose_peaks(tied, find_peaks(tied)[0], config, .95)[0]) == 2
    assert integer_config(dict(c_s=.5, p=.55, c_d=.9, c_b=.5), 5) == dict(smoothing_width=2, peak_distance=4, boundary_radius=2)
    assert min(range(6), key=lambda i: q_rank(dict(TP=0, FP=6-i, FN=1), i)) == 5
    return dict(positive_affine_invariance=True, zero_and_one_peak_fallback=True,
                threshold_equality_and_ties=True, original_rounding=True, deterministic_F1_tie_rule=True)


def run_dataset(name, data):
    records, subjects, kp = data["records"], data["subjects"], data["kp"]
    geometry, training_counts = {}, {}
    invariants = dict(native_peak_set_equivalence_checks=0, quantile_distance_equivalence_checks=0,
                      held_out_subject_exclusion_checks=0, selected_q_outer_video_evaluations=0)

    def get_geometry(config, index):
        key = (config_id(config), index)
        if key not in geometry:
            geometry[key] = peak_geometry(records[index]["score"], config, kp)
            curve, peaks, ints = geometry[key]
            kept, tau, _ = choose_peaks(curve, peaks, config)
            assert np.array_equal(kept, find_peaks(curve, height=tau, distance=ints["peak_distance"])[0])
            invariants["native_peak_set_equivalence_checks"] += 1
        return geometry[key]

    def decode_quantile(config, index, q):
        curve, peaks, ints = get_geometry(config, index)
        kept, tau, fallback = choose_peaks(curve, peaks, config, q)
        # Check that quantile filtering retained exactly the existing distance rule.
        assert np.array_equal(kept, find_peaks(curve, height=tau, distance=ints["peak_distance"])[0])
        invariants["quantile_distance_equivalence_checks"] += 1
        return events_for(kept, ints["boundary_radius"], METHODS[2]), tau, fallback

    outer, selections, survival = [], [], []
    for held in subjects:
        config = data["configs"][held]
        train_subjects = [s for s in subjects if s != held]
        train_indices = [i for s in train_subjects for i in data["by_subject"][s]]
        assert held not in {str(records[i]["subject"]) for i in train_indices}
        invariants["held_out_subject_exclusion_checks"] += 1
        scores = []
        for q in Q_VALUES:
            pooled = native.counts_empty()
            for index in train_indices:
                key = (config_id(config), index, q)
                if key not in training_counts:
                    events, _, _ = decode_quantile(config, index, q)
                    training_counts[key] = native.evaluate(records[index], events)
                add(pooled, training_counts[key])
            scores.append(pooled)
        best = min(range(len(Q_VALUES)), key=lambda i: q_rank(scores[i], i))
        q = Q_VALUES[best]
        selections.append(dict(dataset=name, outer_subject=held, selected_q=q, **config, **integer_config(config, kp),
                               train_subject_count=len(train_subjects), train_video_count=len(train_indices),
                               train_subjects=json.dumps(train_subjects), selected_train_F1=native.f1(scores[best]),
                               selected_train_TP=scores[best]["TP"], selected_train_FP=scores[best]["FP"], selected_train_FN=scores[best]["FN"],
                               all_q_train_metrics=json.dumps([dict(q=x, **native.metrics(c)) for x, c in zip(Q_VALUES, scores)])))
        totals = {method: native.counts_empty() for method in METHODS}
        fold_survival = {method: [] for method in METHODS}
        for index in data["by_subject"][held]:
            record = records[index]
            for method in METHODS:
                cfg = AUTHOR_CONFIG if method == METHODS[0] else config
                curve, peaks, ints = get_geometry(cfg, index)
                if method == METHODS[2]:
                    events, tau, fallback = decode_quantile(config, index, q)
                    kept = np.asarray([e["peak"] for e in events], dtype=int)
                    invariants["selected_q_outer_video_evaluations"] += 1
                else:
                    kept, tau, fallback = choose_peaks(curve, peaks, cfg)
                    events = events_for(kept, ints["boundary_radius"], method)
                    reference = native.author_decode(record, kp) if method == METHODS[0] else native.tuned_decode(record, config, kp)
                    assert [(e["peak"], e["onset"], e["offset"]) for e in events] == [(e["peak"], e["onset"], e["offset"]) for e in reference]
                counts = native.evaluate(record, events)
                add(totals[method], counts)
                row = dict(dataset=name, outer_subject=held, video=str(record["video"]), method=method,
                           selected_q=q if method == METHODS[2] else None, **cfg, **ints, k_p=kp,
                           score_length=len(record["score"]), legal_peak_count=len(peaks), kept_peak_count=len(events),
                           peak_survival_rate=len(events)/len(peaks) if len(peaks) else None,
                           threshold=tau, fallback_to_native=bool(fallback), fallback_reason="fewer_than_two_legal_peaks" if fallback else "",
                           peak_times=json.dumps(kept.tolist()), intervals=json.dumps([[e["onset"],e["offset"]] for e in events]), **counts)
                fold_survival[method].append(row)
                survival.append(row)
        for method in METHODS:
            if method in METHODS[:2]:
                assert totals[method] == data["baseline"][method][held]
            peak_count = sum(r["legal_peak_count"] for r in fold_survival[method])
            outer.append(dict(dataset=name, outer_subject=held, row_type="outer_subject", method=method,
                              selected_q=q if method == METHODS[2] else None,
                              **native.metrics(totals[method]), video_count=len(data["by_subject"][held]),
                              predicted_events_per_video=totals[method]["event_count"]/len(data["by_subject"][held]),
                              legal_peak_count=peak_count, peak_survival_rate=totals[method]["event_count"]/peak_count if peak_count else None,
                              fallback_video_count=sum(r["fallback_to_native"] for r in fold_survival[method])))
    assert invariants["selected_q_outer_video_evaluations"] == len(records)
    aggregates = {}
    for method in METHODS:
        rows = [r for r in outer if r["method"] == method]
        counts = counts_sum(rows)
        peaks = sum(r["legal_peak_count"] for r in rows)
        rates = [r["peak_survival_rate"] for r in survival if r["method"] == method and r["peak_survival_rate"] is not None]
        aggregates[method] = dict(**native.metrics(counts), video_count=len(records),
                                  predicted_events_per_video=counts["event_count"]/len(records), legal_peak_count=peaks,
                                  peak_survival_rate=counts["event_count"]/peaks if peaks else None,
                                  mean_video_peak_survival_rate=float(np.mean(rates)) if rates else None,
                                  fallback_video_count=sum(r["fallback_video_count"] for r in rows))
    per_subject = {method: {r["outer_subject"]: r for r in outer if r["method"] == method} for method in METHODS}
    comparison = Counter(improved=0, equal=0, worse=0)
    sensitivity = []
    for subject in subjects:
        delta = per_subject[METHODS[2]][subject]["F1"] - per_subject[METHODS[1]][subject]["F1"]
        comparison["improved" if delta > 1e-12 else "worse" if delta < -1e-12 else "equal"] += 1
        values = {}
        for method in METHODS[1:]:
            remaining = {k: aggregates[method][k] - per_subject[method][subject][k] for k in COUNT_KEYS}
            values[method] = native.f1(remaining)
        omit_delta = values[METHODS[2]] - values[METHODS[1]]
        sensitivity.append(dict(omitted_subject=subject, remaining_quantile_F1=values[METHODS[2]],
                                remaining_strong_F1=values[METHODS[1]], delta_F1=omit_delta))
        for method in METHODS:
            per_subject[method][subject]["subject_delta_F1_vs_strong"] = per_subject[method][subject]["F1"] - per_subject[METHODS[1]][subject]["F1"]
            if method == METHODS[2]:
                per_subject[method][subject]["aggregate_delta_F1_omitting_this_subject"] = omit_delta
    qs = [r["selected_q"] for r in selections]
    frequency = {f"{q:.2f}": qs.count(q) for q in Q_VALUES}
    result = dict(metrics=aggregates, selected_q_frequency=frequency, selected_q_median=float(np.median(qs)),
                  selected_q_mode=max(frequency, key=frequency.get), selected_q_range=[min(qs),max(qs)],
                  delta_F1=aggregates[METHODS[2]]["F1"]-aggregates[METHODS[1]]["F1"],
                  subject_comparison=dict(comparison), leave_one_subject_out=sensitivity,
                  min_leave_one_subject_out_delta=min(r["delta_F1"] for r in sensitivity),
                  max_leave_one_subject_out_delta=max(r["delta_F1"] for r in sensitivity), invariants=invariants)
    for method in METHODS:
        outer.append(dict(dataset=name, outer_subject="aggregate", row_type="aggregate", method=method, **aggregates[method]))
    return result, outer, selections, survival, per_subject


def bootstrap(subjects, per_subject):
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPS, len(subjects)))
    samples = {}
    for method in METHODS[1:]:
        counts = np.array([[per_subject[method][s][k] for k in COUNT_KEYS[:3]] for s in subjects], dtype=np.int64)
        sums = counts[indices].sum(axis=1)
        den = 2*sums[:,0]+sums[:,1]+sums[:,2]
        samples[method] = np.divide(2*sums[:,0], den, out=np.zeros(BOOTSTRAP_REPS), where=den!=0)
    deltas = samples[METHODS[2]]-samples[METHODS[1]]
    return dict(repetitions=BOOTSTRAP_REPS, seed=SEED, pairing="same subject indices for both methods in each repetition",
                subjects=subjects, mean_delta_F1=float(np.mean(deltas)), ci95=np.quantile(deltas,[.025,.975]).tolist(),
                fraction_delta_positive=float(np.mean(deltas>0)), replicate_delta_F1=deltas.tolist(),
                sampled_subject_indices=indices.tolist())


def decision(results, boot):
    a,b = (results[name] for name in DATASETS)
    gap = abs(a["selected_q_median"]-b["selected_q_median"])
    contradictory = gap >= REGIME_GAP-1e-12
    overlap = sum(min(a["selected_q_frequency"][f"{q:.2f}"]/29,b["selected_q_frequency"][f"{q:.2f}"]/94) for q in Q_VALUES)
    portable = dict(median_q_gap=gap, grossly_contradictory=contradictory, gross_gap_threshold=REGIME_GAP,
                    frequency_overlap=overlap, similar_q_regime=gap<=.05+1e-12,
                    both_at_upper_grid_endpoint=all(r["selected_q_mode"]=="0.95" for r in (a,b)))
    improved, tied, degraded, stable = {}, {}, {}, {}
    for name,r in results.items():
        delta = r["delta_F1"]
        ci = boot[name]["ci95"]
        improved[name] = delta>0
        tied[name] = abs(delta)<=TIE_TOLERANCE and ci[0]<=0<=ci[1]
        degraded[name] = delta < -TIE_TOLERANCE or ci[1]<0
        stable[name] = r["min_leave_one_subject_out_delta"]>0
    if not contradictory and all(improved.values()) and all(stable.values()):
        label = "QUANTILE-CALIBRATION-STRONG-GO"
    elif not contradictory and not any(degraded.values()) and ((improved["SAMMLV"] and tied["CASME3"]) or (improved["CASME3"] and tied["SAMMLV"])):
        label = "QUANTILE-CALIBRATION-WEAK-GO"
    else:
        label = "QUANTILE-CALIBRATION-NO-GO"
    return label, portable, dict(observed_improvement=improved, effectively_tied=tied, clearly_degraded=degraded,
                                  positive_after_omitting_any_subject=stable, effective_tie_tolerance=TIE_TOLERANCE)


def report(summary, output_paths):
    lines = ["# Self-Calibrated Threshold Decoder — Stage A", "", f"判定：`{summary['decision']}`。", "",
             "## 设置与 provenance gate", "",
             "输入仅为已验证 frozen score caches。沿用当前逐 outer-fold Native 非阈值配置，合法 peaks 由原 distance 规则生成；仅将 mean-to-max height threshold 换成这些 peaks 高度的线性分位数。", "",
             "每个 outer fold 将自己的冻结配置用于该 fold 的全部 train/test videos。q 仅按 outer-train pooled Spotting F1 选择；没有重新选择 Native 非阈值参数。小于两个合法 peaks 时使用该 fold 原 p 的 Native threshold。", "",
             "q={0.70,0.75,0.80,0.85,0.90,0.95}；同分取更少 FP，再按 grid 顺序。整数 rounding、时间顺序、inclusive IoU≥0.5 greedy one-to-one matching 与原实现相同。", ""]
    for name,g in summary["provenance_anchor_gate"].items():
        lines += [f"- {name}：{g['subjects']} subjects / {g['videos']} videos / {g['gt']} GT / k_p={g['k_p']}；cache `{g['cache_path']}`；SHA-256 `{g['cache_sha256']}`。"]
        for m,row in g["anchors"].items():
            lines.append(f"  - {m} gate PASS：{row['TP']}/{row['FP']}/{row['FN']}；F1={row['F1']:.6f}。")
    lines += ["", "## 结果", "", "| Dataset | Method | TP | FP | FN | Precision | Recall | F1 | peak survival | events/video | fallback videos |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in summary["datasets"].items():
        for method,m in r["metrics"].items():
            lines.append(f"| {name} | {method} | {m['TP']} | {m['FP']} | {m['FN']} | {m['Precision']:.6f} | {m['Recall']:.6f} | {m['F1']:.6f} | {m['peak_survival_rate']:.6f} | {m['predicted_events_per_video']:.4f} | {m['fallback_video_count']} |")
    lines += ["", "peak survival 主表口径为 pooled kept peaks / pooled legal pre-height peaks；每个方法按自身冻结 distance 计算分母。逐视频比例与 dataset video-mean 也保存在 CSV/JSON。", "",
              "## Subject 稳定性与 paired bootstrap", "", "| Dataset | observed ΔF1 | improved/equal/worse | leave-one-subject-out ΔF1 min / max | bootstrap mean ΔF1 | 95% CI |",
              "|---|---:|---|---|---:|---|"]
    for name,r in summary["datasets"].items():
        c=r["subject_comparison"]; b=summary["bootstrap_summary"][name]
        lines.append(f"| {name} | {r['delta_F1']:+.6f} | {c['improved']}/{c['equal']}/{c['worse']} | {r['min_leave_one_subject_out_delta']:+.6f} / {r['max_leave_one_subject_out_delta']:+.6f} | {b['mean_delta_F1']:+.6f} | [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] |")
    lines += ["", "1000 次 subject bootstrap，seed=20260905；同一次抽样在两方法间共享 subject indices。bootstrap 和逐 subject 删除分析均固定已获得的 outer predictions，不重新选 q。", "",
              "## q regime 跨数据集稳定性", "", "| q | SAMMLV folds | CASME3 folds |", "|---:|---:|---:|"]
    for q in Q_VALUES:
        lines.append(f"| {q:.2f} | {summary['datasets']['SAMMLV']['selected_q_frequency'][f'{q:.2f}']} | {summary['datasets']['CASME3']['selected_q_frequency'][f'{q:.2f}']} |")
    portable=summary["cross_dataset_stability"]
    lines += ["", f"selected-q median 差={portable['median_q_gap']:.4f}，频率分布重叠={portable['frequency_overlap']:.4f}；similar q regime={portable['similar_q_regime']}，grossly contradictory={portable['grossly_contradictory']}。"]
    if portable["both_at_upper_grid_endpoint"]:
        lines.append("两套数据均主要选择 q=0.95，处于本次固定 grid 上端。频率相似仅说明所测范围内选择接近，不能证明真实最优 q 或可迁移性能；本次不扩 grid。")
    lines += ["", "## 判定依据与限制", "", f"`{summary['decision']}`", ""]
    for name,r in summary["datasets"].items():
        native_m, qm = r["metrics"][METHODS[1]], r["metrics"][METHODS[2]]
        lines.append(f"- {name}：ΔTP={qm['TP']-native_m['TP']:+d}、ΔFP={qm['FP']-native_m['FP']:+d}、ΔF1={r['delta_F1']:+.6f}；clearly degrades={summary['decision_checks']['clearly_degraded'][name]}。")
    lines += ["", "判读约定在运行前记录于 LOCKED_PROTOCOL_CN.md：effectively tied 要求 |ΔF1|≤0.001 且 CI 包含 0；selected-q median 差≥0.15 视为 grossly contradictory。STRONG-GO 还要求删去任何一个 subject 后两侧增益仍为正。", "",
              "该实验是既有 SAMMLV/CASME3 诊断之后的可行性检验；当前 fold 的 q 选择未使用其 test GT，但不构成对全新未见 dataset 的验证。", "",
              "CASME3 cache 中已有 1 个 malformed GT interval。遵循冻结 evaluator 原样保留该 GT，未清洗或重标注。", "",
              "## 实施与核验", "", "- PASS：两份 cache SHA 与 subjects/videos/GT/k_p 匹配。",
              "- PASS：四个 aggregate anchors 精确复现，并与可用逐 fold 历史计数核对。",
              "- PASS：逐 fold 冻结配置原样复用，outer-test subject 从 q 选择中排除。",
              "- PASS：所有计算的 quantile peak 子集与原 find_peaks distance 规则核对一致。",
              "- PASS：每个 outer video 只记录并评估所选 q 的一次测试结果；counts 守恒。",
              "- PASS：正仿射不变性、阈值 equality、ties、0/1 peak fallback 和原 rounding 测试通过。",
              "- PASS：所读 cache / Native 源码 / 历史锚点文件在本次结束时 SHA 未变。", "",
              "编码时使用 karpathy-guidelines：复用 Native-only 实现，新增代码限于分位数、LOSO 选择与结果统计。Stage A 到此结束，未启动 Stage B。", "",
              "## 输出", ""]
    lines += [f"- `{path}`" for path in output_paths]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    args=parser.parse_args()
    inputs=[NATIVE_SOURCE, native.NATIVE_AUDIT, native.PAPER_METRICS, SAMM_SELECTED, CAS_SELECTED, CAS_METRICS,
            native.SAMMLV_CACHE, native.CASME3_CACHE]
    before={str(p): native.sha256(p) for p in inputs}
    try:
        data, gate_result=gate()
    except Exception as exc:
        print(json.dumps(dict(status="BLOCKED-STAGE-A-ANCHOR-OR-PROVENANCE", reason=repr(exc))), flush=True)
        raise
    tests=self_tests()
    if args.gate_only:
        print(json.dumps(dict(gate=gate_result, self_tests=tests), ensure_ascii=False, indent=2), flush=True)
        return
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment: {OUT}")
    output=OUT/"outputs"
    output.mkdir(parents=True)
    dump(output/"quantile_stage_a_summary.json", dict(status="RUNNING", provenance_anchor_gate=gate_result))
    results, boot, selections, survival, rows_by_name = {}, {}, [], [], {}
    for name,d in data.items():
        results[name], rows, selected, trace, per_subject=run_dataset(name,d)
        rows_by_name[name]=rows
        boot[name]=bootstrap(d["subjects"],per_subject)
        selections.extend(selected); survival.extend(trace)
        print(json.dumps(dict(dataset=name, metrics=results[name]["metrics"], selected_q=results[name]["selected_q_frequency"]), ensure_ascii=False), flush=True)
    label, portable, checks=decision(results,boot)
    after={str(p):native.sha256(p) for p in inputs}
    assert before==after
    assert all(r['q'] in Q_VALUES for row in selections for r in json.loads(row['all_q_train_metrics']))
    assert len(survival)==3*(79+462) and len(selections)==29+94
    for name,rows in rows_by_name.items():
        for method in METHODS:
            total=counts_sum(r for r in rows if r["row_type"]=="outer_subject" and r["method"]==method)
            assert total["TP"]+total["FN"]==DATASETS[name]["gt"]
            assert total["TP"]+total["FP"]==total["event_count"]
        native.write_csv(output/f"quantile_outer_metrics_{name.lower()}.csv",rows)
    native.write_csv(output/"quantile_selected_q.csv",selections)
    native.write_csv(output/"quantile_peak_survival.csv",survival)
    dump(output/"quantile_bootstrap.json",boot)
    summary=dict(status="COMPLETE", decision=label, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 provenance_anchor_gate=gate_result, input_sha256=before, inputs_unchanged=before==after,
                 code_sha256=native.sha256(Path(__file__)), protocol_sha256=native.sha256(HERE/"LOCKED_PROTOCOL_CN.md"),
                 environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                 q_grid=list(Q_VALUES), quantile_method="linear", frozen_non_threshold_settings="per saved outer fold; used on all train/test videos of that fold",
                 bootstrap_summary={name:{k:b[k] for k in ("repetitions","seed","mean_delta_F1","ci95","fraction_delta_positive")} for name,b in boot.items()},
                 datasets=results, cross_dataset_stability=portable, decision_checks=checks, self_tests=tests,
                 integrity=dict(only_threshold_changed=True, no_native_retuning=True, no_test_GT_in_q_selection=True,
                                no_backbone_forward=True, no_recognition=True, no_morphology=True, no_LR_or_parameter_predictor=True,
                                no_new_boundary_or_NMS=True, no_stage_B=True))
    dump(output/"quantile_stage_a_summary.json",summary)
    paths=[OUT/"SELF_CALIBRATED_THRESHOLD_STAGE_A_CN.md"]+sorted(output.iterdir())
    paths += [Path(__file__).resolve(),HERE/"LOCKED_PROTOCOL_CN.md"]
    (OUT/"SELF_CALIBRATED_THRESHOLD_STAGE_A_CN.md").write_text(report(summary,paths),encoding="utf-8")
    print(json.dumps(dict(decision=label, bootstrap=summary["bootstrap_summary"], cross_dataset_stability=portable),indent=2),flush=True)


if __name__=="__main__":
    main()
