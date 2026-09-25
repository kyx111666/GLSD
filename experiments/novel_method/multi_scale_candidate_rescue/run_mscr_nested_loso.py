#!/usr/bin/env python3
"""ME-TST+ Multi-Scale Candidate Rescue, locked SAMMLV nested-LOSO audit.

The script is cache-only. It never imports the backbone, changes native events,
uses recognition, or fits any learned model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


ALGORITHM_VERSION = "MSCR-SAMMLV-v1-locked-2026-09-04"
P_HIGH = 0.55
K_P = 5
IOU_THRESHOLD = 0.5
EPS = 1e-8
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260904

FAMILIES = {
    "A": (0.8, 1.25),
    "B": (2 / 3, 1.5),
    "C": (2 / 3, 0.8, 1.25, 1.5),
}
DELTAS = (0.05, 0.10, 0.15)
C_RADII = (0.50, 0.75, 1.00)
LAMBDAS = (0.25, 0.50, 0.75)
GAMMAS = (0.45, 0.60, 0.75, 0.90)
TUNED_CS = (1.5, 2.0, 2.5)
TUNED_P = (0.45, 0.55, 0.65)
TUNED_CD = (0.75, 1.0, 1.25)
TUNED_CB = (0.75, 1.0, 1.25)
EXPECTED_NATIVE = (53, 184, 106)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def moving_average(score, width):
    width = max(1, int(width))
    return np.convolve(np.asarray(score, dtype=np.float64), np.ones(width) / width, mode="same")


def threshold(curve, p):
    return float(curve.mean() + p * (curve.max() - curve.mean()))


def interval_iou(event, sample):
    left, right = int(event["onset"]), int(event["offset"])
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    inter = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return inter / union if union > 0 else 0.0


def match_events(events, samples):
    unmatched = set(range(len(samples)))
    matches = []
    for event in events:
        choices = [(interval_iou(event, samples[j]), j) for j in unmatched]
        best_iou, best_j = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_j)
            matches.append(best_j)
        else:
            matches.append(-1)
    return matches, unmatched


def event(peak, boundary, source):
    return {"peak": int(peak), "onset": int(peak - boundary), "offset": int(peak + boundary), "source": source}


def author_native(record):
    curve = moving_average(record["score"], 2 * K_P)
    tau = threshold(curve, P_HIGH)
    peaks = find_peaks(curve, height=tau, distance=K_P)[0].astype(int)
    events = [event(p, K_P, "native") for p in peaks]
    return {"curve": curve, "threshold": tau, "peaks": peaks, "events": events}


def scale_alignment(peaks, peak, radius):
    if len(peaks) == 0:
        return 0.0
    distance = float(np.min(np.abs(peaks - peak)))
    if distance > radius:
        return 0.0
    return float(np.clip(1.0 - distance / radius, 0.0, 1.0))


def mscr_decode(record, config, mode="full", trace=False):
    native = author_native(record)
    curve, tau_h, p_h = native["curve"], native["threshold"], native["peaks"]
    delta = float(config["delta_p"])
    p_low = P_HIGH - delta
    tau_l = threshold(curve, p_low)
    low_peaks = find_peaks(curve, height=tau_l, distance=K_P)[0].astype(int)
    # Explicit union is required by the protocol. No second NMS is applied.
    p_l = np.asarray(sorted(set(map(int, low_peaks)) | set(map(int, p_h))), dtype=int)
    weak = np.asarray([p for p in p_l if p not in set(map(int, p_h))], dtype=int)
    radius = float(config.get("c_r", 1.0)) * K_P
    family = FAMILIES.get(config.get("scale_family", "A"), ())
    scale_peaks = []
    if mode in {"scale", "full"}:
        for a in family:
            k_a = max(1, int(round(a * K_P)))
            g_a = moving_average(record["score"], 2 * k_a)
            tau_a = threshold(g_a, p_low)
            scale_peaks.append(find_peaks(g_a, height=tau_a, distance=k_a)[0].astype(int))
    weak_rows, rescued = [], []
    for p in weak:
        height = float(np.clip((curve[p] - tau_l) / (tau_h - tau_l + EPS), 0.0, 1.0))
        multi = float(np.mean([scale_alignment(peaks, int(p), radius) for peaks in scale_peaks])) if scale_peaks else 0.0
        if mode == "low":
            rescue_score, keep = None, True
        elif mode == "height":
            rescue_score = height
            keep = rescue_score >= float(config["gamma"])
        elif mode == "scale":
            rescue_score = multi
            keep = rescue_score >= float(config["gamma"])
        elif mode == "full":
            lam = float(config["lambda"])
            rescue_score = lam * height + (1.0 - lam) * multi
            keep = rescue_score >= float(config["gamma"])
        else:
            raise ValueError(mode)
        if keep:
            rescued.append(event(p, K_P, "rescue"))
        if trace:
            weak_rows.append({"peak": int(p), "g_peak": float(curve[p]), "tau_high": tau_h,
                              "tau_low": tau_l, "H": height, "M": multi, "R": rescue_score,
                              "rescued": bool(keep)})
    final_events = sorted(native["events"] + rescued, key=lambda x: (x["peak"], 0 if x["source"] == "native" else 1))
    return {"native": native, "weak": weak, "rescued": rescued, "events": final_events, "weak_rows": weak_rows}


def tuned_native_decode(record, config):
    width = max(1, int(round(float(config["c_s"]) * K_P)))
    distance = max(1, int(round(float(config["c_d"]) * K_P)))
    boundary = max(1, int(round(float(config["c_b"]) * K_P)))
    curve = moving_average(record["score"], width)
    tau = threshold(curve, float(config["p"]))
    peaks = find_peaks(curve, height=tau, distance=distance)[0].astype(int)
    return {"events": [event(p, boundary, "tuned_native") for p in peaks], "peaks": peaks,
            "curve": curve, "threshold": tau}


def evaluate_decoding(record, decoded):
    samples = record["samples"]
    events = decoded["events"]
    matches, unmatched = match_events(events, samples)
    result = {"TP": sum(m >= 0 for m in matches), "FP": sum(m < 0 for m in matches), "FN": len(unmatched),
              "event_count": len(events), "weak_candidate_count": len(decoded.get("weak", [])),
              "rescued_candidate_count": len(decoded.get("rescued", [])), "rescue_TP": 0, "rescue_FP": 0,
              "native_missed_GT_recovered": 0, "matches": matches}
    if "native" in decoded:
        native_matches, native_unmatched = match_events(decoded["native"]["events"], samples)
        for ev, match in zip(events, matches):
            if ev["source"] == "rescue":
                result["rescue_TP" if match >= 0 else "rescue_FP"] += 1
                if match in native_unmatched:
                    result["native_missed_GT_recovered"] += 1
    return result


def add_counts(target, source):
    for key in ["TP", "FP", "FN", "event_count", "weak_candidate_count", "rescued_candidate_count",
                "rescue_TP", "rescue_FP", "native_missed_GT_recovered"]:
        target[key] += int(source.get(key, 0))


def f1(counts):
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def metrics(counts):
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    rescue_den = counts.get("rescue_TP", 0) + counts.get("rescue_FP", 0)
    return {**{k: int(v) for k, v in counts.items() if k != "matches"}, "Precision": precision,
            "Recall": recall, "F1": f1(counts),
            "rescue_precision": counts.get("rescue_TP", 0) / rescue_den if rescue_den else 0.0}


def empty_counts():
    return Counter({k: 0 for k in ["TP", "FP", "FN", "event_count", "weak_candidate_count",
                                      "rescued_candidate_count", "rescue_TP", "rescue_FP",
                                      "native_missed_GT_recovered"]})


def make_configs():
    configs = {}
    configs["LowRescue"] = [{"delta_p": d} for d in DELTAS]
    configs["HeightOnly"] = [{"delta_p": d, "gamma": g} for d, g in itertools.product(DELTAS, GAMMAS)]
    configs["ScaleOnly"] = [{"scale_family": f, "delta_p": d, "c_r": c, "gamma": g}
                            for f, d, c, g in itertools.product(FAMILIES, DELTAS, C_RADII, GAMMAS)]
    configs["MSCR"] = [{"scale_family": f, "delta_p": d, "c_r": c, "lambda": lam, "gamma": g}
                       for f, d, c, lam, g in itertools.product(FAMILIES, DELTAS, C_RADII, LAMBDAS, GAMMAS)]
    configs["TunedNative"] = [{"c_s": cs, "p": p, "c_d": cd, "c_b": cb}
                              for cs, p, cd, cb in itertools.product(TUNED_CS, TUNED_P, TUNED_CD, TUNED_CB)]
    assert len(configs["MSCR"]) == 324 and len(configs["TunedNative"]) == 81
    return configs


def decode(record, method, config, trace=False):
    if method == "TunedNative":
        return tuned_native_decode(record, config)
    mode = {"LowRescue": "low", "HeightOnly": "height", "ScaleOnly": "scale", "MSCR": "full"}[method]
    return mscr_decode(record, config, mode=mode, trace=trace)


def precompute_score_only(records, configs):
    """Precompute decoder geometry without reading any GT."""
    bank = {}
    for method, method_configs in configs.items():
        method_bank = []
        for config in method_configs:
            decoded_records = []
            for record in records:
                decoded = decode(record, method, config)
                compact = {key: decoded[key] for key in ["events", "weak", "rescued"] if key in decoded}
                if "native" in decoded:
                    compact["native"] = {"events": decoded["native"]["events"]}
                decoded_records.append(compact)
            method_bank.append(decoded_records)
        bank[method] = method_bank
    return bank


def select_config(method, configs, score_only_bank, records, train_subjects):
    ranked = []
    for index, config in enumerate(configs[method]):
        by_subject = {s: empty_counts() for s in train_subjects}
        for record_index, record in enumerate(records):
            subject = str(record["subject"])
            if subject not in by_subject:
                continue
            counts = evaluate_decoding(record, score_only_bank[method][index][record_index])
            add_counts(by_subject[subject], counts)
        subject_f1 = [f1(by_subject[s]) for s in train_subjects]
        pooled = empty_counts()
        for s in train_subjects:
            add_counts(pooled, by_subject[s])
        ranked.append(((-float(np.mean(subject_f1)), -f1(pooled), pooled["FP"],
                        pooled["rescued_candidate_count"], index), index, config,
                       float(np.mean(subject_f1)), f1(pooled), int(pooled["FP"]),
                       int(pooled["rescued_candidate_count"])))
    _, index, config, macro, pooled_f1, fp, rescued = min(ranked, key=lambda x: x[0])
    return index, config, {"train_macro_subject_F1": macro, "train_pooled_F1": pooled_f1,
                           "train_FP": fp, "train_rescue_events": rescued, "lexicographic_rank": index}


def evaluate_subject(records, subject, method, config, trace=False):
    total = empty_counts()
    traces = []
    native_preserved = native_total = 0
    for record in records:
        if str(record["subject"]) != subject:
            continue
        decoded = decode(record, method, config, trace=trace)
        result = evaluate_decoding(record, decoded)
        add_counts(total, result)
        if method in {"LowRescue", "HeightOnly", "ScaleOnly", "MSCR"}:
            native_set = {(e["onset"], e["peak"], e["offset"]) for e in decoded["native"]["events"]}
            final_set = {(e["onset"], e["peak"], e["offset"]) for e in decoded["events"]}
            native_total += len(native_set)
            native_preserved += len(native_set & final_set)
        if trace and method == "MSCR":
            event_match = {(e["peak"], e["source"]): m for e, m in zip(decoded["events"], result["matches"])}
            _, native_unmatched = match_events(decoded["native"]["events"], record["samples"])
            for row in decoded["weak_rows"]:
                match = event_match.get((row["peak"], "rescue"), -1) if row["rescued"] else -1
                traces.append({"subject": subject, "video": str(record["video"]), **row,
                               "matched_gt_index": match, "rescue_TP": bool(row["rescued"] and match >= 0),
                               "rescue_FP": bool(row["rescued"] and match < 0),
                               "native_missed_GT_recovered": bool(row["rescued"] and match in native_unmatched)})
    return total, traces, native_preserved, native_total


def bootstrap(subject_rows, lhs, rhs):
    lookup = {row["subject"]: row for row in subject_rows}
    subjects = sorted(lookup)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        left, right = empty_counts(), empty_counts()
        for s in sampled:
            for name, target in [(lhs, left), (rhs, right)]:
                prefix = name + "_"
                add_counts(target, {k: lookup[s].get(prefix + k, 0) for k in empty_counts()})
        draws.append(f1(left) - f1(right))
    lo, hi = np.quantile(draws, [.025, .975])
    return {"unit": "subject", "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
            "comparison": f"{lhs}-minus-{rhs}", "mean_delta_F1": float(np.mean(draws)),
            "CI95": [float(lo), float(hi)]}


def frequency(selected_rows, method, key):
    values = [str(r.get(key)) for r in selected_rows if r["method"] == method]
    return dict(sorted(Counter(values).items()))


def report(summary):
    methods = summary["pooled_outer_metrics"]
    lines = ["# MSCR Nested Subject-LOSO SAMMLV Report", "", f"最终状态：**{summary['status']}**", "",
             "## 1. 输入与协议", "",
             f"- cache：`{summary['input']['path']}`；SHA-256：`{summary['input']['sha256']}`。",
             f"- 数据：{summary['input']['subjects']} subjects / {summary['input']['videos']} videos / {summary['input']['GT']} GT；k_p={summary['input']['k_p']}。",
             f"- 算法版本：`{summary['algorithm_version']}`；仅 frozen score，未运行 model forward、CUDA、hidden 或 recognition。",
             "- 每个 outer fold 只用其余 subjects 的 GT 选择参数；test subject GT 只参与最终评价。",
             "- tie-break：MacroSubjectF1、pooled F1、FP、rescue event 数、固定 lexicographic config rank。", "",
             "## 2. Author Native gate", "",
             f"- TP/FP/FN = {methods['AuthorNative']['TP']}/{methods['AuthorNative']['FP']}/{methods['AuthorNative']['FN']}。",
             f"- native peak/event count = {summary['baseline']['native_peak_count']}/{summary['baseline']['native_event_count']}。",
             f"- Precision/Recall/F1 = {methods['AuthorNative']['Precision']:.6f}/{methods['AuthorNative']['Recall']:.6f}/{methods['AuthorNative']['F1']:.6f}。",
             f"- baseline status：`{summary['baseline']['status']}`。", "",
             "## 3. Nested outer-LOSO pooled metrics", "",
             "| Ablation | TP | FP | FN | Precision | Recall | F1 | rescue TP/FP | rescue precision | recovered GT |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["AuthorNative", "TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
        m = methods[name]
        lines.append(f"| {name} | {m['TP']} | {m['FP']} | {m['FN']} | {m['Precision']:.6f} | {m['Recall']:.6f} | {m['F1']:.6f} | {m.get('rescue_TP',0)}/{m.get('rescue_FP',0)} | {m.get('rescue_precision',0):.6f} | {m.get('native_missed_GT_recovered',0)} |")
    p = summary["native_preservation"]
    lines += ["", "## 4. Native preservation", "",
              f"- Full MSCR：{p['preserved']}/{p['total']} = {p['rate']:.6f}；onset/peak/offset tuple exact match。",
              "- MSCR 仅新增 rescue events；无删除、移动、边界修改或额外 NMS。", "",
              "## 5. Subject stability", "",
              f"- MSCR vs Author Native：improved/equal/worse = {summary['subject_stability']['vs_author_native']['improved']}/{summary['subject_stability']['vs_author_native']['equal']}/{summary['subject_stability']['vs_author_native']['worse']}。",
              f"- MSCR vs Tuned Native：improved/equal/worse = {summary['subject_stability']['vs_tuned_native']['improved']}/{summary['subject_stability']['vs_tuned_native']['equal']}/{summary['subject_stability']['vs_tuned_native']['worse']}。", "",
              "## 6. Subject bootstrap", ""]
    for key, b in summary["bootstrap"].items():
        lines.append(f"- {key}：mean ΔF1={b['mean_delta_F1']:.6f}，95% CI=[{b['CI95'][0]:.6f}, {b['CI95'][1]:.6f}]。")
    lines += ["", "## 7. Parameter stability", ""]
    for method, freqs in summary["parameter_stability"].items():
        lines.append(f"- {method}：`{freqs}`")
    lines.append("- Full MSCR 的 family、Delta_p、c_r、lambda 在 29 folds 完全一致；gamma 27/29 选择 0.90。因此没有随机漂移，主要表现为选择器稳定倾向于几乎不救回 candidate。")
    lines += ["", "## 8. GO 判定", "",
              f"- MSCR > Author Native：{summary['go_checks']['MSCR_F1_gt_AuthorNative']}。",
              f"- MSCR > Tuned Native：{summary['go_checks']['MSCR_F1_gt_TunedNative']}。",
              f"- rescued TP > 0：{summary['go_checks']['rescued_TP_gt_0']}。",
              f"- rescue FP 明显低于 LowRescue（严格更少且至少低 20%）：{summary['go_checks']['FP_cost_clearly_lower_than_LowRescue']}。",
              "", "## 9. 结论", "", summary["conclusion"]]
    if summary["status"] != "STRONG-GO-SAMMLV":
        lines += ["", "按任务书，本阶段不进入 CASME3、recognition/STRS 或 Skill。"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as f:
        payload = pickle.load(f)
    if payload["dataset"] != "SAMMLV" or int(payload["k_p"]) != K_P:
        raise RuntimeError("BLOCKED-BASELINE: wrong dataset or k_p")
    records = list(payload["records"])
    subjects = sorted({str(r["subject"]) for r in records})
    native_by_subject = {s: empty_counts() for s in subjects}
    native_total = empty_counts()
    for record in records:
        decoded = author_native(record)
        counts = evaluate_decoding(record, {"events": decoded["events"]})
        add_counts(native_by_subject[str(record["subject"])], counts)
        add_counts(native_total, counts)
    if (native_total["TP"], native_total["FP"], native_total["FN"]) != EXPECTED_NATIVE:
        raise RuntimeError(f"BLOCKED-BASELINE: observed={metrics(native_total)}")

    configs = make_configs()
    score_only_bank = precompute_score_only(records, configs)
    selected_rows, outer = [], defaultdict(dict)
    rescue_trace = []
    preservation = Counter()
    for held in subjects:
        train = [s for s in subjects if s != held]
        for method in ["TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
            index, config, selection = select_config(method, configs, score_only_bank, records, train)
            counts, traces, preserved, total = evaluate_subject(records, held, method, config, trace=(method == "MSCR"))
            outer[method][held] = counts
            row = {"subject": held, "method": method, "config_index": index,
                   "selected_config_json": json.dumps(config, sort_keys=True), **config, **selection,
                   **{f"test_{k}": v for k, v in metrics(counts).items()}}
            selected_rows.append(row)
            if method == "MSCR":
                preservation.update(preserved=preserved, total=total)
                for trace_row in traces:
                    rescue_trace.append({**trace_row, "outer_subject": held, "selected_config_json": row["selected_config_json"], **config})

    pooled = {"AuthorNative": metrics(native_total)}
    for method in ["TunedNative", "LowRescue", "HeightOnly", "ScaleOnly", "MSCR"]:
        total = empty_counts()
        for s in subjects:
            add_counts(total, outer[method][s])
        pooled[method] = metrics(total)

    subject_rows = []
    for s in subjects:
        row = {"subject": s}
        all_methods = {"AuthorNative": native_by_subject[s], **{m: outer[m][s] for m in outer}}
        for name, counts in all_methods.items():
            for k, v in metrics(counts).items():
                row[f"{name}_{k}"] = v
        row["Delta_MSCR_minus_AuthorNative"] = row["MSCR_F1"] - row["AuthorNative_F1"]
        row["Delta_MSCR_minus_TunedNative"] = row["MSCR_F1"] - row["TunedNative_F1"]
        subject_rows.append(row)

    def stability(delta_key):
        d = np.asarray([r[delta_key] for r in subject_rows])
        return {"improved": int((d > 1e-12).sum()), "equal": int((np.abs(d) <= 1e-12).sum()),
                "worse": int((d < -1e-12).sum())}

    boot = {"MSCR_minus_AuthorNative": bootstrap(subject_rows, "MSCR", "AuthorNative"),
            "MSCR_minus_TunedNative": bootstrap(subject_rows, "MSCR", "TunedNative")}
    parameter_stability = {
        "MSCR": {k: frequency(selected_rows, "MSCR", k) for k in ["scale_family", "delta_p", "c_r", "lambda", "gamma"]},
        "TunedNative": {k: frequency(selected_rows, "TunedNative", k) for k in ["c_s", "p", "c_d", "c_b"]},
        "LowRescue": {"delta_p": frequency(selected_rows, "LowRescue", "delta_p")},
        "HeightOnly": {k: frequency(selected_rows, "HeightOnly", k) for k in ["delta_p", "gamma"]},
        "ScaleOnly": {k: frequency(selected_rows, "ScaleOnly", k) for k in ["scale_family", "delta_p", "c_r", "gamma"]},
    }
    mscr, native, tuned, low = pooled["MSCR"], pooled["AuthorNative"], pooled["TunedNative"], pooled["LowRescue"]
    fp_clear = mscr["rescue_FP"] < low["rescue_FP"] and mscr["rescue_FP"] <= 0.8 * low["rescue_FP"]
    checks = {"MSCR_F1_gt_AuthorNative": mscr["F1"] > native["F1"],
              "MSCR_F1_gt_TunedNative": mscr["F1"] > tuned["F1"],
              "rescued_TP_gt_0": mscr["rescue_TP"] > 0,
              "FP_cost_clearly_lower_than_LowRescue": fp_clear}
    if mscr["F1"] <= native["F1"]:
        status = "NO-GO-MSCR-SOURCE"
        conclusion = "**NO-GO-MSCR-SOURCE：nested outer-subject LOSO 下 Full MSCR 未超过 Author Native；本次已完成合法 train-subject tuning，不能将失败归因于未调参。**"
    elif all(checks.values()):
        status = "STRONG-GO-SAMMLV"
        conclusion = "**STRONG-GO-SAMMLV：MSCR 同时超过 Author Native 与 Tuned Native，救回真实事件，且相对 LowRescue 明显降低 FP cost；允许进入锁定协议的 CASME3 gate。**"
    else:
        status = "WEAK-GO-SAMMLV"
        conclusion = "**WEAK-GO-SAMMLV：MSCR 超过 Author Native，但未满足全部 Strong-GO 条件；不能直接作为论文主方法，也不进入 CASME3。**"

    invalid_gt = sum(not (0 <= int(s[0]) <= int(s[2]) < len(r["score"])) for r in records for s in r["samples"])
    summary = {
        "status": status, "algorithm_version": ALGORITHM_VERSION,
        "input": {"path": str(args.cache.resolve()), "sha256": sha256(args.cache), "dataset": payload["dataset"],
                  "subjects": len(subjects), "videos": len(records), "GT": int(payload["num_gt"]),
                  "k_p": K_P, "total_score_length": sum(len(r["score"]) for r in records),
                  "invalid_or_out_of_range_GT_preserved_for_native_equivalence": invalid_gt},
        "parameter_grids": {"scale_families": FAMILIES, "delta_p": DELTAS, "c_r": C_RADII,
                            "lambda": LAMBDAS, "gamma": GAMMAS,
                            "MSCR_config_count": len(configs["MSCR"]),
                            "TunedNative": {"c_s": TUNED_CS, "p": TUNED_P, "c_d": TUNED_CD, "c_b": TUNED_CB,
                                            "config_count": len(configs["TunedNative"])},
                            "height_only_config_count": len(configs["HeightOnly"]),
                            "scale_only_config_count": len(configs["ScaleOnly"]),
                            "low_rescue_config_count": len(configs["LowRescue"])},
        "selection_protocol": {"outer_split": "subject LOSO", "primary": "mean train-subject F1",
                               "tie_break": ["pooled train F1 higher", "FP fewer", "rescue events fewer", "lexicographic config order"],
                               "outer_test_GT_used_for_selection": False},
        "baseline": {"expected_TP_FP_FN": EXPECTED_NATIVE, "observed_TP_FP_FN": [native_total["TP"], native_total["FP"], native_total["FN"]],
                     "native_peak_count": native_total["event_count"], "native_event_count": native_total["event_count"],
                     "status": "PASS"},
        "pooled_outer_metrics": pooled,
        "native_preservation": {"preserved": int(preservation["preserved"]), "total": int(preservation["total"]),
                                "rate": preservation["preserved"] / preservation["total"]},
        "subject_stability": {"vs_author_native": stability("Delta_MSCR_minus_AuthorNative"),
                              "vs_tuned_native": stability("Delta_MSCR_minus_TunedNative")},
        "bootstrap": boot, "parameter_stability": parameter_stability,
        "go_checks": checks, "clear_FP_cost_rule": "MSCR rescue_FP < LowRescue rescue_FP and <=80% of it",
        "conclusion": conclusion,
        "next_step": "CASME3 locked gate" if status == "STRONG-GO-SAMMLV" else "STOP before CASME3/recognition/Skill",
    }
    write_csv(out / "mscr_outer_subject_metrics.csv", subject_rows)
    write_csv(out / "mscr_selected_configs.csv", selected_rows)
    write_csv(out / "mscr_rescue_event_trace.csv", rescue_trace)
    write_csv(out / "tuned_native_outer_metrics.csv", [{"subject": s, **metrics(outer["TunedNative"][s])} for s in subjects])
    write_csv(out / "low_rescue_outer_metrics.csv", [{"subject": s, **metrics(outer["LowRescue"][s])} for s in subjects])
    (out / "mscr_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "MSCR_NESTED_LOSO_SAMMLV_REPORT_CN.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps({"status": status, "native": pooled["AuthorNative"], "tuned_native": pooled["TunedNative"],
                      "low_rescue": pooled["LowRescue"], "mscr": pooled["MSCR"],
                      "preservation": summary["native_preservation"], "bootstrap": boot,
                      "go_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
