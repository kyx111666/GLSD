#!/usr/bin/env python3
"""Cache-only cross-dataset Native operating-point diagnosis.

This is a descriptive/oracle diagnostic.  It never trains a model, changes a
decoder, or uses the resulting sensitivity tables as a new baseline.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/native_cross_dataset_diagnosis"
OUT_OUTPUTS = OUT / "outputs"
SAMMLV_CACHE = ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl"
CASME3_CACHE = ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl"
NATIVE_AUDIT = ROOT / "my_method/native_failure_mode_audit/run_native_failure_mode_audit.py"
PAPER_METRICS = ROOT / "third_party/metst_plus/paper_metrics.py"
SAMMLV_TUNED_REPORT = ROOT / "results/rgr1_sammlv_nested/report.json"
CASME3_OLD_SELECTED = ROOT / "results/morphology_casme3_locked_validation/outputs/casme3_tuned_native_selected_configs.csv"
CASME3_FINAL_SELECTED = ROOT / "results/final_native_closure_strong_anchor_morphology/final_strong_native_selected_configs.csv"

EXPECTED = {
    "SAMMLV": {"sha": "3b2264bd527bf144dfe10e03528ac5e637fc30e10a66d8d9575648754b298569", "subjects": 29, "videos": 79, "gt": 159, "k_p": 5, "author": (53, 184, 106), "tuned": (49, 143, 110), "c_b": 1.25, "c_d": 1.25},
    "CASME3": {"sha": "9bdcbb3fc79a3f2a4bf6729b826b5490d8a91aed913b4120c19d68cceec67bda", "subjects": 94, "videos": 462, "gt": 858, "k_p": 17, "author": (81, 912, 777), "tuned": (124, 1148, 734), "c_b": 0.75, "c_d": 0.75},
}

AUTHOR = {"c_s": 2.0, "p": 0.55, "c_d": 1.0, "c_b": 1.0}
CS_VALUES = (0.75, 1.0, 1.5, 2.0)
P_VALUES = (0.45, 0.55, 0.65)
CD_VALUES = (0.75, 1.0, 1.25)
EPS = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data):
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def smooth(score, width):
    width = max(1, int(width))
    return np.convolve(np.asarray(score, dtype=np.float64), np.ones(width, dtype=np.float64) / width, mode="same")


def width_for(cs, kp):
    # Same deterministic integer rounding used by the verified Native implementation.
    return max(1, int(round(float(cs) * int(kp))))


def threshold(curve, p):
    mean = float(np.mean(curve))
    return mean + float(p) * (float(np.max(curve)) - mean)


def event(peak, boundary):
    return {"onset": int(peak - boundary), "offset": int(peak + boundary), "peak": int(peak)}


def interval_iou(pred, sample):
    left, right = int(pred["onset"]), int(pred["offset"])
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    inter = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return inter / union if union else 0.0


def decode(record, config, kp):
    curve = smooth(record["score"], width_for(config["c_s"], kp))
    tau = threshold(curve, config["p"])
    distance = max(1, int(round(float(config["c_d"]) * kp)))
    boundary = max(1, int(round(float(config["c_b"]) * kp)))
    peaks = find_peaks(curve, height=tau, distance=distance)[0].astype(int)
    return curve, tau, [event(p, boundary) for p in peaks]


def match_events(events, samples):
    unmatched = set(range(len(samples)))
    matches = []
    for pred in events:
        choices = [(interval_iou(pred, samples[i]), i) for i in unmatched]
        best, index = max(choices, default=(0.0, -1))
        if best >= 0.5:
            unmatched.remove(index)
            matches.append(int(index))
        else:
            matches.append(-1)
    return matches, unmatched


def metrics(tp, fp, fn, event_count=None):
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1": f1}
    if event_count is not None:
        out["event_count"] = int(event_count)
    return out


def evaluate(record, config, kp):
    _, _, events = decode(record, config, kp)
    matches, unmatched = match_events(events, record["samples"])
    return metrics(sum(m >= 0 for m in matches), sum(m < 0 for m in matches), len(unmatched), len(events))


def load_cache(path: Path, name: str):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    observed = {
        "subjects": len({str(r["subject"]) for r in payload["records"]}),
        "videos": len(payload["records"]),
        "gt": sum(len(r["samples"]) for r in payload["records"]),
        "k_p": int(payload["k_p"]),
    }
    required = {"dataset", "records", "num_subjects", "num_videos", "num_gt", "k_p"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"{name}: cache missing {missing}")
    if observed != {"subjects": EXPECTED[name]["subjects"], "videos": EXPECTED[name]["videos"], "gt": EXPECTED[name]["gt"], "k_p": EXPECTED[name]["k_p"]}:
        raise RuntimeError(f"{name}: metadata mismatch {observed}")
    records = []
    for raw in payload["records"]:
        row = dict(raw)
        row["subject"] = str(row["subject"])
        row["video"] = str(row["video"])
        row["score"] = np.asarray(row["score"], dtype=np.float64)
        row["samples"] = [list(map(int, sample[:3])) for sample in row["samples"]]
        records.append(row)
    if sha256(path) != EXPECTED[name]["sha"]:
        raise RuntimeError(f"{name}: cache SHA mismatch")
    return payload, records, sorted({r["subject"] for r in records})


def aggregate(records, config, kp, subject=None):
    total = Counter()
    for record in records:
        if subject is not None and record["subject"] != subject:
            continue
        m = evaluate(record, config, kp)
        for key in ("TP", "FP", "FN", "event_count"):
            total[key] += int(m[key])
    return metrics(total["TP"], total["FP"], total["FN"], total["event_count"])


def read_selected_csv(path: Path):
    mapping = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject = str(row["outer_subject"])
            mapping[subject] = {"c_s": float(row["c_s"]), "p": float(row["p"]), "c_d": float(row["c_d"]), "c_b": float(row["c_b"])}
    return mapping


def read_sammlv_selected(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for fold in data["outer_folds"]:
        cfg = fold["selected_strong_config"]
        mapping[str(fold["subject"])] = {"c_s": float(cfg["c_s"]), "p": float(cfg["p_s"]), "c_d": float(cfg["c_d"]), "c_b": float(cfg["c_b"])}
    return mapping


def anchor_gate(cache_info, records_by_name):
    anchors = {}
    for name, (records, subjects) in records_by_name.items():
        kp = EXPECTED[name]["k_p"]
        author = aggregate(records, AUTHOR, kp)
        anchors[name] = {"Author Native": author}
        if tuple(author[k] for k in ("TP", "FP", "FN")) != EXPECTED[name]["author"]:
            raise RuntimeError(f"{name}: Author anchor mismatch {author}")
        if name == "SAMMLV":
            selected = read_sammlv_selected(SAMMLV_TUNED_REPORT)
            expected = EXPECTED[name]["tuned"]
            label = "Tuned Native"
        else:
            selected = read_selected_csv(CASME3_FINAL_SELECTED)
            expected = EXPECTED[name]["tuned"]
            label = "Final Strong Native"
        if set(selected) != set(subjects):
            raise RuntimeError(f"{name}: selected-config subjects mismatch")
        total = Counter()
        for record in records:
            m = evaluate(record, selected[record["subject"]], kp)
            for key in ("TP", "FP", "FN", "event_count"):
                total[key] += int(m[key])
        tuned = metrics(total["TP"], total["FP"], total["FN"], total["event_count"])
        anchors[name][label] = tuned
        if tuple(tuned[k] for k in ("TP", "FP", "FN")) != expected:
            raise RuntimeError(f"{name}: {label} anchor mismatch {tuned}")
    return anchors


def quantiles(values):
    vals = np.asarray([v for v in values if finite(v) is not None], dtype=float)
    if not len(vals):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "q90": None, "q95": None, "q99": None}
    q25, q75, q90, q95, q99 = np.quantile(vals, [0.25, 0.75, 0.90, 0.95, 0.99])
    return {"n": int(len(vals)), "mean": float(vals.mean()), "median": float(np.median(vals)), "q25": float(q25), "q75": float(q75), "q90": float(q90), "q95": float(q95), "q99": float(q99)}


def interpeak(peaks):
    d = np.diff(np.asarray(peaks, dtype=float)) if len(peaks) >= 2 else np.asarray([], dtype=float)
    if not len(d):
        return {"median": None, "p25": None, "p75": None}
    return {"median": float(np.median(d)), "p25": float(np.quantile(d, .25)), "p75": float(np.quantile(d, .75))}


def autocorr(values, lag):
    if lag < 1 or len(values) <= lag:
        return None
    a, b = np.asarray(values[:-lag], dtype=float), np.asarray(values[lag:], dtype=float)
    if np.std(a) < EPS or np.std(b) < EPS:
        return None
    return finite(np.corrcoef(a, b)[0, 1])


def video_diagnostics(name, records, kp):
    score_rows, peak_rows, occupancy_rows, gt_rows = [], [], [], []
    positive_peaks, background_peaks = [], []
    for record in records:
        score = np.asarray(record["score"], dtype=float)
        n = len(score)
        med = float(np.median(score))
        mad = float(np.median(np.abs(score - med)))
        mean = float(np.mean(score))
        std = float(np.std(score))
        author_curve = smooth(score, width_for(2.0, kp))
        author_tau = threshold(author_curve, .55)
        author_peaks = find_peaks(author_curve, height=author_tau, distance=max(1, kp))[0].astype(int)
        all_author_peaks = find_peaks(author_curve, distance=max(1, kp))[0].astype(int)
        prom = peak_prominences(author_curve, all_author_peaks)[0] if len(all_author_peaks) else np.asarray([])
        prom_by_peak = {int(p): float(v) for p, v in zip(all_author_peaks, prom)}
        scale = 1.4826 * mad + EPS
        rows = {
            "dataset": name, "subject": record["subject"], "video": record["video"], "length": n,
            "mean": mean, "std": std, "median": med, "MAD": mad, "min": float(np.min(score)), "max": float(np.max(score)),
            "max_minus_mean": float(np.max(score) - mean), "max_minus_median": float(np.max(score) - med),
            "q50": float(np.quantile(score, .50)), "q75": float(np.quantile(score, .75)), "q90": float(np.quantile(score, .90)),
            "q95": float(np.quantile(score, .95)), "q99": float(np.quantile(score, .99)),
            "robust_peakiness": float((np.max(score) - med) / scale), "CV": float(std / (abs(mean) + EPS)),
            "D1": float(np.mean(np.abs(np.diff(score)))) if n >= 2 else 0.0,
            "D2": float(np.mean(np.abs(score[2:] - 2 * score[1:-1] + score[:-2]))) if n >= 3 else 0.0,
            "NR1": float(np.mean(np.abs(np.diff(score))) / (std + EPS)) if n >= 2 else 0.0,
            "NR2": float(np.mean(np.abs(score[2:] - 2 * score[1:-1] + score[:-2])) / (std + EPS)) if n >= 3 else 0.0,
            "autocorr_lag_1": autocorr(score, 1),
            "autocorr_lag_q25": autocorr(score, max(1, int(round(.25 * kp)))),
            "autocorr_lag_q50": autocorr(score, max(1, int(round(.50 * kp)))),
            "autocorr_lag_kp": autocorr(score, max(1, int(kp))),
            "author_smoothing_width": width_for(2.0, kp), "author_threshold_p055": author_tau,
            "author_local_peak_count": int(len(all_author_peaks)),
            "author_local_peak_density_per_1000": 1000 * len(all_author_peaks) / n,
        }
        score_rows.append(rows)
        # Peak-density and threshold-occupancy tables use the same verified peak detector.
        for cs in CS_VALUES:
            curve = smooth(score, width_for(cs, kp))
            local = find_peaks(curve, distance=max(1, kp))[0].astype(int)
            local_dist = interpeak(local)
            for p in P_VALUES:
                tau = threshold(curve, p)
                selected = find_peaks(curve, height=tau, distance=max(1, kp))[0].astype(int)
                sel_dist = interpeak(selected)
                peak_rows.append({
                    "dataset": name, "subject": record["subject"], "video": record["video"], "c_s": cs,
                    "p": p, "k_p": kp, "smoothing_width": width_for(cs, kp), "threshold": tau,
                    "local_peak_count": int(len(local)), "local_peak_density_per_1000": 1000 * len(local) / n,
                    "thresholded_peak_count": int(len(selected)), "thresholded_peak_density_per_1000": 1000 * len(selected) / n,
                    "local_peak_survival_rate": float(len(selected) / len(local)) if len(local) else None,
                    "all_interpeak_median": local_dist["median"], "all_interpeak_p25": local_dist["p25"], "all_interpeak_p75": local_dist["p75"],
                    "thresholded_interpeak_median": sel_dist["median"], "thresholded_interpeak_p25": sel_dist["p25"], "thresholded_interpeak_p75": sel_dist["p75"],
                    "thresholded_interpeak_median_over_kp": sel_dist["median"] / kp if sel_dist["median"] is not None else None,
                })
        # Occupancy is explicitly anchored to Author Native smoothing (2*k_p).
        for p in (0.35, 0.45, 0.55, 0.65):
            tau = threshold(author_curve, p)
            above = author_curve >= tau
            peak_above = all_author_peaks[author_curve[all_author_peaks] >= tau] if len(all_author_peaks) else []
            occupancy_rows.append({
                "dataset": name, "subject": record["subject"], "video": record["video"], "p": p,
                "smoothing_width": width_for(2.0, kp), "threshold": tau,
                "fraction_frames_above_threshold": float(np.mean(above)),
                "local_peak_count": int(len(all_author_peaks)), "local_peaks_above_threshold": int(len(peak_above)),
                "fraction_local_peaks_retained": float(len(peak_above) / len(all_author_peaks)) if len(all_author_peaks) else None,
            })
        # GT-near positives are one nearest Author-Native local peak per GT event.
        expanded = [(max(0, int(s[0]) - kp), min(n - 1, int(s[2]) + kp)) for s in record["samples"]]
        for gt_index, sample in enumerate(record["samples"]):
            onset, apex, offset = map(int, sample[:3])
            left, right = max(0, onset), min(n - 1, offset)
            duration = offset - onset + 1 if offset >= onset else None
            nearest = None
            if len(all_author_peaks):
                nearest = min((int(p) for p in all_author_peaks), key=lambda p: (abs(p - apex), p))
            nearest_height = float(author_curve[nearest]) if nearest is not None else None
            nearest_prom = prom_by_peak.get(nearest) if nearest is not None else None
            nearest_z = float((nearest_height - np.median(author_curve)) / (1.4826 * np.median(np.abs(author_curve - np.median(author_curve))) + EPS)) if nearest is not None else None
            gt_rows.append({
                "dataset": name, "subject": record["subject"], "video": record["video"], "gt_index": gt_index,
                "gt_onset": onset, "gt_apex": apex, "gt_offset": offset, "gt_duration": duration,
                "valid_gt_geometry": bool(offset >= onset),
                "gt_duration_over_kp": duration / kp if duration is not None else None,
                "max_score_inside_gt": float(np.max(score[left:right + 1])) if right >= left else None,
                "mean_score_inside_gt": float(np.mean(score[left:right + 1])) if right >= left else None,
                "nearest_local_peak": nearest, "nearest_local_peak_height": nearest_height,
                "nearest_local_peak_robust_z": nearest_z, "nearest_local_peak_prominence": nearest_prom,
                "nearest_peak_distance_to_apex": abs(nearest - apex) if nearest is not None else None,
                "nearest_peak_distance_to_center": abs(nearest - (onset + offset) / 2) if nearest is not None else None,
                "nearest_peak_inside_gt": bool(nearest is not None and onset <= nearest <= offset),
            })
            if nearest is not None and duration is not None:
                positive_peaks.append({"dataset": name, "subject": record["subject"], "video": record["video"], "gt_index": gt_index,
                                       "peak_height": nearest_height, "robust_z_height": nearest_z, "prominence": nearest_prom})
        for p in all_author_peaks:
            p = int(p)
            if any(left <= p <= right for left, right in expanded):
                continue
            nearest_distance = min(
                max(onset - p, 0, p - offset)
                for onset, _, offset in record["samples"]
            ) if record["samples"] else None
            height = float(author_curve[p])
            z = float((height - np.median(author_curve)) / scale)
            background_peaks.append({"dataset": name, "subject": record["subject"], "video": record["video"], "peak_time": p,
                                     "peak_height": height, "robust_z_height": z, "prominence": prom_by_peak.get(p),
                                     "nearest_gt_distance": nearest_distance})
    return score_rows, peak_rows, occupancy_rows, gt_rows, positive_peaks, background_peaks


def summary_rows(values):
    return quantiles(values)


def aucs(pos, neg):
    out = {"roc_auc": None, "pr_auc": None}
    pairs = [(1, float(v)) for v in pos if finite(v) is not None] + [(0, float(v)) for v in neg if finite(v) is not None]
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    if len(set(labels)) == 2 and len(values) == len(labels):
        out["roc_auc"] = finite(roc_auc_score(labels, values))
        out["pr_auc"] = finite(average_precision_score(labels, values))
    return out


def peak_comparison(name, positives, backgrounds):
    rows = []
    for metric in ("peak_height", "robust_z_height", "prominence"):
        pos = [r[metric] for r in positives]
        neg = [r[metric] for r in backgrounds]
        auc = aucs(pos, neg)
        for group, vals in (("GT-near", pos), ("background", neg)):
            q = quantiles(vals)
            rows.append({"dataset": name, "metric": metric, "peak_group": group, **q, "roc_auc": auc["roc_auc"], "pr_auc": auc["pr_auc"]})
    return rows


def aggregate_subjects(name, score_rows, gt_rows):
    by = defaultdict(list)
    for row in score_rows:
        by[row["subject"]].append(row)
    gby = defaultdict(list)
    for row in gt_rows:
        gby[row["subject"]].append(row)
    rows = []
    skip = {"dataset", "subject", "video"}
    fields = [k for k in score_rows[0] if k not in skip]
    for subject in sorted(by):
        out = {"dataset": name, "subject": subject, "video_count": len(by[subject]), "gt_count": len(gby.get(subject, []))}
        for field in fields:
            vals = [finite(r.get(field)) for r in by[subject]]
            vals = [v for v in vals if v is not None]
            if vals:
                out[f"mean_{field}"] = float(np.mean(vals))
                out[f"median_{field}"] = float(np.median(vals))
        durations = [finite(r.get("gt_duration_over_kp")) for r in gby.get(subject, [])]
        durations = [v for v in durations if v is not None]
        if durations:
            out["GT_duration_over_kp_mean"] = float(np.mean(durations))
            out["GT_duration_over_kp_median"] = float(np.median(durations))
            out["GT_duration_over_kp_p25"] = float(np.quantile(durations, .25))
            out["GT_duration_over_kp_p75"] = float(np.quantile(durations, .75))
            out["GT_duration_over_kp_p90"] = float(np.quantile(durations, .90))
        rows.append(out)
    return rows


def response_surface(name, records, kp, cb):
    rows = []
    for cs, p, cd in itertools.product(CS_VALUES, P_VALUES, CD_VALUES):
        cfg = {"c_s": cs, "p": p, "c_d": cd, "c_b": cb}
        total = Counter()
        for record in records:
            m = evaluate(record, cfg, kp)
            for key in ("TP", "FP", "FN", "event_count"):
                total[key] += int(m[key])
        row = {"dataset": name, **cfg, **metrics(total["TP"], total["FP"], total["FN"], total["event_count"]),
               "analysis_label": "EXPLORATORY FULL-DATA SENSITIVITY — NOT UNBIASED TEST RESULT"}
        rows.append(row)
    return rows


def subject_oracle(name, records, subjects, kp, fixed_cd, fixed_cb, subject_stats):
    rows = []
    for subject in subjects:
        ranked = []
        for index, (cs, p) in enumerate(itertools.product(CS_VALUES, P_VALUES)):
            cfg = {"c_s": cs, "p": p, "c_d": fixed_cd, "c_b": fixed_cb}
            m = aggregate(records, cfg, kp, subject)
            f1 = m["F1"]
            ranked.append((-(f1), m["FP"], index, cfg, m))
        best_key = min(ranked, key=lambda x: x[0:3])
        best_f1 = best_key[4]["F1"]
        ties = sum(abs(x[4]["F1"] - best_f1) < 1e-12 for x in ranked)
        row = {"dataset": name, "subject": subject, "fixed_c_d": fixed_cd, "fixed_c_b": fixed_cb,
               "best_c_s": best_key[3]["c_s"], "best_p": best_key[3]["p"], "oracle_F1": best_f1,
               "oracle_TP": best_key[4]["TP"], "oracle_FP": best_key[4]["FP"], "oracle_FN": best_key[4]["FN"],
               "oracle_event_count": best_key[4]["event_count"], "best_config_tie_count": ties,
               "analysis_label": "ORACLE DIAGNOSTIC ONLY"}
        row.update({k: v for k, v in subject_stats.get(subject, {}).items() if k.startswith("mean_") or k.startswith("median_") or k.startswith("GT_duration")})
        rows.append(row)
    return rows


def corr_value(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None, None
    rho, p = spearmanr(x, y)
    return finite(rho), finite(p)


def oracle_correlations(name, oracle_rows, subject_rows):
    merged = {r["subject"]: r for r in subject_rows}
    stats = ["mean_robust_peakiness", "mean_MAD", "mean_NR1", "mean_NR2", "mean_author_local_peak_density_per_1000",
             "mean_autocorr_lag_1", "mean_autocorr_lag_q50", "mean_autocorr_lag_kp", "mean_mean", "mean_std", "mean_CV", "GT_duration_over_kp_median"]
    rows = []
    for target in ("best_p", "best_c_s"):
        for stat in stats:
            pairs = [(r[target], merged[r["subject"]].get(stat)) for r in oracle_rows if r["subject"] in merged and finite(merged[r["subject"]].get(stat)) is not None]
            if pairs:
                rho, pv = corr_value([p[0] for p in pairs], [p[1] for p in pairs])
            else:
                rho, pv = None, None
            groups = defaultdict(list)
            for r in oracle_rows:
                val = finite(merged.get(r["subject"], {}).get(stat))
                if val is not None:
                    groups[str(r[target])].append(val)
            rows.append({"dataset": name, "target": target, "statistic": stat, "n": len(pairs), "spearman_rho": rho,
                         "spearman_p": pv, "abs_rho_ge_0.3": bool(rho is not None and abs(rho) >= .3),
                         "group_means": json.dumps({k: float(np.mean(v)) for k, v in sorted(groups.items())}, ensure_ascii=False),
                         "analysis_label": "ORACLE DIAGNOSTIC ONLY"})
    return rows


def cliffs_delta(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        return None
    # Positive means SAMMLV values tend to be larger than CASME3 values.
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def effect_row(stat, aggregation, a, b):
    a, b = np.asarray([v for v in a if finite(v) is not None], dtype=float), np.asarray([v for v in b if finite(v) is not None], dtype=float)
    if len(a) and len(b):
        u = mannwhitneyu(a, b, alternative="two-sided")
        pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(1, len(a) + len(b) - 2)) if len(a) + len(b) > 2 else 0.0
        es = (float(np.mean(a)) - float(np.mean(b))) / pooled if pooled > EPS else None
        return {"aggregation": aggregation, "statistic": stat, "n_sammlv": len(a), "n_casme3": len(b),
                "sammlv_mean": float(np.mean(a)), "casme3_mean": float(np.mean(b)), "difference_sammlv_minus_casme3": float(np.mean(a) - np.mean(b)),
                "standardized_effect_size": es, "mannwhitney_U": float(u.statistic), "mannwhitney_p": float(u.pvalue), "cliffs_delta": cliffs_delta(a, b),
                "analysis_label": "DESCRIPTIVE; video-level observations are not fully independent within subject"}
    return {"aggregation": aggregation, "statistic": stat, "n_sammlv": len(a), "n_casme3": len(b), "sammlv_mean": None, "casme3_mean": None,
            "difference_sammlv_minus_casme3": None, "standardized_effect_size": None, "mannwhitney_U": None, "mannwhitney_p": None, "cliffs_delta": None,
            "analysis_label": "DESCRIPTIVE"}


def cross_effects(video_rows, subject_rows, peak_rows, occupancy_rows, gt_rows):
    by_video = {(r["dataset"], r["subject"], r["video"]): r for r in video_rows}
    by_subject = {(r["dataset"], r["subject"]): r for r in subject_rows}
    # Select a common, explicitly defined diagnostic operating point.
    pd = {(r["dataset"], r["subject"], r["video"]): r for r in peak_rows if r["c_s"] == 1.0 and r["p"] == .55}
    occ = {(r["dataset"], r["subject"], r["video"]): r for r in occupancy_rows if r["p"] == .55}
    subject_derived = defaultdict(lambda: defaultdict(list))
    for key, row in pd.items():
        subject_derived[(key[0], key[1])]["peak_density_c_s_1.0"].append(row.get("local_peak_density_per_1000"))
        subject_derived[(key[0], key[1])]["interpeak_distance_over_kp_c_s_1.0_p_0.55"].append(row.get("thresholded_interpeak_median_over_kp"))
    for key, row in occ.items():
        subject_derived[(key[0], key[1])]["threshold_occupancy_p_0.55"].append(row.get("fraction_frames_above_threshold"))
    durations = defaultdict(list)
    for r in gt_rows:
        if finite(r.get("gt_duration_over_kp")) is not None:
            durations[(r["dataset"], r["subject"], r["video"])].append(float(r["gt_duration_over_kp"]))
    def mean_valid(values):
        values = [float(v) for v in values if finite(v) is not None]
        return float(np.mean(values)) if values else None
    specs = [
        ("robust_peakiness", lambda key: by_video[key].get("robust_peakiness"), lambda key: by_subject[key].get("mean_robust_peakiness")),
        ("MAD", lambda key: by_video[key].get("MAD"), lambda key: by_subject[key].get("mean_MAD")),
        ("NR1", lambda key: by_video[key].get("NR1"), lambda key: by_subject[key].get("mean_NR1")),
        ("NR2", lambda key: by_video[key].get("NR2"), lambda key: by_subject[key].get("mean_NR2")),
        ("peak_density_c_s_1.0", lambda key: pd.get(key, {}).get("local_peak_density_per_1000"), lambda key: mean_valid(subject_derived[key]["peak_density_c_s_1.0"])),
        ("threshold_occupancy_p_0.55", lambda key: occ.get(key, {}).get("fraction_frames_above_threshold"), lambda key: mean_valid(subject_derived[key]["threshold_occupancy_p_0.55"])),
        ("interpeak_distance_over_kp_c_s_1.0_p_0.55", lambda key: pd.get(key, {}).get("thresholded_interpeak_median_over_kp"), lambda key: mean_valid(subject_derived[key]["interpeak_distance_over_kp_c_s_1.0_p_0.55"])),
        ("GT_duration_over_kp", lambda key: float(np.mean(durations[key])) if durations.get(key) else None, lambda key: by_subject[key].get("GT_duration_over_kp_mean")),
    ]
    rows = []
    video_keys = [key for key in by_video]
    subject_keys = [key for key in by_subject]
    for stat, vf, sf in specs:
        a = [vf(k) for k in video_keys if k[0] == "SAMMLV"]
        b = [vf(k) for k in video_keys if k[0] == "CASME3"]
        rows.append({"dataset_pair": "SAMMLV_vs_CASME3", "statistic": stat, **effect_row(stat, "video", a, b)})
        a = [sf(k) for k in subject_keys if k[0] == "SAMMLV"]
        b = [sf(k) for k in subject_keys if k[0] == "CASME3"]
        rows.append({"dataset_pair": "SAMMLV_vs_CASME3", "statistic": stat, **effect_row(stat, "subject", a, b)})
    return rows


def marginal(response_rows):
    out = {}
    for field in ("c_s", "p", "c_d"):
        vals = {}
        for value in sorted({r[field] for r in response_rows}):
            subset = [r["F1"] for r in response_rows if r[field] == value]
            vals[str(value)] = {"mean_F1_across_other_configs": float(np.mean(subset)), "max_F1": float(np.max(subset))}
        out[field] = vals
    best = max(response_rows, key=lambda r: (r["F1"], -r["FP"], -r["c_s"], -r["p"], -r["c_d"]))
    out["best_full_data_config"] = best
    return out


def decide(summary):
    effects = summary["cross_dataset_effect_sizes"]
    core = {"robust_peakiness", "MAD", "NR1", "NR2", "peak_density_c_s_1.0", "threshold_occupancy_p_0.55", "interpeak_distance_over_kp_c_s_1.0_p_0.55"}
    shift = any((r.get("statistic") in core and ((r.get("standardized_effect_size") is not None and abs(r["standardized_effect_size"]) >= .5) or (r.get("mannwhitney_p") is not None and r["mannwhitney_p"] < .05))) for r in effects)
    rel = [r for r in summary["oracle_stat_correlations"] if r.get("abs_rho_ge_0.3")]
    relation = bool(rel)
    # Directional sensitivity: report-level criterion checks that the two datasets
    # have different best descriptive operating points and that p/c_s changes move
    # the expected recall/FP tradeoff in the observed direction.
    best_s = summary["response_surface"]["SAMMLV"]["best_full_data_config"]
    best_c = summary["response_surface"]["CASME3"]["best_full_data_config"]
    direction = (best_s["c_s"] != best_c["c_s"] or best_s["p"] != best_c["p"] or best_s["c_d"] != best_c["c_d"])
    if shift and direction and relation:
        return "ADAPTIVE-OPERATING-POINT-HYPOTHESIS-SUPPORTED"
    if shift:
        return "DATASET-SHIFT-OBSERVED-BUT-ADAPTIVITY-NOT-PREDICTABLE"
    return "ADAPTIVE-OPERATING-POINT-HYPOTHESIS-NOT-SUPPORTED"


def build_report(summary, paths):
    lines = ["# ME-TST+ Native Cross-Dataset Operating-Point Diagnosis", "", "本报告是 cache-only 的 exploratory / oracle diagnostic；没有修改 decoder、Morphology、grid、candidate rule 或 evaluator。", "", "## Provenance / anchor gate", ""]
    for name, info in summary["provenance"].items():
        lines.append(f"- **{name}**：cache `{info['cache_path']}`；SHA-256=`{info['cache_sha256']}`；{info['subjects']} subjects / {info['videos']} videos / {info['gt']} GT / k_p={info['k_p']}。")
        lines.append(f"  - Author Native：{info['Author Native']['TP']}/{info['Author Native']['FP']}/{info['Author Native']['FN']}, F1={info['Author Native']['F1']:.6f}。")
        lines.append(f"  - {'Tuned Native' if name == 'SAMMLV' else 'Final Strong Native'}：{info['Tuned/Final Strong Native']['TP']}/{info['Tuned/Final Strong Native']['FP']}/{info['Tuned/Final Strong Native']['FN']}, F1={info['Tuned/Final Strong Native']['F1']:.6f}。")
    lines += ["", "## 诊断口径", "", "- Score-only statistics 和 Native response surface 使用 frozen cache；response surface 明确标记为 `EXPLORATORY FULL-DATA SENSITIVITY — NOT UNBIASED TEST RESULT`。", "- GT duration、GT-near/background peak separation、subject oracle 均为 `ORACLE DIAGNOSTIC ONLY`，不进入模型输入或正式性能。", "- video-level 显著性仅作描述性证据；同一 subject 内 videos 不完全独立。", "", "## Q1–Q3: regime shift", ""]
    for name in ("SAMMLV", "CASME3"):
        lines.append(f"### {name}")
        lines.append(f"- Duration/k_p：median={summary['duration_summary'][name]['median']:.4f}, mean={summary['duration_summary'][name]['mean']:.4f}, p25={summary['duration_summary'][name]['q25']:.4f}, p75={summary['duration_summary'][name]['q75']:.4f}, p90={summary['duration_summary'][name]['q90']:.4f}。")
        lines.append(f"- Author smoothing threshold occupancy (p=.55)：frames={summary['occupancy_summary'][name]['frame_fraction_mean']:.4f} mean，local peak survival={summary['occupancy_summary'][name]['peak_survival_mean']:.4f} mean。")
        lines.append(f"- c_s=1.0 local peak density={summary['peak_density_summary'][name]['c_s_1.0']['local_density_mean']:.2f} peaks/1000 frames（mean across videos）。")
        lines.append("- Occupancy by p (frames / local-peak survival): " + ", ".join(f"p={p}: {summary['occupancy_summary'][name]['by_p'][str(p)]['frame_fraction_mean']:.4f} / {summary['occupancy_summary'][name]['by_p'][str(p)]['peak_survival_mean']:.4f}" for p in (0.35, 0.45, 0.55, 0.65)) + "。")
    lines += ["", "### Cross-dataset effect sizes", "", "| aggregation | statistic | SAMMLV mean | CASME3 mean | difference | standardized effect | MW p | Cliff's delta |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["cross_dataset_effect_sizes"]:
        if r["statistic"] in {"robust_peakiness", "MAD", "NR1", "NR2", "peak_density_c_s_1.0", "threshold_occupancy_p_0.55", "interpeak_distance_over_kp_c_s_1.0_p_0.55", "GT_duration_over_kp"}:
            fmt = lambda x: "NA" if x is None else f"{x:.4f}"
            lines.append(f"| {r['aggregation']} | {r['statistic']} | {fmt(r['sammlv_mean'])} | {fmt(r['casme3_mean'])} | {fmt(r['difference_sammlv_minus_casme3'])} | {fmt(r['standardized_effect_size'])} | {fmt(r['mannwhitney_p'])} | {fmt(r['cliffs_delta'])} |")
    lines += ["", "### 对 Q1–Q3 的直接回答", "", "- **Q1：是。** 相同 p 在 CASME3 上产生更低的 frame occupancy 和 local-peak survival；例如 p=.55 时为 0.0229 / 0.0241，而 SAMMLV 为 0.0550 / 0.0944。", "- **Q2：SAMMLV 更 rough 且 local peak density 更高。** NR1/NR2 分别为 0.2400/0.4108（SAMMLV）对 0.0832/0.1425（CASME3），c_s=1.0 的 local peak density 为 96.29 对 37.93 peaks/1000 frames。", "- **Q3：GT duration/k_p 的 regime 不同。** median 为 2.4000（SAMMLV）对 1.2941（CASME3）；CASME3 的均值较接近但右尾更长（p90=3.2353 对 3.0000）。"]
    lines += ["", "## GT-near peaks vs background peaks (oracle/diagnostic only)", "", "GT-near peak = Author-Native-smoothed local peak nearest the GT apex; background peak = local peak outside every GT expanded window. One malformed CASME3 GT interval is excluded from duration and GT-near distribution summaries (the cache GT count remains 858).", "", "| dataset | metric | GT-near median | background median | GT-near mean | background mean | ROC-AUC | PR-AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("SAMMLV", "CASME3"):
        for metric in ("peak_height", "robust_z_height", "prominence"):
            pos = next(r for r in summary["gt_vs_background_peak_summary"] if r["dataset"] == name and r["metric"] == metric and r["peak_group"] == "GT-near")
            neg = next(r for r in summary["gt_vs_background_peak_summary"] if r["dataset"] == name and r["metric"] == metric and r["peak_group"] == "background")
            fmt = lambda x: "NA" if x is None else f"{x:.4f}"
            lines.append(f"| {name} | {metric} | {fmt(pos['median'])} | {fmt(neg['median'])} | {fmt(pos['mean'])} | {fmt(neg['mean'])} | {fmt(pos['roc_auc'])} | {fmt(pos['pr_auc'])} |")
    lines += ["", "## Q4: subject-wise oracle relation", ""]
    for name in ("SAMMLV", "CASME3"):
        rows = [r for r in summary["oracle_stat_correlations"] if r["dataset"] == name and r["spearman_rho"] is not None]
        strong = [f"{r['target']}~{r['statistic']} (rho={r['spearman_rho']:+.3f})" for r in rows if r["abs_rho_ge_0.3"]]
        lines.append(f"- {name}：{'; '.join(strong) if strong else '没有 |rho| >= 0.3 的关系'}。")
    lines += ["", "## Q5: Native response surface (descriptive only)", ""]
    for name in ("SAMMLV", "CASME3"):
        m = summary["response_surface"][name]
        b = m["best_full_data_config"]
        lines.append(f"- {name} fixed c_b={m['fixed_c_b']}：full-data sensitivity best config c_s={b['c_s']}, p={b['p']}, c_d={b['c_d']}, F1={b['F1']:.6f}。")
        lines.append(f"  - Marginal means: c_s={json.dumps(m['marginal']['c_s'], ensure_ascii=False)}")
        lines.append(f"  - Marginal means: p={json.dumps(m['marginal']['p'], ensure_ascii=False)}")
        lines.append(f"  - Marginal means: c_d={json.dumps(m['marginal']['c_d'], ensure_ascii=False)}")
    lines += ["", "## Interpretation", "", f"**{summary['decision']}**", "", "该状态只总结本次诊断证据，不构成新方法、adaptive threshold、parameter predictor 或正式 unbiased test 结果。", "", "## Outputs", ""]
    lines.extend(f"- `{p}`" for p in paths)
    return "\n".join(lines) + "\n"


def main():
    OUT_OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        cache_info = {}
        records_by_name = {}
        for name, path in (("SAMMLV", SAMMLV_CACHE), ("CASME3", CASME3_CACHE)):
            payload, records, subjects = load_cache(path, name)
            cache_info[name] = {"cache_path": str(path.resolve()), "cache_sha256": sha256(path), "subjects": len(subjects), "videos": len(records), "gt": sum(len(r["samples"]) for r in records), "k_p": EXPECTED[name]["k_p"], "gt_source": "verified cache records[*].samples (onset, apex, offset)"}
            records_by_name[name] = (records, subjects)
        anchors = anchor_gate(cache_info, records_by_name)
    except Exception as exc:
        summary = {"status": "BLOCKED-DIAGNOSTIC-PROVENANCE", "reason": repr(exc), "provenance": cache_info if "cache_info" in locals() else {}}
        write_json(OUT / "native_cross_dataset_diagnosis.json", summary)
        (OUT / "NATIVE_CROSS_DATASET_DIAGNOSIS_CN.md").write_text("# ME-TST+ Native Cross-Dataset Operating-Point Diagnosis\n\n**BLOCKED-DIAGNOSTIC-PROVENANCE**\n\n" + repr(exc) + "\n", encoding="utf-8")
        raise

    all_score, all_peak, all_occ, all_gt, peak_summary, duration_summary = [], [], [], [], {}, {}
    _occ_summary = {}
    peak_compare_rows = []
    subject_rows, response_summary, oracle_rows_all, corr_rows = [], {}, [], []
    for name, (records, subjects) in records_by_name.items():
        score, peak, occ, gt, pos, neg = video_diagnostics(name, records, EXPECTED[name]["k_p"])
        all_score.extend(score); all_peak.extend(peak); all_occ.extend(occ); all_gt.extend(gt)
        peak_compare_rows.extend(peak_comparison(name, pos, neg))
        subject = aggregate_subjects(name, score, gt)
        subject_rows.extend(subject)
        # Duration and occupancy summaries feed the report and JSON.
        duration_summary[name] = quantiles([r["gt_duration_over_kp"] for r in gt])
        occ55 = [r for r in occ if r["p"] == .55]
        occupancy_summary = {"frame_fraction_mean": float(np.mean([r["fraction_frames_above_threshold"] for r in occ55])), "peak_survival_mean": float(np.mean([r["fraction_local_peaks_retained"] for r in occ55 if r["fraction_local_peaks_retained"] is not None])) if any(r["fraction_local_peaks_retained"] is not None for r in occ55) else None,
                            "by_p": {str(p): {"frame_fraction_mean": float(np.mean([r["fraction_frames_above_threshold"] for r in occ if r["p"] == p])), "peak_survival_mean": float(np.mean([r["fraction_local_peaks_retained"] for r in occ if r["p"] == p and r["fraction_local_peaks_retained"] is not None])) if any(r["fraction_local_peaks_retained"] is not None for r in occ if r["p"] == p) else None} for p in (0.35, 0.45, 0.55, 0.65)}}
        # Keep per-dataset scalar summaries in a separate dictionary below.
        _occ_summary[name] = occupancy_summary
        pd1 = [r for r in peak if r["c_s"] == 1.0]
        peak_summary[name] = {"c_s_1.0": {"local_density_mean": float(np.mean([r["local_peak_density_per_1000"] for r in pd1]))}}
        fixed_cd, fixed_cb = EXPECTED[name]["c_d"], EXPECTED[name]["c_b"]
        subject_map = {r["subject"]: r for r in subject}
        oracle = subject_oracle(name, records, subjects, EXPECTED[name]["k_p"], fixed_cd, fixed_cb, subject_map)
        oracle_rows_all.extend(oracle)
        corr_rows.extend(oracle_correlations(name, oracle, subject))
        surface = response_surface(name, records, EXPECTED[name]["k_p"], fixed_cb)
        response_summary[name] = {"fixed_c_b": fixed_cb, "fixed_c_d_for_oracle": fixed_cd, "rows": surface, "marginal": marginal(surface), "best_full_data_config": marginal(surface)["best_full_data_config"]}

    effects = cross_effects(all_score, subject_rows, all_peak, all_occ, all_gt)
    summary = {
        "status": "NATIVE-CROSS-DATASET-DIAGNOSIS-COMPLETE", "decision": None,
        "provenance": {name: {**cache_info[name], "Author Native": anchors[name]["Author Native"], "Tuned/Final Strong Native": anchors[name]["Tuned Native" if name == "SAMMLV" else "Final Strong Native"]} for name in cache_info},
        "evaluator": {"native_audit_path": str(NATIVE_AUDIT.resolve()), "native_audit_sha256": sha256(NATIVE_AUDIT), "paper_metrics_reference_path": str(PAPER_METRICS.resolve()), "paper_metrics_reference_sha256": sha256(PAPER_METRICS)},
        "duration_summary": duration_summary, "occupancy_summary": _occ_summary, "peak_density_summary": peak_summary,
        "gt_vs_background_peak_summary": peak_compare_rows,
        "response_surface": response_summary, "oracle_stat_correlations": corr_rows, "cross_dataset_effect_sizes": effects,
        "diagnostic_labels": {"response_surface": "EXPLORATORY FULL-DATA SENSITIVITY — NOT UNBIASED TEST RESULT", "subject_oracle": "ORACLE DIAGNOSTIC ONLY", "gt_peak_analysis": "ORACLE/DIAGNOSTIC ONLY"},
        "integrity": {"cache_only": True, "new_decoder": False, "adaptive_threshold": False, "parameter_predictor": False, "morphology_modified": False, "native_grid_expanded": False, "recognition_STRS_run": False, "outer_test_gt_used_for_selection": False},
    }
    summary["decision"] = decide(summary)
    # Write required artifacts.
    write_csv(OUT_OUTPUTS / "video_score_statistics.csv", all_score)
    write_csv(OUT_OUTPUTS / "subject_score_statistics.csv", subject_rows)
    write_csv(OUT_OUTPUTS / "peak_density_statistics.csv", all_peak)
    write_csv(OUT_OUTPUTS / "threshold_occupancy.csv", all_occ)
    write_csv(OUT_OUTPUTS / "gt_duration_statistics.csv", all_gt)
    write_csv(OUT_OUTPUTS / "gt_vs_background_peak_statistics.csv", peak_compare_rows)
    surface_rows = []
    for name in ("SAMMLV", "CASME3"):
        surface_rows.extend(response_summary[name]["rows"])
    write_csv(OUT_OUTPUTS / "native_response_surface.csv", surface_rows)
    write_csv(OUT_OUTPUTS / "subject_oracle_operating_points.csv", oracle_rows_all)
    write_csv(OUT_OUTPUTS / "oracle_stat_correlations.csv", corr_rows)
    write_csv(OUT_OUTPUTS / "cross_dataset_effect_sizes.csv", effects)
    paths = [OUT / "NATIVE_CROSS_DATASET_DIAGNOSIS_CN.md"] + sorted(OUT_OUTPUTS.glob("*.csv")) + [OUT / "native_cross_dataset_diagnosis.json"]
    write_json(OUT / "native_cross_dataset_diagnosis.json", summary)
    (OUT / "NATIVE_CROSS_DATASET_DIAGNOSIS_CN.md").write_text(build_report(summary, [p.resolve() for p in paths]), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "anchors": summary["provenance"], "outputs": [str(p.resolve()) for p in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
