#!/usr/bin/env python3
"""Locked Hidden Representation Event-Localization Predictability Audit.

This is an inference-cache-only diagnostic. It never imports or runs ME-TST.
It uses fresh hidden/score NPZ files, fixed native peaks, one 41-D cosine
trajectory, and strict subject-LOSO Ridge(alpha=1.0).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


P = 0.55
K_P = 5
IOU_THRESHOLD = 0.5
PATCH_POINTS = 41
RIDGE_ALPHA = 1.0
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 20260904
EXPECTED = {"subjects": 29, "videos": 79, "total_T": 44295, "hidden_dim": 384,
            "native": (53, 184, 106, 237)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-root", type=Path, required=True)
    p.add_argument("--compact-cache", type=Path, required=True)
    p.add_argument("--failure-gt-trace", type=Path, required=True)
    p.add_argument("--failure-prediction-trace", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def smooth(score):
    return np.convolve(np.asarray(score, dtype=np.float64), np.ones(2 * K_P) / (2 * K_P), mode="same")


def native_decode(score):
    curve = smooth(score)
    threshold = float(curve.mean() + P * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=K_P)[0].astype(int)
    return curve, threshold, peaks


def iou(interval, sample):
    left, right = map(int, interval)
    gt_left, gt_right = int(sample[0]), int(sample[2])
    if right < left or gt_right < gt_left:
        return 0.0
    inter = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return inter / union if union else 0.0


def greedy_match(peaks, samples):
    unmatched = set(range(len(samples)))
    matches = []
    for peak in peaks:
        choices = [(iou((peak - K_P, peak + K_P), samples[j]), j) for j in unmatched]
        best_iou, best_j = max(choices, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched.remove(best_j)
            matches.append(best_j)
        else:
            matches.append(-1)
    return matches, unmatched


def valid_gt(sample, length):
    onset, _, offset = map(int, sample[:3])
    return 0 <= onset <= offset < length


def resample_scalar(values, peak):
    coords = np.linspace(peak - 2 * K_P, peak + 2 * K_P, PATCH_POINTS)
    coords = np.clip(coords, 0, len(values) - 1)
    return np.interp(coords, np.arange(len(values)), values)


def score_patch(curve, peak):
    values = resample_scalar(curve, peak)
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo + 1e-12)


def hidden_cosine_patch(hidden, peak):
    coords = np.linspace(peak - 2 * K_P, peak + 2 * K_P, PATCH_POINTS)
    coords = np.clip(coords, 0, len(hidden) - 1)
    grid = np.arange(len(hidden))
    sampled = np.column_stack([np.interp(coords, grid, hidden[:, d]) for d in range(hidden.shape[1])])
    anchor = np.asarray(hidden[peak], dtype=np.float64)
    denom = np.linalg.norm(sampled, axis=1) * np.linalg.norm(anchor)
    return np.divide(sampled @ anchor, denom, out=np.zeros(PATCH_POINTS), where=denom > 1e-12)


def patch_is_valid(valid_mask, peak):
    # The same association rows are used by Native, Score, and Hidden. A row is
    # eligible only if all real indices touched by the ±2kp patch are valid.
    left, right = max(0, peak - 2 * K_P), min(len(valid_mask) - 1, peak + 2 * K_P)
    return bool(valid_mask[peak]) and bool(np.asarray(valid_mask[left:right + 1]).all())


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {k: None for k in ["mean", "median", "q25", "q75", "IQR", "p10", "p90", "min", "max"]} | {"n": 0}
    q10, q25, q75, q90 = np.quantile(a, [0.10, 0.25, 0.75, 0.90])
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "q25": float(q25), "q75": float(q75), "IQR": float(q75 - q25),
            "p10": float(q10), "p90": float(q90), "min": float(a.min()), "max": float(a.max())}


def load_and_audit_assets(hidden_root, compact_cache):
    manifest_path = hidden_root / "sammlv_full_hidden_dump_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with compact_cache.open("rb") as f:
        payload = pickle.load(f)
    records = {(str(r["subject"]), str(r["video"])): r for r in payload["records"]}
    npz_paths = sorted(hidden_root.glob("subject_*/*.npz"))
    fresh = {}
    tree = hashlib.sha256()
    total_t = invalid_positions = 0
    file_hashes = {}
    for path in npz_paths:
        digest = sha256(path)
        rel = str(path.relative_to(hidden_root))
        file_hashes[rel] = digest
        tree.update(f"{rel}:{digest}\n".encode())
        z = np.load(path, allow_pickle=False)
        if set(z.files) != {"hidden", "score", "valid_mask", "subject", "video_id"}:
            raise RuntimeError(f"BLOCKED-HIDDEN-ASSET fields: {path}")
        hidden, score, valid = z["hidden"], z["score"], z["valid_mask"].astype(bool)
        subject, video = str(z["subject"].item()), str(z["video_id"].item())
        if hidden.shape != (len(score), EXPECTED["hidden_dim"]) or valid.shape != score.shape:
            raise RuntimeError(f"BLOCKED-HIDDEN-ASSET shape: {path}")
        if (subject, video) not in records or len(records[(subject, video)]["score"]) != len(score):
            raise RuntimeError(f"BLOCKED-HIDDEN-ASSET mapping: {subject}/{video}")
        fresh[(subject, video)] = {"hidden": hidden.astype(np.float64), "score": score.astype(np.float64),
                                   "valid_mask": valid, "path": str(path.resolve()), "sha256": digest}
        total_t += len(score)
        invalid_positions += int((~valid).sum())
    observed = (len({s for s, _ in fresh}), len(fresh), total_t)
    expected = (EXPECTED["subjects"], EXPECTED["videos"], EXPECTED["total_T"])
    if observed != expected or manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"BLOCKED-HIDDEN-ASSET observed={observed}, expected={expected}")
    checkpoints = {}
    for row in manifest.get("records", []):
        checkpoints[str(row.get("subject"))] = {"path": row.get("checkpoint_path"), "sha256": row.get("checkpoint_sha256")}
    inventory = {
        "hidden_root": str(hidden_root.resolve()), "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path), "hidden_tree_sha256": tree.hexdigest(),
        "file_count": len(npz_paths), "subjects": observed[0], "videos": observed[1], "total_T": total_t,
        "hidden_dim": EXPECTED["hidden_dim"], "invalid_positions": invalid_positions,
        "input_cache_path_recorded": manifest.get("input_cache_path"),
        "input_cache_sha256_recorded": manifest.get("input_cache_sha256"),
        "batch_size": manifest.get("batch_size"), "window_length": manifest.get("window_length"),
        "model_eval": manifest.get("model_eval"), "torch_no_grad": manifest.get("torch_no_grad"),
        "inference_only": manifest.get("inference_only"), "checkpoints": checkpoints,
        "npz_sha256": file_hashes,
    }
    return payload, records, fresh, inventory


def build_associations(payload, records, fresh, gt_trace, pred_trace):
    old_pred = {r["prediction_id"]: r for r in pred_trace if r["dataset"] == "SAMMLV"}
    old_peak_to_id = {(r["subject"], r["video"], int(r["peak_time"])): r["prediction_id"]
                      for r in pred_trace if r["dataset"] == "SAMMLV"}
    sources = defaultdict(set)
    features = {}
    totals = Counter()
    peak_differences = []
    all_candidate_validity = Counter()
    for key, record in records.items():
        subject, video = key
        item = fresh[key]
        curve, threshold, peaks = native_decode(item["score"])
        matches, unmatched = greedy_match(peaks, record["samples"])
        totals.update(tp=sum(x >= 0 for x in matches), fp=sum(x < 0 for x in matches), fn=len(unmatched), peaks=len(peaks))
        for pred_index, (peak, match) in enumerate(zip(peaks, matches)):
            prediction_id = f"SAMMLV:{subject}:{video}:{pred_index}"
            trace = old_pred.get(prediction_id)
            if trace is None:
                raise AssertionError(f"missing historical deterministic candidate id {prediction_id}")
            old_peak = int(trace["peak_time"])
            if old_peak != int(peak):
                peak_differences.append({"prediction_id": prediction_id, "historical_peak": old_peak, "fresh_peak": int(peak)})
            valid = patch_is_valid(item["valid_mask"], int(peak))
            all_candidate_validity["eligible" if valid else "invalid_patch"] += 1
            features[prediction_id] = {
                "subject": subject, "video": video, "prediction_id": prediction_id,
                "peak": int(peak), "historical_peak": old_peak, "video_length": len(item["score"]),
                "threshold": threshold, "peak_height": float(curve[peak]), "patch_valid": valid,
                "score_patch": score_patch(curve, int(peak)),
                "hidden_patch": hidden_cosine_patch(item["hidden"], int(peak)),
                "hidden_npz_path": item["path"], "hidden_npz_sha256": item["sha256"],
            }
            if match >= 0:
                sources[(prediction_id, int(match))].add("A_NATIVE_TP")
    observed = (totals["tp"], totals["fp"], totals["fn"], totals["peaks"])
    if observed != EXPECTED["native"]:
        raise RuntimeError(f"BLOCKED-BASELINE fresh native={observed}, expected={EXPECTED['native']}")

    for row in gt_trace:
        if row["dataset"] == "SAMMLV" and row["fn_category"] == "FN-C_BOUNDARY":
            pid = old_peak_to_id.get((row["subject"], row["video"], int(row["diagnostic_peak"])))
            if pid is None:
                raise AssertionError(f"FN-C peak not in fixed candidate trace: {row}")
            sources[(pid, int(row["gt_index"]))].add("B_FNC")
    for row in pred_trace:
        if row["dataset"] == "SAMMLV" and row["fp_category"] == "FP-B_NEAR_GT_LOCALIZATION":
            sources[(row["prediction_id"], int(row["closest_gt_id"]))].add("B_NEAR_FP")

    rows, excluded = [], Counter()
    for (pid, gt_index), src in sorted(sources.items()):
        feat = features[pid]
        sample = records[(feat["subject"], feat["video"])]["samples"][gt_index]
        if not valid_gt(sample, feat["video_length"]):
            excluded["invalid_gt"] += 1
            continue
        if not feat["patch_valid"]:
            excluded["invalid_hidden_patch"] += 1
            continue
        onset, apex, offset = map(int, sample[:3])
        rows.append({**feat, "gt_index": gt_index, "gt_onset": onset, "gt_apex": apex, "gt_offset": offset,
                     "gt_duration": offset - onset + 1, "y_on": onset - feat["peak"], "y_off": offset - feat["peak"],
                     "r_on": (onset - feat["peak"]) / K_P, "r_off": (offset - feat["peak"]) / K_P,
                     "cohort": "A_NATIVE_TP" if "A_NATIVE_TP" in src else "B_LOCALIZATION",
                     "is_native_tp": "A_NATIVE_TP" in src, "is_fnc": "B_FNC" in src,
                     "is_near_fp": "B_NEAR_FP" in src, "association_sources": "+".join(sorted(src))})
    candidate_targets = defaultdict(set)
    for row in rows:
        candidate_targets[row["prediction_id"]].add(row["gt_index"])
    audit = {"fresh_native_TP_FP_FN_peaks": list(observed), "candidate_count": totals["peaks"],
             "candidate_patch_validity": dict(all_candidate_validity), "association_rows": len(rows),
             "unique_associated_candidates": len(candidate_targets),
             "multi_target_candidate_count": sum(len(v) > 1 for v in candidate_targets.values()),
             "excluded_associations": dict(excluded), "fresh_vs_historical_peak_differences": peak_differences}
    return rows, audit


def loso(rows, feature_key):
    x = np.asarray([r[feature_key] for r in rows], dtype=np.float64)
    y = np.asarray([[r["r_on"], r["r_off"]] for r in rows], dtype=np.float64)
    subjects = np.asarray([r["subject"] for r in rows])
    out = np.full_like(y, np.nan)
    for held in sorted(set(subjects)):
        test = subjects == held
        train = ~test
        if not train.any() or not test.any():
            raise RuntimeError(f"invalid LOSO fold {held}")
        scaler = StandardScaler().fit(x[train])
        x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
        for side in range(2):
            out[test, side] = Ridge(alpha=RIDGE_ALPHA).fit(x_train, y[train, side]).predict(x_test)
    if not np.isfinite(out).all():
        raise AssertionError("non-finite OOF predictions")
    return out


def event_from_r(row, r):
    onset = int(np.clip(np.rint(row["peak"] + K_P * float(r[0])), 0, row["video_length"] - 1))
    offset = int(np.clip(np.rint(row["peak"] + K_P * float(r[1])), 0, row["video_length"] - 1))
    return onset, offset


def attach_predictions(rows):
    score_oof = loso(rows, "score_patch")
    hidden_oof = loso(rows, "hidden_patch")
    output = []
    for row, score_r, hidden_r in zip(rows, score_oof, hidden_oof):
        gt = (row["gt_onset"], row["gt_apex"], row["gt_offset"])
        clean = {k: v for k, v in row.items() if k not in {"score_patch", "hidden_patch"}}
        clean["loso_held_out_subject"] = row["subject"]
        for name, pred_r in [("native", np.asarray([-1.0, 1.0])), ("score", score_r), ("hidden", hidden_r)]:
            interval = event_from_r(row, pred_r)
            clean[f"{name}_pred_r_on"] = float(pred_r[0])
            clean[f"{name}_pred_r_off"] = float(pred_r[1])
            clean[f"{name}_pred_y_on"] = float(K_P * pred_r[0])
            clean[f"{name}_pred_y_off"] = float(K_P * pred_r[1])
            clean[f"{name}_onset"] = interval[0]
            clean[f"{name}_offset"] = interval[1]
            clean[f"{name}_interval_valid"] = interval[0] <= interval[1]
            clean[f"{name}_iou"] = iou(interval, gt)
            clean[f"{name}_onset_AE"] = abs(K_P * float(pred_r[0]) - row["y_on"])
            clean[f"{name}_offset_AE"] = abs(K_P * float(pred_r[1]) - row["y_off"])
        output.append(clean)
    return output


def predictor_metrics(rows, name):
    ious = np.asarray([r[f"{name}_iou"] for r in rows])
    return {"n": len(rows), "onset_AE": distribution([r[f"{name}_onset_AE"] for r in rows]),
            "offset_AE": distribution([r[f"{name}_offset_AE"] for r in rows]), "IoU": distribution(ious),
            "fraction_IoU_ge_0.3": float(np.mean(ious >= .3)), "fraction_IoU_ge_0.5": float(np.mean(ious >= .5)),
            "fraction_IoU_ge_0.7": float(np.mean(ious >= .7)),
            "invalid_interval_count": sum(not r[f"{name}_interval_valid"] for r in rows),
            "invalid_interval_fraction": float(np.mean([not r[f"{name}_interval_valid"] for r in rows]))}


def paired(rows, a, b):
    delta = np.asarray([r[f"{a}_iou"] - r[f"{b}_iou"] for r in rows])
    return {"delta_IoU": distribution(delta), "improved": int((delta > 1e-12).sum()),
            "equal": int((np.abs(delta) <= 1e-12).sum()), "worse": int((delta < -1e-12).sum()),
            "fraction_improved": float(np.mean(delta > 1e-12)), "fraction_equal": float(np.mean(np.abs(delta) <= 1e-12)),
            "fraction_worse": float(np.mean(delta < -1e-12))}


def summarize_cohort(rows):
    return {"n": len(rows),
            "targets": {"y_on": distribution([r["y_on"] for r in rows]), "y_off": distribution([r["y_off"] for r in rows]),
                        "r_on": distribution([r["r_on"] for r in rows]), "r_off": distribution([r["r_off"] for r in rows])},
            "predictors": {name: predictor_metrics(rows, name) for name in ["native", "score", "hidden"]},
            "paired": {"Hidden_minus_Native": paired(rows, "hidden", "native"),
                       "Hidden_minus_Score": paired(rows, "hidden", "score")}}


def bootstrap(rows, cohort, a="hidden", b="native"):
    groups = defaultdict(list)
    for r in rows:
        groups[r["subject"]].append(r[f"{a}_iou"] - r[f"{b}_iou"])
    subjects = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        values = [v for s in sampled for v in groups[s]]
        draws.append(float(np.mean(values)))
    lo, hi = np.quantile(draws, [.025, .975])
    return {"cohort": cohort, "comparison": f"{a}_minus_{b}", "unit": "subject",
            "repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED,
            "mean_delta": float(np.mean(draws)), "CI95": [float(lo), float(hi)]}


def subject_metrics(rows):
    output = []
    for subject in sorted({r["subject"] for r in rows}):
        for cohort in ["A_NATIVE_TP", "B_LOCALIZATION", "B_FNC", "B_NEAR_FP", "COMBINED"]:
            subset = [r for r in rows if r["subject"] == subject and
                      (cohort == "COMBINED" or r["cohort"] == cohort or
                       (cohort == "B_FNC" and r["is_fnc"]) or (cohort == "B_NEAR_FP" and r["is_near_fp"]))]
            if not subset:
                continue
            output.append({"subject": subject, "cohort": cohort, "n": len(subset),
                           "native_mean_iou": np.mean([r["native_iou"] for r in subset]),
                           "score_mean_iou": np.mean([r["score_iou"] for r in subset]),
                           "hidden_mean_iou": np.mean([r["hidden_iou"] for r in subset]),
                           "hidden_minus_native_mean_iou": np.mean([r["hidden_iou"] - r["native_iou"] for r in subset]),
                           "hidden_minus_score_mean_iou": np.mean([r["hidden_iou"] - r["score_iou"] for r in subset])})
    return output


def fmt(x):
    return "N/A" if x is None else f"{x:.6f}" if isinstance(x, float) else str(x)


def make_report(result):
    inv, audit = result["asset_audit"], result["candidate_association_audit"]
    lines = ["# Hidden Representation Event-Localization Predictability Audit", "",
             f"最终状态：**{result['status']}**", "",
             "## 1. 冻结资产与协议", "",
             f"- hidden source：`{inv['hidden_root']}`；tree SHA-256：`{inv['hidden_tree_sha256']}`。",
             f"- manifest：`{inv['manifest_path']}`；SHA-256：`{inv['manifest_sha256']}`。",
             f"- 完整性：{inv['subjects']} subjects / {inv['videos']} videos / total T={inv['total_T']} / hidden dim={inv['hidden_dim']}。",
             f"- frozen 记录：model.eval={inv['model_eval']}，torch.no_grad={inv['torch_no_grad']}，inference_only={inv['inference_only']}；batch={inv['batch_size']}，window={inv['window_length']}。",
             f"- valid_mask 无效位置：{inv['invalid_positions']}；涉及无效 hidden patch 的 association 同时从三种 predictor 排除。",
             "- 本审计没有导入或运行 ME-TST，没有训练或微调 backbone。唯一拟合为 Ridge(alpha=1.0)。", "",
             "## 2. Candidate 与 association", "",
             f"- fresh native TP/FP/FN/peaks：`{audit['fresh_native_TP_FP_FN_peaks']}`。",
             f"- eligible association rows：{audit['association_rows']}；unique candidates：{audit['unique_associated_candidates']}；multi-target candidates：{audit['multi_target_candidate_count']}。",
             f"- exclusions：`{audit['excluded_associations']}`。",
             f"- fresh/historical peak-list 数值差异：`{audit['fresh_vs_historical_peak_differences']}`；HREP 始终使用 fresh peak。",
             "- A 使用 fresh native greedy TP；B 沿用已锁定 FN-C diagnostic 与 FP-B closest-GT association。所有重复关联留在同一 subject LOSO fold。", "",
             "## 3. 固定表示与预测器", "",
             "- Hidden：`[p-2k_p,p+2k_p]` hidden 线性重采样至 41 点，计算每点与 `h_p` 的 cosine，得到唯一 41-D trajectory。",
             "- Score negative control：同源 fresh smoothed score 的 41 点 patch，逐 patch min-max normalization。",
             "- target：`r_on=(onset-p)/k_p`，`r_off=(offset-p)/k_p`；严格 subject-LOSO；scaler 仅 fit 训练 subjects。",
             "- 输出直接 `np.rint` 后 clip；不 swap、不 clamp duration、不强制包含 peak；onset>offset 记 invalid、IoU=0。", "",
             "## 4. Signed target 分布", "",
             "| Cohort | N | y_on mean/median/IQR/p10/p90/min/max | y_off mean/median/IQR/p10/p90/min/max |", "|---|---:|---:|---:|"]
    for cohort, summary in result["cohorts"].items():
        yon, yoff = summary["targets"]["y_on"], summary["targets"]["y_off"]
        lines.append(f"| {cohort} | {summary['n']} | {fmt(yon['mean'])}/{fmt(yon['median'])}/{fmt(yon['IQR'])}/{fmt(yon['p10'])}/{fmt(yon['p90'])}/{fmt(yon['min'])}/{fmt(yon['max'])} | {fmt(yoff['mean'])}/{fmt(yoff['median'])}/{fmt(yoff['IQR'])}/{fmt(yoff['p10'])}/{fmt(yoff['p90'])}/{fmt(yoff['min'])}/{fmt(yoff['max'])} |")
    lines += ["", "## 5. Cohort 结果", "",
             "| Cohort | N | Predictor | onset mean/median AE | offset mean/median AE | mean/median IoU | IoU≥.3/.5/.7 | invalid |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for cohort, summary in result["cohorts"].items():
        for name in ["native", "score", "hidden"]:
            m = summary["predictors"][name]
            lines.append(f"| {cohort} | {summary['n']} | {name} | {fmt(m['onset_AE']['mean'])}/{fmt(m['onset_AE']['median'])} | {fmt(m['offset_AE']['mean'])}/{fmt(m['offset_AE']['median'])} | {fmt(m['IoU']['mean'])}/{fmt(m['IoU']['median'])} | {fmt(m['fraction_IoU_ge_0.3'])}/{fmt(m['fraction_IoU_ge_0.5'])}/{fmt(m['fraction_IoU_ge_0.7'])} | {m['invalid_interval_count']} |")
    lines += ["", "## 6. Paired IoU", "", "| Cohort | Comparison | mean/median ΔIoU | improved/equal/worse |", "|---|---|---:|---:|"]
    for cohort, summary in result["cohorts"].items():
        for comp, m in summary["paired"].items():
            d = m["delta_IoU"]
            lines.append(f"| {cohort} | {comp} | {fmt(d['mean'])}/{fmt(d['median'])} | {m['improved']}/{m['equal']}/{m['worse']} |")
    s = result["safety"]
    lines += ["", "## 7. Safety / rescue", "",
              f"- Native TP preserved/lost：{s['native_TP_preserved']}/{s['native_TP_lost']}，preservation={fmt(s['native_TP_preservation_rate'])}。",
              f"- FN-C rescued/remained：{s['FN_C_rescued']}/{s['FN_C_remained']}。",
              f"- NetGain = FN-C rescued - Native TP lost = **{s['NetGain']}**。",
              f"- Near-FP converted/remained：{s['near_FP_converted']}/{s['near_FP_remained']}；与 FN-C 重合 pair={s['FNC_near_FP_overlap_pairs']}，未重复计入 NetGain。", "",
              "## 8. Subject bootstrap", "", "| Cohort | mean ΔIoU Hidden-Native | 95% CI |", "|---|---:|---:|"]
    for cohort, b in result["bootstrap_hidden_minus_native"].items():
        lines.append(f"| {cohort} | {fmt(b['mean_delta'])} | [{fmt(b['CI95'][0])}, {fmt(b['CI95'][1])}] |")
    lines += ["", "## 9. 预固定 GO gate", "",
              "含糊项在看结果前操作化为：Native TP loss≤10%，invalid≤5%，10 项中至少 7 项通过；若 combined bootstrap CI 上界<0，则直接 NO-GO。", "",
              "| Check | Pass |", "|---|---:|"]
    for k, v in result["go_gate"]["checks"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"通过 {result['go_gate']['passed']}/10；hard negative={result['go_gate']['hard_negative']}。", "",
              "## 10. 结论", "", result["conclusion"], "",
              "本实验与此前 hidden-relation classification/ranking audit 不重复：此前问 hidden 能否区分 true/noise candidate；本实验在 candidate anchor 固定后，仅问 hidden 能否预测 signed onset/offset。"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    payload, records, fresh, inventory = load_and_audit_assets(args.hidden_root, args.compact_cache)
    rows, association_audit = build_associations(payload, records, fresh,
                                                  read_csv(args.failure_gt_trace), read_csv(args.failure_prediction_trace))
    if len({r["subject"] for r in rows}) < 2:
        raise RuntimeError("insufficient LOSO subjects")
    oof = attach_predictions(rows)
    cohorts = {
        "A_NATIVE_TP": [r for r in oof if r["cohort"] == "A_NATIVE_TP"],
        "B_LOCALIZATION": [r for r in oof if r["cohort"] == "B_LOCALIZATION"],
        "B_FNC": [r for r in oof if r["is_fnc"]],
        "B_NEAR_FP": [r for r in oof if r["is_near_fp"]],
        "COMBINED": oof,
    }
    summaries = {k: summarize_cohort(v) for k, v in cohorts.items()}
    a, fnc, near = cohorts["A_NATIVE_TP"], cohorts["B_FNC"], cohorts["B_NEAR_FP"]
    preserved = sum(r["hidden_iou"] >= .5 for r in a)
    lost = len(a) - preserved
    rescued = sum(r["hidden_iou"] >= .5 for r in fnc)
    converted = sum(r["hidden_iou"] >= .5 for r in near)
    overlap = {(r["prediction_id"], r["gt_index"]) for r in fnc} & {(r["prediction_id"], r["gt_index"]) for r in near}
    safety = {"native_TP_total": len(a), "native_TP_preserved": preserved, "native_TP_lost": lost,
              "native_TP_preservation_rate": preserved / len(a) if a else None,
              "FN_C_total": len(fnc), "FN_C_rescued": rescued, "FN_C_remained": len(fnc) - rescued,
              "NetGain": rescued - lost, "near_FP_total": len(near), "near_FP_converted": converted,
              "near_FP_remained": len(near) - converted, "FNC_near_FP_overlap_pairs": len(overlap)}
    bootstrap_native = {k: bootstrap(cohorts[k], k) for k in ["A_NATIVE_TP", "B_LOCALIZATION", "COMBINED"]}
    bootstrap_score = {k: bootstrap(cohorts[k], k, "hidden", "score") for k in ["A_NATIVE_TP", "B_LOCALIZATION", "COMBINED"]}
    combined = summaries["COMBINED"]
    n, sc, hi = (combined["predictors"][k] for k in ["native", "score", "hidden"])
    dn = combined["paired"]["Hidden_minus_Native"]["delta_IoU"]
    ds = combined["paired"]["Hidden_minus_Score"]["delta_IoU"]
    checks = {
        "1_hidden_both_endpoint_mean_AE_below_native": hi["onset_AE"]["mean"] < n["onset_AE"]["mean"] and hi["offset_AE"]["mean"] < n["offset_AE"]["mean"],
        "2_hidden_both_endpoint_mean_AE_below_score_and_mean_IoU_above_score": hi["onset_AE"]["mean"] < sc["onset_AE"]["mean"] and hi["offset_AE"]["mean"] < sc["offset_AE"]["mean"] and ds["mean"] > 0,
        "3_hidden_mean_IoU_above_native": dn["mean"] > 0,
        "4_hidden_median_IoU_not_below_native": dn["median"] >= 0,
        "5_hidden_IoU_ge_0.5_fraction_above_native": hi["fraction_IoU_ge_0.5"] > n["fraction_IoU_ge_0.5"],
        "6_native_TP_loss_at_most_10_percent": lost / len(a) <= .10 if a else False,
        "7_FN_C_rescue_exceeds_native_TP_loss": safety["NetGain"] > 0,
        "8_B_LOCALIZATION_mean_delta_positive": summaries["B_LOCALIZATION"]["paired"]["Hidden_minus_Native"]["delta_IoU"]["mean"] > 0,
        "9_combined_bootstrap_lower_CI_nonnegative": bootstrap_native["COMBINED"]["CI95"][0] >= 0,
        "10_hidden_invalid_interval_at_most_5_percent": hi["invalid_interval_fraction"] <= .05,
    }
    hard_negative = bootstrap_native["COMBINED"]["CI95"][1] < 0
    passed = sum(checks.values())
    go = passed >= 7 and not hard_negative
    status = "GO-HREP-SOURCE" if go else "NO-GO-HREP-SOURCE"
    conclusion = ("**GO-HREP-SOURCE：ME-TST+ frozen pre-head temporal representation 在固定低容量、subject-LOSO localization probe 下显示了可利用的 event-boundary information；下一步才允许检查 CASME3 同规格 hidden 资产。**" if go else
                  "**NO-GO-HREP-SOURCE：ME-TST+ frozen pre-head temporal representation 在固定低容量、subject-LOSO localization probe 下未显示足够稳定的 event-boundary information。结合 score-level audits，停止当前 frozen-backbone inference refinement 路线。**")
    result = {
        "status": status, "dataset": "SAMMLV", "asset_audit": inventory,
        "compact_cache_metadata_source": {"path": str(args.compact_cache.resolve()), "sha256": sha256(args.compact_cache),
                                          "use": "GT/sample metadata only; candidate scores come from fresh hidden dump"},
        "candidate_association_audit": association_audit,
        "protocol": {"candidate_source": "fresh score from same NPZ as hidden", "P": P, "k_p": K_P,
                     "patch": "[p-2kp,p+2kp], edge padding, deterministic linear resample to 41",
                     "hidden_representation": "41-D cosine(resampled h_t, h_p) trajectory only",
                     "score_control": "41-point smoothed-score patch with per-patch min-max normalization",
                     "target": "signed normalized r_on=(onset-p)/kp, r_off=(offset-p)/kp",
                     "predictor": "two independent Ridge(alpha=1.0)", "split": "strict subject LOSO",
                     "scaler": "StandardScaler fit on training subjects only",
                     "event_construction": "p+kp*r; np.rint; clip; no swap/clamp/correction; invalid IoU=0"},
        "cohorts": summaries, "safety": safety,
        "bootstrap_hidden_minus_native": bootstrap_native, "bootstrap_hidden_minus_score": bootstrap_score,
        "go_gate": {"rule": ">=7/10 pre-operationalized checks and no hard-negative combined CI",
                    "hard_negative_rule": "combined bootstrap CI upper bound < 0", "checks": checks,
                    "passed": passed, "hard_negative": hard_negative},
        "conclusion": conclusion,
        "next_step": "inspect CASME3 hidden assets" if go else "STOP; no CASME3 export and no further hidden/heuristic search",
    }
    write_csv(out / "hrep_oof_predictions.csv", oof)
    write_csv(out / "hrep_subject_metrics.csv", subject_metrics(oof))
    (out / "hrep_localization_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "HIDDEN_EVENT_LOCALIZATION_PREDICTABILITY_AUDIT_CN.md").write_text(make_report(result), encoding="utf-8")
    print(json.dumps({"status": status, "association_rows": len(oof), "checks_passed": passed,
                      "hard_negative": hard_negative, "combined_hidden_minus_native_mean_iou": dn["mean"],
                      "combined_hidden_minus_score_mean_iou": ds["mean"], "safety": safety}, indent=2))


if __name__ == "__main__":
    main()
