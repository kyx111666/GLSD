#!/usr/bin/env python3
"""Phase-1 frozen temporal feature-relation audit: ME-TST+ x SAMMLV.

Fresh score/hidden arrays come only from the frozen full dump.  The historical
compact cache is read only for GT intervals and fixed protocol metadata (k_p,
frame_skip).  This script trains no backbone and does not modify any input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SEED = 20260903
SCALES = (1, 2)
RELATION_SIZE = 21
HEIGHT_CALIPER = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.name).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def smooth(score: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(score, np.ones(width, dtype=float) / width, mode="same")


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - median) / (1.4826 * mad)


def interval_iou(left: int, right: int, gt: list) -> float:
    gt_left, gt_right = int(gt[0]), int(gt[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def native_match(curve: np.ndarray, samples: list, k_p: int):
    threshold = float(curve.mean() + 0.55 * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0]
    unmatched = set(range(len(samples)))
    roles = {}
    for peak in peaks:
        overlaps = [
            (interval_iou(int(peak - k_p), int(peak + k_p), gt), index)
            for index, gt in enumerate(samples) if index in unmatched
        ]
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            roles[int(peak)] = ("native_tp", int(best_index))
        else:
            roles[int(peak)] = ("native_fp", None)
    return roles, unmatched, {
        "tp": sum(role == "native_tp" for role, _ in roles.values()),
        "fp": sum(role == "native_fp" for role, _ in roles.values()),
        "fn": len(unmatched),
    }


def load_gt_only(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"records", "k_p", "frame_skip"}
    if not required.issubset(payload):
        raise KeyError(f"GT cache lacks fields: {sorted(required - set(payload))}")
    records = {}
    forbidden_used = []
    for row in payload["records"]:
        key = (str(row["subject"]), str(row["video"]))
        records[key] = {"samples": row["samples"]}
    # Explicit provenance guard: numeric historical predictions are never copied.
    for field in ("score", "logits", "emotion", "mirror_score", "mirror_logits"):
        if any(field not in row for row in payload["records"]):
            continue
        forbidden_used.append(field)
    return records, int(payload["k_p"]), int(payload["frame_skip"]), forbidden_used


def resize_square(matrix: np.ndarray, size: int = RELATION_SIZE) -> np.ndarray:
    old = np.linspace(0.0, 1.0, matrix.shape[0])
    new = np.linspace(0.0, 1.0, size)
    rows = np.vstack([np.interp(new, old, row) for row in matrix])
    return np.vstack([np.interp(new, old, rows[:, col]) for col in range(size)]).T


def relation_features(hidden: np.ndarray, peak: int, radius: int):
    padded = np.pad(hidden, ((radius, radius), (0, 0)), mode="edge")
    window = np.asarray(padded[peak:peak + 2 * radius + 1], dtype=np.float64)
    normalized = window / np.maximum(np.linalg.norm(window, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    transition = float(np.mean(1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)))
    center = normalized[radius]
    context = np.concatenate((normalized[:radius], normalized[radius + 1:]), axis=0)
    center_context = float(1.0 - np.mean(context @ center))
    return transition, center_context, resize_square(similarity).reshape(-1)


def fit_probe(train_x: np.ndarray, train_y: np.ndarray):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=SEED,
    ).fit(scaler.transform(train_x), train_y)
    return scaler, model


def loso_probe(features: np.ndarray, rows: list[dict], label: str) -> np.ndarray:
    y = np.asarray([row[label] for row in rows], dtype=int)
    subjects = np.asarray([row["subject"] for row in rows])
    output = np.full(len(rows), np.nan, dtype=float)
    for subject in sorted(set(subjects.tolist())):
        test = subjects == subject
        train = ~test
        if len(np.unique(y[train])) != 2:
            raise RuntimeError(f"LOSO training fold {subject} lacks both classes")
        scaler, model = fit_probe(features[train], y[train])
        output[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    if np.isnan(output).any():
        raise RuntimeError("LOSO output contains NaN")
    return output


def safe_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "positive": int(y.sum()), "prevalence": None,
                "roc_auc": None, "pr_auc": None}
    return {
        "n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def metric_table(rows: list[dict], scores: dict[str, np.ndarray], label: str,
                 indices: np.ndarray | None = None) -> dict:
    if indices is None:
        indices = np.arange(len(rows))
    y = np.asarray([rows[index][label] for index in indices], dtype=int)
    return {name: safe_metrics(y, values[indices]) for name, values in scores.items()}


def subject_bootstrap(rows: list[dict], scores: dict[str, np.ndarray], label: str,
                      repeats: int) -> dict:
    by_subject = defaultdict(list)
    for index, row in enumerate(rows):
        by_subject[row["subject"]].append(index)
    usable = []
    values = defaultdict(lambda: defaultdict(list))
    for subject in sorted(by_subject):
        indices = np.asarray(by_subject[subject], dtype=int)
        y = np.asarray([rows[index][label] for index in indices], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        usable.append(subject)
        for name, score in scores.items():
            values[name]["roc_auc"].append(float(roc_auc_score(y, score[indices])))
            values[name]["pr_auc"].append(float(average_precision_score(y, score[indices])))
    if not usable:
        raise RuntimeError("no subject has both classes for subject bootstrap")
    rng = np.random.default_rng(SEED)
    result = {"unit": "subject", "subjects_with_both_classes": len(usable),
              "repeats": repeats, "metrics": {}}
    for name in scores:
        result["metrics"][name] = {}
        for metric in ("roc_auc", "pr_auc"):
            current = np.asarray(values[name][metric])
            baseline = np.asarray(values["height_robust"][metric])
            draws = rng.integers(0, len(current), size=(repeats, len(current)))
            boot = current[draws].mean(axis=1)
            delta = (current - baseline)[draws].mean(axis=1)
            result["metrics"][name][metric] = {
                "mean": float(current.mean()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "delta_vs_height": float((current - baseline).mean()),
                "delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            }
    return result


def height_matched_indices(rows: list[dict]):
    selected, differences = [], []
    by_subject = defaultdict(list)
    for index, row in enumerate(rows):
        by_subject[row["subject"]].append(index)
    for subject in sorted(by_subject):
        positives = sorted(
            [i for i in by_subject[subject] if rows[i]["inside_gt"]],
            key=lambda i: (rows[i]["height_robust"], i),
        )
        negatives = {i for i in by_subject[subject] if not rows[i]["inside_gt"]}
        for positive in positives:
            if not negatives:
                break
            negative = min(negatives, key=lambda i: (
                abs(rows[i]["height_robust"] - rows[positive]["height_robust"]), i))
            difference = abs(rows[negative]["height_robust"] - rows[positive]["height_robust"])
            if difference <= HEIGHT_CALIPER:
                selected.extend((positive, negative))
                differences.append(difference)
                negatives.remove(negative)
    return np.asarray(sorted(selected), dtype=int), {
        "method": "within-subject greedy 1:1 nearest matching without replacement",
        "caliper": HEIGHT_CALIPER, "matched_pairs": len(differences),
        "mean_abs_height_difference": float(np.mean(differences)) if differences else None,
        "max_abs_height_difference": float(np.max(differences)) if differences else None,
    }


def matched_fp(rows: list[dict], scores: dict[str, np.ndarray], fp_budget: int,
               missed_total: int) -> dict:
    result = {}
    for name, score in scores.items():
        order = sorted(range(len(rows)), key=lambda i: (
            -float(score[i]), rows[i]["subject"], rows[i]["video_id"], rows[i]["peak"]))
        fp = 0
        selected = 0
        positive_candidates = 0
        recovered = set()
        for index in order:
            selected += 1
            if rows[index]["inside_gt"]:
                positive_candidates += 1
            else:
                fp += 1
            recovered.update(rows[index]["missed_gt_ids"])
            if fp >= fp_budget:
                break
        result[name] = {
            "fp_budget": fp_budget, "fp": fp, "selected_candidates": selected,
            "positive_candidates": positive_candidates,
            "recovered_missed_gt": len(recovered),
            "precision_recovered_vs_fp": len(recovered) / (len(recovered) + fp)
            if len(recovered) + fp else 0.0,
            "recall_of_native_missed_gt": len(recovered) / missed_total if missed_total else 0.0,
        }
    return result


def best_f1(y: np.ndarray, score: np.ndarray) -> dict:
    thresholds = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 501)))
    f1, threshold = max(
        ((float(f1_score(y, score >= threshold)), float(threshold)) for threshold in thresholds),
        key=lambda pair: (pair[0], pair[1]),
    )
    return {"oof_best_f1": f1, "oof_threshold": threshold,
            "note": "diagnostic only; not an unbiased final decoder estimate"}


def build_dataset(fresh_root: Path, gt_cache: Path):
    manifest_path = fresh_root / "sammlv_full_hidden_dump_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("num_videos") != 79:
        raise RuntimeError("fresh dump manifest is not COMPLETE with 79 videos")
    if manifest.get("historical_compact_numeric_outputs_used") is not False:
        raise RuntimeError("fresh dump provenance guard failed")
    gt, k_p, frame_skip, historical_numeric_fields_present = load_gt_only(gt_cache)
    rows, features = [], {scale: [] for scale in SCALES}
    native_counts = {"tp": 0, "fp": 0, "fn": 0}
    missed_gt_ids = set()
    excluded = 0
    raw_candidates = 0
    npz_paths = []
    max_radius = max(SCALES) * k_p

    for record in sorted(manifest["records"], key=lambda row: row["global_video_index"]):
        subject, video = str(record["subject"]), str(record["video_id"])
        key = (subject, video)
        if key not in gt:
            raise KeyError(f"missing GT metadata for {key}")
        path = fresh_root / f"subject_{subject}" / f"{video}_frozen_hidden_score.npz"
        npz_paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            hidden = np.asarray(saved["hidden"], dtype=np.float32)
            score = np.asarray(saved["score"], dtype=np.float64)
            valid = np.asarray(saved["valid_mask"], dtype=bool)
        if hidden.shape != (len(score), 384) or valid.shape != score.shape:
            raise RuntimeError(f"unaligned fresh arrays: {path}")
        samples = gt[key]["samples"]
        curve = smooth(score, 2 * k_p)
        normalized_height = robust_z(curve)
        roles, unmatched, counts = native_match(curve, samples, k_p)
        for field in native_counts:
            native_counts[field] += counts[field]
        video_gt_ids = {i: f"{subject}/{video}/gt_{i}" for i in range(len(samples))}
        missed_gt_ids.update(video_gt_ids[i] for i in unmatched)
        peaks = find_peaks(curve)[0]
        raw_candidates += len(peaks)
        for peak in peaks:
            left, right = max(0, peak - max_radius), min(len(valid), peak + max_radius + 1)
            if not valid[peak] or not np.all(valid[left:right]):
                excluded += 1
                continue
            containing = [i for i, sample in enumerate(samples)
                          if int(sample[0]) <= peak <= int(sample[2])]
            missed_containing = [i for i in containing if i in unmatched]
            hit_containing = [i for i in containing if i not in unmatched]
            native_role, native_gt = roles.get(int(peak), ("not_native_selected", None))
            row = {
                "subject": subject, "video_id": video, "peak": int(peak),
                "inside_gt": int(bool(containing)),
                "iou_positive": int(max((interval_iou(peak-k_p, peak+k_p, s)
                                         for s in samples), default=0.0) >= 0.5),
                "inside_missed_gt": int(bool(missed_containing)),
                "inside_native_hit_gt": int(bool(hit_containing)),
                "native_role": native_role,
                "native_gt_index": native_gt,
                "missed_gt_ids": [video_gt_ids[i] for i in missed_containing],
                "height_raw": float(curve[peak]),
                "height_robust": float(normalized_height[peak]),
            }
            for scale in SCALES:
                transition, center_context, relation = relation_features(
                    hidden, int(peak), scale * k_p)
                row[f"transition_L{scale}"] = transition
                row[f"center_context_L{scale}"] = center_context
                features[scale].append(relation)
            rows.append(row)

    if set(gt) != {(str(r["subject"]), str(r["video_id"])) for r in manifest["records"]}:
        raise RuntimeError("fresh manifest and GT metadata video sets differ")
    metadata = {
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "fresh_npz_aggregate_sha256": aggregate_sha256(npz_paths),
        "gt_cache_path": str(gt_cache), "gt_cache_sha256": sha256(gt_cache),
        "gt_cache_fields_consumed": ["subject", "video", "samples", "k_p", "frame_skip"],
        "gt_cache_historical_numeric_fields_present_but_not_consumed": historical_numeric_fields_present,
        "fresh_numeric_fields_consumed": ["hidden", "score", "valid_mask"],
        "k_p": k_p, "frame_skip": frame_skip,
        "num_subjects": len({row["subject"] for row in rows}),
        "num_videos": len(manifest["records"]), "gt_events": sum(len(v["samples"]) for v in gt.values()),
        "raw_local_maxima": raw_candidates, "eligible_candidates": len(rows),
        "excluded_invalid_relation_window": excluded,
        "native_counts_full_curve": native_counts,
        "native_missed_gt_count": len(missed_gt_ids),
        "unwritten_positions": sum(int(r["num_unwritten_zero_positions"]) for r in manifest["records"]),
    }
    return rows, {scale: np.asarray(features[scale], dtype=np.float32) for scale in SCALES}, metadata


def write_candidates(path: Path, rows: list[dict], probe_scores: dict[int, np.ndarray]) -> None:
    fields = [key for key in rows[0] if key != "missed_gt_ids"]
    fields += [f"relation_probe_L{scale}" for scale in SCALES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {key: value for key, value in row.items() if key != "missed_gt_ids"}
            output.update({f"relation_probe_L{scale}": float(probe_scores[scale][index])
                           for scale in SCALES})
            writer.writerow(output)


def render_report(result: dict) -> str:
    lines = [
        "# ME-TST+ × SAMMLV Frozen Temporal Feature Relation Audit",
        "",
        "> 状态：SAMMLV Phase-1 已计算；最终跨数据集/跨 backbone GO/NO-GO 仍为 PENDING。",
        "",
        "## 数据与边界",
        "",
        f"- Fresh videos：{result['metadata']['num_videos']}；subjects：{result['metadata']['num_subjects']}。",
        f"- GT events：{result['metadata']['gt_events']}；eligible candidates：{result['metadata']['eligible_candidates']}。",
        f"- `valid_mask=False` 共 {result['metadata']['unwritten_positions']} 个位置；因关系窗口触及无效位置排除 {result['metadata']['excluded_invalid_relation_window']} 个候选。",
        "- 历史 compact cache 只读取 subject/video/samples/k_p/frame_skip，不读取 score/logits/emotion。",
        "- Probe 是信息上限诊断，不是最终 decoder；backbone 没有训练或更新。",
        "",
        "## 主要结果",
        "",
        "| Window | Evidence | All PR-AUC | All ROC-AUC | Height-matched PR-AUC | Missed-vs-bg PR-AUC | Recovered missed GT @ native FP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scale in SCALES:
        block = result["scales"][f"L_{scale}kp"]
        for name in ("height_robust", "transition", "center_context", "relation_probe"):
            all_m = block["all_candidates_primary"][name]
            hm = block["height_conditioned"]["metrics"][name]
            missed = block["native_missed_gt_vs_background"][name]
            recovered = block["matched_fp_native_budget"][name]["recovered_missed_gt"]
            def fmt(value): return "N/A" if value is None else f"{value:.4f}"
            lines.append(f"| L={scale}k_p | {name} | {fmt(all_m['pr_auc'])} | {fmt(all_m['roc_auc'])} | {fmt(hm['pr_auc'])} | {fmt(missed['pr_auc'])} | {recovered} |")
    lines += [
        "", "## 解释边界", "",
        "本报告只回答 SAMMLV 上是否出现关系信号。即使数值为正，也必须完成 ME-TST+ × CASME3 locked transfer，才能判断跨数据集证据；在获得第二 backbone 的可信 hidden 之前，也不能声称 model-agnostic Skill。",
        "", "## 人工判定", "",
        "- 当前 verdict：**PENDING**。",
        "- 请先核对两个窗口的 PR-AUC、height-conditioned、missed-GT recovery 与 subject-bootstrap delta 是否同方向。",
        "- 若 SAMMLV 信号成立，下一步仅做 CASME3 fresh hidden dump 与 locked transfer；不先设计正式 Skill。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, relation, metadata = build_dataset(args.fresh_root, args.gt_cache)
    if not rows:
        raise RuntimeError("no eligible candidates")
    labels = np.asarray([row["inside_gt"] for row in rows], dtype=int)
    matched_indices, matching = height_matched_indices(rows)
    missed_indices = np.asarray([i for i, row in enumerate(rows)
                                 if row["inside_missed_gt"] or not row["inside_gt"]], dtype=int)
    probe_scores = {scale: loso_probe(relation[scale], rows, "inside_gt") for scale in SCALES}
    result = {
        "status": "SAMMLV_PHASE1_COMPLETE_FINAL_VERDICT_PENDING",
        "metadata": metadata,
        "protocol": {
            "candidate_generation": "all scipy local maxima after moving-average smoothing width 2*k_p; GT-free",
            "common_eligibility": "candidate and clipped +/-2*k_p relation window must be valid_mask=True",
            "height_baseline": "per-video robust z using median and MAD",
            "windows": ["L=k_p", "L=2*k_p"], "relation_resize": [RELATION_SIZE, RELATION_SIZE],
            "deterministic_scalars": ["transition magnitude", "center-context dissimilarity"],
            "probe": "StandardScaler + LogisticRegression(C=1,class_weight=balanced,solver=liblinear), strict subject-LOSO",
            "seed": SEED, "bootstrap_repeats": args.bootstrap_repeats,
            "final_decision_policy": "PENDING until CASME3 locked transfer; cross-backbone unverified",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "scales": {},
    }
    for scale in SCALES:
        scores = {
            "height_robust": np.asarray([row["height_robust"] for row in rows]),
            "transition": np.asarray([row[f"transition_L{scale}"] for row in rows]),
            "center_context": np.asarray([row[f"center_context_L{scale}"] for row in rows]),
            "relation_probe": probe_scores[scale],
        }
        result["scales"][f"L_{scale}kp"] = {
            "radius": scale * metadata["k_p"],
            "all_candidates_primary": metric_table(rows, scores, "inside_gt"),
            "all_candidates_secondary": metric_table(rows, scores, "iou_positive"),
            "height_conditioned": {"matching": matching,
                                   "metrics": metric_table(rows, scores, "inside_gt", matched_indices)},
            "native_missed_gt_vs_background": metric_table(
                rows, scores, "inside_missed_gt", missed_indices),
            "matched_fp_native_budget": matched_fp(
                rows, scores, metadata["native_counts_full_curve"]["fp"],
                metadata["native_missed_gt_count"]),
            "subject_bootstrap_primary": subject_bootstrap(
                rows, scores, "inside_gt", args.bootstrap_repeats),
            "source_oof_threshold_diagnostic": best_f1(labels, probe_scores[scale]),
        }
    json_path = args.output_dir / "sammlv_relation_audit_results.json"
    csv_path = args.output_dir / "sammlv_relation_candidates.csv"
    report_path = args.output_dir / "SAMMLV_RELATION_AUDIT_CN.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_candidates(csv_path, rows, probe_scores)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "candidates": len(rows),
        "json": str(json_path), "csv": str(csv_path), "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
