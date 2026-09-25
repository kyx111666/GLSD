#!/usr/bin/env python3
"""Audit information in local spotting-curve morphology beyond peak height.

This is a cache-level falsification test.  The logistic probes are diagnostic
upper bounds, not a proposed decoder.  Candidate generation never uses GT.
"""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 20260902
BOOTSTRAP_REPEATS = 1000
RESAMPLE_POINTS = 31
HEIGHT_MATCH_CALIPER = 0.15
SCALES = (1, 2)
REPRESENTATIONS = ("raw_patch", "shape_minmax", "relative_l2")
CACHES = {
    ("me_tst_plus", "SAMMLV"): ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
    ("me_tst_plus", "CASME_3"): ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
    ("boostingvrme", "SAMMLV"): ROOT / "caches/boostingvrme/sammlv_curves.pkl",
    ("boostingvrme", "CASME_3"): ROOT / "caches/boostingvrme/casme3_curves.pkl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smooth(score: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    return np.convolve(score, np.ones(width, dtype=float) / width, mode="same")


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - median) / (1.4826 * mad)


def interval_iou(left: int, right: int, gt) -> float:
    gt_left, gt_right = int(gt[0]), int(gt[2])
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def extract_patch(curve: np.ndarray, peak: int, radius: int) -> np.ndarray:
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.asarray(padded[peak:peak + 2 * radius + 1], dtype=float)


def resample_patch(patch: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(patch))
    target = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
    return np.interp(target, source, patch).astype(float)


def representations(patch: np.ndarray) -> dict[str, np.ndarray]:
    patch = resample_patch(patch)
    span = float(np.max(patch) - np.min(patch))
    shape = (patch - np.min(patch)) / (span + 1e-8)
    relative = patch - patch[RESAMPLE_POINTS // 2]
    relative /= float(np.linalg.norm(relative) + 1e-8)
    return {
        "raw_patch": patch,
        "shape_minmax": shape,
        "relative_l2": relative,
    }


def load_cache(backbone: str, dataset: str, path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    videos = []
    if "records" in payload:
        for row in payload["records"]:
            videos.append({
                "subject": str(row["subject"]),
                "video": str(row["video"]),
                "score": np.asarray(row["score"], dtype=float),
                "samples": row["samples"],
            })
        schema = "me_tst_strategy_output_records"
    else:
        for subject in payload["subject_curves"]:
            for index, score in enumerate(subject["score"]):
                videos.append({
                    "subject": str(subject["subject"]),
                    "video": str(subject["videos"][index]),
                    "score": np.asarray(score, dtype=float),
                    "samples": subject["samples"][index],
                })
        schema = "boostingvrme_subject_curves"
    metadata = {
        "backbone": backbone,
        "dataset": dataset,
        "input_path": str(path.resolve()),
        "sha256": sha256(path),
        "schema": schema,
        "subjects": len({video["subject"] for video in videos}),
        "videos": len(videos),
        "gt_events": sum(len(video["samples"]) for video in videos),
        "k_p": int(payload["k_p"]),
        "frame_skip": int(payload["frame_skip"]),
    }
    return videos, metadata


def native_match(curve: np.ndarray, samples, k_p: int):
    threshold = float(curve.mean() + 0.55 * (curve.max() - curve.mean()))
    peaks = find_peaks(curve, height=threshold, distance=k_p)[0]
    unmatched = set(range(len(samples)))
    roles = {}
    for peak in peaks:
        overlaps = [
            (interval_iou(int(peak - k_p), int(peak + k_p), gt), index)
            for index, gt in enumerate(samples)
            if index in unmatched
        ]
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            roles[int(peak)] = ("native_tp", int(best_index))
        else:
            roles[int(peak)] = ("native_fp", None)
    counts = {
        "tp": sum(role == "native_tp" for role, _ in roles.values()),
        "fp": sum(role == "native_fp" for role, _ in roles.values()),
        "fn": len(unmatched),
    }
    return roles, unmatched, counts


def build_candidates(backbone: str, dataset: str, path: Path):
    videos, metadata = load_cache(backbone, dataset, path)
    k_p = metadata["k_p"]
    rows = []
    features = {
        scale: {representation: [] for representation in REPRESENTATIONS}
        for scale in SCALES
    }
    native_counts = {"tp": 0, "fp": 0, "fn": 0}
    missed_gt_ids = set()
    category_counts = defaultdict(int)

    for video in videos:
        curve = smooth(video["score"], 2 * k_p)
        normalized = robust_z(curve)
        peaks = find_peaks(curve)[0]
        native_roles, missed, counts = native_match(curve, video["samples"], k_p)
        for key in native_counts:
            native_counts[key] += counts[key]
        video_gt_ids = [
            f"{video['subject']}::{video['video']}::{index}"
            for index in range(len(video["samples"]))
        ]
        missed_gt_ids.update(video_gt_ids[index] for index in missed)
        native_hit = set(range(len(video["samples"]))) - missed

        for peak in peaks:
            containing = {
                index for index, gt in enumerate(video["samples"])
                if int(gt[0]) <= int(peak) <= int(gt[2])
            }
            missed_containing = containing & missed
            hit_containing = containing & native_hit
            best_iou = max([
                interval_iou(int(peak - k_p), int(peak + k_p), gt)
                for gt in video["samples"]
            ] or [0.0])
            native_role, native_gt = native_roles.get(int(peak), (None, None))
            if native_role is not None:
                category = native_role
            elif missed_containing:
                category = "native_missed_gt_local_peak"
            elif hit_containing:
                category = "native_hit_gt_other_peak"
            else:
                category = "ordinary_background_noise"
            category_counts[category] += 1
            rows.append({
                "subject": video["subject"],
                "video": video["video"],
                "peak": int(peak),
                "inside_gt": int(bool(containing)),
                "iou_positive": int(best_iou >= 0.5),
                "inside_missed_gt": int(bool(missed_containing)),
                "inside_native_hit_gt": int(bool(hit_containing)),
                "native_role": native_role,
                "native_gt_index": native_gt,
                "category": category,
                "missed_gt_ids": [video_gt_ids[index] for index in sorted(missed_containing)],
                "height_raw": float(curve[peak]),
                "height_robust": float(normalized[peak]),
            })
            for scale in SCALES:
                patch = extract_patch(curve, int(peak), scale * k_p)
                patch_representations = representations(patch)
                for name in REPRESENTATIONS:
                    features[scale][name].append(patch_representations[name])

    arrays = {
        scale: {
            name: np.asarray(values, dtype=float)
            for name, values in representations_by_name.items()
        }
        for scale, representations_by_name in features.items()
    }
    metadata.update({
        "candidate_count": len(rows),
        "candidate_generation": "all scipy local maxima after moving-average smoothing width 2*k_p",
        "native_counts_audit_protocol": native_counts,
        "native_missed_gt_count": len(missed_gt_ids),
        "candidate_category_counts": dict(sorted(category_counts.items())),
    })
    return {"rows": rows, "features": arrays, "metadata": metadata,
            "missed_gt_ids": missed_gt_ids}


def fit_probe(train_x: np.ndarray, train_y: np.ndarray):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(scaler.transform(train_x), train_y)
    return scaler, model


def loso_probe(features: np.ndarray, rows: list[dict], label: str) -> np.ndarray:
    y = np.asarray([row[label] for row in rows], dtype=int)
    subjects = np.asarray([row["subject"] for row in rows])
    output = np.full(len(rows), np.nan, dtype=float)
    for subject in sorted(set(subjects.tolist())):
        test = subjects == subject
        train = ~test
        if len(np.unique(y[train])) < 2:
            raise RuntimeError(f"training fold for subject {subject} lacks both classes")
        scaler, model = fit_probe(features[train], y[train])
        output[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    if np.isnan(output).any():
        raise RuntimeError("LOSO predictions contain NaN")
    return output


def safe_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "positive": int(np.sum(y)),
                "roc_auc": None, "pr_auc": None}
    return {
        "n": int(len(y)),
        "positive": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
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
                      baseline: str = "height_robust") -> dict:
    subjects = sorted({row["subject"] for row in rows})
    subject_indices = {
        subject: np.asarray([i for i, row in enumerate(rows) if row["subject"] == subject])
        for subject in subjects
    }
    per_subject = defaultdict(lambda: defaultdict(list))
    usable = []
    for subject in subjects:
        indices = subject_indices[subject]
        y = np.asarray([rows[index][label] for index in indices], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        usable.append(subject)
        for name, values in scores.items():
            per_subject[name]["roc_auc"].append(float(roc_auc_score(y, values[indices])))
            per_subject[name]["pr_auc"].append(float(average_precision_score(y, values[indices])))

    rng = np.random.default_rng(SEED)
    output = {"subjects_with_both_classes": len(usable), "repeats": BOOTSTRAP_REPEATS,
              "subject_balanced_metrics": {}}
    for name in scores:
        output["subject_balanced_metrics"][name] = {}
        for metric in ("roc_auc", "pr_auc"):
            values = np.asarray(per_subject[name][metric], dtype=float)
            baseline_values = np.asarray(per_subject[baseline][metric], dtype=float)
            boot = np.asarray([
                np.mean(values[rng.integers(0, len(values), len(values))])
                for _ in range(BOOTSTRAP_REPEATS)
            ])
            delta_values = values - baseline_values
            delta_boot = np.asarray([
                np.mean(delta_values[rng.integers(0, len(values), len(values))])
                for _ in range(BOOTSTRAP_REPEATS)
            ])
            output["subject_balanced_metrics"][name][metric] = {
                "mean": float(np.mean(values)),
                "ci95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
                f"delta_vs_{baseline}": float(np.mean(delta_values)),
                "delta_ci95": [float(value) for value in np.quantile(delta_boot, [0.025, 0.975])],
            }
    return output


def height_matched_indices(rows: list[dict]) -> tuple[np.ndarray, dict]:
    selected = []
    differences = []
    by_subject = defaultdict(list)
    for index, row in enumerate(rows):
        by_subject[row["subject"]].append(index)
    for subject in sorted(by_subject):
        indices = by_subject[subject]
        positives = sorted(
            [index for index in indices if rows[index]["inside_gt"]],
            key=lambda index: (rows[index]["height_robust"], index),
        )
        negatives = {index for index in indices if not rows[index]["inside_gt"]}
        for positive in positives:
            if not negatives:
                break
            negative = min(
                negatives,
                key=lambda index: (abs(rows[index]["height_robust"] -
                                       rows[positive]["height_robust"]), index),
            )
            difference = abs(rows[negative]["height_robust"] -
                             rows[positive]["height_robust"])
            if difference <= HEIGHT_MATCH_CALIPER:
                selected.extend((positive, negative))
                differences.append(difference)
                negatives.remove(negative)
    selected_array = np.asarray(sorted(selected), dtype=int)
    details = {
        "matching": "within-subject greedy 1:1 nearest matching without replacement",
        "height": "per-video robust normalized peak height",
        "caliper": HEIGHT_MATCH_CALIPER,
        "matched_pairs": len(differences),
        "matched_candidates": len(selected),
        "mean_absolute_height_difference": float(np.mean(differences)) if differences else None,
        "median_absolute_height_difference": float(np.median(differences)) if differences else None,
        "p95_absolute_height_difference": float(np.quantile(differences, 0.95)) if differences else None,
    }
    return selected_array, details


def matched_fp(rows: list[dict], scores: dict[str, np.ndarray], fp_budget: int,
               missed_total: int) -> dict:
    output = {}
    for name, values in scores.items():
        order = sorted(range(len(rows)), key=lambda index: (
            -float(values[index]), rows[index]["subject"], rows[index]["video"],
            rows[index]["peak"],
        ))
        fp = 0
        selected = 0
        positive_candidates = 0
        recovered = set()
        for index in order:
            row = rows[index]
            selected += 1
            if row["inside_gt"]:
                positive_candidates += 1
            else:
                fp += 1
            recovered.update(row["missed_gt_ids"])
            if fp >= fp_budget:
                break
        recovered_count = len(recovered)
        output[name] = {
            "fp_budget": int(fp_budget),
            "fp": int(fp),
            "selected_candidates": int(selected),
            "positive_candidates": int(positive_candidates),
            "recovered_missed_gt": int(recovered_count),
            "event_precision_recovered_vs_fp": (
                recovered_count / (recovered_count + fp)
                if recovered_count + fp else 0.0
            ),
            "candidate_precision": positive_candidates / selected if selected else 0.0,
            "recall_of_native_missed_gt": (
                recovered_count / missed_total if missed_total else 0.0
            ),
        }
    return output


def best_f1_threshold(y: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 501)))
    return max(
        ((float(f1_score(y, values >= threshold)), float(threshold))
         for threshold in thresholds),
        key=lambda pair: (pair[0], pair[1]),
    )


def threshold_metrics(y: np.ndarray, values: np.ndarray, threshold: float) -> dict:
    predicted = values >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, predicted)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def analyze_one(key: tuple[str, str], path: Path):
    analysis = build_candidates(*key, path)
    rows = analysis["rows"]
    labels = np.asarray([row["inside_gt"] for row in rows], dtype=int)
    base_scores = {
        "height_raw": np.asarray([row["height_raw"] for row in rows]),
        "height_robust": np.asarray([row["height_robust"] for row in rows]),
    }
    probe_scores = {}
    for scale in SCALES:
        probe_scores[scale] = {}
        for representation in REPRESENTATIONS:
            probe_scores[scale][representation] = loso_probe(
                analysis["features"][scale][representation], rows, "inside_gt"
            )

    matched_indices, matching_details = height_matched_indices(rows)
    missed_indices = np.asarray([
        index for index, row in enumerate(rows)
        if row["inside_missed_gt"] or not row["inside_gt"]
    ], dtype=int)
    result = {
        "metadata": analysis["metadata"],
        "labels": {
            "primary": "candidate peak lies within GT onset-offset",
            "secondary": "peak-centered +/-k_p interval has IoU >= 0.5",
        },
        "scales": {},
    }
    for scale in SCALES:
        scores = dict(base_scores)
        scores.update({
            f"probe_{representation}": values
            for representation, values in probe_scores[scale].items()
        })
        result["scales"][f"L_{scale}kp"] = {
            "radius": int(scale * analysis["metadata"]["k_p"]),
            "all_candidates_primary": metric_table(rows, scores, "inside_gt"),
            "all_candidates_secondary": metric_table(rows, scores, "iou_positive"),
            "native_missed_gt_vs_background": metric_table(
                rows, scores, "inside_missed_gt", missed_indices
            ),
            "subject_bootstrap_primary": subject_bootstrap(rows, scores, "inside_gt"),
            "height_conditioned": {
                "matching": matching_details,
                "metrics": metric_table(rows, scores, "inside_gt", matched_indices),
            },
            "matched_fp_native_budget": matched_fp(
                rows, scores, analysis["metadata"]["native_counts_audit_protocol"]["fp"],
                analysis["metadata"]["native_missed_gt_count"],
            ),
        }
    analysis["base_scores"] = base_scores
    analysis["probe_scores"] = probe_scores
    analysis["labels"] = labels
    return analysis, result


def locked_transfer(source: dict, target: dict) -> dict:
    source_y = source["labels"]
    target_y = target["labels"]
    output = {}
    for scale in SCALES:
        methods = {
            "height_robust": (
                source["base_scores"]["height_robust"],
                target["base_scores"]["height_robust"],
            )
        }
        for representation in REPRESENTATIONS:
            train_x = source["features"][scale][representation]
            target_x = target["features"][scale][representation]
            scaler, model = fit_probe(train_x, source_y)
            target_score = model.predict_proba(scaler.transform(target_x))[:, 1]
            methods[f"probe_{representation}"] = (
                source["probe_scores"][scale][representation], target_score
            )
        scale_result = {}
        for name, (source_oof, target_score) in methods.items():
            source_f1, threshold = best_f1_threshold(source_y, source_oof)
            scale_result[name] = {
                "source_threshold_selection": "best candidate F1 on SAMMLV LOSO OOF predictions",
                "source_oof_f1": source_f1,
                "target_ranking": safe_metrics(target_y, target_score),
                "target_locked_threshold": threshold_metrics(target_y, target_score, threshold),
            }
        output[f"L_{scale}kp"] = scale_result
    return output


def decision(report: dict) -> dict:
    """Apply a conservative, predeclared reading to the native-scale result."""
    checks = {}
    for dataset in ("SAMMLV", "CASME_3"):
        name = f"me_tst_plus__{dataset}"
        scale = report["datasets"][name]["scales"]["L_1kp"]
        metrics = scale["all_candidates_primary"]
        matched = scale["matched_fp_native_budget"]
        conditioned = scale["height_conditioned"]["metrics"]
        bootstrap = scale["subject_bootstrap_primary"]["subject_balanced_metrics"]
        checks[f"me_tst_{dataset}_pr_auc_improves"] = (
            metrics["probe_shape_minmax"]["pr_auc"] > metrics["height_robust"]["pr_auc"]
        )
        checks[f"me_tst_{dataset}_matched_fp_improves"] = (
            matched["probe_shape_minmax"]["recovered_missed_gt"] >
            matched["height_robust"]["recovered_missed_gt"]
        )
        checks[f"me_tst_{dataset}_height_conditioned_improves"] = (
            conditioned["probe_shape_minmax"]["pr_auc"] >
            conditioned["height_robust"]["pr_auc"]
        )
        checks[f"me_tst_{dataset}_subject_pr_delta_ci_above_zero"] = (
            bootstrap["probe_shape_minmax"]["pr_auc"]["delta_ci95"][0] > 0.0
        )
    for backbone in ("me_tst_plus", "boostingvrme"):
        transfer = report["locked_transfer"][backbone]["L_1kp"]
        checks[f"{backbone}_locked_target_pr_auc_improves"] = (
            transfer["probe_shape_minmax"]["target_ranking"]["pr_auc"] >
            transfer["height_robust"]["target_ranking"]["pr_auc"]
        )
        checks[f"{backbone}_locked_target_f1_not_worse"] = (
            transfer["probe_shape_minmax"]["target_locked_threshold"]["f1"] >=
            transfer["height_robust"]["target_locked_threshold"]["f1"]
        )
    for dataset in ("SAMMLV", "CASME_3"):
        name = f"boostingvrme__{dataset}"
        metrics = report["datasets"][name]["scales"]["L_1kp"]["all_candidates_primary"]
        checks[f"boostingvrme_{dataset}_pr_direction_not_reverse"] = (
            metrics["probe_shape_minmax"]["pr_auc"] >= metrics["height_robust"]["pr_auc"]
        )
    go = all(checks.values())
    return {
        "policy": (
            "GO only if native-scale shape-only improves ME-TST+ PR-AUC, matched-FP "
            "recovery, height-conditioned PR-AUC and subject-bootstrap PR delta on both "
            "datasets; transfer survives; BoostingVRME is not systematically reverse."
        ),
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "verdict": "GO" if go else "NO-GO",
        "required_final_statement": (
            "GO：local temporal morphology 在 peak height 之外提供稳定独立信息，可以进一步设计 "
            "training-free morphology-based inference algorithm。"
            if go else
            "NO-GO：local temporal morphology 没有提供足够稳定的独立信息，不建议继续从 frozen "
            "spotting curve 设计 morphology decoder。"
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "purpose": "temporal morphology information audit; not a decoder",
            "seed": SEED,
            "candidate_pool": "all local maxima on native-smoothed frozen spotting curve",
            "smoothing": "moving average width 2*k_p",
            "candidate_generation_uses_gt": False,
            "same_candidate_pool_for_height_and_morphology": True,
            "patch_scales": ["L=k_p", "L=2*k_p"],
            "edge_handling": "edge-value padding",
            "resample_points": RESAMPLE_POINTS,
            "representations": {
                "raw_patch": "resampled smoothed score patch; amplitude retained",
                "shape_minmax": "(patch-min)/(max-min+1e-8); amplitude removed",
                "relative_l2": "subtract center value then divide by L2 norm",
            },
            "probe": {
                "model": "sklearn LogisticRegression",
                "configuration": {
                    "C": 1.0, "class_weight": "balanced", "solver": "liblinear",
                    "max_iter": 2000, "random_state": SEED,
                },
                "outer_evaluation": "subject-level LOSO",
                "scaler": "StandardScaler fit on training subjects only",
                "interpretation": "diagnostic information upper bound, not final algorithm",
            },
            "bootstrap": {
                "unit": "subject", "repeats": BOOTSTRAP_REPEATS,
                "seed": SEED, "summary": "mean of within-subject metrics",
            },
            "height_matching": {
                "within_subject": True, "one_to_one": True,
                "without_replacement": True, "caliper_robust_z": HEIGHT_MATCH_CALIPER,
            },
            "matched_fp_budget": "audit-side native decoder FP on the same cache",
            "locked_transfer": (
                "fit probe on all SAMMLV candidates; choose threshold on SAMMLV LOSO OOF; "
                "apply model and threshold directly to CASME3"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "datasets": {},
        "locked_transfer": {},
    }
    analyses = {}
    for key, path in CACHES.items():
        print(f"Analyzing {key[0]} x {key[1]} ...", flush=True)
        analysis, result = analyze_one(key, path)
        analyses[key] = analysis
        report["datasets"][f"{key[0]}__{key[1]}"] = result
    for backbone in ("me_tst_plus", "boostingvrme"):
        report["locked_transfer"][backbone] = locked_transfer(
            analyses[(backbone, "SAMMLV")], analyses[(backbone, "CASME_3")]
        )
    report["decision"] = decision(report)
    output_path = OUT / "temporal_morphology_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "output": str(output_path),
        "verdict": report["decision"]["verdict"],
        "checks_passed": report["decision"]["passed"],
        "checks_total": report["decision"]["total"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
