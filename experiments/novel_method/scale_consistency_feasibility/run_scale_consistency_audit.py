#!/usr/bin/env python3
"""Cache-only falsification audit for cross-scale peak consistency.

Only ``record['score']`` is read from each frozen ME-TST+ cache.  The script
does not train, forward a model, inspect mirror outputs, or use recognition
logits.  GT is used only after candidate generation for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHES = {
    "SAMMLV": ROOT / "caches/me_tst/sammlv_strategy1_outputs.pkl",
    "CASME_3": ROOT / "caches/me_tst/casme3_strategy1_outputs.pkl",
}
SEED = 20260903
BOOTSTRAP_REPEATS = 1000
SCALE_RATIOS = (2 / 3, 4 / 5, 1.0, 5 / 4, 3 / 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_half_up(value: float) -> int:
    return max(1, int(np.floor(value + 0.5)))


def fixed_scales(k_p: int) -> list[int]:
    return list(dict.fromkeys(round_half_up(k_p * ratio) for ratio in SCALE_RATIOS))


def moving_average(score: np.ndarray, width: int) -> np.ndarray:
    kernel = np.full(max(1, int(width)), 1.0 / max(1, int(width)))
    return np.convolve(score, kernel, mode="same")


def generate_scale_peaks(score: np.ndarray, scale: int) -> list[dict]:
    curve = moving_average(score, 2 * scale)
    threshold = float(curve.mean() + 0.55 * (curve.max() - curve.mean()))
    positions = find_peaks(curve, height=threshold, distance=scale)[0]
    return [
        {"position": int(position), "scale": int(scale),
         "height": float(curve[position])}
        for position in positions
    ]


def load_cache(path: Path) -> tuple[list[dict], dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    videos = []
    for record in payload["records"]:
        # Deliberately do not read mirror_score, logits, mirror_logits, emotion.
        videos.append({
            "subject": str(record["subject"]),
            "video": str(record["video"]),
            "score": np.asarray(record["score"], dtype=float),
            "samples": [tuple(map(int, sample[:3])) for sample in record["samples"]],
        })
    metadata = {
        "dataset": str(payload["dataset"]),
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "subjects": len({video["subject"] for video in videos}),
        "videos": len(videos),
        "gt": sum(len(video["samples"]) for video in videos),
        "native_k_p": int(payload["k_p"]),
        "scales": fixed_scales(int(payload["k_p"])),
        "score_only": True,
    }
    return videos, metadata


def connected_components(peaks: list[dict], radius: int) -> list[list[dict]]:
    """Single-link temporal components; adjacent sorted peaks link within radius."""
    ordered = sorted(peaks, key=lambda item: (item["position"], item["scale"]))
    if not ordered:
        return []
    components = [[ordered[0]]]
    for peak in ordered[1:]:
        if peak["position"] - components[-1][-1]["position"] <= radius:
            components[-1].append(peak)
        else:
            components.append([peak])
    return components


def reduce_component(component: list[dict]) -> dict:
    """Keep at most one peak per scale, then use the rounded median position."""
    center = float(np.median([item["position"] for item in component]))
    by_scale = defaultdict(list)
    for item in component:
        by_scale[item["scale"]].append(item)
    members = []
    for scale in sorted(by_scale):
        members.append(min(
            by_scale[scale],
            key=lambda item: (abs(item["position"] - center),
                              -item["height"], item["position"]),
        ))
    representative = round_half_up(float(np.median(
        [item["position"] for item in members]
    )))
    return {
        "representative": representative,
        "support": len(members),
        "height": max(item["height"] for item in members),
        "members": members,
    }


def containing_gt(position: int, samples: list[tuple[int, int, int]]) -> list[int]:
    return [index for index, sample in enumerate(samples)
            if sample[0] <= position <= sample[2]]


def interval_iou(left: int, right: int, sample: tuple[int, int, int]) -> float:
    gt_left, gt_right = sample[0], sample[2]
    intersection = max(0, min(right, gt_right) - max(left, gt_left) + 1)
    union = max(right, gt_right) - min(left, gt_left) + 1
    return intersection / union if union else 0.0


def ranking_metrics(labels: np.ndarray, values: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "pr_auc": float(average_precision_score(labels, values)),
    }


def threshold_curve(rows: list[dict]) -> list[dict]:
    total_positive = sum(row["positive"] for row in rows)
    output = []
    for threshold in range(1, 6):
        selected = [row for row in rows if row["support"] >= threshold]
        tp = sum(row["positive"] for row in selected)
        fp = len(selected) - tp
        precision = tp / len(selected) if selected else 0.0
        recall = tp / total_positive if total_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        output.append({
            "threshold": "=5" if threshold == 5 else f">={threshold}",
            "tp_related_clusters": tp,
            "fp_noise_clusters": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return output


def support_table(rows: list[dict]) -> list[dict]:
    output = []
    for support in range(1, 6):
        selected = [row for row in rows if row["support"] == support]
        positives = sum(row["positive"] for row in selected)
        negatives = len(selected) - positives
        output.append({
            "support": support,
            "clusters": len(selected),
            "positive": positives,
            "negative": negatives,
            "positive_rate": positives / len(selected) if selected else None,
            "precision": positives / len(selected) if selected else None,
        })
    return output


def height_matched(rows: list[dict]) -> dict:
    """Optimal, without-replacement nearest-height matching within subject."""
    selected_indices = []
    by_subject = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(rows):
        by_subject[row["subject"]][row["positive"]].append(index)
    pair_differences = []
    for groups in by_subject.values():
        positives, negatives = groups[1], groups[0]
        if not positives or not negatives:
            continue
        costs = np.abs(
            np.asarray([rows[i]["height"] for i in positives])[:, None]
            - np.asarray([rows[i]["height"] for i in negatives])[None, :]
        )
        positive_rows, negative_cols = linear_sum_assignment(costs)
        for p_row, n_col in zip(positive_rows, negative_cols):
            selected_indices.extend([positives[p_row], negatives[n_col]])
            pair_differences.append(float(costs[p_row, n_col]))
    labels = np.asarray([rows[i]["positive"] for i in selected_indices])
    height = np.asarray([rows[i]["height"] for i in selected_indices])
    support = np.asarray([rows[i]["support"] / 5 for i in selected_indices])
    return {
        "pairs": len(selected_indices) // 2,
        "mean_abs_height_difference": float(np.mean(pair_differences))
        if pair_differences else None,
        "median_abs_height_difference": float(np.median(pair_differences))
        if pair_differences else None,
        "height": ranking_metrics(labels, height),
        "support": ranking_metrics(labels, support),
    }


def matched_fp_primary(rows: list[dict], evidence: str, fp_budget: int,
                       missed_gt_ids: set[str]) -> dict:
    ordered = sorted(
        rows,
        key=lambda row: (-row[evidence], row["subject"], row["video"],
                         row["representative"]),
    )
    selected = []
    fp = 0
    for row in ordered:
        if not row["positive"]:
            if fp == fp_budget:
                break
            fp += 1
        selected.append(row)
    covered_missed = set()
    for row in selected:
        covered_missed.update(set(row["gt_ids"]) & missed_gt_ids)
    return {
        "evidence": evidence,
        "gt_related_clusters": sum(row["positive"] for row in selected),
        "unique_gt_covered": len({gt for row in selected for gt in row["gt_ids"]}),
        "native_missed_recovered": len(covered_missed),
        "fp": fp,
        "fp_budget": fp_budget,
        "selected_clusters": len(selected),
    }


def match_iou_candidates(candidates: list[dict], gt_ids: list[str],
                         samples: list[tuple[int, int, int]], k_p: int
                         ) -> tuple[int, int, set[str]]:
    """Greedy one-to-one IoU matching used by the earlier fixed audits."""
    unmatched = set(range(len(samples)))
    matched_ids = set()
    fp = 0
    for candidate in candidates:
        overlaps = [
            (interval_iou(candidate["position"] - k_p,
                          candidate["position"] + k_p, samples[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_iou >= 0.5:
            unmatched.remove(best_index)
            matched_ids.add(gt_ids[best_index])
        else:
            fp += 1
    return len(matched_ids), fp, matched_ids


def matched_fp_iou(rows: list[dict], evidence: str, fp_budget: int,
                   all_gt_ids: set[str], native_missed_gt_ids: set[str]) -> dict:
    """Rank clusters and stop immediately before exceeding the native FP budget."""
    ordered = sorted(
        rows,
        key=lambda row: (-row[evidence], row["subject"], row["video"],
                         row["representative"]),
    )
    unmatched = set(all_gt_ids)
    matched_ids = set()
    selected = 0
    fp = 0
    for row in ordered:
        available = [
            (item["iou"], item["gt_id"])
            for item in row["iou_matches"] if item["gt_id"] in unmatched
        ]
        best_iou, best_id = max(available, default=(0.0, ""))
        is_tp = best_iou >= 0.5
        if not is_tp and fp == fp_budget:
            break
        selected += 1
        if is_tp:
            unmatched.remove(best_id)
            matched_ids.add(best_id)
        else:
            fp += 1
    return {
        "evidence": evidence,
        "gt_related_clusters": len(matched_ids),
        "unique_gt_covered": len(matched_ids),
        "native_missed_recovered": len(matched_ids & native_missed_gt_ids),
        "fp": fp,
        "fp_budget": fp_budget,
        "selected_clusters": selected,
    }


def bootstrap_delta(rows: list[dict]) -> dict:
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        batch = [row for subject in sampled for row in by_subject[subject]]
        labels = np.asarray([row["positive"] for row in batch])
        if len(np.unique(labels)) < 2:
            continue
        height = np.asarray([row["height"] for row in batch])
        joint = np.asarray([row["joint"] for row in batch])
        deltas.append(float(
            average_precision_score(labels, joint)
            - average_precision_score(labels, height)
        ))
    return {
        "seed": SEED,
        "requested_repeats": BOOTSTRAP_REPEATS,
        "valid_repeats": len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "ci95": [float(value) for value in np.percentile(deltas, [2.5, 97.5])],
    }


def audit_dataset(path: Path) -> tuple[dict, list[dict]]:
    videos, metadata = load_cache(path)
    scales = metadata["scales"]
    k_p = metadata["native_k_p"]
    scale_hits = np.zeros((metadata["gt"], len(scales)), dtype=bool)
    gt_cursor = 0
    rows = []
    native_fp_budget = 0
    native_primary_fp_budget = 0
    missed_gt_ids = set()
    native_decoder_matched_gt_ids = set()
    all_gt_ids = set()

    for video in videos:
        scale_peaks = {
            scale: generate_scale_peaks(video["score"], scale)
            for scale in scales
        }
        video_gt_ids = [
            f"{video['subject']}::{video['video']}::{index}"
            for index in range(len(video["samples"]))
        ]
        all_gt_ids.update(video_gt_ids)
        for scale_index, scale in enumerate(scales):
            for gt_index, sample in enumerate(video["samples"]):
                scale_hits[gt_cursor + gt_index, scale_index] = any(
                    sample[0] <= peak["position"] <= sample[2]
                    for peak in scale_peaks[scale]
                )
        native_index = scales.index(k_p)
        for gt_index, hit in enumerate(scale_hits[
                gt_cursor:gt_cursor + len(video["samples"]), native_index]):
            if not hit:
                missed_gt_ids.add(video_gt_ids[gt_index])
        _, video_native_fp, video_native_matches = match_iou_candidates(
            scale_peaks[k_p], video_gt_ids, video["samples"], k_p
        )
        native_fp_budget += video_native_fp
        native_decoder_matched_gt_ids.update(video_native_matches)
        native_primary_fp_budget += sum(
            not containing_gt(peak["position"], video["samples"])
            for peak in scale_peaks[k_p]
        )

        all_peaks = [peak for scale in scales for peak in scale_peaks[scale]]
        for component in connected_components(all_peaks, k_p // 2):
            cluster = reduce_component(component)
            contained = containing_gt(cluster["representative"], video["samples"])
            best_iou = max([
                interval_iou(cluster["representative"] - k_p,
                             cluster["representative"] + k_p, sample)
                for sample in video["samples"]
            ] or [0.0])
            iou_matches = [
                {"gt_id": video_gt_ids[index],
                 "iou": interval_iou(cluster["representative"] - k_p,
                                     cluster["representative"] + k_p, sample)}
                for index, sample in enumerate(video["samples"])
            ]
            support_fraction = cluster["support"] / 5
            rows.append({
                "dataset": metadata["dataset"],
                "subject": video["subject"],
                "video": video["video"],
                "representative": cluster["representative"],
                "support": cluster["support"],
                "height": cluster["height"],
                "support_fraction": support_fraction,
                "joint": cluster["height"] * support_fraction,
                "positive": int(bool(contained)),
                "iou_positive": int(best_iou >= 0.5),
                "gt_ids": [video_gt_ids[index] for index in contained],
                "iou_matches": iou_matches,
                "member_positions": [item["position"] for item in cluster["members"]],
                "member_scales": [item["scale"] for item in cluster["members"]],
            })
        gt_cursor += len(video["samples"])

    native = scale_hits[:, scales.index(k_p)]
    union = scale_hits.any(axis=1)
    missed = ~native
    native_decoder_missed_gt_ids = all_gt_ids - native_decoder_matched_gt_ids
    missed_support = scale_hits[missed].sum(axis=1)
    labels = np.asarray([row["positive"] for row in rows])
    metric_values = {
        "height": np.asarray([row["height"] for row in rows]),
        "support": np.asarray([row["support_fraction"] for row in rows]),
        "height_x_support": np.asarray([row["joint"] for row in rows]),
    }
    metrics = {name: ranking_metrics(labels, values)
               for name, values in metric_values.items()}
    result = {
        "metadata": metadata,
        "implementation": {
            "moving_average_width": "2*a",
            "threshold": "mean + 0.55*(max-mean)",
            "peak_minimum_distance": "a",
            "association_radius": k_p // 2,
            "association": "single-link components over temporally sorted peaks",
            "per_scale_dedup": "nearest to raw component median; tie: higher height, earlier position",
            "representative": "round-half-up median of retained member positions",
            "height": "maximum uncalibrated smoothed peak score among retained members",
            "primary_label": "representative lies inside inclusive GT onset-offset",
            "pr_auc_definition": "sklearn average_precision_score",
        },
        "scale_complementarity": {
            "native_hit_gt": int(native.sum()),
            "union_hit_gt": int(union.sum()),
            "native_missed_gt": int(missed.sum()),
            "native_missed_recovered_by_at_least_1_non_native": int(
                np.sum(missed_support >= 1)),
            "native_missed_support_at_least_2": int(np.sum(missed_support >= 2)),
            "native_missed_support_at_least_3": int(np.sum(missed_support >= 3)),
            "native_missed_support_at_least_4": int(np.sum(missed_support >= 4)),
            "native_missed_support_5": int(np.sum(missed_support == 5)),
        },
        "cluster_counts": {
            "total": len(rows),
            "primary_positive": int(labels.sum()),
            "primary_negative": int(len(labels) - labels.sum()),
            "secondary_iou_positive": sum(row["iou_positive"] for row in rows),
            "mean_support_positive": float(np.mean(
                [row["support"] for row in rows if row["positive"]]
            )),
            "mean_support_negative": float(np.mean(
                [row["support"] for row in rows if not row["positive"]]
            )),
        },
        "support_table": support_table(rows),
        "ranking_metrics": metrics,
        "height_matched": height_matched(rows),
        "support_threshold_curve": threshold_curve(rows),
        "matched_fp": {
            "budget_definition": "native-scale +/-k intervals failing greedy one-to-one IoU>=0.5 matching",
            "native_fp_budget": native_fp_budget,
            "native_decoder_tp": len(native_decoder_matched_gt_ids),
            "native_decoder_fn": len(native_decoder_missed_gt_ids),
            "height": matched_fp_iou(
                rows, "height", native_fp_budget, all_gt_ids,
                native_decoder_missed_gt_ids),
            "height_x_support": matched_fp_iou(
                rows, "joint", native_fp_budget, all_gt_ids,
                native_decoder_missed_gt_ids),
            "primary_label_supplement": {
                "budget_definition": "native-scale peaks outside every GT",
                "native_fp_budget": native_primary_fp_budget,
                "height": matched_fp_primary(
                    rows, "height", native_primary_fp_budget, missed_gt_ids),
                "height_x_support": matched_fp_primary(
                    rows, "joint", native_primary_fp_budget, missed_gt_ids),
            },
        },
        "subject_bootstrap": bootstrap_delta(rows),
    }
    return result, rows


def write_cluster_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset", "subject", "video", "representative", "support",
        "height", "support_fraction", "joint", "positive", "iou_positive",
        "gt_ids", "member_positions", "member_scales",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable.pop("iou_matches")
            for key in ("gt_ids", "member_positions", "member_scales"):
                serializable[key] = json.dumps(serializable[key], ensure_ascii=False)
            writer.writerow(serializable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sammlv-cache", type=Path,
                        default=DEFAULT_CACHES["SAMMLV"])
    parser.add_argument("--casme3-cache", type=Path,
                        default=DEFAULT_CACHES["CASME_3"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {
        "status": "COMPLETE",
        "restrictions": {
            "training": False,
            "model_forward": False,
            "hidden_features": False,
            "recognition_logits": False,
            "mirror_branch": False,
            "gt_used_for_candidate_generation": False,
            "scale_grid_search": False,
        },
        "datasets": {},
    }
    for name, cache in (("SAMMLV", args.sammlv_cache),
                        ("CASME_3", args.casme3_cache)):
        result, rows = audit_dataset(cache)
        all_results["datasets"][name] = result
        write_cluster_csv(args.output_dir / f"{name.lower()}_clusters.csv", rows)
        metrics = result["ranking_metrics"]
        bootstrap = result["subject_bootstrap"]
        print(
            f"{name}: clusters={len(rows)}, "
            f"AP(H)={metrics['height']['pr_auc']:.6f}, "
            f"AP(M)={metrics['support']['pr_auc']:.6f}, "
            f"AP(H*M)={metrics['height_x_support']['pr_auc']:.6f}, "
            f"delta_CI={bootstrap['ci95']}"
        )
    output = args.output_dir / "scale_consistency_results.json"
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
