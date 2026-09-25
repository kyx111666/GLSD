"""Fixed equal-weight persistence, matched ablations, and threshold sensitivity.

Reuse signed count caches from the prior unified experiment. No previous code
or result is changed, and no threshold perturbation is used to select a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import equiscale_fair_validation as fair
import scheme11_persistence_selector as reference
import unified_persistence as engine


RESULT_NAME = "pure_persistence_matched_v1"
OFFSETS = (-0.1, -0.05, 0.0, 0.05, 0.1)
METHODS = ("original_native", "native", "pure", "global_tuned", "local_tuned",
           "global_at_pure_settings", "local_at_pure_settings",
           "fusion_at_global_candidates", "local_at_global_candidates")


@dataclass(frozen=True)
class PersistenceConfig:
    reference: float
    radius: float
    threshold: float

    def as_engine_config(self):
        return engine.Config("unified", self.reference, 0.0, self.threshold, self.radius)


def detect_curve(curve, k, config, backbone="boostingvrme", intervals="native"):
    return engine.detect_curve(curve, k, config.as_engine_config(), backbone, intervals)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed_pools(configs):
    return {"pure": [i for i, c in enumerate(configs) if c.family == "unified" and c.height_weight == 0],
            "global": [i for i, c in enumerate(configs) if c.family == "global"],
            "local": [i for i, c in enumerate(configs) if c.family == "local"]}


def selected_configs(train, configs):
    pools = indexed_pools(configs)
    lookup = {c: i for i, c in enumerate(configs)}
    pure_id = engine.choose(train, pools["pure"])
    global_id = engine.choose(train, pools["global"])
    pure, global_config = configs[pure_id], configs[global_id]
    same_reference = lambda pool: [i for i in pool if configs[i].reference == global_config.reference]
    return {
        "original_native": 0, "native": 0, "pure": pure_id, "global_tuned": global_id,
        "local_tuned": engine.choose(train, pools["local"]),
        "global_at_pure_settings": lookup[engine.Config("global", pure.reference, 0.0, pure.threshold)],
        "local_at_pure_settings": lookup[engine.Config("local", pure.reference, 0.0, pure.threshold, pure.radius)],
        "fusion_at_global_candidates": engine.choose(train, same_reference(pools["pure"])),
        "local_at_global_candidates": engine.choose(train, same_reference(pools["local"])),
    }


def perturb(config, offset):
    return replace(config, threshold=max(0.0, round(config.threshold + offset, 10)))


def resolved_config(name, selected, configs):
    config = configs[selected[name]]
    if name == "global_at_pure_settings":
        config = replace(config, radius=configs[selected["pure"]].radius)
    return config


def overlap_matrix(intervals, gt):
    events = np.asarray(gt, dtype=int).reshape(-1, 3)
    intervals = np.asarray(intervals, dtype=int).reshape(-1, 2)
    left = np.maximum(intervals[:, 0, None], events[:, 0])
    right = np.minimum(intervals[:, 1, None], events[:, 2])
    intersection = np.maximum(0, right - left + 1)
    union = (intervals[:, 1] - intervals[:, 0] + 1)[:, None] + events[:, 2] - events[:, 0] + 1 - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)


def error_audit(record, pure, native, prepared, config):
    rows = []
    pure_ids = {p["matched_gt"] for p in pure if p["matched_gt"] >= 0}
    native_ids = {p["matched_gt"] for p in native if p["matched_gt"] >= 0}
    overlaps = overlap_matrix(prepared.intervals, record["gt"])
    scores = engine.evidence_scores(prepared.evidence, config)
    mask = scores >= config.threshold
    base = {"subject": record["subject"], "video": record["video"], "threshold": config.threshold,
            "reference": config.reference, "radius": config.radius}
    native_fp = {(p["onset"], p["peak"], p["offset"]) for p in native if p["matched_gt"] < 0}
    point_index = {c[0]: i for i, c in enumerate(prepared.clusters)}
    for gt in sorted(native_ids - pure_ids):
        eligible = overlaps[:, gt] >= 0.5
        if not eligible.any():
            reason = "candidate_or_interval_miss"
        elif not (eligible & mask).any():
            reason = "score_rejected"
        else:
            reason = "suppression_or_greedy_matching"
        candidates = np.flatnonzero(eligible)
        best = int(candidates[np.argmax(scores[candidates])]) if len(candidates) else -1
        rows.append({**base, "kind": "lost_gt", "reason": reason, "gt_index": gt,
                     "onset": record["gt"][gt][0], "peak": record["gt"][gt][1], "offset": record["gt"][gt][2],
                     "best_iou": float(overlaps[:, gt].max()) if len(overlaps) else 0.0,
                     "G": float(prepared.evidence[best, 1]) if best >= 0 else None,
                     "L": float(prepared.evidence[best, 2]) if best >= 0 else None,
                     "score": float(scores[best]) if best >= 0 else None})
    for pred in pure:
        if pred["matched_gt"] >= 0 or (pred["onset"], pred["peak"], pred["offset"]) in native_fp:
            continue
        index = point_index[pred["peak"]]
        best_iou = float(overlaps[index].max()) if overlaps.shape[1] else 0.0
        reason = ("matching_conflict" if best_iou >= 0.5 else
                  "partial_overlap" if best_iou >= 0.1 else "low_overlap")
        rows.append({**base, "kind": "new_fp", "reason": reason, "gt_index": -1,
                     "onset": pred["onset"], "peak": pred["peak"], "offset": pred["offset"],
                     "best_iou": best_iou, "G": float(prepared.evidence[index, 1]),
                     "L": float(prepared.evidence[index, 2]), "score": float(scores[index])})
    return rows


def verify_predictions(predictions, record, counts):
    tuples = [(p["onset"], p["offset"], p["peak"]) for p in predictions]
    expected = reference.count_video(tuples, record)
    np.testing.assert_array_equal(counts, (expected.tp, expected.fp, expected.fn))
    matched = set()
    for prediction, interval in zip(predictions, tuples):
        ious = [reference.interval_iou(interval, gt) for gt in record["gt"]]
        gt = int(np.argmax(ious)) if ious else -1
        found = gt if gt >= 0 and ious[gt] >= 0.5 and gt not in matched else -1
        assert prediction["matched_gt"] == found
        if found >= 0:
            matched.add(found)


def load_signed_cache(backbone, dataset, cache_path, outer, inner):
    root = fair.WORKSPACE / backbone / "results" / engine.RESULT_NAME / dataset
    manifest = load_json(root / "PROTOCOL.json")
    signed = {key: value for key, value in manifest.items() if key != "signature"}
    assert hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest() == manifest["signature"]
    assert fair.digest(cache_path) == manifest["input_sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert fair.digest(Path(__file__).with_name(name)) == digest
    configs = [engine.Config(**row) for row in manifest["protocol"]["configurations"]]
    assert configs == engine.configuration_grid()
    modes = ("native", "fixed") if backbone == "boostingvrme" else ("fixed",)
    stats = {mode: {} for mode in modes}
    for k in sorted(set(outer.tolist()) | set(inner[inner > 0].tolist())):
        with np.load(root / f"stats_k{k}.npz", allow_pickle=False) as cached:
            assert str(cached["signature"]) == manifest["signature"]
            np.testing.assert_array_equal(cached["needed"], fair.required_subjects(k, outer, inner))
            for mode in modes:
                stats[mode][k] = cached[mode]
    return root, manifest, configs, stats


def protocol_spec():
    return {"version": RESULT_NAME, "primary_score": "(G+L)/2", "height_weight": 0,
            "primary_search_size": 90, "same_grid_for_every_backbone_and_dataset": True,
            "scales": engine.REFERENCE_SCALES, "radii": engine.LOCAL_RADII, "thresholds": engine.THRESHOLDS,
            "primary_local_contribution": "fusion_at_global_candidates vs global_tuned",
            "candidate_anchor": "select reference and threshold using G on inner subjects; freeze reference for matched comparisons",
            "matched_retuning": "L and G+L may retune radius/threshold on inner subjects using the identical G-anchored candidate pool",
            "exact_drop_one": "freeze primary G+L reference/radius/threshold and substitute G or L without reselection; calibration-sensitive diagnostic",
            "global_radius": "G is independent of local radius; cache uses canonical radius=1",
            "threshold_offsets": OFFSETS, "threshold_perturbation": "freeze selected config; clamp tau+offset at zero and round to ten decimals; never choose offset using test results",
            "baseline": "original native and train-k native both replayed and paired by subject",
            "validation": "outer/inner subjects excluded from duration and selection; frozen backbone OOF cache remains exploratory",
            "error_audit": "numeric candidate/score/suppression and interval-overlap categories; not semantic classification of video content"}


def run(backbone, dataset):
    records, subjects, legacy_k, input_path = fair.load_data(backbone, dataset)
    outer, inner = fair.fold_priors(records, subjects, backbone)
    prior_root, prior_manifest, configs, all_stats = load_signed_cache(backbone, dataset, input_path, outer, inner)
    target = fair.WORKSPACE / backbone / "results" / RESULT_NAME / dataset
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": protocol_spec(), "prior_manifest_signature": prior_manifest["signature"],
                "backbone": backbone, "dataset": dataset, "input_path": str(input_path),
                "input_sha256": fair.digest(input_path), "engine_sha256": fair.digest(__file__)}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["signature"] = signature
    path = target / "PROTOCOL.json"
    if path.exists() and load_json(path)["signature"] != signature:
        raise RuntimeError("Changed protocol or code; preserve this experiment and use a new result version")
    fair.json_write(path, manifest)
    sid_lookup = {subject: i for i, subject in enumerate(subjects)}
    for mode, stats in all_stats.items():
        output = target / mode
        output.mkdir(exist_ok=True)
        choices, selection_rows = [], []
        for held, subject in enumerate(subjects):
            train = engine.inner_counts(stats, held, inner)
            selected = selected_configs(train, configs)
            choices.append(selected)
            for name in METHODS:
                config = resolved_config(name, selected, configs)
                selection_rows.append({"family": name, "subject": subject, "config_id": selected[name],
                                       "config": config.identifier, "k": legacy_k if name == "original_native" else int(outer[held]),
                                       "inner_F1": None if name == "original_native" else fair.metrics(train[selected[name]])["F1"]})
        totals = {name: np.zeros((len(subjects), 3), dtype=int) for name in METHODS}
        sensitivity_counts = {offset: np.zeros((len(subjects), 3), dtype=int) for offset in OFFSETS}
        predictions = {name: [] for name in METHODS}
        sensitivity_predictions = {offset: [] for offset in OFFSETS}
        video_rows, errors = [], []
        for record in records:
            sid = sid_lookup[record["subject"]]
            chosen = choices[sid]
            k = int(outer[sid])
            features = {k: engine.CurveFeatures(record["curve"], k)}
            prepared_cache = {}
            for name in METHODS:
                config = resolved_config(name, chosen, configs)
                current_k = legacy_k if name == "original_native" else k
                key = current_k, config.reference, config.radius
                if current_k not in features:
                    features[current_k] = engine.CurveFeatures(record["curve"], current_k)
                if key not in prepared_cache:
                    prepared_cache[key] = engine.prepare(record, features[current_k], config.reference, backbone, mode, config.radius)
                prepared = prepared_cache[key]
                counts, details = prepared.evaluate([config], details=True)
                verify_predictions(details[0], record, counts[0])
                totals[name][sid] += counts[0]
                predictions[name].append(details[0])
            for names in (("pure", "global_at_pure_settings", "local_at_pure_settings"),
                          ("global_tuned", "fusion_at_global_candidates", "local_at_global_candidates")):
                prepared_group = [prepared_cache[k, configs[chosen[name]].reference,
                                               configs[chosen["pure"]].radius if name == "global_at_pure_settings" else configs[chosen[name]].radius]
                                  for name in names]
                for prepared in prepared_group[1:]:
                    np.testing.assert_array_equal(prepared_group[0].intervals, prepared.intervals)
                    np.testing.assert_array_equal(prepared_group[0].conflicts, prepared.conflicts)
                    assert prepared_group[0].clusters == prepared.clusters
            pure_config = configs[chosen["pure"]]
            pure_prepared = prepared_cache[k, pure_config.reference, pure_config.radius]
            perturbed = [perturb(pure_config, offset) for offset in OFFSETS]
            values, details = pure_prepared.evaluate(perturbed, details=True)
            for i, offset in enumerate(OFFSETS):
                verify_predictions(details[i], record, values[i])
                sensitivity_counts[offset][sid] += values[i]
                sensitivity_predictions[offset].append(details[i])
            errors.extend(error_audit(record, predictions["pure"][-1], predictions["original_native"][-1], pure_prepared, pure_config))
            video_rows.append({"subject": record["subject"], "video": record["video"], "gt": record["gt"],
                               "predictions": {name: predictions[name][-1] for name in METHODS},
                               "sensitivity_predictions": {str(offset): sensitivity_predictions[offset][-1] for offset in OFFSETS}})
        for sid, selected in enumerate(choices):
            for name in METHODS:
                if name != "original_native":
                    np.testing.assert_array_equal(totals[name][sid], stats[int(outer[sid])][selected[name], sid])
        np.testing.assert_array_equal(sensitivity_counts[0.0], totals["pure"])
        prior = load_json(prior_root / mode / "report.json")
        assert fair.metrics(totals["pure"].sum(axis=0)) == prior["metrics"]["persistence_only"]
        assert fair.metrics(totals["native"].sum(axis=0)) == prior["metrics"]["native"]
        if (backbone == "metst" or mode == "native"):
            assert tuple(totals["original_native"].sum(axis=0)) == fair.EXPECTED[backbone, dataset]
        pairs = [("pure", name) for name in ("original_native", "native", "global_tuned", "local_tuned",
                                              "global_at_pure_settings", "local_at_pure_settings")]
        pairs += [(name, "global_tuned") for name in ("fusion_at_global_candidates", "local_at_global_candidates")]
        comparisons = {}
        for first, second in pairs:
            changes = fair.event_changes(predictions[first], predictions[second])
            diff = totals[first].sum(axis=0) - totals[second].sum(axis=0)
            assert changes["net_TP"] == diff[0] and changes["net_FP"] == diff[1]
            comparisons[first + "_vs_" + second] = {**fair.paired_bootstrap(totals[first], totals[second]), **changes}
        sensitivity = {str(offset): {"metrics": fair.metrics(values.sum(axis=0)),
                                    "vs_original_native": fair.paired_bootstrap(values, totals["original_native"]),
                                    "vs_selected_threshold": fair.paired_bootstrap(values, totals["pure"])}
                       for offset, values in sensitivity_counts.items()}
        report = {"backbone": backbone, "dataset": dataset, "intervals": mode,
                  "primary_intervals": mode == ("native" if backbone == "boostingvrme" else "fixed"),
                  "manifest_signature": signature, "metrics": {name: fair.metrics(value.sum(axis=0)) for name, value in totals.items()},
                  "comparisons": comparisons, "threshold_sensitivity": sensitivity,
                  "selection_frequency": {name: dict(Counter(resolved_config(name, choice, configs).identifier for choice in choices)) for name in METHODS},
                  "error_audit_counts": dict(Counter(row["kind"] + ":" + row["reason"] for row in errors)),
                  "verification": {"signed_prior_counts_reused": True, "all_predictions_independently_recounted": True,
                                   "exact_matched_candidates_intervals_and_NMS": True, "pure_matches_previous_ablation": True,
                                   "threshold_perturbations_not_selected": True}}
        fair.json_write(output / "report.json", report)
        fair.json_write(output / "selected_predictions.json", video_rows)
        fair.json_write(output / "error_audit.json", errors)
        if errors:
            fair.csv_write(output / "error_audit.csv", errors)
        fair.csv_write(output / "outer_loso_selections.csv", selection_rows)
        fair.csv_write(output / "subject_counts.csv", [{"family": name, "subject": subject, **fair.metrics(totals[name][sid])}
                                                       for name in METHODS for sid, subject in enumerate(subjects)])
        fair.csv_write(output / "threshold_sensitivity.csv", [{"offset": offset, "subject": subject, **fair.metrics(sensitivity_counts[offset][sid])}
                                                              for offset in OFFSETS for sid, subject in enumerate(subjects)])
        fair.log(f"[{backbone}/{dataset}/{mode}] " + str({name: round(value["F1"], 6) for name, value in report["metrics"].items()}))


def main(default_backbone="boostingvrme"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("metst", "boostingvrme", "both"), default=default_backbone)
    parser.add_argument("--dataset", choices=("sammlv", "casme3", "both"), default="both")
    args = parser.parse_args()
    for backbone in (("metst", "boostingvrme") if args.backbone == "both" else (args.backbone,)):
        for dataset in (("sammlv", "casme3") if args.dataset == "both" else (args.dataset,)):
            run(backbone, dataset)


if __name__ == "__main__":
    main()
