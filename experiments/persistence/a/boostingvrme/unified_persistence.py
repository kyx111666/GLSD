"""Unified, height-gate-free persistence on frozen subject-LOSO curves.

The same grid and scoring rule serve both backbones and datasets. Backbone
interval decoders are preserved; no inner-fold backbone is retrained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences

import equiscale_fair_validation as fair


SCALES = (1.0, 1.5, 2.0)
REFERENCE_SCALES = (1.0, 1.5, 2.0)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)
HEIGHT_WEIGHTS = (0.0, 0.25, 0.5, 0.75)
LOCAL_RADII = (1.0, 2.0, 3.0)
RESULT_NAME = "unified_persistence_final"


@dataclass(frozen=True)
class Config:
    family: str
    reference: float
    height_weight: float
    threshold: float
    radius: float = 1.0

    @property
    def identifier(self):
        return (f"{self.family}|scale={self.reference:g}"
                f"|height={self.height_weight:g}|tau={self.threshold:g}|radius={self.radius:g}")


def configuration_grid():
    configs = [Config("native", 2.0, 1.0, 0.55)]
    for reference in REFERENCE_SCALES:
        for radius in LOCAL_RADII:
            for weight in HEIGHT_WEIGHTS:
                configs.extend(Config("unified", reference, weight, tau, radius) for tau in THRESHOLDS)
            configs.extend(Config("local", reference, 0.0, tau, radius) for tau in THRESHOLDS)
        for family in ("height", "global"):
            configs.extend(Config(family, reference, 1.0 if family == "height" else 0.0, tau)
                           for tau in THRESHOLDS)
    return configs


class CurveFeatures:
    """Label-free candidate generation and evidence calculation."""

    def __init__(self, curve, k):
        self.curve = np.asarray(curve, dtype=float)
        if self.curve.ndim != 1 or len(self.curve) < 3 or not np.isfinite(self.curve).all():
            raise ValueError("Expected a finite one-dimensional curve with at least three samples")
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.scales = {}
        self.local_cache = {}
        self.evidence_cache = {}
        physical = {}
        for scale in SCALES:
            width = min(len(self.curve), max(1, round(scale * k)))
            if width not in physical:
                smooth = np.convolve(self.curve, np.ones(width) / width, mode="same")
                peaks = (find_peaks(smooth, distance=k)[0] if np.ptp(self.curve) > 0
                         else np.empty(0, dtype=int))
                spread = max(float(np.ptp(smooth)), 1e-12)
                mean = float(smooth.mean())
                height = np.maximum(0.0, (smooth[peaks] - mean) / max(float(smooth.max() - mean), 1e-12))
                global_values = peak_prominences(smooth, peaks)[0] / spread
                physical[width] = (peaks, height, global_values, smooth, spread)
            self.scales[scale] = physical[width]
        self.effective_scales = physical

    def evidence(self, reference, radius=1.0):
        if (reference, radius) in self.evidence_cache:
            return self.evidence_cache[reference, radius]
        peaks, height, global_values, _, _ = self.scales[reference]
        tolerance = max(1, round(0.5 * self.k))
        window = max(1, round(radius * self.k))
        for width, (other_peaks, _, _, smooth, spread) in self.effective_scales.items():
            if (width, window) not in self.local_cache:
                self.local_cache[width, window] = np.array([
                    max(0.0, float(smooth[p]) - max(float(np.min(smooth[max(0, p-window):p+1])),
                                                  float(np.min(smooth[p:min(len(smooth), p+window+1)])))) / spread
                    for p in other_peaks
                ])
        local = []
        for point in peaks:
            aligned = []
            for width, (other_peaks, _, _, _, _) in self.effective_scales.items():
                values = self.local_cache[width, window]
                indexes = np.flatnonzero(np.abs(other_peaks - point) <= tolerance)
                if not len(indexes):
                    aligned.append(0.0)
                    continue
                chosen = min(indexes, key=lambda i: (abs(int(other_peaks[i]) - point), -values[i]))
                aligned.append(float(values[chosen]))
            local.append(float(np.median(aligned)))
        output = peaks, np.column_stack((height, global_values, local))
        self.evidence_cache[reference, radius] = output
        return output


def evidence_scores(evidence, config):
    height, global_values, local = evidence.T
    if config.family in ("native", "height"):
        return height
    if config.family == "global":
        return global_values
    if config.family == "local":
        return local
    if config.family != "unified":
        raise ValueError(f"Unknown family: {config.family}")
    return config.height_weight * height + (1 - config.height_weight) * 0.5 * (global_values + local)


class Prepared(fair.Prepared):
    def __init__(self, decoder, peaks, evidence):
        base = decoder.prepare([(int(p), 1, 0.0, 0.0, 0.0) for p in peaks])
        super().__init__(base.clusters, base.intervals, base.conflicts, base.best, base.num_gt)
        lookup = {int(p): i for i, p in enumerate(peaks)}
        indexes = [lookup[c[0]] for c in self.clusters]
        self.evidence = evidence[indexes]

    def masks(self, configs):
        return np.array([evidence_scores(self.evidence, config) >= config.threshold for config in configs])


def prepare(record, features, reference, backbone, intervals, radius=1.0):
    decoder = fair.VideoFeatures(record, features.k, backbone if intervals == "native" else "metst")
    return Prepared(decoder, *features.evidence(reference, radius))


def detect_curve(curve, k, config, backbone="boostingvrme", intervals="native"):
    record = {"curve": np.asarray(curve, dtype=float), "gt": []}
    features = CurveFeatures(curve, k)
    _, predictions = prepare(record, features, config.reference, backbone, intervals, config.radius).evaluate([config], details=True)
    return [{key: value for key, value in row.items() if key != "matched_gt"} for row in predictions[0]]


def choose(counts, indexes):
    indexes = np.asarray(indexes, dtype=int)
    tp, fp, fn = counts[indexes].astype(float).T
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return int(indexes[np.lexsort((indexes, fp, -precision, -f1))[0]])


def inner_counts(stats, held, inner_k):
    sample = stats[next(iter(stats))]
    total = np.zeros((sample.shape[0], 3), dtype=np.int64)
    for validation in range(len(inner_k)):
        if validation != held:
            total += stats[int(inner_k[held, validation])][:, validation]
    return total


def selection_pools(configs):
    return {
        "native": [0],
        "unified": [i for i, c in enumerate(configs) if c.family == "unified"],
        "persistence_only": [i for i, c in enumerate(configs) if c.family == "unified" and c.height_weight == 0],
        "height_only": [i for i, c in enumerate(configs) if c.family == "height"],
        "global_only": [i for i, c in enumerate(configs) if c.family == "global"],
        "local_only": [i for i, c in enumerate(configs) if c.family == "local"],
    }


def evaluate_k(records, subjects, backbone, k, needed, configs, modes):
    stats = {mode: np.zeros((len(configs), len(subjects), 3), dtype=np.int32) for mode in modes}
    subject_ids = {subject: i for i, subject in enumerate(subjects)}
    batches = {(reference, radius): [i for i, c in enumerate(configs) if (c.reference, c.radius) == (reference, radius)]
               for reference in REFERENCE_SCALES for radius in LOCAL_RADII}
    for vid, record in enumerate(records):
        sid = subject_ids[record["subject"]]
        if not needed[sid]:
            continue
        features = CurveFeatures(record["curve"], k)
        for (reference, radius), indexes in batches.items():
            selected = [configs[i] for i in indexes]
            for mode in modes:
                stats[mode][indexes, sid] += prepare(record, features, reference, backbone, mode, radius).evaluate(selected)
        if (vid + 1) % 100 == 0:
            fair.log(f"[{backbone}] k={k}: {vid+1}/{len(records)} videos")
    return stats


def select_and_replay(records, subjects, backbone, mode, configs, stats, outer_k, inner_k, target):
    pools = selection_pools(configs)
    counts = {name: np.zeros((len(subjects), 3), dtype=np.int32) for name in pools}
    choices = {name: [] for name in pools}
    rows = []
    for held, subject in enumerate(subjects):
        train = inner_counts(stats, held, inner_k)
        for name, indexes in pools.items():
            selected = choose(train, indexes)
            choices[name].append(selected)
            counts[name][held] = stats[int(outer_k[held])][selected, held]
            rows.append({"family": name, "subject": subject, "k_train_only": int(outer_k[held]),
                         "config_id": selected, "config": configs[selected].identifier,
                         "inner_F1": fair.metrics(train[selected])["F1"], **fair.metrics(counts[name][held])})
    fair.csv_write(target / "outer_loso_selections.csv", rows)
    predictions = {name: [] for name in pools}
    replay = {name: np.zeros_like(counts[name]) for name in pools}
    videos = []
    subject_ids = {subject: i for i, subject in enumerate(subjects)}
    for record in records:
        sid = subject_ids[record["subject"]]
        features = CurveFeatures(record["curve"], int(outer_k[sid]))
        prepared = {}
        for name in pools:
            config = configs[choices[name][sid]]
            key = config.reference, config.radius
            if key not in prepared:
                prepared[key] = prepare(record, features, config.reference, backbone, mode, config.radius)
            values, details = prepared[key].evaluate([config], details=True)
            replay[name][sid] += values[0]
            predictions[name].append(details[0])
        videos.append({"subject": record["subject"], "video": record["video"], "gt": record["gt"],
                       "predictions": {name: predictions[name][-1] for name in pools}})
    for name in pools:
        if not np.array_equal(replay[name], counts[name]):
            raise AssertionError(f"Replay mismatch: {name}")
    fair.json_write(target / "selected_predictions.json", videos)
    fair.csv_write(target / "subject_counts.csv", [{"family": name, "subject": subject, **fair.metrics(counts[name][i])}
                                                   for name in pools for i, subject in enumerate(subjects)])
    comparisons = {}
    pairs = [(name, "native") for name in pools if name != "native"]
    pairs += [("unified", reference) for reference in ("height_only", "global_only", "persistence_only")]
    for name, reference in pairs:
        changes = fair.event_changes(predictions[name], predictions[reference])
        delta = counts[name].sum(axis=0) - counts[reference].sum(axis=0)
        if changes["net_TP"] != delta[0] or changes["net_FP"] != delta[1]:
            raise AssertionError("Event accounting mismatch")
        comparisons[f"{name}_vs_{reference}"] = {**fair.paired_bootstrap(counts[name], counts[reference]), **changes}
    return {"metrics": {name: fair.metrics(value.sum(axis=0)) for name, value in counts.items()},
            "comparisons": comparisons,
            "selection_frequency": {name: dict(Counter(configs[i].identifier for i in indexes)) for name, indexes in choices.items()},
            "verification": {"prediction_replay": True, "event_accounting": True,
                             "outer_and_inner_duration_exclusion": True}}


def protocol_spec(configs):
    return {"version": RESULT_NAME, "score": "w*H + (1-w)*(G+L)/2",
            "H": "max(0, (peak-mean)/(max-mean)) on reference curve; continuous evidence, no height gate",
            "G": "global prominence / reference smoothed curve range",
            "L": "median of aligned local prominences / respective smoothed ranges; missing scale is zero",
            "scales": SCALES, "reference_scales": REFERENCE_SCALES, "local_radius_multipliers": LOCAL_RADII,
            "alignment_tolerance": "max(1, round(0.5*k))", "peak_distance": "k",
            "scale_deduplication": "one vote per distinct integer smoothing width",
            "constant_curve_policy": "no candidates; avoid artificial zero-padding plateaus",
            "routing": False, "height_candidate_gate": False, "global_peak_snapping": False,
            "height_weights": HEIGHT_WEIGHTS, "thresholds": THRESHOLDS,
            "configurations": [asdict(c) for c in configs], "primary_family": "unified",
            "selection": "aggregate inner-subject raw F1; ties precision, fewer FP, then grid order",
            "outer_k": "training subjects only", "inner_k": "exclude outer test and inner validation subjects",
            "intervals": "ME-TST fixed; BoostingVRME native expansion/NMS primary, fixed secondary",
            "limitations": ["Exploratory design using previously examined datasets.",
                            "Frozen backbone OOF caches; not fully nested backbone retraining.",
                            "Raw spotting only; existing BoostingVRME reference-kernel cache.",
                            "Paired bootstrap conditions on selected configurations and frozen curves."]}


def run(backbone, dataset, result_name=RESULT_NAME):
    records, subjects, legacy_k, cache_path = fair.load_data(backbone, dataset)
    outer_k, inner_k = fair.fold_priors(records, subjects, backbone)
    legacy, native_train = fair.baseline_audit(records, subjects, backbone, legacy_k, outer_k)
    if tuple(legacy) != fair.EXPECTED[backbone, dataset]:
        raise AssertionError("Original baseline did not reproduce")
    configs = configuration_grid()
    modes = ("native", "fixed") if backbone == "boostingvrme" else ("fixed",)
    target = fair.WORKSPACE / backbone / "results" / result_name / dataset
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": protocol_spec(configs), "backbone": backbone, "dataset": dataset,
                "input_path": str(cache_path), "input_sha256": fair.digest(cache_path),
                "source_sha256": {name: fair.digest(Path(__file__).with_name(name)) for name in
                                  ("unified_persistence.py", "equiscale_fair_validation.py", "tune_equiscale.py")}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["signature"] = signature
    manifest_path = target / "PROTOCOL.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise RuntimeError("Protocol/code changed. Use --result-name to preserve the prior experiment.")
    fair.json_write(manifest_path, manifest)
    fair.json_write(target / "fold_duration_priors.json", {"subjects": subjects, "outer": outer_k.tolist(), "inner": inner_k.tolist()})
    stats = {mode: {} for mode in modes}
    for k in sorted(set(outer_k.tolist()) | set(inner_k[inner_k > 0].tolist())):
        needed = fair.required_subjects(k, outer_k, inner_k)
        path = target / f"stats_k{k}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cache:
                if str(cache["signature"]) != signature or not np.array_equal(cache["needed"], needed):
                    raise AssertionError("Stale count cache")
                values = {mode: cache[mode] for mode in modes}
        else:
            fair.log(f"[{backbone}/{dataset}] k={k}: {len(configs)} configurations")
            values = evaluate_k(records, subjects, backbone, k, needed, configs, modes)
            np.savez_compressed(path, **values, needed=needed, signature=np.array(signature))
        for mode in modes:
            stats[mode][k] = values[mode]
    reports = {}
    for mode in modes:
        mode_target = target / mode
        mode_target.mkdir(exist_ok=True)
        report = select_and_replay(records, subjects, backbone, mode, configs, stats[mode], outer_k, inner_k, mode_target)
        primary = mode == modes[0]
        if primary and report["metrics"]["native"] != fair.metrics(native_train.sum(axis=0)):
            raise AssertionError("Training-fold baseline mismatch")
        report.update({"backbone": backbone, "dataset": dataset, "intervals": mode, "primary_intervals": primary,
                       "legacy_baseline": fair.metrics(legacy), "manifest_signature": signature,
                       "outer_k_frequency": dict(Counter(str(k) for k in outer_k)),
                       "inner_k_frequency": dict(Counter(str(k) for k in inner_k[inner_k > 0]))})
        fair.json_write(mode_target / "report.json", report)
        reports[mode] = report
        fair.log(f"[{backbone}/{dataset}/{mode}] " + str({name: round(m["F1"], 6) for name, m in report["metrics"].items()}))
    return reports


def main(default_backbone="boostingvrme"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("metst", "boostingvrme", "both"), default=default_backbone)
    parser.add_argument("--dataset", choices=("sammlv", "casme3", "both"), default="both")
    parser.add_argument("--result-name", default=RESULT_NAME)
    args = parser.parse_args()
    if Path(args.result_name).name != args.result_name or args.result_name in (".", ".."):
        parser.error("--result-name must be a directory name, not a path")
    for backbone in (("boostingvrme", "metst") if args.backbone == "both" else (args.backbone,)):
        for dataset in (("sammlv", "casme3") if args.dataset == "both" else (args.dataset,)):
            run(backbone, dataset, args.result_name)


if __name__ == "__main__":
    main()
