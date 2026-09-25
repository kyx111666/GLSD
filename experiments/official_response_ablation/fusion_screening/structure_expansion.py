"""Reference-scale extension with the sealed local support scales preserved."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences

import run_full_fusion_tuning as full

EXPANDED_SCALES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
EXPANDED_RADII = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
REFERENCE_FILE = Path(__file__).with_name("refined_reference_sammlv.json")


def counts_hash(table):
    return hashlib.sha256(np.asarray(table, dtype="<i8").tobytes(order="C")).hexdigest()


class ExtendedFeatures:
    def __init__(self, base, response, k, scales):
        self.sealed = base.GLSDFeatures(response, k)
        self.physical_evidence = {}
        # Add references to scale_data only. effective_scales remains exactly the
        # sealed three-scale (integer-width-deduplicated) set used for median L.
        physical = dict(self.sealed.effective_scales)
        for scale in scales:
            width = min(len(self.sealed.response), max(1, round(scale * self.sealed.k)))
            if width not in physical:
                smooth = np.convolve(self.sealed.response, np.ones(width, dtype=float) / width, mode="same")
                peaks = (find_peaks(smooth, distance=self.sealed.k)[0]
                         if np.ptp(self.sealed.response) > 0 else np.empty(0, dtype=int))
                spread = max(float(np.ptp(smooth)), 1e-12)
                physical[width] = (peaks, smooth, spread, peak_prominences(smooth, peaks)[0] / spread)
            if scale not in self.sealed.scale_data:
                self.sealed.scale_data[scale] = physical[width]

    def __getattr__(self, name):
        return getattr(self.sealed, name)

    def evidence(self, reference, radius):
        width = min(len(self.response), max(1, round(reference * self.k)))
        window = max(1, round(radius * self.k))
        key = width, window
        if key not in self.physical_evidence:
            self.physical_evidence[key] = self.sealed.evidence(reference, radius)
        return self.physical_evidence[key]

    def selected_peaks(self, config):
        # Independent sealed mean implementation, including at new references.
        return self.sealed.selected_peaks(config)


class ExpandedCore:
    def __init__(self, base, scales):
        if tuple(base.SCALES) != full.LEGACY_SCALES:
            raise RuntimeError("Unexpected sealed support scales")
        self.base, self.scales, self.cache = base, tuple(scales), {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest()
        if key not in self.cache:
            self.cache[key] = ExtendedFeatures(self.base, response, k, self.scales)
        return self.cache[key]


def subset_indexes(grid, variant):
    return [i for i, c in enumerate(grid)
            if (variant == "a0_only" or c.reference_scale in full.LEGACY_SCALES)
            and (variant == "rho_only" or c.local_radius in full.LEGACY_RADII)]


class StructureAudit:
    """Re-select in each structural subset without consulting held-out counts."""
    def __init__(self, setting, output, subjects):
        self.setting, self.output, self.subjects = setting, output, subjects
        self.summaries, self.rows, self.selections = [], [], []
        self.reference = None
        if setting == "metst_sammlv":
            self.reference = json.loads(REFERENCE_FILE.read_text())
            if list(map(str, subjects)) != self.reference["subjects"]:
                raise RuntimeError("Previous refined run subject order differs")
        self.replay_checks = []

    def evaluate(self, method, grid, table, context, core):
        for variant in ("original", "a0_only", "rho_only"):
            indexes = subset_indexes(grid, variant)
            subset = table[indexes]
            if variant == "original" and self.reference:
                expected = self.reference["methods"][method]
                if counts_hash(subset) != expected["raw_counts_sha256"]:
                    raise RuntimeError(f"Previous refined search counts differ: {method}")
            raw_rows, final_rows = [], []
            with gzip.open(self.output / "structure_subset_predictions.jsonl.gz", "at", encoding="utf-8") as handle:
                for si, subject in enumerate(self.subjects):
                    winner, pooled = full.shared.choose(subset, si)
                    config = grid[indexes[winner]]
                    raw, final, predictions, _ = full.decode_with_context(
                        context, core, si, subject, config, True, self.setting, method)
                    if tuple(subset[winner, si]) != raw:
                        raise RuntimeError("Structural subset held-out replay mismatch")
                    if variant == "original" and self.reference:
                        expected = self.reference["methods"][method]["selected"][str(subject)]
                        values = [config.reference_scale, config.local_radius, config.threshold, config.alpha, config.beta]
                        if values != expected["parameters"] or list(raw) != expected["raw"] or list(final) != expected["full"]:
                            raise RuntimeError(f"Previous refined selected result differs: {method}/{subject}")
                    raw_rows.append(raw)
                    final_rows.append(final)
                    common = dict(setting=self.setting, variant=variant, method=method, subject=subject)
                    self.selections.append(dict(**common, config_id=config.config_id,
                        reference_scale=config.reference_scale, local_radius=config.local_radius,
                        threshold=config.threshold, alpha=config.alpha, beta=config.beta,
                        inner_subject_count=len(self.subjects)-1,
                        inner_raw_F1=full.shared.metrics(pooled[winner])["F1"]))
                    for stage, counts in (("raw", raw), ("full", final)):
                        self.rows.append(dict(**common, metric_stage=stage, **full.shared.metrics(counts)))
                    handle.write(json.dumps(full.shared.serializable(dict(**common,
                        raw_counts=raw, full_counts=final, predictions=predictions))) + "\n")
            for stage, counts in (("raw", raw_rows), ("full", final_rows)):
                self.summaries.append(dict(setting=self.setting, variant=variant, method=method,
                    metric_stage=stage, configs=len(indexes), **full.shared.metrics(np.sum(counts, axis=0))))
            if variant == "original" and self.reference:
                self.replay_checks.append(dict(method=method, all_search_counts_equal=True,
                    all_selected_parameters_equal=True, all_selected_raw_full_counts_equal=True))
                full.shared.write_json(self.output / "previous_refined_replay.json", self.replay_checks)
                print(f"PREVIOUS_REFINED_REPLAY {method} = PASS", flush=True)
        self.save()

    def save(self):
        for name, rows in (("summary", self.summaries), ("per_subject_counts", self.rows),
                           ("selected_configs", self.selections)):
            full.shared.write_csv(self.output / f"structure_subset_{name}.csv", rows)

    def finish(self, summaries, rows, selections, feature_base):
        self.summaries.extend(dict(variant="expanded", **row) for row in summaries)
        self.rows.extend(dict(variant="expanded", **row) for row in rows)
        self.selections.extend(dict(variant="expanded", **row) for row in selections)
        self.save()
        diagnostics = []
        for (k, shape, digest), features in feature_base.cache.items():
            for a0 in feature_base.scales:
                for rho in full.RADII:
                    peaks, values = features.evidence(a0, rho)
                    diagnostics.append(dict(response_sha256=digest, temporal_scale=k,
                        response_length=shape[0], a0=a0, rho=rho,
                        reference_width=min(shape[0], max(1, round(a0*k))),
                        local_half_window=max(1, round(rho*k)),
                        support_widths="|".join(map(str, features.effective_scales)),
                        candidate_count=len(peaks),
                        G_mean=float(values[:, 0].mean()) if len(peaks) else None,
                        L_mean=float(values[:, 1].mean()) if len(peaks) else None,
                        L_zero_fraction=float(np.mean(values[:, 1] == 0)) if len(peaks) else None))
        full.shared.write_csv(self.output / "structure_feature_diagnostics.csv", diagnostics)
