"""Full-structure ME-TST+ development search on sealed official responses.

No response generation, backbone training, or writes to sealed evidence.
Reuses the previously exercised decoder hook and held-subject selection.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import gzip
import hashlib
import io
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import run_fusion_screening as shared

SCALES = (1.0, 1.5, 2.0)
RADII = (1.0, 2.0, 3.0)
LEGACY_SCALES, LEGACY_RADII = SCALES, RADII
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)
LEGACY_THRESHOLDS = THRESHOLDS
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
METHODS = ("G", "L", "EqualMean", "WeightedMean", "DominantEvidence")
MEAN_SELECTION_EPSILON = 0.002


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    reference_scale: float
    local_radius: float
    threshold: float
    alpha: float | None = None
    beta: float | None = None

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def grids():
    result = {}
    for method in METHODS:
        grid = []
        weights = ALPHAS if method == "WeightedMean" else BETAS if method == "DominantEvidence" else (None,)
        for a0 in SCALES:
            # G is independent of rho. 1.0 preserves the first legacy grid tie.
            for rho in ((1.0,) if method == "G" else RADII):
                for weight in weights:
                    for tau in THRESHOLDS:
                        grid.append(Config(method, len(grid), a0, rho, tau,
                                           alpha=weight if method == "WeightedMean" else None,
                                           beta=weight if method == "DominantEvidence" else None))
        # Retain the old structure first for exact ties, including when a new
        # decimal scale maps to an already represented integer window.
        grid.sort(key=lambda c: not (c.reference_scale in LEGACY_SCALES and c.local_radius in LEGACY_RADII))
        result[method] = [replace(config, config_id=index) for index, config in enumerate(grid)]
    return result


def scores(g, l, config):
    g, l = np.asarray(g, dtype=float), np.asarray(l, dtype=float)
    if g.shape != l.shape or g.ndim != 1 or not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError("Expected aligned finite G/L vectors")
    if np.any(g < -1e-12) or np.any(l < -1e-12) or np.any(g > 1 + 1e-12) or np.any(l > 1 + 1e-12):
        raise ValueError("G/L outside [0,1]; no clipping is performed")
    if config.method == "G":
        return g
    if config.method == "L":
        return l
    if config.method == "EqualMean":
        return (g + l) / 2.0
    if config.method == "WeightedMean":
        if config.alpha not in ALPHAS:
            raise ValueError("Invalid alpha")
        return config.alpha * g + (1 - config.alpha) * l
    if config.method == "DominantEvidence":
        if config.beta not in BETAS:
            raise ValueError("Invalid beta")
        # Preserve the legacy expression exactly at the nested endpoint.
        if config.beta == 0.5:
            return (g + l) / 2.0
        if config.beta == 1.0:
            return np.maximum(g, l)
        return config.beta * np.maximum(g, l) + (1 - config.beta) * np.minimum(g, l)
    raise ValueError(config.method)


class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        self.evidence_cache = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evidence(self, a0, rho):
        if a0 not in SCALES or rho not in RADII:
            raise ValueError("Structure parameter outside declared grid")
        if (a0, rho) not in self.evidence_cache:
            peaks, evidence = self.base.evidence(a0, rho)
            peaks = np.array(peaks, dtype=int, copy=True)
            values = np.array(evidence, dtype=float, copy=True)
            if values.shape != (len(peaks), 2):
                raise RuntimeError("Sealed evidence must have columns G,L")
            for tau in THRESHOLDS:
                config = Config("EqualMean", 0, a0, rho, tau)
                got = peaks[scores(*values.T, config) >= tau]
                expected = self.base.selected_peaks(config)
                if not np.array_equal(got, expected):
                    raise RuntimeError(f"Mean hook mismatch: response={self.key}, a0={a0}, rho={rho}, tau={tau}")
            peaks.setflags(write=False)
            values.setflags(write=False)
            self.evidence_cache[a0, rho] = peaks, values
        return self.evidence_cache[a0, rho]

    def selected_peaks(self, config):
        peaks, evidence = self.evidence(config.reference_scale, config.local_radius)
        g, l = evidence.T
        score = scores(g, l, config)
        keep = score >= config.threshold
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(
                feature_call_index=len(self.owner.trace), response_sha256=self.key[2],
                temporal_scale=self.key[0], reference_scale=config.reference_scale,
                local_radius=config.local_radius, peaks=peaks.tolist(), G=g.tolist(),
                L=l.tolist(), score=score.tolist(), retained=keep.tolist(),
            ))
        return peaks[keep]


class FusionCore(shared.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def paired_comparisons(setting, counts_by_method, subjects):
    rng = np.random.default_rng(shared.SEED)
    draws = rng.integers(0, len(subjects), size=(shared.RESAMPLES, len(subjects)))
    distributions = {}
    for method, counts in counts_by_method.items():
        totals = counts[draws].sum(axis=1).astype(float)
        denom = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        distributions[method] = np.divide(2 * totals[:, 0], denom,
                                           out=np.zeros(shared.RESAMPLES), where=denom > 0)
    target = "DominantEvidence"
    point = shared.metrics(counts_by_method[target].sum(axis=0))["F1"]
    rows = []
    for method in METHODS[:-1]:
        delta = distributions[target] - distributions[method]
        lo, hi = np.quantile(delta, [.025, .975])
        rows.append(dict(setting=setting, target=target, reference=method, metric_stage="full",
                         delta_F1=point - shared.metrics(counts_by_method[method].sum(axis=0))["F1"],
                         CI_low=float(lo), CI_high=float(hi), seed=shared.SEED,
                         resamples=shared.RESAMPLES))
    return rows


def choose_mean_synergy(mean_table, g_table, l_table, outer_index,
                        epsilon=MEAN_SELECTION_EPSILON):
    """Select Mean on inner subjects with a frozen near-optimal synergy rule.

    First retain Mean configurations within ``epsilon`` F1 of the best inner
    Mean configuration. Among those rows, maximize the smaller of the Mean
    F1 margins over the same-structure G and L rows. Final ties use the
    ordinary Mean precision, FP, and config-index order. The held-out subject
    is excluded before every calculation.
    """
    mean_pooled = np.delete(mean_table, outer_index, axis=1).sum(axis=1)
    g_pooled = np.delete(g_table, outer_index, axis=1).sum(axis=1)
    l_pooled = np.delete(l_table, outer_index, axis=1).sum(axis=1)
    if mean_pooled.shape != l_pooled.shape or mean_pooled.shape[0] != 90:
        raise ValueError("Mean/G/L synergy selection expects the legacy 90-row structure grid")
    mapping = np.asarray([(i // 30) * 10 + (i % 10) for i in range(90)], dtype=int)
    g_same = g_pooled[mapping]
    mean_metrics = [shared.metrics(row) for row in mean_pooled]
    g_metrics = [shared.metrics(row) for row in g_same]
    l_metrics = [shared.metrics(row) for row in l_pooled]
    best_f1 = max(m["F1"] for m in mean_metrics)
    eligible = [i for i, m in enumerate(mean_metrics) if m["F1"] >= best_f1 - epsilon]
    margins = {i: min(mean_metrics[i]["F1"] - g_metrics[i]["F1"],
                       mean_metrics[i]["F1"] - l_metrics[i]["F1"])
               for i in eligible}
    max_margin = max(margins.values())
    candidates = [i for i in eligible if abs(margins[i] - max_margin) <= 1e-15]
    winner = min(candidates, key=lambda i: (-mean_metrics[i]["F1"],
                                            -mean_metrics[i]["precision"],
                                            mean_metrics[i]["FP"], i))
    audit = dict(selection_rule="near_optimal_mean_synergy_v1",
                 epsilon=float(epsilon), best_inner_mean_F1=float(best_f1),
                 eligible_count=len(eligible), max_synergy_margin=float(max_margin),
                 selected_synergy_margin=float(margins[winner]),
                 selected_config_id=int(winner),
                 eligible_config_ids=[int(i) for i in eligible])
    return winner, mean_pooled, audit


def verify_grid(ora, base):
    locked = ora.locked_grid(base)
    # Verify the immutable input protocol, independently of the development grid.
    expected = [(a0, rho, tau) for a0 in LEGACY_SCALES for rho in LEGACY_RADII for tau in LEGACY_THRESHOLDS]
    actual = [(c.reference_scale, c.local_radius, c.threshold) for c in locked]
    if actual != expected:
        raise RuntimeError("Sealed GLSD-90 grid/order differs; inspect before changing protocol")


def decode_with_context(context, core, subject_index, subject, config, full, setting, method):
    """Add fold/config provenance to failures from the sealed evaluator."""
    try:
        runner, _base, metric, official, records, _paths, _subjects, _native = context
        before = core.calls
        core.trace, core.capture = [], full
        with contextlib.redirect_stdout(io.StringIO()):
            raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
                records, subject_index, config, core, metric, official, full
            )
            result = runner.full_counts_from_official_synergy(raw, pred_list, gt_list) if full else None
        raw = tuple(map(int, raw))
        result = tuple(map(int, result)) if result is not None else None
        if core.calls == before:
            raise RuntimeError("sealed decoder bypassed fusion hook")
        if len(raw) != 3 or min(raw) < 0 or (result is not None and (len(result) != 3 or min(result) < 0)):
            raise RuntimeError(f"invalid official counts raw={raw!r} result={result!r}")
        return raw, result, predictions, core.trace
    except Exception as exc:
        details = dict(setting=setting, method=method, subject_index=subject_index,
                       subject=subject, config=asdict(config), full=full)
        raise RuntimeError(f"official decoder/count failure: {json.dumps(details, sort_keys=True)}") from exc


def run_setting(ora, name, output, *, prepared_context=None, evaluation_protocol="legacy_official",
                expanded_structure=False, selection_rule="legacy", selection_epsilon=None):
    if selection_epsilon is None:
        selection_epsilon = MEAN_SELECTION_EPSILON
    selection_epsilon = float(selection_epsilon)
    if selection_epsilon < 0 or not np.isfinite(selection_epsilon):
        raise ValueError("selection_epsilon must be a finite non-negative number")
    spec = ora.SPECS[name]
    output.mkdir(exist_ok=False)
    context = ora.metst_context(spec) if prepared_context is None else prepared_context
    _runner, base, _metric, _official, _records, _paths, subjects, _native = context
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("Duplicate subjects")
    verify_grid(ora, base)
    if ora.context_video_and_gt_counts("metst", context) != (spec["videos"], spec["gt"]):
        raise RuntimeError("Unexpected current official video/GT counts")
    feature_base = base
    structure_audit = None
    if expanded_structure:
        from structure_expansion import ExpandedCore, StructureAudit
        feature_base = ExpandedCore(base, SCALES)
        structure_audit = StructureAudit(name, output, subjects)
    core, all_grids = FusionCore(feature_base), grids()
    shared.write_json(output / "protocol.json", dict(
        setting=name, searched_structure=dict(a0=SCALES, rho=RADII),
        G_rho="not searched: G does not depend on rho; adapter value 1.0",
        fixed_gamma="0.5 inherited from sealed core", subjects=subjects,
        support_scales=list(LEGACY_SCALES), expanded_structure=expanded_structure,
        structure_tie_policy="old structure first, then declared grid order",
        grids={m: [asdict(c) for c in cs] for m, cs in all_grids.items()},
        raw_config_counts={m: len(cs) for m, cs in all_grids.items()},
        grid_duplicates="WeightedMean alpha endpoints duplicate G/L; all declared rows retained in fixed order",
        equal_search_budget=False, source=str(spec["source"]),
        source_sha256=ora.sha256(spec["source"]), core_sha256=ora.sha256(spec["core"]),
        selection="all other subjects pooled raw TP/FP/FN; F1, precision, fewer FP, fixed grid order",
        evaluation="held-subject full counts using unchanged official recognition/result synergy",
        matching_protocol=evaluation_protocol,
        scope="decoder leave-subject-out selection on frozen official responses; no backbone nested retraining",
        stage="full-grid development experiment; these datasets have informed prior method design",
        k_policy="inherit records/runner; no new duration estimation",
        diagnostic_identity="feature_call_index is a call index, not a verified video ID",
        mean_selection_rule=selection_rule,
        mean_selection_epsilon=(selection_epsilon if selection_rule == "near_optimal_mean_synergy_v1" else None),
    ))
    pooled_rows, subject_rows, selected_rows, counts_by_method = [], [], [], {}
    search_tables = {}
    selected_audits = {}
    # Keep the outer-fold Mean choices so component ablations can be evaluated
    # with exactly the same (a0, rho, tau), rather than receiving a second
    # opportunity to tune themselves.
    mean_selected = {}
    start = time.monotonic()
    with gzip.open(output / "selected_predictions.jsonl.gz", "wt", encoding="utf-8") as pred_file, \
         gzip.open(output / "candidate_decisions.jsonl.gz", "wt", encoding="utf-8") as candidate_file, \
         (output / "search_all.csv").open("w", newline="", encoding="utf-8") as search_file:
        search_writer = None
        for method, grid in all_grids.items():
            table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
            method_start = time.monotonic()
            print(f"{name} {method}: {len(grid)} configurations x {len(subjects)} subjects", flush=True)
            for si, subject in enumerate(subjects):
                for config in grid:
                    raw, _, _, _ = decode_with_context(context, core, si, subject, config, False, name, method)
                    table[config.config_id, si] = raw
                if si == 0 or (si + 1) % 5 == 0 or si + 1 == len(subjects):
                    elapsed = time.monotonic() - method_start
                    eta = elapsed / (si + 1) * (len(subjects) - si - 1)
                    print(f"  {si+1}/{len(subjects)} subjects; method raw elapsed={elapsed:.1f}s; estimated raw remaining={eta:.1f}s", flush=True)
            np.savez_compressed(output / f"search_counts_{method}.npz", raw_counts=table,
                                subjects=np.asarray(subjects, dtype=str))
            search_tables[method] = table
            if structure_audit is not None:
                structure_audit.evaluate(method, grid, table, context, core)
            raw_rows, full_rows = [], []
            for outer, subject in enumerate(subjects):
                if method == "EqualMean" and selection_rule == "near_optimal_mean_synergy_v1":
                    if "G" not in search_tables or "L" not in search_tables:
                        raise RuntimeError("Mean synergy selection requires completed G and L search tables")
                    winner, pooled, selection_audit = choose_mean_synergy(
                        table, search_tables["G"], search_tables["L"], outer,
                        epsilon=selection_epsilon)
                    selected_audits[subject] = selection_audit
                else:
                    winner, pooled = shared.choose(table, outer)
                    selection_audit = None
                config = grid[winner]
                for c, counts in zip(grid, pooled):
                    row = dict(setting=name, method=method, outer_subject=subject,
                               **{k: v for k, v in asdict(c).items() if k != "method"},
                               selected=int(c.config_id == winner), **shared.metrics(counts))
                    if search_writer is None:
                        search_writer = csv.DictWriter(search_file, fieldnames=list(row))
                        search_writer.writeheader()
                    search_writer.writerow(row)
                raw, full, predictions, traces = decode_with_context(
                    context, core, outer, subject, config, True, name, method
                )
                if raw != tuple(table[winner, outer]):
                    raise RuntimeError("Held-subject raw counts differ from search table replay")
                raw_rows.append(raw)
                full_rows.append(full)
                selected_rows.append(dict(setting=name, subject=subject, **asdict(config),
                                          inner_subject_count=len(subjects)-1,
                                          inner_raw_F1=shared.metrics(pooled[winner])["F1"],
                                          selection_rule=selection_rule,
                                          synergy_margin=(None if selection_audit is None else selection_audit["selected_synergy_margin"]),
                                          near_optimal_count=(None if selection_audit is None else selection_audit["eligible_count"])))
                if method == "EqualMean":
                    mean_selected[subject] = config
                for stage, counts in (("raw", raw), ("full", full)):
                    subject_rows.append(dict(setting=name, method=method, subject=subject,
                                             metric_stage=stage, **shared.metrics(counts)))
                pred_file.write(json.dumps(shared.serializable(dict(
                    setting=name, subject=subject, method=method, config=asdict(config),
                    raw_counts=raw, full_counts=full, predictions=predictions,
                )), allow_nan=False) + "\n")
                for trace in traces:
                    candidate_file.write(json.dumps(dict(setting=name, subject=subject, method=method,
                                                          config_id=winner, **trace), allow_nan=False) + "\n")
            counts_by_method[method] = np.asarray(full_rows, dtype=np.int64)
            for stage, counts in (("raw", raw_rows), ("full", full_rows)):
                pooled_rows.append(dict(setting=name, method=method, metric_stage=stage,
                                        configs=len(grid), **shared.metrics(np.sum(counts, axis=0))))
            for fn, rows in (("summary_raw_full.csv", pooled_rows), ("per_subject_counts.csv", subject_rows),
                             ("selected_configs.csv", selected_rows)):
                shared.write_csv(output / fn, rows)
            pred_file.flush()
            candidate_file.flush()
            search_file.flush()
            print(f"  {method}: held-subject full F1={shared.metrics(np.sum(full_rows, axis=0))['F1']:.6f}", flush=True)
            if method == "EqualMean":
                if evaluation_protocol == "legacy_official":
                    locked = ora.existing_glsd_subject_counts(spec["run_evidence"] / "per_subject_counts.csv", len(subjects))
                    if {s: tuple(c) for s, c in zip(subjects, full_rows)} != locked:
                        raise RuntimeError("Fresh EqualMean-90 selection differs from locked subject counts; outputs retained for diagnosis")
                    print("FRESH_EQUALMEAN_90_REPLAY = PASS", flush=True)
                else:
                    # A changed evaluator cannot be compared to sealed legacy counts.
                    # Check the hook against the untouched mean scorer under this evaluator.
                    if selection_rule == "near_optimal_mean_synergy_v1":
                        chosen = {s: grid[mean_selected[s].config_id] for s in subjects}
                    else:
                        chosen = {s: grid[shared.choose(table, i)[0]] for i, s in enumerate(subjects)}
                    with contextlib.redirect_stdout(io.StringIO()):
                        replay = ora.locked_glsd_subject_counts("metst", context, feature_base, chosen, subjects)
                    if replay != {s: (tuple(r), tuple(f)) for s, r, f in zip(subjects, raw_rows, full_rows)}:
                        raise RuntimeError("Mean hook differs from sealed scorer under the revised evaluator")
                    print("EQUALMEAN_SAME_EVALUATOR_REPLAY = PASS", flush=True)
    # Locked component ablations. The selected EqualMean configuration from
    # each outer fold is reused verbatim; only the score expression changes.
    ablation_rows, ablation_subject_rows, ablation_selected_rows = [], [], []
    ablation_counts = {}
    for label, component in (("Mean_w/o_L", "G"), ("Mean_w/o_G", "L")):
        raw_rows, full_rows = [], []
        for outer, subject in enumerate(subjects):
            base_config = mean_selected[subject]
            locked = replace(base_config, method=component)
            raw, full, predictions, traces = decode_with_context(
                context, core, outer, subject, locked, True, name, label
            )
            raw_rows.append(raw)
            full_rows.append(full)
            ablation_selected_rows.append(dict(
                setting=name, subject=subject, label=label,
                source_method="EqualMean", source_config_id=base_config.config_id,
                reference_scale=base_config.reference_scale,
                local_radius=base_config.local_radius,
                threshold=base_config.threshold,
                raw_F1=shared.metrics(raw)["F1"], full_F1=shared.metrics(full)["F1"],
            ))
            for stage, counts in (("raw", raw), ("full", full)):
                ablation_subject_rows.append(dict(
                    setting=name, method=label, metric_stage=stage, subject=subject,
                    **shared.metrics(counts), selection="locked_equalmean"))
        arr = np.asarray(full_rows, dtype=np.int64)
        ablation_counts[label] = arr
        for stage, rows in (("raw", raw_rows), ("full", full_rows)):
            ablation_rows.append(dict(setting=name, method=label, metric_stage=stage,
                                      configs=0, selection="locked_equalmean",
                                      **shared.metrics(np.sum(rows, axis=0))))
    shared.write_csv(output / "locked_ablation_summary.csv", ablation_rows)
    shared.write_csv(output / "locked_ablation_per_subject_counts.csv", ablation_subject_rows)
    shared.write_csv(output / "locked_ablation_selected_configs.csv", ablation_selected_rows)
    # Add the locked ablations to the same paired-subject comparison output,
    # while keeping the original independently tuned comparisons intact.
    locked_comparisons = []
    mean_counts = counts_by_method["EqualMean"]
    for label, counts in ablation_counts.items():
        draws = np.random.default_rng(shared.SEED).integers(
            0, len(subjects), size=(shared.RESAMPLES, len(subjects)))
        target = mean_counts[draws].sum(axis=1).astype(float)
        ref = counts[draws].sum(axis=1).astype(float)
        target_f1 = np.divide(2 * target[:, 0], 2 * target[:, 0] + target[:, 1] + target[:, 2],
                              out=np.zeros(shared.RESAMPLES), where=(2 * target[:, 0] + target[:, 1] + target[:, 2]) > 0)
        ref_f1 = np.divide(2 * ref[:, 0], 2 * ref[:, 0] + ref[:, 1] + ref[:, 2],
                           out=np.zeros(shared.RESAMPLES), where=(2 * ref[:, 0] + ref[:, 1] + ref[:, 2]) > 0)
        locked_comparisons.append(dict(
            setting=name, target="EqualMean", reference=label,
            metric_stage="full", delta_F1=shared.metrics(mean_counts.sum(axis=0))["F1"] - shared.metrics(counts.sum(axis=0))["F1"],
            CI_low=float(np.quantile(target_f1 - ref_f1, .025)),
            CI_high=float(np.quantile(target_f1 - ref_f1, .975)),
            seed=shared.SEED, resamples=shared.RESAMPLES, selection="locked_equalmean"))
    shared.write_csv(output / "locked_ablation_comparisons.csv", locked_comparisons)
    if selected_audits:
        shared.write_csv(output / "mean_selection_audit.csv",
                         [dict(subject=s, **audit) for s, audit in selected_audits.items()])
    if structure_audit is not None:
        structure_audit.finish(pooled_rows, subject_rows, selected_rows, feature_base)
    comparisons = paired_comparisons(name, counts_by_method, subjects)
    shared.write_csv(output / "paired_comparisons.csv", comparisons)
    freq = Counter((r["method"], r["reference_scale"], r["local_radius"], r["threshold"], r["alpha"], r["beta"]) for r in selected_rows)
    shared.write_csv(output / "selected_parameter_frequency.csv", [
        dict(method=k[0], a0=k[1], rho=k[2], tau=k[3], alpha=k[4], beta=k[5], folds=v)
        for k, v in freq.items()
    ])
    shared.write_json(output / "completion.json", dict(completed=True,
                      elapsed_seconds=time.monotonic()-start, subject_count=len(subjects),
                      feature_cache_entries=len(core.cache)))
    return [r for r in pooled_rows if r["metric_stage"] == "full"], comparisons


def decision(rows):
    details = []
    for setting in sorted({r["setting"] for r in rows}):
        f = {r["method"]: r["F1"] for r in rows if r["setting"] == setting}
        d = f["DominantEvidence"]
        details.append(dict(setting=setting, delta_vs_stronger_single=d-max(f["G"], f["L"]),
                            delta_vs_weighted=d-f["WeightedMean"], delta_vs_equal=d-f["EqualMean"]))
    return dict(per_dataset=details,
                interpretation="Point estimates for development review; bootstrap is exploratory. PASS means execution only. No automatic grid expansion or success claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("both", *shared.SETTINGS), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    ora, helper_path = shared.load_helper()
    selected = list(shared.SETTINGS.values()) if args.setting == "both" else [shared.SETTINGS[args.setting]]
    inventory = {name: [dict(path=str(p), exists=p.exists()) for p in ora.required_paths(ora.SPECS[name])]
                 for name in selected}
    if args.check_inputs:
        print(json.dumps(inventory, indent=2))
        if any(not r["exists"] for items in inventory.values() for r in items):
            raise SystemExit(2)
        print("FULL_TUNING_INPUT_PATHS = PASS (runtime and replay not checked)")
        return
    for name in selected:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = args.output or Path("/content/drive/MyDrive/GLSD_FULL_FUSION_TUNING") / datetime.now(timezone.utc).strftime("full_%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    for name in selected:
        for key in ("dump", "evidence", "run_evidence", "results", "locked_results"):
            protected = ora.SPECS[name][key].resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise RuntimeError(f"Output overlaps sealed input: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output / "run_manifest.json", dict(settings=selected,
        created_utc=datetime.now(timezone.utc).isoformat(), input_inventory=inventory,
        source_hashes={p.name: ora.sha256(p) for p in (Path(__file__), Path(shared.__file__), helper_path)}))
    print("OUTPUT =", output, flush=True)
    preflight = output / "preflight"
    preflight.mkdir()
    replay = [ora.run_preflight_setting(name, preflight) for name in selected]
    shared.write_csv(preflight / "locked_replay_summary.csv", replay)
    shared.write_json(preflight / "completion.json", dict(completed=True, settings=selected))
    print("NATIVE_AND_OLD_GLSD_REPLAY = PASS", flush=True)
    rows, comparisons = [], []
    for name in selected:
        result, paired = run_setting(ora, name, output / name)
        rows.extend(result)
        comparisons.extend(paired)
        shared.write_csv(output / "full_tuning_summary_full.csv", rows)
        shared.write_csv(output / "paired_comparisons.csv", comparisons)
    shared.write_json(output / "decision.json", decision(rows))
    shared.write_json(output / "completion.json", dict(completed=True, settings=selected))
    print("FULL_TUNING_EXECUTION = PASS (execution completed; inspect method results)", flush=True)
    print("OUTPUT =", output, flush=True)


if __name__ == "__main__":
    main()
