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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import run_generalized_mean_screening as shared

SCALES = (1.0, 1.5, 2.0)
RADII = (1.0, 2.0, 3.0)
THRESHOLDS = tuple(i / 20 for i in range(1, 20))
LEGACY_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.75)
METHODS = ("G", "L", "Mean", "SimpleBonus")


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    reference_scale: float
    local_radius: float
    threshold: float

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def grids():
    result = {}
    for method in METHODS:
        grid = []
        for a0 in SCALES:
            for rho in ((1.0,) if method == "G" else RADII):
                for tau in THRESHOLDS:
                    grid.append(Config(method, len(grid), a0, rho, tau))
        result[method] = grid
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
    if config.method == "Mean":
        return (g + l) / 2.0
    if config.method == "SimpleBonus":
        # Same arithmetic as the previously tested lambda=1, gamma=0 endpoint.
        maximum, minimum = np.maximum(g, l), np.minimum(g, l)
        return maximum + minimum * (1 - maximum)
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
                config = Config("Mean", 0, a0, rho, tau)
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
    target = "SimpleBonus"
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


def verify_grid(ora, base):
    locked = ora.locked_grid(base)
    # Verify the immutable input protocol, independently of the development grid.
    expected = [(a0, rho, tau) for a0 in SCALES for rho in RADII for tau in LEGACY_THRESHOLDS]
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


def run_setting(ora, name, output, *, prepared_context=None, evaluation_protocol="legacy_official"):
    spec = ora.SPECS[name]
    output.mkdir(exist_ok=False)
    context = ora.metst_context(spec) if prepared_context is None else prepared_context
    _runner, base, _metric, _official, _records, _paths, subjects, _native = context
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("Duplicate subjects")
    verify_grid(ora, base)
    if ora.context_video_and_gt_counts("metst", context) != (spec["videos"], spec["gt"]):
        raise RuntimeError("Unexpected current official video/GT counts")
    core, all_grids = FusionCore(base), grids()
    shared.write_json(output / "protocol.json", dict(
        setting=name, searched_structure=dict(a0=SCALES, rho=RADII),
        G_rho="not searched: G does not depend on rho; adapter value 1.0",
        fixed_gamma="0.5 inherited from sealed core", subjects=subjects,
        grids={m: [asdict(c) for c in cs] for m, cs in all_grids.items()},
        raw_config_counts={m: len(cs) for m, cs in all_grids.items()},
        grid_duplicates="G radius duplicates removed; no fusion hyperparameters",
        fusion_formula="G+L-G*L; evaluated as max+min*(1-max) to preserve previous endpoint",
        equal_search_budget=False, source=str(spec["source"]),
        source_sha256=ora.sha256(spec["source"]), core_sha256=ora.sha256(spec["core"]),
        selection="all other subjects pooled raw TP/FP/FN; F1, precision, fewer FP, fixed grid order",
        evaluation="held-subject full counts using unchanged official recognition/result synergy",
        matching_protocol=evaluation_protocol,
        scope="decoder leave-subject-out selection on frozen official responses; no backbone nested retraining",
        stage="full-grid development experiment; these datasets have informed prior method design",
        k_policy="inherit records/runner; no new duration estimation",
        diagnostic_identity="feature_call_index is a call index, not a verified video ID",
    ))
    pooled_rows, subject_rows, selected_rows, counts_by_method = [], [], [], {}
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
            raw_rows, full_rows = [], []
            for outer, subject in enumerate(subjects):
                winner, pooled = shared.choose(table, outer)
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
                                          inner_raw_F1=shared.metrics(pooled[winner])["F1"]))
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
            if method == "Mean":
                # The common 19-tau search differs from legacy GLSD-90.
                # Verify selected predictions against the untouched mean scorer,
                # not against selections from a different threshold grid.
                chosen = {s: grid[shared.choose(table, i)[0]] for i, s in enumerate(subjects)}
                with contextlib.redirect_stdout(io.StringIO()):
                    replay = ora.locked_glsd_subject_counts("metst", context, base, chosen, subjects)
                if replay != {s: (tuple(r), tuple(f)) for s, r, f in zip(subjects, raw_rows, full_rows)}:
                    raise RuntimeError("Mean hook differs from sealed mean scorer")
                print("MEAN_SELECTED_SCORER_REPLAY = PASS", flush=True)
    comparisons = paired_comparisons(name, counts_by_method, subjects)
    shared.write_csv(output / "paired_comparisons.csv", comparisons)
    freq = Counter((r["method"], r["reference_scale"], r["local_radius"], r["threshold"]) for r in selected_rows)
    shared.write_csv(output / "selected_parameter_frequency.csv", [
        dict(method=k[0], a0=k[1], rho=k[2], tau=k[3], folds=v)
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
        d = f["SimpleBonus"]
        details.append(dict(setting=setting, delta_vs_stronger_single=d-max(f["G"], f["L"]),
                            delta_vs_mean=d-f["Mean"]))
    return dict(per_dataset=details,
                interpretation="Point estimates for development review; bootstrap is exploratory. PASS means execution only. No automatic grid expansion or success claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("both", *shared.SETTINGS), default="sammlv")
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
        print("STRUCTURE_BONUS_INPUT_PATHS = PASS (runtime and replay not checked)")
        return
    for name in selected:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = args.output or Path("/content/drive/MyDrive/GLSD_STRUCTURE_BONUS") / datetime.now(timezone.utc).strftime("full_%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    for name in selected:
        for key in ("dump", "evidence", "run_evidence", "results", "locked_results"):
            protected = ora.SPECS[name][key].resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise RuntimeError(f"Output overlaps sealed input: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output / "run_manifest.json", dict(settings=selected,
        created_utc=datetime.now(timezone.utc).isoformat(), input_inventory=inventory,
        source_hashes={p.name: ora.sha256(p) for p in Path(__file__).parent.glob("*.py")}))
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
        shared.write_csv(output / "structure_summary_full.csv", rows)
        shared.write_csv(output / "paired_comparisons.csv", comparisons)
    shared.write_json(output / "decision.json", decision(rows))
    shared.write_json(output / "completion.json", dict(completed=True, settings=selected))
    print("STRUCTURE_BONUS_EXECUTION = PASS (execution completed; inspect method results)", flush=True)
    print("OUTPUT =", output, flush=True)


if __name__ == "__main__":
    main()
