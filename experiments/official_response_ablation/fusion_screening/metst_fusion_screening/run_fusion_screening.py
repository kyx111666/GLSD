"""Six-method ME-TST+ fusion screening using the sealed Colab decoders.

Run in the original ME-TST+ Colab environment after mounting Drive. This
module never imports or runs a backbone. Official runners remain read-only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


A0, RHO, GAMMA = 1.5, 2.0, 0.5
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
ETAS = (0.25, 0.5, 0.75)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ("G", "L", "EqualMean", "WeightedMean", "DualOR", "CrossSupported")
SETTINGS = {"sammlv": "metst_sammlv", "casme3": "metst_casme3"}
SEED, RESAMPLES = 100, 10_000


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    threshold: float = 1.0
    tau_g: float | None = None
    tau_l: float | None = None
    eta: float | None = None
    alpha: float | None = None
    reference_scale: float = A0
    local_radius: float = RHO

    @property
    def identifier(self):
        return json.dumps(asdict(self), sort_keys=True)


def grids():
    result = {}
    for method in METHODS:
        configs = []
        if method in ("G", "L", "EqualMean"):
            configs = [Config(method, i, threshold=t) for i, t in enumerate(THRESHOLDS)]
        elif method == "WeightedMean":
            for alpha in ALPHAS:
                for t in THRESHOLDS:
                    configs.append(Config(method, len(configs), threshold=t, alpha=alpha))
        else:
            for eta in ((0.0,) if method == "DualOR" else ETAS):
                for tg in THRESHOLDS:
                    for tl in THRESHOLDS:
                        configs.append(Config(method, len(configs), tau_g=tg, tau_l=tl, eta=eta))
        result[method] = configs
    return result


def accept(g, l, config):
    g, l = np.asarray(g, dtype=float), np.asarray(l, dtype=float)
    if g.shape != l.shape or g.ndim != 1 or not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError("G/L must be finite aligned vectors")
    if np.any(g < -1e-12) or np.any(l < -1e-12) or np.any(g > 1 + 1e-12) or np.any(l > 1 + 1e-12):
        raise ValueError("G/L out of expected [0,1] range; no clipping is applied")
    if config.method == "G":
        return g >= config.threshold
    if config.method == "L":
        return l >= config.threshold
    if config.method == "EqualMean":
        return (g + l) / 2 >= config.threshold
    if config.method == "WeightedMean":
        return config.alpha * g + (1 - config.alpha) * l >= config.threshold
    if config.method in ("DualOR", "CrossSupported"):
        return ((g >= config.tau_g) & (l >= config.eta * config.tau_l)) | (
            (l >= config.tau_l) & (g >= config.eta * config.tau_g)
        )
    raise ValueError(config.method)


def metrics(counts):
    tp, fp, fn = map(int, counts)
    return dict(TP=tp, FP=fp, FN=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def choose(table, outer_index):
    """Exclude the held subject before all selection metrics and tie-breaks."""
    pooled = np.delete(table, outer_index, axis=1).sum(axis=1)
    m = [metrics(row) for row in pooled]
    winner = min(range(len(m)), key=lambda i: (-m[i]["F1"], -m[i]["precision"], m[i]["FP"], i))
    return winner, pooled


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(serializable(obj), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FeatureView:
    def __init__(self, base, key, owner):
        self.base, self.key, self.owner = base, key, owner
        peaks, evidence = base.evidence(A0, RHO)
        self.peaks = np.array(peaks, dtype=int, copy=True)
        self.values = np.array(evidence, dtype=float, copy=True)
        if self.values.shape != (len(self.peaks), 2):
            raise RuntimeError("official evidence must have two columns G,L")
        self.peaks.setflags(write=False)
        self.values.setflags(write=False)
        for t in THRESHOLDS:
            c = SimpleNamespace(reference_scale=A0, local_radius=RHO, threshold=t)
            expected = base.selected_peaks(c)
            got = self.peaks[accept(*self.values.T, Config("EqualMean", 0, threshold=t))]
            if not np.array_equal(got, expected):
                raise RuntimeError("new EqualMean hook does not exactly reproduce the sealed scorer")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def selected_peaks(self, config):
        if config.reference_scale != A0 or config.local_radius != RHO:
            raise RuntimeError("screening structure parameters changed")
        self.owner.calls += 1
        g, l = self.values.T
        keep = accept(g, l, config)
        if self.owner.capture:
            row = dict(feature_call_index=len(self.owner.trace), response_sha256=self.key[2],
                       temporal_scale=self.key[0], peaks=self.peaks.tolist(),
                       G=g.tolist(), L=l.tolist(), retained=keep.tolist())
            if config.method in ("DualOR", "CrossSupported"):
                row["global_branch"] = ((g >= config.tau_g) & (l >= config.eta * config.tau_l)).tolist()
                row["local_branch"] = ((l >= config.tau_l) & (g >= config.eta * config.tau_g)).tolist()
            self.owner.trace.append(row)
        return self.peaks[keep]


class FusionCore:
    """Reuse label-free official features, keyed by response content and k."""
    def __init__(self, base):
        self.base, self.cache = base, {}
        self.calls, self.capture, self.trace = 0, False, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]


def decode(context, core, subject_index, config, full):
    runner, _base, metric, official, records, _paths, _subjects, _native = context
    before = core.calls
    core.trace, core.capture = [], full
    # The sealed runner is the only owner of geometry, matching and synergy.
    with contextlib.redirect_stdout(io.StringIO()):
        raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
            records, subject_index, config, core, metric, official, full
        )
        result = runner.full_counts_from_official_synergy(raw, pred_list, gt_list) if full else None
    if core.calls == before:
        raise RuntimeError("sealed decoder bypassed the fusion hook; cannot trust this run")
    raw = tuple(map(int, raw))
    result = tuple(map(int, result)) if result is not None else None
    if len(raw) != 3 or min(raw) < 0 or (result is not None and (len(result) != 3 or min(result) < 0)):
        raise RuntimeError("invalid official counts")
    return raw, result, predictions, core.trace


def paired_comparisons(setting, counts_by_method, subjects):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subjects), size=(RESAMPLES, len(subjects)))
    distribution = {}
    for method, counts in counts_by_method.items():
        totals = counts[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        distribution[method] = np.divide(2 * totals[:, 0], denominator,
                                         out=np.zeros(RESAMPLES), where=denominator > 0)
    rows = []
    new = metrics(counts_by_method["CrossSupported"].sum(axis=0))["F1"]
    for method in METHODS[:-1]:
        delta = distribution["CrossSupported"] - distribution[method]
        lo, hi = np.quantile(delta, [.025, .975])
        rows.append(dict(setting=setting, reference=method, metric_stage="full",
                         delta_F1=new - metrics(counts_by_method[method].sum(axis=0))["F1"],
                         CI_low=float(lo), CI_high=float(hi), seed=SEED, resamples=RESAMPLES))
    return rows


def load_helper():
    name = "official_response_component_ablation.py"
    directory = Path(__file__).resolve().parent
    path = next((p for p in (directory / name, directory.parent / name) if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Place the unchanged {name} beside this script")
    spec = importlib.util.spec_from_file_location("fusion_screening_original_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def run_setting(ora, name, output):
    spec = ora.SPECS[name]
    output.mkdir(exist_ok=False)
    context = ora.metst_context(spec)
    runner, base, _metric, _official, records, _paths, subjects, _native = context
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("duplicate subject IDs")
    core = FusionCore(base)
    all_grids = grids()
    write_json(output / "protocol.json", dict(
        setting=name, fixed_structure=dict(a0=A0, rho=RHO, gamma=GAMMA),
        gamma_source="inherited sealed evidence() implementation; not overridden",
        grids={m: [asdict(c) for c in cs] for m, cs in all_grids.items()},
        subjects=subjects, source=str(spec["source"]), source_sha256=ora.sha256(spec["source"]),
        core_sha256=ora.sha256(spec["core"]),
        selection="pool raw counts from all other subjects; F1, precision, fewer FP, config order",
        evaluation="held-subject full official counts including original result synergy",
        scope="decoder selection on existing official frozen responses; no backbone nested retraining",
        stage="exploratory screening, not a final full-parameter result", equal_search_budget=False,
        k_policy="preserve sealed subject records and runner; no new duration estimate",
    ))
    pooled_rows, subject_rows, selected_rows, search_rows, counts_by_method = [], [], [], [], {}
    start = time.monotonic()
    with gzip.open(output / "selected_predictions.jsonl.gz", "wt", encoding="utf-8") as predictions_file, \
         gzip.open(output / "candidate_decisions.jsonl.gz", "wt", encoding="utf-8") as candidate_file:
        for method, grid in all_grids.items():
            table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
            print(f"{name} {method}: {len(grid)} configurations x {len(subjects)} subjects", flush=True)
            for subject_index, subject in enumerate(subjects):
                for config in grid:
                    raw, _, _, _ = decode(context, core, subject_index, config, False)
                    table[config.config_id, subject_index] = raw
                if (subject_index + 1) % 5 == 0 or subject_index + 1 == len(subjects):
                    print(f"  {subject_index + 1}/{len(subjects)} subjects; elapsed {time.monotonic()-start:.1f}s", flush=True)
            np.savez_compressed(output / f"search_counts_{method}.npz", raw_counts=table,
                                subjects=np.asarray(subjects, dtype=str))
            full_rows, raw_rows = [], []
            for outer, subject in enumerate(subjects):
                winner, pooled = choose(table, outer)
                config = grid[winner]
                for c, counts in zip(grid, pooled):
                    search_rows.append(dict(setting=name, outer_subject=subject,
                                            **asdict(c), selected=int(c.config_id == winner),
                                            **metrics(counts)))
                raw, full, predictions, traces = decode(context, core, outer, config, True)
                if raw != tuple(table[winner, outer]):
                    raise RuntimeError("selected raw prediction does not replay its search counts")
                full_rows.append(full)
                raw_rows.append(raw)
                selected_rows.append(dict(setting=name, subject=subject, **asdict(config),
                                          inner_subject_count=len(subjects)-1,
                                          inner_raw_F1=metrics(pooled[winner])["F1"]))
                for stage, counts in (("raw", raw), ("full", full)):
                    subject_rows.append(dict(setting=name, method=method, subject=subject,
                                             metric_stage=stage, **metrics(counts)))
                predictions_file.write(json.dumps(serializable(dict(
                    setting=name, subject=subject, method=method, config=asdict(config),
                    raw_counts=raw, full_counts=full, predictions=predictions,
                )), ensure_ascii=False, allow_nan=False) + "\n")
                for trace in traces:
                    candidate_file.write(json.dumps(dict(setting=name, subject=subject, method=method,
                                                        config_id=winner, **trace), allow_nan=False) + "\n")
            counts_by_method[method] = np.asarray(full_rows, dtype=np.int64)
            for stage, counts in (("raw", raw_rows), ("full", full_rows)):
                pooled_rows.append(dict(setting=name, method=method, metric_stage=stage,
                                        configs=len(grid), **metrics(np.sum(counts, axis=0))))
            # Persist completed methods even if a later method fails.
            write_csv(output / "screening_summary.csv", pooled_rows)
            write_csv(output / "per_subject_counts.csv", subject_rows)
            write_csv(output / "selected_configs.csv", selected_rows)
            write_csv(output / "search_all.csv", search_rows)
            print(f"  {method}: held-subject full F1={metrics(np.sum(full_rows, axis=0))['F1']:.6f}", flush=True)
    comparisons = paired_comparisons(name, counts_by_method, subjects)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_json(output / "completion.json", dict(completed=True, elapsed_seconds=time.monotonic()-start,
                                                feature_cache_entries=len(core.cache), subject_count=len(subjects)))
    return [r for r in pooled_rows if r["metric_stage"] == "full"], comparisons


def decision(rows, comparisons):
    """Budget recommendation, not proof of inferiority/superiority."""
    settings = sorted({r["setting"] for r in rows})
    details = []
    for setting in settings:
        f = {r["method"]: r["F1"] for r in rows if r["setting"] == setting}
        value = f["CrossSupported"]
        details.append(dict(setting=setting,
                            no_point_loss_to_G_L_weighted=value >= max(f["G"], f["L"], f["WeightedMean"]),
                            improves_mean=value > f["EqualMean"],
                            improves_OR=value > f["DualOR"],
                            below_both_strong_single_and_weighted=value < max(f["G"], f["L"]) and value < f["WeightedMean"]))
    if len(settings) != 2:
        recommendation = "WAIT_FOR_OTHER_DATASET"
    elif all(d["no_point_loss_to_G_L_weighted"] and d["improves_mean"] and d["improves_OR"] for d in details):
        recommendation = "PROMISING_REVIEW_BEFORE_FORMAL"
    elif all(d["below_both_strong_single_and_weighted"] for d in details):
        recommendation = "STOP_EXPANSION_REVIEW_FUSION"
    else:
        recommendation = "MIXED_REVIEW_COUNTS_AND_CIS"
    return dict(recommendation=recommendation, per_dataset=details, comparisons=comparisons,
                interpretation="exploratory fixed-structure screening; no automatic formal run; point estimates are not significance claims")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("both", *SETTINGS), default="both")
    parser.add_argument("--output", type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--check-inputs", action="store_true", help="Read-only path inventory; no decoding")
    parser.add_argument("--preflight-only", action="store_true", help="Replay Native and old GLSD only")
    args = parser.parse_args()
    ora, helper_path = load_helper()
    selected = list(SETTINGS.values()) if args.setting == "both" else [SETTINGS[args.setting]]
    inventory = {name: [{"path": str(p), "exists": p.exists()} for p in ora.required_paths(ora.SPECS[name])]
                 for name in selected}
    if args.check_inputs:
        print(json.dumps(inventory, indent=2))
        if any(not row["exists"] for items in inventory.values() for row in items):
            raise SystemExit(2)
        print("SCREENING_INPUT_PATHS = PASS (runtime and replay not checked)")
        return
    for name in selected:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = args.output or Path("/content/drive/MyDrive/GLSD_FUSION_SCREENING") / datetime.now(timezone.utc).strftime("screen_%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    for name in selected:
        for key in ("dump", "evidence", "run_evidence", "results", "locked_results"):
            protected = ora.SPECS[name][key].resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise RuntimeError(f"output overlaps protected input directory: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                                  settings=selected, runner_sha256=ora.sha256(Path(__file__)),
                                                  helper_sha256=ora.sha256(helper_path), input_inventory=inventory))
    print("OUTPUT =", output, flush=True)
    preflight = output / "preflight"
    preflight.mkdir()
    replay_rows = [ora.run_preflight_setting(name, preflight) for name in selected]
    write_csv(preflight / "locked_replay_summary.csv", replay_rows)
    write_json(preflight / "completion.json", dict(completed=True, settings=selected))
    print("SCREENING_NATIVE_AND_OLD_GLSD_REPLAY = PASS", flush=True)
    if args.preflight_only:
        return
    rows, comparisons = [], []
    for name in selected:
        result, paired = run_setting(ora, name, output / name)
        rows.extend(result)
        comparisons.extend(paired)
    write_csv(output / "screening_summary_full.csv", rows)
    write_csv(output / "paired_comparisons.csv", comparisons)
    report = decision(rows, comparisons)
    write_json(output / "decision.json", report)
    write_json(output / "completion.json", dict(completed=True, settings=selected))
    print("SCREENING_EXECUTION = PASS", flush=True)
    print("RECOMMENDATION =", report["recommendation"], flush=True)
    print("OUTPUT =", output, flush=True)


if __name__ == "__main__":
    main()
