"""Phase-1 ME-TST+ generalized-mean screening on sealed official responses.

This runner is deliberately limited to the first risk-screening stage:
``a0=2.0`` and ``rho=1.0`` are frozen, p is the only fusion variable, and
the held-out subject is never used to select tau or p.  The official ME-TST+
decoder, geometry, matching, recognition, and result-synergy code remains
sealed; this module replaces only the candidate score after evidence() has
already produced G/L.
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
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


A0, RHO = 2.0, 1.0
P_FINITE = (1.0, 2.0, 4.0, 8.0)
P_DIAGNOSTIC = math.inf
P_ALL = P_FINITE + (P_DIAGNOSTIC,)
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
SETTINGS = {"sammlv": "metst_sammlv", "casme3": "metst_casme3"}
SEED, RESAMPLES = 100, 10_000


@dataclass(frozen=True)
class Config:
    method: str
    config_id: int
    threshold: float
    p: float | None = None
    reference_scale: float = A0
    local_radius: float = RHO

    @property
    def identifier(self) -> str:
        return json.dumps(config_dict(self), sort_keys=True)


def p_label(p: float | None) -> str:
    if p is None:
        return "none"
    return "inf" if math.isinf(float(p)) else f"{float(p):g}"


def config_dict(config: Config) -> dict:
    row = asdict(config)
    row["p"] = p_label(config.p)
    return row


def validate_p(p: float) -> float:
    value = float(p)
    if math.isnan(value) or value < 1.0 or (not math.isfinite(value) and not math.isinf(value)):
        raise ValueError("p must be one or more, finite or +inf")
    return value


def generalized_mean(g, l, p: float):
    """Compute S_p with exact p=1 and explicit +inf endpoints."""
    p = validate_p(p)
    g = np.asarray(g, dtype=float)
    l = np.asarray(l, dtype=float)
    if g.shape != l.shape:
        raise ValueError("G/L shapes must match")
    if not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError("G/L must be finite; no clipping is applied")
    if np.any(g < -1e-12) or np.any(l < -1e-12) or np.any(g > 1 + 1e-12) or np.any(l > 1 + 1e-12):
        raise ValueError("G/L out of expected [0,1] range; no clipping is applied")
    if p == 1.0:
        return (g + l) / 2.0
    if math.isinf(p):
        return np.maximum(g, l)
    return ((g ** p + l ** p) / 2.0) ** (1.0 / p)


def accept(g, l, config: Config):
    g = np.asarray(g, dtype=float)
    l = np.asarray(l, dtype=float)
    if config.method == "G":
        if not np.isfinite(g).all() or np.any(g < -1e-12) or np.any(g > 1 + 1e-12):
            raise ValueError("G must be finite and in [0,1]")
        score = g
    elif config.method == "L":
        if not np.isfinite(l).all() or np.any(l < -1e-12) or np.any(l > 1 + 1e-12):
            raise ValueError("L must be finite and in [0,1]")
        score = l
    elif config.method == "GM":
        if config.p is None:
            raise ValueError("generalized-mean config requires p")
        score = generalized_mean(g, l, config.p)
    else:
        raise ValueError(config.method)
    threshold = float(config.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    return score >= threshold


def metrics(counts):
    tp, fp, fn = map(int, counts)
    return dict(TP=tp, FP=fp, FN=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def grids():
    singles = {
        method: [Config(method, i, threshold=t) for i, t in enumerate(THRESHOLDS)]
        for method in ("G", "L")
    }
    gm = [Config("GM", i, threshold=t, p=p)
          for i, (p, t) in enumerate((p, t) for p in P_ALL for t in THRESHOLDS)]
    return {**singles, "GM": gm}


def finite_gm_configs(grid):
    return [c for c in grid if c.p in P_FINITE]


def choose(table: np.ndarray, outer_index: int, allowed=None):
    """Select from inner subjects only, with a fixed deterministic tie-break."""
    candidates = list(range(table.shape[0])) if allowed is None else list(allowed)
    inner = np.delete(table[:, :, :], outer_index, axis=1)
    pooled = inner.sum(axis=1)
    winner = min(candidates, key=lambda i: (
        -metrics(pooled[i])["F1"], -metrics(pooled[i])["precision"],
        metrics(pooled[i])["FP"], i))
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
    if isinstance(value, float) and math.isinf(value):
        return "inf"
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
        self.peaks = np.asarray(peaks, dtype=int).copy()
        self.values = np.asarray(evidence, dtype=float).copy()
        if self.values.shape != (len(self.peaks), 2):
            raise RuntimeError("official evidence must have exactly two columns G,L")
        if not np.isfinite(self.values).all() or np.any(self.values < -1e-12) or np.any(self.values > 1 + 1e-12):
            raise RuntimeError(f"invalid G/L evidence for response {key[2]}")
        self.peaks.setflags(write=False)
        self.values.setflags(write=False)
        # p=1 is the exact historical arithmetic-mean hook replay gate.
        g, l = self.values.T
        for threshold in THRESHOLDS:
            legacy = SimpleNamespace(reference_scale=A0, local_radius=RHO, threshold=threshold)
            expected = base.selected_peaks(legacy)
            got = self.peaks[accept(g, l, Config("GM", 0, threshold=threshold, p=1.0))]
            if not np.array_equal(got, expected):
                raise RuntimeError("p=1 does not exactly reproduce the sealed arithmetic-mean scorer")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def selected_peaks(self, config):
        if config.reference_scale != A0 or config.local_radius != RHO:
            raise RuntimeError("phase-1 structure parameters are fixed at a0=2.0,rho=1.0")
        self.owner.calls += 1
        g, l = self.values.T
        keep = accept(g, l, config)
        if self.owner.capture:
            score = g if config.method == "G" else l if config.method == "L" else generalized_mean(g, l, config.p)
            self.owner.trace.append({
                "response_sha256": self.key[2], "k": self.key[0],
                "candidate_count": len(self.peaks), "peaks": self.peaks.tolist(),
                "G": g.tolist(), "L": l.tolist(), "score": score.tolist(),
                "method": config.method, "p": p_label(config.p),
                "threshold": config.threshold, "retained": keep.tolist(),
                "provenance_note": "response hash and peak index are stable; sealed runner did not expose video ID",
            })
        return self.peaks[keep]


class FusionCore:
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


def decode(context, core, subject_index, config, full, capture=False):
    runner, _base, metric, official, records, _paths, _subjects, _native = context
    before = core.calls
    core.trace, core.capture = [], capture
    with contextlib.redirect_stdout(io.StringIO()):
        raw, predictions, pred_list, gt_list, *_ = runner.decode_glsd_subject(
            records, subject_index, config, core, metric, official, full
        )
        result = runner.full_counts_from_official_synergy(raw, pred_list, gt_list) if full else None
    if core.calls == before:
        raise RuntimeError("sealed decoder bypassed the fusion hook")
    raw = tuple(map(int, raw))
    result = tuple(map(int, result)) if result is not None else None
    return raw, result, predictions, list(core.trace)


def load_helper():
    directory = Path(__file__).resolve().parent
    path = directory / "official_response_component_ablation.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing unchanged helper: {path}")
    spec = importlib.util.spec_from_file_location("generalized_mean_official_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def inversion_rate(reference, other):
    """Exact pairwise inversion rate with stable value-order tie handling."""
    reference = np.asarray(reference, dtype=float)
    other = np.asarray(other, dtype=float)
    n = len(reference)
    if n < 2:
        return 0.0
    ref_order = np.argsort(-reference, kind="mergesort")
    other_order = np.argsort(-other, kind="mergesort")
    rank = np.empty(n, dtype=np.int64)
    rank[other_order] = np.arange(n)
    values = rank[ref_order]
    work = values.copy()

    def sort_count(left, right):
        if right - left <= 1:
            return 0
        mid = (left + right) // 2
        count = sort_count(left, mid) + sort_count(mid, right)
        i, j, k = left, mid, left
        while i < mid and j < right:
            if values[i] <= values[j]:
                work[k] = values[i]; i += 1
            else:
                work[k] = values[j]; j += 1; count += mid - i
            k += 1
        while i < mid:
            work[k] = values[i]; i += 1; k += 1
        while j < right:
            work[k] = values[j]; j += 1; k += 1
        values[left:right] = work[left:right]
        return count

    inversions = sort_count(0, n)
    return float(inversions / (n * (n - 1) / 2))


def paired_comparisons(setting, counts_by_variant, subjects):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subjects), size=(RESAMPLES, len(subjects)))
    distribution = {}
    for variant, counts in counts_by_variant.items():
        totals = counts[draws].sum(axis=1).astype(float)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        distribution[variant] = np.divide(2 * totals[:, 0], denominator,
                                           out=np.zeros(RESAMPLES), where=denominator > 0)
    reference = "GM_joint"
    rows = []
    for variant in counts_by_variant:
        if variant == reference:
            continue
        delta = distribution[reference] - distribution[variant]
        rows.append(dict(setting=setting, reference=variant, compared=reference,
                         metric_stage="full",
                         delta_F1=metrics(counts_by_variant[reference].sum(axis=0))["F1"] - metrics(counts_by_variant[variant].sum(axis=0))["F1"],
                         CI_low=float(np.quantile(delta, .025)),
                         CI_high=float(np.quantile(delta, .975)), seed=SEED, resamples=RESAMPLES))
    return rows


def diagnostic_rows(core: FusionCore, setting: str):
    score_rows, rank_rows, transition_rows, disagreement_rows, mask_rows = [], [], [], [], []
    for view in core.cache.values():
        g, l = view.values.T
        base = generalized_mean(g, l, 1.0)
        bins = np.where((g >= 2 * l) & (l > 0), "G-dominant",
                        np.where((l >= 2 * g) & (g > 0), "L-dominant", "near-equal"))
        for p in P_ALL:
            score = generalized_mean(g, l, p)
            score_rows.append({"setting": setting, "response_sha256": view.key[2], "p": p_label(p),
                               "candidate_count": len(score),
                               **{f"q{q:g}": float(np.quantile(score, q)) for q in (0, .25, .5, .75, 1)}})
            base_order = np.argsort(-base, kind="mergesort")
            score_order = np.argsort(-score, kind="mergesort")
            base_rank = np.empty(len(base), dtype=np.int64)
            score_rank = np.empty(len(score), dtype=np.int64)
            base_rank[base_order] = np.arange(len(base))
            score_rank[score_order] = np.arange(len(score))
            rank_rows.append({"setting": setting, "response_sha256": view.key[2], "p": p_label(p),
                              "candidate_count": len(score), "pairwise_inversion_vs_p1": inversion_rate(base, score),
                              "mean_abs_rank_shift": float(np.mean(np.abs(base_rank - score_rank))) if len(score) else 0.0})
            if p != 1.0:
                other = score >= .5
                ref = base >= .5
                transition_rows.append({"setting": setting, "response_sha256": view.key[2], "p": p_label(p),
                                        "tau": .5, "p1_retained": int(ref.sum()), "p_retained": int(other.sum()),
                                        "rejected_to_retained": int((~ref & other).sum()),
                                        "retained_to_rejected": int((ref & ~other).sum())})
            for label in ("G-dominant", "L-dominant", "near-equal"):
                mask = bins == label
                disagreement_rows.append({"setting": setting, "response_sha256": view.key[2], "p": p_label(p),
                                          "group": label, "candidate_count": int(mask.sum()),
                                          "mean_G": float(np.mean(g[mask])) if mask.any() else 0.0,
                                          "mean_L": float(np.mean(l[mask])) if mask.any() else 0.0,
                                          "mean_score": float(np.mean(score[mask])) if mask.any() else 0.0})
        # Count distinct masks over all fixed tau values, which exposes p/tau equivalence.
        hashes = {}
        for p in P_ALL:
            for tau in THRESHOLDS:
                mask = accept(g, l, Config("GM", 0, threshold=tau, p=p))
                digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
                hashes.setdefault(digest, []).append(f"p={p_label(p)},tau={tau:g}")
        mask_rows.append({"setting": setting, "response_sha256": view.key[2],
                         "candidate_count": len(g), "total_p_tau_pairs": len(P_ALL) * len(THRESHOLDS),
                         "unique_retained_masks": len(hashes),
                         "equivalence_collapsed_pairs": len(P_ALL) * len(THRESHOLDS) - len(hashes)})
    return score_rows, rank_rows, transition_rows, disagreement_rows, mask_rows


def run_setting(ora, name, output):
    spec = ora.SPECS[name]
    output.mkdir(exist_ok=False)
    context = ora.metst_context(spec)
    _runner, base, _metric, _official, _records, _paths, subjects, _native = context
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("duplicate subject IDs")
    core = FusionCore(base)
    all_grids = grids()
    gm_grid = all_grids["GM"]
    finite_ids = [c.config_id for c in finite_gm_configs(gm_grid)]
    fixed_ids = {p_label(p): [c.config_id for c in gm_grid if c.p == p] for p in P_ALL}
    write_json(output / "protocol.json", {
        "setting": name, "stage": "phase-1 fusion-risk screen",
        "fixed_structure": {"a0": A0, "rho": RHO},
        "p_finite_selectable": [p_label(p) for p in P_FINITE], "p_infinity": "diagnostic_only",
        "tau_grid": list(THRESHOLDS), "nested_selection": "outer subject excluded before every F1/precision/FP tie-break",
        "evaluation": "held-subject full official counts including result synergy",
        "scope": "frozen official ME-TST+ responses; no backbone inference or retraining",
        "grids": {method: [config_dict(c) for c in values] for method, values in all_grids.items()},
        "source": str(spec["source"]), "source_sha256": ora.sha256(spec["source"]),
        "core_sha256": ora.sha256(spec["core"]), "selection_tie_break": ["higher F1", "higher precision", "fewer FP", "fixed grid order"],
    })
    start = time.monotonic()
    tables = {}
    for method, grid in all_grids.items():
        table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
        print(f"{name} {method}: {len(grid)} configurations x {len(subjects)} subjects", flush=True)
        for subject_index, subject in enumerate(subjects):
            for config in grid:
                raw, _, _, _ = decode(context, core, subject_index, config, False)
                table[config.config_id, subject_index] = raw
            if (subject_index + 1) % 5 == 0 or subject_index + 1 == len(subjects):
                print(f"  {subject_index + 1}/{len(subjects)} subjects; elapsed {time.monotonic()-start:.1f}s", flush=True)
        tables[method] = table
        np.savez_compressed(output / f"search_counts_{method}.npz", raw_counts=table, subjects=np.asarray(subjects, dtype=str))

    selected_rows, subject_rows, search_rows, pooled_rows = [], [], [], []
    predictions_path = output / "selected_predictions.jsonl.gz"
    counts_by_variant = {}
    variants = [("G", "G", None), ("L", "L", None)]
    variants += [(f"GM_p{p_label(p)}", "GM", p) for p in P_ALL]
    variants += [("GM_joint", "GM", "joint")]
    with gzip.open(predictions_path, "wt", encoding="utf-8") as predictions_file, gzip.open(output / "candidate_trace.jsonl.gz", "wt", encoding="utf-8") as trace_file:
        for variant, method, selector in variants:
            grid = all_grids[method]
            selected_full, selected_raw = [], []
            allowed = None
            if method == "GM" and selector == "joint":
                allowed = finite_ids
            elif method == "GM" and selector is not None:
                allowed = fixed_ids[p_label(selector)]
            for outer, subject in enumerate(subjects):
                winner, pooled = choose(tables[method], outer, allowed)
                config = grid[winner]
                raw, full, predictions, traces = decode(context, core, outer, config, True, capture=True)
                if raw != tuple(tables[method][winner, outer]):
                    raise RuntimeError("selected raw prediction does not replay its search counts")
                selected_raw.append(raw); selected_full.append(full)
                selected_rows.append({"setting": name, "variant": variant, "subject": subject,
                                      **config_dict(config), "inner_subject_count": len(subjects) - 1,
                                      "inner_raw_F1": metrics(pooled[winner])["F1"],
                                      "inner_raw_precision": metrics(pooled[winner])["precision"]})
                for stage, counts in (("raw", raw), ("full", full)):
                    subject_rows.append({"setting": name, "variant": variant, "subject": subject,
                                         "metric_stage": stage, **metrics(counts)})
                predictions_file.write(json.dumps(serializable({"setting": name, "variant": variant, "subject": subject,
                                                                 "config": config_dict(config), "raw_counts": raw,
                                                                 "full_counts": full, "predictions": predictions}), ensure_ascii=False, allow_nan=False) + "\n")
                for trace in traces:
                    trace_file.write(json.dumps(serializable({"setting": name, "variant": variant,
                                                              "subject": subject, "config": config_dict(config), **trace}), ensure_ascii=False, allow_nan=False) + "\n")
                for candidate, counts in zip(grid, pooled):
                    search_rows.append({"setting": name, "selection_variant": variant, "outer_subject": subject,
                                        **config_dict(candidate), "selected": int(candidate.config_id == winner), **metrics(counts)})
            selected_full = np.asarray(selected_full, dtype=np.int64)
            selected_raw = np.asarray(selected_raw, dtype=np.int64)
            counts_by_variant[variant] = selected_full
            for stage, counts in (("raw", selected_raw), ("full", selected_full)):
                pooled_rows.append({"setting": name, "variant": variant, "metric_stage": stage,
                                    "configs": len(allowed) if allowed is not None else len(grid), **metrics(counts.sum(axis=0))})
            print(f"  {variant}: held-subject full F1={metrics(selected_full.sum(axis=0))['F1']:.6f}", flush=True)

    score_rows, rank_rows, transition_rows, disagreement_rows, mask_rows = diagnostic_rows(core, name)
    write_csv(output / "screening_summary.csv", pooled_rows)
    write_csv(output / "per_subject_counts.csv", subject_rows)
    write_csv(output / "selected_configs.csv", selected_rows)
    write_csv(output / "search_all.csv", search_rows)
    write_csv(output / "candidate_score_distribution.csv", score_rows)
    write_csv(output / "ranking_change.csv", rank_rows)
    write_csv(output / "candidate_transitions.csv", transition_rows)
    write_csv(output / "disagreement_analysis.csv", disagreement_rows)
    write_csv(output / "mask_equivalence.csv", mask_rows)
    write_csv(output / "event_mechanism.csv", [{"setting": name, "available": 0,
                                                 "reason": "sealed ME-TST decoder does not expose stable video/event IDs in the candidate hook; candidate-level trace is provided"}])
    write_csv(output / "selected_p_frequency.csv", [
        {"setting": name, "variant": variant, "p": p, "count": sum(row["variant"] == variant and row["p"] == p for row in selected_rows),
         "total_subjects": len(subjects)}
        for variant in [f"GM_p{p_label(p)}" for p in P_ALL] + ["GM_joint"]
        for p in ([p_label(p) for p in P_ALL] if variant == "GM_joint" else [variant[len("GM_p"):]])
    ])
    comparisons = paired_comparisons(name, counts_by_variant, subjects)
    write_csv(output / "paired_comparisons.csv", comparisons)
    return [r for r in pooled_rows if r["metric_stage"] == "full"], comparisons


def decision(rows, comparisons):
    report = {"recommendation": "REVIEW_PHASE1_RESULTS", "comparisons": comparisons, "per_dataset": []}
    for setting in sorted({r["setting"] for r in rows}):
        f = {r["variant"]: r["F1"] for r in rows if r["setting"] == setting}
        finite = [f.get(f"GM_p{p_label(p)}", 0.0) for p in P_FINITE]
        report["per_dataset"].append({"setting": setting, "GM_joint_F1": f.get("GM_joint"),
                                      "p1_F1": f.get("GM_p1"), "G_F1": f.get("G"), "L_F1": f.get("L"),
                                      "best_fixed_p_F1": max(finite),
                                      "joint_below_both_single": f.get("GM_joint", 0.0) < max(f.get("G", 0.0), f.get("L", 0.0)),
                                      "infinity_is_diagnostic_only": True})
    report["interpretation"] = "Phase-1 screen only; point estimates and bootstrap intervals are not claims of significance. Proceed to structure/p formal search only after checking FP-dominated gains and mask equivalence."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("both", *SETTINGS), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    ora, helper_path = load_helper()
    selected = list(SETTINGS.values()) if args.setting == "both" else [SETTINGS[args.setting]]
    inventory = {name: [{"path": str(p), "exists": p.exists()} for p in ora.required_paths(ora.SPECS[name])] for name in selected}
    if args.check_inputs:
        print(json.dumps(inventory, indent=2))
        if any(not row["exists"] for items in inventory.values() for row in items):
            raise SystemExit(2)
        print("GENERALIZED_MEAN_INPUT_PATHS = PASS (runtime and replay not checked)")
        return
    for name in selected:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = args.output or Path("/content/drive/MyDrive/GLSD_GENERALIZED_MEAN_SCREENING") / datetime.now(timezone.utc).strftime("screen_%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    for name in selected:
        for key in ("dump", "evidence", "run_evidence", "results", "locked_results"):
            protected = ora.SPECS[name][key].resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise RuntimeError(f"output overlaps protected input directory: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                               "settings": selected, "runner_sha256": ora.sha256(Path(__file__)),
                                               "helper_sha256": ora.sha256(helper_path), "input_inventory": inventory,
                                               "stage": "phase-1 fixed-structure generalized-mean screening"})
    preflight = output / "preflight"; preflight.mkdir()
    replay_rows = [ora.run_preflight_setting(name, preflight) for name in selected]
    write_csv(preflight / "locked_replay_summary.csv", replay_rows)
    write_json(preflight / "completion.json", {"completed": True, "settings": selected, "ablation_search_started": False})
    print("SCREENING_NATIVE_AND_OLD_GLSD_REPLAY = PASS", flush=True)
    if args.preflight_only:
        return
    rows, comparisons = [], []
    for name in selected:
        result, paired = run_setting(ora, name, output / name)
        rows.extend(result); comparisons.extend(paired)
    write_csv(output / "screening_summary_full.csv", rows)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_json(output / "decision.json", decision(rows, comparisons))
    write_json(output / "completion.json", {"completed": True, "settings": selected,
                                             "elapsed_seconds": None, "stage": "phase-1 fixed-structure generalized-mean screening"})
    print("GENERALIZED_MEAN_SCREENING_EXECUTION = PASS", flush=True)
    print("OUTPUT =", output, flush=True)


if __name__ == "__main__":
    main()
