"""Nested LOSO sensitivity analysis for S_alpha = alpha*G + (1-alpha)*L.

This runner keeps the GLSD structure and official-response evaluator fixed,
then evaluates a pre-declared alpha grid.  For every alpha and every outer
subject, (a0, rho, tau) is selected from the other subjects' raw counts and
the held-out subject is evaluated with the unchanged full official pipeline.
The one-to-one evaluator patch is installed by the CLI, so all alpha values
share the same matching protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import one_to_one_evaluator as evaluator
import run_full_fusion_tuning as full

shared = full.shared
DEFAULT_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_alphas(value: str) -> tuple[float, ...]:
    """Parse a comma-separated alpha grid and reject ambiguous duplicates."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("alpha grid is empty")
    if any(not np.isfinite(x) or x < 0.0 or x > 1.0 for x in values):
        raise argparse.ArgumentTypeError("each alpha must be finite and in [0, 1]")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("alpha grid contains duplicates")
    return tuple(values)


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def alpha_grid(alpha: float):
    """Use the same 90 (a0, rho, tau) configurations for one fixed alpha."""
    configs = []
    for a0 in full.SCALES:
        for rho in full.RADII:
            for tau in full.THRESHOLDS:
                configs.append(full.Config(
                    method="WeightedMean", config_id=len(configs),
                    reference_scale=a0, local_radius=rho, threshold=tau,
                    alpha=float(alpha), beta=None))
    return configs


def validate_counts(counts, gt_count: int, label: str):
    values = tuple(int(x) for x in counts)
    if len(values) != 3 or min(values) < 0:
        raise RuntimeError(f"invalid {label} counts: {values}")
    if values[0] + values[2] != gt_count or values[0] > gt_count:
        raise RuntimeError(f"GT conservation failed for {label}: {values}, GT={gt_count}")
    return values


def subject_gt(context, index: int) -> int:
    runner, _, _, _, records, *_ = context
    return sum(len(video) for video in runner.final_samples(records)[index])


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    shared.write_csv(path, rows)


def paired_alpha_comparisons(counts_by_alpha, subjects, baseline: float):
    """Paired subject bootstrap of baseline alpha minus each alternative."""
    draws = np.random.default_rng(shared.SEED).integers(
        0, len(subjects), size=(shared.RESAMPLES, len(subjects)))
    distributions = {}
    for alpha, counts in counts_by_alpha.items():
        totals = counts[draws].sum(axis=1).astype(float)
        denom = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        distributions[alpha] = np.divide(
            2 * totals[:, 0], denom, out=np.zeros(shared.RESAMPLES), where=denom > 0)
    if baseline not in distributions:
        raise RuntimeError(f"baseline alpha={baseline} is absent from the grid")
    base_counts = counts_by_alpha[baseline].sum(axis=0)
    rows = []
    for alpha in counts_by_alpha:
        if alpha == baseline:
            continue
        delta = distributions[baseline] - distributions[alpha]
        lo, hi = np.quantile(delta, [.025, .975])
        rows.append(dict(
            target_alpha=baseline,
            reference_alpha=alpha,
            metric_stage="full",
            delta_F1=shared.metrics(base_counts)["F1"] - shared.metrics(counts_by_alpha[alpha].sum(axis=0))["F1"],
            CI_low=float(lo), CI_high=float(hi),
            seed=shared.SEED, resamples=shared.RESAMPLES,
            subjects=len(subjects),
        ))
    return rows


def run_setting(ora, name: str, output: Path, alphas: tuple[float, ...], *,
                prepared_context=None, probe_only=False):
    spec = ora.SPECS[name]
    output.mkdir(parents=False, exist_ok=False)
    context = ora.metst_context(spec) if prepared_context is None else prepared_context
    runner, base, _, _, _, _, subjects, _ = context
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("duplicate subject IDs")
    if ora.context_video_and_gt_counts("metst", context) != (spec["videos"], spec["gt"]):
        raise RuntimeError("unexpected official video/GT counts")

    # The imported scorer validates alpha membership, so configure its globals
    # locally without changing the existing full-tuning defaults on disk.
    full.SCALES = full.LEGACY_SCALES
    full.RADII = full.LEGACY_RADII
    full.THRESHOLDS = full.LEGACY_THRESHOLDS
    full.ALPHAS = tuple(alphas)
    core = full.FusionCore(base)
    protocol = dict(
        experiment="weighted_alpha_hparam_ablation_v1",
        setting=name,
        formula="S_alpha = alpha*G + (1-alpha)*L",
        alphas=list(alphas),
        main_alpha=0.5,
        structure_grid=dict(a0=list(full.SCALES), rho=list(full.RADII), tau=list(full.THRESHOLDS)),
        configs_per_alpha=len(alpha_grid(alphas[0])),
        subjects=list(subjects),
        selection="for each fixed alpha, exclude the held subject; select pooled raw F1, precision, fewer FP, fixed grid order",
        evaluation="held-subject full official counts using unchanged official recognition/result synergy",
        matching_protocol=evaluator.PROTOCOL,
        raw_stage="inner selection and diagnostic counts",
        full_stage="primary official-response Spotting F1",
        scope="frozen official responses; no backbone training or response generation",
        alpha_endpoints="alpha=0 and alpha=1 are represented by the separate L/G module ablation; this run uses interior weights",
        source=str(spec["source"]), source_sha256=ora.sha256(spec["source"]),
        core_sha256=ora.sha256(spec["core"]),
    )
    shared.write_json(output / "protocol.json", protocol)

    if probe_only:
        config = alpha_grid(0.5)[0]
        gt = subject_gt(context, 0)
        raw, full_counts, _, _ = full.decode_with_context(
            context, core, 0, subjects[0], config, True, name, "alpha_probe")
        validate_counts(raw, gt, "probe raw")
        validate_counts(full_counts, gt, "probe full")
        shared.write_json(output / "probe.json", dict(
            completed=True, setting=name, alpha=0.5, config=asdict(config),
            subject=subjects[0], raw_counts=raw, full_counts=full_counts,
            matching_protocol=evaluator.PROTOCOL))
        shared.write_json(output / "completion.json", dict(completed=True, probe_only=True, setting=name))
        return [], []

    pooled_rows, subject_rows, selected_rows, search_rows = [], [], [], []
    counts_by_alpha = {}
    start = time.monotonic()
    predictions_path = output / "selected_predictions.jsonl.gz"
    candidate_path = output / "candidate_decisions.jsonl.gz"
    with gzip.open(predictions_path, "wt", encoding="utf-8") as predictions_file, \
         gzip.open(candidate_path, "wt", encoding="utf-8") as candidate_file:
        for alpha in alphas:
            tag = alpha_tag(alpha)
            variant = f"alpha_{tag}"
            grid = alpha_grid(alpha)
            table = np.zeros((len(grid), len(subjects), 3), dtype=np.int64)
            method_start = time.monotonic()
            print(f"{name} {variant}: {len(grid)} configurations x {len(subjects)} subjects", flush=True)
            for subject_index, subject in enumerate(subjects):
                for config in grid:
                    raw, _, _, _ = full.decode_with_context(
                        context, core, subject_index, subject, config, False, name, variant)
                    table[config.config_id, subject_index] = raw
                if subject_index == 0 or (subject_index + 1) % 5 == 0 or subject_index + 1 == len(subjects):
                    elapsed = time.monotonic() - method_start
                    remaining = elapsed / (subject_index + 1) * (len(subjects) - subject_index - 1)
                    print(f"  {subject_index + 1}/{len(subjects)} subjects; estimated raw remaining={remaining:.1f}s", flush=True)
            np.savez_compressed(output / f"search_counts_{variant}.npz",
                                raw_counts=table, subjects=np.asarray(subjects, dtype=str))

            raw_rows, full_rows = [], []
            for outer, subject in enumerate(subjects):
                winner, pooled = shared.choose(table, outer)
                config = grid[winner]
                for c, counts in zip(grid, pooled):
                    search_rows.append(dict(
                        setting=name, variant=variant,
                        outer_subject=subject,
                        **{k: v for k, v in asdict(c).items() if k != "alpha"},
                        alpha=alpha, selected=int(c.config_id == winner),
                        **shared.metrics(counts)))
                raw, full_counts, predictions, traces = full.decode_with_context(
                    context, core, outer, subject, config, True, name, variant)
                gt = subject_gt(context, outer)
                validate_counts(raw, gt, f"{variant} raw {subject}")
                validate_counts(full_counts, gt, f"{variant} full {subject}")
                if raw != tuple(table[winner, outer]):
                    raise RuntimeError(f"held-subject raw mismatch for {variant}/{subject}")
                raw_rows.append(raw)
                full_rows.append(full_counts)
                selected_rows.append(dict(
                    setting=name, variant=variant, subject=subject,
                    **{k: v for k, v in asdict(config).items() if k != "alpha"},
                    alpha=alpha, inner_subject_count=len(subjects) - 1,
                    inner_raw_F1=shared.metrics(pooled[winner])["F1"],
                ))
                for stage, counts in (("raw", raw), ("full", full_counts)):
                    subject_rows.append(dict(
                        setting=name, variant=variant, alpha=alpha, subject=subject,
                        metric_stage=stage, **shared.metrics(counts)))
                predictions_file.write(json.dumps(shared.serializable(dict(
                    setting=name, variant=variant, alpha=alpha, subject=subject,
                    config=asdict(config), raw_counts=raw, full_counts=full_counts,
                    predictions=predictions)), allow_nan=False) + "\n")
                for trace in traces:
                    candidate_file.write(json.dumps(dict(
                        setting=name, variant=variant, alpha=alpha, subject=subject,
                        config_id=winner, **trace), allow_nan=False) + "\n")
            counts_by_alpha[alpha] = np.asarray(full_rows, dtype=np.int64)
            for stage, rows in (("raw", raw_rows), ("full", full_rows)):
                pooled_rows.append(dict(
                    setting=name, variant=variant, alpha=alpha,
                    metric_stage=stage, configs=len(grid), **shared.metrics(np.sum(rows, axis=0))))
            write_csv(output / "alpha_summary.csv", pooled_rows)
            write_csv(output / "alpha_per_subject_counts.csv", subject_rows)
            write_csv(output / "alpha_selected_configs.csv", selected_rows)
            write_csv(output / "alpha_search_all.csv", search_rows)
            print(f"  {variant}: held-subject full F1={shared.metrics(np.sum(full_rows, axis=0))['F1']:.6f}", flush=True)

    baseline = 0.5
    comparisons = paired_alpha_comparisons(counts_by_alpha, subjects, baseline)
    write_csv(output / "alpha_paired_comparisons.csv", comparisons)
    write_csv(output / "alpha_summary.csv", pooled_rows)
    write_csv(output / "alpha_per_subject_counts.csv", subject_rows)
    write_csv(output / "alpha_selected_configs.csv", selected_rows)
    write_csv(output / "alpha_search_all.csv", search_rows)
    shared.write_json(output / "completion.json", dict(
        completed=True, probe_only=False, setting=name, alphas=list(alphas),
        baseline_alpha=baseline, subject_count=len(subjects),
        elapsed_seconds=time.monotonic() - start, matching_protocol=evaluator.PROTOCOL))
    return [row for row in pooled_rows if row["metric_stage"] == "full"], comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("sammlv", "casme3", "both"), default="sammlv")
    parser.add_argument("--alphas", type=parse_alphas,
                        default=DEFAULT_ALPHAS,
                        help="comma-separated fixed alpha values; default: 0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if 0.5 not in args.alphas:
        parser.error("the alpha grid must include the main-method value 0.5")

    ora, helper_path = shared.load_helper()
    selected = list(shared.SETTINGS.values()) if args.setting == "both" else [shared.SETTINGS[args.setting]]
    inventory = {name: [dict(path=str(path), exists=path.exists())
                        for path in ora.required_paths(ora.SPECS[name])] for name in selected}
    if args.check_inputs:
        print(json.dumps(inventory, indent=2))
        if any(not item["exists"] for paths in inventory.values() for item in paths):
            raise SystemExit(2)
        print("ALPHA_HPARAM_INPUT_PATHS = PASS (runtime and replay not checked)")
        return
    for name in selected:
        ora.verify_sealed_inputs(ora.SPECS[name])
    output = (args.output or Path("/content/drive/MyDrive/GLSD_WEIGHTED_ALPHA_HPARAM_ABLATION") /
              datetime.now(timezone.utc).strftime("alpha_%Y%m%dT%H%M%S_%fZ")).resolve()
    for name in selected:
        for key in ("dump", "evidence", "run_evidence", "results", "locked_results"):
            protected = ora.SPECS[name][key].resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise RuntimeError(f"output overlaps protected input: {protected}")
    output.mkdir(parents=False, exist_ok=False)
    shared.write_json(output / "run_manifest.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), settings=selected,
        alphas=list(args.alphas), baseline_alpha=0.5,
        runner_sha256=ora.sha256(Path(__file__)), full_tuning_sha256=ora.sha256(Path(full.__file__)),
        evaluator_sha256=ora.sha256(Path(evaluator.__file__)), helper_sha256=ora.sha256(helper_path),
        input_inventory=inventory, matching_protocol=evaluator.PROTOCOL))
    print("OUTPUT =", output, flush=True)
    preflight = output / "preflight"
    preflight.mkdir()
    replay = [ora.run_preflight_setting(name, preflight) for name in selected]
    write_csv(preflight / "locked_replay_summary.csv", replay)
    shared.write_json(preflight / "completion.json", dict(completed=True, settings=selected))
    print("LEGACY_NATIVE_AND_GLSD_REPLAY = PASS", flush=True)
    all_rows, all_comparisons = [], []
    for name in selected:
        context = ora.metst_context(ora.SPECS[name])
        metric = context[2]
        audit = output / f"{name}_evaluation_audit"
        audit.mkdir()
        with evaluator.install(metric) as info:
            shared.write_json(audit / "matching_protocol.json", info)
            result, comparisons = run_setting(
                ora, name, output / name, tuple(args.alphas),
                prepared_context=context, probe_only=args.probe_only)
        all_rows.extend(result)
        all_comparisons.extend(comparisons)
        write_csv(output / "alpha_summary_full.csv", all_rows) if all_rows else None
        write_csv(output / "alpha_paired_comparisons.csv", all_comparisons) if all_comparisons else None
    shared.write_json(output / "completion.json", dict(
        completed=True, probe_only=args.probe_only, settings=selected,
        alphas=list(args.alphas), baseline_alpha=0.5,
        matching_protocol=evaluator.PROTOCOL))
    print("ALPHA_HPARAM_PROBE = PASS" if args.probe_only else "ALPHA_HPARAM_ABLATION = PASS", flush=True)
    print("PASS means execution completed; inspect alpha_summary.csv and alpha_paired_comparisons.csv.", flush=True)
    print("OUTPUT =", output, flush=True)


if __name__ == "__main__":
    main()
